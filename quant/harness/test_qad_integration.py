#!/usr/bin/env python3
"""capture_moe.py -> qad_l2.py -> apply -> stream_ref (ハーネス) / qad_l3.py (export) の結合テスト (CPU のみ)。

GPU・CUDA 拡張・実重みは使わない。実パックと同じ名前規則・shard 割り当て (層 N = shard N+3、
Engram と MTP は別 shard、shard に非 expert のテンソルも同居) の小さな偽パックを作って、
  1. capture: 本物の model.py MoE/Gate + stream_ref.moe_forward を偽 Transformer で回し、2 part で記録
     (片方は pack 済み safetensors、片方は生ファイルのまま)
  2. qad_l2 train (CLI, --device cpu): 2 part をまとめて読み、expert mode -> layer mode (--init-diff)
  3. 差分のキー・dtype・形状が pack と同じ、base_sha256 が付く
  4. apply shards / overlay (CLI): ヘッダ不変、expert shard だけ別 inode、他は hardlink、元 pack 不変、
     stream_ref.TensorIndex と qad_l3.SafetensorsIndex の両方が新しい値を読む、overlay と shards の値が同じ
  5. ハーネスの経路 (stream_ref.ExpertStore._read_expert -> get_weight_tensor 式 -> stream_ref.expert_forward)
     で計った教師との誤差が、L2 の学習で下がっている (L2 の式と ハーネスの式が同じものを最適化している)
  6. base の取り違え検出 (apply を別 pack に当てる、L3 export を別 pack に当てる)
  7. overlay の上に overlay を重ねても元の overlay を壊さない
  8. qad_l3: Exl3ResidentSource の base 読み込み (CPU) -> ScaleParams -> export が L2 の pack の上に乗る

  sudo docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v .../ref:/ref:ro -v .../harness:/h:ro \
    -v <scratch>:/s --entrypoint python3 dsv41-quant:151 /h/test_qad_integration.py /s/integ
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capture_moe as C  # noqa: E402
import qad_l2 as Q  # noqa: E402
import qad_l3 as L3  # noqa: E402
import stream_ref as SR  # noqa: E402
import test_qad_l2 as T  # noqa: E402  (exllamav3 の get_weight_tensor の CPU 移植と FP4 量子化)

M = SR.M
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qad_integ"
shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(ROOT)
RES = []


def check(name, cond, detail=""):
    RES.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


DIM, INTER, E, K, L, HAD = 256, 128, 8, 2, 2, 32
SHAPES = {"w1": (DIM, INTER), "w3": (DIM, INTER), "w2": (INTER, DIM)}  # [in, out]
NSH = 48


def shard(i):
    return f"model-{i:05d}-of-{NSH:05d}.safetensors"


# --------------------------------------------------------------------------- 偽 trellis の復号 (2 bit x 256 / tile)
def inner_from_trellis(tr):
    """trellis [in/16, out/16, 32] int16 -> inner [in, out] (2bit 符号を {-1.5,-.5,.5,1.5})。決定的なだけの模擬。"""
    t = tr.to(torch.int32) & 0xFFFF
    v = (t[..., None] >> (2 * torch.arange(8))) & 3  # [ti, to, 32, 8]
    ti, to = tr.shape[:2]
    v = (v.reshape(ti, to, 16, 16).float() - 1.5).permute(0, 2, 1, 3).reshape(ti * 16, to * 16)
    return v.half()


def mock_qtilde(tr, mcg, device):
    inner = inner_from_trellis(tr)
    return T.mock_get_weight_tensor(inner, torch.ones(inner.shape[0]).half(), torch.ones(inner.shape[1]).half(), HAD)


def mock_wt(d):
    """LinearEXL3(...).get_weight_tensor() の CPU 版 (suh/svh 込み、[in, out])。"""
    return T.mock_get_weight_tensor(inner_from_trellis(d["trellis"]), d["suh"], d["svh"], HAD)


# --------------------------------------------------------------------------- 偽の教師 ckpt と v1 パック
g = torch.Generator().manual_seed(0)
TEACH, PACK = os.path.join(ROOT, "orig"), os.path.join(ROOT, "v1")
truth = {}
pk_t, tc_t = {}, {}
for l in range(L):
    for e in range(E):
        for p, (i, o) in SHAPES.items():
            tr = torch.randint(-2**15, 2**15 - 1, (i // 16, o // 16, 32), dtype=torch.int16, generator=g)
            qt = mock_qtilde(tr, None, "cpu").float()
            a = torch.sign(torch.randn(i, generator=g)) * 0.05 * (0.5 + torch.rand(i, generator=g))
            if p == "w2":
                a = a * 4
            b = torch.sign(torch.randn(o, generator=g)) * (1 + 0.03 * torch.randn(o, generator=g))
            w_eff = a[:, None] * qt * b[None, :]
            w_eff = w_eff + 0.02 * w_eff.std() * torch.randn(i, o, generator=g)
            wq, sc = T.fp4_quantize(w_eff.t().contiguous())
            tc_t[Q.ename(l, e, p, "weight")] = wq
            tc_t[Q.ename(l, e, p, "scale")] = sc.view(torch.float8_e8m0fnu)
            pk_t[Q.ename(l, e, p, "trellis")] = tr
            pk_t[Q.ename(l, e, p, "suh")] = (a * (1 + 0.15 * torch.randn(i, generator=g))).half()
            pk_t[Q.ename(l, e, p, "svh")] = (b * (1 + 0.15 * torch.randn(o, generator=g))).half()
            pk_t[Q.ename(l, e, p, "mcg")] = torch.tensor(-878968851, dtype=torch.int32)


def write_ckpt(d, experts):
    os.makedirs(os.path.join(d, "inference"))
    os.makedirs(os.path.join(d, "encoding"))
    wm = {}
    groups = {1: {"embed.weight": torch.randn(16, DIM, generator=g).to(torch.bfloat16)},
              2: {"layers.0.engram.embed.weight": torch.randint(0, 255, (64, DIM), dtype=torch.uint8, generator=g)},
              44: {"mtp.0.ffn.gate.weight": torch.randn(E, DIM, generator=g).to(torch.bfloat16)}}
    for l in range(L):
        t = {f"layers.{l}.attn_norm.weight": torch.randn(DIM, generator=g).to(torch.bfloat16),
             f"layers.{l}.ffn.gate.weight": torch.randn(E, DIM, generator=g).to(torch.bfloat16),
             f"layers.{l}.ffn.gate.bias": torch.randn(E, generator=g)}
        t.update({k: v for k, v in experts.items() if k.startswith(f"layers.{l}.")})
        groups[l + 3] = t
    for i, t in groups.items():
        Q.st_save(os.path.join(d, shard(i)), t)
        wm.update({k: shard(i) for k in t})
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump({"quantization_config": {"quant_method": "exl3", "bits": 2}}, open(os.path.join(d, "config.json"), "w"))
    json.dump({"swiglu_limit": 10.0}, open(os.path.join(d, "inference", "config.json"), "w"))
    open(os.path.join(d, "encoding", "encoding.py"), "w").write("# fake\n")
    open(os.path.join(d, "tokenizer.json"), "w").write("{}")


write_ckpt(TEACH, tc_t)
write_ckpt(PACK, pk_t)
pack_sums = {f: sha(os.path.join(PACK, f)) for f in os.listdir(PACK) if f.endswith(".safetensors")}
print(f"[setup] 偽 ckpt / v1 パック: {L} 層 x {E} experts、shard {sorted(int(f[6:11]) for f in pack_sums)}", flush=True)

# --------------------------------------------------------------------------- 1. capture (2 part)
print("[1] capture (本物の MoE/Gate + stream_ref.moe_forward、2 part)", flush=True)
torch.set_default_dtype(torch.bfloat16)
M.default_dtype = torch.bfloat16
margs = M.ModelArgs(dim=DIM, moe_inter_dim=INTER, n_routed_experts=E, n_activated_experts=K, n_layers=L,
                    swiglu_limit=10.0, route_scale=1.5, score_func="sqrtsoftplus", expert_dtype=None, dtype="bf16")
moes = []
for l in range(L):
    m = M.MoE(l, margs)
    with torch.no_grad():
        for name, prm in m.named_parameters():
            prm.normal_(0, 0.3 if "gate.weight" in name else (0.02 if "gate.bias" in name else 0.05))
    m.swiglu_limit = 10.0
    moes.append(m)
norms = [M.RMSNorm(DIM, 1e-6) for _ in range(L)]
with torch.no_grad():
    for n_ in norms:
        n_.weight.fill_(4.0)


class FakeStore:
    counts, stats = {}, {}

    def get(self, layer, used):
        mm = moes[layer]
        return {e: ("bf16", mm.experts[e].w1.weight.t().contiguous(), mm.experts[e].w3.weight.t().contiguous(),
                    mm.experts[e].w2.weight.t().contiguous()) for e in used}


SR.STORE = FakeStore()
M.MoE.forward = SR.moe_forward  # stream_ref.build と同じ差し替え
VOCAB = 300
emb = torch.randn(VOCAB, DIM, generator=g)


def fake_forward(ids):
    n = len(ids)
    h = emb[torch.tensor(ids)].float()
    for l in range(L):
        mix = torch.cumsum(h, 0) / torch.arange(1, n + 1)[:, None]
        xx = norms[l]((h + mix).to(torch.bfloat16))
        h = h + moes[l](xx.view(1, n, DIM)).view(n, DIM).float()


def tok_decode(ids):
    return bytes([t - 4 for t in ids if 4 <= t < 260]).decode("utf-8", errors="replace")


def enc(t):
    return [4 + b for b in t.encode()]


ja = ["量子化の誤差を減らすために日本語のデータを多めに入れます。", "この関数はエラーを返すので修正してください。",
      "ログを確認したところメモリが足りないようです。"]
en = ["the code returns an error when x = 0\n", "def f(x): return x\n", "please fix the error in the code\n"]
rng = np.random.default_rng(0)
inp = os.path.join(ROOT, "calib.jsonl")
with open(inp, "w") as f:
    for q in range(24):
        body = "".join(rng.choice(ja if q % 3 == 0 else en, size=7))
        ids = [0] + enc("system: be helpful.\n") + enc(body)
        # build_calib.py の行形式 (区分は lang に入る)
        f.write(json.dumps({"ids": ids, "score_from": 10, "src": f"syn{q}", "off": q,
                            "lang": "ja" if q % 3 == 0 else ("agent" if q % 3 == 1 else "code")}) + "\n")


def cap_args(out, part):
    ap = C.argparse.ArgumentParser()
    C.add_common(ap)
    return ap.parse_args(["--input", inp, "--out", out, "--dim", str(DIM), "--topk", str(K), "--n-experts", str(E),
                          "--n-layers", str(L), "--window", "30", "--rows-per-layer", "3000", "--expert-floor", "60",
                          "--part", part])


CAPS = []
for part in ("0/2", "1/2"):
    out = os.path.join(ROOT, "cap_p" + part[0])
    a = cap_args(out, part)
    plan = C.make_plan(a, decode=tok_decode, special_ids={0, 1, 2, 3})
    st = C.CaptureStore(out, C.build_meta(a, plan, "/fake"), fsync=False)
    cp = C.Capturer(st, plan)
    cp.install(M)
    try:
        with torch.inference_mode():
            C.run_capture(st, plan, cp, fake_forward, log=lambda *a_, **k_: None)
    finally:
        cp.uninstall()
    CAPS.append(out)
C.pack(CAPS[1])  # part 1 は pack 済み、part 0 は生ファイル
torch.set_default_dtype(torch.float32)
rows = [C.load_layer(CAPS, l, as_torch=False) for l in range(L)]
check("capture: 2 part とも記録、part1 は pack 済み", all(os.path.exists(os.path.join(CAPS[1], f"layer_{l:02d}.safetensors")) for l in range(L))
      and not os.path.exists(os.path.join(CAPS[0], "layer_00.safetensors")), f"rows/layer {[len(r['seq']) for r in rows]}")
d0 = Q.load_capture(CAPS, 0)
check("load_capture: 生 + pack の 2 part を連結し、F_DUP と kp=0 を除く",
      d0["x"].shape[0] == int((((rows[0]["flags"] & C.F_DUP) == 0) & (rows[0]["kp"] > 0)).sum()) and d0["x"].dtype == torch.bfloat16
      and d0["indices"].shape[1] == K and int(d0["indices"].max()) < E, f"N={d0['x'].shape[0]}, ja={int(d0['lang'].sum())}")
check("load_capture: seq は part をまたいで重ならない (gid)",
      not (set(C.load_layer(CAPS[0], 0, False)["seq"].tolist()) & set(C.load_layer(CAPS[1], 0, False)["seq"].tolist())))
check("load_capture: stream=teacher を報告 (学生 stream 推奨の警告の元)", d0["stream"] == ["teacher"])
meta0 = json.load(open(os.path.join(CAPS[0], "meta.json")))
dom_ok = all(s_["domain"] == ("ja" if s_["off"] % 3 == 0 else ("agent" if s_["off"] % 3 == 1 else "code")) for s_ in meta0["seqs"])
st0, _ = C.load_stats(CAPS[0])
check("capture: build_calib の lang (ja/agent/code) を domain に使う (dom_* の routing 統計が区分別になる)",
      meta0["domains"] == ["agent", "code", "ja"] and dom_ok and all(f"cnt.dom_{d_}" in st0 for d_ in meta0["domains"]),
      str(meta0["domains"]))

# --------------------------------------------------------------------------- 2. qad_l2 train (CLI)
print("[2] qad_l2 train (CLI, cpu)", flush=True)
Q.PackEXL3.qtilde_fn = staticmethod(mock_qtilde)
DIFF, DIFF2 = os.path.join(ROOT, "diff_expert"), os.path.join(ROOT, "diff_layer")
common = ["--orig-ckpt", TEACH, "--pack", PACK, "--layers", "0-1", "--device", "cpu", "--n-experts", str(E),
          "--min-tokens", "40", "--min-val", "8", "--min-val-ja", "8", "--batch", "128", "--lr", "5e-3", "--l2", "0.01",
          "--max-steps", "150", "--min-steps", "40", "--eval-batch", "4096", "--layer-batch", "512", "--patience", "4"]
for c in CAPS:
    common += ["--capture", c]
Q.main(["train", "--out", DIFF, "--loss", "expert"] + common)
Q.main(["train", "--out", DIFF2, "--loss", "layer", "--epochs", "20", "--eval-every", "5",
        "--init-diff", os.path.join(DIFF, "qad_l2_L{L:02d}.safetensors")] + common)
pk_hdr = {}
for f in pack_sums:
    h, _ = Q.st_header(os.path.join(PACK, f))
    pk_hdr.update({k: v for k, v in h.items() if k != "__metadata__"})
for dd in (DIFF, DIFF2):
    for l in range(L):
        tens, md = Q.st_load(os.path.join(dd, f"qad_l2_L{l:02d}.safetensors"))
        rep = json.load(open(os.path.join(dd, f"qad_l2_L{l:02d}.json")))
        ok_keys = all(k in pk_hdr and pk_hdr[k]["dtype"] == "F16" and pk_hdr[k]["shape"] == list(v.shape) and v.dtype == torch.float16
                      and k.startswith(f"layers.{l}.ffn.experts.") for k, v in tens.items())
        signs = all(bool((torch.sign(v.float()) == torch.sign(pk_t[k].float())).all()) for k, v in tens.items())
        lv = rep.get("layer_val", {}).get("weighted", {})
        check(f"{os.path.basename(dd)} L{l}: 差分のキー/dtype/形状 = pack、符号保存、base_sha256 付き",
              len(tens) > 0 and ok_keys and signs and len(md.get("base_sha256", "")) == 64 and md.get("pack") == os.path.abspath(PACK),
              f"{len(tens)} tensors, rel-MSE v1 {lv.get('v1', float('nan')):.4f} -> {lv.get('qad', float('nan')):.4f}")

# --------------------------------------------------------------------------- 4. apply (CLI)
print("[4] apply shards / overlay", flush=True)
OUT_S, OUT_O = os.path.join(ROOT, "v1-qad"), os.path.join(ROOT, "v1-qad-ov")
Q.main(["apply", "--pack", PACK, "--diff", DIFF2, "--out", OUT_S, "--mode", "shards"])
Q.main(["apply", "--pack", PACK, "--diff", DIFF2, "--out", OUT_O, "--mode", "overlay"])
check("apply: 元の v1 パックは全 shard の sha が不変", all(sha(os.path.join(PACK, f)) == h for f, h in pack_sums.items()))
exp_sh = {shard(l + 3) for l in range(L)}
link_ok = all(os.path.samefile(os.path.join(PACK, f), os.path.join(OUT_S, f)) == (f not in exp_sh) for f in pack_sums)
cfg_ok = all(os.path.samefile(os.path.join(PACK, f), os.path.join(OUT_S, f))
             for f in ("config.json", "model.safetensors.index.json", "tokenizer.json", "inference/config.json", "encoding/encoding.py"))
check("apply shards: 層 N の expert shard (N+3) だけ別 inode、他 shard (Engram/MTP/embed) と設定は hardlink", link_ok and cfg_ok)
hdr_ok = all(Q.st_header(os.path.join(PACK, f)) == Q.st_header(os.path.join(OUT_S, f)) for f in pack_sums)
check("apply shards: 全 shard のヘッダ (キー・dtype・形状・offset) が v1 と同一", hdr_ok)
new_all = {}
for l in range(L):
    new_all.update(Q.st_load(os.path.join(DIFF2, f"qad_l2_L{l:02d}.safetensors"))[0])
ti_s, ti_o = Q.stream_ref_tensor_index()(OUT_S), Q.stream_ref_tensor_index()(OUT_O)
si_s, si_o = L3.SafetensorsIndex(OUT_S), L3.SafetensorsIndex(OUT_O)
same = True
for k in pk_hdr:
    a_, b_ = Q._tensor_bytes(ti_s.load(k)), Q._tensor_bytes(ti_o.load(k))
    want = new_all.get(k, pk_t.get(k))
    if want is not None and a_ != Q._tensor_bytes(want):
        same = False
    if a_ != b_:
        same = False
    if pk_hdr[k]["dtype"] in L3._ST_NP and not (si_s.read_numpy(k).tobytes() == si_o.read_numpy(k).tobytes() == a_):
        same = False
check("apply: shards と overlay で全テンソルが同じ値 (stream_ref.TensorIndex と qad_l3.SafetensorsIndex の両方)、差分は新しい値", same,
      f"{len(new_all)} 個の suh/svh を差し替え")
check("apply overlay: マーカーと zz- 差分、index だけ新しい", os.path.exists(os.path.join(OUT_O, "QAD_OVERLAY_HARNESS_ONLY"))
      and os.path.exists(os.path.join(OUT_O, "zz-qad_l2_L00.safetensors"))
      and not os.path.samefile(os.path.join(PACK, "model.safetensors.index.json"), os.path.join(OUT_O, "model.safetensors.index.json")))

# --------------------------------------------------------------------------- 5. ハーネスの経路で効果が出るか
print("[5] ハーネスの経路 (ExpertStore._read_expert -> get_weight_tensor 式 -> expert_forward)", flush=True)


def harness_err(pack_dir, l, x, idx, w):
    store = SR.ExpertStore(SR.TensorIndex(pack_dir), L, 1, "cpu", SR.TensorIndex(pack_dir))
    tch = Q.TeacherFP4(TEACH)
    num = den = 0.0
    for e in range(E):
        tok, top = torch.where(idx == e)
        if tok.numel() == 0:
            continue
        kind, parts = store._read_expert(l, e)
        assert kind == "exl3"
        W = ("bf16",) + tuple(mock_wt(d).float() for d in parts)
        xe = x[tok].float()
        ge = w[tok, top, None].float()
        yS = SR.expert_forward(xe, W, ge, 10.0).float()
        yT = Q.teacher_expert(x[tok], Q.TeacherFP4.weights(tch.raw(l, e), "cpu", torch.float32), ge, 10.0, True, torch.float32).float()
        num += (yT - yS).pow(2).sum().item()
        den += yT.pow(2).sum().item()
    return num / den


for l in range(L):
    dd = Q.load_capture(CAPS, l)
    e_v1 = harness_err(PACK, l, dd["x"], dd["indices"], dd["weights"])
    e_s = harness_err(OUT_S, l, dd["x"], dd["indices"], dd["weights"])
    e_o = harness_err(OUT_O, l, dd["x"], dd["indices"], dd["weights"])
    check(f"L{l}: ハーネス経路の教師誤差 v1 > L2 (shards = overlay)", e_s < 0.8 * e_v1 and abs(e_s - e_o) < 1e-12,
          f"rel-MSE v1 {e_v1:.4f} -> shards {e_s:.4f} / overlay {e_o:.4f}")
# L2 の式 ((x*su)@Q~)*sv と ハーネスの式 x @ get_weight_tensor(su, sv) の一致 (保存した fp16 の値で)
x = d0["x"][:64].float()
kmax = 0.0
for e in range(E):
    for p in Q.PARTS:
        nm = Q.ename(0, e, p, "suh")
        if nm not in new_all:
            continue
        d = {"trellis": pk_t[Q.ename(0, e, p, "trellis")], "suh": new_all[nm], "svh": new_all[Q.ename(0, e, p, "svh")]}
        xin = x if p != "w2" else torch.randn(64, INTER)
        y_h = xin @ mock_wt(d).float()
        y_l2 = ((xin * d["suh"].float()) @ mock_qtilde(d["trellis"], None, "cpu").float()) * d["svh"].float()
        kmax = max(kmax, ((y_h - y_l2).norm() / y_h.norm()).item())
check("L2 の分解式とハーネスの get_weight_tensor 式が保存値で一致 (fp16 丸め程度)", kmax < 2e-3, f"max rel {kmax:.2e}")

# --------------------------------------------------------------------------- 6. base の取り違え検出
print("[6] base の取り違え検出", flush=True)
try:
    Q.main(["apply", "--pack", OUT_S, "--diff", DIFF, "--out", os.path.join(ROOT, "wrong"), "--mode", "overlay"])
    check("apply: 学習時と違う pack (L2 済み) に当てると拒否", False)
except SystemExit as ex:
    check("apply: 学習時と違う pack (L2 済み) に当てると拒否", "base" in str(ex) or "--pack" in str(ex), str(ex)[:60])
shutil.rmtree(os.path.join(ROOT, "wrong"), ignore_errors=True)

try:
    Q.apply_pack(OUT_O, DIFF2, os.path.join(ROOT, "wrong_s"), "shards")
    check("apply: overlay pack から shards は作らない (ds41rt 用にならない)", False)
except SystemExit as ex:
    check("apply: overlay pack から shards は作らない (ds41rt 用にならない)", "overlay" in str(ex), str(ex)[:50])

# --------------------------------------------------------------------------- 7. overlay の上に overlay
print("[7] overlay の上に overlay を重ねる", flush=True)
ov_sums = {f: sha(os.path.join(OUT_O, f)) for f in os.listdir(OUT_O) if f.endswith(".safetensors")}
DIFF3 = os.path.join(ROOT, "diff_l0_again")
os.makedirs(DIFF3)
k0 = Q.ename(0, 0, "w1", "suh")
ti_o2 = Q.stream_ref_tensor_index()(OUT_O)
again = {k0: (ti_o2.load(k0).float() * 1.01).half()}
Q.st_save(os.path.join(DIFF3, "qad_l2_L00.safetensors"), again,
          {"base_sha256": Q.base_digest(lambda k: Q._tensor_bytes(ti_o2.load(k)), again)})
OUT_O2 = os.path.join(ROOT, "v1-qad-ov2")
Q.apply_pack(OUT_O, DIFF3, OUT_O2, "overlay")
check("overlay の上の overlay: 元の overlay の zz- 差分を上書きしない", all(sha(os.path.join(OUT_O, f)) == h for f, h in ov_sums.items()))
v_new = Q.stream_ref_tensor_index()(OUT_O2).load(k0)
check("overlay の上の overlay: 新しい値が読める", torch.equal(v_new, again[k0]))

# --------------------------------------------------------------------------- 8. L3 export を L2 の pack の上に
print("[8] qad_l3: Exl3ResidentSource の base (CPU) -> ScaleParams -> export", flush=True)
src = L3.Exl3ResidentSource(OUT_S, list(range(L)), "cpu", n_experts=E, workers=4)
check("L3: Exl3ResidentSource が L2 shards パックの trellis/base を読む (K=2)", src.K == 2 and
      torch.equal(src.base_scales(0)[0][1].half(), (new_all.get(Q.ename(0, 1, "w1", "suh"), pk_t[Q.ename(0, 1, "w1", "suh")]))))
sp = L3.ScaleParams({l: src.base_scales(l) for l in range(L)})
with torch.no_grad():
    for t_ in sp.theta.values():
        t_.normal_(0, 0.03)
ident = {"pack": os.path.abspath(OUT_S),
         "base_sha256": L3.base_digests(lambda: [(f"{l}_{k}", getattr(sp, f"base_{l}_{k}")) for l in sp.layers for k in range(6)])}
thp = os.path.join(ROOT, "theta_r0_best.pt")
torch.save({"theta": sp.state_dict(), **ident}, thp)
l2_sums = {f: sha(os.path.join(OUT_S, f)) for f in pack_sums}
OUT3 = os.path.join(ROOT, "v1-qad-l3")
from types import SimpleNamespace  # noqa: E402
rep = L3.export_main(SimpleNamespace(pack=OUT_S, theta=[thp], out=OUT3, dry_run=False))
si3 = L3.SafetensorsIndex(OUT3)
nm = Q.ename(1, 3, "w2", "svh")
want = (torch.from_numpy(si_s.read_numpy(nm)).float() * torch.exp(sp.theta["1_5"][3].detach())).half()
check("L3 export: suh/svh = (L2 の値) * exp(theta)、base 照合 ok", np.array_equal(si3.read_numpy(nm), want.numpy())
      and rep["base_check"] == ["ok"] and rep["n"] == L * E * 6)
check("L3 export: L2 パックは不変、ヘッダは同一、expert shard 以外は hardlink",
      all(sha(os.path.join(OUT_S, f)) == h for f, h in l2_sums.items())
      and all(Q.st_header(os.path.join(OUT_S, f)) == Q.st_header(os.path.join(OUT3, f)) for f in pack_sums)
      and all(os.path.samefile(os.path.join(OUT_S, f), os.path.join(OUT3, f)) == (f not in exp_sh) for f in pack_sums))
check("L3 export: stream_ref.TensorIndex でも読める", torch.equal(Q.stream_ref_tensor_index()(OUT3).load(nm), want))
try:
    L3.export_main(SimpleNamespace(pack=PACK, theta=[thp], out=os.path.join(ROOT, "wrong3"), dry_run=False))
    check("L3 export: 学習時と違う pack (v1) に当てると拒否", False)
except ValueError as ex:
    check("L3 export: 学習時と違う pack (v1) に当てると拒否", "base" in str(ex) or "pack" in str(ex), str(ex)[:60])

bad = [n for n, ok in RES if not ok]
print(f"\n{len(RES) - len(bad)}/{len(RES)} passed" + (f"  FAIL: {bad}" if bad else ""), flush=True)
sys.exit(1 if bad else 0)
