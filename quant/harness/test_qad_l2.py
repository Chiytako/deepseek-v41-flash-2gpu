#!/usr/bin/env python3
"""qad_l2.py の CPU テスト (GPU・CUDA 拡張・実重みは使わない)。

  python3 test_qad_l2.py            # 全部
  python3 test_qad_l2.py decomp     # 名前に decomp を含むものだけ

Hadamard は Sylvester 型を自作し、exllamav3 1.5.1 の preapply_had_l / preapply_had_r
(exl3_lib/quantize.py:401-418) と get_weight_tensor (modules/quant/exl3.py:239-248) を同じ演算順・同じ dtype で移植した。
"""
import hashlib
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qad_l2 as Q  # noqa: E402

torch.manual_seed(0)
RESULTS = []


# --------------------------------------------------------------------------- exllamav3 の移植 (CPU)
def hadamard(n, dtype=torch.float32):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.shape[0] == n
    return (h / math.sqrt(n)).to(dtype)


def preapply_had_l(x, had_dim, cdt=torch.float):
    k, n = x.shape
    x_dtype = x.dtype
    x = x.to(cdt)
    had = hadamard(had_dim, cdt)
    x = (had @ x.view(-1, had_dim, n)).view(k, n)
    return x.to(x_dtype)


def preapply_had_r(x, had_dim, cdt=torch.float):
    k, n = x.shape
    x_dtype = x.dtype
    x = x.to(cdt)
    had = hadamard(had_dim, cdt)
    x = (x.view(k, -1, had_dim) @ had).view(k, n)
    return x.to(x_dtype)


def mock_get_weight_tensor(inner, suh, svh, had=32, cdt=torch.float):
    """LinearEXL3.get_weight_tensor と同じ: w=had_l(inner); w*=suh[:,None]; w=had_r(w); w*=svh[None,:]。"""
    w = inner.clone()
    w = preapply_had_l(w, had, cdt)
    w *= suh.to(w.dtype).unsqueeze(1)
    w = preapply_had_r(w, had, cdt)
    w *= svh.to(w.dtype).unsqueeze(0)
    return w


# --------------------------------------------------------------------------- FP4 量子化 (テスト用の教師作成)
POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fp4_quantize(w):
    """[out, in] float -> (int8 [out, in/2], uint8 E8M0 [out, in/32])。ブロック 32 は in 方向。"""
    o, i = w.shape
    wb = w.float().view(o, i // 32, 32)
    amax = wb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 6.0))
    sc = torch.pow(2.0, e)
    v = wb / sc
    mag = (v.abs().unsqueeze(-1) - POS).abs().argmin(-1)
    code = (mag | torch.where(v < 0, 8, 0)).to(torch.uint8).view(o, i)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
    return packed.contiguous(), (e.view(o, i // 32) + 127).to(torch.uint8)


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)


# --------------------------------------------------------------------------- tests
def test_decomp():
    """W = diag(su)·Q̃·diag(sv) が get_weight_tensor 式と一致すること (実数で厳密、fp16 で丸め誤差のみ)。"""
    for in_f, out_f, had in ((256, 128, 32), (512, 256, 128)):
        inner = torch.randn(in_f, out_f).half()
        su = (torch.sign(torch.randn(in_f)) * 0.017 * (0.3 + 0.9 * torch.rand(in_f))).half()
        sv = (torch.sign(torch.randn(out_f)) * (1 + 0.03 * torch.randn(out_f))).half()
        ones_u, ones_v = torch.ones(in_f).half(), torch.ones(out_f).half()
        # (1) fp16 の実経路
        w_ref = mock_get_weight_tensor(inner, su, sv, had).float()
        qt = mock_get_weight_tensor(inner, ones_u, ones_v, had).float()
        w_fac = su.float()[:, None] * qt * sv.float()[None, :]
        rel16 = ((w_fac - w_ref).norm() / w_ref.norm()).item()
        check(f"decomp fp16 {in_f}x{out_f} had{had}", rel16 < 2e-3, f"rel {rel16:.2e} (fp16 の段ごとの丸めだけ)")
        # (2) float64 で厳密
        w64 = mock_get_weight_tensor(inner.double(), su.double(), sv.double(), had, torch.float64)
        q64 = mock_get_weight_tensor(inner.double(), ones_u.double(), ones_v.double(), had, torch.float64)
        f64 = su.double()[:, None] * q64 * sv.double()[None, :]
        rel64 = ((f64 - w64).norm() / w64.norm()).item()
        check(f"decomp fp64 {in_f}x{out_f} had{had}", rel64 < 1e-13, f"rel {rel64:.2e} (結合法則で厳密に一致)")
        # (3) 対照: su を had_l の前 (inner 側) に掛けると一致しない = diag(su) と H_in は非可換
        wrong = preapply_had_r(preapply_had_l(inner.double() * su.double()[:, None], had, torch.float64), had,
                               torch.float64) * sv.double()[None, :]
        relw = ((wrong - w64).norm() / w64.norm()).item()
        check(f"decomp 対照 (su を had_l の前) {in_f}x{out_f}", relw > 0.05, f"rel {relw:.2f} (一致しないのが正しい)")
        H = hadamard(had, torch.float64)
        D = torch.diag(su.double()[:had])
        check(f"decomp diag(su)H != H diag(su) (had{had})", (D @ H - H @ D).abs().max().item() > 1e-3)
        # (4) 学生の forward ((x⊙su)@Q̃)⊙sv が x@W_ref と一致
        x = torch.randn(64, in_f)
        bank = Q.ScaleBank([su[None]] * 3, [sv[None]] * 3, [[qt, qt, qt]], 0.0, torch.float32)
        y_bank = bank.lin(0, 0, x)
        y_ref = x @ w_ref
        relf = ((y_bank - y_ref).norm() / y_ref.norm()).item()
        check(f"student lin == x@W {in_f}x{out_f}", relf < 2e-3, f"rel {relf:.2e}")


def test_fp4():
    b = torch.tensor([[0x21, 0xF9]], dtype=torch.uint8).view(torch.int8)  # 下位 1(0.5) 上位 2(1.0) / 下位 9(-0.5) 上位 F(-6)
    s = torch.full((1, 1), 127, dtype=torch.uint8)
    w = torch.zeros(1, 16, dtype=torch.int8)  # 32 値 = 1 ブロック
    w[0, :2] = b[0]
    v = Q.fp4_dequant(w, s, torch.float32)[0, :4].tolist()
    check("fp4 nibble 順 (下位が偶数番目)", v == [0.5, 1.0, -0.5, -6.0], str(v))
    s2 = torch.tensor([[130, 120]], dtype=torch.uint8)  # 2^3, 2^-7
    w2 = torch.zeros(1, 16, dtype=torch.int8)
    w2[0, 0] = 0x77
    w64 = torch.cat([w2, w2], 1)
    v2 = Q.fp4_dequant(w64, s2, torch.float32)
    check("fp4 E8M0 scale /32 ブロック", v2[0, 0].item() == 48.0 and v2[0, 32].item() == 6.0 * 2 ** -7,
          f"{v2[0, 0].item()} {v2[0, 32].item()}")
    s3 = torch.tensor([[130, 120]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    check("fp4 E8M0 dtype でも同じ", torch.equal(Q.fp4_dequant(w64, s3, torch.float32), v2))
    W = torch.randn(64, 256) * 0.02
    qw, qs = fp4_quantize(W)
    d32 = Q.fp4_dequant(qw, qs, torch.float32)
    d16 = Q.fp4_dequant(qw, qs, torch.bfloat16).float()
    check("fp4 dequant は bf16 で厳密", torch.equal(d32, d16))
    rel = ((d32 - W).norm() / W.norm()).item()
    check("fp4 往復誤差 (e2m1 相当)", rel < 0.2, f"rel {rel:.3f}")


def test_act_quant():
    x = torch.randn(16, 128) * torch.logspace(-3, 3, 16)[:, None]
    q = Q.act_quant_sim(x)
    ok = True
    for r in range(16):
        for b in range(4):
            blk = x[r, b * 32:(b + 1) * 32].double()
            amax = max(blk.abs().max().item(), 1e-4)
            v = np.float32(amax) * np.float32(1 / 448.0)
            sc = 2.0 ** math.ceil(math.log2(float(v)))
            ref = (blk / sc).clamp(-448, 448).float().to(torch.float8_e4m3fn).double() * sc
            ok &= torch.allclose(q[r, b * 32:(b + 1) * 32].double(), ref, rtol=0, atol=0)
    check("act_quant 2 冪 scale と e4m3 丸め", ok)
    x2 = torch.zeros(1, 32)
    x2[0, 0] = 448.0 * 4  # amax/448 = 4 ちょうど -> scale 4 (切り上げない)
    x2[0, 1] = 4 * 2 ** -9 * 1.25  # scale 4 なら e4m3 の非正規化 1 段 (4*2^-9) に、scale 8 なら 8*2^-9 に丸まる
    q2 = Q.act_quant_sim(x2)
    check("act_quant 境界 (amax/448 が 2 冪なら切り上げない)", q2[0, 0].item() == 1792.0 and q2[0, 1].item() == 4 * 2 ** -9,
          f"{q2[0, :2].tolist()}")
    check("act_quant 結果は bf16 で厳密", torch.equal(q, q.to(torch.bfloat16).float()))


def test_clamp():
    dim, inter = 32, 32
    x = torch.zeros(1, dim)
    x[0, 0] = 1.0
    w1 = torch.zeros(inter, dim)
    w3 = torch.zeros(inter, dim)
    w2 = torch.zeros(dim, inter)
    w1[0, 0], w3[0, 0], w2[0, 0] = -20.0, 20.0, 1.0  # gate=-20, up=20
    w1[1, 0], w3[1, 0], w2[1, 1] = 20.0, -20.0, 1.0  # gate=+20, up=-20
    g = torch.ones(1, 1)
    y = Q.teacher_expert(x, (w1, w3, w2), g, 10.0, act_quant=False, cdt=torch.float32).float()
    e0 = (torch.nn.functional.silu(torch.tensor(-20.0)) * 10).item()  # gate は下側を clamp しない
    e1 = (torch.nn.functional.silu(torch.tensor(10.0)) * -10).item()
    check("teacher clamp: gate 上側だけ / up ±10", abs(y[0, 0].item() - e0) < 1e-4 * abs(e0) + 1e-6
          and abs(y[0, 1].item() - e1) < 1e-2, f"{y[0, :2].tolist()} vs {[e0, e1]}")
    ones = [torch.ones(1, dim), torch.ones(1, dim), torch.ones(1, inter)]
    onesv = [torch.ones(1, inter), torch.ones(1, inter), torch.ones(1, dim)]
    bank = Q.ScaleBank(ones, onesv, [[w1.t().contiguous(), w3.t().contiguous(), w2.t().contiguous()]], 10.0, torch.float32)
    ys = bank(0, x, g)
    check("student clamp が教師と同じ向き", torch.allclose(ys, y, rtol=1e-3, atol=1e-5), f"{ys[0, :2].tolist()}")


def test_split_val():
    n = 20000
    seq = torch.arange(n) // 200  # 100 系列
    lang = (seq % 3 == 0)
    val = Q.split_val(lang, seq, 0.1, 0)
    vs = torch.unique(seq[val])
    check("val 分割: 系列単位", all(bool(val[seq == s].all()) for s in vs.tolist()))
    check("val 分割: 約 10%", 0.07 < val.float().mean().item() < 0.13, f"{val.float().mean().item():.3f}")
    check("val 分割: 日本語系列と他の両方を含む", bool(lang[val].any()) and bool((~lang[val]).any()))
    val2 = Q.split_val(lang, torch.full((n,), -1), 0.1, 0)
    check("val 分割: seq 無しは塊単位", 0.05 < val2.float().mean().item() < 0.15)


# --------------------------------------------------------------------------- 合成の層
class Synth:
    """小さな次元の合成 MoE 層。教師 = FP4(diag(a*) Q̃ diag(b*) + 雑音)、学生の初期値 = a*, b* を 15% ずらしたもの。"""

    def __init__(self, dim=256, inter=128, had=32, E=8, k=2, N=6000, cold=7, seed=0, noise=0.02, pert=0.15,
                 ja_subspace=False):
        g = torch.Generator().manual_seed(seed)
        self.dim, self.inter, self.had, self.E, self.k = dim, inter, had, E, k
        shapes = {"w1": (dim, inter), "w3": (dim, inter), "w2": (inter, dim)}
        self.pack, self.teach = {}, {}
        for e in range(E):
            pp, tt = {}, {}
            for p, (i, o) in shapes.items():
                inner = torch.randn(i, o, generator=g).half()
                qt = mock_get_weight_tensor(inner, torch.ones(i).half(), torch.ones(o).half(), had).float()
                a = torch.sign(torch.randn(i, generator=g)) * 0.017 * (0.5 + torch.rand(i, generator=g))
                if p == "w2":
                    a = a * 8  # down の入力 (h) は小さいので大きめ
                b = torch.sign(torch.randn(o, generator=g)) * (1 + 0.03 * torch.randn(o, generator=g))
                w_eff = a[:, None] * qt * b[None, :]
                w_eff = w_eff + noise * w_eff.std() * torch.randn(i, o, generator=g)
                tt[p] = fp4_quantize(w_eff.t().contiguous())  # 教師は [out, in]
                su0 = (a * (1 + pert * torch.randn(i, generator=g))).half()
                sv0 = (b * (1 + pert * torch.randn(o, generator=g))).half()
                pp[p] = {"trellis": inner, "suh": su0, "svh": sv0, "mcg": torch.tensor(-878968851, dtype=torch.int32)}
            self.pack[e], self.teach[e] = pp, tt
        x = torch.randn(N, dim, generator=g) * 15
        logits = torch.randn(N, E, generator=g)
        logits[:, cold] -= 6.0  # cold expert はほとんど選ばれない
        top = logits.topk(k, -1)
        w = torch.softmax(top.values, -1) * 1.5
        idx = top.indices.clone()
        idx[:60, k - 1] = cold  # cold expert は 60 トークンだけ (min_tokens=100 未満)
        seq = torch.arange(N) // 100
        lang = (seq % 10) < 3
        if ja_subspace:  # 日本語トークンは前半の次元、他は後半の次元が大きい (入力分布が違う)
            h = dim // 2
            x[lang, :h] *= 3.0
            x[~lang, h:] *= 3.0
        kp = torch.where(lang, 0.5, 0.25)
        self.data = {"x": x.to(torch.bfloat16), "indices": idx, "weights": w, "lang": lang, "seq": seq,
                     "imp": 1.0 / kp, "stream": ["student:synthetic"]}
        self.cold = cold


class MockTeacher:
    weights = staticmethod(Q.TeacherFP4.weights)

    def __init__(self, s):
        self.s = s

    def raw(self, layer, e):
        return {p: self.s.teach[e][p] for p in Q.PARTS}


class MockPack:
    def __init__(self, s):
        self.s = s
        had = s.had
        self.qtilde_fn = lambda tr, mcg, dev: mock_get_weight_tensor(
            tr, torch.ones(tr.shape[0]).half(), torch.ones(tr.shape[1]).half(), had)

    def raw(self, layer, e):
        return {p: dict(self.s.pack[e][p]) for p in Q.PARTS}


def make_cfg(**kw):
    c = Q.Cfg()
    base = dict(loss="expert", ja_weight=2.0, lr=5e-3, l2=0.01, epochs=4, batch=256, layer_batch=1024, eval_batch=4096,
                min_steps=60, max_steps=300, eval_every=10, patience=4, es_tol=1e-4, val_frac=0.1, min_tokens=100,
                min_val=16, ja_expert_ratio=4.0, min_val_ja=32, n_experts=8, limit=10.0, teacher_act_quant=True,
                layer_report=True, seed=0, verbose=False, init_diff="", cdt=torch.float32)
    base.update(kw)
    for k, v in base.items():
        setattr(c, k, v)
    return c


def test_train_expert():
    s = Synth()
    t0 = time.time()
    new, rep = Q.train_layer(0, s.data, MockTeacher(s), MockPack(s), make_cfg(), "cpu")
    dt = time.time() - t0
    ex = rep["experts"]
    trained = [e for e, v in ex.items() if v["status"] in ("accepted", "rejected")]
    drops = {e: ex[e]["final_val_fp16"] / ex[e]["init_val"] for e in trained}
    check("expert mode: cold expert は学習しない", ex[s.cold]["status"] == "cold" and
          not any(f".experts.{s.cold}." in k for k in new), f"n_train={ex[s.cold]['n_train']}")
    check("expert mode: 全 warm expert で検証損失が下がる", len(trained) == 7 and all(v < 0.7 for v in drops.values()),
          "final/init = " + ", ".join(f"e{e}:{v:.3f}" for e, v in sorted(drops.items())))
    lv = rep["layer_val"]
    check("expert mode: MoE 出力 (top-k 和) の検証誤差も下がる", lv["weighted"]["qad"] < 0.7 * lv["weighted"]["v1"],
          f"v1 {lv['weighted']['v1']:.4f} -> qad {lv['weighted']['qad']:.4f}; ja {lv['ja']['v1']:.4f}->{lv['ja']['qad']:.4f}")
    keys_ok = all(k.startswith("layers.0.ffn.experts.") and k.rsplit(".", 2)[1] in Q.PARTS and k.endswith((".suh", ".svh"))
                  and v.dtype == torch.float16 for k, v in new.items())
    check("expert mode: 出力キー名と fp16", keys_ok and len(new) == 7 * 6, f"{len(new)} tensors")
    e0 = trained[0]
    su_new = new[f"layers.0.ffn.experts.{e0}.w1.suh"].float()
    su_v1 = s.pack[e0]["w1"]["suh"].float()
    check("expert mode: 符号は保たれる", bool((torch.sign(su_new) == torch.sign(su_v1)).all()))
    print(f"    expert mode {dt:.1f}s, steps " + ", ".join(f"e{e}:{ex[e]['steps']}" for e in trained))
    return s, new, rep


def test_train_layer(s=None, new_expert=None):
    s = s or Synth()
    with tempfile.TemporaryDirectory() as d:
        init = ""
        if new_expert is not None:
            init = os.path.join(d, "init.safetensors")
            Q.st_save(init, new_expert)
        t0 = time.time()
        new, rep = Q.train_layer(0, s.data, MockTeacher(s), MockPack(s),
                                 make_cfg(loss="layer", epochs=30, eval_every=5, patience=4, init_diff=init), "cpu")
        dt = time.time() - t0
    lv = rep["layer_val"]
    tag = "layer mode (expert mode の結果から)" if new_expert is not None else "layer mode (v1 から)"
    check(f"{tag}: 検証の MoE 出力誤差が下がる", rep["status"] == "accepted" and lv["weighted"]["qad"] < lv["weighted"]["v1"],
          f"v1 {lv['weighted']['v1']:.4f} -> {lv['weighted']['qad']:.4f} (ja {lv['ja']['v1']:.4f}->{lv['ja']['qad']:.4f}), "
          f"steps {rep['steps']}, {dt:.1f}s")
    check(f"{tag}: cold expert は凍結", not any(f".experts.{s.cold}." in k for k in new) and rep["trainable"] == 7)
    return rep


def test_ja_weight():
    """日本語重みを 1 -> 2 にすると、収束後の日本語トークンの誤差が下がる (日本語と他で入力分布が違う合成データ)。
    suh/svh は 22K 自由度しかないので差は小さい (r3_qad §4.2 の通り、日本語の主な手段はデータに入れること)。"""
    s = Synth(noise=0.25, pert=0.05, seed=3, ja_subspace=True, N=12000)
    out = {}
    for jw in (1.0, 2.0):
        _, rep = Q.train_layer(0, s.data, MockTeacher(s), MockPack(s),
                               make_cfg(ja_weight=jw, l2=0.0, min_steps=1500, max_steps=1500, patience=1000, lr=2e-3), "cpu")
        out[jw] = rep["layer_val"]
    j1, j2 = out[1.0]["ja"]["qad"], out[2.0]["ja"]["qad"]
    o1, o2 = out[1.0]["other"]["qad"], out[2.0]["other"]["qad"]
    check("ja_weight 1->2: 日本語の誤差が下がる (収束後)", j2 < j1 and (j2 / o2) < (j1 / o1),
          f"ja {j1:.4f}->{j2:.4f}, other {o1:.4f}->{o2:.4f} (v1: ja {out[1.0]['ja']['v1']:.4f}, other {out[1.0]['other']['v1']:.4f})")


def _write_fake_pack(d, layers=(0, 1), E=3, dim=64, inter=32):
    os.makedirs(d)
    wm = {}
    rng = torch.Generator().manual_seed(1)
    for L in layers:
        name = f"model-{L + 3:05d}-of-00048.safetensors"
        t = {f"layers.{L}.attn_norm.weight": torch.randn(dim, generator=rng).to(torch.bfloat16)}
        for e in range(E):
            for p, (i, o) in {"w1": (dim, inter), "w3": (dim, inter), "w2": (inter, dim)}.items():
                t[Q.ename(L, e, p, "trellis")] = torch.randint(-2**15, 2**15 - 1, (i // 16, o // 16, 32), dtype=torch.int16, generator=rng)
                t[Q.ename(L, e, p, "suh")] = torch.randn(i, generator=rng).half()
                t[Q.ename(L, e, p, "svh")] = torch.randn(o, generator=rng).half()
                t[Q.ename(L, e, p, "mcg")] = torch.tensor(-878968851, dtype=torch.int32)
        Q.st_save(os.path.join(d, name), t)
        wm.update({k: name for k in t})
    Q.st_save(os.path.join(d, "model-00001-of-00048.safetensors"), {"embed.weight": torch.randn(8, dim).to(torch.bfloat16)})
    wm["embed.weight"] = "model-00001-of-00048.safetensors"
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump({"quantization_config": {"out_scales": "never"}}, open(os.path.join(d, "config.json"), "w"))
    os.makedirs(os.path.join(d, "inference"))
    open(os.path.join(d, "inference", "config.json"), "w").write("{}")


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_apply():
    TI = Q.stream_ref_tensor_index()
    with tempfile.TemporaryDirectory() as root:
        pack = os.path.join(root, "v1")
        _write_fake_pack(pack)
        sums = {f: _sha(os.path.join(pack, f)) for f in os.listdir(pack) if f.endswith(".safetensors")}
        diff = os.path.join(root, "diff")
        os.makedirs(diff)
        new = {Q.ename(0, 1, "w1", "suh"): torch.full((64,), 0.25).half(),
               Q.ename(0, 1, "w2", "svh"): torch.full((64,), -0.5).half(),
               Q.ename(0, 2, "w3", "svh"): torch.full((32,), 2.0).half()}
        Q.st_save(os.path.join(diff, "qad_l2_L00.safetensors"), new)
        # shards
        out = os.path.join(root, "v1-qad")
        Q.apply_pack(pack, diff, out, "shards")
        sh = "model-00003-of-00048.safetensors"
        src, dst = os.path.join(pack, sh), os.path.join(out, sh)
        check("apply shards: 元の v1 shard は不変", all(_sha(os.path.join(pack, f)) == h for f, h in sums.items()))
        check("apply shards: 書き換えた shard は別 inode", not os.path.samefile(src, dst))
        check("apply shards: 他の shard / 設定は hardlink",
              os.path.samefile(os.path.join(pack, "model-00004-of-00048.safetensors"), os.path.join(out, "model-00004-of-00048.safetensors"))
              and os.path.samefile(os.path.join(pack, "config.json"), os.path.join(out, "config.json"))
              and os.path.samefile(os.path.join(pack, "model.safetensors.index.json"), os.path.join(out, "model.safetensors.index.json")))
        a, _ = Q.st_load(src)
        b, _ = Q.st_load(dst)
        same = all(torch.equal(a[k], b[k]) for k in a if k not in new)
        patched = all(torch.equal(b[k], v) for k, v in new.items())
        check("apply shards: suh/svh だけ差し替わり他は同一", same and patched and st_hdr_same(src, dst))
        try:
            Q.apply_pack(pack, diff, out, "shards")
            check("apply: 空でない出力先は拒否", False)
        except SystemExit:
            check("apply: 空でない出力先は拒否", True)
        # overlay (ハーネス用)
        ov = os.path.join(root, "v1-ov")
        Q.apply_pack(pack, diff, ov, "overlay")
        ti = TI(ov)
        got = ti.load(Q.ename(0, 1, "w1", "suh"))
        keep = ti.load(Q.ename(0, 1, "w3", "suh"))
        check("apply overlay: stream_ref.TensorIndex が差分の値を読む", torch.equal(got, new[Q.ename(0, 1, "w1", "suh")])
              and torch.equal(keep, a[Q.ename(0, 1, "w3", "suh")]))
        check("apply overlay: 元パック不変・shard は hardlink",
              all(_sha(os.path.join(pack, f)) == h for f, h in sums.items()) and os.path.samefile(src, os.path.join(ov, sh)))
        check("apply overlay: index.json は新しいファイル",
              not os.path.samefile(os.path.join(pack, "model.safetensors.index.json"), os.path.join(ov, "model.safetensors.index.json")))


def st_hdr_same(a, b):
    return Q.st_header(a) == Q.st_header(b)


def test_capture_loader():
    """capture_moe.py の生形式 (layer_LL/<field>.bin + meta.json + state.npz) を load_capture で読めること。"""
    import capture_moe as C
    with tempfile.TemporaryDirectory() as d:
        dim, k, n = 64, 6, 50
        meta = {"schema": C.SCHEMA, "dim": dim, "topk": k, "n_layers": 1, "stream": "student:v1"}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
        np.savez(os.path.join(d, "state.npz"), rows=np.array([n]), json=np.frombuffer(b"{}", np.uint8))
        os.makedirs(os.path.join(d, "layer_00"))
        x = torch.randn(n, dim).to(torch.bfloat16)
        flags = np.zeros(n, np.uint8)
        flags[:10] |= C.F_JA
        flags[45:] |= C.F_DUP
        kp = np.where(flags & C.F_JA, 0.5, 0.25).astype(np.float32)
        kp[45:] = 0
        cols = {"x": x.view(torch.int16).numpy(), "idx": np.tile(np.arange(k, dtype=np.int16), (n, 1)),
                "w": np.full((n, k), 0.1, np.float32), "seq": (np.arange(n) // 10).astype(np.int32),
                "pos": np.arange(n, dtype=np.int32), "flags": flags, "dom": np.zeros(n, np.uint8), "kp": kp}
        for name, arr in cols.items():
            arr.tofile(os.path.join(d, "layer_00", f"{name}.bin"))
        t = Q.load_capture([d], 0)
        check("capture: F_DUP 行を捨てる", t["x"].shape == (45, dim))
        check("capture: x は bf16 のまま", t["x"].dtype == torch.bfloat16 and torch.equal(t["x"], x[:45]))
        check("capture: F_JA -> lang、1/kp -> imp", int(t["lang"].sum()) == 10 and float(t["imp"][0]) == 2.0
              and float(t["imp"][20]) == 4.0)
        check("capture: stream を報告", t["stream"] == ["student:v1"])


def test_estimate():
    e = Q.estimate(600000)
    print("    estimate(600K, 一様):", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in e.items()
                                                  if not isinstance(v, list)}, ensure_ascii=False),
          "compute_s", [round(v, 1) for v in e["compute_s"]])
    p = "/t/agent24_teacher.pt"
    if os.path.exists(p):
        c = torch.load(p, weights_only=False)["counts"]
        for L in (0, 20, 39):
            arr = torch.as_tensor(c[L], dtype=torch.float64)
            for n in (196608, 600000):
                cnt = (arr / arr.sum() * n * 6).tolist()
                ee = Q.estimate(n, counts=cnt)
                el = Q.estimate(n, counts=cnt, loss="layer")
                print(f"    L{L} N={n}: expert {ee['total_TFLOP']:.0f} TFLOP ({ee['compute_s'][0]:.0f}-{ee['compute_s'][1]:.0f}s) "
                      f"mem {ee['gpu_mem_GiB']:.1f}GiB hot {ee['hot_expert_pairs']:.0f} | layer {el['total_TFLOP']:.0f} TFLOP "
                      f"({el['compute_s'][0]:.0f}-{el['compute_s'][1]:.0f}s) mem {el['gpu_mem_GiB']:.1f}GiB | io {ee['io_GB']:.1f}GB")
    check("estimate 実行", e["total_TFLOP"] > 0)


def test_cli_parse():
    check("parse_layers", Q.parse_layers("0-2,5") == [0, 1, 2, 5])
    try:
        Q.main(["estimate", "--tokens", "1000"])
        check("CLI estimate", True)
    except Exception as ex:  # noqa: BLE001
        check("CLI estimate", False, repr(ex))


def main():
    only = sys.argv[1:]
    tests = [test_decomp, test_fp4, test_act_quant, test_clamp, test_split_val, test_capture_loader, test_apply,
             test_cli_parse, test_estimate]
    for t in tests:
        if only and not any(o in t.__name__ for o in only):
            continue
        print(f"[{t.__name__}]", flush=True)
        t()
    if not only or any(o in "train" for o in only):
        print("[test_train_expert]", flush=True)
        s, new, _ = test_train_expert()
        print("[test_train_layer]", flush=True)
        test_train_layer(s)
        test_train_layer(s, new)
        print("[test_ja_weight]", flush=True)
        test_ja_weight()
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} passed", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
