"""capture_moe.py の CPU テスト (GPU・CUDA 拡張を使わない)。

  sudo docker run --rm --network none --cpus 8 --memory 16g -e CUDA_VISIBLE_DEVICES= \
    -v $DSV41/ref:/ref:ro -v $DSV41/harness:/h:ro \
    -v $MODELS/DeepSeek-V4.1-Flash-1017:/model:ro -v $DSV41/work/teacher:/t:ro \
    -v <scratch>:/s --entrypoint bash dsv41-quant:151 -c 'cd /h && python3 test_capture_moe.py /s/capture_test'

1. 言語タグ (UTF-8 が途中で切れるバイト単位トークナイザの模擬)
2. MoE.forward / Gate.forward の包み: model.py の MoE (小さい ModelArgs) と stream_ref.moe_forward で、
   出力が変わらないこと、記録した x / indices / weights が gate の値と一致すること
3. 小さな偽 Transformer (model.py の MoE と RMSNorm を 4 層) で end-to-end:
   保存した x の再計算一致、重複接頭辞の除外、日本語の保持率、専門家の最低行数、
   中断 (層の途中で例外 + 書きかけのゴミ) → 再開で無中断と完全一致、設定違いの再開拒否、
   part 分割の和集合 = 一括、pack → safetensors で同じ値
4. 実トークナイザで ja16 / agent24 の plan (あれば)
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import json
import os
import shutil
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture_moe as C  # noqa: E402
import stream_ref as SR  # noqa: E402

M = SR.M
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/capture_test"
shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(ROOT)
torch.manual_seed(0)
OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print("  ok:", msg, flush=True)


# --------------------------------------------------------------------------- 偽トークナイザ (バイト単位)
N_SPECIAL = 4
WORDS = [b" the", b" def", b" return", b"(", b")", b"\n", b" =", b" x", b" code", b" error"]


def tok_encode(text, special=None):
    out = [] if special is None else [special]
    b = text.encode()
    i = 0
    while i < len(b):
        for wi, w in enumerate(WORDS):
            if b.startswith(w, i):
                out.append(N_SPECIAL + 256 + wi)
                i += len(w)
                break
        else:
            out.append(N_SPECIAL + b[i])
            i += 1
    return out


def tok_decode(ids):
    bs = b""
    for t in ids:
        if t < N_SPECIAL:
            bs += b"<s>"
        elif t < N_SPECIAL + 256:
            bs += bytes([t - N_SPECIAL])
        else:
            bs += WORDS[t - N_SPECIAL - 256]
    return bs.decode("utf-8", errors="replace")


VOCAB = N_SPECIAL + 256 + len(WORDS)
SPECIAL = set(range(N_SPECIAL))

# --------------------------------------------------------------------------- 1. タグ
print("[1] 言語タグ", flush=True)
txt_en = "the quick brown fox returns an error code x = 1\n" * 2
txt_ja = "日本語の文章です。量子化しても性能を保ちたいと考えています。"
txt_zh = "我们希望量化之后模型性能保持不变这是中文句子测试"
ids = tok_encode(txt_en, 0) + tok_encode(txt_ja) + tok_encode(txt_en) + tok_encode(txt_zh)
fl = C.tag_tokens(ids, tok_decode, SPECIAL, score_from=10, window=30)  # バイト単位なので 30 トークン ≈ 10 文字
n_en = len(tok_encode(txt_en)) + 1
n_ja = len(tok_encode(txt_ja))
ja_part = fl[n_en: n_en + n_ja]
check((fl[0] & C.F_SPECIAL) and not (fl[1:] & C.F_SPECIAL).any(), "特殊トークンだけに special")
check(((ja_part & C.F_JA) > 0).mean() > 0.8, f"日本語区間の ja 率 {((ja_part & C.F_JA) > 0).mean():.2f} > 0.8 (境界の窓で落ちる分を除く)")
check(((ja_part[20:-20] & C.F_JA) > 0).all(), "日本語区間の内側は全部 ja")
check(((ja_part & C.F_JA) > 0)[(ja_part & C.F_KANA) > 0].all(), "日本語区間の仮名トークンは境界でも ja")
check(((ja_part & C.F_KANA) > 0).sum() >= 3 * 15, "仮名 (3 バイトずつ分割) が run 単位で付く")
check(((fl[n_en + n_ja + n_en:] & C.F_JA) > 0).mean() < 0.1, "中国語区間は ja でない")
check(((fl[-30:] & C.F_ZH) > 0).mean() > 0.9, "中国語区間は zh")
check(((fl[5:n_en - 31] & C.F_JA) == 0).all(), "英語区間 (日本語から窓幅以上離れた所) は ja でない")
check(((fl[:10] & C.F_SCORED) == 0).all() and ((fl[10:] & C.F_SCORED) > 0).all(), "scored = pos >= score_from")
f_ja = np.full(10, C.F_JA, np.uint8)
f_mix = np.array([C.F_JA] * 2 + [0] * 8, np.uint8)  # ja 20% (< 25%)
check(C.infer_domain({"domain": "code"}, "ja", f_ja) == "code" and C.infer_domain({}, "zh-multi", f_ja) == "zh-multi"
      and C.infer_domain({"kana_ratio": 0.2}, None, f_mix) == "ja" and C.infer_domain({"tools": [1]}, None, f_mix) == "agent"
      and C.infer_domain({"tools": [1]}, None, f_ja) == "ja" and C.infer_domain({}, None, f_mix) == "en",
      "domain の推定順 (jsonl の domain > --input の ::domain > kana_ratio > 窓の ja 率 25% > tools > en)")

# --------------------------------------------------------------------------- 2. 包みの単体
print("[2] MoE.forward / Gate.forward の包み", flush=True)
DIM, INTER, E, K, L = 64, 32, 16, 6, 4
torch.set_default_dtype(torch.bfloat16)
M.default_dtype = torch.bfloat16
margs = M.ModelArgs(dim=DIM, moe_inter_dim=INTER, n_routed_experts=E, n_activated_experts=K, n_layers=L,
                    swiglu_limit=10.0, route_scale=1.5, score_func="sqrtsoftplus", expert_dtype=None, dtype="bf16")


def make_moe(l):
    m = M.MoE(l, margs)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "gate.weight" in name:
                p.normal_(0, 0.3)
            elif "gate.bias" in name:
                p.normal_(0, 0.05)
            else:
                p.normal_(0, 0.6)  # clamp(±10) にも時々当たる大きさ
    m.swiglu_limit = margs.swiglu_limit
    return m


moes = [make_moe(l) for l in range(L)]
norms = [M.RMSNorm(DIM, 1e-6) for _ in range(L)]


class FakeStore:
    def __init__(self):
        self.counts = {}
        self.stats = {}

    def get(self, layer, used):
        m = moes[layer]
        return {e: ("bf16", m.experts[e].w1.weight.t().contiguous(), m.experts[e].w3.weight.t().contiguous(),
                    m.experts[e].w2.weight.t().contiguous()) for e in used}


SR.STORE = FakeStore()
ORIG_MOE = M.MoE.forward  # model.py の元の forward (stream_ref の import では差し替わらない)
x = torch.randn(1, 97, DIM).to(torch.bfloat16) * 3
with torch.inference_mode():
    y_ref = moes[0](x)
    M.MoE.forward = SR.moe_forward  # stream_ref.build と同じ差し替え
    y_sr = moes[0](x)
    w_g, i_g = moes[0].gate(x.view(-1, DIM))


class RecCap(C.Capturer):
    def __init__(self):
        self.cur = {"on": True}
        self.slot = None
        self.n_layers = L
        self.rec = []

    def on_layer(self, l, x2d, w, i):
        self.rec.append((l, x2d.clone(), w.clone(), i.clone()))


rc = RecCap()
rc.install(M)
with torch.inference_mode():
    y_cap = moes[0](x)
    rc.cur = None
    y_off = moes[0](x)
rc.uninstall()
check(M.MoE.forward is SR.moe_forward and M.Gate.forward.__name__ == "forward", "uninstall で元に戻る")
check((y_ref.float() - y_sr.float()).abs().max().item() <= 0.05 * y_ref.float().abs().max().item(),
      f"stream_ref.moe_forward ≈ model.py MoE.forward (max diff {(y_ref.float() - y_sr.float()).abs().max().item():.3g})")
check(torch.equal(y_cap, y_sr) and torch.equal(y_off, y_sr), "包んでも出力はビット一致")
l0, xr, wr, ir = rc.rec[0]
check(len(rc.rec) == 1 and l0 == 0, "1 回の MoE で 1 回記録")
check(torch.equal(xr, x.view(-1, DIM)), "記録した x = MoE 入力")
check(torch.equal(ir, i_g) and torch.equal(wr, w_g), "記録した indices / weights = gate の出力")
check(wr.dtype == torch.float32 and ir.shape == (97, K), "weights は fp32 [n, topk]")

# --------------------------------------------------------------------------- 3. end-to-end
print("[3] 偽 Transformer で end-to-end", flush=True)
emb = torch.randn(VOCAB, DIM)


def fake_forward(ids, rec=None, crash_at=None):
    n = len(ids)
    h = emb[torch.tensor(ids)].float()
    for l in range(L):
        mix = torch.cumsum(h, 0) / torch.arange(1, n + 1)[:, None]  # 因果的に前の位置を混ぜる
        xx = norms[l]((h + mix).to(torch.bfloat16))
        if rec is not None:
            rec.append(xx.clone())
        y = moes[l](xx.view(1, n, DIM)).view(n, DIM)
        h = h + y.float()
        if crash_at is not None and crash_at == l:
            raise RuntimeError("模擬の中断")
    return h


# 合成の校正データ: 共通の system prompt + 日本語 / 英語 / コード
sysp = "You are a helpful agent. Follow the tools schema strictly and return code.\n"
texts = []
rng = np.random.default_rng(0)
ja_s = ["量子化の誤差を減らすために、日本語のデータを多めに入れます。", "この関数はエラーを返すので修正してください。",
        "ログを確認したところ、メモリが足りないようです。", "次の手順で実行してください。まず環境を整えます。"]
en_s = ["the code returns an error when x = 0\n", "def f(x): return x\n", "please fix the error in the code\n"]
for q in range(14):
    body = "".join(rng.choice(ja_s if q % 2 == 0 else en_s, size=6))
    texts.append((sysp + body, "ja" if q % 2 == 0 else None))
inp = os.path.join(ROOT, "calib.jsonl")
with open(inp, "w") as f:
    for q, (t, dom) in enumerate(texts):
        ids_ = tok_encode(t, 0)
        r = {"ids": ids_, "score_from": len(tok_encode(sysp, 0)), "src": f"syn{q}", "off": q}
        if dom:
            r["domain"] = dom
        f.write(json.dumps(r) + "\n")
n_tok = sum(len(tok_encode(t, 0)) for t, _ in texts)
print(f"  合成 {len(texts)} 系列 {n_tok} tok", flush=True)


def parse(extra):
    ap = C.argparse.ArgumentParser()
    C.add_common(ap)
    return ap.parse_args(["--input", inp, "--dim", str(DIM), "--topk", str(K), "--n-experts", str(E),
                          "--n-layers", str(L), "--window", "30"] + extra)


def do_run(out, extra, crash=None, garbage=False):
    a = parse(["--out", out] + extra)
    plan = C.make_plan(a, decode=tok_decode, special_ids=SPECIAL)
    meta = C.build_meta(a, plan, "/fake")
    st = C.CaptureStore(out, meta, fsync=False)
    cap = C.Capturer(st, plan)
    cap.install(M)
    calls = {"n": 0}

    def fwd(ids_):
        calls["n"] += 1
        with torch.inference_mode():
            fake_forward(ids_, crash_at=(2 if crash is not None and calls["n"] == crash else None))

    try:
        C.run_capture(st, plan, cap, fwd, log=lambda *a_, **k_: None)
    finally:
        cap.uninstall()
    return plan, st


ROWS_TGT = 600
common = ["--rows-per-layer", str(ROWS_TGT), "--expert-floor", "40", "--ja-boost", "2.5"]
A = os.path.join(ROOT, "A")
planA, stA = do_run(A, common)
smp = planA["sampling"]
print("  sampling:", {k: smp[k] for k in ("p_base", "p_ja", "expected_rows_per_layer", "max_rows_per_layer_this_part")})
print("  rows/layer:", stA.rows.tolist(), "tokens:", planA["tokens"])
check(len(stA.info["done"]) == len(texts), "全系列を記録")
# 3a. 保存した x = 再計算した MoE 入力
seq_by_gid = {s["gid"]: s for s in planA["seqs"]}
recx = {}
for s in planA["seqs"]:
    r = []
    with torch.inference_mode():
        fake_forward(s["ids"], rec=r)
    recx[s["gid"]] = r
bad = 0
for l in range(L):
    d = C.load_layer(A, l)
    for j in range(len(d["seq"])):
        g, p = int(d["seq"][j]), int(d["pos"][j])
        if not torch.equal(d["x"][j], recx[g][l][p]):
            bad += 1
    with torch.inference_mode():
        for g in set(d["seq"].tolist()):
            m = d["seq"] == g
            w_, i_ = moes[l].gate(recx[g][l])  # 記録時と同じく系列全体で gate を計算してから選ぶ
            w_, i_ = w_[d["pos"][m].long()], i_[d["pos"][m].long()]
            if not (torch.equal(i_.to(torch.int16), d["idx"][m]) and torch.equal(w_, d["w"][m])):
                bad += 1
check(bad == 0, "保存した x / idx / w が再計算とビット一致 (全層・全行)")
# 3b. 重複接頭辞
first = min(planA["seqs"], key=lambda s: s["gid"])
plen = len(tok_encode(sysp, 0)) - 1
dup_rows = 0
for l in range(L):
    d = C.load_layer(A, l, as_torch=False)
    dup_rows += int(((d["seq"] != first["gid"]) & (d["pos"] < plen)).sum())
check(dup_rows == 0, "共通 system prompt の位置は最初の系列からしか保存しない")
other = [s for s in planA["seqs"] if s["gid"] != first["gid"]][0]
check(torch.equal(recx[other["gid"]][3][:plen], recx[first["gid"]][3][:plen]), "同じ接頭辞なら x も同じ (除外の前提)")
# 3c. 保持率と flag
d0 = C.load_layer(A, 0, as_torch=False)
ja_rows = (d0["flags"] & C.F_JA) > 0
nf = (d0["flags"] & C.F_FLOOR) == 0
check(np.allclose(d0["kp"][ja_rows], smp["p_ja"]) and np.allclose(d0["kp"][~ja_rows], smp["p_base"]),
      "kp = 日本語は p_ja、他は p_base")
live_ja = sum(int((((s["flags"] & C.F_JA) > 0) & ((s["flags"] & C.F_DUP) == 0)).sum()) for s in planA["seqs"])
live_ot = sum(int((((s["flags"] & C.F_JA) == 0) & ((s["flags"] & C.F_DUP) == 0)).sum()) for s in planA["seqs"])
r_ja = (ja_rows & nf).sum() / live_ja
r_ot = ((~ja_rows) & nf).sum() / live_ot
check(abs(r_ja - smp["p_ja"]) < 0.08 and abs(r_ot - smp["p_base"]) < 0.08,
      f"実測の保持率 ja {r_ja:.3f} (期待 {smp['p_ja']:.3f}) / 他 {r_ot:.3f} (期待 {smp['p_base']:.3f})")
check(abs(stA.rows.mean() - smp["expected_rows_per_layer"]) < 0.25 * smp["expected_rows_per_layer"] + 40 * E / K,
      f"1 層の行数 {stA.rows.mean():.0f} ≈ 期待 {smp['expected_rows_per_layer']:.0f} (+floor 分)")
# 3d. 専門家の最低行数
avail = stA.stats["cnt.all"]
check((stA.kept >= np.minimum(40, avail)).all(), "kept[l,e] >= min(floor, その専門家に routing された行数)")
kept_re = np.zeros((L, E), np.int64)
for l in range(L):
    d = C.load_layer(A, l, as_torch=False)
    kept_re[l] = np.bincount(d["idx"].astype(np.int64).ravel(), minlength=E)
check((kept_re == stA.kept).all(), "state の kept = 保存行から数え直した値")
check(int(avail.sum()) == (len(texts) * 0 + sum(int(((s["flags"] & C.F_DUP) == 0).sum()) for s in planA["seqs"])) * K * L,
      "routing 統計は重複を除く全トークン × topk × 層")
# 3e. 中断 → 再開
B = os.path.join(ROOT, "B")
try:
    do_run(B, common, crash=5)
    raise AssertionError("中断しなかった")
except RuntimeError as e:
    check("模擬" in str(e), "5 本目の層 2 で中断")
stB = C.CaptureStore(B, json.load(open(os.path.join(B, "meta.json"))), fsync=False)
check(len(stB.info["done"]) == 4, "中断時点のコミットは 4 系列")
for l in (0, 3):
    with open(stB.fpath(l, "x"), "ab") as f:
        f.write(b"\x7f" * 1234)  # 書きかけのゴミ
    with open(stB.fpath(l, "flags"), "ab") as f:
        f.write(b"\x01" * 7)
planB, stB = do_run(B, common)
same = True
for l in range(L):
    for name, *_ in C.FIELDS:
        if open(stA.fpath(l, name), "rb").read() != open(stB.fpath(l, name), "rb").read():
            same = False
check(same, "中断 + ゴミ → 再開で、全層・全 field のファイルが無中断とバイト一致")
check((stA.rows == stB.rows).all() and (stA.kept == stB.kept).all()
      and all(np.array_equal(stA.stats[k], stB.stats[k]) for k in stA.stats), "state (rows, kept, 統計) も一致")
try:
    do_run(B, common + ["--seed", "7"])
    raise AssertionError("設定違いで再開できてしまった")
except SystemExit as e:
    check("再開不可" in str(e), "seed を変えた再開は拒否")
# 3f. part 分割 (floor なし) の和集合 = 一括
full = os.path.join(ROOT, "full")
nofloor = ["--rows-per-layer", str(ROWS_TGT), "--expert-floor", "0", "--max-rows-per-layer", "100000"]
do_run(full, nofloor)
P = [os.path.join(ROOT, f"part{i}") for i in range(2)]
for i in range(2):
    do_run(P[i], nofloor + ["--part", f"{i}/2"])
ok = True
for l in range(L):
    a_ = C.load_layer(full, l, as_torch=False)
    b_ = C.load_layer(P, l, as_torch=False)
    ka = np.lexsort((a_["pos"], a_["seq"]))
    kb = np.lexsort((b_["pos"], b_["seq"]))
    for name, *_ in C.FIELDS:
        if not np.array_equal(a_[name][ka], b_[name][kb]):
            ok = False
check(ok, "part 0/2 + 1/2 の連結 = 一括の記録 (行の並べ替え後に全 field 一致)")
# 3g. pack
Bp = os.path.join(ROOT, "B_packed")
shutil.copytree(B, Bp)
C.pack(Bp, rm_raw=True)
from safetensors import safe_open  # noqa: E402

ok = True
for l in range(L):
    raw = C.load_layer(B, l)
    pk = C.load_layer(Bp, l)
    with safe_open(os.path.join(Bp, f"layer_{l:02d}.safetensors"), "pt") as f:
        st_x = f.get_tensor("x")
        st_w = f.get_tensor("w")
        md = f.metadata()
    if not (torch.equal(raw["x"], pk["x"]) and torch.equal(raw["x"], st_x) and torch.equal(raw["w"], st_w)
            and st_x.dtype == torch.bfloat16 and md["schema"] == C.SCHEMA):
        ok = False
check(ok, "pack 後の safetensors (safetensors ライブラリで読む) = 生ファイル")
check(not os.path.isdir(os.path.join(Bp, "layer_00")), "--rm-raw で生ファイルを消す")
try:
    C.CaptureStore(Bp, json.load(open(os.path.join(Bp, "meta.json"))), fsync=False)
    raise AssertionError("pack 済みに追記できてしまった")
except SystemExit:
    check(True, "pack 済みへの追記は拒否")

# 3h. verify (gate の再計算照合)
vr = [C.verify_rows(C.load_layer(A, l), moes[l].gate, 300) for l in range(L)]
check(all(r["idx_set_agree"] == 1.0 and r["w_maxdiff"] < 1e-5 for r in vr), f"verify_rows: 一致 ({vr[0]})")
bad = C.load_layer(A, 1)
bad["x"] = bad["x"].flip(0)  # 行をずらすと不一致になること
rb_ = C.verify_rows(bad, moes[1].gate, 300)
check(rb_["idx_set_agree"] < 0.5, f"verify_rows: 行がずれると検出 (agree {rb_['idx_set_agree']:.2f})")

# --------------------------------------------------------------------------- 4. 実トークナイザ
print("[4] 実トークナイザ + 保存量の表", flush=True)
rb = C.row_bytes(5120, 6)
check(rb == 10290, f"1 行 {rb} B (x 10240 + idx 12 + w 24 + seq/pos/flags/dom/kp 14)")
for name, (rows, floor) in C.PRESETS.items():
    print(f"  preset {name}: {rows} 行/層 × 40 層 = {rows * 40 * rb / 1e9:.1f} GB (floor {floor}、上限 ×1.15 で "
          f"{rows * 1.15 * 40 * rb / 1e9:.1f} GB + floor 枠 {384 * floor // 6 * 40 * rb / 1e9:.1f} GB)")
if os.path.exists("/model/tokenizer.json") and os.path.exists("/t/ja16.jsonl"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/model")
    for files in (["/t/ja16.jsonl"], ["/t/agent24.jsonl"], ["/t/ja16.jsonl", "/t/agent24.jsonl"]):
        ap = C.argparse.ArgumentParser()
        C.add_common(ap)
        a = ap.parse_args(sum((["--input", f] for f in files), []) + ["--out", ROOT + "/real", "--preset", "64k"])
        plan = C.make_plan(a, tok)
        sm = C.plan_summary(plan, 5120, 6, 40, ROOT)
        fl_all = np.concatenate([s["flags"] for s in plan["seqs"]])
        live = (fl_all & C.F_DUP) == 0
        sc = live & ((fl_all & C.F_SCORED) > 0)
        print(f"  {[os.path.basename(f) for f in files]}: tokens {sm['tokens']}, ja 率 (重複除外) "
              f"{((fl_all & C.F_JA) > 0)[live].mean():.3f}, 応答側の ja 率 {((fl_all & C.F_JA) > 0)[sc].mean():.3f}, "
              f"by_domain {sm['by_domain_this_part']}, p_base {sm['sampling']['p_base']:.3f} p_ja {sm['sampling']['p_ja']:.3f}, "
              f"見込み {sm['expected_GB_this_part']} GB", flush=True)
    s = next(q for q in plan["seqs"] if ((q["flags"][q["score_from"]:] & C.F_JA) > 0).mean() > 0.5)
    own = (fl_all & C.F_KANA) > 0
    check((((fl_all & C.F_JA) > 0) | ~own)[live].mean() > 0.995,
          f"仮名を持つトークンは ja ({((fl_all & C.F_JA) > 0)[own & live].mean():.4f})")
    seg = [(tok.decode([t]), int(f)) for t, f in zip(s["ids"][s["score_from"]: s["score_from"] + 24],
                                                    s["flags"][s["score_from"]: s["score_from"] + 24])]
    print("  例:", [(t, "".join(n[0] for b, n in C.FLAG_NAMES.items() if f & b)) for t, f in seg])
    # 原重みの gate を CPU に作る (verify サブコマンドの読み込み部分)
    torch.set_default_dtype(torch.float32)
    g0 = C.load_real_gate(0, "/model")
    xg = torch.randn(64, 5120).to(torch.bfloat16)
    with torch.inference_mode():
        wg, ig = g0(xg)
    check(ig.shape == (64, 6) and wg.dtype == torch.float32 and torch.allclose(wg.sum(-1), torch.full((64,), 1.5), atol=1e-3)
          and g0.weight.dtype == torch.bfloat16 and float(g0.weight.float().abs().sum()) > 0,
          f"原重みの gate (層 0) を CPU で構築: weight {tuple(g0.weight.shape)} {g0.weight.dtype}、Σw = route_scale 1.5")
else:
    print("  /model か /t が無いので省略")

print(f"\nALL OK ({len(OK)} checks)")
