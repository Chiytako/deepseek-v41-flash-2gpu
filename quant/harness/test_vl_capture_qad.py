"""画像対応 (vl_rows.py、capture_moe.py の画像経路、qad_l2.py の画像重みと守り) の CPU テスト。GPU は使わない。

  sudo docker run --rm --network none --cpus 8 --memory 16g -e CUDA_VISIBLE_DEVICES= \
    -v $DSV41/ref:/ref:ro -v $DSV41/harness:/h:ro \
    -v $MODELS/DeepSeek-V4.1-Flash-1017:/model:ro -v <scratch>:/s \
    --entrypoint bash dsv41-quant:151 -c 'cd /h && python3 test_vl_capture_qad.py /s/vlcap_test'

1. vl_rows: 区間の並び = image_processor、build_row = 公式 generate.prepare_case (ids / token_types / patches)、
   回答の継ぎ目、相対パス / file://、区間の検査、salt と digest
2. capture_moe: 画像なしの入力は従来と同じ (row_bytes 10290、指紋・sampling に画像の鍵なし)。画像ありでは
   tt を保存、画像行の routing = bias_vl (verify)、画像トークンの kp = p_img、tok_img 統計、違う画像は重複扱いしない、
   forward に images / token_types が渡る、画像なしの part と混ぜて読める、pack 後も同じ、再開
3. qad_l2: load_capture の img、画像なしなら --img-weight を変えてもビット一致、画像の守り (tol で拒否)、
   検証の分割に画像系列が入る、layer_val に img、layer mode の守り
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capture_moe as C  # noqa: E402
import stream_ref as SR  # noqa: E402  (model.py を import する。差し替えはしない)
import vl_rows as V  # noqa: E402

M = SR.M
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vlcap_test"
shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(os.path.join(ROOT, "img"))
torch.manual_seed(0)
OK = []


def check(cond, msg, detail=""):
    if not cond:
        raise AssertionError(f"{msg} {detail}")
    OK.append(msg)
    print("  ok:", msg, detail, flush=True)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


def make_img(name, w, h, seed):
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (w, h), tuple(int(v) for v in rng.integers(0, 255, 3)))
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x0, y0 = int(rng.integers(0, w - 10)), int(rng.integers(0, h - 10))
        d.rectangle((x0, y0, x0 + int(rng.integers(5, w // 3)), y0 + int(rng.integers(5, h // 3))),
                    fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    p = os.path.join(ROOT, "img", name)
    im.save(p)
    return p


P = {"a.png": make_img("a.png", 640, 480, 1), "b.png": make_img("b.png", 300, 900, 2),
     "c.png": make_img("c.png", 1600, 500, 3)}

# --------------------------------------------------------------------------- 1. vl_rows
print("[1] vl_rows", flush=True)
from image_processor import image_token_types  # noqa: E402

check(all(V.span_types(h, w) == image_token_types(h, w).tolist() and V.span_len(h, w) == len(V.span_types(h, w))
          for h in (1, 3, 7) for w in (1, 4, 9)), "span_types = image_processor.image_token_types、長さ h(w+1)+2")

row = {"ids": [5, 6] + [129264] * V.span_len(2, 3) + [7, 8], "score_from": 2 + V.span_len(2, 3),
       "images": [{"path": "img/a.png", "start": 2, "grid": [2, 3], "sha256": "x"}]}
tt = V.row_token_types(row)
check(tt[:2].tolist() == [-1, -1] and tt[2] == 0 and tt[2 + V.span_len(2, 3) - 1] == 3 and (tt[-2:] == -1).all()
      and int((tt == 1).sum()) == 6 and int((tt == 2).sum()) == 2, "row_token_types: 区間を START/IMAGE/NL/END で埋める")
bad1 = dict(row, ids=row["ids"][:-2] + [129264, 8])
bad2 = dict(row, images=[dict(row["images"][0], grid=[2, 4])])
bad3 = dict(row, score_from=3)
check(raises(lambda: V.row_token_types(bad1), ValueError) and raises(lambda: V.row_token_types(bad2), ValueError)
      and raises(lambda: V.check_scored(bad3), ValueError), "区間外の image id / grid の食い違い / 採点区間内の画像を検出")
row_b = dict(row, images=[dict(row["images"][0], sha256="y")])
s1, s2 = V.image_salt(row), V.image_salt(row_b)
check(V.image_salt({"ids": [1, 2]}) is None and (s1[:2] == 0).all() and (s1[2:2 + 10] != 0).all() and not np.array_equal(s1, s2),
      "image_salt: テキストは 0、画像区間は画像ごとに違う値")
check(V.row_digest({"ids": [1, 2]}) == hashlib.sha256(np.asarray([1, 2], np.int64).tobytes()).hexdigest()
      and V.row_digest(row) != V.row_digest(row_b), "row_digest: テキスト行は従来の hid と同じ、画像が違えば違う")

bs_row = {"id": "bs0", "vl_ids": row["ids"], "token_types": tt.tolist(), "score_from": row["score_from"],
          "image_spans": [{"start": 2, "len": V.span_len(2, 3), "path": "img/a.png", "n_llm_h": 2, "n_llm_w": 3}]}
nr = V.normalize_row(bs_row)
check(nr["ids"] == row["ids"] and "vl_ids" not in nr and nr["images"][0]["grid"] == [2, 3]
      and V.row_token_types(nr).tolist() == tt.tolist() and V.normalize_row(row) is row,
      "normalize_row: tools/vl/build_sets.py の行 (vl_ids / image_spans) を変換、ids の行はそのまま")
bad_bs = dict(bs_row, token_types=[-1] * len(row["ids"]))
check(raises(lambda: V.normalize_row(bad_bs), ValueError), "normalize_row: token_types の食い違いを検出")
CKPT = os.environ.get("DSV41_CKPT", "/model")
HAVE_MODEL = os.path.exists(os.path.join(CKPT, "tokenizer.json"))
if HAVE_MODEL:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    args = V.model_args()
    import generate as G  # noqa: E402  (公式の prepare_case)

    def user(*parts):
        c = [{"type": "image_url", "image_url": {"url": P[x]}} if x in P else {"type": "text", "text": x} for x in parts]
        return {"role": "user", "content": c}

    case = {"id": "q1", "messages": [user("この画像は何ですか？", "a.png", "2枚目", "b.png", "比べてください。")],
            "answer": {"content": "1枚目は横長、2枚目は縦長です。"}}
    out_dir = ROOT
    r1 = V.build_row(case, tok, args, ROOT, out_dir)
    full = case["messages"] + [{"role": "assistant", **case["answer"]}]
    _, g_tok, g_types, g_imgs = G.prepare_case({"messages": full}, "chat", tok, args)
    check(r1["ids"] == g_tok and V.row_token_types(r1).tolist() == g_types,
          "build_row: ids / token_types = 公式 generate.prepare_case (会話 + 回答を一括)")
    imgs = V.image_inputs(r1, args, out_dir)
    check(len(imgs) == 2 and all(a.start == b.start and torch.equal(a.patches, b.patches) and torch.equal(a.types, b.types)
                                 for a, b in zip(imgs, g_imgs)), "image_inputs: ImageInput (start, patches, types) も一致")
    _, p_tok, _, _ = G.prepare_case({"messages": case["messages"]}, "chat", tok, args)
    check(r1["score_from"] == len(p_tok) and tok.decode(r1["ids"][r1["score_from"]:]).startswith(case["answer"]["content"])
          and V.image_end(r1) <= r1["score_from"], "score_from = 回答の先頭、画像区間は回答より前")
    check(r1["images"][0]["path"] == "img/a.png" and r1["images"][0]["sha256"] == hashlib.sha256(open(P["a.png"], "rb").read()).hexdigest(),
          "path は出力 jsonl からの相対、sha256 は画像のバイト列")
    case_f = {"id": "q2", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file://" + P["a.png"]}}, {"type": "text", "text": "説明"}]}]}
    case_r = {"id": "q3", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "img/a.png"}}, {"type": "text", "text": "説明"}]}]}
    rf, rr = V.build_row(case_f, tok, args, ROOT, ROOT), V.build_row(case_r, tok, args, ROOT, ROOT)
    check(rf["ids"] == rr["ids"] and rf["images"][0]["sha256"] == rr["images"][0]["sha256"] and rf["score_from"] == len(rf["ids"]),
          "file:// と相対パスで同じ行、回答なしは score_from = len(ids)")
    bad_sha = dict(r1, images=[dict(r1["images"][0], sha256="0" * 64), r1["images"][1]])
    check(raises(lambda: V.image_inputs(bad_sha, args, out_dir), ValueError), "image_inputs: sha256 の食い違いを検出")
    cases_p = os.path.join(ROOT, "cases.jsonl")
    with open(cases_p, "w") as f:
        for c_ in (case, case_r):
            f.write(json.dumps(c_, ensure_ascii=False) + "\n")
    V.main(["build", "--cases", cases_p, "--out", os.path.join(ROOT, "rows_vl.jsonl")])
    V.main(["check", os.path.join(ROOT, "rows_vl.jsonl")])
    check(True, "CLI build / check")
else:
    print("  (skip: 実トークナイザが無い)", flush=True)

# --------------------------------------------------------------------------- 2. capture_moe
print("[2] capture_moe の画像経路", flush=True)
IMG_ID = 300  # 偽語彙の中の画像 id
DIM, INTER, E, K, L = 64, 32, 16, 4, 3
torch.set_default_dtype(torch.bfloat16)
M.default_dtype = torch.bfloat16
margs = M.ModelArgs(dim=DIM, moe_inter_dim=INTER, n_routed_experts=E, n_activated_experts=K, n_layers=L,
                    swiglu_limit=10.0, route_scale=1.5, score_func="sqrtsoftplus", expert_dtype=None, dtype="bf16",
                    vision_n_layers=1)


def make_moe(l):
    m = M.MoE(l, margs)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "gate.weight" in name:
                p.normal_(0, 0.3)
            elif name.endswith("gate.bias"):
                p.normal_(0, 0.05)
            elif name.endswith("gate.bias_vl"):
                p.copy_(torch.linspace(1.0, -1.0, E))  # 画像は前半の専門家に強く寄る
            else:
                p.normal_(0, 0.6)
    m.swiglu_limit = margs.swiglu_limit
    return m


moes = [make_moe(l) for l in range(L)]
check(all(m.gate.bias_vl is not None for m in moes), "vision ありの ModelArgs で Gate に bias_vl")
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
M.MoE.forward = SR.moe_forward
VOCAB = 320
emb = torch.randn(VOCAB, DIM)
img_emb = {}  # 画像 (sha) ごとの区間の埋め込み


def fake_forward(ids, images=None, token_types=None):
    n = len(ids)
    h = emb[torch.tensor(ids)].float()
    im = None
    if images is not None:
        im = token_types[0] >= 0
        for img in images[0]:
            key = float(img.patches.float().mean())
            g = torch.Generator().manual_seed(int(abs(key) * 1e6) % (2**31))
            h[img.start:img.start + img.types.numel()] = torch.randn(img.types.numel(), DIM, generator=g) * 2
    for l in range(L):
        mix = torch.cumsum(h, 0) / torch.arange(1, n + 1)[:, None]
        xx = norms[l]((h + mix).to(torch.bfloat16))
        y = moes[l](xx.view(1, n, DIM), None if im is None else im.view(1, n)).view(n, DIM)
        h = h + y.float()
    return h


def tok_decode(ids):
    return "".join("<img>" if t == IMG_ID else ("<s>" if t < 4 else chr(0x61 + t % 26)) for t in ids)


SPECIAL = {0, 1, 2, 3}
rng = np.random.default_rng(0)
texts = [[0] + rng.integers(4, 290, 60).tolist() for _ in range(8)]
inp_txt = os.path.join(ROOT, "calib_txt.jsonl")
with open(inp_txt, "w") as f:
    for q, ids_ in enumerate(texts):
        f.write(json.dumps({"ids": ids_, "score_from": 20, "src": f"t{q}", "off": q}) + "\n")
# 画像付きの行: 偽の画像 (実ファイル、grid は行に書いた値を使うので image_inputs は偽にする)
vl_rows_ = []
for q, (nm, grid) in enumerate([("a.png", [2, 3]), ("b.png", [3, 2]), ("a.png", [2, 3]), ("c.png", [2, 3])]):
    pre = [0] + rng.integers(4, 290, 10).tolist()
    n_sp = V.span_len(*grid)
    post = rng.integers(4, 290, 30).tolist()
    sha = hashlib.sha256(open(P[nm], "rb").read()).hexdigest()
    vl_rows_.append({"ids": pre + [IMG_ID] * n_sp + post, "score_from": len(pre) + n_sp,
                     "images": [{"path": "img/" + nm, "start": len(pre), "grid": grid, "sha256": sha}],
                     "domain": "vl", "src": f"v{q}", "off": q})
# 行 0 と 行 3 は ids が同じで画像だけ違う (重複扱いしてはいけない)、行 0 と 行 2 は ids も画像も同じ (重複)
vl_rows_[2]["ids"] = list(vl_rows_[0]["ids"])
vl_rows_[3]["ids"] = list(vl_rows_[0]["ids"])
inp_vl = os.path.join(ROOT, "calib_vl.jsonl")
with open(inp_vl, "w") as f:
    for q, r in enumerate(vl_rows_):
        if q == 1:  # 1 行は tools/vl/build_sets.py の形式で書く (sha256 なし、path で salt)
            im = r["images"][0]
            r = {"vl_ids": r["ids"], "score_from": r["score_from"], "domain": "vl", "src": r["src"], "off": r["off"],
                 "token_types": V.row_token_types(r, IMG_ID).tolist(),
                 "image_spans": [{"start": im["start"], "len": V.span_len(*im["grid"]), "path": im["path"],
                                  "n_llm_h": im["grid"][0], "n_llm_w": im["grid"][1]}], "image_token_id": IMG_ID}
        f.write(json.dumps(r) + "\n")


def parse(inputs, extra):
    ap = C.argparse.ArgumentParser()
    C.add_common(ap)
    argv = []
    for p in inputs:
        argv += ["--input", p]
    return ap.parse_args(argv + ["--dim", str(DIM), "--topk", str(K), "--n-experts", str(E), "--n-layers", str(L),
                                 "--image-token-id", str(IMG_ID), "--no-shuffle"] + extra)


class FakeImage:
    def __init__(self, start, patches, types):
        self.start, self.patches, self.types = start, patches, types


class FakeVL:
    """capture の run が vl_rows.image_inputs を呼ぶ代わり (patches は画像の sha から決まる偽物)。"""

    @staticmethod
    def image_inputs(row, args_, base_dir):
        return [FakeImage(im["start"], torch.full((4, 3), int(hashlib.sha256((im.get("sha256") or im["path"]).encode()).hexdigest()[:6], 16) / 16**6),
                          torch.tensor(V.span_types(*im["grid"]))) for im in row["images"]]


def do_run(out, inputs, extra, seen_images):
    a = parse(inputs, ["--out", out] + extra)
    plan = C.make_plan(a, decode=tok_decode, special_ids=SPECIAL)
    meta = C.build_meta(a, plan, "/fake")
    st = C.CaptureStore(out, meta, fsync=False)
    cap = C.Capturer(st, plan)
    cap.install(M)

    def fwd(ids_):  # capture_moe.main の fwd と同じ分岐
        s = cap.cur["s"]
        with torch.inference_mode():
            if "tt" not in s:
                fake_forward(ids_)
                return
            imgs = FakeVL.image_inputs({"ids": ids_, "images": s["images"]}, None, s["base_dir"])
            seen_images.append((s["gid"], [im.start for im in imgs]))
            fake_forward(ids_, [imgs], torch.from_numpy(s["tt"].astype(np.int64))[None])

    try:
        C.run_capture(st, plan, cap, fwd, log=lambda *a_, **k_: None)
    finally:
        cap.uninstall()
    return plan, st, meta


# 2a. 画像なし: 従来の形式
seen = []
T = os.path.join(ROOT, "T")
planT, stT, metaT = do_run(T, [inp_txt], ["--preset", "all"], seen)
check(C.row_bytes(5120, 6) == 10290 and C.row_bytes(5120, 6, True) == 10291, "row_bytes: 画像なし 10290、画像あり 10291")
check("vl" not in metaT and "img_boost" not in metaT["fingerprint"]["sampling"] and "img_boost" not in planT["sampling"]
      and [f["name"] for f in metaT["fields"]] == [f[0] for f in C.FIELDS] and "tok_img" not in stT.buckets
      and "img" not in planT["tokens"], "画像なしの入力: meta・指紋・sampling・統計は従来どおり (画像の鍵なし)")
dT = C.load_layer(T, 0, as_torch=False)
check("tt" not in dT and not seen and not os.path.exists(os.path.join(T, "layer_00", "tt.bin")), "画像なし: field tt を作らない")

# 2b. 画像あり
seen = []
A = os.path.join(ROOT, "A")
planA, stA, metaA = do_run(A, [inp_txt, inp_vl], ["--rows-per-layer", "400", "--img-boost", "3", "--expert-floor", "0"], seen)
smp = planA["sampling"]
check(metaA.get("vl") and metaA["image_token_id"] == IMG_ID and metaA["fingerprint"]["sampling"]["img_boost"] == 3.0
      and [f["name"] for f in metaA["fields"]][-1] == "tt" and "tok_img" in stA.buckets, "画像あり: meta.vl、fields に tt、bucket tok_img")
check(abs(smp["p_img"] - min(1.0, 3 * smp["p_base"])) < 1e-12 and smp["p_base"] < 1.0, f"p_img = min(1, p × img_boost)",
      f"(p {smp['p_base']:.3f}, p_img {smp['p_img']:.3f})")
gid_of = {s["src"]: s["gid"] for s in planA["seqs"]}
sq = {s["gid"]: s for s in planA["seqs"]}
check(sorted(g for g, _ in seen) == sorted(gid_of[f"v{q}"] for q in range(4)), "画像の行だけ forward に images / token_types を渡す")
s0, s2, s3 = sq[gid_of["v0"]], sq[gid_of["v2"]], sq[gid_of["v3"]]
first = min(s0["gid"], s2["gid"])
dup_later = s2 if s2["gid"] > s0["gid"] else s0
span0 = slice(vl_rows_[0]["images"][0]["start"], V.image_end(vl_rows_[0]))
check(((dup_later["flags"] & C.F_DUP) > 0).all(), "ids も画像も同じ行は全位置が重複 (F_DUP)")
check(not ((s3["flags"][span0] & C.F_DUP) > 0).any() and not ((s3["flags"][span0.stop:] & C.F_DUP) > 0).any(),
      "ids が同じでも画像が違う行は画像区間とその後を重複扱いしない (prefix hash に画像を混ぜる)")
check(((s0["flags"][s0["tt"] >= 0] & (C.F_JA | C.F_KANA | C.F_HAN | C.F_ZH)) == 0).all()
      and ((s0["flags"][s0["tt"] >= 0] & C.F_SPECIAL) > 0).all(), "画像の位置: 言語ビットなし、special")
live_img = (s0["tt"] >= 0) & ((s0["flags"] & C.F_DUP) == 0)
check(np.allclose(s0["kp"][live_img], smp["p_img"]), "画像トークンの kp = p_img")
d0 = C.load_layer(A, 0, as_torch=False)
img_rows = d0["tt"] >= 0
check(d0["tt"].dtype == np.int8 and img_rows.any() and (~img_rows).any()
      and set(np.unique(d0["tt"][img_rows]).tolist()) <= {0, 1, 2, 3}, "保存行に tt (int8)、画像とテキストが混ざる")
for gid in (s0["gid"], s3["gid"]):
    m = d0["seq"] == gid
    want = sq[gid]["tt"][d0["pos"][m]]
    check(np.array_equal(d0["tt"][m], want), f"seq {gid}: tt = 行の token_types (pos で引く)")
vr = [C.verify_rows(C.load_layer(A, l), moes[l].gate, 500) for l in range(L)]
check(all(r["idx_set_agree"] == 1.0 and r["w_maxdiff"] < 1e-5 and r["checked_img"] > 0 for r in vr),
      "verify: 画像行を bias_vl で再計算して idx / w が一致", str([r["checked_img"] for r in vr]))
with torch.inference_mode():
    x_img = C.load_layer(A, 1)["x"][torch.from_numpy(C.load_layer(A, 1, False)["tt"] >= 0)]
    _, i_txt = moes[1].gate(x_img)
    _, i_vl = moes[1].gate(x_img, torch.ones(x_img.shape[0], dtype=torch.bool))
d1 = C.load_layer(A, 1, as_torch=False)
rec_idx = torch.from_numpy(d1["idx"][d1["tt"] >= 0].astype(np.int64))
check(torch.equal(torch.sort(rec_idx, -1).values, torch.sort(i_vl, -1).values)
      and not torch.equal(torch.sort(rec_idx, -1).values, torch.sort(i_txt, -1).values),
      "記録した画像行の idx は bias_vl の選択 (テキスト用 bias の選択とは違う)")
cnt_img = stA.stats["cnt.tok_img"]
n_img_live = sum(int(((s["tt"] >= 0) & ((s["flags"] & C.F_DUP) == 0)).sum()) for s in planA["seqs"] if "tt" in s)
check(np.allclose(cnt_img.sum(1), n_img_live * K) and (cnt_img[:, : E // 2].sum() > 3 * cnt_img[:, E // 2:].sum()),
      "統計 cnt.tok_img = 画像トークン (重複除く) × topk、bias_vl の寄りが見える")
check(all("n_img" in stA.info["seqs"][str(g)] for g in (s0["gid"], s3["gid"])), "系列の info に n_img")

# 2c. 画像なしの part と混ぜて読む、pack、再開
U = C.load_layer([T, A], 0, as_torch=True)
nT = len(dT["seq"])
check("tt" in U and (U["tt"][:nT] == -1).all() and torch.equal(U["tt"][nT:], torch.from_numpy(d0["tt"])),
      "画像なしの part と混ぜると tt = -1 で埋める")
Ap = os.path.join(ROOT, "Ap")
shutil.copytree(A, Ap)
C.pack(Ap)
pk = C.load_layer(Ap, 2, as_torch=False)
raw = C.load_layer(A, 2, as_torch=False)
check(all(np.array_equal(pk[k_], raw[k_]) for k_ in raw), "pack 後も tt を含め全 field が同じ")
st_again = C.CaptureStore(A, C.build_meta(parse([inp_txt, inp_vl], ["--out", A, "--rows-per-layer", "400", "--img-boost", "3",
                                                                    "--expert-floor", "0"]), planA, "/fake"), fsync=False)
check(int(st_again.rows.sum()) == int(stA.rows.sum()), "同じ設定なら再開できる")
check(raises(lambda: C.CaptureStore(A, C.build_meta(parse([inp_txt, inp_vl], ["--out", A, "--rows-per-layer", "400",
                                                                                "--img-boost", "1", "--expert-floor", "0"]),
                                                     C.make_plan(parse([inp_txt, inp_vl], ["--out", A, "--rows-per-layer", "400",
                                                                                           "--img-boost", "1", "--expert-floor", "0"]),
                                                                 decode=tok_decode, special_ids=SPECIAL), "/fake"), fsync=False),
             SystemExit), "img_boost を変えた再開は拒否")
plan_sum = C.plan_summary(planA, DIM, K, L)
check(plan_sum["row_bytes"] == C.row_bytes(DIM, K, True) and plan_sum["by_domain_this_part"]["vl"]["img"] > 0,
      "plan_summary: 画像ありの row_bytes と区分ごとの画像トークン数")
over = os.path.join(ROOT, "calib_over.jsonl")
with open(over, "w") as f:
    f.write(json.dumps(vl_rows_[1]) + "\n")
po = C.make_plan(parse([over], ["--out", os.path.join(ROOT, "O"), "--max-seq-len", "15"]), decode=tok_decode, special_ids=SPECIAL)
check(po["skipped_image_overflow"] == 1 and not po["seqs"], "画像区間が --max-seq-len を超える行は切らずに飛ばす")
stray = os.path.join(ROOT, "calib_stray.jsonl")
with open(stray, "w") as f:
    f.write(json.dumps({"ids": [0, 5, IMG_ID, 6], "score_from": 1}) + "\n")
check(raises(lambda: C.make_plan(parse([stray], ["--out", os.path.join(ROOT, "S")]), decode=tok_decode, special_ids=SPECIAL),
             SystemExit), "images の無い行に image id があれば止める")

# --------------------------------------------------------------------------- 3. qad_l2
print("[3] qad_l2 の画像重みと守り", flush=True)
torch.set_default_dtype(torch.float32)
import qad_l2 as Q  # noqa: E402
import test_qad_l2 as TQ  # noqa: E402  (合成の MoE 層と mock を流用)

tq = Q.load_capture([A], 0)
tqT = Q.load_capture([T], 0)
check("img" in tq and int(tq["img"].sum()) == int((d0["tt"] >= 0)[((d0["flags"] & C.F_DUP) == 0) & (d0["kp"] > 0)].sum())
      and "img" not in tqT, "load_capture: 画像付きの記録だけ img を返す (tt >= 0)")

s = TQ.Synth(seed=1)
n = s.data["x"].shape[0]
seq = s.data["seq"]
img = (seq % 10) >= 8  # 2 割の系列が画像
xi = s.data["x"].float().clone()
xi[img] = xi[img] * 0.3 + 12.0 * torch.sin(torch.arange(xi.shape[1]).float() / 3)[None]  # 画像は入力分布が違う
s.data["x"] = xi.to(torch.bfloat16)

new0, rep0 = Q.train_layer(0, s.data, TQ.MockTeacher(s), TQ.MockPack(s), TQ.make_cfg(), "cpu")
d_no = dict(s.data, img=torch.zeros(n, dtype=torch.bool))
new1, rep1 = Q.train_layer(0, d_no, TQ.MockTeacher(s), TQ.MockPack(s), TQ.make_cfg(img_weight=3.0, img_guard_tol=-1.0), "cpu")
check(new0.keys() == new1.keys() and all(torch.equal(new0[k_], new1[k_]) for k_ in new0)
      and "img" not in rep1["layer_val"] and "rejected_img" not in rep1,
      "画像が無ければ img_weight / 守りを変えてもビット一致 (報告にも画像の鍵なし)")
val = Q.split_val(s.data["lang"], seq, 0.1, 0)
val_i = Q.split_val(s.data["lang"], seq, 0.1, 0, img=img)
check(torch.equal(val, Q.split_val(s.data["lang"], seq, 0.1, 0, img=torch.zeros(n, dtype=torch.bool)))
      and bool((val_i & img).any()) and bool((val_i & ~img).any()), "split_val: 画像なしなら従来と同じ、画像ありなら画像系列も検証に入る")
val_f = Q.split_val(s.data["lang"], seq, 0.1, 0, img=img, frac_img=0.5)
n_iseq = lambda v: int(torch.unique(seq[v & img]).numel())  # noqa: E731
check(torch.equal(val, Q.split_val(s.data["lang"], seq, 0.1, 0, frac_img=0.5))
      and torch.equal(val_i, Q.split_val(s.data["lang"], seq, 0.1, 0, img=img, frac_img=None))
      and n_iseq(val_f) > n_iseq(val_i) and torch.equal(val_f & ~img & ~torch.isin(seq, torch.unique(seq[img])),
                                                         val_i & ~img & ~torch.isin(seq, torch.unique(seq[img]))),
      "split_val frac_img: 画像系列の検証だけ増え、テキスト系列の分割と画像なしの分割は変わらない",
      f"(画像系列の検証 {n_iseq(val_i)} -> {n_iseq(val_f)})")
_, rep_f = Q.train_layer(0, dict(s.data, img=img), TQ.MockTeacher(s), TQ.MockPack(s),
                         TQ.make_cfg(min_val_img=8, img_guard_tol=1e9, val_frac_img=0.5), "cpu")
check("img_guarded" in rep_f and rep_f["img_guarded"] == sum("img_init_val" in v for v in rep_f["experts"].values()),
      "報告に img_guarded (守りを掛けた expert 数)")
d_img = dict(s.data, img=img)
cfgk = dict(min_val_img=8)
new2, rep2 = Q.train_layer(0, d_img, TQ.MockTeacher(s), TQ.MockPack(s), TQ.make_cfg(img_guard_tol=1e9, **cfgk), "cpu")
ex2 = rep2["experts"]
guarded = [e for e, v in ex2.items() if "img_init_val" in v]
check(len(guarded) >= 5 and "img" in rep2["layer_val"] and "text" in rep2["layer_val"] and rep2["rejected_img"] == 0,
      "画像の検証対が十分な expert は画像誤差を測り、layer_val に img / text", f"(measured {len(guarded)} experts)")
lv = rep2["layer_val"]
print(f"    layer_val img v1 {lv['img']['v1']:.4f} -> {lv['img']['qad']:.4f}, text {lv['text']['v1']:.4f} -> {lv['text']['qad']:.4f}")
new3, rep3 = Q.train_layer(0, d_img, TQ.MockTeacher(s), TQ.MockPack(s), TQ.make_cfg(img_guard_tol=-1.0, **cfgk), "cpu")
ex3 = rep3["experts"]
rej = [e for e, v in ex3.items() if v["status"] == "rejected_img"]
check(set(rej) == {e for e in guarded if ex2[e]["status"] == "accepted"} and rep3["rejected_img"] == len(rej)
      and not any(f".experts.{e}." in k_ for e in rej for k_ in new3), "守り: 画像誤差の条件を満たさない expert は rejected_img で v1 のまま",
      f"(rejected_img {len(rej)})")
worse = [e for e in guarded if ex2[e]["status"] == "accepted" and ex2[e]["img_final_val"] > ex2[e]["img_init_val"]]
new4, rep4 = Q.train_layer(0, d_img, TQ.MockTeacher(s), TQ.MockPack(s), TQ.make_cfg(**cfgk), "cpu")
check({e for e, v in rep4["experts"].items() if v["status"] == "rejected_img"} == set(worse),
      "守り (tol 0): 画像誤差が v1 より悪くなる expert だけ拒否", f"(悪化 {len(worse)} / 測定 {len(guarded)})")
rel_img = {}
for w_ in (1.0, 4.0):
    _, r_ = Q.train_layer(0, d_img, TQ.MockTeacher(s), TQ.MockPack(s),
                          TQ.make_cfg(img_weight=w_, img_guard_tol=1e9, l2=0.0, **cfgk), "cpu")
    rel_img[w_] = (r_["layer_val"]["img"]["qad"], r_["layer_val"]["text"]["qad"])
print(f"    img_weight 1 -> 4: img {rel_img[1.0][0]:.4f} -> {rel_img[4.0][0]:.4f}, text {rel_img[1.0][1]:.4f} -> {rel_img[4.0][1]:.4f}")
check(rel_img[4.0][0] <= rel_img[1.0][0] * 1.02, "img_weight 1 -> 4: 画像の誤差が悪くならない (重みの向き)")
_, rep5 = Q.train_layer(0, d_img, TQ.MockTeacher(s), TQ.MockPack(s),
                        TQ.make_cfg(loss="layer", epochs=15, eval_every=5, img_guard_tol=-1.0), "cpu")
check(rep5["status"] == "rejected_img" and "img" in rep5["layer_val"], "layer mode: 守り (tol -1) で層ごと拒否、layer_val に img")
try:
    Q.main(["train", "--help"])
except SystemExit:
    pass
check(True, "CLI train --help (--img-weight / --img-guard-tol / --min-val-img)")

print(f"\nALL OK ({len(OK)} checks)", flush=True)
