"""stream_ref_vl.py の CPU テスト (GPU・CUDA 拡張を使わない)。

  sudo docker run --rm --network none --cpus 8 --memory 16g -e CUDA_VISIBLE_DEVICES= \
    -v $DSV41/ref:/ref:ro -v $DSV41/harness:/h:ro \
    -v $MODELS/DeepSeek-V4.1-Flash-1017:/model:ro -v <scratch>:/s \
    --entrypoint bash dsv41-quant:151 -c 'cd /h && python3 test_stream_ref_vl.py /s/vl_test'

1. 入力の準備 (実トークナイザ・実 config・合成画像): 公式 prepare_case と一致、画像区間の位置、
   回答の継ぎ目 (一括トークナイズと一致)、thinking、data:/file:/相対パス、上限、エラー
2. 画像区間への埋め込み: 小さな ViT + Aligner (実 patch 14 / downsample 3) と model.py の
   merge_image_embeddings で、IMAGE の位置に aligner の行が順に入り、区切りとテキストは正しい
3. routing の text / image 分け: model.py の MoE (bias_vl あり) + stream_ref.moe_forward
4. 教師 / 学生 / 生成の流れ: 偽モデルで run() を回し、stream_ref.teacher と同じ数値、
   画像は prefill だけに渡る、学生の KL=0 (同じモデル)、不一致の検出
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import base64
import io
import json
import os
import shutil
import sys
import types as pytypes

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream_ref as SR  # noqa: E402
import stream_ref_vl as V  # noqa: E402

M = SR.M
from image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, TEXT  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vl_test"
shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(os.path.join(ROOT, "img"))
torch.manual_seed(0)
OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print("  ok:", msg, flush=True)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


check(M.MoE.forward.__qualname__ == "MoE.forward" and M.Gate.forward.__qualname__ == "Gate.forward",
      "import しただけでは model.py を差し替えない (stream_ref と同じ)")

# --------------------------------------------------------------------------- 合成画像
rng = np.random.default_rng(0)


def make_img(name, w, h):
    im = Image.new("RGB", (w, h), (int(rng.integers(256)), 80, 160))
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x0, y0 = int(rng.integers(w)), int(rng.integers(h))
        d.rectangle([x0, y0, x0 + w // 5, y0 + h // 5], fill=tuple(int(c) for c in rng.integers(0, 256, 3)))
    p = os.path.join(ROOT, "img", name)
    im.save(p)
    return p


P = {n: make_img(n, w, h) for n, w, h in [("a.png", 640, 480), ("tall.png", 120, 2400), ("wide.png", 3000, 150),
                                          ("tiny.png", 40, 30), ("big.jpg", 4000, 3000), ("b.png", 900, 600)]}

# --------------------------------------------------------------------------- 1. 入力の準備
print("[1] 入力の準備 (実トークナイザ / 実 config)", flush=True)
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(SR.CKPT)
args = V.load_model_args({"max_batch_size": 1, "max_seq_len": 16384})
IMG_ID = args.image_token_id
EOS = tok.eos_token_id
check(args.vision_enabled and IMG_ID == 129264 and tok.convert_tokens_to_ids("<｜deepseek_image｜>") == IMG_ID,
      "config の vision と image_token_id 129264 = トークナイザの <｜deepseek_image｜>")


def user(*blocks):
    content = []
    for b in blocks:
        if isinstance(b, str) and b in P:
            content.append({"type": "image_url", "image_url": {"url": P[b]}})
        else:
            content.append({"type": "text", "text": b})
    return {"role": "user", "content": content}


import generate as G  # noqa: E402  (公式の prepare_case)

rec1 = {"id": "one", "messages": [user("この画像に何が写っていますか？", "a.png", "日本語で答えてください。")]}
c1 = V.prepare_vl_case(rec1, tok, args)
_, g_tok, g_types, g_imgs = G.prepare_case({"messages": rec1["messages"]}, "chat", tok, args)
check(c1["ids"] == g_tok and c1["token_types"] == g_types, "ids / token_types が公式 generate.prepare_case と一致")
check(len(c1["images"]) == 1 and all(a.start == b.start and torch.equal(a.patches, b.patches) and torch.equal(a.types, b.types)
                                     for a, b in zip(c1["images"], g_imgs)), "ImageInput (start, patches, types) も一致")
from encoding import encode_case  # noqa: E402

prompt1, _ = encode_case({"messages": rec1["messages"]}, "chat")
raw = tok.encode(prompt1)
check(raw.count(IMG_ID) == 1 and c1["images"][0].start == raw.index(IMG_ID), "画像区間の開始 = 置換前の placeholder の位置")
sp = V.image_spans(c1)[0]
s, n = sp["start"], sp["n_tokens"]
check(c1["token_types"][s] == IMAGE_START and c1["token_types"][s + n - 1] == IMAGE_END
      and sum(t == IMAGE_NEW_LINE for t in c1["token_types"]) == sp["n_llm_h"]
      and sum(t == IMAGE for t in c1["token_types"]) == sp["n_llm_h"] * sp["n_llm_w"], "START / IMAGE 行 + NEW_LINE / END の並び")
check(n == sp["n_llm_h"] * (sp["n_llm_w"] + 1) + 2 and c1["ids"][s:s + n] == [IMG_ID] * n, "区間の長さ h(w+1)+2、全位置が id 129264")
check(c1["ids"][:s] == raw[:s] and c1["ids"][s + n:] == raw[s + 1:], "区間の前後のテキストは置換前と同じ")
check(c1["ids"][0] == 0 and c1["ids"].count(0) == 1 and c1["ids"][-2:] == [128804, 128822],
      "BOS は 1 個 (tokenizer は BOS を足さない)、末尾は <｜Assistant｜></think>")
check(c1["score_from"] is None and c1["prompt_len"] == len(c1["ids"]) and c1["answer_source"] == "teacher_greedy",
      "answer なし → 生成待ち (score_from=None)")
print(f"    a.png 640x480: vit {sp['n_vit_h']}x{sp['n_vit_w']} → llm {sp['n_llm_h']}x{sp['n_llm_w']} = {n} tok, start {s}")

# 2 枚 + 履歴付きの多ターン
rec2 = {"id": "multi", "messages": [user("1枚目", "a.png", "2枚目", "tall.png", "比べて"),
                                    {"role": "assistant", "content": "縦長の画像です。"},
                                    user("次はこれ", "wide.png", "と", "tiny.png")],
        "answer": "横長の画像と小さい画像です。"}
c2 = V.prepare_vl_case(rec2, tok, args)
prompt2, recs2 = encode_case({"messages": rec2["messages"]}, "chat")
raw2 = tok.encode(prompt2)
ph = [i for i, t in enumerate(raw2) if t == IMG_ID]
sps = V.image_spans(c2)
exp, shift = [], 0
for i, p in enumerate(ph):
    exp.append(p + shift)
    shift += sps[i]["n_tokens"] - 1
check(len(ph) == 4 and [x["start"] for x in sps] == exp, f"4 枚の開始位置 = placeholder の位置 + 前の区間の伸び ({exp})")
check([r["url"] for r in recs2] == [P["a.png"], P["tall.png"], P["wide.png"], P["tiny.png"]], "画像の順 = プロンプト内の順 (履歴の画像も含む)")
check(all(x["n_tokens"] <= args.vision_max_n_token for x in sps), "各画像 ≤ vision_max_n_token (1024)")
check(V.check_spans(c2, IMG_ID), "check_spans: 重なりなし、プロンプト内、区間外に画像 id なし、aligner の格子")
check(c2["score_from"] == c2["prompt_len"] and all(x["start"] + x["n_tokens"] <= c2["score_from"] for x in sps),
      "score_from = 回答の先頭、画像区間はすべて prefill 側")
full2 = rec2["messages"] + [{"role": "assistant", "content": rec2["answer"]}]
_, g2_tok, g2_types, _ = G.prepare_case({"messages": full2}, "chat", tok, args)
check(c2["ids"] == g2_tok and c2["token_types"] == g2_types, "プロンプト + 回答の継ぎ足し = 会話全体を一括で prepare_case したもの")
check(tok.decode(c2["ids"][c2["score_from"]:]) == rec2["answer"] + "<｜end▁of▁sentence｜>", "回答区間 = 回答本文 + EOS")
check(all(t == TEXT for t in c2["token_types"][c2["score_from"]:]), "回答区間は全部 TEXT")
print("    multi:", [(x["start"], x["n_tokens"], x["n_llm_h"], x["n_llm_w"]) for x in sps])

# thinking mode + reasoning_content
rec3 = {"id": "think", "thinking_mode": "thinking", "messages": [user("b.png", "何色？")],
        "answer": {"content": "赤と緑です。", "reasoning_content": "画像を見ると四角形がある。"}}
c3 = V.prepare_vl_case(rec3, tok, args)
check(c3["ids"][c3["score_from"] - 1] == 128821, "thinking: プロンプトは <think> で終わる")
check(tok.decode(c3["ids"][c3["score_from"]:]) == "画像を見ると四角形がある。</think>赤と緑です。<｜end▁of▁sentence｜>",
      "thinking: 回答区間 = 推論 + </think> + 本文 + EOS")
_, g3_tok, _, _ = G.prepare_case({"messages": rec3["messages"] + [{"role": "assistant", **rec3["answer"]}],
                                  "thinking_mode": "thinking"}, "chat", tok, args)
check(c3["ids"] == g3_tok, "thinking: 継ぎ足し = 一括")
c3b = V.prepare_vl_case(dict(rec3, thinking_mode=None), tok, args, thinking="thinking")
check(c3b["ids"] == c3["ids"], "--thinking thinking でも同じ (行の thinking_mode が優先、無ければ CLI)")

# 上限: 大きい画像は 1024 トークン以下に縮む、tall/wide も
cbig = V.prepare_vl_case({"messages": [user("big.jpg", "説明")]}, tok, args)
spb = V.image_spans(cbig)[0]
check(spb["n_tokens"] <= 1024 and spb["n_tokens"] > 900, f"4000x3000 は {spb['n_tokens']} トークン (≤1024 に収まる最大近く)")
ctiny = V.image_spans(V.prepare_vl_case({"messages": [user("tiny.png")]}, tok, args))[0]
check(ctiny["n_vit_h"] * ctiny["n_vit_w"] * 14 * 14 >= 0.9 * args.vision_min_pixels, "40x30 は vision_min_pixels まで拡大")

# URL の書き方: 相対パス / file:// / data: で同じ入力
b64 = base64.b64encode(open(P["a.png"], "rb").read()).decode()
variants = {"rel": "img/a.png", "file": "file://" + P["a.png"], "data": "data:image/png;base64," + b64}
same = True
for k, u in variants.items():
    r = {"messages": [{"role": "user", "content": [{"type": "text", "text": "この画像に何が写っていますか？"},
                                                   {"type": "image_url", "image_url": {"url": u}},
                                                   {"type": "text", "text": "日本語で答えてください。"}]}]}
    c = V.prepare_vl_case(r, tok, args, base_dir=ROOT)
    same &= c["ids"] == c1["ids"] and torch.equal(c["images"][0].patches, c1["images"][0].patches) and c["image_sha"] == c1["image_sha"]
check(same, "相対パス (jsonl 基準) / file:// / data:base64 で ids・patches・sha256 が同じ")
check(rec1["messages"][0]["content"][1]["image_url"]["url"] == P["a.png"], "入力の messages は書き換えない")
check(raises(lambda: V.prepare_vl_case({"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "nope.png"}}]}]}, tok, args, base_dir=ROOT), FileNotFoundError),
      "存在しない画像は FileNotFoundError (build の前に落ちる)")
check(raises(lambda: V.prepare_vl_case({"messages": [{"role": "user", "content": "<｜deepseek_image｜>です"}]}, tok, args),
             ValueError), "テキストに placeholder を書くと ValueError (encoding の検査)")
check(raises(lambda: V.prepare_vl_case({"messages": rec1["messages"], "context": [{"role": "user", "content": "x"}]}, tok, args),
             ValueError), "context は未対応 (公式と同じ)")

# 既存形式 {ids, score_from} の通過
crow = V.prepare_vl_case({"ids": [0, 128803, 6366, 128804, 128822, 6366, 1], "score_from": 5}, tok, args)
check(crow["images"] is None and crow["token_types"] == [TEXT] * 7 and crow["score_from"] == 5 and crow["answer_source"] == "ids",
      "既存の {ids, score_from} 行はそのまま (画像なし)")

# check_spans の否定
bad = dict(c1, token_types=list(c1["token_types"]))
bad["token_types"][s + 3] = TEXT
check(raises(lambda: V.check_spans(bad, IMG_ID), AssertionError), "check_spans: 型の並びが崩れると検出")
bad = dict(c1, prompt_len=s + 5)
check(raises(lambda: V.check_spans(bad, IMG_ID), AssertionError), "check_spans: 画像区間がプロンプトからはみ出すと検出")

# jsonl + --plan-only
jl = os.path.join(ROOT, "cases.jsonl")
with open(jl, "w") as f:
    for r in [{"id": "rel", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "img/b.png"}},
                                                                      {"type": "text", "text": "説明して"}]}]},
              rec2, rec3, {"ids": crow["ids"], "score_from": 5}]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
cases = V.load_vl_cases(jl, tok, args)
check(len(cases) == 4 and cases[0]["images"][0].patches.shape[0] == V.image_spans(cases[0])[0]["n_vit_h"] * V.image_spans(cases[0])[0]["n_vit_w"],
      "load_vl_cases: jsonl のディレクトリを基準に相対パスを解決")
V.main(["--vl-in", jl, "--plan-only", "--max-seq-len", "8192"])
check(True, "--plan-only は GPU なしで最後まで動く")

# ds41rt 用の要求本体: 画像は data: URL、同じ messages なら同じ ids / patches
ex = os.path.join(ROOT, "ds41rt.jsonl")
V.main(["--vl-in", jl, "--export-openai", ex])
bodies = [json.loads(l) for l in open(ex)]
urls = [b_["image_url"]["url"] for x in bodies for m in x["body"]["messages"] if isinstance(m["content"], list)
        for b_ in m["content"] if b_.get("type") == "image_url"]
rt = [V.prepare_vl_case({"messages": x["body"]["messages"], "answer": r_.get("answer"), "thinking_mode": r_.get("thinking_mode")},
                        tok, args) for x, r_ in zip(bodies, [json.loads(l) for l in open(jl)][:3])]
check(len(bodies) == 3 and all(u.startswith("data:image/png;base64,") for u in urls) and len(urls) == 6
      and all(a_["ids"] == b_["ids"] and all(torch.equal(p_.patches, q_.patches) for p_, q_ in zip(a_["images"], b_["images"]))
              for a_, b_ in zip(rt, cases[:3])) and bodies[0]["body"]["temperature"] == 0,
      "--export-openai: 画像を data: URL に埋め、同じ ids / patches に戻る (既存形式の行は書かない)")

# --------------------------------------------------------------------------- 2. 画像区間への埋め込み
print("[2] merge_image_embeddings (小さな ViT + Aligner、実 patch/downsample)", flush=True)
torch.set_default_dtype(torch.bfloat16)
DIM = 16
vargs = M.ModelArgs(dim=DIM, vision_n_layers=2, vision_dim=32, vision_n_heads=2, vision_inter_dim=64,
                    vision_patch_size=args.vision_patch_size, vision_downsample_ratio=args.vision_downsample_ratio)
from vision import ViT, Aligner  # noqa: E402

vit, ali = ViT(vargs), Aligner(vargs)
with torch.no_grad():
    for p in list(vit.parameters()) + list(ali.parameters()):
        p.normal_(0, 0.05)
fake = pytypes.SimpleNamespace(image_start=torch.randn(DIM), image_end=torch.randn(DIM), image_newline=torch.randn(DIM))
fake.encode_image = lambda pt, h, w: ali(vit(pt, h, w), h, w)
c = c2
L = len(c["ids"])
h = torch.arange(L, dtype=torch.float32)[:, None].repeat(1, DIM).to(torch.bfloat16)[None].clone()
h0 = h.clone()
with torch.inference_mode():
    M.Transformer.merge_image_embeddings(fake, [c["images"]], h)
    emb = [fake.encode_image(img.patches, img.n_vit_h, img.n_vit_w) for img in c["images"]]
good = True
tt = torch.tensor(c["token_types"])
for img, e in zip(c["images"], emb):
    st, nn_ = img.start, img.types.numel()
    seg = h[0, st:st + nn_]
    good &= e.shape[0] == int((img.types == IMAGE).sum())
    good &= torch.equal(seg[img.types == IMAGE], e.to(h.dtype))
    good &= bool((seg[img.types == IMAGE_START] == fake.image_start).all() and (seg[img.types == IMAGE_END] == fake.image_end).all()
                 and (seg[img.types == IMAGE_NEW_LINE] == fake.image_newline).all())
check(good, "4 枚とも: IMAGE の位置に aligner の行が行優先で入り、START/END/NEW_LINE は学習済みベクトル")
check(torch.equal(h[0, tt == TEXT], h0[0, tt == TEXT]), "テキスト位置 (回答区間を含む) は埋め込みのまま")

# --------------------------------------------------------------------------- 3. routing counts
print("[3] routing の text / image 分け (model.py MoE + bias_vl)", flush=True)
E, K = 16, 6
M.default_dtype = torch.bfloat16
margs = M.ModelArgs(dim=64, moe_inter_dim=32, n_routed_experts=E, n_activated_experts=K, n_layers=2, vision_n_layers=1,
                    swiglu_limit=10.0, route_scale=1.5, score_func="sqrtsoftplus", expert_dtype=None, dtype="bf16")
moe = M.MoE(0, margs)
with torch.no_grad():
    for name, p in moe.named_parameters():
        p.normal_(0, 0.3 if "gate.weight" in name else 0.05 if "bias" in name else 0.6)
    moe.gate.bias_vl.zero_()
    moe.gate.bias_vl[10:16] = 100.0  # 画像トークンは専門家 10..15 に寄せる
moe.swiglu_limit = margs.swiglu_limit
check(moe.gate.bias_vl is not None, "vision_enabled の Gate は bias_vl を持つ")


class FakeStore:
    def __init__(self):
        self.counts, self.stats = {}, {}

    def get(self, layer, used):
        return {e: ("bf16", moe.experts[e].w1.weight.t().contiguous(), moe.experts[e].w3.weight.t().contiguous(),
                    moe.experts[e].w2.weight.t().contiguous()) for e in used}


SR.STORE = FakeStore()
M.MoE.forward = SR.moe_forward
x = torch.randn(1, 80, 64).to(torch.bfloat16) * 3
mask = torch.zeros(1, 80, dtype=torch.bool)
mask[0, 20:50] = True
with torch.inference_mode():
    y_plain = moe(x, mask)
    rc = V.RouteCounter()
    rc.install(M)
    SR.STORE.counts = {}  # 包む前の 1 回を除く
    y_cnt = moe(x, mask)
    tot_after_img = SR.STORE.counts[0].copy()
    y_txt = moe(x, None)
    rc.uninstall()
    w_i, i_i = moe.gate(x.view(-1, 64), mask.flatten())
    w_t, i_t = moe.gate(x.view(-1, 64), None)
check(M.MoE.forward is SR.moe_forward and M.Gate.forward.__qualname__ == "Gate.forward", "uninstall で元に戻る")
check(torch.equal(y_plain, y_cnt), "数えても出力はビット一致")
exp_img = np.bincount(i_i[mask.flatten()].flatten().numpy(), minlength=E)
check(np.array_equal(rc.image[0], exp_img) and rc.n_image_tokens == 30, "画像区間の回数 = gate(bias_vl) の画像行の bincount、トークン数 30")
check(set(np.flatnonzero(exp_img)) == set(range(10, 16)) and exp_img[10:].min() == 30, "bias_vl が画像トークンの選択を変える (10..15 に全部)")
check(not np.isin(i_t[mask.flatten()].numpy(), np.arange(10, 16)).all(), "同じ入力でもテキスト扱いなら別の専門家 (bias)")
txt, img = rc.split({0: tot_after_img})
exp_txt = np.bincount(i_i[~mask.flatten()].flatten().numpy(), minlength=E)
check(np.array_equal(txt[0], exp_txt) and np.array_equal(img[0], exp_img), "split: テキスト = 全体 − 画像 (テキスト行の bincount と一致)")
check(np.array_equal(rc.image[0], exp_img) and int(SR.STORE.counts[0].sum()) == 2 * 80 * K,
      "image_mask=None の呼び出しは画像に数えない、STORE.counts は従来どおり全トークン")
check(rc.seq_array(2, E).shape == (2, E) and np.array_equal(rc.seq_array(2, E)[0], exp_img), "系列ごとの画像 counts [n_layers, E]")

# --------------------------------------------------------------------------- 4. run() (偽モデル)
print("[4] 教師 / 学生 / 生成 (偽モデル)", flush=True)
torch.set_default_dtype(torch.float32)
VOC = args.vocab_size


class FakeModel:
    """位置ごとの logits = 直前トークンと位置の関数 + 画像の要約 (prefill で受けて decode でも持つ)。"""

    def __init__(self, seed, stop_after=6):
        g = torch.Generator().manual_seed(seed)
        self.E = torch.randn(VOC, 4, generator=g)
        self.W = torch.randn(4, VOC, generator=g)
        self.Wi = torch.randn(VOC, generator=g)
        self.calls = []
        self.stop_after = stop_after
        self.prompt_end = None
        self.img = 0.0

    def forward(self, input_ids, start_pos=0, images=None, token_types=None):
        b, n = input_ids.shape
        self.calls.append((start_pos, n, images is not None, None if token_types is None else tuple(token_types.shape)))
        if images is not None:
            assert start_pos == 0 and token_types.shape == input_ids.shape
            self.img = float(sum(im.patches.float().mean() for im in images[0]))
            assert (input_ids[0][token_types[0] >= 0] == IMG_ID).all()
        elif start_pos == 0:
            self.img = 0.0
        if start_pos == 0:
            self.prompt_end = n
        pos = torch.arange(start_pos, start_pos + n, dtype=torch.float32)[:, None]
        lg = (self.E[input_ids[0]] * (1 + 0.01 * pos)) @ self.W + self.img * self.Wi
        if start_pos > 0 and start_pos - self.prompt_end >= self.stop_after:
            lg[:, EOS] = 1e4
        lg = lg[None]
        out = lg[:, -1].argmax(-1)
        return out, (lg if SR.FULL_LOGITS else lg[:, -1]), None


SR.STORE = FakeStore()
SR.STORE.counts = {0: np.zeros(E, dtype=np.int64)}
V.COUNTER = V.RouteCounter()


class CountingFake:
    """偽モデルの forward ごとに STORE.counts / COUNTER を「トークン数」だけ進める (生成分の巻き戻しの検査用)。"""

    def __init__(self, inner):
        self.inner, self.calls = inner, inner.calls

    def forward(self, input_ids, start_pos=0, images=None, token_types=None):
        n = input_ids.shape[1]
        SR.STORE.counts[0][0] += n
        if token_types is not None:
            ni = int((token_types >= 0).sum())
            V.COUNTER.image.setdefault(0, np.zeros(E, dtype=np.int64))[0] += ni
            V.COUNTER.n_image_tokens += ni
        return self.inner.forward(input_ids, start_pos, images, token_types)
targs = V.load_model_args({"max_batch_size": 1, "max_seq_len": 8192})
tm = FakeModel(1)
tpt, gout = os.path.join(ROOT, "teacher.pt"), os.path.join(ROOT, "gen.jsonl")
V.run(CountingFake(tm), tok, targs, cases, tpt, gen_out=gout, max_new=32)
T = torch.load(tpt, weights_only=False)
n_all = sum(len(r["ids"]) for r in T["recs"])
n_img = sum(r["n_image_tokens"] for r in T["recs"])
check(int(T["counts"][0][0]) == n_all and int(T["counts_image"][0][0]) == n_img and T["n_image_tokens"] == n_img
      and int(T["counts_text"][0][0]) == n_all - n_img,
      "counts は採点の prefill だけ (教師の生成に使った prefill / decode は巻き戻す)")
r0 = T["recs"][0]
check(r0["answer_source"] == "teacher_greedy" and r0["ids"][-1] == EOS and r0["score_from"] == cases[0]["prompt_len"]
      and len(r0["ids"]) - r0["score_from"] == 8 and not r0["truncated"], "answer なし → 教師 greedy (EOS まで) を回答に、score_from = プロンプト長")
gcalls = [cl for cl in tm.calls]
# 画像あり 3 問: 0 番は生成の prefill + 採点の prefill、1・2 番は採点の prefill だけ → 4 回。3 番 (ids) は画像なし
check(sum(1 for cl in gcalls if cl[2]) == 4 and all(cl[0] == 0 for cl in gcalls if cl[2]),
      "画像は start_pos=0 の prefill にだけ渡る (生成の decode には渡らない)")
dec = [cl for cl in gcalls if cl[0] > 0]
check(len(dec) == 7 and [cl[0] for cl in dec] == list(range(cases[0]["prompt_len"], cases[0]["prompt_len"] + 7))
      and all(cl[1] == 1 for cl in dec), "decode は 1 トークンずつ、位置は連続")
check(set(T) >= {"recs", "counts", "counts_text", "counts_image", "n_image_tokens", "kl", "agree"} and T["vl"],
      ".pt に counts (全体) / counts_text / counts_image")
check(all(set(r) >= {"i", "src", "ids", "score_from", "top_v", "top_i", "tgt_lp", "argmax_match", "images", "image_sha",
                     "prompt_len", "counts_image"} for r in T["recs"]), "rec は既存のキー + images / prompt_len / counts_image")
check(r0["top_v"].shape == (8, 32) and r0["top_v"].dtype == torch.float16 and r0["top_i"].dtype == torch.int32,
      "top-32 logprob は既存と同じ形 (fp16 / int32)")
check(T["recs"][1]["images"][1]["sha256"] == V.image_sha256({"url": P["tall.png"]}) and T["recs"][1]["n_image_tokens"] == sum(x["n_tokens"] for x in sps),
      "images 情報 (sha256、区間) と画像トークン数")
g = [json.loads(l) for l in open(gout)]
check(len(g) == 4 and g[0]["answer_ids"] == r0["ids"][r0["score_from"]:] and g[1]["answer_source"] == "given",
      "--gen-out に回答 (生成 / 与えた) を書く")

# stream_ref.teacher と同じ数値 (既存形式の行)
seqs = [{"ids": cases[3]["ids"], "score_from": cases[3]["score_from"]}]
spt = os.path.join(ROOT, "sr_teacher.pt")
SR.teacher(tm, seqs, spt)
S0 = torch.load(spt, weights_only=False)["recs"][0]
r3 = T["recs"][3]
check(torch.equal(S0["top_v"], r3["top_v"]) and torch.equal(S0["top_i"], r3["top_i"]) and torch.equal(S0["tgt_lp"], r3["tgt_lp"])
      and S0["argmax_match"] == r3["argmax_match"], "既存形式の行は stream_ref.teacher とビット一致")

# 画像が logits に効いている (偽モデルで画像の要約を足している) ことの確認: 画像なしで回すと違う
nimg = dict(cases[1], images=None)
ri = V.score(V.prefill_logits(tm, nimg["ids"], nimg["token_types"], None), nimg["ids"], nimg["score_from"])[0]
check(not torch.equal(ri["top_v"], T["recs"][1]["top_v"]), "画像を渡さないと教師 logits が変わる (画像が forward に届いている)")

# 学生: 同じモデルなら KL 0 / top1 1.0、別モデルなら KL > 0 で式は stream_ref.teacher と同じ
tm2 = FakeModel(1)
spt2 = os.path.join(ROOT, "student_same.pt")
V.run(tm2, tok, targs, cases, spt2, ref=T)
St = torch.load(spt2, weights_only=False)
kl_all = torch.cat(St["kl"])
check(kl_all.abs().max().item() < 5e-3 and torch.cat(St["agree"]).all(), f"同じモデルの学生は KL≈0 (max {kl_all.abs().max().item():.2e}) / top1 100%")
check(not any(cl[0] > 0 for cl in tm2.calls), "学生は生成しない (ids は教師 .pt から)")
check(all(a["ids"] == b["ids"] for a, b in zip(St["recs"], T["recs"])), "学生の ids = 教師の ids (生成した回答も)")
tm3 = FakeModel(2)
V.run(tm3, tok, targs, cases, os.path.join(ROOT, "student_other.pt"), ref=T)
S3 = torch.load(os.path.join(ROOT, "student_other.pt"), weights_only=False)
spt3 = os.path.join(ROOT, "sr_student.pt")
SR.teacher(FakeModel(2), [dict(s_) for s_ in seqs], spt3, ref={"recs": [T["recs"][3]]})
check(torch.cat(S3["kl"]).mean().item() > 0.01 and torch.allclose(torch.load(spt3, weights_only=False)["kl"][0], S3["kl"][3]),
      "別モデルの学生: KL > 0、既存形式の行の KL は stream_ref.teacher の学生と一致")

# 不一致の検出
other = [dict(c_) for c_ in cases]
other[0] = V.prepare_vl_case({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": P["a.png"]}},
                                                                        {"type": "text", "text": "説明して"}]}]}, tok, args)
check(raises(lambda: V.run(FakeModel(1), tok, targs, other, "", ref=T), AssertionError), "画像やプロンプトが教師と違えば学生は止まる")
other = [dict(c_) for c_ in cases]
other[1] = V.prepare_vl_case(dict(rec2, answer="違う回答"), tok, args)
check(raises(lambda: V.run(FakeModel(1), tok, targs, other, "", ref=T), AssertionError), "与えた回答が教師と違えば止まる")
check(raises(lambda: V.run(FakeModel(1), tok, V.load_model_args({"max_seq_len": cases[0]["prompt_len"] + 4}), cases[:1], "",
                           max_new=32), AssertionError), "生成の長さが max_seq_len を超えるなら始める前に止まる")

# 生成だけ (teacher_logits=False)
tm4 = FakeModel(1)
V.run(tm4, tok, targs, cases[:1], "", gen_out=os.path.join(ROOT, "gen_only.jsonl"), teacher_logits=False)
check(json.loads(open(os.path.join(ROOT, "gen_only.jsonl")).readline())["answer_ids"] == g[0]["answer_ids"]
      and sum(1 for cl in tm4.calls if cl[0] == 0) == 1, "生成だけのモード (prefill 1 回 + decode、logits の prefill なし)")

# --------------------------------------------------------------------------- 5. build_sets.py の行 (vl_kl / vl_calib)
print("[5] tools/vl/build_sets.py の行形式 (vl_ids / score_from / spans / q_spans)", flush=True)
A_ID = tok.convert_tokens_to_ids("<｜Assistant｜>")


def bs_row(messages, rid="bs"):
    """build_sets.Enc.render と同じ規則で行を作る (公式 prepare_case + <｜Assistant｜> の次から EOS までを spans)。"""
    _, ids_, tt_, ims_ = G.prepare_case({"messages": messages}, "chat", tok, args)
    spans_, j_ = [], 0
    while j_ < len(ids_):
        if ids_[j_] == A_ID:
            e_ = next(x for x in range(j_, len(ids_)) if ids_[x] == EOS) + 1
            spans_.append([j_ + 1, e_])
            j_ = e_
        else:
            j_ += 1
    last_ = max(im.start + im.types.numel() for im in ims_)
    q_end = next(x for x in range(last_, len(ids_)) if ids_[x] == A_ID)
    return {"id": rid, "lang": "ja", "cat": "vqa", "sub": "t", "thinking_mode": "chat", "messages": messages,
            "vl_ids": ids_, "token_types": tt_, "score_from": spans_[0][0], "prompt_len": spans_[0][0], "spans": spans_,
            "q_spans": [[last_, q_end]] if q_end > last_ else [],
            "image_spans": [{"start": im.start, "len": int(im.types.numel()), "n_vit_h": im.n_vit_h, "n_vit_w": im.n_vit_w}
                            for im in ims_]}


msgs5 = [user("この画像に何が写っていますか？", "a.png"), {"role": "assistant", "content": "四角形がいくつかあります。"}]
row5 = bs_row(msgs5)
c5 = V.prepare_vl_case(row5, tok, args)
check(c5["ids"] == row5["vl_ids"] and c5["token_types"] == row5["token_types"] and c5["score_from"] == row5["score_from"]
      and c5["answer_source"] == "given", "vl_ids の行: 公式経路で描画し直した ids / types と一致し、行の score_from で採点する")
check(c5["ids"][c5["score_from"] - 1] == A_ID and tok.decode(c5["ids"][c5["score_from"]:]).endswith("<｜end▁of▁sentence｜>"),
      "vl_ids の行: 採点は <｜Assistant｜> の次から EOS まで (build_calib と同じ規則)")
c5m = V.prepare_vl_case({"id": "m", "messages": msgs5}, tok, args)
check(c5m["score_from"] is not None and c5m["ids"] == row5["vl_ids"] and c5m["answer_source"] == "given"
      and c5m["score_from"] == c5m["prompt_len"],
      "最後が assistant の会話 (answer 無し) は、その発話を回答として採点する (続きを生成しない)")
check(c5m["score_from"] == c5["score_from"] + 1, "answer 経路は </think> の後ろから、build_sets の行は </think> の予測も含む (1 トークン差)")
bad5 = dict(row5, vl_ids=row5["vl_ids"][:-2] + [5, row5["vl_ids"][-1]])
check(raises(lambda: V.prepare_vl_case(bad5, tok, args), ValueError), "vl_ids が描画と違えば止まる")
check(raises(lambda: V.prepare_vl_case(dict(row5, score_from=row5["score_from"] + 1), tok, args), ValueError),
      "score_from が最初の <｜Assistant｜> の次でなければ止まる")
check(raises(lambda: V.prepare_vl_case(dict(row5, image_spans=[dict(row5["image_spans"][0], start=3)]), tok, args), ValueError),
      "image_spans が描画と違えば止まる")
# 多ターン (vl_calib の JA-VG と同じ形): score_from は最初の回答、spans は全回答
msgs6 = [user("b.png", "何色？"), {"role": "assistant", "content": "赤です。"}, user("いくつ？"),
         {"role": "assistant", "content": "三つです。"}]
row6 = bs_row(msgs6, "multi6")
c6 = V.prepare_vl_case(row6, tok, args)
check(c6["ids"] == row6["vl_ids"] and c6["score_from"] == row6["spans"][0][0] and len(c6["spans"]) == 2
      and c6["spans"][-1][1] == len(c6["ids"]), "多ターンの行: score_from = 最初の回答、spans を rec に持つ")
c6q = V.apply_score_q(dict(c6))
check(c6q["score_from"] == row6["q_spans"][0][0] and c6q["answer_from"] == row6["score_from"], "--score-q: q_spans の先頭から採点")
jl5 = os.path.join(ROOT, "bs.jsonl")
with open(jl5, "w") as f:
    for r in (row5, row6):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
cs5 = V.load_vl_cases(jl5, tok, args)
tpt5 = os.path.join(ROOT, "teacher_bs.pt")
tm5 = FakeModel(1)
V.run(tm5, tok, targs, cs5, tpt5)
T5 = torch.load(tpt5, weights_only=False)
check(not any(cl[0] > 0 for cl in tm5.calls) and all(r["answer_source"] == "given" for r in T5["recs"])
      and T5["recs"][1]["spans"] == row6["spans"] and T5["recs"][0]["lang"] == "ja"
      and T5["recs"][0]["top_v"].shape[0] == len(row5["vl_ids"]) - row5["score_from"],
      "vl_ids の行の教師: 生成しない、採点の長さ = len − score_from、spans / lang を rec に残す")
St5 = V.run(FakeModel(1), tok, targs, cs5, "", ref=T5)
check(all(r["ids"] == t["ids"] for r, t in zip(St5, T5["recs"])), "vl_ids の行の学生: 教師と同じ ids で採点")
csq = [V.apply_score_q(dict(c_)) for c_ in cs5]
check(raises(lambda: V.run(FakeModel(1), tok, targs, csq, "", ref=T5), AssertionError), "--score-q の有無が教師と違えば止まる")
ex5 = os.path.join(ROOT, "bs_ds41rt.jsonl")
V.main(["--vl-in", jl5, "--export-openai", ex5])
b5 = [json.loads(l)["body"] for l in open(ex5)]
check(all(b_["messages"][-1]["role"] == "user" for b_ in b5), "--export-openai: 最後の assistant (参照回答) は送らない")
check(all(b_["thinking"] == {"type": "disabled"} for b_ in b5 + [x["body"] for x in bodies[:2]])
      and bodies[2]["body"]["thinking"] == {"type": "enabled"}
      and V.to_openai_request(dict(row5, thinking_mode="thinking"), ROOT)["thinking"] == {"type": "enabled"},
      "--export-openai: thinking_mode を明示 (ds41rt の既定は思考あり)")
# 実データ (work/vl を /v にマウントしたとき): 全行が描画と一致して読める
for name in ("vl_kl.jsonl", "vl_calib.jsonl"):
    pth = os.path.join("/v", name)
    if not os.path.exists(pth):
        print(f"  (skip: {pth} が無い)", flush=True)
        continue
    rows_ = [json.loads(l) for l in open(pth) if l.strip()]
    cs_ = V.load_vl_cases(pth, tok, args)
    check(len(cs_) == len(rows_) and all(c_["ids"] == r_["vl_ids"] and c_["score_from"] == r_["score_from"]
                                         and c_["answer_source"] == "given" for c_, r_ in zip(cs_, rows_)),
          f"{name}: {len(rows_)} 行すべてで ids / score_from が行と一致 (生成待ちの行なし)")

print(f"\nALL OK ({len(OK)} checks)", flush=True)
