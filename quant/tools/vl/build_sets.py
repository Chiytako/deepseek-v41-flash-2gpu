"""画像の評価セット (教師 KL 用 / ds41rt 生成評価用) と QAD 校正用の画像系列を work/vl/ に作る。

出力 (すべて work/vl/):
  vl_kl.jsonl      教師 KL 用 (日本語が半分以上)。messages + 展開済み ids/token_types/image_spans/score_from
  vl_gen.jsonl     ds41rt 生成評価用の選択式・短答 (tools/vl_eval.py が読む)
  vl_calib.jsonl   QAD 校正用の画像入り系列 (calib*.jsonl と同じ列 + 画像の列)
  img/<出典>/...   画像 (公開データは元のバイト列のまま、自作は PNG)
  sources.json     出典・ライセンス・件数、作成スクリプトの sha256

画像の扱いは参照実装に合わせる: encoding.py の encode_messages(return_multi_modal_data=True) でプロンプトを作り、
<｜deepseek_image｜> (id 129264) 1 個を image_processor.py と同じ規則の [START] + ([IMAGE]*w + [NEWLINE])*h + [END] に展開する
(patch 14、downsample 3、最大 1024 トークン、最小 544x544 画素)。ids は ref の prepare_vl_inputs の出力と同じになる想定で、
ハーネス側で prepare_vl_inputs を使って作り直した場合と一致するかは GPU 段で確認する (同じ ids なら assert が通る)。

vl_kl.jsonl / vl_calib.jsonl の列:
  id, set, lang (ja|en), cat, sub, source, license, ref (元データの位置)
  messages       OpenAI 形式 (image_url.url は work/vl からの相対パス)。thinking_mode="chat" で encode_case に渡せる
  vl_ids         展開済みトークン列 (画像位置はすべて 129264)。わざと "ids" にしていない: 画像を扱えない今の
                 stream_ref.py / capture_moe.py に渡すと、画像位置がテキストの埋め込みと text 用 bias で処理されて
                 黙って誤るため、KeyError / TypeError で止まるようにした
  token_types    -1=テキスト、0=START、1=IMAGE、2=NEWLINE、3=END (ref image_processor.py と同じ)
  image_spans    [{start, len, path, width, height, n_llm_h, n_llm_w, n_vit_h, n_vit_w}]
                 → ImageInput(start, load_image(path) の patches, n_vit_h, n_vit_w, types) を作る材料
  score_from     採点の開始位置 (最初の assistant 区間の先頭)。spans は assistant 区間 [a, b) のすべて
  q_spans        補助の採点区間 (画像の直後の user の問い)。prompt_len は教師に自分で答えさせる場合の切り位置
  n_tok, n_img_tok, ja_tok (採点区間のうち仮名・漢字を含むトークンの割合)
ids は tools/vl/check_ref_ids.py で ref の encode_case + prepare_vl_inputs と一致することを確かめた (CPU コンテナ)。

使い方: tools/.venv/bin/python tools/vl/build_sets.py [--no-render]  (CPU のみ、数分。Firefox headless を使う)
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402

D = DSV41
OUT = D + "/work/vl"
RAW = OUT + "/raw"
MODEL = MODELS + "/DeepSeek-V4.1-Flash-1017"
SEED = 20261018
IMAGE_TOKEN_ID = 129264
TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = -1, 0, 1, 2, 3
PATCH, DOWN, MAX_TOK, MIN_PIX = 14, 3, 1024, 295936  # ref/inference_config.json:57-61

LIC = {
    "javlm": ("SakanaAI/JA-VLM-Bench-In-the-Wild", "画像: Unsplash License (販売・競合サービス不可)。問いと参照回答は GPT-4V 補助 + 人手", "https://huggingface.co/datasets/SakanaAI/JA-VLM-Bench-In-the-Wild"),
    "heron": ("turing-motors/Japanese-Heron-Bench", "画像ごと (LICENCE.md): Public Domain / CC0 1.0 / CC BY 2.0 のものだけ使用。参照回答は GPT-4 (answers_gpt4.jsonl)", "https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench"),
    "javg": ("SakanaAI/JA-VG-VQA-500", "CC BY 4.0 (Japanese Visual Genome VQA / Visual Genome)", "https://huggingface.co/datasets/SakanaAI/JA-VG-VQA-500"),
    "mecha": ("llm-jp/MECHA-ja", "Apache-2.0", "https://huggingface.co/datasets/llm-jp/MECHA-ja"),
    "jmmmu": ("JMMMU/JMMMU", "MIT (データセット)。画像は行ごとの image_1_license (CC BY / CC BY-SA / CC0 / PD のみ使用、著作権表示のみの画像は除外)", "https://huggingface.co/datasets/JMMMU/JMMMU"),
    "ccocr": ("wulipc/CC-OCR (multi_lan_ocr/Japanese)", "MIT", "https://huggingface.co/datasets/wulipc/CC-OCR"),
    "chartqa": ("lmms-lab/ChartQA (test)", "ChartQA 本体は GPL-3.0 (github.com/vis-nlp/ChartQA)。図は Statista/Pew/OWID 由来。内部評価のみ・再配布しない", "https://huggingface.co/datasets/lmms-lab/ChartQA"),
    "docvqa": ("lmms-lab/DocVQA (validation)", "HF のタグは Apache-2.0。文書画像は UCSF Industry Documents Library (DocVQA の利用条件: 研究目的)。内部評価のみ", "https://huggingface.co/datasets/lmms-lab/DocVQA"),
    "textvqa": ("lmms-lab/textvqa (validation 先頭 200 行)", "注釈 CC BY 4.0 (TextVQA)、画像は OpenImages (Flickr、CC BY 2.0)", "https://huggingface.co/datasets/lmms-lab/textvqa"),
    "mmbench": ("lmms-lab/MMBench (en dev)", "MMBench は Apache-2.0 (open-compass)。画像は各元データ (ScienceQA ほか) の条件に従う。内部評価のみ", "https://huggingface.co/datasets/lmms-lab/MMBench"),
    "synth": ("自作 (tools/vl/synth.py で描画)", "自作 (制約なし)。フォントは Noto CJK / DejaVu (SIL OFL / 自由)", ""),
}


# ------------------------------------------------------------------------------------------- image grid (ref と同じ規則)
def image_size(data):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        return im.width, im.height


def plan_grid(width, height):
    """ref/image_processor.py plan_image_grid + safe_resize (vision_max_wh_ratio=None)。返り値 (n_llm_h, n_llm_w, n_vit_h, n_vit_w)。"""
    if 0 < width * height < MIN_PIX:
        r = (MIN_PIX / (width * height)) ** 0.5
        width, height = int(width * r), int(height * r)
    bw, bh = math.ceil(width / PATCH) * PATCH, math.ceil(height / PATCH) * PATCH

    def grid(h, w):
        return math.ceil((h // PATCH) / DOWN), math.ceil((w // PATCH) / DOWN)

    def ntok(h, w):
        return h * (w + 1) + 2
    lh, lw = grid(bh, bw)
    if ntok(lh, lw) > MAX_TOK:
        r = height / width
        mw = math.sqrt((MAX_TOK - 2) / r + 0.25) - 0.5
        mh = mw * r
        cell = PATCH * DOWN
        if mw < 1.0:
            bh, bw = (MAX_TOK - 2) // 2 * cell, cell
        elif mh < 1.0:
            bh, bw = cell, (MAX_TOK - 3) * cell
        else:
            beta = min(math.floor(mw) * cell / width, math.floor(mh) * cell / height)
            bh, bw = math.floor(height * beta / PATCH) * PATCH, math.floor(width * beta / PATCH) * PATCH
        lh, lw = grid(bh, bw)
        assert ntok(lh, lw) <= MAX_TOK
    return lh, lw, bh // PATCH, bw // PATCH


class Enc:
    def __init__(self):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(MODEL + "/tokenizer.json")
        spec = importlib.util.spec_from_file_location("v41_encoding", MODEL + "/encoding/encoding.py")
        self.enc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.enc)
        assert self.tok.token_to_id(self.enc.IMAGE_PLACEHOLDER) == IMAGE_TOKEN_ID
        self.A = self.tok.token_to_id("<｜Assistant｜>")
        self.U = self.tok.token_to_id("<｜User｜>")
        self.E = self.tok.token_to_id("<｜end▁of▁sentence｜>")
        n = self.tok.get_vocab_size()
        self.cjk = [False] * n
        for i in range(n):
            t = self.tok.decode([i], skip_special_tokens=False)
            self.cjk[i] = any(0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF for c in t)

    def render(self, messages, thinking_mode="chat"):
        """messages の image_url は OUT からの相対パス。返り値 dict(ids, token_types, image_spans, spans)。"""
        text, mm = self.enc.encode_messages(messages, thinking_mode=thinking_mode, return_multi_modal_data=True)
        raw = self.tok.encode(text, add_special_tokens=False).ids
        imgs = mm["images"]
        assert sum(t == IMAGE_TOKEN_ID for t in raw) == len(imgs), (len(imgs), text[:200])
        ids, types, spans_img = [], [], []
        it = iter(imgs)
        for t in raw:
            if t != IMAGE_TOKEN_ID:
                ids.append(t)
                types.append(TEXT)
                continue
            rec = next(it)
            path = rec["url"]
            w, h = image_size(open(os.path.join(OUT, path), "rb").read())
            lh, lw, vh, vw = plan_grid(w, h)
            ty = [IMAGE_START] + ([IMAGE] * lw + [IMAGE_NEW_LINE]) * lh + [IMAGE_END]
            spans_img.append({"start": len(ids), "len": len(ty), "path": path, "width": w, "height": h,
                              "n_llm_h": lh, "n_llm_w": lw, "n_vit_h": vh, "n_vit_w": vw})
            ids += [IMAGE_TOKEN_ID] * len(ty)
            types += ty
        # assistant spans (build_calib.Env.render と同じ規則: <｜Assistant｜> の次から EOS まで)
        out, inside, a = [], False, 0
        for i, t in enumerate(ids):
            if t == self.U and inside:
                out.append([a, i])
                inside = False
            if t == self.A:
                inside, a = True, i + 1
            elif t == self.E and inside:
                out.append([a, i + 1])
                inside = False
        if inside:
            out.append([a, len(ids)])
        return {"ids": ids, "token_types": types, "image_spans": spans_img, "spans": [s for s in out if s[1] > s[0]]}


# ------------------------------------------------------------------------------------------- helpers
def sha(b):
    return hashlib.sha256(b).hexdigest()


def save_img(sub, name, data):
    ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg" if data[:2] == b"\xff\xd8" else ".img"
    rel = f"img/{sub}/{name}{ext}"
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    return rel


def user_msg(images, text):
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": p}} for p in images] + [{"type": "text", "text": text}]}


def lit(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return x


def read_pq(path, cols=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=cols).to_pylist()


SUFFIX_JA = "\n答えだけを簡潔に答えてください。"
SUFFIX_EN = "\nAnswer the question using a single word or phrase."


def mcq_text(q, opts, lang, labels="ABCDEFGHIJ"):
    body = "\n".join(f"{labels[i]}. {o}" for i, o in enumerate(opts))
    if lang == "ja":
        return f"{q}\n\n選択肢:\n{body}\n\n選択肢の記号（{', '.join(labels[:len(opts)])}）だけを答えてください。"
    return f"{q}\n{body}\nAnswer with the option's letter from the given choices directly."


# ------------------------------------------------------------------------------------------- public sources
def load_public(rng):
    """返り値 {src: [record]}。record は dict(key, img_bytes(list), q, a, ...) で、key は画像の同一性 (分割の単位)。"""
    P = {}
    # JA-VLM-Bench-In-the-Wild (50 問 / 42 画像): 自由記述 + 参照回答
    rows = read_pq(RAW + "/SakanaAI__JA-VLM-Bench-In-the-Wild/data/test-00000-of-00001.parquet")
    P["javlm"] = [dict(key=sha(r["image"]["bytes"]), imgs=[r["image"]["bytes"]], q=r["question"], a=r["answer"], src=r["page_url"]) for r in rows]
    # Heron-Bench: ライセンスの明確な画像だけ
    ok = {3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21}
    hb = RAW + "/turing-motors__Japanese-Heron-Bench/"
    qs = [json.loads(l) for l in open(hb + "questions_ja.jsonl")]
    ans = {json.loads(l)["question_id"]: json.loads(l) for l in open(hb + "answers_gpt4.jsonl")}
    P["heron"] = []
    for q in qs:
        n = int(q["image"][:3])
        if n not in ok:
            continue
        P["heron"].append(dict(key=f"heron-{n:03d}", imgs=[open(hb + "images/" + q["image"], "rb").read()], q=q["text"],
                               a=ans[q["question_id"]]["text"], src=f"Heron-Bench question_id={q['question_id']} image={q['image']} ({q['category']})"))
    # JA-VG-VQA-500
    rows = read_pq(RAW + "/SakanaAI__JA-VG-VQA-500/data/test-00000-of-00001.parquet")
    P["javg"] = [dict(key=f"vg-{r['image_id']}", imgs=[r["image"]["bytes"]], qas=lit(r["qas"]), src=f"VG image_id={r['image_id']}") for r in rows]
    # MECHA-ja (default, shard 0)
    rows = read_pq(RAW + "/llm-jp__MECHA-ja/data/test-00000-of-00002.parquet")
    P["mecha"] = [dict(key=sha(r["image"]["bytes"]), imgs=[r["image"]["bytes"]], q=r["question"], bg=r["background_text"] or "",
                       opts=list(r["options"]), ans=int(r["answer"]), expl=r["explanation"] or "", src=f"MECHA-ja id={r['id']}") for r in rows
                  if r["image"] and r["image"].get("bytes")]
    # JMMMU (Japanese_Heritage + Computer_Science)、画像 1 枚・選択式・ライセンス明確なものだけ
    P["jmmmu"] = []
    for sub in ("Japanese_Heritage", "Computer_Science"):
        for r in read_pq(RAW + f"/JMMMU__JMMMU/{sub}/test-00000-of-00001.parquet"):
            lic = r["image_1_license"] or ""
            if r["question_type"] != "multiple-choice" or r["image_2"] is not None or not r["image_1"]:
                continue
            if not (lic.startswith("CC") or lic.startswith("Public Domain")):
                continue
            P["jmmmu"].append(dict(key=r["id"], imgs=[r["image_1"]["bytes"]], q=r["question"].replace("<image 1>", "画像").strip(),
                                   opts=lit(r["options"]), ans="ABCDEFGHIJ".index(r["answer"]), expl=r["explanation"] if r["explanation"] not in (None, "None") else "",
                                   src=f"JMMMU {r['id']} (image: {lic}, {r['image_1_attribution']})", sub=sub))
    # CC-OCR 日本語
    csv.field_size_limit(1 << 30)
    import base64
    P["ccocr"] = []
    with open(RAW + "/wulipc__CC-OCR/multi_lan_ocr/Japanese/Japanese_150.tsv", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            P["ccocr"].append(dict(key=r["image_name"], imgs=[base64.b64decode(r["image"])], a=r["answer"], src=f"CC-OCR Japanese {r['image_name']}"))
    # ChartQA test
    rows = read_pq(RAW + "/lmms-lab__ChartQA/data/test-00000-of-00001.parquet")
    P["chartqa"] = [dict(key=sha(r["image"]["bytes"]), imgs=[r["image"]["bytes"]], q=r["question"], a=r["answer"], src=f"ChartQA test #{i} ({r['type']})") for i, r in enumerate(rows)]
    # DocVQA validation shard 0
    rows = read_pq(RAW + "/lmms-lab__DocVQA/DocVQA/validation-00000-of-00006.parquet")
    P["docvqa"] = [dict(key=f"doc-{r['docId']}", imgs=[r["image"]["bytes"]], q=r["question"], answers=lit(r["answers"]),
                        src=f"DocVQA val questionId={r['questionId']} ucsf={r['ucsf_document_id']} p{r['ucsf_document_page_no']}") for r in rows]
    # TextVQA validation (先頭 200 行)
    rows = read_pq(RAW + "/lmms-lab__textvqa/validation-00000-rg0-1.parquet")
    P["textvqa"] = [dict(key=f"tv-{r['image_id']}", imgs=[r["image"]["bytes"]], q=r["question"], answers=lit(r["answers"]),
                         src=f"TextVQA val question_id={r['question_id']} ({r['flickr_original_url']})") for r in rows]
    # MMBench en dev
    rows = read_pq(RAW + "/lmms-lab__MMBench/en/dev-00000-of-00001.parquet")
    P["mmbench"] = []
    for r in rows:
        opts = [r[k] for k in "ABCD" if r[k] not in (None, "nan", "")]
        hint = r["hint"] if r["hint"] not in (None, "nan") else ""
        P["mmbench"].append(dict(key=sha(r["image"]["bytes"]), imgs=[r["image"]["bytes"]], q=(hint + "\n" + r["question"]).strip(), opts=opts,
                                 ans="ABCD".index(r["answer"]), src=f"MMBench en dev index={r['index']} ({r['category']}, {r['source']})", cat=r["category"]))
    return P


def split_by_key(rng, recs, n_gen, n_kl, n_cal, filt_gen=None):
    """画像 (key) 単位で gen / kl / calib を互いに素に選ぶ。key ごとに 1 件だけ取る。"""
    keys = sorted({r["key"] for r in recs})
    rng.shuffle(keys)
    by = {}
    for r in recs:
        by.setdefault(r["key"], []).append(r)
    used = set()
    out = {"gen": [], "kl": [], "calib": []}
    for part, n, f in (("gen", n_gen, filt_gen), ("kl", n_kl, None), ("calib", n_cal, None)):
        for k in keys:
            if len(out[part]) >= n:
                break
            if k in used:
                continue
            cands = [r for r in by[k] if f is None or f(r)]
            if not cands:
                continue
            out[part].append(rng.choice(cands))
            used.add(k)
        assert len(out[part]) == n, (part, n, len(out[part]))
    return out


# JA-VG-VQA の短答: 色・数を問う問いに限る (正解の表記ゆれが小さい)
def javg_pick(qas):
    good = []
    for qa in qas:
        q, a = qa["question"], qa["answer"].strip()
        if len(a) > 6:
            continue
        if any(k in q for k in ("何色", "何人", "いくつ", "何本", "何台", "何匹", "何羽", "何頭", "何個", "何枚")):
            good.append(qa)
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true", help="自作画像の描画を省く (既存の PNG を使う)")
    a = ap.parse_args()
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    enc = Enc()
    P = load_public(rng)
    print({k: len(v) for k, v in P.items()}, flush=True)

    gen, kl, cal = [], [], []

    def gen_item(src, lang, cat, sub, rec, images, prompt, answer, atype, **kw):
        it = dict(id=f"{sub}-{len([g for g in gen if g['sub'] == sub]):03d}", set="gen", lang=lang, cat=cat, sub=sub, source=src,
                  license=LIC[src][1], ref=rec.get("src", ""), images=images, prompt=prompt, answer=answer, atype=atype)
        it.update(kw)
        gen.append(it)

    def seq_item(store, src, lang, cat, sub, ref, messages):
        r = enc.render(messages)
        ids = r["ids"]
        img_tok = sum(s["len"] for s in r["image_spans"])
        sc = [i for s in r["spans"] for i in range(s[0], s[1])]
        # 補助の採点区間: 最初の画像より後ろにある user のテキスト (問い) 。画像の表現が後続テキストへ伝わる度合いを見る
        last = max(s_["start"] + s_["len"] for s_ in r["image_spans"])
        q_end = next(i for i in range(last, len(ids)) if ids[i] == enc.A)
        q_spans = [[last, q_end]] if q_end > last else []
        it = dict(id=f"{store_name[id(store)]}-{sub}-{len([x for x in store if x['sub'] == sub]):03d}", set=store_name[id(store)], lang=lang, cat=cat, sub=sub,
                  source=src, license=LIC[src][1], ref=ref, thinking_mode="chat", messages=messages,
                  images=[s["path"] for s in r["image_spans"]], vl_ids=ids, token_types=r["token_types"], image_spans=r["image_spans"],
                  score_from=r["spans"][0][0], spans=r["spans"], q_spans=q_spans, prompt_len=r["spans"][0][0], n_tok=len(ids), n_img_tok=img_tok,
                  ja_tok=round(sum(enc.cjk[ids[i]] for i in sc) / max(1, len(sc)), 4))
        store.append(it)

    store_name = {id(kl): "kl", id(cal): "calib"}

    # ---------------------------------------------------------------- 公開データ
    # JA-VLM (gen 0 / kl 6 / calib 0; 42 画像しかなく自由記述なので KL 専用)
    s = split_by_key(rng, P["javlm"], 0, 6, 0)
    for r in s["kl"]:
        im = save_img("javlm", "kl-" + r["key"][:12], r["imgs"][0])
        seq_item(kl, "javlm", "ja", "vqa", "javlm", r["src"], [user_msg([im], r["q"]), {"role": "assistant", "content": r["a"]}])
    # Heron (kl 4、detail/complex を優先)
    s = split_by_key(rng, P["heron"], 0, 4, 0)
    for r in s["kl"]:
        im = save_img("heron", r["key"], r["imgs"][0])
        seq_item(kl, "heron", "ja", "vqa", "heron", r["src"], [user_msg([im], r["q"]), {"role": "assistant", "content": r["a"]}])
    # JA-VG-VQA (gen 20 短答 / calib 4 複数ターン)
    s = split_by_key(rng, P["javg"], 20, 0, 4, filt_gen=lambda r: bool(javg_pick(r["qas"])))
    for r in s["gen"]:
        qa = rng.choice(javg_pick(r["qas"]))
        im = save_img("javg", r["key"], r["imgs"][0])
        gen_item("javg", "ja", "vqa", "javg", r, [im], qa["question"] + "\n単語または短い句で答えてください。", qa["answer"].strip(), "ja_short")
    for r in s["calib"]:
        im = save_img("javg", r["key"], r["imgs"][0])
        qas = r["qas"][:6]
        msgs = [user_msg([im], qas[0]["question"]), {"role": "assistant", "content": qas[0]["answer"]}]
        for qa in qas[1:]:
            msgs += [{"role": "user", "content": qa["question"]}, {"role": "assistant", "content": qa["answer"]}]
        seq_item(cal, "javg", "ja", "vqa", "javg", r["src"], msgs)
    # MECHA-ja (gen 30 / kl 2 / calib 3)
    s = split_by_key(rng, P["mecha"], 30, 2, 2)
    for part in ("gen", "kl", "calib"):
        for r in s[part]:
            im = save_img("mecha", r["key"][:16], r["imgs"][0])
            q = (r["bg"] + "\n" + r["q"]).strip()
            if part == "gen":
                gen_item("mecha", "ja", "knowledge", "mecha", r, [im], mcq_text(q, r["opts"], "ja"), "ABCD"[r["ans"]], "mcq", options=r["opts"])
            else:
                body = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(r["opts"]))
                ans = f"正解は {'ABCD'[r['ans']]}（{r['opts'][r['ans']]}）です。{r['expl']}"
                seq_item(kl if part == "kl" else cal, "mecha", "ja", "knowledge", "mecha", r["src"],
                         [user_msg([im], f"{q}\n\n選択肢:\n{body}\n\n理由も説明してください。"), {"role": "assistant", "content": ans}])
    # JMMMU (gen 20)
    s = split_by_key(rng, P["jmmmu"], 20, 0, 0)
    for r in s["gen"]:
        im = save_img("jmmmu", r["key"], r["imgs"][0])
        gen_item("jmmmu", "ja", "knowledge", "jmmmu", r, [im], mcq_text(r["q"], r["opts"], "ja"), "ABCDEFGHIJ"[r["ans"]], "mcq", options=r["opts"])
    # CC-OCR 日本語 (gen 10 / kl 3 / calib 3)
    s = split_by_key(rng, P["ccocr"], 10, 3, 2)
    OCR_Q = "画像に写っている文字をすべて書き出してください。説明は不要です。"
    for part in ("gen", "kl", "calib"):
        for r in s[part]:
            im = save_img("ccocr", r["key"].rsplit(".", 1)[0], r["imgs"][0])
            if part == "gen":
                gen_item("ccocr", "ja", "ocr", "ccocr", r, [im], OCR_Q, r["a"], "ned", max_tokens=512)
            else:
                seq_item(kl if part == "kl" else cal, "ccocr", "ja", "ocr", "ccocr", r["src"], [user_msg([im], OCR_Q), {"role": "assistant", "content": r["a"]}])
    # ChartQA (gen 20 / kl 4 / calib 2)
    s = split_by_key(rng, P["chartqa"], 20, 4, 2)
    for part in ("gen", "kl", "calib"):
        for r in s[part]:
            im = save_img("chartqa", r["key"][:16], r["imgs"][0])
            if part == "gen":
                gen_item("chartqa", "en", "chart", "chartqa", r, [im], r["q"] + SUFFIX_EN, r["a"], "relaxed")
            else:
                seq_item(kl if part == "kl" else cal, "chartqa", "en", "chart", "chartqa", r["src"],
                         [user_msg([im], r["q"]), {"role": "assistant", "content": f"The answer is {r['a']}."}])
    # DocVQA (gen 20 / kl 4 / calib 2)
    s = split_by_key(rng, P["docvqa"], 20, 4, 1)
    for part in ("gen", "kl", "calib"):
        for r in s[part]:
            im = save_img("docvqa", r["key"], r["imgs"][0])
            if part == "gen":
                gen_item("docvqa", "en", "doc", "docvqa", r, [im], r["q"] + SUFFIX_EN, r["answers"][0], "anls", answers=r["answers"])
            else:
                seq_item(kl if part == "kl" else cal, "docvqa", "en", "doc", "docvqa", r["src"],
                         [user_msg([im], r["q"]), {"role": "assistant", "content": f"The answer is {r['answers'][0]}."}])
    # TextVQA (gen 20 / kl 3 / calib 2)
    s = split_by_key(rng, P["textvqa"], 20, 3, 1)
    for part in ("gen", "kl", "calib"):
        for r in s[part]:
            im = save_img("textvqa", r["key"], r["imgs"][0])
            maj = max(sorted(set(r["answers"])), key=r["answers"].count)
            if part == "gen":
                gen_item("textvqa", "en", "vqa", "textvqa", r, [im], r["q"] + SUFFIX_EN, maj, "vqa", answers=r["answers"])
            else:
                seq_item(kl if part == "kl" else cal, "textvqa", "en", "vqa", "textvqa", r["src"],
                         [user_msg([im], r["q"]), {"role": "assistant", "content": f"The answer is {maj}."}])
    # MMBench (gen 20 / kl 3 / calib 2)
    s = split_by_key(rng, P["mmbench"], 20, 3, 2)
    for part in ("gen", "kl", "calib"):
        for r in s[part]:
            im = save_img("mmbench", r["key"][:16], r["imgs"][0])
            if part == "gen":
                gen_item("mmbench", "en", "mcq", "mmbench", r, [im], mcq_text(r["q"], r["opts"], "en"), "ABCD"[r["ans"]], "mcq", options=r["opts"])
            else:
                seq_item(kl if part == "kl" else cal, "mmbench", "en", "mcq", "mmbench", r["src"],
                         [user_msg([im], mcq_text(r["q"], r["opts"], "en")), {"role": "assistant", "content": f"{'ABCD'[r['ans']]}. {r['opts'][r['ans']]}"}])

    # ---------------------------------------------------------------- 自作
    R = None if a.no_render else synth.Renderer(OUT + "/.render")
    T = synth.TEMPLATES

    def make(split, kinds, lang, n, seed):
        """kinds: TEMPLATES のキー。テンプレートを順に回して n 個作る。"""
        r = random.Random(seed)
        fns = [f for k in kinds for f in T[k]]
        if split == "calib":
            r.shuffle(fns)
        outs = []
        for i in range(n):
            fn = fns[i % len(fns)]
            inst = fn(r, lang)
            key = f"{split}-{lang}-{inst.kind}-{i:03d}"
            rel = f"img/synth/{split}/{key}.png"
            if R is not None:
                R.render(inst, os.path.join(OUT, rel))
            else:
                inst.fig = None
            outs.append((inst, rel, r))
        return outs

    CAT = {"chart": "chart", "ocr": "ocr", "screenshot": "screenshot"}

    def add_gen(outs, sub_prefix):
        for inst, rel, r in outs:
            qa = r.choice(inst.qas)
            sfx = SUFFIX_JA if inst.lang == "ja" else SUFFIX_EN
            gen_item("synth", inst.lang, CAT[inst.cat], f"synth-{inst.kind}", {"src": f"synth {inst.kind}"}, [rel], qa["q"] + sfx, qa["a"],
                     qa["atype"], aliases=qa.get("aliases", []))

    def add_seq(store, outs):
        for inst, rel, r in outs:
            seq_item(store, "synth", inst.lang, CAT[inst.cat], f"synth-{inst.kind}", f"synth {inst.kind}",
                     [user_msg([rel], inst.kl_q), {"role": "assistant", "content": inst.kl_ref}])

    # 生成評価: ja 図表 20 / 文書 OCR 20 / スクショ 25 (code 8, terminal 9, web 8)、en スクショ 20 (code 6, terminal 7, web 7)
    add_gen(make("gen", ["chart"], "ja", 20, SEED + 1), "")
    add_gen(make("gen", ["ocr-ja"], "ja", 20, SEED + 2), "")
    add_gen(make("gen", ["code"], "ja", 8, SEED + 3), "")
    add_gen(make("gen", ["terminal"], "ja", 9, SEED + 4), "")
    add_gen(make("gen", ["web"], "ja", 8, SEED + 5), "")
    add_gen(make("gen", ["code"], "en", 6, SEED + 6), "")
    add_gen(make("gen", ["terminal"], "en", 7, SEED + 7), "")
    add_gen(make("gen", ["web"], "en", 7, SEED + 8), "")
    # KL: ja 図表 3 / OCR 3 / code 2 / terminal 3 / web 2、en 図表 1 / invoice 1 / code 2 / terminal 2 / web 2
    kl_chart = make("kl", ["chart"], "ja", 4, SEED + 11)
    add_seq(kl, [kl_chart[0], kl_chart[2], kl_chart[3]])  # 棒・折れ線・円
    add_seq(kl, make("kl", ["ocr-ja"], "ja", 3, SEED + 12))
    add_seq(kl, make("kl", ["code"], "ja", 2, SEED + 13))
    kl_term = make("kl", ["terminal"], "ja", 4, SEED + 14)
    add_seq(kl, [kl_term[0], kl_term[2], kl_term[3]])  # pytest, git(日本語), CLI(日本語)
    add_seq(kl, make("kl", ["web"], "ja", 2, SEED + 15))
    add_seq(kl, make("kl", ["chart"], "en", 1, SEED + 16))
    add_seq(kl, make("kl", ["ocr-en"], "en", 1, SEED + 17))
    add_seq(kl, make("kl", ["code"], "en", 2, SEED + 18))
    add_seq(kl, make("kl", ["terminal"], "en", 2, SEED + 19))
    add_seq(kl, make("kl", ["web"], "en", 2, SEED + 20))
    # 校正: ja 図表 2 / OCR 2 / スクショ 3 / 複数画像 (2 枚) 1、en スクショ 2。合計 ~1.25 万トークン (SPEC §2.1 の画像 2% = 1.2 万)
    add_seq(cal, make("calib", ["chart"], "ja", 2, SEED + 21))
    add_seq(cal, make("calib", ["ocr-ja"], "ja", 2, SEED + 22))
    add_seq(cal, make("calib", ["code", "terminal", "web"], "ja", 3, SEED + 23))
    add_seq(cal, make("calib", ["code", "terminal", "web"], "en", 2, SEED + 24))
    for lang, n, sd in (("ja", 1, SEED + 25),):
        pairs = make("calib", ["terminal", "web"], lang, 2 * n, sd)
        for k in range(n):
            (i1, p1, _), (i2, p2, _) = pairs[2 * k], pairs[2 * k + 1]
            if lang == "ja":
                q = "2 枚のスクリーンショットについて、それぞれ何が表示されているかを説明してください。"
                ans = f"1 枚目: {i1.kl_ref}\n\n2 枚目: {i2.kl_ref}"
            else:
                q = "Describe what each of the two screenshots shows."
                ans = f"First screenshot: {i1.kl_ref}\n\nSecond screenshot: {i2.kl_ref}"
            seq_item(cal, "synth", lang, "screenshot", "synth-multi", f"synth {i1.kind}+{i2.kind}",
                     [user_msg([p1, p2], q), {"role": "assistant", "content": ans}])
    if R is not None:
        R.close()

    # ---------------------------------------------------------------- 書き出し
    for name, rows in (("vl_gen.jsonl", gen), ("vl_kl.jsonl", kl), ("vl_calib.jsonl", cal)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 画像の重複検査 (セット間で同じ画像ファイル内容が無いこと)
    def hashes(rows):
        return {sha(open(os.path.join(OUT, p), "rb").read()) for r in rows for p in r["images"]}
    hg, hk, hc = hashes(gen), hashes(kl), hashes(cal)
    assert not (hg & hk) and not (hg & hc) and not (hk & hc), "画像がセット間で重複"

    def summ(rows, tok=False):
        d = {}
        for r in rows:
            k = f"{r['lang']}/{r['cat']}/{r['sub']}"
            e = d.setdefault(k, {"n": 0})
            e["n"] += 1
            if tok:
                e["tok"] = e.get("tok", 0) + r["n_tok"]
                e["img_tok"] = e.get("img_tok", 0) + r["n_img_tok"]
                e["scored"] = e.get("scored", 0) + sum(b - a_ for a_, b in r["spans"])
        return dict(sorted(d.items()))
    meta = {
        "schema": "dsv41.vl-sets.v1", "seed": SEED,
        "script": {p: sha(open(p, "rb").read()) for p in (os.path.abspath(__file__), os.path.abspath(synth.__file__))},
        "image_rule": {"patch": PATCH, "downsample": DOWN, "max_tokens": MAX_TOK, "min_pixels": MIN_PIX, "image_token_id": IMAGE_TOKEN_ID},
        "sources": {k: {"dataset": v[0], "license": v[1], "url": v[2]} for k, v in LIC.items()},
        "gen": {"n": len(gen), "ja": sum(r["lang"] == "ja" for r in gen), "by": summ(gen)},
        "kl": {"n": len(kl), "ja": sum(r["lang"] == "ja" for r in kl), "tok": sum(r["n_tok"] for r in kl),
               "img_tok": sum(r["n_img_tok"] for r in kl), "scored": sum(b - a_ for r in kl for a_, b in r["spans"]), "by": summ(kl, True)},
        "calib": {"n": len(cal), "ja": sum(r["lang"] == "ja" for r in cal), "tok": sum(r["n_tok"] for r in cal),
                  "img_tok": sum(r["n_img_tok"] for r in cal), "scored": sum(b - a_ for r in cal for a_, b in r["spans"]), "by": summ(cal, True)},
        "disjoint_image_sha256": True,
    }
    json.dump(meta, open(os.path.join(OUT, "sources.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: meta[k] for k in ("gen", "kl", "calib")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
