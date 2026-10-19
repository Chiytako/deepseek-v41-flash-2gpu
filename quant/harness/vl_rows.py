"""画像付き系列の「行」形式 (dsv41.vl-row.v1) と、その作成・検査・画像の読み込み (capture_moe.py の画像対応で使う)。

ids を前もって展開した行を扱う。会話 (messages) を実行時に展開する教師 / 学生の採点は harness/stream_ref_vl.py の
担当で、本ファイルはそれと独立に動く (stream_ref.py も stream_ref_vl.py も import しない)。

画像経路 (公式 ref/model.py):
  - 画像 1 枚は ids 上で [IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h + [IMAGE_END] の区間になり、
    区間の全位置が image_token_id (129264) を持つ。区別は token_types だけ (image_processor.py:1-10, 161-172)。
  - Transformer.forward(input_ids, start_pos, images, token_types) が区間を ViT/aligner 出力と学習済み区切りで
    上書きし (model.py:1231-1242)、token_types >= 0 の位置は Engram を引かず (DEAD、model.py:1252-1255)、
    MoE の gate は bias の代わりに bias_vl を使う (model.py:820-822)。
  - 画像区間は prefill の 1 チャンク (start_pos 0) に収まる必要がある (model.py:1258)。

行の形式 (jsonl、1 行 1 系列。画像の無い行は stream_ref.py --teacher-in の行と同じ):
  {"ids": [...], "score_from": int, "domain": "vl", "id": "...", "src": "...",
   "images": [{"path": "img/a.png", "start": s, "grid": [h, w], "n_tokens": n, "sha256": "..."}]}
  - ids[start : start+n] はすべて image_token_id。n = h*(w+1)+2。token_types は (start, grid) から再構成する。
  - path は jsonl のあるディレクトリからの相対パス (絶対パス、data: URL も可)。sha256 は画像ファイルのバイト列。
  - 採点に使う行では、画像区間は score_from より前にあること (画像の位置の次トークン予測は意味が無い)。

サブコマンド
  build : OpenAI 形式の会話 (画像 = image_url ブロック、任意で answer) を公式 encoding.py で描画し、行を作る (CPU)
  check : 行を検査する (区間、画像の読み込みと grid、sha256)。CPU

usage (dsv41-quant イメージ):
  python3 vl_rows.py build --cases /c/vl_cases.jsonl --out /c/calib_vl.jsonl
  python3 vl_rows.py check /c/calib_vl.jsonl
"""
import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np

REF = os.environ.get("DSV41_REF", "/ref")
CKPT = os.environ.get("DSV41_CKPT", "/model")
SCHEMA = "dsv41.vl-row.v1"
TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = -1, 0, 1, 2, 3  # image_processor.py:22-23
DEFAULT_IMAGE_TOKEN_ID = 129264


# --------------------------------------------------------------------------- 純粋な補助 (numpy のみ)
def span_types(h, w):
    """image_processor.image_token_types と同じ並び (list[int])。"""
    return [IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h + [IMAGE_END]


def span_len(h, w):
    return h * (w + 1) + 2


def resolve_path(path, base_dir):
    if path.startswith("file://"):
        path = path[len("file://"):]
    if path.startswith(("data:", "http://", "https://")) or os.path.isabs(path) or not base_dir:
        return path
    return os.path.join(base_dir, path)


def normalize_row(row):
    """tools/vl/build_sets.py の行 (vl_ids / token_types / image_spans、ids をわざと持たない) をこの形式に直す。
    image_spans の path は jsonl のあるディレクトリ (work/vl) からの相対。token_types があれば再構成と照合する。
    vl_ids を持たない行はそのまま返す。"""
    if "vl_ids" not in row:
        return row
    r = dict(row)
    r["ids"] = [int(t) for t in row["vl_ids"]]
    r["images"] = [{"path": sp["path"], "start": int(sp["start"]), "grid": [int(sp["n_llm_h"]), int(sp["n_llm_w"])],
                    "n_tokens": int(sp.get("len", span_len(int(sp["n_llm_h"]), int(sp["n_llm_w"])))),
                    **({"sha256": sp["sha256"]} if sp.get("sha256") else {})} for sp in row.get("image_spans") or []]
    if row.get("token_types") is not None:
        tt = row_token_types(r, int(row.get("image_token_id", DEFAULT_IMAGE_TOKEN_ID)))
        if tt.tolist() != [int(t) for t in row["token_types"]]:
            raise ValueError(f"{row.get('id')}: token_types が image_spans からの再構成と違う")
    r.pop("vl_ids")
    return r


def has_images(row):
    return bool(row.get("images"))


def row_token_types(row, image_token_id=DEFAULT_IMAGE_TOKEN_ID):
    """行の token_types (np.int8 [len(ids)]、TEXT = -1)。区間と ids の整合を検査する。"""
    ids = row["ids"]
    n = len(ids)
    tt = np.full(n, TEXT, np.int8)
    for k, im in enumerate(row.get("images") or []):
        s = int(im["start"])
        h, w = (int(v) for v in im["grid"])
        e = s + span_len(h, w)
        if "n_tokens" in im and int(im["n_tokens"]) != e - s:
            raise ValueError(f"image {k}: n_tokens {im['n_tokens']} != grid から {e - s}")
        if s < 0 or e > n:
            raise ValueError(f"image {k}: 区間 [{s}, {e}) が系列長 {n} を超える")
        if (tt[s:e] != TEXT).any():
            raise ValueError(f"image {k}: 区間が他の画像と重なる")
        if any(int(t) != image_token_id for t in ids[s:e]):
            raise ValueError(f"image {k}: 区間 [{s}, {e}) に image_token_id 以外がある")
        tt[s:e] = span_types(h, w)
    ids_a = np.asarray(ids)
    stray = (ids_a == image_token_id) & (tt == TEXT)
    if stray.any():
        raise ValueError(f"画像区間の外に image_token_id がある (位置 {np.flatnonzero(stray)[:5].tolist()})")
    return tt


def image_end(row):
    """最後の画像区間の終わり (画像が無ければ 0)。"""
    return max((int(im["start"]) + span_len(*im["grid"]) for im in row.get("images") or []), default=0)


def check_scored(row):
    sf = int(row.get("score_from", 0))
    if has_images(row) and sf < image_end(row):
        raise ValueError(f"score_from {sf} が画像区間の終わり {image_end(row)} より前 (画像は採点区間の前に置く)")


def image_salt(row, n=None):
    """画像区間の位置ごとの 64bit 値 (画像の sha256 と区間内の位置から)。接頭辞ハッシュに混ぜ、
    ids が同じで画像が違う系列を同じ接頭辞と見なさないために使う。画像が無ければ None。"""
    if not has_images(row):
        return None
    n = len(row["ids"]) if n is None else n
    salt = np.zeros(n, np.uint64)
    for im in row["images"]:
        s = int(im["start"])
        e = min(n, s + span_len(*im["grid"]))
        key = im.get("sha256") or im.get("path", "")
        base = int(hashlib.sha256(str(key).encode()).hexdigest()[:16], 16)
        with np.errstate(over="ignore"):
            salt[s:e] = np.uint64(base) + np.arange(e - s, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        salt[s:e] |= np.uint64(1)  # 0 (テキスト) と区別する
    return salt


def row_digest(row):
    """系列の同一性 (ids と画像の sha256)。画像が無ければ ids だけ (capture_moe の従来の hid と同じ)。"""
    h = hashlib.sha256(np.asarray(row.get("ids") if "ids" in row else row.get("tokens"), np.int64).tobytes())
    if has_images(row):
        for im in row["images"]:
            h.update(f"|{im['start']}:{im.get('sha256') or im.get('path')}".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- 画像の読み込み (torch、ref)
def _ref_imports():
    if REF not in sys.path:
        sys.path.insert(0, REF)
    enc = os.path.join(CKPT, "encoding")
    if enc not in sys.path:
        sys.path.insert(0, enc)


def model_args(ckpt=None):
    """inference/config.json から画像前処理に要る値を持つ args (ModelArgs を作らずに済む)。"""
    import types

    cfg = json.load(open(os.path.join(ckpt or CKPT, "inference", "config.json")))
    a = types.SimpleNamespace(**cfg)
    for k, v in dict(vision_rope_theta=10000.0, vision_max_wh_ratio=None, vision_n_layers=0).items():
        if not hasattr(a, k):
            setattr(a, k, v)
    a.vision_enabled = a.vision_n_layers > 0
    return a


def image_inputs(row, args, base_dir="", verify_sha=True):
    """行の画像を ImageInput (image_processor.py) の list にする。grid と sha256 を照合する。"""
    _ref_imports()
    from image_processor import ImageInput, image_token_types, load_image, load_image_bytes

    out = []
    for k, im in enumerate(row["images"]):
        src = resolve_path(im["path"], base_dir)
        rec = {"url": src}
        if verify_sha and im.get("sha256"):
            got = hashlib.sha256(load_image_bytes(rec)).hexdigest()
            if got != im["sha256"]:
                raise ValueError(f"image {k} ({src}): sha256 が行の記録と違う")
        patches, nvh, nvw, h, w = load_image(rec, args)
        if [h, w] != [int(v) for v in im["grid"]]:
            raise ValueError(f"image {k} ({src}): grid {h}x{w} != 行の記録 {im['grid']} (config か画像が違う)")
        out.append(ImageInput(int(im["start"]), patches, nvh, nvw, image_token_types(h, w)))
    return out


# --------------------------------------------------------------------------- build (CPU)
def _image_record_bytes(rec, base_dir):
    _ref_imports()
    from image_processor import load_image_bytes

    r = dict(rec)
    if isinstance(r.get("url"), str):
        r["url"] = resolve_path(r["url"], base_dir)
    return load_image_bytes(r), r.get("url")


def build_row(case, tok, args, cases_dir="", out_dir="", thinking_mode="chat"):
    """case = {"id", "messages": [OpenAI 形式、画像は image_url ブロック (url はパスか data: URL)],
               "answer": {"content": str, "reasoning_content"?: str} (任意), "domain"?, "thinking_mode"?}
    返り値: 行 dict。answer があれば score_from = 回答の先頭、無ければ len(ids) (capture 専用)。"""
    _ref_imports()
    from encoding import encode_messages
    from image_processor import plan_image_grid
    from PIL import Image
    import io

    tm = case.get("thinking_mode") or thinking_mode
    msgs = copy.deepcopy(case["messages"])
    prompt, mm = encode_messages(copy.deepcopy(msgs), thinking_mode=tm, return_multi_modal_data=True)
    full = prompt
    if case.get("answer"):
        ans = {"role": "assistant", **case["answer"]}
        full, mm2 = encode_messages(copy.deepcopy(msgs) + [ans], thinking_mode=tm, return_multi_modal_data=True)
        if not full.startswith(prompt):
            raise ValueError(f"{case.get('id')}: 回答付きの描画が prompt で始まらない (thinking_mode={tm})")
        if len(mm2["images"]) != len(mm["images"]):
            raise ValueError(f"{case.get('id')}: 回答に画像を入れない")
    image_token_id = int(args.image_token_id)
    ids_p = tok.encode(prompt, add_special_tokens=False)
    ids_f = tok.encode(full, add_special_tokens=False)
    if ids_f[: len(ids_p)] != ids_p:
        raise ValueError(f"{case.get('id')}: 回答付きのトークン列が prompt のトークン列で始まらない")
    recs = mm["images"]
    n_ph = sum(t == image_token_id for t in ids_f)
    if n_ph != len(recs):
        raise ValueError(f"{case.get('id')}: placeholder {n_ph} 個 != 画像 {len(recs)} 枚 "
                         f"(トークナイザが placeholder を {image_token_id} にしていない)")
    ids, images, it = [], [], iter(recs)
    score_from = None
    for j, t in enumerate(ids_f):
        if j == len(ids_p):
            score_from = len(ids)
        if t != image_token_id:
            ids.append(int(t))
            continue
        rec = next(it)
        data, src = _image_record_bytes(rec, cases_dir)
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
        h, w, _, _ = plan_image_grid(width, height, args)
        path = src
        if isinstance(src, str) and not src.startswith(("data:", "http://", "https://")) and out_dir:
            path = os.path.relpath(os.path.abspath(src), os.path.abspath(out_dir))
        images.append({"path": path, "start": len(ids), "grid": [h, w], "n_tokens": span_len(h, w),
                       "sha256": hashlib.sha256(data).hexdigest(), "orig_size": [width, height]})
        ids += [image_token_id] * span_len(h, w)
    if score_from is None:
        score_from = len(ids)
    row = {"schema": SCHEMA, "id": case.get("id"), "src": case.get("src", case.get("id")),
           "domain": case.get("domain", "vl"), "ids": ids, "score_from": score_from, "images": images,
           "n_img_tokens": int(sum(im["n_tokens"] for im in images)), "thinking_mode": tm}
    row_token_types(row, image_token_id)
    if case.get("answer"):
        check_scored(row)
    return row


def cmd_build(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    args = model_args()
    cases_dir = os.path.dirname(os.path.abspath(a.cases))
    out_dir = os.path.dirname(os.path.abspath(a.out))
    n = n_img = n_tok = 0
    with open(a.out + ".tmp", "w") as f:
        for l in open(a.cases):
            if not l.strip():
                continue
            row = build_row(json.loads(l), tok, args, cases_dir, out_dir, a.thinking)
            if len(row["ids"]) > a.max_seq_len:
                print(f"[build] {row['id']}: {len(row['ids'])} tok > {a.max_seq_len}、飛ばす", flush=True)
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            n_img += len(row["images"])
            n_tok += len(row["ids"])
    os.replace(a.out + ".tmp", a.out)
    print(f"[build] {n} 行、画像 {n_img} 枚、{n_tok} tok -> {a.out}", flush=True)


def cmd_check(a):
    args = model_args()
    base = os.path.dirname(os.path.abspath(a.rows))
    n = 0
    for li, l in enumerate(open(a.rows)):
        row = json.loads(l)
        tt = row_token_types(row, int(args.image_token_id))
        if has_images(row):
            check_scored(row)
            if not a.no_load:
                imgs = image_inputs(row, args, base)
                assert all(im.types.numel() == span_len(*r["grid"]) for im, r in zip(imgs, row["images"]))
        n += 1
        print(f"[check] {li}: {row.get('id')} {len(row['ids'])} tok, 画像 {len(row.get('images') or [])} 枚 "
              f"({int((tt >= 0).sum())} tok), score_from {row.get('score_from')}", flush=True)
    print(f"[check] OK ({n} 行)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cases", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--thinking", default="chat")
    b.add_argument("--max-seq-len", type=int, default=8192)
    c = sub.add_parser("check")
    c.add_argument("rows")
    c.add_argument("--no-load", action="store_true", help="画像を読まずに区間だけ検査する")
    a = ap.parse_args(argv)
    {"build": cmd_build, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
