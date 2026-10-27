"""日本語 OCR の劣化を測る評価セットと、専門家選びに使う較正セットを作る (CPU のみ)。

CC-OCR 日本語 (150 枚) のうち、vl_gen / vl_kl / vl_calib / vl_gen_hard で使っていない画像を sha256 で選び、
評価 work/vl/vl_ocr_ja.jsonl と較正 work/vl/vl_ocr_ja_calib.jsonl に分ける (画像は重ならない)。
評価の列は vl_gen.jsonl と同じなので tools/vl_eval.py --set でそのまま使える。較正は harness/stream_ref_vl.py の入力形式
{id, messages, answer} (回答 = 正解の書き起こし。画像パスは work/vl からの相対) で、教師 / 学生の採点や routing の集計に使う。

使い方: tools/.venv/bin/python tools/vl/build_ocr.py [--n-calib 35]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sets as B  # noqa: E402

SEED = 20261027
OCR_Q = "画像に写っている文字をすべて書き出してください。説明は不要です。"  # vl_gen_hard の ccocr と同じ問い


def used_image_hashes():
    hs = set()
    for name in ("vl_gen.jsonl", "vl_kl.jsonl", "vl_calib.jsonl", "vl_gen_hard.jsonl"):
        for l in open(os.path.join(B.OUT, name), encoding="utf-8"):
            for p in json.loads(l)["images"]:
                hs.add(B.sha(open(os.path.join(B.OUT, p), "rb").read()))
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=35)
    a = ap.parse_args()
    rng = random.Random(SEED)
    used = used_image_hashes()
    P = B.load_public(rng)
    recs = sorted((r for r in P["ccocr"] if B.sha(r["imgs"][0]) not in used), key=lambda r: r["key"])
    rng.shuffle(recs)
    cal, ev = recs[: a.n_calib], recs[a.n_calib:]
    with open(os.path.join(B.OUT, "vl_ocr_ja.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(ev):
            im = B.save_img("ocr/eval", r["key"].rsplit(".", 1)[0], r["imgs"][0])
            it = dict(id=f"ocr-ccocr-{i:03d}", set="ocr_ja", lang="ja", cat="ocr", sub="ccocr", source="ccocr",
                      license=B.LIC["ccocr"][1], ref=r["src"], images=[im], prompt=OCR_Q, answer=r["a"], atype="ned", max_tokens=512)
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(os.path.join(B.OUT, "vl_ocr_ja_calib.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(cal):
            im = B.save_img("ocr/calib", r["key"].rsplit(".", 1)[0], r["imgs"][0])
            row = dict(id=f"ocrcal-ccocr-{i:03d}", lang="ja", cat="ocr", sub="ccocr", ref=r["src"], thinking_mode="chat",
                       messages=[B.user_msg([im], OCR_Q)], answer=r["a"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"CC-OCR 日本語 {len(P['ccocr'])} 枚、未使用 {len(recs)} 枚 → 評価 {len(ev)} / 較正 {len(cal)}")


if __name__ == "__main__":
    main()
