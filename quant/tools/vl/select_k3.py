"""日本語 OCR で使われる routed expert を層ごとに順位付けし、K3 に上げる候補を決める。

入力は harness/stream_ref_vl.py の教師出力 .pt (vl_ocr_ja_calib.jsonl を元の重みで teacher forcing したもの)。
counts_image (画像トークン) と counts_text (問いと書き起こしのトークン) を層ごとに割合へ直し、
score = (1 - w) * 画像の割合 + w * テキストの割合 で並べる。画像トークンは 1 問約 1,000 個、書き起こしは数十〜数百個と
数が大きく違うので、割合にしてから混ぜる (OCR のループは書き起こしの decode で起きるので、テキスト側も外せない)。

出力 (--out): {"ranked": {"<layer>": [expert, ...] (全 384 個、score の高い順)},
               "coverage": {"<n>": 上位 n 個が占める画像 / テキストの選択回数の割合 (全層平均)}, ...}
dsv41-quant:151 の中で動かす (torch が要る)。build/ocr_k3.sh select。
"""
import argparse
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt", nargs="+")
    ap.add_argument("--text-weight", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    img, txt = {}, {}
    n_img = 0
    for p in a.pt:
        d = torch.load(p, weights_only=False)
        for src, dst in ((d["counts_image"], img), (d["counts_text"], txt)):
            for l, v in src.items():
                dst[l] = dst.get(l, 0) + np.asarray(v, dtype=np.int64)
        n_img += int(d["n_image_tokens"])
    layers = sorted(l for l in img if l < 40)
    ranked, cov = {}, {n: [0.0, 0.0] for n in (16, 24, 32, 40, 48)}
    for l in layers:
        pi = img[l] / max(1, img[l].sum())
        pt = txt[l] / max(1, txt[l].sum())
        score = (1 - a.text_weight) * pi + a.text_weight * pt
        order = np.argsort(-score, kind="stable")
        ranked[str(l)] = [int(e) for e in order]
        for n in cov:
            cov[n][0] += pi[order[:n]].sum() / len(layers)
            cov[n][1] += pt[order[:n]].sum() / len(layers)
    out = {"ranked": ranked, "text_weight": a.text_weight, "n_image_tokens": n_img, "sources": a.pt,
           "coverage": {str(n): {"image": round(v[0], 4), "text": round(v[1], 4)} for n, v in cov.items()}}
    json.dump(out, open(a.out, "w"), indent=1)
    for n, v in out["coverage"].items():
        print(f"上位 {n:>2} 個/層 ({int(n) / 384:.1%}): 画像の選択の {v['image']:.1%}、テキストの選択の {v['text']:.1%}")


if __name__ == "__main__":
    main()
