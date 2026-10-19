"""画像 routing の CPU 近似と、画像の評価・校正の試作に使う合成画像を作る (ライセンスの問題が無い自作画像)。

実トラフィックの画像 (GLM 要求ログ、105 種) は PNG が 101 枚で、1024x768 (64 枚) と 3000x2000 (16 枚) が多い。
内容は読んでいない (個人データの扱いのため)。ここでは画面写し型の合成画像 (日本語文書、コード、グラフ、表、UI) を作り、
写真は配布条件の明らかなもの (モデル同梱の carrots/corn = MIT、matplotlib 同梱の grace_hopper = パブリックドメイン) を写す。

usage: python3 r6_make_images.py <out_dir> [--font /fonts/NotoSansCJK-Regular.ttc] [--model /model]
"""
import argparse
import os
import shutil

from PIL import Image, ImageDraw, ImageFont

JA = [
    "第3章 量子化と蒸留の手順",
    "本章では、専門家ごとにビット数を変える静的な混合精度について説明する。",
    "校正データの40%を日本語にし、日本語トークンの重みを2倍にする。",
    "1. 教師の出力を記録する (GPU 1枚、約30分)",
    "2. 層ごとに suh / svh を学習する (局所蒸留)",
    "3. ハーネスで KL を測り、v1 と比べる",
    "注意: 画像トークンは Engram を引かず、専用の補正バイアスで専門家を選ぶ。",
    "表 3.1 に区分ごとの KL を示す。日本語の KL は英語の 1.25 倍である。",
]
CODE = [
    "def moe_forward(self, x, image_mask=None):",
    "    shape = x.size()",
    "    x = x.view(-1, self.dim)",
    "    weights, indices = self.gate(x, image_mask)",
    "    y = torch.zeros_like(x, dtype=torch.float32)",
    "    for i in used:",
    "        idx, top = torch.where(indices == i)",
    "        y[idx] += expert(x[idx], weights[idx, top, None])",
    "    return y.type_as(x).view(shape)",
    "",
    "# 画像区間は prefill の 1 チャンク内に必要",
    "assert start_pos == 0, 'image spans must be prefilled in a single chunk'",
]


def doc(font, w=1024, h=768):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    f1, f2 = ImageFont.truetype(font, 34), ImageFont.truetype(font, 22)
    d.text((40, 30), JA[0], fill="black", font=f1)
    y = 100
    for line in JA[1:] * 2:
        d.text((40, y), line, fill=(30, 30, 30), font=f2)
        y += 38
    return im


def code(font, w=1024, h=768):
    im = Image.new("RGB", (w, h), (30, 32, 40))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, 20)
    for i, line in enumerate(CODE * 2):
        d.text((20, 20 + i * 30), f"{i + 1:3d}", fill=(120, 120, 140), font=f)
        col = (220, 180, 120) if line.startswith(("def", "assert")) else (200, 220, 200)
        d.text((80, 20 + i * 30), line, fill=col, font=f)
    return im


def chart(font, w=1024, h=768):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, 22)
    d.text((300, 20), "版ごとの KL (agent / ja)", fill="black", font=f)
    vals = [("v0", 0.158, 0.206), ("v1", 0.059, 0.074), ("L2", 0.052, 0.063), ("K3mix", 0.040, 0.049)]
    d.line((100, 680, 980, 680), fill="black", width=2)
    d.line((100, 80, 100, 680), fill="black", width=2)
    for i, (n, a, j) in enumerate(vals):
        x = 160 + i * 210
        d.rectangle((x, 680 - a * 2800, x + 70, 680), fill=(70, 110, 200))
        d.rectangle((x + 80, 680 - j * 2800, x + 150, 680), fill=(220, 120, 60))
        d.text((x + 40, 690), n, fill="black", font=f)
    return im


def table(font, w=1280, h=960):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, 20)
    cols = ["層", "上位25%", "下位30%", "Gini", "未使用", "備考"]
    for r in range(22):
        for c in range(6):
            x0, y0 = 40 + c * 200, 40 + r * 40
            d.rectangle((x0, y0, x0 + 200, y0 + 40), outline=(160, 160, 160), fill=(235, 240, 250) if r == 0 else None)
            t = cols[c] if r == 0 else (str(r * 2) if c == 0 else ("日本語" if c == 5 and r % 3 == 0 else f"0.{(r * 37 + c * 11) % 1000:03d}"))
            d.text((x0 + 10, y0 + 8), t, fill="black", font=f)
    return im


def ui(font, w=3000, h=2000):
    im = Image.new("RGB", (w, h), (246, 247, 249))
    d = ImageDraw.Draw(im)
    f1, f2 = ImageFont.truetype(font, 56), ImageFont.truetype(font, 40)
    d.rectangle((0, 0, w, 140), fill=(40, 60, 110))
    d.text((60, 40), "ダッシュボード  |  量子化ジョブ", fill="white", font=f1)
    d.rectangle((0, 140, 520, h), fill=(230, 233, 240))
    for i, s in enumerate(["概要", "ジョブ一覧", "評価", "設定", "ログ"]):
        d.text((60, 200 + i * 90), s, fill=(40, 40, 60), font=f2)
    for i in range(6):
        x0, y0 = 600 + (i % 3) * 790, 220 + (i // 3) * 600
        d.rounded_rectangle((x0, y0, x0 + 740, y0 + 540), 24, fill="white", outline=(200, 200, 210), width=3)
        d.text((x0 + 40, y0 + 40), f"GPU{i % 2} ジョブ {i + 1}", fill=(30, 30, 30), font=f2)
        d.rectangle((x0 + 40, y0 + 140, x0 + 40 + 100 * (i + 1), y0 + 180), fill=(70, 160, 90))
        d.text((x0 + 40, y0 + 220), f"進捗 {(i + 1) * 15}%  残り {(6 - i) * 7} 分", fill=(80, 80, 90), font=f2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--font", default="/fonts/NotoSansCJK-Regular.ttc")
    ap.add_argument("--model", default="/model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in (("syn_doc_ja.png", doc), ("syn_code.png", code), ("syn_chart_ja.png", chart),
                     ("syn_table_ja.png", table), ("syn_ui_ja.png", ui)):
        fn(a.font).save(os.path.join(a.out, name))
    srcs = [os.path.join(a.model, "inference/examples/images/carrots.jpeg"),
            os.path.join(a.model, "inference/examples/images/corn.jpeg")]
    try:
        import matplotlib

        srcs.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/sample_data/grace_hopper.jpg"))
    except ImportError:
        pass
    for s in srcs:
        if os.path.exists(s):
            shutil.copy(s, os.path.join(a.out, "photo_" + os.path.basename(s)))
    print(sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
