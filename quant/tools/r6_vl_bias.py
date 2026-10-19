"""画像トークン用の gate.bias_vl とテキスト用の gate.bias の差を層ごとに調べる (CPU、GPU 不要)。

model.py:811-830 の Gate は topk(scores + bias) で専門家を選ぶ。画像区間 (token_types >= 0) は bias の代わりに
bias_vl を使う。topk は bias 全体に定数を足しても変わらないので、差は中央値を引いた dc = (bias_vl - bias) - median で見る。

入力:
  - 原重み (DSV41_CKPT、既定 /model) の layers.N.ffn.gate.{bias,bias_vl,weight} と mtp.N.ffn.gate.{bias,bias_vl}
  - テキストの routing counts: stream_ref.py teacher() の .pt ({"counts": {layer: int64[384]}})。label=path で複数
出力: --out の JSON (層ごとの統計、画像で押し上げ/押し下げられる専門家、テキスト上位・下位との重なり、K3 被覆の試算)

usage (dsv41-quant イメージ、CPU だけ):
  python3 r6_vl_bias.py --counts agent=/t/agent24_teacher.pt --counts ja=/t/ja16_teacher.pt --out /w/r6_vl_bias.json
"""
import argparse
import json
import os

import numpy as np

CKPT = os.environ.get("DSV41_CKPT", "/model")
N_LAYERS, N_MTP, N_EXP, TOPK = 40, 3, 384, 6


class ST:
    """safetensors のヘッダだけ読み、名前で 1 テンソルを取り出す (numpy)。"""

    DT = {"F32": np.float32, "BF16": np.uint16, "F16": np.float16}

    def __init__(self, ckpt):
        self.ckpt = ckpt
        self.wm = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}

    def _h(self, fn):
        if fn not in self.hdr:
            with open(os.path.join(self.ckpt, fn), "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self.hdr[fn] = (8 + n, json.loads(f.read(n)))
        return self.hdr[fn]

    def get(self, name):
        fn = self.wm[name]
        base, h = self._h(fn)
        m = h[name]
        a, b = m["data_offsets"]
        with open(os.path.join(self.ckpt, fn), "rb") as f:
            f.seek(base + a)
            raw = np.frombuffer(f.read(b - a), dtype=self.DT[m["dtype"]])
        if m["dtype"] == "BF16":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.reshape(m["shape"]).astype(np.float64)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def load_counts(path):
    import torch

    d = torch.load(path, weights_only=False, map_location="cpu")
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in d["counts"].items()}


def k3_mask(shares, pis, frac, m):
    """層内で I = (1-m) Σ π_d S_d + m/384 の上位 frac を K3 にする (SPEC §2.3 の層内版、α は層内で無関係)。"""
    imp = sum(pis[d] * shares[d] for d in pis)
    imp = (1 - m) * imp + m / N_EXP
    k = int(round(frac * N_EXP))
    mask = np.zeros(N_EXP, bool)
    mask[np.argsort(-imp)[:k]] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="append", default=[], help="label=path (teacher .pt)")
    ap.add_argument("--top-frac", type=float, default=0.05, help="画像候補の割合 (層内、既定 5% = 19 個)")
    ap.add_argument("--k3-frac", type=float, default=0.193, help="K3 の割合 (SPEC §3 案 B の 19.3%)")
    ap.add_argument("--mix", type=float, default=0.3, help="一様分布との混合 m")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    st = ST(CKPT)
    counts = {}
    for spec in a.counts:
        lab, p = spec.split("=", 1)
        counts[lab] = load_counts(p)
    labs = list(counts)
    ntop = int(round(a.top_frac * N_EXP))
    rep = {"ckpt": CKPT, "top_frac": a.top_frac, "n_top": ntop, "labels": labs, "layers": {}, "mtp": {}}

    rows = []
    for L in range(N_LAYERS):
        b = st.get(f"layers.{L}.ffn.gate.bias")
        bv = st.get(f"layers.{L}.ffn.gate.bias_vl")
        d = bv - b
        dc = d - np.median(d)
        sd_b, sd_bv, sd_dc = b.std(), bv.std(), dc.std()
        up = np.argsort(-dc)[:ntop]       # bias_vl で選ばれやすくなる (画像候補)
        down = np.argsort(dc)[:ntop]      # bias_vl で選ばれにくくなる
        z = dc / sd_dc
        r = {
            "bias": {"mean": b.mean(), "std": sd_b, "min": b.min(), "max": b.max()},
            "bias_vl": {"mean": bv.mean(), "std": sd_bv, "min": bv.min(), "max": bv.max()},
            "diff": {"mean": d.mean(), "median": float(np.median(d)), "std": sd_dc,
                     "absmax_c": float(np.abs(dc).max()), "p95_abs_c": float(np.quantile(np.abs(dc), 0.95))},
            "pearson_b_bv": float(np.corrcoef(b, bv)[0, 1]),
            "spearman_b_bv": spearman(b, bv),
            "sd_ratio": sd_dc / sd_b,
            "n_z2_up": int((z >= 2).sum()), "n_z2_down": int((z <= -2).sum()),
            "up": [int(e) for e in up], "down": [int(e) for e in down],
            "up_dc": [float(dc[e]) for e in up], "down_dc": [float(dc[e]) for e in down],
        }
        # テキストの routing との関係
        shares = {}
        for lab in labs:
            c = counts[lab].get(L)
            if c is None:
                continue
            s = c / c.sum()
            shares[lab] = s
            top96 = set(np.argsort(-s)[:96].tolist())
            cold = set(np.argsort(s)[:115].tolist())
            r[lab] = {
                "up_in_top96": len(top96 & set(up.tolist())), "down_in_top96": len(top96 & set(down.tolist())),
                "up_in_cold115": len(cold & set(up.tolist())), "down_in_cold115": len(cold & set(down.tolist())),
                "up_share": float(s[up].sum()), "down_share": float(s[down].sum()),
                "spearman_dc_logshare": spearman(dc, np.log(s + 1e-7)),
                "spearman_b_logshare": spearman(b, np.log(s + 1e-7)),
                "spearman_bv_logshare": spearman(bv, np.log(s + 1e-7)),
                "up_rank_pct_mean": float(np.mean([(s < s[e]).mean() for e in up])),
            }
        # K3 被覆の試算: テキスト配分 (agent 0.7 / ja 0.3) で選んだ K3 が、各分布の routing のどれだけを覆うか
        if "agent" in shares and "ja" in shares:
            uni = np.full(N_EXP, 1 / N_EXP)
            for name, pis in {"text": {"agent": 0.7, "ja": 0.3},
                              "text+img_uni0.2": {"agent": 0.5, "ja": 0.3, "img": 0.2}}.items():
                sh = dict(shares, img=uni)
                msk = k3_mask(sh, pis, a.k3_frac, a.mix)
                r.setdefault("k3_cover", {})[name] = {
                    "agent": float(shares["agent"][msk].sum()), "ja": float(shares["ja"][msk].sum()),
                    "img_uniform": float(msk.mean()),
                    "img_up_experts_in_k3": int(msk[up].sum()),
                }
        rep["layers"][L] = r
        rows.append(r)

    for j in range(N_MTP):
        b = st.get(f"mtp.{j}.ffn.gate.bias")
        bv = st.get(f"mtp.{j}.ffn.gate.bias_vl")
        d = bv - b
        rep["mtp"][j] = {"n": len(b), "diff_std": float((d - np.median(d)).std()), "bias_std": float(b.std()),
                         "pearson_b_bv": float(np.corrcoef(b, bv)[0, 1]), "identical": bool(np.array_equal(b, bv))}

    def mean(key, sub=None):
        v = [(r[sub][key] if sub else r[key]) for r in rows if (sub is None or sub in r)]
        return float(np.mean(v)) if v else None

    summ = {"pearson_b_bv": mean("pearson_b_bv"), "spearman_b_bv": mean("spearman_b_bv"),
            "sd_ratio": mean("sd_ratio"), "n_z2_up": mean("n_z2_up"), "n_z2_down": mean("n_z2_down"),
            "expect_random_in_top96": ntop * 96 / N_EXP, "expect_random_in_cold115": ntop * 115 / N_EXP}
    for lab in labs:
        summ[lab] = {k: mean(k, lab) for k in ("up_in_top96", "down_in_top96", "up_in_cold115", "down_in_cold115",
                                               "up_share", "down_share", "spearman_dc_logshare",
                                               "spearman_b_logshare", "spearman_bv_logshare", "up_rank_pct_mean")}
    if all("k3_cover" in r for r in rows):
        summ["k3_cover"] = {name: {k: float(np.mean([r["k3_cover"][name][k] for r in rows]))
                                   for k in rows[0]["k3_cover"][name]} for name in rows[0]["k3_cover"]}
    rep["summary"] = summ

    print(f"{'L':>2} {'sd(b)':>7} {'sd(bv)':>7} {'sd(dc)':>7} {'ratio':>5} {'r(b,bv)':>7} {'z2+':>3} {'z2-':>3} "
          + " ".join(f"{lab}:up∩top96/cold up%/dn%" for lab in labs))
    for L, r in rep["layers"].items():
        extra = " ".join(f"{r[lab]['up_in_top96']:2d}/{r[lab]['up_in_cold115']:2d} {100*r[lab]['up_share']:4.1f}/{100*r[lab]['down_share']:4.1f}"
                         for lab in labs if lab in r)
        print(f"{L:2d} {r['bias']['std']:7.4f} {r['bias_vl']['std']:7.4f} {r['diff']['std']:7.4f} {r['sd_ratio']:5.2f} "
              f"{r['pearson_b_bv']:7.3f} {r['n_z2_up']:3d} {r['n_z2_down']:3d} {extra}")
    print("MTP", rep["mtp"])
    print("SUMMARY", json.dumps(summ, indent=1, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
