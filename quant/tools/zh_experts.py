"""言語別の routing counts から「言語に偏った expert」を層ごとに挙げ、その gate.bias の値と並べる (CPU)。

入力は label=path の並び。path は次のどれか:
  - stream_ref.py teacher() の .pt ({"counts": {layer: np.int64[384]}})。系列全体 (prompt を含む) の counts
  - harness/steer_collect.py の .pt ({"routing": {group: {layer: np.int64[384]}}, "ntok_routing": {group: n}})。
    この場合 label は "label=path:group" と書き、group (例: ja_answer, zh_answer) の位置だけの counts を使う
指標 (SteerMoE, Fayyaz et al. 2025 と同じ考え方): 1 トークンあたりの選択率 p_g(e) = counts_g(e) / ntok_g の差 (risk
difference) RD = p_A - p_B と比 p_A / p_B。top-6 なので Σ_e p_g(e) = 6。

usage: python -I tools/zh_experts.py A=work/teacher/ja16_teacher.pt B=work/teacher/agent24_teacher.pt \
         [--top 8] [--ratio 4] [--min-rate 0.005] [--out work/r5/experts_ja_vs_agent.json]
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse
import json
import os
import sys

import numpy as np

CKPT = os.environ.get("DSV41_CKPT", MODELS + "/DeepSeek-V4.1-Flash-1017")
TOPK = 6


def load_counts(spec):
    import torch  # .pt を読むときだけ

    path, _, group = spec.partition(":")
    d = torch.load(path, weights_only=False, map_location="cpu")
    if group:
        c = {int(k): np.asarray(v, dtype=np.float64) for k, v in d["routing"][group].items()}
    else:
        c = {int(k): np.asarray(v, dtype=np.float64) for k, v in d["counts"].items()}
    return c


def gate_bias(layers):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from zh_ortho_fp8 import ST

    st = ST(CKPT)
    return {L: np.asarray(st.raw(f"layers.{L}.ffn.gate.bias")[0], dtype=np.float64) for L in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs=2, help="A=path[:group] B=path[:group]")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=4.0)
    ap.add_argument("--min-rate", type=float, default=0.005, help="A 側の 1 トークンあたり選択率の下限")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    (la, pa), (lb, pb) = [p.split("=", 1) for p in a.pairs]
    A, B = load_counts(pa), load_counts(pb)
    layers = sorted(set(A) & set(B))
    layers = [L for L in layers if L < 40]
    bias = gate_bias(layers)
    rep = {"A": la, "B": lb, "layers": {}}
    tot_spec = []
    for L in layers:
        na, nb = A[L].sum() / TOPK, B[L].sum() / TOPK
        ra, rb = A[L] / na, B[L] / nb
        rd = ra - rb
        ratio = (ra + 1e-6) / (rb + 1e-6)
        spec = np.where((ratio >= a.ratio) & (ra >= a.min_rate))[0]
        order = np.argsort(-rd)[: a.top]
        b = bias[L]
        cos = float(ra @ rb / np.linalg.norm(ra) / np.linalg.norm(rb))
        rep["layers"][L] = {
            "ntok_A": float(na), "ntok_B": float(nb), "cos": cos,
            "spec_A": [int(e) for e in spec],
            "spec_A_share_of_A_routing": float(ra[spec].sum() / TOPK),
            "spec_A_share_of_B_routing": float(rb[spec].sum() / TOPK),
            "top_rd": [{"e": int(e), "rate_A": float(ra[e]), "rate_B": float(rb[e]), "bias": float(b[e]),
                        "bias_rank_pct": float((b < b[e]).mean())} for e in order],
            "bias_stats": {"mean": float(b.mean()), "std": float(b.std()), "min": float(b.min()),
                           "max": float(b.max()), "p10": float(np.quantile(b, 0.1)), "p90": float(np.quantile(b, 0.9))},
            "bias_spec_mean": float(b[spec].mean()) if len(spec) else None,
        }
        tot_spec.append(len(spec))
    rep["mean_spec_per_layer"] = float(np.mean(tot_spec))
    for L in layers[:: max(1, len(layers) // 10)]:
        r = rep["layers"][L]
        t = r["top_rd"][:4]
        print(f"L{L:2d} cos {r['cos']:.3f} spec {len(r['spec_A']):2d} (A {r['spec_A_share_of_A_routing']:.3f} / B {r['spec_A_share_of_B_routing']:.3f}) "
              f"bias mean {r['bias_stats']['mean']:+.4f} std {r['bias_stats']['std']:.4f} | top: "
              + " ".join(f"e{x['e']}({x['rate_A']:.3f}/{x['rate_B']:.3f}, b{x['bias']:+.3f})" for x in t))
    print(f"mean spec experts/layer (ratio>={a.ratio}, rate_A>={a.min_rate}): {rep['mean_spec_per_layer']:.1f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
