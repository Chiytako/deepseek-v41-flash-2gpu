#!/usr/bin/env python3
"""Routing-skew summary of public V4.1-Flash pass-0 statistics.

Input: a directory of layer_XX.json files in the format of
sudoingX/dgx-spark-deepseek-v41 data/p0_stats_v3/merged_bal (per-expert 'count' = routed slots,
'tshare' = per-token share of the routed output). Calibration: 853,106 tokens, V4.1-Flash (not 1017).

Usage: python3 -I r2_routing_skew.py <stats_dir> [--key count|tshare]
"""
import json
import os
import sys


def gini(xs):
    xs = sorted(xs)
    n = len(xs)
    s = sum(xs)
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * s) - (n + 1) / n


def cover(xs, frac):
    xs = sorted(xs, reverse=True)
    s, acc = sum(xs), 0.0
    for i, x in enumerate(xs):
        acc += x
        if acc >= frac * s:
            return i + 1
    return len(xs)


def main():
    d = sys.argv[1]
    key = sys.argv[sys.argv.index("--key") + 1] if "--key" in sys.argv else "count"
    files = sorted(f for f in os.listdir(d) if f.startswith("layer_") and f.endswith(".json"))
    allv = []
    print(f"key={key}")
    print("layer  max/mean  min/mean  top10%  top25%  top50%  cold30%  n@80%  n@90%  n@95%  gini")
    agg = {"cold30": [], "n90": [], "gini": []}
    for f in files:
        st = json.load(open(os.path.join(d, f)))
        v = [float(x) for x in st[key]]
        n, s = len(v), sum(v)
        m = s / n
        sv = sorted(v, reverse=True)
        top = lambda p: sum(sv[: round(n * p)]) / s
        cold30 = sum(sorted(v)[: round(n * 0.3)]) / s
        g = gini(v)
        n80, n90, n95 = cover(v, .8), cover(v, .9), cover(v, .95)
        agg["cold30"].append(cold30); agg["n90"].append(n90); agg["gini"].append(g)
        print(f"{st['layer']:5d}  {max(v)/m:8.2f}  {min(v)/m:8.3f}  {top(.1):6.3f}  {top(.25):6.3f}  {top(.5):6.3f}"
              f"  {cold30:7.4f}  {n80:5d}  {n90:5d}  {n95:5d}  {g:.3f}")
        allv += [x / s for x in v]  # each layer normalised to 1
    # global (all 15,360 experts, each layer weighted equally) coldest 30%, as in diffbot make_plan.py --norm global
    allv.sort()
    tot = sum(allv)
    k = round(len(allv) * 0.3)
    print(f"\nglobal coldest 30% of {len(allv)} experts: {sum(allv[:k]) / tot:.4f} of routings")
    print(f"global coldest 50%: {sum(allv[: len(allv) // 2]) / tot:.4f}")
    print(f"mean over layers: cold30 {sum(agg['cold30'])/len(files):.4f}, n@90% {sum(agg['n90'])/len(files):.1f}, "
          f"gini {sum(agg['gini'])/len(files):.3f}")


if __name__ == "__main__":
    main()
