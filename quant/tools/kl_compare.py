#!/usr/bin/env python3
"""ハーネス学生採点 (.pt) を 2 版で対にして比べる。QAD の採否判定用 (research/QAD_PLAN.md §5)。

入力は stream_ref.py --teacher-ref <教師.pt> --teacher-out <学生.pt> の出力 (stream_ref.py:465-466 の
{"recs", "kl", "agree"})。同じ教師・同じ系列で採点した base と new を系列ごとに対にして、
  - トークン平均の KL (教師 top-32 上の近似、QUALITY.md と同じ指標) と top-1 一致
  - new/base − 1 (相対変化) と、系列単位の対応のある bootstrap の 95% 区間 (区分ごとに層化して再標本化)
を区分 (--set の名前、jsonl の lang / sub) ごとと全体で出す。GPU は使わない。

使い方:
  python3 kl_compare.py --set agent24 /t/agent24_v1.pt /t/agent24_l2.pt \
                        --set ja16 /t/ja16_v1.pt /t/ja16_l2.pt \
                        --set jakl32 /t/jakl32_v1.pt /t/jakl32_l2.pt --jsonl jakl32=/c/jakl32.jsonl \
                        [--ja-sets ja16,jakl32] [--boot 10000] [--json out.json]
"""
import argparse
import json
import math
import sys

import numpy as np


def load_pt(path):
    import torch
    d = torch.load(path, weights_only=False, map_location="cpu")
    kl, ag = d.get("kl"), d.get("agree")
    if not kl or any(k is None for k in kl):
        raise SystemExit(f"{path}: 学生採点の kl が無い (--teacher-ref を付けて採点した .pt か?)")
    recs = d["recs"]
    if len(recs) != len(kl):
        raise SystemExit(f"{path}: recs {len(recs)} と kl {len(kl)} の数が違う (途中で止まった採点?)")
    out = []
    for r, k, a in zip(recs, kl, ag):
        out.append({"ids_len": len(r["ids"]), "score_from": int(r["score_from"]), "src": r.get("src"),
                    "kl": k.float().numpy().astype(np.float64), "agree": a.float().numpy().astype(np.float64),
                    "ids_head": tuple(r["ids"][:64]), "ids_tail": tuple(r["ids"][-64:])})
    return out


def pair(base, new, name):
    if len(base) != len(new):
        raise SystemExit(f"{name}: 系列数が違う base {len(base)} / new {len(new)}")
    for i, (b, n) in enumerate(zip(base, new)):
        if (b["ids_len"], b["score_from"], b["ids_head"], b["ids_tail"]) != (n["ids_len"], n["score_from"], n["ids_head"], n["ids_tail"]):
            raise SystemExit(f"{name}: 系列 {i} が base と new で違う (別の jsonl で採点した?)")
        if b["kl"].shape != n["kl"].shape:
            raise SystemExit(f"{name}: 系列 {i} の採点位置数が違う")
    return [{"set": name, "src": b["src"], "n": b["kl"].size,
             "kl_b": b["kl"].sum(), "kl_n": n["kl"].sum(),
             "ag_b": b["agree"].sum(), "ag_n": n["agree"].sum()} for b, n in zip(base, new)]


def summarize(rows, boot, rng):
    """rows: 系列ごとの和。トークン平均 (系列の長さで重み付け) と、set で層化した系列 bootstrap。"""
    if not rows:
        return None
    a = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in ("n", "kl_b", "kl_n", "ag_b", "ag_n")}
    N = a["n"].sum()
    kb, kn = a["kl_b"].sum() / N, a["kl_n"].sum() / N
    res = {"seqs": len(rows), "tokens": int(N), "kl_base": kb, "kl_new": kn, "kl_rel": kn / kb - 1 if kb > 0 else float("nan"),
           "top1_base": a["ag_b"].sum() / N, "top1_new": a["ag_n"].sum() / N}
    res["top1_diff_pt"] = 100 * (res["top1_new"] - res["top1_base"])
    if boot > 0 and len(rows) >= 2:
        sets = np.array([r["set"] for r in rows])
        strata = [np.flatnonzero(sets == s) for s in dict.fromkeys(sets)]
        rel, dtop = np.empty(boot), np.empty(boot)
        for b in range(boot):
            idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
            n = a["n"][idx].sum()
            rel[b] = a["kl_n"][idx].sum() / a["kl_b"][idx].sum() - 1
            dtop[b] = 100 * (a["ag_n"][idx].sum() - a["ag_b"][idx].sum()) / n
        res["kl_rel_ci95"] = [float(np.quantile(rel, 0.025)), float(np.quantile(rel, 0.975))]
        res["top1_diff_ci95"] = [float(np.quantile(dtop, 0.025)), float(np.quantile(dtop, 0.975))]
        res["p_worse"] = float((rel >= 0).mean())  # 片側: KL が下がっていない確率 (bootstrap)
    return res


def fmt(name, r):
    if r is None:
        return f"{name:<22} (なし)"
    ci = r.get("kl_rel_ci95")
    cis = f"[{100 * ci[0]:+6.1f}%, {100 * ci[1]:+6.1f}%]" if ci else " " * 18
    tci = r.get("top1_diff_ci95")
    tcis = f"[{tci[0]:+.2f}, {tci[1]:+.2f}]" if tci else ""
    return (f"{name:<22} seq {r['seqs']:>3} tok {r['tokens']:>7}  KL {r['kl_base']:.4f} -> {r['kl_new']:.4f} "
            f"({100 * r['kl_rel']:+6.1f}%) {cis}  top1 {100 * r['top1_base']:.2f} -> {100 * r['top1_new']:.2f} "
            f"({r['top1_diff_pt']:+.2f} pt) {tcis}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs=3, action="append", required=True, metavar=("NAME", "BASE_PT", "NEW_PT"))
    ap.add_argument("--jsonl", action="append", default=[], help="NAME=path。jsonl の lang / sub で区分を細分する")
    ap.add_argument("--ja-sets", default="ja16,jakl32", help="「日本語」としてまとめる set の名前")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    jl = dict(s.split("=", 1) for s in a.jsonl)
    rows = []
    for name, bp, np_ in a.set:
        r = pair(load_pt(bp), load_pt(np_), name)
        if name in jl:
            meta = [json.loads(l) for l in open(jl[name])]
            if len(meta) != len(r):
                raise SystemExit(f"{name}: jsonl {len(meta)} 行と .pt {len(r)} 系列が違う")
            for x, m in zip(r, meta):
                x["sub"] = m.get("sub") or m.get("lang")
        rows += r
    out = {"by_set": {}, "by_sub": {}}
    names = list(dict.fromkeys(r["set"] for r in rows))
    for s in names:
        out["by_set"][s] = summarize([r for r in rows if r["set"] == s], a.boot, rng)
    for s in sorted({(r["set"], r["sub"]) for r in rows if r.get("sub")}):
        out["by_sub"][f"{s[0]}:{s[1]}"] = summarize([r for r in rows if (r["set"], r.get("sub")) == s], a.boot, rng)
    ja = set(x for x in a.ja_sets.split(",") if x)
    out["ja"] = summarize([r for r in rows if r["set"] in ja], a.boot, rng)
    out["non_ja"] = summarize([r for r in rows if r["set"] not in ja], a.boot, rng)
    out["all"] = summarize(rows, a.boot, rng)
    for s in names:
        print(fmt(s, out["by_set"][s]))
    for k, v in out["by_sub"].items():
        print(fmt("  " + k, v))
    print(fmt("JA (" + ",".join(sorted(ja & set(names))) + ")", out["ja"]))
    print(fmt("non-JA", out["non_ja"]))
    print(fmt("ALL", out["all"]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    return out


if __name__ == "__main__":
    main()
