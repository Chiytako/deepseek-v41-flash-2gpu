"""X2 stage 4: turn x2_sim.py output into tables (markdown) + derived IO numbers.
usage: x2_report.py <sim.json> <prep_dir>
"""
import json
import sys

import numpy as np

r = json.load(open(sys.argv[1]))
prep = json.load(open(sys.argv[2] + "/prep_stats.json"))
f = r["sample_frac"]
caps = r["caps_gb"]
H = np.array(r["hit_ev"], float)   # [pol5, cap, cat6, attr16]
HR = np.array(r["hit_raw"], float)
T = np.array(r["tot_ev"], float)   # [cat6, attr16]
TR = np.array(r["tot_raw"], float)
RU = np.array(r["reuse"], float)   # [pol5, cap, phase2, 6]
CO = np.array(r["cold"], float)
PH = np.array(r["phit"], float)    # [2, cap, cat6, attr16]
PT = np.array(r["ptot"], float)
tok = np.array(r["tok_cnt_half_phase_cls"], float)  # [half, phase, cls]
test_tok = tok[1]  # [phase, cls]
pols = r["pol_row"]

# attr bits: L | order<<1 | rank<<3  (order 0..2 = 2/3/4-gram)
attr = np.arange(16)
A_L = attr & 1
A_O = (attr >> 1) & 3
A_R = (attr >> 3) & 1
cat = np.arange(6)
C_PH = cat // 3
C_CL = cat % 3
CLS = ["ja", "code", "other"]


def sel(cm=None, am=None):
    cm = np.ones(6, bool) if cm is None else cm
    am = np.ones(16, bool) if am is None else am
    return np.outer(cm, am)


def rate(Hx, Tx, m):
    return (Hx[..., m].sum(-1) / max(1.0, Tx[m].sum()))


def p(x):
    return f"{100 * x:.1f}%"


out = []
w = out.append
w(f"sample: {len(r['parts'])}/{r['W']} page partitions (frac {f:.3f}); test half = requests after #{r['split_q']}; "
  f"Engram tokens total {r['total_engram_tokens']:,}, test-half tokens {int(test_tok.sum()):,} "
  f"(prefill {int(test_tok[0].sum()):,}, decode {int(test_tok[1].sum()):,}), image tokens skipped {r['img_tokens']:,}")
w(f"test half: raw lookups {TR.sum() / f:,.0f}, deduped accesses {T.sum() / f:,.0f}; "
  f"rows seen train {r['train_rows'] / f:,.0f} test {r['test_rows'] / f:,.0f} both {r['both_rows'] / f:,.0f}")

# dedupe per phase / class
w("\n### dedupe (unique rows / lookups inside a group: prefill 8k chunk, decode 4-token step)")
w("| phase | class | lookups | unique | unique/lookup |")
w("|---|---|---|---|---|")
for ph in range(2):
    for c in range(3):
        m = sel((C_PH == ph) & (C_CL == c))
        w(f"| {'prefill' if ph == 0 else 'decode'} | {CLS[c]} | {TR[m].sum() / f:,.0f} | {T[m].sum() / f:,.0f} | "
          f"{T[m].sum() / max(1, TR[m].sum()):.3f} |")
    m = sel(C_PH == ph)
    w(f"| {'prefill' if ph == 0 else 'decode'} | all | {TR[m].sum() / f:,.0f} | {T[m].sum() / f:,.0f} | "
      f"{T[m].sum() / max(1, TR[m].sum()):.3f} |")

# main hit table (deduped accesses)
w("\n### hit rate on deduped accesses (test half), all phases")
w("| policy | " + " | ".join(f"{c}GB" for c in caps) + " |")
w("|---|" + "---|" * len(caps))
m = sel()
for pi, pn in enumerate(pols):
    w(f"| row {pn} | " + " | ".join(p(rate(H[pi, ci], T, m)) for ci in range(len(caps))) + " |")
for pi, pn in enumerate(r["pol_page"]):
    w(f"| page4K {pn} | " + " | ".join(p(rate(PH[pi, ci], PT, m)) for ci in range(len(caps))) + " |")
w(f"| cold (first-ever access) | {p(CO.sum() / T.sum())} |")

w("\n### static row cache, raw-lookup hit rate (round-1 comparable, no dedupe)")
w("| phase | " + " | ".join(f"{c}GB" for c in caps) + " |")
w("|---|" + "---|" * len(caps))
for ph in range(2):
    mm = sel(C_PH == ph)
    w(f"| {'prefill' if ph == 0 else 'decode'} | " + " | ".join(p(rate(HR[0, ci], TR, mm)) for ci in range(len(caps))) + " |")

for ph in range(2):
    w(f"\n### {'prefill' if ph == 0 else 'decode'}: deduped hit rate by policy")
    w("| policy | " + " | ".join(f"{c}GB" for c in caps) + " |")
    w("|---|" + "---|" * len(caps))
    mm = sel(C_PH == ph)
    for pi, pn in enumerate(pols):
        w(f"| row {pn} | " + " | ".join(p(rate(H[pi, ci], T, mm)) for ci in range(len(caps))) + " |")
    for pi, pn in enumerate(r["pol_page"]):
        w(f"| page4K {pn} | " + " | ".join(p(rate(PH[pi, ci], PT, mm)) for ci in range(len(caps))) + " |")

w("\n### by n-gram order (row lru / row static), all phases")
w("| order | share of accesses | " + " | ".join(f"lru {c}GB" for c in caps) + " | " + " | ".join(f"st {c}GB" for c in caps) + " |")
w("|---|---|" + "---|" * (2 * len(caps)))
for o in range(3):
    mm = sel(None, A_O == o)
    w(f"| {o + 2}-gram | {p(T[mm].sum() / T.sum())} | " + " | ".join(p(rate(H[2, ci], T, mm)) for ci in range(len(caps)))
      + " | " + " | ".join(p(rate(H[0, ci], T, mm)) for ci in range(len(caps))) + " |")

w("\n### by TP rank (rows split at ceil(N/2) per layer)")
w("| rank | share of raw lookups | share of deduped | shared lru 16/32/64GB | per-rank lru (K/2 each) 16/32/64GB | per-rank static 16/32/64GB |")
w("|---|---|---|---|---|---|")
ix = [caps.index(c) for c in (16, 32, 64)]
for rk in range(2):
    mm = sel(None, A_R == rk)
    w(f"| {rk} | {p(TR[mm].sum() / TR.sum())} | {p(T[mm].sum() / T.sum())} | "
      + " / ".join(p(rate(H[2, ci], T, mm)) for ci in ix) + " | "
      + " / ".join(p(rate(H[3, ci], T, mm)) for ci in ix) + " | "
      + " / ".join(p(rate(H[4, ci], T, mm)) for ci in ix) + " |")

w("\n### by text class (row lru), prefill / decode")
w("| class | phase | test tokens | dedupe | " + " | ".join(f"lru {c}GB" for c in caps) + " | static 64GB |")
w("|---|---|---|---|" + "---|" * len(caps) + "---|")
for c in range(3):
    for ph in range(2):
        mm = sel((C_CL == c) & (C_PH == ph))
        w(f"| {CLS[c]} | {'prefill' if ph == 0 else 'decode'} | {int(test_tok[ph, c]):,} | {T[mm].sum() / max(1, TR[mm].sum()):.3f} | "
          + " | ".join(p(rate(H[2, ci], T, mm)) for ci in range(len(caps))) + f" | {p(rate(H[0, -1], T, mm))} |")

w("\n### miss reuse: of the misses, fraction re-accessed within N Engram tokens (later in trace)")
lab = ["<=8k", "<=128k", "<=1M", "<=10M", ">10M", "never"]
w("| policy | cap | phase | " + " | ".join(lab) + " |")
w("|---|---|---|" + "---|" * 6)
for pi in (0, 2):
    for ci in (caps.index(16), caps.index(64)):
        for ph in range(2):
            v = RU[pi, ci, ph]
            v = v / max(1, v.sum())
            w(f"| {pols[pi]} | {caps[ci]}GB | {'prefill' if ph == 0 else 'decode'} | " + " | ".join(p(x) for x in v) + " |")

# IO per token
w("\n### NVMe IO per Engram token (test half). row cache: realigned table (1 IO/row); x1.0625 if original layout")
w("| policy | phase | " + " | ".join(f"{c}GB" for c in caps) + " |")
w("|---|---|" + "---|" * len(caps))
io = {}
for ph in range(2):
    mm = sel(C_PH == ph)
    ntok = test_tok[ph].sum()
    for pi, pn in enumerate(pols):
        if pn not in ("static", "lru", "oracle"):
            continue
        vals = [(T[mm].sum() - H[pi, ci][mm].sum()) / f / ntok for ci in range(len(caps))]
        io[(pn, ph)] = vals
        w(f"| row {pn} | {'prefill' if ph == 0 else 'decode'} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    for pi, pn in enumerate(r["pol_page"]):
        vals = [(PT[mm].sum() - PH[pi, ci][mm].sum()) / f / ntok for ci in range(len(caps))]
        w(f"| page4K {pn} | {'prefill' if ph == 0 else 'decode'} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    vals = [T[mm].sum() / f / ntok]
    w(f"| no cache (rows) | {'prefill' if ph == 0 else 'decode'} | {vals[0]:.2f} |")
    io[("none", ph)] = vals * len(caps)

w("\n### derived: prefill IOPS ceiling (230k IOPS) and decode c16 per-step IO (64 verified tokens/step)")
w("| policy | cap | prefill IO/tok | prefill tok/s at 230k IOPS | decode IO/tok | c16 IO/step (64 tok) | c16 IO/s at 30 step/s |")
w("|---|---|---|---|---|---|---|")
for pn in ("none", "static", "lru"):
    for ci, c in enumerate(caps):
        a = io[(pn, 0)][ci]
        b = io[(pn, 1)][ci]
        w(f"| {pn} | {c}GB | {a:.2f} | {230000 / a:,.0f} | {b:.2f} | {64 * b:.0f} | {64 * b * 30:,.0f} |")
        if pn == "none":
            break

# unseen-task subset
if "u_tot_ev" in r:
    UH = np.array(r["u_hit_ev"], float)
    UT = np.array(r["u_tot_ev"], float)
    UC = np.array(r["u_cold"], float)
    ts = r["task_stats"]
    w(f"\n### unseen-task subset (test requests whose first user turn never appears in the first half): "
      f"{ts['tasks_test_unseen']}/{ts['tasks_test']} test tasks, {ts['test_tokens_unseen']:,}/{ts['test_tokens']:,} test tokens")
    w("| policy | phase | " + " | ".join(f"{c}GB" for c in caps) + " |")
    w("|---|---|" + "---|" * len(caps))
    for ph in range(2):
        mm = sel(C_PH == ph)
        for pi, pn in enumerate(pols):
            if pn in ("lru_rk", "static_rk"):
                continue
            w(f"| row {pn} | {'prefill' if ph == 0 else 'decode'} | "
              + " | ".join(p(rate(UH[pi, ci], UT, mm)) for ci in range(len(caps))) + " |")
        w(f"| cold | {'prefill' if ph == 0 else 'decode'} | {p(UC[mm].sum() / max(1, UT[mm].sum()))} |")
    # IO/token on the unseen subset (lookups/token assumed equal to all-test)
    w("\nIO/token (unseen subset, misses / token, using the subset's deduped accesses per token = all-test value)")
    for ph in range(2):
        mm = sel(C_PH == ph)
        acc_per_tok = T[mm].sum() / f / test_tok[ph].sum()
        for pi in (0, 2):
            vals = [acc_per_tok * (1 - rate(UH[pi, ci], UT, mm)) for ci in range(len(caps))]
            w(f"- {pols[pi]} {'prefill' if ph == 0 else 'decode'}: " + ", ".join(f"{c}GB {v:.2f}" for c, v in zip(caps, vals)))

if "first_hist" in r:
    fh = np.array(r["first_hist"], float) / f
    cumr = np.cumsum(fh)
    w("\n### distinct rows (both layers) vs Engram tokens processed (first-ever accesses, whole trace)")
    w("| tokens | distinct rows | as row-cache GB (x264B) | new rows per 1M tokens in that 1M |")
    w("|---|---|---|---|")
    n = int(np.ceil(r["total_engram_tokens"] / 1e6))
    for m in (1, 2, 5, 10, 20, 30, 40, 50, n):
        if m <= n:
            w(f"| {m}M | {cumr[m - 1]:,.0f} | {cumr[m - 1] * 264 / 1e9:.1f} | {fh[m - 1]:,.0f} |")

if "req_miss" in r:
    rqz = np.load(sys.argv[2] + "/req.npz")
    RM = np.array(r["req_miss"], float)[:-1] / f  # [n_req, cap(16,32), phase]
    newp = (rqz["plen"] - rqz["newstart"]).astype(float)
    outl = rqz["olen"].astype(float)
    test = np.arange(len(newp)) > r["split_q"]
    w("\n### per-request prefill IO/token (test half, LRU row cache), token-weighted percentiles")
    w("| subset | cap | requests | mean | p50 | p90 | p99 | max |")
    w("|---|---|---|---|---|---|---|---|")

    def wpct(x, wt, qs):
        o = np.argsort(x)
        c = np.cumsum(wt[o]) / wt.sum()
        return [x[o][np.searchsorted(c, q_)] for q_ in qs]

    for lab_, m in (("new prefill >= 1 token", test & (newp > 0)), ("new prefill >= 8k", test & (newp >= 8192)),
                    ("new prefill >= 64k", test & (newp >= 65536))):
        for ci, c in enumerate((16, 32)):
            x = RM[m, ci, 0] / newp[m]
            pc = wpct(x, newp[m], (0.5, 0.9, 0.99))
            w(f"| {lab_} | {c}GB | {int(m.sum())} | {RM[m, ci, 0].sum() / newp[m].sum():.2f} | "
              + " | ".join(f"{v:.2f}" for v in pc) + f" | {x.max():.2f} |")

print("\n".join(out))
