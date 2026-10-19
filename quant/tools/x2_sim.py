"""X2 stage 3: Engram row / page cache simulation on the real traffic (CPU only).

Row ids are computed with a re-implementation of ref/engram.py (compressed vocab 99092, primes,
multipliers, rolling XOR hash, offsets, pad at sequence start, image tokens DEAD + masked).

Engram-relevant tokens per request = prompt tokens after the prefix-cache hit (x2_prep newstart)
+ all output tokens. Prefill is cut into 8192-token chunks (from newstart), decode into groups of
4 tokens (one DSpark verify step of one sequence). Inside a group a row is fetched once (dedupe).

Sampling: the 2 tables are split into 4KiB pages of the original safetensors layout
(row r of layer L starts at base_L + 256 r, base = 664 / 672). A worker owns all pages with
page % W == j (stratified 1/W sample of pages and of rows). Capacities are scaled by 1/W
(SHARDS-style spatial sampling). Every lookup touching an owned row/page is simulated exactly.

Policies (all caches warm from the first half, measured on the second half):
  row  : cache unit = one 264B row (fp8 256B + scale 8B)
     static  : top-K rows by first-half (deduped) access count
     oracle  : top-K rows by second-half count (upper bound for a static table)
     lru     : LRU over the deduped access stream, shared by both TP ranks
     lru_rk  : two LRUs of K/2 each, one per TP rank (rows split at ceil(N/2))
     static_rk: per-rank top-K/2
  page : cache unit = one 4KiB page of the original weight tensor (scale tensor assumed resident)
     static / lru as above
usage: x2_sim.py <prep_dir> <out.json> [W=64] [parts=0,4,8,...] [nproc=8]
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import json
import multiprocessing as mp
import os
import sys
import time

import numba as nb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # engram_sim.py = round-1 copy

TOK_PATH = MODELS + "/DeepSeek-V4.1-Flash-1017/tokenizer.json"
LAYER_IDS = (1, 14)
NROWS = (384006168, 384016682)
BASE = (664, 672)
ROWB = 264
CAPS_GB = (1, 4, 16, 32, 48, 64)
CHUNK = 8192
DGROUP = 4
IMG_TOKEN = 129264
DEAD = -1
REUSE_EDGES = (8192, 131072, 1 << 20, 10 << 20)  # tokens; buckets: <=8k,<=128k,<=1M,<=10M,later,never
POL_ROW = ("static", "oracle", "lru", "lru_rk", "static_rk")
POL_PAGE = ("static", "lru")


def build_constants():
    from tokenizers import Tokenizer
    import engram_sim as S  # round-1 simulator (same map / primes / multipliers as ref/engram.py)
    tok = Tokenizer.from_file(TOK_PATH)
    cmap, cv = S.build_map(tok)
    primes = S.primes_layout()  # [2,3,8]
    mults = S.multipliers(cv)  # [2,4]
    flat = primes.reshape(2, -1)
    offs = np.concatenate([np.zeros((2, 1), np.int64), np.cumsum(flat, 1)[:, :-1]], 1)  # [2,24]
    pad_c = int(cmap[2])
    cm = cmap.astype(np.int64).copy()
    cm[IMG_TOKEN] = DEAD
    assert [int(primes[l].sum()) for l in range(2)] == list(NROWS), "rows mismatch"
    return cm, cv, primes, mults, offs, pad_c


# ----------------------------------------------------------------------------- event generation
@nb.njit
def gen_events(tokens, cls, rq_start, rq_plen, rq_olen, rq_new, cmap, pad_c, primes, mults, offs,
               W, j, rpl, ppl, ev_cap, pev_cap):
    # outputs
    ev_row = np.empty(ev_cap, np.int32)
    ev_w = np.zeros(ev_cap, np.uint16)
    ev_cat = np.empty(ev_cap, np.uint8)
    ev_tok = np.empty(ev_cap, np.int32)
    pev_pg = np.empty(pev_cap, np.int32)
    pev_cat = np.empty(pev_cap, np.uint8)
    pev_tok = np.empty(pev_cap, np.int32)
    attr_row = np.full(2 * rpl, 255, np.uint8)
    attr_pg = np.full(2 * ppl, 255, np.uint8)
    stamp = np.full(2 * rpl, -1, np.int32)
    evof = np.zeros(2 * rpl, np.int32)
    pstamp = np.full(2 * ppl, -1, np.int32)
    n_ev = 0
    n_pev = 0
    half_rows = np.empty(2, np.int64)
    for L in range(2):
        tot = 0
        for a in range(3):
            for h in range(8):
                tot += primes[L, a, h]
        half_rows[L] = (tot + 1) // 2
    gid = -1
    tcount = 0  # global counter of Engram-relevant tokens (all rows, all partitions)
    vals = np.empty(4, np.int64)
    for q in range(rq_start.shape[0]):
        s = rq_start[q]
        P = rq_plen[q]
        E = P + rq_olen[q]
        t0 = rq_new[q]
        # group boundaries
        t = t0
        while t < E:
            if t < P:
                g_end = min(P, t + CHUNK)
                phase = 0
            else:
                g_end = min(E, t + DGROUP)
                phase = 1
            gid += 1
            for tt in range(t, g_end):
                pos = s + tt
                tc = tcount
                tcount += 1
                if tokens[pos] == IMG_TOKEN:
                    continue
                c = cls[pos]
                if c > 2:
                    c = 2
                cat = phase * 3 + c
                blocked = False
                for k in range(4):
                    if tt - k < 0:
                        blocked = True
                    else:
                        src = cmap[tokens[pos - k]]
                        if src == DEAD:
                            blocked = True
                    vals[k] = pad_c if blocked else cmap[tokens[pos - k]]
                for L in range(2):
                    base = 664 if L == 0 else 672
                    rolling = vals[0] * mults[L, 0]
                    for i in range(1, 4):
                        rolling = rolling ^ (vals[i] * mults[L, i])
                        for h in range(8):
                            r = rolling % primes[L, i - 1, h] + offs[L, (i - 1) * 8 + h]
                            byte = base + 256 * r
                            page = byte >> 12
                            within = byte & 4095
                            rk = 1 if r >= half_rows[L] else 0
                            at = L | ((i - 1) << 1) | (rk << 3)
                            if page % W == j:
                                lr = L * rpl + (page // W) * 16 + (within >> 8)
                                attr_row[lr] = at
                                if stamp[lr] == gid:
                                    e = evof[lr]
                                    if ev_w[e] < 65535:
                                        ev_w[e] += 1
                                else:
                                    if n_ev >= ev_cap:
                                        return -1, n_pev, tcount, ev_row, ev_w, ev_cat, ev_tok, pev_pg, pev_cat, pev_tok, attr_row, attr_pg
                                    stamp[lr] = gid
                                    evof[lr] = n_ev
                                    ev_row[n_ev] = lr
                                    ev_w[n_ev] = 1
                                    ev_cat[n_ev] = cat
                                    ev_tok[n_ev] = tc
                                    n_ev += 1
                            npg = 2 if within > 3840 else 1
                            for dp in range(npg):
                                pg = page + dp
                                if pg % W == j:
                                    lp = L * ppl + pg // W
                                    if attr_pg[lp] == 255:
                                        attr_pg[lp] = at
                                    if pstamp[lp] != gid:
                                        if n_pev >= pev_cap:
                                            return n_ev, -1, tcount, ev_row, ev_w, ev_cat, ev_tok, pev_pg, pev_cat, pev_tok, attr_row, attr_pg
                                        pstamp[lp] = gid
                                        pev_pg[n_pev] = lp
                                        pev_cat[n_pev] = cat
                                        pev_tok[n_pev] = tc
                                        n_pev += 1
            t = g_end
    return n_ev, n_pev, tcount, ev_row, ev_w, ev_cat, ev_tok, pev_pg, pev_cat, pev_tok, attr_row, attr_pg


# ----------------------------------------------------------------------------- LRU stack distance
@nb.njit
def stack_dist(keys, sel, nkeys):
    """keys[i] for the events listed in sel (in time order). returns d per listed event (-1 = cold)."""
    n = sel.shape[0]
    bit = np.zeros(n + 1, np.int32)
    last = np.full(nkeys, -1, np.int64)
    d = np.empty(n, np.int32)
    for ii in range(n):
        k = keys[sel[ii]]
        p = last[k]
        if p >= 0:
            # markers in (p, ii) = prefix(ii-1) - prefix(p)
            s1 = 0
            x = ii  # prefix over [0, ii-1] -> 1-based index ii
            while x > 0:
                s1 += bit[x]
                x -= x & (-x)
            s0 = 0
            x = p + 1
            while x > 0:
                s0 += bit[x]
                x -= x & (-x)
            d[ii] = s1 - s0
            x = p + 1
            while x <= n:
                bit[x] -= 1
                x += x & (-x)
        else:
            d[ii] = -1
        x = ii + 1
        while x <= n:
            bit[x] += 1
            x += x & (-x)
        last[k] = ii
    return d


@nb.njit
def next_gap(keys, toks, n, nkeys):
    last = np.full(nkeys, -1, np.int64)
    gap = np.full(n, -1, np.int32)
    for i in range(n - 1, -1, -1):
        k = keys[i]
        if last[k] >= 0:
            gap[i] = toks[last[k]] - toks[i]
        last[k] = i
    return gap


@nb.njit
def count_keys(keys, toks, n, nkeys, split):
    ctr = np.zeros(nkeys, np.int32)
    cte = np.zeros(nkeys, np.int32)
    for i in range(n):
        if toks[i] < split:
            ctr[keys[i]] += 1
        else:
            cte[keys[i]] += 1
    return ctr, cte


def ranks_from_counts(cnt, salt):
    """rank 0 = most frequent; rows with count 0 get rank = big (never cached)."""
    n = cnt.shape[0]
    rnd = (np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15 + salt)) >> np.uint64(40)
    key = cnt.astype(np.uint64) << np.uint64(24) | (rnd & np.uint64(0xFFFFFF))
    order = np.argsort(-key.astype(np.int64), kind="stable")
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    rank[cnt == 0] = np.iinfo(np.int64).max
    return rank


def ranks_per_rank(cnt, attr, salt):
    rk = (attr >> 3) & 1
    out = np.full(cnt.shape[0], np.iinfo(np.int64).max, np.int64)
    for r in (0, 1):
        idx = np.nonzero(rk == r)[0]
        sub = ranks_from_counts(cnt[idx], salt + r)
        out[idx] = sub
    return out


@nb.njit
def tally_row(ev_row, ev_w, ev_cat, ev_tok, n, split, attr_row, ranks, d_lru, d_rk, gap, kcap, edges, evmask):
    """evmask[i] == 0 -> event not tallied (used for the unseen-task subset). ranks: [3, nkeys] static, oracle, static_rk ; kcap: [ncap] rows per partition (total cache).
    returns hit_ev[5, ncap, 6, 16], hit_raw[5, ncap, 6, 16], tot_ev[6,16], tot_raw[6,16],
    reuse[5, ncap, 2(phase), 6 buckets] (misses by reuse distance), cold[6, 16]"""
    ncap = kcap.shape[0]
    hit_ev = np.zeros((5, ncap, 6, 16), np.int64)
    hit_raw = np.zeros((5, ncap, 6, 16), np.int64)
    tot_ev = np.zeros((6, 16), np.int64)
    tot_raw = np.zeros((6, 16), np.int64)
    reuse = np.zeros((5, ncap, 2, 6), np.int64)
    cold = np.zeros((6, 16), np.int64)
    nb_ = edges.shape[0]
    for i in range(n):
        if ev_tok[i] < split or evmask[i] == 0:
            continue
        r = ev_row[i]
        a = attr_row[r]
        c = ev_cat[i]
        w = ev_w[i]
        ph = c // 3
        tot_ev[c, a] += 1
        tot_raw[c, a] += w
        if d_lru[i] < 0:
            cold[c, a] += 1
        g = gap[i]
        if g < 0:
            bk = nb_ + 1
        else:
            bk = nb_
            for e in range(nb_):
                if g <= edges[e]:
                    bk = e
                    break
        for ci in range(ncap):
            K = kcap[ci]
            hs = (ranks[0, r] < K, ranks[1, r] < K, d_lru[i] >= 0 and d_lru[i] < K,
                  d_rk[i] >= 0 and d_rk[i] < K // 2, ranks[2, r] < K // 2)
            for p in range(5):
                if hs[p]:
                    hit_ev[p, ci, c, a] += 1
                    hit_raw[p, ci, c, a] += w
                else:
                    reuse[p, ci, ph, bk] += 1
    return hit_ev, hit_raw, tot_ev, tot_raw, reuse, cold


@nb.njit
def tally_page(pev_pg, pev_cat, pev_tok, n, split, attr_pg, rank, d_lru, kcap):
    ncap = kcap.shape[0]
    hit = np.zeros((2, ncap, 6, 16), np.int64)
    tot = np.zeros((6, 16), np.int64)
    for i in range(n):
        if pev_tok[i] < split:
            continue
        pg = pev_pg[i]
        a = attr_pg[pg]
        if a == 255:
            a = 0
        c = pev_cat[i]
        tot[c, a] += 1
        for ci in range(ncap):
            K = kcap[ci]
            if rank[pg] < K:
                hit[0, ci, c, a] += 1
            if d_lru[i] >= 0 and d_lru[i] < K:
                hit[1, ci, c, a] += 1
    return hit, tot


@nb.njit
def per_req_miss(ev_cat, d_lru, req_of_ev, n, kcap2, n_req):
    """misses per request for two LRU capacities (rows per partition): [n_req, 2 cap, 2 phase]"""
    out = np.zeros((n_req, 2, 2), np.int64)
    for i in range(n):
        q = req_of_ev[i]
        ph = ev_cat[i] // 3
        for c in range(2):
            if d_lru[i] < 0 or d_lru[i] >= kcap2[c]:
                out[q, c, ph] += 1
    return out


# ----------------------------------------------------------------------------- worker
G = {}


def _init(prep_dir, consts, tok_cum, unseen):
    G["prep"] = prep_dir
    G["consts"] = consts
    G["tok_cum"] = tok_cum  # end (exclusive) of each request in the Engram token counter
    G["unseen"] = unseen


def worker(args):
    j, W, split_tok, ev_cap, pev_cap = args
    t0 = time.time()
    prep = G["prep"]
    cm, cv, primes, mults, offs, pad_c = G["consts"]
    tokens = np.load(os.path.join(prep, "tokens.npy"), mmap_mode="r")
    cls = np.load(os.path.join(prep, "cls.npy"), mmap_mode="r")
    rq = np.load(os.path.join(prep, "req.npz"))
    pages_per_layer = max((BASE[L] + 256 * NROWS[L]) // 4096 + 2 for L in range(2))
    ppl = pages_per_layer // W + 2
    rpl = ppl * 16
    while True:
        res = gen_events(tokens, cls, rq["start"], rq["plen"], rq["olen"], rq["newstart"], cm, pad_c, primes,
                         mults, offs, W, j, rpl, ppl, ev_cap, pev_cap)
        n_ev, n_pev = res[0], res[1]
        if n_ev >= 0 and n_pev >= 0:
            break
        ev_cap = int(ev_cap * 1.5)
        pev_cap = int(pev_cap * 1.5)
    (_, _, tcount, ev_row, ev_w, ev_cat, ev_tok, pev_pg, pev_cat, pev_tok, attr_row, attr_pg) = res
    ev_row, ev_w, ev_cat, ev_tok = ev_row[:n_ev], ev_w[:n_ev], ev_cat[:n_ev], ev_tok[:n_ev]
    pev_pg, pev_cat, pev_tok = pev_pg[:n_pev], pev_cat[:n_pev], pev_tok[:n_pev]
    t1 = time.time()
    nkeys = 2 * rpl
    # static ranks
    ctr, cte = count_keys(ev_row, ev_tok, n_ev, nkeys, split_tok)
    ranks = np.stack([ranks_from_counts(ctr, 1), ranks_from_counts(cte, 2), ranks_per_rank(ctr, attr_row, 3)])
    train_rows = int((ctr > 0).sum())
    test_rows = int((cte > 0).sum())
    both_rows = int(((ctr > 0) & (cte > 0)).sum())
    # LRU
    allsel = np.arange(n_ev, dtype=np.int64)
    d_lru = stack_dist(ev_row, allsel, nkeys)
    rk_of_ev = (attr_row[ev_row] >> 3) & 1
    d_rk = np.empty(n_ev, np.int32)
    for r in (0, 1):
        sel = np.nonzero(rk_of_ev == r)[0]
        d_rk[sel] = stack_dist(ev_row, sel, nkeys)
    del rk_of_ev
    gap = next_gap(ev_row, ev_tok, n_ev, nkeys)
    kcap = np.array([int(c * 1e9 / ROWB / W) for c in CAPS_GB], np.int64)
    edges = np.array(REUSE_EDGES, np.int64)
    hit_ev, hit_raw, tot_ev, tot_raw, reuse, cold = tally_row(ev_row, ev_w, ev_cat, ev_tok, n_ev, split_tok,
                                                              attr_row, ranks, d_lru, d_rk, gap, kcap,
                                                              edges, np.ones(n_ev, np.uint8))
    # unseen-task subset: test requests whose task (first user turn) never occurs in the first half
    req_of_ev = np.searchsorted(G["tok_cum"], ev_tok, side="right")
    um = G["unseen"][req_of_ev].astype(np.uint8)
    prm = per_req_miss(ev_cat, d_lru, req_of_ev, n_ev,
                       np.array([int(c * 1e9 / ROWB / W) for c in (16, 32)], np.int64), len(G["tok_cum"]) + 1)
    del req_of_ev
    u_hit_ev, _, u_tot_ev, _, _, u_cold = tally_row(ev_row, ev_w, ev_cat, ev_tok, n_ev, split_tok, attr_row,
                                                    ranks, d_lru, d_rk, gap, kcap, edges, um)
    del um
    # distinct-row growth: first-ever accesses per 1M Engram tokens
    first_hist = np.bincount(ev_tok[d_lru < 0] // 1_000_000, minlength=64)[:64]
    del d_lru, d_rk, gap, ranks
    # pages
    npk = 2 * ppl
    pctr, _ = count_keys(pev_pg, pev_tok, n_pev, npk, split_tok)
    prank = ranks_from_counts(pctr, 4)
    pd = stack_dist(pev_pg, np.arange(n_pev, dtype=np.int64), npk)
    kpg = np.array([int(c * 1e9 / 4096 / W) for c in CAPS_GB], np.int64)
    phit, ptot = tally_page(pev_pg, pev_cat, pev_tok, n_pev, split_tok, attr_pg, prank, pd, kpg)
    t2 = time.time()
    return {"j": j, "n_ev": int(n_ev), "n_pev": int(n_pev), "tcount": int(tcount),
            "train_rows": train_rows, "test_rows": test_rows, "both_rows": both_rows,
            "hit_ev": hit_ev, "hit_raw": hit_raw, "tot_ev": tot_ev, "tot_raw": tot_raw, "reuse": reuse,
            "cold": cold, "phit": phit, "ptot": ptot, "u_hit_ev": u_hit_ev, "u_tot_ev": u_tot_ev,
            "u_cold": u_cold, "first_hist": first_hist, "req_miss": prm, "t_gen": t1 - t0, "t_sim": t2 - t1}


def main():
    prep, out = sys.argv[1], sys.argv[2]
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    parts = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else list(range(0, W, 4))
    nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    consts = build_constants()
    print("compressed vocab", consts[1], flush=True)
    rq = np.load(os.path.join(prep, "req.npz"))
    cls = np.load(os.path.join(prep, "cls.npy"), mmap_mode="r")
    seg = (rq["plen"] + rq["olen"] - rq["newstart"]).astype(np.int64)
    cum = np.cumsum(seg)
    total = int(cum[-1])
    split_q = int(np.searchsorted(cum, total // 2))
    split_tok = int(cum[split_q])  # first token index (in the Engram token counter) of the test half
    # global token counts (test half) by phase x class, image tokens excluded
    tokens = np.load(os.path.join(prep, "tokens.npy"), mmap_mode="r")
    tok_cnt = np.zeros((2, 2, 3), np.int64)  # half, phase, cls
    img_cnt = 0
    for q in range(len(seg)):
        s, P, E, ns = int(rq["start"][q]), int(rq["plen"][q]), int(rq["plen"][q] + rq["olen"][q]), int(rq["newstart"][q])
        h = 1 if q > split_q else 0
        for ph, (a, b) in enumerate(((ns, P), (P, E))):
            if b <= a:
                continue
            tk = np.asarray(tokens[s + a:s + b])
            c = np.minimum(np.asarray(cls[s + a:s + b]), 2)
            ok = tk != IMG_TOKEN
            img_cnt += int((~ok).sum())
            tok_cnt[h, ph] += np.bincount(c[ok], minlength=3)[:3]
    # task id = hash of the first user turn (tokens after the first <｜User｜> up to the next role token)
    import hashlib
    task = []
    for q in range(len(seg)):
        s0, P = int(rq["start"][q]), int(rq["plen"][q])
        head = np.asarray(tokens[s0:s0 + min(P, 60000)])
        u = np.nonzero(head == 128803)[0]
        if len(u) == 0:
            task.append("nouser")
            continue
        a = int(u[0]) + 1
        e = np.nonzero(np.isin(head[a:], (128803, 128804, 128799)))[0]
        b = a + (int(e[0]) if len(e) else min(4096, len(head) - a))
        task.append(hashlib.blake2b(head[a:b].tobytes(), digest_size=8).hexdigest())
    train_tasks = set(task[:split_q + 1])
    unseen = np.array([q > split_q and task[q] not in train_tasks for q in range(len(seg))], bool)
    unseen = np.append(unseen, False)
    test_tasks = set(task[split_q + 1:])
    task_stats = {"tasks_total": len(set(task)), "tasks_train": len(train_tasks), "tasks_test": len(test_tasks),
                  "tasks_test_unseen": len(test_tasks - train_tasks),
                  "test_tokens_unseen": int(seg[unseen[:-1]].sum()),
                  "test_tokens": int(seg[split_q + 1:].sum())}
    print(task_stats, flush=True)
    est = int(total * 48 / W * 1.2) + 1_000_000
    jobs = [(j, W, split_tok, est, est) for j in parts]
    agg = None
    per = []
    with mp.Pool(nproc, initializer=_init, initargs=(prep, consts, cum, unseen)) as pool:
        for r in pool.imap_unordered(worker, jobs):
            print("part", r["j"], "ev", r["n_ev"], "pev", r["n_pev"], "gen %.0fs sim %.0fs" % (r["t_gen"], r["t_sim"]),
                  flush=True)
            per.append({k: r[k] for k in ("j", "n_ev", "n_pev", "train_rows", "test_rows", "both_rows")})
            if agg is None:
                agg = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
            else:
                for k, v in r.items():
                    if isinstance(v, np.ndarray):
                        agg[k] += v
                    elif k in ("train_rows", "test_rows", "both_rows", "n_ev", "n_pev"):
                        agg[k] += v
    res = {"W": W, "parts": parts, "sample_frac": len(parts) / W, "split_q": split_q, "split_tok": split_tok,
           "total_engram_tokens": total, "img_tokens": img_cnt, "tok_cnt_half_phase_cls": tok_cnt.tolist(),
           "caps_gb": CAPS_GB, "pol_row": POL_ROW, "pol_page": POL_PAGE, "reuse_edges": REUSE_EDGES,
           "per_part": per, "task_stats": task_stats}
    for k, v in agg.items():
        if isinstance(v, np.ndarray):
            res[k] = v.tolist()
        elif k in ("train_rows", "test_rows", "both_rows", "n_ev", "n_pev"):
            res[k] = v
    json.dump(res, open(out, "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
