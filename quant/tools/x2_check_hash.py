"""Cross-check x2_sim.gen_events row ids against the round-1 numpy hash (engram_sim.hash_ids,
a transcription of ref/engram.py) on a few real requests. usage: x2_check_hash.py <prep_dir>"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import sys
import numpy as np
sys.path.insert(0, DSV41 + "/tools")
import x2_sim as X
import engram_sim as S

prep = sys.argv[1]
cm, cv, primes, mults, offs, pad_c = X.build_constants()
print("compressed vocab", cv, "rows/layer", [int(primes[l].sum()) for l in range(2)])
tokens = np.load(prep + "/tokens.npy", mmap_mode="r")
cls = np.load(prep + "/cls.npy", mmap_mode="r")
rq = np.load(prep + "/req.npz")
W = 64
pages = max((X.BASE[L] + 256 * X.NROWS[L]) // 4096 + 2 for L in range(2))
ppl = pages // W + 2
rpl = ppl * 16
for q in (5, 1000, 20000):
    sel = slice(q, q + 1)
    st = rq["start"][sel]; pl = rq["plen"][sel]; ol = rq["olen"][sel]; ns = rq["newstart"][sel]
    tk = np.asarray(tokens[int(st[0]):int(st[0] + pl[0] + ol[0])]).astype(np.int64)
    ref = S.hash_ids(cm[tk].clip(0), pad_c, primes, mults)[int(ns[0]):]  # [T, 2, 24]
    for j in (0, 17):
        res = X.gen_events(tokens, cls, st, pl, ol, ns, cm, pad_c, primes, mults, offs, W, j, rpl, ppl,
                           10_000_000, 10_000_000)
        n_ev = res[0]
        got = set(res[3][:n_ev].tolist())
        exp = set()
        for L in range(2):
            r = ref[:, L, :].ravel()
            byte = X.BASE[L] + 256 * r
            pg = byte >> 12
            m = pg % W == j
            lr = L * rpl + (pg[m] // W) * 16 + ((byte[m] & 4095) >> 8)
            exp |= set(lr.tolist())
        print("req", q, "part", j, "tokens", len(ref), "rows", len(got), "match", got == exp)
