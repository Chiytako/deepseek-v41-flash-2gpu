"""X2 stage 2: time-order requests, apply a prefix cache, classify text (CPU only).

Inputs: <req_dir>/*.u32 + *.meta.json from x2_extract.py.
Outputs in <out_dir>:
  tokens.u32   all prompt+output tokens, requests in arrival-time order
  req.npz      per request: start (offset into tokens.u32), plen, olen, newstart (first prompt token
               that is not served by the prefix cache), glm_new (GLM-reported uncached prompt tokens),
               glm_out, tools, n_img, ts (unix seconds), ord (row index into meta)
  cls.u8       per token class for the Engram-relevant region (newstart .. plen+olen): 0=ja/CJK, 1=code,
               2=other; 255 elsewhere
  prep_stats.json

Prefix cache model: global, unbounded, block-granular (BLOCK tokens, chain hash over full blocks of
prompt+output; a request reuses the longest chain of previously inserted blocks). This assumes the
V4.1 server keeps every earlier session prefix (KV is ~0.9KB/token, so 50M tokens ~ 45GB host tier).

usage: x2_prep.py <req_dir> <out_dir> [block=256]
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import glob
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime

import numba as nb
import numpy as np

TOK_PATH = MODELS + "/DeepSeek-V4.1-Flash-1017/tokenizer.json"
WIN = 256


@nb.njit(cache=True)
def _mix(h, x):
    h ^= x + np.uint64(0x9E3779B97F4A7C15) + (h << np.uint64(6)) + (h >> np.uint64(2))
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(31)
    return h


@nb.njit(cache=True)
def prefix_cache(tokens, starts, plens, olens, block, table_bits):
    cap = np.uint64(1) << np.uint64(table_bits)
    mask = cap - np.uint64(1)
    table = np.zeros(cap, np.uint64)
    n = starts.shape[0]
    newstart = np.zeros(n, np.int64)
    inserted = 0
    for q in range(n):
        s = starts[q]
        P = plens[q]
        L = P + olens[q]
        h = np.uint64(1469598103934665603)
        nblk_all = L // block
        hit_blocks = 0
        missed = False
        for b in range(nblk_all):
            for t in range(b * block, (b + 1) * block):
                h = _mix(h, np.uint64(tokens[s + t]) + np.uint64(1))
            if h == 0:
                h = np.uint64(1)
            slot = h & mask
            found = False
            while table[slot] != 0:
                if table[slot] == h:
                    found = True
                    break
                slot = (slot + np.uint64(1)) & mask
            if found:
                if not missed and (b + 1) * block <= P:
                    hit_blocks = b + 1
            else:
                missed = True
                table[slot] = h
                inserted += 1
                if inserted * 10 > cap * 7:
                    return newstart, -1
        # at least the last prompt token is always computed
        ns = hit_blocks * block
        if ns >= P:
            ns = P - 1 if P > 0 else 0
        newstart[q] = ns
    return newstart, inserted


def classify(text):
    ns = 0
    cjk = 0
    kana = 0
    sym = 0
    for ch in text:
        if ch.isspace():
            continue
        ns += 1
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF or 0xFF66 <= o <= 0xFF9D:
            kana += 1
            cjk += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFF65:
            cjk += 1
        elif ch in "{}[]();=<>_/\\|$#&*+`%^~":
            sym += 1
    if ns == 0:
        return 2
    if cjk / ns >= 0.25:
        return 0
    if sym / ns >= 0.06:
        return 1
    return 2


_tok = None


def _init():
    global _tok
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(TOK_PATH)


def main():
    req_dir, out_dir = sys.argv[1], sys.argv[2]
    block = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    os.makedirs(out_dir, exist_ok=True)
    metas = []
    for mpath in sorted(glob.glob(os.path.join(req_dir, "*.meta.json"))):
        name = os.path.basename(mpath)[:-len(".meta.json")]
        ms = json.load(open(mpath))
        off = 0
        for m in ms:
            m["foff"] = off
            m["fname"] = name
            off += m["plen"] + m["olen"]
            metas.append(m)
    for m in metas:
        m["t"] = datetime.fromisoformat(m["ts"]).timestamp()
    metas.sort(key=lambda m: m["t"])
    n = len(metas)
    plen = np.array([m["plen"] for m in metas], np.int64)
    olen = np.array([m["olen"] for m in metas], np.int64)
    total = int((plen + olen).sum())
    out_tok = np.lib.format.open_memmap(os.path.join(out_dir, "tokens.npy"), mode="w+", dtype=np.uint32,
                                        shape=(total,))
    starts = np.zeros(n, np.int64)
    cur = 0
    mm = {}
    for i, m in enumerate(metas):
        if m["fname"] not in mm:
            mm[m["fname"]] = np.memmap(os.path.join(req_dir, m["fname"] + ".u32"), dtype=np.uint32, mode="r")
        L = m["plen"] + m["olen"]
        out_tok[cur:cur + L] = mm[m["fname"]][m["foff"]:m["foff"] + L]
        starts[i] = cur
        cur += L
    out_tok.flush()
    del mm
    tokens = np.load(os.path.join(out_dir, "tokens.npy"), mmap_mode="r")
    newstart, ins = prefix_cache(np.asarray(tokens), starts, plen, olen, block, 28)
    assert ins >= 0, "prefix table overflow"
    glm_new = np.array([(m["glm_prompt"] or 0) - (m["glm_cached"] or 0) for m in metas], np.int64)
    np.savez(os.path.join(out_dir, "req.npz"), start=starts, plen=plen, olen=olen, newstart=newstart,
             glm_new=glm_new, glm_out=np.array([m["glm_completion"] or 0 for m in metas], np.int64),
             tools=np.array([m["tools"] for m in metas]), n_img=np.array([m["n_img"] for m in metas]),
             ts=np.array([m["t"] for m in metas]), block=block)
    # classes for the Engram region
    cls = np.lib.format.open_memmap(os.path.join(out_dir, "cls.npy"), mode="w+", dtype=np.uint8, shape=(total,))
    cls[:] = 255
    jobs, cur_job, cur_sz = [], [], 0
    for i in range(n):
        a = int(starts[i] + newstart[i])
        b = int(starts[i] + plen[i] + olen[i])
        if b > a:
            cur_job.append((a, b))
            cur_sz += b - a
        if cur_sz > 2_000_000:
            jobs.append((os.path.join(out_dir, "tokens.npy"), cur_job))
            cur_job, cur_sz = [], 0
    if cur_job:
        jobs.append((os.path.join(out_dir, "tokens.npy"), cur_job))
    # tokens.npy has a .npy header; workers memmap past it
    hdr = np.load(os.path.join(out_dir, "tokens.npy"), mmap_mode="r").offset
    with mp.Pool(28, initializer=_init) as pool:
        for res in pool.imap_unordered(_cls_job_hdr, [(p, r, hdr) for p, r in jobs]):
            for a, b, c in res:
                cls[a:b] = np.repeat(c, WIN)[:b - a]
    cls.flush()
    newp = int((plen - newstart).sum())
    st = {"requests": n, "prompt_tokens_v41": int(plen.sum()), "output_tokens_v41": int(olen.sum()),
          "new_prefill_tokens_v41": newp, "prefix_hit_frac": 1 - newp / max(1, int(plen.sum())),
          "glm_new_prefill": int(glm_new.sum()), "glm_prompt": int(sum((m["glm_prompt"] or 0) for m in metas)),
          "glm_out": int(sum((m["glm_completion"] or 0) for m in metas)), "block": block,
          "unique_blocks": int(ins), "t0": metas[0]["ts"], "t1": metas[-1]["ts"]}
    for c in range(3):
        st[f"cls{c}_tokens"] = int((cls == c).sum())
    json.dump(st, open(os.path.join(out_dir, "prep_stats.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))


def _cls_job_hdr(args):
    path, ranges, hdr = args
    toks = np.memmap(path, dtype=np.uint32, mode="r", offset=hdr)
    out = []
    for a, b in ranges:
        texts = [toks[w:min(b, w + WIN)].tolist() for w in range(a, b, WIN)]
        decoded = _tok.decode_batch(texts, skip_special_tokens=True)
        out.append((a, b, np.array([classify(t) for t in decoded], np.uint8)))
    return out


if __name__ == "__main__":
    main()
