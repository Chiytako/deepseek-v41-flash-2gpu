"""Engram hash-id simulation (CPU only), replicating ref/engram.py exactly for config of DeepSeek-V4.1-Flash-1017.

usage: engram_sim.py <tokenizer.json> <out.json> <train_files...> -- <test_files...>
"""
import json
import sys
from collections import Counter

import numpy as np
from sympy import isprime
from tokenizers import Regex, Tokenizer, normalizers

LAYER_IDS = (1, 14)
MAX_N = 4
N_HEADS = 8
ENGRAM_VOCAB = 16000000
PAD_ID = 2
EXPECTED_CVOCAB = 99092
ROW_BYTES = 256 + 8  # fp8 row + 8 ue8m0 scales


def build_map(tok: Tokenizer):
    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    n = tok.get_vocab_size(with_added_tokens=True)
    key_to_new, lookup = {}, [0] * n
    for t in range(n):
        text = tok.decode([t], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(t)
        else:
            nn = norm.normalize_str(text)
            key = nn if nn else text
        v = key_to_new.get(key)
        if v is None:
            v = len(key_to_new)
            key_to_new[key] = v
        lookup[t] = v
    return np.array(lookup, dtype=np.int64), len(key_to_new)


def primes_layout():
    primes, seen = [], set()
    for _ in LAYER_IDS:
        per = []
        for _ in range(MAX_N - 1):
            sizes, cur = [], ENGRAM_VOCAB - 1
            for _ in range(N_HEADS):
                c = cur + 1
                while not isprime(c) or c in seen:
                    c += 1
                cur = c
                seen.add(c)
                sizes.append(c)
            per.append(sizes)
        primes.append(per)
    return np.array(primes, dtype=np.int64)  # [L, 3, 8]


def multipliers(cvocab):
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // cvocab) // 2)
    rows = []
    for lid in LAYER_IDS:
        g = np.random.default_rng(10007 * lid)
        v = g.integers(low=0, high=bound, size=(MAX_N,), dtype=np.int64)
        rows.append(v * 2 + 1)
    return np.stack(rows)  # [L, 4]


def hash_ids(cids, pad_c, primes, mults):
    """cids: [T] compressed ids of one document. returns [T, L, 24] global row ids."""
    T = len(cids)
    toks = []
    pos = np.arange(T)
    blocked = np.zeros(T, dtype=bool)
    for s in range(MAX_N):
        src = cids[np.clip(pos - s, 0, None)]
        blocked = blocked | (pos < s)
        toks.append(np.where(blocked, pad_c, src))
    toks = np.stack(toks, -1)  # [T, 4]
    with np.errstate(over="ignore"):
        prod = toks[:, None, :] * mults[None]  # [T, L, 4]
    rolling = prod[..., 0]
    out = []
    for i in range(1, MAX_N):
        rolling = np.bitwise_xor(rolling, prod[..., i])
        out.append(rolling[..., None] % primes[None, :, i - 1])  # [T, L, 8]
    h = np.concatenate(out, -1)  # [T, L, 24]
    flat = primes.reshape(len(LAYER_IDS), -1)
    offs = np.concatenate([np.zeros((len(LAYER_IDS), 1), np.int64), np.cumsum(flat, 1)[:, :-1]], 1)
    return h + offs[None]


def load_ids(tok, cmap, files, chunk_chars=200000):
    docs = []
    for f in files:
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for i in range(0, len(text), chunk_chars):
            ids = tok.encode(text[i:i + chunk_chars], add_special_tokens=False).ids
            if ids:
                docs.append(cmap[np.array(ids, dtype=np.int64)])
    return docs


def main():
    tok_path, out_path = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    sep = rest.index("--")
    train_files, test_files = rest[:sep], rest[sep + 1:]
    tok = Tokenizer.from_file(tok_path)
    cmap, cvocab = build_map(tok)
    res = {"compressed_vocab": cvocab, "expected": EXPECTED_CVOCAB}
    primes = primes_layout()
    res["table_rows_per_layer"] = [int(primes[l].sum()) for l in range(len(LAYER_IDS))]
    mults = multipliers(cvocab)
    pad_c = int(cmap[PAD_ID])
    train = load_ids(tok, cmap, train_files)
    test = load_ids(tok, cmap, test_files)
    res["train_tokens"] = int(sum(len(d) for d in train))
    res["test_tokens"] = int(sum(len(d) for d in test))
    for li, lid in enumerate(LAYER_IDS):
        ctr = Counter()
        for d in train:
            ctr.update(hash_ids(d, pad_c, primes, mults)[:, li, :].ravel().tolist())
        total = sum(ctr.values())
        freqs = np.array(sorted(ctr.values(), reverse=True), dtype=np.int64)
        cum = np.cumsum(freqs) / total
        r = {"train_lookups": int(total), "train_unique_rows": int(len(freqs)),
             "train_unique_frac_of_lookups": len(freqs) / total,
             "train_singleton_rows": int((freqs == 1).sum())}
        for k in (10_000, 100_000, 1_000_000, 4_000_000):
            if k <= len(freqs):
                r[f"train_cov_top{k}"] = float(cum[k - 1])
        # per n-gram order uniqueness (cols 0-7 = 2gram, 8-15 = 3gram, 16-23 = 4gram)
        test_ids = [hash_ids(d, pad_c, primes, mults)[:, li, :] for d in test]
        tl = np.concatenate(test_ids, 0)
        r["test_lookups"] = int(tl.size)
        r["test_unique_rows"] = int(np.unique(tl).size)
        ranked = [row for row, _ in ctr.most_common()]
        for k in (10_000, 100_000, 1_000_000, 4_000_000, len(ranked)):
            hot = np.array(ranked[:k], dtype=np.int64)
            hit = np.isin(tl, hot)
            key = "all_train" if k == len(ranked) else str(k)
            r[f"test_hit_cache{key}"] = {
                "rows": int(min(k, len(ranked))),
                "bytes_GB": min(k, len(ranked)) * ROW_BYTES / 1e9,
                "hit_all": float(hit.mean()),
                "hit_2gram": float(hit[:, 0:8].mean()),
                "hit_3gram": float(hit[:, 8:16].mean()),
                "hit_4gram": float(hit[:, 16:24].mean()),
            }
        res[f"layer{lid}"] = r
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
