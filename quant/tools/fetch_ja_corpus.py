"""Fetch small samples of public Japanese corpora for build_calib.py (CPU, network, stdlib only).

Uses the Hugging Face datasets-server `rows` API (no full parquet download). Each source is written as
JSONL (one row per line, the dataset's own fields plus `_row`) to <out_dir>/<name>.jsonl, and the
licence / offsets used are recorded in <out_dir>/_sources.json.

Sources (licence checked on the dataset card via https://huggingface.co/api/datasets/<id>, 2026-10-18):
  wiki_ja   wikimedia/wikipedia 20231101.ja   CC BY-SA 3.0 / GFDL   (1,389,467 rows)
  aozora    globis-university/aozorabunko-clean  CC BY 4.0 (dataset); only 作品著作権フラグ=なし kept
  oasst2_ja llm-jp/oasst2-33k-ja              Apache-2.0  (OASST2 translated to Japanese)
  dolly_ja  kunishou/databricks-dolly-15k-ja  CC BY-SA 3.0

usage: python -I fetch_ja_corpus.py <out_dir> [--scale 1.0]
"""
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"
SEED = 20261018

# name: (dataset, config, licence, total_rows, n_pages, page_len)
SOURCES = {
    "wiki_ja": ("wikimedia/wikipedia", "20231101.ja", "CC-BY-SA-3.0 / GFDL", 1389467, 10, 100),
    "aozora": ("globis-university/aozorabunko-clean", "default", "CC-BY-4.0 (texts public domain)", 16951, 40, 10),
    "oasst2_ja": ("llm-jp/oasst2-33k-ja", "default", "Apache-2.0", 32702, 8, 100),
    "dolly_ja": ("kunishou/databricks-dolly-15k-ja", "default", "CC-BY-SA-3.0", 15015, 4, 100),
}
MAX_BYTES = 300 * 1024 * 1024  # hard cap over all sources


def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            print("retry", url, type(e).__name__, e, file=sys.stderr, flush=True)
            time.sleep(3 * (k + 1))


def main():
    out_dir = sys.argv[1]
    scale = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 1.0
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(SEED)
    meta = {}
    total = 0
    for name, (ds, cfg, lic, nrows, npages, plen) in SOURCES.items():
        npages = max(1, int(round(npages * scale)))
        offs = sorted(rng.sample(range(0, nrows - plen), npages))
        path = os.path.join(out_dir, name + ".jsonl")
        n = 0
        with open(path, "w") as f:
            for off in offs:
                q = urllib.parse.urlencode({"dataset": ds, "config": cfg, "split": "train", "offset": off,
                                            "length": plen})
                d = get(API + "?" + q)
                for r in d.get("rows", []):
                    row = dict(r["row"])
                    row["_row"] = r["row_idx"]
                    if r.get("truncated_cells"):
                        row["_truncated"] = r["truncated_cells"]
                    s = json.dumps(row, ensure_ascii=False)
                    total += len(s.encode())
                    if total > MAX_BYTES:
                        raise SystemExit(f"size cap {MAX_BYTES} exceeded")
                    f.write(s + "\n")
                    n += 1
                time.sleep(0.3)
        meta[name] = {"dataset": ds, "config": cfg, "license": lic, "offsets": offs, "page_len": plen, "rows": n,
                      "bytes": os.path.getsize(path)}
        print(name, n, "rows", os.path.getsize(path), "bytes", flush=True)
    meta["_fetched"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["_seed"] = SEED
    with open(os.path.join(out_dir, "_sources.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("total bytes", total)


if __name__ == "__main__":
    main()
