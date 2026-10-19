"""zh_detect の機能語を選び、偽陽性率と検出率を測る (CPU)。

日本語コーパス (偽陽性): work/calib/src の wiki_ja / aozora / oasst2_ja / dolly_ja、work/r3ja/data の JSQuAD
  文脈、MMLU-ProX ja、ELYZA-tasks-100、Japanese MT-Bench、M-IFEval-Ja、research/*.md (r5_* を除く)。
中国語コーパス (検出率): work/zh/data の alpaca-zh (簡体字の対話)、zh wikipedia (簡繁混在)。
合成: 日本語の回答 (oasst2_ja の assistant) に中国語の 1 文を差し込んだもの。差し込む文が ZS 字を含まない場合
  (機能語と句の規則だけで拾う必要がある難しい場合) を別に数える。

出力: tools/zh_words.json (strong / weak の機能語と、各語の出現率)、標準出力に表。
usage: zh_calib.py [--write]
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import glob
import json
import os
import random
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zh_detect as Z  # noqa: E402

ROOT = DSV41


def _jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ja_corpora():
    """name -> list[str] (文書)"""
    C = {}
    src = f"{ROOT}/work/calib/src"
    C["wiki_ja"] = [r["text"] for r in _jsonl(f"{src}/wiki_ja.jsonl")]
    C["aozora"] = [r["text"] for r in _jsonl(f"{src}/aozora.jsonl")]
    oa = []
    oa_asst = []
    for r in _jsonl(f"{src}/oasst2_ja.jsonl"):
        for m in r.get("conversations", []):
            oa.append(m.get("content", ""))
            if m.get("role") == "assistant":
                oa_asst.append(m.get("content", ""))
    C["oasst2_ja"] = oa
    C["dolly_ja"] = [" ".join(str(r.get(k, "")) for k in ("instruction", "input", "output")) for r in _jsonl(f"{src}/dolly_ja.jsonl")]
    try:
        import pandas as pd
        d = pd.read_parquet(f"{ROOT}/work/r3ja/data/jsquad_val.parquet")
        C["jsquad"] = sorted(set(d["context"].tolist()))
        d = pd.read_parquet(f"{ROOT}/work/r3ja/data/mmluprox_ja.parquet")
        C["mmluprox_ja"] = [" ".join(str(r[k]) for k in ["question"] + [f"option_{i}" for i in range(10)] + ["cot_content"] if r.get(k) is not None) for r in d.to_dict("records")]
        d = pd.read_csv(f"{ROOT}/work/r3ja/data/elyza100_test.csv")
        C["elyza100"] = [f"{a}\n{b}" for a, b in zip(d["input"], d["output"])]
    except ImportError:
        pass
    C["jmtbench"] = ["\n".join(r["turns"]) for r in _jsonl(f"{ROOT}/work/r3ja/data/jmtbench_q.jsonl")]
    C["mifeval_ja"] = [r["prompt"] for r in _jsonl(f"{ROOT}/work/r3ja/data/mifeval_ja.jsonl")]
    md = []
    for p in sorted(glob.glob(f"{ROOT}/research/*.md")):
        if os.path.basename(p).startswith("r5_"):
            continue
        md.append(open(p, encoding="utf-8").read())
    C["research_md"] = md
    return C, oa_asst


def zh_corpora():
    C = {"alpaca_zh": [], "wiki_zh": []}
    for p in sorted(glob.glob(f"{ROOT}/work/zh/data/alpaca_zh_*.json")):
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            r = r["row"]
            C["alpaca_zh"].append(f"{r['instruction']}\n{r['input']}\n{r['output']}")
    for p in sorted(glob.glob(f"{ROOT}/work/zh/data/wiki_zh_*.json")):
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            C["wiki_zh"].append(r["row"]["text"][:6000])
    return C


def count_words(docs, words):
    rx = re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)))
    c = Counter()
    nch = 0
    for d in docs:
        t = Z.clean(d)
        nch += len(t)
        for m in rx.finditer(t):
            c[m.group()] += 1
    return c, nch


def sentences_zh(docs):
    out = []
    for d in docs:
        for s in re.split(r"(?<=[。！？])", d):
            s = s.strip()
            if 8 <= len(s) <= 80 and not any(Z.is_kana(ord(ch)) for ch in s) and sum(Z.is_han(ord(ch)) for ch in s) >= 0.7 * len(s):
                out.append(s)
    return out


def main(argv):
    write = "--write" in argv
    ja, oa_asst = ja_corpora()
    zh = zh_corpora()
    ja_all = [d for v in ja.values() for d in v]
    zh_s = zh["alpaca_zh"]

    # ---- 1. 機能語候補の出現率 (100 万字あたり)
    cj, nj = count_words(ja_all, Z.CANDIDATES)
    cz, nz = count_words(zh_s, Z.CANDIDATES)
    cw, nw = count_words(zh["wiki_zh"], Z.CANDIDATES)
    print(f"[corpus] ja {len(ja_all)} docs {nj/1e6:.2f}M chars | alpaca_zh {len(zh_s)} docs {nz/1e6:.3f}M | wiki_zh {len(zh['wiki_zh'])} docs {nw/1e6:.3f}M")
    rows = []
    for w in Z.CANDIDATES:
        rj = cj[w] / nj * 1e6
        rz = (cz[w] + cw[w]) / (nz + nw) * 1e6
        has_zs = any(Z.char_class(c) == "ZS" for c in w)
        rows.append((w, cj[w], rj, cz[w] + cw[w], rz, has_zs))
    strong, weak = [], []
    # 日本語の技術文で連接として出得るもの (完了一覧、京都不動産、その他是非、自我是、来看) は weak に落とす
    DEMOTE = {"了一", "都不", "他是", "我是", "来看", "是一", "基本上", "是否", "知道", "然而"}
    for w, nj_, rj, nz_, rz, has_zs in rows:
        if len(w) == 1:
            continue  # 1 字は句の規則で扱う
        if nz_ < 3:
            continue  # 中国語でも稀なものは使わない
        if w in DEMOTE:
            weak.append(w)
        elif nj_ == 0 or (rj < 0.3 and rz / max(rj, 1e-9) > 1000):
            strong.append(w)
        elif rj < 5 and rz / rj > 100:
            weak.append(w)
    print("\n[words] w ja_n ja_per_M zh_n zh_per_M has_ZS -> class")
    for w, nj_, rj, nz_, rz, has_zs in sorted(rows, key=lambda r: -r[4]):
        cls = "strong" if w in strong else "weak" if w in weak else "-"
        print(f"  {w}\t{nj_}\t{rj:.2f}\t{nz_}\t{rz:.0f}\t{int(has_zs)}\t{cls}")

    # 日本語コーパスで出た語の文脈 (偽陽性の確認用)
    print("\n[ja contexts of weak/rejected words] (先頭 1 例ずつ)")
    for w, nj_, rj, nz_, rz, has_zs in rows:
        if nj_ and len(w) > 1 and w not in strong and rz > 200:
            for d in ja_all:
                i = d.find(w)
                if i >= 0:
                    print(f"  {w}: …{d[max(0, i-15):i+15]!r}")
                    break

    if write:
        json.dump({"strong": strong, "weak": weak,
                   "rates": {w: {"ja_n": a, "ja_per_M": round(b, 3), "zh_n": c, "zh_per_M": round(d, 1)} for w, a, b, c, d, _ in rows},
                   "corpus_chars": {"ja": nj, "zh": nz + nw}},
                  open(f"{HERE}/zh_words.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        Z._words.cache_clear()
        Z._word_res.cache_clear()
        print(f"\n[write] strong {len(strong)} weak {len(weak)} -> {HERE}/zh_words.json")

    # ---- 2. ZS / ZT 字の日本語コーパスでの出現
    zs_c, zt_c = Counter(), Counter()
    for d in ja_all:
        for ch in d:
            k = Z.char_class(ch)
            if k == "ZS":
                zs_c[ch] += 1
            elif k == "ZT":
                zt_c[ch] += 1
    print(f"\n[ja chars] ZS 字の出現 {sum(zs_c.values())} 回 / {len(zs_c)} 種、ZT {sum(zt_c.values())} 回 / {len(zt_c)} 種 (全 {nj/1e6:.2f}M 字)")
    print("  ZS 上位:", " ".join(f"{c}{n}" for c, n in zs_c.most_common(40)))
    print("  ZT 上位:", " ".join(f"{c}{n}" for c, n in zt_c.most_common(30)))

    # ---- 3. 文書単位の判定率 (偽陽性)
    print("\n[FP] 日本語コーパスでの文書単位の判定 (level>=1 / >=2 / =3)、1 万字あたりの level>=2 文書数")
    for name, docs in ja.items():
        lv = Counter()
        nch = 0
        ex = []
        for d in docs:
            r = Z.detect(d)
            nch += r["chars"]
            lv[r["level"]] += 1
            if r["level"] >= 2 and len(ex) < 3:
                ex.append((r["level"], r["zs"][:10], r["strong_words"][:4], [c["text"][:40] for c in r["clauses"][:1]]))
        n = len(docs)
        ge1 = sum(v for k, v in lv.items() if k >= 1)
        ge2 = lv[2] + lv[3]
        print(f"  {name:12s} n={n:6d} chars={nch/1e6:6.2f}M  >=1 {ge1/n:7.2%}  >=2 {ge2/n:7.2%}  =3 {lv[3]/n:7.2%}  per10k {ge2/nch*1e4:.3f}  例 {ex}")

    # ---- 4. 検出率 (中国語)
    print("\n[recall] 中国語の文書 (level>=2 / =3)")
    for name, docs in zh.items():
        lv = Counter(Z.detect(d)["level"] for d in docs)
        n = len(docs)
        print(f"  {name:12s} n={n}  >=2 {(lv[2]+lv[3])/n:.2%}  =3 {lv[3]/n:.2%}")

    # ---- 5. 合成: 日本語の回答に中国語 1 文を差し込む
    rnd = random.Random(20261018)
    sents = sentences_zh(zh_s)
    s_zs = [s for s in sents if any(Z.char_class(c) == "ZS" for c in s)]
    s_j = [s for s in sents if not any(Z.char_class(c) in ("ZS", "ZT") for c in s)]
    base = [a for a in oa_asst if Z.is_japanese(a) and Z.detect(a)["level"] == 0][:600]
    print(f"\n[synthetic] base ja answers {len(base)}、zh 文 {len(sents)} (ZS あり {len(s_zs)}、J 字のみ {len(s_j)})")
    for label, pool in (("ZS を含む文", s_zs), ("J 字だけの文", s_j)):
        hit = Counter()
        for a in base:
            parts = re.split(r"(?<=[。\n])", a)
            i = rnd.randrange(len(parts) + 1)
            s = rnd.choice(pool)
            t = "".join(parts[:i]) + s + "".join(parts[i:])
            lv = Z.detect(t)["level"]
            hit[lv] += 1
        n = len(base)
        print(f"  {label}: >=1 {(n-hit[0])/n:.1%} >=2 {(hit[2]+hit[3])/n:.1%} =3 {hit[3]/n:.1%}")
    print("  J 字だけの文の例:", s_j[:5])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
