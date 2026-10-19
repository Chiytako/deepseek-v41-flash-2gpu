"""V4.1 tokenizer の語彙を走査し、「中国語専用」トークンの id 集合を作る (CPU)。

分類 (1 トークンの復号バイト列に対して):
  zs    : ZS 字 (簡体字。GB2312 にあり JIS X 0208 / cp932 / 常用 / 人名用にない字) を 1 字以上含む
  zt    : ZS は無いが ZT 字 (日本の規格にも GB2312 にもない漢字。繁体字専用など) を含む
  zw    : 漢字だけからなり、zh_words.json の strong 機能語を含む (例: 可以 没有 但是)。字は日本語にもある
  frag  : UTF-8 として途中で切れたバイト列を含み、完全な文字だけでは判定できない CJK 断片 (先頭バイト 0xE3-0xE9 など)
  ja    : 仮名を含む、または J 字 (日本語の漢字) だけからなる CJK トークン
zh_only = zs ∪ zt (日本語で使われない字を含むトークン)。zh_lex = zw。

出力: tools/zh_only_tokens.json
  {"vocab_size", "sets": {"zs": [...], "zt": [...], "zw": [...], "frag": [...]}, "zh_only": [...], "counts", "examples"}
追加で、日本語コーパス / 中国語コーパス / GLM 日本語回答をトークン化し、各集合のトークン出現率を出す。
usage: zh_vocab_r5.py [--no-corpus]   (r5_zh_mixing.md 用。zh_teacher_mass.py が import する)
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zh_detect as Z  # noqa: E402

TOK = MODELS + "/DeepSeek-V4.1-Flash-1017/tokenizer.json"


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def token_bytes(tj):
    """id -> bytes (added tokens は None)"""
    dec = {v: k for k, v in bytes_to_unicode().items()}
    vocab = tj["model"]["vocab"]
    n = max(max(vocab.values()), max(a["id"] for a in tj["added_tokens"])) + 1
    out = [None] * n
    for s, i in vocab.items():
        try:
            out[i] = bytes(dec[c] for c in s)
        except KeyError:
            out[i] = s.encode("utf-8")
    for a in tj["added_tokens"]:
        out[a["id"]] = None
    return out


def classify(b, strong_re):
    """bytes -> (class, text)"""
    t = b.decode("utf-8", errors="replace")
    has_frag = "�" in t
    cls = [Z.char_class(c) for c in t]
    if "ZS" in cls:
        return "zs", t
    if "ZT" in cls:
        return "zt", t
    core = t.strip()
    if core and all(Z.is_han(ord(c)) for c in core) and strong_re and strong_re.search(core):
        return "zw", t
    if has_frag:
        # CJK 領域の先頭バイト (E3..E9 は U+3000..U+9FFF) か継続バイトだけからなる断片
        if any(0xE2 <= x <= 0xEF or 0xF0 <= x <= 0xF4 for x in b) or all(0x80 <= x <= 0xBF for x in b):
            return "frag", t
        return "other", t
    if any(Z.is_kana(ord(c)) for c in t):
        return "ja", t
    if any(c == "J" or c == "JX" for c in cls):
        return "ja", t  # J 字だけの漢字トークン (日本語と中国語で共有)
    return "other", t


def build():
    tj = json.load(open(TOK, encoding="utf-8"))
    tb = token_bytes(tj)
    strong_re, _ = Z._word_res()
    sets = {"zs": [], "zt": [], "zw": [], "frag": [], "ja": []}
    ex = {k: [] for k in sets}
    for i, b in enumerate(tb):
        if b is None:
            continue
        c, t = classify(b, strong_re)
        if c in sets:
            sets[c].append(i)
            if len(ex[c]) < 40 and (i % 7 == 0):
                ex[c].append(t)
    zh_only = sorted(sets["zs"] + sets["zt"])
    counts = {k: len(v) for k, v in sets.items()}
    counts["zh_only"] = len(zh_only)
    counts["vocab"] = len(tb)
    # 長さの分布 (zs)
    lens = Counter(len(tb[i].decode("utf-8", errors="replace")) for i in sets["zs"])
    return tb, {"vocab_size": len(tb), "tokenizer": TOK, "sets": {k: v for k, v in sets.items() if k != "ja"},
                "zh_only": zh_only, "counts": counts, "zs_len_hist": dict(sorted(lens.items())), "examples": ex,
                "def": "zh_only = zs ∪ zt。zs: ZS 字 (GB2312 にあり JIS X 0208/cp932/常用/人名用にない) を含む。zt: 日本の規格にも GB2312 にもない漢字を含む。zw: 漢字のみで中国語の strong 機能語を含む。frag: CJK の途中で切れたバイト断片"}


def corpus_rates(sets_ids):
    from tokenizers import Tokenizer
    import zh_calib as C
    tok = Tokenizer.from_file(TOK)
    ja, _ = C.ja_corpora()
    zh = C.zh_corpora()
    S = {k: set(v) for k, v in sets_ids.items()}
    print("\n[token rate] コーパス別: トークン数、各集合のトークン割合 (100 万トークンあたり)、集合のトークンを 1 つ以上含む文書の割合")
    rows = []
    for group, corp in (("ja", ja), ("zh", zh)):
        for name, docs in corp.items():
            docs = [Z.clean(d) for d in docs]
            enc = tok.encode_batch(docs, add_special_tokens=False)
            ntok = 0
            hit = Counter()
            dochit = Counter()
            top = Counter()
            for e in enc:
                ids = e.ids
                ntok += len(ids)
                seen = set()
                for t in ids:
                    for k, s in S.items():
                        if t in s:
                            hit[k] += 1
                            seen.add(k)
                            if k == "zh_only" and group == "ja":
                                top[t] += 1
                for k in seen:
                    dochit[k] += 1
            nd = len(docs)
            r = {k: hit[k] / max(1, ntok) * 1e6 for k in S}
            d = {k: dochit[k] / max(1, nd) for k in S}
            print(f"  {group}:{name:12s} docs {nd:6d} tok {ntok:9d} | " + "  ".join(f"{k} {r[k]:8.1f}/M ({d[k]:6.2%})" for k in S))
            if top:
                print("      ja 側で出た zh_only の上位:", " ".join(f"{tok.decode([t])!r}x{n}" for t, n in top.most_common(12)))
            rows.append((group, name, nd, ntok, r, d))
    return rows


def main(argv):
    tb, out = build()
    print("[vocab]", json.dumps(out["counts"], ensure_ascii=False))
    print("[zs len]", out["zs_len_hist"])
    for k in ("zs", "zt", "zw", "frag"):
        print(f"[ex {k}]", " | ".join(repr(x) for x in out["examples"][k][:25]))
    if "--no-corpus" not in argv:
        rows = corpus_rates({"zh_only": out["zh_only"], "zw": out["sets"]["zw"], "frag": out["sets"]["frag"]})
        out["corpus_rates"] = [{"group": g, "name": n, "docs": nd, "tokens": nt, "per_M": r, "doc_frac": d} for g, n, nd, nt, r, d in rows]
    del out["examples"]["ja"]
    json.dump(out, open(f"{HERE}/zh_only_tokens.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n[write] {HERE}/zh_only_tokens.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
