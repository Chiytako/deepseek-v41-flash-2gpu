#!/usr/bin/env python3
"""ds41rt (OpenAI 互換 /v1/chat/completions) に画像を base64 で送り、画像の生成評価セットの正答率を出す。標準ライブラリのみ。

セット: work/vl/vl_gen.jsonl (tools/vl/build_sets.py が作る)。各行の列:
  id, lang (ja|en), cat (vqa|ocr|chart|doc|screenshot|knowledge|mcq), sub (出典・テンプレート), images (work/vl からの相対パス),
  prompt, answer, atype (str|num|mcq|ja_short|ned|relaxed|anls|vqa), aliases, answers, options, max_tokens

採点 (1 問あたり 0〜1、正答率はその平均):
  str      NFKC・小文字・空白と句読点を除いて、正解 (または別表記) と一致、または答えが短い予測に含まれれば 1
  num      予測中の数値のどれかが正解と一致 (カンマ・全角を正規化)
  mcq      予測から選択肢の記号を取り出して一致 (記号が無ければ選択肢本文の一致で判定)
  ja_short JA-VG-VQA の短答。正規化後に一方が他方を含む、または数 (漢数字含む) が一致すれば 1
  ned      OCR 全文の書き起こし。1 - 正規化編集距離 (空白を除く)
  relaxed  ChartQA。数値は相対誤差 5% 以内、文字列は str と同じ
  anls     DocVQA の ANLS (閾値 0.5)
  vqa      TextVQA の VQA 正答率 min(1, 一致した注釈者数 / 3)

日本語の問いの回答に、簡体字専用字 (zh_only、tools/zh_regex.py と同じ定義) が出た数も数える (中国語混入の目安)。

例:
  tools/.venv/bin/python tools/vl_eval.py --out work/vl/results/v1.jsonl
  tools/.venv/bin/python tools/vl_eval.py --out work/vl/results/qad.jsonl --compare work/vl/results/v1.jsonl
  tools/.venv/bin/python tools/vl_eval.py --limit 1 --out /tmp/x.jsonl      # 形式の確認
  tools/.venv/bin/python tools/vl_eval.py --report work/vl/results/v1.jsonl # 採点だけやり直す (要求しない)
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse
import base64
import concurrent.futures as cf
import functools
import json
import os
import random
import re
import string
import sys
import time
import unicodedata
import urllib.error
import urllib.request

VL = DSV41 + "/work/vl"
# 簡体字専用字 (zh_only): GB2312 にあって JIS (euc_jp / euc_jis_2004 / shift_jis_2004) に無い CJK 統合漢字。
# tools/zh_regex.py と同じ定義 (テキストの中国語混入の評価と揃える)。以前の手書きの約 70 字の表は、そのうちの一部だけだった
@functools.lru_cache(maxsize=None)
def is_zh_only(ch):
    o = ord(ch)
    if not (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF):
        return False
    for enc in ("euc_jp", "euc_jis_2004", "shift_jis_2004"):
        try:
            ch.encode(enc)
            return False
        except UnicodeEncodeError:
            pass
    try:
        ch.encode("gb2312")
        return True
    except UnicodeEncodeError:
        return False


PUNCT = set(string.punctuation) | set("、。，．・「」『』（）【】［］〔〕“”‘’：；！？…〜～　｡､")


# ------------------------------------------------------------------------------------------------ 正規化と採点
def nfkc(s):
    return unicodedata.normalize("NFKC", s or "")


def norm(s):
    s = nfkc(s).lower().strip()
    return "".join(c for c in s if not c.isspace() and c not in PUNCT)


def strip_answer(pred):
    """思考の残りや「答え:」などの前置きを取る。"""
    p = nfkc(pred or "")
    if "</think>" in p:
        p = p.split("</think>")[-1]
    p = p.strip()
    p = re.sub(r"^(答え|回答|正解|answer)\s*(は|is)?\s*[:：]?\s*", "", p, flags=re.I)
    return p.strip()


def first_line(p):
    for line in p.splitlines():
        if line.strip():
            return line.strip()
    return p.strip()


def numbers(p):
    p = nfkc(p)
    p = re.sub(r"(?<=\d),(?=\d{3})", "", p)
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", p)]


KANJI_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "ひとつ": 1, "ふたつ": 2, "みっつ": 3,
             "よっつ": 4, "いつつ": 5, "ひとり": 1, "ふたり": 2}


def ja_numbers(p):
    out = numbers(p)
    for k, v in KANJI_NUM.items():
        if k in p:
            out.append(float(v))
    return out


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _ascii_word(g):
    return bool(g) and all(c.isascii() and (c.isalnum() or c in " -_./:") for c in g)


def s_str(pred, golds):
    sp = strip_answer(pred)
    p = norm(sp)
    words = " " + " ".join(re.sub(r"[^0-9a-z]+", " ", nfkc(sp).lower()).split()) + " "
    for g0 in golds:
        g = norm(g0)
        if not g:
            continue
        if p == g:
            return 1.0
        if g in p and len(p) <= 3 * len(g) + 12:
            gw = " ".join(re.sub(r"[^0-9a-z]+", " ", nfkc(g0).lower()).split())
            # 英数字の正解は語の境界で一致させる (「no」が「not sure」に、「2」が「12」に含まれて当たらないように)
            if _ascii_word(nfkc(g0).strip()) and gw and f" {gw} " not in words:
                continue
            return 1.0
    return 0.0


def s_num(pred, gold, tol=0.0):
    t = numbers(gold)
    if not t:
        return s_str(pred, [gold])
    t = t[0]
    return float(any(abs(x - t) <= max(tol * abs(t), 1e-6) for x in numbers(strip_answer(pred))))


def s_mcq(pred, gold, options):
    p = nfkc(strip_answer(pred))
    labels = "ABCDEFGHIJ"[:max(len(options or []), 4)]
    # 先頭の「A 」は英語の冠詞と区別できないので、A の後ろが空白のときは続きが選択肢 A の本文で始まる場合だけ採る
    m = re.match(rf"^\(?([{labels}])(?:[\).:：、．]|$|(\s+)(.*))", p, flags=re.S)
    if m and not (m.group(1) == "A" and m.group(2) and options
                  and not (norm(options[0]) and norm(m.group(3)).startswith(norm(options[0])[:8]))):
        return float(m.group(1) == gold)
    m = re.search(rf"(?i:答え|正解|answer)\s*(?:は|is)?\s*[:：]?\s*\(?([{labels}])(?![A-Za-z])", p)
    if m:
        return float(m.group(1) == gold)
    for m in re.finditer(rf"(?<![A-Za-z])([{labels}])(?![A-Za-z])", p):
        if m.group(1) == "A" and re.match(r"\s+[a-z]", p[m.end():]):
            continue  # 英語の冠詞 (a dog の A)
        return float(m.group(1) == gold)
    hits = [i for i, o in enumerate(options or []) if norm(o) and norm(o) in norm(p)]
    if len(hits) == 1:
        return float(labels[hits[0]] == gold)
    return 0.0


def ja_core(s):
    """JA-VG の短答の語尾 (です/でした/色) を落とす。「白です。」「白色」「白」を同じに扱う。"""
    s = norm(s)
    s = re.sub(r"(です|でした|である|だ)$", "", s)
    return re.sub(r"色$", "", s)


def s_ja_short(pred, gold):
    p, g = ja_core(strip_answer(pred)), ja_core(gold)
    if not p or not g:
        return 0.0
    if g in p and len(p) <= 3 * len(g) + 12:
        return 1.0
    if p in g:
        return 1.0
    gn, pn = ja_numbers(gold), ja_numbers(strip_answer(pred))
    if gn and pn and gn[0] in pn:
        return 1.0
    return 0.0


def s_ned(pred, gold):
    p, g = norm(strip_answer(pred)), norm(gold)
    if not g:
        return float(not p)
    return max(0.0, 1.0 - lev(p, g) / max(len(p), len(g)))


def s_relaxed(pred, gold):
    g = gold.strip().rstrip("%").replace("$", "")
    try:
        t = float(g.replace(",", ""))
    except ValueError:
        return s_str(first_line(strip_answer(pred)), [gold])
    xs = numbers(first_line(strip_answer(pred)).replace("$", ""))
    return float(any(abs(x - t) <= 0.05 * abs(t) + 1e-9 for x in xs))


def s_anls(pred, answers):
    p = " ".join(first_line(strip_answer(pred)).lower().split()).rstrip(".")
    best = 0.0
    for a in answers:
        a = " ".join(a.lower().split())
        d = lev(a, p) / max(len(a), len(p), 1)
        best = max(best, 1 - d if d < 0.5 else 0.0)
    return best


ART = {"a", "an", "the"}


def vqa_norm(s):
    s = nfkc(s).lower().replace("\n", " ").strip().rstrip(".")
    s = "".join(" " if c in string.punctuation and c not in "'" else c for c in s)
    return " ".join(w for w in s.split() if w not in ART)


def s_vqa(pred, answers):
    p = vqa_norm(first_line(strip_answer(pred)))
    m = sum(vqa_norm(a) == p for a in answers)
    return min(1.0, m / 3.0)


def score(item, pred):
    t = item["atype"]
    golds = [item["answer"]] + list(item.get("aliases") or [])
    if t == "str":
        return s_str(pred, golds)
    if t == "num":
        return s_num(pred, item["answer"], item.get("tol", 0.0))
    if t == "mcq":
        return s_mcq(pred, item["answer"], item.get("options"))
    if t == "ja_short":
        return max(s_ja_short(pred, g) for g in golds)
    if t == "ned":
        return s_ned(pred, item["answer"])
    if t == "relaxed":
        return s_relaxed(pred, item["answer"])
    if t == "anls":
        return s_anls(pred, item.get("answers") or golds)
    if t == "vqa":
        return s_vqa(pred, item.get("answers") or golds)
    raise ValueError(t)


# ------------------------------------------------------------------------------------------------ 要求
def data_url(path):
    b = open(path, "rb").read()
    mime = "image/png" if b[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg" if b[:2] == b"\xff\xd8" else "image/webp" if b[8:12] == b"WEBP" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(b).decode()


def ask(args, item):
    content = [{"type": "image_url", "image_url": {"url": data_url(os.path.join(args.root, p))}} for p in item["images"]]
    content.append({"type": "text", "text": item["prompt"]})
    body = {"model": args.model, "messages": [{"role": "user", "content": content}], "temperature": args.temperature,
            "max_tokens": item.get("max_tokens") or args.max_tokens, "thinking": {"type": args.thinking}}
    if args.temperature > 0:
        body["top_p"] = args.top_p
    req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    err = None
    for attempt in range(args.retries + 1):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                res = json.load(r)
            msg = res["choices"][0]["message"]
            return {"pred": msg.get("content") or "", "reasoning": msg.get("reasoning_content"), "finish": res["choices"][0].get("finish_reason"),
                    "usage": res.get("usage"), "latency": round(time.time() - t0, 2)}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code < 500:
                break
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        time.sleep(2 * (attempt + 1))
    return {"pred": "", "error": err}


# ------------------------------------------------------------------------------------------------ 集計
def boot_ci(xs, n=2000, seed=0):
    if not xs:
        return (float("nan"), float("nan"))
    r = random.Random(seed)
    ms = sorted(sum(r.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def zh_count(text, gold=""):
    """回答中の zh_only の字の数。正解 (画像そのものに中国語が写っている OCR など) に含まれる字は数えない。"""
    g = set(gold or "")
    return sum(is_zh_only(c) and c not in g for c in text or "")


def report(rows, base=None):
    groups = {"ALL": rows}
    for r in rows:
        groups.setdefault(f"lang={r['lang']}", []).append(r)
        groups.setdefault(f"{r['lang']}/{r['cat']}", []).append(r)
        groups.setdefault(f"  {r['lang']}/{r['cat']}/{r['sub']}", []).append(r)
    bmap = {b["id"]: b for b in base} if base else {}
    order = ["ALL", "lang=ja", "lang=en"] + sorted(k for k in groups if "/" in k and not k.startswith("  ")) + \
        sorted(k for k in groups if k.startswith("  "))
    hdr = f"{'group':<44}{'n':>5}{'acc':>8}{'95% CI':>17}"
    if base:
        hdr += f"{'base':>8}{'diff':>8}{'diff 95% CI':>19}"
    print(hdr)
    for k in order:
        g = groups[k]
        xs = [r["score"] for r in g]
        lo, hi = boot_ci(xs)
        line = f"{k:<44}{len(g):>5}{100 * sum(xs) / len(xs):>7.1f}%  [{100 * lo:5.1f},{100 * hi:5.1f}]"
        if base:
            pairs = [(r["score"], bmap[r["id"]]["score"]) for r in g if r["id"] in bmap]
            if pairs:
                d = [a - b for a, b in pairs]
                dlo, dhi = boot_ci(d)
                line += f"{100 * sum(b for _, b in pairs) / len(pairs):>7.1f}%{100 * sum(d) / len(d):>+7.1f}  [{100 * dlo:+5.1f},{100 * dhi:+5.1f}]"
        print(line)
    ja = [r for r in rows if r["lang"] == "ja"]
    zh = [r for r in ja if zh_count(r.get("pred"), r.get("answer")) > 0]
    err = [r for r in rows if r.get("error")]
    print(f"\n日本語の問い {len(ja)} 件のうち簡体字 (zh_only) を含む回答: {len(zh)} 件" + (f" 例: {zh[0]['id']} {zh[0]['pred'][:60]!r}" if zh else ""))
    print(f"要求エラー: {len(err)} 件" + (f" 例: {err[0]['id']} {err[0]['error'][:120]}" if err else ""))
    lat = [r["latency"] for r in rows if r.get("latency")]
    if lat:
        print(f"レイテンシ 中央値 {sorted(lat)[len(lat) // 2]:.2f}s、最大 {max(lat):.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=os.path.join(VL, "vl_gen.jsonl"))
    ap.add_argument("--root", default=VL, help="images の相対パスの基準")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--out", default="", help="1 問 1 行の結果 jsonl (途中から再開できる)")
    ap.add_argument("--compare", default="", help="比べる結果 jsonl (同じ id で対にして差と 95%% 区間を出す)")
    ap.add_argument("--report", default="", help="結果 jsonl を読み直して採点・集計だけ行う (要求しない)")
    ap.add_argument("--filter", default="", help="id / sub / cat / lang の部分一致 (カンマ区切りで OR)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--thinking", default="disabled", choices=["disabled", "enabled"])
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.set, encoding="utf-8")]
    if a.filter:
        keys = a.filter.split(",")
        items = [it for it in items if any(k in it["id"] or k == it["sub"] or k == it["cat"] or k == it["lang"] for k in keys)]
    if a.limit:
        items = items[:a.limit]
    base = [json.loads(l) for l in open(a.compare, encoding="utf-8")] if a.compare else None

    if a.report:
        rows = [json.loads(l) for l in open(a.report, encoding="utf-8")]
        imap = {it["id"]: it for it in items}
        for r in rows:  # 採点規則を変えたときのために採点し直す
            if r["id"] in imap:
                r["score"] = score(imap[r["id"]], r.get("pred", ""))
        report(rows, base)
        return

    done = {}
    if a.out and os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("error"):
                done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} 問 (済み {len(items) - len(todo)})、{a.base_url} model={a.model} thinking={a.thinking} T={a.temperature}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fo = open(a.out, "a", encoding="utf-8") if a.out else None
    rows = list(done.values())
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(ask, a, it): it for it in todo}
        for k, f in enumerate(cf.as_completed(futs), 1):
            it = futs[f]
            res = f.result()
            row = {"id": it["id"], "lang": it["lang"], "cat": it["cat"], "sub": it["sub"], "answer": it["answer"], **res}
            row["score"] = score(it, res.get("pred", "")) if not res.get("error") else 0.0
            rows.append(row)
            if fo:
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                fo.flush()
            if k % 20 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    if fo:
        fo.close()
    order = {it["id"]: i for i, it in enumerate(items)}
    rows = sorted([r for r in rows if r["id"] in order], key=lambda r: order[r["id"]])
    if a.limit and a.limit <= 3:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False)[:800])
    report(rows, base)


if __name__ == "__main__":
    main()
