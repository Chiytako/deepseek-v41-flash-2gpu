"""生成での中国語混入率を測る: プロンプト集 (zh_promptset.py) を OpenAI 互換サーバーで生成し、zh_detect で採点する。

  gen   : prompts.jsonl を /v1/chat/completions に投げ、出力 (reasoning_content / content / tool_calls) を jsonl に保存
  score : 出力 jsonl (このスクリプトか harness/steer_zhgen.py のもの) を採点し、区分別・版別の表を出す

ds41rt (既定 :8000) と vLLM (:8001 など) の両方に使える。ds41rt は temperature を省くと greedy になるので
必ず明示する (r2_ds41rt §7)。reasoning_effort は low / high / max のどれかに丸める (medium は high)。
生成は思考ありが既定。思考と回答の両方を採点する (思考は英語が多いので、ZS 字と中国語の句だけを見る)。

usage:
  zh_gen_eval.py gen --url http://127.0.0.1:8000 --label v0 --temp 1.0 --top-p 1.0 --seeds 1 \
      [--prompts work/zh/promptset/prompts.jsonl] [--only-harness] [--max-tokens 4096] [--concurrency 8] [--dry-run]
  zh_gen_eval.py score work/zh/gen/v0_t1.0.jsonl [work/zh/gen/v1_t1.0.jsonl ...] [--out table.json]
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zh_detect as Z  # noqa: E402

ROOT = DSV41
EFFORT = {"low": "low", "minimal": "low", "none": "none", "medium": "high", "high": "high", "max": "max", "xhigh": "max"}


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join((b.get("text") or "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def context_text(p):
    """採点の copied 判定用に、プロンプト側の全テキスト (system / user / tool / 以前の assistant)。"""
    parts = []
    for m in p["messages"]:
        parts.append(_text(m.get("content")))
        for tc in m.get("tool_calls") or []:
            parts.append(json.dumps(tc.get("function", {}), ensure_ascii=False))
    return "\n".join(parts)


def payload(p, a, seed):
    body = {"model": a.model, "messages": p["messages"], "max_tokens": a.max_tokens, "temperature": a.temp,
            "top_p": a.top_p, "seed": seed, "stream": False}
    if p.get("tools"):
        body["tools"] = p["tools"]
    if a.thinking == "off" or not p.get("thinking", True):
        body["thinking"] = {"type": "disabled"}
        body["chat_template_kwargs"] = {"thinking": False}
    else:
        body["thinking"] = {"type": "enabled"}
        body["chat_template_kwargs"] = {"thinking": True}
        eff = EFFORT.get(str(a.effort or p.get("reasoning_effort") or "high"), "high")
        body["reasoning_effort"] = eff
    for kv in a.extra:
        k, v = kv.split("=", 1)
        body[k] = json.loads(v)
    return body


def post(url, body, timeout):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def cmd_gen(a):
    P = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    if a.only_harness:
        P = [p for p in P if p.get("harness")]
    if a.cats:
        P = [p for p in P if p["cat"] in a.cats.split(",")]
    jobs = [(p, s) for s in range(a.seeds) for p in P]
    if a.dry_run:
        print(json.dumps(payload(jobs[0][0], a, 0), ensure_ascii=False)[:2000])
        print(f"[dry-run] {len(jobs)} 件")
        return 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["id"], r["seed"]))
    jobs = [(p, s) for p, s in jobs if (p["id"], s) not in done]
    print(f"[gen] {len(jobs)} 件 (済み {len(done)}) -> {a.out}", flush=True)

    def run(job):
        p, s = job
        body = payload(p, a, 1000 + s)
        try:
            d, dt = post(a.url, body, a.timeout)
            ch = d["choices"][0]
            m = ch.get("message") or {}
            return {"id": p["id"], "cat": p["cat"], "seed": s, "label": a.label, "temp": a.temp, "top_p": a.top_p,
                    "reasoning": m.get("reasoning_content") or m.get("reasoning") or "", "content": m.get("content") or "",
                    "tool_calls": m.get("tool_calls") or [], "finish": ch.get("finish_reason"), "usage": d.get("usage"), "sec": dt}
        except Exception as e:  # noqa: BLE001
            return {"id": p["id"], "cat": p["cat"], "seed": s, "label": a.label, "error": f"{type(e).__name__}: {e}"}

    n = 0
    with open(a.out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.concurrency) as ex:
        for r in ex.map(run, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 10 == 0 or "error" in r:
                print(f"  {n}/{len(jobs)} {r['id']} {r.get('error') or r.get('finish')}", flush=True)
    return 0


def score_one(r, ctx):
    ans = Z.detect(r.get("content") or "", context=ctx)
    th = Z.detect(r.get("reasoning") or "", context=ctx) if r.get("reasoning") else None
    return ans, th


def cmd_score(a):
    P = {json.loads(l)["id"]: json.loads(l) for l in open(a.prompts, encoding="utf-8")}
    table = {}
    for path in a.files:
        R = [json.loads(l) for l in open(path, encoding="utf-8")]
        S = defaultdict(Counter)
        bad = []
        zc = Counter()
        for r in R:
            if "error" in r:
                S["ALL"]["error"] += 1
                continue
            p = P.get(r["id"])
            ctx = context_text(p) if p else None
            ans, th = score_one(r, ctx)
            for key in ("ALL", r["cat"]):
                s = S[key]
                s["n"] += 1
                s["finish_length"] += r.get("finish") == "length"
                if ans["chars"] >= 8:
                    s["ans"] += 1
                    s["ans_ja"] += ans["is_ja"]
                    s["ans_chars"] += ans["chars"]
                    if ans["is_ja"]:
                        s["ja_lv1"] += ans["level"] >= 1
                        s["ja_lv2"] += ans["level"] >= 2
                        s["ja_lv3"] += ans["level"] >= 3
                        s["ja_zs"] += bool(ans["zs"])
                        s["ja_zs3"] += len(ans["zs"]) >= 3
                        s["ja_zs_chars"] += len(ans["zs"])
                        s["ja_chars"] += ans["chars"]
                    else:
                        # 日本語の指示に日本語で答えなかった (中国語の回答か英語の回答)
                        s["nonja_zh"] += ans["level"] >= 3
                if th is not None:
                    s["think"] += 1
                    s["think_ja"] += th["is_ja"]
                    s["think_zs"] += bool(th["zs"])
                    s["think_clause"] += bool(th["clauses"])
                    # 思考の主言語: 漢字と仮名が多ければ日本語、ZS / 句が多ければ中国語、それ以外は英語
                    k, h, n = Z.text_stats(Z.clean(r["reasoning"]))
                    zsn = len(th["zs"])
                    s["think_lang_zh"] += (k < 0.02 * max(1, n) and h > 0.3 * max(1, n)) or zsn > 0.05 * max(1, n)
                    s["think_lang_ja"] += k >= 0.05 * max(1, n)
            if ans["is_ja"] and ans["level"] >= 2 and len(bad) < 30:
                bad.append({"id": r["id"], "seed": r["seed"], "level": ans["level"], "zs": ans["zs"][:20],
                            "strong": ans["strong_words"][:5], "clause": [c["text"][:50] for c in ans["clauses"][:1]]})
            zc.update(ans["zs"])
        out = {}
        for key, s in S.items():
            nj = max(1, s["ans_ja"])
            nt = max(1, s["think"])
            out[key] = {"n": s["n"], "err": s["error"], "ans": s["ans"], "ja_answer_rate": s["ans_ja"] / max(1, s["ans"]),
                        "ja_lv>=1": s["ja_lv1"] / nj, "ja_lv>=2": s["ja_lv2"] / nj, "ja_lv3": s["ja_lv3"] / nj,
                        "ja_zs>=1": s["ja_zs"] / nj, "ja_zs>=3": s["ja_zs3"] / nj,
                        "zs_per_10k_chars": s["ja_zs_chars"] / max(1, s["ja_chars"]) * 1e4,
                        "nonja_zh_answer": s["nonja_zh"], "mix_any": (s["ja_lv2"] + s["nonja_zh"]) / max(1, s["ans"]), "finish_length": s["finish_length"] / max(1, s["n"]),
                        "think_n": s["think"], "think_zs>=1": s["think_zs"] / nt, "think_clause": s["think_clause"] / nt,
                        "think_lang_zh": s["think_lang_zh"] / nt, "think_lang_ja": s["think_lang_ja"] / nt}
        table[path] = {"by_cat": out, "examples": bad, "zs_top": zc.most_common(30)}
        print(f"\n=== {path}")
        print("  cat         n   ja率   混入計  lv>=2   lv3    ZS>=1  ZS>=3  ZS/1万字 中国語答 思考ZS  思考句  思考zh  思考ja  上限到達")
        print("  (混入計 = 日本語回答で level>=2 + 中国語で答えた回答、の回答数に対する割合)")
        for key in ["ALL"] + sorted(k for k in out if k != "ALL"):
            o = out[key]
            print(f"  {key:10s}{o['n']:4d} {o['ja_answer_rate']:6.1%} {o['mix_any']:6.2%} {o['ja_lv>=2']:6.2%} {o['ja_lv3']:6.2%} {o['ja_zs>=1']:6.2%} {o['ja_zs>=3']:6.2%} {o['zs_per_10k_chars']:7.3f} {o['nonja_zh_answer']:5d} "
                  f"{o['think_zs>=1']:6.2%} {o['think_clause']:6.2%} {o['think_lang_zh']:6.2%} {o['think_lang_ja']:6.2%} {o['finish_length']:6.2%}")
        print("  ZS 上位:", " ".join(f"{c}{k}" for c, k in zc.most_common(20)))
        for b in bad[:8]:
            print("  例", json.dumps(b, ensure_ascii=False))
    if a.out:
        json.dump(table, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--url", default="http://127.0.0.1:8000")
    g.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    g.add_argument("--label", required=True)
    g.add_argument("--prompts", default=f"{ROOT}/work/zh/promptset/prompts.jsonl")
    g.add_argument("--out", default="")
    g.add_argument("--temp", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--seeds", type=int, default=1)
    g.add_argument("--max-tokens", type=int, default=4096)
    g.add_argument("--thinking", choices=["on", "off"], default="on")
    g.add_argument("--effort", default="")
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--timeout", type=float, default=1800)
    g.add_argument("--only-harness", action="store_true")
    g.add_argument("--cats", default="")
    g.add_argument("--extra", action="append", default=[], help='追加のリクエスト項目 key=JSON (例: response_format={"type":"regex",...})')
    g.add_argument("--dry-run", action="store_true")
    s = sp.add_parser("score")
    s.add_argument("files", nargs="+")
    s.add_argument("--prompts", default=f"{ROOT}/work/zh/promptset/prompts.jsonl")
    s.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "gen":
        if not a.out:
            a.out = f"{ROOT}/work/zh/gen/{a.label}_t{a.temp}_p{a.top_p}.jsonl"
        return cmd_gen(a)
    return cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())
