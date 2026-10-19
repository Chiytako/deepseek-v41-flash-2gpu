#!/usr/bin/env python3
# GLM 時代の要求ログ (SGLang reqlog) の chat 要求を、クライアント別に無作為抽出して ds41gw に max_tokens=1・非ストリーム・
# 少数並列で送り、受理/拒否を数える (r7_proxy_spec.md §16.4 の 1)。model は記録どおり (GLM-5.3-Flash など)。
# クライアントの型の判別規則は環境変数 REPLAY_LABELS (JSON) で与える (label() の説明を参照)。
# 使い方: python3 tests/replay_compat.py [--url http://127.0.0.1:30001] [--per 6] [--files 16] [--conc 2] [--max-bytes 80000]
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse, collections, glob, json, os, random, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30001")
ap.add_argument("--reqlog", default=GLM_REQLOG)
ap.add_argument("--per", type=int, default=6)
ap.add_argument("--files", type=int, default=16)
ap.add_argument("--conc", type=int, default=2)
ap.add_argument("--max-bytes", type=int, default=80000)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default="")
ap.add_argument("--file-list", default="", help="カンマ区切りのファイル名 (reqlog 内)。指定すると --files は使わない")
A = ap.parse_args()
assert A.conc <= 4
random.seed(A.seed)


def label(o, _rules=[]):
    """要求をクライアントの型に分ける (r7_clients.md §3.1 の A〜E)。判別の規則は環境変数 REPLAY_LABELS の JSON ファイルで与える。
    公開版には、利用者のクライアントの system プロンプトの文面を入れていない。規則の例:
      [{"label": "A", "system_startswith": ["<自動コーディングの system プロンプトの書き出し>"]},
       {"label": "A", "no_tools": true, "temperature_in": [0.7, 0.3], "max_tokens": 12000},
       {"label": "B", "system_contains": ["<対話型エージェントの system プロンプトの一部>"]},
       {"label": "E", "clear_thinking": true, "max_tokens": 120}]
    上から順に見て、書かれた条件をすべて満たした最初の規則の label を返す。どれにも当たらなければ、
    ツールなし・2 メッセージ以下・system なしは "test"、それ以外は "other"。"""
    if not _rules:
        import json as _json, os as _os2
        _p = _os2.environ.get("REPLAY_LABELS")
        _rules.append(_json.load(open(_p)) if _p else [])
    msgs = o.get("messages") or []
    tools = o.get("tools") or []
    s = ""
    for m in msgs:
        if m.get("role") in ("system", "developer"):
            c = m.get("content")
            s = c if isinstance(c, str) else " ".join(p.get("text") or "" for p in c or [] if isinstance(p, dict))
            break
    ctk = o.get("chat_template_kwargs") or {}
    for r in _rules[0]:
        if "system_startswith" in r and not s.startswith(tuple(r["system_startswith"])):
            continue
        if "system_contains" in r and not any(w in s for w in r["system_contains"]):
            continue
        if r.get("no_tools") and tools:
            continue
        if "temperature_in" in r and o.get("temperature") not in r["temperature_in"]:
            continue
        if "max_tokens" in r and o.get("max_tokens") != r["max_tokens"]:
            continue
        if r.get("clear_thinking") and not ctk.get("clear_thinking"):
            continue
        return r["label"]
    if not tools and len(msgs) <= 2 and not s:
        return "test"
    return "other"


def has_tool_reference(o):
    for m in o.get("messages") or []:
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") == "tool_reference" for p in c):
            return True
    return False


files = sorted(glob.glob(os.path.join(A.reqlog, "*.log*")))
files = [os.path.join(A.reqlog, f) for f in A.file_list.split(",")] if A.file_list else random.sample(files, min(A.files, len(files)))
pool = collections.defaultdict(list)
for f in files:
    with open(f, errors="replace") as fh:
        for line in fh:
            if '"request.received.openai"' not in line:
                continue
            tr = '"tool_reference"' in line
            if len(line) > A.max_bytes and not tr:
                continue
            if tr and len(line) > 400_000:
                continue
            try:
                o = json.loads(line[line.find("{"):])["obj"]
            except Exception:
                continue
            if not o.get("messages"):
                continue
            lab = label(o) + ("+tool_reference" if has_tool_reference(o) else "")
            pool[lab].append(o)
print({k: len(v) for k, v in pool.items()}, flush=True)

jobs = []
for lab, lst in sorted(pool.items()):
    for o in random.sample(lst, min(A.per, len(lst))):
        body = dict(o)
        body.update(stream=False, max_tokens=1)
        body.pop("max_completion_tokens", None)
        jobs.append((lab, body))


def send(job):
    lab, body = job
    t0 = time.time()
    data = json.dumps(body).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request(A.url + "/v1/chat/completions", data, {"Content-Type": "application/json"}), timeout=600)
        v = json.loads(r.read())
        code, msg = r.status, (v["choices"][0].get("finish_reason") or "")
        warn = r.headers.get("x-ds41gw-warning") or ""
    except urllib.error.HTTPError as e:
        code, msg, warn = e.code, e.read().decode(errors="replace")[:200], ""
    except Exception as e:
        code, msg, warn = "exc", str(e)[:200], ""
    res = {"label": lab, "code": code, "sec": round(time.time() - t0, 1), "bytes": len(data), "msg": msg, "warn": warn}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


with ThreadPoolExecutor(A.conc) as ex:
    results = list(ex.map(send, jobs))
summary = collections.defaultdict(collections.Counter)
for r in results:
    summary[r["label"]][r["code"]] += 1
print(json.dumps({k: dict(v) for k, v in summary.items()}, ensure_ascii=False))
ok = sum(1 for r in results if r["code"] == 200)
print(f"{ok}/{len(results)} returned 200")
if A.out:
    json.dump({"summary": {k: dict(v) for k, v in summary.items()}, "results": results}, open(A.out, "w"), ensure_ascii=False, indent=1)
sys.exit(0 if ok == len(results) else 1)
