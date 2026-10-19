#!/usr/bin/env python3
# GLM 時代の実要求から再生用の要求集 (JSONL) を作る (r7 の再生試験、tests/replay_real.py の入力)。
#   chat      : reqlog の `request.received.openai` の obj (クライアントが送った Chat 本文。SGLang が足した既定値は KEEP で落とす)。
#   responses : reqlog に生 JSON は無い (r7_clients.md §1(c))。`request.received` (rid が resp_) の描画済みプロンプト
#               (GLM-5.3 のチャットテンプレート) を逆変換して Responses の要求を復元する (§4.2 の推定復元と同じ規則)。
#               reasoning 項目は 3 形式 (SGLang の encrypted_content / summary / content) を順に使い分ける。
# 使い方: python3 tests/build_replay_corpus.py --out tests/replay_corpus.jsonl
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse, base64, collections, importlib.util, json, os, random, re, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--reqlog", default=GLM_REQLOG)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_corpus.jsonl"))
ap.add_argument("--seed", type=int, default=19)
ap.add_argument("--max-bytes", type=int, default=900_000)
A = ap.parse_args()
random.seed(A.seed)

_spec = importlib.util.spec_from_file_location("replay_compat_label", os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_compat.py"))
_src = open(_spec.origin).read()
_ns = {}
exec(_src[_src.index("def label"):_src.index("def has_tool_reference")], _ns)
label = _ns["label"]

# replay_openai.py の KEEP + stream / stream_options / response_format (クライアントが実際に送るもの)
KEEP = {"messages", "model", "tools", "tool_choice", "parallel_tool_calls", "max_tokens", "max_completion_tokens", "temperature",
        "top_p", "top_k", "presence_penalty", "frequency_penalty", "reasoning_effort", "chat_template_kwargs", "stop", "seed",
        "stream", "stream_options", "response_format"}

# (ファイル, クライアントの型 (replay_compat.label、r7_clients.md §3.1), 件数, 大きさの条件)。件数は合計 65 前後になるように選ぶ
# publish: 元の要求ログのファイル名は伏せて reqlog-X.log に置き換えた。自分の要求ログのファイル名に書き換えて使う
CHAT_PLAN = [
    ("reqlog-A.log", "B", 5, None),
    ("reqlog-B.log", "B", 3, (200_000, None)),
    ("reqlog-C.log", "B", 3, (200_000, None)),
    ("reqlog-D.log", "B", 2, None),
    ("reqlog-E.log", "B", 2, None),
    ("TOOLREF", "B", 4, None),
    ("reqlog-A.log", "B-aux", 3, None),
    ("reqlog-C.log", "B-aux", 2, None),
    ("reqlog-F.log", "C-new", 4, (None, 200_000)),
    ("reqlog-F.log", "C-new", 3, (200_000, 500_000)),
    ("reqlog-F.log", "C-new", 3, (500_000, None)),
    ("reqlog-F.log", "C-title", 2, None),
    ("reqlog-G.log", "C-old", 5, None),
    ("reqlog-A.log", "D", 5, None),
    ("reqlog-H.log", "A", 12, None),
    ("reqlog-I.log", "E", 3, None),
    ("reqlog-B.log", "E", 2, None),
    ("reqlog-J.log", "other", 3, None),
]
TOOLREF_FILES = ["reqlog-K.log", "reqlog-L.log", "reqlog-M.log"]

RESP_FILES = ["reqlog-N.log", "reqlog-O.log"]
N_RESP = 30


def read_openai(fname):
    out = []
    with open(os.path.join(A.reqlog, fname), errors="replace") as fh:
        for line in fh:
            if '"request.received.openai"' not in line:
                continue
            try:
                d = json.loads(line[line.find("{"):])
            except Exception:
                continue
            o = d.get("obj") or {}
            if not o.get("messages"):
                continue
            body = {k: v for k, v in o.items() if k in KEEP and v is not None}
            out.append((d.get("rid"), body, len(line)))
    return out


def has_tool_reference(o):
    return any(isinstance(p, dict) and p.get("type") == "tool_reference"
               for m in o.get("messages") or [] if isinstance(m.get("content"), list) for p in m["content"])


# ---------------- Responses の復元 ----------------
SEG = re.compile(r"(<\|system\|>|<\|user\|>|<\|assistant\|>|<\|observation\|>)")
TOOLCALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
ARG = re.compile(r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", re.S)
RESP = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)


def arg_value(v, schema):
    t = (schema or {}).get("type")
    if t == "string" or (isinstance(t, list) and "string" in t and len(t) == 1):
        return v
    try:
        return json.loads(v)
    except Exception:
        return v


def reasoning_item(text, k):
    mode = k % 3
    if mode == 0:
        blob = "sglang-reasoning-v1." + base64.urlsafe_b64encode(zlib.compress(text.encode())).decode()
        return {"type": "reasoning", "summary": [], "encrypted_content": blob}
    if mode == 1:
        return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}
    return {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": text}]}


def rebuild_responses(rid, text, sp, k):
    assert text.startswith("[gMASK]<sop>"), text[:40]
    parts = SEG.split(text[len("[gMASK]<sop>"):])
    segs = [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    effort, tools, inputs, calls_pending = None, [], [], []
    schemas = {}
    ncall = 0
    for role, body in segs:
        if role == "<|system|>":
            if body.startswith("Reasoning Effort: "):
                effort = body[len("Reasoning Effort: "):].strip().lower()
            elif body.startswith("\n# Tools"):
                blk = body[body.index("<tools>\n") + 8:body.index("\n</tools>")]
                for ln in blk.split("\n"):
                    t = json.loads(ln)
                    tools.append({"type": "function", "name": t["name"], "description": t.get("description", ""),
                                  "parameters": t.get("parameters") or {"type": "object", "properties": {}}})
                    schemas[t["name"]] = (t.get("parameters") or {}).get("properties") or {}
            else:
                txt = body[len("Developer instructions:\n"):] if body.startswith("Developer instructions:\n") else body
                inputs.append({"type": "message", "role": "developer", "content": [{"type": "input_text", "text": txt}]})
        elif role == "<|user|>":
            inputs.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": body}]})
        elif role == "<|assistant|>":
            if body in ("<think>", ""):
                continue  # 生成の開始位置
            rest = body
            if rest.startswith("<think>"):
                j = rest.index("</think>")
                think = rest[7:j]
                rest = rest[j + 8:]
                if think.strip():
                    inputs.append(reasoning_item(think, k + ncall))
            i = rest.find("<tool_call>")
            msg = (rest if i < 0 else rest[:i]).strip()
            if msg:
                inputs.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": msg}]})
            for m in TOOLCALL.finditer(rest if i >= 0 else ""):
                inner = m.group(1)
                a = inner.find("<arg_key>")
                name = (inner if a < 0 else inner[:a]).strip()
                args = {kk: arg_value(vv, schemas.get(name, {}).get(kk)) for kk, vv in ARG.findall(inner)}
                ncall += 1
                cid = f"call_{rid[5:13]}_{ncall}"
                calls_pending.append(cid)
                inputs.append({"type": "function_call", "call_id": cid, "name": name,
                               "arguments": json.dumps(args, ensure_ascii=False)})
        elif role == "<|observation|>":
            for m in RESP.finditer(body):
                cid = calls_pending.pop(0) if calls_pending else f"call_{rid[5:13]}_orphan{len(inputs)}"
                inputs.append({"type": "function_call_output", "call_id": cid, "output": m.group(1)})
    # 実データは 100% ストリームだが、非ストリームの集約経路も見るため 6 件に 1 件を非ストリームにする
    req = {"model": "GLM-5.3-Flash", "stream": k % 6 != 5, "input": inputs,
           "max_output_tokens": 128000 if sp.get("max_new_tokens", 0) > 1000 else 64}
    if tools:
        req["tools"] = tools
    if effort:
        req["reasoning"] = {"effort": effort}
    return req


def main():
    corpus = []
    cache = {}
    for fname, lb, n, size in CHAT_PLAN:
        if fname == "TOOLREF":
            pool = []
            for f in TOOLREF_FILES:
                pool += [(rid, b, sz, f) for rid, b, sz in cache.setdefault(f, read_openai(f)) if has_tool_reference(b) and sz < A.max_bytes]
        else:
            pool = [(rid, b, sz, fname) for rid, b, sz in cache.setdefault(fname, read_openai(fname))
                    if label(b) == lb and not has_tool_reference(b) and sz < A.max_bytes]
        if size:
            lo, hi = size
            pool = [p for p in pool if (lo is None or p[2] >= lo) and (hi is None or p[2] < hi)]
        pick = random.sample(pool, min(n, len(pool)))
        for rid, b, sz, f in pick:
            corpus.append({"id": f"chat-{len(corpus):03d}", "api": "chat", "client": lb + ("+tool_reference" if fname == "TOOLREF" else ""),
                           "src": f"{f}:{rid}", "bytes": sz, "stream": bool(b.get("stream")), "body": b})
        print(f"{fname} {lb}: {len(pick)}/{len(pool)}")
    # Responses
    cand = []
    for f in RESP_FILES:
        with open(os.path.join(A.reqlog, f), errors="replace") as fh:
            for line in fh:
                if '"request.received", "rid": "resp_' not in line:
                    continue
                d = json.loads(line[line.find("{"):])
                o = d["obj"]
                cand.append((f, d["rid"], o["text"], o.get("sampling_params") or {}))
    # 長さで層別して選ぶ (先頭ターン・タイトル・途中・長い後半)
    cand.sort(key=lambda c: len(c[2]))
    titles = [c for c in cand if c[3].get("max_new_tokens", 0) < 1000]
    mains = [c for c in cand if c[3].get("max_new_tokens", 0) >= 1000 and len(c[2]) < A.max_bytes]
    pick = titles[:1]
    step = len(mains) / (N_RESP - len(pick))
    pick += [mains[int(i * step)] for i in range(N_RESP - len(pick))]
    pick[-1] = mains[-1]
    for k, (f, rid, text, sp) in enumerate(pick):
        req = rebuild_responses(rid, text, sp, k)
        kinds = collections.Counter(it["type"] for it in req["input"])
        corpus.append({"id": f"resp-{k:03d}", "api": "responses", "client": "F" + ("-title" if "tools" not in req else ""),
                       "src": f"{f}:{rid}", "bytes": len(json.dumps(req)), "stream": req["stream"], "body": req, "items": dict(kinds)})
    with open(A.out, "w") as fh:
        for c in corpus:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(corpus)} ({sum(c['api'] == 'chat' for c in corpus)} chat, {sum(c['api'] == 'responses' for c in corpus)} responses) -> {A.out}")


if __name__ == "__main__":
    main()
