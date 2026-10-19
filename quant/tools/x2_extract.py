"""X2 stage 1: GLM reqlog -> V4.1 token streams (CPU only).

For every OpenAI chat request in the GLM SGLang request log:
  * pair `request.received.openai` (messages/tools) with the following `request.received`
    (rid) by FIFO order inside one log file, verified by a text probe;
  * take the GLM output text from `request.finished` (rid), parse it into
    reasoning / content / tool_calls (GLM-4.5 XML tool format);
  * render prompt with the official V4.1 encoding.py (thinking mode, effort "high" when
    enable_thinking/thinking is not False; tools put on the system message), then render
    prompt + assistant message; output tokens = suffix after the prompt.
  * tokenize both with the V4.1 tokenizer.

Output per log file (in <out_dir>): <name>.u32 (prompt tokens then output tokens per request,
concatenated) and <name>.meta.json (list of per-request dicts).

usage: x2_extract.py <reqlog_dir> <out_dir> [n_workers]
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import glob
import json
import multiprocessing as mp
import os
import re
import sys
from collections import deque

import numpy as np
import orjson

ENC_DIR = MODELS + "/DeepSeek-V4.1-Flash-1017/encoding"
TOK_PATH = MODELS + "/DeepSeek-V4.1-Flash-1017/tokenizer.json"

_tok = None
_enc = None


def _init():
    global _tok, _enc
    import importlib.util

    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(TOK_PATH)
    spec = importlib.util.spec_from_file_location("v41_encoding", os.path.join(ENC_DIR, "encoding.py"))
    _enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_enc)


TC_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)(?:</arg_value>|$)", re.S)


def parse_glm_output(text, thinking):
    reasoning = ""
    rest = text
    if "</think>" in text:
        reasoning, rest = text.split("</think>", 1)
    elif thinking:
        # truncated inside reasoning
        reasoning, rest = text, ""
    tool_calls = []
    for m in TC_RE.finditer(rest):
        body = m.group(1)
        k = body.find("<arg_key>")
        name = (body if k < 0 else body[:k]).strip()
        args = {}
        for a in ARG_RE.finditer(body):
            key, val = a.group(1).strip(), a.group(2)
            v = val
            s = val.strip()
            if s[:1] in "[{" or s in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", s or "x"):
                try:
                    v = json.loads(s)
                except ValueError:
                    v = val
            args[key] = v
        tool_calls.append({"type": "function", "function": {"name": name or "unknown",
                                                              "arguments": json.dumps(args, ensure_ascii=False)}})
    content = TC_RE.sub("", rest).strip()
    msg = {"role": "assistant", "content": content}
    if reasoning.strip():
        msg["reasoning_content"] = reasoning.strip()
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def norm_messages(messages, tools):
    out = []
    n_img = 0
    for m in messages:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            blocks = []
            for b in c:
                if not isinstance(b, dict):
                    blocks.append({"type": "text", "text": str(b)})
                    continue
                t = b.get("type")
                if t in ("image_url", "image"):
                    n_img += 1
                    blocks.append({"type": "image_url", "image_url": {"url": "x"}})
                elif t == "text":
                    blocks.append({"type": "text", "text": b.get("text") or ""})
                else:
                    blocks.append({"type": "text", "text": json.dumps(b, ensure_ascii=False)[:20000]})
            if m.get("role") in ("assistant", "system"):
                m["content"] = "\n\n".join(x.get("text", "") for x in blocks if x["type"] == "text")
            elif m.get("role") == "tool":
                m["content"] = "\n\n".join(x.get("text", "<image>") if x["type"] == "text" else "[image]"
                                           for x in blocks)
            else:
                m["content"] = blocks
        elif c is None:
            m["content"] = ""
        if m.get("role") == "developer":
            m["role"] = "system"
        tc = m.get("tool_calls")
        if tc:
            fixed = []
            for x in tc:
                f = dict(x.get("function") or {})
                if not isinstance(f.get("arguments"), str):
                    f["arguments"] = json.dumps(f.get("arguments") or {}, ensure_ascii=False)
                fixed.append({"type": "function", "function": f})
            m["tool_calls"] = fixed
        out.append(m)
    if tools:
        if out and out[0].get("role") == "system":
            out[0] = dict(out[0], tools=tools)
        else:
            out.insert(0, {"role": "system", "content": "", "tools": tools})
    return out, n_img


def render(messages, tools, thinking, out_text):
    msgs, n_img = norm_messages(messages, tools)
    mode = "thinking" if thinking else "chat"
    eff = "high" if thinking else None
    prompt = _enc.encode_messages(msgs, thinking_mode=mode, reasoning_effort=eff)
    asst = parse_glm_output(out_text, thinking)
    full = _enc.encode_messages(msgs + [asst], thinking_mode=mode, reasoning_effort=eff)
    if full.startswith(prompt):
        out = full[len(prompt):]
        pref_ok = True
    else:
        # template dropped something: fall back to the assistant block alone
        i = 0
        lim = min(len(full), len(prompt))
        while i < lim and full[i] == prompt[i]:
            i += 1
        out = full[i:]
        pref_ok = False
    return prompt, out, n_img, pref_ok


def first_text(msgs):
    if not msgs:
        return ""
    last = msgs[-1].get("content")
    if isinstance(last, list):
        last = " ".join(b.get("text", "") for b in last if isinstance(b, dict))
    return (last or "")[:40]


def process(args):
    path, out_dir = args
    name = os.path.basename(path)
    pend_oa = deque()
    pending = {}
    metas = []
    chunks = []
    stats = {"oa": 0, "pair_ok": 0, "pair_ng": 0, "render_err": 0, "done": 0, "unfinished": 0, "prefix_ng": 0}
    with open(path, "rb") as f:
        for line in f:
            head = line[:400]
            i = line.find(b"{")
            if i < 0:
                continue
            if b'"event": "request.received.openai"' in head:
                try:
                    d = orjson.loads(line[i:])
                except orjson.JSONDecodeError:
                    continue
                o = d.get("obj") or {}
                ctk = o.get("chat_template_kwargs") or {}
                th = ctk.get("enable_thinking", ctk.get("thinking", True))
                pend_oa.append({"ts": d.get("timestamp"), "messages": o.get("messages") or [],
                                "tools": o.get("tools"), "thinking": th is not False})
                stats["oa"] += 1
            elif b'"event": "request.received"' in head:
                if not pend_oa:
                    continue
                d = orjson.loads(line[i:])
                o = d.get("obj") or {}
                rec = pend_oa.popleft()
                txt = o.get("text") or ""
                if first_text(rec["messages"]) in txt:
                    stats["pair_ok"] += 1
                    pending[d.get("rid")] = rec
                else:
                    stats["pair_ng"] += 1
            elif b'"event": "request.finished"' in head:
                m = re.search(rb'"rid": "([0-9a-f]+)"', head)
                if not m:
                    continue
                rid = m.group(1).decode()
                rec = pending.pop(rid, None)
                if rec is None:
                    continue
                d = orjson.loads(line[i:])
                out = d.get("out") or {}
                mi = out.get("meta_info") or {}
                try:
                    prompt, otext, n_img, pref_ok = render(rec["messages"], rec["tools"], rec["thinking"],
                                                           out.get("text") or "")
                except Exception as e:  # noqa: BLE001
                    stats["render_err"] += 1
                    stats.setdefault("err_" + type(e).__name__, 0)
                    stats["err_" + type(e).__name__] += 1
                    continue
                if not pref_ok:
                    stats["prefix_ng"] += 1
                p_ids = _tok.encode(prompt, add_special_tokens=False).ids
                o_ids = _tok.encode(otext, add_special_tokens=False).ids
                chunks.append(np.asarray(p_ids, dtype=np.uint32))
                chunks.append(np.asarray(o_ids, dtype=np.uint32))
                metas.append({"ts": rec["ts"], "rid": rid, "file": name, "plen": len(p_ids), "olen": len(o_ids),
                              "glm_prompt": mi.get("prompt_tokens"), "glm_cached": mi.get("cached_tokens"),
                              "glm_completion": mi.get("completion_tokens"), "tools": bool(rec["tools"]),
                              "n_img": n_img, "thinking": rec["thinking"], "nmsg": len(rec["messages"]),
                              "finish": (mi.get("finish_reason") or {}).get("type")})
                stats["done"] += 1
    stats["unfinished"] = len(pending)
    arr = np.concatenate(chunks) if chunks else np.zeros(0, np.uint32)
    arr.tofile(os.path.join(out_dir, name + ".u32"))
    with open(os.path.join(out_dir, name + ".meta.json"), "w") as f:
        json.dump(metas, f)
    return name, stats


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.log*")), key=os.path.getsize, reverse=True)
    tot = {}
    with mp.Pool(nw, initializer=_init) as pool:
        for name, st in pool.imap_unordered(process, [(f, out_dir) for f in files]):
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
            if st["oa"]:
                print(name, st, flush=True)
    print("TOTAL", tot)
    with open(os.path.join(out_dir, "_extract_stats.json"), "w") as f:
        json.dump(tot, f, indent=1)


if __name__ == "__main__":
    main()
