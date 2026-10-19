#!/usr/bin/env python3
# GLM 時代の実要求 (tests/build_replay_corpus.py で作った要求集) を ds41gw 経由で ds41rt に再生し、
#   1) エラー率、2) 応答の形 (SSE の順序・ツール呼び出しの JSON・名前・id)、3) 追加遅延 (ds41gw 経由と ds41rt 直の TTFT 差)
# を測る。ds41rt は稼働中なので同時 4 本まで、出力は --max-tokens (既定 256) で抑える。
#
#   python3 tests/replay_real.py run   --out tests/results_replay_real.jsonl            # 段 1: 全件を ds41gw 経由で
#   python3 tests/replay_real.py pairs --in tests/results_replay_real.jsonl --out tests/results_replay_pairs.jsonl
#                                                                                      # 段 2: ストリーム要求を ds41rt 直と ds41gw 経由で交互に
#   python3 tests/replay_real.py report --in ... --pairs ...
#
# 段 2 の ds41rt 直の本文は、ds41gw の要求ログ (logs/req/*.jsonl) の `obj` (ds41gw が ds41rt に送った Chat 本文) を
# x-request-id で引いて使う。両方を同じ接頭辞キャッシュの状態で比べるため、段 1 の後に、組ごとに順番を入れ替えて続けて送る。
import argparse, glob, http.client, json, os, random, statistics, sys, threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = "http://127.0.0.1:30001"
DIRECT = "http://127.0.0.1:8000"
DS41RT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


def cap_body(c, cap):
    b = json.loads(json.dumps(c["body"]))
    if c["api"] == "chat":
        for k in ("max_tokens", "max_completion_tokens"):
            if k in b and b[k] is not None:
                b[k] = min(int(b[k]), cap)
        if "max_tokens" not in b and "max_completion_tokens" not in b:
            b["max_tokens"] = cap
    else:
        b["max_output_tokens"] = min(int(b.get("max_output_tokens") or cap), cap)
    return b


# ---------------------------------------------------------------- HTTP
def post(base, path, body, timeout=1800):
    u = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    data = json.dumps(body).encode()
    t0 = time.perf_counter()
    conn.request("POST", path, data, {"Content-Type": "application/json", "User-Agent": "ds41gw-replay-real/1"})
    r = conn.getresponse()
    return conn, r, t0


def read_sse(r, t0):
    """(events, ttft, t_first_byte)。events は (event 名 or None, data 文字列, 到着時刻)。"""
    events, ev_name, buf = [], None, []
    t_first = None
    while True:
        line = r.readline()
        if not line:
            break
        if t_first is None:
            t_first = time.perf_counter() - t0
        s = line.decode("utf-8", "replace").rstrip("\r\n")
        if s == "":
            if buf:
                events.append((ev_name, "\n".join(buf), time.perf_counter() - t0))
            ev_name, buf = None, []
            continue
        if s.startswith(":"):
            continue
        if s.startswith("event:"):
            ev_name = s[6:].strip()
        elif s.startswith("data:"):
            buf.append(s[5:].lstrip(" "))
    if buf:
        events.append((ev_name, "\n".join(buf), time.perf_counter() - t0))
    return events, t_first


# ---------------------------------------------------------------- 検査
class Check:
    def __init__(self):
        self.errors, self.notes = [], []

    def err(self, m):
        if len(self.errors) < 20:
            self.errors.append(m)

    def note(self, m):
        if len(self.notes) < 20:
            self.notes.append(m)


def tool_names(c):
    b = c["body"]
    out = set()
    for t in b.get("tools") or []:
        if c["api"] == "chat":
            f = t.get("function") or t
            if f.get("name"):
                out.add(f["name"])
        elif t.get("name"):
            out.add(t["name"])
    return out


def tool_schemas(c):
    out = {}
    for t in c["body"].get("tools") or []:
        f = (t.get("function") or t) if c["api"] == "chat" else t
        if f.get("name"):
            out[f["name"]] = f.get("parameters") or {}
    return out


def check_call(ck, c, name, args, cid, finish, seen_ids, calls):
    names, schemas = tool_names(c), tool_schemas(c)
    if not cid:
        ck.err("tool_call id 空")
    elif cid in seen_ids:
        ck.err(f"tool_call id 重複 {cid}")
    seen_ids.add(cid)
    if not name:
        ck.err("tool_call name 空")
    elif names and name not in names:
        ck.err(f"未定義のツール名 {name!r}")
    try:
        a = json.loads(args)
        ok = isinstance(a, dict)
        if not ok:
            ck.err(f"arguments が object でない ({type(a).__name__})")
    except Exception as e:
        a, ok = None, False
        if finish in ("length", "incomplete"):
            ck.note(f"上限で切れた arguments ({len(args)} 字)")
        else:
            ck.err(f"arguments が JSON でない: {e}: {args[:120]!r}")
    if ok:
        sch = schemas.get(name) or {}
        req = [k for k in sch.get("required") or [] if k not in a]
        if req:
            ck.note(f"{name}: required 欠落 {req}")
        props = sch.get("properties")
        if isinstance(props, dict) and sch.get("additionalProperties") is False:
            extra = [k for k in a if k not in props]
            if extra:
                ck.note(f"{name}: 余分なキー {extra}")
    calls.append({"name": name, "args_ok": ok, "args_len": len(args)})


def check_chat_stream(ck, c, body, events, t_start):
    want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    acc = {}
    ttft = None
    done = False
    usage = None
    ids = set()
    for name, data, t in events:
        if done:
            ck.err("[DONE] の後にデータ")
        if data == "[DONE]":
            done = True
            continue
        try:
            ev = json.loads(data)
        except Exception:
            ck.err(f"data が JSON でない: {data[:80]!r}")
            continue
        if "error" in ev:
            ck.err(f"ストリーム内エラー: {json.dumps(ev['error'], ensure_ascii=False)[:200]}")
            continue
        if ev.get("object") != "chat.completion.chunk":
            ck.err(f"object={ev.get('object')}")
        if ev.get("model") != body.get("model"):
            ck.err(f"model={ev.get('model')!r} (要求 {body.get('model')!r})")
        ids.add(ev.get("id"))
        if ev.get("usage"):
            usage = ev["usage"]
            if want_usage and ev.get("choices"):
                ck.err("usage チャンクの choices が空でない")
        for ch in ev.get("choices") or []:
            st = acc.setdefault(ch.get("index", 0), {"content": "", "reasoning": "", "tc": {}, "finish": None})
            d = ch.get("delta") or {}
            if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls")):
                ttft = t
            st["content"] += d.get("content") or ""
            st["reasoning"] += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                i = tc.get("index")
                if i is None:
                    ck.err("tool_calls[].index が無い")
                    i = 0
                slot = st["tc"].setdefault(i, {"id": None, "name": None, "args": ""})
                if tc.get("id"):
                    if slot["id"] and slot["id"] != tc["id"]:
                        ck.err(f"同じ index に別の id ({slot['id']} → {tc['id']})")
                    slot["id"] = tc["id"]
                f = tc.get("function") or {}
                if f.get("name"):
                    if slot["name"] and slot["name"] != f["name"]:
                        ck.err("name が途中で変わった")
                    slot["name"] = f["name"]
                slot["args"] += f.get("arguments") or ""
            if ch.get("finish_reason"):
                if st["finish"]:
                    ck.err("finish_reason が 2 回")
                st["finish"] = ch["finish_reason"]
    if not done:
        ck.err("[DONE] が無い")
    if len(ids - {None}) > 1:
        ck.err(f"chunk id が複数 {len(ids)}")
    if want_usage and not usage:
        ck.err("include_usage なのに usage が無い")
    n = body.get("n") or 1
    if len(acc) != n:
        ck.err(f"choices の数 {len(acc)} != n {n}")
    calls = []
    finish = None
    seen = set()
    content_len = reasoning_len = 0
    for i, st in sorted(acc.items()):
        finish = st["finish"]
        if not st["finish"]:
            ck.err(f"choice {i}: finish_reason が無い")
        if st["finish"] == "tool_calls" and not st["tc"]:
            ck.err("finish_reason=tool_calls なのに呼び出しが無い")
        if st["tc"] and st["finish"] not in ("tool_calls", "length"):
            ck.err(f"呼び出しがあるのに finish_reason={st['finish']}")
        if "<｜DSML｜" in st["content"] or "｜DSML｜" in st["content"]:
            ck.err("content に DSML が漏れた")
        for j, slot in sorted(st["tc"].items()):
            check_call(ck, c, slot["name"], slot["args"], slot["id"], st["finish"], seen, calls)
        content_len += len(st["content"])
        reasoning_len += len(st["reasoning"])
    return {"ttft": ttft, "finish": finish, "usage": usage, "tool_calls": calls, "content_chars": content_len,
            "reasoning_chars": reasoning_len}


def check_chat_json(ck, c, d):
    if d.get("object") != "chat.completion":
        ck.err(f"object={d.get('object')}")
    if d.get("model") != c["body"].get("model"):
        ck.err(f"model={d.get('model')!r}")
    calls, seen = [], set()
    finish = None
    content_len = reasoning_len = 0
    for ch in d.get("choices") or []:
        m = ch.get("message") or {}
        finish = ch.get("finish_reason")
        if not finish:
            ck.err("finish_reason が無い")
        if m.get("role") != "assistant":
            ck.err(f"role={m.get('role')}")
        tcs = m.get("tool_calls") or []
        if finish == "tool_calls" and not tcs:
            ck.err("finish_reason=tool_calls なのに呼び出しが無い")
        if tcs and finish not in ("tool_calls", "length"):
            ck.err(f"呼び出しがあるのに finish_reason={finish}")
        if "｜DSML｜" in (m.get("content") or ""):
            ck.err("content に DSML が漏れた")
        for tc in tcs:
            if tc.get("type") != "function":
                ck.err(f"tool_call type={tc.get('type')}")
            f = tc.get("function") or {}
            if not isinstance(f.get("arguments"), str):
                ck.err("arguments が文字列でない")
            check_call(ck, c, f.get("name"), f.get("arguments") or "", tc.get("id"), finish, seen, calls)
        content_len += len(m.get("content") or "")
        reasoning_len += len(m.get("reasoning_content") or "")
    if not d.get("choices"):
        ck.err("choices が空")
    if not d.get("usage"):
        ck.err("usage が無い")
    return {"ttft": None, "finish": finish, "usage": d.get("usage"), "tool_calls": calls, "content_chars": content_len,
            "reasoning_chars": reasoning_len}


DELTA_TYPES = {"response.output_text.delta", "response.reasoning_text.delta", "response.reasoning_summary_text.delta",
               "response.function_call_arguments.delta", "response.refusal.delta"}


def check_resp_output(ck, c, resp, finish, calls, seen):
    content_len = reasoning_len = 0
    for it in resp.get("output") or []:
        t = it.get("type")
        if not it.get("id"):
            ck.err(f"output 項目 {t} に id が無い")
        if t == "function_call":
            check_call(ck, c, it.get("name"), it.get("arguments") or "", it.get("call_id"), finish, seen, calls)
            if it.get("status") not in ("completed", "incomplete"):
                ck.err(f"function_call status={it.get('status')}")
        elif t == "message":
            for p in it.get("content") or []:
                if p.get("type") == "output_text":
                    content_len += len(p.get("text") or "")
                    if "｜DSML｜" in (p.get("text") or ""):
                        ck.err("output_text に DSML が漏れた")
        elif t == "reasoning":
            for p in (it.get("content") or []) + (it.get("summary") or []):
                reasoning_len += len(p.get("text") or "")
        else:
            ck.err(f"未知の output 項目 {t}")
    return content_len, reasoning_len


def check_resp_stream(ck, c, body, events):
    ttft = None
    seq = -1
    final = None
    open_item = None
    items_done = {}
    arg_acc = {}
    types = []
    for name, data, t in events:
        try:
            ev = json.loads(data)
        except Exception:
            ck.err(f"data が JSON でない: {data[:80]!r}")
            continue
        ty = ev.get("type")
        types.append(ty)
        if name and name != ty:
            ck.err(f"event 名 {name} と type {ty} が違う")
        sn = ev.get("sequence_number")
        if sn != seq + 1:
            ck.err(f"sequence_number {sn} (期待 {seq + 1}) @ {ty}")
        seq = sn if isinstance(sn, int) else seq + 1
        if ttft is None and ty in DELTA_TYPES and ev.get("delta"):
            ttft = t
        if ty == "response.output_item.added":
            if open_item is not None:
                ck.err("前の項目が閉じる前に次の項目が開いた")
            open_item = ev["item"].get("id")
        elif ty == "response.output_item.done":
            if ev["item"].get("id") != open_item:
                ck.err("output_item.done の id が開いた項目と違う")
            items_done[ev.get("output_index")] = ev["item"]
            open_item = None
        elif ty in DELTA_TYPES or ty.endswith(".done") and ty.startswith(("response.output_text", "response.reasoning", "response.function_call")):
            if ev.get("item_id") and ev.get("item_id") != open_item:
                ck.err(f"{ty} の item_id が開いた項目でない")
            if ty == "response.function_call_arguments.delta":
                arg_acc[ev["item_id"]] = arg_acc.get(ev["item_id"], "") + ev.get("delta", "")
            if ty == "response.function_call_arguments.done":
                if ev.get("arguments") != arg_acc.get(ev["item_id"], ""):
                    ck.err("function_call_arguments.done と delta の連結が違う")
        elif ty in ("response.completed", "response.incomplete", "response.failed"):
            final = ev
        elif ty == "error":
            ck.err(f"error イベント: {json.dumps(ev, ensure_ascii=False)[:200]}")
    if not types or types[0] != "response.created":
        ck.err(f"最初のイベントが response.created でない ({types[:1]})")
    if final is None:
        ck.err(f"終了イベントが無い (最後 {types[-3:]})")
        return {"ttft": ttft, "finish": None, "usage": None, "tool_calls": [], "content_chars": 0, "reasoning_chars": 0}
    if final["type"] == "response.failed":
        ck.err(f"response.failed: {json.dumps(final['response'].get('error'), ensure_ascii=False)[:200]}")
    if types[-1] != final["type"]:
        ck.err("終了イベントの後にイベント")
    resp = final["response"]
    status = resp.get("status")
    if resp.get("model") != body.get("model"):
        ck.err(f"model={resp.get('model')!r}")
    if final["type"] == "response.incomplete":
        finish = "incomplete"
    else:
        finish = status
    # 完了オブジェクトの output と output_item.done の一致
    out = resp.get("output") or []
    if len(out) != len(items_done):
        ck.err(f"output {len(out)} 件と output_item.done {len(items_done)} 件が違う")
    for i, it in enumerate(out):
        d = items_done.get(i)
        if d is not None and json.dumps(d, sort_keys=True) != json.dumps(it, sort_keys=True):
            ck.err(f"output[{i}] と output_item.done が違う")
    calls, seen = [], set()
    cl, rl = check_resp_output(ck, c, resp, finish, calls, seen)
    if not resp.get("usage"):
        ck.err("usage が無い")
    return {"ttft": ttft, "finish": finish, "usage": resp.get("usage"), "tool_calls": calls, "content_chars": cl,
            "reasoning_chars": rl, "resp_status": status, "incomplete_details": resp.get("incomplete_details")}


def check_resp_json(ck, c, d):
    if d.get("object") != "response":
        ck.err(f"object={d.get('object')}")
    if d.get("model") != c["body"].get("model"):
        ck.err(f"model={d.get('model')!r}")
    status = d.get("status")
    if status not in ("completed", "incomplete"):
        ck.err(f"status={status} {json.dumps(d.get('error'), ensure_ascii=False)[:200]}")
    finish = "incomplete" if status == "incomplete" else status
    calls, seen = [], set()
    cl, rl = check_resp_output(ck, c, d, finish, calls, seen)
    if not d.get("usage"):
        ck.err("usage が無い")
    return {"ttft": None, "finish": finish, "usage": d.get("usage"), "tool_calls": calls, "content_chars": cl,
            "reasoning_chars": rl, "resp_status": status, "incomplete_details": d.get("incomplete_details")}


def run_proxy_one(c, cap, base=PROXY):
    body = cap_body(c, cap)
    path = "/v1/chat/completions" if c["api"] == "chat" else "/v1/responses"
    ck = Check()
    rec = {"id": c["id"], "api": c["api"], "client": c["client"], "stream": bool(body.get("stream")), "bytes": c["bytes"]}
    try:
        conn, r, t0 = post(base, path, body)
        rec["status"] = r.status
        rec["rid"] = r.getheader("x-request-id")
        rec["warning"] = r.getheader("x-ds41gw-warning")
        ctype = r.getheader("content-type") or ""
        if r.status != 200:
            rec["error_body"] = r.read().decode("utf-8", "replace")[:600]
            ck.err(f"HTTP {r.status}")
        elif "text/event-stream" in ctype:
            if not body.get("stream"):
                ck.err("非ストリーム要求に SSE が返った")
            events, tfb = read_sse(r, t0)
            rec["t_first_byte"] = tfb
            rec.update(check_chat_stream(ck, c, body, events, t0) if c["api"] == "chat" else check_resp_stream(ck, c, body, events))
        else:
            if body.get("stream"):
                ck.err(f"ストリーム要求に {ctype} が返った")
            raw = r.read()
            try:
                d = json.loads(raw)
            except Exception:
                d = {}
                ck.err(f"本文が JSON でない: {raw[:100]!r}")
            rec.update(check_chat_json(ck, c, d) if c["api"] == "chat" else check_resp_json(ck, c, d))
        rec["e2e"] = time.perf_counter() - t0
        conn.close()
    except Exception as e:
        rec["status"] = rec.get("status") or -1
        ck.err(f"例外: {type(e).__name__}: {e}")
    rec["errors"], rec["notes"] = ck.errors, ck.notes
    rec["ok"] = not ck.errors
    return rec


# ---------------------------------------------------------------- ds41rt 直
def run_direct_one(obj, max_tokens=None):
    body = dict(obj, stream=True, stream_options={"include_usage": True})
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    body["model"] = DS41RT_MODEL
    rec = {}
    try:
        conn, r, t0 = post(DIRECT, "/v1/chat/completions", body)
        rec["status"] = r.status
        if r.status != 200:
            rec["error_body"] = r.read().decode("utf-8", "replace")[:400]
            conn.close()
            return rec
        events, tfb = read_sse(r, t0)
        ttft = None
        usage = None
        for _, data, t in events:
            if data == "[DONE]":
                continue
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                    ttft = t
        rec.update(ttft=ttft, t_first_byte=tfb, e2e=time.perf_counter() - t0, usage=usage)
        conn.close()
    except Exception as e:
        rec["status"] = -1
        rec["error_body"] = f"{type(e).__name__}: {e}"
    return rec


def load_proxy_log(rids):
    want = set(rids)
    recv, fin = {}, {}
    for f in sorted(glob.glob(os.path.join(HERE, "..", "logs", "req", "*.jsonl"))):
        for line in open(f, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("rid") not in want:
                continue
            if d.get("event") == "request.received.openai":
                recv[d["rid"]] = d
            elif d.get("event") == "request.finished":
                fin[d["rid"]] = d
    return recv, fin


def pmap(fn, items, conc):
    out = [None] * len(items)
    lock = threading.Lock()
    done = [0]

    def wrap(i):
        out[i] = fn(items[i])
        with lock:
            done[0] += 1
            r = out[i] if isinstance(out[i], dict) else {}
            print(f"[{done[0]}/{len(items)}] {r.get('id', '')} {r.get('status', '')} {'OK' if r.get('ok', True) else 'NG ' + '; '.join(r.get('errors', [])[:2])}",
                  flush=True)

    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(wrap, range(len(items))))
    return out


def cmd_run(a):
    corpus = [json.loads(l) for l in open(a.corpus)]
    if a.only:
        ids = set(a.only.split(","))
        corpus = [c for c in corpus if c["id"] in ids]
    order = corpus[:]
    random.Random(a.seed).shuffle(order)  # クライアントが混ざった到着順にする
    t0 = time.time()
    res = pmap(lambda c: run_proxy_one(c, a.max_tokens), order, a.conc)
    print(f"elapsed {time.time() - t0:.1f}s")
    time.sleep(1.0)
    recv, fin = load_proxy_log([r.get("rid") for r in res if r.get("rid")])
    for r in res:
        f = fin.get(r.get("rid")) or {}
        v = recv.get(r.get("rid")) or {}
        r["gw"] = {k: f.get(k) for k in ("status", "upstream_status", "queue_ms", "ttft_ms", "dur_ms", "upstream_retries", "finish_reason",
                                         "reinjected", "dsml_leaks", "midstream_failures", "error", "warnings", "reasoning_tokens")}
        r["gw"]["normalized"] = v.get("normalized")
        r["gw"]["recv_warnings"] = v.get("warnings")
    with open(a.out, "w") as fh:
        for r in sorted(res, key=lambda r: r["id"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summarize(res)


def summarize(res):
    n = len(res)
    bad = [r for r in res if not r["ok"]]
    print(f"\n{n} 件、合格 {n - len(bad)}、不合格 {len(bad)}")
    for r in bad:
        print(f"  {r['id']} {r['client']} status={r.get('status')} {r['errors'][:3]} {r.get('error_body', '')[:200]}")
    calls = [t for r in res for t in r.get("tool_calls") or []]
    print(f"ツール呼び出し {len(calls)} 本、arguments が JSON object {sum(t['args_ok'] for t in calls)} 本")


def cmd_pairs(a):
    res = [json.loads(l) for l in open(a.inp)]
    corpus = {json.loads(l)["id"]: json.loads(l) for l in open(a.corpus)}
    recv, _ = load_proxy_log([r.get("rid") for r in res if r.get("rid")])
    todo = []
    for r in res:
        if r.get("status") != 200 or not r.get("stream"):
            continue
        v = recv.get(r.get("rid"))
        if not v or not isinstance(v.get("obj"), dict):
            continue
        if '"image_url"' in json.dumps(v["obj"]):
            continue  # 要求ログの画像はハッシュに置き換わっているので ds41rt 直には送れない
        todo.append((corpus[r["id"]], v["obj"]))
    rnd = random.Random(a.seed)
    jobs = []
    for i, (c, obj) in enumerate(todo):
        for rep in range(a.reps):
            jobs.append((c, obj, rnd.random() < 0.5))
    rnd.shuffle(jobs)

    def one(job):
        c, obj, direct_first = job
        out = {"id": c["id"], "api": c["api"], "direct_first": direct_first}
        # 接頭辞キャッシュを両者で同じ状態にするため、先に ds41rt 直で 1 トークンだけ生成して温める
        out["warm"] = run_direct_one(obj, max_tokens=1)
        mt = a.max_tokens
        dobj = dict(obj, max_tokens=min(int(obj.get("max_tokens") or mt), mt))
        if a.abba:
            # D P P D か P D D P。順番による差 (キャッシュの状態・混み具合の変化) を打ち消す
            seq = ["d", "p", "p", "d"] if direct_first else ["p", "d", "d", "p"]
        else:
            seq = ["d", "p"] if direct_first else ["p", "d"]
        got = {"p": [], "d": []}
        for k in seq:
            got[k].append(run_direct_one(dobj) if k == "d" else run_proxy_one(c, mt))
        out["proxy"], out["direct"] = got["p"][0], got["d"][0]
        out["proxy_all"] = [{"ttft": x.get("ttft"), "t_first_byte": x.get("t_first_byte"), "usage": x.get("usage")} for x in got["p"]]
        out["direct_all"] = [{"ttft": x.get("ttft"), "t_first_byte": x.get("t_first_byte"), "usage": x.get("usage")} for x in got["d"]]
        pt = [x["ttft"] for x in got["p"] if x.get("ttft") is not None]
        dt = [x["ttft"] for x in got["d"] if x.get("ttft") is not None]
        if len(pt) == len(seq) // 2 and len(dt) == len(seq) // 2:
            out["proxy"] = dict(out["proxy"], ttft=statistics.mean(pt))
            out["direct"] = dict(out["direct"], ttft=statistics.mean(dt))
        if any(not x.get("ok") for x in got["p"]) or any(x.get("status") != 200 for x in got["d"]):
            out["proxy"] = next((x for x in got["p"] if not x.get("ok")), out["proxy"])
            out["direct"] = next((x for x in got["d"] if x.get("status") != 200), out["direct"])
        p, d = out["proxy"], out["direct"]
        out["ok"] = p.get("ok") and d.get("status") == 200
        out["status"] = f"{p.get('status')}/{d.get('status')}"
        out["errors"] = p.get("errors", []) + ([d.get("error_body", "")] if d.get("status") != 200 else [])
        if p.get("ttft") is not None and d.get("ttft") is not None:
            out["d_ttft_ms"] = (p["ttft"] - d["ttft"]) * 1000
        cached = lambda u: (u or {}).get("prompt_cache_hit_tokens", ((u or {}).get("input_tokens_details") or (u or {}).get("prompt_tokens_details") or {}).get("cached_tokens"))
        out["cached"] = {"proxy": cached(p.get("usage")), "direct": cached(d.get("usage"))}
        if p.get("t_first_byte") is not None and d.get("t_first_byte") is not None:
            out["d_first_byte_ms"] = (p["t_first_byte"] - d["t_first_byte"]) * 1000
        return out

    print(f"{len(jobs)} pairs, conc {a.conc}")
    out = pmap(one, jobs, a.conc)
    with open(a.out, "w") as fh:
        for o in out:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    report_pairs(out)


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def report_pairs(out):
    for api in ("chat", "responses", None):
        xs = [o["d_ttft_ms"] for o in out if "d_ttft_ms" in o and (api is None or o["api"] == api)]
        if xs:
            print(f"{api or 'all'}: n={len(xs)} TTFT 差 (ds41gw − ds41rt 直) p50 {pct(xs, .5):+.1f} ms  p10 {pct(xs, .1):+.1f}  p90 {pct(xs, .9):+.1f}  mean {statistics.mean(xs):+.1f}")
        ys = [o["proxy"]["ttft"] * 1000 for o in out if "d_ttft_ms" in o and (api is None or o["api"] == api)]
        zs = [o["direct"]["ttft"] * 1000 for o in out if "d_ttft_ms" in o and (api is None or o["api"] == api)]
        if ys:
            print(f"   TTFT p50 ds41gw {pct(ys, .5):.0f} ms / 直 {pct(zs, .5):.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "pairs"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", default=os.path.join(HERE, "replay_corpus.jsonl"))
        p.add_argument("--conc", type=int, default=4)
        p.add_argument("--max-tokens", type=int, default=256)
        p.add_argument("--seed", type=int, default=3)
        p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--only", default="")
        else:
            p.add_argument("--in", dest="inp", required=True)
            p.add_argument("--reps", type=int, default=1)
            p.add_argument("--abba", action="store_true", help="組ごとに D P P D / P D D P で 4 回送り、平均の差を取る")
    p = sub.add_parser("report")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--pairs", default="")
    a = ap.parse_args()
    assert getattr(a, "conc", 1) <= 4, "ds41rt は稼働中なので同時 4 本まで"
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "pairs":
        cmd_pairs(a)
    else:
        summarize([json.loads(l) for l in open(a.inp)])
        if a.pairs:
            report_pairs([json.loads(l) for l in open(a.pairs)])


if __name__ == "__main__":
    main()
