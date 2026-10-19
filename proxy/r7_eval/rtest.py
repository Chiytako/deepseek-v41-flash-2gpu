# Responses / Messages probe against a proxy. usage: rtest.py URL KEY MODEL CASE
import json, sys, time, urllib.request
url, key, model, case = sys.argv[1:5]
TOOLS_R = [{"type": "function", "name": "read", "description": "read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
TOOLS_A = [{"name": "read", "description": "read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
def post(path, body, stream):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json",
        "Authorization": "Bearer " + key, "x-api-key": key, "anthropic-version": "2023-06-01"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=300)
    if not stream:
        return json.loads(r.read()), time.time() - t0
    evs = []; ev = None
    for raw in r:
        line = raw.decode().rstrip("\n")
        if line.startswith("event:"): ev = line[6:].strip()
        elif line.startswith("data:"):
            d = line[5:].strip()
            if d == "[DONE]": evs.append(("[DONE]", None)); continue
            j = json.loads(d); evs.append((j.get("type", ev), j))
    return evs, time.time() - t0
def summarize(evs):
    out = []; last = None; n = 0
    for t, j in evs:
        if t == last: n += 1; continue
        if last: out.append(f"{last}" + (f" x{n}" if n > 1 else ""))
        last, n = t, 1
    out.append(f"{last}" + (f" x{n}" if n > 1 else ""))
    return out
if case == "r_stream_tool":
    evs, dt = post("/v1/responses", {"model": model, "stream": True, "max_output_tokens": 300, "reasoning": {"effort": "low", "summary": "auto"},
        "store": False, "include": ["reasoning.encrypted_content"], "tools": TOOLS_R,
        "input": [{"role": "user", "content": "Read a.txt and b.txt with the tool, in parallel."}]}, True)
    print(dt); print("\n".join(summarize(evs)))
    for t, j in evs:
        if t == "response.output_item.done": print("ITEM", json.dumps(j["item"])[:400])
        if t == "response.completed": print("COMPLETED usage", j["response"].get("usage"), "status", j["response"].get("status"))
elif case == "r_nostream":
    j, dt = post("/v1/responses", {"model": model, "max_output_tokens": 200, "reasoning": {"effort": "low"}, "tools": TOOLS_R,
        "input": "Read a.txt with the tool."}, False)
    print(dt); print(json.dumps(j)[:1500])
elif case == "r_roundtrip":
    j, dt = post("/v1/responses", {"model": model, "max_output_tokens": 120, "reasoning": {"effort": "low"}, "tools": TOOLS_R, "store": False,
        "input": [{"role": "user", "content": "Read a.txt with the tool and tell me its content."},
                  {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "SUMMARY-THOUGHT: I will read a.txt"}],
                   "content": [{"type": "reasoning_text", "text": "CONTENT-THOUGHT: I will read a.txt"}]},
                  {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read", "arguments": "{\"path\":\"a.txt\"}"},
                  {"type": "function_call_output", "call_id": "call_1", "output": "hello from a"}]}, False)
    print(dt); print(json.dumps(j)[:1200])
elif case == "r_prev":
    j1, _ = post("/v1/responses", {"model": model, "max_output_tokens": 60, "reasoning": {"effort": "none"}, "store": True,
        "input": "My name is Taro. Reply OK."}, False)
    print("first", j1.get("id"), json.dumps(j1.get("output"))[:300])
    try:
        j2, _ = post("/v1/responses", {"model": model, "max_output_tokens": 60, "reasoning": {"effort": "none"},
            "previous_response_id": j1["id"], "input": "What is my name?"}, False)
        print("second", json.dumps(j2.get("output"))[:400])
    except urllib.error.HTTPError as e:
        print("second ERR", e.code, e.read()[:400])
elif case == "a_stream_tool":
    evs, dt = post("/v1/messages", {"model": model, "stream": True, "max_tokens": 400, "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tools": TOOLS_A, "messages": [{"role": "user", "content": "Read a.txt and b.txt with the tool, in parallel."}]}, True)
    print(dt); print("\n".join(summarize(evs)))
    for t, j in evs:
        if t in ("content_block_start", "message_delta"): print(t, json.dumps(j)[:300])
elif case == "a_roundtrip":
    j, dt = post("/v1/messages", {"model": model, "max_tokens": 200, "thinking": {"type": "enabled", "budget_tokens": 1024}, "tools": TOOLS_A,
        "messages": [{"role": "user", "content": "Read a.txt with the tool and tell me its content."},
                     {"role": "assistant", "content": [{"type": "thinking", "thinking": "ANTH-THOUGHT: read a.txt", "signature": "sig"},
                                                       {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a.txt"}}]},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "hello from a"}]}]}, False)
    print(dt); print(json.dumps(j)[:1200])
if case == "r_order":
    evs, dt = post("/v1/responses", {"model": model, "stream": True, "max_output_tokens": 300, "reasoning": {"effort": "low", "summary": "auto"},
        "store": False, "tools": TOOLS_R, "input": [{"role": "user", "content": "First say one short sentence, then read a.txt and b.txt with the tool, in parallel."}]}, True)
    for t, j in evs:
        if j is None: print(t); continue
        if t.endswith(".delta"): continue
        print(j.get("sequence_number"), t, "oi=", j.get("output_index"), "id=", j.get("item_id") or (j.get("item") or {}).get("id"), "ci=", j.get("content_index"), "si=", j.get("summary_index"))
