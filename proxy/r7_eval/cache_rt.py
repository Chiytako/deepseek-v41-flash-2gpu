import json, urllib.request, uuid
U = "http://localhost:8000/v1/chat/completions"; M = "deepseek-ai/DeepSeek-V4.1-Flash"
tools = [{"type": "function", "function": {"name": "read", "description": "read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
def post(b):
    r = urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=300)
    return json.loads(r.read())
nonce = uuid.uuid4().hex
sys_msg = {"role": "system", "content": f"session {nonce}. " + "You are a coding agent. " * 200}
user = {"role": "user", "content": "Read a.txt with the tool."}
r1 = post({"model": M, "messages": [sys_msg, user], "tools": tools, "max_tokens": 200, "reasoning_effort": "low"})
m = r1["choices"][0]["message"]; print("r1 usage", r1["usage"]["prompt_tokens"], "finish", r1["choices"][0]["finish_reason"], "reasoning len", len(m.get("reasoning_content") or ""))
tc = m["tool_calls"]
tool = {"role": "tool", "tool_call_id": tc[0]["id"], "content": "hello"}
with_r = {"role": "assistant", "content": m.get("content") or "", "reasoning_content": m.get("reasoning_content"), "tool_calls": tc}
without_r = {"role": "assistant", "content": m.get("content") or "", "tool_calls": tc}
for name, a in [("without_reasoning", without_r), ("with_reasoning", with_r)]:
    r = post({"model": M, "messages": [sys_msg, user, a, tool], "tools": tools, "max_tokens": 1, "thinking": {"type": "enabled"}, "reasoning_effort": "low"})
    u = r["usage"]; print(name, "prompt", u["prompt_tokens"], "cached", u.get("prompt_cache_hit_tokens"), "miss", u.get("prompt_cache_miss_tokens"))
