# mock ds41rt-like chat server on :8010. stream: 1 role chunk, R reasoning, C content, 1 tool call (8 arg chunks), finish, usage. every chunk DT seconds apart.
import asyncio, json, time, uuid
from aiohttp import web
R, C, DT = 40, 40, 0.010
def ch(i, delta, fin=None, usage=None):
    return {"id": i, "object": "chat.completion.chunk", "created": int(time.time()), "model": "deepseek-ai/DeepSeek-V4.1-Flash",
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": fin}], "usage": usage}
async def chat(req):
    body = await req.json()
    i = "chatcmpl-" + str(uuid.uuid4())
    usage = {"prompt_tokens": 1000, "completion_tokens": R + C + 10, "total_tokens": 1090 + R + C, "prompt_tokens_details": {"cached_tokens": 900}}
    if not body.get("stream"):
        await asyncio.sleep(DT * (R + C + 10))
        return web.json_response({"id": i, "object": "chat.completion", "created": int(time.time()), "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "x" * C, "reasoning_content": "r" * R,
              "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"a.txt\"}"}}]},
              "finish_reason": "tool_calls"}], "usage": usage})
    resp = web.StreamResponse(headers={"content-type": "text/event-stream", "cache-control": "no-cache"})
    await resp.prepare(req)
    async def send(o):
        await resp.write(("data: " + json.dumps(o) + "\n\n").encode())
    await send(ch(i, {"role": "assistant", "content": None, "reasoning_content": ""}))
    t0 = time.time(); k = 0
    async def tick():
        nonlocal k
        k += 1
        d = t0 + k * DT - time.time()
        if d > 0: await asyncio.sleep(d)
    for n in range(R):
        await tick(); await send(ch(i, {"content": None, "reasoning_content": f"r{n} "}))
    for n in range(C):
        await tick(); await send(ch(i, {"content": f"c{n} ", "reasoning_content": None}))
    await tick(); await send(ch(i, {"tool_calls": [{"index": 0, "id": "call_0", "type": "function", "function": {"name": "read", "arguments": ""}}]}))
    for piece in ["{", "\"", "path", "\"", ": ", "\"", "a.txt", "\"}"]:
        await tick(); await send(ch(i, {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}))
    await send(ch(i, {"content": "", "reasoning_content": None}, fin="tool_calls"))
    if (body.get("stream_options") or {}).get("include_usage"):
        o = ch(i, {}); o["choices"] = []; o["usage"] = usage; await send(o)
    await resp.write(b"data: [DONE]\n\n"); await resp.write_eof(); return resp
async def models(req):
    return web.json_response({"object": "list", "data": [{"id": "deepseek-ai/DeepSeek-V4.1-Flash", "object": "model", "owned_by": "x"}]})
app = web.Application(); app.router.add_post("/v1/chat/completions", chat); app.router.add_get("/v1/models", models)
web.run_app(app, host="127.0.0.1", port=8010, print=None)
