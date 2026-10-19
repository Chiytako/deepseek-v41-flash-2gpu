# minimal translating proxy (Responses stream -> chat stream) to estimate per-chunk cost of a hand-written Python proxy. :8012
import json, os, uuid, aiohttp
from aiohttp import web
UP = os.environ.get("UP", "http://127.0.0.1:8010")
async def responses(req):
    b = await req.json()
    msgs = [{"role": "user", "content": b["input"]}] if isinstance(b["input"], str) else b["input"]
    up = {"model": "deepseek-ai/DeepSeek-V4.1-Flash", "messages": msgs, "stream": True, "stream_options": {"include_usage": True}}
    rid = "resp_" + uuid.uuid4().hex
    resp = web.StreamResponse(headers={"content-type": "text/event-stream"}); await resp.prepare(req)
    seq = 0
    async def ev(o):
        nonlocal seq
        o["sequence_number"] = seq; seq += 1
        await resp.write(("event: " + o["type"] + "\ndata: " + json.dumps(o, separators=(",", ":")) + "\n\n").encode())
    await ev({"type": "response.created", "response": {"id": rid, "object": "response", "status": "in_progress", "output": []}})
    async with req.app["s"].post(UP + "/v1/chat/completions", json=up) as r:
        buf = b""
        async for chunk in r.content.iter_any():
            buf += chunk
            while b"\n\n" in buf:
                line, buf = buf.split(b"\n\n", 1)
                if not line.startswith(b"data: ") or line[6:] == b"[DONE]": continue
                j = json.loads(line[6:])
                for c in j.get("choices") or []:
                    d = c.get("delta") or {}
                    if d.get("reasoning_content"): await ev({"type": "response.reasoning_text.delta", "item_id": "rs_0", "output_index": 0, "content_index": 0, "delta": d["reasoning_content"]})
                    if d.get("content"): await ev({"type": "response.output_text.delta", "item_id": "msg_0", "output_index": 1, "content_index": 0, "delta": d["content"]})
                    for t in d.get("tool_calls") or []:
                        a = (t.get("function") or {}).get("arguments")
                        if a: await ev({"type": "response.function_call_arguments.delta", "item_id": "fc_0", "output_index": 2, "delta": a})
    await ev({"type": "response.completed", "response": {"id": rid, "object": "response", "status": "completed"}})
    await resp.write_eof(); return resp
async def init(app):
    app["s"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))
app = web.Application(); app.on_startup.append(init); app.router.add_post("/v1/responses", responses)
web.run_app(app, host="127.0.0.1", port=8012, print=None)
