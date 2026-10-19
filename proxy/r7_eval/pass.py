# logging pass-through: 127.0.0.1:8009 -> 127.0.0.1:8000, request bodies appended to tee.jsonl
import json, time, aiohttp
from aiohttp import web
import os
UP = os.environ.get("UP", "http://127.0.0.1:8000")
async def h(req):
    body = await req.read()
    if os.environ.get("NOLOG") is None:
      with open("tee.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "path": req.path, "body": body.decode("utf-8", "replace")}) + "\n")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        async with s.request(req.method, UP + req.path_qs, data=body, headers={"content-type": req.headers.get("content-type", "application/json")}) as r:
            resp = web.StreamResponse(status=r.status, headers={"content-type": r.headers.get("content-type", "")})
            await resp.prepare(req)
            async for chunk in r.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp
app = web.Application(client_max_size=64 << 20)
app.router.add_route("*", "/{tail:.*}", h)
web.run_app(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8009")), print=None)
