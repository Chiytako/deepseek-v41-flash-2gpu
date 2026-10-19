# concurrency bench against mock-backed endpoints. usage: bench.py NAME URL PATH MODEL KEY CONC ROUNDS
import asyncio, json, sys, time, statistics, aiohttp
name, url, path, model, key, conc, rounds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7])
def body():
    if path.endswith("responses"):
        return {"model": model, "input": "hi", "stream": True}
    if path.endswith("messages"):
        return {"model": model, "max_tokens": 200, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    return {"model": model, "stream": True, "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": "hi"}]}
async def one(s, res):
    t0 = time.perf_counter(); first = None; last = t0; gaps = []; n = 0
    async with s.post(url + path, json=body(), headers={"Authorization": "Bearer " + key, "x-api-key": key, "anthropic-version": "2023-06-01"}) as r:
        assert r.status == 200, (r.status, await r.text())
        async for line in r.content:
            if not line.startswith(b"data:"): continue
            s_ = line[5:].strip()
            if s_ == b"[DONE]": continue
            if (b'"delta"' in s_ or b'_delta' in s_) and (b'r0 ' in s_ or b'c' in s_ or b'arguments' in s_ or b'partial_json' in s_):
                now = time.perf_counter(); n += 1
                if first is None:
                    if b'r0 ' in s_: first = now - t0
                    else: n -= 1; continue
                else: gaps.append(now - last)
                last = now
    res.append((first, time.perf_counter() - t0, gaps, n))
async def main():
    res = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), connector=aiohttp.TCPConnector(limit=0)) as s:
        await one(s, [])  # warmup
        for _ in range(rounds):
            await asyncio.gather(*[one(s, res) for _ in range(conc)])
    ttft = sorted(r[0] for r in res); tot = sorted(r[1] for r in res); gaps = sorted(g for r in res for g in r[2])
    q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))] * 1000
    print(f"{name:22s} n={len(res)} deltas/req={statistics.mean(r[3] for r in res):.0f} TTFT p50 {q(ttft,.5):6.1f} p95 {q(ttft,.95):6.1f} ms | total p50 {q(tot,.5):6.0f} p95 {q(tot,.95):6.0f} ms | gap p50 {q(gaps,.5):5.1f} p99 {q(gaps,.99):5.1f} max {gaps[-1]*1000:5.1f} ms")
asyncio.run(main())
