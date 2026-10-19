#!/usr/bin/env python3
# replay_real.py の要求集を少し変えて、実クライアントの文脈のまま別の経路を通す (同時 4 本、出力 256 まで)。
#   V1 ストリーム ↔ 非ストリームの入れ替え (型 B・C・D を非ストリーム、型 A をストリーム + include_usage)
#   V2 Responses に include:["reasoning.encrypted_content"] と reasoning.summary:"auto" (Codex 系の送り方)
#   V3 Responses の往復: ds41gw の応答 (reasoning + function_call) を履歴に足し、ツール結果を付けて次のターンを送る。
#      2 ターン目の上流本文 (要求ログの obj) に 1 ターン目の思考が入っているかを確かめる (encrypted_content だけ / 項目そのまま)
# 使い方: python3 tests/replay_variants.py --out tests/results_replay_variants.jsonl
import argparse, importlib.util, json, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("replay_real", os.path.join(HERE, "replay_real.py"))
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default=os.path.join(HERE, "replay_corpus.jsonl"))
ap.add_argument("--out", required=True)
ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--seed", type=int, default=5)
A = ap.parse_args()
assert A.conc <= 4
rnd = random.Random(A.seed)
corpus = [json.loads(l) for l in open(A.corpus)]


def variant(c, vid, **kw):
    v = json.loads(json.dumps(c))
    v["id"] = f"{vid}:{c['id']}"
    for k, val in kw.items():
        if val is None:
            v["body"].pop(k, None)
        else:
            v["body"][k] = val
    v["stream"] = bool(v["body"].get("stream"))
    return v


jobs = []
streamers = [c for c in corpus if c["api"] == "chat" and c["stream"] and c["client"].split("+")[0] in ("B", "C-new", "C-old", "D")]
for c in rnd.sample(streamers, 8):
    jobs.append(variant(c, "V1n", stream=False, stream_options=None))
for c in rnd.sample([c for c in corpus if c["client"] == "A"], 6):
    jobs.append(variant(c, "V1s", stream=True, stream_options={"include_usage": True}))
resp = [c for c in corpus if c["api"] == "responses" and "tools" in c["body"]]
for i, c in enumerate(rnd.sample(resp, 6)):
    jobs.append(variant(c, "V2", include=["reasoning.encrypted_content"], reasoning={"effort": "high", "summary": "auto"}, stream=i % 3 != 2))

res = rr.pmap(lambda c: rr.run_proxy_one(c, A.max_tokens), jobs, A.conc)


# V3: 往復 (逐次でよいが、2 本ずつ並べる)
def roundtrip(c, mode):
    """mode: 'enc' = reasoning 項目を encrypted_content だけにして返す / 'asis' = 出力項目をそのまま返す"""
    body = rr.cap_body(c, A.max_tokens)
    # 256 トークンで呼び出しまで届くよう effort は low にする (2 ターン目も同じ)
    body.update(stream=False, include=["reasoning.encrypted_content"], reasoning={"effort": "low"})
    out = {"id": f"V3{mode}:{c['id']}", "api": "responses", "client": c["client"], "stream": False, "bytes": c["bytes"], "errors": [], "notes": []}
    for attempt in range(3):
        conn, r, t0 = rr.post(rr.PROXY, "/v1/responses", body)
        d = json.loads(r.read() or b"{}")
        conn.close()
        calls = [it for it in d.get("output") or [] if it.get("type") == "function_call"]
        if r.status == 200 and calls and d.get("status") == "completed":
            break
    else:
        out.update(status=r.status, ok=True, skipped="1 ターン目でツール呼び出しが完了しなかった")
        return out
    reasoning = [it for it in d["output"] if it.get("type") == "reasoning"]
    think = "".join(p.get("text", "") for it in reasoning for p in (it.get("content") or []))
    hist = []
    for it in d["output"]:
        if it["type"] == "reasoning" and mode == "enc":
            hist.append({"type": "reasoning", "id": it["id"], "summary": [], "encrypted_content": it.get("encrypted_content")})
        else:
            hist.append(it)
    for cl in calls:
        hist.append({"type": "function_call_output", "call_id": cl["call_id"], "output": "[status: completed, exit code: 0]\n(replay: 出力は省略)"})
    c2 = json.loads(json.dumps(c))
    c2["body"]["input"] = c["body"]["input"] + hist
    c2["body"]["reasoning"] = {"effort": "low"}
    c2["body"]["include"] = ["reasoning.encrypted_content"]
    rec = rr.run_proxy_one(c2, A.max_tokens)
    rec["id"] = out["id"]
    recv, _ = rr.load_proxy_log([rec.get("rid")]) if rec.get("rid") else ({}, {})
    time.sleep(0.3)
    if rec.get("rid") and not recv:
        recv, _ = rr.load_proxy_log([rec["rid"]])
    obj = (recv.get(rec.get("rid")) or {}).get("obj") or {}
    msgs = obj.get("messages") or []
    last_asst = [m for m in msgs if m.get("role") == "assistant"]
    got = (last_asst[-1].get("reasoning_content") or "") if last_asst else ""
    rec["roundtrip"] = {"turn1_reasoning_chars": len(think), "turn2_upstream_reasoning_chars": len(got), "match": got == think,
                        "turn1_calls": [cl["name"] for cl in calls]}
    if think and got != think:
        rec["errors"].append(f"2 ターン目の上流本文に 1 ターン目の思考が戻っていない ({len(got)} / {len(think)} 字)")
        rec["ok"] = False
    return rec


v3 = [(c, m) for c in rnd.sample([c for c in resp if c["bytes"] < 120_000], 4) for m in ("enc", "asis")]
res += rr.pmap(lambda cm: roundtrip(*cm), v3, min(A.conc, 2))

with open(A.out, "w") as fh:
    for r in res:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
rr.summarize(res)
for r in res:
    if r.get("roundtrip") or r.get("skipped"):
        print(r["id"], r.get("roundtrip") or r.get("skipped"))
