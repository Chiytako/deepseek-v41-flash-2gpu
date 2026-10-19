#!/usr/bin/env python3
# 同じ seed・temperature 0 で、プロキシ経由 (Chat / Responses / Messages の SDK) と ds41rt 直 (:8000) の出力を突き合わせる。逐次、max_tokens 64。
# 使い方: tests/.venv/bin/python -I tests/effort_direct.py
import json, os, sys, urllib.request, warnings
import anthropic, openai

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_effort_direct.json")
OA = openai.OpenAI(base_url="http://127.0.0.1:30001/v1", api_key="none", max_retries=0, timeout=300)
AN = anthropic.Anthropic(base_url="http://127.0.0.1:30001", api_key="none", max_retries=0, timeout=300)
Q = "How many letters r are in the word strawberry? Think, then answer with one number."
MSG = [{"role": "user", "content": Q}]
MT = 64
SEED = 7


def direct(effort):
    body = {"model": "deepseek-ai/DeepSeek-V4.1-Flash", "messages": MSG, "max_tokens": MT, "temperature": 0, "seed": SEED}
    if effort is None:
        body["thinking"] = {"type": "disabled"}
    else:
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = effort
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    m = r["choices"][0]["message"]
    return {"reasoning": m.get("reasoning_content") or "", "content": m.get("content") or "", "prompt_tokens": r["usage"]["prompt_tokens"]}


def via_chat(v):
    kw = {} if v == "(unspec)" else {"reasoning_effort": v}
    r = OA.chat.completions.create(model="x", messages=MSG, max_tokens=MT, temperature=0, seed=SEED, **kw)
    m = r.choices[0].message
    return {"reasoning": (m.model_extra or {}).get("reasoning_content") or "", "content": m.content or "", "prompt_tokens": r.usage.prompt_tokens}


def via_resp(v):
    kw = {} if v == "(unspec)" else {"reasoning": {"effort": v}}
    r = OA.responses.create(model="x", input=Q, max_output_tokens=MT, temperature=0, extra_body={"seed": SEED}, **kw)
    rs = "".join(c.text for o in r.output if o.type == "reasoning" for c in (o.content or []))
    rs = rs or "".join(s.text for o in r.output if o.type == "reasoning" for s in (o.summary or []))
    return {"reasoning": rs, "content": r.output_text, "prompt_tokens": r.usage.input_tokens}


def via_msg(v):
    kw = {} if v == "(unspec)" else {"output_config": {"effort": v}}
    m = AN.messages.create(model="x", messages=MSG, max_tokens=MT, extra_body={"seed": SEED, "temperature": 0}, **kw)
    return {"reasoning": "".join(b.thinking for b in m.content if b.type == "thinking"),
            "content": "".join(b.text for b in m.content if b.type == "text"), "prompt_tokens": m.usage.input_tokens + (m.usage.cache_read_input_tokens or 0)}


def via_msg_budget(b):
    m = AN.messages.create(model="x", messages=MSG, max_tokens=MT, extra_body={"seed": SEED, "temperature": 0}, thinking={"type": "enabled", "budget_tokens": b})
    return {"reasoning": "".join(b.thinking for b in m.content if b.type == "thinking"),
            "content": "".join(b.text for b in m.content if b.type == "text"), "prompt_tokens": m.usage.input_tokens + (m.usage.cache_read_input_tokens or 0)}


ref = {}
for e in [1, 37, 50, 60, 62, 75, 88, 100, None]:
    ref[e] = direct(e)
    ref[f"{e}#2"] = direct(e)  # 直の再現性
    print("direct", e, ref[e]["prompt_tokens"], repr(ref[e]["reasoning"][:60]), "repeat_same:", ref[e] == ref[f"{e}#2"], flush=True)

plan = [
    ("chat", via_chat, [(1, 1), (37, 37), (60, 60), (100, 100), ("60", 60), ("minimal", 50), ("low", 50), ("medium", 62), ("high", 75), ("xhigh", 88), ("max", 100), ("none", None), ("(unspec)", 75)]),
    ("responses", via_resp, [(1, 1), (37, 37), (60, 60), (100, 100), ("60", 60), ("low", 50), ("medium", 62), ("xhigh", 88), ("max", 100), ("none", None), ("(unspec)", 75)]),
    ("messages", via_msg, [(1, 1), (37, 37), (60, 60), (100, 100), ("60", 60), ("low", 50), ("medium", 62), ("xhigh", 88), ("max", 100), ("(unspec)", 75)]),
    ("messages.budget", via_msg_budget, [(2048, 50), (16384, 75), (40000, 100)]),
]
rows = []
for api, fn, vals in plan:
    for v, want in vals:
        got = fn(v)
        r = ref[want]
        same = got["reasoning"] == r["reasoning"] and got["content"] == r["content"]
        rows.append({"api": api, "sent": v, "want": want, "same_output": same, "prompt_tokens": [got["prompt_tokens"], r["prompt_tokens"]],
                     "got": got if not same else None, "ref": r if not same else None})
        print(api, repr(v), "->", want, "same:", same, got["prompt_tokens"], r["prompt_tokens"], flush=True)

distinct = {str(k): v["reasoning"][:80] for k, v in ref.items() if "#" not in str(k)}
json.dump({"ref": {str(k): v for k, v in ref.items()}, "rows": rows, "distinct": distinct}, open(OUT, "w"), ensure_ascii=False, indent=1)
print(sum(r["same_output"] for r in rows), "/", len(rows), "same")
