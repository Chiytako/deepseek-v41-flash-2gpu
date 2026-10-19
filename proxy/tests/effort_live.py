#!/usr/bin/env python3
# reasoning effort の実機試験。公式 openai / anthropic SDK で数値・数値文字列・段階名・未指定・不正値をプロキシ経由で 5 つの口に送り、
# 要求ログ (logs/req) の obj で ds41rt に届いた thinking と reasoning_effort を確かめる。逐次、max_tokens 48 以下。
# 使い方: tests/.venv/bin/python -I tests/effort_live.py [out.json]
import glob, json, os, sys, time, uuid, warnings
import anthropic, openai

warnings.filterwarnings("ignore")
URL = "http://127.0.0.1:30001"
LOGDIR = os.environ.get("DS41GW_LOGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "req"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_effort_live.json")
RUN = uuid.uuid4().hex[:8]
OA = openai.OpenAI(base_url=URL + "/v1", api_key="none", max_retries=0, timeout=300)
AN = anthropic.Anthropic(base_url=URL, api_key="none", max_retries=0, timeout=300)
MODEL = "GLM-5.3-Flash"
MSG = [{"role": "user", "content": "Say ok."}]
NAMES = {"none": None, "off": None, "disabled": None, "minimal": 50, "low": 50, "medium": 62, "high": 75, "xhigh": 88, "max": 100,
         "MAX": 100, "High": 75}
VALID = [(1, 1), (37, 37), (60, 60), (100, 100), ("60", 60), ("37", 37), (0, None), (0.37, 37), (59.5, 60)] + list(NAMES.items())
INVALID = [101, -1, 100.5, 1000, "abc", "inf", "nan", "1e3", True, [60], {"a": 1}, "very-high"]
cases = []


def ua(cid):
    return {"User-Agent": f"effort-verify/{RUN}/{cid}"}


def run(cid, api, sent, want, fn, expect_400=False):
    """want: None = 思考なし、int = その effort。fn(headers) は (info dict) を返す。"""
    rec = {"id": cid, "api": api, "sent": sent, "want": want, "expect_400": expect_400}
    t0 = time.time()
    try:
        rec["info"] = fn(ua(cid))
        rec["status"] = 200
    except (openai.APIStatusError, anthropic.APIStatusError) as e:
        rec["status"] = e.status_code
        body = getattr(e, "body", None)
        rec["err"] = body if isinstance(body, (dict, list, str)) else str(e)
    except Exception as e:
        rec["status"] = -1
        rec["err"] = f"{type(e).__name__}: {e}"[:500]
    rec["sec"] = round(time.time() - t0, 2)
    cases.append(rec)
    print(cid, api, repr(sent), "->", rec["status"], json.dumps(rec.get("info", rec.get("err")), ensure_ascii=False)[:200], flush=True)


def chat_info(r):
    m = r.choices[0].message
    rc = getattr(m, "reasoning_content", None) or (m.model_extra or {}).get("reasoning_content")
    return {"reasoning": bool(rc), "content": m.content, "usage_in": r.usage.prompt_tokens if r.usage else None}


def chat_stream(h, **kw):
    rc, txt = "", ""
    for ch in OA.chat.completions.create(model=MODEL, max_tokens=48, messages=MSG, stream=True, extra_headers=h, **kw):
        if ch.choices:
            d = ch.choices[0].delta
            rc += (d.model_extra or {}).get("reasoning_content") or ""
            txt += d.content or ""
    return {"reasoning": bool(rc), "content": txt}


def resp_info(r):
    return {"reasoning": "reasoning" in [o.type for o in r.output], "echo": r.reasoning.effort if r.reasoning else None, "text": r.output_text}


def msg_info(m):
    return {"reasoning": "thinking" in [b.type for b in m.content], "usage_in": m.usage.input_tokens + (m.usage.cache_read_input_tokens or 0)}


# ------------------------------------------------------------ Chat: reasoning_effort
for i, (v, want) in enumerate(VALID):
    run(f"chat-{i}", "chat.reasoning_effort", v, want,
        lambda h, v=v: chat_info(OA.chat.completions.create(model=MODEL, max_tokens=48, messages=MSG, reasoning_effort=v, extra_headers=h)))
run("chat-unspec", "chat.reasoning_effort", "(未指定)", 75,
    lambda h: chat_info(OA.chat.completions.create(model=MODEL, max_tokens=48, messages=MSG, extra_headers=h)))
run("chat-stream-37", "chat.reasoning_effort(stream)", 37, 37, lambda h: chat_stream(h, reasoning_effort=37))
run("chat-thinking-enabled-37", "chat.thinking+reasoning_effort", 37, 37,
    lambda h: chat_info(OA.chat.completions.create(model=MODEL, max_tokens=48, messages=MSG, reasoning_effort=37, extra_body={"thinking": {"type": "enabled"}}, extra_headers=h)))
run("chat-thinking-disabled-60", "chat.thinking disabled+reasoning_effort", 60, None,
    lambda h: chat_info(OA.chat.completions.create(model=MODEL, max_tokens=48, messages=MSG, reasoning_effort=60, extra_body={"thinking": {"type": "disabled"}}, extra_headers=h)))
for i, v in enumerate(INVALID):
    run(f"chat-bad-{i}", "chat.reasoning_effort", v, "400",
        lambda h, v=v: chat_info(OA.chat.completions.create(model=MODEL, max_tokens=8, messages=MSG, reasoning_effort=v, extra_headers=h)), True)

# ------------------------------------------------------------ Chat: chat_template_kwargs.reasoning_effort
for i, (v, want) in enumerate(VALID):
    run(f"ctk-{i}", "chat.chat_template_kwargs.reasoning_effort", v, want,
        lambda h, v=v: chat_info(OA.chat.completions.create(model=MODEL, max_tokens=48, messages=MSG, extra_body={"chat_template_kwargs": {"reasoning_effort": v}}, extra_headers=h)))
run("ctk-enable-false-60", "chat.ctk enable_thinking=false+reasoning_effort", 60, None,
    lambda h: chat_info(OA.chat.completions.create(model=MODEL, max_tokens=48, messages=MSG, extra_body={"chat_template_kwargs": {"reasoning_effort": 60, "enable_thinking": False}}, extra_headers=h)))
run("ctk-top-wins", "chat.reasoning_effort 37 + ctk 90", 37, 37,
    lambda h: chat_info(OA.chat.completions.create(model=MODEL, max_tokens=48, messages=MSG, reasoning_effort=37, extra_body={"chat_template_kwargs": {"reasoning_effort": 90}}, extra_headers=h)))
for i, v in enumerate([101, "abc", -1]):
    run(f"ctk-bad-{i}", "chat.chat_template_kwargs.reasoning_effort", v, "400",
        lambda h, v=v: chat_info(OA.chat.completions.create(model=MODEL, max_tokens=8, messages=MSG, extra_body={"chat_template_kwargs": {"reasoning_effort": v}}, extra_headers=h)), True)

# ------------------------------------------------------------ Responses: reasoning.effort
for i, (v, want) in enumerate(VALID):
    run(f"resp-{i}", "responses.reasoning.effort", v, want,
        lambda h, v=v: resp_info(OA.responses.create(model=MODEL, input="Say ok.", max_output_tokens=48, reasoning={"effort": v}, extra_headers=h)))
run("resp-unspec", "responses.reasoning.effort", "(未指定)", 75,
    lambda h: resp_info(OA.responses.create(model=MODEL, input="Say ok.", max_output_tokens=48, extra_headers=h)))
run("resp-summary-only", "responses.reasoning={summary:auto}", "(未指定)", 75,
    lambda h: resp_info(OA.responses.create(model=MODEL, input="Say ok.", max_output_tokens=48, reasoning={"summary": "auto"}, extra_headers=h)))


def resp_stream(h):
    evs = []
    with OA.responses.stream(model=MODEL, input="Say ok.", max_output_tokens=48, reasoning={"effort": 37}, extra_headers=h) as s:
        for e in s:
            evs.append(e.type)
        f = s.get_final_response()
    return {"reasoning": "reasoning" in [o.type for o in f.output], "echo": f.reasoning.effort, "events": len(evs)}


run("resp-stream-37", "responses.reasoning.effort(stream)", 37, 37, resp_stream)
for i, v in enumerate(INVALID):
    run(f"resp-bad-{i}", "responses.reasoning.effort", v, "400",
        lambda h, v=v: resp_info(OA.responses.create(model=MODEL, input="x", max_output_tokens=16, reasoning={"effort": v}, extra_headers=h)), True)

# ------------------------------------------------------------ Messages: output_config.effort / thinking.budget_tokens
for i, (v, want) in enumerate(VALID):
    run(f"msg-{i}", "messages.output_config.effort", v, want,
        lambda h, v=v: msg_info(AN.messages.create(model="claude-x", max_tokens=48, messages=MSG, output_config={"effort": v}, extra_headers=h)))
run("msg-unspec", "messages.output_config.effort", "(未指定)", 75,
    lambda h: msg_info(AN.messages.create(model="claude-x", max_tokens=48, messages=MSG, extra_headers=h)))
for b, want in [(1024, 50), (8191, 50), (8192, 75), (32767, 75), (32768, 100)]:
    run(f"msg-budget-{b}", "messages.thinking.budget_tokens", b, want,
        lambda h, b=b: msg_info(AN.messages.create(model="claude-x", max_tokens=48, messages=MSG, thinking={"type": "enabled", "budget_tokens": b}, extra_headers=h)))
run("msg-budget-str", "messages.thinking.budget_tokens(文字列)", "40000", 100,
    lambda h: msg_info(AN.messages.create(model="claude-x", max_tokens=48, messages=MSG, thinking={"type": "enabled", "budget_tokens": "40000"}, extra_headers=h)))
for i, b in enumerate([-1, "abc", 2048.5]):
    run(f"msg-budget-bad-{i}", "messages.thinking.budget_tokens", b, "400",
        lambda h, b=b: msg_info(AN.messages.create(model="claude-x", max_tokens=8, messages=MSG, thinking={"type": "enabled", "budget_tokens": b}, extra_headers=h)), True)
run("msg-budget+effort", "messages.budget 2048 + output_config max", "max", 100,
    lambda h: msg_info(AN.messages.create(model="claude-x", max_tokens=48, messages=MSG, thinking={"type": "enabled", "budget_tokens": 2048}, output_config={"effort": "max"}, extra_headers=h)))
run("msg-adaptive-37", "messages.thinking adaptive + effort 37", 37, 37,
    lambda h: msg_info(AN.messages.create(model="claude-x", max_tokens=48, messages=MSG, thinking={"type": "adaptive"}, output_config={"effort": 37}, extra_headers=h)))
run("msg-adaptive", "messages.thinking adaptive", "(未指定)", 75,
    lambda h: msg_info(AN.messages.create(model="claude-x", max_tokens=48, messages=MSG, thinking={"type": "adaptive"}, extra_headers=h)))
run("msg-disabled-60", "messages.thinking disabled + effort 60", 60, None,
    lambda h: msg_info(AN.messages.create(model="claude-x", max_tokens=48, messages=MSG, thinking={"type": "disabled"}, output_config={"effort": 60}, extra_headers=h)))
run("msg-extra-body-37", "messages.extra_body output_config", 37, 37,
    lambda h: msg_info(AN.messages.create(model="claude-x", max_tokens=48, messages=MSG, extra_body={"output_config": {"effort": 37}}, extra_headers=h)))


def msg_stream(h):
    kinds = []
    with AN.messages.stream(model="claude-x", max_tokens=48, messages=MSG, output_config={"effort": 37}, extra_headers=h) as s:
        for e in s:
            kinds.append(e.type)
        f = s.get_final_message()
    return {"reasoning": "thinking" in [b.type for b in f.content], "usage_in": f.usage.input_tokens}


run("msg-stream-37", "messages.output_config.effort(stream)", 37, 37, msg_stream)
for i, v in enumerate(INVALID):
    run(f"msg-bad-{i}", "messages.output_config.effort", v, "400",
        lambda h, v=v: msg_info(AN.messages.create(model="claude-x", max_tokens=8, messages=MSG, output_config={"effort": v}, extra_headers=h)), True)

# ------------------------------------------------------------ legacy completions
for i, (v, want) in enumerate(VALID):
    run(f"cmpl-{i}", "completions.reasoning_effort", v, want,
        lambda h, v=v: {"text": OA.completions.create(model=MODEL, prompt="Say ok.", max_tokens=32, extra_body={"reasoning_effort": v}, extra_headers=h).choices[0].text})
run("cmpl-unspec", "completions.reasoning_effort", "(未指定)", None,
    lambda h: {"text": OA.completions.create(model=MODEL, prompt="Say ok.", max_tokens=16, extra_headers=h).choices[0].text})
run("cmpl-ctk-37", "completions.chat_template_kwargs.reasoning_effort", 37, 37,
    lambda h: {"text": OA.completions.create(model=MODEL, prompt="Say ok.", max_tokens=32, extra_body={"chat_template_kwargs": {"reasoning_effort": 37}}, extra_headers=h).choices[0].text})
for i, v in enumerate([101, "abc", -1, True]):
    run(f"cmpl-bad-{i}", "completions.reasoning_effort", v, "400",
        lambda h, v=v: {"text": OA.completions.create(model=MODEL, prompt="x", max_tokens=8, extra_body={"reasoning_effort": v}, extra_headers=h).choices[0].text}, True)

# ------------------------------------------------------------ count_tokens と実際の input_tokens
ct = []
for v in [1, 37, 60, 100, "60", "max", "low", "none", None]:
    kw = {"output_config": {"effort": v}} if v is not None else {}
    n = AN.messages.count_tokens(model="claude-x", messages=MSG, **kw).input_tokens
    m = AN.messages.create(model="claude-x", max_tokens=16, messages=MSG, **kw)
    a = m.usage.input_tokens + (m.usage.cache_read_input_tokens or 0) + (m.usage.cache_creation_input_tokens or 0)
    ct.append({"effort": v, "count_tokens": n, "actual": a, "match": n == a})
    print("count_tokens", v, n, a, flush=True)
for b in [2048, 40000]:
    kw = {"thinking": {"type": "enabled", "budget_tokens": b}}
    n = AN.messages.count_tokens(model="claude-x", messages=MSG, **kw).input_tokens
    m = AN.messages.create(model="claude-x", max_tokens=16, messages=MSG, **kw)
    a = m.usage.input_tokens + (m.usage.cache_read_input_tokens or 0) + (m.usage.cache_creation_input_tokens or 0)
    ct.append({"budget": b, "count_tokens": n, "actual": a, "match": n == a})
    print("count_tokens budget", b, n, a, flush=True)

# ------------------------------------------------------------ 要求ログで上流本文を確かめる
time.sleep(1.5)
logs = {}
for f in sorted(glob.glob(LOGDIR + "/*.jsonl"))[-2:]:
    for line in open(f, encoding="utf-8"):
        if f"effort-verify/{RUN}/" not in line:
            continue
        d = json.loads(line)
        if d.get("event") == "request.received.openai":
            cid = d["user_agent"].rsplit("/", 1)[1]
            logs[cid] = d.get("obj")

bad = []
for c in cases:
    if c["expect_400"]:
        ok = c["status"] == 400
        c["upstream"] = "not sent" if not (logs.get(c["id"]) or {}).get("messages") else logs.get(c["id"])
        ok = ok and c["upstream"] == "not sent"
    else:
        obj = logs.get(c["id"])
        if obj is None:
            c["upstream"] = "NO LOG"
            ok = False
        else:
            th = (obj.get("thinking") or {}).get("type")
            re_ = obj.get("reasoning_effort", "<absent>")
            c["upstream"] = {"thinking": th, "reasoning_effort": re_}
            if c["want"] is None:
                ok = th == "disabled" and re_ == "<absent>"
            else:
                ok = th == "enabled" and type(re_) is int and re_ == c["want"]
        ok = ok and c["status"] == 200
        info = c.get("info") or {}
        if ok and "reasoning" in info and c["want"] is not None and not c["api"].endswith("(stream)"):
            pass  # max_tokens が小さいので思考の有無は参考 (思考なしで ok が出ることはない)
        if ok and "reasoning" in info and c["want"] is None:
            ok = info["reasoning"] is False
        if ok and c["api"].startswith("responses") and "echo" in info:
            exp = c["sent"] if c["sent"] != "(未指定)" else "high"
            ok = info["echo"] == exp
            if not ok:
                c["echo_mismatch"] = [info["echo"], exp]
    c["ok"] = ok
    if not ok:
        bad.append(c)

summary = {"run": RUN, "total": len(cases), "ok": len(cases) - len(bad), "count_tokens": ct,
           "count_tokens_all_match": all(x["match"] for x in ct), "bad": bad, "cases": cases}
json.dump(summary, open(OUT, "w"), ensure_ascii=False, indent=1, default=str)
print(f"\n{summary['ok']}/{summary['total']} ok; count_tokens match: {summary['count_tokens_all_match']}")
for c in bad:
    print("BAD", c["id"], c["api"], repr(c["sent"]), "want", c["want"], "status", c["status"], "upstream", c.get("upstream"), c.get("err", "")[:300] if isinstance(c.get("err"), str) else json.dumps(c.get("err"))[:300], c.get("echo_mismatch", ""))
sys.exit(0 if not bad and summary["count_tokens_all_match"] else 1)
