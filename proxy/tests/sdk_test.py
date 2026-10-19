#!/usr/bin/env python3
# ds41gw の実機試験 (r7_proxy_spec.md §16.3)。公式 openai / anthropic SDK で、ds41rt 経由の小さな要求を逐次に送る。
# 使い方: tests/.venv/bin/python tests/sdk_test.py [--url http://127.0.0.1:30001] [--only T-C1,T-R2] [--conc 1]
# 制約: ds41rt への同時数は 4 本以内 (既定は逐次)。max_tokens は小さく (思考ありは effort low か ≤ 512)。
import argparse, base64, json, struct, sys, threading, time, traceback, warnings, zlib
from concurrent.futures import ThreadPoolExecutor

import anthropic
import openai

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30001")
ap.add_argument("--only", default="")
ap.add_argument("--conc", type=int, default=1)
ap.add_argument("--out", default="")
A = ap.parse_args()
warnings.filterwarnings("ignore")  # SDK 自身の pydantic 直列化の警告 (Responses の union 型) を黙らせる
assert A.conc <= 4

OA = openai.OpenAI(base_url=A.url + "/v1", api_key="none", max_retries=0, timeout=300)
AN = anthropic.Anthropic(base_url=A.url, api_key="none", max_retries=0, timeout=300)
MODEL = "GLM-5.3-Flash"
TESTS = []
LOG = threading.Lock()


def test(tid):
    def deco(f):
        TESTS.append((tid, f))
        return f
    return deco


def png(r, g, b, w=32, h=32):
    raw = b"".join(b"\x00" + bytes([r, g, b]) * w for _ in range(h))
    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")
    return base64.b64encode(data).decode()


RED, BLUE = png(230, 20, 20), png(20, 20, 230)
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def check_resp_events(evs):
    seq = 0; open_item = None; open_part = None; terminal = 0
    for e in evs:
        assert e["sequence_number"] == seq, f"seq {e['sequence_number']} != {seq} ({e['type']})"
        seq += 1
        assert terminal == 0, "event after terminal"
        t = e["type"]
        if t == "response.output_item.added":
            assert open_item is None; open_item = e["output_index"]
        elif t == "response.output_item.done":
            assert open_item == e["output_index"] and open_part is None; open_item = None
        elif t in ("response.content_part.added", "response.reasoning_summary_part.added"):
            assert open_part is None and open_item == e["output_index"]; open_part = e["output_index"]
        elif t in ("response.content_part.done", "response.reasoning_summary_part.done"):
            assert open_part == e["output_index"]; open_part = None
        elif t in ("response.completed", "response.incomplete", "response.failed"):
            terminal += 1
    assert terminal == 1, f"{terminal} terminal events"


# ---------------------------------------------------------------- Chat
@test("T-C1")
def t_c1():
    r = OA.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Reply with the single word: pong"}],
                                   max_tokens=200, reasoning_effort="low")
    assert r.model == MODEL and r.choices[0].message.content.strip(), r
    text = ""
    with OA.chat.completions.create(model=MODEL, stream=True, max_tokens=200, reasoning_effort="low",
                                    messages=[{"role": "user", "content": "Count 1 to 5 separated by spaces."}]) as s:
        for ch in s:
            if ch.choices and ch.choices[0].delta.content:
                text += ch.choices[0].delta.content
    assert "3" in text, text
    return {"nonstream": r.choices[0].message.content, "stream": text}


@test("T-C2")
def t_c2():
    msgs = [{"role": "user", "content": "What is the weather in Tokyo and in Osaka? Call get_weather for both cities in parallel."}]
    raw = OA.chat.completions.with_raw_response.create(model=MODEL, messages=msgs, tools=[WEATHER], tool_choice="required",
                                                       max_tokens=512, reasoning_effort="low")
    r = raw.parse()
    m = r.choices[0].message
    assert m.tool_calls, m
    msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [tc.model_dump() for tc in m.tool_calls]})  # 思考は返さない
    for tc in m.tool_calls:
        city = json.loads(tc.function.arguments).get("city", "?")
        msgs.append({"role": "tool", "tool_call_id": tc.id, "content": f"{city}: sunny, 21C"})
    raw2 = OA.chat.completions.with_raw_response.create(model=MODEL, messages=msgs, tools=[WEATHER], max_tokens=300, reasoning_effort="low")
    r2 = raw2.parse()
    assert r2.choices[0].message.content, r2
    return {"n_calls": len(m.tool_calls), "rid2": raw2.headers.get("x-request-id"),
            "cache_hit_2": r2.usage.model_extra.get("prompt_cache_hit_tokens"), "answer": r2.choices[0].message.content[:120]}


@test("T-C3")
def t_c3():
    r = OA.chat.completions.create(model=MODEL, max_tokens=300, reasoning_effort="low", messages=[{"role": "user", "content": [
        {"type": "text", "text": "What colors are these two images? Answer briefly."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + RED}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + BLUE}}]}])
    c1 = r.choices[0].message.content
    # tool メッセージの画像
    r2 = OA.chat.completions.create(model=MODEL, max_tokens=300, reasoning_effort="low", tools=[{"type": "function", "function": {"name": "screenshot", "parameters": {"type": "object", "properties": {}}}}],
        messages=[{"role": "user", "content": "Take a screenshot and tell me its color."},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}]},
                  {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "screenshot:"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + RED}}]}])
    c2 = r2.choices[0].message.content
    assert "red" in (c1 + c2).lower(), (c1, c2)
    return {"two_images": c1, "tool_image": c2}


@test("T-C4")
def t_c4():
    schema = {"type": "object", "properties": {"capital": {"type": "string"}, "population_millions": {"type": "number"}},
              "required": ["capital", "population_millions"], "additionalProperties": False}
    r = OA.chat.completions.create(model=MODEL, max_tokens=400, reasoning_effort="low",
        response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema, "strict": True}},
        messages=[{"role": "user", "content": "Capital of Japan and its population in millions, as JSON."}])
    v = json.loads(r.choices[0].message.content)
    assert set(v) == {"capital", "population_millions"}, v
    return v


@test("T-C5")
def t_c5():
    msgs = [{"role": "user", "content": "Check the weather in Tokyo."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_x1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]},
            {"role": "tool", "tool_call_id": "call_x1", "content": "Tokyo: rainy, 15C"},
            {"role": "user", "content": "Now summarize the result in one sentence. Do not call tools."}]
    raw = OA.chat.completions.with_raw_response.create(model=MODEL, messages=msgs, tools=[WEATHER], tool_choice="none", max_tokens=300, reasoning_effort="low")
    r = raw.parse()
    c = r.choices[0].message.content
    assert "DSML" not in c and "｜" not in c, c
    assert not r.choices[0].message.tool_calls
    return {"content": c, "warning": raw.headers.get("x-ds41gw-warning")}


@test("T-C6")
def t_c6():
    chunks = list(OA.chat.completions.create(model=MODEL, stream=True, stream_options={"include_usage": True}, max_tokens=200,
                                             reasoning_effort="low", messages=[{"role": "user", "content": "Say hello."}]))
    last = chunks[-1]
    assert last.choices == [] and last.usage, last
    fin = [c for c in chunks if c.choices and c.choices[0].finish_reason]
    assert fin and fin[0].usage is None
    rt = last.usage.completion_tokens_details.reasoning_tokens
    assert rt is not None and rt > 0, last.usage
    return {"usage": last.usage.model_dump()}


@test("T-C7")
def t_c7():
    r = OA.chat.completions.create(model=MODEL, n=2, max_tokens=60, reasoning_effort="none", temperature=1.0,
                                   messages=[{"role": "user", "content": "Give one random fruit name."}])
    assert len(r.choices) == 2 and r.choices[1].index == 1
    try:
        OA.chat.completions.create(model=MODEL, logprobs=True, max_tokens=5, messages=[{"role": "user", "content": "x"}])
        raise AssertionError("logprobs accepted")
    except openai.BadRequestError as e:
        assert e.status_code == 400
    r3 = OA.chat.completions.create(model=MODEL, max_tokens=60, extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                                    messages=[{"role": "user", "content": "Say ok."}])
    assert not (r3.choices[0].message.model_extra or {}).get("reasoning_content"), r3
    return {"n2": [c.message.content for c in r.choices], "no_think": r3.choices[0].message.content}


# ---------------------------------------------------------------- Responses
@test("T-R1")
def t_r1():
    r = OA.responses.create(model=MODEL, instructions="Answer in exactly one word.", input="What color is the sky on a clear day?",
                            max_output_tokens=300, reasoning={"effort": "low"})
    assert r.status == "completed" and r.output_text.strip(), r
    assert r.usage.output_tokens_details.reasoning_tokens > 0
    return {"text": r.output_text, "usage": r.usage.model_dump()}


@test("T-R2")
def t_r2():
    events = []
    with OA.responses.stream(model=MODEL, input="Write a haiku about rain.", max_output_tokens=512, reasoning={"effort": "low"}) as s:
        for e in s:
            events.append(e.model_dump())
    check_resp_events(events)
    # temperature 1 なので、思考が max_output_tokens を使い切ることがある (effort low = 50 でも時々 400 を超える)。
    # そのときは response.incomplete (reason max_output_tokens) で終わるのが正しい。SDK の get_final_response は completed しか受けない
    last = events[-1]
    if last["type"] == "response.incomplete":
        r = last["response"]
        assert r["status"] == "incomplete" and r["incomplete_details"]["reason"] == "max_output_tokens", r.get("incomplete_details")
        assert any(o["type"] == "reasoning" for o in r["output"]), [o["type"] for o in r["output"]]
        return {"events": len(events), "incomplete": "max_output_tokens"}
    assert last["type"] == "response.completed", last["type"]
    final = s.get_final_response()
    assert final.output_text.strip()
    return {"events": len(events), "text": final.output_text}


@test("T-R3")
def t_r3():
    tools = [{"type": "function", "name": "get_weather", "description": "Get the current weather for a city.",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
    inp = [{"role": "user", "content": "Get the weather for Tokyo and Osaka (call get_weather twice in parallel)."}]
    r = OA.responses.create(model=MODEL, input=inp, tools=tools, tool_choice="required", store=False,
                            include=["reasoning.encrypted_content"], max_output_tokens=512, reasoning={"effort": "low"})
    calls = [o for o in r.output if o.type == "function_call"]
    assert calls, r.output
    for o in r.output:
        d = o.model_dump(exclude_none=True)
        if d["type"] == "reasoning":
            assert d.get("encrypted_content", "").startswith("ds41gw1.")
            d.pop("content", None)  # 本文を削って encrypted_content だけで返す
            d.pop("status", None)
        inp.append(d)
    for c in calls:
        inp.append({"type": "function_call_output", "call_id": c.call_id, "output": f"{json.loads(c.arguments).get('city')}: sunny"})
    r2 = OA.responses.create(model=MODEL, input=inp, tools=tools, store=False, max_output_tokens=300, reasoning={"effort": "low"})
    assert r2.output_text.strip(), r2
    return {"calls": len(calls), "answer": r2.output_text[:120]}


@test("T-R4")
def t_r4():
    r1 = OA.responses.create(model=MODEL, input="My name is Hanako. Just say OK.", max_output_tokens=200, reasoning={"effort": "low"})
    r2 = OA.responses.create(model=MODEL, previous_response_id=r1.id, input="I live in Kyoto. Just say OK.", max_output_tokens=200, reasoning={"effort": "low"})
    r3 = OA.responses.create(model=MODEL, previous_response_id=r2.id, input="What is my name and city? Answer briefly.", max_output_tokens=300, reasoning={"effort": "low"})
    t = r3.output_text
    assert "Hanako" in t and "Kyoto" in t, t
    g = OA.responses.retrieve(r2.id)
    assert g.id == r2.id
    items = OA.responses.input_items.list(r2.id)
    assert items.data
    OA.responses.delete(r1.id)
    try:
        OA.responses.create(model=MODEL, previous_response_id=r2.id, input="x", max_output_tokens=20)
        raise AssertionError("deleted parent accepted")
    except openai.BadRequestError as e:
        assert e.status_code == 400
    return {"answer": t}


@test("T-R5")
def t_r5():
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": False}
    r = OA.responses.create(model=MODEL, input="What is 6*7? Respond as JSON.", max_output_tokens=300, reasoning={"effort": "low"},
                            text={"format": {"type": "json_schema", "name": "ans", "schema": schema, "strict": True}})
    assert json.loads(r.output_text)["answer"] == 42, r.output_text
    out = {}
    for eff in ["none", "low", "xhigh"]:
        rr = OA.responses.create(model=MODEL, input="Say hi.", max_output_tokens=300 if eff != "xhigh" else 512, reasoning={"effort": eff})
        out[eff] = [o.type for o in rr.output]
    assert "reasoning" not in out["none"] and "reasoning" in out["low"]
    types = []
    with OA.responses.stream(model=MODEL, input="Say hi.", max_output_tokens=300, reasoning={"effort": "low", "summary": "auto"}) as s:
        for e in s:
            types.append(e.type)
    assert "response.reasoning_summary_text.delta" in types and "response.reasoning_text.delta" not in types, types
    return {"json": r.output_text, "efforts": out}


@test("T-R6")
def t_r6():
    for kw in [dict(tools=[{"type": "web_search"}]), dict(background=True)]:
        try:
            OA.responses.create(model=MODEL, input="x", max_output_tokens=10, **kw)
            raise AssertionError(f"{kw} accepted")
        except openai.BadRequestError as e:
            assert e.status_code == 400
    return {}


@test("T-R7")
def t_r7():
    r = OA.responses.create(model=MODEL, max_output_tokens=300, reasoning={"effort": "low"}, input=[{"role": "user", "content": [
        {"type": "input_text", "text": "What color is this image? One word."},
        {"type": "input_image", "image_url": "data:image/png;base64," + BLUE}]}])
    tools = [{"type": "function", "name": "screenshot", "parameters": {"type": "object", "properties": {}}}]
    r2 = OA.responses.create(model=MODEL, max_output_tokens=300, reasoning={"effort": "low"}, tools=tools, input=[
        {"role": "user", "content": "Take a screenshot and tell me its color in one word."},
        {"type": "function_call", "call_id": "c1", "name": "screenshot", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": "screenshot:"},
                                                                     {"type": "input_image", "image_url": "data:image/png;base64," + RED}]}])
    assert "blue" in r.output_text.lower() and "red" in r2.output_text.lower(), (r.output_text, r2.output_text)
    return {"input_image": r.output_text, "fco_image": r2.output_text}


# ---------------------------------------------------------------- Anthropic
@test("T-A1")
def t_a1():
    m = AN.messages.create(model="claude-sonnet-4-5", max_tokens=300, system="You answer in one short sentence.",
                           messages=[{"role": "user", "content": "What is the capital of France?"}])
    assert m.stop_reason == "end_turn" and any(b.type == "text" for b in m.content), m
    return {"text": [b.text for b in m.content if b.type == "text"][0]}


@test("T-A2")
def t_a2():
    with AN.messages.stream(model="claude-x", max_tokens=600, thinking={"type": "enabled", "budget_tokens": 2048},
                            messages=[{"role": "user", "content": "What is 17*3?"}]) as s:
        for _ in s:
            pass
        m = s.get_final_message()
    th = [b for b in m.content if b.type == "thinking"]
    assert th and th[0].signature and th[0].thinking, m
    assert any(b.type == "text" and "51" in b.text for b in m.content), m
    return {"signature": th[0].signature[:24]}


@test("T-A3")
def t_a3():
    tools = [{"name": "get_weather", "description": "Get the weather for a city.", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
    msgs = [{"role": "user", "content": "Get the weather for Tokyo and Osaka with two parallel tool calls."}]
    m = AN.messages.create(model="claude-x", max_tokens=600, tools=tools, tool_choice={"type": "any"}, messages=msgs,
                           thinking={"type": "enabled", "budget_tokens": 1024})
    uses = [b for b in m.content if b.type == "tool_use"]
    assert uses and m.stop_reason == "tool_use", m
    msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in m.content]})
    results = []
    for i, u in enumerate(uses):
        results.append({"type": "tool_result", "tool_use_id": u.id, "content": "sunny" if i == 0 else "station offline", "is_error": i == 1})
    msgs.append({"role": "user", "content": results})
    m2 = AN.messages.create(model="claude-x", max_tokens=400, tools=tools, messages=msgs, thinking={"type": "enabled", "budget_tokens": 1024})
    # 2 ターン目は回答でも、失敗した呼び出しの再試行 (tool_use) でもよい。往復が 200 で通ることを見る
    assert m2.stop_reason in ("end_turn", "tool_use") and m2.content, m2
    return {"uses": len(uses), "stop_reason_2": m2.stop_reason, "blocks_2": [b.type for b in m2.content]}


@test("T-A4")
def t_a4():
    m = AN.messages.create(model="claude-x", max_tokens=300, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": RED}},
        {"type": "text", "text": "What color is this image? One word."}]}], thinking={"type": "enabled", "budget_tokens": 1024})
    t = [b.text for b in m.content if b.type == "text"][0]
    assert "red" in t.lower(), t
    return {"text": t}


@test("T-A5")
def t_a5():
    m = AN.messages.create(model="claude-x", max_tokens=300, stop_sequences=["7"], thinking={"type": "disabled"},
                           messages=[{"role": "user", "content": "Count from 1 to 10, separated by commas."}])
    assert m.stop_reason == "stop_sequence" and m.stop_sequence == "7", m
    t = m.content[-1].text
    assert "7" not in t and "6" in t, t
    return {"text": t}


@test("T-A6")
def t_a6():
    kw = dict(model="claude-x", system="You are a terse assistant.", messages=[{"role": "user", "content": "Explain TCP slow start in two sentences."}])
    n = AN.messages.count_tokens(**kw).input_tokens
    m = AN.messages.create(max_tokens=200, thinking={"type": "enabled", "budget_tokens": 1024}, **kw)
    actual = m.usage.input_tokens + (m.usage.cache_read_input_tokens or 0)
    diff = abs(n - actual) / actual
    assert diff <= 0.05, (n, actual)
    return {"count_tokens": n, "actual": actual}


@test("T-A9")
def t_a9():
    r = OA.chat.completions.create(model=MODEL, max_tokens=120, reasoning_effort="low", extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                                   messages=[{"role": "user", "content": "Summarize in one line: the meeting moved from Monday to Wednesday because the room was booked."}])
    s = OA.responses.create(model=MODEL, input="Generate a short title for a chat about fixing a Rust borrow checker error.", max_output_tokens=64, reasoning={"effort": "high"})
    return {"summary_content": r.choices[0].message.content, "summary_finish": r.choices[0].finish_reason,
            "title_text": s.output_text, "title_status": s.status}


# ---------------------------------------------------------------- その他
@test("T-E1")
def t_e1():
    """reasoning effort の数値 (1〜100、数値文字列) と段階名を 3 API で。0 は思考なし、101 は 400 invalid_reasoning_effort。"""
    out = {}
    for eff in [60, "60", "high", "max", 0]:
        think = eff not in (0,)
        r = OA.chat.completions.create(model=MODEL, max_tokens=64, reasoning_effort=eff, messages=[{"role": "user", "content": "Say ok."}])
        rc = getattr(r.choices[0].message, "reasoning_content", None) or (r.model_extra or {}).get("reasoning_content")
        rc = rc or (r.choices[0].message.model_extra or {}).get("reasoning_content")
        assert bool(rc) == think, (eff, rc)
        rr = OA.responses.create(model=MODEL, input="Say ok.", max_output_tokens=64, reasoning={"effort": eff})
        assert ("reasoning" in [o.type for o in rr.output]) == think, (eff, rr.output)
        assert rr.reasoning.effort == eff, (eff, rr.reasoning.effort)
        m = AN.messages.create(model="claude-x", max_tokens=64, messages=[{"role": "user", "content": "Say ok."}], extra_body={"output_config": {"effort": eff}})
        assert ("thinking" in [b.type for b in m.content]) == think, (eff, m.content)
        out[str(eff)] = "ok"
    try:
        OA.chat.completions.create(model=MODEL, max_tokens=5, reasoning_effort=101, messages=[{"role": "user", "content": "x"}])
        raise AssertionError("101 was accepted")
    except openai.BadRequestError as e:
        assert e.code == "invalid_reasoning_effort" and e.param == "reasoning_effort", (e.code, e.param)
    try:
        OA.responses.create(model=MODEL, input="x", max_output_tokens=16, reasoning={"effort": 101})
        raise AssertionError("101 was accepted")
    except openai.BadRequestError as e:
        assert e.code == "invalid_reasoning_effort" and e.param == "reasoning.effort", (e.code, e.param)
    try:
        AN.messages.create(model="claude-x", max_tokens=5, messages=[{"role": "user", "content": "x"}], extra_body={"output_config": {"effort": 101}})
        raise AssertionError("101 was accepted")
    except anthropic.BadRequestError as e:
        assert "output_config.effort" in str(e), str(e)
    out["101"] = "400"
    return out


@test("T-X1")
def t_x1():
    r = OA.completions.create(model=MODEL, prompt="The capital of Italy is", max_tokens=30)
    text = "".join(c.choices[0].text for c in OA.completions.create(model=MODEL, prompt="List three colors:", max_tokens=30, stream=True) if c.choices)
    assert r.choices[0].text.strip() and text.strip()
    return {"nonstream": r.choices[0].text, "stream": text}


@test("T-X2")
def t_x2():
    a = [m.id for m in OA.models.list()]
    b = [m.id for m in AN.models.list()]
    assert MODEL in a and MODEL in b and "deepseek-ai/DeepSeek-V4.1-Flash" in a
    return {"openai": a, "anthropic": b}


def run(tid, f):
    t0 = time.time()
    try:
        info = f()
        res = {"id": tid, "ok": True, "sec": round(time.time() - t0, 2), "info": info}
    except Exception as e:
        res = {"id": tid, "ok": False, "sec": round(time.time() - t0, 2), "error": f"{type(e).__name__}: {e}"[:800],
               "tb": traceback.format_exc()[-1500:]}
    with LOG:
        print(("PASS " if res["ok"] else "FAIL ") + tid + f" ({res['sec']}s) " + json.dumps(res.get("info", res.get("error")), ensure_ascii=False)[:300], flush=True)
    return res


def main():
    sel = [t for t in TESTS if not A.only or t[0] in A.only.split(",")]
    with ThreadPoolExecutor(A.conc) as ex:
        results = list(ex.map(lambda t: run(*t), sel))
    ok = sum(r["ok"] for r in results)
    print(f"\n{ok}/{len(results)} passed")
    if A.out:
        json.dump(results, open(A.out, "w"), ensure_ascii=False, indent=1)
    sys.exit(0 if ok == len(results) else 1)


main()
