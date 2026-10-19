#!/usr/bin/env python3
# ds41gw の SDK 適合試験。公式 openai / anthropic Python SDK でプロキシ (:30001) を叩き、
#   (1) SDK のパーサ・ストリーム補助 (responses.stream / chat.completions.stream / messages.stream / parse) が例外を出さないこと、
#   (2) 生の応答 (SSE の各イベント、非ストリームの本文) が SDK の型に pydantic で厳密に通ること、
#   (3) イベント列の順序・対応・連結結果が仕様 (r7_proxy_spec.md §5.4、§6.8、§7.4、OpenAI / Anthropic の公開仕様) どおりであること
# を確かめる。sdk_test.py (T-*) が機能を見るのに対し、こちら (K-*) は形と順序を見る。
# 使い方: tests/.venv/bin/python tests/sdk_conformance.py [--url http://127.0.0.1:30001] [--only K-C1,K-R2] [--out results.json]
# 制約: ds41rt への同時数は 4 本以内 (この試験は逐次)。max_tokens は小さく、思考は effort low。
import argparse, base64, json, os, re, struct, sys, threading, time, traceback, typing, warnings, zlib

import anthropic
import httpx2
import openai
import pydantic
from anthropic.types import Message as AMessage, RawMessageStreamEvent
from openai.types import Completion
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response, ResponseStreamEvent
from openai.types.responses.response_item_list import ResponseItemList

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30001")
ap.add_argument("--only", default="")
ap.add_argument("--out", default="")
A = ap.parse_args()
warnings.filterwarnings("ignore")  # SDK 自身の pydantic 直列化警告 (Responses の union 型)

# ---------------------------------------------------------------- 生の応答の記録 (SDK の HTTP 層に挟む)
CAP = []
CAP_LOCK = threading.Lock()


class _TeeStream(httpx2.SyncByteStream):
    def __init__(self, inner, rec):
        self.inner, self.rec = inner, rec

    def __iter__(self):
        for part in self.inner:
            self.rec["body"] += part
            yield part

    def close(self):
        self.rec["closed"] = True
        self.inner.close()


class TeeTransport(httpx2.HTTPTransport):
    def handle_request(self, request):
        resp = super().handle_request(request)
        rec = {"method": request.method, "url": str(request.url), "status": resp.status_code,
               "headers": dict(resp.headers), "body": bytearray(), "closed": False}
        with CAP_LOCK:
            CAP.append(rec)
        resp.stream = _TeeStream(resp.stream, rec)
        return resp


def cap_reset():
    with CAP_LOCK:
        CAP.clear()


def caps(path=None):
    with CAP_LOCK:
        return [c for c in CAP if path is None or path in c["url"]]


HTTP = httpx2.Client(transport=TeeTransport(), timeout=300)
OA = openai.OpenAI(base_url=A.url + "/v1", api_key="none", max_retries=0, timeout=300, http_client=HTTP)
AN = anthropic.Anthropic(base_url=A.url, api_key="none", max_retries=0, timeout=300, http_client=HTTP)
MODEL = "GLM-5.3-Flash"
AMODEL = "claude-sonnet-4-5"
TESTS = []
NOTES = threading.local()


def test(tid):
    def deco(f):
        TESTS.append((tid, f))
        return f
    return deco


def note(k, v):
    NOTES.d[k] = v


# ---------------------------------------------------------------- SSE と型の検査
def parse_sse(body: bytes):
    """[(event, data_str)]。コメント行 (: keep-alive) は捨てる。"""
    out = []
    text = body.decode("utf-8")
    for block in re.split(r"\r?\n\r?\n", text):
        ev, data = None, []
        for line in block.splitlines():
            if not line or line.startswith(":"):
                continue
            k, _, v = line.partition(":")
            v = v[1:] if v.startswith(" ") else v
            if k == "event":
                ev = v
            elif k == "data":
                data.append(v)
        if ev is not None or data:
            out.append((ev, "\n".join(data)))
    return out


RSE = pydantic.TypeAdapter(ResponseStreamEvent)
AMSE = pydantic.TypeAdapter(RawMessageStreamEvent)
# 各 Responses イベント型名 → SDK のクラス (SDK が推測で別の型に当てはめていないかを見る)
RESP_EVENT_CLASSES = {}
for cls in typing.get_args(typing.get_args(ResponseStreamEvent)[0]):
    lit = cls.model_fields["type"].annotation
    RESP_EVENT_CLASSES[typing.get_args(lit)[0]] = cls


def strict(model_or_adapter, obj, what):
    try:
        if isinstance(model_or_adapter, pydantic.TypeAdapter):
            return model_or_adapter.validate_python(obj)
        return model_or_adapter.model_validate(obj)
    except pydantic.ValidationError as e:
        raise AssertionError(f"{what}: SDK 型の検証に失敗: {str(e)[:600]}\nobj={json.dumps(obj, ensure_ascii=False)[:600]}")


def resp_events_from_cap(rec):
    evs = []
    assert rec["headers"].get("content-type", "").startswith("text/event-stream"), rec["headers"]
    for ev, data in parse_sse(bytes(rec["body"])):
        assert data != "[DONE]", "Responses の SSE に [DONE] がある (OpenAI は送らない)"
        obj = json.loads(data)
        assert ev == obj["type"], f"event: 行 {ev!r} と type {obj['type']!r} が違う"
        assert obj["type"] in RESP_EVENT_CLASSES, f"SDK が知らないイベント型 {obj['type']}"
        strict(RSE, obj, obj["type"])
        evs.append(obj)
    return evs


def check_response_obj(r: dict, what="response"):
    strict(Response, r, what)
    for k in ["id", "object", "created_at", "model", "status", "output", "tools", "tool_choice", "parallel_tool_calls"]:
        assert k in r, f"{what}: {k} が無い"
    assert r["object"] == "response" and r["id"].startswith("resp_"), r["id"]
    assert r["model"] == MODEL, r["model"]


def check_resp_stream(evs, expect_status="completed"):
    """Responses のイベント列の文法 (§6.8.1)。最終 response を返す。"""
    assert evs, "イベントが無い"
    for i, e in enumerate(evs):
        assert e["sequence_number"] == i, f"sequence_number {e['sequence_number']} != {i} ({e['type']})"
    assert evs[0]["type"] == "response.created", evs[0]["type"]
    assert evs[1]["type"] == "response.in_progress", evs[1]["type"]
    assert evs[0]["response"]["status"] == "in_progress" and evs[0]["response"]["output"] == []
    rid = evs[0]["response"]["id"]
    for e in evs[:2]:
        check_response_obj(e["response"], e["type"])
    terms = [e for e in evs if e["type"] in ("response.completed", "response.incomplete", "response.failed")]
    assert len(terms) == 1 and evs[-1] is terms[0], [e["type"] for e in terms]
    final = terms[0]["response"]
    check_response_obj(final, terms[0]["type"])
    assert final["id"] == rid and final["status"] == expect_status, (final["status"], expect_status)
    assert terms[0]["type"] == "response." + {"completed": "completed", "incomplete": "incomplete", "failed": "failed"}[expect_status]

    items, cur, part, buf = [], None, None, ""
    for e in evs[2:-1]:
        t = e["type"]
        if t == "response.output_item.added":
            assert cur is None, "項目が二重に開いている"
            assert e["output_index"] == len(items), (e["output_index"], len(items))
            cur = {"item": e["item"], "args": "", "parts": [], "summ": []}
            assert e["item"]["id"], e
            assert e["item"].get("status") in ("in_progress", None), e["item"]
            continue
        assert cur is not None or t.startswith("response.in_progress"), f"項目の外の {t}"
        if "output_index" in e:
            assert e["output_index"] == len(items), (t, e["output_index"], len(items))
        if "item_id" in e:
            assert e["item_id"] == cur["item"]["id"], (t, e["item_id"], cur["item"]["id"])
        if t in ("response.content_part.added", "response.reasoning_summary_part.added"):
            assert part is None
            idx = e.get("content_index", e.get("summary_index"))
            lst = cur["parts"] if t == "response.content_part.added" else cur["summ"]
            assert idx == len(lst), (t, idx)
            part = {"kind": t, "type": e["part"]["type"], "idx": idx}
            buf = ""
            want = {"message": ("output_text", "refusal"), "reasoning": ("reasoning_text", "summary_text")}[cur["item"]["type"]]
            assert part["type"] in want, (cur["item"]["type"], part["type"])
        elif t in ("response.output_text.delta", "response.reasoning_text.delta", "response.reasoning_summary_text.delta", "response.refusal.delta"):
            assert part is not None, f"{t} が part の外"
            ptype = {"response.output_text.delta": "output_text", "response.reasoning_text.delta": "reasoning_text",
                     "response.reasoning_summary_text.delta": "summary_text", "response.refusal.delta": "refusal"}[t]
            assert part["type"] == ptype, (t, part["type"])
            assert e.get("content_index", e.get("summary_index")) == part["idx"]
            buf += e["delta"]
        elif t in ("response.output_text.done", "response.reasoning_text.done", "response.reasoning_summary_text.done", "response.refusal.done"):
            key = "refusal" if t == "response.refusal.done" else "text"
            assert e[key] == buf, f"{t} の全文が delta の連結と違う"
        elif t in ("response.content_part.done", "response.reasoning_summary_part.done"):
            assert part is not None and part["kind"].replace(".added", "") == t.replace(".done", "")
            body = e["part"].get("text", e["part"].get("refusal"))
            assert body == buf, f"{t} の part 本文が delta の連結と違う"
            (cur["parts"] if t == "response.content_part.done" else cur["summ"]).append(e["part"])
            part = None
        elif t == "response.function_call_arguments.delta":
            assert cur["item"]["type"] == "function_call" and part is None
            cur["args"] += e["delta"]
        elif t == "response.function_call_arguments.done":
            assert e["arguments"] == cur["args"], "arguments.done が delta の連結と違う"
            json.loads(e["arguments"])
        elif t == "response.output_item.done":
            assert part is None, "part を閉じずに項目を閉じた"
            it = e["item"]
            assert it["id"] == cur["item"]["id"] and it["type"] == cur["item"]["type"]
            assert it["status"] in ("completed", "incomplete"), it.get("status")
            if it["type"] == "message":
                assert [p.get("text") for p in it["content"]] == [p.get("text") for p in cur["parts"]]
                assert it["role"] == "assistant"
            elif it["type"] == "function_call":
                assert it["arguments"] == cur["args"] and it["call_id"] and it["name"]
            elif it["type"] == "reasoning":
                summ = [p["text"] for p in it.get("summary") or []]
                assert summ == [p["text"] for p in cur["summ"]]
                content = [p["text"] for p in it.get("content") or []]
                # summary で流した場合、done の項目は content にも同じ本文を持つ (§6.8.2。二重に流してはいない)
                assert content == [p["text"] for p in cur["parts"]] or (not cur["parts"] and content == summ), (content, summ)
            items.append(it)
            cur = None
        else:
            raise AssertionError(f"想定外のイベント {t}")
    assert cur is None, "項目が閉じていない"
    assert json.dumps(final["output"], sort_keys=True) == json.dumps(items, sort_keys=True), "最終 output が output_item.done の列と違う"
    return final


def chat_chunks_from_cap(rec, n=1, usage_chunk=False):
    """Chat の SSE (§5.4.1)。各チャンクを SDK 型で検証し、本ごとに集約して返す。"""
    assert rec["headers"].get("content-type", "").startswith("text/event-stream"), rec["headers"]
    ev = parse_sse(bytes(rec["body"]))
    assert ev and ev[-1][1] == "[DONE]", "最後が [DONE] でない"
    assert all(d != "[DONE]" for _, d in ev[:-1])
    chunks = [json.loads(d) for _, d in ev[:-1]]
    ids = {c["id"] for c in chunks}
    assert len(ids) == 1, ids
    agg = {i: {"role": None, "content": "", "reasoning": "", "tools": {}, "finish": None, "order": []} for i in range(n)}
    usage = None
    for c in chunks:
        strict(ChatCompletionChunk, c, "chat.completion.chunk")
        assert c["object"] == "chat.completion.chunk" and c["model"] == MODEL, (c["object"], c["model"])
        if not c["choices"]:
            assert usage_chunk and c.get("usage") and usage is None, c
            usage = c["usage"]
            continue
        assert usage is None, "usage チャンクの後に choices がある"
        for ch in c["choices"]:
            a = agg[ch["index"]]
            assert a["finish"] is None, "finish_reason の後に差分がある"
            d = ch["delta"]
            if a["role"] is None:
                assert d.get("role") == "assistant", f"最初の差分に role:assistant が無い {d}"
                a["role"] = "assistant"
            if d.get("reasoning_content"):
                a["reasoning"] += d["reasoning_content"]
                a["order"].append("r")
            if d.get("content"):
                a["content"] += d["content"]
                a["order"].append("c")
            for tc in d.get("tool_calls") or []:
                a["order"].append("t")
                slot = a["tools"].get(tc["index"])
                if slot is None:
                    assert tc.get("id") and tc.get("type") == "function" and tc["function"].get("name"), f"tool_call の最初の差分に id/type/name が無い {tc}"
                    assert tc["index"] == len(a["tools"]), "tool_call の index が連番でない"
                    slot = a["tools"][tc["index"]] = {"id": tc["id"], "name": tc["function"]["name"], "arguments": ""}
                else:
                    assert not tc.get("id") or tc["id"] == slot["id"]
                slot["arguments"] += (tc.get("function") or {}).get("arguments") or ""
            if ch.get("finish_reason"):
                a["finish"] = ch["finish_reason"]
                if usage_chunk:
                    assert c.get("usage") is None, "include_usage なのに finish チャンクに usage がある"
    for i, a in agg.items():
        assert a["finish"], f"本 {i} に finish_reason が無い"
        if a["reasoning"] and "r" in a["order"]:
            first_other = next((k for k, x in enumerate(a["order"]) if x != "r"), len(a["order"]))
            assert all(x == "r" for x in a["order"][:first_other]) and "r" not in a["order"][first_other:], "思考が本文・ツールの後に来た"
        for t in a["tools"].values():
            json.loads(t["arguments"])
    if usage_chunk:
        assert usage, "include_usage なのに usage チャンクが無い"
    return agg, usage, chunks


def msg_events_from_cap(rec):
    """Anthropic の SSE (§7.4)。ping 以外を SDK 型で検証し、ブロックを組み立てて返す。"""
    assert rec["headers"].get("content-type", "").startswith("text/event-stream"), rec["headers"]
    out = []
    for ev, data in parse_sse(bytes(rec["body"])):
        obj = json.loads(data)
        assert ev == obj["type"], (ev, obj["type"])
        if obj["type"] == "ping":
            continue
        assert obj["type"] != "error", f"ストリーム内エラー {obj}"
        strict(AMSE, obj, obj["type"])
        out.append(obj)
    assert out[0]["type"] == "message_start" and out[-1]["type"] == "message_stop", (out[0]["type"], out[-1]["type"])
    assert out[-2]["type"] == "message_delta", out[-2]["type"]
    m = out[0]["message"]
    assert m["content"] == [] and m["role"] == "assistant" and m["model"] == AMODEL and m["id"].startswith("msg_"), m
    assert m["stop_reason"] is None
    blocks, cur = [], None
    for e in out[1:-2]:
        t = e["type"]
        if t == "content_block_start":
            assert cur is None and e["index"] == len(blocks), ("ブロックが重なった", e)
            cb = dict(e["content_block"])
            cur = {"block": cb, "json": "", "sig": ""}
            if cb["type"] == "tool_use":
                assert cb["input"] == {} and cb["id"] and cb["name"], cb
            elif cb["type"] == "text":
                assert cb["text"] == ""
            elif cb["type"] == "thinking":
                assert cb["thinking"] == ""
            else:
                raise AssertionError(f"想定外のブロック {cb['type']}")
        elif t == "content_block_delta":
            assert cur is not None and e["index"] == len(blocks)
            d, bt = e["delta"], cur["block"]["type"]
            allowed = {"text": ("text_delta",), "thinking": ("thinking_delta", "signature_delta"), "tool_use": ("input_json_delta",)}[bt]
            assert d["type"] in allowed, (bt, d["type"])
            if d["type"] == "text_delta":
                cur["block"]["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                assert not cur["sig"], "signature の後に thinking_delta"
                cur["block"]["thinking"] += d["thinking"]
            elif d["type"] == "signature_delta":
                cur["sig"] += d["signature"]
            else:
                cur["json"] += d["partial_json"]
        elif t == "content_block_stop":
            assert cur is not None and e["index"] == len(blocks)
            b = cur["block"]
            if b["type"] == "tool_use":
                b["input"] = json.loads(cur["json"] or "{}")
                assert isinstance(b["input"], dict)
            if b["type"] == "thinking":
                assert cur["sig"], "thinking ブロックに signature_delta が無い"
                b["signature"] = cur["sig"]
            blocks.append(b)
            cur = None
        else:
            raise AssertionError(f"想定外のイベント {t}")
    assert cur is None
    md = out[-2]
    assert md["delta"]["stop_reason"] in ("end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"), md
    assert md["usage"]["output_tokens"] > 0, md
    return out, blocks, md


def openai_error_shape(rec, status, code=None, param=None):
    assert rec["status"] == status, (rec["status"], status)
    body = json.loads(bytes(rec["body"]))
    err = body["error"]
    assert set(["message", "type", "param", "code"]) <= set(err), err
    assert isinstance(err["message"], str) and err["message"]
    if code:
        assert err["code"] == code, err
    if param:
        assert err["param"] == param, err
    return err


def anthropic_error_shape(rec, status, etype):
    assert rec["status"] == status, (rec["status"], status)
    body = json.loads(bytes(rec["body"]))
    assert body["type"] == "error" and body["error"]["type"] == etype and body["error"]["message"], body
    return body["error"]


def expect(exc_cls, fn):
    try:
        fn()
    except exc_cls as e:
        return e
    raise AssertionError(f"{exc_cls.__name__} が出なかった")


def png(r, g, b, w=32, h=32):
    raw = b"".join(b"\x00" + bytes([r, g, b]) * w for _ in range(h))
    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")
    return base64.b64encode(data).decode()


RED, GREEN, BLUE = png(230, 20, 20), png(20, 200, 20), png(20, 20, 230)
WEATHER_PARAMS = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}
CHAT_WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.", "parameters": WEATHER_PARAMS}}
RESP_WEATHER = {"type": "function", "name": "get_weather", "description": "Get the current weather for a city.", "parameters": WEATHER_PARAMS, "strict": True}
ANTH_WEATHER = {"name": "get_weather", "description": "Get the current weather for a city.", "input_schema": WEATHER_PARAMS}
PARALLEL_Q = "What is the weather in Tokyo and in Osaka? Call get_weather for both cities in parallel (two calls in one turn)."
WX = {"tokyo": "sunny, 21C", "osaka": "rainy, 17C"}


def wx(city):
    return WX.get(str(city).lower(), "cloudy, 10C")


LOG_DIR = os.environ.get("DS41GW_LOGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "req"))


def log_finished(ua, timeout=10.0):
    """要求ログ (§13.1) から User-Agent で要求を探し、request.finished を返す。ほかの要求と並走していても判別できる。"""
    import datetime, glob, os
    t0 = time.time()
    while time.time() - t0 < timeout:
        files = sorted(glob.glob(LOG_DIR + "/*.jsonl"), key=os.path.getmtime)
        if files:
            with open(files[-1], "rb") as fh:
                fh.seek(max(0, os.path.getsize(files[-1]) - (8 << 20)))
                lines = fh.read().splitlines()
            rid, fin = None, None
            for l in lines:
                if ua.encode() in l[:3000] and b'"request.received' in l[:200]:
                    rid = json.loads(l)["rid"]
            if rid:
                for l in lines:
                    if rid.encode() in l[:300] and b'"request.finished"' in l[:100]:
                        fin = json.loads(l)
                if fin:
                    ts = datetime.datetime.fromisoformat(fin["ts"].replace("Z", "+00:00")).timestamp()
                    fin["_ts"] = ts
                    return fin
        time.sleep(0.2)
    raise AssertionError(f"要求ログに {ua} の request.finished が {timeout}s 出ない (上流が止まっていない)")


def check_cancelled(ua, closed_at, label):
    fin = log_finished(ua)
    lag = round(fin["_ts"] - closed_at, 2)
    assert fin["client_disconnected"] is True, f"{label}: client_disconnected が {fin['client_disconnected']}"
    assert lag < 3.0, f"{label}: 切断から終了まで {lag}s"
    comp = (fin.get("usage") or {}).get("completion_tokens")
    return {"lag_s": lag, "dur_ms": fin["dur_ms"], "completion_tokens": comp, "finish_reason": fin.get("finish_reason")}


def ua_hdr(tag):
    ua = f"ds41gw-conformance/{tag}-{time.time_ns()}"
    return ua, {"User-Agent": ua}


class Weather(pydantic.BaseModel):
    city: str


class CityInfo(pydantic.BaseModel):
    capital: str
    country: str
    population_millions: float


# ================================================================ Chat Completions
@test("K-C1")
def k_c1():
    """非ストリーム: 本文を ChatCompletion で厳密に検証。"""
    cap_reset()
    r = OA.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Reply with the single word: pong"}],
                                   max_tokens=300, reasoning_effort="low")
    body = json.loads(bytes(caps("/chat/completions")[-1]["body"]))
    strict(ChatCompletion, body, "chat.completion")
    assert body["object"] == "chat.completion" and body["model"] == MODEL
    c = r.choices[0]
    assert c.finish_reason == "stop" and c.message.role == "assistant" and c.message.content.strip()
    assert c.message.model_extra.get("reasoning_content"), "thinking on なのに reasoning_content が無い"
    u = r.usage
    assert u.total_tokens == u.prompt_tokens + u.completion_tokens and u.completion_tokens_details.reasoning_tokens > 0, u
    return {"content": c.message.content}


@test("K-C2")
def k_c2():
    """ストリーム: SDK の chat.completions.stream 補助と生チャンクの文法、include_usage。"""
    cap_reset()
    seen = set()
    with OA.chat.completions.stream(model=MODEL, max_tokens=300, reasoning_effort="low", stream_options={"include_usage": True},
                                    messages=[{"role": "user", "content": "Count 1 to 5 separated by spaces."}]) as s:
        for e in s:
            seen.add(e.type)
        fin = s.get_final_completion()
    agg, usage, _ = chat_chunks_from_cap(caps("/chat/completions")[-1], usage_chunk=True)
    assert "content.delta" in seen and "content.done" in seen and "chunk" in seen, seen
    assert fin.choices[0].message.content == agg[0]["content"] and "3" in agg[0]["content"]
    assert agg[0]["reasoning"] and agg[0]["finish"] == "stop"
    assert usage["completion_tokens_details"]["reasoning_tokens"] > 0
    assert fin.usage and fin.usage.total_tokens == usage["total_tokens"]
    return {"events": sorted(seen), "content": agg[0]["content"]}


@test("K-C3")
def k_c3():
    """ツールの往復 (ストリーム): 並列呼び出し → 結果 → 最終回答。SDK の tool_calls.function.arguments.* イベント。"""
    msgs = [{"role": "user", "content": PARALLEL_Q}]
    cap_reset()
    ev_types = []
    with OA.chat.completions.stream(model=MODEL, messages=msgs, tools=[CHAT_WEATHER], tool_choice="required",
                                    max_tokens=800, reasoning_effort="low") as s:
        for e in s:
            ev_types.append(e.type)
        fin = s.get_final_completion()
    agg, _, _ = chat_chunks_from_cap(caps("/chat/completions")[-1])
    m = fin.choices[0].message
    assert fin.choices[0].finish_reason == "tool_calls" and agg[0]["finish"] == "tool_calls"
    assert len(m.tool_calls) >= 2, f"並列呼び出しが {len(m.tool_calls or [])} 本"
    assert [t.id for t in m.tool_calls] == [t["id"] for t in agg[0]["tools"].values()]
    assert ev_types.count("tool_calls.function.arguments.done") == len(m.tool_calls), ev_types
    ids = {t.id for t in m.tool_calls}
    assert len(ids) == len(m.tool_calls), "tool_call id が重複"
    msgs.append({"role": "assistant", "content": m.content, "tool_calls": [tc.model_dump(exclude_none=True, exclude={"parsed_arguments"}) for tc in m.tool_calls]})
    for tc in m.tool_calls:
        msgs.append({"role": "tool", "tool_call_id": tc.id, "content": wx(json.loads(tc.function.arguments).get("city"))})
    cap_reset()
    with OA.chat.completions.stream(model=MODEL, messages=msgs, tools=[CHAT_WEATHER], max_tokens=600, reasoning_effort="low") as s:
        for _ in s:
            pass
        fin2 = s.get_final_completion()
    agg2, _, _ = chat_chunks_from_cap(caps("/chat/completions")[-1])
    a = fin2.choices[0].message.content or ""
    assert fin2.choices[0].finish_reason == "stop" and a == agg2[0]["content"], fin2
    assert "21" in a or "sunny" in a.lower(), a
    return {"calls": [(t.function.name, t.function.arguments) for t in m.tool_calls], "answer": a[:150]}


@test("K-C4")
def k_c4():
    """ツール (非ストリーム) と strict な pydantic_function_tool の parse。"""
    msgs = [{"role": "user", "content": "What's the weather in Sapporo? Use the tool."}]
    r = OA.chat.completions.parse(model=MODEL, messages=msgs, tools=[openai.pydantic_function_tool(Weather, name="get_weather")],
                                  tool_choice={"type": "function", "function": {"name": "get_weather"}}, max_tokens=600, reasoning_effort="low")
    m = r.choices[0].message
    assert m.tool_calls and m.tool_calls[0].function.name == "get_weather", m
    p = m.tool_calls[0].function.parsed_arguments
    assert isinstance(p, Weather) and p.city, p
    assert r.choices[0].finish_reason == "tool_calls"
    return {"parsed": p.model_dump()}


@test("K-C5")
def k_c5():
    """json_schema: chat.completions.parse (非ストリーム) と stream(response_format=...) の content.done.parsed。"""
    q = [{"role": "user", "content": "Give the capital of Japan, the country, and the capital's population in millions."}]
    r = OA.chat.completions.parse(model=MODEL, messages=q, response_format=CityInfo, max_tokens=600, reasoning_effort="low")
    p = r.choices[0].message.parsed
    assert isinstance(p, CityInfo) and "tokyo" in p.capital.lower(), r.choices[0].message.content
    parsed_done = None
    with OA.chat.completions.stream(model=MODEL, messages=q, response_format=CityInfo, max_tokens=600, reasoning_effort="low") as s:
        for e in s:
            if e.type == "content.done":
                parsed_done = e.parsed
        fin = s.get_final_completion()
    assert isinstance(parsed_done, CityInfo) and isinstance(fin.choices[0].message.parsed, CityInfo), parsed_done
    # json_object
    r3 = OA.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Return a JSON object with key \"ok\" set to true."}],
                                    response_format={"type": "json_object"}, max_tokens=300, reasoning_effort="low")
    assert isinstance(json.loads(r3.choices[0].message.content), dict), r3.choices[0].message.content
    return {"parse": p.model_dump(), "stream": parsed_done.model_dump()}


@test("K-C6")
def k_c6():
    """画像入力 (ストリーム、2 枚) と reasoning の切り替え (none で思考なし)。"""
    cap_reset()
    s = OA.chat.completions.create(model=MODEL, stream=True, max_tokens=400, reasoning_effort="low", messages=[{"role": "user", "content": [
        {"type": "text", "text": "Name the color of each image, in order, one word each."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + GREEN}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + BLUE, "detail": "low"}}]}])
    text = "".join(c.choices[0].delta.content or "" for c in s if c.choices)
    agg, _, _ = chat_chunks_from_cap(caps("/chat/completions")[-1])
    assert agg[0]["content"] == text and "green" in text.lower() and "blue" in text.lower(), text
    cap_reset()
    r = OA.chat.completions.create(model=MODEL, stream=True, max_tokens=100, reasoning_effort="none", messages=[{"role": "user", "content": "Say ok."}])
    list(r)
    agg2, _, _ = chat_chunks_from_cap(caps("/chat/completions")[-1])
    assert agg2[0]["reasoning"] == "" and agg2[0]["content"], agg2
    return {"colors": text}


@test("K-C7")
def k_c7():
    """n=2 ストリーム: 各本の index と finish、[DONE] は 1 回。"""
    cap_reset()
    list(OA.chat.completions.create(model=MODEL, stream=True, n=2, max_tokens=60, reasoning_effort="none",
                                    messages=[{"role": "user", "content": "Name one random animal."}]))
    agg, _, _ = chat_chunks_from_cap(caps("/chat/completions")[-1], n=2)
    assert agg[0]["content"] and agg[1]["content"]
    return {"n": [agg[0]["content"], agg[1]["content"]]}


@test("K-C8")
def k_c8():
    """エラー: 不正入力は SDK の型付き例外になり、本文は OpenAI 形式。"""
    out = {}
    def bad(kw, code=None, param=None, exc=openai.BadRequestError):
        cap_reset()
        e = expect(exc, lambda: OA.chat.completions.create(**kw))
        err = openai_error_shape(caps()[-1], e.status_code, code, param)
        assert e.code == err["code"] and e.type == err["type"] and e.param == err["param"], (e.code, e.type, e.param, err)
        return f"{e.status_code} {err['type']}/{err['code']}/{err['param']}"
    base = dict(model=MODEL, max_tokens=5, messages=[{"role": "user", "content": "x"}])
    out["logprobs"] = bad({**base, "logprobs": True}, "unsupported_parameter", "logprobs")
    out["effort"] = bad({**base, "reasoning_effort": "bogus"}, "invalid_reasoning_effort")
    out["max_tokens0"] = bad({**base, "max_tokens": 0})
    out["no_messages"] = bad({**base, "messages": []})
    out["bad_role"] = bad({**base, "messages": [{"role": "wizard", "content": "x"}]})
    out["n_too_big"] = bad({**base, "n": 99})
    out["bad_tool"] = bad({**base, "tools": [{"type": "function", "function": {"description": "no name"}}]})
    # 数値の content は文字列にして受理する (仕様 §5.3 の吸収。OpenAI は 400)
    r0 = OA.chat.completions.create(**{**base, "reasoning_effort": "none", "messages": [{"role": "user", "content": 12345}]})
    assert r0.choices[0].message.content is not None
    # JSON の構文誤り (SDK を通さず送る)
    r = HTTP.post(A.url + "/v1/chat/completions", content=b"{not json", headers={"content-type": "application/json"})
    out["bad_json"] = r.status_code
    assert r.status_code == 400 and "error" in r.json(), r.text
    # 未知モデル: alias_mode=any なので生成は受理する。一覧に無い名前の取得は 404 (§4.1、§9.1)
    r2 = OA.chat.completions.create(model="no-such-model-xyz", max_tokens=30, reasoning_effort="none", messages=[{"role": "user", "content": "Say ok."}])
    assert r2.model == "no-such-model-xyz"
    cap_reset()
    e = expect(openai.NotFoundError, lambda: OA.models.retrieve("no-such-model-xyz"))
    out["model_retrieve"] = openai_error_shape(caps()[-1], 404)["code"]
    cap_reset()
    e = expect(openai.NotFoundError, lambda: OA.get("/no/such/path", cast_to=object))
    out["404_path"] = openai_error_shape(caps()[-1], 404, "not_found")["code"]
    return out


@test("K-C9")
def k_c9():
    """キャンセル: ストリームを途中で閉じる / 非ストリームの時間切れで、上流も止まる (§11.8、要求ログの client_disconnected)。"""
    ua, h = ua_hdr("chat-stream")
    s = OA.chat.completions.create(model=MODEL, stream=True, max_tokens=3000, reasoning_effort="none", extra_headers=h,
                                   messages=[{"role": "user", "content": "Count from 1 to 2000, one number per line."}])
    got = 0
    for c in s:
        if c.choices and c.choices[0].delta.content:
            got += 1
            if got >= 5:
                break
    s.close()
    out = {"stream": check_cancelled(ua, time.time(), "chat stream")}
    ua2, h2 = ua_hdr("chat-nonstream")
    c2 = OA.with_options(timeout=2.0)
    expect(openai.APITimeoutError, lambda: c2.chat.completions.create(model=MODEL, max_tokens=3000, reasoning_effort="none", extra_headers=h2,
                                  messages=[{"role": "user", "content": "Count from 1 to 2000, one number per line."}]))
    out["nonstream"] = check_cancelled(ua2, time.time(), "chat nonstream")
    return out


# ================================================================ Responses
@test("K-R1")
def k_r1():
    """非ストリーム: 本文を Response で厳密に検証し、§6.7 の補完項目を見る。"""
    cap_reset()
    r = OA.responses.create(model=MODEL, instructions="Answer in exactly one word.", input="What color is the sky on a clear day?",
                            max_output_tokens=400, reasoning={"effort": "low"}, metadata={"k": "v"}, temperature=0.7)
    body = json.loads(bytes(caps("/responses")[-1]["body"]))
    check_response_obj(body)
    assert body["status"] == "completed" and body["instructions"] == "Answer in exactly one word."
    assert body["metadata"] == {"k": "v"} and body["temperature"] == 0.7 and body["reasoning"]["effort"] == "low", body
    assert body["max_output_tokens"] == 400 and body["store"] is True and body["background"] is False
    assert r.output_text.strip() and [o.type for o in r.output] == ["reasoning", "message"], [o.type for o in r.output]
    u = r.usage
    assert u.total_tokens == u.input_tokens + u.output_tokens and u.output_tokens_details.reasoning_tokens > 0
    assert u.input_tokens_details.cached_tokens is not None
    return {"text": r.output_text}


@test("K-R2")
def k_r2():
    """ストリーム: responses.stream の型付きイベントと生 SSE の文法 (§6.8.1)、get_final_response の一致。"""
    cap_reset()
    typed = []
    with OA.responses.stream(model=MODEL, input="Write a haiku about rain.", max_output_tokens=500, reasoning={"effort": "low"}) as s:
        for e in s:
            typed.append(e)
        final = s.get_final_response()
    evs = resp_events_from_cap(caps("/responses")[-1])
    fr = check_resp_stream(evs)
    assert len(typed) == len(evs), (len(typed), len(evs))
    for te, ce in zip(typed, evs):
        assert te.type == ce["type"]
        assert isinstance(te, RESP_EVENT_CLASSES[te.type]) or te.type.endswith(".delta") or te.type.endswith(".done"), (type(te).__name__, te.type)
    assert final.id == fr["id"] and final.output_text == "".join(c["text"] for o in fr["output"] if o["type"] == "message" for c in o["content"])
    types = [e["type"] for e in evs]
    assert "response.reasoning_text.delta" in types and "response.output_text.delta" in types
    return {"n_events": len(evs), "types": sorted(set(types)), "text": final.output_text}


@test("K-R3")
def k_r3():
    """ツールの往復 (ストリーム): 並列 function_call → previous_response_id + function_call_output → 最終回答。"""
    cap_reset()
    with OA.responses.stream(model=MODEL, input=PARALLEL_Q, tools=[RESP_WEATHER], tool_choice="required",
                             max_output_tokens=800, reasoning={"effort": "low"}) as s:
        argdone = [e for e in s if e.type == "response.function_call_arguments.done"]
        r1 = s.get_final_response()
    fr = check_resp_stream(resp_events_from_cap(caps("/responses")[-1]))
    calls = [o for o in r1.output if o.type == "function_call"]
    assert len(calls) >= 2 and len(argdone) == len(calls), (len(calls), len(argdone))
    assert len({c.call_id for c in calls}) == len(calls)
    for c in calls:
        assert c.status == "completed" and c.name == "get_weather" and json.loads(c.arguments).get("city")
    out_items = [{"type": "function_call_output", "call_id": c.call_id, "output": wx(json.loads(c.arguments)["city"])} for c in calls]
    cap_reset()
    with OA.responses.stream(model=MODEL, previous_response_id=r1.id, input=out_items, tools=[RESP_WEATHER],
                             max_output_tokens=600, reasoning={"effort": "low"}) as s:
        for _ in s:
            pass
        r2 = s.get_final_response()
    fr2 = check_resp_stream(resp_events_from_cap(caps("/responses")[-1]))
    assert fr2["previous_response_id"] == r1.id, fr2["previous_response_id"]
    a = r2.output_text
    assert ("21" in a or "sunny" in a.lower()) and ("17" in a or "rain" in a.lower()), a
    # input_items には function_call_output が入っている
    cap_reset()
    items = OA.responses.input_items.list(r2.id)
    strict(ResponseItemList, json.loads(bytes(caps()[-1]["body"])), "input_items")
    assert any(i.type == "function_call_output" for i in items.data), [i.type for i in items.data]
    return {"calls": [(c.name, c.arguments) for c in calls], "answer": a[:160]}


@test("K-R4")
def k_r4():
    """previous_response_id の多ターン (非ストリームとストリームの混在) と retrieve / input_items / delete。"""
    r1 = OA.responses.create(model=MODEL, input="My name is Hanako. Just say OK.", max_output_tokens=300, reasoning={"effort": "low"})
    with OA.responses.stream(model=MODEL, previous_response_id=r1.id, input="I live in Kyoto. Just say OK.", max_output_tokens=300, reasoning={"effort": "low"}) as s:
        for _ in s:
            pass
        r2 = s.get_final_response()
    r3 = OA.responses.create(model=MODEL, previous_response_id=r2.id, input=[{"role": "user", "content": [{"type": "input_text", "text": "What is my name and city? Answer briefly."}]}],
                             max_output_tokens=400, reasoning={"effort": "low"})
    assert "Hanako" in r3.output_text and "Kyoto" in r3.output_text, r3.output_text
    assert r3.previous_response_id == r2.id
    cap_reset()
    g = OA.responses.retrieve(r2.id)
    check_response_obj(json.loads(bytes(caps()[-1]["body"])), "retrieve")
    assert g.id == r2.id and g.output_text == r2.output_text
    cap_reset()
    items = OA.responses.input_items.list(r3.id, order="asc")
    lb = json.loads(bytes(caps()[-1]["body"]))
    strict(ResponseItemList, lb, "input_items")
    assert lb["object"] == "list" and lb["first_id"] == lb["data"][0]["id"] and lb["last_id"] == lb["data"][-1]["id"] and lb["has_more"] is False, lb
    assert items.data[-1].type == "message" and items.data[-1].role == "user", items.data[-1]
    assert items.data[-1].content[0].type == "input_text"
    page = OA.responses.input_items.list(r3.id, limit=1, order="asc")
    assert len(page.data) == 1 and page.has_more is (len(items.data) > 1), (len(items.data), page.has_more)
    # ページ送り: 2 項目の input を limit=1 で after を使ってたどる
    r4 = OA.responses.create(model=MODEL, input=[{"role": "user", "content": "Remember: blue."}, {"role": "user", "content": "Just say OK."}],
                             max_output_tokens=100, reasoning={"effort": "none"})
    p1 = OA.responses.input_items.list(r4.id, limit=1, order="asc")
    assert len(p1.data) == 1 and p1.has_more is True, p1
    p2 = OA.responses.input_items.list(r4.id, limit=1, order="asc", after=p1.last_id)
    assert len(p2.data) == 1 and p2.has_more is False and p2.data[0].id != p1.data[0].id, p2
    allp = list(OA.responses.input_items.list(r4.id, limit=1, order="asc"))  # SDK の自動ページ送り
    assert [i.content[0].text for i in allp] == ["Remember: blue.", "Just say OK."], allp
    d = OA.responses.delete(r1.id)
    e = expect(openai.NotFoundError, lambda: OA.responses.retrieve(r1.id))
    e2 = expect(openai.BadRequestError, lambda: OA.responses.create(model=MODEL, previous_response_id=r2.id, input="x", max_output_tokens=20))
    assert e2.code == "previous_response_not_found" and e2.param == "previous_response_id", (e2.code, e2.param)
    return {"answer": r3.output_text, "n_items": len(items.data)}


@test("K-R5")
def k_r5():
    """json_schema: responses.parse(text_format=...) と stream(text_format=...)、strict な pydantic ツールの parsed_arguments。"""
    r = OA.responses.parse(model=MODEL, input="Give the capital of France, the country, and the capital's population in millions.",
                           text_format=CityInfo, max_output_tokens=600, reasoning={"effort": "low"})
    p = r.output_parsed
    assert isinstance(p, CityInfo) and "paris" in p.capital.lower(), r.output_text
    assert r.text.format.type == "json_schema", r.text
    with OA.responses.stream(model=MODEL, input="Give the capital of Italy, the country, and the capital's population in millions.",
                             text_format=CityInfo, max_output_tokens=600, reasoning={"effort": "low"}) as s:
        for _ in s:
            pass
        f = s.get_final_response()
    assert isinstance(f.output_parsed, CityInfo) and "rome" in f.output_parsed.capital.lower(), f.output_text
    r3 = OA.responses.parse(model=MODEL, input="What's the weather in Sapporo? Use the tool.", tools=[openai.pydantic_function_tool(Weather, name="get_weather")],
                            tool_choice="required", max_output_tokens=600, reasoning={"effort": "low"})
    fc = [o for o in r3.output if o.type == "function_call"]
    assert fc and isinstance(fc[0].parsed_arguments, Weather), r3.output
    return {"parse": p.model_dump(), "stream": f.output_parsed.model_dump(), "tool": fc[0].parsed_arguments.model_dump()}


@test("K-R6")
def k_r6():
    """reasoning: none / summary auto (reasoning_summary_* イベント) / encrypted_content の往復。"""
    r0 = OA.responses.create(model=MODEL, input="Say hi.", max_output_tokens=100, reasoning={"effort": "none"})
    assert [o.type for o in r0.output] == ["message"], [o.type for o in r0.output]
    assert not r0.usage.output_tokens_details.reasoning_tokens
    cap_reset()
    with OA.responses.stream(model=MODEL, input="What is 12*12?", max_output_tokens=500, reasoning={"effort": "low", "summary": "auto"},
                             include=["reasoning.encrypted_content"], store=False) as s:
        for _ in s:
            pass
        f = s.get_final_response()
    evs = resp_events_from_cap(caps("/responses")[-1])
    check_resp_stream(evs)
    types = [e["type"] for e in evs]
    assert "response.reasoning_summary_text.delta" in types and "response.reasoning_summary_part.added" in types
    assert "response.reasoning_text.delta" not in types, "summary と reasoning_text を二重に流した"
    rs = [o for o in f.output if o.type == "reasoning"][0]
    assert rs.summary and rs.summary[0].type == "summary_text" and rs.summary[0].text
    assert rs.encrypted_content and rs.encrypted_content.startswith("ds41gw1."), rs.encrypted_content
    assert "144" in f.output_text
    # 暗号化された思考だけを返して続ける (store=false のクライアントの形)
    inp = [{"role": "user", "content": "What is 12*12?"},
           {"type": "reasoning", "id": rs.id, "summary": [], "encrypted_content": rs.encrypted_content},
           {"role": "assistant", "content": f.output_text}, {"role": "user", "content": "Now divide that by 4. Number only."}]
    raw = OA.responses.with_raw_response.create(model=MODEL, input=inp, max_output_tokens=400, reasoning={"effort": "low"}, store=False)
    r2 = raw.parse()
    assert "36" in r2.output_text, r2.output_text
    assert "reasoning_decrypt" not in (raw.headers.get("x-ds41gw-warning") or ""), raw.headers.get("x-ds41gw-warning")
    return {"summary": rs.summary[0].text[:100], "r2": r2.output_text}


@test("K-R7")
def k_r7():
    """画像入力 (ストリーム、input_image の data URL)。"""
    cap_reset()
    with OA.responses.stream(model=MODEL, max_output_tokens=400, reasoning={"effort": "low"}, input=[{"role": "user", "content": [
            {"type": "input_text", "text": "What color is this image? One word."},
            {"type": "input_image", "image_url": "data:image/png;base64," + GREEN, "detail": "auto"}]}]) as s:
        for _ in s:
            pass
        f = s.get_final_response()
    check_resp_stream(resp_events_from_cap(caps("/responses")[-1]))
    assert "green" in f.output_text.lower(), f.output_text
    return {"text": f.output_text}


@test("K-R8")
def k_r8():
    """max_output_tokens 到達: response.incomplete と incomplete_details (ストリーム / 非ストリーム)。"""
    cap_reset()
    # SDK の get_final_response は response.completed しか受け付けない (OpenAI 本家でも incomplete では例外) ので、イベントを直接見る
    with OA.responses.stream(model=MODEL, input="Write a 500-word essay about the sea.", max_output_tokens=40, reasoning={"effort": "none"}) as s:
        last = [e for e in s][-1]
    assert last.type == "response.incomplete" and type(last).__name__ == "ResponseIncompleteEvent", type(last).__name__
    f = last.response
    fr = check_resp_stream(resp_events_from_cap(caps("/responses")[-1]), expect_status="incomplete")
    assert fr["incomplete_details"] == {"reason": "max_output_tokens"}, fr["incomplete_details"]
    assert f.status == "incomplete" and f.output[-1].status == "incomplete", f.output[-1]
    r = OA.responses.create(model=MODEL, input="Write a 500-word essay about the sea.", max_output_tokens=40, reasoning={"effort": "none"})
    assert r.status == "incomplete" and r.incomplete_details.reason == "max_output_tokens"
    return {"text": f.output_text[:80], "out_tokens": r.usage.output_tokens}


@test("K-R9")
def k_r9():
    """エラーとキャンセル API: 不正入力・未対応・存在しない id・cancel (background 無しなので 400)。"""
    out = {}
    def bad(fn, exc=openai.BadRequestError, code=None, param=None):
        cap_reset()
        e = expect(exc, fn)
        err = openai_error_shape(caps()[-1], e.status_code, code, param)
        return f"{e.status_code} {err['type']}/{err['code']}/{err['param']}"
    c = lambda **kw: (lambda: OA.responses.create(model=MODEL, max_output_tokens=20, **kw))
    out["web_search"] = bad(c(input="x", tools=[{"type": "web_search"}]), code="unsupported_tool")
    out["background"] = bad(c(input="x", background=True), code="unsupported_parameter", param="background")
    out["prev_missing"] = bad(c(input="x", previous_response_id="resp_" + "0" * 32), code="previous_response_not_found", param="previous_response_id")
    out["bad_item"] = bad(c(input=[{"type": "no_such_item", "foo": 1}]))
    out["bad_role"] = bad(c(input=[{"role": "wizard", "content": "x"}]))
    out["no_input"] = bad(lambda: OA.post("/responses", body={"model": MODEL, "max_output_tokens": 20}, cast_to=object))
    out["effort"] = bad(c(input="x", reasoning={"effort": "bogus"}))
    out["fc_out_orphan_ok"] = None
    out["retrieve_404"] = bad(lambda: OA.responses.retrieve("resp_" + "f" * 32), exc=openai.NotFoundError)
    out["delete_404"] = bad(lambda: OA.responses.delete("resp_" + "f" * 32), exc=openai.NotFoundError)
    out["items_404"] = bad(lambda: OA.responses.input_items.list("resp_" + "f" * 32), exc=openai.NotFoundError)
    r = OA.responses.create(model=MODEL, input="Say ok.", max_output_tokens=50, reasoning={"effort": "none"})
    out["cancel"] = bad(lambda: OA.responses.cancel(r.id))
    out.pop("fc_out_orphan_ok")
    return out


@test("K-R10")
def k_r10():
    """キャンセル: Responses のストリームを途中で閉じると上流も止まる。"""
    ua, h = ua_hdr("resp-stream")
    with OA.responses.stream(model=MODEL, input="Count from 1 to 2000, one number per line.", max_output_tokens=3000,
                             reasoning={"effort": "none"}, extra_headers=h) as s:
        n = 0
        for e in s:
            if e.type == "response.output_text.delta":
                n += 1
                if n >= 5:
                    break
    return check_cancelled(ua, time.time(), "responses stream")


@test("K-R11")
def k_r11():
    """ツールの指定: tool_choice の名前指定 + parallel_tool_calls:false (ストリーム)、応答オブジェクトへの反映。"""
    cap_reset()
    with OA.responses.stream(model=MODEL, input=PARALLEL_Q, tools=[RESP_WEATHER], tool_choice={"type": "function", "name": "get_weather"},
                             parallel_tool_calls=False, max_output_tokens=800, reasoning={"effort": "low"}) as s:
        for _ in s:
            pass
        r = s.get_final_response()
    fr = check_resp_stream(resp_events_from_cap(caps("/responses")[-1]))
    calls = [o for o in r.output if o.type == "function_call"]
    assert len(calls) == 1, len(calls)
    assert fr["parallel_tool_calls"] is False and fr["tool_choice"] == {"type": "function", "name": "get_weather"}, (fr["parallel_tool_calls"], fr["tool_choice"])
    assert fr["tools"][0]["name"] == "get_weather" and fr["tools"][0]["strict"] is True
    return {"call": calls[0].arguments}


# ================================================================ Completions (legacy)
@test("K-X1")
def k_x1():
    """completions: 非ストリーム / ストリームを Completion で厳密に検証、エラー。"""
    cap_reset()
    raw = OA.completions.with_raw_response.create(model=MODEL, prompt="The capital of Italy is", max_tokens=30)
    r = raw.parse()
    assert "completions_emulated_via_chat" in (raw.headers.get("x-ds41gw-warning") or "")
    strict(Completion, json.loads(bytes(caps()[-1]["body"])), "text_completion")
    assert r.object == "text_completion" and r.choices[0].text.strip() and r.model == MODEL
    cap_reset()
    text = "".join(c.choices[0].text for c in OA.completions.create(model=MODEL, prompt="List three colors:", max_tokens=30, stream=True) if c.choices)
    ev = parse_sse(bytes(caps()[-1]["body"]))
    assert ev[-1][1] == "[DONE]"
    for k, (_, d) in enumerate(ev[:-1]):
        o = json.loads(d)
        # SDK の Completion 型は finish_reason を非 null にしているが、ストリームの途中のチャンクは OpenAI 本家でも null
        if o["choices"] and o["choices"][0]["finish_reason"] is None:
            assert k < len(ev) - 2, "最後のチャンクに finish_reason が無い"
            o["choices"][0]["finish_reason"] = "stop"
        strict(Completion, o, "text_completion chunk")
    assert text.strip()
    out = {}
    for kw in [dict(echo=True), dict(prompt=[1, 2, 3]), dict(logprobs=2), dict(best_of=3)]:
        cap_reset()
        e = expect(openai.BadRequestError, lambda: OA.completions.create(model=MODEL, **{"prompt": "x", "max_tokens": 5, **kw}))
        out[list(kw)[0]] = openai_error_shape(caps()[-1], 400)["code"]
    return {"nonstream": r.choices[0].text, "stream": text, "errors": out}


# ================================================================ Anthropic Messages
@test("K-A1")
def k_a1():
    """非ストリーム: 本文を Message で厳密に検証。thinking 有りの block 順。"""
    cap_reset()
    m = AN.messages.create(model=AMODEL, max_tokens=500, system="You answer in one short sentence.",
                           thinking={"type": "enabled", "budget_tokens": 1024}, messages=[{"role": "user", "content": "What is the capital of France?"}])
    body = json.loads(bytes(caps("/messages")[-1]["body"]))
    strict(AMessage, body, "message")
    assert body["type"] == "message" and body["role"] == "assistant" and body["model"] == AMODEL and body["id"].startswith("msg_")
    assert [b.type for b in m.content] == ["thinking", "text"], [b.type for b in m.content]
    assert m.content[0].signature and m.stop_reason == "end_turn" and m.stop_sequence is None
    assert m.usage.input_tokens > 0 and m.usage.output_tokens > 0
    return {"text": m.content[1].text}


@test("K-A2")
def k_a2():
    """ストリーム: messages.stream の型付きイベントと生 SSE の文法、get_final_message の一致。"""
    cap_reset()
    typed = []
    with AN.messages.stream(model=AMODEL, max_tokens=600, thinking={"type": "enabled", "budget_tokens": 1024},
                            messages=[{"role": "user", "content": "What is 17*3? Answer with the number."}]) as s:
        for e in s:
            typed.append(e.type)
        text = s.get_final_text()
        m = s.get_final_message()
    raw, blocks, md = msg_events_from_cap(caps("/messages")[-1])
    assert "51" in text and m.stop_reason == md["delta"]["stop_reason"] == "end_turn"
    got = [b.model_dump(exclude_none=True) for b in m.content]
    for g, b in zip(got, blocks):
        for k, v in b.items():
            assert g.get(k) == v, (k, g.get(k), v)
    assert len(got) == len(blocks) and [b["type"] for b in blocks] == ["thinking", "text"]
    assert {"message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop", "text", "thinking", "signature"} <= set(typed), set(typed)
    assert m.usage.output_tokens == md["usage"]["output_tokens"]
    # thinking 無し
    cap_reset()
    with AN.messages.stream(model=AMODEL, max_tokens=100, messages=[{"role": "user", "content": "Say ok."}], thinking={"type": "disabled"}) as s:
        for _ in s:
            pass
    _, b2, _ = msg_events_from_cap(caps("/messages")[-1])
    assert [b["type"] for b in b2] == ["text"], b2
    return {"text": text, "n_events": len(raw)}


@test("K-A3")
def k_a3():
    """ツールの往復 (ストリーム): 並列 tool_use → tool_result → 最終回答。input_json_delta の連結。"""
    msgs = [{"role": "user", "content": PARALLEL_Q}]
    cap_reset()
    with AN.messages.stream(model=AMODEL, max_tokens=800, tools=[ANTH_WEATHER], tool_choice={"type": "any"}, messages=msgs,
                            thinking={"type": "enabled", "budget_tokens": 1024}) as s:
        for _ in s:
            pass
        m = s.get_final_message()
    _, blocks, md = msg_events_from_cap(caps("/messages")[-1])
    uses = [b for b in m.content if b.type == "tool_use"]
    assert len(uses) >= 2 and m.stop_reason == "tool_use" == md["delta"]["stop_reason"], (len(uses), m.stop_reason)
    assert [u.input for u in uses] == [b["input"] for b in blocks if b["type"] == "tool_use"]
    assert len({u.id for u in uses}) == len(uses) and all(u.id.startswith("toolu_") for u in uses)
    msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in m.content]})
    msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u.id, "content": [{"type": "text", "text": wx(u.input.get("city"))}]} for u in uses]})
    cap_reset()
    with AN.messages.stream(model=AMODEL, max_tokens=600, tools=[ANTH_WEATHER], messages=msgs, thinking={"type": "enabled", "budget_tokens": 1024}) as s:
        for _ in s:
            pass
        m2 = s.get_final_message()
    msg_events_from_cap(caps("/messages")[-1])
    a = "".join(b.text for b in m2.content if b.type == "text")
    assert m2.stop_reason == "end_turn" and ("21" in a or "sunny" in a.lower()), (m2.stop_reason, a)
    # tool_choice: 名前指定 (非ストリーム)、disable_parallel_tool_use
    m3 = AN.messages.create(model=AMODEL, max_tokens=600, tools=[ANTH_WEATHER], messages=[{"role": "user", "content": PARALLEL_Q}],
                            tool_choice={"type": "tool", "name": "get_weather", "disable_parallel_tool_use": True}, thinking={"type": "disabled"})
    u3 = [b for b in m3.content if b.type == "tool_use"]
    assert len(u3) == 1 and u3[0].name == "get_weather", [b.type for b in m3.content]
    return {"uses": [u.input for u in uses], "answer": a[:160]}


@test("K-A4")
def k_a4():
    """画像入力 (ストリーム、base64)。"""
    cap_reset()
    with AN.messages.stream(model=AMODEL, max_tokens=400, thinking={"type": "enabled", "budget_tokens": 1024}, messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": BLUE}},
            {"type": "text", "text": "What color is this image? One word."}]}]) as s:
        t = s.get_final_text()
    msg_events_from_cap(caps("/messages")[-1])
    assert "blue" in t.lower(), t
    return {"text": t}


@test("K-A5")
def k_a5():
    """構造化出力: messages.parse(output_format=...) と output_config.format (json_schema)。"""
    r = AN.messages.parse(model=AMODEL, max_tokens=600, output_format=CityInfo, thinking={"type": "disabled"},
                          messages=[{"role": "user", "content": "Give the capital of Germany, the country, and the capital's population in millions."}])
    p = r.parsed_output
    assert isinstance(p, CityInfo) and "berlin" in p.capital.lower(), [b for b in r.content]
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": False}
    with AN.messages.stream(model=AMODEL, max_tokens=600, thinking={"type": "enabled", "budget_tokens": 1024},
                            output_config={"format": {"type": "json_schema", "schema": schema}},
                            messages=[{"role": "user", "content": "What is 6*7?"}]) as s:
        t = s.get_final_text()
    assert json.loads(t) == {"answer": 42}, t
    return {"parse": p.model_dump(), "stream": t}


@test("K-A6")
def k_a6():
    """max_tokens 到達と stop_sequences (ストリーム)。"""
    cap_reset()
    with AN.messages.stream(model=AMODEL, max_tokens=30, thinking={"type": "disabled"}, messages=[{"role": "user", "content": "Write a 500-word essay about the sea."}]) as s:
        m = s.get_final_message()
    _, _, md = msg_events_from_cap(caps("/messages")[-1])
    assert m.stop_reason == "max_tokens" == md["delta"]["stop_reason"], m.stop_reason
    cap_reset()
    with AN.messages.stream(model=AMODEL, max_tokens=200, stop_sequences=["7"], thinking={"type": "disabled"},
                            messages=[{"role": "user", "content": "Count from 1 to 10, separated by commas."}]) as s:
        m2 = s.get_final_message()
    _, _, md2 = msg_events_from_cap(caps("/messages")[-1])
    assert m2.stop_reason == "stop_sequence" and m2.stop_sequence == "7" and md2["delta"]["stop_sequence"] == "7", m2
    assert "7" not in m2.content[-1].text
    return {"max_tokens_text": m.content[-1].text[:60], "stop_text": m2.content[-1].text}


@test("K-A7")
def k_a7():
    """エラー: Anthropic 形式の本文と SDK の型付き例外。未知モデル。"""
    out = {}
    def bad(fn, exc=anthropic.BadRequestError, status=400, etype="invalid_request_error"):
        cap_reset()
        e = expect(exc, fn)
        err = anthropic_error_shape(caps()[-1], status, etype)
        assert e.status_code == status
        return f"{status} {err['type']}: {err['message'][:60]}"
    msg = [{"role": "user", "content": "x"}]
    out["no_max_tokens"] = bad(lambda: AN.post("/v1/messages", body={"model": AMODEL, "messages": msg}, cast_to=object))
    out["document"] = bad(lambda: AN.messages.create(model=AMODEL, max_tokens=10, messages=[{"role": "user", "content": [
        {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "hi"}}]}]))
    out["builtin_tool"] = bad(lambda: AN.messages.create(model=AMODEL, max_tokens=10, messages=msg, tools=[{"type": "bash_20250124", "name": "bash"}]))
    out["bad_role"] = bad(lambda: AN.messages.create(model=AMODEL, max_tokens=10, messages=[{"role": "wizard", "content": "x"}]))
    out["empty_messages"] = bad(lambda: AN.messages.create(model=AMODEL, max_tokens=10, messages=[]))
    cap_reset()
    rj = HTTP.post(A.url + "/v1/messages", content=b"{oops", headers={"content-type": "application/json", "anthropic-version": "2023-06-01"})
    out["bad_json"] = anthropic_error_shape(caps()[-1], 400, "invalid_request_error")["message"][:60]
    out["404_path"] = bad(lambda: AN.get("/v1/no/such", cast_to=object), exc=anthropic.NotFoundError, status=404, etype="not_found_error")
    m = AN.messages.create(model="claude-unknown-9", max_tokens=30, thinking={"type": "disabled"}, messages=[{"role": "user", "content": "Say ok."}])
    assert m.model == "claude-unknown-9"
    cap_reset()
    expect(anthropic.NotFoundError, lambda: AN.models.retrieve("claude-unknown-9"))
    out["model_retrieve"] = caps()[-1]["status"]
    n = AN.messages.count_tokens(model=AMODEL, messages=[{"role": "user", "content": "hello"}])
    assert n.input_tokens > 0
    return out


@test("K-A8")
def k_a8():
    """キャンセル: Messages のストリームを途中で閉じると上流も止まる。"""
    ua, h = ua_hdr("msg-stream")
    with AN.messages.stream(model=AMODEL, max_tokens=3000, thinking={"type": "disabled"}, extra_headers=h,
                            messages=[{"role": "user", "content": "Count from 1 to 2000, one number per line."}]) as s:
        n = 0
        for _ in s.text_stream:
            n += 1
            if n >= 5:
                break
    return check_cancelled(ua, time.time(), "messages stream")


# ================================================================ 実行
def run(tid, f):
    NOTES.d = {}
    t0 = time.time()
    try:
        info = f()
        res = {"id": tid, "doc": (f.__doc__ or "").strip(), "ok": True, "sec": round(time.time() - t0, 2), "info": info}
    except Exception as e:
        res = {"id": tid, "doc": (f.__doc__ or "").strip(), "ok": False, "sec": round(time.time() - t0, 2),
               "error": f"{type(e).__name__}: {e}"[:1500], "tb": traceback.format_exc()[-2500:]}
    print(("PASS " if res["ok"] else "FAIL ") + tid + f" ({res['sec']}s) " + json.dumps(res.get("info", res.get("error")), ensure_ascii=False)[:400], flush=True)
    if not res["ok"]:
        print(res["tb"], flush=True)
    return res


def main():
    print(f"openai {openai.__version__} / anthropic {anthropic.__version__} / pydantic {pydantic.__version__} -> {A.url}")
    sel = [t for t in TESTS if not A.only or t[0] in A.only.split(",")]
    results = [run(*t) for t in sel]
    ok = sum(r["ok"] for r in results)
    print(f"\n{ok}/{len(results)} passed")
    if A.out:
        json.dump({"openai": openai.__version__, "anthropic": anthropic.__version__, "url": A.url, "results": results},
                  open(A.out, "w"), ensure_ascii=False, indent=1)
    sys.exit(0 if ok == len(results) else 1)


main()
