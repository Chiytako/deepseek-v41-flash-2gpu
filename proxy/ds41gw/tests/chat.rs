//! /v1/chat/completions と共通の動作 (待ち行列、キャンセル、エラー、ログ、認証) の結合試験。
//! 上流は tests/common の mock。

mod common;

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use common::*;
use serde_json::{json, Value};

#[tokio::test]
async fn chat_nonstream_defaults_and_alias() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = client()
        .post(format!("{}/v1/chat/completions", px.url))
        .header("content-type", "text/plain") // Content-Type を問わない
        .body(json!({"model": "GLM-5.3-Flash", "messages": [{"role": "user", "content": "hi"}]}).to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers().get("x-request-id").unwrap().to_str().unwrap().starts_with("gw_"));
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "GLM-5.3-Flash");
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(v["choices"][0]["message"]["reasoning_content"], "Let me think.");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], 100);
    assert!(v["usage"]["completion_tokens_details"]["reasoning_tokens"].as_u64().unwrap() > 0);
    let b = mock.last_body();
    assert_eq!(b["model"], "deepseek-ai/DeepSeek-V4.1-Flash");
    assert_eq!(b["stream"], true);
    assert_eq!(b["temperature"], 1.0);
    assert_eq!(b["top_p"], 1.0);
    assert_eq!(b["max_tokens"], 262144);
    assert_eq!(b["thinking"]["type"], "enabled");
    assert_eq!(b["reasoning_effort"], 75);
    assert!(b.get("stream_options").is_none());
}

#[tokio::test]
async fn chat_stream_include_usage_split() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = client()
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.6,
            "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "text/event-stream");
    let evs = read_sse(r).await;
    let last = evs.last().unwrap();
    assert_eq!(last.1, "[DONE]");
    let ds: Vec<Value> = evs.iter().filter(|e| e.1 != "[DONE]" && e.0 != "#comment").map(|e| serde_json::from_str(&e.1).unwrap()).collect();
    assert_eq!(ds[0]["choices"][0]["delta"]["role"], "assistant");
    let text: String = ds.iter().filter_map(|d| d["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(text, "Hello world");
    let reasoning: String = ds.iter().filter_map(|d| d["choices"][0]["delta"]["reasoning_content"].as_str()).collect();
    assert_eq!(reasoning, "Let me think.");
    let fin = ds.iter().find(|d| d["choices"][0]["finish_reason"] == "stop").unwrap();
    assert!(fin["usage"].is_null());
    let u = ds.last().unwrap();
    assert_eq!(u["choices"], json!([]));
    assert_eq!(u["usage"]["total_tokens"], 120);
    assert!(u["usage"]["completion_tokens_details"]["reasoning_tokens"].is_number());
    // 明示した temperature は残し、top_p だけ補う
    let b = mock.last_body();
    assert_eq!(b["temperature"], 0.6);
    assert_eq!(b["top_p"], 1.0);
}

#[tokio::test]
async fn chat_stream_without_include_usage_keeps_usage_in_final_chunk() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = client()
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "stream": true, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    let evs = read_sse(r).await;
    let ds = datas(&evs);
    let fin = ds.iter().find(|d| d["choices"][0]["finish_reason"] == "stop").unwrap();
    assert_eq!(fin["usage"]["prompt_tokens"], 100);
    assert!(ds.iter().all(|d| d["choices"].as_array().map_or(false, |a| !a.is_empty())));
}

#[tokio::test]
async fn chat_normalization_table() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let req = json!({
        "model": "GLM-5.3-Flash",
        "stream": false,
        "stream_options": {"include_usage": true},
        "max_completion_tokens": 777,
        "reasoning_effort": "HIGH",
        "chat_template_kwargs": {"enable_thinking": false},
        "presence_penalty": 0.5,
        "user": "u1", "metadata": {"a": 1}, "store": true,
        "tools": [
            {"type": "function", "function": {"name": "a.b", "description": "dotted", "parameters": {"type": ["object"], "properties": {"x": {"type": "string"}}}}},
            {"function": {"name": "plain", "parameters": {}}},
            {"type": "function", "name": "flat", "parameters": {"properties": {}}}
        ],
        "messages": [
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": null},
            {"role": "user", "content": [{"type": "input_text", "text": "q"}, {"type": "image_url", "image_url": "data:image/png;base64,AAAA"}]},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "function": {"name": "a.b", "arguments": {"x": "1"}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "tool_reference", "name": "schedule_task"}]},
            {"role": "assistant", "content": null, "reasoning": "old thought"},
            {"role": "user", "content": "#mock:tools go"}
        ]
    });
    let r = client().post(format!("{}/v1/chat/completions", px.url)).json(&req).send().await.unwrap();
    let st = r.status();
    let warn = r.headers().get("x-ds41gw-warning").map(|v| v.to_str().unwrap().to_string()).unwrap_or_default();
    let v: Value = r.json().await.unwrap();
    assert_eq!(st, 200, "{v}");
    assert!(warn.contains("ignored=presence_penalty"), "{warn}");
    let b = mock.last_body();
    assert_eq!(b["max_tokens"], 777);
    assert!(b.get("max_completion_tokens").is_none());
    assert_eq!(b["thinking"]["type"], "disabled");
    assert!(b.get("chat_template_kwargs").is_none());
    assert!(b.get("stream_options").is_none());
    assert!(b.get("user").is_none() && b.get("metadata").is_none() && b.get("store").is_none());
    let m = b["messages"].as_array().unwrap();
    assert_eq!(m[0]["role"], "system");
    assert_eq!(m[1]["content"], "");
    assert_eq!(m[2]["content"][0], json!({"type": "text", "text": "q"}));
    assert_eq!(m[2]["content"][1]["image_url"]["url"], "data:image/png;base64,AAAA");
    assert_eq!(m[3]["tool_calls"][0]["function"]["arguments"], "{\"x\":\"1\"}");
    assert_eq!(m[3]["tool_calls"][0]["function"]["name"], "a_b");
    assert_eq!(m[4]["content"][0]["text"], "Tool loaded: schedule_task");
    assert_eq!(m[5]["content"], "");
    assert_eq!(m[5]["reasoning_content"], "old thought");
    let tools = b["tools"].as_array().unwrap();
    assert_eq!(tools[0]["function"]["name"], "a_b");
    assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    assert_eq!(tools[1]["type"], "function");
    assert_eq!(tools[1]["function"]["parameters"], json!({"type": "object", "properties": {}}));
    assert_eq!(tools[2]["function"]["name"], "flat");
    assert_eq!(tools[2]["function"]["parameters"]["type"], "object");
    // 上流の本文が recipe (ds41rt と同じ変換器) で通ること
    recipe_accepts(&b);
    // 出力の名前は元に戻す
    let tcs = v["choices"][0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(tcs.len(), 2);
    assert_eq!(tcs[0]["function"]["name"], "a.b");
    assert_eq!(tcs[0]["function"]["arguments"], "{\"city\": \"Tokyo\"}");
    assert_eq!(tcs[1]["function"]["name"], "plain");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert!(v["choices"][0]["message"].get("reasoning_content").is_none());
}

/// ds41rt と同じ recipe の Chat 変換が Ok を返すか (T-U2)。
fn recipe_accepts(b: &Value) {
    use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
    let mut b = b.clone();
    // ds41rt は tool_choice:"required" を auto に書き換えてから recipe に渡す
    if b.get("tool_choice").is_some_and(|t| t != "none") {
        b.as_object_mut().unwrap().remove("tool_choice");
    }
    let req: deepseek_recipe::openai::ChatCompletionRequest = serde_json::from_value(b.clone()).unwrap_or_else(|e| panic!("deser: {e}\n{b}"));
    if let Err(e) = req.convert(ConversionOptions::new()) {
        panic!("recipe rejected: {e}\n{b:#}");
    }
}

#[tokio::test]
async fn tool_history_repair() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let tools = json!([{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {}}}}]);
    let req = json!({"model": "x", "tools": tools, "messages": [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
            {"id": "a2", "type": "function", "function": {"name": "read", "arguments": "{}"}},
            {"id": "a3", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a2", "content": "r2"},
        {"role": "user", "content": "interjection"},
        {"role": "tool", "tool_call_id": "a1", "content": "r1"},
        {"role": "tool", "tool_call_id": "a1", "content": "dup"},
        {"role": "tool", "tool_call_id": "zz", "content": "orphan"},
        {"role": "assistant", "content": "next"},
        {"role": "tool", "tool_call_id": "late", "content": "orphan2"},
        {"role": "user", "content": "end"}
    ]});
    let r = client().post(format!("{}/v1/chat/completions", px.url)).json(&req).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers()["x-ds41gw-warning"].to_str().unwrap().contains("repaired_tool_history"));
    let b = mock.last_body();
    let m = b["messages"].as_array().unwrap();
    let roles: Vec<&str> = m.iter().map(|x| x["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "tool", "tool", "user", "user", "assistant", "user", "user"]);
    assert_eq!(m[2]["tool_call_id"], "a1");
    assert_eq!(m[2]["content"], "r1");
    assert_eq!(m[3]["tool_call_id"], "a2");
    assert_eq!(m[4]["tool_call_id"], "a3");
    assert!(m[4]["content"].as_str().unwrap().contains("no result"));
    assert!(m[6]["content"].as_str().unwrap().starts_with("[tool result zz]"));
    recipe_accepts(&b);
}

#[tokio::test]
async fn tool_choice_none_flattens_history_and_strips_dsml() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let req = json!({"model": "x", "tool_choice": "none",
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
        "messages": [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a1", "type": "function", "function": {"name": "read", "arguments": "{\"p\":1}"}}]},
        {"role": "tool", "tool_call_id": "a1", "content": "file body"},
        {"role": "user", "content": "#mock:dsml summarize"}
    ]});
    let r = client().post(format!("{}/v1/chat/completions", px.url)).json(&req).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    let b = mock.last_body();
    assert!(b.get("tools").is_none());
    assert!(b.get("tool_choice").is_none());
    let m = b["messages"].as_array().unwrap();
    assert!(m.iter().all(|x| x.get("tool_calls").is_none() && x["role"] != "tool"));
    assert_eq!(m[1]["content"], "[tool call] read({\"p\":1})");
    assert_eq!(m[2]["content"], "[tool result] file body");
    recipe_accepts(&b);
    let c = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!c.contains("DSML"), "{c}");
    assert_eq!(c, "Answer: ok\n\n\nbye");
}

#[tokio::test]
async fn dsml_stripped_in_stream() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = client()
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "stream": true, "messages": [{"role": "user", "content": "#mock:dsml"}]}))
        .send()
        .await
        .unwrap();
    let ds = datas(&read_sse(r).await);
    let text: String = ds.iter().filter_map(|d| d["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(text, "Answer: ok\n\n\nbye");
}

#[tokio::test]
async fn rejects_and_n() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let c = client();
    let post = |b: Value| c.post(format!("{}/v1/chat/completions", px.url)).json(&b).send();
    let r = post(json!({"model": "x", "logprobs": true, "messages": [{"role": "user", "content": "hi"}]})).await.unwrap();
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["code"], "unsupported_parameter");
    assert_eq!(v["error"]["param"], "logprobs");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    let r = post(json!({"model": "x", "reasoning_effort": "bogus", "messages": [{"role": "user", "content": "hi"}]})).await.unwrap();
    assert_eq!(r.status(), 400);
    let r = post(json!({"model": "x", "n": 9, "messages": [{"role": "user", "content": "hi"}]})).await.unwrap();
    assert_eq!(r.status(), 400);
    let r = c.post(format!("{}/v1/chat/completions", px.url)).body("{not json").send().await.unwrap();
    assert_eq!(r.status(), 400);
    assert_eq!(r.json::<Value>().await.unwrap()["error"]["code"], "invalid_json");
    // n=2: 2 本に分けて送る
    let before = mock.st.calls.load(Ordering::SeqCst);
    let r = post(json!({"model": "x", "n": 2, "seed": 5, "messages": [{"role": "user", "content": "hi"}]})).await.unwrap();
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["choices"].as_array().unwrap().len(), 2);
    assert_eq!(v["choices"][1]["index"], 1);
    assert_eq!(v["usage"]["completion_tokens"], 40);
    assert_eq!(mock.st.calls.load(Ordering::SeqCst) - before, 2);
    let seeds: Vec<i64> = mock.bodies().iter().rev().take(2).map(|b| b["seed"].as_i64().unwrap()).collect();
    assert!(seeds.contains(&5) && seeds.contains(&6));
    // n=2 のストリーム
    let r = post(json!({"model": "x", "n": 2, "stream": true, "messages": [{"role": "user", "content": "hi"}]})).await.unwrap();
    let evs = read_sse(r).await;
    assert_eq!(evs.last().unwrap().1, "[DONE]");
    let ds = datas(&evs);
    let idx: std::collections::HashSet<u64> = ds.iter().filter_map(|d| d["choices"][0]["index"].as_u64()).collect();
    assert_eq!(idx.len(), 2);
    // include_usage が無ければ choices:[] のチャンクは送らない (choices[0] を読むクライアントが落ちる)
    assert!(ds.iter().all(|d| !d["choices"].as_array().unwrap().is_empty()), "{ds:?}");
    // seed を省いても、本ごとに別の seed を入れる (同じ出力にならないように)
    let s2: Vec<i64> = mock.bodies().iter().rev().take(2).map(|b| b["seed"].as_i64().unwrap()).collect();
    assert_ne!(s2[0], s2[1]);
    // include_usage あり: 合算 usage を choices:[] の最後のチャンクで 1 回だけ送る
    let r = post(json!({"model": "x", "n": 2, "stream": true, "stream_options": {"include_usage": true},
        "messages": [{"role": "user", "content": "hi"}]})).await.unwrap();
    let ds = datas(&read_sse(r).await);
    let last = ds.last().unwrap();
    assert_eq!(last["choices"], json!([]));
    assert_eq!(last["usage"]["completion_tokens"], 40);
    assert_eq!(ds.iter().filter(|d| d["choices"].as_array().unwrap().is_empty()).count(), 1);
}

#[tokio::test]
async fn upstream_errors_are_mapped() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let c = client();
    let send = |path: &str, b: Value| c.post(format!("{}{path}", px.url)).json(&b).send();
    let chat = |s: &str| json!({"model": "x", "messages": [{"role": "user", "content": format!("#mock:{s}")}]});
    let r = send("/v1/chat/completions", chat("err400")).await.unwrap();
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["code"], "upstream_rejected");
    assert!(v["error"]["message"].as_str().unwrap().starts_with("upstream: invalid thing"));
    let r = send("/v1/chat/completions", chat("err500")).await.unwrap();
    assert_eq!(r.status(), 502);
    assert_eq!(r.json::<Value>().await.unwrap()["error"]["type"], "server_error");
    let r = send("/v1/chat/completions", chat("err503")).await.unwrap();
    assert_eq!(r.status(), 503);
    // Anthropic 形式
    let r = send("/v1/messages", json!({"model": "claude-x", "max_tokens": 10, "messages": [{"role": "user", "content": "#mock:err400"}]})).await.unwrap();
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    let r = send("/v1/messages", json!({"model": "claude-x", "max_tokens": 10, "messages": [{"role": "user", "content": "#mock:err503"}]})).await.unwrap();
    assert_eq!(r.status(), 529);
    assert_eq!(r.json::<Value>().await.unwrap()["error"]["type"], "overloaded_error");
    // 429 は proxy が再試行する
    let t0 = Instant::now();
    let r = send("/v1/chat/completions", chat("r429x2")).await.unwrap();
    assert_eq!(r.status(), 200);
    assert!(t0.elapsed() >= Duration::from_millis(1400));
    assert_eq!(px.state.metrics.counter("ds41gw_upstream_retries_total", &[("reason", "429")]), 2);
    // 未定義のパス
    let r = c.get(format!("{}/v1/nothing", px.url)).send().await.unwrap();
    assert_eq!(r.status(), 404);
    assert_eq!(r.json::<Value>().await.unwrap()["error"]["code"], "not_found");
}

#[tokio::test]
async fn upstream_down_gives_503() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| {
        c.upstream.url = "http://127.0.0.1:9".into();
        c.upstream.connect_retry_total_s = 1.0;
    })
    .await;
    let t0 = Instant::now();
    let r = client()
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
    assert!(t0.elapsed() >= Duration::from_millis(400));
    assert_eq!(r.json::<Value>().await.unwrap()["error"]["code"], "upstream_unavailable");
    let r = client().get(format!("{}/health/ready", px.url)).send().await.unwrap();
    assert_eq!(r.status(), 503);
    let r = client().get(format!("{}/health", px.url)).send().await.unwrap();
    assert_eq!(r.status(), 200);
}

#[tokio::test]
async fn midstream_failures() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let c = client();
    // ストリーム: エラーイベントを送り、[DONE] は送らない
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "stream": true, "messages": [{"role": "user", "content": "#mock:midfail"}]}))
        .send()
        .await
        .unwrap();
    let evs = read_sse(r).await;
    assert!(evs.iter().all(|e| e.1 != "[DONE]"));
    let last: Value = serde_json::from_str(&evs.last().unwrap().1).unwrap();
    assert_eq!(last["error"]["code"], "upstream_disconnected");
    // 非ストリーム: 1 回再試行して成功
    let before = mock.st.calls.load(Ordering::SeqCst);
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "#mock:midfail_once"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(mock.st.calls.load(Ordering::SeqCst) - before, 2);
    // 非ストリーム: 再試行しても失敗 → 502
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "#mock:midfail"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
}

async fn wait_for(f: impl Fn() -> bool, ms: u64) -> bool {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(ms) {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    f()
}

#[tokio::test]
async fn client_disconnect_cancels_upstream_stream() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let mut r = client()
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "stream": true, "messages": [{"role": "user", "content": "#mock:slow"}]}))
        .send()
        .await
        .unwrap();
    let _ = r.chunk().await.unwrap();
    let _ = r.chunk().await.unwrap();
    assert_eq!(mock.st.disconnects.load(Ordering::SeqCst), 0);
    let t0 = Instant::now();
    drop(r);
    // T-M5: 上流の切断を観測する
    assert!(wait_for(|| mock.st.disconnects.load(Ordering::SeqCst) == 1, 1000).await);
    eprintln!("stream cancel observed after {:?}", t0.elapsed());
    assert!(t0.elapsed() < Duration::from_millis(500));
}

#[tokio::test]
async fn client_disconnect_cancels_upstream_nonstream() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let c = reqwest::Client::builder().timeout(Duration::from_millis(300)).build().unwrap();
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "#mock:slow"}]}))
        .send()
        .await;
    assert!(r.is_err());
    // T-M6: 非ストリームでも、クライアントの切断で上流を切る
    assert!(wait_for(|| mock.st.disconnects.load(Ordering::SeqCst) == 1, 1500).await, "upstream was not cancelled");
}

#[tokio::test]
async fn queue_limits_upstream_concurrency() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| c.upstream.max_concurrency = 3).await;
    let c = client();
    let futs = (0..12).map(|_| {
        c.post(format!("{}/v1/chat/completions", px.url))
            .json(&json!({"model": "x", "tools": [{"type": "function", "function": {"name": "f"}}], "messages": [{"role": "user", "content": "#mock:work"}]}))
            .send()
    });
    let rs = futures::future::join_all(futs).await;
    for r in rs {
        assert_eq!(r.unwrap().status(), 200);
    }
    assert!(mock.st.max_inflight.load(Ordering::SeqCst) <= 3, "max inflight {}", mock.st.max_inflight.load(Ordering::SeqCst));
    assert_eq!(mock.st.completed.load(Ordering::SeqCst), 12);
}

#[tokio::test]
async fn queue_full_and_timeout() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| {
        c.upstream.max_concurrency = 1;
        c.queue.max_waiting = 1;
        c.queue.timeout_s = 0.3;
    })
    .await;
    let c = client();
    let mk = || {
        c.post(format!("{}/v1/chat/completions", px.url))
            .json(&json!({"model": "x", "messages": [{"role": "user", "content": "#mock:slow"}]}))
            .send()
    };
    let (a, b, d) = tokio::join!(mk(), async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        mk().await
    }, async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        mk().await
    });
    assert_eq!(a.unwrap().status(), 200);
    let b = b.unwrap();
    assert_eq!(b.status(), 429);
    assert!(b.headers().get("retry-after").is_some());
    assert_eq!(b.json::<Value>().await.unwrap()["error"]["code"], "queue_timeout");
    let d = d.unwrap();
    assert_eq!(d.status(), 429);
    assert_eq!(d.json::<Value>().await.unwrap()["error"]["code"], "queue_full");
    // Anthropic は 529
    let px2 = start_proxy(&mock, |c| {
        c.upstream.max_concurrency = 1;
        c.queue.max_waiting = 0;
    })
    .await;
    let slow = c
        .post(format!("{}/v1/chat/completions", px2.url))
        .json(&json!({"model": "x", "stream": true, "messages": [{"role": "user", "content": "#mock:slow"}]}))
        .send()
        .await
        .unwrap();
    let r = c
        .post(format!("{}/v1/messages", px2.url))
        .json(&json!({"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 529);
    drop(slow);
}

#[tokio::test]
async fn keepalive_and_early_headers() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| {
        c.client.early_headers_after_s = 0.1;
        c.client.keepalive_interval_s = 0.08;
    })
    .await;
    let t0 = Instant::now();
    let r = client()
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "stream": true, "messages": [{"role": "user", "content": "#mock:silent"}]}))
        .send()
        .await
        .unwrap();
    // 上流のヘッダは 400 ms 後だが、proxy は 100 ms で先にヘッダを返す (T-M4)
    assert!(t0.elapsed() < Duration::from_millis(350), "headers after {:?}", t0.elapsed());
    assert_eq!(r.status(), 200);
    let evs = read_sse(r).await;
    assert!(evs.iter().any(|e| e.0 == "#comment" && e.1 == "keep-alive"));
    assert_eq!(evs.last().unwrap().1, "[DONE]");
    // 上流のヘッダ後の無音でも keep-alive
    let r = client()
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "stream": true, "messages": [{"role": "user", "content": "#mock:silent_body"}]}))
        .send()
        .await
        .unwrap();
    let evs = read_sse(r).await;
    assert!(evs.iter().filter(|e| e.0 == "#comment").count() >= 2);
    // Messages の keep-alive は ping
    let r = client()
        .post(format!("{}/v1/messages", px.url))
        .json(&json!({"model": "x", "stream": true, "max_tokens": 20, "messages": [{"role": "user", "content": "#mock:silent"}]}))
        .send()
        .await
        .unwrap();
    let evs = read_sse(r).await;
    assert_eq!(evs[0].0, "ping");
    assert!(evs.iter().any(|e| e.0 == "message_stop"));
}

#[tokio::test]
async fn reasoning_is_reinjected() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let c = client();
    let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]);
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "tools": tools, "messages": [{"role": "user", "content": "#mock:tools weather?"}]}))
        .send()
        .await
        .unwrap();
    let v: Value = r.json().await.unwrap();
    let msg = v["choices"][0]["message"].clone();
    let tcs = msg["tool_calls"].clone();
    // 2 ターン目: 思考を返さない (履歴に思考を返さないクライアントの形)
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "tools": tools, "messages": [
            {"role": "user", "content": "#mock:tools weather?"},
            {"role": "assistant", "content": msg["content"], "tool_calls": tcs},
            {"role": "tool", "tool_call_id": tcs[0]["id"], "content": "sunny"},
            {"role": "tool", "tool_call_id": tcs[1]["id"], "content": "rainy"}
        ]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let b = mock.last_body();
    assert_eq!(b["messages"][1]["reasoning_content"], "Let me think.");
    // 本文のハッシュでも引ける
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    let v: Value = r.json().await.unwrap();
    let _ = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": v["choices"][0]["message"]["content"]}, {"role": "user", "content": "more"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(mock.last_body()["messages"][1]["reasoning_content"], "Let me think.");
    // clear_thinking では消す
    let _ = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "chat_template_kwargs": {"clear_thinking": true}, "messages": [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello world", "reasoning_content": "secret"}, {"role": "user", "content": "more"}]}))
        .send()
        .await
        .unwrap();
    assert!(mock.last_body()["messages"][1].get("reasoning_content").is_none());
}

#[tokio::test]
async fn request_log_is_replayable() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| c.log.images = "hash".into()).await;
    let img = format!("data:image/png;base64,{}", "A".repeat(400));
    let r = client()
        .post(format!("{}/v1/chat/completions", px.url))
        .header("user-agent", "test-agent/1")
        .json(&json!({"model": "GLM-5.3-Flash", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": img}}]}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let _ = r.text().await;
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let dir = px.dir.path().join("logs/req");
    let f = std::fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&f).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
    }
    let text = std::fs::read_to_string(&f).unwrap();
    let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let recv = lines.iter().find(|l| l["event"] == "request.received.openai").unwrap();
    assert_eq!(recv["user_agent"], "test-agent/1");
    assert_eq!(recv["client_ip"], "127.0.0.1");
    assert_eq!(recv["obj"]["model"], "deepseek-ai/DeepSeek-V4.1-Flash");
    assert!(recv["obj"]["messages"].is_array());
    assert_eq!(recv["orig"]["model"], "GLM-5.3-Flash");
    // 画像はハッシュに置き換える
    assert!(recv["obj"]["messages"][0]["content"][1]["image_url"]["url"]["sha256"].is_string());
    let fin = lines.iter().find(|l| l["event"] == "request.finished").unwrap();
    assert_eq!(fin["status"], 200);
    assert_eq!(fin["client_disconnected"], false);
    assert!(fin["dur_ms"].is_number());
}

#[tokio::test]
async fn api_keys() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| {
        c.auth.keys = vec![ds41gw::config::ApiKey {
            name: "k1".into(),
            sha256: ds41gw::util::sha256_hex(b"secret-key"),
            max_concurrency: Some(2),
        }];
        c.auth.allow_without_key = vec![];
    })
    .await;
    let c = client();
    let body = json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]});
    let r = c.post(format!("{}/v1/chat/completions", px.url)).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 401);
    assert_eq!(r.json::<Value>().await.unwrap()["error"]["type"], "authentication_error");
    let r = c.post(format!("{}/v1/chat/completions", px.url)).bearer_auth("secret-key").json(&body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let r = c
        .post(format!("{}/v1/messages", px.url))
        .header("x-api-key", "secret-key")
        .json(&json!({"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let r = c
        .post(format!("{}/v1/messages", px.url))
        .header("x-api-key", "wrong")
        .json(&json!({"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);
    assert_eq!(r.json::<Value>().await.unwrap()["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn models_and_metrics() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| c.cors.allow_origins = vec!["http://ui.local".into()]).await;
    let c = client();
    let v: Value = c.get(format!("{}/v1/models", px.url)).send().await.unwrap().json().await.unwrap();
    let ids: Vec<&str> = v["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"deepseek-ai/DeepSeek-V4.1-Flash") && ids.contains(&"GLM-5.3-Flash"));
    assert_eq!(v["data"][0]["object"], "model");
    assert_eq!(v["data"][0]["type"], "model");
    assert!(v["data"][0]["display_name"].is_string());
    assert!(v["data"][0]["created_at"].is_string());
    assert_eq!(v["has_more"], false);
    let r = c.get(format!("{}/v1/models/GLM-5.3-Flash", px.url)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let r = c.get(format!("{}/v1/models/nope", px.url)).send().await.unwrap();
    assert_eq!(r.status(), 404);
    let _ = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    let m = c.get(format!("{}/metrics", px.url)).send().await.unwrap().text().await.unwrap();
    assert!(m.contains("ds41gw_requests_total{api=\"chat\",stream=\"false\",status=\"200\"} 1"), "{m}");
    assert!(m.contains("ds41gw_queue_wait_seconds_count"));
    let r = c.get(format!("{}/health/ready", px.url)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    // CORS
    let r = c.request(reqwest::Method::OPTIONS, format!("{}/v1/chat/completions", px.url)).header("origin", "http://ui.local").send().await.unwrap();
    assert_eq!(r.status(), 204);
    assert_eq!(r.headers()["access-control-allow-origin"], "http://ui.local");
}
