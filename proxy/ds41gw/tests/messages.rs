//! /v1/messages、/v1/messages/count_tokens、/v1/completions の結合試験。

mod common;

use common::*;
use serde_json::{json, Value};

async fn post(px: &Proxy, path: &str, b: Value) -> reqwest::Response {
    client().post(format!("{}{path}", px.url)).header("anthropic-version", "2023-06-01").json(&b).send().await.unwrap()
}

fn tools() -> Value {
    json!([{"name": "get_weather", "description": "weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}},
           {"name": "read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}])
}

#[tokio::test]
async fn messages_nonstream_with_tools_and_thinking() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = post(&px, "/v1/messages", json!({"model": "claude-sonnet-x", "max_tokens": 1000, "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "thinking": {"type": "enabled", "budget_tokens": 2048}, "tools": tools(),
        "messages": [{"role": "user", "content": "#mock:tools weather?"}]})).await;
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["model"], "claude-sonnet-x");
    assert!(v["id"].as_str().unwrap().starts_with("msg_"));
    let c = v["content"].as_array().unwrap();
    assert_eq!(c[0]["type"], "thinking");
    assert_eq!(c[0]["thinking"], "Let me think.");
    assert!(c[0]["signature"].as_str().unwrap().starts_with("ds41gw-sig-v1."));
    assert_eq!(c[1], json!({"type": "text", "text": "Checking."}));
    assert_eq!(c[2]["type"], "tool_use");
    assert_eq!(c[2]["name"], "get_weather");
    assert_eq!(c[2]["input"], json!({"city": "Tokyo"}));
    assert_eq!(c[3]["name"], "read");
    assert_eq!(v["stop_reason"], "tool_use");
    assert_eq!(v["usage"]["input_tokens"], 60);
    assert_eq!(v["usage"]["cache_read_input_tokens"], 40);
    assert_eq!(v["usage"]["output_tokens"], 20);
    let b = mock.last_body();
    assert_eq!(b["reasoning_effort"], 50);
    assert_eq!(b["max_tokens"], 1000);
    assert_eq!(b["messages"][0], json!({"role": "system", "content": "sys"}));
    assert_eq!(b["tools"][0]["function"]["parameters"]["type"], "object");

    // 2 ターン目: tool_result (is_error あり) と thinking を返す
    let tid0 = c[2]["id"].as_str().unwrap();
    let tid1 = c[3]["id"].as_str().unwrap();
    let r = post(&px, "/v1/messages", json!({"model": "claude-sonnet-x", "max_tokens": 100, "tools": tools(),
        "tool_choice": {"type": "any", "disable_parallel_tool_use": true},
        "messages": [
            {"role": "user", "content": "#mock:tools weather?"},
            {"role": "assistant", "content": [c[0].clone(), c[1].clone(), c[2].clone(), c[3].clone(), {"type": "redacted_thinking", "data": "x"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid0, "content": "sunny"},
                {"type": "tool_result", "tool_use_id": tid1, "is_error": true, "content": [{"type": "text", "text": "no file"}]},
                {"type": "text", "text": "and?"}]}
        ]})).await;
    let st = r.status();
    let v2: Value = r.json().await.unwrap();
    assert_eq!(st, 200, "{v2}");
    let b = mock.last_body();
    let m = b["messages"].as_array().unwrap();
    assert_eq!(m[1]["reasoning_content"], "Let me think.");
    assert_eq!(m[1]["tool_calls"][0]["function"]["arguments"], "{\"city\": \"Tokyo\"}");
    assert_eq!(m[2]["content"], "sunny");
    assert_eq!(m[3]["content"], "[error] no file");
    assert_eq!(m[4]["content"], "and?");
    assert_eq!(b["tool_choice"], "required");
    assert_eq!(b["parallel_tool_calls"], false);
}

#[tokio::test]
async fn messages_stream_events_and_stop_sequence() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 50, "stream": true, "stop_sequences": ["END"],
        "messages": [{"role": "user", "content": "#mock:stopseq"}]})).await;
    assert_eq!(r.status(), 200);
    let evs = read_sse(r).await;
    let names: Vec<&str> = evs.iter().map(|e| e.0.as_str()).collect();
    assert_eq!(names[0], "message_start");
    assert_eq!(names[1], "content_block_start");
    assert_eq!(names[2], "ping");
    assert_eq!(*names.last().unwrap(), "message_stop");
    let ds = datas(&evs);
    // block は逐次に開いて閉じる
    let mut open: Option<u64> = None;
    for d in &ds {
        match d["type"].as_str().unwrap() {
            "content_block_start" => {
                assert!(open.is_none());
                open = d["index"].as_u64();
            }
            "content_block_delta" => assert_eq!(open, d["index"].as_u64()),
            "content_block_stop" => {
                assert_eq!(open, d["index"].as_u64());
                open = None;
            }
            _ => {}
        }
    }
    let sig = ds.iter().find(|d| d["delta"]["type"] == "signature_delta").unwrap();
    assert!(sig["delta"]["signature"].as_str().unwrap().starts_with("ds41gw-sig-v1."));
    let text: String = ds.iter().filter_map(|d| d["delta"]["text"].as_str()).collect();
    assert_eq!(text, "Hello world ");
    let md = ds.iter().find(|d| d["type"] == "message_delta").unwrap();
    assert_eq!(md["delta"]["stop_reason"], "stop_sequence");
    assert_eq!(md["delta"]["stop_sequence"], "END");
    assert!(mock.last_body().get("stop").is_none());
    // 非ストリームでも同じ
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 50, "stop_sequences": ["END"],
        "messages": [{"role": "user", "content": "#mock:stopseq"}]})).await;
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["stop_reason"], "stop_sequence");
    assert_eq!(v["stop_sequence"], "END");
    assert_eq!(v["content"][1]["text"], "Hello world ");
}

#[tokio::test]
async fn messages_errors_and_thinking_off() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = post(&px, "/v1/messages", json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 10, "tools": [{"type": "bash_20250124", "name": "bash"}],
        "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(r.status(), 400);
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 10, "messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}]})).await;
    assert_eq!(r.status(), 400);
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 10, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "hi"}]})).await;
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(mock.last_body()["thinking"]["type"], "disabled");
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 10, "stream": true, "messages": [{"role": "user", "content": "#mock:midfail"}]})).await;
    let evs = read_sse(r).await;
    assert_eq!(evs.last().unwrap().0, "error");
    let e: Value = serde_json::from_str(&evs.last().unwrap().1).unwrap();
    assert_eq!(e["error"]["type"], "api_error");
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 10, "messages": [{"role": "user", "content": "#mock:midfail"}]})).await;
    assert_eq!(r.status(), 500);
    // 画像 (base64)
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 10, "top_k": 20, "output_config": {"effort": "max"}, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}, {"type": "text", "text": "what?"}]}]})).await;
    assert_eq!(r.status(), 200);
    let b = mock.last_body();
    assert_eq!(b["top_k"], 20);
    assert_eq!(b["reasoning_effort"], 100);
    assert_eq!(b["messages"][0]["content"][0]["image_url"]["url"], "data:image/png;base64,iVBORw0KGgo=");
    assert_eq!(b["messages"][1]["content"], "what?");
}

#[tokio::test]
async fn count_tokens() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let before = mock.st.calls.load(std::sync::atomic::Ordering::SeqCst);
    let r = post(&px, "/v1/messages/count_tokens", json!({"model": "x", "system": "You are helpful.", "tools": tools(),
        "messages": [{"role": "user", "content": "How many tokens is this?"}]})).await;
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert!(v["input_tokens"].as_u64().unwrap() > 50, "{v}");
    assert_eq!(mock.st.calls.load(std::sync::atomic::Ordering::SeqCst), before);
}

#[tokio::test]
async fn completions_emulated() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = post(&px, "/v1/completions", json!({"model": "x", "prompt": "Once upon a time"})).await;
    assert_eq!(r.status(), 200);
    assert!(r.headers()["x-ds41gw-warning"].to_str().unwrap().contains("completions_emulated_via_chat"));
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"][0]["text"], "Hello world");
    assert!(v["id"].as_str().unwrap().starts_with("cmpl-"));
    let b = mock.last_body();
    assert_eq!(b["thinking"]["type"], "disabled");
    assert_eq!(b["max_tokens"], 16);
    assert_eq!(b["messages"][0], json!({"role": "user", "content": "Once upon a time"}));
    // 複数プロンプトと stream
    let r = post(&px, "/v1/completions", json!({"model": "x", "prompt": ["a", "b"], "stream": true, "max_tokens": 5})).await;
    let evs = read_sse(r).await;
    assert_eq!(evs.last().unwrap().1, "[DONE]");
    let ds = datas(&evs);
    let idx: std::collections::HashSet<u64> = ds.iter().filter_map(|d| d["choices"][0]["index"].as_u64()).collect();
    assert_eq!(idx.len(), 2);
    assert!(ds.iter().all(|d| d["object"] == "text_completion"));
    let r = post(&px, "/v1/completions", json!({"model": "x", "prompt": "a", "echo": true})).await;
    assert_eq!(r.status(), 400);
    let r = post(&px, "/v1/completions", json!({"model": "x", "prompt": [1, 2, 3]})).await;
    assert_eq!(r.status(), 400);
}

#[tokio::test]
async fn messages_structured_output_maps_to_response_format() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": false});
    // output_config.format (anthropic SDK の messages.parse(output_format=...) が送る形)
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 50, "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(r.status(), 200);
    let b = mock.last_body();
    assert_eq!(b["response_format"], json!({"type": "json_schema", "json_schema": {"name": "output", "schema": schema, "strict": true}}));
    assert_eq!(b["reasoning_effort"], 50);
    assert!(b.get("output_config").is_none());
    // 旧 beta の output_format
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 50, "output_format": {"type": "json_schema", "schema": schema},
        "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(r.status(), 200);
    let b = mock.last_body();
    assert_eq!(b["response_format"]["json_schema"]["schema"], schema);
    assert!(b.get("output_format").is_none());
    // 不正な format は 400 (黙って捨てない)
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 50, "output_config": {"format": {"type": "xml"}},
        "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    let r = post(&px, "/v1/messages", json!({"model": "x", "max_tokens": 50, "output_config": {"format": {"type": "json_schema"}},
        "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(r.status(), 400);
}
