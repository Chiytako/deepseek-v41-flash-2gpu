//! /v1/responses の結合試験 (非ストリーム・ストリーム、ツール往復、状態保持、イベント順序、エラー)。

mod common;

use common::*;
use serde_json::{json, Value};

fn tools() -> Value {
    json!([
        {"type": "function", "name": "get_weather", "description": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}, "strict": true},
        {"type": "function", "name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}
    ])
}

async fn post(px: &Proxy, b: Value) -> reqwest::Response {
    client().post(format!("{}/v1/responses", px.url)).json(&b).send().await.unwrap()
}

#[tokio::test]
async fn nonstream_basic_and_fields() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = post(&px, json!({"model": "GLM-5.3-Flash", "instructions": "You are terse.", "input": "hello", "max_output_tokens": 300,
        "metadata": {"k": "v"}})).await;
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["object"], "response");
    assert!(v["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(v["status"], "completed");
    assert_eq!(v["model"], "GLM-5.3-Flash");
    assert_eq!(v["instructions"], "You are terse.");
    assert_eq!(v["metadata"], json!({"k": "v"}));
    assert_eq!(v["store"], true);
    assert_eq!(v["max_output_tokens"], 300);
    assert_eq!(v["truncation"], "disabled");
    assert_eq!(v["background"], false);
    assert_eq!(v["reasoning"]["effort"], "high");
    let out = v["output"].as_array().unwrap();
    assert_eq!(out[0]["type"], "reasoning");
    assert_eq!(out[0]["content"][0]["text"], "Let me think.");
    assert_eq!(out[1]["type"], "message");
    assert_eq!(out[1]["content"][0]["text"], "Hello world");
    assert_eq!(v["usage"]["input_tokens"], 100);
    assert_eq!(v["usage"]["input_tokens_details"]["cached_tokens"], 40);
    // openai SDK 3.x の ResponseUsage は cache_write_tokens を必須にしている
    assert_eq!(v["usage"]["input_tokens_details"]["cache_write_tokens"], 0);
    assert_eq!(v["usage"]["output_tokens"], 20);
    assert!(v["usage"]["output_tokens_details"]["reasoning_tokens"].as_u64().unwrap() > 0);
    let b = mock.last_body();
    assert_eq!(b["messages"][0], json!({"role": "system", "content": "You are terse."}));
    assert_eq!(b["messages"][1], json!({"role": "user", "content": "hello"}));
    assert_eq!(b["max_tokens"], 300);
    assert_eq!(b["temperature"], 1.0);
    // GET と input_items と DELETE (§6.6)
    let id = v["id"].as_str().unwrap().to_string();
    let c = client();
    let g: Value = c.get(format!("{}/v1/responses/{id}", px.url)).send().await.unwrap().json().await.unwrap();
    assert_eq!(g["id"], id.as_str());
    assert_eq!(g["output"][1]["content"][0]["text"], "Hello world");
    let items: Value = c.get(format!("{}/v1/responses/{id}/input_items?order=asc", px.url)).send().await.unwrap().json().await.unwrap();
    assert_eq!(items["object"], "list");
    // OpenAI の ResponseItem の形 (type / status / content の配列) で返す
    assert_eq!(items["data"][0]["type"], "message");
    assert_eq!(items["data"][0]["role"], "user");
    assert_eq!(items["data"][0]["status"], "completed");
    assert_eq!(items["data"][0]["content"], json!([{"type": "input_text", "text": "hello"}]));
    assert!(items["data"][0]["id"].as_str().unwrap().starts_with("msg_"));
    let d: Value = c.delete(format!("{}/v1/responses/{id}", px.url)).send().await.unwrap().json().await.unwrap();
    assert_eq!(d["deleted"], true);
    let r = c.get(format!("{}/v1/responses/{id}", px.url)).send().await.unwrap();
    assert_eq!(r.status(), 404);
    let r = post(&px, json!({"model": "x", "previous_response_id": id, "input": "again"})).await;
    assert_eq!(r.status(), 400);
    let e: Value = r.json().await.unwrap();
    assert_eq!(e["error"]["code"], "previous_response_not_found");
    assert_eq!(e["error"]["param"], "previous_response_id");
}

#[tokio::test]
async fn previous_response_id_chain() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r1: Value = post(&px, json!({"model": "x", "instructions": "sys1", "input": "my name is Taro"})).await.json().await.unwrap();
    let r2: Value = post(&px, json!({"model": "x", "previous_response_id": r1["id"], "input": [{"role": "user", "content": "and I like tea"}]}))
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(r2["previous_response_id"], r1["id"]);
    // 省略形 ({"role","content"}、type 無し) の input も ResponseItem の形で返す
    let c = client();
    let items: Value = c.get(format!("{}/v1/responses/{}/input_items", px.url, r2["id"].as_str().unwrap())).send().await.unwrap().json().await.unwrap();
    assert_eq!(items["data"][0]["type"], "message");
    assert_eq!(items["data"][0]["content"], json!([{"type": "input_text", "text": "and I like tea"}]));
    let _r3: Value = post(&px, json!({"model": "x", "previous_response_id": r2["id"], "instructions": "sys3", "input": "what is my name?"}))
        .await
        .json()
        .await
        .unwrap();
    let b = mock.last_body();
    let m = b["messages"].as_array().unwrap();
    // instructions は引き継がない。履歴の思考と回答は残る
    assert_eq!(m[0], json!({"role": "system", "content": "sys3"}));
    assert_eq!(m[1]["content"], "my name is Taro");
    assert_eq!(m[2]["role"], "assistant");
    assert_eq!(m[2]["content"], "Hello world");
    assert_eq!(m[2]["reasoning_content"], "Let me think.");
    assert_eq!(m[3]["content"], "and I like tea");
    assert_eq!(m[5]["content"], "what is my name?");
    // store:false の応答は参照できない
    let r4: Value = post(&px, json!({"model": "x", "store": false, "input": "x"})).await.json().await.unwrap();
    assert_eq!(r4["store"], false);
    let r = post(&px, json!({"model": "x", "previous_response_id": r4["id"], "input": "y"})).await;
    assert_eq!(r.status(), 400);
}

#[tokio::test]
async fn stream_event_order_with_parallel_tools() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = post(&px, json!({"model": "x", "stream": true, "tools": tools(), "input": "#mock:tools weather in two cities"})).await;
    assert_eq!(r.status(), 200);
    let evs = read_sse(r).await;
    let ds = datas(&evs);
    // SSE の event 名と type が一致
    for (e, d) in evs.iter().filter(|e| e.0 != "#comment") {
        let v: Value = serde_json::from_str(d).unwrap();
        assert_eq!(e, v["type"].as_str().unwrap());
    }
    check_responses_events(&ds).unwrap();
    let types: Vec<&str> = ds.iter().map(|d| d["type"].as_str().unwrap()).collect();
    assert_eq!(types[0], "response.created");
    assert_eq!(types[1], "response.in_progress");
    assert_eq!(*types.last().unwrap(), "response.completed");
    assert!(types.contains(&"response.reasoning_text.delta"));
    assert!(!types.contains(&"response.reasoning_summary_text.delta"));
    let fin = ds.last().unwrap()["response"].clone();
    let out = fin["output"].as_array().unwrap();
    let kinds: Vec<&str> = out.iter().map(|o| o["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["reasoning", "message", "function_call", "function_call"]);
    assert_eq!(out[1]["phase"], "commentary");
    assert_eq!(out[2]["name"], "get_weather");
    assert_eq!(out[2]["arguments"], "{\"city\": \"Tokyo\"}");
    assert_eq!(out[3]["name"], "read");
    assert!(out[2]["call_id"].as_str().unwrap().starts_with("call_resp_"));
    assert!(out[2]["id"].as_str().unwrap().len() <= 64);
    assert_eq!(fin["usage"]["output_tokens"], 20);
    assert_eq!(fin["tools"], tools());
    // strict は上流に渡る
    let b = mock.last_body();
    assert_eq!(b["tools"][0]["function"]["strict"], true);
    assert!(b.get("stream_options").is_none());

    // 2 ターン目: 関数の結果を返す。reasoning 項目は content を削り encrypted_content だけ (T-R3)
    let r = post(&px, json!({"model": "x", "stream": true, "tools": tools(), "include": ["reasoning.encrypted_content"], "store": false,
        "input": "#mock:tools weather"})).await;
    let ds = datas(&read_sse(r).await);
    check_responses_events(&ds).unwrap();
    let out = ds.last().unwrap()["response"]["output"].as_array().unwrap().clone();
    let enc = out[0]["encrypted_content"].as_str().unwrap().to_string();
    assert!(enc.starts_with("ds41gw1."));
    let mut input = vec![json!({"role": "user", "content": "#mock:tools weather"})];
    input.push(json!({"type": "reasoning", "id": out[0]["id"], "summary": [], "encrypted_content": enc}));
    input.push(json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": out[1]["content"][0]["text"]}]}));
    for (k, fc) in out[2..].iter().enumerate() {
        input.push(json!({"type": "function_call", "call_id": fc["call_id"], "name": fc["name"], "arguments": fc["arguments"]}));
        let _ = k;
    }
    for (k, fc) in out[2..].iter().enumerate() {
        input.push(json!({"type": "function_call_output", "call_id": fc["call_id"], "output": format!("result {k}")}));
    }
    input.push(json!({"role": "user", "content": "thanks"}));
    let r = post(&px, json!({"model": "x", "tools": tools(), "input": input})).await;
    assert_eq!(r.status(), 200);
    let b = mock.last_body();
    let m = b["messages"].as_array().unwrap();
    assert_eq!(m[1]["role"], "assistant");
    assert_eq!(m[1]["reasoning_content"], "Let me think.");
    assert_eq!(m[1]["content"], "Checking.");
    assert_eq!(m[1]["tool_calls"].as_array().unwrap().len(), 2);
    assert_eq!(m[2]["role"], "tool");
    assert_eq!(m[3]["role"], "tool");
    assert_eq!(m[4]["content"], "thanks");
}

#[tokio::test]
async fn summary_mode_and_effort() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = post(&px, json!({"model": "x", "stream": true, "reasoning": {"effort": "xhigh", "summary": "auto"}, "input": "hi"})).await;
    let ds = datas(&read_sse(r).await);
    check_responses_events(&ds).unwrap();
    let types: Vec<&str> = ds.iter().map(|d| d["type"].as_str().unwrap()).collect();
    assert!(types.contains(&"response.reasoning_summary_part.added"));
    assert!(types.contains(&"response.reasoning_summary_text.delta"));
    assert!(types.contains(&"response.reasoning_summary_text.done"));
    assert!(types.contains(&"response.reasoning_summary_part.done"));
    // 同じ本文を 2 種類のイベントで流さない
    assert!(!types.contains(&"response.reasoning_text.delta"));
    let item = ds.iter().find(|d| d["type"] == "response.output_item.done" && d["item"]["type"] == "reasoning").unwrap();
    assert_eq!(item["item"]["summary"][0]["text"], "Let me think.");
    assert_eq!(item["item"]["summary"][0]["type"], "summary_text");
    assert_eq!(mock.last_body()["reasoning_effort"], 88, "xhigh は独自の対応で 88 (公式は 75)");
    // effort none → thinking off
    let r = post(&px, json!({"model": "x", "reasoning": {"effort": "none"}, "input": "hi"})).await;
    let v: Value = r.json().await.unwrap();
    assert_eq!(mock.last_body()["thinking"]["type"], "disabled");
    assert_eq!(v["output"][0]["type"], "message");
    // effort low
    let _ = post(&px, json!({"model": "x", "reasoning": {"effort": "LOW"}, "input": "hi"})).await;
    assert_eq!(mock.last_body()["reasoning_effort"], 50);
}

#[tokio::test]
async fn text_format_json_schema_and_tool_choice() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let schema = json!({"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": false});
    let r = post(&px, json!({"model": "x", "input": "#mock:json give json", "text": {"format": {"type": "json_schema", "name": "ans", "schema": schema, "strict": true}}})).await;
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["text"]["format"]["type"], "json_schema");
    let b = mock.last_body();
    assert_eq!(b["response_format"]["type"], "json_schema");
    assert_eq!(b["response_format"]["json_schema"]["name"], "ans");
    assert_eq!(b["response_format"]["json_schema"]["strict"], true);
    assert_eq!(b["response_format"]["json_schema"]["schema"], schema);
    // tool_choice の named と required は thinking のままで上流に渡す (ds41rt と同じ)
    let _ = post(&px, json!({"model": "x", "tools": tools(), "tool_choice": {"type": "function", "name": "read"}, "input": "hi"})).await;
    let b = mock.last_body();
    assert_eq!(b["tool_choice"]["function"]["name"], "read");
    assert_eq!(b["thinking"]["type"], "enabled");
    let _ = post(&px, json!({"model": "x", "tools": tools(), "tool_choice": {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "get_weather"}]}, "input": "hi"})).await;
    let b = mock.last_body();
    assert_eq!(b["tool_choice"], "required");
    assert_eq!(b["tools"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn rejects_builtin_and_unsupported() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = post(&px, json!({"model": "x", "input": "hi", "tools": [{"type": "web_search"}]})).await;
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["code"], "unsupported_tool");
    assert_eq!(v["error"]["param"], "tools[0].type");
    let r = post(&px, json!({"model": "x", "input": "hi", "background": true})).await;
    assert_eq!(r.status(), 400);
    let r = post(&px, json!({"model": "x", "input": "hi", "include": ["message.output_text.logprobs"]})).await;
    assert_eq!(r.status(), 400);
    let r = post(&px, json!({"model": "x", "input": [{"role": "user", "content": [{"type": "input_file", "file_data": "x"}]}]})).await;
    assert_eq!(r.status(), 400);
    // 結果の無い呼び出し・呼び出しの無い結果・重複は補修して送る (recipe はそのままだと 400 にする)
    let r = post(&px, json!({"model": "x", "tools": tools(), "input": [
        {"role": "user", "content": "q"},
        {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
        {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "zz", "output": "orphan"},
        {"role": "user", "content": "next"}]})).await;
    let st = r.status();
    let warn = r.headers().get("x-ds41gw-warning").map(|v| v.to_str().unwrap().to_string()).unwrap_or_default();
    let body = r.text().await.unwrap();
    assert_eq!(st, 200, "{body}");
    assert!(warn.contains("repaired_tool_history"), "{warn}");
    let m = mock.last_body()["messages"].as_array().unwrap().clone();
    let roles: Vec<&str> = m.iter().map(|x| x["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "user", "user"]);
    assert_eq!(m[1]["tool_calls"].as_array().unwrap().len(), 1);
    assert!(m[2]["content"].as_str().unwrap().contains("no result"));
    assert!(m[3]["content"].as_str().unwrap().starts_with("[tool result zz]"));
    let r = client().post(format!("{}/v1/responses/resp_x/cancel", px.url)).send().await.unwrap();
    assert_eq!(r.status(), 400);
}

#[tokio::test]
async fn namespace_tools_and_images() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let tools = json!([{"type": "namespace", "name": "fs", "description": "files", "tools": [
        {"type": "function", "name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]}]);
    let img = "data:image/png;base64,iVBORw0KGgo=";
    let r = post(&px, json!({"model": "x", "tools": tools, "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "#mock:tools look"}, {"type": "input_image", "image_url": img}]},
        {"type": "function_call", "call_id": "c0", "name": "read", "namespace": "fs", "arguments": "{\"path\":\"a\"}"},
        {"type": "function_call_output", "call_id": "c0", "output": [{"type": "input_text", "text": "see"}, {"type": "input_image", "image_url": img}]}
    ]})).await;
    let st = r.status();
    let v: Value = r.json().await.unwrap();
    assert_eq!(st, 200, "{v}");
    let b = mock.last_body();
    assert_eq!(b["tools"][0]["function"]["name"], "fs__read");
    let m = b["messages"].as_array().unwrap();
    assert_eq!(m[0]["content"][0]["text"], "#mock:tools look");
    assert_eq!(m[0]["content"][1]["image_url"]["url"], img);
    assert_eq!(m[1]["tool_calls"][0]["function"]["name"], "fs__read");
    // function_call_output の画像は tool メッセージに入れる (§1.2)
    assert_eq!(m[2]["role"], "tool");
    assert_eq!(m[2]["content"][0]["text"], "see");
    assert_eq!(m[2]["content"][1]["type"], "image_url");
    // 出力は namespace に戻す
    let fc: Vec<&Value> = v["output"].as_array().unwrap().iter().filter(|o| o["type"] == "function_call").collect();
    assert_eq!(fc[0]["name"], "read");
    assert_eq!(fc[0]["namespace"], "fs");
}

#[tokio::test]
async fn failures_and_incomplete() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = post(&px, json!({"model": "x", "stream": true, "input": "#mock:midfail"})).await;
    let ds = datas(&read_sse(r).await);
    check_responses_events(&ds).unwrap();
    let last = ds.last().unwrap();
    assert_eq!(last["type"], "response.failed");
    assert_eq!(last["response"]["status"], "failed");
    assert_eq!(last["response"]["error"]["code"], "server_error");
    let r = post(&px, json!({"model": "x", "input": "#mock:midfail"})).await;
    assert_eq!(r.status(), 502);
    let r = post(&px, json!({"model": "x", "stream": true, "input": "#mock:length"})).await;
    let ds = datas(&read_sse(r).await);
    check_responses_events(&ds).unwrap();
    let last = ds.last().unwrap();
    assert_eq!(last["type"], "response.incomplete");
    assert_eq!(last["response"]["incomplete_details"]["reason"], "max_output_tokens");
    let r = post(&px, json!({"model": "x", "input": "#mock:length"})).await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap()["status"], "incomplete");
    // 早期ヘッダの後に上流が失敗したら response.created → response.failed
    let px2 = start_proxy(&mock, |c| {
        c.client.early_headers_after_s = 0.05;
        c.upstream.max_concurrency = 1;
        c.queue.timeout_s = 0.3;
    })
    .await;
    let slow = client().post(format!("{}/v1/chat/completions", px2.url)).json(&json!({"model": "x", "stream": true,
        "messages": [{"role": "user", "content": "#mock:slow"}]})).send().await.unwrap();
    let r = client().post(format!("{}/v1/responses", px2.url)).json(&json!({"model": "x", "stream": true, "input": "hi"})).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let ds = datas(&read_sse(r).await);
    check_responses_events(&ds).unwrap();
    assert_eq!(ds[0]["type"], "response.created");
    assert_eq!(ds.last().unwrap()["type"], "response.failed");
    drop(slow);
}

#[tokio::test]
async fn sglang_encrypted_reasoning_is_decoded() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    // python: "sglang-reasoning-v1." + base64.urlsafe_b64encode(zlib.compress(b"hello world")).decode()
    let enc = "sglang-reasoning-v1.eJzLSM3JyVcozy_KSQEAGgsEXQ==";
    let r = post(&px, json!({"model": "x", "input": [
        {"role": "user", "content": "q"},
        {"type": "reasoning", "summary": [], "encrypted_content": enc},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a"}]},
        {"role": "user", "content": "q2"}
    ]})).await;
    assert_eq!(r.status(), 200);
    let b = mock.last_body();
    assert_eq!(b["messages"][1]["reasoning_content"], "hello world");
    assert_eq!(b["messages"][1]["content"], "a");
    // 改ざんされた自前形式は空として扱い、会話は止めない
    let r = post(&px, json!({"model": "x", "input": [
        {"role": "user", "content": "q"},
        {"type": "reasoning", "summary": [], "encrypted_content": "ds41gw1.AAAA.BBBB"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "zzz"}]},
        {"role": "user", "content": "q2"}
    ]})).await;
    assert_eq!(r.status(), 200);
    assert!(r.headers()["x-ds41gw-warning"].to_str().unwrap().contains("bad_encrypted_content"));
}

/// 実要求の再生 (r7_replay_real.md §4.1) で見つかった不一致: include encrypted_content で、思考の項目の done と
/// response.completed の間で秒をまたぐと、encrypted_content (ts 入り) が作り直されて食い違っていた。
#[tokio::test]
async fn encrypted_content_is_identical_in_done_and_completed() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    for summary in [Value::Null, json!("auto")] {
        let r = post(&px, json!({"model": "x", "stream": true, "tools": tools(), "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "high", "summary": summary}, "input": "#mock:slowtools weather"})).await;
        assert_eq!(r.status(), 200);
        let ds = datas(&read_sse(r).await);
        check_responses_events(&ds).unwrap();
        let fin = &ds.last().unwrap()["response"];
        let enc = fin["output"][0]["encrypted_content"].as_str().unwrap();
        assert!(enc.starts_with("ds41gw1."));
        // 保存された応答 (GET) も同じ
        let id = fin["id"].as_str().unwrap();
        let g: Value = client().get(format!("{}/v1/responses/{id}", px.url)).send().await.unwrap().json().await.unwrap();
        assert_eq!(g["output"][0]["encrypted_content"], fin["output"][0]["encrypted_content"]);
    }
}
