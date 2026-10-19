//! 敵対的レビュー (r7) で見つけた不具合の再現と、ストリームの端の場合の検査。
//! 上流は tests/common の mock。

mod common;

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use common::*;
use serde_json::{json, Value};

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

fn tools() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])
}

fn resp_tools() -> Value {
    json!([{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}])
}

fn msg_tools() -> Value {
    json!([{"name": "get_weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}])
}

/// Anthropic のイベント順序: message_start → (block start → delta* → stop)* → message_delta → message_stop。
fn check_messages_events(evs: &[(String, String)]) -> Result<(), String> {
    let names: Vec<&str> = evs.iter().map(|e| e.0.as_str()).filter(|n| *n != "ping" && *n != "#comment").collect();
    if names.first() != Some(&"message_start") {
        return Err(format!("first event {:?}", names.first()));
    }
    if names.last() != Some(&"message_stop") {
        return Err(format!("last event {:?}", names.last()));
    }
    let mut open: Option<u64> = None;
    let mut next_index = 0u64;
    let mut saw_delta = false;
    for (n, d) in evs.iter().filter(|e| e.0 != "ping" && e.0 != "#comment") {
        let v: Value = serde_json::from_str(d).map_err(|e| format!("bad json {e}"))?;
        if v["type"] != n.as_str() {
            return Err(format!("event name {n} != type {}", v["type"]));
        }
        match n.as_str() {
            "content_block_start" => {
                if open.is_some() || saw_delta {
                    return Err("block start while open / after message_delta".into());
                }
                let i = v["index"].as_u64().unwrap();
                if i != next_index {
                    return Err(format!("block index {i}, expected {next_index}"));
                }
                next_index += 1;
                open = Some(i);
            }
            "content_block_delta" => {
                if open != v["index"].as_u64() {
                    return Err("delta outside open block".into());
                }
            }
            "content_block_stop" => {
                if open != v["index"].as_u64() {
                    return Err("stop for block that is not open".into());
                }
                open = None;
            }
            "message_delta" => {
                if open.is_some() {
                    return Err("message_delta while a block is open".into());
                }
                saw_delta = true;
            }
            _ => {}
        }
    }
    if !saw_delta {
        return Err("no message_delta".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 待ち行列

/// 待っている間にクライアントが切ると、待ち人数 (waiting) が減らずに残っていた。
#[tokio::test]
async fn queue_waiting_is_released_when_client_cancels() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| {
        c.upstream.max_concurrency = 1;
        c.queue.max_waiting = 2;
    })
    .await;
    let c = client();
    let slow = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "stream": true, "messages": [{"role": "user", "content": "#mock:slow"}]}))
        .send()
        .await
        .unwrap();
    let short = reqwest::Client::builder().timeout(Duration::from_millis(200)).build().unwrap();
    for _ in 0..3 {
        let r = short
            .post(format!("{}/v1/chat/completions", px.url))
            .json(&json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]}))
            .send()
            .await;
        assert!(r.is_err(), "queued request should have timed out on the client");
    }
    let up = px.state.up.clone();
    assert!(wait_for(|| up.waiting.load(Ordering::Relaxed) == 0, 1000).await, "waiting leaked: {}", up.waiting.load(Ordering::Relaxed));
    drop(slow);
    // 枠が空けば普通に通る (queue_full にならない)
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
}

/// n 本のストリームは n 枠をまとめて取る。上限を超える本数は 400 (待ち続けて全体を塞がない)。
#[tokio::test]
async fn stream_n_slots_are_acquired_atomically() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| {
        c.upstream.max_concurrency = 3;
        c.queue.timeout_s = 5.0;
    })
    .await;
    let c = client();
    // completions: プロンプト 4 本 × n 1 = 4 本 > 3 枠 → 400
    let t0 = Instant::now();
    let r = c
        .post(format!("{}/v1/completions", px.url))
        .json(&json!({"model": "x", "stream": true, "prompt": ["a", "b", "c", "d"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "{:?}", r.text().await);
    assert!(t0.elapsed() < Duration::from_secs(2));
    // 1 本が枠を持っている間に n=2 を 2 本: 交互に枠を取り合って詰まらないこと
    let hold = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "stream": true, "messages": [{"role": "user", "content": "#mock:work"}]}))
        .send();
    let mk = || {
        c.post(format!("{}/v1/chat/completions", px.url))
            .json(&json!({"model": "x", "stream": true, "n": 2, "temperature": 1.0, "messages": [{"role": "user", "content": "#mock:work"}]}))
            .send()
    };
    let (h, a, b) = tokio::join!(hold, mk(), mk());
    for r in [h, a, b] {
        let r = r.unwrap();
        assert_eq!(r.status(), 200);
        let evs = read_sse(r).await;
        assert_eq!(evs.last().unwrap().1, "[DONE]");
    }
    assert!(mock.st.max_inflight.load(Ordering::SeqCst) <= 3);
}

/// ds41rt が応答ヘッダを返さないまま固まっても、idle_timeout で諦める。
#[tokio::test]
async fn upstream_header_timeout() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| c.upstream.idle_timeout_s = 0.3).await;
    let t0 = Instant::now();
    let r = client()
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "#mock:hang_headers"}]}))
        .send()
        .await
        .unwrap();
    assert!(r.status().is_server_error(), "{}", r.status());
    assert!(t0.elapsed() < Duration::from_millis(2500), "{:?}", t0.elapsed());
    assert_eq!(px.state.up.inflight.load(Ordering::Relaxed), 0);
}

// ---------------------------------------------------------------------------
// 認証・ヘッダ

#[tokio::test]
async fn auth_accepts_either_header_and_blocks_cross_site_simple_posts() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| {
        c.auth.keys = vec![ds41gw::config::ApiKey { name: "k1".into(), sha256: ds41gw::util::sha256_hex(b"secret-key"), max_concurrency: None }];
        c.auth.allow_without_key = vec![];
    })
    .await;
    let c = client();
    // Authorization にダミー、x-api-key に本物 (Anthropic 系のクライアントでありうる)
    let r = c
        .post(format!("{}/v1/messages", px.url))
        .bearer_auth("dummy")
        .header("x-api-key", "secret-key")
        .json(&json!({"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    // GET /v1/responses/:id も鍵が要る
    let r = c.get(format!("{}/v1/responses/resp_x", px.url)).send().await.unwrap();
    assert_eq!(r.status(), 401);

    // 鍵なしで動く既定の構成: ブラウザの別サイトからの "simple request" (text/plain) は断る
    let px2 = start_proxy(&mock, |_| {}).await;
    let body = json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let r = c
        .post(format!("{}/v1/chat/completions", px2.url))
        .header("origin", "http://evil.example")
        .header("content-type", "text/plain")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    let calls = mock.st.calls.load(Ordering::SeqCst);
    // Origin の無い普通のクライアントは content-type を問わない
    let r = c.post(format!("{}/v1/chat/completions", px2.url)).header("content-type", "text/plain").body(body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(mock.st.calls.load(Ordering::SeqCst), calls + 1);
}

#[tokio::test]
async fn warning_header_is_bounded() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let parts: Vec<Value> = (0..200).map(|i| json!({"type": format!("weird_{i}_{}", "x".repeat(200)), "v": i})).collect();
    let r = client()
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": parts}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let w = r.headers().get("x-ds41gw-warning").map(|v| v.len()).unwrap_or(0);
    assert!(w > 0 && w <= 4096, "warning header length {w}");
}

// ---------------------------------------------------------------------------
// ストリームの端の場合

#[tokio::test]
async fn chat_edge_streams() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let c = client();
    let post = |content: &str, stream: bool| {
        c.post(format!("{}/v1/chat/completions", px.url))
            .json(&json!({"model": "x", "stream": stream, "tools": tools(), "messages": [{"role": "user", "content": content}]}))
            .send()
    };
    // 思考だけで終わる
    let ds = datas(&read_sse(post("#mock:thinkonly", true).await.unwrap()).await);
    let reasoning: String = ds.iter().filter_map(|d| d["choices"][0]["delta"]["reasoning_content"].as_str()).collect();
    assert_eq!(reasoning, "Let me think.");
    assert_eq!(ds.last().unwrap()["choices"][0]["finish_reason"], "length");
    let v: Value = post("#mock:thinkonly", false).await.unwrap().json().await.unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "");
    assert_eq!(v["choices"][0]["message"]["reasoning_content"], "Let me think.");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    // 1 チャンクに 2 本
    let ds = datas(&read_sse(post("#mock:multitool_chunk", true).await.unwrap()).await);
    let tcs: Vec<&Value> = ds.iter().filter_map(|d| d["choices"][0]["delta"]["tool_calls"].as_array()).flatten().collect();
    assert_eq!(tcs.iter().filter(|t| t.get("id").is_some()).count(), 2);
    let v: Value = post("#mock:multitool_chunk", false).await.unwrap().json().await.unwrap();
    let tc = v["choices"][0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(tc.len(), 2);
    assert_eq!(tc[0]["function"]["arguments"], "{\"city\": \"Tokyo\"}");
    assert_eq!(tc[1]["function"]["arguments"], "{\"city\": \"Osaka\"}");
    // 引数の途中で上流が切れる
    let evs = read_sse(post("#mock:midfail_tool", true).await.unwrap()).await;
    let last: Value = serde_json::from_str(&evs.last().unwrap().1).unwrap();
    assert_eq!(last["error"]["code"], "upstream_disconnected");
    let r = post("#mock:midfail_tool", false).await.unwrap();
    assert_eq!(r.status(), 502);
}

#[tokio::test]
async fn responses_edge_streams() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let c = client();
    let post = |input: &str| {
        c.post(format!("{}/v1/responses", px.url))
            .json(&json!({"model": "x", "stream": true, "tools": resp_tools(), "input": input}))
            .send()
    };
    for (sc, status, kinds) in [
        ("#mock:thinkonly", "incomplete", vec!["reasoning"]),
        ("#mock:multitool_chunk", "completed", vec!["reasoning", "function_call", "function_call"]),
        ("#mock:nothink", "completed", vec!["message"]),
    ] {
        let ds = datas(&read_sse(post(sc).await.unwrap()).await);
        check_responses_events(&ds).map_err(|e| format!("{sc}: {e}")).unwrap();
        let fin = &ds.last().unwrap()["response"];
        assert_eq!(fin["status"], status, "{sc}");
        let k: Vec<&str> = fin["output"].as_array().unwrap().iter().map(|o| o["type"].as_str().unwrap()).collect();
        assert_eq!(k, kinds, "{sc}");
        if sc == "#mock:multitool_chunk" {
            assert_eq!(fin["output"][1]["arguments"], "{\"city\": \"Tokyo\"}");
            assert_eq!(fin["output"][2]["arguments"], "{\"city\": \"Osaka\"}");
        }
    }
    // 引数の途中で切れる → response.failed で終わり、開いていた呼び出しは output に入れない
    let ds = datas(&read_sse(post("#mock:midfail_tool").await.unwrap()).await);
    let last = ds.last().unwrap();
    assert_eq!(last["type"], "response.failed");
    let seqs: Vec<u64> = ds.iter().map(|d| d["sequence_number"].as_u64().unwrap()).collect();
    assert_eq!(seqs, (0..ds.len() as u64).collect::<Vec<_>>());
    assert!(last["response"]["output"].as_array().unwrap().iter().all(|o| o["type"] != "function_call"));
}

/// response.completed を受けた直後に切っても、次のターンで previous_response_id が引ける。
#[tokio::test]
async fn responses_stored_before_terminal_event() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| c.state.backend = "memory".into()).await;
    let c = client();
    for _ in 0..5 {
        let mut r = c
            .post(format!("{}/v1/responses", px.url))
            .json(&json!({"model": "x", "stream": true, "input": "hello"}))
            .send()
            .await
            .unwrap();
        let mut buf = String::new();
        let id = loop {
            let ch = r.chunk().await.unwrap().unwrap();
            buf.push_str(&String::from_utf8_lossy(&ch));
            if let Some(p) = buf.find("event: response.completed") {
                let rest = &buf[p..];
                if let Some(e) = rest.find("\n\n") {
                    let data = rest[..e].lines().find_map(|l| l.strip_prefix("data: ")).unwrap();
                    let v: Value = serde_json::from_str(data).unwrap();
                    break v["response"]["id"].as_str().unwrap().to_string();
                }
            }
        };
        drop(r);
        let r = c
            .post(format!("{}/v1/responses", px.url))
            .json(&json!({"model": "x", "previous_response_id": id, "input": "again"}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    }
}

#[tokio::test]
async fn messages_edge_streams() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let c = client();
    let post = |content: &str| {
        c.post(format!("{}/v1/messages", px.url))
            .json(&json!({"model": "x", "max_tokens": 50, "stream": true, "tools": msg_tools(), "messages": [{"role": "user", "content": content}]}))
            .send()
    };
    for (sc, stop, kinds) in [
        ("#mock:thinkonly", "max_tokens", vec!["thinking"]),
        ("#mock:multitool_chunk", "tool_use", vec!["thinking", "tool_use", "tool_use"]),
        // 思考 on で思考が空でも、recipe は空の thinking ブロック (署名付き) を出す。Anthropic の形として正しい
        ("#mock:nothink", "end_turn", vec!["thinking", "text"]),
        ("#mock:tools", "tool_use", vec!["thinking", "text", "tool_use", "tool_use"]),
    ] {
        let evs = read_sse(post(sc).await.unwrap()).await;
        check_messages_events(&evs).map_err(|e| format!("{sc}: {e}")).unwrap();
        let ds = datas(&evs);
        let md = ds.iter().find(|d| d["type"] == "message_delta").unwrap();
        assert_eq!(md["delta"]["stop_reason"], stop, "{sc}");
        let k: Vec<&str> = ds.iter().filter(|d| d["type"] == "content_block_start").map(|d| d["content_block"]["type"].as_str().unwrap()).collect();
        assert_eq!(k, kinds, "{sc}");
        if sc == "#mock:multitool_chunk" {
            let js: Vec<String> = (1..=2)
                .map(|i| ds.iter().filter(|d| d["index"] == i).filter_map(|d| d["delta"]["partial_json"].as_str()).collect())
                .collect();
            assert_eq!(js, ["{\"city\": \"Tokyo\"}", "{\"city\": \"Osaka\"}"]);
        }
    }
    let evs = read_sse(post("#mock:midfail_tool").await.unwrap()).await;
    assert_eq!(evs.last().unwrap().0, "error");
}

/// 思考だけで終わった応答 (本文が空) は、次のターンで別の空の応答に思考を再注入しない。
#[tokio::test]
async fn reinject_does_not_match_empty_content() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let c = client();
    let _ = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "#mock:thinkonly"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await;
    let _ = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "b"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await;
    let b = mock.last_body();
    assert!(b["messages"][1].get("reasoning_content").is_none(), "{}", b["messages"][1]);
}

/// 数十 MB の data URL: 3 つの API で通り、上流には元のまま届き、要求ログでは伏せられる。
#[tokio::test]
async fn large_data_url_images() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| c.state.memory_mb = 64).await;
    let c = client();
    let b64 = "QUJD".repeat(10 << 20); // 40 MiB
    let url = format!("data:image/png;base64,{b64}");
    let t0 = Instant::now();
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": [{"type": "text", "text": "what"}, {"type": "image_url", "image_url": {"url": url}}]}]}))
        .send()
        .await
        .unwrap();
    let st = r.status();
    assert_eq!(st, 200, "{}", r.text().await.unwrap());
    let up = mock.last_body();
    assert_eq!(up["messages"][0]["content"][1]["image_url"]["url"].as_str().unwrap().len(), url.len());
    let r = c
        .post(format!("{}/v1/messages", px.url))
        .json(&json!({"model": "x", "max_tokens": 10, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}, {"type": "text", "text": "what"}]}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(mock.last_body()["messages"][0]["content"][0]["image_url"]["url"].as_str().unwrap().len(), url.len());
    for _ in 0..3 {
        let r = c
            .post(format!("{}/v1/responses", px.url))
            .json(&json!({"model": "x", "input": [{"role": "user", "content": [{"type": "input_image", "image_url": url}, {"type": "input_text", "text": "what"}]}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
    }
    assert!(px.state.store.memory_usage().1 <= 64 << 20);
    eprintln!("large images: {:?}", t0.elapsed());
    // 要求ログ: 画像はハッシュだけ
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let dir = px.dir.path().join("logs/req");
    let mut total = 0u64;
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let t = std::fs::read_to_string(e.path()).unwrap();
        total += t.len() as u64;
        assert!(!t.contains("QUJDQUJDQUJDQUJDQUJDQUJD"), "image bytes leaked into the request log");
    }
    assert!(total > 0 && total < 1 << 20, "log size {total}");
}

#[tokio::test]
async fn compressed_request_bodies() {
    use std::io::Write;
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| c.server.max_body_mb = 1).await;
    let c = client();
    let body = json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(body.as_bytes()).unwrap();
    let gz = gz.finish().unwrap();
    let zs = zstd::bulk::compress(body.as_bytes(), 3).unwrap();
    for (enc, data) in [("gzip", gz), ("zstd", zs)] {
        let r = c
            .post(format!("{}/v1/chat/completions", px.url))
            .header("content-type", "application/json")
            .header("content-encoding", enc)
            .body(data)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "{enc}");
    }
    // 展開爆弾: 圧縮後は小さいが展開すると上限 (1 MB) を超える
    let big = format!("{{\"model\":\"x\",\"messages\":[{{\"role\":\"user\",\"content\":\"{}\"}}]}}", "a".repeat(4 << 20));
    let bomb = zstd::bulk::compress(big.as_bytes(), 3).unwrap();
    assert!(bomb.len() < 64 << 10);
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .header("content-encoding", "zstd")
        .body(bomb)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 413);
    let r = c.post(format!("{}/v1/chat/completions", px.url)).header("content-encoding", "br").body("x").send().await.unwrap();
    assert_eq!(r.status(), 400);
}

/// ds41rt がストリームの途中で送る error イベントの message を、クライアントへの途中失敗に載せる。
#[tokio::test]
async fn upstream_error_event_message_is_propagated() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let c = client();
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "stream": true, "messages": [{"role": "user", "content": "#mock:stream_error_event"}]}))
        .send()
        .await
        .unwrap();
    let evs = read_sse(r).await;
    let last: Value = serde_json::from_str(&evs.last().unwrap().1).unwrap();
    assert_eq!(last["error"]["code"], "upstream_disconnected");
    assert!(last["error"]["message"].as_str().unwrap().contains("boom"), "{last}");
    let r = c
        .post(format!("{}/v1/responses", px.url))
        .json(&json!({"model": "x", "stream": true, "input": "#mock:stream_error_event"}))
        .send()
        .await
        .unwrap();
    let ds = datas(&read_sse(r).await);
    let last = ds.last().unwrap();
    assert_eq!(last["type"], "response.failed");
    assert!(last["response"]["error"]["message"].as_str().unwrap().contains("boom"));
    // 非ストリームは 1 回やり直してから 502
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .json(&json!({"model": "x", "messages": [{"role": "user", "content": "#mock:stream_error_event"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
    assert!(r.text().await.unwrap().contains("boom"));
}
