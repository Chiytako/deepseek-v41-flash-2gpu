//! 最終段で直した点の回帰試験: max_tool_calls:0、reasoning 項目の id だけからの思考 (§6.5 の 4)、
//! DNS rebinding (Host の検査)、slowloris (ヘッダ・本文の読み取りの時間切れ)、ds41gw_overhead_seconds。
//! 上流は tests/common の mock。

mod common;

use std::time::Duration;

use common::*;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn resp_tools() -> Value {
    json!([{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}])
}

async fn post(px: &Proxy, path: &str, b: Value) -> reqwest::Response {
    client().post(format!("{}{path}", px.url)).json(&b).send().await.unwrap()
}

#[tokio::test]
async fn max_tool_calls_zero_means_no_tool_calls() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = post(&px, "/v1/responses", json!({"model": "x", "input": "weather?", "tools": resp_tools(), "max_tool_calls": 0})).await;
    assert_eq!(r.status(), 200);
    let warn = r.headers().get("x-ds41gw-warning").map(|v| v.to_str().unwrap().to_string()).unwrap_or_default();
    assert!(warn.contains("max_tool_calls_zero_as_tool_choice_none"), "{warn}");
    let v: Value = r.json().await.unwrap();
    // 応答の tool_choice は要求どおり (省略 = auto)
    assert_eq!(v["tool_choice"], "auto");
    assert_eq!(mock.last_body()["tool_choice"], "none");
    // 1 以上はそのまま (tool_choice を変えない)
    let r = post(&px, "/v1/responses", json!({"model": "x", "input": "weather?", "tools": resp_tools(), "max_tool_calls": 2})).await;
    assert_eq!(r.status(), 200);
    assert!(mock.last_body().get("tool_choice").is_none_or(|t| t != "none"));
}

/// reasoning 項目を id だけで送り返す (content・summary・encrypted_content 無し)。
fn id_only_history(rs_id: &str, answer: &str) -> Value {
    json!([
        {"role": "user", "content": "hello"},
        {"type": "reasoning", "id": rs_id, "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": answer}]},
        {"role": "user", "content": "next"}
    ])
}

#[tokio::test]
async fn reasoning_is_found_by_item_id_in_reinject_store() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let v: Value = post(&px, "/v1/responses", json!({"model": "x", "input": "hello", "store": false})).await.json().await.unwrap();
    let rs_id = v["output"][0]["id"].as_str().unwrap().to_string();
    assert!(rs_id.starts_with("rs_resp_"), "{rs_id}");
    // 回答の本文を変えて、本文ハッシュでの再注入には当たらないようにする
    let r = post(&px, "/v1/responses", json!({"model": "x", "store": false, "input": id_only_history(&rs_id, "edited answer")})).await;
    assert_eq!(r.status(), 200);
    let m = mock.last_body()["messages"].as_array().unwrap().clone();
    let a = m.iter().find(|x| x["role"] == "assistant").unwrap();
    assert_eq!(a["content"], "edited answer");
    assert_eq!(a["reasoning_content"], "Let me think.");
}

#[tokio::test]
async fn reasoning_is_found_by_item_id_in_saved_response() {
    let mock = start_mock().await;
    // 再注入ストアを切って、保存済みの応答 (sqlite) から引くことを確かめる
    let px = start_proxy(&mock, |c| c.reasoning.reinject = false).await;
    let v: Value = post(&px, "/v1/responses", json!({"model": "x", "input": "hello"})).await.json().await.unwrap();
    let rs_id = v["output"][0]["id"].as_str().unwrap().to_string();
    let r = post(&px, "/v1/responses", json!({"model": "x", "input": id_only_history(&rs_id, "edited answer")})).await;
    assert_eq!(r.status(), 200);
    let m = mock.last_body()["messages"].as_array().unwrap().clone();
    let a = m.iter().find(|x| x["role"] == "assistant").unwrap();
    assert_eq!(a["reasoning_content"], "Let me think.");
    // 知らない rs_resp_ の id は空のまま通し、警告を付ける
    let r = post(&px, "/v1/responses", json!({"model": "x", "input": id_only_history("rs_resp_0000_0", "x")})).await;
    assert_eq!(r.status(), 200);
    let warn = r.headers().get("x-ds41gw-warning").map(|v| v.to_str().unwrap().to_string()).unwrap_or_default();
    assert!(warn.contains("reasoning_id_not_found"), "{warn}");
}

#[tokio::test]
async fn dns_rebinding_is_rejected() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| c.server.allowed_hosts = vec!["gpu.example.ts.net".into()]).await;
    let port = px.url.rsplit(':').next().unwrap().to_string();
    let c = client();
    let body = json!({"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8});
    // 攻撃者の名前を 127.0.0.1 に向けたページからの同じ Origin の POST (JSON なので preflight は無い)
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .header("host", format!("evil.example.com:{port}"))
        .header("origin", format!("http://evil.example.com:{port}"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    let e: Value = r.json().await.unwrap();
    assert_eq!(e["error"]["code"], "host_not_allowed");
    // 同じ Origin の GET は Origin を付けない。Sec-Fetch-Site で見分ける (保存済みの応答の読み出しを防ぐ)
    let r = c
        .get(format!("{}/v1/models", px.url))
        .header("host", format!("evil.example.com:{port}"))
        .header("sec-fetch-site", "same-origin")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    // IP リテラル・localhost・allowed_hosts の Host は通す
    for host in [format!("127.0.0.1:{port}"), format!("localhost:{port}"), format!("gpu.example.ts.net:{port}")] {
        let r = c.get(format!("{}/v1/models", px.url)).header("host", host.clone()).header("sec-fetch-site", "same-origin").send().await.unwrap();
        assert_eq!(r.status(), 200, "{host}");
    }
    // ブラウザでない (Origin も Sec-Fetch-Site も無い) クライアントは Host を問わない (MagicDNS の名前など)
    let r = c.post(format!("{}/v1/chat/completions", px.url)).header("host", format!("other.example.net:{port}")).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    // Node の fetch は Sec-Fetch-Mode だけを付ける。これも通す
    let r = c
        .post(format!("{}/v1/chat/completions", px.url))
        .header("host", format!("other.example.net:{port}"))
        .header("sec-fetch-mode", "cors")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
}

/// 時間切れを短くした待ち受けを立てる (start_proxy は既定の ServeOpts)。
async fn start_short_timeouts(mock: &Mock) -> (String, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(mock, dir.path());
    cfg.server.header_read_timeout_s = 0.3;
    cfg.server.body_idle_timeout_s = 0.3;
    cfg.server.body_read_timeout_s = 1.0;
    let opts = ds41gw::ServeOpts::from_config(&cfg.server);
    let state = ds41gw::build_state(cfg);
    let app = ds41gw::router(state);
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        ds41gw::serve_listener_with(l, app, opts, std::future::pending()).await.unwrap();
    });
    (addr, dir)
}

async fn read_all(s: &mut tokio::net::TcpStream, limit: Duration) -> (String, bool) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let closed = loop {
        match tokio::time::timeout(limit, s.read(&mut tmp)).await {
            Ok(Ok(0)) | Ok(Err(_)) => break true,
            Ok(Ok(n)) => buf.extend_from_slice(&tmp[..n]),
            Err(_) => break false,
        }
    };
    (String::from_utf8_lossy(&buf).to_string(), closed)
}

#[tokio::test]
async fn slow_headers_are_cut() {
    let mock = start_mock().await;
    let (addr, _dir) = start_short_timeouts(&mock).await;
    let mut s = tokio::net::TcpStream::connect(&addr).await.unwrap();
    s.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n").await.unwrap();
    // ヘッダを最後まで送らずに置いておく → 0.3 秒で切られる
    let (_, closed) = read_all(&mut s, Duration::from_secs(3)).await;
    assert!(closed, "connection with incomplete headers was not closed");
    // 普通の要求は通る
    let r = client().get(format!("http://{addr}/health")).send().await.unwrap();
    assert_eq!(r.status(), 200);
}

#[tokio::test]
async fn slow_body_is_cut_with_408() {
    let mock = start_mock().await;
    let (addr, _dir) = start_short_timeouts(&mock).await;
    let mut s = tokio::net::TcpStream::connect(&addr).await.unwrap();
    s.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 200\r\n\r\n{\"model\":")
        .await
        .unwrap();
    let (text, _) = read_all(&mut s, Duration::from_secs(3)).await;
    assert!(text.starts_with("HTTP/1.1 408"), "{text}");
    assert!(text.contains("request_timeout"), "{text}");
    assert_eq!(mock.st.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    // 少しずつでも全体の上限 (1 秒) で切る
    let mut s = tokio::net::TcpStream::connect(&addr).await.unwrap();
    s.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 200\r\n\r\n")
        .await
        .unwrap();
    let t0 = std::time::Instant::now();
    let mut got = String::new();
    while t0.elapsed() < Duration::from_secs(3) {
        if s.write_all(b" ").await.is_err() {
            break;
        }
        let mut tmp = [0u8; 1024];
        if let Ok(Ok(n)) = tokio::time::timeout(Duration::from_millis(200), s.read(&mut tmp)).await {
            got.push_str(&String::from_utf8_lossy(&tmp[..n]));
            break;
        }
    }
    assert!(got.starts_with("HTTP/1.1 408"), "{got}");
    assert!(t0.elapsed() < Duration::from_millis(2500));
}

#[tokio::test]
async fn overhead_metric_and_log_field() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let r = post(&px, "/v1/chat/completions", json!({"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})).await;
    assert_eq!(r.status(), 200);
    let _ = r.text().await;
    let m = client().get(format!("{}/metrics", px.url)).send().await.unwrap().text().await.unwrap();
    assert!(m.contains("ds41gw_overhead_seconds_count{api=\"chat\"} 1"), "{m}");
    // 要求ログの終了行に overhead_ms、受信行に obj と orig (書き込みスレッドで整形する)
    tokio::time::sleep(Duration::from_millis(300)).await;
    let dir = px.dir.path().join("logs/req");
    let mut text = String::new();
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        text.push_str(&std::fs::read_to_string(e.path()).unwrap_or_default());
    }
    let lines: Vec<Value> = text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    let recv = lines.iter().find(|l| l["event"] == "request.received.openai").expect("received line");
    assert_eq!(recv["orig"]["messages"][0]["content"], "hi");
    assert_eq!(recv["obj"]["model"], "deepseek-ai/DeepSeek-V4.1-Flash");
    let fin = lines.iter().find(|l| l["event"] == "request.finished").expect("finished line");
    assert!(fin["overhead_ms"].as_f64().is_some(), "{fin}");
}
