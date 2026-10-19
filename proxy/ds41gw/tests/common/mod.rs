//! 試験用の共通部品: ds41rt を模した上流 (mock) と、proxy の起動。
//! mock の振る舞いは、要求本文に含まれる `#mock:<名前>` で選ぶ。

#![allow(dead_code)]

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use futures::StreamExt;
use serde_json::{json, Value};

#[derive(Default)]
pub struct MockState {
    pub bodies: Mutex<Vec<Value>>,
    pub calls: AtomicUsize,
    pub disconnects: AtomicUsize,
    pub completed: AtomicUsize,
    pub inflight: AtomicUsize,
    pub max_inflight: AtomicUsize,
    pub scenario_calls: Mutex<std::collections::HashMap<String, usize>>,
}

pub struct Mock {
    pub addr: SocketAddr,
    pub st: Arc<MockState>,
}

impl Mock {
    pub fn bodies(&self) -> Vec<Value> {
        self.st.bodies.lock().unwrap().clone()
    }
    pub fn last_body(&self) -> Value {
        self.bodies().last().cloned().unwrap_or(Value::Null)
    }
}

fn scenario(body: &Value) -> String {
    let s = body.to_string();
    match s.find("#mock:") {
        Some(p) => s[p + 6..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect(),
        None => "text".into(),
    }
}

struct Guard {
    st: Arc<MockState>,
    done: bool,
}
impl Drop for Guard {
    fn drop(&mut self) {
        self.st.inflight.fetch_sub(1, Ordering::SeqCst);
        if self.done {
            self.st.completed.fetch_add(1, Ordering::SeqCst);
        } else {
            self.st.disconnects.fetch_add(1, Ordering::SeqCst);
        }
    }
}

fn chunk(id: &str, delta: Value, fin: Option<&str>, usage: Option<Value>) -> Bytes {
    let v = json!({"id": id, "object": "chat.completion.chunk", "created": 1792339106u64, "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "system_fingerprint": "mock-fp", "choices": [{"index": 0, "delta": delta, "logprobs": null, "finish_reason": fin}], "usage": usage});
    Bytes::from(format!("data: {v}\n\n"))
}

fn usage() -> Value {
    json!({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "prompt_tokens_details": {"cached_tokens": 40},
           "prompt_cache_hit_tokens": 40, "prompt_cache_miss_tokens": 60})
}

async fn chat(State(st): State<Arc<MockState>>, body: Bytes) -> Response {
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    st.bodies.lock().unwrap().push(body.clone());
    st.calls.fetch_add(1, Ordering::SeqCst);
    let sc = scenario(&body);
    let nth = {
        let mut m = st.scenario_calls.lock().unwrap();
        let e = m.entry(sc.clone()).or_insert(0);
        *e += 1;
        *e
    };
    match sc.as_str() {
        "err400" => {
            return (StatusCode::BAD_REQUEST, axum::Json(json!({"error": {"message": "invalid thing from ds41rt", "type": "native_v41_error"}})))
                .into_response()
        }
        "err500" => return (StatusCode::INTERNAL_SERVER_ERROR, axum::Json(json!({"error": {"message": "worker failed", "type": "native_v41_error"}}))).into_response(),
        "err503" => return (StatusCode::SERVICE_UNAVAILABLE, "worker queue is closed").into_response(),
        "r429x2" if nth <= 2 => {
            return (StatusCode::TOO_MANY_REQUESTS, axum::Json(json!({"error": {"message": "request queue is full", "type": "native_v41_error"}}))).into_response()
        }
        "silent" => tokio::time::sleep(Duration::from_millis(400)).await,
        "hang_headers" => tokio::time::sleep(Duration::from_millis(3000)).await,
        _ => {}
    }
    let thinking = body.pointer("/thinking/type").and_then(Value::as_str) == Some("enabled");
    let tool_names: Vec<String> = body
        .get("tools")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|t| t.pointer("/function/name").and_then(Value::as_str).map(str::to_string)).collect())
        .unwrap_or_default();
    let id = format!("chatcmpl-mock{}", st.calls.load(Ordering::SeqCst));
    let st2 = st.clone();
    st.inflight.fetch_add(1, Ordering::SeqCst);
    let cur = st.inflight.load(Ordering::SeqCst);
    st.max_inflight.fetch_max(cur, Ordering::SeqCst);
    let stream = async_stream::stream! {
        let mut guard = Guard { st: st2, done: false };
        let first = if thinking { json!({"role": "assistant", "content": null, "reasoning_content": ""}) } else { json!({"role": "assistant", "content": ""}) };
        yield chunk(&id, first, None, None);
        let r = |s: &str| json!({"content": null, "reasoning_content": s});
        let c = |s: &str| if thinking { json!({"content": s, "reasoning_content": null}) } else { json!({"content": s}) };
        if thinking && sc != "nothink" {
            for p in ["Let me ", "think."] { yield chunk(&id, r(p), None, None); }
        }
        match sc.as_str() {
            "tools" | "slowtools" => {
                if sc == "slowtools" {
                    // 思考の項目と残りの項目の間で秒をまたがせる (encrypted_content の ts が変わる)
                    tokio::time::sleep(Duration::from_millis(1100)).await;
                }
                yield chunk(&id, c("Checking."), None, None);
                let n0 = tool_names.first().cloned().unwrap_or_else(|| "get_weather".into());
                let n1 = tool_names.get(1).cloned().unwrap_or_else(|| n0.clone());
                for (k, name) in [n0, n1].iter().enumerate() {
                    yield chunk(&id, json!({"tool_calls": [{"index": k, "id": format!("call_{id}_{k}"), "type": "function", "function": {"name": name, "arguments": ""}}]}), None, None);
                    for piece in ["{\"city\"", ": \"", if k == 0 { "Tokyo" } else { "Osaka" }, "\"}"] {
                        yield chunk(&id, json!({"tool_calls": [{"index": k, "function": {"arguments": piece}}]}), None, None);
                    }
                }
                yield chunk(&id, c(""), Some("tool_calls"), Some(usage()));
            }
            "dsml" => {
                for p in ["Answer: ", "ok\n\n<｜DS", "ML｜ calls>\n<｜DSML｜ invoke name=\"x\">\n</｜DSML｜ invoke>\n</｜DSML｜ ca", "lls>\nbye"] {
                    yield chunk(&id, c(p), None, None);
                }
                yield chunk(&id, c(""), Some("stop"), Some(usage()));
            }
            "thinkonly" => {
                yield chunk(&id, c(""), Some("length"), Some(usage()));
            }
            "nothink" => {
                yield chunk(&id, c("plain"), None, None);
                yield chunk(&id, c(""), Some("stop"), Some(usage()));
            }
            "multitool_chunk" => {
                // 1 つのチャンクに 2 本の呼び出し (引数も全部)
                yield chunk(&id, json!({"tool_calls": [
                    {"index": 0, "id": format!("call_{id}_0"), "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}},
                    {"index": 1, "id": format!("call_{id}_1"), "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Osaka\"}"}}
                ]}), None, None);
                yield chunk(&id, c(""), Some("tool_calls"), Some(usage()));
            }
            "midfail_tool" => {
                yield chunk(&id, json!({"tool_calls": [{"index": 0, "id": format!("call_{id}_0"), "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}), None, None);
                yield chunk(&id, json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"ci"}}]}), None, None);
                guard.done = true;
                return;
            }
            "midfail" => {
                yield chunk(&id, c("partial"), None, None);
                // finish を送らずに切る
                guard.done = true;
                return;
            }
            "stream_error_event" => {
                yield chunk(&id, c("partial"), None, None);
                yield Bytes::from_static(b"event: error\ndata: {\"error\":{\"message\":\"real-full streaming executor error: boom\",\"type\":\"server_error\",\"param\":null,\"code\":\"backend_error\"}}\n\n");
                yield Bytes::from_static(b"data: [DONE]\n\n");
                guard.done = true;
                return;
            }
            "midfail_once" if nth == 1 => {
                yield chunk(&id, c("partial"), None, None);
                guard.done = true;
                return;
            }
            "slow" => {
                for i in 0..200 {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    yield chunk(&id, c(&format!("t{i} ")), None, None);
                }
                yield chunk(&id, c(""), Some("stop"), Some(usage()));
            }
            "work" => {
                tokio::time::sleep(Duration::from_millis(150)).await;
                yield chunk(&id, c("done"), None, None);
                yield chunk(&id, c(""), Some("stop"), Some(usage()));
            }
            "silent_body" => {
                tokio::time::sleep(Duration::from_millis(350)).await;
                yield chunk(&id, c("late"), None, None);
                yield chunk(&id, c(""), Some("stop"), Some(usage()));
            }
            "length" => {
                yield chunk(&id, c("cut"), None, None);
                yield chunk(&id, c(""), Some("length"), Some(usage()));
            }
            "stopseq" => {
                for p in ["Hello wo", "rld EN", "D tail"] { yield chunk(&id, c(p), None, None); }
                yield chunk(&id, c(""), Some("stop"), Some(usage()));
            }
            "json" => {
                yield chunk(&id, c("{\"answer\": 42}"), None, None);
                yield chunk(&id, c(""), Some("stop"), Some(usage()));
            }
            _ => {
                for p in ["Hello", " world"] { yield chunk(&id, c(p), None, None); }
                yield chunk(&id, c(""), Some("stop"), Some(usage()));
            }
        }
        yield Bytes::from_static(b"data: [DONE]\n\n");
        guard.done = true;
    };
    let mut r = Response::new(Body::from_stream(stream.map(Ok::<_, std::convert::Infallible>)));
    r.headers_mut().insert("content-type", "text/event-stream".parse().unwrap());
    r
}

async fn models() -> Response {
    axum::Json(json!({"object": "list", "data": [{"id": "deepseek-ai/DeepSeek-V4.1-Flash", "object": "model", "owned_by": "deepseek-ai",
        "max_context_tokens": 1048576, "max_output_tokens": 393216}]}))
    .into_response()
}

pub async fn start_mock() -> Mock {
    let st = Arc::new(MockState::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(chat).layer(axum::extract::DefaultBodyLimit::max(256 << 20)))
        .route("/v1/models", get(models))
        .route("/health", get(|| async { "ok" }))
        .route("/v1/stats", get(|| async { axum::Json(json!({"http_queue_rejects": 0, "totals": {"admitted": 1}})) }))
        .with_state(st.clone());
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(l, app).await.unwrap();
    });
    Mock { addr, st }
}

pub struct Proxy {
    pub url: String,
    pub dir: tempfile::TempDir,
    pub state: ds41gw::core::AppState,
}

pub fn test_config(mock: &Mock, dir: &std::path::Path) -> ds41gw::config::Config {
    let mut cfg = ds41gw::config::Config::default();
    cfg.upstream.url = format!("http://{}", mock.addr);
    cfg.upstream.connect_retry_total_s = 1.0;
    cfg.state.path = dir.join("state/responses.sqlite");
    cfg.state.secret_key_path = dir.join("state/secret.key");
    cfg.log.dir = dir.join("logs");
    cfg.tokens.tokenizer = dir.join("no-tokenizer.json");
    cfg
}

pub async fn start_proxy(mock: &Mock, f: impl FnOnce(&mut ds41gw::config::Config)) -> Proxy {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(mock, dir.path());
    f(&mut cfg);
    let state = ds41gw::build_state(cfg);
    let app = ds41gw::router(state.clone());
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move {
        ds41gw::serve_listener(l, app, std::future::pending()).await.unwrap();
    });
    Proxy { url: format!("http://{addr}"), dir, state }
}

pub fn client() -> reqwest::Client {
    reqwest::Client::builder().build().unwrap()
}

/// SSE を読み切って (event 名, data) の列にする。コメント行は ("#comment", 本文) にする。
pub async fn read_sse(resp: reqwest::Response) -> Vec<(String, String)> {
    let text = resp.text().await.unwrap();
    parse_sse(&text)
}

pub fn parse_sse(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        if block.trim().is_empty() {
            continue;
        }
        let mut ev = String::new();
        let mut data = String::new();
        for line in block.lines() {
            if let Some(c) = line.strip_prefix(':') {
                out.push(("#comment".into(), c.trim().to_string()));
            } else if let Some(e) = line.strip_prefix("event:") {
                ev = e.trim().to_string();
            } else if let Some(d) = line.strip_prefix("data:") {
                data.push_str(d.trim_start());
            }
        }
        if !data.is_empty() || !ev.is_empty() {
            out.push((ev, data));
        }
    }
    out
}

pub fn datas(evs: &[(String, String)]) -> Vec<Value> {
    evs.iter().filter(|(e, _)| e != "#comment").filter_map(|(_, d)| serde_json::from_str(d).ok()).collect()
}

/// Responses のイベント順序の検査器 (T-U4)。
pub fn check_responses_events(evs: &[Value]) -> Result<(), String> {
    let mut seq_expect = 0u64;
    let mut open_item: Option<u64> = None;
    let mut open_part: Option<(u64, u64)> = None;
    let mut terminal = 0;
    let mut items_added = std::collections::HashSet::new();
    for (i, e) in evs.iter().enumerate() {
        let ty = e["type"].as_str().ok_or(format!("event {i} has no type"))?;
        let seq = e["sequence_number"].as_u64().ok_or(format!("event {i} ({ty}) has no sequence_number"))?;
        if seq != seq_expect {
            return Err(format!("event {i} ({ty}): sequence_number {seq}, expected {seq_expect}"));
        }
        seq_expect += 1;
        if terminal > 0 {
            return Err(format!("event {i} ({ty}) after terminal"));
        }
        match ty {
            "response.output_item.added" => {
                if open_item.is_some() {
                    return Err(format!("event {i}: item added while another is open"));
                }
                let oi = e["output_index"].as_u64().unwrap();
                if !items_added.insert(oi) {
                    return Err(format!("event {i}: output_index {oi} added twice"));
                }
                open_item = Some(oi);
            }
            "response.output_item.done" => {
                if open_item != e["output_index"].as_u64() {
                    return Err(format!("event {i}: done for item that is not open"));
                }
                if open_part.is_some() {
                    return Err(format!("event {i}: item done while a part is open"));
                }
                open_item = None;
            }
            "response.content_part.added" | "response.reasoning_summary_part.added" => {
                if open_item != e["output_index"].as_u64() || open_part.is_some() {
                    return Err(format!("event {i}: part added in a bad state"));
                }
                open_part = Some((e["output_index"].as_u64().unwrap(), 0));
            }
            "response.content_part.done" | "response.reasoning_summary_part.done" => {
                if open_part.map(|p| p.0) != e["output_index"].as_u64() {
                    return Err(format!("event {i}: part done without open part"));
                }
                open_part = None;
            }
            "response.output_text.delta" | "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" | "response.output_text.done"
            | "response.reasoning_text.done" | "response.reasoning_summary_text.done" => {
                if open_part.map(|p| p.0) != e["output_index"].as_u64() {
                    return Err(format!("event {i} ({ty}): text outside an open part"));
                }
            }
            "response.function_call_arguments.delta" | "response.function_call_arguments.done" => {
                if open_item != e["output_index"].as_u64() {
                    return Err(format!("event {i} ({ty}): arguments outside the open item"));
                }
            }
            "response.completed" | "response.incomplete" | "response.failed" => terminal += 1,
            _ => {}
        }
    }
    if terminal != 1 {
        return Err(format!("{terminal} terminal events"));
    }
    // output_item.done の項目と、終了イベントの response.output は同じでなければならない
    let done: std::collections::HashMap<u64, &Value> = evs
        .iter()
        .filter(|e| e["type"] == "response.output_item.done")
        .filter_map(|e| Some((e["output_index"].as_u64()?, &e["item"])))
        .collect();
    let last = evs.last().unwrap();
    if last["type"] != "response.failed" {
        let out = last["response"]["output"].as_array().ok_or("terminal response has no output")?;
        if out.len() != done.len() {
            return Err(format!("output has {} items but {} output_item.done", out.len(), done.len()));
        }
        for (k, it) in out.iter().enumerate() {
            if done.get(&(k as u64)) != Some(&it) {
                return Err(format!("output[{k}] differs from its output_item.done"));
            }
        }
    }
    Ok(())
}
