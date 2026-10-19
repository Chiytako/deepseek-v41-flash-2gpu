//! `/v1/models`、ヘルスチェック、計測、統計 (r7_proxy_spec.md §9)。

use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

use crate::core::AppState;
use crate::error::{ApiError, Flavor, Kind};

/// 正規名と別名の一覧。
pub fn model_ids(app: &AppState) -> Vec<String> {
    let mut v = vec![app.cfg.upstream.model.clone()];
    for a in &app.cfg.models.aliases {
        if !v.contains(a) {
            v.push(a.clone());
        }
    }
    v
}

fn model_obj(app: &AppState, id: &str) -> Value {
    let info = app.model_info.read().unwrap().clone();
    let canon = &app.cfg.upstream.model;
    let display = if id == canon { "DeepSeek-V4.1-Flash".to_string() } else { format!("DeepSeek-V4.1-Flash (alias {id})") };
    // OpenAI (`object`/`created`/`owned_by`) と Anthropic (`type`/`created_at`/`display_name`) の和集合
    json!({"id": id, "object": "model", "type": "model", "created": 1_792_339_200u64, "created_at": "2026-10-19T00:00:00Z",
           "owned_by": "ds41gw", "display_name": display, "root": canon,
           "max_context_tokens": info.max_context_tokens, "max_output_tokens": info.max_output_tokens})
}

pub async fn list(State(app): State<AppState>) -> Response {
    let data: Vec<Value> = model_ids(&app).iter().map(|id| model_obj(&app, id)).collect();
    let first = data.first().and_then(|d| d.get("id")).cloned().unwrap_or(Value::Null);
    let last = data.last().and_then(|d| d.get("id")).cloned().unwrap_or(Value::Null);
    axum::Json(json!({"object": "list", "data": data, "has_more": false, "first_id": first, "last_id": last})).into_response()
}

pub async fn get(State(app): State<AppState>, Path(id): Path<String>) -> Response {
    if model_ids(&app).contains(&id) {
        return axum::Json(model_obj(&app, &id)).into_response();
    }
    ApiError::new(Kind::NotFound, "model_not_found", format!("The model `{id}` does not exist")).into_response_with(Flavor::OpenAI)
}

pub async fn health() -> Response {
    axum::Json(json!({"status": "ok"})).into_response()
}

pub async fn ready(State(app): State<AppState>) -> Response {
    let up = app
        .up
        .http()
        .get(app.up.url("/health"))
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    if app.up.shutting_down.load(Ordering::Relaxed) {
        return (StatusCode::SERVICE_UNAVAILABLE, axum::Json(json!({"status": "unavailable", "reason": "shutting_down"}))).into_response();
    }
    if !up {
        return (StatusCode::SERVICE_UNAVAILABLE, axum::Json(json!({"status": "unavailable", "reason": "upstream_down"}))).into_response();
    }
    if app.up.queue_full() {
        return (StatusCode::SERVICE_UNAVAILABLE, axum::Json(json!({"status": "unavailable", "reason": "queue_full"}))).into_response();
    }
    axum::Json(json!({"status": "ready"})).into_response()
}

pub async fn metrics(State(app): State<AppState>) -> Response {
    let mut gauges = vec![
        ("ds41gw_queue_waiting", app.up.waiting.load(Ordering::Relaxed) as f64),
        ("ds41gw_upstream_inflight", app.up.inflight.load(Ordering::Relaxed) as f64),
        ("ds41gw_reinject_store_entries", app.reinject.len() as f64),
        ("ds41gw_uptime_seconds", app.started.elapsed().as_secs_f64()),
    ];
    let stats = app.upstream_stats.read().unwrap().clone();
    let mut extra = Vec::new();
    for k in ["http_queue_waits", "http_queue_rejects", "http_queue_len", "http_queue_wait_ms_sum"] {
        if let Some(v) = stats.get(k).and_then(Value::as_f64) {
            extra.push((format!("ds41rt_{k}"), v));
        }
    }
    if let Some(t) = stats.get("totals").and_then(Value::as_object) {
        for (k, v) in t {
            if let Some(x) = v.as_f64() {
                extra.push((format!("ds41rt_totals_{k}"), x));
            }
        }
    }
    let extra_refs: Vec<(&str, f64)> = extra.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    gauges.extend(extra_refs);
    let body = app.metrics.render(&gauges);
    ([("content-type", "text/plain; version=0.0.4")], body).into_response()
}

pub async fn proxy_stats(State(app): State<AppState>) -> Response {
    let up = app.upstream_stats.read().unwrap().clone();
    axum::Json(json!({
        "queue_waiting": app.up.waiting.load(Ordering::Relaxed),
        "upstream_inflight": app.up.inflight.load(Ordering::Relaxed),
        "max_concurrency": app.cfg.upstream.max_concurrency,
        "reinject_entries": app.reinject.len(),
        "uptime_s": app.started.elapsed().as_secs(),
        "tokenizer": app.counter.has_tokenizer(),
        "ds41rt": up,
    }))
    .into_response()
}

pub async fn not_found(uri: Uri, headers: HeaderMap) -> Response {
    let anthropic = uri.path().starts_with("/v1/messages") || headers.contains_key("anthropic-version");
    let flavor = if anthropic { Flavor::Anthropic } else { Flavor::OpenAI };
    ApiError::new(Kind::NotFound, "not_found", format!("Unknown path: {}", uri.path())).into_response_with(flavor)
}

/// 背景の定期処理: ds41rt の /v1/models (60 秒)、/v1/stats (10 秒)、状態の掃除 (1 時間)。
pub fn spawn_pollers(app: AppState) {
    let a = app.clone();
    tokio::spawn(async move {
        loop {
            if let Ok(r) = a.up.http().get(a.up.url("/v1/models")).timeout(Duration::from_secs(5)).send().await {
                if let Ok(v) = r.json::<Value>().await {
                    if let Some(m) = v.pointer("/data/0") {
                        let mut info = a.model_info.write().unwrap();
                        if let Some(x) = m.get("max_context_tokens").and_then(Value::as_u64) {
                            info.max_context_tokens = x;
                        }
                        if let Some(x) = m.get("max_output_tokens").and_then(Value::as_u64) {
                            info.max_output_tokens = x;
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    });
    let a = app.clone();
    tokio::spawn(async move {
        loop {
            if let Ok(r) = a.up.http().get(a.up.url("/v1/stats")).timeout(Duration::from_secs(5)).send().await {
                if let Ok(v) = r.json::<Value>().await {
                    *a.upstream_stats.write().unwrap() = v;
                }
            }
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
    });
    let a = app;
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            let a2 = a.clone();
            let _ = tokio::task::spawn_blocking(move || a2.store.sweep()).await;
        }
    });
}
