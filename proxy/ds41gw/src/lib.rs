//! ds41gw: ds41rt (DeepSeek-V4.1-Flash) の前段に置く API 変換プロキシ。
//! 仕様: docs/research/r7_proxy_spec.md

pub mod auth;
pub mod chat;
pub mod completions;
pub mod config;
pub mod core;
pub mod dsml;
pub mod error;
pub mod messages;
pub mod metrics;
pub mod models;
pub mod names;
pub mod reasoning;
pub mod reqlog;
pub mod responses;
pub mod reverse;
pub mod store;
pub mod thinking;
pub mod tokens;
pub mod upstream;
pub mod util;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use axum::extract::DefaultBodyLimit;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use tokio::sync::Semaphore;

use crate::config::Config;
use crate::core::{App, AppState, ModelInfo};
use crate::error::{ApiError, Flavor, Kind};

/// アプリの状態を作る。
pub fn build_state(cfg: Config) -> AppState {
    let cfg = Arc::new(cfg);
    let metrics = Arc::new(metrics::Metrics::default());
    let up = Arc::new(upstream::Upstream::new(cfg.clone(), metrics.clone()));
    let signer = match reasoning::Signer::load_or_create(&cfg.state.secret_key_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ds41gw: 鍵 {} を読めない・作れない ({e})。一時的な鍵を使う", cfg.state.secret_key_path.display());
            let mut k = vec![0u8; 32];
            let _ = getrandom::getrandom(&mut k);
            reasoning::Signer::new(k)
        }
    };
    let store = store::Store::open(&cfg.state);
    let counter = tokens::Counter::load(&cfg.tokens.tokenizer, cfg.tokens.image_estimate);
    let reqlog = reqlog::ReqLog::start(&cfg.log);
    let mut key_sems = HashMap::new();
    for k in &cfg.auth.keys {
        if let Some(n) = k.max_concurrency {
            key_sems.insert(k.name.clone(), Arc::new(Semaphore::new(n.max(1))));
        }
    }
    Arc::new(App {
        reinject: reasoning::ReinjectStore::new(cfg.reasoning.reinject_store_mb, cfg.reasoning.reinject_ttl_h),
        model_info: RwLock::new(ModelInfo { max_context_tokens: cfg.limits.max_context_tokens, max_output_tokens: 393_216 }),
        upstream_stats: RwLock::new(serde_json::Value::Null),
        started: Instant::now(),
        cfg,
        up,
        metrics,
        signer,
        store,
        counter,
        reqlog,
        key_sems,
    })
}

/// axum の既定の拒否 (text/plain) を API の JSON エラーにする。
async fn json_rejections(req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let anthropic = req.uri().path().starts_with("/v1/messages");
    let r = next.run(req).await;
    let st = r.status();
    let is_text = r
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/plain"));
    if is_text && (st == StatusCode::PAYLOAD_TOO_LARGE || st == StatusCode::METHOD_NOT_ALLOWED || st == StatusCode::UNSUPPORTED_MEDIA_TYPE) {
        let flavor = if anthropic { Flavor::Anthropic } else { Flavor::OpenAI };
        let e = match st {
            StatusCode::PAYLOAD_TOO_LARGE => ApiError::new(Kind::TooLarge, "request_too_large", "request body is too large"),
            StatusCode::METHOD_NOT_ALLOWED => ApiError::new(Kind::NotFound, "method_not_allowed", "method not allowed"),
            _ => ApiError::bad("invalid_request", "unsupported media type"),
        };
        let mut resp = e.into_response_with(flavor);
        if st == StatusCode::METHOD_NOT_ALLOWED {
            *resp.status_mut() = StatusCode::METHOD_NOT_ALLOWED;
        }
        return resp;
    }
    r
}

/// POST の本文を読み切ってから先に渡す。読み取りに時間切れを付け (slowloris 対策)、
/// `Content-Encoding: gzip | deflate | zstd` の本文を展開する。展開後も max_body_mb で抑える (展開爆弾)。
/// 展開しないと、圧縮して送るクライアントは「JSON として読めない」400 になる。
async fn decode_body(
    axum::extract::State(app): axum::extract::State<AppState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if req.method() != axum::http::Method::POST {
        return next.run(req).await;
    }
    let enc = req
        .headers()
        .get("content-encoding")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let flavor = if req.uri().path().starts_with("/v1/messages") { Flavor::Anthropic } else { Flavor::OpenAI };
    let limit = app.cfg.server.max_body_mb.saturating_mul(1 << 20);
    let (mut parts, body) = req.into_parts();
    let raw = match read_body(body, limit, &app.cfg.server).await {
        Ok(b) => b,
        Err(BodyError::TooLarge) => {
            return ApiError::new(Kind::TooLarge, "request_too_large", "request body is too large").into_response_with(flavor)
        }
        Err(BodyError::Timeout) => {
            app.metrics.inc("ds41gw_body_read_timeouts_total", &[]);
            let mut r = ApiError::bad("request_timeout", "timed out reading the request body").into_response_with(flavor);
            *r.status_mut() = StatusCode::REQUEST_TIMEOUT;
            r.headers_mut().insert("connection", HeaderValue::from_static("close"));
            return r;
        }
        Err(BodyError::Io) => return ApiError::bad("invalid_request", "could not read the request body").into_response_with(flavor),
    };
    if enc.is_empty() || enc == "identity" {
        return next.run(axum::extract::Request::from_parts(parts, axum::body::Body::from(raw))).await;
    }
    use std::io::Read;
    let mut out = Vec::new();
    let res = match enc.as_str() {
        "gzip" | "x-gzip" => flate2::read::MultiGzDecoder::new(&raw[..]).take(limit as u64 + 1).read_to_end(&mut out),
        "deflate" => {
            // HTTP の deflate は zlib 形式 (生の deflate を送るものもあるので両方試す)
            let r = flate2::read::ZlibDecoder::new(&raw[..]).take(limit as u64 + 1).read_to_end(&mut out);
            if r.is_err() {
                out.clear();
                flate2::read::DeflateDecoder::new(&raw[..]).take(limit as u64 + 1).read_to_end(&mut out)
            } else {
                r
            }
        }
        "zstd" => match zstd::stream::read::Decoder::new(&raw[..]) {
            Ok(d) => d.take(limit as u64 + 1).read_to_end(&mut out),
            Err(e) => Err(e),
        },
        other => {
            return ApiError::new(Kind::BadRequest, "unsupported_content_encoding", format!("unsupported Content-Encoding: {other}"))
                .into_response_with(flavor)
        }
    };
    if res.is_err() {
        return ApiError::bad("invalid_request", format!("could not decode the {enc} request body")).into_response_with(flavor);
    }
    if out.len() > limit {
        return ApiError::new(Kind::TooLarge, "request_too_large", "request body is too large").into_response_with(flavor);
    }
    parts.headers.remove("content-encoding");
    parts.headers.remove("content-length");
    next.run(axum::extract::Request::from_parts(parts, axum::body::Body::from(out))).await
}

enum BodyError {
    TooLarge,
    Timeout,
    Io,
}

/// 本文を読み切る。かたまりの間が `body_idle_timeout_s`、全体が `body_read_timeout_s` を超えたら諦める。
async fn read_body(body: axum::body::Body, limit: usize, cfg: &config::Server) -> Result<bytes::Bytes, BodyError> {
    use futures::StreamExt;
    let idle = (cfg.body_idle_timeout_s > 0.0).then(|| std::time::Duration::from_secs_f64(cfg.body_idle_timeout_s));
    let total = (cfg.body_read_timeout_s > 0.0).then(|| std::time::Duration::from_secs_f64(cfg.body_read_timeout_s));
    let read = async {
        let mut s = body.into_data_stream();
        let mut chunks: Vec<bytes::Bytes> = Vec::new();
        let mut len = 0usize;
        loop {
            let next = match idle {
                Some(d) => tokio::time::timeout(d, s.next()).await.map_err(|_| BodyError::Timeout)?,
                None => s.next().await,
            };
            match next {
                None => break,
                Some(Err(_)) => return Err(BodyError::Io),
                Some(Ok(b)) => {
                    len += b.len();
                    if len > limit {
                        return Err(BodyError::TooLarge);
                    }
                    chunks.push(b);
                }
            }
        }
        Ok(match chunks.len() {
            0 => bytes::Bytes::new(),
            1 => chunks.pop().unwrap(),
            _ => {
                let mut v = Vec::with_capacity(len);
                for c in &chunks {
                    v.extend_from_slice(c);
                }
                bytes::Bytes::from(v)
            }
        })
    };
    match total {
        Some(d) => tokio::time::timeout(d, read).await.map_err(|_| BodyError::Timeout)?,
        None => read.await,
    }
}

pub fn router(state: AppState) -> Router {
    let limit = state.cfg.server.max_body_mb.saturating_mul(1 << 20);
    Router::new()
        .route("/v1/chat/completions", post(chat::handle))
        .route("/chat/completions", post(chat::handle))
        .route("/v1/responses", post(responses::handle))
        .route("/responses", post(responses::handle))
        .route("/v1/responses/:id", get(responses::get_response).delete(responses::delete_response))
        .route("/v1/responses/:id/input_items", get(responses::input_items))
        .route("/v1/responses/:id/cancel", post(responses::cancel_response))
        .route("/v1/messages", post(messages::handle))
        .route("/v1/messages/count_tokens", post(messages::count_tokens))
        .route("/v1/completions", post(completions::handle))
        .route("/completions", post(completions::handle))
        .route("/v1/models", get(models::list))
        .route("/models", get(models::list))
        .route("/v1/models/:id", get(models::get))
        .route("/health", get(models::health))
        .route("/health/ready", get(models::ready))
        .route("/metrics", get(models::metrics))
        .route("/v1/proxy/stats", get(models::proxy_stats))
        .fallback(models::not_found)
        .layer(DefaultBodyLimit::max(limit))
        .layer(axum::middleware::from_fn_with_state(state.clone(), decode_body))
        .layer(axum::middleware::from_fn(json_rejections))
        .layer(axum::middleware::from_fn_with_state(state.clone(), auth::cors_layer))
        .with_state(state)
}

/// 待ち受けの設定。
#[derive(Debug, Clone, Copy)]
pub struct ServeOpts {
    /// 要求ヘッダを読み終えるまでの時間 (None で無制限)。keep-alive で次の要求を待つ間にも数える。
    pub header_read_timeout: Option<std::time::Duration>,
}

impl Default for ServeOpts {
    fn default() -> Self {
        Self::from_config(&config::Server::default())
    }
}

impl ServeOpts {
    pub fn from_config(c: &config::Server) -> Self {
        Self { header_read_timeout: (c.header_read_timeout_s > 0.0).then(|| std::time::Duration::from_secs_f64(c.header_read_timeout_s)) }
    }
}

/// 1 つの口で待ち受ける (試験からも使う)。既定の ServeOpts で。
pub async fn serve_listener(
    listener: tokio::net::TcpListener,
    app: Router,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    serve_listener_with(listener, app, ServeOpts::default(), shutdown).await
}

/// 1 つの口で待ち受ける。axum::serve はヘッダの読み取りに時間切れを付けられない (hyper に timer を渡さない) ので、
/// hyper の http1 を直に使う。ヘッダを少しずつ送って接続を持ち続ける slowloris を header_read_timeout で切る。
pub async fn serve_listener_with(
    listener: tokio::net::TcpListener,
    app: Router,
    opts: ServeOpts,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let mut shutdown = std::pin::pin!(shutdown);
    loop {
        let (stream, remote) = tokio::select! {
            r = listener.accept() => match r {
                Ok(x) => x,
                Err(e) => {
                    // 接続ごとのエラー (相手が先に切ったなど) は無視する。fd 不足などは少し待つ
                    if !matches!(e.kind(), std::io::ErrorKind::ConnectionAborted | std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionRefused) {
                        eprintln!("ds41gw: accept: {e}");
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    }
                    continue;
                }
            },
            _ = &mut shutdown => break,
        };
        // TCP_NODELAY: SSE の小さな書き込みが Nagle と遅延 ACK で 40 ms 待たされないように
        let _ = stream.set_nodelay(true);
        let app = app.clone();
        let mut stop = stop_rx.clone();
        tokio::spawn(async move {
            let svc = hyper::service::service_fn(move |mut req: hyper::Request<hyper::body::Incoming>| {
                req.extensions_mut().insert(axum::extract::ConnectInfo(remote));
                tower::ServiceExt::oneshot(app.clone(), req)
            });
            let mut b = hyper::server::conn::http1::Builder::new();
            b.timer(hyper_util::rt::TokioTimer::new());
            b.header_read_timeout(opts.header_read_timeout);
            let conn = b.serve_connection(hyper_util::rt::TokioIo::new(stream), svc);
            let mut conn = std::pin::pin!(conn);
            tokio::select! {
                _ = conn.as_mut() => return,
                _ = stop.changed() => conn.as_mut().graceful_shutdown(),
            }
            let _ = conn.await;
        });
    }
    let _ = stop_tx.send(true);
    Ok(())
}

pub fn version_header() -> HeaderValue {
    HeaderValue::from_static(concat!("ds41gw/", env!("CARGO_PKG_VERSION")))
}

pub fn unavailable() -> Response {
    (StatusCode::SERVICE_UNAVAILABLE, "shutting down").into_response()
}
