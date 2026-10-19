//! API キーと CORS (r7_proxy_spec.md §12)。既定はどちらも off。

use std::net::IpAddr;
use std::sync::Arc;

use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use tokio::sync::Semaphore;

use crate::core::AppState;
use crate::error::{ApiError, Kind};
use crate::util::{ip_in_cidr, sha256_hex};

/// (キーの名前, キーごとの同時数の枠)。
pub fn check(app: &AppState, headers: &HeaderMap, ip: Option<IpAddr>) -> Result<(Option<String>, Option<Arc<Semaphore>>), ApiError> {
    let keys = &app.cfg.auth.keys;
    if keys.is_empty() {
        return Ok((None, None));
    }
    // Authorization: Bearer と x-api-key の両方を見る (片方にダミーを入れてくるクライアントがある)
    let bearer = headers
        .get_all("authorization")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .filter_map(|v| {
            let v = v.trim();
            let (scheme, rest) = v.split_once(' ')?;
            scheme.eq_ignore_ascii_case("bearer").then(|| rest.trim())
        });
    let xkey = headers.get_all("x-api-key").iter().filter_map(|v| v.to_str().ok()).map(str::trim);
    for k in bearer.chain(xkey).filter(|k| !k.is_empty()) {
        let h = sha256_hex(k.as_bytes());
        for key in keys {
            if ct_eq(h.as_bytes(), key.sha256.to_ascii_lowercase().as_bytes()) {
                return Ok((Some(key.name.clone()), app.key_sems.get(&key.name).cloned()));
            }
        }
    }
    if let Some(ip) = ip {
        if app.cfg.auth.allow_without_key.iter().any(|c| ip_in_cidr(ip, c)) {
            return Ok((None, None));
        }
    }
    Err(ApiError::new(Kind::Auth, "invalid_api_key", "Incorrect API key provided."))
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// CORS のヘッダを付ける (許可された Origin のときだけ)。
pub fn cors_headers(app: &AppState, origin: Option<&HeaderValue>, h: &mut HeaderMap) {
    let allow = &app.cfg.cors.allow_origins;
    if allow.is_empty() {
        return;
    }
    let Some(o) = origin.and_then(|o| o.to_str().ok()) else { return };
    if !allow.iter().any(|a| a == "*" || a == o) {
        return;
    }
    if let Ok(v) = HeaderValue::from_str(o) {
        h.insert("access-control-allow-origin", v);
    }
    h.insert("vary", HeaderValue::from_static("origin"));
    h.insert(
        "access-control-allow-headers",
        HeaderValue::from_static("authorization, content-type, x-api-key, anthropic-version, anthropic-beta, openai-beta"),
    );
    h.insert("access-control-allow-methods", HeaderValue::from_static("GET, POST, DELETE, OPTIONS"));
    h.insert("access-control-expose-headers", HeaderValue::from_static("x-request-id, x-ds41gw-warning, retry-after"));
}

/// axum のミドルウェア: preflight と CORS ヘッダ。
pub async fn cors_layer(
    axum::extract::State(app): axum::extract::State<AppState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let origin = req.headers().get("origin").cloned();
    if let Some(r) = reject_rebinding(&app, &req) {
        return r;
    }
    if let Some(r) = reject_cross_site_simple_request(&app, &req) {
        return r;
    }
    if app.cfg.cors.allow_origins.is_empty() {
        return next.run(req).await;
    }
    if req.method() == Method::OPTIONS {
        let mut r = StatusCode::NO_CONTENT.into_response();
        cors_headers(&app, origin.as_ref(), r.headers_mut());
        return r;
    }
    let mut r = next.run(req).await;
    cors_headers(&app, origin.as_ref(), r.headers_mut());
    r
}

/// 別サイトのブラウザからの "simple request" (preflight の無い POST) を断る。
/// 本文の Content-Type を問わずに JSON として読むので、許可していない Origin からの
/// `text/plain` などの POST がそのまま生成を起こせてしまう (応答は読めないが GPU を使わせられる)。
/// Origin を付けない普通のクライアント (SDK、curl) には影響しない。
fn reject_cross_site_simple_request(app: &AppState, req: &axum::extract::Request) -> Option<Response> {
    if req.method() != Method::POST {
        return None;
    }
    let origin = req.headers().get("origin")?.to_str().unwrap_or("");
    let allow = &app.cfg.cors.allow_origins;
    if allow.iter().any(|a| a == "*" || a == origin) {
        return None;
    }
    let ct = req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(';').next().unwrap_or("").trim().to_ascii_lowercase())
        .unwrap_or_default();
    let simple = ct.is_empty() || ct == "text/plain" || ct == "application/x-www-form-urlencoded" || ct == "multipart/form-data";
    if !simple {
        return None;
    }
    app.metrics.inc("ds41gw_cross_site_rejects_total", &[]);
    let flavor = if req.uri().path().starts_with("/v1/messages") { crate::error::Flavor::Anthropic } else { crate::error::Flavor::OpenAI };
    Some(ApiError::new(Kind::Forbidden, "cross_site_request", "cross-site browser requests are not allowed").into_response_with(flavor))
}

/// DNS rebinding を断る。攻撃者の名前を 127.0.0.1 や Tailscale の IP に向け直すと、ブラウザからは同じ Origin になり、
/// preflight 無しで JSON の POST や GET (保存済みの応答の読み出し) ができてしまう。そのとき Host は攻撃者の名前になる。
/// ブラウザの要求 (Origin か Sec-Fetch-Site がある。同じ Origin の GET は Origin を付けないので後者も見る) で、
/// 許可していない Origin のものは、Host が IP リテラル・localhost・`server.allowed_hosts` のどれかでなければ 403 にする。
/// SDK・curl はどちらのヘッダも付けないので影響しない (Node の fetch が付けるのは Sec-Fetch-Mode だけ)。
fn reject_rebinding(app: &AppState, req: &axum::extract::Request) -> Option<Response> {
    let h = req.headers();
    let origin = h.get("origin").and_then(|v| v.to_str().ok());
    if origin.is_none() && !h.contains_key("sec-fetch-site") {
        return None;
    }
    let allow = &app.cfg.cors.allow_origins;
    if let Some(o) = origin {
        if allow.iter().any(|a| a == "*" || a == o) {
            return None;
        }
    }
    let host = req
        .headers()
        .get("host")
        .and_then(|v| v.to_str().ok())
        .or_else(|| req.uri().host())
        .unwrap_or("");
    if host_is_allowed(host, &app.cfg.server.allowed_hosts) {
        return None;
    }
    app.metrics.inc("ds41gw_cross_site_rejects_total", &[]);
    let flavor = if req.uri().path().starts_with("/v1/messages") { crate::error::Flavor::Anthropic } else { crate::error::Flavor::OpenAI };
    Some(ApiError::new(Kind::Forbidden, "host_not_allowed", "browser requests must use an IP address or an allowed host name").into_response_with(flavor))
}

/// Host ヘッダ (`name[:port]`、`[v6]:port`) の名前が IP リテラル・localhost・許可された名前か。
pub fn host_is_allowed(host: &str, allowed: &[String]) -> bool {
    let host = host.trim();
    let name = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        match host.rsplit_once(':') {
            Some((n, p)) if p.chars().all(|c| c.is_ascii_digit()) => n,
            _ => host,
        }
    };
    let name = name.trim_end_matches('.').to_ascii_lowercase();
    if name.is_empty() {
        return true;
    }
    if name.parse::<IpAddr>().is_ok() || name == "localhost" || name.ends_with(".localhost") {
        return true;
    }
    allowed.iter().any(|a| a.trim_end_matches('.').eq_ignore_ascii_case(&name))
}

#[cfg(test)]
mod tests {
    use super::host_is_allowed;

    #[test]
    fn host_check() {
        let none: Vec<String> = vec![];
        assert!(host_is_allowed("127.0.0.1:30001", &none));
        assert!(host_is_allowed("100.64.0.1:30001", &none));
        assert!(host_is_allowed("[::1]:30001", &none));
        assert!(host_is_allowed("localhost:30001", &none));
        assert!(host_is_allowed("", &none));
        assert!(!host_is_allowed("evil.example.com:30001", &none));
        assert!(!host_is_allowed("127.0.0.1.nip.io:30001", &none));
        let a = vec!["gpu.tailnet.ts.net".to_string()];
        assert!(host_is_allowed("GPU.tailnet.ts.net.:30001", &a));
        assert!(!host_is_allowed("other.tailnet.ts.net", &a));
    }
}
