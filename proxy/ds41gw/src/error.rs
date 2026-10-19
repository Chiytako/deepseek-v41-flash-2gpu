//! エラーの形を API ごとに 1 つに揃える (r7_proxy_spec.md §4.6)。

use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

/// どの API 形式でエラーを返すか。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    OpenAI,
    Anthropic,
}

/// エラーの種類。HTTP 状態と各 API の type はここから決まる。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// 400。code は個別。
    BadRequest,
    /// 上流 (ds41rt) の 400。
    UpstreamRejected,
    Auth,
    /// 403 (別サイトのブラウザからの要求など)。
    Forbidden,
    NotFound,
    TooLarge,
    /// 待ち行列の時間切れ・満杯 (OpenAI 429 / Anthropic 529)。
    Queue,
    /// ds41rt に接続できない、503。
    Unavailable,
    /// ds41rt の 500、途中失敗。
    Upstream,
    /// proxy の内部エラー。
    Internal,
}

#[derive(Debug, Clone)]
pub struct ApiError {
    pub kind: Kind,
    pub code: String,
    pub message: String,
    pub param: Option<String>,
    pub retry_after: Option<u64>,
}

impl ApiError {
    pub fn new(kind: Kind, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self { kind, code: code.into(), message: message.into(), param: None, retry_after: None }
    }
    pub fn bad(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(Kind::BadRequest, code, message)
    }
    pub fn unsupported(param: &str) -> Self {
        Self::new(Kind::BadRequest, "unsupported_parameter", format!("unsupported parameter: {param}")).with_param(param)
    }
    pub fn with_param(mut self, p: impl Into<String>) -> Self {
        self.param = Some(p.into());
        self
    }
    pub fn with_retry_after(mut self, s: u64) -> Self {
        self.retry_after = Some(s);
        self
    }

    pub fn status(&self, flavor: Flavor) -> StatusCode {
        match (self.kind, flavor) {
            (Kind::BadRequest | Kind::UpstreamRejected, _) => StatusCode::BAD_REQUEST,
            (Kind::Auth, _) => StatusCode::UNAUTHORIZED,
            (Kind::Forbidden, _) => StatusCode::FORBIDDEN,
            (Kind::NotFound, _) => StatusCode::NOT_FOUND,
            (Kind::TooLarge, _) => StatusCode::PAYLOAD_TOO_LARGE,
            (Kind::Queue, Flavor::OpenAI) => StatusCode::TOO_MANY_REQUESTS,
            (Kind::Queue | Kind::Unavailable, Flavor::Anthropic) => StatusCode::from_u16(529).unwrap(),
            (Kind::Unavailable, Flavor::OpenAI) => StatusCode::SERVICE_UNAVAILABLE,
            (Kind::Upstream, Flavor::OpenAI) => StatusCode::BAD_GATEWAY,
            (Kind::Upstream | Kind::Internal, Flavor::Anthropic) => StatusCode::INTERNAL_SERVER_ERROR,
            (Kind::Internal, Flavor::OpenAI) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn openai_type(&self) -> &'static str {
        match self.kind {
            Kind::BadRequest | Kind::UpstreamRejected | Kind::NotFound | Kind::TooLarge => "invalid_request_error",
            Kind::Auth => "authentication_error",
            Kind::Forbidden => "permission_error",
            Kind::Queue => "rate_limit_error",
            Kind::Unavailable | Kind::Upstream | Kind::Internal => "server_error",
        }
    }

    pub fn anthropic_type(&self) -> &'static str {
        match self.kind {
            Kind::BadRequest | Kind::UpstreamRejected => "invalid_request_error",
            Kind::Auth => "authentication_error",
            Kind::Forbidden => "permission_error",
            Kind::NotFound => "not_found_error",
            Kind::TooLarge => "request_too_large",
            Kind::Queue | Kind::Unavailable => "overloaded_error",
            Kind::Upstream | Kind::Internal => "api_error",
        }
    }

    pub fn body(&self, flavor: Flavor) -> Value {
        match flavor {
            Flavor::OpenAI => json!({"error": {
                "message": self.message, "type": self.openai_type(), "param": self.param, "code": self.code}}),
            Flavor::Anthropic => json!({"type": "error", "error": {"type": self.anthropic_type(), "message": self.message}}),
        }
    }

    pub fn into_response_with(self, flavor: Flavor) -> Response {
        let status = self.status(flavor);
        let mut resp = (status, axum::Json(self.body(flavor))).into_response();
        if let Some(s) = self.retry_after {
            if let Ok(v) = HeaderValue::from_str(&s.to_string()) {
                resp.headers_mut().insert("retry-after", v);
            }
        }
        resp
    }

    /// HTTP 状態の数値 (ログ用)。
    pub fn status_u16(&self, flavor: Flavor) -> u16 {
        self.status(flavor).as_u16()
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

/// recipe の変換エラーを 400 / 500 にする。
impl From<deepseek_recipe::request::ConversionError> for ApiError {
    fn from(e: deepseek_recipe::request::ConversionError) -> Self {
        use deepseek_recipe::request::ConversionError as C;
        match e {
            C::BadRequest { detail } => ApiError::bad("invalid_request", detail),
            C::Internal { detail } => ApiError::new(Kind::Internal, "internal_error", detail),
        }
    }
}
