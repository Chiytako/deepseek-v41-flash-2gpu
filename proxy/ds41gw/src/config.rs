//! 設定ファイル (TOML)。既定値は r7_proxy_spec.md §14 に従う。
//! 未指定の項目はすべて既定値になるので、空のファイルでも起動できる。

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub server: Server,
    pub upstream: Upstream,
    pub queue: Queue,
    pub client: Client,
    pub models: Models,
    pub sampling: Sampling,
    pub thinking: Thinking,
    pub limits: Limits,
    pub compat: Compat,
    pub reasoning: Reasoning,
    pub state: State,
    pub tokens: Tokens,
    pub log: Log,
    pub auth: Auth,
    pub cors: Cors,
    pub completions: Completions,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Server {
    pub listen: Vec<String>,
    pub max_body_mb: usize,
    /// SIGTERM の後に実行中の要求を待つ秒数。
    pub shutdown_grace_s: u64,
    /// 要求ヘッダを読み終えるまでの秒数 (slowloris 対策)。keep-alive で次の要求を待つ間にも数える。0 で無効。
    pub header_read_timeout_s: f64,
    /// 本文の読み取りで、次のかたまりが届くまで待つ秒数。0 で無効。
    pub body_idle_timeout_s: f64,
    /// 本文を読み終えるまでの合計秒数。0 で無効。
    pub body_read_timeout_s: f64,
    /// ブラウザ (Origin 付き) の要求で許す Host 名。IP リテラルと localhost は常に許す (DNS rebinding 対策)。
    pub allowed_hosts: Vec<String>,
}
impl Default for Server {
    fn default() -> Self {
        Self {
            listen: vec!["127.0.0.1:30001".into()],
            max_body_mb: 96,
            shutdown_grace_s: 30,
            header_read_timeout_s: 60.0,
            body_idle_timeout_s: 60.0,
            body_read_timeout_s: 600.0,
            allowed_hosts: vec![],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Upstream {
    pub url: String,
    pub model: String,
    pub max_concurrency: usize,
    pub connect_timeout_s: f64,
    pub idle_timeout_s: f64,
    pub retry_midstream_nonstream: u32,
    /// 接続拒否・リセットの再試行を続ける合計秒数 (§11.3)。
    pub connect_retry_total_s: f64,
    /// 429 の再試行の上限秒数 (queue.timeout の残りとの小さい方)。
    pub retry_429_max_s: f64,
}
impl Default for Upstream {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:8000".into(),
            model: "deepseek-ai/DeepSeek-V4.1-Flash".into(),
            max_concurrency: 16,
            connect_timeout_s: 2.0,
            idle_timeout_s: 600.0,
            retry_midstream_nonstream: 1,
            connect_retry_total_s: 30.0,
            retry_429_max_s: 600.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Queue {
    pub max_waiting: usize,
    pub timeout_s: f64,
}
impl Default for Queue {
    fn default() -> Self {
        Self { max_waiting: 256, timeout_s: 600.0 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Client {
    pub keepalive_interval_s: f64,
    pub early_headers_after_s: f64,
    pub nonstream_whitespace_keepalive: bool,
}
impl Default for Client {
    fn default() -> Self {
        Self { keepalive_interval_s: 15.0, early_headers_after_s: 10.0, nonstream_whitespace_keepalive: false }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Models {
    /// "any" | "listed"
    pub alias_mode: String,
    pub aliases: Vec<String>,
    pub echo_requested: bool,
}
impl Default for Models {
    fn default() -> Self {
        Self {
            alias_mode: "any".into(),
            aliases: vec!["DeepSeek-V4.1-Flash".into(), "deepseek-v4.1-flash".into(), "GLM-5.3-Flash".into()],
            echo_requested: true,
        }
    }
}

/// `default = "recommended" | "greedy"`、または `temperature` / `top_p` を直接書く。
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Sampling {
    pub default: String,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
}
impl Default for Sampling {
    fn default() -> Self {
        Self { default: "recommended".into(), temperature: None, top_p: None }
    }
}
impl Sampling {
    /// temperature を省いた要求に入れる値。None なら何も入れない (greedy)。
    pub fn default_pair(&self) -> Option<(f64, f64)> {
        if let Some(t) = self.temperature {
            return Some((t, self.top_p.unwrap_or(1.0)));
        }
        match self.default.as_str() {
            "greedy" => None,
            _ => Some((1.0, 1.0)),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Thinking {
    /// "enabled" | "disabled"
    pub default: String,
    /// effort を省いた要求の値。段階名 (effort_map の鍵) か "1"〜"100"。
    pub default_effort: String,
    /// 段階名 → effort の値 (1〜100、0 は思考なし)。鍵は大小文字を区別しない。
    /// 設定に書いた鍵だけを既定の表 ([`default_effort_map`]) に上書きする。
    /// 値は整数のほか、既定の表にある段階名 ("low" など) でもよい (旧形式の設定のため)。
    #[serde(deserialize_with = "effort_map_de")]
    pub effort_map: BTreeMap<String, u8>,
    pub auto_off_below_max_tokens: u64,
    /// budget_tokens → effort の境界 (§4.3.3)。[low (50) の上限, high (75) の上限)。それ以上は max (100)。
    pub budget_low_below: u64,
    pub budget_high_below: u64,
    /// true: ds41rt に reasoning_effort を整数 (1〜100) で送る (ds41rt 改修版 v15-effort 以降)。
    /// false: 段階名 (≤61 low、62〜88 high、他 max) に戻して送る (段階名しか受けない旧 ds41rt 用)。
    pub numeric_upstream: bool,
}
impl Default for Thinking {
    fn default() -> Self {
        Self {
            default: "enabled".into(),
            default_effort: "high".into(),
            effort_map: default_effort_map(),
            auto_off_below_max_tokens: 0,
            budget_low_below: 8192,
            budget_high_below: 32768,
            numeric_upstream: true,
        }
    }
}

/// 段階名 → effort の既定の表。low 50、high 75、max 100 は DeepSeek の公式の encoding どおり。
/// medium 62 と xhigh 88 は独自の対応 (公式は low/high/max の 3 つだけを定め、deepseek-recipe は medium/xhigh を 75 に写す)。
/// ds41rt 改修版の reasoning_effort の解き方 (minimal は low、medium 62、xhigh 88、none/off/disabled は思考なし) と揃える。
pub fn default_effort_map() -> BTreeMap<String, u8> {
    [
        ("none", 0),
        ("off", 0),
        ("disabled", 0),
        ("minimal", 50),
        ("low", 50),
        ("medium", 62),
        ("high", 75),
        ("xhigh", 88),
        ("max", 100),
        ("ultra", 100),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

fn effort_map_de<'de, D: serde::Deserializer<'de>>(d: D) -> Result<BTreeMap<String, u8>, D::Error> {
    use serde::de::Error;
    let defaults = default_effort_map();
    let user: BTreeMap<String, toml::Value> = BTreeMap::deserialize(d)?;
    let mut m = defaults.clone();
    for (k, v) in user {
        let k = k.trim().to_ascii_lowercase();
        let n = match &v {
            toml::Value::Integer(i) if (0..=100).contains(i) => *i as u8,
            toml::Value::String(s) => match defaults.get(&s.trim().to_ascii_lowercase()) {
                Some(n) => *n,
                None => match s.trim().parse::<u8>() {
                    Ok(n) if n <= 100 => n,
                    _ => return Err(D::Error::custom(format!("thinking.effort_map.{k}: unknown level {s:?} (use an integer 0-100)"))),
                },
            },
            other => return Err(D::Error::custom(format!("thinking.effort_map.{k}: expected an integer 0-100, got {other}"))),
        };
        m.insert(k, n);
    }
    Ok(m)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Limits {
    pub default_max_tokens: u64,
    pub max_n: usize,
    pub max_tool_calls_per_response: usize,
    /// "drop_oldest" | "reject"
    pub image_overflow: String,
    pub max_images: usize,
    /// これを超える文字数の要求だけ、文脈長を事前に数える (§11.6)。
    pub precheck_chars: usize,
    pub max_context_tokens: u64,
    pub max_stop: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            default_max_tokens: 262_144,
            max_n: 8,
            max_tool_calls_per_response: 32,
            image_overflow: "drop_oldest".into(),
            max_images: 16,
            precheck_chars: 3_000_000,
            max_context_tokens: 1_048_576,
            max_stop: 16,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Compat {
    /// "reject" | "ignore"
    pub unsupported: String,
    pub repair_tool_history: bool,
    /// "flatten" | "keep_tools"
    pub tool_none_strategy: String,
    pub strip_dsml: bool,
    pub null_content_for_tool_calls: bool,
    pub ignore_builtin_tools: Vec<String>,
    /// include_usage 指定時に usage を別チャンクに分ける (§5.4.1)。
    pub split_usage_chunk: bool,
    /// include_usage 指定が無いときに最終チャンクの usage を消す。
    pub drop_usage_without_include: bool,
}
impl Default for Compat {
    fn default() -> Self {
        Self {
            unsupported: "reject".into(),
            repair_tool_history: true,
            tool_none_strategy: "flatten".into(),
            strip_dsml: true,
            null_content_for_tool_calls: false,
            ignore_builtin_tools: vec![],
            split_usage_chunk: true,
            drop_usage_without_include: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Reasoning {
    /// "auto" | "text" | "summary"
    pub output_mode: String,
    pub always_encrypted: bool,
    pub reinject: bool,
    pub reinject_store_mb: usize,
    pub reinject_ttl_h: f64,
}
impl Default for Reasoning {
    fn default() -> Self {
        Self { output_mode: "auto".into(), always_encrypted: false, reinject: true, reinject_store_mb: 512, reinject_ttl_h: 24.0 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct State {
    /// "sqlite" | "memory"
    pub backend: String,
    pub path: PathBuf,
    pub memory_items: usize,
    /// プロセス内 LRU の上限 (MB)。件数だけだと、画像 (data URL) を含む input を持つ応答で
    /// 2000 件 × 数十 MB まで膨らみうるので、バイト数でも抑える。
    pub memory_mb: usize,
    pub ttl_days: f64,
    pub max_gb: f64,
    /// HMAC 鍵 (§10.2)。無ければ初回起動時に生成する。
    pub secret_key_path: PathBuf,
}
impl Default for State {
    fn default() -> Self {
        Self {
            backend: "sqlite".into(),
            path: "/var/lib/ds41gw/responses.sqlite".into(),
            memory_items: 2000,
            memory_mb: 1024,
            ttl_days: 30.0,
            max_gb: 20.0,
            secret_key_path: "/var/lib/ds41gw/secret.key".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Tokens {
    pub tokenizer: PathBuf,
    pub image_estimate: usize,
}
impl Default for Tokens {
    fn default() -> Self {
        Self { tokenizer: "/models/tokenizer.json".into(), image_estimate: 1024 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Log {
    pub dir: PathBuf,
    /// "full" | "meta"
    pub bodies: String,
    pub outputs: bool,
    /// "hash" | "full"
    pub images: String,
    pub retention_days: u64,
    pub max_total_gb: f64,
    /// false にすると要求ログを書かない (試験用)。
    pub enabled: bool,
}
impl Default for Log {
    fn default() -> Self {
        Self {
            dir: "/var/log/ds41gw".into(),
            bodies: "full".into(),
            outputs: false,
            images: "hash".into(),
            retention_days: 14,
            max_total_gb: 50.0,
            enabled: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiKey {
    pub name: String,
    /// キーの SHA-256 (16 進、小文字)。
    pub sha256: String,
    pub max_concurrency: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Auth {
    pub keys: Vec<ApiKey>,
    pub allow_without_key: Vec<String>,
}
impl Default for Auth {
    fn default() -> Self {
        Self { keys: vec![], allow_without_key: vec!["127.0.0.1/32".into(), "::1/128".into(), "172.17.0.0/16".into()] }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct Cors {
    pub allow_origins: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Completions {
    pub enabled: bool,
    pub thinking: bool,
    pub default_max_tokens: u64,
}
impl Default for Completions {
    fn default() -> Self {
        Self { enabled: true, thinking: false, default_max_tokens: 16 }
    }
}

impl Config {
    pub fn load(path: &std::path::Path) -> anyhow_lite::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| anyhow_lite::err(format!("read {}: {e}", path.display())))?;
        let cfg: Config = toml::from_str(&text).map_err(|e| anyhow_lite::err(format!("parse {}: {e}", path.display())))?;
        crate::thinking::validate_config(&cfg.thinking).map_err(|e| anyhow_lite::err(format!("{}: {e}", path.display())))?;
        Ok(cfg)
    }
}

/// 依存を増やさないための最小のエラー型。
pub mod anyhow_lite {
    #[derive(Debug)]
    pub struct Error(pub String);
    impl std::fmt::Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }
    impl std::error::Error for Error {}
    pub type Result<T> = std::result::Result<T, Error>;
    pub fn err(s: impl Into<String>) -> Error {
        Error(s.into())
    }
}
