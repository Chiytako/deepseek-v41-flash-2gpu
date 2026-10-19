//! 全 API 共通の部品: アプリの状態、要求ごとの文脈とログ、上流の差分の読み手 (Source)、
//! 早期ヘッダ付きの接続、SSE 応答の組み立て。

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::config::Config;
use crate::error::{ApiError, Flavor, Kind};
use crate::metrics::Metrics;
use crate::names::NameMap;
use crate::reasoning::{ReinjectStore, Signer};
use crate::reqlog::ReqLog;
use crate::store::Store;
use crate::tokens::Counter;
use crate::upstream::{Ev, UpStream, Upstream};
use crate::util::{now_unix_ms, rfc3339_now};

/// ds41rt の /v1/models から取る値 (§9.1)。
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub max_context_tokens: u64,
    pub max_output_tokens: u64,
}

pub struct App {
    pub cfg: Arc<Config>,
    pub up: Arc<Upstream>,
    pub metrics: Arc<Metrics>,
    pub reinject: ReinjectStore,
    pub signer: Signer,
    pub store: Store,
    pub counter: Counter,
    pub reqlog: ReqLog,
    pub model_info: RwLock<ModelInfo>,
    pub key_sems: HashMap<String, Arc<Semaphore>>,
    pub upstream_stats: RwLock<Value>,
    pub started: Instant,
}

pub type AppState = Arc<App>;

/// 要求ごとの統計 (終了の行に書く)。
#[derive(Debug, Default)]
pub struct Stats {
    pub status: u16,
    pub upstream_status: Option<u16>,
    pub queue_ms: u64,
    pub ttft_ms: Option<u64>,
    pub upstream_retries: u32,
    pub finish_reason: Option<String>,
    pub usage: Option<Value>,
    pub reasoning_tokens: usize,
    pub tool_calls: Vec<String>,
    pub reinjected: usize,
    pub warnings: Vec<String>,
    pub normalized: Vec<String>,
    pub error: Option<String>,
    pub done: bool,
    pub stream: bool,
    pub model_requested: Option<String>,
    pub outputs: Option<Value>,
    pub dsml_leaks: usize,
    pub midstream_failures: u32,
    /// 受信 (本文を読み終えた後) から上流へ送り出すまでの proxy 内の時間 (待ち行列を除く)。
    pub overhead_ms: Option<f64>,
}

pub struct Ctx {
    pub app: AppState,
    pub rid: String,
    pub api: &'static str,
    pub path: String,
    pub t0: Instant,
    pub client_ip: Option<IpAddr>,
    pub user_agent: String,
    pub key_name: Option<String>,
    pub key_sem: Option<Arc<Semaphore>>,
    pub flavor: Flavor,
    pub stats: Mutex<Stats>,
}

impl Ctx {
    /// 警告を足す。要求の値 (未知のパーツの型など) を含むので、1 件の長さと件数に上限を付ける
    /// (x-ds41gw-warning ヘッダが数十 KB になり、クライアントや中継が応答を捨てるのを防ぐ)。
    pub fn warn(&self, w: impl Into<String>) {
        const MAX_LEN: usize = 96;
        const MAX_COUNT: usize = 24;
        let mut w: String = w.into();
        if w.len() > MAX_LEN {
            let mut end = MAX_LEN;
            while !w.is_char_boundary(end) {
                end -= 1;
            }
            w.truncate(end);
            w.push('~');
        }
        let mut s = self.stats.lock().unwrap();
        if s.warnings.contains(&w) {
            return;
        }
        if s.warnings.len() >= MAX_COUNT {
            if s.warnings.last().map(String::as_str) != Some("more_warnings") {
                s.warnings.push("more_warnings".into());
            }
            return;
        }
        s.warnings.push(w);
    }
    pub fn normalized(&self, n: impl Into<String>) {
        let n = n.into();
        self.app.metrics.inc("ds41gw_normalizations_total", &[("kind", &n)]);
        let mut s = self.stats.lock().unwrap();
        if !s.normalized.contains(&n) {
            s.normalized.push(n);
        }
    }
    pub fn warnings_header(&self) -> Option<String> {
        let s = self.stats.lock().unwrap();
        if s.warnings.is_empty() {
            None
        } else {
            Some(s.warnings.join(", "))
        }
    }
    pub fn set_status(&self, status: u16) {
        self.stats.lock().unwrap().status = status;
    }
    pub fn fail(&self, e: &ApiError) {
        let mut s = self.stats.lock().unwrap();
        s.status = e.status_u16(self.flavor);
        s.error = Some(e.to_string());
        s.done = true;
    }
    pub fn mark_done(&self) {
        self.stats.lock().unwrap().done = true;
    }
    pub fn mark_ttft(&self) {
        let mut s = self.stats.lock().unwrap();
        if s.ttft_ms.is_none() {
            s.ttft_ms = Some(self.t0.elapsed().as_millis() as u64);
            let v = self.t0.elapsed().as_secs_f64();
            drop(s);
            self.app.metrics.observe("ds41gw_ttft_seconds", &[("api", self.api)], v);
        }
    }

    /// 受信の行 (replay_openai.py が読む形)。`raw` は受け取った本文 (展開後)。
    /// 画像を伏せる複製と直列化は要求ログの書き込みスレッドで行う (上流へ送る前の待ち時間に入れない)。
    pub fn log_received(&self, obj: &Value, raw: &Bytes, extra: Value) {
        if !self.app.reqlog.enabled() {
            return;
        }
        let s = self.stats.lock().unwrap();
        let mut line = json!({
            "event": "request.received.openai",
            "ts": rfc3339_now(),
            "rid": self.rid,
            "api": self.api,
            "path": self.path,
            "method": "POST",
            "client_ip": self.client_ip.map(|i| i.to_string()),
            "user_agent": self.user_agent,
            "key": self.key_name,
            "stream": s.stream,
            "model_requested": s.model_requested,
            "normalized": s.normalized,
            "warnings": s.warnings,
        });
        drop(s);
        let o = line.as_object_mut().unwrap();
        if let Value::Object(e) = extra {
            for (k, v) in e {
                o.insert(k, v);
            }
        }
        let bodies = self.app.reqlog.keeps_bodies();
        // obj が無い (検査で断った) 行は replay が読まないように obj を書かない
        let obj = (bodies && !obj.is_null()).then(|| obj.clone());
        let raw = bodies.then(|| raw.clone());
        self.app.reqlog.write_received(line, obj, raw);
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        let s = self.stats.get_mut().unwrap();
        let disconnected = !s.done;
        if disconnected {
            self.app.metrics.inc("ds41gw_client_disconnects_total", &[]);
        }
        let status = if s.status == 0 { 499 } else { s.status };
        let st = status.to_string();
        let stream = if s.stream { "true" } else { "false" };
        self.app.metrics.inc("ds41gw_requests_total", &[("api", self.api), ("stream", stream), ("status", &st)]);
        if !self.app.reqlog.enabled() {
            return;
        }
        let mut line = json!({
            "event": "request.finished",
            "ts": rfc3339_now(),
            "rid": self.rid,
            "api": self.api,
            "status": status,
            "upstream_status": s.upstream_status,
            "queue_ms": s.queue_ms,
            "ttft_ms": s.ttft_ms,
            "overhead_ms": s.overhead_ms.map(|x| (x * 100.0).round() / 100.0),
            "dur_ms": self.t0.elapsed().as_millis() as u64,
            "upstream_retries": s.upstream_retries,
            "finish_reason": s.finish_reason,
            "usage": s.usage,
            "reasoning_tokens": s.reasoning_tokens,
            "tool_calls": s.tool_calls,
            "reinjected": s.reinjected,
            "warnings": s.warnings,
            "error": s.error,
            "client_disconnected": disconnected,
            "dsml_leaks": s.dsml_leaks,
            "midstream_failures": s.midstream_failures,
        });
        if let Some(o) = s.outputs.take() {
            line.as_object_mut().unwrap().insert("outputs".into(), o);
        }
        self.app.reqlog.write(line);
    }
}

/// 認証と要求 id の発行 (§3.1 の 2、§12.1)。
pub fn begin(
    app: &AppState,
    api: &'static str,
    path: &str,
    headers: &HeaderMap,
    addr: Option<std::net::SocketAddr>,
    flavor: Flavor,
) -> Result<Arc<Ctx>, ApiError> {
    let rid = format!("gw_{}", crate::util::hex32());
    let ip = addr.map(|a| a.ip());
    let ua = headers.get("user-agent").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let (key_name, key_sem) = crate::auth::check(app, headers, ip)?;
    Ok(Arc::new(Ctx {
        app: app.clone(),
        rid,
        api,
        path: path.to_string(),
        t0: Instant::now(),
        client_ip: ip,
        user_agent: ua,
        key_name,
        key_sem,
        flavor,
        stats: Mutex::new(Stats::default()),
    }))
}

/// 要求本文を JSON として読む (Content-Type は問わない、§3.1 の 1)。
pub fn parse_json(body: &Bytes) -> Result<Value, ApiError> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::bad("invalid_json", format!("Failed to parse the request body as JSON: {e}")))?;
    if !v.is_object() {
        return Err(ApiError::bad("invalid_json", "request body must be a JSON object"));
    }
    Ok(v)
}

/// 応答の 1 回分を集めたもの。
#[derive(Debug, Default, Clone)]
pub struct Collected {
    pub id: String,
    pub created: u64,
    pub fingerprint: Option<String>,
    pub reasoning: String,
    pub content: String,
    pub tools: Vec<ToolAcc>,
    pub finish: Option<String>,
    pub stop: Option<String>,
    pub usage: Option<Value>,
    pub reasoning_tokens: usize,
}

#[derive(Debug, Default, Clone)]
pub struct ToolAcc {
    pub index: usize,
    pub id: String,
    pub name: String,
    pub args: String,
}

impl Collected {
    pub fn feed(&mut self, ev: &Ev) {
        match ev {
            Ev::Start { id, created, fingerprint } => {
                self.id = id.clone();
                self.created = *created;
                self.fingerprint = fingerprint.clone();
            }
            Ev::Reasoning(s) => self.reasoning.push_str(s),
            Ev::Content(s) => self.content.push_str(s),
            Ev::ToolStart { index, id, name } => {
                self.tools.push(ToolAcc { index: *index, id: id.clone(), name: name.clone(), args: String::new() })
            }
            Ev::ToolArgs { index, args } => {
                if let Some(t) = self.tools.iter_mut().rev().find(|t| t.index == *index) {
                    t.args.push_str(args);
                }
            }
            Ev::Finish { reason, usage, stop } => {
                self.finish = Some(reason.clone());
                self.usage = usage.clone();
                self.stop = stop.clone();
            }
        }
    }
}

/// proxy 側での stop 文字列の照合 (Messages、§7.3)。
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    hold: String,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), hold: String::new() }
    }
    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }
    /// (送ってよい文字列, 一致した stop)。
    pub fn push(&mut self, delta: &str) -> (String, Option<String>) {
        let mut buf = std::mem::take(&mut self.hold);
        buf.push_str(delta);
        let mut best: Option<(usize, &String)> = None;
        for s in &self.stops {
            if let Some(p) = buf.find(s.as_str()) {
                if best.map_or(true, |(bp, _)| p < bp) {
                    best = Some((p, s));
                }
            }
        }
        if let Some((p, s)) = best {
            return (buf[..p].to_string(), Some(s.clone()));
        }
        // 最長の stop の長さ − 1 だけ保留する
        let mut keep_from = buf.len();
        for s in &self.stops {
            let max = s.len().saturating_sub(1).min(buf.len());
            for len in (1..=max).rev() {
                let start = buf.len() - len;
                if buf.is_char_boundary(start) && s.starts_with(&buf[start..]) {
                    keep_from = keep_from.min(start);
                    break;
                }
            }
        }
        self.hold = buf[keep_from..].to_string();
        (buf[..keep_from].to_string(), None)
    }
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.hold)
    }
}

/// 上流の差分を API の出力器に渡す前に通す層:
/// ツール呼び出しの上限、`keep_tools` での呼び出しの破棄、stop の照合、集計。
pub struct Source {
    up: UpStream,
    pub col: Collected,
    cap: usize,
    drop_tools: bool,
    stop: StopMatcher,
    pending: std::collections::VecDeque<Ev>,
    ended: bool,
    finalized: bool,
    pub prompt_estimate: usize,
    pub ctx: Arc<Ctx>,
}

impl Source {
    pub fn new(up: UpStream, ctx: Arc<Ctx>, opts: &SourceOpts) -> Self {
        {
            let mut s = ctx.stats.lock().unwrap();
            s.queue_ms = s.queue_ms.max(up.queue_ms);
            s.upstream_retries += up.retries;
            s.upstream_status = Some(200);
        }
        Self {
            up,
            col: Collected::default(),
            cap: opts.max_tool_calls,
            drop_tools: opts.drop_tool_calls,
            stop: StopMatcher::new(opts.stop.clone()),
            pending: Default::default(),
            ended: false,
            finalized: false,
            prompt_estimate: opts.prompt_estimate,
            ctx,
        }
    }

    fn synth_usage(&self) -> Value {
        let out = self.ctx.app.counter.count(&self.col.reasoning)
            + self.ctx.app.counter.count(&self.col.content)
            + self.col.tools.iter().map(|t| self.ctx.app.counter.count(&t.args) + 4).sum::<usize>();
        json!({"prompt_tokens": self.prompt_estimate, "completion_tokens": out, "total_tokens": self.prompt_estimate + out,
               "estimated": true})
    }

    /// 次の差分。None は正常終了、Err は途中失敗。キャンセル安全。
    pub async fn next(&mut self) -> Option<Result<Ev, String>> {
        loop {
            if let Some(mut ev) = self.pending.pop_front() {
                if let Ev::Finish { usage, .. } = &mut ev {
                    // reasoning_tokens を補う (ds41rt は出さない、§5.4.1)
                    let rt = self.ctx.app.counter.count(&self.col.reasoning);
                    self.col.reasoning_tokens = rt;
                    let u = match usage.take() {
                        Some(u) => u,
                        None => self.synth_usage(),
                    };
                    *usage = Some(crate::upstream::enrich_usage(&u, rt));
                }
                self.col.feed(&ev);
                if matches!(ev, Ev::Reasoning(_) | Ev::Content(_) | Ev::ToolStart { .. }) {
                    self.ctx.mark_ttft();
                }
                if matches!(ev, Ev::Finish { .. }) {
                    // 終わりの差分を渡す前に後始末する。クライアントが finish を受けてすぐ切ると、
                    // この Source は次の poll を待たずに捨てられ、再注入ストアへの保存が抜ける。
                    self.finalize();
                }
                return Some(Ok(ev));
            }
            if self.ended {
                return None;
            }
            let item = self.up.next().await;
            let ev = match item {
                None => {
                    self.ended = true;
                    continue;
                }
                Some(Err(e)) => {
                    self.ended = true;
                    self.ctx.app.metrics.inc("ds41gw_midstream_failures_total", &[]);
                    self.ctx.stats.lock().unwrap().midstream_failures += 1;
                    return Some(Err(e.0));
                }
                Some(Ok(ev)) => ev,
            };
            match ev {
                Ev::ToolStart { .. } | Ev::ToolArgs { .. } if self.drop_tools => {}
                Ev::ToolStart { index, id, name } => {
                    if self.cap > 0 && self.col.tools.len() >= self.cap {
                        // 上限 (§11.7): 以降を捨てて、上流を切って閉じる
                        self.ctx.app.metrics.inc("ds41gw_tool_call_cap_total", &[]);
                        self.ctx.warn("tool_call_cap");
                        let usage = Some(self.synth_usage());
                        self.pending.push_back(Ev::Finish { reason: "tool_calls".into(), usage, stop: None });
                        self.ended = true;
                        continue;
                    }
                    self.pending.push_back(Ev::ToolStart { index, id, name });
                }
                Ev::Content(c) if !self.stop.is_empty() => {
                    let (out, hit) = self.stop.push(&c);
                    if !out.is_empty() {
                        self.pending.push_back(Ev::Content(out));
                    }
                    if let Some(s) = hit {
                        let usage = Some(self.synth_usage_after(&self.pending));
                        self.pending.push_back(Ev::Finish { reason: "stop_sequence".into(), usage, stop: Some(s) });
                        self.ended = true;
                    }
                }
                Ev::Finish { reason, usage, stop } => {
                    if !self.stop.is_empty() {
                        let rest = self.stop.finish();
                        if !rest.is_empty() {
                            self.pending.push_back(Ev::Content(rest));
                        }
                    }
                    let reason = if self.drop_tools && reason == "tool_calls" { "stop".to_string() } else { reason };
                    self.pending.push_back(Ev::Finish { reason, usage, stop });
                }
                other => self.pending.push_back(other),
            }
        }
    }

    fn synth_usage_after(&self, pending: &std::collections::VecDeque<Ev>) -> Value {
        let mut extra = 0;
        for e in pending {
            if let Ev::Content(c) = e {
                extra += self.ctx.app.counter.count(c);
            }
        }
        let mut u = self.synth_usage();
        if let Some(n) = u.get("completion_tokens").and_then(Value::as_u64) {
            let n = n as usize + extra;
            u["completion_tokens"] = json!(n);
            u["total_tokens"] = json!(self.prompt_estimate + n);
        }
        u
    }

    pub fn dsml_leaks(&self) -> usize {
        self.up.dsml_leaks
    }

    /// 正常終了の後始末: reasoning_tokens の計数、再注入ストアへの保存、統計。
    pub fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        self.finalized = true;
        let app = &self.ctx.app;
        if app.cfg.reasoning.reinject && !self.col.reasoning.is_empty() {
            let ids: Vec<String> = self.col.tools.iter().map(|t| t.id.clone()).collect();
            app.reinject.put(&self.col.reasoning, &ids, &self.col.content);
        }
        let leaks = self.up.dsml_leaks;
        if leaks > 0 {
            app.metrics.add("ds41gw_dsml_leak_total", &[], leaks as u64);
            self.ctx.warn("dsml_leak");
        }
        let mut s = self.ctx.stats.lock().unwrap();
        s.finish_reason = self.col.finish.clone();
        s.usage = self.col.usage.clone();
        s.reasoning_tokens += self.col.reasoning_tokens;
        s.tool_calls.extend(self.col.tools.iter().map(|t| t.name.clone()));
        s.dsml_leaks += leaks;
        if app.reqlog.cfg.outputs {
            let o = json!({"content": self.col.content, "reasoning": self.col.reasoning,
                "tool_calls": self.col.tools.iter().map(|t| json!({"id": t.id, "name": t.name, "arguments": t.args})).collect::<Vec<_>>()});
            match &mut s.outputs {
                Some(Value::Array(a)) => a.push(o),
                _ => s.outputs = Some(json!([o])),
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SourceOpts {
    pub max_tool_calls: usize,
    pub drop_tool_calls: bool,
    pub stop: Vec<String>,
    pub prompt_estimate: usize,
}

/// 上流への接続 (n 本)。いずれかが失敗したら全部を閉じてエラーを返す。
/// n>1 のときは n 枠をまとめて取ってから送る (1 枠ずつだと、n 本の要求どうしが残りの枠を待って詰まる)。
pub async fn connect_all(
    ctx: Arc<Ctx>,
    bodies: Vec<Value>,
    names: Arc<NameMap>,
    opts: SourceOpts,
) -> Result<Vec<Source>, ApiError> {
    let app = ctx.app.clone();
    {
        // ds41gw_overhead_seconds (§13.4): 受信から上流へ送り出すまで (待ち行列の前まで)。再試行の 2 回目は数えない
        let mut s = ctx.stats.lock().unwrap();
        if s.overhead_ms.is_none() {
            let d = ctx.t0.elapsed().as_secs_f64();
            s.overhead_ms = Some(d * 1000.0);
            app.metrics.observe("ds41gw_overhead_seconds", &[("api", ctx.api)], d);
        }
    }
    let results: Vec<Result<crate::upstream::UpStream, ApiError>> = if bodies.len() <= 1 {
        let futs = bodies.iter().map(|b| app.up.call(b, names.clone(), ctx.key_sem.clone()));
        futures::future::join_all(futs).await
    } else {
        let n = bodies.len();
        let key_max = ctx
            .key_name
            .as_ref()
            .and_then(|k| app.cfg.auth.keys.iter().find(|x| &x.name == k))
            .and_then(|k| k.max_concurrency)
            .map(|m| m.max(1));
        if let Some(m) = key_max.filter(|m| n > *m) {
            return Err(ApiError::bad("too_many_choices", format!("this request needs {n} concurrent upstream streams; this API key allows {m}"))
                .with_param("n"));
        }
        let t0 = std::time::Instant::now();
        let deadline = t0 + Duration::from_secs_f64(app.cfg.queue.timeout_s);
        let slots = app.up.acquire_n(ctx.key_sem.clone(), n, deadline).await?;
        let queue_ms = t0.elapsed().as_millis() as u64;
        let futs = bodies.iter().zip(slots).map(|(b, slot)| app.up.call_with_slot(b, names.clone(), slot, queue_ms, deadline));
        futures::future::join_all(futs).await
    };
    let mut out = Vec::new();
    for r in results {
        match r {
            Ok(u) => out.push(Source::new(u, ctx.clone(), &opts)),
            Err(e) => {
                let mut s = ctx.stats.lock().unwrap();
                s.upstream_status = Some(match e.kind {
                    Kind::UpstreamRejected => 400,
                    Kind::Queue => 429,
                    Kind::Unavailable => 503,
                    _ => 500,
                });
                return Err(e);
            }
        }
    }
    Ok(out)
}

pub type ConnectFut = BoxFuture<'static, Result<Vec<Source>, ApiError>>;

/// 早期ヘッダ (§11.4): ストリームで `early_headers_after` を超えて待ったら、先にヘッダを送る。
pub enum Connect {
    Ready(Vec<Source>),
    Late(ConnectFut),
}

pub async fn connect_early(app: &AppState, mut fut: ConnectFut, stream: bool) -> Result<Connect, ApiError> {
    let early = app.cfg.client.early_headers_after_s;
    if !stream || early <= 0.0 {
        return fut.await.map(Connect::Ready);
    }
    tokio::select! {
        r = &mut fut => r.map(Connect::Ready),
        _ = tokio::time::sleep(Duration::from_secs_f64(early)) => Ok(Connect::Late(fut)),
    }
}

/// 差分の項目 (複数本を合流させるとき用)。
pub enum Item {
    Ev(Ev),
    Mid(String),
    /// 1 本が正常に終わった。集計を返す。
    Done(Box<Collected>),
}

/// Source を (添字, 項目) のストリームにする。
pub fn source_stream(i: usize, src: Source) -> BoxStream<'static, (usize, Item)> {
    futures::stream::unfold(Some(src), move |st| async move {
        let mut s = st?;
        match s.next().await {
            Some(Ok(ev)) => Some(((i, Item::Ev(ev)), Some(s))),
            Some(Err(m)) => Some(((i, Item::Mid(m)), None)),
            None => {
                s.finalize();
                let col = Box::new(s.col.clone());
                Some(((i, Item::Done(col)), None))
            }
        }
    })
    .boxed()
}

pub fn merge_sources(srcs: Vec<Source>) -> BoxStream<'static, (usize, Item)> {
    futures::stream::select_all(srcs.into_iter().enumerate().map(|(i, s)| source_stream(i, s))).boxed()
}

pub fn keepalive_interval(app: &AppState) -> Duration {
    Duration::from_secs_f64(app.cfg.client.keepalive_interval_s.max(0.05))
}

/// SSE の応答を作る。
pub fn sse_response(ctx: &Ctx, stream: BoxStream<'static, Bytes>) -> Response {
    ctx.set_status(200);
    let body = Body::from_stream(stream.map(Ok::<_, std::convert::Infallible>));
    let mut r = Response::new(body);
    *r.status_mut() = StatusCode::OK;
    let h = r.headers_mut();
    h.insert("content-type", HeaderValue::from_static("text/event-stream"));
    h.insert("cache-control", HeaderValue::from_static("no-cache"));
    h.insert("x-accel-buffering", HeaderValue::from_static("no"));
    decorate(ctx, h);
    r
}

pub fn json_response(ctx: &Ctx, status: StatusCode, v: &Value) -> Response {
    ctx.set_status(status.as_u16());
    let mut r = Response::new(Body::from(serde_json::to_vec(v).unwrap_or_default()));
    *r.status_mut() = status;
    r.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    decorate(ctx, r.headers_mut());
    r
}

pub fn error_response(ctx: &Ctx, e: ApiError) -> Response {
    ctx.fail(&e);
    let mut r = e.into_response_with(ctx.flavor);
    decorate(ctx, r.headers_mut());
    r
}

pub fn decorate(ctx: &Ctx, h: &mut HeaderMap) {
    if let Ok(v) = HeaderValue::from_str(&ctx.rid) {
        h.insert("x-request-id", v);
    }
    if let Some(w) = ctx.warnings_header() {
        // 要求由来の文字 (非 ASCII、制御文字) が混じるとヘッダごと落ちるので置き換える
        let w: String = w.chars().map(|c| if c.is_ascii_graphic() || c == ' ' { c } else { '?' }).collect();
        if let Ok(v) = HeaderValue::from_str(&w) {
            h.insert("x-ds41gw-warning", v);
        }
    }
}

pub fn sse_data(v: &Value) -> Bytes {
    Bytes::from(format!("data: {}\n\n", v))
}

pub fn sse_event(name: &str, v: &Value) -> Bytes {
    Bytes::from(format!("event: {name}\ndata: {v}\n\n"))
}

pub fn created_now() -> u64 {
    (now_unix_ms() / 1000) as u64
}

/// 非ストリームの集約 (§5.4.2)。途中失敗は `retry_midstream_nonstream` 回だけ最初からやり直す。
pub async fn aggregate_with_retry(
    ctx: Arc<Ctx>,
    body: Value,
    names: Arc<NameMap>,
    opts: SourceOpts,
    first: Option<Source>,
) -> Result<Collected, ApiError> {
    let mut attempt = 0u32;
    let mut src = first;
    loop {
        let mut s = match src.take() {
            Some(s) => s,
            None => {
                let mut v = connect_all(ctx.clone(), vec![body.clone()], names.clone(), opts.clone()).await?;
                v.pop().unwrap()
            }
        };
        let mut failed = None;
        loop {
            match s.next().await {
                Some(Ok(_)) => {}
                Some(Err(m)) => {
                    failed = Some(m);
                    break;
                }
                None => break,
            }
        }
        match failed {
            None => {
                s.finalize();
                return Ok(s.col);
            }
            Some(m) => {
                drop(s);
                if attempt < ctx.app.cfg.upstream.retry_midstream_nonstream {
                    attempt += 1;
                    ctx.app.metrics.inc("ds41gw_upstream_retries_total", &[("reason", "midstream")]);
                    ctx.stats.lock().unwrap().upstream_retries += 1;
                    continue;
                }
                return Err(ApiError::new(Kind::Upstream, "upstream_error", format!("upstream: {m}")));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stop_matcher_boundaries() {
        let text = "Hello world. END here";
        let idx: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        for &i in &idx {
            let (a, b) = text.split_at(i);
            let mut m = StopMatcher::new(vec!["END".into(), "zzz".into()]);
            let mut out = String::new();
            let mut hit = None;
            for part in [a, b] {
                if hit.is_some() {
                    break;
                }
                let (o, h) = m.push(part);
                out.push_str(&o);
                hit = h;
            }
            assert_eq!(out, "Hello world. ", "split {i}");
            assert_eq!(hit.as_deref(), Some("END"));
        }
        let mut m = StopMatcher::new(vec!["END".into()]);
        let (o, h) = m.push("no stop E");
        assert_eq!((o.as_str(), h), ("no stop ", None));
        assert_eq!(m.finish(), "E");
    }
}

/// 非ストリームの応答。`nonstream_whitespace_keepalive` が on なら、待つ間に空白を送る (§11.4)。
pub async fn nonstream_response(
    ctx: Arc<Ctx>,
    fut: BoxFuture<'static, Result<(StatusCode, Value), ApiError>>,
) -> Response {
    if ctx.app.cfg.client.nonstream_whitespace_keepalive {
        let ka = keepalive_interval(&ctx.app);
        let c2 = ctx.clone();
        let s = async_stream::stream! {
            let mut fut = fut;
            loop {
                match tokio::time::timeout(ka, &mut fut).await {
                    Err(_) => yield Bytes::from_static(b" "),
                    Ok(Ok((_, v))) => { c2.mark_done(); yield Bytes::from(serde_json::to_vec(&v).unwrap_or_default()); break; }
                    Ok(Err(e)) => { c2.fail(&e); yield Bytes::from(serde_json::to_vec(&e.body(c2.flavor)).unwrap_or_default()); break; }
                }
            }
        };
        ctx.set_status(200);
        let mut r = Response::new(Body::from_stream(s.map(Ok::<_, std::convert::Infallible>)));
        r.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
        decorate(&ctx, r.headers_mut());
        return r;
    }
    match fut.await {
        Ok((st, v)) => {
            ctx.mark_done();
            json_response(&ctx, st, &v)
        }
        Err(e) => error_response(&ctx, e),
    }
}
