//! ds41rt への接続: 待ち行列 (同時数の制限)、再試行、SSE の解析と差分イベントへの写像。
//! 上流には常に `stream:true` で送る (r7_proxy_spec.md §3、§11)。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::config::Config;
use crate::dsml::DsmlFilter;
use crate::error::{ApiError, Kind};
use crate::metrics::Metrics;
use crate::names::NameMap;
use crate::util::truncate;

/// 上流の 1 応答から取り出す差分 (§3.2)。
#[derive(Debug, Clone, PartialEq)]
pub enum Ev {
    Start { id: String, created: u64, fingerprint: Option<String> },
    Reasoning(String),
    Content(String),
    ToolStart { index: usize, id: String, name: String },
    ToolArgs { index: usize, args: String },
    /// `finish_reason` と usage (ds41rt の値のまま)。stop は proxy が照合した stop 文字列。
    Finish { reason: String, usage: Option<Value>, stop: Option<String> },
}

/// 途中失敗 (finish_reason の前に EOF・読み取りエラー・無音の時間切れ)。
#[derive(Debug, Clone)]
pub struct MidstreamError(pub String);

pub struct Upstream {
    pub cfg: Arc<Config>,
    http: reqwest::Client,
    sem: Arc<Semaphore>,
    pub waiting: AtomicUsize,
    pub inflight: Arc<AtomicUsize>,
    /// 直近の待ち時間の指数平均 (ミリ秒)。Retry-After の計算に使う。
    avg_wait_ms: AtomicU64,
    pub metrics: Arc<Metrics>,
    pub shutting_down: std::sync::atomic::AtomicBool,
}

/// 上流の枠 (ストリームの終わりか drop で解放する)。
pub struct Slot {
    _permit: OwnedSemaphorePermit,
    _key_permit: Option<OwnedSemaphorePermit>,
    inflight: Arc<AtomicUsize>,
}
impl Drop for Slot {
    fn drop(&mut self) {
        self.inflight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// 接続済みの上流ストリーム。
pub struct UpStream {
    resp: futures::stream::BoxStream<'static, Result<Bytes, reqwest::Error>>,
    buf: Vec<u8>,
    pending: VecDeque<Ev>,
    names: Arc<NameMap>,
    dsml: Option<DsmlFilter>,
    pub dsml_leaks: usize,
    finished: bool,
    done: bool,
    err: Option<String>,
    err_reported: bool,
    idle: Duration,
    _slot: Slot,
    /// 呼び出しごとの統計
    pub queue_ms: u64,
    pub retries: u32,
    pub first_byte: Option<Instant>,
}

pub struct CallInfo {
    pub queue_ms: u64,
    pub retries: u32,
}

impl Upstream {
    pub fn new(cfg: Arc<Config>, metrics: Arc<Metrics>) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs_f64(cfg.upstream.connect_timeout_s))
            .pool_idle_timeout(Duration::from_secs(30))
            .tcp_nodelay(true)
            .build()
            .expect("reqwest client");
        Self {
            sem: Arc::new(Semaphore::new(cfg.upstream.max_concurrency.max(1))),
            cfg,
            http,
            waiting: AtomicUsize::new(0),
            inflight: Arc::new(AtomicUsize::new(0)),
            avg_wait_ms: AtomicU64::new(1000),
            metrics,
            shutting_down: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.cfg.upstream.url.trim_end_matches('/'), path)
    }

    pub fn queue_full(&self) -> bool {
        self.waiting.load(Ordering::Relaxed) >= self.cfg.queue.max_waiting
    }

    fn retry_after_s(&self) -> u64 {
        (self.avg_wait_ms.load(Ordering::Relaxed) / 1000).clamp(1, 60)
    }

    /// 待ち行列に入って枠を得る (§11.2)。
    async fn acquire(&self, key_sem: Option<Arc<Semaphore>>, deadline: Instant) -> Result<Slot, ApiError> {
        let mut v = self.acquire_n(key_sem, 1, deadline).await?;
        Ok(v.pop().unwrap())
    }

    /// n 枠をまとめて得る。n 本のストリーム (n>1、複数プロンプト) が 1 枠ずつ取り合うと、
    /// 互いに残りの枠を待って詰まるので、必ずまとめて取る (tokio の Semaphore は先着順)。
    pub async fn acquire_n(&self, key_sem: Option<Arc<Semaphore>>, n: usize, deadline: Instant) -> Result<Vec<Slot>, ApiError> {
        if self.shutting_down.load(Ordering::Relaxed) {
            return Err(ApiError::new(Kind::Unavailable, "shutting_down", "ds41gw is shutting down"));
        }
        let n = n.max(1);
        let max = self.cfg.upstream.max_concurrency.max(1);
        if n > max {
            return Err(ApiError::bad("too_many_choices", format!("this request needs {n} concurrent upstream streams; at most {max} are available"))
                .with_param("n"));
        }
        let split = |mut p: OwnedSemaphorePermit, mut kp: Option<OwnedSemaphorePermit>, inflight: &Arc<AtomicUsize>| -> Vec<Slot> {
            let mut out = Vec::with_capacity(n);
            for _ in 1..n {
                let one = p.split(1).expect("permit split");
                let kone = kp.as_mut().map(|k| k.split(1).expect("key permit split"));
                out.push(Slot { _permit: one, _key_permit: kone, inflight: inflight.clone() });
            }
            out.push(Slot { _permit: p, _key_permit: kp, inflight: inflight.clone() });
            inflight.fetch_add(n, Ordering::Relaxed);
            out
        };
        // 枠がすぐ空いていて、待っている人がいなければ待たない
        if key_sem.is_none() && self.waiting.load(Ordering::Relaxed) == 0 {
            if let Ok(p) = self.sem.clone().try_acquire_many_owned(n as u32) {
                self.metrics.observe("ds41gw_queue_wait_seconds", &[], 0.0);
                return Ok(split(p, None, &self.inflight));
            }
        }
        if self.queue_full() {
            self.metrics.inc("ds41gw_queue_rejects_total", &[("reason", "full")]);
            return Err(ApiError::new(Kind::Queue, "queue_full", "proxy queue is full").with_retry_after(5));
        }
        // 待っている間にクライアントが切ると、この future は await の途中で捨てられる。
        // 減算を後ろに書くと実行されずに waiting が増え続ける (やがて全要求が queue_full) ので、drop で減らす。
        struct Waiting<'a>(&'a AtomicUsize);
        impl Drop for Waiting<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let guard = Waiting(&self.waiting);
        let t0 = Instant::now();
        let res = async {
            let key_permit = match key_sem {
                Some(s) => Some(tokio::time::timeout_at(deadline.into(), s.acquire_many_owned(n as u32)).await.map_err(|_| ())?.map_err(|_| ())?),
                None => None,
            };
            let p = tokio::time::timeout_at(deadline.into(), self.sem.clone().acquire_many_owned(n as u32)).await.map_err(|_| ())?.map_err(|_| ())?;
            Ok::<_, ()>((p, key_permit))
        }
        .await;
        drop(guard);
        let waited = t0.elapsed();
        self.metrics.observe("ds41gw_queue_wait_seconds", &[], waited.as_secs_f64());
        let prev = self.avg_wait_ms.load(Ordering::Relaxed);
        self.avg_wait_ms.store((prev * 7 + waited.as_millis() as u64) / 8, Ordering::Relaxed);
        match res {
            Ok((p, kp)) => Ok(split(p, kp, &self.inflight)),
            Err(()) => {
                self.metrics.inc("ds41gw_queue_rejects_total", &[("reason", "timeout")]);
                Err(ApiError::new(Kind::Queue, "queue_timeout", "timed out waiting for an upstream slot")
                    .with_retry_after(self.retry_after_s()))
            }
        }
    }

    /// 上流に送って、応答ヘッダまで待つ。429 と接続失敗は再試行する (§11.3)。
    pub async fn call(
        &self,
        body: &Value,
        names: Arc<NameMap>,
        key_sem: Option<Arc<Semaphore>>,
    ) -> Result<UpStream, ApiError> {
        let t0 = Instant::now();
        let queue_deadline = t0 + Duration::from_secs_f64(self.cfg.queue.timeout_s);
        let slot = self.acquire(key_sem, queue_deadline).await?;
        let queue_ms = t0.elapsed().as_millis() as u64;
        self.call_with_slot(body, names, slot, queue_ms, queue_deadline).await
    }

    /// 取ってある枠で上流に送る。
    pub async fn call_with_slot(
        &self,
        body: &Value,
        names: Arc<NameMap>,
        slot: Slot,
        queue_ms: u64,
        queue_deadline: Instant,
    ) -> Result<UpStream, ApiError> {
        // 本文を複製せずに、stream:true にし stream_options を除いた形で直列化する (大きな本文で 1 回分の複製を省く)
        let payload = Bytes::from(serde_json::to_vec(&StreamBody(body)).unwrap_or_default());
        let url = self.url("/v1/chat/completions");
        let mut retries = 0u32;
        let mut backoff = 0.5f64;
        let conn_deadline = Instant::now() + Duration::from_secs_f64(self.cfg.upstream.connect_retry_total_s);
        let r429_deadline = (Instant::now() + Duration::from_secs_f64(self.cfg.upstream.retry_429_max_s)).min(queue_deadline);
        // 応答ヘッダの待ち時間にも上限を付ける (ds41rt が受け付けたまま固まったときに枠を持ち続けない)
        let header_timeout = Duration::from_secs_f64(self.cfg.upstream.idle_timeout_s.max(0.05));
        loop {
            let res = tokio::time::timeout(
                header_timeout,
                self.http
                    .post(&url)
                    .header("content-type", "application/json")
                    .header("accept", "text/event-stream")
                    .body(payload.clone())
                    .send(),
            )
            .await;
            let res = match res {
                Ok(r) => r,
                Err(_) => {
                    self.metrics.inc("ds41gw_upstream_header_timeouts_total", &[]);
                    return Err(ApiError::new(
                        Kind::Unavailable,
                        "upstream_timeout",
                        format!("upstream did not send response headers within {}s", header_timeout.as_secs_f64()),
                    ));
                }
            };
            match res {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    if status == 200 {
                        let idle = Duration::from_secs_f64(self.cfg.upstream.idle_timeout_s);
                        let dsml = if self.cfg.compat.strip_dsml { Some(DsmlFilter::new()) } else { None };
                        return Ok(UpStream {
                            resp: resp.bytes_stream().boxed(),
                            buf: Vec::new(),
                            pending: VecDeque::new(),
                            names,
                            dsml,
                            dsml_leaks: 0,
                            finished: false,
                            done: false,
                            err: None,
                            err_reported: false,
                            idle,
                            _slot: slot,
                            queue_ms,
                            retries,
                            first_byte: None,
                        });
                    }
                    let retry_after = resp
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<f64>().ok());
                    let text = resp.text().await.unwrap_or_default();
                    if status == 429 {
                        let wait = backoff.max(retry_after.unwrap_or(1.0));
                        if Instant::now() + Duration::from_secs_f64(wait) < r429_deadline {
                            retries += 1;
                            self.metrics.inc("ds41gw_upstream_retries_total", &[("reason", "429")]);
                            tokio::time::sleep(Duration::from_secs_f64(wait)).await;
                            backoff = (backoff * 2.0).min(4.0);
                            continue;
                        }
                        return Err(ApiError::new(Kind::Queue, "queue_timeout", format!("upstream: {}", upstream_message(&text)))
                            .with_retry_after(1));
                    }
                    return Err(map_upstream_status(status, &text));
                }
                Err(e) => {
                    // 接続の拒否・リセット (ds41rt の再起動中など)
                    if (e.is_connect() || e.is_request()) && Instant::now() + Duration::from_secs_f64(backoff) < conn_deadline {
                        retries += 1;
                        self.metrics.inc("ds41gw_upstream_retries_total", &[("reason", "connect")]);
                        tokio::time::sleep(Duration::from_secs_f64(backoff)).await;
                        backoff = (backoff * 2.0).min(4.0);
                        continue;
                    }
                    return Err(ApiError::new(Kind::Unavailable, "upstream_unavailable", format!("upstream: {}", truncate(&e.to_string(), 512))));
                }
            }
        }
    }
}

/// 上流に送る形: `stream` を true にし (元の位置のまま。無ければ末尾)、`stream_options` を除く。
struct StreamBody<'a>(&'a Value);

impl serde::Serialize for StreamBody<'_> {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let Some(o) = self.0.as_object() else { return self.0.serialize(ser) };
        let mut m = ser.serialize_map(None)?;
        let mut wrote_stream = false;
        for (k, v) in o {
            match k.as_str() {
                "stream_options" => {}
                "stream" => {
                    m.serialize_entry("stream", &true)?;
                    wrote_stream = true;
                }
                _ => m.serialize_entry(k, v)?,
            }
        }
        if !wrote_stream {
            m.serialize_entry("stream", &true)?;
        }
        m.end()
    }
}

/// ds41rt のエラー本文から message を取り出す (JSON 2 種類と text/plain、§4.6)。
pub fn upstream_message(text: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        if let Some(m) = v.pointer("/error/message").and_then(Value::as_str) {
            return truncate(m, 512);
        }
    }
    truncate(text.trim(), 512)
}

pub fn map_upstream_status(status: u16, text: &str) -> ApiError {
    let msg = format!("upstream: {}", upstream_message(text));
    let param = serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|v| v.pointer("/error/param").and_then(Value::as_str).map(str::to_string));
    match status {
        400 | 413 | 415 | 422 => {
            let mut e = ApiError::new(Kind::UpstreamRejected, "upstream_rejected", msg);
            e.param = param;
            e
        }
        503 => ApiError::new(Kind::Unavailable, "upstream_unavailable", msg),
        404 => ApiError::new(Kind::Upstream, "upstream_error", msg),
        _ => ApiError::new(Kind::Upstream, "upstream_error", msg),
    }
}

impl UpStream {
    /// 次の差分。None は正常終了。Err は途中失敗。キャンセル安全 (await は上流の読み取りだけ)。
    pub async fn next(&mut self) -> Option<Result<Ev, MidstreamError>> {
        loop {
            if let Some(ev) = self.pending.pop_front() {
                return Some(Ok(ev));
            }
            if self.done {
                if self.finished || self.err_reported {
                    return None;
                }
                self.err_reported = true;
                let msg = self.err.take().unwrap_or_else(|| "upstream stream terminated unexpectedly".into());
                return Some(Err(MidstreamError(msg)));
            }
            let item = match tokio::time::timeout(self.idle, self.resp.next()).await {
                Ok(x) => x,
                Err(_) => {
                    self.done = true;
                    self.err = Some(format!("upstream idle for {}s", self.idle.as_secs()));
                    continue;
                }
            };
            match item {
                Some(Ok(bytes)) => {
                    if self.first_byte.is_none() {
                        self.first_byte = Some(Instant::now());
                    }
                    self.buf.extend_from_slice(&bytes);
                    self.drain_lines();
                }
                Some(Err(e)) => {
                    self.done = true;
                    self.err = Some(format!("upstream read error: {e}"));
                }
                None => {
                    self.done = true;
                    if !self.buf.is_empty() {
                        let rest = std::mem::take(&mut self.buf);
                        self.handle_line(&rest);
                    }
                }
            }
        }
    }

    fn drain_lines(&mut self) {
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            self.handle_line(&line);
        }
    }

    fn handle_line(&mut self, line: &[u8]) {
        let line = String::from_utf8_lossy(line);
        let line = line.trim_end_matches(['\r', '\n']);
        let Some(data) = line.strip_prefix("data:") else { return };
        let data = data.trim_start();
        if data == "[DONE]" {
            self.done = true;
            return;
        }
        if self.finished {
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else { return };
        // ds41rt (real_full) はストリームの途中の失敗を `event: error` + `data: {"error":{...}}` + [DONE] で送る。
        // 読み飛ばすと「理由不明の途中切断」になるので、その message を途中失敗の理由にする
        if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
            let msg = e.get("message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| e.to_string());
            self.err = Some(format!("upstream error: {}", truncate(&msg, 512)));
            self.done = true;
            return;
        }
        self.handle_chunk(&v);
    }

    fn filter(&mut self, s: &str) -> String {
        match &mut self.dsml {
            Some(f) => {
                let out = f.push(s);
                self.dsml_leaks = f.leaks;
                out
            }
            None => s.to_string(),
        }
    }

    /// ds41rt のチャンク 1 つを差分に写す (§3.2)。
    pub fn handle_chunk(&mut self, v: &Value) {
        let ch = v.pointer("/choices/0");
        let delta = ch.and_then(|c| c.get("delta"));
        if let Some(d) = delta {
            if d.get("role").is_some() {
                self.pending.push_back(Ev::Start {
                    id: v.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
                    created: v.get("created").and_then(Value::as_u64).unwrap_or_else(crate::util::now_unix),
                    fingerprint: v.get("system_fingerprint").and_then(Value::as_str).map(str::to_string),
                });
            }
            if let Some(r) = d.get("reasoning_content").and_then(Value::as_str) {
                if !r.is_empty() {
                    self.pending.push_back(Ev::Reasoning(r.to_string()));
                }
            }
            if let Some(c) = d.get("content").and_then(Value::as_str) {
                if !c.is_empty() {
                    let out = self.filter(c);
                    if !out.is_empty() {
                        self.pending.push_back(Ev::Content(out));
                    }
                }
            }
            if let Some(tcs) = d.get("tool_calls").and_then(Value::as_array) {
                for tc in tcs {
                    let index = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                    let args = tc.pointer("/function/arguments").and_then(Value::as_str).unwrap_or_default();
                    if let Some(id) = tc.get("id").and_then(Value::as_str) {
                        let name = tc.pointer("/function/name").and_then(Value::as_str).unwrap_or_default();
                        self.pending.push_back(Ev::ToolStart { index, id: id.to_string(), name: self.names.unmap(name) });
                    }
                    if !args.is_empty() {
                        self.pending.push_back(Ev::ToolArgs { index, args: args.to_string() });
                    }
                }
            }
        }
        let fin = ch.and_then(|c| c.get("finish_reason")).and_then(Value::as_str);
        if let Some(reason) = fin {
            // 保留していた本文を出す
            if let Some(f) = &mut self.dsml {
                let rest = f.finish();
                self.dsml_leaks = f.leaks;
                if !rest.is_empty() {
                    self.pending.push_back(Ev::Content(rest));
                }
            }
            let usage = v.get("usage").filter(|u| !u.is_null()).cloned();
            self.pending.push_back(Ev::Finish { reason: reason.to_string(), usage, stop: None });
            self.finished = true;
            // 残りは [DONE] だけなので読まない
            self.done = true;
        }
    }
}

/// 試験用: チャンクの列から差分を作る。
pub fn parse_chunks_for_test(chunks: &[Value], names: Arc<NameMap>) -> Vec<Ev> {
    let sem = Arc::new(Semaphore::new(1));
    let permit = sem.try_acquire_owned().unwrap();
    let inflight = Arc::new(AtomicUsize::new(1));
    let mut u = UpStream {
        resp: futures::stream::empty().boxed(),
        buf: vec![],
        pending: VecDeque::new(),
        names,
        dsml: Some(DsmlFilter::new()),
        dsml_leaks: 0,
        finished: false,
        done: false,
        err: None,
        err_reported: false,
        idle: Duration::from_secs(1),
        _slot: Slot { _permit: permit, _key_permit: None, inflight },
        queue_ms: 0,
        retries: 0,
        first_byte: None,
    };
    for c in chunks {
        u.handle_chunk(c);
    }
    u.pending.into_iter().collect()
}

/// 上流の usage を補う: `completion_tokens_details.reasoning_tokens` を加える。
pub fn enrich_usage(usage: &Value, reasoning_tokens: usize) -> Value {
    let mut u = usage.clone();
    if let Some(o) = u.as_object_mut() {
        let d = o.entry("completion_tokens_details").or_insert_with(|| json!({}));
        if !d.is_object() {
            *d = json!({});
        }
        d.as_object_mut().unwrap().insert("reasoning_tokens".into(), json!(reasoning_tokens));
    }
    u
}

pub fn usage_num(usage: Option<&Value>, key: &str) -> usize {
    usage.and_then(|u| u.get(key)).and_then(Value::as_u64).unwrap_or(0) as usize
}
