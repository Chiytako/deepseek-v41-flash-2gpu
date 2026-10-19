//! 要求ログ (r7_proxy_spec.md §13.1〜13.3)。
//! `logs/req/YYYYMMDD.jsonl` (UTC) に 1 行 1 JSON。前日分は zstd で圧縮する。
//! 受信の行は `"event":"request.received.openai"` と `obj` (ds41rt に送った Chat 本文) を持ち、
//! GLM 時代の再生器 `replay_openai.py` (非公開) や `tests/replay_capped.py` でそのまま再生できる。

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::config::Log as Cfg;
use crate::util::{now_unix, sha256_hex, yyyymmdd};

pub struct ReqLog {
    tx: Option<mpsc::Sender<Msg>>,
    pub cfg: Cfg,
}

/// 書き込みスレッドへの通知。受信の行は本文の整形と直列化をスレッド側で行う。
enum Msg {
    Line(String),
    Received { line: Value, obj: Option<Value>, raw: Option<bytes::Bytes> },
}

impl ReqLog {
    pub fn disabled() -> Self {
        Self { tx: None, cfg: Cfg { enabled: false, ..Cfg::default() } }
    }

    pub fn start(cfg: &Cfg) -> Self {
        if !cfg.enabled {
            return Self { tx: None, cfg: cfg.clone() };
        }
        let dir = cfg.dir.join("req");
        if let Err(e) = create_private_dir(&dir) {
            eprintln!("ds41gw: 要求ログの置き場 {} を作れない: {e}", dir.display());
        }
        let (tx, rx) = mpsc::channel::<Msg>();
        let c = cfg.clone();
        std::thread::Builder::new()
            .name("ds41gw-reqlog".into())
            .spawn(move || writer(dir, c, rx))
            .expect("spawn reqlog");
        Self { tx: Some(tx), cfg: cfg.clone() }
    }

    pub fn enabled(&self) -> bool {
        self.tx.is_some()
    }

    pub fn write(&self, v: Value) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Msg::Line(v.to_string()));
        }
    }

    /// 本文を残す設定か (`log.bodies = "full"`)。
    pub fn keeps_bodies(&self) -> bool {
        self.cfg.bodies != "meta"
    }

    /// 受信の行。`obj` (上流に送る本文) と `raw` (受け取った本文) は書き込みスレッドで伏せて直列化する。
    pub fn write_received(&self, line: Value, obj: Option<Value>, raw: Option<bytes::Bytes>) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Msg::Received { line, obj, raw });
        }
    }

    /// 本文を残すか、画像を伏せるかを設定どおりに整える。
    pub fn prepare_body(&self, v: &Value) -> Option<Value> {
        if self.cfg.bodies == "meta" {
            return None;
        }
        if self.cfg.images == "full" {
            return Some(v.clone());
        }
        // 先に丸ごと複製すると、数十 MB の data URL を 1 回余分に写すので、伏せながら複製する
        Some(scrubbed(v))
    }
}

pub fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
        if let Some(p) = dir.parent() {
            let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o700));
        }
    }
    Ok(())
}

fn open_private(path: &Path) -> std::io::Result<std::fs::File> {
    let mut o = std::fs::OpenOptions::new();
    o.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(path)
}

fn image_digest(body: &str, mime: &str) -> Value {
    json!({"sha256": sha256_hex(body.as_bytes()), "bytes": body.len() * 3 / 4, "mime": mime})
}

/// 画像を `{"sha256","bytes","mime"}` に置き換えた複製を作る。
/// data: URL の文字列 (OpenAI / Responses) と、Anthropic の `{"type":"base64","media_type","data"}` の data を伏せる。
pub fn scrubbed(v: &Value) -> Value {
    match v {
        Value::String(s) => {
            if s.len() > 128 && s.starts_with("data:") {
                if let Some((head, body)) = s.split_once(',') {
                    let mime = head.trim_start_matches("data:").split(';').next().unwrap_or("");
                    return image_digest(body, mime);
                }
            }
            v.clone()
        }
        Value::Array(a) => Value::Array(a.iter().map(scrubbed).collect()),
        Value::Object(o) => {
            let anthropic_b64 = o.get("type").and_then(Value::as_str) == Some("base64");
            let mut out = serde_json::Map::with_capacity(o.len());
            for (k, x) in o {
                let nv = match (k.as_str(), x) {
                    ("data", Value::String(d)) if anthropic_b64 && d.len() > 128 => {
                        image_digest(d, o.get("media_type").and_then(Value::as_str).unwrap_or(""))
                    }
                    _ => scrubbed(x),
                };
                out.insert(k.clone(), nv);
            }
            Value::Object(out)
        }
        _ => v.clone(),
    }
}

/// その場で伏せる版 (互換のため残す)。
pub fn scrub_images(v: &mut Value) {
    *v = scrubbed(v);
}

/// 受信の行を仕上げる (書き込みスレッドで)。
fn render(msg: Msg, cfg: &Cfg) -> String {
    match msg {
        Msg::Line(s) => s,
        Msg::Received { mut line, obj, raw } => {
            let prep = |v: Value| -> Value {
                if cfg.images == "full" {
                    v
                } else {
                    scrubbed(&v)
                }
            };
            if let Some(o) = line.as_object_mut() {
                if let Some(v) = obj {
                    o.insert("obj".into(), prep(v));
                }
                if let Some(v) = raw.and_then(|b| serde_json::from_slice::<Value>(&b).ok()) {
                    o.insert("orig".into(), prep(v));
                }
            }
            line.to_string()
        }
    }
}

fn writer(dir: PathBuf, cfg: Cfg, rx: mpsc::Receiver<Msg>) {
    let mut day = String::new();
    let mut file: Option<std::io::BufWriter<std::fs::File>> = None;
    let mut last_maint = Instant::now() - Duration::from_secs(7200);
    loop {
        let msg = rx.recv_timeout(Duration::from_secs(1));
        let today = yyyymmdd(now_unix());
        if today != day {
            if let Some(mut f) = file.take() {
                let _ = f.flush();
            }
            day = today.clone();
            match open_private(&dir.join(format!("{day}.jsonl"))) {
                Ok(f) => file = Some(std::io::BufWriter::new(f)),
                Err(e) => eprintln!("ds41gw: 要求ログを開けない: {e}"),
            }
            last_maint = Instant::now() - Duration::from_secs(7200);
        }
        match msg {
            Ok(m) => {
                let line = render(m, &cfg);
                if let Some(f) = &mut file {
                    let _ = f.write_all(line.as_bytes());
                    let _ = f.write_all(b"\n");
                }
                // 溜まっている分をまとめて書く
                while let Ok(m) = rx.try_recv() {
                    let line = render(m, &cfg);
                    if let Some(f) = &mut file {
                        let _ = f.write_all(line.as_bytes());
                        let _ = f.write_all(b"\n");
                    }
                }
                if let Some(f) = &mut file {
                    let _ = f.flush();
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                if let Some(mut f) = file.take() {
                    let _ = f.flush();
                }
                return;
            }
        }
        if last_maint.elapsed() > Duration::from_secs(3600) {
            last_maint = Instant::now();
            let d = dir.clone();
            let c = cfg.clone();
            let t = day.clone();
            std::thread::spawn(move || maintain(&d, &c, &t));
        }
    }
}

/// 前日以前の .jsonl を zstd で圧縮し、保存期間と容量の上限で古い日から消す。
fn maintain(dir: &Path, cfg: &Cfg, today: &str) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<(String, PathBuf, u64)> = Vec::new();
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(date) = name.get(..8).filter(|d| d.chars().all(|c| c.is_ascii_digit())) else { continue };
        let date = date.to_string();
        let path = e.path();
        if name.ends_with(".jsonl") && date.as_str() < today {
            let zpath = path.with_extension("jsonl.zst");
            if compress_file(&path, &zpath).is_ok() {
                let _ = std::fs::remove_file(&path);
                let size = std::fs::metadata(&zpath).map(|m| m.len()).unwrap_or(0);
                files.push((date, zpath, size));
                continue;
            }
        }
        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
        files.push((date, path, size));
    }
    files.sort();
    let cutoff = yyyymmdd(now_unix().saturating_sub(cfg.retention_days * 86_400));
    let mut total: u64 = files.iter().map(|f| f.2).sum();
    let max = (cfg.max_total_gb * (1u64 << 30) as f64) as u64;
    for (date, path, size) in &files {
        if date.as_str() >= today {
            break;
        }
        if date.as_str() < cutoff.as_str() || total > max {
            if std::fs::remove_file(path).is_ok() {
                total = total.saturating_sub(*size);
            }
        }
    }
}

fn compress_file(src: &Path, dst: &Path) -> std::io::Result<()> {
    let input = std::fs::File::open(src)?;
    let out = {
        let mut o = std::fs::OpenOptions::new();
        o.create(true).write(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        o.open(dst)?
    };
    zstd::stream::copy_encode(input, out, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maintain_compresses_and_expires() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let today = yyyymmdd(now_unix());
        let yesterday = yyyymmdd(now_unix() - 86_400);
        let old = yyyymmdd(now_unix() - 20 * 86_400);
        std::fs::write(d.join(format!("{today}.jsonl")), "{\"a\":1}\n").unwrap();
        std::fs::write(d.join(format!("{yesterday}.jsonl")), "{\"b\":2}\n".repeat(100)).unwrap();
        std::fs::write(d.join(format!("{old}.jsonl.zst")), "x").unwrap();
        let cfg = Cfg { retention_days: 14, ..Cfg::default() };
        maintain(d, &cfg, &today);
        assert!(d.join(format!("{today}.jsonl")).exists());
        assert!(!d.join(format!("{yesterday}.jsonl")).exists());
        let z = std::fs::read(d.join(format!("{yesterday}.jsonl.zst"))).unwrap();
        assert_eq!(zstd::stream::decode_all(&z[..]).unwrap(), "{\"b\":2}\n".repeat(100).into_bytes());
        assert!(!d.join(format!("{old}.jsonl.zst")).exists());
    }

    #[test]
    fn scrub() {
        let mut v = serde_json::json!({"u": format!("data:image/png;base64,{}", "A".repeat(200)), "short": "data:x"});
        scrub_images(&mut v);
        assert_eq!(v["u"]["mime"], "image/png");
        assert_eq!(v["short"], "data:x");
        // Anthropic の base64 画像
        let mut v = serde_json::json!({"messages": [{"role": "user", "content": [{"type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "B".repeat(500)}}]}]});
        scrub_images(&mut v);
        let src = &v["messages"][0]["content"][0]["source"];
        assert_eq!(src["data"]["mime"], "image/jpeg");
        assert!(src["data"]["sha256"].is_string());
        assert_eq!(src["media_type"], "image/jpeg");
    }
}
