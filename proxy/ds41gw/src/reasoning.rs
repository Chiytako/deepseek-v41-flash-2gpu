//! 思考の再注入ストアと自前の encrypted_content (r7_proxy_spec.md §10)。

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::util::{b64url, b64url_decode, now_unix, sha256_hex};

/// tool_call id か本文のハッシュをキーに、生成時の思考を覚えておく LRU。
pub struct ReinjectStore {
    inner: Mutex<Inner>,
    max_bytes: usize,
    ttl: Duration,
}

struct Inner {
    map: HashMap<String, (std::sync::Arc<str>, Instant)>,
    order: VecDeque<String>,
    bytes: usize,
}

impl ReinjectStore {
    pub fn new(max_mb: usize, ttl_h: f64) -> Self {
        Self {
            inner: Mutex::new(Inner { map: HashMap::new(), order: VecDeque::new(), bytes: 0 }),
            max_bytes: max_mb.saturating_mul(1 << 20),
            ttl: Duration::from_secs_f64((ttl_h * 3600.0).max(1.0)),
        }
    }

    pub fn content_key(content: &str) -> String {
        format!("h:{}", sha256_hex(content.trim().as_bytes()))
    }

    /// 1 応答の思考を、tool_call id 群 (あれば) と本文のハッシュで保存する。
    pub fn put(&self, reasoning: &str, tool_call_ids: &[String], content: &str) {
        if reasoning.is_empty() || self.max_bytes == 0 {
            return;
        }
        let text: std::sync::Arc<str> = std::sync::Arc::from(reasoning);
        let mut keys: Vec<String> = tool_call_ids.iter().map(|i| format!("t:{i}")).collect();
        if tool_call_ids.is_empty() && !content.trim().is_empty() {
            keys.push(Self::content_key(content));
        }
        let mut g = self.inner.lock().unwrap();
        let now = Instant::now();
        for k in keys {
            let add = k.len() + text.len() + 64;
            if let Some((old, _)) = g.map.insert(k.clone(), (text.clone(), now)) {
                g.bytes = g.bytes.saturating_sub(k.len() + old.len() + 64);
            } else {
                g.order.push_back(k);
            }
            g.bytes += add;
        }
        while g.bytes > self.max_bytes {
            let Some(k) = g.order.pop_front() else { break };
            if let Some((old, _)) = g.map.remove(&k) {
                g.bytes = g.bytes.saturating_sub(k.len() + old.len() + 64);
            }
        }
    }

    /// Responses の reasoning 項目の id (`rs_…`) で思考を覚える (§6.5 の 4)。
    pub fn put_item(&self, item_id: &str, reasoning: &str) {
        if item_id.is_empty() || reasoning.is_empty() || self.max_bytes == 0 {
            return;
        }
        let k = format!("r:{item_id}");
        let text: std::sync::Arc<str> = std::sync::Arc::from(reasoning);
        let mut g = self.inner.lock().unwrap();
        let add = k.len() + text.len() + 64;
        if let Some((old, _)) = g.map.insert(k.clone(), (text, Instant::now())) {
            g.bytes = g.bytes.saturating_sub(k.len() + old.len() + 64);
        } else {
            g.order.push_back(k);
        }
        g.bytes += add;
        while g.bytes > self.max_bytes {
            let Some(k) = g.order.pop_front() else { break };
            if let Some((old, _)) = g.map.remove(&k) {
                g.bytes = g.bytes.saturating_sub(k.len() + old.len() + 64);
            }
        }
    }

    pub fn get_item(&self, item_id: &str) -> Option<String> {
        self.get(&format!("r:{item_id}"))
    }

    pub fn get_tool(&self, tool_call_id: &str) -> Option<String> {
        self.get(&format!("t:{tool_call_id}"))
    }

    pub fn get_content(&self, content: &str) -> Option<String> {
        if content.trim().is_empty() {
            return None;
        }
        self.get(&Self::content_key(content))
    }

    fn get(&self, key: &str) -> Option<String> {
        let mut g = self.inner.lock().unwrap();
        let expired = match g.map.get(key) {
            Some((_, t)) => t.elapsed() > self.ttl,
            None => return None,
        };
        if expired {
            // 容量の計数からも引く (引かないと、期限切れが溜まるほど bytes が実際より大きくなり、
            // やがて put するたびに入れたものをすぐ追い出す = 再注入が効かなくなる)
            if let Some((old, _)) = g.map.remove(key) {
                g.bytes = g.bytes.saturating_sub(key.len() + old.len() + 64);
            }
            if let Some(pos) = g.order.iter().position(|k| k == key) {
                g.order.remove(pos);
            }
            return None;
        }
        g.map.get(key).map(|(t, _)| t.to_string())
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().map.len()
    }

    pub fn bytes(&self) -> usize {
        self.inner.lock().unwrap().bytes
    }
}

type HmacSha256 = Hmac<Sha256>;

/// HMAC 鍵。encrypted_content と Anthropic の thinking signature に使う。
pub struct Signer {
    key: Vec<u8>,
}

pub const ENC_PREFIX: &str = "ds41gw1.";
pub const SGLANG_PREFIX: &str = "sglang-reasoning-v1.";
pub const SIG_PREFIX: &str = "ds41gw-sig-v1.";

impl Signer {
    pub fn new(key: Vec<u8>) -> Self {
        Self { key }
    }

    /// 鍵ファイルを読む。無ければ 32 バイトを生成して 0600 で書く。
    pub fn load_or_create(path: &std::path::Path) -> std::io::Result<Self> {
        if let Ok(k) = std::fs::read(path) {
            if k.len() >= 16 {
                return Ok(Self::new(k));
            }
        }
        let mut key = vec![0u8; 32];
        getrandom::getrandom(&mut key).map_err(|e| std::io::Error::other(e.to_string()))?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(path)?;
        f.write_all(&key)?;
        Ok(Self::new(key))
    }

    fn mac16(&self, data: &[u8]) -> Vec<u8> {
        let mut m = HmacSha256::new_from_slice(&self.key).expect("hmac key");
        m.update(data);
        m.finalize().into_bytes()[..16].to_vec()
    }

    /// `ds41gw1.<payload>.<mac>`。payload = base64url(zstd(JSON))。
    pub fn encrypt(&self, text: &str) -> String {
        let json = serde_json::json!({"v": 1, "t": text, "ts": now_unix()}).to_string();
        let z = zstd::bulk::compress(json.as_bytes(), 3).unwrap_or_else(|_| json.into_bytes());
        let payload = b64url(&z);
        let mac = b64url(&self.mac16(payload.as_bytes()));
        format!("{ENC_PREFIX}{payload}.{mac}")
    }

    /// 復号。mac が合わない・形式が違う場合は Err (呼び出し側は空として扱う)。
    pub fn decrypt(&self, blob: &str) -> Result<String, &'static str> {
        if let Some(rest) = blob.strip_prefix(SGLANG_PREFIX) {
            return decode_sglang(rest).ok_or("bad_sglang_reasoning");
        }
        let rest = blob.strip_prefix(ENC_PREFIX).ok_or("unknown_encrypted_content")?;
        let (payload, mac) = rest.rsplit_once('.').ok_or("bad_encrypted_content")?;
        let want = self.mac16(payload.as_bytes());
        let got = b64url_decode(mac).ok_or("bad_encrypted_content")?;
        if !constant_eq(&want, &got) {
            return Err("bad_encrypted_content");
        }
        let z = b64url_decode(payload).ok_or("bad_encrypted_content")?;
        let raw = zstd::stream::decode_all(&z[..]).map_err(|_| "bad_encrypted_content")?;
        let v: serde_json::Value = serde_json::from_slice(&raw).map_err(|_| "bad_encrypted_content")?;
        v.get("t").and_then(|t| t.as_str()).map(str::to_string).ok_or("bad_encrypted_content")
    }

    /// Anthropic の thinking signature (非空)。
    pub fn signature(&self, thinking: &str) -> String {
        format!("{SIG_PREFIX}{}", b64url(&self.mac16(thinking.as_bytes())))
    }
}

fn constant_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut d = 0u8;
    for (x, y) in a.iter().zip(b) {
        d |= x ^ y;
    }
    d == 0
}

/// SGLang の `sglang-reasoning-v1.` + urlsafe_b64(zlib(text))。
pub fn decode_sglang(rest: &str) -> Option<String> {
    use base64::Engine;
    let raw = base64::engine::general_purpose::URL_SAFE
        .decode(rest.trim())
        .ok()
        .or_else(|| b64url_decode(rest.trim()))?;
    let mut d = flate2::read::ZlibDecoder::new(&raw[..]);
    let mut s = String::new();
    d.read_to_string(&mut s).ok()?;
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_tamper() {
        let s = Signer::new(vec![7u8; 32]);
        let blob = s.encrypt("思考の本文 thinking");
        assert!(blob.starts_with(ENC_PREFIX));
        assert_eq!(s.decrypt(&blob).unwrap(), "思考の本文 thinking");
        // 改ざん
        let mut t = blob.clone();
        let i = ENC_PREFIX.len() + 3;
        let c = if &t[i..i + 1] == "A" { "B" } else { "A" };
        t.replace_range(i..i + 1, c);
        assert!(s.decrypt(&t).is_err());
        // 別の鍵
        let s2 = Signer::new(vec![8u8; 32]);
        assert!(s2.decrypt(&blob).is_err());
        assert!(s.decrypt("garbage").is_err());
        let sig = s.signature("abc");
        assert!(sig.starts_with(SIG_PREFIX) && sig.len() > SIG_PREFIX.len());
    }

    #[test]
    fn sglang_format() {
        // python: "sglang-reasoning-v1." + base64.urlsafe_b64encode(zlib.compress("hello 世界".encode())).decode()
        let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all("hello 世界".as_bytes()).unwrap();
        let z = e.finish().unwrap();
        use base64::Engine;
        let blob = format!("{SGLANG_PREFIX}{}", base64::engine::general_purpose::URL_SAFE.encode(z));
        let s = Signer::new(vec![1; 32]);
        assert_eq!(s.decrypt(&blob).unwrap(), "hello 世界");
        // python で作った実物
        let py = "sglang-reasoning-v1.eJzLSM3JyVcozy_KSQEAGgsEXQ==";
        assert_eq!(s.decrypt(py).unwrap(), "hello world");
    }

    #[test]
    fn reinject_expiry_releases_bytes() {
        let r = ReinjectStore::new(1, 0.0); // ttl は最小 1 秒
        r.put("think", &["call_x".into()], "");
        assert!(r.bytes() > 0);
        std::thread::sleep(Duration::from_millis(1100));
        assert!(r.get_tool("call_x").is_none());
        assert_eq!(r.bytes(), 0);
        assert_eq!(r.inner.lock().unwrap().order.len(), 0);
        r.put("think2", &["call_x".into()], "");
        assert_eq!(r.get_tool("call_x").unwrap(), "think2");
    }

    #[test]
    fn reinject_lru() {
        let r = ReinjectStore::new(1, 24.0);
        r.put("think A", &["call_1".into(), "call_2".into()], "");
        r.put("think B", &[], " answer ");
        assert_eq!(r.get_tool("call_2").unwrap(), "think A");
        assert_eq!(r.get_content("answer").unwrap(), "think B");
        assert!(r.get_tool("nope").is_none());
        // 容量を超えたら古い順に消える
        let big = "x".repeat(400_000);
        for i in 0..5 {
            r.put(&big, &[format!("big{i}")], "");
        }
        assert!(r.get_tool("call_1").is_none());
        assert!(r.get_tool("big4").is_some());
    }
}
