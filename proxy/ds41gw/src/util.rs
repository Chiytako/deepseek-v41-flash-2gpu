//! 小さな共通関数 (時刻、ID、JSON の補助)。

use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn now_unix_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

/// 1970-01-01 からの日数 → (年, 月, 日)。H. Hinnant の civil_from_days。
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// UTC の RFC3339 (ミリ秒付き)。
pub fn rfc3339_ms(ms: u128) -> String {
    let secs = (ms / 1000) as i64;
    let milli = (ms % 1000) as u32;
    let (y, mo, d) = civil_from_days(secs.div_euclid(86_400));
    let sod = secs.rem_euclid(86_400);
    format!(
        "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}.{milli:03}Z",
        sod / 3600,
        (sod / 60) % 60,
        sod % 60
    )
}

pub fn rfc3339_now() -> String {
    rfc3339_ms(now_unix_ms())
}

/// UTC の YYYYMMDD。
pub fn yyyymmdd(secs: u64) -> String {
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}{m:02}{d:02}")
}

pub fn hex32() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

pub fn sha256_hex(data: &[u8]) -> String {
    let h = Sha256::digest(data);
    h.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn b64url(data: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
}

pub fn b64url_decode(s: &str) -> Option<Vec<u8>> {
    let t = s.trim_end_matches('=');
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(t).ok()
}

/// 文字境界を守って最大 n バイトに切り詰め、切った場合は印を付ける。
pub fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut end = n;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… (truncated)", &s[..end])
}

/// 整数として読む (`16.0` も受け付ける)。負や小数は None。
pub fn as_u64_lenient(v: &Value) -> Option<u64> {
    if let Some(u) = v.as_u64() {
        return Some(u);
    }
    let f = v.as_f64()?;
    if f >= 0.0 && f.fract() == 0.0 && f < 1.8e19 {
        return Some(f as u64);
    }
    None
}

pub fn is_truthy(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) != 0.0,
        Some(Value::String(s)) => matches!(s.as_str(), "true" | "1" | "yes"),
        _ => false,
    }
}

/// content (文字列か text パーツの配列) からテキストだけを連結する。
pub fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            let mut out = Vec::new();
            for p in parts {
                if let Some(t) = p.get("text").and_then(Value::as_str) {
                    out.push(t.to_string());
                } else if let Some(s) = p.as_str() {
                    out.push(s.to_string());
                }
            }
            out.join("\n\n")
        }
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// IPv4/IPv6 の CIDR 照合 (依存を増やさない最小実装)。
pub fn ip_in_cidr(ip: std::net::IpAddr, cidr: &str) -> bool {
    use std::net::IpAddr;
    let (net, bits) = match cidr.split_once('/') {
        Some((n, b)) => (n, b.parse::<u32>().unwrap_or(128)),
        None => (cidr, 128),
    };
    let Ok(net): Result<IpAddr, _> = net.parse() else { return false };
    let ip = match (ip, net) {
        (IpAddr::V6(v6), IpAddr::V4(_)) => match v6.to_ipv4_mapped() {
            Some(v4) => IpAddr::V4(v4),
            None => return false,
        },
        _ => ip,
    };
    match (ip, net) {
        (IpAddr::V4(a), IpAddr::V4(b)) => {
            let bits = bits.min(32);
            if bits == 0 {
                return true;
            }
            let mask = u32::MAX << (32 - bits);
            (u32::from(a) & mask) == (u32::from(b) & mask)
        }
        (IpAddr::V6(a), IpAddr::V6(b)) => {
            let bits = bits.min(128);
            if bits == 0 {
                return true;
            }
            let mask = u128::MAX << (128 - bits);
            (u128::from(a) & mask) == (u128::from(b) & mask)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn time_fmt() {
        assert_eq!(rfc3339_ms(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(rfc3339_ms(1_792_339_106_123), "2026-10-18T15:58:26.123Z");
        assert_eq!(yyyymmdd(1_792_339_106), "20261018");
    }
    #[test]
    fn cidr() {
        assert!(ip_in_cidr("127.0.0.1".parse().unwrap(), "127.0.0.1/32"));
        assert!(ip_in_cidr("172.17.3.4".parse().unwrap(), "172.17.0.0/16"));
        assert!(!ip_in_cidr("100.1.2.3".parse().unwrap(), "172.17.0.0/16"));
        assert!(ip_in_cidr("::1".parse().unwrap(), "::1/128"));
        assert!(ip_in_cidr("::ffff:127.0.0.1".parse().unwrap(), "127.0.0.1/32"));
    }
    #[test]
    fn lenient() {
        assert_eq!(as_u64_lenient(&serde_json::json!(16.0)), Some(16));
        assert_eq!(as_u64_lenient(&serde_json::json!(16.5)), None);
        assert_eq!(as_u64_lenient(&serde_json::json!(-1)), None);
    }
}
