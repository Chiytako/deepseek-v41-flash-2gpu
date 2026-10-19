//! ツール名の写像 (r7_proxy_spec.md §5.3.8)。
//! ds41rt は `^[A-Za-z0-9_-]{1,128}$` 以外の名前を 400 にする。合わない名前 (例 `a.b`、
//! Responses の namespace を平坦化した `ns::name`) を要求ごとに写し、出力で元に戻す。

use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone)]
pub struct NameMap {
    to_up: HashMap<String, String>,
    to_orig: HashMap<String, String>,
    used: HashSet<String>,
}

pub fn is_valid(name: &str) -> bool {
    !name.is_empty() && name.len() <= 128 && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn sanitize(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    if s.is_empty() {
        s.push('_');
    }
    if s.len() > 128 {
        s.truncate(128);
    }
    s
}

impl NameMap {
    /// 先に、正しい名前をすべて登録しておく (写した名前が衝突しないように)。
    pub fn reserve(&mut self, name: &str) {
        if is_valid(name) {
            self.used.insert(name.to_string());
        }
    }

    /// 送る名前を返す。合わない名前は写して記録する。
    pub fn map(&mut self, name: &str) -> String {
        if is_valid(name) {
            self.used.insert(name.to_string());
            return name.to_string();
        }
        if let Some(m) = self.to_up.get(name) {
            return m.clone();
        }
        let base = sanitize(name);
        let mut cand = base.clone();
        let mut i = 2;
        while self.used.contains(&cand) {
            let suffix = format!("_{i}");
            let mut b = base.clone();
            if b.len() + suffix.len() > 128 {
                b.truncate(128 - suffix.len());
            }
            cand = format!("{b}{suffix}");
            i += 1;
        }
        self.used.insert(cand.clone());
        self.to_up.insert(name.to_string(), cand.clone());
        self.to_orig.insert(cand.clone(), name.to_string());
        cand
    }

    /// 出力の名前を元に戻す。
    pub fn unmap(&self, name: &str) -> String {
        self.to_orig.get(name).cloned().unwrap_or_else(|| name.to_string())
    }

    pub fn is_empty(&self) -> bool {
        self.to_up.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mapping() {
        let mut m = NameMap::default();
        m.reserve("a_b");
        assert_eq!(m.map("ok-name"), "ok-name");
        assert_eq!(m.map("a.b"), "a_b_2");
        assert_eq!(m.map("ns::name"), "ns__name");
        assert_eq!(m.map("a.b"), "a_b_2");
        assert_eq!(m.unmap("a_b_2"), "a.b");
        assert_eq!(m.unmap("ns__name"), "ns::name");
        assert_eq!(m.unmap("other"), "other");
        let long = "x.".repeat(100);
        let mapped = m.map(&long);
        assert!(is_valid(&mapped));
        assert_eq!(m.unmap(&mapped), long);
    }
}
