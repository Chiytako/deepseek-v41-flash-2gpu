//! 出力の content に漏れた DSML (ツール呼び出しの markup) を除く (r7_proxy_spec.md §5.3.9)。
//! `<｜DSML｜` が現れたら、`</｜DSML｜ calls>` (V4 は `</｜DSML｜tool_calls>`) までを捨てる。
//! ストリームでは、開始の印の途中で切れているかもしれない末尾だけを保留する。

const BEGIN: &str = "<｜DSML｜";
const ENDS: [&str; 2] = ["</｜DSML｜ calls>", "</｜DSML｜tool_calls>"];

#[derive(Debug, Default)]
pub struct DsmlFilter {
    /// 保留中の文字列 (開始の印の接頭辞か、ブロック内の末尾)。
    hold: String,
    in_block: bool,
    /// 見つけて除いたブロックの数。
    pub leaks: usize,
}

/// `s` の末尾で、`pat` の真の接頭辞になっている最長の部分のバイト位置。
fn partial_suffix_start(s: &str, pat: &str) -> Option<usize> {
    let max = pat.len().saturating_sub(1).min(s.len());
    for len in (1..=max).rev() {
        let start = s.len() - len;
        if !s.is_char_boundary(start) {
            continue;
        }
        if pat.starts_with(&s[start..]) {
            return Some(start);
        }
    }
    None
}

impl DsmlFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// 差分を入れて、クライアントに送ってよい文字列を返す。
    pub fn push(&mut self, delta: &str) -> String {
        let mut buf = std::mem::take(&mut self.hold);
        buf.push_str(delta);
        let mut out = String::new();
        loop {
            if self.in_block {
                // 終わりの印を探す。
                let mut found: Option<(usize, usize)> = None;
                for e in ENDS {
                    if let Some(p) = buf.find(e) {
                        if found.map_or(true, |(fp, _)| p < fp) {
                            found = Some((p, e.len()));
                        }
                    }
                }
                match found {
                    Some((p, l)) => {
                        self.in_block = false;
                        buf = buf[p + l..].to_string();
                        continue;
                    }
                    None => {
                        // 終わりの印の途中かもしれない末尾だけ残す。
                        let keep = ENDS.iter().filter_map(|e| partial_suffix_start(&buf, e)).min();
                        self.hold = match keep {
                            Some(k) => buf[k..].to_string(),
                            None => String::new(),
                        };
                        return out;
                    }
                }
            }
            match buf.find(BEGIN) {
                Some(p) => {
                    out.push_str(&buf[..p]);
                    self.in_block = true;
                    self.leaks += 1;
                    buf = buf[p + BEGIN.len()..].to_string();
                }
                None => {
                    match partial_suffix_start(&buf, BEGIN) {
                        Some(k) => {
                            out.push_str(&buf[..k]);
                            self.hold = buf[k..].to_string();
                        }
                        None => out.push_str(&buf),
                    }
                    return out;
                }
            }
        }
    }

    /// 終わりに残りを出す。ブロックの中で終わった場合は捨てる。
    pub fn finish(&mut self) -> String {
        let h = std::mem::take(&mut self.hold);
        if self.in_block {
            return String::new();
        }
        h
    }
}

/// 一括版 (非ストリーム用)。
pub fn strip(s: &str) -> (String, usize) {
    let mut f = DsmlFilter::new();
    let mut out = f.push(s);
    out.push_str(&f.finish());
    (out, f.leaks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str]) -> (String, usize) {
        let mut f = DsmlFilter::new();
        let mut out = String::new();
        for c in chunks {
            out.push_str(&f.push(c));
        }
        out.push_str(&f.finish());
        (out, f.leaks)
    }

    #[test]
    fn passthrough() {
        assert_eq!(run(&["hello ", "world"]), ("hello world".into(), 0));
        assert_eq!(run(&["a <", "b"]), ("a <b".into(), 0));
        assert_eq!(run(&["x<｜", "y"]), ("x<｜y".into(), 0));
    }

    #[test]
    fn strips_block_split_everywhere() {
        let full = "I will call.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">x</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\nDone.";
        let expect = "I will call.\n\n\nDone.";
        // すべての文字境界で 2 分割する
        let idx: Vec<usize> = full.char_indices().map(|(i, _)| i).collect();
        for &i in &idx {
            let (a, b) = full.split_at(i);
            let (out, leaks) = run(&[a, b]);
            assert_eq!(out, expect, "split at {i}");
            assert_eq!(leaks, 1);
        }
        // 1 文字ずつ
        let chars: Vec<String> = full.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chars.iter().map(|s| s.as_str()).collect();
        assert_eq!(run(&refs).0, expect);
    }

    #[test]
    fn unterminated_block_is_dropped() {
        assert_eq!(run(&["ok <｜DSML｜ calls>\n<｜DSML｜ invoke"]).0, "ok ");
    }

    #[test]
    fn hold_is_bounded() {
        let mut f = DsmlFilter::new();
        let out = f.push("abcdef<｜DS");
        assert_eq!(out, "abcdef");
        assert!(f.hold.chars().count() <= 16);
        assert_eq!(f.push("x"), "<｜DSx");
    }
}
