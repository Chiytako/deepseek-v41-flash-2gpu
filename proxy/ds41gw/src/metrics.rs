//! Prometheus 形式の計測 (r7_proxy_spec.md §13.4)。依存を増やさない最小実装。

use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::Mutex;

const BUCKETS: [f64; 16] = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0];

#[derive(Default)]
struct Hist {
    counts: [u64; BUCKETS.len()],
    sum: f64,
    count: u64,
}

#[derive(Default)]
pub struct Metrics {
    counters: Mutex<BTreeMap<(String, String), u64>>,
    hists: Mutex<BTreeMap<(String, String), Hist>>,
    gauges: Mutex<BTreeMap<(String, String), f64>>,
}

fn labels(l: &[(&str, &str)]) -> String {
    if l.is_empty() {
        return String::new();
    }
    let inner: Vec<String> = l.iter().map(|(k, v)| format!("{k}=\"{}\"", v.replace('"', "'"))).collect();
    format!("{{{}}}", inner.join(","))
}

impl Metrics {
    pub fn inc(&self, name: &str, l: &[(&str, &str)]) {
        self.add(name, l, 1);
    }
    pub fn add(&self, name: &str, l: &[(&str, &str)], n: u64) {
        *self.counters.lock().unwrap().entry((name.to_string(), labels(l))).or_default() += n;
    }
    pub fn observe(&self, name: &str, l: &[(&str, &str)], v: f64) {
        let mut g = self.hists.lock().unwrap();
        let h = g.entry((name.to_string(), labels(l))).or_default();
        for (i, b) in BUCKETS.iter().enumerate() {
            if v <= *b {
                h.counts[i] += 1;
            }
        }
        h.sum += v;
        h.count += 1;
    }
    pub fn set_gauge(&self, name: &str, l: &[(&str, &str)], v: f64) {
        self.gauges.lock().unwrap().insert((name.to_string(), labels(l)), v);
    }
    pub fn counter(&self, name: &str, l: &[(&str, &str)]) -> u64 {
        self.counters.lock().unwrap().get(&(name.to_string(), labels(l))).copied().unwrap_or(0)
    }

    pub fn render(&self, extra_gauges: &[(&str, f64)]) -> String {
        let mut out = String::new();
        let mut last = String::new();
        for ((n, l), v) in self.counters.lock().unwrap().iter() {
            if *n != last {
                let _ = writeln!(out, "# TYPE {n} counter");
                last = n.clone();
            }
            let _ = writeln!(out, "{n}{l} {v}");
        }
        for (n, v) in extra_gauges {
            let _ = writeln!(out, "# TYPE {n} gauge\n{n} {v}");
        }
        last.clear();
        for ((n, l), v) in self.gauges.lock().unwrap().iter() {
            if *n != last {
                let _ = writeln!(out, "# TYPE {n} gauge");
                last = n.clone();
            }
            let _ = writeln!(out, "{n}{l} {v}");
        }
        last.clear();
        for ((n, l), h) in self.hists.lock().unwrap().iter() {
            if *n != last {
                let _ = writeln!(out, "# TYPE {n} histogram");
                last = n.clone();
            }
            let base = l.trim_start_matches('{').trim_end_matches('}');
            let sep = if base.is_empty() { "" } else { "," };
            for (i, b) in BUCKETS.iter().enumerate() {
                let _ = writeln!(out, "{n}_bucket{{{base}{sep}le=\"{b}\"}} {}", h.counts[i]);
            }
            let _ = writeln!(out, "{n}_bucket{{{base}{sep}le=\"+Inf\"}} {}", h.count);
            let _ = writeln!(out, "{n}_sum{l} {}", h.sum);
            let _ = writeln!(out, "{n}_count{l} {}", h.count);
        }
        out
    }
}
