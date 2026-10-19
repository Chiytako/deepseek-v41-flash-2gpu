//! 思考 (thinking) と effort の決定 (r7_proxy_spec.md §4.3)。
//! 要求の指定を優先し、指定が無ければ thinking on・effort high (75) とする。
//!
//! effort の内部表現は「思考なし」か「整数 1〜100」(V4.1 の `Reasoning Effort: N`) の 2 つだけにする。
//! - 段階名は `thinking.effort_map` で数値に写す (既定: none/off/disabled → 思考なし、minimal/low 50、medium 62、
//!   high 75、xhigh 88、max/ultra 100)。大小文字は区別しない。low/high/max は公式の値、medium 62 と xhigh 88 は独自の対応
//!   (公式は 3 段階だけを定め、deepseek-recipe は medium/xhigh を 75 に写す)。
//! - 数値 (JSON number と数値文字列): 1〜100 の整数はそのまま。0 は思考なし。0 < x < 1 は ×100 (旧来の互換)。
//!   それ以外の小数は四捨五入。範囲外 (負、100.5 以上) と非数は 400 `invalid_reasoning_effort`。
//! - `thinking.budget_tokens` は境界で low/high/max の値 (50/75/100) にする。
//!
//! ds41rt には `thinking:{"type":"enabled"|"disabled"}` と `reasoning_effort` (既定は JSON の整数) だけを送る。
//! `thinking.numeric_upstream=false` のときは、段階名 (≤61 low、62〜88 high、他 max) に戻して送る。

use serde_json::{Map, Value};

use crate::config::Thinking as Cfg;
use crate::error::ApiError;
use crate::util::is_truthy;

/// effort の最小値と最大値 (V4.1 の 1〜100)。
pub const MIN_EFFORT: u8 = 1;
pub const MAX_EFFORT: u8 = 100;
/// default_effort が解けないときの値 (公式の high)。
pub const FALLBACK_EFFORT: u8 = 75;

/// 1 つの effort 指定の解釈。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    /// 思考なし (none / off / disabled / 0)。
    Off,
    /// 1〜100。
    Score(u8),
}

fn invalid(got: impl std::fmt::Display, param: &str) -> ApiError {
    ApiError::bad(
        "invalid_reasoning_effort",
        format!(
            "invalid {param}: {got}. It must be an integer from {MIN_EFFORT} to {MAX_EFFORT} (a JSON number or a numeric string such as \"60\"; 0 disables thinking) \
             or one of none, off, disabled, minimal, low, medium, high, xhigh, max"
        ),
    )
    .with_param(param)
}

/// 文字列を誤りの文言に出す形 (40 文字まで)。
fn shown(s: &str) -> String {
    let head: String = s.chars().take(40).collect();
    if s.chars().count() > 40 {
        format!("{head:?}...")
    } else {
        format!("{head:?}")
    }
}

/// effort の値 (段階名・数値・数値文字列) を解く。null と空文字列は「指定なし」(None)。
/// `param` は誤りの文言と `param` に出す名前 (例 `reasoning.effort`)。
pub fn map_effort(v: &Value, cfg: &Cfg, param: &str) -> Result<Option<Effort>, ApiError> {
    match v {
        Value::Null => Ok(None),
        Value::Number(n) => match n.as_f64() {
            Some(x) => numeric_effort(x).map(Some).ok_or_else(|| invalid(n, param)),
            None => Err(invalid(n, param)),
        },
        Value::String(s) => {
            let k = s.trim().to_ascii_lowercase();
            if k.is_empty() {
                return Ok(None);
            }
            if let Some(&m) = cfg.effort_map.get(&k) {
                return Ok(Some(if m == 0 { Effort::Off } else { Effort::Score(m.min(MAX_EFFORT)) }));
            }
            // 数値文字列 ("60"、"0.6")。inf / nan / 1e999 などは弾く
            let numeric = !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'+' | b'-' | b'e'));
            match k.parse::<f64>() {
                Ok(x) if numeric => numeric_effort(x).map(Some).ok_or_else(|| invalid(shown(s), param)),
                _ => Err(invalid(shown(s), param)),
            }
        }
        other => Err(invalid(other, param)),
    }
}

/// 数値の effort。0 → 思考なし、0 < x < 1 → x×100 (旧来の互換)、他は四捨五入して 1〜100。範囲外と非有限は None。
pub fn numeric_effort(x: f64) -> Option<Effort> {
    if !x.is_finite() || x < 0.0 {
        return None;
    }
    if x == 0.0 {
        return Some(Effort::Off);
    }
    let x = if x < 1.0 { x * 100.0 } else { x };
    let r = x.round().max(f64::from(MIN_EFFORT));
    if r > f64::from(MAX_EFFORT) {
        return None;
    }
    Some(Effort::Score(r as u8))
}

/// budget_tokens → effort (§4.3.3)。low 50、high 75、max 100。
pub fn budget_effort(b: u64, cfg: &Cfg) -> u8 {
    let name = if b < cfg.budget_low_below {
        "low"
    } else if b < cfg.budget_high_below {
        "high"
    } else {
        "max"
    };
    level_score(name)
}

/// `thinking.budget_tokens` を読む。0 以上の整数 (JSON number、`2048.0`、数値文字列 `"2048"`) を受け、null と空文字列は指定なし。
/// 負・小数・非数・真偽値・配列などは、黙って既定の effort にせず 400 `invalid_thinking` にする。
pub fn budget_tokens(v: Option<&Value>) -> Result<Option<u64>, ApiError> {
    let bad = |got: &dyn std::fmt::Display| {
        ApiError::bad("invalid_thinking", format!("invalid thinking.budget_tokens: {got}. It must be a non-negative integer"))
            .with_param("thinking.budget_tokens")
    };
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => crate::util::as_u64_lenient(&Value::Number(n.clone())).map(Some).ok_or_else(|| bad(n)),
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                return Ok(None);
            }
            if !t.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad(&shown(s)));
            }
            t.parse::<u64>().map(Some).map_err(|_| bad(&shown(s)))
        }
        Some(other) => Err(bad(other)),
    }
}

/// 公式の段階の値 (low 50、high 75、max 100)。
pub fn level_score(name: &str) -> u8 {
    match name {
        "low" => 50,
        "max" => 100,
        _ => 75,
    }
}

/// 整数の effort を段階名にする (≤61 low、62〜88 high、他 max)。
/// ほぼ最寄りの段階だが、medium (62) と xhigh (88) は low / max より 1 だけ遠い high に含める。
/// 段階名しか受けない旧 ds41rt では medium / xhigh が公式どおり high になるようにするため。
pub fn nearest_level(score: u8) -> &'static str {
    match score {
        0..=61 => "low",
        62..=88 => "high",
        _ => "max",
    }
}

/// 設定の検査: `default_effort` は 1〜100 に解けること (none / 0 / 範囲外 / 未知の名前は誤り)。
pub fn validate_config(cfg: &Cfg) -> Result<(), String> {
    match map_effort(&Value::String(cfg.default_effort.clone()), cfg, "thinking.default_effort") {
        Ok(Some(Effort::Score(_))) => Ok(()),
        _ => Err(format!(
            "thinking.default_effort = {:?}: must be a level name that maps to 1-100 (e.g. \"high\") or \"1\"-\"100\"; \
             use thinking.default = \"disabled\" to turn thinking off by default",
            cfg.default_effort
        )),
    }
}

/// `thinking.default_effort` の値 (1〜100)。
pub fn default_score(cfg: &Cfg) -> u8 {
    match map_effort(&Value::String(cfg.default_effort.clone()), cfg, "thinking.default_effort") {
        Ok(Some(Effort::Score(n))) => n,
        _ => FALLBACK_EFFORT,
    }
}

/// ds41rt に送る effort の値と、ds41rt がプロンプトに描く値 (count_tokens 用)。
pub fn upstream_effort(score: u8, cfg: &Cfg) -> (Value, u8) {
    if cfg.numeric_upstream {
        (Value::from(score), score)
    } else {
        let name = nearest_level(score);
        (Value::String(name.into()), level_score(name))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub enabled: bool,
    /// enabled のときの effort (1〜100)。
    pub effort: u8,
    /// 要求が on/off を明示したか (auto_off の判定に使う)。
    pub explicit: bool,
    /// `chat_template_kwargs.clear_thinking:true`。
    pub clear: bool,
}

/// Chat 本文から思考の指定を読み、指定のフィールドを取り除く (§4.3.2)。
pub fn resolve(body: &mut Map<String, Value>, cfg: &Cfg, normalized: &mut Vec<String>) -> Result<Decision, ApiError> {
    let thinking = body.remove("thinking");
    let effort_top = body.remove("reasoning_effort");
    let ctk = body.remove("chat_template_kwargs");
    let enable_top = body.remove("enable_thinking");
    if ctk.is_some() {
        normalized.push("chat_template_kwargs".into());
    }

    let mut enabled: Option<bool> = None;
    let mut budget: Option<u64> = None;
    // 1, 2: thinking (object / bool)
    match &thinking {
        Some(Value::Object(o)) => {
            if let Some(t) = o.get("type").and_then(Value::as_str) {
                match t.to_ascii_lowercase().as_str() {
                    "enabled" | "adaptive" | "auto" => enabled = Some(true),
                    "disabled" | "none" | "off" => enabled = Some(false),
                    other => {
                        return Err(ApiError::bad("invalid_thinking", format!("invalid thinking.type: {other:?}")).with_param("thinking.type"))
                    }
                }
            }
            budget = budget_tokens(o.get("budget_tokens"))?;
        }
        Some(Value::Bool(b)) => {
            enabled = Some(*b);
            normalized.push("thinking_bool".into());
        }
        Some(Value::Null) | None => {}
        Some(other) => {
            return Err(ApiError::bad("invalid_thinking", format!("invalid thinking: {other}")).with_param("thinking"))
        }
    }

    let effort_top = match &effort_top {
        Some(v) => map_effort(v, cfg, "reasoning_effort")?,
        None => None,
    };
    let ctk_obj = ctk.as_ref().and_then(Value::as_object);
    let effort_ctk = match ctk_obj.and_then(|o| o.get("reasoning_effort")) {
        Some(v) => map_effort(v, cfg, "chat_template_kwargs.reasoning_effort")?,
        None => None,
    };

    // 3: reasoning_effort none / 0
    if enabled.is_none() && effort_top == Some(Effort::Off) {
        enabled = Some(false);
    }
    // 4: chat_template_kwargs.enable_thinking / thinking / enable_thinking
    if enabled.is_none() {
        let cands = [
            ctk_obj.and_then(|o| o.get("enable_thinking")),
            ctk_obj.and_then(|o| o.get("thinking")),
            enable_top.as_ref(),
        ];
        for c in cands.into_iter().flatten() {
            if c.is_boolean() || c.is_number() || c.is_string() {
                enabled = Some(is_truthy(Some(c)));
                break;
            }
        }
    }
    // chat_template_kwargs.reasoning_effort の none / 0 は、トップレベルの reasoning_effort が無いときだけ効く
    // (effort はトップレベルが優先なので、{"reasoning_effort":60, "chat_template_kwargs":{"reasoning_effort":0}} は 60)
    if enabled.is_none() && effort_top.is_none() && effort_ctk == Some(Effort::Off) {
        enabled = Some(false);
    }
    let explicit = enabled.is_some() || effort_top.is_some() || effort_ctk.is_some();
    let enabled = enabled.unwrap_or(cfg.default != "disabled");

    let score = |e: Option<Effort>| match e {
        Some(Effort::Score(n)) => Some(n),
        _ => None,
    };
    let effort = score(effort_top)
        .or(score(effort_ctk))
        .or(budget.map(|b| budget_effort(b, cfg)))
        .unwrap_or_else(|| default_score(cfg));
    let clear = is_truthy(ctk_obj.and_then(|o| o.get("clear_thinking")));
    Ok(Decision { enabled, effort, explicit, clear })
}

/// 決定を ds41rt の本文に書く。
pub fn apply(body: &mut Map<String, Value>, d: &Decision, cfg: &Cfg) {
    if d.enabled {
        body.insert("thinking".into(), serde_json::json!({"type": "enabled"}));
        body.insert("reasoning_effort".into(), upstream_effort(d.effort, cfg).0);
    } else {
        body.insert("thinking".into(), serde_json::json!({"type": "disabled"}));
        body.remove("reasoning_effort");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn r(v: Value) -> Decision {
        let mut m = v.as_object().unwrap().clone();
        resolve(&mut m, &Cfg::default(), &mut vec![]).unwrap()
    }
    fn err(v: Value) -> ApiError {
        let mut m = v.as_object().unwrap().clone();
        resolve(&mut m, &Cfg::default(), &mut vec![]).unwrap_err()
    }

    #[test]
    fn defaults_and_named_levels_follow_the_official_scores() {
        let d = r(json!({}));
        assert!(d.enabled && d.effort == 75 && !d.explicit);
        for (name, want) in [
            ("minimal", 50),
            ("low", 50),
            ("LOW", 50),
            ("medium", 62),
            ("Medium", 62),
            ("high", 75),
            ("HIGH", 75),
            (" xhigh ", 88),
            ("XHIGH", 88),
            ("max", 100),
            ("Max", 100),
            ("ultra", 100),
        ] {
            let d = r(json!({"reasoning_effort": name}));
            assert!(d.enabled && d.explicit, "{name}");
            assert_eq!(d.effort, want, "{name}");
        }
        for off in ["none", "off", "disabled", "NONE", "Off"] {
            assert!(!r(json!({"reasoning_effort": off})).enabled, "{off}");
        }
        assert_eq!(r(json!({"reasoning_effort": ""})).effort, 75);
        assert!(!r(json!({"reasoning_effort": ""})).explicit);
    }

    #[test]
    fn numbers_and_numeric_strings() {
        for (v, want) in [
            (json!(1), 1),
            (json!(60), 60),
            (json!(100), 100),
            (json!("60"), 60),
            (json!(" 7 "), 7),
            (json!("100"), 100),
            (json!(0.9), 90),
            (json!(0.5), 50),
            (json!("0.6"), 60),
            (json!(0.001), 1),
            (json!(1.0), 1),
            (json!(59.5), 60),
            (json!(60.4), 60),
            (json!(100.4), 100),
            (json!(3), 3),
            (json!(75), 75),
            (json!("1e2"), 100),
        ] {
            let d = r(json!({"reasoning_effort": v.clone()}));
            assert!(d.enabled, "{v}");
            assert_eq!(d.effort, want, "{v}");
        }
        assert!(!r(json!({"reasoning_effort": 0})).enabled);
        assert!(!r(json!({"reasoning_effort": 0.0})).enabled);
        assert!(!r(json!({"reasoning_effort": "0"})).enabled);
        assert!(r(json!({"reasoning_effort": 0})).explicit);
    }

    #[test]
    fn out_of_range_and_malformed_values_are_400() {
        for v in [
            json!(101),
            json!(100.5),
            json!(-1),
            json!(-0.5),
            json!(1000000),
            json!("101"),
            json!("-3"),
            json!("abc"),
            json!("ultra-mega"),
            json!("inf"),
            json!("nan"),
            json!("6 0"),
            json!(true),
            json!([60]),
            json!({"x": 1}),
        ] {
            let e = err(json!({"reasoning_effort": v.clone()}));
            assert_eq!(e.code, "invalid_reasoning_effort", "{v}");
            assert_eq!(e.param.as_deref(), Some("reasoning_effort"), "{v}");
            assert!(e.message.contains("integer from 1 to 100"), "{v}: {}", e.message);
        }
        let e = err(json!({"chat_template_kwargs": {"reasoning_effort": 101}}));
        assert_eq!(e.param.as_deref(), Some("chat_template_kwargs.reasoning_effort"));
        assert!(e.message.contains("101"), "{}", e.message);
    }

    #[test]
    fn thinking_switches_budget_and_priority() {
        assert!(!r(json!({"thinking": false})).enabled);
        assert!(r(json!({"thinking": true})).enabled);
        assert!(!r(json!({"chat_template_kwargs": {"enable_thinking": false}})).enabled);
        assert!(!r(json!({"enable_thinking": false})).enabled);
        assert!(r(json!({"thinking": {"type": "adaptive"}})).enabled);
        assert_eq!(r(json!({"thinking": {"type": "enabled", "budget_tokens": 2048}})).effort, 50);
        assert_eq!(r(json!({"thinking": {"type": "enabled", "budget_tokens": 16000}})).effort, 75);
        assert_eq!(r(json!({"thinking": {"type": "enabled", "budget_tokens": 64000}})).effort, 100);
        // budget_tokens の数値文字列と整数値の小数も受ける。null と空文字列は指定なし
        assert_eq!(r(json!({"thinking": {"type": "enabled", "budget_tokens": "40000"}})).effort, 100);
        assert_eq!(r(json!({"thinking": {"type": "enabled", "budget_tokens": " 2048 "}})).effort, 50);
        assert_eq!(r(json!({"thinking": {"type": "enabled", "budget_tokens": 16000.0}})).effort, 75);
        assert_eq!(r(json!({"thinking": {"type": "enabled", "budget_tokens": null}})).effort, 75);
        assert_eq!(r(json!({"thinking": {"type": "enabled", "budget_tokens": ""}})).effort, 75);
        // 不正な budget_tokens は 400 (黙って既定の 75 にしない)
        for v in [json!(-1), json!(2048.5), json!("abc"), json!("-5"), json!("1e4"), json!(true), json!([2048]), json!({"n": 1})] {
            let e = err(json!({"thinking": {"type": "enabled", "budget_tokens": v.clone()}}));
            assert_eq!(e.code, "invalid_thinking", "{v}");
            assert_eq!(e.param.as_deref(), Some("thinking.budget_tokens"), "{v}");
        }
        assert_eq!(r(json!({"chat_template_kwargs": {"reasoning_effort": "low"}})).effort, 50);
        assert_eq!(r(json!({"chat_template_kwargs": {"reasoning_effort": 33}})).effort, 33);
        assert_eq!(r(json!({"chat_template_kwargs": {"reasoning_effort": "44"}})).effort, 44);
        assert!(!r(json!({"chat_template_kwargs": {"reasoning_effort": 0}})).enabled);
        // reasoning_effort が chat_template_kwargs と budget_tokens より優先
        assert_eq!(r(json!({"reasoning_effort": 60, "chat_template_kwargs": {"reasoning_effort": 20}, "thinking": {"type": "enabled", "budget_tokens": 64000}})).effort, 60);
        // thinking の明示が優先
        assert!(!r(json!({"thinking": {"type": "disabled"}, "reasoning_effort": "max"})).enabled);
        assert!(!r(json!({"thinking": {"type": "disabled"}, "reasoning_effort": 90})).enabled);
        // thinking on と effort 0: 思考は on のまま、effort は次の候補か既定
        let d = r(json!({"thinking": {"type": "enabled", "budget_tokens": 2048}, "reasoning_effort": 0}));
        assert!(d.enabled && d.effort == 50);
        assert!(r(json!({"chat_template_kwargs": {"clear_thinking": true}})).clear);
        // chat_template_kwargs の 0 / none より、トップレベルの effort が優先 (on のまま、effort はトップレベル)
        let d = r(json!({"reasoning_effort": 60, "chat_template_kwargs": {"reasoning_effort": 0}}));
        assert!(d.enabled && d.effort == 60);
        let d = r(json!({"reasoning_effort": "high", "chat_template_kwargs": {"reasoning_effort": "none"}}));
        assert!(d.enabled && d.effort == 75);
        // トップレベルが none なら off
        assert!(!r(json!({"reasoning_effort": "none", "chat_template_kwargs": {"reasoning_effort": 60}})).enabled);
        // トップレベルが無ければ chat_template_kwargs の 0 で off
        assert!(!r(json!({"chat_template_kwargs": {"reasoning_effort": "none"}})).enabled);
        // 空白だけは指定なし
        assert!(!r(json!({"reasoning_effort": " \t"})).explicit);
    }

    #[test]
    fn default_effort_is_validated() {
        let mut cfg = Cfg::default();
        assert!(validate_config(&cfg).is_ok());
        for ok in ["high", "XHIGH", "60", "1", "100", "minimal"] {
            cfg.default_effort = ok.into();
            assert!(validate_config(&cfg).is_ok(), "{ok}");
        }
        for bad in ["none", "off", "0", "101", "bogus", "", "-5"] {
            cfg.default_effort = bad.into();
            assert!(validate_config(&cfg).is_err(), "{bad}");
        }
        // 設定ファイルの読み込みで弾く
        let dir = std::env::temp_dir().join(format!("ds41gw-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("bad.toml");
        std::fs::write(&f, "[thinking]\ndefault_effort = \"none\"\n").unwrap();
        let e = crate::config::Config::load(&f).unwrap_err().to_string();
        assert!(e.contains("default_effort"), "{e}");
        std::fs::write(&f, "[thinking]\ndefault_effort = \"max\"\neffort_map = { xhigh = \"max\" }\n").unwrap();
        let c = crate::config::Config::load(&f).unwrap();
        assert_eq!(default_score(&c.thinking), 100);
        assert_eq!(c.thinking.effort_map["xhigh"], 100);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_overrides_and_upstream_forms() {
        let mut cfg = Cfg::default();
        assert_eq!(cfg.effort_map["xhigh"], 88);
        assert_eq!(cfg.effort_map["medium"], 62);
        cfg.effort_map.insert("xhigh".into(), 90);
        cfg.default_effort = "60".into();
        let mut m = json!({"reasoning_effort": "XHIGH"}).as_object().unwrap().clone();
        assert_eq!(resolve(&mut m, &cfg, &mut vec![]).unwrap().effort, 90);
        let mut m = json!({}).as_object().unwrap().clone();
        assert_eq!(resolve(&mut m, &cfg, &mut vec![]).unwrap().effort, 60);
        cfg.default_effort = "bogus".into();
        assert_eq!(default_score(&cfg), FALLBACK_EFFORT);

        // ds41rt に送る形
        let d = Decision { enabled: true, effort: 60, explicit: true, clear: false };
        let mut body = Map::new();
        apply(&mut body, &d, &Cfg::default());
        assert_eq!(Value::Object(body.clone()), json!({"thinking": {"type": "enabled"}, "reasoning_effort": 60}));
        let mut old = Cfg::default();
        old.numeric_upstream = false;
        for (score, name, rendered) in [(1, "low", 50), (50, "low", 50), (61, "low", 50), (62, "high", 75), (75, "high", 75), (88, "high", 75), (89, "max", 100), (100, "max", 100)] {
            let mut body = Map::new();
            apply(&mut body, &Decision { effort: score, ..d.clone() }, &old);
            assert_eq!(body["reasoning_effort"], name, "{score}");
            assert_eq!(upstream_effort(score, &old), (json!(name), rendered));
        }
        let mut body = json!({"reasoning_effort": 5}).as_object().unwrap().clone();
        apply(&mut body, &Decision { enabled: false, ..d }, &Cfg::default());
        assert_eq!(Value::Object(body), json!({"thinking": {"type": "disabled"}}));
    }

    #[test]
    fn effort_map_in_toml_accepts_numbers_and_legacy_names() {
        let c: crate::config::Config = toml::from_str("[thinking]\neffort_map = { xhigh = 90, legacy = \"max\", Custom = 0 }\n").unwrap();
        let m = &c.thinking.effort_map;
        assert_eq!(m["xhigh"], 90);
        assert_eq!(m["legacy"], 100);
        assert_eq!(m["custom"], 0);
        assert_eq!(m["low"], 50, "keys not in the file keep the defaults");
        assert_eq!(m["medium"], 62, "keys not in the file keep the defaults");
        assert!(c.thinking.numeric_upstream);
        // 旧形式 (値が段階名) の表もそのまま読める
        let c: crate::config::Config = toml::from_str(
            "[thinking]\neffort_map = { none = \"none\", minimal = \"low\", low = \"low\", medium = \"high\", high = \"high\", xhigh = \"max\", max = \"max\", ultra = \"max\" }\nnumeric_upstream = false\n",
        )
        .unwrap();
        assert_eq!(c.thinking.effort_map["xhigh"], 100);
        assert_eq!(c.thinking.effort_map["none"], 0);
        assert!(!c.thinking.numeric_upstream);
        assert!(toml::from_str::<crate::config::Config>("[thinking]\neffort_map = { x = 101 }\n").is_err());
        assert!(toml::from_str::<crate::config::Config>("[thinking]\neffort_map = { x = \"bogus\" }\n").is_err());
    }
}
