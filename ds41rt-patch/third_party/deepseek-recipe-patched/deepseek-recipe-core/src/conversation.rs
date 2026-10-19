use serde::{Deserialize, Serialize};

use crate::messages::InputMessage;
use crate::tools::{ToolChoice, ToolDefinition};

/// A conversation and its prompt configuration.
///
/// Sampling parameters and transport options belong to the caller. Thinking is
/// enabled by default.
#[derive(Debug, Clone)]
pub struct Conversation {
    /// Messages in conversation order.
    pub messages: Vec<InputMessage>,
    /// Enable thinking mode for the next assistant turn.
    pub thinking_mode: bool,
    /// Available client tools, in declaration order.
    pub tools: Vec<ToolDefinition>,
    /// Tool selection after protocol-specific normalization.
    pub tool_choice: ToolChoice,
    /// Reasoning effort used when thinking mode is enabled.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Requested answer format used during prompt rendering.
    pub response_format: ResponseFormat,
}

impl Default for Conversation {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            thinking_mode: true,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            reasoning_effort: None,
            response_format: ResponseFormat::Text,
        }
    }
}

/// Output format described in the inference prompt.
#[derive(Debug, Clone, Default)]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
}

/// Protocol-independent reasoning effort.
///
/// The named levels follow the official DeepSeek encoding (`low` = 50,
/// `high`/`xhigh` = 75, `max` = 100 on the V4.1 1-100 scale). The protocol
/// adapters of this patched copy map the request names `medium` and `xhigh`
/// to `Budget(MEDIUM_SCORE)` (62) and `Budget(XHIGH_SCORE)` (88) instead of
/// the official 75 (a local extension; DeepSeek defines only low/high/max).
/// `Budget` carries
/// an explicit V4.1 score in [`ReasoningEffort::MIN_BUDGET`,
/// `ReasoningEffort::MAX_BUDGET`]; construct it with
/// [`ReasoningEffort::budget`] so the range is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    High,
    Xhigh,
    Max,
    /// Explicit V4.1 reasoning effort score (1-100).
    Budget(u8),
}

impl ReasoningEffort {
    /// Smallest explicit V4.1 reasoning effort score.
    pub const MIN_BUDGET: u8 = 1;
    /// Largest explicit V4.1 reasoning effort score.
    pub const MAX_BUDGET: u8 = 100;
    /// V4.1 score used when thinking is enabled without an explicit effort.
    pub const DEFAULT_SCORE: u8 = 75;
    /// Score for the request name `medium` (local extension: midway between
    /// low 50 and high 75, rounded down; the official mapping is 75).
    pub const MEDIUM_SCORE: u8 = 62;
    /// Score for the request name `xhigh` (local extension: midway between
    /// high 75 and max 100, rounded up; the official mapping is 75).
    pub const XHIGH_SCORE: u8 = 88;

    /// Return an explicit score effort, or `None` outside 1-100.
    pub fn budget(score: u64) -> Option<Self> {
        (u64::from(Self::MIN_BUDGET)..=u64::from(Self::MAX_BUDGET))
            .contains(&score)
            .then(|| Self::Budget(score as u8))
    }

    /// The V4.1 1-100 score this effort renders as.
    pub fn score(self) -> u8 {
        match self {
            Self::Low => 50,
            Self::High | Self::Xhigh => Self::DEFAULT_SCORE,
            Self::Max => 100,
            Self::Budget(score) => score.clamp(Self::MIN_BUDGET, Self::MAX_BUDGET),
        }
    }

    /// The nearest named level, for encodings without a numeric scale (V4).
    ///
    /// Scores map to low (50), high (75) or max (100): 1-61 low, 62-88 high,
    /// 89-100 max. The high band includes [`Self::MEDIUM_SCORE`] (62) and
    /// [`Self::XHIGH_SCORE`] (88) so `medium` and `xhigh` fall back to high,
    /// as in the official mapping, on encodings without a numeric scale.
    pub fn level(self) -> Self {
        match self {
            Self::Budget(score) => match score {
                0..=61 => Self::Low,
                62..=88 => Self::High,
                _ => Self::Max,
            },
            named => named,
        }
    }
}

/// Error returned when a protocol value is not a supported reasoning effort.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningEffortParseError {
    /// Human-readable description of the rejected value.
    pub got: String,
}

impl std::fmt::Display for ReasoningEffortParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "reasoning effort must be an integer from {} to {} (a JSON number or a numeric string such as \"60\") or one of none, off, disabled, minimal, low, medium, high, xhigh, max; got {}",
            ReasoningEffort::MIN_BUDGET,
            ReasoningEffort::MAX_BUDGET,
            self.got
        )
    }
}

impl std::error::Error for ReasoningEffortParseError {}

/// A reasoning effort value as supplied by a protocol request: either a level
/// name (lowercased, trimmed) or an explicit 1-100 score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestedReasoningEffort {
    Name(String),
    Score(u8),
}

/// Quote a rejected string for an error message, bounded to 40 characters.
fn shown(value: &str) -> String {
    const MAX_CHARS: usize = 40;
    let mut chars = value.chars();
    let head: String = chars.by_ref().take(MAX_CHARS).collect();
    if chars.next().is_some() {
        format!("{head:?}...")
    } else {
        format!("{head:?}")
    }
}

impl RequestedReasoningEffort {
    fn parse_str(value: &str) -> Result<Self, ReasoningEffortParseError> {
        let trimmed = value.trim();
        if !trimmed.is_empty() && trimmed.bytes().all(|b| b.is_ascii_digit() || b == b'+') {
            return match trimmed.parse::<u64>().ok().and_then(ReasoningEffort::budget) {
                Some(ReasoningEffort::Budget(score)) => Ok(Self::Score(score)),
                _ => Err(ReasoningEffortParseError {
                    got: shown(value),
                }),
            };
        }
        if trimmed.is_empty()
            || !trimmed
                .bytes()
                .all(|b| b.is_ascii_alphabetic() || b == b'_' || b == b'-')
        {
            return Err(ReasoningEffortParseError {
                got: shown(value),
            });
        }
        Ok(Self::Name(trimmed.to_ascii_lowercase()))
    }

    /// Deserialize a string, integer or numeric string effort value.
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = RequestedReasoningEffort;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a reasoning effort level name or an integer from 1 to 100")
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                RequestedReasoningEffort::parse_str(v).map_err(E::custom)
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                match ReasoningEffort::budget(v) {
                    Some(ReasoningEffort::Budget(score)) => {
                        Ok(RequestedReasoningEffort::Score(score))
                    }
                    _ => Err(E::custom(ReasoningEffortParseError { got: v.to_string() })),
                }
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                match u64::try_from(v) {
                    Ok(v) => self.visit_u64(v),
                    Err(_) => Err(E::custom(ReasoningEffortParseError { got: v.to_string() })),
                }
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                Err(E::custom(ReasoningEffortParseError {
                    // `{v:?}` keeps the fraction visible: 60.0 shows as "60.0", not "60".
                    got: format!("{v:?} (not an integer)"),
                }))
            }

            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Err(E::custom(ReasoningEffortParseError { got: v.to_string() }))
            }
        }
        deserializer.deserialize_any(Visitor)
    }

    /// Build an "unsupported level name" error for a protocol adapter.
    pub fn unsupported(&self) -> ReasoningEffortParseError {
        ReasoningEffortParseError {
            got: match self {
                Self::Name(name) => shown(name),
                Self::Score(score) => score.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod reasoning_effort_tests {
    use super::{ReasoningEffort, RequestedReasoningEffort};
    use serde_json::json;

    fn parse(value: serde_json::Value) -> Result<RequestedReasoningEffort, String> {
        RequestedReasoningEffort::deserialize(value).map_err(|e| e.to_string())
    }

    #[test]
    fn accepts_integers_numeric_strings_and_names() {
        assert_eq!(parse(json!(1)), Ok(RequestedReasoningEffort::Score(1)));
        assert_eq!(parse(json!(60)), Ok(RequestedReasoningEffort::Score(60)));
        assert_eq!(parse(json!(100)), Ok(RequestedReasoningEffort::Score(100)));
        assert_eq!(parse(json!("60")), Ok(RequestedReasoningEffort::Score(60)));
        assert_eq!(parse(json!(" 7 ")), Ok(RequestedReasoningEffort::Score(7)));
        assert_eq!(
            parse(json!("High")),
            Ok(RequestedReasoningEffort::Name("high".into()))
        );
    }

    #[test]
    fn rejects_out_of_range_fractional_and_malformed_values() {
        for value in [
            json!(0),
            json!(101),
            json!(-1),
            json!(60.5),
            json!(60.0),
            json!("0"),
            json!("101"),
            json!("-5"),
            json!("60.5"),
            json!(""),
            json!("6 0"),
            json!(true),
            json!(null),
            json!([60]),
        ] {
            let error = parse(value.clone()).unwrap_err();
            assert!(
                error.contains("integer from 1 to 100") || error.contains("invalid type"),
                "{value}: {error}"
            );
        }
        assert!(parse(json!(101)).unwrap_err().contains("got 101"));
        assert!(parse(json!(60.0)).unwrap_err().contains("got 60.0 (not an integer)"));
        assert!(parse(json!(3.5)).unwrap_err().contains("got 3.5 (not an integer)"));
    }

    #[test]
    fn scores_and_levels_follow_the_official_mapping() {
        assert_eq!(ReasoningEffort::Low.score(), 50);
        assert_eq!(ReasoningEffort::High.score(), 75);
        assert_eq!(ReasoningEffort::Xhigh.score(), 75);
        assert_eq!(ReasoningEffort::Max.score(), 100);
        assert_eq!(ReasoningEffort::Budget(42).score(), 42);
        assert_eq!(ReasoningEffort::budget(0), None);
        assert_eq!(ReasoningEffort::budget(101), None);
        assert_eq!(ReasoningEffort::budget(1), Some(ReasoningEffort::Budget(1)));
        assert_eq!(ReasoningEffort::Budget(61).level(), ReasoningEffort::Low);
        assert_eq!(ReasoningEffort::Budget(62).level(), ReasoningEffort::High);
        assert_eq!(ReasoningEffort::Budget(88).level(), ReasoningEffort::High);
        assert_eq!(ReasoningEffort::Budget(89).level(), ReasoningEffort::Max);
        assert_eq!(ReasoningEffort::MEDIUM_SCORE, 62);
        assert_eq!(ReasoningEffort::XHIGH_SCORE, 88);
        assert_eq!(ReasoningEffort::Xhigh.level(), ReasoningEffort::Xhigh);
    }
}
