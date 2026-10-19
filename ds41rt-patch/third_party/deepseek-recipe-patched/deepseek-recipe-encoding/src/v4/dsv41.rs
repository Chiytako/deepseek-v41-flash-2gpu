use deepseek_recipe_core::conversation::ReasoningEffort;

use super::{EncodingV4, SYSTEM_SP_TOKEN};
use crate::TokenizerEncoder;

fn reasoning_effort_template(effort: Option<ReasoningEffort>) -> String {
    // Named levels follow the official encoding; `Budget(n)` renders `n`.
    let effort_score = effort.map_or(ReasoningEffort::DEFAULT_SCORE, ReasoningEffort::score);
    format!(
        "Reasoning Effort: {effort_score} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    )
}

/// Prompt rendering and token encoding for DeepSeek V4.1.
pub struct DeepseekV41Encoding {
    tokenizer: Option<Box<dyn TokenizerEncoder>>,
}

impl DeepseekV41Encoding {
    /// Create an encoding without an attached tokenizer.
    ///
    /// Token encoding requires a tokenizer, attached with `with_tokenizer`.
    pub fn new() -> Self {
        Self { tokenizer: None }
    }

    /// Attach the tokenizer used by token encoding.
    pub fn with_tokenizer(mut self, tokenizer: impl TokenizerEncoder + 'static) -> Self {
        self.tokenizer = Some(Box::new(tokenizer));
        self
    }
}

impl Default for DeepseekV41Encoding {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodingV4 for DeepseekV41Encoding {
    fn tokenizer(&self) -> Option<&dyn TokenizerEncoder> {
        self.tokenizer.as_deref()
    }

    fn supports_mid_conversation_system(&self) -> bool {
        true
    }

    fn system_token(&self) -> &'static str {
        SYSTEM_SP_TOKEN
    }

    fn tool_calls_block_name(&self) -> &'static str {
        " calls"
    }

    fn tool_call_tag_name(&self) -> &'static str {
        " invoke"
    }

    fn tool_parameter_tag_name(&self) -> &'static str {
        " parameter"
    }

    fn render_reasoning_effort(
        &self,
        index: usize,
        thinking_mode: bool,
        effort: Option<ReasoningEffort>,
    ) -> String {
        if index == 0 && thinking_mode {
            return reasoning_effort_template(effort);
        }
        String::new()
    }
}

#[cfg(test)]
mod reasoning_effort_tests {
    use super::reasoning_effort_template;
    use deepseek_recipe_core::conversation::ReasoningEffort;

    fn score_line(score: u8) -> String {
        format!(
            "Reasoning Effort: {score} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        )
    }

    #[test]
    fn named_levels_keep_the_official_scores_and_budgets_render_verbatim() {
        assert_eq!(reasoning_effort_template(None), score_line(75));
        assert_eq!(reasoning_effort_template(Some(ReasoningEffort::Low)), score_line(50));
        assert_eq!(reasoning_effort_template(Some(ReasoningEffort::High)), score_line(75));
        assert_eq!(reasoning_effort_template(Some(ReasoningEffort::Xhigh)), score_line(75));
        assert_eq!(reasoning_effort_template(Some(ReasoningEffort::Max)), score_line(100));
        for score in [1, 33, 60, 62, 88, 99, 100] {
            assert_eq!(
                reasoning_effort_template(Some(ReasoningEffort::Budget(score))),
                score_line(score)
            );
        }
    }
}
