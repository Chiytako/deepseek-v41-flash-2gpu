//! Render the V4.1 prompt for one chat request: `render_effort '<json effort>'`.
//! Prints the prompt head, or the deserialization/conversion error.
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_encoding::{PromptEncoding, v4::dsv41::DeepseekV41Encoding};

fn main() {
    for arg in std::env::args().skip(1) {
        let effort: serde_json::Value = serde_json::from_str(&arg).expect("JSON effort value");
        let body = serde_json::json!({"model": "m", "reasoning_effort": effort,
            "messages": [{"role": "user", "content": "2+2?"}]});
        let rendered = serde_json::from_value::<ChatCompletionRequest>(body)
            .map_err(|e| e.to_string())
            .and_then(|r| {
                r.convert(ConversionOptions::default().with_default_thinking_mode(true))
                    .map_err(|e| format!("{e:?}"))
            })
            .map(|c| DeepseekV41Encoding::new().render_conversation(&c.conversation).prompt);
        match rendered {
            Ok(prompt) => println!("{arg:>8} => {prompt:?}"),
            Err(error) => println!("{arg:>8} => 400: {error}"),
        }
    }
}
