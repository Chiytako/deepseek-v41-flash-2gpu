//! recipe の `Conversation` を Chat の messages / tools に戻す (r7_proxy_spec.md §6.1 の 3)。
//! ds41rt は Chat を同じ recipe で `Conversation` に戻すので、描画されるプロンプトが一致するように書く
//! (`tests/lossless.rs` の T-U3 で検査する)。

use deepseek_recipe_core::conversation::{Conversation, ResponseFormat};
use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::multimodal::{ImageSource, IMAGE_SPECIAL_TOKEN};
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{json, Value};

fn image_part(src: &ImageSource) -> Value {
    let (url, detail) = match src {
        ImageSource::DataUrl { data_url, detail } => (data_url.clone(), *detail),
        ImageSource::Url { url, detail } => (url.clone(), *detail),
        ImageSource::Bytes { data, detail } => {
            use base64::Engine;
            (format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(data)), *detail)
        }
    };
    let detail = serde_json::to_value(detail).unwrap_or(json!("high"));
    json!({"type": "image_url", "image_url": {"url": url, "detail": detail}})
}

/// 画像プレースホルダを含む文字列を Chat のパーツに戻す。
/// recipe はブロックを "\n\n" で連結しているので、画像の前後の区切りを 1 つずつ取り除く。
pub fn content_with_images(content: &str, images: &[ImageSource]) -> Value {
    if images.is_empty() {
        return Value::String(content.to_string());
    }
    let segs: Vec<&str> = content.split(IMAGE_SPECIAL_TOKEN).collect();
    let nimg = segs.len() - 1;
    let mut parts = Vec::new();
    for (k, seg) in segs.iter().enumerate() {
        let before_img = k < nimg;
        let after_img = k > 0;
        let text: Option<String> = match (after_img, before_img) {
            (false, false) => Some(seg.to_string()),
            (false, true) => {
                if seg.is_empty() {
                    None
                } else {
                    Some(seg.strip_suffix("\n\n").unwrap_or(seg).to_string())
                }
            }
            (true, true) => {
                if *seg == "\n\n" {
                    None
                } else {
                    let s = seg.strip_prefix("\n\n").unwrap_or(seg);
                    Some(s.strip_suffix("\n\n").unwrap_or(s).to_string())
                }
            }
            (true, false) => {
                if seg.is_empty() {
                    None
                } else {
                    Some(seg.strip_prefix("\n\n").unwrap_or(seg).to_string())
                }
            }
        };
        if let Some(t) = text {
            parts.push(json!({"type": "text", "text": t}));
        }
        if before_img {
            match images.get(k) {
                Some(src) => parts.push(image_part(src)),
                None => parts.push(json!({"type": "text", "text": "[image omitted]"})),
            }
        }
    }
    Value::Array(parts)
}

pub fn messages_to_chat(messages: &[InputMessage]) -> Vec<Value> {
    let mut out = Vec::new();
    for m in messages {
        match m {
            InputMessage::System { content } => out.push(json!({"role": "system", "content": content})),
            InputMessage::User { content, image_sources } => {
                out.push(json!({"role": "user", "content": content_with_images(content, image_sources)}))
            }
            InputMessage::Assistant { content, reasoning_content, tool_calls } => {
                let mut a = json!({"role": "assistant", "content": content});
                if let Some(r) = reasoning_content {
                    a["reasoning_content"] = json!(r);
                }
                if let Some(tcs) = tool_calls {
                    if !tcs.is_empty() {
                        a["tool_calls"] = Value::Array(
                            tcs.iter()
                                .map(|t| json!({"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}}))
                                .collect(),
                        );
                    }
                }
                out.push(a)
            }
            InputMessage::Tool { content, image_sources, tool_call_id } => out.push(json!({
                "role": "tool", "tool_call_id": tool_call_id, "content": content_with_images(content, image_sources)})),
            InputMessage::LatestReminder { content } => out.push(json!({"role": "latest_reminder", "content": content})),
        }
    }
    out
}

pub fn tools_to_chat(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            let mut f = json!({"name": t.name});
            if let Some(d) = &t.description {
                f["description"] = json!(d);
            }
            // `{}` は省略と同じ描画になる。ds41rt は `{}` を 400 にするので送らない。
            let empty = t.parameters.as_object().is_some_and(|o| o.is_empty()) || t.parameters.is_null();
            if !empty {
                f["parameters"] = t.parameters.clone();
            }
            if let Some(s) = t.strict {
                f["strict"] = json!(s);
            }
            json!({"type": "function", "function": f})
        })
        .collect()
}

/// Conversation 全体を Chat 本文の断片 (messages, tools, response_format) にする。
pub fn conversation_to_chat(conv: &Conversation) -> serde_json::Map<String, Value> {
    let mut o = serde_json::Map::new();
    o.insert("messages".into(), Value::Array(messages_to_chat(&conv.messages)));
    let tools = tools_to_chat(&conv.tools);
    if !tools.is_empty() {
        o.insert("tools".into(), Value::Array(tools));
    }
    if matches!(conv.response_format, ResponseFormat::JsonObject) {
        o.insert("response_format".into(), json!({"type": "json_object"}));
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_recipe_core::multimodal::ImageDetail;

    fn img() -> ImageSource {
        ImageSource::DataUrl { data_url: "data:image/png;base64,AAAA".into(), detail: ImageDetail::High }
    }

    /// recipe の Chat 側の連結 (text は "\n\n" で結合、画像はプレースホルダ) を再現する。
    fn rejoin(v: &Value) -> String {
        match v {
            Value::String(s) => s.clone(),
            Value::Array(a) => a
                .iter()
                .map(|p| if p["type"] == "text" { p["text"].as_str().unwrap().to_string() } else { IMAGE_SPECIAL_TOKEN.to_string() })
                .collect::<Vec<_>>()
                .join("\n\n"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn image_split_roundtrip() {
        let t = IMAGE_SPECIAL_TOKEN;
        let cases = [
            format!("A\n\n{t}"),
            format!("{t}"),
            format!("{t}\n\n{t}"),
            format!("{t}\n\nB"),
            format!("A\n\n{t}\n\nB\n\n{t}\n\nC"),
            format!("\n\n{t}"),
            format!("{t}\n\n\n\n{t}"),
            format!("A\n\nB\n\n{t}\n\n"),
            format!("x\n\n\n\n{t}"),
        ];
        for c in cases {
            let n = c.matches(t).count();
            let imgs = vec![img(); n];
            let v = content_with_images(&c, &imgs);
            assert_eq!(rejoin(&v), c, "case {c:?} → {v}");
        }
    }
}
