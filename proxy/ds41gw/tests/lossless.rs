//! T-U3: 逆写像の無損失性。
//! Responses / Messages の要求 → recipe → Conversation (A) → proxy の逆写像で Chat → recipe の Chat 変換
//! (ds41rt と同じ) → Conversation (B) として、描画したプロンプトが一致することを確かめる。

use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_core::conversation::Conversation;
use deepseek_recipe_encoding::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::PromptEncoding;
use serde_json::{json, Value};

fn render(c: &Conversation) -> (String, usize) {
    let r = DeepseekV41Encoding::new().render_conversation(c);
    (r.prompt, r.image_sources.len())
}

/// proxy と同じく、ds41rt の規則に合わない名前 (ns::name) を写す。描画は写した名前で比べる。
fn map_names(a: &mut Conversation, names: &mut ds41gw::names::NameMap) {
    for t in &a.tools {
        names.reserve(&t.name);
    }
    for t in a.tools.iter_mut() {
        t.name = names.map(&t.name);
    }
    for m in a.messages.iter_mut() {
        if let deepseek_recipe_core::messages::InputMessage::Assistant { tool_calls: Some(tcs), .. } = m {
            for tc in tcs.iter_mut() {
                tc.name = names.map(&tc.name);
            }
        }
    }
}

fn via_chat(a: &Conversation) -> Conversation {
    let mut chat = ds41gw::reverse::conversation_to_chat(a);
    chat.insert("model".into(), json!("deepseek-ai/DeepSeek-V4.1-Flash"));
    // 思考は proxy が ds41rt に明示する
    chat.insert("thinking".into(), json!({"type": if a.thinking_mode { "enabled" } else { "disabled" }}));
    let body = Value::Object(chat);
    let req: ChatCompletionRequest = serde_json::from_value(body.clone()).unwrap_or_else(|e| panic!("{e}: {body}"));
    let mut b = req.convert(ConversionOptions::new()).unwrap_or_else(|e| panic!("chat convert: {e}\n{body:#}")).conversation;
    b.reasoning_effort = a.reasoning_effort;
    b.tool_choice = a.tool_choice;
    b
}

fn check(mut a: Conversation) {
    let mut names = ds41gw::names::NameMap::default();
    map_names(&mut a, &mut names);
    let b = via_chat(&a);
    let (pa, ia) = render(&a);
    let (pb, ib) = render(&b);
    assert_eq!(ia, ib, "image count");
    if pa != pb {
        let pos = pa.chars().zip(pb.chars()).position(|(x, y)| x != y).unwrap_or(pa.len().min(pb.len()));
        panic!("prompt mismatch at char {pos}\nA: {}\nB: {}", pa.chars().skip(pos.saturating_sub(80)).take(200).collect::<String>(),
            pb.chars().skip(pos.saturating_sub(80)).take(200).collect::<String>());
    }
    assert!(pa.len() > 50);
}

fn responses(v: Value) -> Conversation {
    let r: ResponsesRequest = serde_json::from_value(v).unwrap();
    r.convert(ConversionOptions::new()).unwrap().conversation
}

fn messages(v: Value) -> Conversation {
    let r: MessagesRequest = serde_json::from_value(v).unwrap();
    r.convert(ConversionOptions::new()).unwrap().conversation
}

const IMG: &str = "data:image/png;base64,iVBORw0KGgo=";

#[test]
fn responses_first_turn_like_c42_example1() {
    check(responses(json!({"model": "m", "max_output_tokens": 128000,
        "tools": [{"type": "function", "name": "bash", "description": "Execute a bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
                  {"type": "function", "name": "noparams", "description": "x"}],
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are an AI agent."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "こんにちは"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Runtime context."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<system-reminder>skills</system-reminder>"}, {"type": "input_text", "text": "second block"}]}
        ]})));
}

#[test]
fn responses_parallel_calls_reasoning_and_images() {
    check(responses(json!({"model": "m", "instructions": "inst",
        "tools": [{"type": "function", "name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}},
                  {"type": "function", "name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}],
        "input": [
            {"role": "developer", "content": "dev text"},
            {"role": "user", "content": [{"type": "input_text", "text": "look"}, {"type": "input_image", "image_url": IMG}, {"type": "input_text", "text": "and"}, {"type": "input_image", "image_url": IMG, "detail": "low"}]},
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "think 1"}]},
            {"type": "function_call", "call_id": "call_2", "name": "bash", "arguments": "{\"command\":\"ls -la\"}"},
            {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{\"path\":\"a\",\"n\":[1,2.5,{\"k\":null}]}"},
            {"type": "function_call_output", "call_id": "call_1", "output": [{"type": "input_text", "text": "see"}, {"type": "input_image", "image_url": IMG}]},
            {"type": "function_call_output", "call_id": "call_2", "output": "total 0\n[status: completed, exit code: 0]"},
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "think 2"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "progress"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "more"}]},
            {"role": "user", "content": "next"},
            {"type": "message", "role": "system", "content": "mid system"},
            {"role": "user", "content": "final"}
        ]})));
}

#[test]
fn responses_namespace_apply_patch_json_object_no_thinking() {
    check(responses(json!({"model": "m", "reasoning": {"effort": "none"}, "text": {"format": {"type": "json_object"}},
        "tools": [{"type": "namespace", "name": "fs", "description": "Filesystem", "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}]},
                  {"type": "custom", "name": "apply_patch"}],
        "input": [
            {"role": "user", "content": "give json"},
            {"type": "function_call", "call_id": "c1", "name": "read", "namespace": "fs", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "x"},
            {"type": "custom_tool_call", "call_id": "c2", "name": "apply_patch", "input": "*** Begin Patch\n*** End Patch\n"},
            {"type": "custom_tool_call_output", "call_id": "c2", "output": "Done"},
            {"role": "user", "content": "json please"}
        ]})));
}

#[test]
fn messages_tools_thinking_images() {
    check(messages(json!({"model": "m", "max_tokens": 100, "system": [{"type": "text", "text": "sys A"}, {"type": "text", "text": "sys B"}],
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}, {"type": "text", "text": "what?"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "s"},
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Tokyo", "n": 1.5}},
                {"type": "tool_use", "id": "toolu_0", "name": "get_weather", "input": {"city": "Osaka"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_0", "content": [{"type": "text", "text": "rain"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}]},
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                {"type": "text", "text": "thanks"}]},
            {"role": "assistant", "content": "plain answer"},
            {"role": "system", "content": "reminder"},
            {"role": "user", "content": "bye"}
        ]})));
}

/// 逆写像した本文を proxy の正規化に通しても、ds41rt の Chat 変換で通る形のままであること。
#[test]
fn reverse_output_is_valid_chat_after_prepare() {
    let a = responses(json!({"model": "m", "tools": [{"type": "namespace", "name": "ns", "tools": [{"type": "function", "name": "f"}]}],
        "input": [{"role": "user", "content": "x"},
                  {"type": "function_call", "call_id": "c1", "name": "f", "namespace": "ns", "arguments": "{}"},
                  {"type": "function_call_output", "call_id": "c1", "output": "y"}]}));
    let chat = ds41gw::reverse::conversation_to_chat(&a);
    // 名前 ns::f は ds41rt の規則に合わないので、そのままでは recipe の Chat 変換が 400 にする
    let mut body = Value::Object(chat);
    body["model"] = json!("m");
    let req: ChatCompletionRequest = serde_json::from_value(body.clone()).unwrap();
    assert!(req.convert(ConversionOptions::new()).is_err());
    // 写すと通る
    let mut names = ds41gw::names::NameMap::default();
    for t in body["tools"].as_array_mut().unwrap() {
        let n = t["function"]["name"].as_str().unwrap().to_string();
        t["function"]["name"] = json!(names.map(&n));
    }
    for m in body["messages"].as_array_mut().unwrap() {
        if let Some(tcs) = m.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for tc in tcs {
                let n = tc["function"]["name"].as_str().unwrap().to_string();
                tc["function"]["name"] = json!(names.map(&n));
            }
        }
    }
    let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
    assert!(req.convert(ConversionOptions::new()).is_ok());
    assert_eq!(names.unmap("ns__f"), "ns::f");
}
