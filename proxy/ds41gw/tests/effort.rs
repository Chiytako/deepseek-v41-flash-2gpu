//! reasoning effort の数値 (1〜100) と段階名を、各 API (Chat、Chat の chat_template_kwargs、Responses、Messages、
//! count_tokens、legacy completions) で受けて、ds41rt に `thinking:{type}` と整数の `reasoning_effort` で送ること。
//! `thinking.numeric_upstream=false` では段階名 (≤61 low、62〜88 high、他 max) に戻して送ること。
//! 段階名の既定は low 50、medium 62 (独自)、high 75、xhigh 88 (独自)、max 100。

mod common;

use common::*;
use serde_json::{json, Value};

async fn post(px: &Proxy, path: &str, b: Value) -> reqwest::Response {
    client().post(format!("{}{path}", px.url)).json(&b).send().await.unwrap()
}

fn chat(extra: Value) -> Value {
    let mut b = json!({"model": "x", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]});
    b.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    b
}

fn responses(extra: Value) -> Value {
    let mut b = json!({"model": "x", "max_output_tokens": 50, "input": "hi"});
    b.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    b
}

fn messages(extra: Value) -> Value {
    let mut b = json!({"model": "x", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]});
    b.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    b
}

/// 送った本文の思考の指定: Some(effort) なら thinking enabled、None なら disabled (effort は送らない)。
fn sent(mock: &Mock) -> Option<Value> {
    let b = mock.last_body();
    match b["thinking"]["type"].as_str() {
        Some("enabled") => Some(b["reasoning_effort"].clone()),
        Some("disabled") => {
            assert!(b.get("reasoning_effort").is_none(), "{b}");
            None
        }
        other => panic!("thinking.type {other:?}: {b}"),
    }
}

/// 数値・数値文字列・段階名と、ds41rt に送る値 (numeric_upstream=true)。
fn accepted() -> Vec<(Value, Option<Value>)> {
    vec![
        (json!(60), Some(json!(60))),
        (json!("60"), Some(json!(60))),
        (json!(1), Some(json!(1))),
        (json!(100), Some(json!(100))),
        (json!(" 33 "), Some(json!(33))),
        (json!(0.9), Some(json!(90))),
        (json!(42.6), Some(json!(43))),
        (json!("minimal"), Some(json!(50))),
        (json!("low"), Some(json!(50))),
        (json!("medium"), Some(json!(62))),
        (json!("Medium"), Some(json!(62))),
        (json!("high"), Some(json!(75))),
        (json!("HIGH"), Some(json!(75))),
        (json!("xhigh"), Some(json!(88))),
        (json!("XHIGH"), Some(json!(88))),
        (json!(62), Some(json!(62))),
        (json!("88"), Some(json!(88))),
        (json!("max"), Some(json!(100))),
        (json!(0), None),
        (json!("0"), None),
        (json!("none"), None),
        (json!("off"), None),
        (json!("disabled"), None),
    ]
}

fn rejected() -> Vec<Value> {
    vec![json!(101), json!("101"), json!(-1), json!(100.5), json!("abc"), json!("inf"), json!(true), json!([60])]
}

async fn openai_error(r: reqwest::Response, param: &str) -> Value {
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["code"], "invalid_reasoning_effort", "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    assert_eq!(v["error"]["param"], param, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("integer from 1 to 100"), "{v}");
    v
}

#[tokio::test]
async fn chat_reasoning_effort_numbers_and_names() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    for (v, want) in accepted() {
        let r = post(&px, "/v1/chat/completions", chat(json!({"reasoning_effort": v.clone()}))).await;
        assert_eq!(r.status(), 200, "{v}");
        assert_eq!(sent(&mock), want, "{v}");
    }
    // 未指定は high (75)
    let _ = post(&px, "/v1/chat/completions", chat(json!({}))).await;
    assert_eq!(sent(&mock), Some(json!(75)));
    // ストリームでも同じ
    let r = post(&px, "/v1/chat/completions", chat(json!({"stream": true, "reasoning_effort": "60"}))).await;
    assert_eq!(read_sse(r).await.last().unwrap().1, "[DONE]");
    assert_eq!(sent(&mock), Some(json!(60)));
    for v in rejected() {
        let r = post(&px, "/v1/chat/completions", chat(json!({"reasoning_effort": v.clone()}))).await;
        let e = openai_error(r, "reasoning_effort").await;
        if v == json!(101) {
            assert!(e["error"]["message"].as_str().unwrap().contains("101"), "{e}");
        }
    }
}

#[tokio::test]
async fn chat_template_kwargs_reasoning_effort() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    for (v, want) in accepted() {
        let r = post(&px, "/v1/chat/completions", chat(json!({"chat_template_kwargs": {"reasoning_effort": v.clone()}}))).await;
        assert_eq!(r.status(), 200, "{v}");
        assert_eq!(sent(&mock), want, "{v}");
    }
    // enable_thinking と一緒に
    let _ = post(&px, "/v1/chat/completions", chat(json!({"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": 40}}))).await;
    assert_eq!(sent(&mock), Some(json!(40)));
    // トップレベルの effort が優先 (chat_template_kwargs の 0 で思考を止めない)
    let _ = post(&px, "/v1/chat/completions", chat(json!({"reasoning_effort": 60, "chat_template_kwargs": {"reasoning_effort": 0}}))).await;
    assert_eq!(sent(&mock), Some(json!(60)));
    for v in rejected() {
        let r = post(&px, "/v1/chat/completions", chat(json!({"chat_template_kwargs": {"reasoning_effort": v}}))).await;
        openai_error(r, "chat_template_kwargs.reasoning_effort").await;
    }
}

#[tokio::test]
async fn responses_reasoning_effort_numbers_and_names() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    for (v, want) in accepted() {
        let r = post(&px, "/v1/responses", responses(json!({"reasoning": {"effort": v.clone()}}))).await;
        assert_eq!(r.status(), 200, "{v}");
        let o: Value = r.json().await.unwrap();
        assert_eq!(sent(&mock), want, "{v}");
        // 応答に返す reasoning.effort は要求の値そのまま
        assert_eq!(o["reasoning"]["effort"], v, "{o}");
        assert_eq!(o["output"][0]["type"], if want.is_some() { "reasoning" } else { "message" }, "{v}");
    }
    // 未指定は high (75)、応答には "high"
    let o: Value = post(&px, "/v1/responses", responses(json!({}))).await.json().await.unwrap();
    assert_eq!(sent(&mock), Some(json!(75)));
    assert_eq!(o["reasoning"]["effort"], "high");
    let o: Value = post(&px, "/v1/responses", responses(json!({"reasoning": {"summary": "auto"}}))).await.json().await.unwrap();
    assert_eq!(o["reasoning"]["effort"], "high");
    // 空白だけ・空文字列は指定なし: 既定の 75 を送り、応答にも "high"
    for blank in [json!(" "), json!("")] {
        let o: Value = post(&px, "/v1/responses", responses(json!({"reasoning": {"effort": blank.clone()}}))).await.json().await.unwrap();
        assert_eq!(sent(&mock), Some(json!(75)), "{blank}");
        assert_eq!(o["reasoning"]["effort"], "high", "{blank}");
    }
    // ストリーム: response.created と response.completed の reasoning.effort
    let r = post(&px, "/v1/responses", responses(json!({"stream": true, "reasoning": {"effort": 60}}))).await;
    let ds = datas(&read_sse(r).await);
    check_responses_events(&ds).unwrap();
    assert_eq!(sent(&mock), Some(json!(60)));
    for ty in ["response.created", "response.completed"] {
        let e = ds.iter().find(|d| d["type"] == ty).unwrap();
        assert_eq!(e["response"]["reasoning"]["effort"], 60, "{ty}");
    }
    // 保存した応答 (GET) も同じ
    let id = ds.iter().find(|d| d["type"] == "response.completed").unwrap()["response"]["id"].as_str().unwrap().to_string();
    let g: Value = client().get(format!("{}/v1/responses/{id}", px.url)).send().await.unwrap().json().await.unwrap();
    assert_eq!(g["reasoning"]["effort"], 60);
    for v in rejected() {
        let r = post(&px, "/v1/responses", responses(json!({"reasoning": {"effort": v}}))).await;
        openai_error(r, "reasoning.effort").await;
    }
}

#[tokio::test]
async fn messages_output_config_effort_and_budget_tokens() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    for (v, want) in accepted() {
        let r = post(&px, "/v1/messages", messages(json!({"output_config": {"effort": v.clone()}}))).await;
        assert_eq!(r.status(), 200, "{v}");
        assert_eq!(sent(&mock), want, "{v}");
    }
    let _ = post(&px, "/v1/messages", messages(json!({"output_config": {"effort": "ultra"}}))).await;
    assert_eq!(sent(&mock), Some(json!(100)));
    let r = post(&px, "/v1/messages", messages(json!({"output_config": {"effort": "  "}}))).await;
    assert_eq!(r.status(), 200);
    assert_eq!(sent(&mock), Some(json!(75)));
    // thinking の明示と一緒に (adaptive + effort は Claude の推奨の形)
    let _ = post(&px, "/v1/messages", messages(json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": 60}}))).await;
    assert_eq!(sent(&mock), Some(json!(60)));
    let _ = post(&px, "/v1/messages", messages(json!({"thinking": {"type": "disabled"}, "output_config": {"effort": 60}}))).await;
    assert_eq!(sent(&mock), None);
    // budget_tokens は従来の境界で low/high/max の値
    for (b, want) in [(1024, 50), (8191, 50), (8192, 75), (16000, 75), (32767, 75), (32768, 100), (64000, 100)] {
        let r = post(&px, "/v1/messages", messages(json!({"max_tokens": 70000, "thinking": {"type": "enabled", "budget_tokens": b}}))).await;
        assert_eq!(r.status(), 200);
        assert_eq!(sent(&mock), Some(json!(want)), "budget {b}");
    }
    // budget_tokens の数値文字列も受ける。不正値は黙って既定 (75) にせず Anthropic の形の 400
    let r = post(&px, "/v1/messages", messages(json!({"max_tokens": 70000, "thinking": {"type": "enabled", "budget_tokens": "40000"}}))).await;
    assert_eq!(r.status(), 200);
    assert_eq!(sent(&mock), Some(json!(100)));
    for v in [json!(-1), json!("abc"), json!(2048.5)] {
        let r = post(&px, "/v1/messages", messages(json!({"thinking": {"type": "enabled", "budget_tokens": v.clone()}}))).await;
        assert_eq!(r.status(), 400, "{v}");
        let e: Value = r.json().await.unwrap();
        assert_eq!(e["error"]["type"], "invalid_request_error", "{e}");
        assert!(e["error"]["message"].as_str().unwrap().contains("thinking.budget_tokens"), "{e}");
        let r = post(&px, "/v1/messages/count_tokens", json!({"model": "x", "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "enabled", "budget_tokens": v.clone()}})).await;
        assert_eq!(r.status(), 400, "count_tokens {v}");
    }
    // effort が budget_tokens より優先
    let _ = post(&px, "/v1/messages", messages(json!({"thinking": {"type": "enabled", "budget_tokens": 2048}, "output_config": {"effort": "max"}}))).await;
    assert_eq!(sent(&mock), Some(json!(100)));
    // ストリーム
    let r = post(&px, "/v1/messages", messages(json!({"stream": true, "output_config": {"effort": "60"}}))).await;
    assert_eq!(r.status(), 200);
    let _ = read_sse(r).await;
    assert_eq!(sent(&mock), Some(json!(60)));
    // 範囲外・非数は Anthropic の形の 400
    for v in rejected() {
        let r = post(&px, "/v1/messages", messages(json!({"output_config": {"effort": v.clone()}}))).await;
        assert_eq!(r.status(), 400, "{v}");
        let e: Value = r.json().await.unwrap();
        assert_eq!(e["type"], "error", "{e}");
        assert_eq!(e["error"]["type"], "invalid_request_error", "{e}");
        let m = e["error"]["message"].as_str().unwrap();
        assert!(m.contains("output_config.effort") && m.contains("integer from 1 to 100"), "{m}");
    }
}

#[tokio::test]
async fn count_tokens_accepts_numeric_effort() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    let before = mock.st.calls.load(std::sync::atomic::Ordering::SeqCst);
    for v in [json!(60), json!("60"), json!("high"), json!("max"), json!(0)] {
        let r = post(&px, "/v1/messages/count_tokens", messages(json!({"output_config": {"effort": v.clone()}}))).await;
        assert_eq!(r.status(), 200, "{v}");
        assert!(r.json::<Value>().await.unwrap()["input_tokens"].as_u64().unwrap() > 0);
    }
    let r = post(&px, "/v1/messages/count_tokens", messages(json!({"output_config": {"effort": 101}}))).await;
    assert_eq!(r.status(), 400);
    assert_eq!(mock.st.calls.load(std::sync::atomic::Ordering::SeqCst), before);
}

/// count_tokens の描画は、ds41rt と同じく `Reasoning Effort: N` を描く (patched recipe の Budget)。
#[test]
fn count_tokens_render_uses_the_numeric_effort() {
    use deepseek_recipe::anthropic::MessagesRequest;
    use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
    use deepseek_recipe_encoding::dsv41::DeepseekV41Encoding;
    use deepseek_recipe_encoding::PromptEncoding;

    let render = |cfg: &ds41gw::config::Thinking, chat: Value| {
        let req: MessagesRequest = serde_json::from_value(json!({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})).unwrap();
        let mut conv = req.convert(ConversionOptions::new()).unwrap().conversation;
        let (enabled, effort) = ds41gw::messages::render_thinking(cfg, &chat).unwrap();
        conv.thinking_mode = enabled;
        conv.reasoning_effort = effort;
        DeepseekV41Encoding::new().render_conversation(&conv).prompt
    };
    let line = |n: u32| format!("Reasoning Effort: {n} (range 1-100");
    let cfg = ds41gw::config::Thinking::default();
    assert!(render(&cfg, json!({"reasoning_effort": 60})).contains(&line(60)));
    assert!(render(&cfg, json!({"reasoning_effort": "60"})).contains(&line(60)));
    assert!(render(&cfg, json!({"reasoning_effort": "low"})).contains(&line(50)));
    assert!(render(&cfg, json!({"reasoning_effort": "medium"})).contains(&line(62)));
    assert!(render(&cfg, json!({"reasoning_effort": "high"})).contains(&line(75)));
    assert!(render(&cfg, json!({"reasoning_effort": "xhigh"})).contains(&line(88)));
    assert!(render(&cfg, json!({"reasoning_effort": "max"})).contains(&line(100)));
    assert!(render(&cfg, json!({})).contains(&line(75)));
    assert!(render(&cfg, json!({"thinking": {"type": "enabled", "budget_tokens": 2048}})).contains(&line(50)));
    assert!(!render(&cfg, json!({"reasoning_effort": 0})).contains("Reasoning Effort"));
    // 旧 ds41rt (段階名しか受けない) に送る設定では、ds41rt が描く段階の値で数える
    let mut old = cfg.clone();
    old.numeric_upstream = false;
    assert!(render(&old, json!({"reasoning_effort": 60})).contains(&line(50)));
    assert!(render(&old, json!({"reasoning_effort": 80})).contains(&line(75)));
    assert!(render(&old, json!({"reasoning_effort": 90})).contains(&line(100)));
    assert!(render(&old, json!({"reasoning_effort": "medium"})).contains(&line(75)));
    assert!(render(&old, json!({"reasoning_effort": "xhigh"})).contains(&line(75)));
}

#[tokio::test]
async fn completions_accept_numeric_effort() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |_| {}).await;
    // 既定は思考 off
    let _ = post(&px, "/v1/completions", json!({"model": "x", "prompt": "a"})).await;
    assert_eq!(sent(&mock), None);
    for (v, want) in accepted() {
        let r = post(&px, "/v1/completions", json!({"model": "x", "prompt": "a", "reasoning_effort": v.clone()})).await;
        assert_eq!(r.status(), 200, "{v}");
        assert_eq!(sent(&mock), want, "{v}");
    }
    let _ = post(&px, "/v1/completions", json!({"model": "x", "prompt": "a", "chat_template_kwargs": {"reasoning_effort": "70"}})).await;
    assert_eq!(sent(&mock), Some(json!(70)));
    for v in rejected() {
        let r = post(&px, "/v1/completions", json!({"model": "x", "prompt": "a", "reasoning_effort": v})).await;
        openai_error(r, "reasoning_effort").await;
    }
}

#[tokio::test]
async fn numeric_upstream_false_sends_nearest_level_names() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| c.thinking.numeric_upstream = false).await;
    let table = [
        (json!(1), "low"),
        (json!(60), "low"),
        (json!(61), "low"),
        (json!("62"), "high"),
        (json!(63), "high"),
        (json!(80), "high"),
        (json!(88), "high"),
        (json!(89), "max"),
        (json!(100), "max"),
        (json!("minimal"), "low"),
        (json!("low"), "low"),
        (json!("medium"), "high"),
        (json!("high"), "high"),
        (json!("xhigh"), "high"),
        (json!("max"), "max"),
    ];
    for (v, want) in &table {
        let _ = post(&px, "/v1/chat/completions", chat(json!({"reasoning_effort": v}))).await;
        assert_eq!(sent(&mock), Some(json!(want)), "chat {v}");
        let o: Value = post(&px, "/v1/responses", responses(json!({"reasoning": {"effort": v}}))).await.json().await.unwrap();
        assert_eq!(sent(&mock), Some(json!(want)), "responses {v}");
        assert_eq!(&o["reasoning"]["effort"], v);
        let _ = post(&px, "/v1/messages", messages(json!({"output_config": {"effort": v}}))).await;
        assert_eq!(sent(&mock), Some(json!(want)), "messages {v}");
        let _ = post(&px, "/v1/completions", json!({"model": "x", "prompt": "a", "reasoning_effort": v})).await;
        assert_eq!(sent(&mock), Some(json!(want)), "completions {v}");
    }
    let _ = post(&px, "/v1/chat/completions", chat(json!({}))).await;
    assert_eq!(sent(&mock), Some(json!("high")));
    let _ = post(&px, "/v1/messages", messages(json!({"max_tokens": 70000, "thinking": {"type": "enabled", "budget_tokens": 64000}}))).await;
    assert_eq!(sent(&mock), Some(json!("max")));
    let _ = post(&px, "/v1/chat/completions", chat(json!({"reasoning_effort": 0}))).await;
    assert_eq!(sent(&mock), None);
    let r = post(&px, "/v1/chat/completions", chat(json!({"reasoning_effort": 101}))).await;
    assert_eq!(r.status(), 400);
}

#[tokio::test]
async fn effort_map_can_be_overridden() {
    let mock = start_mock().await;
    let px = start_proxy(&mock, |c| {
        c.thinking.effort_map.insert("xhigh".into(), 90);
        c.thinking.effort_map.insert("quick".into(), 20);
        c.thinking.default_effort = "60".into();
    })
    .await;
    let _ = post(&px, "/v1/chat/completions", chat(json!({"reasoning_effort": "xhigh"}))).await;
    assert_eq!(sent(&mock), Some(json!(90)));
    let _ = post(&px, "/v1/responses", responses(json!({"reasoning": {"effort": "Quick"}}))).await;
    assert_eq!(sent(&mock), Some(json!(20)));
    let o: Value = post(&px, "/v1/responses", responses(json!({}))).await.json().await.unwrap();
    assert_eq!(sent(&mock), Some(json!(60)));
    assert_eq!(o["reasoning"]["effort"], "60");
}

/// vendoring した recipe は数値の effort を解ける (crates.io の 0.1.0 は解けない)。
#[test]
fn vendored_recipe_parses_numeric_effort() {
    use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
    use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
    use deepseek_recipe_core::conversation::ReasoningEffort;

    let r: ResponsesRequest = serde_json::from_value(json!({"model": "m", "input": "hi", "reasoning": {"effort": 60}})).unwrap();
    let c = r.convert(ConversionOptions::new().with_default_thinking_mode(true)).unwrap().conversation;
    assert_eq!(c.reasoning_effort, Some(ReasoningEffort::Budget(60)));
    let r: ChatCompletionRequest =
        serde_json::from_value(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "60"})).unwrap();
    let c = r.convert(ConversionOptions::new().with_default_thinking_mode(true)).unwrap().conversation;
    assert_eq!(c.reasoning_effort, Some(ReasoningEffort::Budget(60)));
    // medium / xhigh は ds41rt 改修版と同じ独自の値 (62 / 88) に写り、ds41gw の既定の effort_map と一致する
    let map = ds41gw::config::default_effort_map();
    for (name, score) in [("medium", 62u8), ("xhigh", 88u8)] {
        assert_eq!(map[name], score);
        let r: ChatCompletionRequest =
            serde_json::from_value(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": name})).unwrap();
        let c = r.convert(ConversionOptions::new().with_default_thinking_mode(true)).unwrap().conversation;
        assert_eq!(c.reasoning_effort, Some(ReasoningEffort::Budget(score)), "{name}");
        let r: ResponsesRequest = serde_json::from_value(json!({"model": "m", "input": "hi", "reasoning": {"effort": name}})).unwrap();
        let c = r.convert(ConversionOptions::new().with_default_thinking_mode(true)).unwrap().conversation;
        assert_eq!(c.reasoning_effort, Some(ReasoningEffort::Budget(score)), "{name}");
    }
    assert_eq!(ReasoningEffort::Budget(62).level(), ReasoningEffort::High);
    assert_eq!(ReasoningEffort::Budget(88).level(), ReasoningEffort::High);
}
