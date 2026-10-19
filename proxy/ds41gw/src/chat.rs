//! `/v1/chat/completions`: 正規化して ds41rt に素通しする (r7_proxy_spec.md §5)。
//! Responses / Messages / completions も、最後はここの `prepare` で Chat 本文に揃えてから送る。

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use serde_json::{json, Map, Value};

use crate::core::{
    aggregate_with_retry, begin, connect_all, connect_early, error_response, keepalive_interval, merge_sources,
    nonstream_response, parse_json, sse_data, sse_response, Collected, Connect, Ctx, Item, SourceOpts,
};
use crate::error::{ApiError, Flavor};
use crate::names::NameMap;
use crate::thinking::{self, Decision};
use crate::upstream::{usage_num, Ev};
use crate::util::{as_u64_lenient, content_text, is_truthy};

/// どの API から来た要求か (既定値の違いに使う)。
#[derive(Debug, Clone, Default)]
pub struct PrepOpts {
    /// 思考の既定を上書きする (completions は false)。
    pub default_thinking: Option<bool>,
    pub default_max_tokens: Option<u64>,
    /// n>1 を許すか。
    pub allow_n: bool,
    /// 名前の写像を先に作ってある場合 (Responses の namespace など)。
    pub names: Option<NameMap>,
}

#[derive(Debug, Clone)]
pub struct Prepared {
    /// ds41rt に送る Chat 本文 (stream は送る直前に付ける)。
    pub body: Value,
    pub names: Arc<NameMap>,
    pub thinking: Decision,
    pub n: usize,
    pub stream: bool,
    pub include_usage: bool,
    pub model_out: String,
    pub drop_tool_calls: bool,
    pub max_tokens: u64,
}

const BUILTIN_TOOL_TYPES: [&str; 12] = [
    "web_search",
    "web_search_preview",
    "web_search_2025_08_26",
    "file_search",
    "code_interpreter",
    "computer_use_preview",
    "computer",
    "image_generation",
    "mcp",
    "local_shell",
    "shell",
    "custom",
];

fn obj(v: &mut Value) -> &mut Map<String, Value> {
    if !v.is_object() {
        *v = Value::Object(Map::new());
    }
    v.as_object_mut().unwrap()
}

/// model 名の解決 (§4.1)。(送る名前, 応答に書く名前)。
pub fn resolve_model(ctx: &Ctx, requested: Option<&str>) -> Result<(String, String), ApiError> {
    let cfg = &ctx.app.cfg;
    let canon = cfg.upstream.model.clone();
    let req = requested.unwrap_or("").to_string();
    ctx.stats.lock().unwrap().model_requested = Some(req.clone());
    if cfg.models.alias_mode == "listed" && req != canon && !cfg.models.aliases.iter().any(|a| *a == req) {
        return Err(ApiError::new(crate::error::Kind::NotFound, "model_not_found", format!("The model `{req}` does not exist"))
            .with_param("model"));
    }
    if req != canon {
        ctx.normalized("model_alias");
    }
    let out = if cfg.models.echo_requested && !req.is_empty() { req } else { canon.clone() };
    Ok((canon, out))
}

/// Chat 本文を ds41rt が受け付ける形に正規化する (§5.2、§5.3)。
pub fn prepare(ctx: &Ctx, mut body: Value, opts: &PrepOpts) -> Result<Prepared, ApiError> {
    let app = &ctx.app;
    let cfg = &app.cfg;
    let o = obj(&mut body);

    // model
    let (canon, model_out) = resolve_model(ctx, o.get("model").and_then(Value::as_str))?;
    o.insert("model".into(), Value::String(canon));

    // stream / stream_options
    let stream = is_truthy(o.get("stream"));
    let include_usage = is_truthy(o.get("stream_options").and_then(|s| s.get("include_usage")));
    if o.remove("stream_options").is_some() && !stream {
        ctx.normalized("stream_options_removed");
    }
    o.remove("stream");
    ctx.stats.lock().unwrap().stream = stream;

    // n
    let n = match o.remove("n") {
        None | Some(Value::Null) => 1,
        Some(v) => as_u64_lenient(&v).ok_or_else(|| ApiError::bad("invalid_n", "n must be a positive integer").with_param("n"))? as usize,
    };
    if n == 0 || n > cfg.limits.max_n || (n > 1 && !opts.allow_n) {
        return Err(ApiError::bad("invalid_n", format!("n must be between 1 and {}", cfg.limits.max_n)).with_param("n"));
    }

    // logprobs
    let reject = cfg.compat.unsupported == "reject";
    if is_truthy(o.get("logprobs")) {
        if reject {
            return Err(ApiError::unsupported("logprobs"));
        }
        ctx.warn("ignored=logprobs");
    }
    if o.get("top_logprobs").and_then(Value::as_f64).unwrap_or(0.0) > 0.0 {
        if reject {
            return Err(ApiError::unsupported("top_logprobs"));
        }
        ctx.warn("ignored=top_logprobs");
    }
    o.remove("logprobs");
    o.remove("top_logprobs");
    for k in ["echo", "suffix", "best_of"] {
        o.remove(k);
    }

    // penalty 系
    let mut ignored = Vec::new();
    for (k, neutral) in [("frequency_penalty", 0.0), ("presence_penalty", 0.0), ("repetition_penalty", 1.0)] {
        if let Some(v) = o.get(k) {
            if v.is_null() {
                o.remove(k);
                continue;
            }
            let x = v.as_f64().ok_or_else(|| ApiError::bad("invalid_parameter", format!("{k} must be a number")).with_param(k))?;
            if k != "repetition_penalty" && !(-2.0..=2.0).contains(&x) {
                return Err(ApiError::bad("invalid_parameter", format!("{k} must be in [-2, 2]")).with_param(k));
            }
            if x != neutral && !(k == "repetition_penalty" && x == 0.0) {
                ignored.push(k.to_string());
            }
            if k == "repetition_penalty" {
                o.remove(k);
            }
        }
    }
    for k in ["logit_bias", "service_tier"] {
        if o.get(k).is_some_and(|v| !v.is_null()) {
            ignored.push(k.to_string());
        }
    }
    if !ignored.is_empty() {
        ctx.warn(format!("ignored={}", ignored.join("+")));
    }
    for k in ["user", "metadata", "store", "prompt_cache_key", "safety_identifier", "prompt_cache_retention", "modalities", "audio", "prediction", "web_search_options", "verbosity"] {
        o.remove(k);
    }

    // max_tokens (§4.4)
    let mt = o.remove("max_tokens").filter(|v| !v.is_null());
    let mct = o.remove("max_completion_tokens").filter(|v| !v.is_null());
    let max_tokens = match mt.or(mct.clone()) {
        Some(v) => {
            let n = as_u64_lenient(&v).ok_or_else(|| ApiError::bad("invalid_max_tokens", "max_tokens must be a positive integer").with_param("max_tokens"))?;
            if n == 0 {
                return Err(ApiError::bad("invalid_max_tokens", "max_tokens must be greater than zero").with_param("max_tokens"));
            }
            if mct.is_some() && !o.contains_key("max_tokens") {
                ctx.normalized("max_completion_tokens");
            }
            n
        }
        None => {
            ctx.normalized("max_tokens_default");
            opts.default_max_tokens.unwrap_or(cfg.limits.default_max_tokens)
        }
    };
    o.insert("max_tokens".into(), json!(max_tokens));

    // temperature / top_p (§4.2)
    let has_temp = o.get("temperature").is_some_and(|v| !v.is_null());
    if !has_temp {
        o.remove("temperature");
        if let Some((t, p)) = cfg.sampling.default_pair() {
            o.insert("temperature".into(), json!(t));
            if !o.get("top_p").is_some_and(|v| !v.is_null()) {
                o.insert("top_p".into(), json!(p));
            }
            ctx.normalized("temperature_default");
        }
    } else if !o.get("top_p").is_some_and(|v| !v.is_null()) {
        o.insert("top_p".into(), json!(1.0));
    }
    if o.get("top_p").is_some_and(Value::is_null) {
        o.remove("top_p");
    }

    // stop
    match o.get("stop") {
        Some(Value::Array(a)) if a.len() > cfg.limits.max_stop => {
            return Err(ApiError::bad("invalid_stop", format!("stop must contain at most {} sequences", cfg.limits.max_stop)).with_param("stop"));
        }
        Some(Value::Null) => {
            o.remove("stop");
        }
        _ => {}
    }

    // thinking (§4.3)
    let mut norm = Vec::new();
    let mut d = thinking::resolve(o, &cfg.thinking, &mut norm)?;
    for n in norm {
        ctx.normalized(n);
    }
    if let Some(def) = opts.default_thinking {
        if !d.explicit {
            d.enabled = def;
        }
    }
    let auto_off = cfg.thinking.auto_off_below_max_tokens;
    if auto_off > 0 && !d.explicit && max_tokens < auto_off && d.enabled {
        d.enabled = false;
        ctx.normalized("thinking_auto_off");
    }
    thinking::apply(o, &d, &cfg.thinking);

    // messages
    let mut messages = match o.remove("messages") {
        Some(Value::Array(a)) => a,
        Some(_) => return Err(ApiError::bad("invalid_messages", "messages must be an array").with_param("messages")),
        None => return Err(ApiError::bad("invalid_messages", "missing field `messages`").with_param("messages")),
    };
    if messages.is_empty() {
        return Err(ApiError::bad("invalid_messages", "messages must not be empty").with_param("messages"));
    }
    normalize_messages(ctx, &mut messages)?;
    if cfg.compat.repair_tool_history {
        let (fixed, repaired) = repair_tool_history(messages);
        messages = fixed;
        if repaired {
            ctx.warn("repaired_tool_history");
            ctx.normalized("repaired_tool_history");
        }
    }

    // 思考の消去と再注入 (§4.3.4、§10.1)
    if d.clear {
        for m in messages.iter_mut() {
            if m.get("role").and_then(Value::as_str) == Some("assistant") {
                if let Some(mo) = m.as_object_mut() {
                    mo.remove("reasoning_content");
                }
            }
        }
        ctx.normalized("clear_thinking");
    } else if d.enabled && cfg.reasoning.reinject {
        let n = reinject(ctx, &mut messages);
        if n > 0 {
            ctx.stats.lock().unwrap().reinjected += n;
            ctx.normalized("reasoning_reinjected");
        }
    }

    // tools と名前の写像 (§5.3.8)
    let mut names = opts.names.clone().unwrap_or_default();
    let tools = normalize_tools(ctx, o.remove("tools"), &mut names)?;
    let has_tools = !tools.is_empty();
    if has_tools {
        o.insert("tools".into(), Value::Array(tools));
    }
    for m in messages.iter_mut() {
        if let Some(tcs) = m.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for tc in tcs {
                if let Some(n) = tc.pointer("/function/name").and_then(Value::as_str).map(str::to_string) {
                    let mapped = names.map(&n);
                    tc["function"]["name"] = Value::String(mapped);
                }
            }
        }
    }
    let mut tool_choice = o.remove("tool_choice").filter(|v| !v.is_null());
    if let Some(tc) = &mut tool_choice {
        normalize_tool_choice(tc, &mut names)?;
    }
    if !names.is_empty() {
        ctx.normalized("tool_name_mapped");
    }

    // tool_choice:"none" と DSML の漏れ (§5.3.9)
    let choice_none = tool_choice.as_ref().and_then(Value::as_str) == Some("none");
    let history_has_tools = messages.iter().any(|m| {
        m.get("role").and_then(Value::as_str) == Some("tool") || m.get("tool_calls").is_some_and(|t| t.as_array().is_some_and(|a| !a.is_empty()))
    });
    let mut drop_tool_calls = false;
    if (choice_none || !has_tools) && history_has_tools {
        if cfg.compat.tool_none_strategy == "keep_tools" && has_tools {
            tool_choice = Some(json!("auto"));
            drop_tool_calls = true;
            ctx.normalized("tool_none_keep_tools");
        } else {
            flatten_tool_history(&mut messages, &names);
            o.remove("tools");
            tool_choice = None;
            ctx.normalized("tool_none_flattened");
            ctx.warn("tool_history_flattened");
        }
    } else if choice_none {
        // 履歴にツールが無ければ ds41rt の "none" で問題ない
    }
    if let Some(tc) = tool_choice {
        if o.contains_key("tools") || tc.as_str() == Some("none") {
            o.insert("tool_choice".into(), tc);
        }
    }
    if !o.contains_key("tools") {
        o.remove("parallel_tool_calls");
    }

    // 画像の枚数 (§6.3)
    limit_images(ctx, &mut messages)?;
    o.insert("messages".into(), Value::Array(messages));

    let prepared = Prepared {
        body,
        names: Arc::new(names),
        thinking: d,
        n,
        stream,
        include_usage,
        model_out,
        drop_tool_calls,
        max_tokens,
    };
    precheck_context(ctx, &prepared)?;
    Ok(prepared)
}

/// §5.3 の 1〜5 (ロール、null の content、パーツの型、思考の別名、tool_calls の整形)。
fn normalize_messages(ctx: &Ctx, messages: &mut Vec<Value>) -> Result<(), ApiError> {
    let mut gen_id = 0usize;
    for (i, m) in messages.iter_mut().enumerate() {
        let Some(mo) = m.as_object_mut() else {
            return Err(ApiError::bad("invalid_messages", format!("messages[{i}] must be an object")).with_param(format!("messages[{i}]")));
        };
        let role = mo.get("role").and_then(Value::as_str).unwrap_or("").to_string();
        let role = match role.as_str() {
            "developer" => {
                ctx.normalized("developer_to_system");
                "system".to_string()
            }
            "function" => {
                ctx.normalized("function_to_tool");
                mo.remove("name");
                if !mo.contains_key("tool_call_id") {
                    mo.insert("tool_call_id".into(), Value::String(String::new()));
                }
                "tool".to_string()
            }
            "system" | "user" | "assistant" | "tool" | "latest_reminder" => role,
            "" => return Err(ApiError::bad("invalid_messages", format!("messages[{i}].role is required")).with_param(format!("messages[{i}].role"))),
            other => {
                return Err(ApiError::bad("invalid_messages", format!("messages[{i}]: unsupported role {other:?}"))
                    .with_param(format!("messages[{i}].role")))
            }
        };
        mo.insert("role".into(), Value::String(role.clone()));
        mo.remove("name");

        // content
        let content = mo.remove("content").unwrap_or(Value::Null);
        let content = normalize_content(ctx, content, &role);
        match (&role[..], content) {
            ("assistant", Value::Null) => {}
            ("latest_reminder", c) => {
                mo.insert("content".into(), Value::String(content_text(&c)));
            }
            (_, Value::Null) => {
                mo.insert("content".into(), Value::String(String::new()));
                ctx.normalized("null_content");
            }
            (_, c) => {
                mo.insert("content".into(), c);
            }
        }

        if role == "assistant" {
            // 思考の別名
            if !mo.get("reasoning_content").is_some_and(Value::is_string) {
                let alias = mo.remove("reasoning").or_else(|| mo.remove("reasoning_text")).filter(Value::is_string);
                match alias {
                    Some(a) => {
                        mo.insert("reasoning_content".into(), a);
                        ctx.normalized("reasoning_alias");
                    }
                    None => {
                        mo.remove("reasoning_content");
                    }
                }
            } else {
                mo.remove("reasoning");
                mo.remove("reasoning_text");
            }
            // tool_calls
            let tcs = mo.remove("tool_calls").and_then(|t| match t {
                Value::Array(a) if !a.is_empty() => Some(a),
                _ => None,
            });
            if let Some(tcs) = tcs {
                let mut out = Vec::new();
                let mut seen = HashSet::new();
                for tc in tcs {
                    let Some(t) = tc.as_object() else { continue };
                    let f = t.get("function").and_then(Value::as_object);
                    let name = f.and_then(|f| f.get("name")).or_else(|| t.get("name")).and_then(Value::as_str).unwrap_or("").to_string();
                    let args = f.and_then(|f| f.get("arguments")).or_else(|| t.get("arguments")).cloned().unwrap_or(Value::Null);
                    let args = match args {
                        Value::String(s) => s,
                        Value::Null => "{}".to_string(),
                        other => {
                            ctx.normalized("tool_arguments_object");
                            other.to_string()
                        }
                    };
                    let id = match t.get("id").and_then(Value::as_str) {
                        Some(s) if !s.is_empty() => s.to_string(),
                        _ => {
                            gen_id += 1;
                            ctx.normalized("tool_call_id_generated");
                            format!("call_ds41gw_{}_{gen_id}", ctx.rid)
                        }
                    };
                    if !seen.insert(id.clone()) {
                        continue;
                    }
                    out.push(json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}}));
                }
                if !out.is_empty() {
                    mo.insert("tool_calls".into(), Value::Array(out));
                }
            }
            if !mo.contains_key("tool_calls") && mo.get("content").map_or(true, Value::is_null) {
                mo.insert("content".into(), Value::String(String::new()));
            }
        }
        if role == "tool" {
            if let Some(id) = mo.get("tool_call_id") {
                if !id.is_string() {
                    let s = if id.is_null() { String::new() } else { id.to_string() };
                    mo.insert("tool_call_id".into(), Value::String(s));
                }
            } else {
                mo.insert("tool_call_id".into(), Value::String(String::new()));
            }
        }
    }
    Ok(())
}

/// content のパーツを ds41rt の型 (text / image_url / file) に揃える。
fn normalize_content(ctx: &Ctx, content: Value, role: &str) -> Value {
    let allow_images = matches!(role, "user" | "tool");
    let parts = match content {
        Value::Null => return Value::Null,
        Value::String(s) => return Value::String(s),
        Value::Array(a) => a,
        Value::Object(o) => vec![Value::Object(o)],
        other => return Value::String(other.to_string()),
    };
    let mut out = Vec::new();
    for p in parts {
        let p = match p {
            Value::String(s) => {
                out.push(json!({"type": "text", "text": s}));
                continue;
            }
            Value::Object(o) => o,
            Value::Null => continue,
            other => {
                out.push(json!({"type": "text", "text": other.to_string()}));
                continue;
            }
        };
        let ty = p.get("type").and_then(Value::as_str).unwrap_or(if p.contains_key("text") { "text" } else { "" });
        let image = |url: String, detail: Option<&Value>| {
            let mut iu = json!({"url": url});
            if let Some(d) = detail.and_then(Value::as_str) {
                iu["detail"] = json!(d);
            }
            json!({"type": "image_url", "image_url": iu})
        };
        let part = match ty {
            "text" | "input_text" | "output_text" | "refusal" => {
                if ty != "text" {
                    ctx.normalized(format!("part_{ty}"));
                }
                let t = p.get("text").or_else(|| p.get("refusal")).map(|t| t.as_str().map(str::to_string).unwrap_or_else(|| t.to_string()));
                json!({"type": "text", "text": t.unwrap_or_default()})
            }
            "tool_reference" => {
                ctx.normalized("tool_reference_to_text");
                let name = p.get("name").or_else(|| p.get("tool_name")).and_then(Value::as_str).unwrap_or("");
                json!({"type": "text", "text": format!("Tool loaded: {name}")})
            }
            "image_url" => match p.get("image_url") {
                Some(Value::String(u)) => {
                    ctx.normalized("image_url_string");
                    image(u.clone(), p.get("detail"))
                }
                Some(Value::Object(iu)) => {
                    let url = iu.get("url").and_then(Value::as_str).unwrap_or("").to_string();
                    image(url, iu.get("detail"))
                }
                _ => json!({"type": "text", "text": "[image omitted]"}),
            },
            "input_image" => {
                ctx.normalized("part_input_image");
                let url = match p.get("image_url") {
                    Some(Value::String(u)) => Some(u.clone()),
                    Some(Value::Object(iu)) => iu.get("url").and_then(Value::as_str).map(str::to_string),
                    _ => p.get("file_data").and_then(Value::as_str).map(str::to_string),
                };
                match url {
                    Some(u) => image(u, p.get("detail")),
                    None => json!({"type": "text", "text": "[image omitted]"}),
                }
            }
            "image" => {
                // Anthropic 形式 {"source": {"type":"base64","media_type","data"} | {"type":"url","url"}}
                let src = p.get("source");
                let url = match src.and_then(|s| s.get("type")).and_then(Value::as_str) {
                    Some("base64") => Some(format!(
                        "data:{};base64,{}",
                        src.and_then(|s| s.get("media_type")).and_then(Value::as_str).unwrap_or("image/png"),
                        src.and_then(|s| s.get("data")).and_then(Value::as_str).unwrap_or("")
                    )),
                    Some("url") => src.and_then(|s| s.get("url")).and_then(Value::as_str).map(str::to_string),
                    _ => None,
                };
                match url {
                    Some(u) => image(u, None),
                    None => json!({"type": "text", "text": "[image omitted]"}),
                }
            }
            "file" => Value::Object(p.clone()),
            _ => {
                ctx.warn(format!("unknown_part={ty}"));
                ctx.normalized("unknown_part_to_text");
                json!({"type": "text", "text": Value::Object(p.clone()).to_string()})
            }
        };
        let is_image = matches!(part.get("type").and_then(Value::as_str), Some("image_url" | "file"));
        if is_image && !allow_images {
            ctx.normalized("image_omitted_role");
            out.push(json!({"type": "text", "text": "[image omitted]"}));
        } else {
            out.push(part);
        }
    }
    if !allow_images {
        // system / assistant はテキストだけなので文字列にまとめる
        return Value::String(out.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n\n"));
    }
    Value::Array(out)
}

/// ツール履歴の補修 (§5.3 の 6)。ds41rt の規則 (assistant の tool_calls の直後に N 件の tool) を満たす形にする。
pub fn repair_tool_history(msgs: Vec<Value>) -> (Vec<Value>, bool) {
    let role = |m: &Value| m.get("role").and_then(Value::as_str).unwrap_or("").to_string();
    let mut out = Vec::with_capacity(msgs.len());
    let mut repaired = false;
    let mut i = 0;
    let n = msgs.len();
    while i < n {
        let m = &msgs[i];
        let r = role(m);
        let calls: Vec<(String, bool)> = m
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|tc| {
                        let id = tc.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                        (id.clone(), id.starts_with("call_ds41gw_"))
                    })
                    .collect()
            })
            .unwrap_or_default();
        if r == "assistant" && !calls.is_empty() {
            out.push(m.clone());
            let ids: Vec<String> = calls.iter().map(|c| c.0.clone()).collect();
            let generated = calls.iter().any(|c| c.1);
            let mut results: HashMap<String, Value> = HashMap::new();
            let mut others: Vec<Value> = Vec::new();
            let mut j = i + 1;
            let mut positions_ok = true;
            let mut saw_other = false;
            while j < n && role(&msgs[j]) != "assistant" {
                let mj = &msgs[j];
                if role(mj) == "tool" {
                    let mut tid = mj.get("tool_call_id").and_then(Value::as_str).unwrap_or("").to_string();
                    if !ids.contains(&tid) && (tid.is_empty() || generated) {
                        // id が無い・合わない場合は順番で対応させる
                        if let Some(free) = ids.iter().find(|id| !results.contains_key(*id)) {
                            tid = free.clone();
                            repaired = true;
                        }
                    }
                    if ids.contains(&tid) {
                        if results.contains_key(&tid) {
                            repaired = true; // 重複は最初の 1 件を残す
                        } else {
                            if saw_other {
                                positions_ok = false;
                            }
                            let mut t = mj.clone();
                            t["tool_call_id"] = Value::String(tid.clone());
                            results.insert(tid, t);
                        }
                    } else {
                        saw_other = true;
                        others.push(orphan_to_user(mj));
                        repaired = true;
                    }
                } else {
                    saw_other = true;
                    others.push(mj.clone());
                }
                j += 1;
            }
            if !positions_ok {
                repaired = true;
            }
            for id in &ids {
                match results.remove(id) {
                    Some(t) => out.push(t),
                    None => {
                        repaired = true;
                        out.push(json!({"role": "tool", "tool_call_id": id, "content": "[no result: the tool call was interrupted]"}));
                    }
                }
            }
            out.extend(others);
            i = j;
        } else if r == "tool" {
            out.push(orphan_to_user(m));
            repaired = true;
            i += 1;
        } else {
            out.push(m.clone());
            i += 1;
        }
    }
    (out, repaired)
}

fn orphan_to_user(m: &Value) -> Value {
    let id = m.get("tool_call_id").and_then(Value::as_str).unwrap_or("");
    let head = format!("[tool result {id}]\n");
    match m.get("content") {
        Some(Value::Array(parts)) => {
            let mut p = vec![json!({"type": "text", "text": head.trim_end()})];
            p.extend(parts.iter().cloned());
            json!({"role": "user", "content": p})
        }
        Some(c) => json!({"role": "user", "content": format!("{head}{}", content_text(c))}),
        None => json!({"role": "user", "content": head}),
    }
}

/// tool_choice:"none" 用に、ツールのやり取りを平文にする。
fn flatten_tool_history(messages: &mut Vec<Value>, names: &NameMap) {
    for m in messages.iter_mut() {
        let r = m.get("role").and_then(Value::as_str).unwrap_or("").to_string();
        if r == "assistant" {
            if let Some(tcs) = m.as_object_mut().and_then(|o| o.remove("tool_calls")) {
                let mut text = m.get("content").map(content_text).unwrap_or_default();
                for tc in tcs.as_array().into_iter().flatten() {
                    let name = names.unmap(tc.pointer("/function/name").and_then(Value::as_str).unwrap_or(""));
                    let args = tc.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("");
                    if !text.is_empty() {
                        text.push_str("\n\n");
                    }
                    text.push_str(&format!("[tool call] {name}({args})"));
                }
                m["content"] = Value::String(text);
            }
        } else if r == "tool" {
            let content = m.get("content").cloned().unwrap_or(Value::Null);
            let new = match content {
                Value::Array(parts) => {
                    let mut p = vec![json!({"type": "text", "text": "[tool result]"})];
                    p.extend(parts);
                    Value::Array(p)
                }
                other => Value::String(format!("[tool result] {}", content_text(&other))),
            };
            *m = json!({"role": "user", "content": new});
        }
    }
}

/// 思考を返さないクライアントに、生成時の思考を補う (§10.1)。
fn reinject(ctx: &Ctx, messages: &mut [Value]) -> usize {
    let store = &ctx.app.reinject;
    let mut n = 0;
    for m in messages.iter_mut() {
        if m.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        if m.get("reasoning_content").and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
            continue;
        }
        let first_id = m.pointer("/tool_calls/0/id").and_then(Value::as_str).map(str::to_string);
        let hit = match first_id {
            Some(id) => store.get_tool(&id),
            None => store.get_content(&m.get("content").map(content_text).unwrap_or_default()),
        };
        match hit {
            Some(r) => {
                m["reasoning_content"] = Value::String(r);
                n += 1;
                ctx.app.metrics.inc("ds41gw_reinject_total", &[("result", "hit")]);
            }
            None => ctx.app.metrics.inc("ds41gw_reinject_total", &[("result", "miss")]),
        }
    }
    n
}

/// ツール定義の正規化 (§5.3.8)。
fn normalize_tools(ctx: &Ctx, tools: Option<Value>, names: &mut NameMap) -> Result<Vec<Value>, ApiError> {
    let tools = match tools {
        Some(Value::Array(a)) => a,
        None | Some(Value::Null) => return Ok(vec![]),
        Some(_) => return Err(ApiError::bad("invalid_tools", "tools must be an array").with_param("tools")),
    };
    let mut defs = Vec::new();
    for (i, t) in tools.into_iter().enumerate() {
        let Some(to) = t.as_object() else {
            return Err(ApiError::bad("invalid_tools", format!("tools[{i}] must be an object")).with_param(format!("tools[{i}]")));
        };
        let ty = to.get("type").and_then(Value::as_str).unwrap_or("function");
        if !to.contains_key("type") {
            ctx.normalized("tool_type_added");
        }
        if ty != "function" {
            if ctx.app.cfg.compat.ignore_builtin_tools.iter().any(|x| x == ty) {
                ctx.warn(format!("ignored_tool={ty}"));
                continue;
            }
            let code = if BUILTIN_TOOL_TYPES.contains(&ty) { "unsupported_tool" } else { "invalid_tools" };
            return Err(ApiError::bad(code, format!("tools[{i}]: tool type {ty:?} is not supported")).with_param(format!("tools[{i}].type")));
        }
        let f = match to.get("function").and_then(Value::as_object) {
            Some(f) => f.clone(),
            None => {
                // 平坦形 (Responses 形式)
                ctx.normalized("tool_flat_form");
                let mut f = Map::new();
                for k in ["name", "description", "parameters", "strict"] {
                    if let Some(v) = to.get(k) {
                        f.insert(k.into(), v.clone());
                    }
                }
                f
            }
        };
        let name = f.get("name").and_then(Value::as_str).unwrap_or("").to_string();
        if name.is_empty() {
            return Err(ApiError::bad("invalid_tools", format!("tools[{i}].function.name is required")).with_param(format!("tools[{i}].function.name")));
        }
        defs.push((i, name, f));
    }
    for (_, n, _) in &defs {
        names.reserve(n);
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for (i, name, mut f) in defs {
        let mapped = names.map(&name);
        if !seen.insert(mapped.clone()) {
            return Err(ApiError::bad("invalid_tools", format!("tools[{i}]: duplicate tool name {name:?}")).with_param(format!("tools[{i}].function.name")));
        }
        f.insert("name".into(), Value::String(mapped));
        if let Some(p) = f.remove("parameters") {
            match fix_schema(p) {
                Ok(Some((p, changed))) => {
                    if changed {
                        ctx.normalized("tool_parameters_fixed");
                    }
                    f.insert("parameters".into(), p);
                }
                Ok(None) => {}
                Err(msg) => {
                    return Err(ApiError::bad("invalid_tool_schema", format!("tools[{i}].function.parameters: {msg}"))
                        .with_param(format!("tools[{i}].function.parameters")))
                }
            }
        }
        if f.get("description").is_some_and(Value::is_null) {
            f.remove("description");
        }
        if f.get("strict").is_some_and(Value::is_null) {
            f.remove("strict");
        }
        out.push(json!({"type": "function", "function": Value::Object(f)}));
    }
    Ok(out)
}

/// parameters を `"type":"object"` に揃える。(新しい値, 変えたか)。None は送らない。
pub fn fix_schema(p: Value) -> Result<Option<(Value, bool)>, String> {
    match p {
        Value::Null => Ok(None),
        Value::Object(mut o) => {
            let mut changed = false;
            match o.get("type").cloned() {
                None => {
                    o.insert("type".into(), json!("object"));
                    changed = true;
                }
                Some(Value::String(s)) if s == "object" => {}
                Some(Value::Array(a)) if a.iter().any(|x| x.as_str() == Some("object")) => {
                    o.insert("type".into(), json!("object"));
                    changed = true;
                }
                Some(other) => return Err(format!("must be a JSON Schema of type object (got {other})")),
            }
            if !o.contains_key("properties") && changed {
                o.insert("properties".into(), json!({}));
            }
            Ok(Some((Value::Object(o), changed)))
        }
        _ => Err("must be an object".into()),
    }
}

fn normalize_tool_choice(tc: &mut Value, names: &mut NameMap) -> Result<(), ApiError> {
    match tc {
        Value::String(s) => {
            let l = s.to_ascii_lowercase();
            match l.as_str() {
                "none" | "auto" | "required" => *s = l,
                "any" => *s = "required".into(),
                _ => return Err(ApiError::bad("invalid_tool_choice", format!("invalid tool_choice {s:?}")).with_param("tool_choice")),
            }
        }
        Value::Object(o) => {
            let name = o
                .get("function")
                .and_then(|f| f.get("name"))
                .or_else(|| o.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string);
            match name {
                Some(n) => *tc = json!({"type": "function", "function": {"name": names.map(&n)}}),
                None => {
                    let ty = o.get("type").and_then(Value::as_str).unwrap_or("");
                    match ty {
                        "auto" | "none" | "required" => *tc = json!(ty),
                        "any" => *tc = json!("required"),
                        _ => return Err(ApiError::bad("invalid_tool_choice", format!("unsupported tool_choice type {ty:?}")).with_param("tool_choice")),
                    }
                }
            }
        }
        _ => return Err(ApiError::bad("invalid_tool_choice", "invalid tool_choice").with_param("tool_choice")),
    }
    Ok(())
}

/// 画像が上限を超えたら古いものから `[image omitted]` にする (§6.3)。
pub fn limit_images(ctx: &Ctx, messages: &mut [Value]) -> Result<(), ApiError> {
    let max = ctx.app.cfg.limits.max_images;
    let is_img = |p: &Value| matches!(p.get("type").and_then(Value::as_str), Some("image_url" | "file"));
    let total: usize = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .map(|a| a.iter().filter(|p| is_img(p)).count())
        .sum();
    if total <= max {
        return Ok(());
    }
    if ctx.app.cfg.limits.image_overflow == "reject" {
        return Err(ApiError::bad("too_many_images", format!("at most {max} images are supported (got {total})")));
    }
    let mut drop = total - max;
    for m in messages.iter_mut() {
        if drop == 0 {
            break;
        }
        if let Some(parts) = m.get_mut("content").and_then(Value::as_array_mut) {
            for p in parts.iter_mut() {
                if drop > 0 && is_img(p) {
                    *p = json!({"type": "text", "text": "[image omitted]"});
                    drop -= 1;
                }
            }
        }
    }
    ctx.warn("images_dropped");
    ctx.normalized("images_dropped");
    Ok(())
}

/// 文脈長の事前検査 (§11.6)。大きな要求だけ数える。
fn precheck_context(ctx: &Ctx, p: &Prepared) -> Result<(), ApiError> {
    let cfg = &ctx.app.cfg;
    if !ctx.app.counter.has_tokenizer() {
        return Ok(());
    }
    // 画像の data URL は文字数に数えない (tokenizer では画像 1 枚 = image_estimate。数えると画像だけで発動する)
    let len = text_len(&p.body);
    if len <= cfg.limits.precheck_chars {
        return Ok(());
    }
    let req: Result<deepseek_recipe::openai::ChatCompletionRequest, _> = serde_json::from_value(p.body.clone());
    let Ok(req) = req else { return Ok(()) };
    use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
    let Ok(conv) = req.convert(ConversionOptions::new()) else { return Ok(()) };
    let n = ctx.app.counter.count_conversation(&conv.conversation) as u64;
    let max = ctx.app.model_info.read().unwrap().max_context_tokens;
    if n >= max {
        return Err(ApiError::bad("context_length_exceeded", format!("This model's maximum context length is {max} tokens. However, your messages resulted in {n} tokens.")));
    }
    Ok(())
}

/// 本文の文字列の長さの合計 (data URL を除く)。直列化せずに数える。
fn text_len(v: &Value) -> usize {
    match v {
        Value::String(s) => {
            if s.starts_with("data:") {
                0
            } else {
                s.len()
            }
        }
        Value::Array(a) => a.iter().map(text_len).sum(),
        Value::Object(o) => o.iter().map(|(k, x)| k.len() + text_len(x)).sum(),
        _ => 0,
    }
}

/// n 本の本文 (seed は i 本目に seed+i)。
pub fn split_n(p: &Prepared, ctx: &Ctx) -> Vec<Value> {
    if p.n <= 1 {
        return vec![p.body.clone()];
    }
    if p.body.get("temperature").and_then(Value::as_f64).is_some_and(|t| t < 1e-5) || p.body.get("temperature").is_none() {
        ctx.warn("n_with_greedy_gives_identical_outputs");
    }
    // seed が無いと ds41rt は時刻から作るので、同時に送った n 本が同じ出力になりうる。本ごとに別の seed を入れる
    let base = p.body.get("seed").and_then(Value::as_i64).unwrap_or_else(|| {
        let mut b = [0u8; 4];
        let _ = getrandom::getrandom(&mut b);
        (u32::from_le_bytes(b) >> 1) as i64
    });
    (0..p.n)
        .map(|i| {
            let mut b = p.body.clone();
            b["seed"] = json!(base + i as i64);
            b
        })
        .collect()
}

pub fn source_opts(ctx: &Ctx, p: &Prepared) -> SourceOpts {
    SourceOpts {
        max_tool_calls: ctx.app.cfg.limits.max_tool_calls_per_response,
        drop_tool_calls: p.drop_tool_calls,
        stop: vec![],
        prompt_estimate: 0,
    }
}

// ---------------------------------------------------------------------------
// ハンドラ

pub async fn handle(
    State(app): State<crate::core::AppState>,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let ctx = match begin(&app, "chat", "/v1/chat/completions", &headers, Some(addr), Flavor::OpenAI) {
        Ok(c) => c,
        Err(e) => return e.into_response_with(Flavor::OpenAI),
    };
    let orig = match parse_json(&body) {
        Ok(v) => v,
        Err(e) => return error_response(&ctx, e),
    };
    let p = match prepare(&ctx, orig, &PrepOpts { allow_n: true, ..Default::default() }) {
        Ok(p) => p,
        Err(e) => {
            ctx.log_received(&Value::Null, &body, json!({}));
            return error_response(&ctx, e);
        }
    };
    ctx.log_received(&p.body, &body, json!({}));
    run_chat(ctx, p).await
}

pub async fn run_chat(ctx: Arc<Ctx>, p: Prepared) -> Response {
    let bodies = split_n(&p, &ctx);
    let opts = source_opts(&ctx, &p);
    let p = Arc::new(p);
    if p.stream {
        let fut = connect_all(ctx.clone(), bodies, p.names.clone(), opts).boxed();
        match connect_early(&ctx.app, fut, true).await {
            Err(e) => error_response(&ctx, e),
            Ok(conn) => {
                let s = chat_stream(ctx.clone(), p, conn);
                sse_response(&ctx, s)
            }
        }
    } else {
        let c2 = ctx.clone();
        let fut = async move {
            let futs = bodies.into_iter().map(|b| aggregate_with_retry(c2.clone(), b, p.names.clone(), opts.clone(), None));
            let cols = futures::future::try_join_all(futs).await?;
            Ok((StatusCode::OK, chat_completion_json(&c2, &p, &cols)))
        }
        .boxed();
        nonstream_response(ctx, fut).await
    }
}

/// 非ストリームの `chat.completion` (§5.4.2)。
pub fn chat_completion_json(ctx: &Ctx, p: &Prepared, cols: &[Collected]) -> Value {
    let first = &cols[0];
    let mut choices = Vec::new();
    for (i, c) in cols.iter().enumerate() {
        let mut msg = Map::new();
        msg.insert("role".into(), json!("assistant"));
        let null_content = ctx.app.cfg.compat.null_content_for_tool_calls && !c.tools.is_empty() && c.content.is_empty();
        msg.insert("content".into(), if null_content { Value::Null } else { json!(c.content) });
        if p.thinking.enabled {
            msg.insert("reasoning_content".into(), json!(c.reasoning));
        }
        if !c.tools.is_empty() {
            let tcs: Vec<Value> = c
                .tools
                .iter()
                .map(|t| json!({"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.args}}))
                .collect();
            msg.insert("tool_calls".into(), Value::Array(tcs));
        }
        choices.push(json!({"index": i, "message": Value::Object(msg), "logprobs": null,
            "finish_reason": c.finish.clone().unwrap_or_else(|| "stop".into())}));
    }
    json!({
        "id": first.id, "object": "chat.completion", "created": first.created, "model": p.model_out,
        "system_fingerprint": first.fingerprint, "choices": choices, "usage": merge_usage(cols),
    })
}

/// n 本の usage をまとめる (prompt は 1 本分、completion は合計)。
pub fn merge_usage(cols: &[Collected]) -> Value {
    if cols.len() == 1 {
        return cols[0].usage.clone().unwrap_or(Value::Null);
    }
    let mut u = cols[0].usage.clone().unwrap_or_else(|| json!({}));
    let comp: usize = cols.iter().map(|c| usage_num(c.usage.as_ref(), "completion_tokens")).sum();
    let reas: usize = cols.iter().map(|c| c.reasoning_tokens).sum();
    let prompt = usage_num(cols[0].usage.as_ref(), "prompt_tokens");
    if let Some(o) = u.as_object_mut() {
        o.insert("completion_tokens".into(), json!(comp));
        o.insert("total_tokens".into(), json!(prompt + comp));
        o.insert("completion_tokens_details".into(), json!({"reasoning_tokens": reas}));
    }
    u
}

struct ChunkCtx {
    id: String,
    created: u64,
    fingerprint: Option<String>,
    model: String,
    include_usage: bool,
}

impl ChunkCtx {
    fn chunk(&self, index: usize, delta: Value, finish: Option<&str>) -> Value {
        let mut c = json!({"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
            "system_fingerprint": self.fingerprint,
            "choices": [{"index": index, "delta": delta, "logprobs": null, "finish_reason": finish}]});
        if self.include_usage {
            c["usage"] = Value::Null;
        }
        c
    }
}

/// ストリームの出力 (§5.4.1)。
fn chat_stream(ctx: Arc<Ctx>, p: Arc<Prepared>, conn: Connect) -> BoxStream<'static, Bytes> {
    async_stream::stream! {
        let ka = keepalive_interval(&ctx.app);
        let srcs = match conn {
            Connect::Ready(s) => s,
            Connect::Late(mut fut) => {
                loop {
                    match tokio::time::timeout(ka, &mut fut).await {
                        Ok(Ok(s)) => break s,
                        Ok(Err(e)) => {
                            yield sse_data(&e.body(Flavor::OpenAI));
                            ctx.fail(&e);
                            return;
                        }
                        Err(_) => yield Bytes::from_static(b": keep-alive\n\n"),
                    }
                }
            }
        };
        let n = srcs.len();
        let split = p.include_usage && ctx.app.cfg.compat.split_usage_chunk;
        let mut cc = ChunkCtx { id: String::new(), created: 0, fingerprint: None, model: p.model_out.clone(), include_usage: p.include_usage };
        let mut have_id = false;
        let mut cols: Vec<Option<Collected>> = vec![None; n];
        let mut merged = merge_sources(srcs);
        loop {
            let item = match tokio::time::timeout(ka, merged.next()).await {
                Err(_) => { yield Bytes::from_static(b": keep-alive\n\n"); continue; }
                Ok(None) => break,
                Ok(Some(x)) => x,
            };
            match item {
                (i, Item::Ev(ev)) => {
                    let think = p.thinking.enabled;
                    let out = match ev {
                        Ev::Start { id, created, fingerprint } => {
                            if !have_id {
                                cc.id = id; cc.created = created; cc.fingerprint = fingerprint; have_id = true;
                            }
                            let d = if think { json!({"role": "assistant", "content": null, "reasoning_content": ""}) }
                                    else { json!({"role": "assistant", "content": ""}) };
                            Some(cc.chunk(i, d, None))
                        }
                        Ev::Reasoning(s) => Some(cc.chunk(i, json!({"content": null, "reasoning_content": s}), None)),
                        Ev::Content(s) => {
                            let d = if think { json!({"content": s, "reasoning_content": null}) } else { json!({"content": s}) };
                            Some(cc.chunk(i, d, None))
                        }
                        Ev::ToolStart { index, id, name } => Some(cc.chunk(i, json!({"tool_calls": [{"index": index, "id": id, "type": "function",
                            "function": {"name": name, "arguments": ""}}]}), None)),
                        Ev::ToolArgs { index, args } => Some(cc.chunk(i, json!({"tool_calls": [{"index": index, "function": {"arguments": args}}]}), None)),
                        Ev::Finish { reason, usage, .. } => {
                            let d = if think { json!({"content": "", "reasoning_content": null}) } else { json!({"content": ""}) };
                            let mut c = cc.chunk(i, d, Some(&reason));
                            if n == 1 && !split && !(ctx.app.cfg.compat.drop_usage_without_include && !p.include_usage) {
                                c["usage"] = usage.unwrap_or(Value::Null);
                            }
                            Some(c)
                        }
                    };
                    if let Some(c) = out { yield sse_data(&c); }
                }
                (_, Item::Mid(m)) => {
                    let e = json!({"error": {"message": format!("upstream stream terminated unexpectedly: {m}"), "type": "server_error", "code": "upstream_disconnected"}});
                    yield sse_data(&e);
                    let mut s = ctx.stats.lock().unwrap();
                    s.error = Some(m);
                    s.done = true;
                    return;
                }
                (i, Item::Done(col)) => { cols[i] = Some(*col); }
            }
        }
        // n>1 の合算 usage は choices:[] のチャンクでしか送れない。OpenAI と同じく include_usage のときだけ送る
        // (指定の無いクライアントは choices[0] を読むので、空の choices で落ちる)
        if split || (n > 1 && p.include_usage) {
            let cs: Vec<Collected> = cols.into_iter().flatten().collect();
            if !cs.is_empty() {
                let c = json!({"id": cc.id, "object": "chat.completion.chunk", "created": cc.created, "model": cc.model,
                    "system_fingerprint": cc.fingerprint, "choices": [], "usage": merge_usage(&cs)});
                yield sse_data(&c);
            }
        }
        yield Bytes::from_static(b"data: [DONE]\n\n");
        ctx.mark_done();
    }
    .boxed()
}

#[cfg(test)]
mod precheck_tests {
    use super::text_len;
    use serde_json::json;

    #[test]
    fn data_urls_do_not_count() {
        let big = format!("data:image/png;base64,{}", "A".repeat(5_000_000));
        let v = json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "abc"}, {"type": "image_url", "image_url": {"url": big}}]}]});
        assert!(text_len(&v) < 100);
        let v = json!({"messages": [{"role": "user", "content": "x".repeat(1000)}]});
        assert!(text_len(&v) >= 1000);
    }
}
