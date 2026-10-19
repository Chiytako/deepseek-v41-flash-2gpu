//! `/v1/completions`: chat に包む擬似実装 (r7_proxy_spec.md §8)。
//! ds41rt に生のプロンプトを流す経路は無いので、prompt を user メッセージ 1 件にする。
//! 思考は既定で off。厳密な補完 (FIM、文の途中からの続き) にはならない。

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use serde_json::{json, Value};

use crate::chat::{merge_usage, prepare, source_opts, PrepOpts, Prepared};
use crate::core::{
    aggregate_with_retry, begin, connect_all, connect_early, error_response, keepalive_interval, merge_sources, nonstream_response,
    parse_json, sse_data, sse_response, AppState, Collected, Connect, Ctx, Item,
};
use crate::error::{ApiError, Flavor};
use crate::upstream::Ev;
use crate::util::{hex32, is_truthy};

pub async fn handle(
    State(app): State<AppState>,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let ctx = match begin(&app, "completions", "/v1/completions", &headers, Some(addr), Flavor::OpenAI) {
        Ok(c) => c,
        Err(e) => return e.into_response_with(Flavor::OpenAI),
    };
    if !app.cfg.completions.enabled {
        return error_response(&ctx, ApiError::new(crate::error::Kind::NotFound, "not_found", "/v1/completions is disabled"));
    }
    ctx.warn("completions_emulated_via_chat");
    let orig = match parse_json(&body) {
        Ok(v) => v,
        Err(e) => return error_response(&ctx, e),
    };
    let r = (|| {
        let o = orig.as_object().unwrap();
        if is_truthy(o.get("echo")) {
            return Err(ApiError::unsupported("echo"));
        }
        if o.get("suffix").is_some_and(|v| !v.is_null() && v != "") {
            return Err(ApiError::unsupported("suffix"));
        }
        if o.get("logprobs").is_some_and(|v| !v.is_null() && v != &json!(0) && v != &json!(false)) {
            return Err(ApiError::unsupported("logprobs"));
        }
        if o.get("best_of").and_then(Value::as_u64).unwrap_or(1) > 1 {
            return Err(ApiError::unsupported("best_of"));
        }
        let prompts: Vec<String> = match o.get("prompt") {
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) if a.iter().all(Value::is_string) && !a.is_empty() => {
                a.iter().map(|s| s.as_str().unwrap().to_string()).collect()
            }
            Some(Value::Array(_)) => return Err(ApiError::bad("unsupported_parameter", "token id prompts are not supported").with_param("prompt")),
            None | Some(Value::Null) => vec![String::new()],
            Some(_) => return Err(ApiError::bad("invalid_prompt", "prompt must be a string or an array of strings").with_param("prompt")),
        };
        let mut chat = orig.clone();
        let co = chat.as_object_mut().unwrap();
        for k in ["prompt", "echo", "suffix", "best_of", "logprobs"] {
            co.remove(k);
        }
        co.insert("messages".into(), json!([{"role": "user", "content": prompts[0]}]));
        let opts = PrepOpts {
            default_thinking: Some(app.cfg.completions.thinking),
            default_max_tokens: Some(app.cfg.completions.default_max_tokens),
            allow_n: true,
            names: None,
        };
        let p = prepare(&ctx, chat, &opts)?;
        Ok::<_, ApiError>((p, prompts))
    })();
    let (p, prompts) = match r {
        Ok(x) => x,
        Err(e) => {
            ctx.log_received(&Value::Null, &body, json!({}));
            return error_response(&ctx, e);
        }
    };
    ctx.log_received(&p.body, &body, json!({}));
    // プロンプト × n 本
    let per = crate::chat::split_n(&p, &ctx);
    let mut bodies = Vec::new();
    for pr in &prompts {
        for b in &per {
            let mut b = b.clone();
            b["messages"][0]["content"] = json!(pr);
            bodies.push(b);
        }
    }
    let id = format!("cmpl-{}", hex32());
    let p = Arc::new(p);
    let opts = source_opts(&ctx, &p);
    if p.stream {
        let fut = connect_all(ctx.clone(), bodies, p.names.clone(), opts).boxed();
        match connect_early(&app, fut, true).await {
            Err(e) => error_response(&ctx, e),
            Ok(conn) => sse_response(&ctx, stream(ctx.clone(), p, conn, id)),
        }
    } else {
        let c2 = ctx.clone();
        let fut = async move {
            let futs = bodies.into_iter().map(|b| aggregate_with_retry(c2.clone(), b, p.names.clone(), opts.clone(), None));
            let cols = futures::future::try_join_all(futs).await?;
            Ok((StatusCode::OK, completion_json(&id, &p, &cols)))
        }
        .boxed();
        nonstream_response(ctx, fut).await
    }
}

fn completion_json(id: &str, p: &Prepared, cols: &[Collected]) -> Value {
    let choices: Vec<Value> = cols
        .iter()
        .enumerate()
        .map(|(i, c)| json!({"index": i, "text": c.content, "logprobs": null, "finish_reason": c.finish.clone().unwrap_or_else(|| "stop".into())}))
        .collect();
    json!({"id": id, "object": "text_completion", "created": cols[0].created, "model": p.model_out,
           "system_fingerprint": cols[0].fingerprint, "choices": choices, "usage": merge_usage(cols)})
}

fn stream(ctx: Arc<Ctx>, p: Arc<Prepared>, conn: Connect, id: String) -> BoxStream<'static, Bytes> {
    async_stream::stream! {
        let ka = keepalive_interval(&ctx.app);
        let srcs = match conn {
            Connect::Ready(s) => s,
            Connect::Late(mut fut) => loop {
                match tokio::time::timeout(ka, &mut fut).await {
                    Ok(Ok(s)) => break s,
                    Ok(Err(e)) => { yield sse_data(&e.body(Flavor::OpenAI)); ctx.fail(&e); return; }
                    Err(_) => yield Bytes::from_static(b": keep-alive\n\n"),
                }
            },
        };
        let n = srcs.len();
        let mut created = crate::core::created_now();
        let mut cols: Vec<Option<Collected>> = vec![None; n];
        let mut merged = merge_sources(srcs);
        let chunk = |i: usize, text: &str, fin: Option<&str>, created: u64| json!({"id": id, "object": "text_completion", "created": created,
            "model": p.model_out, "choices": [{"index": i, "text": text, "logprobs": null, "finish_reason": fin}]});
        loop {
            let item = match tokio::time::timeout(ka, merged.next()).await {
                Err(_) => { yield Bytes::from_static(b": keep-alive\n\n"); continue; }
                Ok(None) => break,
                Ok(Some(x)) => x,
            };
            match item {
                (_, Item::Ev(Ev::Start { created: c, .. })) => created = c,
                (i, Item::Ev(Ev::Content(s))) => yield sse_data(&chunk(i, &s, None, created)),
                (i, Item::Ev(Ev::Finish { reason, .. })) => yield sse_data(&chunk(i, "", Some(&reason), created)),
                (_, Item::Ev(_)) => {}
                (_, Item::Mid(m)) => {
                    yield sse_data(&json!({"error": {"message": format!("upstream stream terminated unexpectedly: {m}"), "type": "server_error", "code": "upstream_disconnected"}}));
                    let mut s = ctx.stats.lock().unwrap();
                    s.error = Some(m);
                    s.done = true;
                    return;
                }
                (i, Item::Done(c)) => cols[i] = Some(*c),
            }
        }
        if p.include_usage {
            let cs: Vec<Collected> = cols.into_iter().flatten().collect();
            if !cs.is_empty() {
                yield sse_data(&json!({"id": id, "object": "text_completion", "created": created, "model": p.model_out, "choices": [], "usage": merge_usage(&cs)}));
            }
        }
        yield Bytes::from_static(b"data: [DONE]\n\n");
        ctx.mark_done();
    }
    .boxed()
}
