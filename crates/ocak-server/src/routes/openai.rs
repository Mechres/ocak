//! OpenAI-compatible chat / text completions and embeddings.
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use ocak_engine::{GenRequest, SamplerParams, ThinkingCfg};
use ocak_tok::template::render;
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;
use crate::*;

#[derive(Debug, Deserialize)]
pub(crate) struct ChatReq {
    pub(crate) model: String,
    pub(crate) messages: Vec<Value>,
    #[serde(default)]
    pub(crate) stream: bool,
    pub(crate) max_tokens: Option<usize>,
    /// OpenAI's newer name for `max_tokens` (takes precedence).
    pub(crate) max_completion_tokens: Option<usize>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<usize>,
    pub(crate) min_p: Option<f32>,
    pub(crate) repeat_penalty: Option<f32>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    /// A single string or a list of strings.
    #[serde(default, deserialize_with = "de_stop")]
    pub(crate) stop: Vec<String>,
    pub(crate) stream_options: Option<Value>,
    pub(crate) reasoning_max_tokens: Option<usize>,
    pub(crate) chat_template_kwargs: Option<Value>,
    pub(crate) response_format: Option<Value>,
    #[serde(default)]
    pub(crate) tools: Vec<Value>,
    #[allow(dead_code)]
    pub(crate) tool_choice: Option<Value>,
}

pub(crate) fn def_max_tok() -> usize { 256 }

pub(crate) fn de_stop<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Stop {
        One(String),
        Many(Vec<String>),
        None(()),
    }
    Ok(match Stop::deserialize(d)? {
        Stop::One(s) => vec![s],
        Stop::Many(v) => v,
        Stop::None(()) => Vec::new(),
    })
}

pub(crate) fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

pub(crate) fn build_sampler(st: &AppState, model_id: &str, req: &ChatReq) -> SamplerParams {
    let p = st.presets.resolve(model_id);
    let f = |k: &str, d: f32| p.get(k).and_then(|s| s.parse().ok()).unwrap_or(d);
    let u = |k: &str, d: usize| p.get(k).and_then(|s| s.parse().ok()).unwrap_or(d);
    SamplerParams {
        temperature: req.temperature.unwrap_or_else(|| f("temp", 0.7)),
        top_p: req.top_p.unwrap_or_else(|| f("top-p", 0.95)),
        top_k: req.top_k.unwrap_or_else(|| u("top-k", 40)),
        min_p: req.min_p.unwrap_or_else(|| f("min-p", 0.0)),
        repeat_penalty: req.repeat_penalty.unwrap_or_else(|| f("repeat-penalty", 1.0)),
        presence_penalty: req.presence_penalty.unwrap_or_else(|| f("presence-penalty", 0.0)),
        frequency_penalty: req.frequency_penalty.unwrap_or_else(|| f("frequency-penalty", 0.0)),
        ..Default::default()
    }
}

pub(crate) async fn chat_completions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatReq>,
) -> Response {
    st.requests_total.fetch_add(1, Ordering::Relaxed);
    if headers.get("x-ocak-tier").and_then(|v| v.to_str().ok()) == Some("fleet")
        || req.model.starts_with("fleet:")
    {
        let body = serde_json::to_vec(&json!({
            "model": req.model.trim_start_matches("fleet:"),
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "stream": false
        }))
        .unwrap();
        return match st.mergen.forward_json("/v1/chat/completions", &body).await {
            Ok((code, b)) => (StatusCode::from_u16(code).unwrap_or(StatusCode::OK), b).into_response(),
            Err(e) => (StatusCode::BAD_GATEWAY, Json(json!({ "error": e }))).into_response(),
        };
    }

    let Some(entry) = st.find_model(&req.model).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": format!("unknown model {}", req.model) }))).into_response();
    };

    if engine_kind_for(&entry) != "native-vulkan" {
        return match st.ensure_bridge(&entry).await {
            Ok(addr) => {
                let body = serde_json::to_vec(&json!({
                    "model": entry.id,
                    "messages": req.messages,
                    "max_tokens": req.max_tokens,
                    "stream": false
                }))
                .unwrap();
                match ocak_federation::bridges::http_request(&addr, "POST", "/v1/chat/completions", "application/json", &body).await {
                    Ok((code, b)) => (StatusCode::from_u16(code).unwrap_or(StatusCode::OK), b).into_response(),
                    Err(e) => (StatusCode::BAD_GATEWAY, Json(json!({ "error": e }))).into_response(),
                }
            }
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
        };
    }

    let (fam, multi_slot) = match st.ensure_native(&entry).await {
        Ok(m) => m,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
    };

    let json_mode = req
        .response_format
        .as_ref()
        .and_then(|v| v.get("type"))
        .and_then(|v| v.as_str())
        .map(|t| t == "json_object" || t == "json_schema")
        .unwrap_or(false);
    let sampler = build_sampler(&st, &entry.id, &req);
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = unix_now();
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|o| o.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model_id = entry.id.clone();
    let tool_defs = tools::extract_openai_tools(&req.tools);
    let has_tools = !tool_defs.is_empty();
    let chat_msgs = tools::openai_to_chat(&req.messages, &tool_defs);

    let enable_thinking = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|v| v.get("enable_thinking"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let mut prompt = render(fam, &chat_msgs, true, enable_thinking);
    // An explicit `enable_thinking: true` on a template that only thinks when asked (Qwen3.5):
    // open the reasoning block as the model's own template would.
    let explicit_think = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|v| v.get("enable_thinking"))
        .and_then(|v| v.as_bool())
        == Some(true);
    let think_opened = explicit_think && fam == ocak_tok::TemplateFamily::ChatMl && st.template_opens_think(&model_id);
    if think_opened {
        prompt.push_str("<think>\n");
    }
    let greq = GenRequest {
        prompt,
        parse_special: true,
        max_tokens: req.max_completion_tokens.or(req.max_tokens).unwrap_or_else(def_max_tok),
        sampler,
        stop: req.stop,
        json_mode,
        thinking: req.reasoning_max_tokens.map(|m| ThinkingCfg { max_tokens: m }),
    };

    let started_in_think = (fam == ocak_tok::TemplateFamily::Xing && enable_thinking) || think_opened;
    if req.stream {
        let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(1024);
        let (tok_tx, mut tok_rx) = mpsc::channel::<String>(1024);
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        st.batcher.submit(
            Arc::clone(&st),
            batch::BatchJob {
                model_id: model_id.clone(),
                greq,
                token_tx: if has_tools { None } else { Some(tok_tx) },
                done_tx,
            },
        );
        tokio::spawn(async move {
            if !has_tools {
                let mut splitter = reasoning::StreamReasoningSplitter::new(started_in_think);
                while let Some(piece) = tok_rx.recv().await {
                    let (r_delta, c_delta) = splitter.push(&piece);
                    if !r_delta.is_empty() {
                        let chunk = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{ "index": 0, "delta": { "reasoning_content": r_delta }, "finish_reason": null }]
                        });
                        if tx.send(Ok(Event::default().data(chunk.to_string()))).await.is_err() {
                            return;
                        }
                    }
                    if !c_delta.is_empty() {
                        let chunk = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{ "index": 0, "delta": { "content": c_delta }, "finish_reason": null }]
                        });
                        if tx.send(Ok(Event::default().data(chunk.to_string()))).await.is_err() {
                            return;
                        }
                    }
                }
                let (r_tail, c_tail) = splitter.flush();
                if !r_tail.is_empty() {
                    let chunk = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                            "created": created,
                        "model": model_id,
                        "choices": [{ "index": 0, "delta": { "reasoning_content": r_tail }, "finish_reason": null }]
                    });
                    let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                }
                if !c_tail.is_empty() {
                    let chunk = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                            "created": created,
                        "model": model_id,
                        "choices": [{ "index": 0, "delta": { "content": c_tail }, "finish_reason": null }]
                    });
                    let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                }
            }
            if let Ok(Ok(gr)) = done_rx.await {
                let mut finish_reason = json!(gr.finish_reason);
                if has_tools {
                    let (r_opt, answer) = reasoning::split_reasoning(&gr.text, started_in_think);
                    if let Some(r_txt) = r_opt {
                        let rc = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{ "index": 0, "delta": { "reasoning_content": r_txt }, "finish_reason": null }]
                        });
                        let _ = tx.send(Ok(Event::default().data(rc.to_string()))).await;
                    }
                    let (clean, calls) = tools::parse_tool_calls(&answer, "call_");
                    if !clean.is_empty() {
                        let c = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{ "index": 0, "delta": { "content": clean }, "finish_reason": null }]
                        });
                        let _ = tx.send(Ok(Event::default().data(c.to_string()))).await;
                    }
                    if !calls.is_empty() {
                        let tc_json: Vec<Value> = calls
                            .iter()
                            .enumerate()
                            .map(|(idx, tc)| json!({
                                "index": idx,
                                "id": tc.id,
                                "type": "function",
                                "function": { "name": tc.name, "arguments": tc.arguments.to_string() }
                            }))
                            .collect();
                        let tc_chunk = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{ "index": 0, "delta": { "tool_calls": tc_json }, "finish_reason": null }]
                        });
                        let _ = tx.send(Ok(Event::default().data(tc_chunk.to_string()))).await;
                        finish_reason = json!("tool_calls");
                    }
                }
                let done_chunk = json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                            "created": created,
                    "model": model_id,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason }],
                    "timings": gr.timings
                });
                let _ = tx.send(Ok(Event::default().data(done_chunk.to_string()))).await;
                if include_usage {
                    // stream_options.include_usage: a final chunk with empty choices and usage.
                    let usage_chunk = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": gr.timings.prompt_n,
                            "completion_tokens": gr.timings.predicted_n,
                            "total_tokens": gr.timings.prompt_n + gr.timings.predicted_n
                        }
                    });
                    let _ = tx.send(Ok(Event::default().data(usage_chunk.to_string()))).await;
                }
                let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
            } else {
                let err_chunk = json!({ "error": { "message": "generation failed or model swapped" } });
                let _ = tx.send(Ok(Event::default().data(err_chunk.to_string()))).await;
                let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
            }
        });
        return Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default()).into_response();
    }

    let _ = multi_slot;
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    st.batcher.submit(
        Arc::clone(&st),
        batch::BatchJob {
            model_id: model_id.clone(),
            greq,
            token_tx: None,
            done_tx,
        },
    );
    let gr = match done_rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    };
    let (reasoning_opt, answer_text) = reasoning::split_reasoning(&gr.text, started_in_think);
    let (mut msg_obj, finish_reason) = if has_tools {
        let (clean, calls) = tools::parse_tool_calls(&answer_text, "call_");
        if !calls.is_empty() {
            let tc_json: Vec<Value> = calls
                .iter()
                .map(|tc| json!({
                    "id": tc.id,
                    "type": "function",
                    "function": { "name": tc.name, "arguments": tc.arguments.to_string() }
                }))
                .collect();
            let content_val = if clean.is_empty() { Value::Null } else { json!(clean) };
            (json!({ "role": "assistant", "content": content_val, "tool_calls": tc_json }), json!("tool_calls"))
        } else {
            (json!({ "role": "assistant", "content": answer_text }), json!(gr.finish_reason))
        }
    } else {
        (json!({ "role": "assistant", "content": answer_text }), json!(gr.finish_reason))
    };
    if let Some(r_txt) = reasoning_opt {
        if let Some(obj) = msg_obj.as_object_mut() {
            obj.insert("reasoning_content".into(), json!(r_txt));
        }
    }
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": msg_obj,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": gr.timings.prompt_n,
            "completion_tokens": gr.timings.predicted_n,
            "total_tokens": gr.timings.prompt_n + gr.timings.predicted_n
        },
        "timings": gr.timings
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub(crate) struct CompReq {
    pub(crate) model: String,
    pub(crate) prompt: String,
    #[serde(default = "def_max_tok")]
    pub(crate) max_tokens: usize,
    pub(crate) temperature: Option<f32>,
}

pub(crate) async fn completions(State(st): State<Arc<AppState>>, Json(req): Json<CompReq>) -> Response {
    st.requests_total.fetch_add(1, Ordering::Relaxed);
    let Some(entry) = st.find_model(&req.model).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "model not found" }))).into_response();
    };
    if let Err(e) = st.ensure_native(&entry).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response();
    }
    let greq = GenRequest {
        prompt: req.prompt,
        parse_special: false,
        max_tokens: req.max_tokens,
        sampler: SamplerParams { temperature: req.temperature.unwrap_or(0.0), ..Default::default() },
        stop: Vec::new(),
        json_mode: false,
        thinking: None,
    };
    let st_c = Arc::clone(&st);
    let model_id = entry.id.clone();
    let res = tokio::task::spawn_blocking(move || {
        let mut guard = st_c.native.blocking_lock();
        let Some(active) = guard.as_mut() else {
            return Err("native engine unloaded before request ran".to_string());
        };
        if active.id != model_id {
            return Err(format!("model mismatch: requested '{model_id}' but '{}' is loaded", active.id));
        }
        let out = active.engine.generate(&greq, |_, _| true).map_err(|e| e.to_string());
        st_c.governor.blocking_lock().touch(&model_id, Instant::now());
        out
    })
    .await;
    match res {
        Ok(Ok(gr)) => {
            st.tokens_generated_total.fetch_add(gr.timings.predicted_n as u64, Ordering::Relaxed);
            st.kv_radix_hits_total.fetch_add(gr.timings.cache_n as u64, Ordering::Relaxed);
            Json(json!({
                "id": format!("cmpl-{}", Uuid::new_v4()),
                "object": "text_completion",
                "model": entry.id,
                "choices": [{ "index": 0, "text": gr.text, "finish_reason": gr.finish_reason }],
                "timings": gr.timings
            }))
            .into_response()
        }
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbReq {
    pub(crate) model: String,
    pub(crate) input: Value,
}

pub(crate) async fn embeddings(State(st): State<Arc<AppState>>, Json(req): Json<EmbReq>) -> Response {
    st.requests_total.fetch_add(1, Ordering::Relaxed);
    let Some(entry) = st.find_model(&req.model).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "model not found" }))).into_response();
    };
    if let Err(e) = st.ensure_native(&entry).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response();
    }
    let inputs: Vec<String> = match req.input {
        Value::String(s) => vec![s],
        Value::Array(a) => a.into_iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect(),
        _ => Vec::new(),
    };
    let st_c = Arc::clone(&st);
    let model_id = entry.id.clone();
    let res = tokio::task::spawn_blocking(move || {
        let mut guard = st_c.native.blocking_lock();
        let Some(active) = guard.as_mut() else {
            return Err("native engine unloaded before request ran".to_string());
        };
        if active.id != model_id {
            return Err(format!("model mismatch: requested '{model_id}' but '{}' is loaded", active.id));
        }
        let mut data = Vec::with_capacity(inputs.len());
        for (idx, txt) in inputs.iter().enumerate() {
            let vec = active.engine.embed_text(txt).map_err(|e| e.to_string())?;
            data.push(json!({ "object": "embedding", "index": idx, "embedding": vec }));
        }
        st_c.governor.blocking_lock().touch(&model_id, Instant::now());
        Ok(data)
    })
    .await;
    match res {
        Ok(Ok(data)) => Json(json!({ "object": "list", "model": entry.id, "data": data })).into_response(),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}
