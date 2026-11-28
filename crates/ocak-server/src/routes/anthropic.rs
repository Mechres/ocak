//! Anthropic-compatible messages and token counting.
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use ocak_engine::{FinishReason, GenRequest, SamplerParams, ThinkingCfg};
use ocak_tok::template::{detect_family, render, ChatMessage};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;
use crate::*;

/// Anthropic Messages API (`/v1/messages`) with streaming SSE + tool_use for Claude Code compatibility.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicReq {
    pub(crate) model: String,
    #[serde(default = "def_max_tok")]
    pub(crate) max_tokens: usize,
    pub(crate) system: Option<Value>,
    pub(crate) messages: Vec<Value>,
    #[serde(default)]
    pub(crate) stream: bool,
    pub(crate) temperature: Option<f32>,
    #[serde(default)]
    pub(crate) tools: Vec<Value>,
    #[allow(dead_code)]
    pub(crate) tool_choice: Option<Value>,
    pub(crate) thinking: Option<Value>,
}

pub(crate) fn anthropic_to_chat(req: &AnthropicReq) -> Vec<ChatMessage> {
    let tool_defs = tools::extract_anthropic_tools(&req.tools);
    tools::anthropic_messages_to_chat(req.system.as_ref(), &req.messages, &tool_defs)
}

pub(crate) async fn anthropic_messages(State(st): State<Arc<AppState>>, Json(req): Json<AnthropicReq>) -> Response {
    st.requests_total.fetch_add(1, Ordering::Relaxed);
    let Some(entry) = st.find_model(&req.model).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "type": "error", "error": { "type": "not_found_error", "message": "model not found" } }))).into_response();
    };
    if let Err(e) = st.ensure_native(&entry).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "type": "error", "error": { "type": "api_error", "message": e } }))).into_response();
    }
    let has_tools = !req.tools.is_empty();
    // `thinking: {"type": "enabled", "budget_tokens": N}` caps the reasoning block.
    let think_budget = req
        .thinking
        .as_ref()
        .and_then(|v| v.get("budget_tokens"))
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let enable_thinking = req
        .thinking
        .as_ref()
        .and_then(|v| v.get("type"))
        .and_then(|v| v.as_str())
        .map(|t| t != "disabled")
        .unwrap_or(true);
    let chat_msgs = anthropic_to_chat(&req);
    let msg_id = format!("msg_{}", Uuid::new_v4().simple());
    let model_id = entry.id.clone();

    if req.stream {
        let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(256);
        let st_c = Arc::clone(&st);
        tokio::task::spawn_blocking(move || {
            let message_start = |input_tokens: usize| {
                let start = json!({
                    "type": "message_start",
                    "message": { "id": msg_id, "type": "message", "role": "assistant", "content": [], "model": model_id, "stop_reason": null, "usage": { "input_tokens": input_tokens, "output_tokens": 0 } }
                });
                Ok(Event::default().event("message_start").data(start.to_string()))
            };

            let mut guard = st_c.native.blocking_lock();
            match guard.as_mut() {
                Some(active) if active.id == model_id => {
                    let fam = detect_family(active.engine.tokenizer.chat_template());
                    let started_in_think = fam == ocak_tok::TemplateFamily::Xing && enable_thinking;
                    let prompt = render(fam, &chat_msgs, true, enable_thinking);
                    let _ = tx.blocking_send(message_start(active.engine.tokenizer.encode(&prompt, true, true).len()));
                    let greq = GenRequest {
                        prompt,
                        parse_special: true,
                        max_tokens: req.max_tokens,
                        sampler: SamplerParams { temperature: req.temperature.unwrap_or(0.7), ..Default::default() },
                        stop: Vec::new(),
                        json_mode: false,
                        thinking: think_budget.map(|m| ThinkingCfg { max_tokens: m }),
                    };
                    let tx_c = tx.clone();
                    let mut splitter = reasoning::StreamReasoningSplitter::new(started_in_think);
                    let mut cur_block_idx = 0usize;
                    let mut cur_block_kind: Option<&'static str> = None;
                    // Never block the engine on a slow client: a full queue cancels the stream
                    // (same policy as the OpenAI path), a closed one stops generation.
                    let live = |ev: Event| match tx_c.try_send(Ok(ev)) {
                        Ok(()) => true,
                        Err(mpsc::error::TrySendError::Full(_)) => {
                            tracing::warn!(model = %model_id, "SSE client channel full; cancelling slow stream");
                            false
                        }
                        Err(mpsc::error::TrySendError::Closed(_)) => false,
                    };
                    let gen_res = active.engine.generate(&greq, |_, piece| {
                        if piece.is_empty() || has_tools { return !tx_c.is_closed(); }
                        let (r_delta, c_delta) = splitter.push(piece);
                        if !r_delta.is_empty() {
                            if cur_block_kind != Some("thinking") {
                                if cur_block_kind.is_some() {
                                    if !live(Event::default().event("content_block_stop").data(json!({ "type": "content_block_stop", "index": cur_block_idx }).to_string())) { return false; }
                                    cur_block_idx += 1;
                                }
                                let blk = json!({ "type": "content_block_start", "index": cur_block_idx, "content_block": { "type": "thinking", "thinking": "" } });
                                if !live(Event::default().event("content_block_start").data(blk.to_string())) { return false; }
                                cur_block_kind = Some("thinking");
                            }
                            let d = json!({ "type": "content_block_delta", "index": cur_block_idx, "delta": { "type": "thinking_delta", "thinking": r_delta } });
                            if !live(Event::default().event("content_block_delta").data(d.to_string())) { return false; }
                        }
                        if !c_delta.is_empty() {
                            if cur_block_kind != Some("text") {
                                if cur_block_kind.is_some() {
                                    if !live(Event::default().event("content_block_stop").data(json!({ "type": "content_block_stop", "index": cur_block_idx }).to_string())) { return false; }
                                    cur_block_idx += 1;
                                }
                                let blk = json!({ "type": "content_block_start", "index": cur_block_idx, "content_block": { "type": "text", "text": "" } });
                                if !live(Event::default().event("content_block_start").data(blk.to_string())) { return false; }
                                cur_block_kind = Some("text");
                            }
                            let d = json!({ "type": "content_block_delta", "index": cur_block_idx, "delta": { "type": "text_delta", "text": c_delta } });
                            if !live(Event::default().event("content_block_delta").data(d.to_string())) { return false; }
                        }
                        true
                    });
                    // Final events are sent without holding the engine.
                    drop(guard);
                    match gen_res {
                        Ok(gr) => {
                            st_c.tokens_generated_total.fetch_add(gr.timings.predicted_n as u64, Ordering::Relaxed);
                            st_c.kv_radix_hits_total.fetch_add(gr.timings.cache_n as u64, Ordering::Relaxed);
                            let mut stop_reason = match gr.finish_reason {
                                FinishReason::Length => "max_tokens",
                                _ => "end_turn",
                            };
                            if has_tools {
                                let (r_opt, answer) = reasoning::split_reasoning(&gr.text, started_in_think);
                                let (clean, calls) = tools::parse_tool_calls(&answer, "toolu_");
                                let mut idx = 0usize;
                                if let Some(r_txt) = r_opt {
                                    let blk = json!({ "type": "content_block_start", "index": idx, "content_block": { "type": "thinking", "thinking": "" } });
                                    let _ = tx.blocking_send(Ok(Event::default().event("content_block_start").data(blk.to_string())));
                                    let d = json!({ "type": "content_block_delta", "index": idx, "delta": { "type": "thinking_delta", "thinking": r_txt } });
                                    let _ = tx.blocking_send(Ok(Event::default().event("content_block_delta").data(d.to_string())));
                                    let _ = tx.blocking_send(Ok(Event::default().event("content_block_stop").data(json!({ "type": "content_block_stop", "index": idx }).to_string())));
                                    idx += 1;
                                }
                                if !clean.is_empty() || calls.is_empty() {
                                    let blk = json!({ "type": "content_block_start", "index": idx, "content_block": { "type": "text", "text": "" } });
                                    let _ = tx.blocking_send(Ok(Event::default().event("content_block_start").data(blk.to_string())));
                                    let txt = if calls.is_empty() { &answer } else { &clean };
                                    if !txt.is_empty() {
                                        let d = json!({ "type": "content_block_delta", "index": idx, "delta": { "type": "text_delta", "text": txt } });
                                        let _ = tx.blocking_send(Ok(Event::default().event("content_block_delta").data(d.to_string())));
                                    }
                                    let _ = tx.blocking_send(Ok(Event::default().event("content_block_stop").data(json!({ "type": "content_block_stop", "index": idx }).to_string())));
                                    idx += 1;
                                }
                                if !calls.is_empty() {
                                    stop_reason = "tool_use";
                                    for tc in &calls {
                                        let blk = json!({
                                            "type": "content_block_start",
                                            "index": idx,
                                            "content_block": { "type": "tool_use", "id": tc.id, "name": tc.name, "input": {} }
                                        });
                                        let _ = tx.blocking_send(Ok(Event::default().event("content_block_start").data(blk.to_string())));
                                        let d = json!({
                                            "type": "content_block_delta",
                                            "index": idx,
                                            "delta": { "type": "input_json_delta", "partial_json": tc.arguments.to_string() }
                                        });
                                        let _ = tx.blocking_send(Ok(Event::default().event("content_block_delta").data(d.to_string())));
                                        let _ = tx.blocking_send(Ok(Event::default().event("content_block_stop").data(json!({ "type": "content_block_stop", "index": idx }).to_string())));
                                        idx += 1;
                                    }
                                }
                            } else {
                                let (r_tail, c_tail) = splitter.flush();
                                if !r_tail.is_empty() {
                                    if cur_block_kind != Some("thinking") {
                                        if cur_block_kind.is_some() {
                                            let _ = tx.blocking_send(Ok(Event::default().event("content_block_stop").data(json!({ "type": "content_block_stop", "index": cur_block_idx }).to_string())));
                                            cur_block_idx += 1;
                                        }
                                        let blk = json!({ "type": "content_block_start", "index": cur_block_idx, "content_block": { "type": "thinking", "thinking": "" } });
                                        let _ = tx.blocking_send(Ok(Event::default().event("content_block_start").data(blk.to_string())));
                                        cur_block_kind = Some("thinking");
                                    }
                                    let d = json!({ "type": "content_block_delta", "index": cur_block_idx, "delta": { "type": "thinking_delta", "thinking": r_tail } });
                                    let _ = tx.blocking_send(Ok(Event::default().event("content_block_delta").data(d.to_string())));
                                }
                                if !c_tail.is_empty() {
                                    if cur_block_kind != Some("text") {
                                        if cur_block_kind.is_some() {
                                            let _ = tx.blocking_send(Ok(Event::default().event("content_block_stop").data(json!({ "type": "content_block_stop", "index": cur_block_idx }).to_string())));
                                            cur_block_idx += 1;
                                        }
                                        let blk = json!({ "type": "content_block_start", "index": cur_block_idx, "content_block": { "type": "text", "text": "" } });
                                        let _ = tx.blocking_send(Ok(Event::default().event("content_block_start").data(blk.to_string())));
                                        cur_block_kind = Some("text");
                                    }
                                    let d = json!({ "type": "content_block_delta", "index": cur_block_idx, "delta": { "type": "text_delta", "text": c_tail } });
                                    let _ = tx.blocking_send(Ok(Event::default().event("content_block_delta").data(d.to_string())));
                                }
                                if cur_block_kind.is_some() {
                                    let _ = tx.blocking_send(Ok(Event::default().event("content_block_stop").data(json!({ "type": "content_block_stop", "index": cur_block_idx }).to_string())));
                                }
                            }
                            let md = json!({ "type": "message_delta", "delta": { "stop_reason": stop_reason }, "usage": { "output_tokens": gr.timings.predicted_n } });
                            let _ = tx.blocking_send(Ok(Event::default().event("message_delta").data(md.to_string())));
                            let _ = tx.blocking_send(Ok(Event::default().event("message_stop").data(json!({ "type": "message_stop" }).to_string())));
                        }
                        Err(e) => {
                            let err = json!({ "type": "error", "error": { "type": "api_error", "message": e.to_string() } });
                            let _ = tx.blocking_send(Ok(Event::default().event("error").data(err.to_string())));
                        }
                    }
                }
                _ => {
                    let _ = tx.blocking_send(message_start(0));
                    let err = json!({ "type": "error", "error": { "type": "api_error", "message": "model unloaded or swapped before generation" } });
                    let _ = tx.blocking_send(Ok(Event::default().event("error").data(err.to_string())));
                }
            }
            st_c.governor.blocking_lock().touch(&model_id, Instant::now());
        });
        return Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default()).into_response();
    }

    let st_c = Arc::clone(&st);
    let model_id_c = model_id.clone();
    let max_tokens = req.max_tokens;
    let temp = req.temperature.unwrap_or(0.7);
    let res = tokio::task::spawn_blocking(move || {
        let mut guard = st_c.native.blocking_lock();
        let Some(active) = guard.as_mut() else {
            return Err("native engine unloaded before request ran".to_string());
        };
        if active.id != model_id_c {
            return Err(format!("model mismatch: requested '{model_id_c}' but '{}' is loaded", active.id));
        }
        let fam = detect_family(active.engine.tokenizer.chat_template());
        let started_in_think = fam == ocak_tok::TemplateFamily::Xing && enable_thinking;
        let prompt = render(fam, &chat_msgs, true, enable_thinking);
        let greq = GenRequest {
            prompt,
            parse_special: true,
            max_tokens,
            sampler: SamplerParams { temperature: temp, ..Default::default() },
            stop: Vec::new(),
            json_mode: false,
            thinking: think_budget.map(|m| ThinkingCfg { max_tokens: m }),
        };
        let out = active.engine.generate(&greq, |_, _| true).map_err(|e| e.to_string());
        st_c.governor.blocking_lock().touch(&model_id_c, Instant::now());
        out.map(|gr| (gr, started_in_think))
    })
    .await;
    match res {
        Ok(Ok((gr, started_in_think))) => {
            st.tokens_generated_total.fetch_add(gr.timings.predicted_n as u64, Ordering::Relaxed);
            st.kv_radix_hits_total.fetch_add(gr.timings.cache_n as u64, Ordering::Relaxed);
            let default_stop = match gr.finish_reason {
                FinishReason::Length => "max_tokens",
                _ => "end_turn",
            };
            let (r_opt, answer) = reasoning::split_reasoning(&gr.text, started_in_think);
            let mut blocks = Vec::new();
            if let Some(r_txt) = r_opt {
                blocks.push(json!({ "type": "thinking", "thinking": r_txt }));
            }
            let stop_reason = if has_tools {
                let (clean, calls) = tools::parse_tool_calls(&answer, "toolu_");
                if !calls.is_empty() {
                    if !clean.is_empty() {
                        blocks.push(json!({ "type": "text", "text": clean }));
                    }
                    for tc in &calls {
                        blocks.push(json!({ "type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments }));
                    }
                    "tool_use"
                } else {
                    blocks.push(json!({ "type": "text", "text": answer }));
                    default_stop
                }
            } else {
                blocks.push(json!({ "type": "text", "text": answer }));
                default_stop
            };
            let content = blocks;
            Json(json!({
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model_id,
                "content": content,
                "stop_reason": stop_reason,
                "usage": { "input_tokens": gr.timings.prompt_n, "output_tokens": gr.timings.predicted_n }
            }))
            .into_response()
        }
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "type": "error", "error": { "type": "api_error", "message": e } }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "type": "error", "error": { "type": "api_error", "message": e.to_string() } }))).into_response(),
    }
}

pub(crate) async fn anthropic_count_tokens(State(st): State<Arc<AppState>>, Json(req): Json<AnthropicReq>) -> Response {
    let Some(entry) = st.find_model(&req.model).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "model not found" }))).into_response();
    };
    let chat_msgs = anthropic_to_chat(&req);
    let text: String = chat_msgs.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n");
    let fallback = text.len() / 4;
    // Opening the GGUF and building the tokenizer is blocking work: off the async workers, and
    // cached per model file.
    let path = entry.path.clone();
    let n = tokio::task::spawn_blocking(move || {
        static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<std::path::PathBuf, Arc<ocak_tok::Tokenizer>>>> = std::sync::OnceLock::new();
        let cache = CACHE.get_or_init(Default::default);
        let cached = cache.lock().unwrap().get(&path).cloned();
        let tok = match cached {
            Some(t) => t,
            None => {
                let gf = ocak_gguf::GgufFile::open(&path).ok()?;
                let t = Arc::new(ocak_tok::Tokenizer::from_gguf(&gf).ok()?);
                cache.lock().unwrap().insert(path, Arc::clone(&t));
                t
            }
        };
        Some(tok.encode(&text, true, true).len())
    })
    .await
    .ok()
    .flatten()
    .unwrap_or(fallback);
    Json(json!({ "input_tokens": n })).into_response()
}
