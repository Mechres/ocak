//! Audio transcription (native Qwen3-ASR, or a llama-server bridge).
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use crate::*;

/// Value of a simple text field `name` in a multipart/form-data body.
pub(crate) fn multipart_field(body: &[u8], name: &str) -> Option<String> {
    let needle = format!("name=\"{name}\"");
    let at = body.windows(needle.len()).position(|w| w == needle.as_bytes())?;
    let rest = &body[at + needle.len()..];
    let start = rest.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
    let val = &rest[start..];
    let end = val.windows(2).position(|w| w == b"\r\n")?;
    let v = std::str::from_utf8(&val[..end]).ok()?.trim();
    (!v.is_empty() && v.len() < 256).then(|| v.to_string())
}

pub(crate) async fn audio_transcriptions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    st.requests_total.fetch_add(1, Ordering::Relaxed);
    // The multipart `model` field picks the model; otherwise any ASR model in the catalog.
    let requested = multipart_field(&body, "model");
    let entry = requested
        .as_deref()
        .and_then(|m| st.find_model(m))
        .or_else(|| st.find_model("Qwen3-ASR-1.7B"))
        .or_else(|| st.catalog.iter().find(|e| !e.is_mmproj && e.id.to_ascii_lowercase().contains("asr")))
        .cloned();
    let Some(entry) = entry else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "no ASR model found in the model catalog" }))).into_response();
    };
    // Native path: if a RIFF/WAVE payload is present (raw or inside multipart) and mmproj exists alongside the model, transcribe natively on Vulkan
    if let Some(riff_pos) = body.windows(12).position(|w| &w[0..4] == b"RIFF" && &w[8..12] == b"WAVE") {
        // The RIFF size field bounds the file: inside multipart, the closing boundary follows it.
        let riff_len = u32::from_le_bytes(body[riff_pos + 4..riff_pos + 8].try_into().unwrap()) as usize + 8;
        let wav_slice = &body[riff_pos..(riff_pos + riff_len).min(body.len())];
        let mmproj_path = entry.path.parent().and_then(|dir| {
            std::fs::read_dir(dir).ok()?.flatten().find_map(|e| {
                let p = e.path();
                let n = p.file_name()?.to_str()?;
                (n.starts_with("mmproj") && n.ends_with(".gguf")).then_some(p)
            })
        });
        if let Some(mmproj) = mmproj_path {
            if let Err(e) = st.ensure_asr(&entry, &mmproj).await {
                tracing::warn!(error = %e, "native ASR load failed, falling back to bridge");
            } else {
                let st_c = Arc::clone(&st);
                let model_id = entry.id.clone();
                let wav_owned = wav_slice.to_vec();
                let res = tokio::task::spawn_blocking(move || {
                    let mut guard = st_c.asr.blocking_lock();
                    let Some((aid, eng)) = guard.as_mut() else {
                        return Err("ASR engine unloaded".to_string());
                    };
                    if aid != &model_id {
                        return Err("ASR model swapped".to_string());
                    }
                    let out = eng.transcribe_wav(&wav_owned, 256).map_err(|e| e.to_string());
                    st_c.governor.blocking_lock().touch(&model_id, Instant::now());
                    out
                })
                .await;
                if let Ok(Ok(raw)) = res {
                    let clean = raw.split("<asr_text>").nth(1).unwrap_or(&raw).trim().to_string();
                    return Json(json!({ "text": clean, "raw": raw, "backend": "native-vk-qwen3a" })).into_response();
                }
            }
        }
    }
    let ct = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("multipart/form-data");
    match st.ensure_bridge(&entry).await {
        Ok(addr) => match ocak_federation::bridges::http_request(&addr, "POST", "/v1/audio/transcriptions", ct, &body).await {
            Ok((code, b)) => (StatusCode::from_u16(code).unwrap_or(StatusCode::OK), b).into_response(),
            Err(e) => (StatusCode::BAD_GATEWAY, Json(json!({ "error": e }))).into_response(),
        },
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
    }
}
