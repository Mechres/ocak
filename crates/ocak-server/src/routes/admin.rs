//! Web UI, health, status / events, model load / sleep, metrics and model list.
use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use crate::*;

pub(crate) async fn ui_page() -> Html<&'static str> {
    Html(include_str!("../ui.html"))
}

pub(crate) async fn health(State(st): State<Arc<AppState>>) -> Json<Value> {
    let snap = st.status_snapshot().await;
    Json(json!({ "status": "ok", "gpu": snap.gpu, "hip": snap.hip }))
}

pub(crate) async fn ocak_status(State(st): State<Arc<AppState>>) -> Json<StatusSnapshot> {
    Json(st.status_snapshot().await)
}

pub(crate) async fn ocak_events(
    State(st): State<Arc<AppState>>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel(8);
    tokio::spawn(async move {
        let mut iv = tokio::time::interval(Duration::from_millis(500));
        loop {
            iv.tick().await;
            let snap = st.status_snapshot().await;
            let data = serde_json::to_string(&snap).unwrap_or_default();
            if tx.send(Ok(Event::default().data(data))).await.is_err() {
                break;
            }
        }
    });
    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

pub(crate) async fn ocak_load(State(st): State<Arc<AppState>>, AxPath(id): AxPath<String>) -> Response {
    let Some(entry) = st.find_model(&id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "model not found" }))).into_response();
    };
    let res = if engine_kind_for(&entry) == "native-vulkan" {
        st.ensure_native(&entry).await.map(|_| ())
    } else {
        st.ensure_bridge(&entry).await.map(|_| ())
    };
    match res {
        Ok(()) => Json(json!({ "status": "loaded", "model": entry.id })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
    }
}

pub(crate) async fn ocak_sleep(State(st): State<Arc<AppState>>, AxPath(id): AxPath<String>) -> Json<Value> {
    st.sleep_model(&id).await;
    Json(json!({ "status": "warm_sleep", "model": id }))
}

pub(crate) async fn metrics(State(st): State<Arc<AppState>>) -> String {
    let snap = st.status_snapshot().await;
    let mut s = String::new();
    if let Some(g) = snap.gpu {
        s.push_str(&format!("ocak_gpu_vram_used_bytes {}\n", g.vram_used_bytes));
        s.push_str(&format!("ocak_gpu_vram_total_bytes {}\n", g.vram_total_bytes));
        s.push_str(&format!("ocak_gpu_gtt_used_bytes {}\n", g.gtt_used_bytes));
        s.push_str(&format!("ocak_gpu_power_watts {:.3}\n", g.power_watts));
        s.push_str(&format!("ocak_gpu_temp_celsius {:.2}\n", g.temp_celsius));
        s.push_str(&format!("ocak_gpu_busy_percent {}\n", g.gpu_busy_percent));
    }
    s.push_str(&format!("ocak_tokens_generated_total {}\n", snap.tokens_generated_total));
    s.push_str(&format!("ocak_kv_radix_hits_total {}\n", snap.kv_radix_hits_total));
    s.push_str(&format!("ocak_requests_total {}\n", snap.requests_total));
    s
}

pub(crate) async fn list_models(State(st): State<Arc<AppState>>) -> Json<Value> {
    let snap = st.status_snapshot().await;
    let data: Vec<Value> = snap
        .models
        .into_iter()
        .filter(|m| !m.is_mmproj)
        .map(|m| {
            json!({
                "id": m.id,
                "object": "model",
                "owned_by": "ocak",
                "ocak": {
                    "arch": m.arch,
                    "quant": m.quant,
                    "size_bytes": m.size_bytes,
                    "sha256": m.sha256,
                    "engine": m.engine,
                    "residency": m.residency,
                }
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}
