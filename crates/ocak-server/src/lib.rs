//! Multi-protocol HTTP/SSE server: OpenAI, Anthropic `/v1/messages`, Embeddings, Audio/ASR,
//! Prometheus `/metrics`, Admin SSE, embedded `/ui`, and RX 6700 XT WarmSleep watchdog.
pub mod batch;
pub mod reasoning;
pub mod tools;
mod routes;
mod security;
mod state;
#[cfg(test)]
mod tests;

use axum::middleware::{self};
use axum::routing::{get, post};
use axum::Router;
use ocak_engine::BackendKind;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::limit::RequestBodyLimitLayer;

pub(crate) use routes::*;
pub(crate) use security::*;
pub use state::*;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub models_dirs: Vec<PathBuf>,
    pub presets_path: PathBuf,
    pub sleep_idle_seconds: u64,
    pub backend: BackendKind,
    pub max_ctx: usize,
    pub api_key: Option<String>,
    pub allow_remote_host: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        let models_dir = std::env::var("OCAK_MODELS_DIR")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var("HOME")
                    .map(|h| PathBuf::from(h).join("models"))
                    .unwrap_or_else(|_| PathBuf::from("./models"))
            });
        let presets_path = models_dir.join("models-preset.ini");
        Self {
            models_dirs: vec![models_dir],
            presets_path,
            sleep_idle_seconds: 120,
            backend: BackendKind::Vulkan,
            max_ctx: 16384,
            api_key: std::env::var("OCAK_API_KEY").ok().filter(|s| !s.is_empty()),
            allow_remote_host: std::env::var("OCAK_ALLOW_REMOTE_HOST").ok().as_deref() == Some("1"),
        }
    }
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(ui_page))
        .route("/ui", get(ui_page))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/ocak/status", get(ocak_status))
        .route("/v1/ocak/events", get(ocak_events))
        .route("/v1/ocak/models/{id}/load", post(ocak_load))
        .route("/v1/ocak/models/{id}/sleep", post(ocak_sleep))
        .layer(middleware::from_fn_with_state(state.clone(), security_middleware))
        .layer(RequestBodyLimitLayer::new(64 * 1024 * 1024))
        .with_state(state)
}
