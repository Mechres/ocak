//! Multi-protocol HTTP/SSE server: OpenAI, Anthropic `/v1/messages`, Embeddings, Audio/ASR,
//! Prometheus `/metrics`, Admin SSE, embedded `/ui`, and RX 6700 XT WarmSleep watchdog.
pub mod batch;
pub mod tools;
use axum::body::Bytes;
use axum::extract::{Path as AxPath, Request as AxRequest, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use ocak_engine::{BackendKind, EngineOptions, FinishReason, GenRequest, NativeEngine, SamplerParams, ThinkingCfg};
use ocak_federation::{
    hip_probe, HipProbe, LlamaBridgeHandle, ManagedLlamaBridge, MergenBridge, StrataBridge,
};
use ocak_gguf::catalog::{scan, CatalogEntry};
use ocak_gguf::preset::Presets;
use ocak_sysfs::{
    cpu_info, discover_amd_gpu, host_memory, Admission, CpuInfo, GpuDevice, GpuTelemetry,
    HostMemory, ModelResidencyState, VramBudgetConfig, VramGovernor,
};
use ocak_tok::template::{detect_family, render, ChatMessage};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::limit::RequestBodyLimitLayer;
use uuid::Uuid;

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
        Self {
            models_dirs: vec![
                PathBuf::from("/home/mechres/models"),
                PathBuf::from("/home/mechres/Projeler/mergen/models"),
            ],
            presets_path: PathBuf::from("/home/mechres/models/models-preset.ini"),
            sleep_idle_seconds: 120,
            backend: BackendKind::Vulkan,
            max_ctx: 2048,
            api_key: std::env::var("OCAK_API_KEY").ok().filter(|s| !s.is_empty()),
            allow_remote_host: std::env::var("OCAK_ALLOW_REMOTE_HOST").ok().as_deref() == Some("1"),
        }
    }
}

struct ActiveNative {
    id: String,
    engine: NativeEngine,
}

pub struct AppState {
    pub cfg: ServerConfig,
    pub catalog: Vec<CatalogEntry>,
    pub presets: Presets,
    pub gpu: Option<GpuDevice>,
    pub hip: HipProbe,
    pub strata: StrataBridge,
    pub mergen: MergenBridge,
    pub llama_bridge: ManagedLlamaBridge,
    governor: Mutex<VramGovernor>,
    load_lock: Mutex<()>,
    native: Mutex<Option<ActiveNative>>,
    active_meta: std::sync::RwLock<Option<(String, ocak_tok::TemplateFamily, bool)>>,
    batcher: Arc<batch::BatchCoordinator>,
    bridges: Mutex<HashMap<String, LlamaBridgeHandle>>,
    next_bridge_port: AtomicU64,
    pub tokens_generated_total: AtomicU64,
    pub kv_radix_hits_total: AtomicU64,
    pub requests_total: AtomicU64,
}

fn engine_kind_for(e: &CatalogEntry) -> &'static str {
    if matches!(e.arch.as_str(), "llama" | "qwen2" | "lfm2" | "lfm3" | "qwen35" | "xing4_0" | "inkling2_loop") {
        "native-vulkan"
    } else {
        "bridge-llama"
    }
}

impl AppState {
    pub fn new(cfg: ServerConfig) -> Arc<Self> {
        let catalog = scan(&cfg.models_dirs);
        let presets = Presets::load(&cfg.presets_path);
        let gpu = discover_amd_gpu();
        let vram_total = gpu.as_ref().map(|g| g.telemetry().vram_total_bytes).unwrap_or(12 << 30);
        let sleep_secs = if cfg.sleep_idle_seconds > 0 {
            cfg.sleep_idle_seconds
        } else {
            presets
                .defaults
                .get("sleep-idle-seconds")
                .and_then(|s| s.parse().ok())
                .unwrap_or(120)
        };
        let gov = VramGovernor::new(VramBudgetConfig {
            vram_total_bytes: vram_total,
            compositor_headroom_bytes: 900 * 1024 * 1024,
            sleep_idle_seconds: sleep_secs,
        });
        Arc::new(Self {
            cfg,
            catalog,
            presets,
            gpu,
            hip: hip_probe(),
            strata: StrataBridge::probe(),
            mergen: MergenBridge::default(),
            llama_bridge: ManagedLlamaBridge::default(),
            governor: Mutex::new(gov),
            load_lock: Mutex::new(()),
            native: Mutex::new(None),
            active_meta: std::sync::RwLock::new(None),
            batcher: Arc::new(batch::BatchCoordinator::new()),
            bridges: Mutex::new(HashMap::new()),
            next_bridge_port: AtomicU64::new(39100),
            tokens_generated_total: AtomicU64::new(0),
            kv_radix_hits_total: AtomicU64::new(0),
            requests_total: AtomicU64::new(0),
        })
    }

    pub fn spawn_watchdog(self: &Arc<Self>) {
        let st = Arc::clone(self);
        tokio::spawn(async move {
            let mut iv = tokio::time::interval(Duration::from_secs(1));
            loop {
                iv.tick().await;
                let idle = {
                    let g = st.governor.lock().await;
                    g.idle_candidates(Instant::now())
                };
                for id in idle {
                    tracing::info!(model = %id, "idle timeout reached -> WarmSleep (releasing GPU)");
                    st.sleep_model(&id).await;
                }
            }
        });
    }

    pub fn find_model(&self, id: &str) -> Option<&CatalogEntry> {
        let clean = id.trim_end_matches(".gguf");
        self.catalog
            .iter()
            .find(|e| e.id == clean || e.id.eq_ignore_ascii_case(clean) || e.path.to_string_lossy() == id)
            .or_else(|| self.catalog.iter().find(|e| !e.is_mmproj && e.id.to_lowercase().contains(&clean.to_lowercase())))
    }

    pub async fn sleep_model(&self, id: &str) {
        {
            let mut nat = self.native.lock().await;
            if nat.as_ref().map(|a| a.id.as_str()) == Some(id) {
                *nat = None; // Drop VkExec -> vkDestroyDevice -> releases RX 6700 XT clocks
                *self.active_meta.write().unwrap() = None;
            }
        }
        {
            let mut br = self.bridges.lock().await;
            if let Some(mut h) = br.remove(id) {
                h.stop();
            }
        }
        let mut g = self.governor.lock().await;
        g.set_state(id, 0, ModelResidencyState::WarmSleep, Instant::now());
    }

    pub async fn ensure_native(&self, entry: &CatalogEntry) -> Result<(ocak_tok::TemplateFamily, bool), String> {
        // Fast lock-free check so concurrent requests can join the continuous batcher immediately!
        let fast = self.active_meta.read().unwrap().clone();
        if let Some((id, fam, multi)) = fast {
            if id == entry.id {
                if let Ok(mut g) = self.governor.try_lock() {
                    g.touch(&entry.id, Instant::now());
                }
                return Ok((fam, multi));
            }
        }
        // Serialize model loads so two cold requests never double-load and OOM VRAM
        let _load_guard = self.load_lock.lock().await;
        let cached = self.active_meta.read().unwrap().clone();
        if let Some((id, fam, multi)) = cached {
            if id == entry.id {
                self.governor.lock().await.touch(&entry.id, Instant::now());
                return Ok((fam, multi));
            }
        }
        {
            let nat = self.native.lock().await;
            if let Some(a) = nat.as_ref() {
                if a.id == entry.id {
                    let fam = detect_family(a.engine.tokenizer.chat_template());
                    let multi = a.engine.supports_multi_slot();
                    *self.active_meta.write().unwrap() = Some((entry.id.clone(), fam, multi));
                    self.governor.lock().await.touch(&entry.id, Instant::now());
                    return Ok((fam, multi));
                }
            }
        }
        let est = entry.size_bytes + 512 * 1024 * 1024;
        let evict = {
            let g = self.governor.lock().await;
            match g.plan(est) {
                Admission::Fits => Vec::new(),
                Admission::EvictFirst(v) => v,
                Admission::TooLarge => Vec::new(), // allow GTT spillover on Vulkan
            }
        };
        for old in evict {
            self.sleep_model(&old).await;
        }
        // Drop previous native model before loading new one
        *self.native.lock().await = None;
        *self.active_meta.write().unwrap() = None;
        self.governor
            .lock()
            .await
            .set_state(&entry.id, est, ModelResidencyState::Loading, Instant::now());

        let path = entry.path.clone();
        let opts = EngineOptions {
            backend: self.cfg.backend,
            max_ctx: self.cfg.max_ctx,
            kv_pages: (self.cfg.max_ctx / 16 * 8).max(1024),
        };
        let eng = tokio::task::spawn_blocking(move || NativeEngine::load(&path, opts))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let vram = eng.estimated_vram_bytes();
        let fam = detect_family(eng.tokenizer.chat_template());
        let multi = eng.supports_multi_slot();
        *self.native.lock().await = Some(ActiveNative { id: entry.id.clone(), engine: eng });
        *self.active_meta.write().unwrap() = Some((entry.id.clone(), fam, multi));
        self.governor
            .lock()
            .await
            .set_state(&entry.id, vram, ModelResidencyState::ActiveGpu, Instant::now());
        Ok((fam, multi))
    }

    pub async fn ensure_bridge(&self, entry: &CatalogEntry) -> Result<String, String> {
        {
            let br = self.bridges.lock().await;
            if let Some(h) = br.get(&entry.id) {
                self.governor.lock().await.touch(&entry.id, Instant::now());
                return Ok(h.base_addr());
            }
        }
        let _load_guard = self.load_lock.lock().await;
        {
            let br = self.bridges.lock().await;
            if let Some(h) = br.get(&entry.id) {
                self.governor.lock().await.touch(&entry.id, Instant::now());
                return Ok(h.base_addr());
            }
        }
        let est = entry.size_bytes + 512 * 1024 * 1024;
        let evict = match self.governor.lock().await.plan(est) {
            Admission::EvictFirst(v) => v,
            _ => Vec::new(),
        };
        for old in evict {
            self.sleep_model(&old).await;
        }
        let mmproj = entry.path.parent().and_then(|dir| {
            self.catalog
                .iter()
                .find(|c| c.is_mmproj && c.path.parent() == Some(dir))
                .map(|c| c.path.as_path())
        });
        let port = self.next_bridge_port.fetch_add(1, Ordering::Relaxed) as u16;
        let handle = self
            .llama_bridge
            .spawn_model(&entry.id, &entry.path, mmproj, &self.presets, port)
            .await?;
        let addr = handle.base_addr();
        self.bridges.lock().await.insert(entry.id.clone(), handle);
        self.governor
            .lock()
            .await
            .set_state(&entry.id, est, ModelResidencyState::ActiveGpu, Instant::now());
        Ok(addr)
    }

    pub async fn status_snapshot(&self) -> StatusSnapshot {
        let g = self.governor.lock().await;
        let models = self
            .catalog
            .iter()
            .map(|e| ModelStatus {
                id: e.id.clone(),
                arch: e.arch.clone(),
                quant: e.quant_summary.clone(),
                size_bytes: e.size_bytes,
                sha256: e.sha256.clone(),
                engine: engine_kind_for(e).into(),
                residency: g.state_of(&e.id),
                is_mmproj: e.is_mmproj,
            })
            .collect();
        StatusSnapshot {
            gpu: self.gpu.as_ref().map(|d| d.telemetry()),
            host_memory: host_memory(),
            cpu: cpu_info(),
            hip: self.hip.clone(),
            tokens_generated_total: self.tokens_generated_total.load(Ordering::Relaxed),
            kv_radix_hits_total: self.kv_radix_hits_total.load(Ordering::Relaxed),
            requests_total: self.requests_total.load(Ordering::Relaxed),
            models,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub arch: String,
    pub quant: String,
    pub size_bytes: u64,
    pub sha256: Option<String>,
    pub engine: String,
    pub residency: ModelResidencyState,
    pub is_mmproj: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct StatusSnapshot {
    pub gpu: Option<GpuTelemetry>,
    pub host_memory: HostMemory,
    pub cpu: CpuInfo,
    pub hip: HipProbe,
    pub tokens_generated_total: u64,
    pub kv_radix_hits_total: u64,
    pub requests_total: u64,
    pub models: Vec<ModelStatus>,
}

fn is_loopback_host(host: &str) -> bool {
    let h = host.trim();
    if let Some(rest) = h.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            return &rest[..end] == "::1";
        }
        return false;
    }
    let bare = h.split(':').next().unwrap_or("");
    matches!(bare, "localhost" | "127.0.0.1" | "::1")
}

fn is_loopback_origin(origin: &str) -> bool {
    let s = origin.trim();
    let rest = s
        .strip_prefix("http://")
        .or_else(|| s.strip_prefix("https://"))
        .unwrap_or("");
    if rest.is_empty() {
        return false;
    }
    let host_port = rest.split('/').next().unwrap_or("");
    is_loopback_host(host_port)
}

async fn security_middleware(
    State(st): State<Arc<AppState>>,
    req: AxRequest,
    next: Next,
) -> Response {
    if !st.cfg.allow_remote_host {
        if let Some(host) = req.headers().get("host").and_then(|v| v.to_str().ok()) {
            if !is_loopback_host(host) {
                return (StatusCode::FORBIDDEN, Json(json!({ "error": "invalid Host header" }))).into_response();
            }
        }
        if let Some(origin) = req.headers().get("origin").and_then(|v| v.to_str().ok()) {
            if !is_loopback_origin(origin) {
                return (StatusCode::FORBIDDEN, Json(json!({ "error": "cross-origin request blocked" }))).into_response();
            }
        }
    }
    if let Some(ref expected) = st.cfg.api_key {
        let path = req.uri().path();
        if path.starts_with("/v1/") {
            let bearer = req
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "));
            let x_api = req.headers().get("x-api-key").and_then(|v| v.to_str().ok());
            let query_key = req.uri().query().and_then(|q| {
                q.split('&')
                    .find_map(|pair| pair.strip_prefix("api_key="))
            });
            if bearer != Some(expected.as_str())
                && x_api != Some(expected.as_str())
                && query_key != Some(expected.as_str())
            {
                return (StatusCode::UNAUTHORIZED, Json(json!({ "error": "unauthorized" }))).into_response();
            }
        }
    }
    next.run(req).await
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

async fn ui_page() -> Html<&'static str> {
    Html(include_str!("ui.html"))
}

async fn health(State(st): State<Arc<AppState>>) -> Json<Value> {
    let snap = st.status_snapshot().await;
    Json(json!({ "status": "ok", "gpu": snap.gpu, "hip": snap.hip }))
}

async fn ocak_status(State(st): State<Arc<AppState>>) -> Json<StatusSnapshot> {
    Json(st.status_snapshot().await)
}

async fn ocak_events(
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

async fn ocak_load(State(st): State<Arc<AppState>>, AxPath(id): AxPath<String>) -> Response {
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

async fn ocak_sleep(State(st): State<Arc<AppState>>, AxPath(id): AxPath<String>) -> Json<Value> {
    st.sleep_model(&id).await;
    Json(json!({ "status": "warm_sleep", "model": id }))
}

async fn metrics(State(st): State<Arc<AppState>>) -> String {
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

async fn list_models(State(st): State<Arc<AppState>>) -> Json<Value> {
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

#[derive(Debug, Deserialize)]
struct ChatReq {
    model: String,
    messages: Vec<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default = "def_max_tok")]
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    repeat_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    #[serde(default)]
    stop: Vec<String>,
    reasoning_max_tokens: Option<usize>,
    response_format: Option<Value>,
    #[serde(default)]
    tools: Vec<Value>,
    #[allow(dead_code)]
    tool_choice: Option<Value>,
}

fn def_max_tok() -> usize { 256 }

fn build_sampler(st: &AppState, model_id: &str, req: &ChatReq) -> SamplerParams {
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
        ..Default::default()
    }
}

async fn chat_completions(
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
    let model_id = entry.id.clone();
    let tool_defs = tools::extract_openai_tools(&req.tools);
    let has_tools = !tool_defs.is_empty();
    let chat_msgs = tools::openai_to_chat(&req.messages, &tool_defs);

    let prompt = render(fam, &chat_msgs, true, true);
    let greq = GenRequest {
        prompt,
        parse_special: true,
        max_tokens: req.max_tokens,
        sampler,
        stop: req.stop,
        json_mode,
        thinking: req.reasoning_max_tokens.map(|m| ThinkingCfg { max_tokens: m }),
    };

    if req.stream {
        let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(256);
        let (tok_tx, mut tok_rx) = mpsc::channel::<String>(256);
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
                while let Some(piece) = tok_rx.recv().await {
                    let chunk = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "model": model_id,
                        "choices": [{ "index": 0, "delta": { "content": piece }, "finish_reason": null }]
                    });
                    if tx.send(Ok(Event::default().data(chunk.to_string()))).await.is_err() {
                        return;
                    }
                }
            }
            if let Ok(Ok(gr)) = done_rx.await {
                let mut finish_reason = json!(gr.finish_reason);
                if has_tools {
                    let (clean, calls) = tools::parse_tool_calls(&gr.text, "call_");
                    if !clean.is_empty() {
                        let c = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
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
                    "model": model_id,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason }],
                    "timings": gr.timings
                });
                let _ = tx.send(Ok(Event::default().data(done_chunk.to_string()))).await;
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
    let (msg_obj, finish_reason) = if has_tools {
        let (clean, calls) = tools::parse_tool_calls(&gr.text, "call_");
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
            (json!({ "role": "assistant", "content": gr.text }), json!(gr.finish_reason))
        }
    } else {
        (json!({ "role": "assistant", "content": gr.text }), json!(gr.finish_reason))
    };
    Json(json!({
        "id": id,
        "object": "chat.completion",
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
struct CompReq {
    model: String,
    prompt: String,
    #[serde(default = "def_max_tok")]
    max_tokens: usize,
    temperature: Option<f32>,
}

async fn completions(State(st): State<Arc<AppState>>, Json(req): Json<CompReq>) -> Response {
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
struct EmbReq {
    model: String,
    input: Value,
}

async fn embeddings(State(st): State<Arc<AppState>>, Json(req): Json<EmbReq>) -> Response {
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

/// Anthropic Messages API (`/v1/messages`) with streaming SSE + tool_use for Claude Code compatibility.
#[derive(Debug, Deserialize)]
struct AnthropicReq {
    model: String,
    #[serde(default = "def_max_tok")]
    max_tokens: usize,
    system: Option<Value>,
    messages: Vec<Value>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f32>,
    #[serde(default)]
    tools: Vec<Value>,
    #[allow(dead_code)]
    tool_choice: Option<Value>,
}

fn anthropic_to_chat(req: &AnthropicReq) -> Vec<ChatMessage> {
    let tool_defs = tools::extract_anthropic_tools(&req.tools);
    tools::anthropic_messages_to_chat(req.system.as_ref(), &req.messages, &tool_defs)
}

async fn anthropic_messages(State(st): State<Arc<AppState>>, Json(req): Json<AnthropicReq>) -> Response {
    st.requests_total.fetch_add(1, Ordering::Relaxed);
    let Some(entry) = st.find_model(&req.model).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "type": "error", "error": { "type": "not_found_error", "message": "model not found" } }))).into_response();
    };
    if let Err(e) = st.ensure_native(&entry).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "type": "error", "error": { "type": "api_error", "message": e } }))).into_response();
    }
    let has_tools = !req.tools.is_empty();
    let chat_msgs = anthropic_to_chat(&req);
    let msg_id = format!("msg_{}", Uuid::new_v4().simple());
    let model_id = entry.id.clone();

    if req.stream {
        let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(256);
        let st_c = Arc::clone(&st);
        tokio::task::spawn_blocking(move || {
            let start = json!({
                "type": "message_start",
                "message": { "id": msg_id, "type": "message", "role": "assistant", "content": [], "model": model_id, "stop_reason": null, "usage": { "input_tokens": 0, "output_tokens": 0 } }
            });
            let _ = tx.blocking_send(Ok(Event::default().event("message_start").data(start.to_string())));
            if !has_tools {
                let blk = json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } });
                let _ = tx.blocking_send(Ok(Event::default().event("content_block_start").data(blk.to_string())));
            }

            let mut guard = st_c.native.blocking_lock();
            match guard.as_mut() {
                Some(active) if active.id == model_id => {
                    let fam = detect_family(active.engine.tokenizer.chat_template());
                    let prompt = render(fam, &chat_msgs, true, true);
                    let greq = GenRequest {
                        prompt,
                        parse_special: true,
                        max_tokens: req.max_tokens,
                        sampler: SamplerParams { temperature: req.temperature.unwrap_or(0.7), ..Default::default() },
                        stop: Vec::new(),
                        json_mode: false,
                        thinking: None,
                    };
                    let tx_c = tx.clone();
                    match active.engine.generate(&greq, |_, piece| {
                        if piece.is_empty() || has_tools { return true; }
                        let d = json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": piece } });
                        tx_c.blocking_send(Ok(Event::default().event("content_block_delta").data(d.to_string()))).is_ok()
                    }) {
                        Ok(gr) => {
                            st_c.tokens_generated_total.fetch_add(gr.timings.predicted_n as u64, Ordering::Relaxed);
                            st_c.kv_radix_hits_total.fetch_add(gr.timings.cache_n as u64, Ordering::Relaxed);
                            let mut stop_reason = match gr.finish_reason {
                                FinishReason::Length => "max_tokens",
                                _ => "end_turn",
                            };
                            if has_tools {
                                let (clean, calls) = tools::parse_tool_calls(&gr.text, "toolu_");
                                let mut idx = 0usize;
                                if !clean.is_empty() || calls.is_empty() {
                                    let blk = json!({ "type": "content_block_start", "index": idx, "content_block": { "type": "text", "text": "" } });
                                    let _ = tx.blocking_send(Ok(Event::default().event("content_block_start").data(blk.to_string())));
                                    let txt = if calls.is_empty() { &gr.text } else { &clean };
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
                                let _ = tx.blocking_send(Ok(Event::default().event("content_block_stop").data(json!({ "type": "content_block_stop", "index": 0 }).to_string())));
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
        let prompt = render(fam, &chat_msgs, true, true);
        let greq = GenRequest {
            prompt,
            parse_special: true,
            max_tokens,
            sampler: SamplerParams { temperature: temp, ..Default::default() },
            stop: Vec::new(),
            json_mode: false,
            thinking: None,
        };
        let out = active.engine.generate(&greq, |_, _| true).map_err(|e| e.to_string());
        st_c.governor.blocking_lock().touch(&model_id_c, Instant::now());
        out
    })
    .await;
    match res {
        Ok(Ok(gr)) => {
            st.tokens_generated_total.fetch_add(gr.timings.predicted_n as u64, Ordering::Relaxed);
            st.kv_radix_hits_total.fetch_add(gr.timings.cache_n as u64, Ordering::Relaxed);
            let default_stop = match gr.finish_reason {
                FinishReason::Length => "max_tokens",
                _ => "end_turn",
            };
            let (content, stop_reason) = if has_tools {
                let (clean, calls) = tools::parse_tool_calls(&gr.text, "toolu_");
                if !calls.is_empty() {
                    let mut blocks = Vec::new();
                    if !clean.is_empty() {
                        blocks.push(json!({ "type": "text", "text": clean }));
                    }
                    for tc in &calls {
                        blocks.push(json!({ "type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments }));
                    }
                    (blocks, "tool_use")
                } else {
                    (vec![json!({ "type": "text", "text": gr.text })], default_stop)
                }
            } else {
                (vec![json!({ "type": "text", "text": gr.text })], default_stop)
            };
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

async fn anthropic_count_tokens(State(st): State<Arc<AppState>>, Json(req): Json<AnthropicReq>) -> Response {
    let Some(entry) = st.find_model(&req.model).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "model not found" }))).into_response();
    };
    let chat_msgs = anthropic_to_chat(&req);
    let text: String = chat_msgs.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n");
    if let Ok(gf) = ocak_gguf::GgufFile::open(&entry.path) {
        if let Ok(tok) = ocak_tok::Tokenizer::from_gguf(&gf) {
            let n = tok.encode(&text, true, true).len();
            return Json(json!({ "input_tokens": n })).into_response();
        }
    }
    Json(json!({ "input_tokens": text.len() / 4 })).into_response()
}

async fn audio_transcriptions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    st.requests_total.fetch_add(1, Ordering::Relaxed);
    let Some(entry) = st.find_model("Qwen3-ASR-1.7B").cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "Qwen3-ASR-1.7B not found in model catalog" }))).into_response();
    };
    // Native path: if a RIFF/WAVE payload is present (raw or inside multipart) and mmproj exists alongside the model, transcribe natively on Vulkan
    if let Some(riff_pos) = body.windows(12).position(|w| &w[0..4] == b"RIFF" && &w[8..12] == b"WAVE") {
        let wav_slice = &body[riff_pos..];
        let mmproj_path = entry.path.parent().and_then(|dir| {
            std::fs::read_dir(dir).ok()?.flatten().find_map(|e| {
                let p = e.path();
                let n = p.file_name()?.to_str()?;
                (n.starts_with("mmproj") && n.ends_with(".gguf")).then_some(p)
            })
        });
        if let Some(mmproj) = mmproj_path {
            let model_path = entry.path.clone();
            let wav_owned = wav_slice.to_vec();
            let res = tokio::task::spawn_blocking(move || {
                let mut asr = ocak_engine::asr::Qwen3AsrEngine::load(&model_path, &mmproj, true)?;
                asr.transcribe_wav(&wav_owned, 256)
            })
            .await;
            if let Ok(Ok(raw)) = res {
                let clean = raw.split("<asr_text>").nth(1).unwrap_or(&raw).trim().to_string();
                return Json(json!({ "text": clean, "raw": raw, "backend": "native-vk-qwen3a" })).into_response();
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, String) {
        let req = match body {
            Some(v) => Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(v.to_string()))
                .unwrap(),
            None => Request::builder().method(method).uri(uri).body(Body::empty()).unwrap(),
        };
        let resp = app.clone().oneshot(req).await.unwrap();
        let st = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (st, String::from_utf8_lossy(&bytes).into_owned())
    }

    #[tokio::test]
    async fn end_to_end_openai_anthropic_embeddings_radix_kv_and_warm_sleep() {
        let model_path = PathBuf::from("/home/mechres/Projeler/mergen/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
        if !model_path.exists() {
            return;
        }
        let mut cfg = ServerConfig::default();
        cfg.sleep_idle_seconds = 2;
        let state = AppState::new(cfg);
        state.spawn_watchdog();
        let app = build_router(state.clone());

        // 1. /v1/models & /ui & /metrics
        let (st, body) = call(&app, "GET", "/v1/models", None).await;
        assert_eq!(st, StatusCode::OK);
        let models: Value = serde_json::from_str(&body).unwrap();
        assert!(models["data"].as_array().unwrap().len() >= 2);
        let (st_ui, ui_html) = call(&app, "GET", "/ui", None).await;
        assert_eq!(st_ui, StatusCode::OK);
        assert!(ui_html.contains("Ocak Control Center"));

        // 2. Turn 1: OpenAI chat completion
        let sys_msg = "You are a helpful, concise computer architecture assistant answering questions accurately in English.";
        let (st1, b1) = call(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
                "max_tokens": 20,
                "temperature": 0.0,
                "messages": [
                    { "role": "system", "content": sys_msg },
                    { "role": "user", "content": "What is the capital of France?" }
                ]
            })),
        )
        .await;
        assert_eq!(st1, StatusCode::OK, "{b1}");
        let j1: Value = serde_json::from_str(&b1).unwrap();
        let reply1 = j1["choices"][0]["message"]["content"].as_str().unwrap().to_string();
        println!("Turn 1: {:?} | timings: {}", reply1, j1["timings"]);

        // 3. Turn 2: Same prefix -> Radix KV prefix cache hit (cache_n >= 16)
        let (st2, b2) = call(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
                "max_tokens": 16,
                "temperature": 0.0,
                "messages": [
                    { "role": "system", "content": sys_msg },
                    { "role": "user", "content": "What is the capital of France?" },
                    { "role": "assistant", "content": reply1 },
                    { "role": "user", "content": "And Spain?" }
                ]
            })),
        )
        .await;
        assert_eq!(st2, StatusCode::OK, "{b2}");
        let j2: Value = serde_json::from_str(&b2).unwrap();
        let cache_n = j2["timings"]["cache_n"].as_u64().unwrap();
        println!("Turn 2: {:?} | cache_n={cache_n}", j2["choices"][0]["message"]["content"]);
        assert!(cache_n >= 16, "expected Radix KV prefix hit >= 16, got {cache_n}");

        // 4. OpenAI streaming SSE
        let (st_sse, sse_body) = call(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
                "stream": true,
                "max_tokens": 8,
                "temperature": 0.0,
                "messages": [{ "role": "user", "content": "Hi" }]
            })),
        )
        .await;
        assert_eq!(st_sse, StatusCode::OK);
        assert!(sse_body.contains("chat.completion.chunk") && sse_body.contains("data: [DONE]"));

        // 5. Anthropic /v1/messages streaming SSE
        let (st_ant, ant_body) = call(
            &app,
            "POST",
            "/v1/messages",
            Some(json!({
                "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
                "stream": true,
                "max_tokens": 10,
                "temperature": 0.0,
                "messages": [{ "role": "user", "content": "Say hello." }]
            })),
        )
        .await;
        assert_eq!(st_ant, StatusCode::OK);
        assert!(ant_body.contains("event: message_start"));
        assert!(ant_body.contains("event: content_block_delta"));
        assert!(ant_body.contains("event: message_stop"));

        // 5b. Security: Host-header DNS rebinding guard + optional API key check
        let bad_host = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("host", "evil.example.com:8090")
            .body(Body::empty())
            .unwrap();
        assert_eq!(app.clone().oneshot(bad_host).await.unwrap().status(), StatusCode::FORBIDDEN);

        let bad_origin = Request::builder()
            .method("POST")
            .uri("/v1/ocak/models/foo/sleep")
            .header("origin", "https://evil.example.com")
            .body(Body::empty())
            .unwrap();
        assert_eq!(app.clone().oneshot(bad_origin).await.unwrap().status(), StatusCode::FORBIDDEN);

        let mut auth_cfg = ServerConfig::default();
        auth_cfg.api_key = Some("secret-test-key".into());
        let auth_app = build_router(AppState::new(auth_cfg));
        let no_key = Request::builder().method("GET").uri("/v1/models").body(Body::empty()).unwrap();
        assert_eq!(auth_app.clone().oneshot(no_key).await.unwrap().status(), StatusCode::UNAUTHORIZED);
        let ok_key = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("authorization", "Bearer secret-test-key")
            .body(Body::empty())
            .unwrap();
        assert_eq!(auth_app.clone().oneshot(ok_key).await.unwrap().status(), StatusCode::OK);

        // 6. /v1/embeddings L2 norm == 1.0
        let (st_emb, emb_body) = call(
            &app,
            "POST",
            "/v1/embeddings",
            Some(json!({ "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M", "input": "paged attention" })),
        )
        .await;
        assert_eq!(st_emb, StatusCode::OK);
        let jemb: Value = serde_json::from_str(&emb_body).unwrap();
        let vec = jemb["data"][0]["embedding"].as_array().unwrap();
        let norm = vec.iter().map(|v| v.as_f64().unwrap().powi(2)).sum::<f64>().sqrt();
        println!("Embeddings: dim={}, L2 norm={norm:.6}", vec.len());
        assert!((norm - 1.0).abs() < 1e-4);

        // 6b. Continuous multi-slot batching: 3 concurrent /v1/chat/completions decoded together!
        let r_a = call(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
                "max_tokens": 16,
                "temperature": 0.0,
                "messages": [{ "role": "user", "content": "What is 2 + 2?" }]
            })),
        );
        let r_b = call(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
                "max_tokens": 16,
                "temperature": 0.0,
                "messages": [{ "role": "user", "content": "Name one planet." }]
            })),
        );
        let r_c = call(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
                "max_tokens": 16,
                "temperature": 0.0,
                "messages": [{ "role": "user", "content": "Say hello." }]
            })),
        );
        let ((sa, ba), (sb, bb), (sc, bc)) = tokio::join!(r_a, r_b, r_c);
        assert_eq!(sa, StatusCode::OK, "{ba}");
        assert_eq!(sb, StatusCode::OK, "{bb}");
        assert_eq!(sc, StatusCode::OK, "{bc}");
        let ja: Value = serde_json::from_str(&ba).unwrap();
        let jb: Value = serde_json::from_str(&bb).unwrap();
        let jc: Value = serde_json::from_str(&bc).unwrap();
        assert!(!ja["choices"][0]["message"]["content"].as_str().unwrap_or("").is_empty());
        assert!(!jb["choices"][0]["message"]["content"].as_str().unwrap_or("").is_empty());
        assert!(!jc["choices"][0]["message"]["content"].as_str().unwrap_or("").is_empty());

        // 7. WarmSleep idle teardown & RX 6700 XT power/VRAM release + automatic wake
        let vram_active = state.status_snapshot().await.gpu.map(|g| g.vram_used_bytes).unwrap_or(0);
        tokio::time::sleep(Duration::from_secs(4)).await;
        let snap_sleep = state.status_snapshot().await;
        let mstate = snap_sleep
            .models
            .iter()
            .find(|m| m.id == "tinyllama-1.1b-chat-v1.0.Q4_K_M")
            .unwrap()
            .residency;
        let vram_sleep = snap_sleep.gpu.map(|g| g.vram_used_bytes).unwrap_or(0);
        let pwr_sleep = snap_sleep.gpu.map(|g| g.power_watts).unwrap_or(0.0);
        println!(
            "WarmSleep: state={mstate:?}, VRAM {:.2} -> {:.2} GiB, Power={pwr_sleep:.1} W",
            vram_active as f64 / (1u64 << 30) as f64,
            vram_sleep as f64 / (1u64 << 30) as f64
        );
        assert_eq!(mstate, ModelResidencyState::WarmSleep);
        assert!(vram_sleep < vram_active, "expected VRAM to drop after WarmSleep");

        // Wake automatically on next request
        let t_w = Instant::now();
        let (st_w, bw) = call(
            &app,
            "POST",
            "/v1/completions",
            Some(json!({ "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M", "prompt": "Hello", "max_tokens": 4, "temperature": 0.0 })),
        )
        .await;
        assert_eq!(st_w, StatusCode::OK, "{bw}");
        println!("Auto-wake completed in {:.0} ms", t_w.elapsed().as_secs_f64() * 1000.0);
    }
}
