//! Shared server state: model catalog, residency (VRAM governor), loaded engines and bridges.
use ocak_engine::{EngineOptions, NativeEngine};
use ocak_federation::{
    hip_probe, HipProbe, LlamaBridgeHandle, ManagedLlamaBridge, MergenBridge, StrataBridge,
};
use ocak_gguf::catalog::{scan, CatalogEntry};
use ocak_gguf::preset::Presets;
use ocak_sysfs::{
    cpu_info, discover_amd_gpu, host_memory, Admission, CpuInfo, GpuDevice, GpuTelemetry,
    HostMemory, ModelResidencyState, VramBudgetConfig, VramGovernor,
};
use ocak_tok::template::detect_family;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use crate::*;

pub(crate) struct ActiveNative {
    pub(crate) id: String,
    pub(crate) engine: NativeEngine,
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
    pub(crate) governor: Mutex<VramGovernor>,
    pub(crate) load_lock: Mutex<()>,
    pub(crate) native: Mutex<Option<ActiveNative>>,
    pub(crate) asr: Mutex<Option<(String, ocak_engine::asr::Qwen3AsrEngine)>>,
    /// Loaded native model: (id, template family, multi-slot, template opens `<think>` only on
    /// an explicit `enable_thinking: true`).
    pub(crate) active_meta: std::sync::RwLock<Option<(String, ocak_tok::TemplateFamily, bool, bool)>>,
    pub(crate) batcher: Arc<batch::BatchCoordinator>,
    pub(crate) bridges: Mutex<HashMap<String, LlamaBridgeHandle>>,
    pub(crate) next_bridge_port: AtomicU64,
    pub tokens_generated_total: AtomicU64,
    pub kv_radix_hits_total: AtomicU64,
    pub requests_total: AtomicU64,
}

pub(crate) fn engine_kind_for(e: &CatalogEntry) -> &'static str {
    if matches!(e.arch.as_str(), "llama" | "qwen2" | "lfm2" | "lfm3" | "qwen35" | "xing4_0" | "inkling2_loop") {
        "native-vulkan"
    } else {
        "bridge-llama"
    }
}

impl AppState {
    /// Stops all llama-server bridge processes (called on shutdown: `AppState` itself is kept
    /// alive by background tasks, so the handles' `Drop` would never run).
    pub async fn shutdown(&self) {
        for (_, mut h) in self.bridges.lock().await.drain() {
            h.stop();
        }
    }

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
            asr: Mutex::new(None),
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
            let mut asr = self.asr.lock().await;
            if asr.as_ref().map(|(aid, _)| aid.as_str()) == Some(id) {
                *asr = None;
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

    pub async fn ensure_asr(&self, entry: &CatalogEntry, mmproj: &std::path::Path) -> Result<(), String> {
        {
            let asr = self.asr.lock().await;
            if asr.as_ref().map(|(aid, _)| aid.as_str()) == Some(entry.id.as_str()) {
                self.governor.lock().await.touch(&entry.id, Instant::now());
                return Ok(());
            }
        }
        let _load_guard = self.load_lock.lock().await;
        {
            let asr = self.asr.lock().await;
            if asr.as_ref().map(|(aid, _)| aid.as_str()) == Some(entry.id.as_str()) {
                self.governor.lock().await.touch(&entry.id, Instant::now());
                return Ok(());
            }
        }
        let mm_size = std::fs::metadata(mmproj).map(|m| m.len()).unwrap_or(0);
        let est = entry.size_bytes + mm_size + 512 * 1024 * 1024;
        let evict = {
            let g = self.governor.lock().await;
            match g.plan(est) {
                Admission::Fits => Vec::new(),
                Admission::EvictFirst(v) => v,
                Admission::TooLarge => Vec::new(),
            }
        };
        for old in evict {
            self.sleep_model(&old).await;
        }
        *self.asr.lock().await = None;
        self.governor
            .lock()
            .await
            .set_state(&entry.id, est, ModelResidencyState::Loading, Instant::now());
        let model_path = entry.path.clone();
        let mmproj_owned = mmproj.to_path_buf();
        let res = tokio::task::spawn_blocking(move || {
            ocak_engine::asr::Qwen3AsrEngine::load(&model_path, &mmproj_owned, true)
        })
        .await
        .map_err(|e| format!("join error: {e}"))
        .and_then(|r| r.map_err(|e| format!("ASR load failed: {e}")));
        let eng = match res {
            Ok(e) => e,
            Err(e) => {
                // Do not keep the estimate reserved for a model that never loaded.
                self.governor.lock().await.set_state(&entry.id, 0, ModelResidencyState::ColdCatalog, Instant::now());
                return Err(e);
            }
        };
        *self.asr.lock().await = Some((entry.id.clone(), eng));
        self.governor
            .lock()
            .await
            .set_state(&entry.id, est, ModelResidencyState::ActiveGpu, Instant::now());
        Ok(())
    }

    /// Whether the loaded model `id`'s chat template opens a `<think>` block only when thinking
    /// is explicitly requested (Qwen3.5 style), so the prompt must do it for `enable_thinking: true`.
    pub(crate) fn template_opens_think(&self, id: &str) -> bool {
        matches!(&*self.active_meta.read().unwrap(), Some((mid, _, _, true)) if mid == id)
    }

    pub async fn ensure_native(&self, entry: &CatalogEntry) -> Result<(ocak_tok::TemplateFamily, bool), String> {
        // Fast lock-free check so concurrent requests can join the continuous batcher immediately!
        let fast = self.active_meta.read().unwrap().clone();
        if let Some((id, fam, multi, _)) = fast {
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
        if let Some((id, fam, multi, _)) = cached {
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
                    let opens = ocak_tok::template::opens_think_on_request(a.engine.tokenizer.chat_template());
                    *self.active_meta.write().unwrap() = Some((entry.id.clone(), fam, multi, opens));
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
        // Room for two full-length contexts (prefix cache + concurrent slots share the pool).
        // NativeEngine::load clamps this, and max_ctx itself, to the VRAM actually free.
        let kv_pages = (self.cfg.max_ctx / 16 * 2).max(1024);
        let opts = EngineOptions {
            backend: self.cfg.backend,
            max_ctx: self.cfg.max_ctx,
            kv_pages,
        };
        let res = tokio::task::spawn_blocking(move || NativeEngine::load(&path, opts))
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| r.map_err(|e| e.to_string()));
        let eng = match res {
            Ok(e) => e,
            Err(e) => {
                // Do not keep the estimate reserved for a model that never loaded.
                self.governor.lock().await.set_state(&entry.id, 0, ModelResidencyState::ColdCatalog, Instant::now());
                return Err(e);
            }
        };
        let vram = eng.estimated_vram_bytes();
        let fam = detect_family(eng.tokenizer.chat_template());
        let opens = ocak_tok::template::opens_think_on_request(eng.tokenizer.chat_template());
        let multi = eng.supports_multi_slot();
        *self.native.lock().await = Some(ActiveNative { id: entry.id.clone(), engine: eng });
        *self.active_meta.write().unwrap() = Some((entry.id.clone(), fam, multi, opens));
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
