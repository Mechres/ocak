//! VRAM admission + idle sleep policy for the 12 GiB RX 6700 XT.
use serde::Serialize;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelResidencyState {
    ColdCatalog,
    Loading,
    ActiveGpu,
    CoResident,
    WarmSleep,
}

#[derive(Debug, Clone, Copy)]
pub struct VramBudgetConfig {
    pub vram_total_bytes: u64,
    pub compositor_headroom_bytes: u64,
    pub sleep_idle_seconds: u64,
}

impl Default for VramBudgetConfig {
    fn default() -> Self {
        Self {
            vram_total_bytes: 12_868_124_672,
            compositor_headroom_bytes: 900 * 1024 * 1024,
            sleep_idle_seconds: 120,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResidentModel {
    pub id: String,
    pub vram_bytes: u64,
    pub state: ModelResidencyState,
    pub last_used: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    /// Fits next to the current residents.
    Fits,
    /// Fits only after sleeping these models (least recently used first).
    EvictFirst(Vec<String>),
    /// Larger than the whole usable budget.
    TooLarge,
}

#[derive(Debug)]
pub struct VramGovernor {
    pub cfg: VramBudgetConfig,
    models: BTreeMap<String, ResidentModel>,
}

impl VramGovernor {
    pub fn new(cfg: VramBudgetConfig) -> Self {
        Self { cfg, models: BTreeMap::new() }
    }

    pub fn usable_bytes(&self) -> u64 {
        self.cfg.vram_total_bytes.saturating_sub(self.cfg.compositor_headroom_bytes)
    }

    pub fn resident_bytes(&self) -> u64 {
        self.models
            .values()
            .filter(|m| matches!(m.state, ModelResidencyState::ActiveGpu | ModelResidencyState::CoResident | ModelResidencyState::Loading))
            .map(|m| m.vram_bytes)
            .sum()
    }

    pub fn plan(&self, candidate_bytes: u64) -> Admission {
        let usable = self.usable_bytes();
        if candidate_bytes > usable {
            return Admission::TooLarge;
        }
        let mut used = self.resident_bytes();
        if used + candidate_bytes <= usable {
            return Admission::Fits;
        }
        let mut live: Vec<&ResidentModel> = self
            .models
            .values()
            .filter(|m| matches!(m.state, ModelResidencyState::ActiveGpu | ModelResidencyState::CoResident))
            .collect();
        live.sort_by_key(|m| m.last_used);
        let mut evict = Vec::new();
        for m in live {
            if used + candidate_bytes <= usable {
                break;
            }
            used -= m.vram_bytes.min(used);
            evict.push(m.id.clone());
        }
        Admission::EvictFirst(evict)
    }

    pub fn set_state(&mut self, id: &str, vram_bytes: u64, state: ModelResidencyState, now: Instant) {
        let e = self.models.entry(id.to_string()).or_insert(ResidentModel {
            id: id.to_string(),
            vram_bytes,
            state,
            last_used: now,
        });
        e.vram_bytes = vram_bytes;
        e.state = state;
        e.last_used = now;
    }

    pub fn touch(&mut self, id: &str, now: Instant) {
        if let Some(m) = self.models.get_mut(id) {
            m.last_used = now;
        }
    }

    pub fn state_of(&self, id: &str) -> ModelResidencyState {
        self.models.get(id).map(|m| m.state).unwrap_or(ModelResidencyState::ColdCatalog)
    }

    /// Models whose idle time exceeded the sleep threshold.
    pub fn idle_candidates(&self, now: Instant) -> Vec<String> {
        let limit = Duration::from_secs(self.cfg.sleep_idle_seconds);
        self.models
            .values()
            .filter(|m| matches!(m.state, ModelResidencyState::ActiveGpu | ModelResidencyState::CoResident))
            .filter(|m| now.duration_since(m.last_used) >= limit)
            .map(|m| m.id.clone())
            .collect()
    }

    pub fn snapshot(&self) -> Vec<ResidentModel> {
        self.models.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;

    #[test]
    fn admission_fits_evicts_and_rejects() {
        let t0 = Instant::now();
        let mut g = VramGovernor::new(VramBudgetConfig::default());
        assert_eq!(g.plan(6 * GIB), Admission::Fits);
        g.set_state("main9b", 6 * GIB, ModelResidencyState::ActiveGpu, t0);
        g.set_state("asr", 2 * GIB, ModelResidencyState::CoResident, t0 + Duration::from_secs(1));
        assert_eq!(g.plan(2 * GIB), Admission::Fits);
        match g.plan(10 * GIB) {
            Admission::EvictFirst(v) => assert_eq!(v, vec!["main9b".to_string(), "asr".to_string()]),
            o => panic!("{o:?}"),
        }
        assert_eq!(g.plan(13 * GIB), Admission::TooLarge);
    }

    #[test]
    fn idle_sleep_selection() {
        let t0 = Instant::now();
        let mut g = VramGovernor::new(VramBudgetConfig { sleep_idle_seconds: 10, ..Default::default() });
        g.set_state("a", GIB, ModelResidencyState::ActiveGpu, t0);
        assert!(g.idle_candidates(t0 + Duration::from_secs(5)).is_empty());
        assert_eq!(g.idle_candidates(t0 + Duration::from_secs(11)), vec!["a".to_string()]);
        g.set_state("a", GIB, ModelResidencyState::WarmSleep, t0);
        assert!(g.idle_candidates(t0 + Duration::from_secs(99)).is_empty());
    }
}
