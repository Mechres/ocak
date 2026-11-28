//! Per-kernel GPU timing (`OCAK_GPU_PROFILE=1`).
use std::sync::Mutex;

/// Accumulated per-pipeline GPU time (name -> (dispatches, ns)) when profiling.
pub(crate) static GPU_PROF: Mutex<Option<std::collections::HashMap<&'static str, (u64, f64)>>> = Mutex::new(None);

/// Timestamp queries per recording (2 per timed dispatch); later dispatches go untimed.
pub(crate) const PROF_QUERIES: u32 = 16384;

/// Per-pipeline GPU time table accumulated since the last call, then reset (None unless
/// OCAK_GPU_PROFILE is set and something was recorded).
pub fn take_gpu_profile_report() -> Option<String> {
    let prof = GPU_PROF.lock().ok()?.take()?;
    if prof.is_empty() {
        return None;
    }
    let mut rows: Vec<_> = prof.iter().map(|(k, v)| (*k, *v)).collect();
    rows.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
    let total: f64 = rows.iter().map(|r| r.1 .1).sum();
    let mut s = format!("ocak GPU profile (dispatches serialized), total {:.1} ms\n", total / 1e6);
    s += &format!("{:<28} {:>9} {:>10} {:>6} {:>9}\n", "pipeline", "calls", "ms", "%", "us/call");
    for (name, (n, ns)) in rows {
        s += &format!(
            "{:<28} {:>9} {:>10.2} {:>6.1} {:>9.1}\n",
            name,
            n,
            ns / 1e6,
            100.0 * ns / total,
            ns / 1e3 / n as f64
        );
    }
    Some(s)
}

pub(crate) fn gpu_profile() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OCAK_GPU_PROFILE").is_ok_and(|v| v != "0"))
}
