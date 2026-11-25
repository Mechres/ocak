use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct GpuDevice {
    pub card: String,
    pub device_path: PathBuf,
    pub vendor_id: u32,
    pub device_id: u32,
    pub hwmon_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct GpuTelemetry {
    pub vram_total_bytes: u64,
    pub vram_used_bytes: u64,
    pub gtt_total_bytes: u64,
    pub gtt_used_bytes: u64,
    pub gpu_busy_percent: u8,
    pub power_watts: f64,
    pub temp_celsius: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct HostMemory {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CpuInfo {
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub logical_cpus: usize,
}

fn read_trim(p: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

fn read_u64(p: impl AsRef<Path>) -> Option<u64> {
    read_trim(p).and_then(|s| s.parse().ok())
}

fn read_hex(p: impl AsRef<Path>) -> Option<u32> {
    read_trim(p).and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
}

/// Finds the first AMD (vendor 0x1002) DRM card. Never hardcodes `card1`.
pub fn discover_amd_gpu() -> Option<GpuDevice> {
    discover_amd_gpu_in(Path::new("/sys/class/drm"))
}

pub fn discover_amd_gpu_in(root: &Path) -> Option<GpuDevice> {
    let mut cards: Vec<_> = fs::read_dir(root)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with("card") && n[4..].chars().all(|c| c.is_ascii_digit()))
        .collect();
    cards.sort();
    for card in cards {
        let dev = root.join(&card).join("device");
        if read_hex(dev.join("vendor")) == Some(0x1002) && dev.join("mem_info_vram_total").exists() {
            let hwmon_path = fs::read_dir(dev.join("hwmon"))
                .ok()
                .and_then(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).min());
            return Some(GpuDevice {
                device_id: read_hex(dev.join("device")).unwrap_or(0),
                vendor_id: 0x1002,
                card,
                device_path: dev,
                hwmon_path,
            });
        }
    }
    None
}

impl GpuDevice {
    pub fn telemetry(&self) -> GpuTelemetry {
        let d = &self.device_path;
        let (mut power, mut temp) = (0.0, 0.0);
        if let Some(h) = &self.hwmon_path {
            power = read_u64(h.join("power1_average"))
                .or_else(|| read_u64(h.join("power1_input")))
                .unwrap_or(0) as f64
                / 1_000_000.0;
            temp = read_u64(h.join("temp1_input")).unwrap_or(0) as f64 / 1000.0;
        }
        GpuTelemetry {
            vram_total_bytes: read_u64(d.join("mem_info_vram_total")).unwrap_or(0),
            vram_used_bytes: read_u64(d.join("mem_info_vram_used")).unwrap_or(0),
            gtt_total_bytes: read_u64(d.join("mem_info_gtt_total")).unwrap_or(0),
            gtt_used_bytes: read_u64(d.join("mem_info_gtt_used")).unwrap_or(0),
            gpu_busy_percent: read_u64(d.join("gpu_busy_percent")).unwrap_or(0).min(100) as u8,
            power_watts: power,
            temp_celsius: temp,
        }
    }
}

pub fn parse_meminfo(text: &str) -> HostMemory {
    let mut m = HostMemory::default();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match (it.next(), it.next().and_then(|v| v.parse::<u64>().ok())) {
            (Some("MemTotal:"), Some(kb)) => m.total_bytes = kb * 1024,
            (Some("MemAvailable:"), Some(kb)) => m.available_bytes = kb * 1024,
            _ => {}
        }
    }
    m
}

pub fn host_memory() -> HostMemory {
    fs::read_to_string("/proc/meminfo").map(|t| parse_meminfo(&t)).unwrap_or_default()
}

pub fn cpu_info() -> CpuInfo {
    #[cfg(target_arch = "x86_64")]
    let (avx2, fma, f16c) = (
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("f16c"),
    );
    #[cfg(not(target_arch = "x86_64"))]
    let (avx2, fma, f16c) = (false, false, false);
    CpuInfo {
        avx2,
        fma,
        f16c,
        logical_cpus: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn meminfo_parses() {
        let m = parse_meminfo("MemTotal:       1000 kB\nMemFree: 5 kB\nMemAvailable:    400 kB\n");
        assert_eq!(m.total_bytes, 1000 * 1024);
        assert_eq!(m.available_bytes, 400 * 1024);
    }
    #[test]
    fn discovery_on_fake_tree() {
        let root = std::env::temp_dir().join(format!("ocak-sysfs-test-{}", std::process::id()));
        let dev = root.join("card7/device");
        fs::create_dir_all(dev.join("hwmon/hwmon3")).unwrap();
        fs::write(dev.join("vendor"), "0x1002\n").unwrap();
        fs::write(dev.join("device"), "0x73df\n").unwrap();
        fs::write(dev.join("mem_info_vram_total"), "1000\n").unwrap();
        fs::write(dev.join("mem_info_vram_used"), "100\n").unwrap();
        fs::write(dev.join("hwmon/hwmon3/power1_average"), "29000000\n").unwrap();
        fs::write(dev.join("hwmon/hwmon3/temp1_input"), "45000\n").unwrap();
        let g = discover_amd_gpu_in(&root).unwrap();
        assert_eq!(g.card, "card7");
        assert_eq!(g.device_id, 0x73df);
        let t = g.telemetry();
        assert_eq!(t.vram_total_bytes, 1000);
        assert!((t.power_watts - 29.0).abs() < 1e-9);
        assert!((t.temp_celsius - 45.0).abs() < 1e-9);
        let _ = fs::remove_dir_all(root);
    }
}
