//! Linux hardware telemetry (amdgpu sysfs, /proc/meminfo, CPU features) and
//! the VRAM/power residency governor.
pub mod amdgpu;
pub mod governor;
pub use amdgpu::{cpu_info, host_memory, discover_amd_gpu, CpuInfo, GpuDevice, GpuTelemetry, HostMemory};
pub use governor::{Admission, ModelResidencyState, ResidentModel, VramBudgetConfig, VramGovernor};
