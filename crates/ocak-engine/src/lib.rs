//! Native transformer engine: llama/qwen2 dense GGUF models on Vulkan or CPU.
pub mod asr;
pub mod config;
pub mod cpu_exec;
pub mod engine;
pub mod exec;
#[cfg(feature = "hip")]
pub mod hip_exec;
#[cfg(feature = "vulkan")]
pub mod inkling_exec;
pub mod mtp;
pub mod sampler;
pub mod xing_exec;
#[cfg(feature = "vulkan")]
pub mod vk_exec;

pub use config::ModelConfig;
pub use engine::{ActiveSlot, BackendKind, EngineOptions, FinishReason, GenRequest, GenResult, NativeEngine, ThinkingCfg, Timings};
pub use exec::{EngineError, Executor, Out, SlotStep};
pub use sampler::{Sampler, SamplerParams};
