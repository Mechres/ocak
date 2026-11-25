//! Native transformer engine: llama/qwen2 dense GGUF models on Vulkan or CPU.
pub mod config;
pub mod cpu_exec;
pub mod engine;
pub mod exec;
pub mod sampler;
pub mod xing_exec;
#[cfg(feature = "vulkan")]
pub mod vk_exec;

pub use config::ModelConfig;
pub use engine::{BackendKind, EngineOptions, FinishReason, GenRequest, GenResult, NativeEngine, ThinkingCfg, Timings};
pub use exec::{EngineError, Executor, Out};
pub use sampler::{Sampler, SamplerParams};
