//! Zero-copy GGUF reader, quantization reference math, `.digest` and preset parsing.
pub mod catalog;
pub mod digest;
pub mod gguf;
pub mod iq_tables;
pub mod preset;
pub mod quant;
pub use gguf::{ArrayValue, GgufError, GgufFile, TensorInfo, Value};
pub use quant::GgmlType;
