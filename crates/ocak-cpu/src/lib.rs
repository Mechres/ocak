//! CPU kernels (AVX2/FMA auto-vectorised) for quantised GEMV and transformer ops.
pub mod gemv;
pub mod ops;
pub use gemv::{gemv, row_dot};
