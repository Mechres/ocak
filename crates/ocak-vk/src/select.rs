//! Kernel-selection helpers, tuning knobs (environment variables read once) and tile sizes.
use crate::*;

/// Weight formats with a tiled prefill matmul (`qmm_tiled.comp`), in `Pipelines::qmmt` order.
pub(crate) fn qmmt_index(ty: WType) -> usize {
    match ty {
        WType::Q4K => 0,
        WType::Q5K => 1,
        WType::Q6K => 2,
        WType::Q2K => 3,
        WType::Q3K => 4,
        WType::Q8_0 => 5,
        WType::Q4_0 => 6,
        WType::F16 => 7,
        WType::F32 => 8,
        WType::IQ3S => 9,
        WType::IQ3XXS => 10,
        WType::IQ2XS => 11,
        WType::IQ2S => 12,
        WType::IQ2XXS => 13,
        WType::IQ1S => 14,
        WType::IQ1M => 15,
        WType::IQ4XS => 16,
        WType::IQ4NL => 17,
        WType::BF16 => 18,
        WType::NVFP4 => 19,
    }
}

/// Formats with an integer-dot (int8 activations) prefill matmul.
pub(crate) fn qmmq_index(ty: WType) -> Option<usize> {
    match ty {
        WType::Q4K => Some(0),
        WType::Q5K => Some(1),
        WType::Q6K => Some(2),
        WType::Q8_0 => Some(3),
        WType::Q2K => Some(4),
        WType::Q3K => Some(5),
        WType::IQ4NL => Some(6),
        WType::IQ4XS => Some(7),
        WType::IQ3S => Some(8),
        WType::IQ3XXS => Some(9),
        WType::IQ2XS => Some(10),
        WType::IQ2S => Some(11),
        WType::IQ2XXS => Some(12),
        WType::IQ1S => Some(13),
        WType::IQ1M => Some(14),
        WType::NVFP4 => Some(15),
        _ => None,
    }
}

/// Integer-dot prefill matmuls are on unless OCAK_MMQ=0.
/// `OCAK_INKLING_ATTN_OLD=1`: the pre-split-K Inkling decode attention kernel (read once).
pub(crate) fn inkling_attn_old() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OCAK_INKLING_ATTN_OLD").is_ok())
}

pub(crate) fn mmq_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OCAK_MMQ").map_or(true, |v| v != "0"))
}

/// Integer-dot decode matvec (qmv_q8.comp) unless OCAK_QMV_INT8=0.
pub(crate) fn qmv_int8_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OCAK_QMV_INT8").map_or(true, |v| v != "0"))
}

/// Device scratch for int8-quantized activations (bump-allocated per recording).
pub(crate) const Q8_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;

/// Whether batched (prefill) matmuls of this weight format use a tiled kernel. Formats without
/// one fall back to a per-token matvec loop, so callers should keep prefill chunks small.
pub fn has_tiled_qmm(ty: WType) -> bool {
    let _ = qmmt_index(ty); // every format has one
    true
}

/// Tile shape of qmm_tiled.comp (16*TM rows x 16*TN tokens); keep in sync with the shader defaults.
pub(crate) const QMMT_BM: u32 = 64;
pub(crate) const QMMT_BN: u32 = 128;
/// Column tile of the grouped variant (16 * TN with -DGROUPED).
pub(crate) const QMMG_BN: u32 = 32;
/// Tokens per workgroup in mla_attn_prefill.comp (its TOKS).
pub(crate) const MLA_PREFILL_TOKS: u32 = 2;

/// Column tile of the grouped integer-dot kernel (qmm_q8.comp -DGROUPED: TN = 4).
pub(crate) const QMMQG_BN: u32 = 64;

/// Whether the generic multi-token matvec (qmv_multi.comp) is faster than the hand-written
/// single-token qmv shader for this format at batch 1. Measured on RX 6700 XT end to end
/// (decode tok/s): big wins for Q8_0 / F16 / IQ3 / IQ1-2 / Q2_K / Q3_K / IQ4 (Qwen3.8-27B Q2_K
/// 9.3 -> 16.7 tok/s), small losses for Q4_K / Q5_K / Q6_K / NVFP4 / IQ2_XS.
pub(crate) fn qmvm_beats_qmv(ty: WType) -> bool {
    !matches!(ty, WType::Q4K | WType::Q5K | WType::Q6K | WType::NVFP4 | WType::IQ2XS)
}

/// Opt-in coarser prefill activation quantization (env `OCAK_PREFILL_Q256=1`): one int8 scale
/// per 256 values instead of 32 for the batched int8 matmuls (lossy; faster fix-up).
pub(crate) fn prefill_q256() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OCAK_PREFILL_Q256").is_ok_and(|v| v != "0"))
}

/// Smallest batch that uses the tiled prefill matmul (env `OCAK_QMMT_MIN`, default 32).
pub(crate) fn qmmt_min_batch() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OCAK_QMMT_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(32))
}
