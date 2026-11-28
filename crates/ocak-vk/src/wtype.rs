//! GPU weight formats.
use ocak_gguf::GgmlType;
use crate::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ1S,
    IQ1M,
    IQ4NL,
    IQ4XS,
    NVFP4,
}

impl WType {
    pub fn from_ggml(t: GgmlType) -> Result<Self, VkError> {
        Ok(match t {
            GgmlType::F32 => WType::F32,
            GgmlType::F16 => WType::F16,
            GgmlType::BF16 => WType::BF16,
            GgmlType::Q8_0 => WType::Q8_0,
            GgmlType::Q4_0 => WType::Q4_0,
            GgmlType::Q2K => WType::Q2K,
            GgmlType::Q3K => WType::Q3K,
            GgmlType::Q4K => WType::Q4K,
            GgmlType::Q5K => WType::Q5K,
            GgmlType::Q6K => WType::Q6K,
            GgmlType::IQ2XXS => WType::IQ2XXS,
            GgmlType::IQ2XS => WType::IQ2XS,
            GgmlType::IQ2S => WType::IQ2S,
            GgmlType::IQ3XXS => WType::IQ3XXS,
            GgmlType::IQ3S => WType::IQ3S,
            GgmlType::IQ1S => WType::IQ1S,
            GgmlType::IQ1M => WType::IQ1M,
            GgmlType::IQ4NL => WType::IQ4NL,
            GgmlType::IQ4XS => WType::IQ4XS,
            GgmlType::NVFP4 => WType::NVFP4,
            o => return Err(VkError::UnsupportedType(o.name())),
        })
    }

    pub fn ggml(self) -> GgmlType {
        match self {
            WType::F32 => GgmlType::F32,
            WType::F16 => GgmlType::F16,
            WType::BF16 => GgmlType::BF16,
            WType::Q8_0 => GgmlType::Q8_0,
            WType::Q4_0 => GgmlType::Q4_0,
            WType::Q2K => GgmlType::Q2K,
            WType::Q3K => GgmlType::Q3K,
            WType::Q4K => GgmlType::Q4K,
            WType::Q5K => GgmlType::Q5K,
            WType::Q6K => GgmlType::Q6K,
            WType::IQ2XXS => GgmlType::IQ2XXS,
            WType::IQ2XS => GgmlType::IQ2XS,
            WType::IQ2S => GgmlType::IQ2S,
            WType::IQ3XXS => GgmlType::IQ3XXS,
            WType::IQ3S => GgmlType::IQ3S,
            WType::IQ1S => GgmlType::IQ1S,
            WType::IQ1M => GgmlType::IQ1M,
            WType::IQ4NL => GgmlType::IQ4NL,
            WType::IQ4XS => GgmlType::IQ4XS,
            WType::NVFP4 => GgmlType::NVFP4,
        }
    }

    pub fn row_bytes(self, k: u32) -> u32 {
        self.ggml().row_bytes(k as usize).expect("k not aligned to block size") as u32
    }
}
