use crate::exec::EngineError;
use ocak_gguf::{GgmlType, GgufFile};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub n_ctx_train: usize,
    pub n_vocab: usize,
    /// 0 = adjacent pairs (llama), 1 = split halves (neox, qwen2)
    pub rope_mode: u32,
    pub qkv_bias: bool,
    pub tied_output: bool,
}

impl ModelConfig {
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    pub fn from_gguf(g: &GgufFile) -> Result<Self, EngineError> {
        let arch = g.architecture().ok_or_else(|| EngineError::Unsupported("no architecture".into()))?.to_string();
        if arch != "llama" && arch != "qwen2" {
            return Err(EngineError::Unsupported(format!("architecture '{arch}' (native engine supports llama, qwen2)")));
        }
        let req = |s: &str| g.arch_u64(s).ok_or_else(|| EngineError::Unsupported(format!("missing {arch}.{s}")));
        let n_head = req("attention.head_count")? as usize;
        let n_embd = req("embedding_length")? as usize;
        let head_dim = g.arch_u64("attention.key_length").map(|v| v as usize).unwrap_or(n_embd / n_head);
        let embd = g.tensor("token_embd.weight").ok_or_else(|| EngineError::Unsupported("no token_embd.weight".into()))?;
        let cfg = ModelConfig {
            n_layer: req("block_count")? as usize,
            n_embd,
            n_ff: req("feed_forward_length")? as usize,
            n_head,
            n_head_kv: g.arch_u64("attention.head_count_kv").map(|v| v as usize).unwrap_or(n_head),
            head_dim,
            n_rot: g.arch_u64("rope.dimension_count").map(|v| v as usize).unwrap_or(head_dim),
            rope_base: g.arch_f64("rope.freq_base").unwrap_or(10000.0) as f32,
            eps: g.arch_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f32,
            n_ctx_train: g.arch_u64("context_length").unwrap_or(2048) as usize,
            n_vocab: embd.n_rows(),
            rope_mode: if arch == "llama" { 0 } else { 1 },
            qkv_bias: g.tensor("blk.0.attn_q.bias").is_some(),
            tied_output: g.tensor("output.weight").is_none(),
            arch,
        };
        if g.get_str("rope.scaling.type").is_some() || g.arch_f64("rope.scaling.factor").is_some() {
            // linear/yarn scaling is not implemented yet; refuse rather than silently diverge
            if g.arch_f64("rope.scaling.factor").map_or(false, |f| (f - 1.0).abs() > 1e-6) {
                return Err(EngineError::Unsupported("rope scaling".into()));
            }
        }
        // check all 2D weight types are supported by the CPU kernels (Vulkan supports the same set)
        for t in &g.tensors {
            if t.n_dims == 2 && t.name.starts_with("blk.") || t.name == "output.weight" || t.name == "token_embd.weight" {
                if !ocak_cpu::gemv::supported(t.ty) {
                    return Err(EngineError::Unsupported(format!("tensor {} has type {}", t.name, t.ty.name())));
                }
            }
        }
        let _ = GgmlType::F32;
        Ok(cfg)
    }
}
