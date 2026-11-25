use crate::exec::EngineError;
use ocak_gguf::GgufFile;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerKind {
    FullAttn {
        n_head_kv: usize,
        gated_q: bool,
        qk_norm: bool,
        /// Index among attention layers (for compact KV buffer allocation).
        kv_layer_idx: usize,
        sliding_window: usize,
        rope_base: f32,
        use_rope_factors: bool,
        head_gate: bool,
    },
    ShortConv {
        l_cache: usize,
    },
    DeltaNet {
        n_k_heads: usize,
        n_v_heads: usize,
        state_dim: usize,
        k_conv: usize,
    },
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,
    pub n_layer: usize,
    pub n_kv_layers: usize,
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
    /// 0 = adjacent pairs (llama), 1 = split halves (neox, qwen2, lfm2, qwen35)
    pub rope_mode: u32,
    pub qkv_bias: bool,
    pub tied_output: bool,
    pub output_norm_name: String,
    pub ffn_norm_suffix: String,
    pub layer_kinds: Vec<LayerKind>,
    pub is_hybrid: bool,
    pub has_post_norms: bool,
}

impl ModelConfig {
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    pub fn supports_native_gguf(g: &GgufFile) -> bool {
        Self::from_gguf(g).is_ok()
    }

    pub fn from_gguf(g: &GgufFile) -> Result<Self, EngineError> {
        let arch = g.architecture().ok_or_else(|| EngineError::Unsupported("no architecture".into()))?.to_string();
        if !matches!(arch.as_str(), "llama" | "qwen2" | "lfm2" | "lfm3" | "qwen35") {
            return Err(EngineError::Unsupported(format!(
                "architecture '{arch}' (native engine supports llama, qwen2, lfm2, qwen35)"
            )));
        }
        let req = |s: &str| g.arch_u64(s).ok_or_else(|| EngineError::Unsupported(format!("missing {arch}.{s}")));
        let block_count = req("block_count")? as usize;
        let nextn = g.arch_u64("nextn_predict_layers").unwrap_or(0) as usize;
        let n_layer = block_count.saturating_sub(nextn);
        let n_head = req("attention.head_count")? as usize;
        let n_embd = req("embedding_length")? as usize;
        let head_dim = g.arch_u64("attention.key_length").map(|v| v as usize).unwrap_or(n_embd / n_head);
        let embd = g.tensor("token_embd.weight").ok_or_else(|| EngineError::Unsupported("no token_embd.weight".into()))?;

        // Per-layer head_count_kv (scalar or int array for lfm2)
        let kv_arr = g.ints(&format!("{arch}.attention.head_count_kv"));
        let kv_scalar = g.arch_u64("attention.head_count_kv").map(|v| v as usize).unwrap_or(n_head);

        let mut layer_kinds = Vec::with_capacity(n_layer);
        let mut n_kv_layers = 0usize;
        let mut n_head_kv_max = 0usize;
        let mut is_hybrid = false;

        let rope_base_main = g.arch_f64("rope.freq_base").unwrap_or(10000.0) as f32;
        let rope_base_swa = g.arch_f64("rope.freq_base_swa").map(|v| v as f32).unwrap_or(rope_base_main);
        let swa_win = g.arch_u64("attention.sliding_window").unwrap_or(0) as usize;
        let swa_pat = g.ints(&format!("{arch}.attention.sliding_window_pattern"));
        let has_rope_factors = g.tensor("rope_freqs.weight").is_some();
        let has_post_norms = arch == "lfm3" && g.tensor("blk.0.post_attention_norm.weight").is_some();
        if arch == "lfm2" || arch == "lfm3" {
            let l_cache = g.arch_u64("shortconv.l_cache").unwrap_or(3) as usize;
            for i in 0..n_layer {
                let hkv = kv_arr.and_then(|a| a.get(i)).map(|&v| v as usize).unwrap_or(0);
                if hkv == 0 {
                    is_hybrid = true;
                    layer_kinds.push(LayerKind::ShortConv { l_cache });
                } else {
                    n_head_kv_max = n_head_kv_max.max(hkv);
                    let is_swa = swa_pat.and_then(|a| a.get(i)).map(|&v| v != 0).unwrap_or(false);
                    layer_kinds.push(LayerKind::FullAttn {
                        n_head_kv: hkv,
                        gated_q: false,
                        qk_norm: g.tensor(&format!("blk.{i}.attn_q_norm.weight")).is_some(),
                        kv_layer_idx: n_kv_layers,
                        sliding_window: if is_swa { swa_win } else { 0 },
                        rope_base: if is_swa { rope_base_swa } else { rope_base_main },
                        use_rope_factors: !is_swa && has_rope_factors,
                        head_gate: g.tensor(&format!("blk.{i}.attn_gate.weight")).is_some(),
                    });
                    n_kv_layers += 1;
                }
            }
        } else if arch == "qwen35" {
            let interval = g.arch_u64("full_attention_interval").unwrap_or(4) as usize;
            let k_conv = g.arch_u64("ssm.conv_kernel").unwrap_or(4) as usize;
            let state_dim = g.arch_u64("ssm.state_size").unwrap_or(128) as usize;
            let n_k_heads = g.arch_u64("ssm.group_count").unwrap_or(16) as usize;
            let n_v_heads = g.arch_u64("ssm.time_step_rank").unwrap_or(32) as usize;
            for i in 0..n_layer {
                let is_full = (i + 1) % interval == 0;
                if is_full {
                    n_head_kv_max = n_head_kv_max.max(kv_scalar);
                    layer_kinds.push(LayerKind::FullAttn {
                        n_head_kv: kv_scalar,
                        gated_q: true,
                        qk_norm: true,
                        kv_layer_idx: n_kv_layers,
                        sliding_window: 0,
                        rope_base: rope_base_main,
                        use_rope_factors: false,
                        head_gate: false,
                    });
                    n_kv_layers += 1;
                } else {
                    is_hybrid = true;
                    layer_kinds.push(LayerKind::DeltaNet { n_k_heads, n_v_heads, state_dim, k_conv });
                }
            }
        } else {
            n_head_kv_max = kv_scalar;
            for i in 0..n_layer {
                layer_kinds.push(LayerKind::FullAttn {
                    n_head_kv: kv_scalar,
                    gated_q: false,
                    qk_norm: g.tensor(&format!("blk.{i}.attn_q_norm.weight")).is_some(),
                    kv_layer_idx: i,
                    sliding_window: 0,
                    rope_base: rope_base_main,
                    use_rope_factors: false,
                    head_gate: false,
                });
            }
            n_kv_layers = n_layer;
        }

        let output_norm_name = if g.tensor("output_norm.weight").is_some() {
            "output_norm.weight".to_string()
        } else if g.tensor("token_embd_norm.weight").is_some() {
            "token_embd_norm.weight".to_string()
        } else {
            return Err(EngineError::Unsupported("missing output_norm/token_embd_norm".into()));
        };
        let ffn_norm_suffix = if arch == "qwen35" {
            "post_attention_norm.weight".to_string()
        } else {
            "ffn_norm.weight".to_string()
        };

        let cfg = ModelConfig {
            n_layer,
            n_kv_layers: n_kv_layers.max(1),
            n_embd,
            n_ff: req("feed_forward_length")? as usize,
            n_head,
            n_head_kv: n_head_kv_max.max(1),
            head_dim,
            n_rot: g.arch_u64("rope.dimension_count").map(|v| v as usize).unwrap_or(head_dim),
            rope_base: g.arch_f64("rope.freq_base").unwrap_or(10000.0) as f32,
            eps: g.arch_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f32,
            n_ctx_train: g.arch_u64("context_length").unwrap_or(2048) as usize,
            n_vocab: embd.n_rows(),
            rope_mode: if arch == "llama" { 0 } else { 1 },
            qkv_bias: g.tensor("blk.0.attn_q.bias").is_some(),
            tied_output: g.tensor("output.weight").is_none(),
            output_norm_name,
            ffn_norm_suffix,
            layer_kinds,
            is_hybrid,
            has_post_norms,
            arch,
        };

        if g.arch_f64("rope.scaling.factor").map_or(false, |f| (f - 1.0).abs() > 1e-6) {
            return Err(EngineError::Unsupported("rope scaling".into()));
        }

        // Check all trunk 2D weight tensor types are supported by our kernels
        for t in &g.tensors {
            if t.n_dims == 2 {
                // Skip MTP extra layers (blk.{n_layer}..)
                if let Some(rest) = t.name.strip_prefix("blk.") {
                    if let Some((num, _)) = rest.split_once('.') {
                        if num.parse::<usize>().map_or(false, |idx| idx >= n_layer) {
                            continue;
                        }
                    }
                }
                if t.name.starts_with("blk.") || t.name == "output.weight" || t.name == "token_embd.weight" {
                    if !ocak_cpu::gemv::supported(t.ty) {
                        return Err(EngineError::Unsupported(format!("tensor {} has type {}", t.name, t.ty.name())));
                    }
                }
            }
        }
        Ok(cfg)
    }
}
