//! Native hybrid GPU (Vulkan 1.4) + CPU (AVX2) executor for `xing4_0` (29B-A4B MoE):
//! - 4-stream Sinkhorn Hyper-Connections
//! - Multi-head Latent Attention (MLA) with YaRN RoPE & absorbed wk_b/wv_b (576-float KV/token)
//! - 64-expert / 4-active MoE + shared expert with configurable `n_cpu_moe` offload for 12 GiB VRAM
use crate::exec::{EngineError, Executor, Out};
use ocak_cpu::{gemv, ops};
use ocak_gguf::quant::dequantize_row;
use ocak_gguf::{GgufFile, TensorInfo};
#[cfg(feature = "vulkan")]
use ocak_vk::{Buffer, VkContext, WType};
use rayon::prelude::*;
use std::sync::Arc;

const XING_PREFILL_BATCH: usize = 1024;
/// Layers per queue submit in batched prefill (keeps each submit far below the amdgpu TDR).
#[cfg(feature = "vulkan")]
const XING_LAYERS_PER_SUBMIT: usize = 8;

#[derive(Debug, Clone)]
pub struct XingConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff_dense: usize,
    pub n_ff_exp: usize,
    pub n_head: usize,
    pub q_rank: usize,
    pub kv_rank: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub n_rot: usize,
    pub n_expert: usize,
    pub n_used: usize,
    pub exp_weights_scale: f32,
    pub exp_weights_norm: bool,
    pub hc_count: usize,
    pub hc_iters: usize,
    pub hc_eps: f32,
    pub eps: f32,
    pub rope_base: f32,
    pub rope_factor: f32,
    pub rope_orig_ctx: usize,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_log_mul: f32,
    pub n_vocab: usize,
    pub n_ctx_train: usize,
}

impl XingConfig {
    pub fn from_gguf(g: &GgufFile) -> Result<Self, EngineError> {
        let arch = g.architecture().unwrap_or("");
        if arch != "xing4_0" {
            return Err(EngineError::Unsupported(format!("expected xing4_0, got {arch}")));
        }
        let u = |k: &str, d: u64| g.arch_u64(k).unwrap_or(d) as usize;
        let f = |k: &str, d: f64| g.arch_f64(k).unwrap_or(d) as f32;
        let block_count = u("block_count", 40);
        let nextn_meta = u("nextn_predict_layers", 0);
        // Some GGUFs strip the NextN/MTP block but keep `nextn_predict_layers`; only treat the
        // trailing blocks as MTP if they actually carry `nextn.*` tensors.
        let nextn = (0..nextn_meta.min(block_count))
            .take_while(|i| g.tensor(&format!("blk.{}.nextn.eh_proj.weight", block_count - 1 - i)).is_some())
            .count();
        let embd = g.tensor("token_embd.weight").ok_or_else(|| EngineError::Unsupported("missing token_embd".into()))?;
        Ok(Self {
            n_layer: block_count.saturating_sub(nextn),
            n_embd: u("embedding_length", 3584),
            n_ff_dense: u("feed_forward_length", 9216),
            n_ff_exp: u("expert_feed_forward_length", 1024),
            n_head: u("attention.head_count", 32),
            q_rank: u("attention.q_lora_rank", 768),
            kv_rank: u("attention.kv_lora_rank", 512),
            key_dim: u("attention.key_length_mla", 192),
            value_dim: u("attention.value_length_mla", 128),
            n_rot: u("rope.dimension_count", 64),
            n_expert: u("expert_count", 64),
            n_used: u("expert_used_count", 4),
            exp_weights_scale: f("expert_weights_scale", 2.0),
            exp_weights_norm: g.get_bool("xing4_0.expert_weights_norm").unwrap_or(true),
            hc_count: u("hyper_connection.count", 4),
            hc_iters: u("hyper_connection.sinkhorn_iterations", 20),
            hc_eps: f("hyper_connection.epsilon", 1e-6),
            eps: f("attention.layer_norm_rms_epsilon", 1e-6),
            rope_base: f("rope.freq_base", 10000.0),
            rope_factor: f("rope.scaling.factor", 64.0),
            rope_orig_ctx: u("rope.scaling.original_context_length", 4096),
            yarn_beta_fast: f("rope.scaling.yarn_beta_fast", 32.0),
            yarn_beta_slow: f("rope.scaling.yarn_beta_slow", 1.0),
            // The HF->GGUF converter stores yarn_log_multiplier pre-multiplied by 0.1; cancel it
            // (matches llama.cpp TAG_DEEPSEEK2_YARN_LOG_MUL_FIX), otherwise kq_scale is ~1.85x too small.
            yarn_log_mul: f("rope.scaling.yarn_log_multiplier", 0.1) / 0.1,
            n_vocab: embd.n_rows(),
            n_ctx_train: u("context_length", 262144),
        })
    }
}

fn req(g: &GgufFile, name: &str) -> Result<TensorInfo, EngineError> {
    g.tensor(name).cloned().ok_or_else(|| EngineError::Unsupported(format!("missing {name}")))
}

fn f32v(g: &GgufFile, name: &str) -> Result<Vec<f32>, EngineError> {
    Ok(g.tensor_to_f32(&req(g, name)?))
}

#[cfg(feature = "vulkan")]
struct VkMat {
    buf: Buffer,
    ty: WType,
    k: u32,
    m: u32,
    slice_bytes: u64,
    /// Lives in host-visible system memory (GTT) because VRAM ran out.
    in_gtt: bool,
}

#[cfg(feature = "vulkan")]
fn try_upload_mat(ctx: &Arc<VkContext>, g: &GgufFile, ti: &TensorInfo) -> Option<VkMat> {
    let ty = WType::from_ggml(ti.ty).ok()?;
    let raw = g.tensor_data(ti);
    let buf = ctx.create_device_buffer(raw.len() as u64).ok()?;
    ctx.upload(&buf, 0, raw).ok()?;
    let k = ti.ne[0] as u32;
    let m = ti.ne[1] as u32;
    let slice_bytes = (ti.row_bytes() * m as usize) as u64;
    Some(VkMat { buf, ty, k, m, slice_bytes, in_gtt: false })
}

#[cfg(feature = "vulkan")]
fn try_upload_mat_gtt(ctx: &Arc<VkContext>, g: &GgufFile, ti: &TensorInfo) -> Option<VkMat> {
    let ty = WType::from_ggml(ti.ty).ok()?;
    let raw = g.tensor_data(ti);
    let buf = ctx.create_gtt_buffer_from_slice(raw).ok()?;
    let k = ti.ne[0] as u32;
    let m = ti.ne[1] as u32;
    let slice_bytes = (ti.row_bytes() * m as usize) as u64;
    Some(VkMat { buf, ty, k, m, slice_bytes, in_gtt: true })
}

#[cfg(feature = "vulkan")]
fn upload_f32_buf(ctx: &Arc<VkContext>, data: &[f32]) -> Option<Buffer> {
    let bytes = bytemuck::cast_slice(data);
    let buf = ctx.create_device_buffer(bytes.len() as u64).ok()?;
    ctx.upload(&buf, 0, bytes).ok()?;
    Some(buf)
}

#[cfg(feature = "vulkan")]
struct XingGpuLayer {
    hc_attn_fn: VkMat,
    hc_attn_params: Buffer,
    hc_ffn_fn: VkMat,
    hc_ffn_params: Buffer,
    attn_norm: Buffer,
    ffn_norm: Buffer,
    q_a_norm: Buffer,
    kv_a_norm: Buffer,
    gate_inp: Option<VkMat>,
    exp_probs_b: Option<Buffer>,
}

#[cfg(feature = "vulkan")]
struct XingGpuState {
    layers: Vec<XingGpuLayer>,
    ones: Buffer,
    streams_a: Buffer,
    streams_b: Buffer,
    streams_norm: Buffer,
    hc_mixes: Buffer,
    hc_out: Buffer,
    inp: Buffer,
    normed: Buffer,
    block_out: Buffer,
    q_a: Buffer,
    q_an: Buffer,
    kv_raw: Buffer,
    q_full: Buffer,
    q_lat: Buffer,
    attn_lat: Buffer,
    val_all: Buffer,
    kv_cache: Buffer,
    router_logits: Buffer,
    top_ids: Buffer,
    /// Identity ids 0..n_head: the per-head absorbed projections (wk_b / wv_b) run as one indexed
    /// matvec over the head slices instead of n_head dispatches each.
    head_ids: Buffer,
    top_w: Buffer,
    dummy_bias: Buffer,
    out_norm: Buffer,
    hidden_buf: Buffer,
    logits_buf: Buffer,
    /// Device-side greedy pick (`forward_argmax`).
    argmax_buf: Buffer,
    moe_tmp: Buffer,
    /// Per-expert entry lists for the grouped MoE matmul (moe_group.comp layout).
    moe_list: Buffer,
}

struct HcWeights {
    fn_w: TensorInfo,
    scale: Vec<f32>,
    base: Vec<f32>,
}

#[allow(clippy::large_enum_variant)]
enum XingFfn {
    Dense {
        gate: TensorInfo,
        up: TensorInfo,
        down: TensorInfo,
        #[cfg(feature = "vulkan")]
        vk_gate: Option<VkMat>,
        #[cfg(feature = "vulkan")]
        vk_up: Option<VkMat>,
        #[cfg(feature = "vulkan")]
        vk_down: Option<VkMat>,
    },
    Moe {
        gate_inp: TensorInfo,
        exp_probs_b: Option<Vec<f32>>,
        gate_exps: TensorInfo,
        up_exps: TensorInfo,
        down_exps: TensorInfo,
        #[cfg(feature = "vulkan")]
        vk_gate_exps: Option<VkMat>,
        #[cfg(feature = "vulkan")]
        vk_up_exps: Option<VkMat>,
        #[cfg(feature = "vulkan")]
        vk_down_exps: Option<VkMat>,
        gate_shexp: TensorInfo,
        up_shexp: TensorInfo,
        down_shexp: TensorInfo,
        #[cfg(feature = "vulkan")]
        vk_gate_sh: Option<VkMat>,
        #[cfg(feature = "vulkan")]
        vk_up_sh: Option<VkMat>,
        #[cfg(feature = "vulkan")]
        vk_down_sh: Option<VkMat>,
    },
}

struct XingLayer {
    hc_attn: HcWeights,
    hc_ffn: HcWeights,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    wq_a: TensorInfo,
    q_a_norm: Vec<f32>,
    wq_b: TensorInfo,
    wkv_a: TensorInfo,
    kv_a_norm: Vec<f32>,
    wk_b: TensorInfo,
    wv_b: TensorInfo,
    wo: TensorInfo,
    #[cfg(feature = "vulkan")]
    vk_wq_a: Option<VkMat>,
    #[cfg(feature = "vulkan")]
    vk_wkv_a: Option<VkMat>,
    #[cfg(feature = "vulkan")]
    vk_wq_b: Option<VkMat>,
    #[cfg(feature = "vulkan")]
    vk_wk_b: Option<VkMat>,
    #[cfg(feature = "vulkan")]
    vk_wv_b: Option<VkMat>,
    #[cfg(feature = "vulkan")]
    vk_wo: Option<VkMat>,
    ffn: XingFfn,
}

pub struct XingExec {
    g: Arc<GgufFile>,
    pub cfg: XingConfig,
    layers: Vec<XingLayer>,
    embd: TensorInfo,
    out_norm: Vec<f32>,
    output: TensorInfo,
    #[cfg(feature = "vulkan")]
    vk: Option<Arc<VkContext>>,
    #[cfg(feature = "vulkan")]
    vk_x: Option<Buffer>,
    #[cfg(feature = "vulkan")]
    vk_y: Option<Buffer>,
    #[cfg(feature = "vulkan")]
    vk_g: Option<Buffer>,
    #[cfg(feature = "vulkan")]
    vk_u: Option<Buffer>,
    #[cfg(feature = "vulkan")]
    vk_a: Option<Buffer>,
    #[cfg(feature = "vulkan")]
    vk_acc: Option<Buffer>,
    #[cfg(feature = "vulkan")]
    vk_output: Option<VkMat>,
    #[cfg(feature = "vulkan")]
    gpu_pipe: Option<XingGpuState>,
    /// VRAM staging buffers (gate, up, down) for one layer's routed experts. When the model does
    /// not fit in VRAM some expert tensors live in GTT; batched prefill DMA-copies them here once
    /// per layer instead of having every matmul tile read host memory over PCIe.
    #[cfg(feature = "vulkan")]
    moe_stage: Option<[Buffer; 3]>,
    /// Compact MLA latent KV cache: [n_layer][max_ctx][kv_rank + n_rot]
    kv_cache: Vec<f32>,
    max_ctx: usize,
    gpu_moe_layers: usize,
    cpu_moe_layers: usize,
    logits: Vec<f32>,
    hidden: Vec<f32>,
}

impl XingExec {
    pub fn new(g: Arc<GgufFile>, use_vulkan: bool, max_ctx: usize, n_cpu_moe: usize) -> Result<Self, EngineError> {
        let cfg = XingConfig::from_gguf(&g)?;
        #[cfg(feature = "vulkan")]
        let vk = if use_vulkan { VkContext::new().ok() } else { None };

        // Count MoE layers so we place the first (total_moe - n_cpu_moe) MoE layers on GPU VRAM
        // and keep the last `n_cpu_moe` MoE layers zero-copy mmap'd in system RAM for Ryzen AVX2.
        let mut moe_indices = Vec::new();
        for i in 0..cfg.n_layer {
            if g.tensor(&format!("blk.{i}.ffn_gate_exps.weight")).is_some() {
                moe_indices.push(i);
            }
        }
        let gpu_moe_target = moe_indices.len().saturating_sub(n_cpu_moe);
        let mut gpu_moe_placed = 0usize;
        let mut cpu_moe_placed = 0usize;
        let mut moe_in_gtt = 0usize;
        // Reserve the expert staging buffers up front, while VRAM is still free, if the model is
        // unlikely to fit entirely (weights + ~1.5 GiB for KV cache and activations).
        #[cfg(feature = "vulkan")]
        let moe_stage: Option<[Buffer; 3]> = match &vk {
            Some(c) if g.file_len() + (3u64 << 29) > c.vram_bytes() && !moe_indices.is_empty() => {
                let max_bytes = |name: &str| {
                    moe_indices.iter().filter_map(|i| g.tensor(&format!("blk.{i}.{name}"))).map(|t| t.n_bytes).max().unwrap_or(0)
                };
                let mk = |name: &str| c.create_device_buffer(max_bytes(name).max(4)).ok();
                match (mk("ffn_gate_exps.weight"), mk("ffn_up_exps.weight"), mk("ffn_down_exps.weight")) {
                    (Some(a), Some(b), Some(d)) => Some([a, b, d]),
                    _ => None,
                }
            }
            _ => None,
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let hc_attn = HcWeights {
                fn_w: req(&g, &p("hc_attn_fn.weight"))?,
                scale: f32v(&g, &p("hc_attn_scale.weight"))?,
                base: f32v(&g, &p("hc_attn_base.weight"))?,
            };
            let hc_ffn = HcWeights {
                fn_w: req(&g, &p("hc_ffn_fn.weight"))?,
                scale: f32v(&g, &p("hc_ffn_scale.weight"))?,
                base: f32v(&g, &p("hc_ffn_base.weight"))?,
            };
            let wq_a = req(&g, &p("attn_q_a.weight"))?;
            let wkv_a = req(&g, &p("attn_kv_a_mqa.weight"))?;
            let wq_b = req(&g, &p("attn_q_b.weight"))?;
            let wk_b = req(&g, &p("attn_k_b.weight"))?;
            let wv_b = req(&g, &p("attn_v_b.weight"))?;
            let wo = req(&g, &p("attn_output.weight"))?;
            #[cfg(feature = "vulkan")]
            let (vk_wq_a, vk_wkv_a, vk_wq_b, vk_wk_b, vk_wv_b, vk_wo) = match &vk {
                Some(c) => (
                    try_upload_mat(c, &g, &wq_a).or_else(|| try_upload_mat_gtt(c, &g, &wq_a)),
                    try_upload_mat(c, &g, &wkv_a).or_else(|| try_upload_mat_gtt(c, &g, &wkv_a)),
                    try_upload_mat(c, &g, &wq_b).or_else(|| try_upload_mat_gtt(c, &g, &wq_b)),
                    try_upload_mat(c, &g, &wk_b).or_else(|| try_upload_mat_gtt(c, &g, &wk_b)),
                    try_upload_mat(c, &g, &wv_b).or_else(|| try_upload_mat_gtt(c, &g, &wv_b)),
                    try_upload_mat(c, &g, &wo).or_else(|| try_upload_mat_gtt(c, &g, &wo)),
                ),
                None => (None, None, None, None, None, None),
            };

            let ffn = if let Some(gate) = g.tensor(&p("ffn_gate.weight")).cloned() {
                let up = req(&g, &p("ffn_up.weight"))?;
                let down = req(&g, &p("ffn_down.weight"))?;
                #[cfg(feature = "vulkan")]
                let (vk_gate, vk_up, vk_down) = match &vk {
                    Some(c) => (try_upload_mat(c, &g, &gate), try_upload_mat(c, &g, &up), try_upload_mat(c, &g, &down)),
                    None => (None, None, None),
                };
                XingFfn::Dense {
                    gate, up, down,
                    #[cfg(feature = "vulkan")]
                    vk_gate,
                    #[cfg(feature = "vulkan")]
                    vk_up,
                    #[cfg(feature = "vulkan")]
                    vk_down,
                }
            } else {
                let gate_inp = req(&g, &p("ffn_gate_inp.weight"))?;
                let exp_probs_b = g.tensor(&p("exp_probs_b.bias")).map(|ti| g.tensor_to_f32(ti));
                let gate_exps = req(&g, &p("ffn_gate_exps.weight"))?;
                let up_exps = req(&g, &p("ffn_up_exps.weight"))?;
                let down_exps = req(&g, &p("ffn_down_exps.weight"))?;
                let gate_shexp = req(&g, &p("ffn_gate_shexp.weight"))?;
                let up_shexp = req(&g, &p("ffn_up_shexp.weight"))?;
                let down_shexp = req(&g, &p("ffn_down_shexp.weight"))?;
                let place_on_gpu = gpu_moe_placed < gpu_moe_target;
                #[cfg(feature = "vulkan")]
                let (vk_gate_exps, vk_up_exps, vk_down_exps, vk_gate_sh, vk_up_sh, vk_down_sh) = match &vk {
                    Some(c) => {
                        let sh = (
                            try_upload_mat(c, &g, &gate_shexp),
                            try_upload_mat(c, &g, &up_shexp),
                            try_upload_mat(c, &g, &down_shexp),
                        );
                        let ex = if place_on_gpu {
                            let v = (
                                try_upload_mat(c, &g, &gate_exps),
                                try_upload_mat(c, &g, &up_exps),
                                try_upload_mat(c, &g, &down_exps),
                            );
                            if v.0.is_some() && v.1.is_some() && v.2.is_some() {
                                v
                            } else {
                                moe_in_gtt += 1;
                                (
                                    try_upload_mat_gtt(c, &g, &gate_exps),
                                    try_upload_mat_gtt(c, &g, &up_exps),
                                    try_upload_mat_gtt(c, &g, &down_exps),
                                )
                            }
                        } else {
                            moe_in_gtt += 1;
                            (
                                try_upload_mat_gtt(c, &g, &gate_exps),
                                try_upload_mat_gtt(c, &g, &up_exps),
                                try_upload_mat_gtt(c, &g, &down_exps),
                            )
                        };
                        if ex.0.is_some() && ex.1.is_some() && ex.2.is_some() {
                            gpu_moe_placed += 1;
                            (ex.0, ex.1, ex.2, sh.0, sh.1, sh.2)
                        } else {
                            cpu_moe_placed += 1;
                            (None, None, None, sh.0, sh.1, sh.2)
                        }
                    }
                    None => {
                        cpu_moe_placed += 1;
                        (None, None, None, None, None, None)
                    }
                };
                #[cfg(not(feature = "vulkan"))]
                { cpu_moe_placed += 1; }
                XingFfn::Moe {
                    gate_inp, exp_probs_b, gate_exps, up_exps, down_exps,
                    #[cfg(feature = "vulkan")]
                    vk_gate_exps,
                    #[cfg(feature = "vulkan")]
                    vk_up_exps,
                    #[cfg(feature = "vulkan")]
                    vk_down_exps,
                    gate_shexp, up_shexp, down_shexp,
                    #[cfg(feature = "vulkan")]
                    vk_gate_sh,
                    #[cfg(feature = "vulkan")]
                    vk_up_sh,
                    #[cfg(feature = "vulkan")]
                    vk_down_sh,
                }
            };

            layers.push(XingLayer {
                hc_attn,
                hc_ffn,
                attn_norm: f32v(&g, &p("attn_norm.weight"))?,
                ffn_norm: f32v(&g, &p("ffn_norm.weight"))?,
                wq_a,
                q_a_norm: f32v(&g, &p("attn_q_a_norm.weight"))?,
                wq_b,
                wkv_a,
                kv_a_norm: f32v(&g, &p("attn_kv_a_norm.weight"))?,
                wk_b,
                wv_b,
                wo,
                #[cfg(feature = "vulkan")]
                vk_wq_a,
                #[cfg(feature = "vulkan")]
                vk_wkv_a,
                #[cfg(feature = "vulkan")]
                vk_wq_b,
                #[cfg(feature = "vulkan")]
                vk_wk_b,
                #[cfg(feature = "vulkan")]
                vk_wv_b,
                #[cfg(feature = "vulkan")]
                vk_wo,
                ffn,
            });
        }

        if moe_in_gtt > 0 || cpu_moe_placed > 0 {
            eprintln!(
                "ocak: xing MoE expert layers: {} in VRAM, {moe_in_gtt} in host memory (GTT), {cpu_moe_placed} on CPU",
                gpu_moe_placed - moe_in_gtt.min(gpu_moe_placed)
            );
        }
        let embd = req(&g, "token_embd.weight")?;
        let out_norm = f32v(&g, "output_norm.weight")?;
        let output = g.tensor("output.weight").cloned().unwrap_or_else(|| embd.clone());
        #[cfg(feature = "vulkan")]
        let (vk_x, vk_y, vk_g, vk_u, vk_a, vk_acc, vk_output) = match &vk {
            Some(c) => {
                let x_cap = cfg.n_ff_dense.max(cfg.n_embd).max(cfg.n_head * cfg.kv_rank);
                let x = c.create_device_buffer((x_cap * 4) as u64).ok();
                let y = c.create_device_buffer((cfg.n_vocab.max(cfg.n_ff_dense).max(cfg.n_head * (cfg.key_dim + cfg.kv_rank)) * 4) as u64).ok();
                let ffn_cap = cfg.n_ff_dense.max((cfg.n_used + 1) * cfg.n_ff_exp);
                let fg = c.create_device_buffer((XING_PREFILL_BATCH * ffn_cap * 4) as u64).ok();
                let fu = c.create_device_buffer((XING_PREFILL_BATCH * ffn_cap * 4) as u64).ok();
                let fa = c.create_device_buffer((XING_PREFILL_BATCH * ffn_cap * 4) as u64).ok();
                let acc = c.create_device_buffer((cfg.n_embd * 4) as u64).ok();
                let vo = try_upload_mat(c, &g, &output);
                (x, y, fg, fu, fa, acc, vo)
            }
            None => (None, None, None, None, None, None, None),
        };

        let hd = cfg.kv_rank + cfg.n_rot;
        #[cfg(feature = "vulkan")]
        let gpu_pipe = if cpu_moe_placed == 0 && vk_output.is_some() {
            if let Some(c) = &vk {
                let mut g_layers = Vec::with_capacity(cfg.n_layer);
                let mut ok = true;
                for l in &layers {
                    let mut p_attn = Vec::with_capacity(27);
                    p_attn.extend_from_slice(&l.hc_attn.scale);
                    p_attn.extend_from_slice(&l.hc_attn.base);
                    let mut p_ffn = Vec::with_capacity(27);
                    p_ffn.extend_from_slice(&l.hc_ffn.scale);
                    p_ffn.extend_from_slice(&l.hc_ffn.base);
                    let (gate_inp, exp_probs_b) = match &l.ffn {
                        XingFfn::Moe { gate_inp, exp_probs_b, .. } => (
                            try_upload_mat(c, &g, gate_inp).or_else(|| try_upload_mat_gtt(c, &g, gate_inp)),
                            exp_probs_b.as_ref().and_then(|b| upload_f32_buf(c, b)),
                        ),
                        XingFfn::Dense { .. } => (None, None),
                    };
                    if let (Some(haf), Some(hap), Some(hff), Some(hfp), Some(an), Some(fn_), Some(qan), Some(kvan)) = (
                        try_upload_mat(c, &g, &l.hc_attn.fn_w).or_else(|| try_upload_mat_gtt(c, &g, &l.hc_attn.fn_w)),
                        upload_f32_buf(c, &p_attn),
                        try_upload_mat(c, &g, &l.hc_ffn.fn_w).or_else(|| try_upload_mat_gtt(c, &g, &l.hc_ffn.fn_w)),
                        upload_f32_buf(c, &p_ffn),
                        upload_f32_buf(c, &l.attn_norm),
                        upload_f32_buf(c, &l.ffn_norm),
                        upload_f32_buf(c, &l.q_a_norm),
                        upload_f32_buf(c, &l.kv_a_norm),
                    ) {
                        g_layers.push(XingGpuLayer {
                            hc_attn_fn: haf, hc_attn_params: hap,
                            hc_ffn_fn: hff, hc_ffn_params: hfp,
                            attn_norm: an, ffn_norm: fn_, q_a_norm: qan, kv_a_norm: kvan,
                            gate_inp, exp_probs_b,
                        });
                    } else {
                        ok = false;
                        break;
                    }
                }
                let mk = |n: usize| c.create_device_buffer((n * 4) as u64).ok();
                let pb = XING_PREFILL_BATCH;
                let ones_v = vec![1.0f32; 4 * cfg.n_embd];
                let zero_b = vec![0.0f32; cfg.n_expert];
                let head_ids_v: Vec<f32> = (0..cfg.n_head as u32).map(f32::from_bits).collect();
                if ok {
                    match (
                        upload_f32_buf(c, &ones_v), mk(pb * 4 * cfg.n_embd), mk(pb * 4 * cfg.n_embd), mk(pb * 4 * cfg.n_embd),
                        mk(pb * 24), mk(pb * 24), mk(pb * cfg.n_embd), mk(pb * cfg.n_embd), mk(pb * cfg.n_embd),
                        mk(pb * cfg.q_rank), mk(pb * cfg.q_rank), mk(pb * hd),
                        mk(pb * cfg.n_head * cfg.key_dim), mk(pb * cfg.n_head * cfg.kv_rank), mk(pb * cfg.n_head * cfg.kv_rank), mk(pb * cfg.n_head * cfg.value_dim),
                        mk(cfg.n_layer * max_ctx * (hd / 2)),
                        mk(pb * cfg.n_expert), mk(pb * cfg.n_used), mk(pb * cfg.n_used), upload_f32_buf(c, &zero_b),
                        upload_f32_buf(c, &out_norm), mk(cfg.n_embd), mk(cfg.n_vocab), mk(pb * cfg.n_used * cfg.n_embd),
                        mk(cfg.n_expert.max(1) * (pb + 1)),
                        upload_f32_buf(c, &head_ids_v), mk(4),
                    ) {
                        (
                            Some(ones), Some(streams_a), Some(streams_b), Some(streams_norm),
                            Some(hc_mixes), Some(hc_out), Some(inp), Some(normed), Some(block_out),
                            Some(q_a), Some(q_an), Some(kv_raw),
                            Some(q_full), Some(q_lat), Some(attn_lat), Some(val_all),
                            Some(kv_cache),
                            Some(router_logits), Some(top_ids), Some(top_w), Some(dummy_bias),
                            Some(out_norm_b), Some(hidden_buf), Some(logits_buf), Some(moe_tmp),
                            Some(moe_list),
                            Some(head_ids), Some(argmax_buf),
                        ) => Some(XingGpuState {
                            layers: g_layers, ones, streams_a, streams_b, streams_norm, hc_mixes, hc_out,
                            inp, normed, block_out, q_a, q_an, kv_raw, q_full, q_lat, attn_lat, val_all,
                            kv_cache, router_logits, top_ids, top_w, dummy_bias, out_norm: out_norm_b, hidden_buf, logits_buf, argmax_buf, moe_tmp, moe_list, head_ids,
                        }),
                        _ => None,
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Host-side MLA KV cache is only used by the hybrid CPU/GPU path; the full-GPU pipeline
        // keeps its cache in VRAM (skipping this saves n_layer * max_ctx * 576 * 4 bytes of RAM).
        #[cfg(feature = "vulkan")]
        let need_host_kv = gpu_pipe.is_none();
        #[cfg(not(feature = "vulkan"))]
        let need_host_kv = true;
        Ok(Self {
            kv_cache: if need_host_kv { vec![0f32; cfg.n_layer * max_ctx * hd] } else { Vec::new() },
            max_ctx,
            gpu_moe_layers: gpu_moe_placed,
            cpu_moe_layers: cpu_moe_placed,
            logits: vec![0f32; cfg.n_vocab],
            hidden: vec![0f32; cfg.n_embd],
            layers,
            embd,
            out_norm,
            output,
            #[cfg(feature = "vulkan")]
            vk,
            #[cfg(feature = "vulkan")]
            vk_x,
            #[cfg(feature = "vulkan")]
            vk_y,
            #[cfg(feature = "vulkan")]
            vk_g,
            #[cfg(feature = "vulkan")]
            vk_u,
            #[cfg(feature = "vulkan")]
            vk_a,
            #[cfg(feature = "vulkan")]
            vk_acc,
            #[cfg(feature = "vulkan")]
            vk_output,
            #[cfg(feature = "vulkan")]
            gpu_pipe,
            #[cfg(feature = "vulkan")]
            moe_stage,
            cfg,
            g,
        })
    }

    fn mv_cpu(&self, ti: &TensorInfo, slice_idx: usize, x: &[f32], y: &mut [f32]) {
        let k = ti.ne[0] as usize;
        let m = ti.ne[1] as usize;
        let slice_bytes = ti.row_bytes() * m;
        let raw = &self.g.tensor_data(ti)[slice_idx * slice_bytes..(slice_idx + 1) * slice_bytes];
        gemv(ti.ty, raw, x, y, k, m);
    }

    #[cfg(feature = "vulkan")]
    fn mv_hybrid(&self, ti: &TensorInfo, vk_mat: Option<&VkMat>, slice_idx: usize, x: &[f32], y: &mut [f32]) {
        if let (Some(ctx), Some(m), Some(bx), Some(by)) = (&self.vk, vk_mat, &self.vk_x, &self.vk_y) {
            if ctx.upload(bx, 0, bytemuck::cast_slice(x)).is_ok() {
                let mut rec = ctx.begin();
                rec.qmv(m.ty, &m.buf, slice_idx as u64 * m.slice_bytes, bx, 0, by, 0, m.k, m.m);
                if rec.finish_and_wait().is_ok() && ctx.download(by, 0, bytemuck::cast_slice_mut(&mut y[..m.m as usize])).is_ok() {
                    return;
                }
            }
        }
        self.mv_cpu(ti, slice_idx, x, y);
    }

    fn hc_pre(&self, streams: &[f32], w: &HcWeights) -> (Vec<f32>, [f32; 4], [[f32; 4]; 4]) {
        let ne = self.cfg.n_embd;
        let hc = 4usize;
        // RMSNorm over flattened [4 * ne]
        let ss: f32 = streams.iter().map(|v| v * v).sum::<f32>() / (hc * ne) as f32;
        let inv = 1.0 / (ss + self.cfg.eps).sqrt();
        let normed: Vec<f32> = streams.iter().map(|v| v * inv).collect();
        let mut mixes = vec![0f32; 24];
        self.mv_cpu(&w.fn_w, 0, &normed, &mut mixes);

        let mut pre = [0f32; 4];
        let mut post = [0f32; 4];
        for h in 0..4 {
            pre[h] = 1.0 / (1.0 + (-(mixes[h] * w.scale[0] + w.base[h])).exp());
            post[h] = 2.0 / (1.0 + (-(mixes[4 + h] * w.scale[1] + w.base[4 + h])).exp());
        }
        let mut c = [[0f32; 4]; 4]; // [src][dst]
        for k in 0..16 {
            c[k % 4][k / 4] = (mixes[8 + k] * w.scale[2] + w.base[8 + k]).clamp(-30.0, 30.0);
        }
        for d in 0..4 {
            let peak = (0..4).map(|s| c[s][d]).fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in 0..4 { c[s][d] = (c[s][d] - peak).exp(); sum += c[s][d]; }
            for s in 0..4 { c[s][d] /= sum; }
        }
        let norm_dst = |c: &mut [[f32; 4]; 4], eps: f32| {
            for s in 0..4 {
                let sum: f32 = eps + (0..4).map(|d| c[s][d]).sum::<f32>();
                for d in 0..4 { c[s][d] /= sum; }
            }
        };
        norm_dst(&mut c, self.cfg.hc_eps);
        for _ in 1..self.cfg.hc_iters {
            for d in 0..4 {
                let sum: f32 = self.cfg.hc_eps + (0..4).map(|s| c[s][d]).sum::<f32>();
                for s in 0..4 { c[s][d] /= sum; }
            }
            norm_dst(&mut c, self.cfg.hc_eps);
        }
        let mut inp = vec![0f32; ne];
        for h in 0..4 {
            let p = pre[h];
            let sl = &streams[h * ne..(h + 1) * ne];
            for e in 0..ne { inp[e] += p * sl[e]; }
        }
        (inp, post, c)
    }

    fn hc_post(streams: &mut [f32], block_out: &[f32], post: &[f32; 4], comb: &[[f32; 4]; 4], ne: usize) {
        let mut next = vec![0f32; 4 * ne];
        for d in 0..4 {
            let pd = post[d];
            let dst = &mut next[d * ne..(d + 1) * ne];
            for e in 0..ne { dst[e] = pd * block_out[e]; }
            for s in 0..4 {
                let cs = comb[s][d];
                let src = &streams[s * ne..(s + 1) * ne];
                for e in 0..ne { dst[e] += cs * src[e]; }
            }
        }
        streams.copy_from_slice(&next);
    }

    fn rope_yarn_normal(&self, v: &mut [f32], n_heads: usize, stride: usize, offset_in_head: usize, pos: usize) {
        let n_rot = self.cfg.n_rot;
        let factor = self.cfg.rope_factor;
        let yarn = factor > 1.0;
        let attn_factor = if yarn { 1.0 / (1.0 + 0.1 * factor.ln()) } else { 1.0 };
        let corr_dim = |n_r: f32| -> f32 {
            (n_rot as f32 * (self.cfg.rope_orig_ctx as f32 / (n_r * 2.0 * std::f32::consts::PI)).ln())
                / (2.0 * self.cfg.rope_base.ln())
        };
        let low = corr_dim(self.cfg.yarn_beta_fast).floor().max(0.0);
        let high = corr_dim(self.cfg.yarn_beta_slow).ceil().min((n_rot - 1) as f32);
        let mscale = if yarn { (1.0 + 0.1 * factor.ln()) * attn_factor } else { 1.0 };

        for i in 0..n_rot / 2 {
            let freq_base_inv = self.cfg.rope_base.powf(-2.0 * i as f32 / n_rot as f32);
            let freq = if yarn {
                let ramp = if (high - low).abs() < 1e-6 {
                    if (i as f32) < low { 0.0 } else { 1.0 }
                } else {
                    ((i as f32 - low) / (high - low)).clamp(0.0, 1.0)
                };
                freq_base_inv * ((1.0 - ramp) + ramp / factor)
            } else {
                freq_base_inv
            };
            let (s, c) = (pos as f32 * freq).sin_cos();
            let (sc, ss) = (c * mscale, s * mscale);
            for h in 0..n_heads {
                let base = h * stride + offset_in_head;
                let (x0, x1) = (v[base + 2 * i], v[base + 2 * i + 1]);
                v[base + 2 * i] = x0 * sc - x1 * ss;
                v[base + 2 * i + 1] = x0 * ss + x1 * sc;
            }
        }
    }
}

impl Executor for XingExec {
    fn name(&self) -> String {
        format!(
            "xing4-hybrid (gpu_moe={}, cpu_moe={})",
            self.gpu_moe_layers, self.cpu_moe_layers
        )
    }

    fn forward(&mut self, token: u32, pos: usize, _page_table: &[u32], out: Out) -> Result<(), EngineError> {
        self.forward_inner(token, pos, out, false).map(|_| ())
    }

    /// Greedy decode with the argmax on the device (only the index crosses PCIe). `None` (caller
    /// falls back to `forward`) unless the whole forward runs on the GPU pipeline.
    fn forward_argmax(&mut self, token: u32, pos: usize, _page_table: &[u32]) -> Result<Option<u32>, EngineError> {
        #[cfg(feature = "vulkan")]
        {
            if self.vk.is_some() && self.gpu_pipe.is_some() && self.vk_g.is_some() && self.vk_u.is_some() && self.vk_a.is_some() && self.vk_output.is_some() {
                if pos >= self.max_ctx {
                    return Err(EngineError::Context(format!("pos {pos} >= max_ctx {}", self.max_ctx)));
                }
                return self.forward_inner(token, pos, Out { logits: true, hidden: false }, true);
            }
        }
        let _ = (token, pos);
        Ok(None)
    }

    fn logits(&self) -> &[f32] { &self.logits }
    fn hidden(&self) -> &[f32] { &self.hidden }

    fn forward_batch(&mut self, tokens: &[u32], pos0: usize, page_table: &[u32], out: Out) -> Result<(), EngineError> {
        if tokens.len() <= 1 {
            if let Some(&tok) = tokens.first() {
                return self.forward(tok, pos0, page_table, out);
            }
            return Ok(());
        }
        #[cfg(feature = "vulkan")]
        if self.gpu_pipe.is_some() && self.vk.is_some() && self.vk_output.is_some() {
            if pos0 + tokens.len() > self.max_ctx {
                return Err(EngineError::Context(format!(
                    "pos {} >= max_ctx {}",
                    pos0 + tokens.len() - 1,
                    self.max_ctx
                )));
            }
            let mut off = 0usize;
            while off < tokens.len() {
                let chunk = &tokens[off..(off + XING_PREFILL_BATCH).min(tokens.len())];
                let is_last = off + chunk.len() == tokens.len();
                self.forward_batch_gpu_chunk(
                    chunk,
                    pos0 + off,
                    Out { logits: is_last && out.logits, hidden: is_last && out.hidden },
                )?;
                off += chunk.len();
            }
            return Ok(());
        }
        for (i, &tok) in tokens.iter().enumerate() {
            let is_last = i + 1 == tokens.len();
            self.forward(tok, pos0 + i, page_table, Out { logits: is_last && out.logits, hidden: is_last && out.hidden })?;
        }
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
fn qmmt_ok(ty: ocak_vk::WType) -> bool {
    ocak_vk::has_tiled_qmm(ty)
}

#[cfg(feature = "vulkan")]
impl XingExec {
    fn forward_batch_gpu_chunk(&mut self, chunk: &[u32], pos0: usize, out: Out) -> Result<(), EngineError> {
        let c = self.cfg.clone();
        let (ne, nh, latent, n_rot, nope, vdim) = (c.n_embd, c.n_head, c.kv_rank, c.n_rot, c.key_dim - c.n_rot, c.value_dim);
        let hd = latent + n_rot;
        let bs = chunk.len();
        let batch = bs as u32;
        let yarn = c.rope_factor > 1.0;
        // Same as `forward`: yarn_log_mul is already the true multiplier (see the config load).
        let mscale = if yarn { 1.0 + 0.1 * c.yarn_log_mul * c.rope_factor.ln() } else { 1.0 };
        let kq_scale = mscale * mscale / (c.key_dim as f32).sqrt();

        let rb = self.embd.row_bytes();
        let emb_raw = self.g.tensor_data(&self.embd);
        let mut streams = vec![0f32; bs * 4 * ne];
        let mut emb = vec![0f32; ne];
        for (t, &tok) in chunk.iter().enumerate() {
            let r = tok as usize;
            dequantize_row(self.embd.ty, &emb_raw[r * rb..(r + 1) * rb], &mut emb, ne);
            for h in 0..4 {
                let dst = (t * 4 + h) * ne;
                streams[dst..dst + ne].copy_from_slice(&emb);
            }
        }

        let (Some(c_vk), Some(gp), Some(bg), Some(bu), Some(ba), Some(vout)) = (
            &self.vk, &self.gpu_pipe, &self.vk_g, &self.vk_u, &self.vk_a, &self.vk_output,
        ) else {
            return Err(EngineError::Backend("xing gpu_pipe unavailable".into()));
        };

        c_vk.upload(&gp.streams_a, 0, bytemuck::cast_slice(&streams)).map_err(|e| EngineError::Backend(e.to_string()))?;
        let corr_dim = |n_r: f32| -> f32 {
            (n_rot as f32 * (c.rope_orig_ctx as f32 / (n_r * 2.0 * std::f32::consts::PI)).ln())
                / (2.0 * c.rope_base.ln())
        };
        let low = corr_dim(c.yarn_beta_fast).floor().max(0.0);
        let high = corr_dim(c.yarn_beta_slow).ceil().min((n_rot - 1) as f32);
        let attn_factor = if yarn { 1.0 / (1.0 + 0.1 * c.rope_factor.ln()) } else { 1.0 };
        let rope_mscale = if yarn { (1.0 + 0.1 * c.rope_factor.ln()) * attn_factor } else { 1.0 };

        let mut rec = c_vk.begin();
        rec.set_defer_barriers(true);
        let mut cur_a = true;
        for li in 0..c.n_layer {
            if li > 0 && li % XING_LAYERS_PER_SUBMIT == 0 {
                rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;
                rec = c_vk.begin();
                rec.set_defer_barriers(true);
            }
            let l = &self.layers[li];
            let gl = &gp.layers[li];
            let (st_cur, st_nxt) = if cur_a { (&gp.streams_a, &gp.streams_b) } else { (&gp.streams_b, &gp.streams_a) };

            // 1. HC pre (attn)
            rec.rms_norm_batch(st_cur, 0, &gp.ones, 0, &gp.streams_norm, 0, (4 * ne) as u32, c.eps, batch);
            rec.barrier();
            rec.qmm(gl.hc_attn_fn.ty, &gl.hc_attn_fn.buf, 0, &gp.streams_norm, 0, &gp.hc_mixes, 0, gl.hc_attn_fn.k, gl.hc_attn_fn.m, batch);
            rec.barrier();
            rec.hc_sinkhorn_batch(&gp.hc_mixes, &gl.hc_attn_params, &gp.hc_out, c.hc_iters as u32, c.hc_eps, batch);
            rec.barrier();
            rec.hc_mix_batch(st_cur, &gp.block_out, &gp.hc_out, &gp.inp, ne as u32, 0, batch);
            rec.barrier();

            // 2. MLA Attention (batched projections + causal batched attention)
            rec.rms_norm_batch(&gp.inp, 0, &gl.attn_norm, 0, &gp.normed, 0, ne as u32, c.eps, batch);
            rec.barrier();
            let va = l.vk_wq_a.as_ref().unwrap();
            let vkva = l.vk_wkv_a.as_ref().unwrap();
            let vqb = l.vk_wq_b.as_ref().unwrap();
            let vkb = l.vk_wk_b.as_ref().unwrap();
            let vvb = l.vk_wv_b.as_ref().unwrap();
            let vwo = l.vk_wo.as_ref().unwrap();
            rec.qmm(va.ty, &va.buf, 0, &gp.normed, 0, &gp.q_a, 0, va.k, va.m, batch);
            rec.qmm(vkva.ty, &vkva.buf, 0, &gp.normed, 0, &gp.kv_raw, 0, vkva.k, vkva.m, batch);
            rec.barrier();
            rec.rms_norm_batch(&gp.q_a, 0, &gl.q_a_norm, 0, &gp.q_an, 0, c.q_rank as u32, c.eps, batch);
            rec.rms_norm_batch_strided(&gp.kv_raw, 0, &gl.kv_a_norm, 0, &gp.kv_raw, 0, latent as u32, hd as u32, c.eps, batch);
            rec.barrier();
            rec.qmm(vqb.ty, &vqb.buf, 0, &gp.q_an, 0, &gp.q_full, 0, vqb.k, vqb.m, batch);
            rec.barrier();
            let kv_layer_off = (li * self.max_ctx * (hd / 2)) as u32;
            rec.rope_yarn_mla_batch(
                &gp.q_full, &gp.kv_raw, &gp.kv_cache,
                nh as u32, c.key_dim as u32, nope as u32, n_rot as u32, latent as u32,
                pos0 as u32, kv_layer_off, c.rope_base, c.rope_factor, low, high, rope_mscale, batch,
            );
            rec.barrier();
            // Per-head absorbed K projection: one grouped matmul (group = head, entry = t*nh + h).
            if !rec.qmm_grouped(vkb.ty, &vkb.buf, 0, vkb.slice_bytes, &gp.q_full, 0, c.key_dim as u32, 1, batch * nh as u32, &gp.q_lat, 0, vkb.k, vkb.m, nh as u32, None, batch) {
                for t in 0..bs {
                    for h in 0..nh {
                        rec.qmv(
                            vkb.ty, &vkb.buf, h as u64 * vkb.slice_bytes,
                            &gp.q_full, ((t * nh + h) * c.key_dim) as u64,
                            &gp.q_lat, ((t * nh + h) * latent) as u64,
                            vkb.k, vkb.m,
                        );
                    }
                }
            }
            rec.barrier();
            rec.mla_attn_batch(
                &gp.q_lat, &gp.q_full, &gp.kv_cache, &gp.attn_lat,
                nh as u32, c.key_dim as u32, nope as u32, n_rot as u32, latent as u32,
                pos0 as u32, kv_layer_off, kq_scale, batch,
            );
            rec.barrier();
            if !rec.qmm_grouped(vvb.ty, &vvb.buf, 0, vvb.slice_bytes, &gp.attn_lat, 0, latent as u32, 1, batch * nh as u32, &gp.val_all, 0, vvb.k, vvb.m, nh as u32, None, batch) {
                for t in 0..bs {
                    for h in 0..nh {
                        rec.qmv(
                            vvb.ty, &vvb.buf, h as u64 * vvb.slice_bytes,
                            &gp.attn_lat, ((t * nh + h) * latent) as u64,
                            &gp.val_all, ((t * nh + h) * vdim) as u64,
                            vvb.k, vvb.m,
                        );
                    }
                }
            }
            rec.barrier();
            rec.qmm(vwo.ty, &vwo.buf, 0, &gp.val_all, 0, &gp.block_out, 0, vwo.k, vwo.m, batch);
            rec.barrier();
            rec.hc_mix_batch(st_cur, &gp.block_out, &gp.hc_out, st_nxt, ne as u32, 1, batch);
            rec.barrier();
            cur_a = !cur_a;

            // 3. HC pre (FFN)
            let (st_cur, st_nxt) = if cur_a { (&gp.streams_a, &gp.streams_b) } else { (&gp.streams_b, &gp.streams_a) };
            rec.rms_norm_batch(st_cur, 0, &gp.ones, 0, &gp.streams_norm, 0, (4 * ne) as u32, c.eps, batch);
            rec.barrier();
            rec.qmm(gl.hc_ffn_fn.ty, &gl.hc_ffn_fn.buf, 0, &gp.streams_norm, 0, &gp.hc_mixes, 0, gl.hc_ffn_fn.k, gl.hc_ffn_fn.m, batch);
            rec.barrier();
            rec.hc_sinkhorn_batch(&gp.hc_mixes, &gl.hc_ffn_params, &gp.hc_out, c.hc_iters as u32, c.hc_eps, batch);
            rec.barrier();
            rec.hc_mix_batch(st_cur, &gp.block_out, &gp.hc_out, &gp.inp, ne as u32, 0, batch);
            rec.barrier();
            rec.rms_norm_batch(&gp.inp, 0, &gl.ffn_norm, 0, &gp.normed, 0, ne as u32, c.eps, batch);
            rec.barrier();

            // 4. Dense or MoE FFN
            match &l.ffn {
                XingFfn::Dense { vk_gate, vk_up, vk_down, .. } => {
                    let (vg, vu, vd) = (vk_gate.as_ref().unwrap(), vk_up.as_ref().unwrap(), vk_down.as_ref().unwrap());
                    rec.qmm(vg.ty, &vg.buf, 0, &gp.normed, 0, bg, 0, vg.k, vg.m, batch);
                    rec.qmm(vu.ty, &vu.buf, 0, &gp.normed, 0, bu, 0, vu.k, vu.m, batch);
                    rec.barrier();
                    rec.swiglu(bg, 0, bu, 0, ba, 0, batch * c.n_ff_dense as u32);
                    rec.barrier();
                    rec.qmm(vd.ty, &vd.buf, 0, ba, 0, &gp.block_out, 0, vd.k, vd.m, batch);
                    rec.barrier();
                }
                XingFfn::Moe { vk_gate_exps, vk_up_exps, vk_down_exps, vk_gate_sh, vk_up_sh, vk_down_sh, .. } => {
                    let vgi = gl.gate_inp.as_ref().unwrap();
                    let (vgs, vus, vds) = (vk_gate_sh.as_ref().unwrap(), vk_up_sh.as_ref().unwrap(), vk_down_sh.as_ref().unwrap());
                    let sh_off = (bs * c.n_used * c.n_ff_exp) as u64;
                    rec.qmm(vgi.ty, &vgi.buf, 0, &gp.normed, 0, &gp.router_logits, 0, vgi.k, vgi.m, batch);
                    rec.qmm(vgs.ty, &vgs.buf, 0, &gp.normed, 0, bg, sh_off, vgs.k, vgs.m, batch);
                    rec.qmm(vus.ty, &vus.buf, 0, &gp.normed, 0, bu, sh_off, vus.k, vus.m, batch);
                    rec.barrier();
                    let (bias_buf, has_bias) = match &gl.exp_probs_b {
                        Some(b) => (b, true),
                        None => (&gp.dummy_bias, false),
                    };
                    let n_used = c.n_used as u32;
                    rec.moe_topk_batch(&gp.router_logits, bias_buf, &gp.top_ids, &gp.top_w, c.n_expert as u32, n_used, has_bias, c.exp_weights_norm, c.exp_weights_scale, batch);
                    rec.swiglu(bg, sh_off, bu, sh_off, ba, sh_off, batch * c.n_ff_exp as u32);
                    rec.barrier();
                    let (vg, vu, vd) = (vk_gate_exps.as_ref().unwrap(), vk_up_exps.as_ref().unwrap(), vk_down_exps.as_ref().unwrap());
                    // Shared down (batched qmm) + all B tokens' routed gate/up in parallel!
                    rec.qmm(vds.ty, &vds.buf, 0, ba, sh_off, &gp.block_out, 0, vds.k, vds.m, batch);
                    // Routed experts: bucket the bs*n_used (token, slot) entries by expert, then one
                    // grouped tiled matmul per projection (each expert's weights read once per tile).
                    let grouped = qmmt_ok(vg.ty) && qmmt_ok(vu.ty) && qmmt_ok(vd.ty) && c.n_expert <= 256;
                    let n_exp = c.n_expert as u32;
                    // GTT-resident experts: one DMA copy per tensor into VRAM staging.
                    let staged: Option<[&Buffer; 3]> = match &self.moe_stage {
                        Some(st) if grouped && vg.in_gtt => {
                            for (src, dst) in [(vg, &st[0]), (vu, &st[1]), (vd, &st[2])] {
                                rec.copy_buf_bytes(&src.buf, dst, src.slice_bytes * c.n_expert as u64);
                            }
                            Some([&st[0], &st[1], &st[2]])
                        }
                        _ => None,
                    };
                    let (wg_buf, wu_buf, wd_buf) = match staged {
                        Some([a, b, d]) => (a, b, d),
                        None => (&vg.buf, &vu.buf, &vd.buf),
                    };
                    if grouped {
                        rec.barrier();
                        rec.moe_group(&gp.top_ids, &gp.moe_list, batch * n_used, n_exp, batch);
                        rec.barrier();
                        rec.qmm_grouped(vg.ty, wg_buf, 0, vg.slice_bytes, &gp.normed, 0, ne as u32, n_used, batch, bg, 0, vg.k, vg.m, n_exp, Some(&gp.moe_list), batch);
                        rec.qmm_grouped(vu.ty, wu_buf, 0, vu.slice_bytes, &gp.normed, 0, ne as u32, n_used, batch, bu, 0, vu.k, vu.m, n_exp, Some(&gp.moe_list), batch);
                    } else {
                        for t in 0..bs {
                            let slot_base = (t * c.n_used) as u32;
                            let y_off = (t * c.n_used * c.n_ff_exp) as u64;
                            rec.qmv_id_batch(vg.ty, &vg.buf, &gp.top_ids, slot_base, n_used, &gp.normed, (t * ne) as u64, 0, bg, y_off, vg.k, vg.m, vg.slice_bytes);
                            rec.qmv_id_batch(vu.ty, &vu.buf, &gp.top_ids, slot_base, n_used, &gp.normed, (t * ne) as u64, 0, bu, y_off, vu.k, vu.m, vu.slice_bytes);
                        }
                    }
                    rec.barrier();
                    rec.swiglu(bg, 0, bu, 0, ba, 0, batch * n_used * c.n_ff_exp as u32);
                    rec.barrier();
                    if grouped {
                        rec.qmm_grouped(vd.ty, wd_buf, 0, vd.slice_bytes, ba, 0, c.n_ff_exp as u32, 1, batch * n_used, &gp.moe_tmp, 0, vd.k, vd.m, n_exp, Some(&gp.moe_list), batch);
                    } else {
                        for t in 0..bs {
                            let slot_base = (t * c.n_used) as u32;
                            let a_off = (t * c.n_used * c.n_ff_exp) as u64;
                            let tmp_off = (t * c.n_used * ne) as u64;
                            rec.qmv_id_batch(vd.ty, &vd.buf, &gp.top_ids, slot_base, n_used, ba, a_off, c.n_ff_exp as u32, &gp.moe_tmp, tmp_off, vd.k, vd.m, vd.slice_bytes);
                        }
                    }
                    rec.barrier();
                    rec.add_scaled_id_tokens(&gp.block_out, 0, &gp.moe_tmp, 0, &gp.top_w, ne as u32, 0, n_used, false, batch);
                    rec.barrier();
                }
            }
            rec.hc_mix_batch(st_cur, &gp.block_out, &gp.hc_out, st_nxt, ne as u32, 1, batch);
            rec.barrier();
            cur_a = !cur_a;
        }

        let st_final = if cur_a { &gp.streams_a } else { &gp.streams_b };
        if out.logits || out.hidden {
            rec.hc_mix_batch(st_final, &gp.block_out, &gp.hc_out, &gp.inp, ne as u32, 2, batch);
            rec.barrier();
            let last_off = ((bs - 1) * ne) as u64;
            rec.rms_norm(&gp.inp, last_off, &gp.out_norm, 0, &gp.hidden_buf, 0, ne as u32, c.eps);
            rec.barrier();
        }
        if out.logits {
            rec.qmv(vout.ty, &vout.buf, 0, &gp.hidden_buf, 0, &gp.logits_buf, 0, vout.k, vout.m);
            rec.barrier();
        }
        rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;
        if out.hidden {
            c_vk.download(&gp.hidden_buf, 0, bytemuck::cast_slice_mut(&mut self.hidden)).map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        if out.logits {
            c_vk.download(&gp.logits_buf, 0, bytemuck::cast_slice_mut(&mut self.logits)).map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        Ok(())
    }
}

impl XingExec {
    /// Single-token forward; with `argmax` (GPU pipeline only) returns the greedy token instead
    /// of reading back logits.
    fn forward_inner(&mut self, token: u32, pos: usize, out: Out, argmax: bool) -> Result<Option<u32>, EngineError> {
        let _ = argmax;
        if pos >= self.max_ctx {
            return Err(EngineError::Context(format!("pos {pos} >= max_ctx {}", self.max_ctx)));
        }
        let c = self.cfg.clone();
        let (ne, nh, latent, n_rot, nope, vdim) = (c.n_embd, c.n_head, c.kv_rank, c.n_rot, c.key_dim - c.n_rot, c.value_dim);
        let hd = latent + n_rot;
        let yarn = c.rope_factor > 1.0;
        let mscale = if yarn { 1.0 + 0.1 * c.yarn_log_mul * c.rope_factor.ln() } else { 1.0 };
        let kq_scale = mscale * mscale / (c.key_dim as f32).sqrt();

        // Embed token & repeat across 4 hyper-connection streams
        let mut emb = vec![0f32; ne];
        let rb = self.embd.row_bytes();
        let r = token as usize;
        dequantize_row(self.embd.ty, &self.g.tensor_data(&self.embd)[r * rb..(r + 1) * rb], &mut emb, ne);
        let mut streams = vec![0f32; 4 * ne];
        for h in 0..4 {
            streams[h * ne..(h + 1) * ne].copy_from_slice(&emb);
        }

        // Fast path: ENTIRE 40-layer forward pass in ONE SINGLE Vulkan vkQueueSubmit!
        #[cfg(feature = "vulkan")]
        if let (Some(c_vk), Some(gp), Some(bg), Some(bu), Some(ba), Some(vout)) = (
            &self.vk, &self.gpu_pipe, &self.vk_g, &self.vk_u, &self.vk_a, &self.vk_output,
        ) {
            c_vk.upload(&gp.streams_a, 0, bytemuck::cast_slice(&streams)).map_err(|e| EngineError::Backend(e.to_string()))?;
            let corr_dim = |n_r: f32| -> f32 {
                (n_rot as f32 * (c.rope_orig_ctx as f32 / (n_r * 2.0 * std::f32::consts::PI)).ln())
                    / (2.0 * c.rope_base.ln())
            };
            let low = corr_dim(c.yarn_beta_fast).floor().max(0.0);
            let high = corr_dim(c.yarn_beta_slow).ceil().min((n_rot - 1) as f32);
            let attn_factor = if yarn { 1.0 / (1.0 + 0.1 * c.rope_factor.ln()) } else { 1.0 };
            let rope_mscale = if yarn { (1.0 + 0.1 * c.rope_factor.ln()) * attn_factor } else { 1.0 };

            let mut rec = c_vk.begin();
            rec.set_defer_barriers(true);
            let mut cur_a = true; // true -> streams_a is active, false -> streams_b is active
            for li in 0..c.n_layer {
                let l = &self.layers[li];
                let gl = &gp.layers[li];
                let (st_cur, st_nxt) = if cur_a { (&gp.streams_a, &gp.streams_b) } else { (&gp.streams_b, &gp.streams_a) };

                // 1. HC pre (attn)
                rec.rms_norm(st_cur, 0, &gp.ones, 0, &gp.streams_norm, 0, (4 * ne) as u32, c.eps);
                rec.barrier();
                rec.qmv(gl.hc_attn_fn.ty, &gl.hc_attn_fn.buf, 0, &gp.streams_norm, 0, &gp.hc_mixes, 0, gl.hc_attn_fn.k, gl.hc_attn_fn.m);
                rec.barrier();
                rec.hc_sinkhorn(&gp.hc_mixes, &gl.hc_attn_params, &gp.hc_out, c.hc_iters as u32, c.hc_eps);
                rec.barrier();
                rec.hc_mix(st_cur, &gp.block_out, &gp.hc_out, &gp.inp, ne as u32, 0);
                rec.barrier();

                // 2. MLA Attention (100% GPU with parallel head projections!)
                rec.rms_norm(&gp.inp, 0, &gl.attn_norm, 0, &gp.normed, 0, ne as u32, c.eps);
                rec.barrier();
                let va = l.vk_wq_a.as_ref().unwrap();
                let vkva = l.vk_wkv_a.as_ref().unwrap();
                let vqb = l.vk_wq_b.as_ref().unwrap();
                let vkb = l.vk_wk_b.as_ref().unwrap();
                let vvb = l.vk_wv_b.as_ref().unwrap();
                let vwo = l.vk_wo.as_ref().unwrap();
                rec.qmv(va.ty, &va.buf, 0, &gp.normed, 0, &gp.q_a, 0, va.k, va.m);
                rec.qmv(vkva.ty, &vkva.buf, 0, &gp.normed, 0, &gp.kv_raw, 0, vkva.k, vkva.m);
                rec.barrier();
                rec.rms_norm(&gp.q_a, 0, &gl.q_a_norm, 0, &gp.q_an, 0, c.q_rank as u32, c.eps);
                let kv_layer_off = (li * self.max_ctx * (hd / 2)) as u32;
                rec.rms_norm(&gp.kv_raw, 0, &gl.kv_a_norm, 0, &gp.kv_raw, 0, latent as u32, c.eps);
                rec.barrier();
                rec.qmv(vqb.ty, &vqb.buf, 0, &gp.q_an, 0, &gp.q_full, 0, vqb.k, vqb.m);
                rec.barrier();
                rec.rope_yarn_mla(
                    &gp.q_full, &gp.kv_raw, &gp.kv_cache,
                    nh as u32, c.key_dim as u32, nope as u32, n_rot as u32, latent as u32,
                    pos as u32, kv_layer_off, c.rope_base, c.rope_factor, low, high, rope_mscale,
                );
                rec.barrier();
                // All heads in one indexed matvec (slice h * slice_bytes, input stride key_dim).
                rec.qmv_id_batch(vkb.ty, &vkb.buf, &gp.head_ids, 0, nh as u32, &gp.q_full, 0, c.key_dim as u32, &gp.q_lat, 0, vkb.k, vkb.m, vkb.slice_bytes);
                rec.barrier();
                rec.mla_attn(
                    &gp.q_lat, &gp.q_full, &gp.kv_cache, &gp.attn_lat,
                    nh as u32, c.key_dim as u32, nope as u32, n_rot as u32, latent as u32,
                    pos as u32, kv_layer_off, kq_scale,
                );
                rec.barrier();
                rec.qmv_id_batch(vvb.ty, &vvb.buf, &gp.head_ids, 0, nh as u32, &gp.attn_lat, 0, latent as u32, &gp.val_all, 0, vvb.k, vvb.m, vvb.slice_bytes);
                rec.barrier();
                rec.qmv(vwo.ty, &vwo.buf, 0, &gp.val_all, 0, &gp.block_out, 0, vwo.k, vwo.m);
                rec.barrier();
                rec.hc_mix(st_cur, &gp.block_out, &gp.hc_out, st_nxt, ne as u32, 1);
                rec.barrier();
                cur_a = !cur_a;

                // 3. HC pre (FFN)
                let (st_cur, st_nxt) = if cur_a { (&gp.streams_a, &gp.streams_b) } else { (&gp.streams_b, &gp.streams_a) };
                rec.rms_norm(st_cur, 0, &gp.ones, 0, &gp.streams_norm, 0, (4 * ne) as u32, c.eps);
                rec.barrier();
                rec.qmv(gl.hc_ffn_fn.ty, &gl.hc_ffn_fn.buf, 0, &gp.streams_norm, 0, &gp.hc_mixes, 0, gl.hc_ffn_fn.k, gl.hc_ffn_fn.m);
                rec.barrier();
                rec.hc_sinkhorn(&gp.hc_mixes, &gl.hc_ffn_params, &gp.hc_out, c.hc_iters as u32, c.hc_eps);
                rec.barrier();
                rec.hc_mix(st_cur, &gp.block_out, &gp.hc_out, &gp.inp, ne as u32, 0);
                rec.barrier();
                rec.rms_norm(&gp.inp, 0, &gl.ffn_norm, 0, &gp.normed, 0, ne as u32, c.eps);
                rec.barrier();

                // 4. Dense or Batched 2D MoE FFN (routed + shared in parallel!)
                match &l.ffn {
                    XingFfn::Dense { vk_gate, vk_up, vk_down, .. } => {
                        let (vg, vu, vd) = (vk_gate.as_ref().unwrap(), vk_up.as_ref().unwrap(), vk_down.as_ref().unwrap());
                        rec.qmv(vg.ty, &vg.buf, 0, &gp.normed, 0, bg, 0, vg.k, vg.m);
                        rec.qmv(vu.ty, &vu.buf, 0, &gp.normed, 0, bu, 0, vu.k, vu.m);
                        rec.barrier();
                        rec.swiglu(bg, 0, bu, 0, ba, 0, c.n_ff_dense as u32);
                        rec.barrier();
                        rec.qmv(vd.ty, &vd.buf, 0, ba, 0, &gp.block_out, 0, vd.k, vd.m);
                        rec.barrier();
                    }
                    XingFfn::Moe { vk_gate_exps, vk_up_exps, vk_down_exps, vk_gate_sh, vk_up_sh, vk_down_sh, .. } => {
                        let vgi = gl.gate_inp.as_ref().unwrap();
                        let (vgs, vus, vds) = (vk_gate_sh.as_ref().unwrap(), vk_up_sh.as_ref().unwrap(), vk_down_sh.as_ref().unwrap());
                        let sh_off = (c.n_used * c.n_ff_exp) as u64;
                        // Router + shared gate/up in parallel!
                        rec.qmv(vgi.ty, &vgi.buf, 0, &gp.normed, 0, &gp.router_logits, 0, vgi.k, vgi.m);
                        rec.qmv(vgs.ty, &vgs.buf, 0, &gp.normed, 0, bg, sh_off, vgs.k, vgs.m);
                        rec.qmv(vus.ty, &vus.buf, 0, &gp.normed, 0, bu, sh_off, vus.k, vus.m);
                        rec.barrier();
                        let (bias_buf, has_bias) = match &gl.exp_probs_b {
                            Some(b) => (b, true),
                            None => (&gp.dummy_bias, false),
                        };
                        let n_used = c.n_used as u32;
                        rec.moe_topk(&gp.router_logits, bias_buf, &gp.top_ids, &gp.top_w, c.n_expert as u32, n_used, has_bias, c.exp_weights_norm, c.exp_weights_scale);
                        rec.swiglu(bg, sh_off, bu, sh_off, ba, sh_off, c.n_ff_exp as u32);
                        rec.barrier();
                        let (vg, vu, vd) = (vk_gate_exps.as_ref().unwrap(), vk_up_exps.as_ref().unwrap(), vk_down_exps.as_ref().unwrap());
                        // Batched 2D routed gate/up + shared down in parallel!
                        rec.qmv_id_batch(vg.ty, &vg.buf, &gp.top_ids, 0, n_used, &gp.normed, 0, 0, bg, 0, vg.k, vg.m, vg.slice_bytes);
                        rec.qmv_id_batch(vu.ty, &vu.buf, &gp.top_ids, 0, n_used, &gp.normed, 0, 0, bu, 0, vu.k, vu.m, vu.slice_bytes);
                        rec.qmv(vds.ty, &vds.buf, 0, ba, sh_off, &gp.block_out, 0, vds.k, vds.m);
                        rec.barrier();
                        rec.swiglu(bg, 0, bu, 0, ba, 0, (c.n_used * c.n_ff_exp) as u32);
                        rec.barrier();
                        rec.qmv_id_batch(vd.ty, &vd.buf, &gp.top_ids, 0, n_used, ba, 0, c.n_ff_exp as u32, &gp.moe_tmp, 0, vd.k, vd.m, vd.slice_bytes);
                        rec.barrier();
                        // Accumulate all 4 weighted routed experts onto shared block_out in 1 dispatch!
                        rec.add_scaled_id_batch(&gp.block_out, 0, &gp.moe_tmp, 0, &gp.top_w, ne as u32, 0, n_used, false);
                        rec.barrier();
                    }
                }
                rec.hc_mix(st_cur, &gp.block_out, &gp.hc_out, st_nxt, ne as u32, 1);
                rec.barrier();
                cur_a = !cur_a;
            }

            // Final collapse + output_norm + lm_head
            let st_final = if cur_a { &gp.streams_a } else { &gp.streams_b };
            if out.logits || out.hidden {
                rec.hc_mix(st_final, &gp.block_out, &gp.hc_out, &gp.inp, ne as u32, 2);
                rec.barrier();
                rec.rms_norm(&gp.inp, 0, &gp.out_norm, 0, &gp.hidden_buf, 0, ne as u32, c.eps);
                rec.barrier();
            }
            if out.logits {
                rec.qmv(vout.ty, &vout.buf, 0, &gp.hidden_buf, 0, &gp.logits_buf, 0, vout.k, vout.m);
                rec.barrier();
            }
            if argmax {
                rec.argmax(&gp.logits_buf, 0, &gp.argmax_buf, c.n_vocab as u32);
                rec.barrier();
                let mut idx = [0u32; 1];
                rec.finish_and_read(&mut [(&gp.argmax_buf, 0, bytemuck::cast_slice_mut(&mut idx))]).map_err(|e| EngineError::Backend(e.to_string()))?;
                return Ok(Some(idx[0]));
            }
            // Read back inside the same submit (no separate download submits).
            let mut reads: Vec<(&Buffer, u64, &mut [u8])> = Vec::with_capacity(2);
            if out.hidden {
                reads.push((&gp.hidden_buf, 0, bytemuck::cast_slice_mut(&mut self.hidden)));
            }
            if out.logits {
                reads.push((&gp.logits_buf, 0, bytemuck::cast_slice_mut(&mut self.logits)));
            }
            rec.finish_and_read(&mut reads).map_err(|e| EngineError::Backend(e.to_string()))?;
            return Ok(None);
        }

        let mut normed = vec![0f32; ne];
        let mut q_a = vec![0f32; c.q_rank];
        let mut q_an = vec![0f32; c.q_rank];
        let mut q_full = vec![0f32; nh * c.key_dim];
        let mut kv_raw = vec![0f32; hd];
        let mut q_lat = vec![0f32; nh * latent];
        let mut attn_lat = vec![0f32; nh * latent];
        let mut val_all = vec![0f32; nh * vdim];
        let mut block_out = vec![0f32; ne];

        for li in 0..c.n_layer {
            // 1. Hyper-Connection pre + MLA attention
            let (inp, post, comb) = self.hc_pre(&streams, &self.layers[li].hc_attn);
            ops::rms_norm(&inp, &self.layers[li].attn_norm, &mut normed, c.eps);

            // Fused GPU Submit 1: wq_a + wkv_a from `normed`
            #[cfg(feature = "vulkan")]
            let did_gpu_a = if let (Some(c_vk), Some(bx), Some(by), Some(va), Some(vkva)) = (
                &self.vk, &self.vk_x, &self.vk_y, &self.layers[li].vk_wq_a, &self.layers[li].vk_wkv_a,
            ) {
                let _ = c_vk.upload(bx, 0, bytemuck::cast_slice(&normed));
                let mut rec = c_vk.begin();
                rec.qmv(va.ty, &va.buf, 0, bx, 0, by, 0, va.k, va.m);
                rec.qmv(vkva.ty, &vkva.buf, 0, bx, 0, by, c.q_rank as u64, vkva.k, vkva.m);
                if rec.finish_and_wait().is_ok() {
                    let mut pack = vec![0f32; c.q_rank + hd];
                    let _ = c_vk.download(by, 0, bytemuck::cast_slice_mut(&mut pack));
                    q_a.copy_from_slice(&pack[..c.q_rank]);
                    kv_raw.copy_from_slice(&pack[c.q_rank..c.q_rank + hd]);
                    true
                } else {
                    false
                }
            } else {
                false
            };
            #[cfg(not(feature = "vulkan"))]
            let did_gpu_a = false;
            if !did_gpu_a {
                self.mv_cpu(&self.layers[li].wq_a, 0, &normed, &mut q_a);
                self.mv_cpu(&self.layers[li].wkv_a, 0, &normed, &mut kv_raw);
            }
            ops::rms_norm(&q_a, &self.layers[li].q_a_norm, &mut q_an, c.eps);

            // Fused GPU Submit 2: wq_b -> 32x wk_b (absorption) in ONE command buffer!
            #[cfg(feature = "vulkan")]
            let did_gpu_qk = if let (Some(c_vk), Some(bx), Some(by), Some(vqb), Some(vkb)) = (
                &self.vk, &self.vk_x, &self.vk_y, &self.layers[li].vk_wq_b, &self.layers[li].vk_wk_b,
            ) {
                let _ = c_vk.upload(bx, 0, bytemuck::cast_slice(&q_an));
                let mut rec = c_vk.begin();
                rec.qmv(vqb.ty, &vqb.buf, 0, bx, 0, by, 0, vqb.k, vqb.m);
                let q_lat_off = (nh * c.key_dim) as u64;
                for h in 0..nh {
                    rec.qmv(
                        vkb.ty,
                        &vkb.buf,
                        h as u64 * vkb.slice_bytes,
                        by,
                        (h * c.key_dim) as u64,
                        by,
                        q_lat_off + (h * latent) as u64,
                        vkb.k,
                        vkb.m,
                    );
                }
                if rec.finish_and_wait().is_ok() {
                    let mut pack = vec![0f32; nh * c.key_dim + nh * latent];
                    let _ = c_vk.download(by, 0, bytemuck::cast_slice_mut(&mut pack));
                    q_full.copy_from_slice(&pack[..nh * c.key_dim]);
                    q_lat.copy_from_slice(&pack[nh * c.key_dim..]);
                    true
                } else {
                    false
                }
            } else {
                false
            };
            #[cfg(not(feature = "vulkan"))]
            let did_gpu_qk = false;
            if !did_gpu_qk {
                #[cfg(feature = "vulkan")]
                self.mv_hybrid(&self.layers[li].wq_b, self.layers[li].vk_wq_b.as_ref(), 0, &q_an, &mut q_full);
                #[cfg(not(feature = "vulkan"))]
                self.mv_cpu(&self.layers[li].wq_b, 0, &q_an, &mut q_full);
            }

            self.rope_yarn_normal(&mut q_full, nh, c.key_dim, nope, pos);
            self.rope_yarn_normal(&mut kv_raw, 1, hd, latent, pos);

            // Normalize kv_latent (first 512 floats) & store in MLA KV cache
            let mut kvl_norm = vec![0f32; latent];
            ops::rms_norm(&kv_raw[..latent], &self.layers[li].kv_a_norm, &mut kvl_norm, c.eps);
            kv_raw[..latent].copy_from_slice(&kvl_norm);

            let kv_layer_base = li * self.max_ctx * hd;
            self.kv_cache[kv_layer_base + pos * hd..kv_layer_base + (pos + 1) * hd].copy_from_slice(&kv_raw);

            // Parallel 32-head latent attention
            let kv_slice = &self.kv_cache[kv_layer_base..kv_layer_base + (pos + 1) * hd];
            if did_gpu_qk {
                attn_lat.par_chunks_mut(latent).enumerate().for_each(|(h, ah)| {
                    let ql = &q_lat[h * latent..(h + 1) * latent];
                    let qp = &q_full[h * c.key_dim + nope..h * c.key_dim + c.key_dim];
                    let mut scores = vec![0f32; pos + 1];
                    let mut mx = f32::NEG_INFINITY;
                    for t in 0..=pos {
                        let kc = &kv_slice[t * hd..(t + 1) * hd];
                        let mut dot = 0f32;
                        for i in 0..latent { dot += ql[i] * kc[i]; }
                        for i in 0..n_rot { dot += qp[i] * kc[latent + i]; }
                        let s = dot * kq_scale;
                        scores[t] = s;
                        if s > mx { mx = s; }
                    }
                    let mut sum = 0f32;
                    for s in &mut scores { *s = (*s - mx).exp(); sum += *s; }
                    let inv = 1.0 / sum;
                    ah.fill(0.0);
                    for t in 0..=pos {
                        let w = scores[t] * inv;
                        let vc = &kv_slice[t * hd..t * hd + latent];
                        for i in 0..latent { ah[i] += w * vc[i]; }
                    }
                });
                // Fused GPU Submit 3: 32x wv_b -> wo in ONE command buffer!
                #[cfg(feature = "vulkan")]
                let did_gpu_vo = if let (Some(c_vk), Some(bx), Some(by), Some(bacc), Some(vvb), Some(vwo)) = (
                    &self.vk, &self.vk_x, &self.vk_y, &self.vk_acc, &self.layers[li].vk_wv_b, &self.layers[li].vk_wo,
                ) {
                    let _ = c_vk.upload(bx, 0, bytemuck::cast_slice(&attn_lat));
                    let mut rec = c_vk.begin();
                    for h in 0..nh {
                        rec.qmv(
                            vvb.ty,
                            &vvb.buf,
                            h as u64 * vvb.slice_bytes,
                            bx,
                            (h * latent) as u64,
                            by,
                            (h * vdim) as u64,
                            vvb.k,
                            vvb.m,
                        );
                    }
                    rec.qmv(vwo.ty, &vwo.buf, 0, by, 0, bacc, 0, vwo.k, vwo.m);
                    if rec.finish_and_wait().is_ok() {
                        let _ = c_vk.download(bacc, 0, bytemuck::cast_slice_mut(&mut block_out));
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                #[cfg(not(feature = "vulkan"))]
                let did_gpu_vo = false;
                if !did_gpu_vo {
                    let wv_b_ti = &self.layers[li].wv_b;
                    let wv_b_raw = self.g.tensor_data(wv_b_ti);
                    let wv_b_stride = wv_b_ti.row_bytes() * wv_b_ti.ne[1] as usize;
                    val_all.par_chunks_mut(vdim).enumerate().for_each(|(h, val_h)| {
                        let ah = &attn_lat[h * latent..(h + 1) * latent];
                        let wvb_h = &wv_b_raw[h * wv_b_stride..(h + 1) * wv_b_stride];
                        for r in 0..vdim {
                            val_h[r] = ocak_cpu::row_dot(wv_b_ti.ty, &wvb_h[r * wv_b_ti.row_bytes()..(r + 1) * wv_b_ti.row_bytes()], ah);
                        }
                    });
                    #[cfg(feature = "vulkan")]
                    self.mv_hybrid(&self.layers[li].wo, self.layers[li].vk_wo.as_ref(), 0, &val_all, &mut block_out);
                    #[cfg(not(feature = "vulkan"))]
                    self.mv_cpu(&self.layers[li].wo, 0, &val_all, &mut block_out);
                }
            } else {
                let wk_b_ti = &self.layers[li].wk_b;
                let wv_b_ti = &self.layers[li].wv_b;
                let wk_b_raw = self.g.tensor_data(wk_b_ti);
                let wv_b_raw = self.g.tensor_data(wv_b_ti);
                let wk_b_stride = wk_b_ti.row_bytes() * wk_b_ti.ne[1] as usize;
                let wv_b_stride = wv_b_ti.row_bytes() * wv_b_ti.ne[1] as usize;
                val_all.par_chunks_mut(vdim).enumerate().for_each(|(h, val_h)| {
                    let qn = &q_full[h * c.key_dim..h * c.key_dim + nope];
                    let qp = &q_full[h * c.key_dim + nope..h * c.key_dim + c.key_dim];
                    let mut ql = vec![0f32; latent];
                    let wkb_h = &wk_b_raw[h * wk_b_stride..(h + 1) * wk_b_stride];
                    for r in 0..latent {
                        ql[r] = ocak_cpu::row_dot(wk_b_ti.ty, &wkb_h[r * wk_b_ti.row_bytes()..(r + 1) * wk_b_ti.row_bytes()], qn);
                    }
                    let mut scores = vec![0f32; pos + 1];
                    let mut mx = f32::NEG_INFINITY;
                    for t in 0..=pos {
                        let kc = &kv_slice[t * hd..(t + 1) * hd];
                        let mut dot = 0f32;
                        for i in 0..latent { dot += ql[i] * kc[i]; }
                        for i in 0..n_rot { dot += qp[i] * kc[latent + i]; }
                        let s = dot * kq_scale;
                        scores[t] = s;
                        if s > mx { mx = s; }
                    }
                    let mut sum = 0f32;
                    for s in &mut scores { *s = (*s - mx).exp(); sum += *s; }
                    let mut ah = vec![0f32; latent];
                    for t in 0..=pos {
                        let w = scores[t] / sum;
                        let vc = &kv_slice[t * hd..t * hd + latent];
                        for i in 0..latent { ah[i] += w * vc[i]; }
                    }
                    let wvb_h = &wv_b_raw[h * wv_b_stride..(h + 1) * wv_b_stride];
                    for r in 0..vdim {
                        val_h[r] = ocak_cpu::row_dot(wv_b_ti.ty, &wvb_h[r * wv_b_ti.row_bytes()..(r + 1) * wv_b_ti.row_bytes()], &ah);
                    }
                });
                #[cfg(feature = "vulkan")]
                self.mv_hybrid(&self.layers[li].wo, self.layers[li].vk_wo.as_ref(), 0, &val_all, &mut block_out);
                #[cfg(not(feature = "vulkan"))]
                self.mv_cpu(&self.layers[li].wo, 0, &val_all, &mut block_out);
            }
            Self::hc_post(&mut streams, &block_out, &post, &comb, ne);

            // 2. Hyper-Connection pre + Dense or MoE FFN
            let (inp, post, comb) = self.hc_pre(&streams, &self.layers[li].hc_ffn);
            ops::rms_norm(&inp, &self.layers[li].ffn_norm, &mut normed, c.eps);

            match &self.layers[li].ffn {
                XingFfn::Dense {
                    gate, up, down,
                    #[cfg(feature = "vulkan")]
                    vk_gate,
                    #[cfg(feature = "vulkan")]
                    vk_up,
                    #[cfg(feature = "vulkan")]
                    vk_down,
                } => {
                    let mut g_buf = vec![0f32; c.n_ff_dense];
                    let mut u_buf = vec![0f32; c.n_ff_dense];
                    let mut a_buf = vec![0f32; c.n_ff_dense];
                    #[cfg(feature = "vulkan")]
                    {
                        self.mv_hybrid(gate, vk_gate.as_ref(), 0, &normed, &mut g_buf);
                        self.mv_hybrid(up, vk_up.as_ref(), 0, &normed, &mut u_buf);
                        ops::swiglu(&g_buf, &u_buf, &mut a_buf);
                        self.mv_hybrid(down, vk_down.as_ref(), 0, &a_buf, &mut block_out);
                    }
                    #[cfg(not(feature = "vulkan"))]
                    {
                        self.mv_cpu(gate, 0, &normed, &mut g_buf);
                        self.mv_cpu(up, 0, &normed, &mut u_buf);
                        ops::swiglu(&g_buf, &u_buf, &mut a_buf);
                        self.mv_cpu(down, 0, &a_buf, &mut block_out);
                    }
                }
                XingFfn::Moe {
                    gate_inp, exp_probs_b, gate_exps, up_exps, down_exps,
                    #[cfg(feature = "vulkan")]
                    vk_gate_exps,
                    #[cfg(feature = "vulkan")]
                    vk_up_exps,
                    #[cfg(feature = "vulkan")]
                    vk_down_exps,
                    gate_shexp, up_shexp, down_shexp,
                    #[cfg(feature = "vulkan")]
                    vk_gate_sh,
                    #[cfg(feature = "vulkan")]
                    vk_up_sh,
                    #[cfg(feature = "vulkan")]
                    vk_down_sh,
                } => {
                    let mut r_logits = vec![0f32; c.n_expert];
                    self.mv_cpu(gate_inp, 0, &normed, &mut r_logits);
                    let probs: Vec<f32> = r_logits.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
                    let mut sel: Vec<(usize, f32)> = probs
                        .iter()
                        .enumerate()
                        .map(|(idx, &p)| (idx, p + exp_probs_b.as_ref().map(|b| b[idx]).unwrap_or(0.0)))
                        .collect();
                    sel.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    sel.truncate(c.n_used);

                    let mut wts: Vec<f32> = sel.iter().map(|&(idx, _)| probs[idx]).collect();
                    if c.exp_weights_norm {
                        let sum = wts.iter().sum::<f32>().max(6.103_515_6e-5);
                        for w in &mut wts { *w /= sum; }
                    }
                    if c.exp_weights_scale != 1.0 && c.exp_weights_scale != 0.0 {
                        for w in &mut wts { *w *= c.exp_weights_scale; }
                    }

                    #[cfg(feature = "vulkan")]
                    let did_fused_gpu = if let (
                        Some(ctx), Some(bx), Some(by), Some(bg), Some(bu), Some(ba), Some(bacc),
                        Some(vg), Some(vu), Some(vd), Some(vgs), Some(vus), Some(vds),
                    ) = (
                        &self.vk, &self.vk_x, &self.vk_y, &self.vk_g, &self.vk_u, &self.vk_a, &self.vk_acc,
                        vk_gate_exps.as_ref(), vk_up_exps.as_ref(), vk_down_exps.as_ref(),
                        vk_gate_sh.as_ref(), vk_up_sh.as_ref(), vk_down_sh.as_ref(),
                    ) {
                        if ctx.upload(bx, 0, bytemuck::cast_slice(&normed)).is_ok() {
                            let mut rec = ctx.begin();
                            rec.zero_buf(bacc);
                            for (k_i, &(exp_id, _)) in sel.iter().enumerate() {
                                rec.qmv(vg.ty, &vg.buf, exp_id as u64 * vg.slice_bytes, bx, 0, bg, 0, vg.k, vg.m);
                                rec.qmv(vu.ty, &vu.buf, exp_id as u64 * vu.slice_bytes, bx, 0, bu, 0, vu.k, vu.m);
                                rec.swiglu(bg, 0, bu, 0, ba, 0, c.n_ff_exp as u32);
                                rec.qmv(vd.ty, &vd.buf, exp_id as u64 * vd.slice_bytes, ba, 0, by, 0, vd.k, vd.m);
                                rec.add_scaled(bacc, 0, by, 0, ne as u32, wts[k_i]);
                            }
                            rec.qmv(vgs.ty, &vgs.buf, 0, bx, 0, bg, 0, vgs.k, vgs.m);
                            rec.qmv(vus.ty, &vus.buf, 0, bx, 0, bu, 0, vus.k, vus.m);
                            rec.swiglu(bg, 0, bu, 0, ba, 0, c.n_ff_exp as u32);
                            rec.qmv(vds.ty, &vds.buf, 0, ba, 0, by, 0, vds.k, vds.m);
                            rec.add(bacc, 0, by, 0, ne as u32);
                            rec.finish_and_wait().is_ok() && ctx.download(bacc, 0, bytemuck::cast_slice_mut(&mut block_out)).is_ok()
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    #[cfg(not(feature = "vulkan"))]
                    let did_fused_gpu = false;

                    if !did_fused_gpu {
                        block_out.fill(0.0);
                        let mut g_e = vec![0f32; c.n_ff_exp];
                        let mut u_e = vec![0f32; c.n_ff_exp];
                        let mut a_e = vec![0f32; c.n_ff_exp];
                        let mut d_e = vec![0f32; ne];
                        for (k_i, &(exp_id, _)) in sel.iter().enumerate() {
                            self.mv_cpu(gate_exps, exp_id, &normed, &mut g_e);
                            self.mv_cpu(up_exps, exp_id, &normed, &mut u_e);
                            ops::swiglu(&g_e, &u_e, &mut a_e);
                            self.mv_cpu(down_exps, exp_id, &a_e, &mut d_e);
                            let w = wts[k_i];
                            for e in 0..ne { block_out[e] += w * d_e[e]; }
                        }
                        self.mv_cpu(gate_shexp, 0, &normed, &mut g_e);
                        self.mv_cpu(up_shexp, 0, &normed, &mut u_e);
                        ops::swiglu(&g_e, &u_e, &mut a_e);
                        self.mv_cpu(down_shexp, 0, &a_e, &mut d_e);
                        ops::add(&mut block_out, &d_e);
                    }
                }
            }
            Self::hc_post(&mut streams, &block_out, &post, &comb, ne);
        }

        if out.logits || out.hidden {
            let mut mean = vec![0f32; ne];
            for h in 0..4 {
                let sl = &streams[h * ne..(h + 1) * ne];
                for e in 0..ne { mean[e] += 0.25 * sl[e]; }
            }
            ops::rms_norm(&mean, &self.out_norm, &mut self.hidden, c.eps);
        }
        if out.logits {
            let h = self.hidden.clone();
            let mut lg = std::mem::take(&mut self.logits);
            #[cfg(feature = "vulkan")]
            self.mv_hybrid(&self.output, self.vk_output.as_ref(), 0, &h, &mut lg);
            #[cfg(not(feature = "vulkan"))]
            self.mv_cpu(&self.output, 0, &h, &mut lg);
            self.logits = lg;
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn xing_vk_vs_cpu_parity() {
        let home = std::env::var("HOME").unwrap_or_default();
        let p = std::path::PathBuf::from(home).join("models/Xing4.0-29B-A4B-IQ3_XXS.gguf");
        if !p.exists() { return; }
        let g = Arc::new(GgufFile::open(&p).unwrap());
        let mut vk = XingExec::new(Arc::clone(&g), true, 64, 0).unwrap();
        let mut cpu = XingExec::new(Arc::clone(&g), false, 64, 40).unwrap();
        let toks = [1u32, 100, 200];
        for (pos, &tok) in toks.iter().enumerate() {
            let out = Out { logits: true, hidden: false };
            vk.forward(tok, pos, &[], out).unwrap();
            cpu.forward(tok, pos, &[], out).unwrap();
            let lv = vk.logits();
            let lc = cpu.logits();
            let mut dot = 0.0f64;
            let mut nv = 0.0f64;
            let mut nc = 0.0f64;
            let mut max_d = 0.0f32;
            for (&a, &b) in lv.iter().zip(lc.iter()) {
                dot += a as f64 * b as f64;
                nv += a as f64 * a as f64;
                nc += b as f64 * b as f64;
                max_d = max_d.max((a - b).abs());
            }
            let cos = dot / (nv.sqrt() * nc.sqrt());
            let top_v = lv.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
            let top_c = lc.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
            println!("pos={pos} cos_sim={cos:.6} max_diff={max_d:.4} top_vk=({}, {:.3}) top_cpu=({}, {:.3})", top_v.0, top_v.1, top_c.0, top_c.1);
            assert!(cos > 0.999, "Xing VK vs CPU mismatch at pos={pos}: cos_sim={cos}");
        }
    }
}
