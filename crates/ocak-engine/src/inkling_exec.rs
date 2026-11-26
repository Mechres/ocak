//! Native hybrid GPU+CPU executor for `inkling2_loop` (`Inkling-2-Mini`, 26.4B looped MoE):
//! - 24 physical layers with middle 12 (`[6..18)`) looped 3x (48 logical layer visits & cache slots)
//! - Depth-wise Attention Residuals (`mix` over block outputs) + cross-pass loop controllers (`gain`/`forget`)
//! - Causal `ShortConv` on K, V, attn_out, ffn_out (with dynamic per-head taps on full-attention layers)
//! - Relative-position bias + per-query attention sinks + Q/K/R L2 clipping + log-scaled full attention
//! - 128-expert / 8-active MoE with `NVFP4` weights, per-expert `.scale`, sigmoid routing + `SITU` activation
//! - Hybrid VRAM+RAM MoE: dense/attention/loop/output + `gpu_moe` expert layers on Vulkan, rest zero-copy `mmap` CPU.

use crate::exec::{EngineError, Executor, Out};
use ocak_gguf::{quant::dequantize_row, GgmlType, GgufFile};
use ocak_vk::{Buffer, VkContext, WType};
use rayon::prelude::*;
use std::sync::Arc;

struct Mat2D {
    name: String,
    ty: GgmlType,
    k: usize,
    m: usize,
    gpu: Option<(Buffer, WType)>,
}

struct MoEExps {
    gu_name: String,
    dn_name: String,
    gu_ty: GgmlType,
    dn_ty: GgmlType,
    gu_stride: usize,
    dn_stride: usize,
    gu_scale: Vec<f32>,
    dn_scale: Vec<f32>,
    gpu_gu: Option<(Buffer, WType)>,
    gpu_dn: Option<(Buffer, WType)>,
}

struct LoopLayer {
    msg_query: Vec<f32>, // [4 * n_embd]
    ctrl_down: [Mat2D; 4],
    ctrl_gain: [Mat2D; 4],
    ctrl_forget: [Mat2D; 4],
}

struct PhysLayer {
    is_swa: bool,
    n_head: usize,
    n_head_kv: usize,
    n_extent: usize,
    wq: Mat2D,
    wk: Mat2D,
    wv: Mat2D,
    wo: Mat2D,
    wqkv_gate: Mat2D,
    attn_q_clip: Vec<f32>,
    attn_sink_proj: Mat2D,
    attn_rel: Mat2D,
    attn_rel_bias: Vec<f32>, // [n_extent * n_rel]
    attn_rel_clip: Vec<f32>,
    attn_k_conv: Vec<f32>,
    attn_v_conv: Vec<f32>,
    attn_post_conv: Vec<f32>,
    ffn_post_conv: Vec<f32>,
    // SWA vs Full norms & dynamic conv proj
    attn_norm: Option<Vec<f32>>,
    attn_q_inp_norm: Option<Vec<f32>>,
    attn_k_inp_norm: Option<Vec<f32>>,
    attn_v_inp_norm: Option<Vec<f32>>,
    attn_k_conv_proj: Option<Mat2D>,
    attn_v_conv_proj: Option<Mat2D>,
    // Attention residuals
    attn_res_query: Option<Vec<f32>>, // [n_embd] (SWA) or [3 * n_embd] (Full)
    ffn_res_query: Vec<f32>,
    // MoE
    ffn_norm: Vec<f32>,
    ffn_gate_inp: Mat2D,
    exp_probs_b: Option<Vec<f32>>,
    exps: MoEExps,
}

struct SlotState {
    // KV cache: [max_ctx * n_head_kv * head_dim]
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    // ShortConv history: (n_conv - 1) slots per channel
    k_hist: Vec<f32>,
    v_hist: Vec<f32>,
    attn_post_hist: Vec<f32>,
    ffn_post_hist: Vec<f32>,
}

struct InklingGpuPhys {
    attn_q_clip: Buffer,
    attn_rel_bias: Buffer,
    attn_rel_clip: Buffer,
    attn_k_conv: Buffer,
    attn_v_conv: Buffer,
    attn_post_conv: Buffer,
    ffn_post_conv: Buffer,
    attn_norm: Option<Buffer>,
    attn_q_inp_norm: Option<Buffer>,
    attn_k_inp_norm: Option<Buffer>,
    attn_v_inp_norm: Option<Buffer>,
    attn_res_query: Option<Buffer>,
    ffn_res_query: Buffer,
    ffn_norm: Buffer,
    exp_probs_b: Option<Buffer>,
    gu_scale: Buffer,
    dn_scale: Buffer,
}

struct InklingGpuLoop {
    msg_query: Buffer,
    ctrl_down: [(Buffer, WType, u32, u32); 4],
    ctrl_gain: [(Buffer, WType, u32, u32); 4],
    ctrl_forget: [(Buffer, WType, u32, u32); 4],
}

struct InklingGpuState {
    phys: Vec<InklingGpuPhys>,
    loops: Vec<InklingGpuLoop>,
    slot_kv_offsets: Vec<u32>,
    slot_caps: Vec<u32>,
    ones: Buffer,
    tok_norm: Buffer,
    output_norm: Buffer,
    output_res_query: Buffer,
    dummy_bias: Buffer,
    dummy_dyn: Buffer,
    // Activations & ping-pong
    cur: Buffer,
    inp_sa: Buffer,
    prelude_cur: Buffer,
    bank_out: Buffer,
    ffn_inp: Buffer,
    blocks_norm: Buffer,
    bank_norm: Buffer,
    inputs_buf: Buffer,   // 3 * ne
    in_norm_buf: Buffer,  // 3 * ne
    msgs_buf: Buffer,     // 4 * ne
    ctrl_cat: Buffer,     // 2 * ne
    ctrl_gate: Buffer,    // 3 * rank
    ctrl_gf: Buffer,      // 6 * ne
    ctrl_rank: u32,
    proj_buf: Buffer,     // q + k + v + r + sinks + gate + k_dyn + v_dyn
    k_conv_buf: Buffer,   // kv_dim
    v_conv_buf: Buffer,   // kv_dim
    attn_cat: Buffer,     // n_head * hd
    o_proj: Buffer,       // ne
    x_moe: Buffer,        // ne
    router_logits: Buffer, // n_expert
    top_ids: Buffer,      // n_expert_used
    top_w: Buffer,        // n_expert_used
    moe_gu: Buffer,       // n_expert_used * 2 * n_ff_exp
    moe_mid: Buffer,      // n_expert_used * n_ff_exp
    moe_tmp: Buffer,      // n_expert_used * ne
    moe_out: Buffer,      // ne
    hidden_buf: Buffer,   // ne
    logits_buf: Buffer,   // n_vocab_padded
    // Persistent slot states (KV + ShortConv histories)
    k_cache: Buffer,
    v_cache: Buffer,
    k_hist: Buffer,
    v_hist: Buffer,
    attn_post_hist: Buffer,
    ffn_post_hist: Buffer,
    kv_dim: usize,
}

fn upload_f32(c: &Arc<VkContext>, v: &[f32]) -> Result<Buffer, EngineError> {
    let bytes = bytemuck::cast_slice(v);
    let b = c.create_device_buffer(bytes.len() as u64).map_err(|e| EngineError::Backend(e.to_string()))?;
    c.upload(&b, 0, bytes).map_err(|e| EngineError::Backend(e.to_string()))?;
    Ok(b)
}

fn upload_mat_gtt_or_vram(g: &GgufFile, c: &Arc<VkContext>, name: &str) -> Result<(Buffer, WType, u32, u32), EngineError> {
    let t = g.tensor(name).ok_or_else(|| EngineError::Unsupported(format!("missing {name}")))?;
    let wt = WType::from_ggml(t.ty).map_err(|e| EngineError::Backend(e.to_string()))?;
    let data = g.tensor_data(t);
    let buf = if let Ok(b) = c.create_device_buffer(data.len() as u64) {
        c.upload(&b, 0, data).map_err(|e| EngineError::Backend(e.to_string()))?;
        b
    } else {
        c.create_gtt_buffer_from_slice(data).map_err(|e| EngineError::Backend(e.to_string()))?
    };
    Ok((buf, wt, t.n_cols() as u32, t.n_rows() as u32))
}

pub struct InklingExec {
    gguf: Arc<GgufFile>,
    vk: Option<Arc<VkContext>>,
    n_phys: usize,
    core_start: usize,
    core_end: usize,
    n_passes: usize,
    n_embd: usize,
    n_ff_exp: usize,
    head_dim: usize,
    n_expert: usize,
    n_expert_used: usize,
    n_rel: usize,
    n_swa: usize,
    n_conv: usize,
    block_size: usize,
    n_vocab_unpadded: usize,
    eps: f32,
    qk_clip: f32,
    rel_clip: f32,
    log_scale_floor: f32,
    log_scale_alpha: f32,
    exp_w_scale: f32,
    situ_beta: f32,
    situ_lin_beta: f32,
    softcap: f32,
    max_ctx: usize,
    tok_embd_name: String,
    tok_embd_ty: GgmlType,
    tok_norm: Vec<f32>,
    output_norm: Vec<f32>,
    output_res_query: Vec<f32>,
    output_w: Mat2D,
    phys: Vec<PhysLayer>,
    loop_layers: Vec<LoopLayer>,
    slots: Vec<SlotState>,
    // GPU scratch buffers
    gpu_x: Option<Buffer>,
    gpu_y: Option<Buffer>,
    gpu_gu: Option<Buffer>,
    gpu_mid: Option<Buffer>,
    gpu_sum: Option<Buffer>,
    gpu_moe_count: usize,
    cpu_moe_count: usize,
    gpu_pipe: Option<InklingGpuState>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
}

fn load_f32_vec(g: &GgufFile, name: &str) -> Result<Vec<f32>, EngineError> {
    let t = g.tensor(name).ok_or_else(|| EngineError::Unsupported(format!("missing {name}")))?;
    let n = t.n_elems() as usize;
    let mut out = vec![0f32; n];
    dequantize_row(t.ty, g.tensor_data(t), &mut out, n);
    Ok(out)
}

fn load_mat(g: &GgufFile, vk: Option<&Arc<VkContext>>, name: &str, to_gpu: bool) -> Result<Mat2D, EngineError> {
    let t = g.tensor(name).ok_or_else(|| EngineError::Unsupported(format!("missing {name}")))?;
    let (k, m) = (t.n_cols(), t.n_rows());
    let gpu = if to_gpu {
        if let (Some(c), Ok(wt)) = (vk, WType::from_ggml(t.ty)) {
            let data = g.tensor_data(t);
            let buf = c.create_device_buffer(data.len() as u64).map_err(|e| EngineError::Backend(e.to_string()))?;
            c.upload(&buf, 0, data).map_err(|e| EngineError::Backend(e.to_string()))?;
            Some((buf, wt))
        } else {
            None
        }
    } else {
        None
    };
    Ok(Mat2D { name: name.to_string(), ty: t.ty, k, m, gpu })
}

#[inline]
fn rms_norm_unweighted(x: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| v * s).collect()
}

#[inline]
fn rms_norm_weighted(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, wt)| v * s * wt).collect()
}

#[inline]
fn clip_heads(x: &mut [f32], scale: Option<&[f32]>, head_dim: usize, n_head: usize, threshold: f32) {
    for h in 0..n_head {
        let sl = &mut x[h * head_dim..(h + 1) * head_dim];
        if let Some(sc) = scale {
            let ssc = &sc[h * head_dim..(h + 1) * head_dim];
            for (v, &s) in sl.iter_mut().zip(ssc) {
                *v *= s;
            }
        }
        let norm = sl.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > threshold {
            let f = threshold / norm;
            for v in sl.iter_mut() {
                *v *= f;
            }
        }
    }
}

#[inline]
fn mix_depth(blocks: &[Vec<f32>], cur: &[f32], query: &[f32], n_embd: usize, n_q: usize) -> Vec<Vec<f32>> {
    // states = [rms_norm(b) for b in blocks] + [rms_norm(cur)]
    let n_st = blocks.len() + 1;
    let mut states: Vec<Vec<f32>> = Vec::with_capacity(n_st);
    for b in blocks {
        states.push(rms_norm_unweighted(b, 1e-6));
    }
    states.push(rms_norm_unweighted(cur, 1e-6));

    let inv_sqrt = 1.0 / (n_embd as f32).sqrt();
    let mut outs = Vec::with_capacity(n_q);
    for q in 0..n_q {
        let qv = &query[q * n_embd..(q + 1) * n_embd];
        let mut sc = vec![0f32; n_st];
        let mut mx = f32::NEG_INFINITY;
        for (i, st) in states.iter().enumerate() {
            let mut dot = 0f32;
            for j in 0..n_embd {
                dot += st[j] * qv[j];
            }
            sc[i] = dot * inv_sqrt;
            if sc[i] > mx { mx = sc[i]; }
        }
        let mut sum = 0f32;
        for v in &mut sc {
            *v = (*v - mx).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        let mut out = vec![0f32; n_embd];
        for (i, st) in states.iter().enumerate() {
            let w = sc[i] * inv;
            for j in 0..n_embd {
                out[j] += w * st[j];
            }
        }
        outs.push(out);
    }
    outs
}

fn convolve_step(
    cur: &[f32],
    weight: &[f32],      // [n_channels * n_conv]
    dynamic: Option<&[f32]>, // [n_heads * n_conv]
    hist: &mut [f32],    // [n_channels * (n_conv - 1)]
    n_conv: usize,
) -> Vec<f32> {
    let n_ch = cur.len();
    let nh = n_conv - 1;
    let mut out = vec![0f32; n_ch];
    let scale = 1.0 / ((if dynamic.is_some() { 2 } else { 1 } * n_conv + 1) as f32).sqrt();
    let n_heads = dynamic.map(|d| d.len() / n_conv).unwrap_or(0);
    let head_ch = if n_heads > 0 { n_ch / n_heads } else { 0 };

    for c in 0..n_ch {
        let h_off = c * nh;
        let w_off = c * n_conv;
        // x = [hist[0], hist[1], ..., hist[nh-1], cur[c]]
        let mut acc = 0f32;
        for k in 0..nh {
            acc += hist[h_off + k] * weight[w_off + k];
        }
        acc += cur[c] * weight[w_off + nh];

        if let Some(dyn_taps) = dynamic {
            let h = c / head_ch;
            let d_off = h * n_conv;
            // lag=0 -> cur[c], lag=1 -> hist[nh-1], ..., lag=nh -> hist[0]
            acc += cur[c] * dyn_taps[d_off];
            for lag in 1..n_conv {
                acc += hist[h_off + (nh - lag)] * dyn_taps[d_off + lag];
            }
        }

        // Shift history left and append cur[c]
        for k in 0..nh - 1 {
            hist[h_off + k] = hist[h_off + k + 1];
        }
        hist[h_off + nh - 1] = cur[c];

        out[c] = (cur[c] + acc) * scale;
    }
    out
}

impl InklingExec {
    pub fn new(g: Arc<GgufFile>, use_vulkan: bool, max_ctx: usize, n_cpu_moe: usize) -> Result<Self, EngineError> {
        let vk = if use_vulkan { Some(VkContext::new().map_err(|e| EngineError::Backend(e.to_string()))?) } else { None };
        let req = |s: &str| g.arch_u64(s).ok_or_else(|| EngineError::Unsupported(format!("missing inkling2_loop.{s}")));
        let n_phys = req("block_count")? as usize;
        let core_start = req("loop.start")? as usize;
        let core_end = req("loop.end")? as usize;
        let n_passes = req("loop.count")? as usize;
        let n_slot = n_phys + (n_passes - 1) * (core_end - core_start);
        let n_embd = req("embedding_length")? as usize;
        let n_ff_exp = g.arch_u64("expert_feed_forward_length").or_else(|| g.arch_u64("feed_forward_length")).unwrap_or(1152) as usize;
        let head_dim = g.arch_u64("attention.key_length").unwrap_or(128) as usize;
        let n_expert = req("expert_count")? as usize;
        let n_expert_used = req("expert_used_count")? as usize;
        let n_rel = g.arch_u64("attention.relative_length").unwrap_or(16) as usize;
        let n_rel_ext = g.arch_u64("attention.relative_extent").unwrap_or(1024) as usize;
        let n_swa = g.arch_u64("attention.sliding_window").unwrap_or(1024) as usize;
        let n_conv = g.arch_u64("shortconv.l_cache").unwrap_or(4) as usize;
        let block_size = g.arch_u64("attn_res.block_size").unwrap_or(6) as usize;
        let n_vocab_padded = g.arch_u64("vocab_size").unwrap_or(201024) as usize;
        let n_vocab_unpadded = g.arch_u64("vocab_size_unpadded").unwrap_or(n_vocab_padded as u64) as usize;
        let eps = g.arch_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32;
        let qk_clip = g.arch_f64("attention.qk_clip").unwrap_or(22.627417) as f32;
        let rel_clip = g.arch_f64("attention.relative_clip").unwrap_or(8.0) as f32;
        let log_scale_floor = g.arch_u64("attention.log_scale_floor").unwrap_or(8192) as f32;
        let log_scale_alpha = g.arch_f64("attention.log_scale_alpha").unwrap_or(0.1) as f32;
        let exp_w_scale = g.arch_f64("expert_weights_scale").unwrap_or(8.0) as f32;
        let situ_beta = g.arch_f64("activation.situ_beta").unwrap_or(4.0) as f32;
        let situ_lin_beta = g.arch_f64("activation.situ_linear_beta").unwrap_or(25.0) as f32;
        let softcap = g.arch_f64("final_logit_softcapping").unwrap_or(30.0) as f32;

        let head_arr = g.ints("inkling2_loop.attention.head_count").map(|s| s.to_vec()).unwrap_or_else(|| vec![24; n_phys]);
        let kv_arr = g.ints("inkling2_loop.attention.head_count_kv").map(|s| s.to_vec()).unwrap_or_else(|| vec![8; n_phys]);
        let swa_arr = g.ints("inkling2_loop.attention.sliding_window_pattern").map(|s| s.to_vec()).unwrap_or_else(|| vec![1; n_phys]);

        let tok_t = g.tensor("token_embd.weight").ok_or_else(|| EngineError::Unsupported("missing token_embd.weight".into()))?;
        let tok_embd_ty = tok_t.ty;
        let tok_norm = load_f32_vec(&g, "token_embd_norm.weight")?;
        let output_norm = load_f32_vec(&g, "output_norm.weight")?;
        let output_res_query = load_f32_vec(&g, "output_res_query.weight")?;
        let output_w = load_mat(&g, vk.as_ref(), "output.weight", vk.is_some())?;

        let gpu_moe_limit = if vk.is_some() { n_phys.saturating_sub(n_cpu_moe) } else { 0 };
        let mut gpu_moe_count = 0usize;
        let mut cpu_moe_count = 0usize;
        let mut phys = Vec::with_capacity(n_phys);

        for i in 0..n_phys {
            let is_swa = swa_arr.get(i).copied().unwrap_or(1) != 0;
            let n_head = head_arr.get(i).copied().unwrap_or(24) as usize;
            let n_head_kv = kv_arr.get(i).copied().unwrap_or(8) as usize;
            let n_extent = if is_swa { n_swa } else { n_rel_ext };

            let wq = load_mat(&g, vk.as_ref(), &format!("blk.{i}.attn_q.weight"), vk.is_some())?;
            let wk = load_mat(&g, vk.as_ref(), &format!("blk.{i}.attn_k.weight"), vk.is_some())?;
            let wv = load_mat(&g, vk.as_ref(), &format!("blk.{i}.attn_v.weight"), vk.is_some())?;
            let wo = load_mat(&g, vk.as_ref(), &format!("blk.{i}.attn_output.weight"), vk.is_some())?;
            let wqkv_gate = load_mat(&g, vk.as_ref(), &format!("blk.{i}.attn_gate.weight"), vk.is_some())?;
            let attn_q_clip = load_f32_vec(&g, &format!("blk.{i}.attn_q_clip.weight"))?;
            let attn_sink_proj = load_mat(&g, vk.as_ref(), &format!("blk.{i}.attn_sink_proj.weight"), vk.is_some())?;
            let attn_rel = load_mat(&g, vk.as_ref(), &format!("blk.{i}.attn_rel.weight"), vk.is_some())?;
            let attn_rel_bias = load_f32_vec(&g, &format!("blk.{i}.attn_rel_bias.weight"))?;
            let attn_rel_clip = load_f32_vec(&g, &format!("blk.{i}.attn_rel_clip.weight"))?;
            let attn_k_conv = load_f32_vec(&g, &format!("blk.{i}.attn_k_conv.weight"))?;
            let attn_v_conv = load_f32_vec(&g, &format!("blk.{i}.attn_v_conv.weight"))?;
            let attn_post_conv = load_f32_vec(&g, &format!("blk.{i}.attn_post_conv.weight"))?;
            let ffn_post_conv = load_f32_vec(&g, &format!("blk.{i}.ffn_post_conv.weight"))?;

            let (attn_norm, attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm, attn_k_conv_proj, attn_v_conv_proj) = if is_swa {
                (Some(load_f32_vec(&g, &format!("blk.{i}.attn_norm.weight"))?), None, None, None, None, None)
            } else {
                (
                    None,
                    Some(load_f32_vec(&g, &format!("blk.{i}.attn_q_inp_norm.weight"))?),
                    Some(load_f32_vec(&g, &format!("blk.{i}.attn_k_inp_norm.weight"))?),
                    Some(load_f32_vec(&g, &format!("blk.{i}.attn_v_inp_norm.weight"))?),
                    Some(load_mat(&g, vk.as_ref(), &format!("blk.{i}.attn_k_conv_proj.weight"), vk.is_some())?),
                    Some(load_mat(&g, vk.as_ref(), &format!("blk.{i}.attn_v_conv_proj.weight"), vk.is_some())?),
                )
            };

            let attn_res_query = if i > 0 {
                Some(load_f32_vec(&g, &format!("blk.{i}.attn_res_query.weight"))?)
            } else {
                None
            };
            let ffn_res_query = load_f32_vec(&g, &format!("blk.{i}.ffn_res_query.weight"))?;
            let ffn_norm = load_f32_vec(&g, &format!("blk.{i}.ffn_norm.weight"))?;
            let ffn_gate_inp = load_mat(&g, vk.as_ref(), &format!("blk.{i}.ffn_gate_inp.weight"), vk.is_some())?;
            let exp_probs_b = load_f32_vec(&g, &format!("blk.{i}.exp_probs_b.bias")).ok();

            let gu_name = format!("blk.{i}.ffn_gate_up_exps.weight");
            let dn_name = format!("blk.{i}.ffn_down_exps.weight");
            let t_gu = g.tensor(&gu_name).ok_or_else(|| EngineError::Unsupported(format!("missing {gu_name}")))?;
            let t_dn = g.tensor(&dn_name).ok_or_else(|| EngineError::Unsupported(format!("missing {dn_name}")))?;
            let gu_stride = t_gu.ty.row_bytes(n_embd).unwrap() * (2 * n_ff_exp);
            let dn_stride = t_dn.ty.row_bytes(n_ff_exp).unwrap() * n_embd;
            let gu_scale = load_f32_vec(&g, &format!("blk.{i}.ffn_gate_up_exps.scale")).unwrap_or_else(|_| vec![1.0; n_expert]);
            let dn_scale = load_f32_vec(&g, &format!("blk.{i}.ffn_down_exps.scale")).unwrap_or_else(|_| vec![1.0; n_expert]);

            // Defer MoE expert GPU placement until after all non-MoE weights are placed so we can
            // prioritize VRAM for the 3x-looped core (core_start..core_end) before prelude/postlude!
            phys.push(PhysLayer {
                is_swa, n_head, n_head_kv, n_extent,
                wq, wk, wv, wo, wqkv_gate, attn_q_clip, attn_sink_proj,
                attn_rel, attn_rel_bias, attn_rel_clip,
                attn_k_conv, attn_v_conv, attn_post_conv, ffn_post_conv,
                attn_norm, attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm,
                attn_k_conv_proj, attn_v_conv_proj,
                attn_res_query, ffn_res_query, ffn_norm, ffn_gate_inp, exp_probs_b,
                exps: MoEExps { gu_name, dn_name, gu_ty: t_gu.ty, dn_ty: t_dn.ty, gu_stride, dn_stride, gu_scale, dn_scale, gpu_gu: None, gpu_dn: None },
            });
        }

        // Upload MoE experts with VRAM priority: 3x-looped core FIRST, then prelude & postlude!
        let mut moe_order: Vec<usize> = Vec::with_capacity(n_phys);
        for i in core_start..core_end { moe_order.push(i); }
        for i in 0..core_start { moe_order.push(i); }
        for i in core_end..n_phys { moe_order.push(i); }
        let mut vram_moe_placed = 0usize;
        for &i in &moe_order {
            let ex = &mut phys[i].exps;
            let t_gu = g.tensor(&ex.gu_name).unwrap();
            let t_dn = g.tensor(&ex.dn_name).unwrap();
            let put_vram = vram_moe_placed < gpu_moe_limit;
            let (gpu_gu, gpu_dn) = if let (Some(c), Ok(wt_gu), Ok(wt_dn)) = (vk.as_ref(), WType::from_ggml(t_gu.ty), WType::from_ggml(t_dn.ty)) {
                if put_vram {
                    let b_gu = c.create_device_buffer(t_gu.n_bytes).ok();
                    let b_dn = c.create_device_buffer(t_dn.n_bytes).ok();
                    if let (Some(b_gu), Some(b_dn)) = (b_gu, b_dn) {
                        c.upload(&b_gu, 0, g.tensor_data(t_gu)).map_err(|e| EngineError::Backend(e.to_string()))?;
                        c.upload(&b_dn, 0, g.tensor_data(t_dn)).map_err(|e| EngineError::Backend(e.to_string()))?;
                        vram_moe_placed += 1;
                        gpu_moe_count += 1;
                        (Some((b_gu, wt_gu)), Some((b_dn, wt_dn)))
                    } else if let (Ok(b_gu), Ok(b_dn)) = (c.create_gtt_buffer_from_slice(g.tensor_data(t_gu)), c.create_gtt_buffer_from_slice(g.tensor_data(t_dn))) {
                        gpu_moe_count += 1;
                        (Some((b_gu, wt_gu)), Some((b_dn, wt_dn)))
                    } else {
                        cpu_moe_count += 1;
                        (None, None)
                    }
                } else if let (Ok(b_gu), Ok(b_dn)) = (c.create_gtt_buffer_from_slice(g.tensor_data(t_gu)), c.create_gtt_buffer_from_slice(g.tensor_data(t_dn))) {
                    gpu_moe_count += 1;
                    (Some((b_gu, wt_gu)), Some((b_dn, wt_dn)))
                } else {
                    cpu_moe_count += 1;
                    (None, None)
                }
            } else {
                cpu_moe_count += 1;
                (None, None)
            };
            ex.gpu_gu = gpu_gu;
            ex.gpu_dn = gpu_dn;
        }

        let mut loop_layers = Vec::with_capacity(core_end - core_start);
        for ip in core_start..core_end {
            let msg_query = load_f32_vec(&g, &format!("blk.{ip}.loop_msg_query.weight"))?;
            let names = ["q", "k", "v", "ffn"];
            let mut down = Vec::with_capacity(4);
            let mut gain = Vec::with_capacity(4);
            let mut forget = Vec::with_capacity(4);
            for nm in names {
                down.push(load_mat(&g, vk.as_ref(), &format!("blk.{ip}.loop_{nm}_ctrl_down.weight"), false)?);
                gain.push(load_mat(&g, vk.as_ref(), &format!("blk.{ip}.loop_{nm}_ctrl_gain.weight"), false)?);
                forget.push(load_mat(&g, vk.as_ref(), &format!("blk.{ip}.loop_{nm}_ctrl_forget.weight"), false)?);
            }
            loop_layers.push(LoopLayer {
                msg_query,
                ctrl_down: down.try_into().ok().unwrap(),
                ctrl_gain: gain.try_into().ok().unwrap(),
                ctrl_forget: forget.try_into().ok().unwrap(),
            });
        }

        let mut slots = Vec::with_capacity(n_slot);
        for il in 0..n_slot {
            let ip = if il < n_phys { il } else { core_start + (il - n_phys) % (core_end - core_start) };
            let kv_dim = phys[ip].n_head_kv * head_dim;
            let ctx_cap = if phys[ip].is_swa { max_ctx.min(n_swa + 16) } else { max_ctx };
            slots.push(SlotState {
                k_cache: vec![0f32; ctx_cap * kv_dim],
                v_cache: vec![0f32; ctx_cap * kv_dim],
                k_hist: vec![0f32; kv_dim * (n_conv - 1)],
                v_hist: vec![0f32; kv_dim * (n_conv - 1)],
                attn_post_hist: vec![0f32; n_embd * (n_conv - 1)],
                ffn_post_hist: vec![0f32; n_embd * (n_conv - 1)],
            });
        }

        let max_dim = n_vocab_padded.max(n_embd * 8);
        let (gpu_x, gpu_y, gpu_gu, gpu_mid, gpu_sum) = if let Some(c) = &vk {
            (
                Some(c.create_device_buffer((n_embd * 3 * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?),
                Some(c.create_device_buffer((max_dim * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?),
                Some(c.create_device_buffer((2 * n_ff_exp * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?),
                Some(c.create_device_buffer((n_ff_exp * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?),
                Some(c.create_device_buffer((n_embd * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?),
            )
        } else {
            (None, None, None, None, None)
        };

        let gpu_pipe = if let Some(c) = &vk {
            if cpu_moe_count == 0 && output_w.gpu.is_some() {
                let mut g_phys = Vec::with_capacity(n_phys);
                for p in &phys {
                    g_phys.push(InklingGpuPhys {
                        attn_q_clip: upload_f32(c, &p.attn_q_clip)?,
                        attn_rel_bias: upload_f32(c, &p.attn_rel_bias)?,
                        attn_rel_clip: upload_f32(c, &p.attn_rel_clip)?,
                        attn_k_conv: upload_f32(c, &p.attn_k_conv)?,
                        attn_v_conv: upload_f32(c, &p.attn_v_conv)?,
                        attn_post_conv: upload_f32(c, &p.attn_post_conv)?,
                        ffn_post_conv: upload_f32(c, &p.ffn_post_conv)?,
                        attn_norm: p.attn_norm.as_ref().map(|v| upload_f32(c, v)).transpose()?,
                        attn_q_inp_norm: p.attn_q_inp_norm.as_ref().map(|v| upload_f32(c, v)).transpose()?,
                        attn_k_inp_norm: p.attn_k_inp_norm.as_ref().map(|v| upload_f32(c, v)).transpose()?,
                        attn_v_inp_norm: p.attn_v_inp_norm.as_ref().map(|v| upload_f32(c, v)).transpose()?,
                        attn_res_query: p.attn_res_query.as_ref().map(|v| upload_f32(c, v)).transpose()?,
                        ffn_res_query: upload_f32(c, &p.ffn_res_query)?,
                        ffn_norm: upload_f32(c, &p.ffn_norm)?,
                        exp_probs_b: p.exp_probs_b.as_ref().map(|v| upload_f32(c, v)).transpose()?,
                        gu_scale: upload_f32(c, &p.exps.gu_scale)?,
                        dn_scale: upload_f32(c, &p.exps.dn_scale)?,
                    });
                }
                let mut g_loops = Vec::with_capacity(core_end - core_start);
                for (idx_l, ip) in (core_start..core_end).enumerate() {
                    let mq = upload_f32(c, &loop_layers[idx_l].msg_query)?;
                    let names = ["q", "k", "v", "ffn"];
                    let mut dn = Vec::with_capacity(4);
                    let mut gn = Vec::with_capacity(4);
                    let mut fg = Vec::with_capacity(4);
                    for nm in names {
                        dn.push(upload_mat_gtt_or_vram(&g, c, &format!("blk.{ip}.loop_{nm}_ctrl_down.weight"))?);
                        gn.push(upload_mat_gtt_or_vram(&g, c, &format!("blk.{ip}.loop_{nm}_ctrl_gain.weight"))?);
                        fg.push(upload_mat_gtt_or_vram(&g, c, &format!("blk.{ip}.loop_{nm}_ctrl_forget.weight"))?);
                    }
                    g_loops.push(InklingGpuLoop {
                        msg_query: mq,
                        ctrl_down: dn.try_into().ok().unwrap(),
                        ctrl_gain: gn.try_into().ok().unwrap(),
                        ctrl_forget: fg.try_into().ok().unwrap(),
                    });
                }
                let kv_dim = phys[0].n_head_kv * head_dim;
                let mut slot_kv_offsets = Vec::with_capacity(n_slot);
                let mut slot_caps = Vec::with_capacity(n_slot);
                let mut total_kv_u32 = 0usize;
                for il in 0..n_slot {
                    let ip = if il < n_phys { il } else { core_start + (il - n_phys) % (core_end - core_start) };
                    let cap = if phys[ip].is_swa { max_ctx.min(n_swa + 16) } else { max_ctx };
                    slot_kv_offsets.push(total_kv_u32 as u32);
                    slot_caps.push(cap as u32);
                    total_kv_u32 += cap * (kv_dim / 2);
                }
                let mk = |n: usize| c.create_device_buffer((n * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()));
                let k_hist = mk(n_slot * kv_dim * (n_conv - 1))?;
                let v_hist = mk(n_slot * kv_dim * (n_conv - 1))?;
                let attn_post_hist = mk(n_slot * n_embd * (n_conv - 1))?;
                let ffn_post_hist = mk(n_slot * n_embd * (n_conv - 1))?;
                c.zero_buffer(&k_hist).map_err(|e| EngineError::Backend(e.to_string()))?;
                c.zero_buffer(&v_hist).map_err(|e| EngineError::Backend(e.to_string()))?;
                c.zero_buffer(&attn_post_hist).map_err(|e| EngineError::Backend(e.to_string()))?;
                c.zero_buffer(&ffn_post_hist).map_err(|e| EngineError::Backend(e.to_string()))?;
                let rank = g_loops.first().map(|l| l.ctrl_down[0].3 as usize).unwrap_or(64);
                let n_head_max = phys[0].n_head;
                Some(InklingGpuState {
                    phys: g_phys,
                    loops: g_loops,
                    slot_kv_offsets,
                    slot_caps,
                    ones: upload_f32(c, &vec![1.0f32; n_embd])?,
                    tok_norm: upload_f32(c, &tok_norm)?,
                    output_norm: upload_f32(c, &output_norm)?,
                    output_res_query: upload_f32(c, &output_res_query)?,
                    dummy_bias: upload_f32(c, &vec![0.0f32; n_expert])?,
                    dummy_dyn: upload_f32(c, &[0.0f32; 4])?,
                    cur: mk(n_embd)?,
                    inp_sa: mk(n_embd)?,
                    prelude_cur: mk(n_embd)?,
                    bank_out: mk(n_embd)?,
                    ffn_inp: mk(n_embd)?,
                    blocks_norm: mk(16 * n_embd)?,
                    bank_norm: mk(16 * n_embd)?,
                    inputs_buf: mk(3 * n_embd)?,
                    in_norm_buf: mk(3 * n_embd)?,
                    msgs_buf: mk(4 * n_embd)?,
                    ctrl_cat: mk(6 * n_embd)?,
                    ctrl_gate: mk(3 * rank)?,
                    ctrl_gf: mk(6 * n_embd)?,
                    ctrl_rank: rank as u32,
                    proj_buf: mk(n_embd * 8)?,
                    k_conv_buf: mk(kv_dim)?,
                    v_conv_buf: mk(kv_dim)?,
                    attn_cat: mk(n_head_max * head_dim)?,
                    o_proj: mk(n_embd)?,
                    x_moe: mk(n_embd)?,
                    router_logits: mk(n_expert)?,
                    top_ids: mk(n_expert_used)?,
                    top_w: mk(n_expert_used)?,
                    moe_gu: mk(n_expert_used * 2 * n_ff_exp)?,
                    moe_mid: mk(n_expert_used * n_ff_exp)?,
                    moe_tmp: mk(n_expert_used * n_embd)?,
                    moe_out: mk(n_embd)?,
                    hidden_buf: mk(n_embd)?,
                    logits_buf: mk(n_vocab_padded)?,
                    k_cache: mk(total_kv_u32)?,
                    v_cache: mk(total_kv_u32)?,
                    k_hist,
                    v_hist,
                    attn_post_hist,
                    ffn_post_hist,
                    kv_dim,
                })
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            gguf: g, vk, n_phys, core_start, core_end, n_passes,
            n_embd, n_ff_exp, head_dim, n_expert, n_expert_used, n_rel, n_swa, n_conv, block_size,
            n_vocab_unpadded, eps, qk_clip, rel_clip, log_scale_floor, log_scale_alpha,
            exp_w_scale, situ_beta, situ_lin_beta, softcap, max_ctx,
            tok_embd_name: "token_embd.weight".into(), tok_embd_ty,
            tok_norm, output_norm, output_res_query, output_w, phys, loop_layers, slots,
            gpu_x, gpu_y, gpu_gu, gpu_mid, gpu_sum, gpu_moe_count, cpu_moe_count, gpu_pipe,
            hidden: vec![0f32; n_embd],
            logits: vec![0f32; n_vocab_unpadded],
        })
    }

    fn matvec(&self, m: &Mat2D, x: &[f32]) -> Result<Vec<f32>, EngineError> {
        let mut y = vec![0f32; m.m];
        if let (Some(c), Some((buf, wt)), Some(bx), Some(by)) = (&self.vk, &m.gpu, &self.gpu_x, &self.gpu_y) {
            c.upload(bx, 0, bytemuck::cast_slice(&x[..m.k])).map_err(|e| EngineError::Backend(e.to_string()))?;
            let mut rec = c.begin();
            rec.qmv(*wt, buf, 0, bx, 0, by, 0, m.k as u32, m.m as u32);
            rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;
            c.download(by, 0, bytemuck::cast_slice_mut(&mut y)).map_err(|e| EngineError::Backend(e.to_string()))?;
        } else {
            let t = self.gguf.tensor(&m.name).unwrap();
            ocak_cpu::gemv(m.ty, self.gguf.tensor_data(t), &x[..m.k], &mut y, m.k, m.m);
        }
        Ok(y)
    }

    fn control(&self, cur: &[f32], msg: &[f32], lp: &LoopLayer, ctrl: usize) -> Result<Vec<f32>, EngineError> {
        let mut cat = rms_norm_unweighted(cur, 1e-6);
        cat.extend_from_slice(&rms_norm_unweighted(msg, 1e-6));
        let (gain, forget) = if let (Some(c), Some(bx), Some(by), Some((bd, wtd)), Some((bg, wtg)), Some((bf, wtf))) = (
            &self.vk, &self.gpu_x, &self.gpu_y,
            &lp.ctrl_down[ctrl].gpu, &lp.ctrl_gain[ctrl].gpu, &lp.ctrl_forget[ctrl].gpu,
        ) {
            let rank = lp.ctrl_down[ctrl].m as u32;
            let ne = self.n_embd as u32;
            c.upload(bx, 0, bytemuck::cast_slice(&cat)).map_err(|e| EngineError::Backend(e.to_string()))?;
            let mut rec = c.begin();
            // down -> by[0..rank], gain -> by[rank..rank+ne], forget -> by[rank+ne..rank+2*ne]
            rec.qmv(*wtd, bd, 0, bx, 0, by, 0, 2 * ne, rank);
            rec.qmv(*wtg, bg, 0, by, 0, by, rank as u64, rank, ne);
            rec.qmv(*wtf, bf, 0, by, 0, by, (rank + ne) as u64, rank, ne);
            rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;
            let mut gf = vec![0f32; 2 * self.n_embd];
            c.download(by, (rank as u64) * 4, bytemuck::cast_slice_mut(&mut gf)).map_err(|e| EngineError::Backend(e.to_string()))?;
            (gf[..self.n_embd].to_vec(), gf[self.n_embd..].to_vec())
        } else {
            let gate = self.matvec(&lp.ctrl_down[ctrl], &cat)?;
            (self.matvec(&lp.ctrl_gain[ctrl], &gate)?, self.matvec(&lp.ctrl_forget[ctrl], &gate)?)
        };
        let mut out = vec![0f32; self.n_embd];
        for i in 0..self.n_embd {
            let g = gain[i].tanh();
            let f = forget[i].tanh();
            out[i] = cur[i] - f * cur[i] + g * msg[i];
        }
        Ok(out)
    }

    fn run_layer(
        &mut self,
        il: usize,
        ip: usize,
        pos: usize,
        tau: f32,
        cur_in: Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        messages: Option<&[Vec<f32>]>,
        loop_idx: Option<usize>,
    ) -> Result<Vec<f32>, EngineError> {
        let is_swa = self.phys[ip].is_swa;
        let n_head = self.phys[ip].n_head;
        let n_head_kv = self.phys[ip].n_head_kv;
        let n_extent = self.phys[ip].n_extent;
        let hd = self.head_dim;
        let kv_dim = n_head_kv * hd;
        let inp_sa = cur_in.clone();

        // 1. Attention residual mix
        let mut inputs = [cur_in.clone(), cur_in.clone(), cur_in];
        if let Some(ref rq) = self.phys[ip].attn_res_query {
            let n_q = if is_swa { 1 } else { 3 };
            let mixed = mix_depth(blocks, &inp_sa, rq, self.n_embd, n_q);
            if is_swa {
                inputs = [mixed[0].clone(), mixed[0].clone(), mixed[0].clone()];
            } else {
                inputs = [mixed[0].clone(), mixed[1].clone(), mixed[2].clone()];
            }
        }

        // 2. Cross-pass loop controllers for Q, K, V
        if let (Some(msgs), Some(li)) = (messages, loop_idx) {
            for j in 0..3 {
                inputs[j] = self.control(&inputs[j], &msgs[j], &self.loop_layers[li], j)?;
            }
        }

        // 3. Input RMSNorm
        if is_swa {
            let nw = self.phys[ip].attn_norm.as_ref().unwrap();
            if messages.is_some() {
                for j in 0..3 { inputs[j] = rms_norm_weighted(&inputs[j], nw, self.eps); }
            } else {
                inputs[0] = rms_norm_weighted(&inputs[0], nw, self.eps);
                inputs[1] = inputs[0].clone();
                inputs[2] = inputs[0].clone();
            }
        } else {
            inputs[0] = rms_norm_weighted(&inputs[0], self.phys[ip].attn_q_inp_norm.as_ref().unwrap(), self.eps);
            inputs[1] = rms_norm_weighted(&inputs[1], self.phys[ip].attn_k_inp_norm.as_ref().unwrap(), self.eps);
            inputs[2] = rms_norm_weighted(&inputs[2], self.phys[ip].attn_v_inp_norm.as_ref().unwrap(), self.eps);
        }

        // 4. Fused Q, K, V, R, Sink, Gate (+ dynamic conv) projections in ONE Vulkan submission
        let (mut q, k_raw, v_raw, mut r, sinks, gate, k_dyn, v_dyn) = if let (Some(c), Some(bx), Some(by)) = (&self.vk, &self.gpu_x, &self.gpu_y) {
            let p = &self.phys[ip];
            if let (Some((bq, wtq)), Some((bk, wtk)), Some((bv, wtv)), Some((br, wtr)), Some((bs, wts)), Some((bg, wtg))) = (
                &p.wq.gpu, &p.wk.gpu, &p.wv.gpu, &p.attn_rel.gpu, &p.attn_sink_proj.gpu, &p.wqkv_gate.gpu,
            ) {
                let mut x_pack = Vec::with_capacity(3 * self.n_embd);
                x_pack.extend_from_slice(&inputs[0]);
                x_pack.extend_from_slice(&inputs[1]);
                x_pack.extend_from_slice(&inputs[2]);
                c.upload(bx, 0, bytemuck::cast_slice(&x_pack)).map_err(|e| EngineError::Backend(e.to_string()))?;
                let o_q = 0u64;
                let o_k = o_q + p.wq.m as u64;
                let o_v = o_k + p.wk.m as u64;
                let o_r = o_v + p.wv.m as u64;
                let o_s = o_r + p.attn_rel.m as u64;
                let o_g = o_s + p.attn_sink_proj.m as u64;
                let o_kd = o_g + p.wqkv_gate.m as u64;
                let mut total = o_kd;
                let mut rec = c.begin();
                let ne = self.n_embd as u32;
                rec.qmv(*wtq, bq, 0, bx, 0, by, o_q, ne, p.wq.m as u32);
                rec.qmv(*wtk, bk, 0, bx, ne as u64, by, o_k, ne, p.wk.m as u32);
                rec.qmv(*wtv, bv, 0, bx, (2 * ne) as u64, by, o_v, ne, p.wv.m as u32);
                rec.qmv(*wtr, br, 0, bx, 0, by, o_r, ne, p.attn_rel.m as u32);
                rec.qmv(*wts, bs, 0, bx, 0, by, o_s, ne, p.attn_sink_proj.m as u32);
                rec.qmv(*wtg, bg, 0, bx, 0, by, o_g, ne, p.wqkv_gate.m as u32);
                let mut o_vd = o_kd;
                if let (Some(kp), Some(vp)) = (&p.attn_k_conv_proj, &p.attn_v_conv_proj) {
                    if let (Some((bkd, wtkd)), Some((bvd, wtvd))) = (&kp.gpu, &vp.gpu) {
                        o_vd = o_kd + kp.m as u64;
                        total = o_vd + vp.m as u64;
                        rec.qmv(*wtkd, bkd, 0, bx, ne as u64, by, o_kd, ne, kp.m as u32);
                        rec.qmv(*wtvd, bvd, 0, bx, (2 * ne) as u64, by, o_vd, ne, vp.m as u32);
                    }
                }
                rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;
                let mut y_pack = vec![0f32; total as usize];
                c.download(by, 0, bytemuck::cast_slice_mut(&mut y_pack)).map_err(|e| EngineError::Backend(e.to_string()))?;
                let q = y_pack[o_q as usize..o_k as usize].to_vec();
                let k_raw = y_pack[o_k as usize..o_v as usize].to_vec();
                let v_raw = y_pack[o_v as usize..o_r as usize].to_vec();
                let r = y_pack[o_r as usize..o_s as usize].to_vec();
                let sinks = y_pack[o_s as usize..o_g as usize].to_vec();
                let gate = y_pack[o_g as usize..o_kd as usize].to_vec();
                let (k_dyn, v_dyn) = if total > o_kd {
                    (Some(y_pack[o_kd as usize..o_vd as usize].to_vec()), Some(y_pack[o_vd as usize..total as usize].to_vec()))
                } else {
                    (None, None)
                };
                (q, k_raw, v_raw, r, sinks, gate, k_dyn, v_dyn)
            } else {
                let q = self.matvec(&self.phys[ip].wq, &inputs[0])?;
                let k_raw = self.matvec(&self.phys[ip].wk, &inputs[1])?;
                let v_raw = self.matvec(&self.phys[ip].wv, &inputs[2])?;
                let k_dyn = if let Some(ref p) = self.phys[ip].attn_k_conv_proj { Some(self.matvec(p, &inputs[1])?) } else { None };
                let v_dyn = if let Some(ref p) = self.phys[ip].attn_v_conv_proj { Some(self.matvec(p, &inputs[2])?) } else { None };
                let r = self.matvec(&self.phys[ip].attn_rel, &inputs[0])?;
                let sinks = self.matvec(&self.phys[ip].attn_sink_proj, &inputs[0])?;
                let gate = self.matvec(&self.phys[ip].wqkv_gate, &inputs[0])?;
                (q, k_raw, v_raw, r, sinks, gate, k_dyn, v_dyn)
            }
        } else {
            let q = self.matvec(&self.phys[ip].wq, &inputs[0])?;
            let k_raw = self.matvec(&self.phys[ip].wk, &inputs[1])?;
            let v_raw = self.matvec(&self.phys[ip].wv, &inputs[2])?;
            let k_dyn = if let Some(ref p) = self.phys[ip].attn_k_conv_proj { Some(self.matvec(p, &inputs[1])?) } else { None };
            let v_dyn = if let Some(ref p) = self.phys[ip].attn_v_conv_proj { Some(self.matvec(p, &inputs[2])?) } else { None };
            let r = self.matvec(&self.phys[ip].attn_rel, &inputs[0])?;
            let sinks = self.matvec(&self.phys[ip].attn_sink_proj, &inputs[0])?;
            let gate = self.matvec(&self.phys[ip].wqkv_gate, &inputs[0])?;
            (q, k_raw, v_raw, r, sinks, gate, k_dyn, v_dyn)
        };
        let mut k = convolve_step(&k_raw, &self.phys[ip].attn_k_conv, k_dyn.as_deref(), &mut self.slots[il].k_hist, self.n_conv);
        let mut v = convolve_step(&v_raw, &self.phys[ip].attn_v_conv, v_dyn.as_deref(), &mut self.slots[il].v_hist, self.n_conv);

        // 5. Clip Q, K, R; RMSNorm V; apply tau on full-attention layers
        clip_heads(&mut q, Some(&self.phys[ip].attn_q_clip), hd, n_head, self.qk_clip);
        clip_heads(&mut k, None, hd, n_head_kv, self.qk_clip);
        for h in 0..n_head_kv {
            let sl = &mut v[h * hd..(h + 1) * hd];
            let ms = sl.iter().map(|x| x * x).sum::<f32>() / hd as f32;
            let inv = 1.0 / (ms + self.eps).sqrt();
            for x in sl.iter_mut() { *x *= inv; }
        }
        clip_heads(&mut r, Some(&self.phys[ip].attn_rel_clip), self.n_rel, n_head, self.rel_clip);
        if !is_swa {
            for x in &mut q { *x *= tau; }
            for x in &mut r { *x *= tau; }
        }

        // 6. Store K, V in slot cache (ring buffer for SWA, linear for Full)
        let cap = self.slots[il].k_cache.len() / kv_dim;
        let slot_pos = if is_swa { pos % cap } else { pos };
        self.slots[il].k_cache[slot_pos * kv_dim..(slot_pos + 1) * kv_dim].copy_from_slice(&k);
        self.slots[il].v_cache[slot_pos * kv_dim..(slot_pos + 1) * kv_dim].copy_from_slice(&v);

        // 7. Attention with relative-position bias + zero sink (parallel across heads)
        let t_start = if is_swa { pos.saturating_sub(self.n_swa - 1) } else { 0 };
        let t_len = pos - t_start + 1;
        let scale = 1.0 / (hd as f32).sqrt();
        let gqa = n_head / n_head_kv;
        let k_cache = &self.slots[il].k_cache;
        let v_cache = &self.slots[il].v_cache;
        let rel_bias = &self.phys[ip].attn_rel_bias;
        let n_rel = self.n_rel;

        let attn_out_heads: Vec<Vec<f32>> = (0..n_head)
            .into_par_iter()
            .map(|h| {
                let qh = &q[h * hd..(h + 1) * hd];
                let rh = &r[h * n_rel..(h + 1) * n_rel];
                let kvh = h / gqa;
                let sink_t = sinks[h];
                let mut scores = vec![0f32; t_len];
                let mut mx = 0.0f32; // zero sink logit = 0.0
                for (idx, t) in (t_start..=pos).enumerate() {
                    let sp = if is_swa { t % cap } else { t };
                    let kh = &k_cache[sp * kv_dim + kvh * hd..sp * kv_dim + (kvh + 1) * hd];
                    let mut dot = 0f32;
                    for d in 0..hd { dot += qh[d] * kh[d]; }
                    let dist = pos - t;
                    let rb = if dist < n_extent {
                        let row = &rel_bias[dist * n_rel..(dist + 1) * n_rel];
                        let mut b = 0f32;
                        for j in 0..n_rel { b += row[j] * rh[j]; }
                        b
                    } else {
                        0.0
                    };
                    let sc = dot * scale + rb - sink_t;
                    scores[idx] = sc;
                    if sc > mx { mx = sc; }
                }
                let mut denom = (-mx).exp(); // zero sink contribution
                for s in &mut scores {
                    *s = (*s - mx).exp();
                    denom += *s;
                }
                let inv = 1.0 / denom;
                let mut oh = vec![0f32; hd];
                for (idx, t) in (t_start..=pos).enumerate() {
                    let w = scores[idx] * inv;
                    let sp = if is_swa { t % cap } else { t };
                    let vh = &v_cache[sp * kv_dim + kvh * hd..sp * kv_dim + (kvh + 1) * hd];
                    for d in 0..hd { oh[d] += w * vh[d]; }
                }
                oh
            })
            .collect();

        let mut attn_cat = vec![0f32; n_head * hd];
        for h in 0..n_head {
            for d in 0..hd {
                let g = 1.0 / (1.0 + (-gate[h * hd + d]).exp());
                attn_cat[h * hd + d] = attn_out_heads[h][d] * g;
            }
        }
        let o_proj = self.matvec(&self.phys[ip].wo, &attn_cat)?;
        let mut cur = convolve_step(&o_proj, &self.phys[ip].attn_post_conv, None, &mut self.slots[il].attn_post_hist, self.n_conv);

        // 8. Block boundary vs residual add
        if ip % self.block_size == 0 {
            blocks.push(inp_sa);
        } else {
            for i in 0..self.n_embd { cur[i] += inp_sa[i]; }
        }
        let ffn_inp = cur.clone();

        // 9. FFN residual mix + loop controller + norm
        cur = mix_depth(blocks, &cur, &self.phys[ip].ffn_res_query, self.n_embd, 1).pop().unwrap();
        if let (Some(msgs), Some(li)) = (messages, loop_idx) {
            cur = self.control(&cur, &msgs[3], &self.loop_layers[li], 3)?;
        }
        let x_moe = rms_norm_weighted(&cur, &self.phys[ip].ffn_norm, self.eps);

        // 10. 128-expert / 8-active MoE (sigmoid router + exp_probs_b + SITU + NVFP4 .scale)
        let r_logits = self.matvec(&self.phys[ip].ffn_gate_inp, &x_moe)?;
        let mut probs = vec![0f32; self.n_expert];
        let mut biased = vec![0f32; self.n_expert];
        for e in 0..self.n_expert {
            let p = 1.0 / (1.0 + (-r_logits[e]).exp());
            probs[e] = p;
            biased[e] = p + self.phys[ip].exp_probs_b.as_ref().map_or(0.0, |b| b[e]);
        }
        let mut idx: Vec<usize> = (0..self.n_expert).collect();
        idx.sort_by(|&a, &b| biased[b].total_cmp(&biased[a]));
        idx.truncate(self.n_expert_used);
        let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>() + 1e-20;
        let top_w: Vec<(usize, f32)> = idx.into_iter().map(|e| (e, probs[e] / wsum * self.exp_w_scale)).collect();

        let exps = &self.phys[ip].exps;
        let n_ff = self.n_ff_exp;
        let beta = self.situ_beta;
        let lb = self.situ_lin_beta;
        let mut moe_out = vec![0f32; self.n_embd];

        if let (Some(c), Some((b_gu, wt_gu)), Some((b_dn, wt_dn)), Some(bx), Some(bgu), Some(bmid), Some(by), Some(bsum)) = (
            &self.vk, &exps.gpu_gu, &exps.gpu_dn, &self.gpu_x, &self.gpu_gu, &self.gpu_mid, &self.gpu_y, &self.gpu_sum,
        ) {
            // Single-command-buffer fused GPU MoE (all 8 active experts: NVFP4 gate_up -> SITU -> NVFP4 down -> add_scaled)
            c.upload(bx, 0, bytemuck::cast_slice(&x_moe)).map_err(|e| EngineError::Backend(e.to_string()))?;
            let mut rec = c.begin();
            rec.zero_buf(bsum);
            for &(e, ew) in &top_w {
                rec.qmv(*wt_gu, b_gu, (e * exps.gu_stride) as u64, bx, 0, bgu, 0, self.n_embd as u32, (2 * n_ff) as u32);
                rec.situ(bgu, 0, bmid, 0, n_ff as u32, exps.gu_scale[e], beta, lb);
                rec.qmv(*wt_dn, b_dn, (e * exps.dn_stride) as u64, bmid, 0, by, 0, n_ff as u32, self.n_embd as u32);
                rec.add_scaled(bsum, 0, by, 0, self.n_embd as u32, ew * exps.dn_scale[e]);
            }
            rec.finish_and_wait().map_err(|er| EngineError::Backend(er.to_string()))?;
            c.download(bsum, 0, bytemuck::cast_slice_mut(&mut moe_out)).map_err(|er| EngineError::Backend(er.to_string()))?;
        } else {
            let t_gu = self.gguf.tensor(&exps.gu_name).unwrap();
            let t_dn = self.gguf.tensor(&exps.dn_name).unwrap();
            let d_gu = self.gguf.tensor_data(t_gu);
            let d_dn = self.gguf.tensor_data(t_dn);
            let mut gu_buf = vec![0f32; 2 * n_ff];
            let mut mid = vec![0f32; n_ff];
            let mut dn_buf = vec![0f32; self.n_embd];
            for &(e, ew) in &top_w {
                ocak_cpu::gemv(exps.gu_ty, &d_gu[e * exps.gu_stride..(e + 1) * exps.gu_stride], &x_moe, &mut gu_buf, self.n_embd, 2 * n_ff);
                let gs = exps.gu_scale[e];
                for j in 0..n_ff {
                    let g = gu_buf[j] * gs;
                    let u = gu_buf[n_ff + j] * gs;
                    let act = beta * (g / beta).tanh() * (1.0 / (1.0 + (-g).exp()));
                    let lin = if lb > 0.0 { lb * (u / lb).tanh() } else { u };
                    mid[j] = act * lin;
                }
                ocak_cpu::gemv(exps.dn_ty, &d_dn[e * exps.dn_stride..(e + 1) * exps.dn_stride], &mid, &mut dn_buf, n_ff, self.n_embd);
                let ds = ew * exps.dn_scale[e];
                for j in 0..self.n_embd { moe_out[j] += ds * dn_buf[j]; }
            }
        }

        // 11. FFN post ShortConv + residual add
        let ffn_out = convolve_step(&moe_out, &self.phys[ip].ffn_post_conv, None, &mut self.slots[il].ffn_post_hist, self.n_conv);
        let mut out = vec![0f32; self.n_embd];
        for i in 0..self.n_embd { out[i] = ffn_inp[i] + ffn_out[i]; }
        Ok(out)
    }
    fn record_layer_gpu(
        &self,
        rec: &mut ocak_vk::Recorder<'_>,
        gp: &InklingGpuState,
        il: usize,
        ip: usize,
        pos: usize,
        tau: f32,
        blk_count: &mut u32,
        loop_idx: Option<usize>,
    ) {
        let p = &self.phys[ip];
        let gph = &gp.phys[ip];
        let ne = self.n_embd as u32;
        let hd = self.head_dim as u32;
        let kv_dim = gp.kv_dim as u32;
        let nh_hist = (self.n_conv - 1) as u32;

        // Save inp_sa + 1. Attention residual mix -> inputs_buf[3 * ne] (parallel!)
        rec.copy(&gp.cur, 0, &gp.inp_sa, 0, ne);
        if let Some(ref rq) = gph.attn_res_query {
            let n_q = if p.is_swa { 1u32 } else { 3u32 };
            rec.inkling_mix(&gp.blocks_norm, &gp.cur, rq, &gp.inputs_buf, ne, 0, *blk_count, n_q);
            rec.barrier();
            if p.is_swa {
                rec.copy(&gp.inputs_buf, 0, &gp.inputs_buf, ne as u64, ne);
                rec.copy(&gp.inputs_buf, 0, &gp.inputs_buf, (2 * ne) as u64, ne);
                rec.barrier();
            }
        } else {
            rec.copy(&gp.cur, 0, &gp.inputs_buf, 0, ne);
            rec.copy(&gp.cur, 0, &gp.inputs_buf, ne as u64, ne);
            rec.copy(&gp.cur, 0, &gp.inputs_buf, (2 * ne) as u64, ne);
            rec.barrier();
        }

        // 2. Cross-pass loop controllers for Q, K, V (all 3 in parallel!)
        if let Some(li) = loop_idx {
            let gl = &gp.loops[li];
            let rk = gp.ctrl_rank;
            for j in 0..3u32 {
                let off = (j * ne) as u64;
                let coff = (j * 2 * ne) as u64;
                rec.rms_norm(&gp.inputs_buf, off, &gp.ones, 0, &gp.ctrl_cat, coff, ne, 1e-6);
                rec.rms_norm(&gp.msgs_buf, off, &gp.ones, 0, &gp.ctrl_cat, coff + ne as u64, ne, 1e-6);
            }
            rec.barrier();
            for j in 0..3u32 {
                let (ref bd, wtd, kd, md) = gl.ctrl_down[j as usize];
                rec.qmv(wtd, bd, 0, &gp.ctrl_cat, (j * 2 * ne) as u64, &gp.ctrl_gate, (j * rk) as u64, kd, md);
            }
            rec.barrier();
            for j in 0..3u32 {
                let (ref bg, wtg, kg, mg) = gl.ctrl_gain[j as usize];
                let (ref bf, wtf, kf, mf) = gl.ctrl_forget[j as usize];
                let goff = (j * rk) as u64;
                let foff = (j * 2 * ne) as u64;
                rec.qmv(wtg, bg, 0, &gp.ctrl_gate, goff, &gp.ctrl_gf, foff, kg, mg);
                rec.qmv(wtf, bf, 0, &gp.ctrl_gate, goff, &gp.ctrl_gf, foff + ne as u64, kf, mf);
            }
            rec.barrier();
            for j in 0..3u32 {
                let off = (j * ne) as u64;
                rec.inkling_ctrl(&gp.inputs_buf, off, &gp.msgs_buf, off, &gp.ctrl_gf, (j * 2 * ne) as u64, ne);
            }
            rec.barrier();
        }

        // 3. Input RMSNorm -> in_norm_buf[3 * ne] (all 3 in parallel!)
        if p.is_swa {
            let nw = gph.attn_norm.as_ref().unwrap();
            for j in 0..3u32 {
                let off = (j * ne) as u64;
                rec.rms_norm(&gp.inputs_buf, off, nw, 0, &gp.in_norm_buf, off, ne, self.eps);
            }
        } else {
            rec.rms_norm(&gp.inputs_buf, 0, gph.attn_q_inp_norm.as_ref().unwrap(), 0, &gp.in_norm_buf, 0, ne, self.eps);
            rec.rms_norm(&gp.inputs_buf, ne as u64, gph.attn_k_inp_norm.as_ref().unwrap(), 0, &gp.in_norm_buf, ne as u64, ne, self.eps);
            rec.rms_norm(&gp.inputs_buf, (2 * ne) as u64, gph.attn_v_inp_norm.as_ref().unwrap(), 0, &gp.in_norm_buf, (2 * ne) as u64, ne, self.eps);
        }
        rec.barrier();

        // 4. Q, K, V, R, Sink, Gate (+ k_dyn, v_dyn) -> proj_buf (all 6-8 in parallel!)
        let o_q = 0u64;
        let o_k = o_q + p.wq.m as u64;
        let o_v = o_k + p.wk.m as u64;
        let o_r = o_v + p.wv.m as u64;
        let o_s = o_r + p.attn_rel.m as u64;
        let o_g = o_s + p.attn_sink_proj.m as u64;
        let o_kd = o_g + p.wqkv_gate.m as u64;
        let (bq, wtq) = p.wq.gpu.as_ref().unwrap();
        let (bk, wtk) = p.wk.gpu.as_ref().unwrap();
        let (bv, wtv) = p.wv.gpu.as_ref().unwrap();
        let (br, wtr) = p.attn_rel.gpu.as_ref().unwrap();
        let (bs, wts) = p.attn_sink_proj.gpu.as_ref().unwrap();
        let (bg, wtg) = p.wqkv_gate.gpu.as_ref().unwrap();
        rec.qmv(*wtq, bq, 0, &gp.in_norm_buf, 0, &gp.proj_buf, o_q, ne, p.wq.m as u32);
        rec.qmv(*wtk, bk, 0, &gp.in_norm_buf, ne as u64, &gp.proj_buf, o_k, ne, p.wk.m as u32);
        rec.qmv(*wtv, bv, 0, &gp.in_norm_buf, (2 * ne) as u64, &gp.proj_buf, o_v, ne, p.wv.m as u32);
        rec.qmv(*wtr, br, 0, &gp.in_norm_buf, 0, &gp.proj_buf, o_r, ne, p.attn_rel.m as u32);
        rec.qmv(*wts, bs, 0, &gp.in_norm_buf, 0, &gp.proj_buf, o_s, ne, p.attn_sink_proj.m as u32);
        rec.qmv(*wtg, bg, 0, &gp.in_norm_buf, 0, &gp.proj_buf, o_g, ne, p.wqkv_gate.m as u32);

        let mut has_dyn = false;
        let mut o_vd = o_kd;
        if let (Some(kp), Some(vp)) = (&p.attn_k_conv_proj, &p.attn_v_conv_proj) {
            let (bkd, wtkd) = kp.gpu.as_ref().unwrap();
            let (bvd, wtvd) = vp.gpu.as_ref().unwrap();
            o_vd = o_kd + kp.m as u64;
            rec.qmv(*wtkd, bkd, 0, &gp.in_norm_buf, ne as u64, &gp.proj_buf, o_kd, ne, kp.m as u32);
            rec.qmv(*wtvd, bvd, 0, &gp.in_norm_buf, (2 * ne) as u64, &gp.proj_buf, o_vd, ne, vp.m as u32);
            has_dyn = true;
        }
        rec.barrier();

        // 5. ShortConv on K and V (both in parallel!)
        let kv_hist_off = (il as u32 * kv_dim * nh_hist) as u64;
        let (dy_buf, kd_off, vd_off) = if has_dyn { (&gp.proj_buf, o_kd, o_vd) } else { (&gp.dummy_dyn, 0, 0) };
        rec.inkling_conv(&gp.proj_buf, o_k, &gph.attn_k_conv, dy_buf, kd_off, &gp.k_hist, kv_hist_off, &gp.k_conv_buf, 0, kv_dim, self.n_conv as u32, has_dyn, hd);
        rec.inkling_conv(&gp.proj_buf, o_v, &gph.attn_v_conv, dy_buf, vd_off, &gp.v_hist, kv_hist_off, &gp.v_conv_buf, 0, kv_dim, self.n_conv as u32, has_dyn, hd);
        rec.barrier();

        // 6. Attention (clip Q/K/R, norm V, KV store, relative bias + zero sink + sigmoid gate) -> attn_cat
        rec.inkling_attn(
            &gp.proj_buf, &gp.k_conv_buf, &gp.v_conv_buf, &gp.proj_buf, &gp.proj_buf, &gp.proj_buf,
            &gph.attn_q_clip, &gph.attn_rel_clip, &gph.attn_rel_bias, &gp.k_cache, &gp.v_cache, &gp.attn_cat,
            o_q as u32, o_r as u32, o_s as u32, o_g as u32,
            p.n_head as u32, p.n_head_kv as u32, hd, self.n_rel as u32, p.n_extent as u32, self.n_swa as u32, p.is_swa,
            gp.slot_caps[il], pos as u32, gp.slot_kv_offsets[il], self.qk_clip, self.rel_clip, tau, self.eps,
        );
        rec.barrier();

        // 7. wo + attn_post_conv -> cur
        let (bwo, wtwo) = p.wo.gpu.as_ref().unwrap();
        rec.qmv(*wtwo, bwo, 0, &gp.attn_cat, 0, &gp.o_proj, 0, p.wo.k as u32, ne);
        rec.barrier();
        let post_hist_off = (il as u32 * ne * nh_hist) as u64;
        rec.inkling_conv(&gp.o_proj, 0, &gph.attn_post_conv, &gp.dummy_dyn, 0, &gp.attn_post_hist, post_hist_off, &gp.cur, 0, ne, self.n_conv as u32, false, 1);

        // 8. Block boundary vs residual add
        if ip % self.block_size == 0 {
            rec.rms_norm(&gp.inp_sa, 0, &gp.ones, 0, &gp.blocks_norm, (*blk_count * ne) as u64, ne, 1e-6);
            *blk_count += 1;
            rec.barrier();
        } else {
            rec.barrier();
            rec.add(&gp.cur, 0, &gp.inp_sa, 0, ne);
            rec.barrier();
        }

        // 9. Save ffn_inp + FFN mix (parallel into o_proj!) + loop controller 3 + ffn_norm -> x_moe
        rec.copy(&gp.cur, 0, &gp.ffn_inp, 0, ne);
        rec.inkling_mix(&gp.blocks_norm, &gp.cur, &gph.ffn_res_query, &gp.o_proj, ne, 0, *blk_count, 1);
        rec.barrier();
        if let Some(li) = loop_idx {
            let gl = &gp.loops[li];
            let off = (3 * ne) as u64;
            rec.rms_norm(&gp.o_proj, 0, &gp.ones, 0, &gp.ctrl_cat, 0, ne, 1e-6);
            rec.rms_norm(&gp.msgs_buf, off, &gp.ones, 0, &gp.ctrl_cat, ne as u64, ne, 1e-6);
            rec.barrier();
            let (ref bd, wtd, kd, md) = gl.ctrl_down[3];
            let (ref bg, wtg, kg, mg) = gl.ctrl_gain[3];
            let (ref bf, wtf, kf, mf) = gl.ctrl_forget[3];
            rec.qmv(wtd, bd, 0, &gp.ctrl_cat, 0, &gp.ctrl_gate, 0, kd, md);
            rec.barrier();
            rec.qmv(wtg, bg, 0, &gp.ctrl_gate, 0, &gp.ctrl_gf, 0, kg, mg);
            rec.qmv(wtf, bf, 0, &gp.ctrl_gate, 0, &gp.ctrl_gf, ne as u64, kf, mf);
            rec.barrier();
            rec.inkling_ctrl(&gp.o_proj, 0, &gp.msgs_buf, off, &gp.ctrl_gf, 0, ne);
            rec.barrier();
        }
        rec.rms_norm(&gp.o_proj, 0, &gph.ffn_norm, 0, &gp.x_moe, 0, ne, self.eps);
        rec.barrier();

        // 10. 128-expert / 8-active batched 2D MoE (4 dispatches total!)
        let (bgi, wtgi) = p.ffn_gate_inp.gpu.as_ref().unwrap();
        rec.qmv(*wtgi, bgi, 0, &gp.x_moe, 0, &gp.router_logits, 0, ne, self.n_expert as u32);
        rec.barrier();
        let (bias_b, has_b) = match &gph.exp_probs_b {
            Some(b) => (b, true),
            None => (&gp.dummy_bias, false),
        };
        let n_used = self.n_expert_used as u32;
        rec.moe_topk(&gp.router_logits, bias_b, &gp.top_ids, &gp.top_w, self.n_expert as u32, n_used, has_b, true, self.exp_w_scale);
        rec.barrier();
        let (b_gu, wt_gu) = p.exps.gpu_gu.as_ref().unwrap();
        let (b_dn, wt_dn) = p.exps.gpu_dn.as_ref().unwrap();
        rec.qmv_id_batch(*wt_gu, b_gu, &gp.top_ids, 0, n_used, &gp.x_moe, 0, 0, &gp.moe_gu, 0, ne, (2 * self.n_ff_exp) as u32, p.exps.gu_stride as u64);
        rec.barrier();
        rec.situ_id_batch(&gp.moe_gu, &gp.moe_mid, &gp.top_ids, &gph.gu_scale, self.n_ff_exp as u32, 0, n_used, self.situ_beta, self.situ_lin_beta);
        rec.barrier();
        rec.qmv_id_batch(*wt_dn, b_dn, &gp.top_ids, 0, n_used, &gp.moe_mid, 0, self.n_ff_exp as u32, &gp.moe_tmp, 0, self.n_ff_exp as u32, ne, p.exps.dn_stride as u64);
        rec.barrier();
        rec.add_scaled_exp_id_batch(&gp.moe_out, &gp.moe_tmp, &gp.top_w, &gp.top_ids, &gph.dn_scale, ne, 0, n_used, true);
        rec.barrier();

        // 11. FFN post ShortConv + residual add -> cur
        rec.inkling_conv(&gp.moe_out, 0, &gph.ffn_post_conv, &gp.dummy_dyn, 0, &gp.ffn_post_hist, post_hist_off, &gp.cur, 0, ne, self.n_conv as u32, false, 1);
        rec.barrier();
        rec.add(&gp.cur, 0, &gp.ffn_inp, 0, ne);
        rec.barrier();
    }
}

impl Executor for InklingExec {
    fn name(&self) -> String {
        if let Some(c) = &self.vk {
            format!("vulkan-inkling-1sub ({}, gpu_moe={}, cpu_moe={})", c.device_name(), self.gpu_moe_count, self.cpu_moe_count)
        } else {
            format!("cpu-inkling (cpu_moe={})", self.cpu_moe_count)
        }
    }

    fn reset_recurrent(&mut self) -> Result<(), EngineError> {
        for s in &mut self.slots {
            s.k_hist.fill(0.0);
            s.v_hist.fill(0.0);
            s.attn_post_hist.fill(0.0);
            s.ffn_post_hist.fill(0.0);
        }
        if let (Some(c), Some(gp)) = (&self.vk, &self.gpu_pipe) {
            c.zero_buffer(&gp.k_hist).map_err(|e| EngineError::Backend(e.to_string()))?;
            c.zero_buffer(&gp.v_hist).map_err(|e| EngineError::Backend(e.to_string()))?;
            c.zero_buffer(&gp.attn_post_hist).map_err(|e| EngineError::Backend(e.to_string()))?;
            c.zero_buffer(&gp.ffn_post_hist).map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        Ok(())
    }

    fn forward(&mut self, token: u32, pos: usize, _page_table: &[u32], out: Out) -> Result<(), EngineError> {
        if pos >= self.max_ctx {
            return Err(EngineError::Context(format!("pos {pos} >= max_ctx {}", self.max_ctx)));
        }
        let tau = 1.0 + self.log_scale_alpha * (((pos as f32 + 1.0) / self.log_scale_floor).max(1.0)).ln();

        // Token embedding
        let te = self.gguf.tensor(&self.tok_embd_name).unwrap();
        let rb = self.tok_embd_ty.row_bytes(self.n_embd).unwrap();
        let row = &self.gguf.tensor_data(te)[token as usize * rb..(token as usize + 1) * rb];
        let mut emb = vec![0f32; self.n_embd];
        dequantize_row(self.tok_embd_ty, row, &mut emb, self.n_embd);

        // Fast path: ENTIRE 48-layer looped forward pass in ONE SINGLE Vulkan vkQueueSubmit!
        if let (Some(c), Some(gp)) = (&self.vk, &self.gpu_pipe) {
            let ne = self.n_embd as u32;
            c.upload(&gp.cur, 0, bytemuck::cast_slice(&emb)).map_err(|e| EngineError::Backend(e.to_string()))?;
            let mut rec = c.begin();
            rec.set_defer_barriers(true);
            rec.rms_norm(&gp.cur, 0, &gp.tok_norm, 0, &gp.cur, 0, ne, self.eps);
            rec.barrier();
            let mut blk_count = 0u32;

            // Prelude: 0..core_start
            for il in 0..self.core_start {
                self.record_layer_gpu(&mut rec, gp, il, il, pos, tau, &mut blk_count, None);
            }
            // Save prelude state
            rec.copy(&gp.cur, 0, &gp.prelude_cur, 0, ne);
            rec.barrier();
            let prelude_blk_count = blk_count;

            // Looped core: n_passes over core_start..core_end
            for pass in 0..self.n_passes {
                let mut bank_count = 0u32;
                if pass > 0 {
                    let bank_start = prelude_blk_count + 1;
                    bank_count = blk_count.saturating_sub(bank_start);
                    if bank_count > 0 {
                        rec.copy(&gp.blocks_norm, (bank_start * ne) as u64, &gp.bank_norm, 0, bank_count * ne);
                    }
                    rec.copy(&gp.cur, 0, &gp.bank_out, 0, ne);
                    rec.barrier();
                    rec.copy(&gp.prelude_cur, 0, &gp.cur, 0, ne);
                    rec.barrier();
                    blk_count = prelude_blk_count;
                }
                for ip in self.core_start..self.core_end {
                    let il = if pass == 0 {
                        ip
                    } else {
                        self.n_phys + (pass - 1) * (self.core_end - self.core_start) + (ip - self.core_start)
                    };
                    if pass == 0 {
                        self.record_layer_gpu(&mut rec, gp, il, ip, pos, tau, &mut blk_count, None);
                    } else {
                        let li = ip - self.core_start;
                        rec.inkling_mix(&gp.bank_norm, &gp.bank_out, &gp.loops[li].msg_query, &gp.msgs_buf, ne, 0, bank_count, 4);
                        rec.barrier();
                        self.record_layer_gpu(&mut rec, gp, il, ip, pos, tau, &mut blk_count, Some(li));
                    }
                }
            }

            // Postlude: core_end..n_phys
            for il in self.core_end..self.n_phys {
                self.record_layer_gpu(&mut rec, gp, il, il, pos, tau, &mut blk_count, None);
            }

            // Output mix + norm + lm_head
            if out.logits || out.hidden {
                rec.inkling_mix(&gp.blocks_norm, &gp.cur, &gp.output_res_query, &gp.o_proj, ne, 0, blk_count, 1);
                rec.barrier();
                rec.rms_norm(&gp.o_proj, 0, &gp.output_norm, 0, &gp.hidden_buf, 0, ne, self.eps);
                rec.barrier();
            }
            if out.logits {
                let (bo, wto) = self.output_w.gpu.as_ref().unwrap();
                rec.qmv(*wto, bo, 0, &gp.hidden_buf, 0, &gp.logits_buf, 0, ne, self.output_w.m as u32);
                rec.barrier();
            }
            rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;
            if out.hidden {
                c.download(&gp.hidden_buf, 0, bytemuck::cast_slice_mut(&mut self.hidden)).map_err(|e| EngineError::Backend(e.to_string()))?;
            }
            if out.logits {
                let mut full = vec![0f32; self.n_vocab_unpadded];
                c.download(&gp.logits_buf, 0, bytemuck::cast_slice_mut(&mut full)).map_err(|e| EngineError::Backend(e.to_string()))?;
                for i in 0..self.n_vocab_unpadded {
                    let v = full[i];
                    self.logits[i] = if self.softcap > 0.0 { self.softcap * (v / self.softcap).tanh() } else { v };
                }
            }
            return Ok(());
        }

        let mut cur = rms_norm_weighted(&emb, &self.tok_norm, self.eps);

        let mut blocks: Vec<Vec<f32>> = Vec::new();
        // Prelude: 0..core_start
        for il in 0..self.core_start {
            cur = self.run_layer(il, il, pos, tau, cur, &mut blocks, None, None)?;
        }

        // Looped core: n_passes over core_start..core_end
        let prelude = cur.clone();
        let prelude_blocks = blocks.clone();
        for pass in 0..self.n_passes {
            let mut bank: Vec<Vec<f32>> = Vec::new();
            let mut bank_out: Vec<f32> = Vec::new();
            if pass > 0 {
                bank.extend_from_slice(&blocks[prelude_blocks.len() + 1..]);
                bank_out = cur;
                cur = prelude.clone();
                blocks = prelude_blocks.clone();
            }
            for ip in self.core_start..self.core_end {
                let il = if pass == 0 {
                    ip
                } else {
                    self.n_phys + (pass - 1) * (self.core_end - self.core_start) + (ip - self.core_start)
                };
                if pass == 0 {
                    cur = self.run_layer(il, ip, pos, tau, cur, &mut blocks, None, None)?;
                } else {
                    let li = ip - self.core_start;
                    let msgs = mix_depth(&bank, &bank_out, &self.loop_layers[li].msg_query, self.n_embd, 4);
                    cur = self.run_layer(il, ip, pos, tau, cur, &mut blocks, Some(&msgs), Some(li))?;
                }
            }
        }

        // Postlude: core_end..n_phys
        for il in self.core_end..self.n_phys {
            cur = self.run_layer(il, il, pos, tau, cur, &mut blocks, None, None)?;
        }

        // Output mix + norm
        cur = mix_depth(&blocks, &cur, &self.output_res_query, self.n_embd, 1).pop().unwrap();
        cur = rms_norm_weighted(&cur, &self.output_norm, self.eps);
        if out.hidden {
            self.hidden.copy_from_slice(&cur);
        }
        if out.logits {
            let full = self.matvec(&self.output_w, &cur)?;
            for i in 0..self.n_vocab_unpadded {
                let v = full[i];
                self.logits[i] = if self.softcap > 0.0 {
                    self.softcap * (v / self.softcap).tanh()
                } else {
                    v
                };
            }
        }
        Ok(())
    }

    fn logits(&self) -> &[f32] { &self.logits }
    fn hidden(&self) -> &[f32] { &self.hidden }
}
