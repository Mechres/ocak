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
            let ffn_gate_inp = load_mat(&g, vk.as_ref(), &format!("blk.{i}.ffn_gate_inp.weight"), false)?;
            let exp_probs_b = load_f32_vec(&g, &format!("blk.{i}.exp_probs_b.bias")).ok();

            let gu_name = format!("blk.{i}.ffn_gate_up_exps.weight");
            let dn_name = format!("blk.{i}.ffn_down_exps.weight");
            let t_gu = g.tensor(&gu_name).ok_or_else(|| EngineError::Unsupported(format!("missing {gu_name}")))?;
            let t_dn = g.tensor(&dn_name).ok_or_else(|| EngineError::Unsupported(format!("missing {dn_name}")))?;
            let gu_stride = t_gu.ty.row_bytes(n_embd).unwrap() * (2 * n_ff_exp);
            let dn_stride = t_dn.ty.row_bytes(n_ff_exp).unwrap() * n_embd;
            let gu_scale = load_f32_vec(&g, &format!("blk.{i}.ffn_gate_up_exps.scale")).unwrap_or_else(|_| vec![1.0; n_expert]);
            let dn_scale = load_f32_vec(&g, &format!("blk.{i}.ffn_down_exps.scale")).unwrap_or_else(|_| vec![1.0; n_expert]);

            let put_gpu = i < gpu_moe_limit;
            let (gpu_gu, gpu_dn) = if put_gpu {
                if let (Some(c), Ok(wt_gu), Ok(wt_dn)) = (vk.as_ref(), WType::from_ggml(t_gu.ty), WType::from_ggml(t_dn.ty)) {
                    let b_gu = c.create_device_buffer(t_gu.n_bytes).ok();
                    let b_dn = c.create_device_buffer(t_dn.n_bytes).ok();
                    if let (Some(b_gu), Some(b_dn)) = (b_gu, b_dn) {
                        c.upload(&b_gu, 0, g.tensor_data(t_gu)).map_err(|e| EngineError::Backend(e.to_string()))?;
                        c.upload(&b_dn, 0, g.tensor_data(t_dn)).map_err(|e| EngineError::Backend(e.to_string()))?;
                        gpu_moe_count += 1;
                        (Some((b_gu, wt_gu)), Some((b_dn, wt_dn)))
                    } else {
                        cpu_moe_count += 1;
                        (None, None)
                    }
                } else {
                    cpu_moe_count += 1;
                    (None, None)
                }
            } else {
                cpu_moe_count += 1;
                (None, None)
            };

            phys.push(PhysLayer {
                is_swa, n_head, n_head_kv, n_extent,
                wq, wk, wv, wo, wqkv_gate, attn_q_clip, attn_sink_proj,
                attn_rel, attn_rel_bias, attn_rel_clip,
                attn_k_conv, attn_v_conv, attn_post_conv, ffn_post_conv,
                attn_norm, attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm,
                attn_k_conv_proj, attn_v_conv_proj,
                attn_res_query, ffn_res_query, ffn_norm, ffn_gate_inp, exp_probs_b,
                exps: MoEExps { gu_name, dn_name, gu_ty: t_gu.ty, dn_ty: t_dn.ty, gu_stride, dn_stride, gu_scale, dn_scale, gpu_gu, gpu_dn },
            });
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

        let max_dim = n_vocab_padded.max(n_embd * 4);
        let (gpu_x, gpu_y, gpu_gu, gpu_mid, gpu_sum) = if let Some(c) = &vk {
            (
                Some(c.create_device_buffer((n_embd * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?),
                Some(c.create_device_buffer((max_dim * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?),
                Some(c.create_device_buffer((2 * n_ff_exp * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?),
                Some(c.create_device_buffer((n_ff_exp * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?),
                Some(c.create_device_buffer((n_embd * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?),
            )
        } else {
            (None, None, None, None, None)
        };

        Ok(Self {
            gguf: g, vk, n_phys, core_start, core_end, n_passes,
            n_embd, n_ff_exp, head_dim, n_expert, n_expert_used, n_rel, n_swa, n_conv, block_size,
            n_vocab_unpadded, eps, qk_clip, rel_clip, log_scale_floor, log_scale_alpha,
            exp_w_scale, situ_beta, situ_lin_beta, softcap, max_ctx,
            tok_embd_name: "token_embd.weight".into(), tok_embd_ty,
            tok_norm, output_norm, output_res_query, output_w, phys, loop_layers, slots,
            gpu_x, gpu_y, gpu_gu, gpu_mid, gpu_sum, gpu_moe_count, cpu_moe_count,
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
        let gate = self.matvec(&lp.ctrl_down[ctrl], &cat)?;
        let gain = self.matvec(&lp.ctrl_gain[ctrl], &gate)?;
        let forget = self.matvec(&lp.ctrl_forget[ctrl], &gate)?;
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

        // 4. Q, K, V, R, Sink, Gate projections + ShortConv on K, V
        let mut q = self.matvec(&self.phys[ip].wq, &inputs[0])?;
        let k_raw = self.matvec(&self.phys[ip].wk, &inputs[1])?;
        let v_raw = self.matvec(&self.phys[ip].wv, &inputs[2])?;
        let k_dyn = if let Some(ref p) = self.phys[ip].attn_k_conv_proj { Some(self.matvec(p, &inputs[1])?) } else { None };
        let v_dyn = if let Some(ref p) = self.phys[ip].attn_v_conv_proj { Some(self.matvec(p, &inputs[2])?) } else { None };
        let mut k = convolve_step(&k_raw, &self.phys[ip].attn_k_conv, k_dyn.as_deref(), &mut self.slots[il].k_hist, self.n_conv);
        let mut v = convolve_step(&v_raw, &self.phys[ip].attn_v_conv, v_dyn.as_deref(), &mut self.slots[il].v_hist, self.n_conv);
        let mut r = self.matvec(&self.phys[ip].attn_rel, &inputs[0])?;
        let sinks = self.matvec(&self.phys[ip].attn_sink_proj, &inputs[0])?;
        let gate = self.matvec(&self.phys[ip].wqkv_gate, &inputs[0])?;

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
            c.upload(bx, 0, bytemuck::cast_slice(&x_moe)).map_err(|e| EngineError::Backend(e.to_string()))?;
            let mut gu_cpu = vec![0f32; 2 * n_ff];
            let mut mid_cpu = vec![0f32; n_ff];
            let mut dn_cpu = vec![0f32; self.n_embd];
            // Dispatch each active expert's NVFP4 gate_up & down on GPU (SITU in f32 on host)
            for &(e, ew) in &top_w {
                let mut rec = c.begin();
                rec.qmv(*wt_gu, b_gu, (e * exps.gu_stride) as u64, bx, 0, bgu, 0, self.n_embd as u32, (2 * n_ff) as u32);
                rec.finish_and_wait().map_err(|er| EngineError::Backend(er.to_string()))?;
                c.download(bgu, 0, bytemuck::cast_slice_mut(&mut gu_cpu)).map_err(|er| EngineError::Backend(er.to_string()))?;
                let gs = exps.gu_scale[e];
                for j in 0..n_ff {
                    let g = gu_cpu[j] * gs;
                    let u = gu_cpu[n_ff + j] * gs;
                    let act = beta * (g / beta).tanh() * (1.0 / (1.0 + (-g).exp()));
                    let lin = if lb > 0.0 { lb * (u / lb).tanh() } else { u };
                    mid_cpu[j] = act * lin;
                }
                c.upload(bmid, 0, bytemuck::cast_slice(&mid_cpu)).map_err(|er| EngineError::Backend(er.to_string()))?;
                let mut rec = c.begin();
                rec.qmv(*wt_dn, b_dn, (e * exps.dn_stride) as u64, bmid, 0, by, 0, n_ff as u32, self.n_embd as u32);
                rec.finish_and_wait().map_err(|er| EngineError::Backend(er.to_string()))?;
                c.download(by, 0, bytemuck::cast_slice_mut(&mut dn_cpu)).map_err(|er| EngineError::Backend(er.to_string()))?;
                let ds = ew * exps.dn_scale[e];
                for j in 0..self.n_embd { moe_out[j] += ds * dn_cpu[j]; }
            }
            let _ = bsum;
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
}

impl Executor for InklingExec {
    fn name(&self) -> String {
        if let Some(c) = &self.vk {
            format!("vulkan-inkling-hybrid ({}, gpu_moe={}, cpu_moe={})", c.device_name(), self.gpu_moe_count, self.cpu_moe_count)
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
        Ok(())
    }

    fn forward(&mut self, token: u32, pos: usize, _page_table: &[u32], out: Out) -> Result<(), EngineError> {
        if pos >= self.max_ctx {
            return Err(EngineError::Context(format!("pos {pos} >= max_ctx {}", self.max_ctx)));
        }
        let tau = 1.0 + self.log_scale_alpha * (((pos as f32 + 1.0) / self.log_scale_floor).max(1.0)).ln();

        // Token embedding + token_embd_norm
        let te = self.gguf.tensor(&self.tok_embd_name).unwrap();
        let rb = self.tok_embd_ty.row_bytes(self.n_embd).unwrap();
        let row = &self.gguf.tensor_data(te)[token as usize * rb..(token as usize + 1) * rb];
        let mut emb = vec![0f32; self.n_embd];
        dequantize_row(self.tok_embd_ty, row, &mut emb, self.n_embd);
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
