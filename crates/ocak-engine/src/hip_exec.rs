//! Native AMD HIP executor (`HipExec`) for dense & hybrid transformer models
//! (`llama`, `qwen2`, `qwen3vl`, `lfm2`, `lfm3`, `qwen35`).
//! Runs a 3-stream HIP Graph DAG for single-token decode (fused Q8 norms/swiglu/rope+kv/attn-merge,
//! GQA-grouped Split-K decode attention, 4-byte GPU argmax) and batched prefill (256-token chunks,
//! paged flash attention, Split-K `qmm`).
use crate::config::{LayerKind, ModelConfig};
use crate::exec::{EngineError, Executor, Out};
use ocak_gguf::{GgmlType, GgufFile, TensorInfo};
use ocak_hip::{
    HipBuffer, HipContext, HipDenseOps, HipEvent, HipGraphExec, HipHybridOps, HipIqKernels,
    HipKernels, HipStream, HipWType,
};
use std::sync::Arc;

const PREFILL_BATCH: usize = 256;
const DECODE_ATTN_SPLITS: u32 = 8;

struct HipWeight {
    buf: HipBuffer,
    ty: HipWType,
    k: u32,
    m: u32,
}

enum HipMixer {
    FullAttn {
        wq: HipWeight,
        wk: HipWeight,
        wv: HipWeight,
        wo: HipWeight,
        bq: Option<HipBuffer>,
        bk: Option<HipBuffer>,
        bv: Option<HipBuffer>,
        q_norm: Option<HipBuffer>,
        k_norm: Option<HipBuffer>,
        gated_q: bool,
        n_head_kv: u32,
        kv_layer_idx: usize,
        sliding_window: u32,
        rope_base: f32,
        use_rope_factors: bool,
        head_gate_w: Option<HipWeight>,
    },
    ShortConv {
        in_proj: HipWeight,
        out_proj: HipWeight,
        conv_w: HipBuffer,
        state: HipBuffer,
        l_cache: u32,
    },
    DeltaNet {
        wqkv: HipWeight,
        wgate: HipWeight,
        walpha: HipWeight,
        wbeta: HipWeight,
        wout: HipWeight,
        conv_w: HipBuffer,
        dt_bias: HipBuffer,
        ssm_a: HipBuffer,
        ssm_norm: HipBuffer,
        conv_state: HipBuffer,
        ssm_state: HipBuffer,
        n_k_heads: u32,
        n_v_heads: u32,
        state_dim: u32,
        k_conv: u32,
        conv_ch: u32,
    },
}

struct HipLayer {
    attn_norm: HipBuffer,
    ffn_norm: HipBuffer,
    post_attn_norm: Option<HipBuffer>,
    post_ffw_norm: Option<HipBuffer>,
    mixer: HipMixer,
    gate: HipWeight,
    up: HipWeight,
    down: HipWeight,
}

pub struct HipExec {
    ctx: Arc<HipContext>,
    stream: HipStream,
    stream_b: HipStream,
    stream_c: HipStream,
    ev_fork: HipEvent,
    ev_join_b: HipEvent,
    ev_join_c: HipEvent,
    kernels: HipKernels,
    iq_kernels: Option<HipIqKernels>,
    iq_tbl: Option<HipBuffer>,
    ops: HipDenseOps,
    hops: HipHybridOps,
    cfg: ModelConfig,
    layers: Vec<HipLayer>,
    embd: HipWeight,
    out_norm: HipBuffer,
    rope_factors: HipBuffer,
    output: Option<HipWeight>,
    kv: HipBuffer,
    layer_stride_bytes: usize,
    pt_buf: HipBuffer,
    pt_last: Vec<u32>,
    max_pages: usize,
    // Scratch buffers sized for PREFILL_BATCH
    toks_buf: HipBuffer,
    x: HipBuffer,
    xn: HipBuffer,
    qg: HipBuffer,
    q: HipBuffer,
    q_gate: HipBuffer,
    head_gate_buf: HipBuffer,
    k: HipBuffer,
    v: HipBuffer,
    attn: HipBuffer,
    o: HipBuffer,
    bcx: HipBuffer,
    qkv_raw: HipBuffer,
    qkv_conv: HipBuffer,
    z_buf: HipBuffer,
    alpha_buf: HipBuffer,
    beta_buf: HipBuffer,
    dn_out: HipBuffer,
    dn_norm: HipBuffer,
    gate: HipBuffer,
    up: HipBuffer,
    act: HipBuffer,
    // Quantized activation scratch
    xq: HipBuffer,
    xsc: HipBuffer,
    // Split-K prefill scratch & decode split-K attention scratch
    split_buf: HipBuffer,
    attn_part_out: HipBuffer,
    attn_part_ml: HipBuffer,
    // Device-resident [token, pos] metadata for static HIP Graph decode
    step_meta: HipBuffer,
    graph_argmax: Option<HipGraphExec>,
    graph_logits: Option<HipGraphExec>,
    // Output buffers
    hidden_buf: HipBuffer,
    logits_buf: HipBuffer,
    argmax_pv: HipBuffer,
    argmax_pi: HipBuffer,
    argmax_out: HipBuffer,
    logits: Vec<f32>,
    hidden: Vec<f32>,
}

fn hip_wtype(ty: GgmlType) -> Result<HipWType, EngineError> {
    match ty {
        GgmlType::F32 => Ok(HipWType::F32),
        GgmlType::F16 => Ok(HipWType::F16),
        GgmlType::Q4_0 => Ok(HipWType::Q4_0),
        GgmlType::Q8_0 => Ok(HipWType::Q8_0),
        GgmlType::Q4K => Ok(HipWType::Q4K),
        GgmlType::Q5K => Ok(HipWType::Q5K),
        GgmlType::Q6K => Ok(HipWType::Q6K),
        GgmlType::Q2K => Ok(HipWType::Q2K),
        GgmlType::Q3K => Ok(HipWType::Q3K),
        GgmlType::IQ1S => Ok(HipWType::IQ1S),
        GgmlType::IQ1M => Ok(HipWType::IQ1M),
        GgmlType::IQ2XXS => Ok(HipWType::IQ2XXS),
        GgmlType::IQ2XS => Ok(HipWType::IQ2XS),
        GgmlType::IQ2S => Ok(HipWType::IQ2S),
        GgmlType::IQ3XXS => Ok(HipWType::IQ3XXS),
        GgmlType::IQ3S => Ok(HipWType::IQ3S),
        GgmlType::IQ4NL => Ok(HipWType::IQ4NL),
        GgmlType::IQ4XS => Ok(HipWType::IQ4XS),
        other => Err(EngineError::Unsupported(format!(
            "HIP backend does not yet support weight format {other:?}"
        ))),
    }
}

fn upload_f32(ctx: &Arc<HipContext>, g: &GgufFile, name: &str) -> Result<HipBuffer, EngineError> {
    let ti = g
        .tensor(name)
        .ok_or_else(|| EngineError::Unsupported(format!("missing {name}")))?;
    let data = g.tensor_to_f32(ti);
    ctx.buffer_from_slice(&data)
        .map_err(|e| EngineError::Backend(e.to_string()))
}

fn upload_weight(
    ctx: &Arc<HipContext>,
    g: &GgufFile,
    ti: &TensorInfo,
) -> Result<HipWeight, EngineError> {
    let ty = hip_wtype(ti.ty)?;
    let raw = g.tensor_data(ti);
    let buf = ctx
        .buffer_from_slice(raw)
        .map_err(|e| EngineError::Backend(e.to_string()))?;
    Ok(HipWeight {
        buf,
        ty,
        k: ti.n_cols() as u32,
        m: ti.n_rows() as u32,
    })
}

fn req_tensor<'a>(g: &'a GgufFile, name: &str) -> Result<&'a TensorInfo, EngineError> {
    g.tensor(name)
        .ok_or_else(|| EngineError::Unsupported(format!("missing {name}")))
}

impl HipExec {
    pub fn new(g: Arc<GgufFile>, cfg: ModelConfig, n_pages: usize) -> Result<Self, EngineError> {
        if cfg.head_dim > 256 {
            return Err(EngineError::Unsupported(format!(
                "HIP attention supports head_dim <= 256 (got {})",
                cfg.head_dim
            )));
        }

        g.prefetch_background();
        let ctx = HipContext::new(0).map_err(|e| EngineError::Backend(e.to_string()))?;
        let stream = ctx
            .create_stream()
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        let stream_b = ctx
            .create_stream()
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        let stream_c = ctx
            .create_stream()
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        let ev_fork = ctx
            .create_event(false)
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        let ev_join_b = ctx
            .create_event(false)
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        let ev_join_c = ctx
            .create_event(false)
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        let kernels = HipKernels::compile(&ctx).map_err(|e| EngineError::Backend(e.to_string()))?;
        let ops = HipDenseOps::compile(&ctx).map_err(|e| EngineError::Backend(e.to_string()))?;
        let hops = HipHybridOps::compile(&ctx).map_err(|e| EngineError::Backend(e.to_string()))?;

        let mk_f32 = |elems: usize| -> Result<HipBuffer, EngineError> {
            ctx.alloc_buffer(elems.max(4) * 4)
                .map_err(|e| EngineError::Backend(e.to_string()))
        };

        let mut max_conv_ch = 4usize;
        let mut max_v_dim = 4usize;
        let mut max_v_heads = 4usize;
        let mut has_gated_q = false;
        let mut has_shortconv = false;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for (i, &kind) in cfg.layer_kinds.iter().enumerate() {
            let p = |s: &str| format!("blk.{i}.{s}");
            let mixer = match kind {
                LayerKind::FullAttn {
                    n_head_kv,
                    gated_q,
                    qk_norm,
                    kv_layer_idx,
                    sliding_window,
                    rope_base,
                    use_rope_factors,
                    head_gate,
                } => {
                    if gated_q {
                        has_gated_q = true;
                    }
                    let opt_bias = |s: &str| -> Result<Option<HipBuffer>, EngineError> {
                        if cfg.qkv_bias && g.tensor(&p(s)).is_some() {
                            Ok(Some(upload_f32(&ctx, &g, &p(s))?))
                        } else {
                            Ok(None)
                        }
                    };
                    HipMixer::FullAttn {
                        wq: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_q.weight"))?)?,
                        wk: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_k.weight"))?)?,
                        wv: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_v.weight"))?)?,
                        wo: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_output.weight"))?)?,
                        bq: opt_bias("attn_q.bias")?,
                        bk: opt_bias("attn_k.bias")?,
                        bv: opt_bias("attn_v.bias")?,
                        q_norm: if qk_norm { Some(upload_f32(&ctx, &g, &p("attn_q_norm.weight"))?) } else { None },
                        k_norm: if qk_norm { Some(upload_f32(&ctx, &g, &p("attn_k_norm.weight"))?) } else { None },
                        gated_q,
                        n_head_kv: n_head_kv as u32,
                        kv_layer_idx,
                        sliding_window: sliding_window as u32,
                        rope_base,
                        use_rope_factors,
                        head_gate_w: if head_gate {
                            Some(upload_weight(&ctx, &g, req_tensor(&g, &p("attn_gate.weight"))?)?)
                        } else {
                            None
                        },
                    }
                }
                LayerKind::ShortConv { l_cache } => {
                    has_shortconv = true;
                    let state = mk_f32(cfg.n_embd * (l_cache - 1))?;
                    state.zero().map_err(|e| EngineError::Backend(e.to_string()))?;
                    HipMixer::ShortConv {
                        in_proj: upload_weight(&ctx, &g, req_tensor(&g, &p("shortconv.in_proj.weight"))?)?,
                        out_proj: upload_weight(&ctx, &g, req_tensor(&g, &p("shortconv.out_proj.weight"))?)?,
                        conv_w: upload_f32(&ctx, &g, &p("shortconv.conv.weight"))?,
                        state,
                        l_cache: l_cache as u32,
                    }
                }
                LayerKind::DeltaNet { n_k_heads, n_v_heads, state_dim, k_conv } => {
                    if state_dim > 128 || !state_dim.is_multiple_of(32) {
                        return Err(EngineError::Unsupported(format!(
                            "HIP DeltaNet requires state_dim <= 128 and multiple of 32 (got {state_dim})"
                        )));
                    }
                    let conv_ch = (2 * n_k_heads + n_v_heads) * state_dim;
                    let v_dim = n_v_heads * state_dim;
                    max_conv_ch = max_conv_ch.max(conv_ch);
                    max_v_dim = max_v_dim.max(v_dim);
                    max_v_heads = max_v_heads.max(n_v_heads);
                    let conv_state = mk_f32(conv_ch * (k_conv - 1))?;
                    let ssm_state = mk_f32(n_v_heads * state_dim * state_dim)?;
                    conv_state.zero().map_err(|e| EngineError::Backend(e.to_string()))?;
                    ssm_state.zero().map_err(|e| EngineError::Backend(e.to_string()))?;
                    HipMixer::DeltaNet {
                        wqkv: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_qkv.weight"))?)?,
                        wgate: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_gate.weight"))?)?,
                        walpha: upload_weight(&ctx, &g, req_tensor(&g, &p("ssm_alpha.weight"))?)?,
                        wbeta: upload_weight(&ctx, &g, req_tensor(&g, &p("ssm_beta.weight"))?)?,
                        wout: upload_weight(&ctx, &g, req_tensor(&g, &p("ssm_out.weight"))?)?,
                        conv_w: upload_f32(&ctx, &g, &p("ssm_conv1d.weight"))?,
                        dt_bias: upload_f32(&ctx, &g, &p("ssm_dt.bias"))?,
                        ssm_a: upload_f32(&ctx, &g, &p("ssm_a"))?,
                        ssm_norm: upload_f32(&ctx, &g, &p("ssm_norm.weight"))?,
                        conv_state,
                        ssm_state,
                        n_k_heads: n_k_heads as u32,
                        n_v_heads: n_v_heads as u32,
                        state_dim: state_dim as u32,
                        k_conv: k_conv as u32,
                        conv_ch: conv_ch as u32,
                    }
                }
            };

            layers.push(HipLayer {
                attn_norm: upload_f32(&ctx, &g, &p("attn_norm.weight"))?,
                ffn_norm: upload_f32(&ctx, &g, &p(&cfg.ffn_norm_suffix))?,
                post_attn_norm: if cfg.has_post_norms {
                    Some(upload_f32(&ctx, &g, &p("post_attention_norm.weight"))?)
                } else {
                    None
                },
                post_ffw_norm: if cfg.has_post_norms {
                    Some(upload_f32(&ctx, &g, &p("post_ffw_norm.weight"))?)
                } else {
                    None
                },
                mixer,
                gate: upload_weight(&ctx, &g, req_tensor(&g, &p("ffn_gate.weight"))?)?,
                up: upload_weight(&ctx, &g, req_tensor(&g, &p("ffn_up.weight"))?)?,
                down: upload_weight(&ctx, &g, req_tensor(&g, &p("ffn_down.weight"))?)?,
            });
        }

        let embd = upload_weight(&ctx, &g, req_tensor(&g, "token_embd.weight")?)?;
        let out_norm = upload_f32(&ctx, &g, &cfg.output_norm_name)?;
        let rope_factors = if g.tensor("rope_freqs.weight").is_some() {
            upload_f32(&ctx, &g, "rope_freqs.weight")?
        } else {
            mk_f32(cfg.head_dim / 2)?
        };
        let output = if cfg.tied_output {
            None
        } else {
            Some(upload_weight(&ctx, &g, req_tensor(&g, "output.weight")?)?)
        };

        let has_iq = embd.ty.is_iq()
            || output.as_ref().is_some_and(|w| w.ty.is_iq())
            || layers.iter().any(|l| {
                l.gate.ty.is_iq()
                    || l.up.ty.is_iq()
                    || l.down.ty.is_iq()
                    || match &l.mixer {
                        HipMixer::FullAttn { wq, wk, wv, wo, head_gate_w, .. } => {
                            wq.ty.is_iq()
                                || wk.ty.is_iq()
                                || wv.ty.is_iq()
                                || wo.ty.is_iq()
                                || head_gate_w.as_ref().is_some_and(|w| w.ty.is_iq())
                        }
                        HipMixer::ShortConv { in_proj, out_proj, .. } => {
                            in_proj.ty.is_iq() || out_proj.ty.is_iq()
                        }
                        HipMixer::DeltaNet { wqkv, wgate, walpha, wbeta, wout, .. } => {
                            wqkv.ty.is_iq()
                                || wgate.ty.is_iq()
                                || walpha.ty.is_iq()
                                || wbeta.ty.is_iq()
                                || wout.ty.is_iq()
                        }
                    }
            });
        let (iq_kernels, iq_tbl) = if has_iq {
            let tbl_data = ocak_gguf::iq_tables::packed_gpu_table();
            let tbl = ctx
                .buffer_from_slice(&tbl_data)
                .map_err(|e| EngineError::Backend(e.to_string()))?;
            let iqk = HipIqKernels::compile(&ctx).map_err(|e| EngineError::Backend(e.to_string()))?;
            (Some(iqk), Some(tbl))
        } else {
            (None, None)
        };

        let q_dim = cfg.n_head * cfg.head_dim;
        let kv_dim = cfg.kv_dim();
        let layer_stride_bytes = n_pages * 2 * 16 * kv_dim * 2;
        let kv = ctx
            .alloc_buffer((cfg.n_kv_layers * layer_stride_bytes).max(16))
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        let pt_buf = ctx
            .alloc_buffer(n_pages.max(16) * 4)
            .map_err(|e| EngineError::Backend(e.to_string()))?;

        let max_k = cfg.n_embd.max(cfg.n_ff).max(q_dim).max(max_v_dim);
        let max_q_elems = max_k * PREFILL_BATCH;
        let xq = ctx
            .alloc_buffer(max_q_elems)
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        let xsc = ctx
            .alloc_buffer((max_q_elems / 32).max(1) * 16)
            .map_err(|e| EngineError::Backend(e.to_string()))?;

        Ok(Self {
            ctx: Arc::clone(&ctx),
            stream,
            stream_b,
            stream_c,
            ev_fork,
            ev_join_b,
            ev_join_c,
            kernels,
            iq_kernels,
            iq_tbl,
            ops,
            hops,
            layers,
            embd,
            out_norm,
            rope_factors,
            output,
            kv,
            layer_stride_bytes,
            pt_buf,
            pt_last: Vec::new(),
            max_pages: n_pages,
            toks_buf: ctx
                .alloc_buffer(PREFILL_BATCH * 4)
                .map_err(|e| EngineError::Backend(e.to_string()))?,
            x: mk_f32(cfg.n_embd * PREFILL_BATCH)?,
            xn: mk_f32(cfg.n_embd * PREFILL_BATCH)?,
            qg: mk_f32(if has_gated_q { 2 * q_dim * PREFILL_BATCH } else { 4 })?,
            q: mk_f32(q_dim * PREFILL_BATCH)?,
            q_gate: mk_f32(if has_gated_q { q_dim * PREFILL_BATCH } else { 4 })?,
            head_gate_buf: mk_f32(cfg.n_head * PREFILL_BATCH)?,
            k: mk_f32(kv_dim * PREFILL_BATCH)?,
            v: mk_f32(kv_dim * PREFILL_BATCH)?,
            attn: mk_f32(q_dim * PREFILL_BATCH)?,
            o: mk_f32(cfg.n_embd * PREFILL_BATCH)?,
            bcx: mk_f32(if has_shortconv { 3 * cfg.n_embd * PREFILL_BATCH } else { 4 })?,
            qkv_raw: mk_f32(max_conv_ch * PREFILL_BATCH)?,
            qkv_conv: mk_f32(max_conv_ch * PREFILL_BATCH)?,
            z_buf: mk_f32(max_v_dim * PREFILL_BATCH)?,
            alpha_buf: mk_f32(max_v_heads * PREFILL_BATCH)?,
            beta_buf: mk_f32(max_v_heads * PREFILL_BATCH)?,
            dn_out: mk_f32(max_v_dim * PREFILL_BATCH)?,
            dn_norm: mk_f32(max_v_dim * PREFILL_BATCH)?,
            gate: mk_f32(cfg.n_ff * PREFILL_BATCH)?,
            up: mk_f32(cfg.n_ff * PREFILL_BATCH)?,
            act: mk_f32(cfg.n_ff * PREFILL_BATCH)?,
            xq,
            xsc,
            split_buf: mk_f32(PREFILL_BATCH * cfg.n_embd.max(q_dim).max(max_v_dim) * 4)?,
            attn_part_out: mk_f32(cfg.n_head * DECODE_ATTN_SPLITS as usize * cfg.head_dim)?,
            attn_part_ml: mk_f32(cfg.n_head * DECODE_ATTN_SPLITS as usize * 2)?,
            step_meta: ctx
                .alloc_buffer(8)
                .map_err(|e| EngineError::Backend(e.to_string()))?,
            graph_argmax: None,
            graph_logits: None,
            hidden_buf: mk_f32(cfg.n_embd)?,
            logits_buf: mk_f32(cfg.n_vocab)?,
            argmax_pv: mk_f32(256)?,
            argmax_pi: mk_f32(256)?,
            argmax_out: mk_f32(1)?,
            logits: vec![0f32; cfg.n_vocab],
            hidden: vec![0f32; cfg.n_embd],
            cfg,
        })
    }

    fn sync_page_table(&mut self, page_table: &[u32]) -> Result<(), EngineError> {
        if page_table.len() > self.max_pages {
            return Err(EngineError::KvExhausted);
        }
        if self.pt_last.as_slice() != page_table {
            self.pt_buf
                .upload(bytemuck::cast_slice(page_table))
                .map_err(|e| EngineError::Backend(e.to_string()))?;
            self.pt_last.clear();
            self.pt_last.extend_from_slice(page_table);
        }
        Ok(())
    }

    fn qmv(
        &self,
        w: &HipWeight,
        s: &HipStream,
        x_f32: &HipBuffer,
        y: &HipBuffer,
    ) -> Result<(), ocak_hip::HipError> {
        if w.ty.is_iq() {
            self.iq_kernels.as_ref().unwrap().qmv(
                w.ty,
                s,
                &w.buf,
                &self.xq,
                &self.xsc,
                y,
                self.iq_tbl.as_ref().unwrap(),
                w.k,
                w.m,
                1,
            )
        } else {
            self.kernels.qmv(w.ty, s, &w.buf, x_f32, &self.xq, &self.xsc, y, w.k, w.m, 1)
        }
    }

    fn qmm(
        &self,
        w: &HipWeight,
        s: &HipStream,
        x_f32: &HipBuffer,
        y: &HipBuffer,
        batch: u32,
    ) -> Result<(), ocak_hip::HipError> {
        if w.ty.is_iq() {
            self.iq_kernels.as_ref().unwrap().qmm_with_split(
                w.ty,
                s,
                &w.buf,
                &self.xq,
                &self.xsc,
                y,
                self.iq_tbl.as_ref().unwrap(),
                Some(&self.split_buf),
                w.k,
                w.m,
                batch,
            )
        } else {
            self.kernels.qmm_with_split(
                w.ty,
                s,
                &w.buf,
                x_f32,
                &self.xq,
                &self.xsc,
                y,
                Some(&self.split_buf),
                w.k,
                w.m,
                batch,
            )
        }
    }

    /// Records the 1-token decode pass onto `s` using fused Q8 norms/swiglu/rope+kv/attn-merge
    /// and a 3-stream HIP Graph DAG (`wq/wk/wv`, `DeltaNet` projections, and `gate/up` concurrent).
    fn record_decode_step(&self, s: &HipStream, with_argmax: bool) -> Result<(), ocak_hip::HipError> {
        let ne = self.cfg.n_embd as u32;
        let nh = self.cfg.n_head as u32;
        let hd = self.cfg.head_dim as u32;
        let nr = self.cfg.n_rot as u32;
        let eps = self.cfg.eps;
        let mode = self.cfg.rope_mode;
        let sb = &self.stream_b;
        let sc = &self.stream_c;

        self.ops.embed_token_meta(
            s,
            &self.embd.buf,
            self.embd.ty,
            self.iq_tbl.as_ref(),
            &self.step_meta,
            &self.x,
            ne,
        )?;
        self.ops
            .rms_norm_q8(s, &self.x, &self.layers[0].attn_norm, &self.xn, &self.xq, &self.xsc, ne, eps)?;

        for (li, l) in self.layers.iter().enumerate() {
            match &l.mixer {
                HipMixer::FullAttn {
                    wq, wk, wv, wo, bq, bk, bv, q_norm, k_norm, gated_q,
                    n_head_kv, kv_layer_idx, sliding_window, rope_base, use_rope_factors, head_gate_w,
                } => {
                    self.ev_fork.record(s)?;
                    sb.wait_event(&self.ev_fork)?;
                    sc.wait_event(&self.ev_fork)?;

                    let q_dst = if *gated_q { &self.qg } else { &self.q };
                    self.qmv(wq, s, &self.xn, q_dst)?;
                    self.qmv(wk, sb, &self.xn, &self.k)?;
                    self.qmv(wv, sc, &self.xn, &self.v)?;
                    if let Some(hgw) = head_gate_w {
                        self.qmv(hgw, sc, &self.xn, &self.head_gate_buf)?;
                    }

                    if *gated_q {
                        self.hops.split_qg(s, &self.qg, &self.q, &self.q_gate, nh, hd, 1)?;
                    }
                    if let Some(bq) = bq { self.ops.add_bias(s, &self.q, bq, nh * hd, 1)?; }
                    if let Some(bk) = bk { self.ops.add_bias(sb, &self.k, bk, *n_head_kv * hd, 1)?; }
                    if let Some(bv) = bv { self.ops.add_bias(sc, &self.v, bv, *n_head_kv * hd, 1)?; }
                    if let Some(qn) = q_norm { self.ops.rms_norm_heads(s, &self.q, qn, hd, nh, eps)?; }
                    if let Some(kn) = k_norm { self.ops.rms_norm_heads(sb, &self.k, kn, hd, *n_head_kv, eps)?; }

                    self.ev_join_b.record(sb)?;
                    self.ev_join_c.record(sc)?;
                    s.wait_event(&self.ev_join_b)?;
                    s.wait_event(&self.ev_join_c)?;

                    let kv_off = *kv_layer_idx * self.layer_stride_bytes;
                    if *use_rope_factors {
                        self.hops.rope_ext_kv_write_meta(
                            s, &self.q, &self.k, &self.v, &self.rope_factors,
                            &self.kv, kv_off, &self.pt_buf, &self.step_meta,
                            nh, *n_head_kv, hd, nr, *rope_base, mode, true,
                        )?;
                    } else {
                        self.ops.rope_qk_kv_write_meta(
                            s, &self.q, &self.k, &self.v, &self.kv, kv_off,
                            &self.pt_buf, &self.step_meta, nh, *n_head_kv, hd, nr, *rope_base, mode,
                        )?;
                    }

                    if *gated_q || head_gate_w.is_some() {
                        self.ops.attention_decode_split_meta(
                            s, &self.q, &self.kv, kv_off, &self.pt_buf, &self.step_meta,
                            &self.attn_part_out, &self.attn_part_ml, &self.attn,
                            nh, *n_head_kv, hd, *sliding_window, DECODE_ATTN_SPLITS,
                        )?;
                        if *gated_q && head_gate_w.is_none() {
                            self.hops.mul_sigmoid_q8(s, &self.attn, &self.q_gate, &self.xq, &self.xsc, nh * hd)?;
                        } else if !*gated_q && head_gate_w.is_some() {
                            self.hops.mul_head_sigmoid_q8(s, &self.attn, &self.head_gate_buf, &self.xq, &self.xsc, nh, hd)?;
                        } else {
                            self.hops.mul_sigmoid(s, &self.attn, &self.q_gate, nh * hd)?;
                            self.hops.mul_head_sigmoid_q8(s, &self.attn, &self.head_gate_buf, &self.xq, &self.xsc, nh, hd)?;
                        }
                    } else {
                        self.ops.attention_decode_split_q8_meta(
                            s, &self.q, &self.kv, kv_off, &self.pt_buf, &self.step_meta,
                            &self.attn_part_out, &self.attn_part_ml, &self.attn,
                            &self.xq, &self.xsc, nh, *n_head_kv, hd, *sliding_window, DECODE_ATTN_SPLITS,
                        )?;
                    }
                    self.qmv(wo, s, &self.attn, &self.o)?;
                }
                HipMixer::ShortConv { in_proj, out_proj, conv_w, state, l_cache } => {
                    self.qmv(in_proj, s, &self.xn, &self.bcx)?;
                    self.hops.shortconv_step_q8(s, &self.bcx, conv_w, state, &self.gate, &self.xq, &self.xsc, ne, *l_cache)?;
                    self.qmv(out_proj, s, &self.gate, &self.o)?;
                }
                HipMixer::DeltaNet {
                    wqkv, wgate, walpha, wbeta, wout, conv_w, dt_bias, ssm_a, ssm_norm,
                    conv_state, ssm_state, n_k_heads, n_v_heads, state_dim, k_conv, conv_ch,
                } => {
                    self.ev_fork.record(s)?;
                    sb.wait_event(&self.ev_fork)?;
                    sc.wait_event(&self.ev_fork)?;

                    self.qmv(wqkv, s, &self.xn, &self.qkv_raw)?;
                    self.qmv(wgate, sb, &self.xn, &self.z_buf)?;
                    self.qmv(walpha, sc, &self.xn, &self.alpha_buf)?;
                    self.qmv(wbeta, sc, &self.xn, &self.beta_buf)?;

                    self.hops.ssm_conv_silu(s, &self.qkv_raw, conv_w, conv_state, &self.qkv_conv, *conv_ch, *k_conv, 1)?;

                    self.ev_join_b.record(sb)?;
                    self.ev_join_c.record(sc)?;
                    s.wait_event(&self.ev_join_c)?;
                    self.hops.deltanet_step(
                        s, &self.qkv_conv, &self.alpha_buf, &self.beta_buf,
                        dt_bias, ssm_a, ssm_state, &self.dn_out,
                        *n_k_heads, *n_v_heads, *state_dim, eps, 1,
                    )?;
                    s.wait_event(&self.ev_join_b)?;
                    self.hops.gated_rms_norm_q8(s, &self.dn_out, ssm_norm, &self.z_buf, &self.dn_norm, &self.xq, &self.xsc, *n_v_heads, *state_dim, eps)?;
                    self.qmv(wout, s, &self.dn_norm, &self.o)?;
                }
            }

            if let Some(pan) = &l.post_attn_norm {
                self.hops.post_rms_add_rms_norm_q8(s, &self.x, &self.o, pan, &l.ffn_norm, &self.xn, &self.xq, &self.xsc, ne, eps)?;
            } else {
                self.ops.add_rms_norm_q8(s, &self.x, &self.o, &l.ffn_norm, &self.xn, &self.xq, &self.xsc, ne, eps)?;
            }

            self.ev_fork.record(s)?;
            sb.wait_event(&self.ev_fork)?;
            self.qmv(&l.gate, s, &self.xn, &self.gate)?;
            self.qmv(&l.up, sb, &self.xn, &self.up)?;
            self.ev_join_b.record(sb)?;
            s.wait_event(&self.ev_join_b)?;

            self.ops.swiglu_q8(s, &self.gate, &self.up, &self.act, &self.xq, &self.xsc, self.cfg.n_ff as u32)?;
            self.qmv(&l.down, s, &self.act, &self.o)?;

            let next_norm = if li + 1 < self.layers.len() {
                &self.layers[li + 1].attn_norm
            } else {
                &self.out_norm
            };
            if let Some(pfn) = &l.post_ffw_norm {
                self.hops.post_rms_add_rms_norm_q8(s, &self.x, &self.o, pfn, next_norm, &self.xn, &self.xq, &self.xsc, ne, eps)?;
            } else {
                self.ops.add_rms_norm_q8(s, &self.x, &self.o, next_norm, &self.xn, &self.xq, &self.xsc, ne, eps)?;
            }
        }

        let head = self.output.as_ref().unwrap_or(&self.embd);
        self.qmv(head, s, &self.xn, &self.logits_buf)?;
        if with_argmax {
            self.ops.argmax(
                s,
                &self.logits_buf,
                &self.argmax_pv,
                &self.argmax_pi,
                &self.argmax_out,
                self.cfg.n_vocab as u32,
            )?;
        }
        Ok(())
    }

    /// Runs all transformer layers for a prefill batch (`2..=PREFILL_BATCH`) starting at `pos0`.
    fn run_layers_prefill(&self, pos0: usize, batch: usize) -> Result<(), EngineError> {
        let b = batch as u32;
        let ne = self.cfg.n_embd as u32;
        let nh = self.cfg.n_head as u32;
        let hd = self.cfg.head_dim as u32;
        let nr = self.cfg.n_rot as u32;
        let eps = self.cfg.eps;
        let mode = self.cfg.rope_mode;
        let s = &self.stream;
        let map_e = |e: ocak_hip::HipError| EngineError::Backend(e.to_string());

        self.kernels
            .rms_norm(s, &self.x, &self.layers[0].attn_norm, &self.xn, ne, b, eps)
            .map_err(map_e)?;

        for (li, l) in self.layers.iter().enumerate() {
            match &l.mixer {
                HipMixer::FullAttn {
                    wq, wk, wv, wo, bq, bk, bv, q_norm, k_norm, gated_q,
                    n_head_kv, kv_layer_idx, sliding_window, rope_base, use_rope_factors, head_gate_w,
                } => {
                    if wq.ty.uses_q8_act() || wk.ty.uses_q8_act() || wv.ty.uses_q8_act()
                        || head_gate_w.as_ref().is_some_and(|w| w.ty.uses_q8_act())
                    {
                        self.kernels
                            .quant_q8(s, &self.xn, &self.xq, &self.xsc, (batch * self.cfg.n_embd / 32) as u32)
                            .map_err(map_e)?;
                    }
                    let q_dst = if *gated_q { &self.qg } else { &self.q };
                    self.qmm(wq, s, &self.xn, q_dst, b).map_err(map_e)?;
                    self.qmm(wk, s, &self.xn, &self.k, b).map_err(map_e)?;
                    self.qmm(wv, s, &self.xn, &self.v, b).map_err(map_e)?;
                    if let Some(hgw) = head_gate_w {
                        self.qmm(hgw, s, &self.xn, &self.head_gate_buf, b).map_err(map_e)?;
                    }

                    if *gated_q {
                        self.hops.split_qg(s, &self.qg, &self.q, &self.q_gate, nh, hd, b).map_err(map_e)?;
                    }
                    if let Some(bq) = bq { self.ops.add_bias(s, &self.q, bq, nh * hd, b).map_err(map_e)?; }
                    if let Some(bk) = bk { self.ops.add_bias(s, &self.k, bk, *n_head_kv * hd, b).map_err(map_e)?; }
                    if let Some(bv) = bv { self.ops.add_bias(s, &self.v, bv, *n_head_kv * hd, b).map_err(map_e)?; }
                    if let Some(qn) = q_norm { self.ops.rms_norm_heads(s, &self.q, qn, hd, nh * b, eps).map_err(map_e)?; }
                    if let Some(kn) = k_norm { self.ops.rms_norm_heads(s, &self.k, kn, hd, *n_head_kv * b, eps).map_err(map_e)?; }

                    if *use_rope_factors {
                        self.hops.rope_ext(s, &self.q, &self.k, &self.rope_factors, nh, *n_head_kv, hd, nr, pos0 as u32, mode, *rope_base, true, b).map_err(map_e)?;
                    } else {
                        self.ops.rope(s, &self.q, nh, hd, nr, pos0 as u32, *rope_base, mode, b).map_err(map_e)?;
                        self.ops.rope(s, &self.k, *n_head_kv, hd, nr, pos0 as u32, *rope_base, mode, b).map_err(map_e)?;
                    }

                    let kv_off = *kv_layer_idx * self.layer_stride_bytes;
                    let kv_dim = *n_head_kv * hd;
                    self.ops.kv_write(s, &self.k, &self.v, &self.kv, kv_off, &self.pt_buf, pos0 as u32, kv_dim, b).map_err(map_e)?;
                    self.ops.attention_prefill(
                        s, &self.q, &self.kv, kv_off, &self.pt_buf, &self.attn,
                        nh, *n_head_kv, hd, pos0 as u32, *sliding_window, b,
                    ).map_err(map_e)?;

                    if *gated_q {
                        self.hops.mul_sigmoid(s, &self.attn, &self.q_gate, b * nh * hd).map_err(map_e)?;
                    }
                    if head_gate_w.is_some() {
                        self.hops.mul_head_sigmoid(s, &self.attn, &self.head_gate_buf, nh, hd, b).map_err(map_e)?;
                    }
                    if wo.ty.uses_q8_act() {
                        self.kernels.quant_q8(s, &self.attn, &self.xq, &self.xsc, (batch * wo.k as usize / 32) as u32).map_err(map_e)?;
                    }
                    self.qmm(wo, s, &self.attn, &self.o, b).map_err(map_e)?;
                }
                HipMixer::ShortConv { in_proj, out_proj, conv_w, state, l_cache } => {
                    if in_proj.ty.uses_q8_act() {
                        self.kernels.quant_q8(s, &self.xn, &self.xq, &self.xsc, (batch * self.cfg.n_embd / 32) as u32).map_err(map_e)?;
                    }
                    self.qmm(in_proj, s, &self.xn, &self.bcx, b).map_err(map_e)?;
                    self.hops.shortconv_step(s, &self.bcx, conv_w, state, &self.gate, ne, *l_cache, b).map_err(map_e)?;
                    if out_proj.ty.uses_q8_act() {
                        self.kernels.quant_q8(s, &self.gate, &self.xq, &self.xsc, (batch * self.cfg.n_embd / 32) as u32).map_err(map_e)?;
                    }
                    self.qmm(out_proj, s, &self.gate, &self.o, b).map_err(map_e)?;
                }
                HipMixer::DeltaNet {
                    wqkv, wgate, walpha, wbeta, wout, conv_w, dt_bias, ssm_a, ssm_norm,
                    conv_state, ssm_state, n_k_heads, n_v_heads, state_dim, k_conv, conv_ch,
                } => {
                    if wqkv.ty.uses_q8_act() || wgate.ty.uses_q8_act() || walpha.ty.uses_q8_act() || wbeta.ty.uses_q8_act() {
                        self.kernels.quant_q8(s, &self.xn, &self.xq, &self.xsc, (batch * self.cfg.n_embd / 32) as u32).map_err(map_e)?;
                    }
                    self.qmm(wqkv, s, &self.xn, &self.qkv_raw, b).map_err(map_e)?;
                    self.qmm(wgate, s, &self.xn, &self.z_buf, b).map_err(map_e)?;
                    self.qmm(walpha, s, &self.xn, &self.alpha_buf, b).map_err(map_e)?;
                    self.qmm(wbeta, s, &self.xn, &self.beta_buf, b).map_err(map_e)?;

                    self.hops.ssm_conv_silu(s, &self.qkv_raw, conv_w, conv_state, &self.qkv_conv, *conv_ch, *k_conv, b).map_err(map_e)?;
                    self.hops.deltanet_step(
                        s, &self.qkv_conv, &self.alpha_buf, &self.beta_buf,
                        dt_bias, ssm_a, ssm_state, &self.dn_out,
                        *n_k_heads, *n_v_heads, *state_dim, eps, b,
                    ).map_err(map_e)?;
                    self.hops.gated_rms_norm(s, &self.dn_out, ssm_norm, &self.z_buf, &self.dn_norm, *n_v_heads, *state_dim, eps, b).map_err(map_e)?;
                    if wout.ty.uses_q8_act() {
                        self.kernels.quant_q8(s, &self.dn_norm, &self.xq, &self.xsc, (batch * wout.k as usize / 32) as u32).map_err(map_e)?;
                    }
                    self.qmm(wout, s, &self.dn_norm, &self.o, b).map_err(map_e)?;
                }
            }

            if let Some(pan) = &l.post_attn_norm {
                self.kernels.rms_norm(s, &self.o, pan, &self.o, ne, b, eps).map_err(map_e)?;
            }
            self.ops.add_rms_norm(s, &self.x, &self.o, &l.ffn_norm, &self.xn, ne, b, eps).map_err(map_e)?;

            if l.gate.ty.uses_q8_act() || l.up.ty.uses_q8_act() {
                self.kernels.quant_q8(s, &self.xn, &self.xq, &self.xsc, (batch * self.cfg.n_embd / 32) as u32).map_err(map_e)?;
            }
            self.qmm(&l.gate, s, &self.xn, &self.gate, b).map_err(map_e)?;
            self.qmm(&l.up, s, &self.xn, &self.up, b).map_err(map_e)?;
            self.ops.swiglu(s, &self.gate, &self.up, &self.act, (batch * self.cfg.n_ff) as u32).map_err(map_e)?;

            if l.down.ty.uses_q8_act() {
                self.kernels.quant_q8(s, &self.act, &self.xq, &self.xsc, (batch * self.cfg.n_ff / 32) as u32).map_err(map_e)?;
            }
            self.qmm(&l.down, s, &self.act, &self.o, b).map_err(map_e)?;

            if let Some(pfn) = &l.post_ffw_norm {
                self.kernels.rms_norm(s, &self.o, pfn, &self.o, ne, b, eps).map_err(map_e)?;
            }
            let next_norm = if li + 1 < self.layers.len() {
                &self.layers[li + 1].attn_norm
            } else {
                &self.out_norm
            };
            self.ops.add_rms_norm(s, &self.x, &self.o, next_norm, &self.xn, ne, b, eps).map_err(map_e)?;
        }
        Ok(())
    }

    fn run_lm_head(&self, hidden_src: &HipBuffer) -> Result<(), EngineError> {
        let head = self.output.as_ref().unwrap_or(&self.embd);
        let map_e = |e: ocak_hip::HipError| EngineError::Backend(e.to_string());
        if head.ty.uses_q8_act() {
            self.kernels
                .quant_q8(&self.stream, hidden_src, &self.xq, &self.xsc, (self.cfg.n_embd / 32) as u32)
                .map_err(map_e)?;
        }
        self.qmv(head, &self.stream, hidden_src, &self.logits_buf).map_err(map_e)
    }
}

impl Executor for HipExec {
    fn name(&self) -> String {
        format!("HIP ({}, {})", self.ctx.info.name, self.ctx.info.gfx_arch)
    }

    fn reset_recurrent(&mut self) -> Result<(), EngineError> {
        let map_e = |e: ocak_hip::HipError| EngineError::Backend(e.to_string());
        for l in &self.layers {
            match &l.mixer {
                HipMixer::ShortConv { state, .. } => {
                    state.zero_async(&self.stream).map_err(map_e)?;
                }
                HipMixer::DeltaNet { conv_state, ssm_state, .. } => {
                    conv_state.zero_async(&self.stream).map_err(map_e)?;
                    ssm_state.zero_async(&self.stream).map_err(map_e)?;
                }
                HipMixer::FullAttn { .. } => {}
            }
        }
        self.stream.synchronize().map_err(map_e)
    }

    fn forward(
        &mut self,
        token: u32,
        pos: usize,
        page_table: &[u32],
        out: Out,
    ) -> Result<(), EngineError> {
        if token as usize >= self.cfg.n_vocab {
            return Err(EngineError::Context(format!(
                "token id {token} out of range (n_vocab={})",
                self.cfg.n_vocab
            )));
        }
        self.sync_page_table(page_table)?;
        let map_e = |e: ocak_hip::HipError| EngineError::Backend(e.to_string());
        let meta = [token, pos as u32];
        self.step_meta
            .upload(bytemuck::cast_slice(&meta))
            .map_err(map_e)?;
        if self.graph_logits.is_none() {
            let g = self
                .stream
                .capture_graph(|s| self.record_decode_step(s, false))
                .map_err(map_e)?;
            self.graph_logits = Some(g);
        }
        self.graph_logits
            .as_ref()
            .unwrap()
            .launch(&self.stream)
            .map_err(map_e)?;
        self.stream.synchronize().map_err(map_e)?;
        if out.hidden {
            self.xn
                .download_at(0, bytemuck::cast_slice_mut(&mut self.hidden))
                .map_err(map_e)?;
        }
        if out.logits {
            self.logits_buf
                .download(bytemuck::cast_slice_mut(&mut self.logits))
                .map_err(map_e)?;
        }
        Ok(())
    }

    fn forward_argmax(
        &mut self,
        token: u32,
        pos: usize,
        page_table: &[u32],
    ) -> Result<Option<u32>, EngineError> {
        if token as usize >= self.cfg.n_vocab {
            return Err(EngineError::Context(format!(
                "token id {token} out of range (n_vocab={})",
                self.cfg.n_vocab
            )));
        }
        self.sync_page_table(page_table)?;
        let map_e = |e: ocak_hip::HipError| EngineError::Backend(e.to_string());
        let meta = [token, pos as u32];
        self.step_meta
            .upload(bytemuck::cast_slice(&meta))
            .map_err(map_e)?;
        if self.graph_argmax.is_none() {
            let g = self
                .stream
                .capture_graph(|s| self.record_decode_step(s, true))
                .map_err(map_e)?;
            self.graph_argmax = Some(g);
        }
        self.graph_argmax
            .as_ref()
            .unwrap()
            .launch(&self.stream)
            .map_err(map_e)?;
        self.stream.synchronize().map_err(map_e)?;
        let mut idx = [0u32; 1];
        self.argmax_out
            .download(bytemuck::cast_slice_mut(&mut idx))
            .map_err(map_e)?;
        Ok(Some(idx[0]))
    }

    fn forward_batch(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        page_table: &[u32],
        out: Out,
    ) -> Result<(), EngineError> {
        if tokens.is_empty() {
            return Ok(());
        }
        if tokens.len() == 1 {
            return self.forward(tokens[0], pos0, page_table, out);
        }
        for &tok in tokens {
            if tok as usize >= self.cfg.n_vocab {
                return Err(EngineError::Context(format!(
                    "token id {tok} out of range (n_vocab={})",
                    self.cfg.n_vocab
                )));
            }
        }
        self.sync_page_table(page_table)?;
        let map_e = |e: ocak_hip::HipError| EngineError::Backend(e.to_string());

        let mut offset = 0usize;
        while offset < tokens.len() {
            let chunk = &tokens[offset..(offset + PREFILL_BATCH).min(tokens.len())];
            let is_last = offset + chunk.len() == tokens.len();
            self.toks_buf
                .upload_at(0, bytemuck::cast_slice(chunk))
                .map_err(map_e)?;
            self.ops
                .embed_tokens_batch(
                    &self.stream,
                    &self.embd.buf,
                    self.embd.ty,
                    self.iq_tbl.as_ref(),
                    &self.toks_buf,
                    &self.x,
                    self.cfg.n_embd as u32,
                    chunk.len() as u32,
                )
                .map_err(map_e)?;
            self.run_layers_prefill(pos0 + offset, chunk.len())?;

            if is_last && (out.logits || out.hidden) {
                let last_off = (chunk.len() - 1) * self.cfg.n_embd * 4;
                self.hidden_buf
                    .copy_from_at_async(0, &self.xn, last_off, self.cfg.n_embd * 4, &self.stream)
                    .map_err(map_e)?;
                if out.logits {
                    self.run_lm_head(&self.hidden_buf)?;
                }
            }
            self.stream.synchronize().map_err(map_e)?;

            if is_last && out.hidden {
                self.hidden_buf
                    .download(bytemuck::cast_slice_mut(&mut self.hidden))
                    .map_err(map_e)?;
            }
            if is_last && out.logits {
                self.logits_buf
                    .download(bytemuck::cast_slice_mut(&mut self.logits))
                    .map_err(map_e)?;
            }
            offset += chunk.len();
        }
        Ok(())
    }

    fn logits(&self) -> &[f32] {
        &self.logits
    }

    fn hidden(&self) -> &[f32] {
        &self.hidden
    }
}
