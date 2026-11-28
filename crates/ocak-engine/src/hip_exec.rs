//! Native AMD HIP executor (`HipExec`) for dense transformer models (`llama`, `qwen2`, `qwen3vl`).
//! Runs single-stream forward passes with paged FP16 KV cache, fused `add_rms_norm`,
//! int8-dot `sdot4` matvec/matmul, and 4-byte GPU argmax readback.
use crate::config::{LayerKind, ModelConfig};
use crate::exec::{EngineError, Executor, Out};
use ocak_gguf::{GgmlType, GgufFile, TensorInfo};
use ocak_hip::{HipBuffer, HipContext, HipDenseOps, HipKernels, HipStream, HipWType};
use std::sync::Arc;

const PREFILL_BATCH: usize = 256;

struct HipWeight {
    buf: HipBuffer,
    ty: HipWType,
    k: u32,
    m: u32,
}

struct HipLayer {
    attn_norm: HipBuffer,
    ffn_norm: HipBuffer,
    wq: HipWeight,
    wk: HipWeight,
    wv: HipWeight,
    wo: HipWeight,
    bq: Option<HipBuffer>,
    bk: Option<HipBuffer>,
    bv: Option<HipBuffer>,
    q_norm: Option<HipBuffer>,
    k_norm: Option<HipBuffer>,
    n_head_kv: u32,
    kv_layer_idx: usize,
    sliding_window: u32,
    rope_base: f32,
    gate: HipWeight,
    up: HipWeight,
    down: HipWeight,
}

pub struct HipExec {
    ctx: Arc<HipContext>,
    stream: HipStream,
    kernels: HipKernels,
    ops: HipDenseOps,
    cfg: ModelConfig,
    layers: Vec<HipLayer>,
    embd: HipWeight,
    out_norm: HipBuffer,
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
    q: HipBuffer,
    k: HipBuffer,
    v: HipBuffer,
    attn: HipBuffer,
    o: HipBuffer,
    gate: HipBuffer,
    up: HipBuffer,
    act: HipBuffer,
    // Quantized activation scratch (max(n_embd, n_ff, q_dim) * PREFILL_BATCH)
    xq: HipBuffer,
    xsc: HipBuffer,
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
        if cfg.is_hybrid || cfg.has_post_norms {
            return Err(EngineError::Unsupported(format!(
                "HIP backend currently supports dense FullAttn models (got arch '{}')",
                cfg.arch
            )));
        }
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
        let kernels = HipKernels::compile(&ctx).map_err(|e| EngineError::Backend(e.to_string()))?;
        let ops = HipDenseOps::compile(&ctx).map_err(|e| EngineError::Backend(e.to_string()))?;

        let embd = upload_weight(&ctx, &g, req_tensor(&g, "token_embd.weight")?)?;
        let out_norm = upload_f32(&ctx, &g, &cfg.output_norm_name)?;
        let output = if cfg.tied_output {
            None
        } else {
            Some(upload_weight(&ctx, &g, req_tensor(&g, "output.weight")?)?)
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for (i, &kind) in cfg.layer_kinds.iter().enumerate() {
            let LayerKind::FullAttn {
                n_head_kv,
                gated_q,
                qk_norm,
                kv_layer_idx,
                sliding_window,
                rope_base,
                use_rope_factors,
                head_gate,
            } = kind
            else {
                return Err(EngineError::Unsupported(
                    "HIP backend only supports FullAttn layers".into(),
                ));
            };
            if gated_q || use_rope_factors || head_gate {
                return Err(EngineError::Unsupported(
                    "HIP backend does not yet support gated_q / rope_factors / head_gate".into(),
                ));
            }
            let attn_norm = upload_f32(&ctx, &g, &format!("blk.{i}.attn_norm.weight"))?;
            let ffn_norm =
                upload_f32(&ctx, &g, &format!("blk.{i}.{}", cfg.ffn_norm_suffix))?;
            let wq = upload_weight(&ctx, &g, req_tensor(&g, &format!("blk.{i}.attn_q.weight"))?)?;
            let wk = upload_weight(&ctx, &g, req_tensor(&g, &format!("blk.{i}.attn_k.weight"))?)?;
            let wv = upload_weight(&ctx, &g, req_tensor(&g, &format!("blk.{i}.attn_v.weight"))?)?;
            let wo =
                upload_weight(&ctx, &g, req_tensor(&g, &format!("blk.{i}.attn_output.weight"))?)?;
            let bq = if cfg.qkv_bias && g.tensor(&format!("blk.{i}.attn_q.bias")).is_some() {
                Some(upload_f32(&ctx, &g, &format!("blk.{i}.attn_q.bias"))?)
            } else {
                None
            };
            let bk = if cfg.qkv_bias && g.tensor(&format!("blk.{i}.attn_k.bias")).is_some() {
                Some(upload_f32(&ctx, &g, &format!("blk.{i}.attn_k.bias"))?)
            } else {
                None
            };
            let bv = if cfg.qkv_bias && g.tensor(&format!("blk.{i}.attn_v.bias")).is_some() {
                Some(upload_f32(&ctx, &g, &format!("blk.{i}.attn_v.bias"))?)
            } else {
                None
            };
            let q_norm = if qk_norm {
                Some(upload_f32(&ctx, &g, &format!("blk.{i}.attn_q_norm.weight"))?)
            } else {
                None
            };
            let k_norm = if qk_norm {
                Some(upload_f32(&ctx, &g, &format!("blk.{i}.attn_k_norm.weight"))?)
            } else {
                None
            };
            let gate =
                upload_weight(&ctx, &g, req_tensor(&g, &format!("blk.{i}.ffn_gate.weight"))?)?;
            let up = upload_weight(&ctx, &g, req_tensor(&g, &format!("blk.{i}.ffn_up.weight"))?)?;
            let down =
                upload_weight(&ctx, &g, req_tensor(&g, &format!("blk.{i}.ffn_down.weight"))?)?;

            layers.push(HipLayer {
                attn_norm,
                ffn_norm,
                wq,
                wk,
                wv,
                wo,
                bq,
                bk,
                bv,
                q_norm,
                k_norm,
                n_head_kv: n_head_kv as u32,
                kv_layer_idx,
                sliding_window: sliding_window as u32,
                rope_base,
                gate,
                up,
                down,
            });
        }

        let mk_f32 = |elems: usize| -> Result<HipBuffer, EngineError> {
            ctx.alloc_buffer(elems.max(1) * 4)
                .map_err(|e| EngineError::Backend(e.to_string()))
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

        let max_k = cfg.n_embd.max(cfg.n_ff).max(q_dim);
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
            kernels,
            ops,
            layers,
            embd,
            out_norm,
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
            q: mk_f32(q_dim * PREFILL_BATCH)?,
            k: mk_f32(kv_dim * PREFILL_BATCH)?,
            v: mk_f32(kv_dim * PREFILL_BATCH)?,
            attn: mk_f32(q_dim * PREFILL_BATCH)?,
            o: mk_f32(cfg.n_embd * PREFILL_BATCH)?,
            gate: mk_f32(cfg.n_ff * PREFILL_BATCH)?,
            up: mk_f32(cfg.n_ff * PREFILL_BATCH)?,
            act: mk_f32(cfg.n_ff * PREFILL_BATCH)?,
            xq,
            xsc,
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

    /// Runs all transformer layers for `batch` tokens (`1..=PREFILL_BATCH`) starting at `pos0`.
    /// Expects `self.x` to hold `[batch, n_embd]` embeddings; leaves final normalized hidden
    /// states in `self.xn` (`[batch, n_embd]`).
    fn run_layers(&self, pos0: usize, batch: usize) -> Result<(), EngineError> {
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
            // Quantize xn once for wq/wk/wv if any of them uses Q8 activations
            if l.wq.ty.uses_q8_act() || l.wk.ty.uses_q8_act() || l.wv.ty.uses_q8_act() {
                self.kernels
                    .quant_q8(s, &self.xn, &self.xq, &self.xsc, (batch * self.cfg.n_embd / 32) as u32)
                    .map_err(map_e)?;
            }
            self.kernels
                .qmm(l.wq.ty, s, &l.wq.buf, &self.xn, &self.xq, &self.xsc, &self.q, l.wq.k, l.wq.m, b)
                .map_err(map_e)?;
            self.kernels
                .qmm(l.wk.ty, s, &l.wk.buf, &self.xn, &self.xq, &self.xsc, &self.k, l.wk.k, l.wk.m, b)
                .map_err(map_e)?;
            self.kernels
                .qmm(l.wv.ty, s, &l.wv.buf, &self.xn, &self.xq, &self.xsc, &self.v, l.wv.k, l.wv.m, b)
                .map_err(map_e)?;

            if let Some(bq) = &l.bq {
                self.ops.add_bias(s, &self.q, bq, l.wq.m, b).map_err(map_e)?;
            }
            if let Some(bk) = &l.bk {
                self.ops.add_bias(s, &self.k, bk, l.wk.m, b).map_err(map_e)?;
            }
            if let Some(bv) = &l.bv {
                self.ops.add_bias(s, &self.v, bv, l.wv.m, b).map_err(map_e)?;
            }
            if let Some(qn) = &l.q_norm {
                self.ops
                    .rms_norm_heads(s, &self.q, qn, hd, nh * b, eps)
                    .map_err(map_e)?;
            }
            if let Some(kn) = &l.k_norm {
                self.ops
                    .rms_norm_heads(s, &self.k, kn, hd, l.n_head_kv * b, eps)
                    .map_err(map_e)?;
            }

            self.ops
                .rope(s, &self.q, nh, hd, nr, pos0 as u32, l.rope_base, mode, b)
                .map_err(map_e)?;
            self.ops
                .rope(s, &self.k, l.n_head_kv, hd, nr, pos0 as u32, l.rope_base, mode, b)
                .map_err(map_e)?;

            let kv_off = l.kv_layer_idx * self.layer_stride_bytes;
            let kv_dim = l.n_head_kv * hd;
            self.ops
                .kv_write(s, &self.k, &self.v, &self.kv, kv_off, &self.pt_buf, pos0 as u32, kv_dim, b)
                .map_err(map_e)?;
            self.ops
                .attention(
                    s,
                    &self.q,
                    &self.kv,
                    kv_off,
                    &self.pt_buf,
                    &self.attn,
                    nh,
                    l.n_head_kv,
                    hd,
                    pos0 as u32,
                    l.sliding_window,
                    b,
                )
                .map_err(map_e)?;

            if l.wo.ty.uses_q8_act() {
                self.kernels
                    .quant_q8(s, &self.attn, &self.xq, &self.xsc, (batch * l.wo.k as usize / 32) as u32)
                    .map_err(map_e)?;
            }
            self.kernels
                .qmm(l.wo.ty, s, &l.wo.buf, &self.attn, &self.xq, &self.xsc, &self.o, l.wo.k, l.wo.m, b)
                .map_err(map_e)?;

            self.ops
                .add_rms_norm(s, &self.x, &self.o, &l.ffn_norm, &self.xn, ne, b, eps)
                .map_err(map_e)?;

            if l.gate.ty.uses_q8_act() || l.up.ty.uses_q8_act() {
                self.kernels
                    .quant_q8(s, &self.xn, &self.xq, &self.xsc, (batch * self.cfg.n_embd / 32) as u32)
                    .map_err(map_e)?;
            }
            self.kernels
                .qmm(l.gate.ty, s, &l.gate.buf, &self.xn, &self.xq, &self.xsc, &self.gate, l.gate.k, l.gate.m, b)
                .map_err(map_e)?;
            self.kernels
                .qmm(l.up.ty, s, &l.up.buf, &self.xn, &self.xq, &self.xsc, &self.up, l.up.k, l.up.m, b)
                .map_err(map_e)?;
            self.ops
                .swiglu(s, &self.gate, &self.up, &self.act, (batch * self.cfg.n_ff) as u32)
                .map_err(map_e)?;

            if l.down.ty.uses_q8_act() {
                self.kernels
                    .quant_q8(s, &self.act, &self.xq, &self.xsc, (batch * self.cfg.n_ff / 32) as u32)
                    .map_err(map_e)?;
            }
            self.kernels
                .qmm(l.down.ty, s, &l.down.buf, &self.act, &self.xq, &self.xsc, &self.o, l.down.k, l.down.m, b)
                .map_err(map_e)?;

            let next_norm = if li + 1 < self.layers.len() {
                &self.layers[li + 1].attn_norm
            } else {
                &self.out_norm
            };
            self.ops
                .add_rms_norm(s, &self.x, &self.o, next_norm, &self.xn, ne, b, eps)
                .map_err(map_e)?;
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
        self.kernels
            .qmv(
                head.ty,
                &self.stream,
                &head.buf,
                hidden_src,
                &self.xq,
                &self.xsc,
                &self.logits_buf,
                head.k,
                head.m,
                1,
            )
            .map_err(map_e)
    }
}

impl Executor for HipExec {
    fn name(&self) -> String {
        format!("HIP ({}, {})", self.ctx.info.name, self.ctx.info.gfx_arch)
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
        self.ops
            .embed_token(&self.stream, &self.embd.buf, self.embd.ty, &self.x, token, self.cfg.n_embd as u32)
            .map_err(map_e)?;
        self.run_layers(pos, 1)?;
        if out.logits {
            self.run_lm_head(&self.xn)?;
        }
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
        self.ops
            .embed_token(&self.stream, &self.embd.buf, self.embd.ty, &self.x, token, self.cfg.n_embd as u32)
            .map_err(map_e)?;
        self.run_layers(pos, 1)?;
        self.run_lm_head(&self.xn)?;
        self.ops
            .argmax(
                &self.stream,
                &self.logits_buf,
                &self.argmax_pv,
                &self.argmax_pi,
                &self.argmax_out,
                self.cfg.n_vocab as u32,
            )
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
                    &self.toks_buf,
                    &self.x,
                    self.cfg.n_embd as u32,
                    chunk.len() as u32,
                )
                .map_err(map_e)?;
            self.run_layers(pos0 + offset, chunk.len())?;

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
