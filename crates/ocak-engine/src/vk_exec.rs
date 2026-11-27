use crate::config::{LayerKind, ModelConfig};
use crate::exec::{EngineError, Executor, Out, SlotStep};
use ocak_gguf::{GgufFile, TensorInfo};
use ocak_vk::{Buffer, VkContext, WType};
use std::sync::Arc;

const PREFILL_BATCH: usize = 256;
/// Layers recorded per queue submit during batched prefill. Splitting a chunk across several
/// submits keeps each one far below amdgpu's ~2 s TDR timeout even on 64-layer 27B models.
const LAYERS_PER_SUBMIT: usize = 8;
/// Prefill chunk used when some layer weight format has no tiled matmul kernel.
/// Max tokens in one MTP verification (current token + drafts).
const MAX_VERIFY: usize = 4;
const SLOW_PREFILL_CHUNK: usize = 16;

struct WeightBuf {
    buf: Arc<Buffer>,
    ty: WType,
    k: u32,
    m: u32,
}

enum VkMixer {
    FullAttn {
        wq: WeightBuf,
        wk: WeightBuf,
        wv: WeightBuf,
        wo: WeightBuf,
        bq: Option<Buffer>,
        bk: Option<Buffer>,
        bv: Option<Buffer>,
        q_norm: Option<Buffer>,
        k_norm: Option<Buffer>,
        gated_q: bool,
        n_head_kv: u32,
        kv_layer_idx: usize,
        sliding_window: u32,
        rope_base: f32,
        use_rope_factors: bool,
        head_gate_w: Option<WeightBuf>,
    },
    ShortConv {
        in_proj: WeightBuf,
        out_proj: WeightBuf,
        conv_w: Buffer,
        state: Buffer,
        l_cache: u32,
    },
    DeltaNet {
        wqkv: WeightBuf,
        wgate: WeightBuf,
        walpha: WeightBuf,
        wbeta: WeightBuf,
        wout: WeightBuf,
        conv_w: Buffer,
        dt_bias: Buffer,
        ssm_a: Buffer,
        ssm_norm: Buffer,
        conv_state: Buffer,
        ssm_state: Buffer,
        n_k_heads: u32,
        n_v_heads: u32,
        state_dim: u32,
        k_conv: u32,
        conv_ch: u32,
    },
}

struct VkLayer {
    attn_norm: Buffer,
    ffn_norm: Buffer,
    post_attn_norm: Option<Buffer>,
    post_ffw_norm: Option<Buffer>,
    mixer: VkMixer,
    gate: WeightBuf,
    up: WeightBuf,
    down: WeightBuf,
}

pub struct VkExec {
    ctx: Arc<VkContext>,
    cfg: ModelConfig,
    /// Largest prefill chunk: PREFILL_BATCH when every layer matrix has a tiled matmul,
    /// otherwise SLOW_PREFILL_CHUNK (untiled formats run one matvec per token).
    prefill_chunk: usize,
    layers: Vec<VkLayer>,
    embd: WeightBuf,
    out_norm: Buffer,
    rope_factors: Buffer,
    head_gate_buf: Buffer,
    output: Option<WeightBuf>,
    kv: Buffer,
    layer_stride: u64,
    pt_buf: Buffer,
    /// Contents last uploaded to pt_buf: the page table changes once per 16-token page, so the
    /// per-step synchronous upload (a submit + fence wait) is skipped when it is unchanged.
    pt_last: Vec<u32>,
    max_pages: usize,
    x: Buffer,
    xn: Buffer,
    qg: Buffer,
    q: Buffer,
    q_gate: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    o: Buffer,
    bcx: Buffer,
    qkv_raw: Buffer,
    qkv_conv: Buffer,
    z_buf: Buffer,
    alpha_buf: Buffer,
    beta_buf: Buffer,
    dn_out: Buffer,
    dn_norm: Buffer,
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    hidden_buf: Buffer,
    logits_buf: Buffer,
    hidden_buf2: Buffer,
    logits_buf2: Buffer,
    multi_logits_buf: Option<(usize, Buffer)>,
    argmax_buf: Buffer,
    rec_snaps: Option<Vec<(Buffer, Option<Buffer>)>>,
    logits: Vec<f32>,
    hidden: Vec<f32>,
}

fn upload_f32(ctx: &Arc<VkContext>, g: &GgufFile, name: &str) -> Result<Buffer, EngineError> {
    let ti = g.tensor(name).ok_or_else(|| EngineError::Unsupported(format!("missing {name}")))?;
    let data = g.tensor_to_f32(ti);
    let buf = ctx.create_device_buffer((data.len() * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?;
    ctx.upload(&buf, 0, bytemuck::cast_slice(&data)).map_err(|e| EngineError::Backend(e.to_string()))?;
    Ok(buf)
}

fn upload_weight(ctx: &Arc<VkContext>, g: &GgufFile, ti: &TensorInfo) -> Result<WeightBuf, EngineError> {
    let ty = WType::from_ggml(ti.ty).map_err(|e| EngineError::Unsupported(e.to_string()))?;
    let raw = g.tensor_data(ti);
    let buf = ctx.create_device_buffer(raw.len() as u64).map_err(|e| EngineError::Backend(e.to_string()))?;
    ctx.upload_async(&buf, 0, raw).map_err(|e| EngineError::Backend(e.to_string()))?;
    Ok(WeightBuf { buf: Arc::new(buf), ty, k: ti.n_cols() as u32, m: ti.n_rows() as u32 })
}

fn req_tensor<'a>(g: &'a GgufFile, name: &str) -> Result<&'a TensorInfo, EngineError> {
    g.tensor(name).ok_or_else(|| EngineError::Unsupported(format!("missing {name}")))
}

impl VkExec {
    pub fn new(g: Arc<GgufFile>, cfg: ModelConfig, n_pages: usize) -> Result<Self, EngineError> {
        if cfg.head_dim > 256 {
            return Err(EngineError::Unsupported(format!(
                "attention.comp shader supports head_dim <= 256 (got {})",
                cfg.head_dim
            )));
        }
        g.prefetch_background();
        let ctx = VkContext::new().map_err(|e| EngineError::Backend(e.to_string()))?;
        let mk = |elems: usize| -> Result<Buffer, EngineError> {
            ctx.create_device_buffer((elems.max(4) * 4) as u64)
                .map_err(|e| EngineError::Backend(e.to_string()))
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let mut max_conv_ch = 4usize;
        let mut max_v_dim = 4usize;
        let mut max_v_heads = 4usize;

        for (i, &kind) in cfg.layer_kinds.iter().enumerate() {
            let p = |s: &str| format!("blk.{i}.{s}");
            let mixer = match kind {
                LayerKind::FullAttn { n_head_kv, gated_q, qk_norm, kv_layer_idx, sliding_window, rope_base, use_rope_factors, head_gate } => {
                    let b = |s: &str| -> Result<Option<Buffer>, EngineError> {
                        if cfg.qkv_bias { Ok(Some(upload_f32(&ctx, &g, &p(s))?)) } else { Ok(None) }
                    };
                    VkMixer::FullAttn {
                        wq: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_q.weight"))?)?,
                        wk: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_k.weight"))?)?,
                        wv: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_v.weight"))?)?,
                        wo: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_output.weight"))?)?,
                        bq: b("attn_q.bias")?,
                        bk: b("attn_k.bias")?,
                        bv: b("attn_v.bias")?,
                        q_norm: if qk_norm { Some(upload_f32(&ctx, &g, &p("attn_q_norm.weight"))?) } else { None },
                        k_norm: if qk_norm { Some(upload_f32(&ctx, &g, &p("attn_k_norm.weight"))?) } else { None },
                        gated_q,
                        n_head_kv: n_head_kv as u32,
                        kv_layer_idx,
                        sliding_window: sliding_window as u32,
                        rope_base,
                        use_rope_factors,
                        head_gate_w: if head_gate { Some(upload_weight(&ctx, &g, req_tensor(&g, &p("attn_gate.weight"))?)?) } else { None },
                    }
                }
                LayerKind::ShortConv { l_cache } => {
                    let state = mk(cfg.n_embd * (l_cache - 1))?;
                    ctx.zero_buffer(&state).map_err(|e| EngineError::Backend(e.to_string()))?;
                    VkMixer::ShortConv {
                        in_proj: upload_weight(&ctx, &g, req_tensor(&g, &p("shortconv.in_proj.weight"))?)?,
                        out_proj: upload_weight(&ctx, &g, req_tensor(&g, &p("shortconv.out_proj.weight"))?)?,
                        conv_w: upload_f32(&ctx, &g, &p("shortconv.conv.weight"))?,
                        state,
                        l_cache: l_cache as u32,
                    }
                }
                LayerKind::DeltaNet { n_k_heads, n_v_heads, state_dim, k_conv } => {
                    let conv_ch = (2 * n_k_heads + n_v_heads) * state_dim;
                    let v_dim = n_v_heads * state_dim;
                    max_conv_ch = max_conv_ch.max(conv_ch);
                    max_v_dim = max_v_dim.max(v_dim);
                    max_v_heads = max_v_heads.max(n_v_heads);
                    let conv_state = mk(conv_ch * (k_conv - 1))?;
                    let ssm_state = mk(n_v_heads * state_dim * state_dim)?;
                    ctx.zero_buffer(&conv_state).map_err(|e| EngineError::Backend(e.to_string()))?;
                    ctx.zero_buffer(&ssm_state).map_err(|e| EngineError::Backend(e.to_string()))?;
                    VkMixer::DeltaNet {
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
            layers.push(VkLayer {
                attn_norm: upload_f32(&ctx, &g, &p("attn_norm.weight"))?,
                ffn_norm: upload_f32(&ctx, &g, &p(&cfg.ffn_norm_suffix))?,
                post_attn_norm: if cfg.has_post_norms { Some(upload_f32(&ctx, &g, &p("post_attention_norm.weight"))?) } else { None },
                post_ffw_norm: if cfg.has_post_norms { Some(upload_f32(&ctx, &g, &p("post_ffw_norm.weight"))?) } else { None },
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
            mk(cfg.head_dim / 2)?
        };
        let head_gate_buf = mk(PREFILL_BATCH * cfg.n_head)?;
        let output = if cfg.tied_output { None } else { Some(upload_weight(&ctx, &g, req_tensor(&g, "output.weight")?)?) };

        // FP16 packed KV cache: 2 f16s per u32 (half the VRAM & attention bandwidth!)
        let layer_stride = (n_pages * 2 * 16 * (cfg.kv_dim() / 2)) as u64;
        let qd = cfg.n_head * cfg.head_dim;
        let all_tiled = g.tensors.iter().filter(|t| t.name.starts_with("blk.") && t.n_dims >= 2).all(|t| {
            WType::from_ggml(t.ty).map(ocak_vk::has_tiled_qmm).unwrap_or(false)
        });
        let prefill_chunk = if all_tiled { PREFILL_BATCH } else { SLOW_PREFILL_CHUNK };
        Ok(Self {
            prefill_chunk,
            kv: mk(layer_stride as usize * cfg.n_kv_layers)?,
            layer_stride,
            pt_buf: mk((n_pages * 4).max(64))?,
            pt_last: Vec::new(),
            max_pages: (n_pages * 4).max(64),
            x: mk(PREFILL_BATCH * cfg.n_embd)?,
            xn: mk(PREFILL_BATCH * cfg.n_embd)?,
            qg: mk(PREFILL_BATCH * 2 * qd)?,
            q: mk(PREFILL_BATCH * qd)?,
            q_gate: mk(PREFILL_BATCH * qd)?,
            k: mk(PREFILL_BATCH * cfg.kv_dim())?,
            v: mk(PREFILL_BATCH * cfg.kv_dim())?,
            attn: mk(PREFILL_BATCH * qd)?,
            o: mk(PREFILL_BATCH * cfg.n_embd)?,
            bcx: mk(PREFILL_BATCH * 3 * cfg.n_embd)?,
            qkv_raw: mk(PREFILL_BATCH * max_conv_ch)?,
            qkv_conv: mk(PREFILL_BATCH * max_conv_ch)?,
            z_buf: mk(PREFILL_BATCH * max_v_dim)?,
            alpha_buf: mk(PREFILL_BATCH * max_v_heads)?,
            beta_buf: mk(PREFILL_BATCH * max_v_heads)?,
            dn_out: mk(PREFILL_BATCH * max_v_dim)?,
            dn_norm: mk(PREFILL_BATCH * max_v_dim)?,
            gate: mk(PREFILL_BATCH * cfg.n_ff)?,
            up: mk(PREFILL_BATCH * cfg.n_ff)?,
            act: mk(PREFILL_BATCH * cfg.n_ff)?,
            hidden_buf: mk(cfg.n_embd)?,
            logits_buf: mk(cfg.n_vocab)?,
            hidden_buf2: mk(MAX_VERIFY * cfg.n_embd)?,
            logits_buf2: mk(MAX_VERIFY * cfg.n_vocab)?,
            multi_logits_buf: None,
            argmax_buf: mk(4)?,
            rec_snaps: None,
            logits: vec![0.0; cfg.n_vocab],
            hidden: vec![0.0; cfg.n_embd],
            layers,
            embd,
            out_norm,
            rope_factors,
            head_gate_buf,
            output,
            cfg,
            ctx,
        })
    }
}

impl Executor for VkExec {
    fn name(&self) -> String { format!("vulkan ({})", self.ctx.device_name()) }

    fn reset_recurrent(&mut self) -> Result<(), EngineError> {
        for l in &self.layers {
            match &l.mixer {
                VkMixer::ShortConv { state, .. } => {
                    self.ctx.zero_buffer(state).map_err(|e| EngineError::Backend(e.to_string()))?;
                }
                VkMixer::DeltaNet { conv_state, ssm_state, .. } => {
                    self.ctx.zero_buffer(conv_state).map_err(|e| EngineError::Backend(e.to_string()))?;
                    self.ctx.zero_buffer(ssm_state).map_err(|e| EngineError::Backend(e.to_string()))?;
                }
                VkMixer::FullAttn { .. } => {}
            }
        }
        Ok(())
    }

    fn forward(&mut self, token: u32, pos: usize, page_table: &[u32], out: Out) -> Result<(), EngineError> {
        self.forward_inner(Some(token), None, pos, page_table, out, false).map(|_| ())
    }

    fn forward_argmax(&mut self, token: u32, pos: usize, page_table: &[u32]) -> Result<Option<u32>, EngineError> {
        self.forward_inner(Some(token), None, pos, page_table, Out { logits: true, hidden: false }, true)
    }

    fn forward_batch(&mut self, tokens: &[u32], pos0: usize, page_table: &[u32], out: Out) -> Result<(), EngineError> {
        if tokens.len() <= 1 {
            if let Some(&tok) = tokens.first() {
                return self.forward(tok, pos0, page_table, out);
            }
            return Ok(());
        }
        if page_table.len() > self.max_pages {
            return Err(EngineError::Context("page table exceeds GPU capacity".into()));
        }
        self.upload_pt(page_table)?;

        // Keep each single vkQueueSubmit well below Linux amdgpu's 2s TDR timeout on 27B+ models
        let chunk_max = std::env::var("OCAK_PREFILL_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(self.prefill_chunk).clamp(1, PREFILL_BATCH);
        let mut offset = 0usize;
        while offset < tokens.len() {
            let chunk = &tokens[offset..(offset + chunk_max).min(tokens.len())];
            let is_last_chunk = offset + chunk.len() == tokens.len();
            let chunk_out = Out { logits: is_last_chunk && out.logits, hidden: is_last_chunk && out.hidden };
            self.forward_batch_chunk(chunk, pos0 + offset, chunk_out, None)?;
            offset += chunk.len();
        }
        Ok(())
    }

    fn forward_batch_collect_hiddens(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        page_table: &[u32],
        out: Out,
    ) -> Result<Vec<Vec<f32>>, EngineError> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        if tokens.len() == 1 {
            self.forward(tokens[0], pos0, page_table, Out { logits: out.logits, hidden: true })?;
            return Ok(vec![self.hidden.clone()]);
        }
        if page_table.len() > self.max_pages {
            return Err(EngineError::Context("page table exceeds GPU capacity".into()));
        }
        self.upload_pt(page_table)?;

        let chunk_max = std::env::var("OCAK_PREFILL_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(self.prefill_chunk).clamp(1, PREFILL_BATCH);
        let mut hiddens = Vec::with_capacity(tokens.len());
        let mut offset = 0usize;
        while offset < tokens.len() {
            let chunk = &tokens[offset..(offset + chunk_max).min(tokens.len())];
            let is_last_chunk = offset + chunk.len() == tokens.len();
            let chunk_out = Out { logits: is_last_chunk && out.logits, hidden: is_last_chunk && out.hidden };
            self.forward_batch_chunk(chunk, pos0 + offset, chunk_out, Some(&mut hiddens))?;
            offset += chunk.len();
        }
        Ok(hiddens)
    }

    fn forward_verify2(
        &mut self,
        tokens: [u32; 2],
        pos0: usize,
        page_table: &[u32],
    ) -> Result<Option<([Vec<f32>; 2], [Vec<f32>; 2])>, EngineError> {
        if page_table.len() > self.max_pages {
            return Err(EngineError::Context("page table exceeds GPU capacity".into()));
        }
        self.ensure_rec_snaps()?;
        self.upload_pt(page_table)?;
        let c = &self.cfg;
        let ne = c.n_embd as u32;
        let mut rec = self.ctx.begin();
        rec.set_defer_barriers(true);
        self.record_batch_layers(&mut rec, &tokens, pos0, true, 0..self.layers.len());
        // Final norm + lm_head for BOTH tokens using a single 2-token qmm!
        rec.rms_norm(&self.x, 0, &self.out_norm, 0, &self.hidden_buf2, 0, ne, c.eps);
        rec.rms_norm(&self.x, ne as u64, &self.out_norm, 0, &self.hidden_buf2, ne as u64, ne, c.eps);
        rec.barrier();
        let w = self.output.as_ref().unwrap_or(&self.embd);
        rec.qmm(w.ty, &w.buf, 0, &self.hidden_buf2, 0, &self.logits_buf2, 0, w.k, w.m, 2);
        rec.barrier();
        rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;

        let mut h_both = vec![0f32; 2 * c.n_embd];
        let mut l_both = vec![0f32; 2 * c.n_vocab];
        self.ctx.download(&self.hidden_buf2, 0, bytemuck::cast_slice_mut(&mut h_both)).map_err(|e| EngineError::Backend(e.to_string()))?;
        self.ctx.download(&self.logits_buf2, 0, bytemuck::cast_slice_mut(&mut l_both)).map_err(|e| EngineError::Backend(e.to_string()))?;
        let h0 = h_both[..c.n_embd].to_vec();
        let h1 = h_both[c.n_embd..].to_vec();
        let l0 = l_both[..c.n_vocab].to_vec();
        let l1 = l_both[c.n_vocab..].to_vec();
        self.hidden.copy_from_slice(&h1);
        self.logits.copy_from_slice(&l1);
        Ok(Some(([l0, l1], [h0, h1])))
    }

    fn forward_verify2_argmax(
        &mut self,
        tokens: [u32; 2],
        pos0: usize,
        page_table: &[u32],
    ) -> Result<Option<([u32; 2], [Vec<f32>; 2])>, EngineError> {
        let Some((idx, mut hs)) = self.forward_verify_argmax(&tokens, pos0, page_table)? else { return Ok(None) };
        let h1 = hs.pop().unwrap();
        let h0 = hs.pop().unwrap();
        Ok(Some(([idx[0], idx[1]], [h0, h1])))
    }

    fn forward_verify_logits(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        page_table: &[u32],
    ) -> Result<Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>, EngineError> {
        let n = tokens.len();
        if !(2..=MAX_VERIFY).contains(&n) {
            return Ok(None);
        }
        if page_table.len() > self.max_pages {
            return Err(EngineError::Context("page table exceeds GPU capacity".into()));
        }
        self.ensure_rec_snaps()?;
        self.upload_pt(page_table)?;
        let c = &self.cfg;
        let ne = c.n_embd as u32;
        let mut rec = self.ctx.begin();
        rec.set_defer_barriers(true);
        self.record_batch_layers(&mut rec, tokens, pos0, true, 0..self.layers.len());
        for t in 0..n as u64 {
            rec.rms_norm(&self.x, t * ne as u64, &self.out_norm, 0, &self.hidden_buf2, t * ne as u64, ne, c.eps);
        }
        rec.barrier();
        let w = self.output.as_ref().unwrap_or(&self.embd);
        rec.qmm(w.ty, &w.buf, 0, &self.hidden_buf2, 0, &self.logits_buf2, 0, w.k, w.m, n as u32);
        rec.barrier();
        let mut l_all = vec![0f32; n * c.n_vocab];
        let mut h_all = vec![0f32; n * c.n_embd];
        rec.finish_and_read(&mut [(&self.logits_buf2, 0, bytemuck::cast_slice_mut(&mut l_all)), (&self.hidden_buf2, 0, bytemuck::cast_slice_mut(&mut h_all))]).map_err(|e| EngineError::Backend(e.to_string()))?;
        let ls: Vec<Vec<f32>> = l_all.chunks(c.n_vocab).map(|l| l.to_vec()).collect();
        let hs: Vec<Vec<f32>> = h_all.chunks(c.n_embd).map(|h| h.to_vec()).collect();
        self.hidden.copy_from_slice(&hs[n - 1]);
        self.logits.copy_from_slice(&ls[n - 1]);
        Ok(Some((ls, hs)))
    }

    fn forward_verify_argmax(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        page_table: &[u32],
    ) -> Result<Option<(Vec<u32>, Vec<Vec<f32>>)>, EngineError> {
        let n = tokens.len();
        if !(2..=MAX_VERIFY).contains(&n) {
            return Ok(None);
        }
        if page_table.len() > self.max_pages {
            return Err(EngineError::Context("page table exceeds GPU capacity".into()));
        }
        self.ensure_rec_snaps()?;
        self.upload_pt(page_table)?;
        let c = &self.cfg;
        let ne = c.n_embd as u32;
        let mut rec = self.ctx.begin();
        rec.set_defer_barriers(true);
        self.record_batch_layers(&mut rec, tokens, pos0, true, 0..self.layers.len());
        for t in 0..n as u64 {
            rec.rms_norm(&self.x, t * ne as u64, &self.out_norm, 0, &self.hidden_buf2, t * ne as u64, ne, c.eps);
        }
        rec.barrier();
        let w = self.output.as_ref().unwrap_or(&self.embd);
        rec.qmm(w.ty, &w.buf, 0, &self.hidden_buf2, 0, &self.logits_buf2, 0, w.k, w.m, n as u32);
        rec.barrier();
        // Greedy: argmax of each row on the device; only n indices and the hidden states cross
        // PCIe instead of n x n_vocab logits.
        for t in 0..n as u32 {
            rec.argmax_at(&self.logits_buf2, t as u64 * c.n_vocab as u64, &self.argmax_buf, t, c.n_vocab as u32);
        }
        rec.barrier();
        let mut idx = vec![0u32; n];
        let mut h_all = vec![0f32; n * c.n_embd];
        rec.finish_and_read(&mut [(&self.argmax_buf, 0, bytemuck::cast_slice_mut(&mut idx)), (&self.hidden_buf2, 0, bytemuck::cast_slice_mut(&mut h_all))]).map_err(|e| EngineError::Backend(e.to_string()))?;
        let hs: Vec<Vec<f32>> = h_all.chunks(c.n_embd).map(|h| h.to_vec()).collect();
        self.hidden.copy_from_slice(&hs[n - 1]);
        Ok(Some((idx, hs)))
    }

    fn snapshot_recurrent(&mut self) -> Result<(), EngineError> {
        self.ensure_rec_snaps()?;
        let snaps = self.rec_snaps.as_ref().unwrap();
        if snaps.is_empty() {
            return Ok(());
        }
        let mut rec = self.ctx.begin();
        let mut idx = 0usize;
        for l in &self.layers {
            match &l.mixer {
                VkMixer::ShortConv { state, .. } => {
                    rec.copy_buf_bytes(state, &snaps[idx].0, state.size());
                    idx += 1;
                }
                VkMixer::DeltaNet { conv_state, ssm_state, .. } => {
                    rec.copy_buf_bytes(conv_state, &snaps[idx].0, conv_state.size());
                    rec.copy_buf_bytes(ssm_state, snaps[idx].1.as_ref().unwrap(), ssm_state.size());
                    idx += 1;
                }
                VkMixer::FullAttn { .. } => {}
            }
        }
        rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;
        Ok(())
    }

    fn restore_recurrent(&mut self) -> Result<(), EngineError> {
        self.restore_recurrent_to(0)
    }

    fn restore_recurrent_to(&mut self, k: usize) -> Result<(), EngineError> {
        let Some(snaps) = self.rec_snaps.as_ref() else { return Ok(()); };
        if snaps.is_empty() {
            return Ok(());
        }
        let mut rec = self.ctx.begin();
        let mut idx = 0usize;
        for l in &self.layers {
            match &l.mixer {
                VkMixer::ShortConv { state, .. } => {
                    if k != 0 {
                        return Err(EngineError::Unsupported("short-conv rollback beyond token 0".into()));
                    }
                    rec.copy_buf_bytes(&snaps[idx].0, state, state.size());
                    idx += 1;
                }
                VkMixer::DeltaNet { conv_state, ssm_state, .. } => {
                    let (cs, ss) = (conv_state.size(), ssm_state.size());
                    rec.copy_buf_region(&snaps[idx].0, k as u64 * cs, conv_state, 0, cs);
                    rec.copy_buf_region(snaps[idx].1.as_ref().unwrap(), k as u64 * ss, ssm_state, 0, ss);
                    idx += 1;
                }
                VkMixer::FullAttn { .. } => {}
            }
        }
        rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;
        Ok(())
    }

    fn mtp_gpu_shared(&self) -> Option<(Arc<VkContext>, Arc<Buffer>, WType, u32, u32)> {
        let w = self.output.as_ref().unwrap_or(&self.embd);
        Some((Arc::clone(&self.ctx), Arc::clone(&w.buf), w.ty, w.k, w.m))
    }

    fn forward_multi_slot(&mut self, slots: &[SlotStep<'_>]) -> Result<Option<Vec<Vec<f32>>>, EngineError> {
        if self.cfg.is_hybrid || slots.is_empty() || slots.len() > PREFILL_BATCH {
            return Ok(None);
        }
        // Pack all slots' page tables contiguously into self.pt_buf
        let mut pt_packed = Vec::new();
        let mut pt_offs = Vec::with_capacity(slots.len());
        for s in slots {
            pt_offs.push(pt_packed.len() as u64);
            pt_packed.extend_from_slice(s.page_table);
        }
        if pt_packed.len() > self.max_pages {
            return Err(EngineError::Context("multi-slot page tables exceed GPU capacity".into()));
        }
        self.upload_pt(&pt_packed)?;

        if slots.len() > 2 {
            let need_alloc = self.multi_logits_buf.as_ref().is_none_or(|(cap, _)| *cap < slots.len());
            if need_alloc {
                let cap = slots.len().max(8);
                let buf = self.ctx.create_device_buffer((cap * self.cfg.n_vocab * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?;
                self.multi_logits_buf = Some((cap, buf));
            }
        }

        let c = &self.cfg;
        let batch = slots.len() as u32;
        let ne = c.n_embd as u32;
        let qd = (c.n_head * c.head_dim) as u32;

        let mut rec = self.ctx.begin();
        rec.set_defer_barriers(true);
        for (b, s) in slots.iter().enumerate() {
            rec.embed(self.embd.ty, &self.embd.buf, s.token, &self.x, (b * c.n_embd) as u64, ne);
        }
        rec.barrier();

        if !self.layers.is_empty() {
            for b in 0..slots.len() {
                rec.rms_norm(&self.x, (b * c.n_embd) as u64, &self.layers[0].attn_norm, 0, &self.xn, (b * c.n_embd) as u64, ne, c.eps);
            }
            rec.barrier();
        }

        for (li, l) in self.layers.iter().enumerate() {
            if let VkMixer::FullAttn { wq, wk, wv, wo, bq, bk, bv, q_norm, k_norm, gated_q, n_head_kv, kv_layer_idx, sliding_window, rope_base, use_rope_factors, head_gate_w } = &l.mixer {
                let kv_dim = (*n_head_kv as usize) * c.head_dim;
                if *gated_q {
                    rec.qmm(wq.ty, &wq.buf, 0, &self.xn, 0, &self.qg, 0, wq.k, wq.m, batch);
                } else {
                    rec.qmm(wq.ty, &wq.buf, 0, &self.xn, 0, &self.q, 0, wq.k, wq.m, batch);
                }
                rec.qmm(wk.ty, &wk.buf, 0, &self.xn, 0, &self.k, 0, wk.k, wk.m, batch);
                rec.qmm(wv.ty, &wv.buf, 0, &self.xn, 0, &self.v, 0, wv.k, wv.m, batch);
                if let Some(hg) = head_gate_w {
                    rec.qmm(hg.ty, &hg.buf, 0, &self.xn, 0, &self.head_gate_buf, 0, hg.k, hg.m, batch);
                }
                rec.barrier();
                if *gated_q {
                    rec.split_qg_batch(&self.qg, &self.q, &self.q_gate, c.n_head as u32, c.head_dim as u32, batch);
                    rec.barrier();
                }
                if bq.is_some() || bk.is_some() || bv.is_some() {
                    for b in 0..slots.len() {
                        if let Some(bias) = bq { rec.add(&self.q, (b * qd as usize) as u64, bias, 0, qd); }
                        if let Some(bias) = bk { rec.add(&self.k, (b * kv_dim) as u64, bias, 0, wk.m); }
                        if let Some(bias) = bv { rec.add(&self.v, (b * kv_dim) as u64, bias, 0, wv.m); }
                    }
                    rec.barrier();
                }
                if q_norm.is_some() || k_norm.is_some() {
                    for b in 0..slots.len() {
                        if let Some(qn) = q_norm { rec.rms_norm_heads(&self.q, (b * qd as usize) as u64, qn, c.n_head as u32, c.head_dim as u32, c.eps); }
                        if let Some(kn) = k_norm { rec.rms_norm_heads(&self.k, (b * kv_dim) as u64, kn, *n_head_kv, c.head_dim as u32, c.eps); }
                    }
                    rec.barrier();
                }
                for (b, s) in slots.iter().enumerate() {
                    rec.rope_ext(&self.q, (b * qd as usize) as u64, &self.k, (b * kv_dim) as u64, &self.rope_factors, *use_rope_factors, c.n_head as u32, *n_head_kv, c.head_dim as u32, c.n_rot as u32, s.pos as u32, *rope_base, c.rope_mode);
                }
                rec.barrier();
                let kv_off = *kv_layer_idx as u64 * self.layer_stride;
                for (b, s) in slots.iter().enumerate() {
                    rec.kv_write(&self.k, (b * kv_dim) as u64, &self.v, (b * kv_dim) as u64, &self.kv, kv_off, &self.pt_buf, pt_offs[b], s.pos as u32, kv_dim as u32);
                }
                rec.barrier();
                for (b, s) in slots.iter().enumerate() {
                    rec.attention_swa(&self.q, (b * qd as usize) as u64, &self.kv, kv_off, &self.pt_buf, pt_offs[b], &self.attn, (b * qd as usize) as u64, c.n_head as u32, *n_head_kv, c.head_dim as u32, s.pos as u32, *sliding_window);
                }
                rec.barrier();
                if *gated_q {
                    rec.mul_sigmoid(&self.attn, &self.q_gate, batch * qd);
                    rec.barrier();
                }
                if head_gate_w.is_some() {
                    rec.mul_head_sigmoid_batch(&self.attn, &self.head_gate_buf, c.n_head as u32, c.head_dim as u32, batch);
                    rec.barrier();
                }
                rec.qmm(wo.ty, &wo.buf, 0, &self.attn, 0, &self.o, 0, wo.k, wo.m, batch);
                rec.barrier();
            }
            if let Some(pan) = &l.post_attn_norm {
                for b in 0..slots.len() {
                    rec.rms_norm(&self.o, (b * c.n_embd) as u64, pan, 0, &self.o, (b * c.n_embd) as u64, ne, c.eps);
                }
                rec.barrier();
            }
            for b in 0..slots.len() {
                rec.add_rms_norm(&self.x, (b * c.n_embd) as u64, &self.o, (b * c.n_embd) as u64, &l.ffn_norm, 0, &self.xn, (b * c.n_embd) as u64, ne, c.eps);
            }
            rec.barrier();
            rec.qmm(l.gate.ty, &l.gate.buf, 0, &self.xn, 0, &self.gate, 0, l.gate.k, l.gate.m, batch);
            rec.qmm(l.up.ty, &l.up.buf, 0, &self.xn, 0, &self.up, 0, l.up.k, l.up.m, batch);
            rec.barrier();
            rec.swiglu(&self.gate, 0, &self.up, 0, &self.act, 0, batch * c.n_ff as u32);
            rec.barrier();
            rec.qmm(l.down.ty, &l.down.buf, 0, &self.act, 0, &self.o, 0, l.down.k, l.down.m, batch);
            rec.barrier();
            if let Some(pfn) = &l.post_ffw_norm {
                for b in 0..slots.len() {
                    rec.rms_norm(&self.o, (b * c.n_embd) as u64, pfn, 0, &self.o, (b * c.n_embd) as u64, ne, c.eps);
                }
                rec.barrier();
            }
            if li + 1 < self.layers.len() {
                for b in 0..slots.len() {
                    rec.add_rms_norm(&self.x, (b * c.n_embd) as u64, &self.o, (b * c.n_embd) as u64, &self.layers[li + 1].attn_norm, 0, &self.xn, (b * c.n_embd) as u64, ne, c.eps);
                }
                rec.barrier();
            } else {
                for b in 0..slots.len() {
                    rec.add_rms_norm(&self.x, (b * c.n_embd) as u64, &self.o, (b * c.n_embd) as u64, &self.out_norm, 0, &self.xn, (b * c.n_embd) as u64, ne, c.eps);
                }
                rec.barrier();
            }
        }

        // Output logits for all B slots using cached multi-slot buffer (zero per-step VRAM allocs)
        let w = self.output.as_ref().unwrap_or(&self.embd);
        let lbuf = if slots.len() > 2 {
            &self.multi_logits_buf.as_ref().unwrap().1
        } else {
            &self.logits_buf2
        };
        rec.qmm(w.ty, &w.buf, 0, &self.xn, 0, lbuf, 0, w.k, w.m, batch);
        rec.barrier();
        rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;

        let mut all = vec![0f32; slots.len() * c.n_vocab];
        self.ctx.download(lbuf, 0, bytemuck::cast_slice_mut(&mut all)).map_err(|e| EngineError::Backend(e.to_string()))?;
        let out: Vec<Vec<f32>> = all.chunks_exact(c.n_vocab).map(|ch| ch.to_vec()).collect();
        Ok(Some(out))
    }

    fn forward_embd(&mut self, embd: &[f32], pos: usize, page_table: &[u32], out: Out) -> Result<(), EngineError> {
        self.forward_inner(None, Some(embd), pos, page_table, out, false).map(|_| ())
    }

    fn logits(&self) -> &[f32] { &self.logits }
    fn hidden(&self) -> &[f32] { &self.hidden }
}

impl VkExec {
    fn forward_inner(&mut self, token: Option<u32>, embd: Option<&[f32]>, pos: usize, page_table: &[u32], out: Out, gpu_argmax: bool) -> Result<Option<u32>, EngineError> {
        if page_table.len() > self.max_pages {
            return Err(EngineError::Context("page table exceeds GPU capacity".into()));
        }
        self.upload_pt(page_table)?;
        if let Some(e) = embd {
            self.ctx.upload(&self.x, 0, bytemuck::cast_slice(e)).map_err(|er| EngineError::Backend(er.to_string()))?;
        }

        let c = &self.cfg;
        let mut rec = self.ctx.begin();
        rec.set_defer_barriers(true);
        if let Some(tok) = token {
            rec.embed(self.embd.ty, &self.embd.buf, tok, &self.x, 0, c.n_embd as u32);
            rec.barrier();
        }

        if !self.layers.is_empty() {
            rec.rms_norm(&self.x, 0, &self.layers[0].attn_norm, 0, &self.xn, 0, c.n_embd as u32, c.eps);
            rec.barrier();
        }

        for (li, l) in self.layers.iter().enumerate() {
            match &l.mixer {
                VkMixer::FullAttn { wq, wk, wv, wo, bq, bk, bv, q_norm, k_norm, gated_q, n_head_kv, kv_layer_idx, sliding_window, rope_base, use_rope_factors, head_gate_w } => {
                    if *gated_q {
                        rec.qmv(wq.ty, &wq.buf, 0, &self.xn, 0, &self.qg, 0, wq.k, wq.m);
                    } else {
                        rec.qmv(wq.ty, &wq.buf, 0, &self.xn, 0, &self.q, 0, wq.k, wq.m);
                    }
                    rec.qmv(wk.ty, &wk.buf, 0, &self.xn, 0, &self.k, 0, wk.k, wk.m);
                    rec.qmv(wv.ty, &wv.buf, 0, &self.xn, 0, &self.v, 0, wv.k, wv.m);
                    if let Some(hg) = head_gate_w {
                        rec.qmv(hg.ty, &hg.buf, 0, &self.xn, 0, &self.head_gate_buf, 0, hg.k, hg.m);
                    }
                    rec.barrier();
                    if *gated_q {
                        rec.split_qg(&self.qg, &self.q, &self.q_gate, c.n_head as u32, c.head_dim as u32);
                        rec.barrier();
                    }
                    if bq.is_some() || bk.is_some() || bv.is_some() {
                        if let Some(b) = bq { rec.add(&self.q, 0, b, 0, (c.n_head * c.head_dim) as u32); }
                        if let Some(b) = bk { rec.add(&self.k, 0, b, 0, wk.m); }
                        if let Some(b) = bv { rec.add(&self.v, 0, b, 0, wv.m); }
                        rec.barrier();
                    }
                    if q_norm.is_some() || k_norm.is_some() {
                        if let Some(qn) = q_norm { rec.rms_norm_heads(&self.q, 0, qn, c.n_head as u32, c.head_dim as u32, c.eps); }
                        if let Some(kn) = k_norm { rec.rms_norm_heads(&self.k, 0, kn, *n_head_kv, c.head_dim as u32, c.eps); }
                        rec.barrier();
                    }
                    rec.rope_ext(&self.q, 0, &self.k, 0, &self.rope_factors, *use_rope_factors, c.n_head as u32, *n_head_kv, c.head_dim as u32, c.n_rot as u32, pos as u32, *rope_base, c.rope_mode);
                    rec.barrier();
                    let kv_off = *kv_layer_idx as u64 * self.layer_stride;
                    rec.kv_write(&self.k, 0, &self.v, 0, &self.kv, kv_off, &self.pt_buf, 0, pos as u32, *n_head_kv * c.head_dim as u32);
                    rec.barrier();
                    rec.attention_swa(&self.q, 0, &self.kv, kv_off, &self.pt_buf, 0, &self.attn, 0, c.n_head as u32, *n_head_kv, c.head_dim as u32, pos as u32, *sliding_window);
                    rec.barrier();
                    if *gated_q {
                        rec.mul_sigmoid(&self.attn, &self.q_gate, (c.n_head * c.head_dim) as u32);
                        rec.barrier();
                    }
                    if head_gate_w.is_some() {
                        rec.mul_head_sigmoid(&self.attn, &self.head_gate_buf, c.n_head as u32, c.head_dim as u32);
                        rec.barrier();
                    }
                    rec.qmv(wo.ty, &wo.buf, 0, &self.attn, 0, &self.o, 0, wo.k, wo.m);
                    rec.barrier();
                }
                VkMixer::ShortConv { in_proj, out_proj, conv_w, state, l_cache } => {
                    rec.qmv(in_proj.ty, &in_proj.buf, 0, &self.xn, 0, &self.bcx, 0, in_proj.k, in_proj.m);
                    rec.barrier();
                    rec.shortconv_step(&self.bcx, conv_w, state, &self.xn, c.n_embd as u32, *l_cache);
                    rec.barrier();
                    rec.qmv(out_proj.ty, &out_proj.buf, 0, &self.xn, 0, &self.o, 0, out_proj.k, out_proj.m);
                    rec.barrier();
                }
                VkMixer::DeltaNet {
                    wqkv, wgate, walpha, wbeta, wout, conv_w, dt_bias, ssm_a, ssm_norm,
                    conv_state, ssm_state, n_k_heads, n_v_heads, state_dim, k_conv, conv_ch,
                } => {
                    rec.qmv(wqkv.ty, &wqkv.buf, 0, &self.xn, 0, &self.qkv_raw, 0, wqkv.k, wqkv.m);
                    rec.qmv(wgate.ty, &wgate.buf, 0, &self.xn, 0, &self.z_buf, 0, wgate.k, wgate.m);
                    rec.qmv(walpha.ty, &walpha.buf, 0, &self.xn, 0, &self.alpha_buf, 0, walpha.k, walpha.m);
                    rec.qmv(wbeta.ty, &wbeta.buf, 0, &self.xn, 0, &self.beta_buf, 0, wbeta.k, wbeta.m);
                    rec.barrier();
                    rec.ssm_conv_silu(&self.qkv_raw, conv_w, conv_state, &self.qkv_conv, *conv_ch, *k_conv);
                    rec.barrier();
                    rec.deltanet_step(&self.qkv_conv, &self.alpha_buf, &self.beta_buf, dt_bias, ssm_a, ssm_state, &self.dn_out, *n_k_heads, *n_v_heads, *state_dim, c.eps);
                    rec.barrier();
                    rec.gated_rms_norm(&self.dn_out, ssm_norm, &self.z_buf, &self.dn_norm, *n_v_heads, *state_dim, c.eps);
                    rec.barrier();
                    rec.qmv(wout.ty, &wout.buf, 0, &self.dn_norm, 0, &self.o, 0, wout.k, wout.m);
                    rec.barrier();
                }
            }
            if let Some(pan) = &l.post_attn_norm {
                rec.rms_norm(&self.o, 0, pan, 0, &self.o, 0, c.n_embd as u32, c.eps);
                rec.barrier();
            }
            rec.add_rms_norm(&self.x, 0, &self.o, 0, &l.ffn_norm, 0, &self.xn, 0, c.n_embd as u32, c.eps);
            rec.barrier();
            rec.qmv(l.gate.ty, &l.gate.buf, 0, &self.xn, 0, &self.gate, 0, l.gate.k, l.gate.m);
            rec.qmv(l.up.ty, &l.up.buf, 0, &self.xn, 0, &self.up, 0, l.up.k, l.up.m);
            rec.barrier();
            rec.swiglu(&self.gate, 0, &self.up, 0, &self.act, 0, c.n_ff as u32);
            rec.barrier();
            rec.qmv(l.down.ty, &l.down.buf, 0, &self.act, 0, &self.o, 0, l.down.k, l.down.m);
            rec.barrier();
            if let Some(pfn) = &l.post_ffw_norm {
                rec.rms_norm(&self.o, 0, pfn, 0, &self.o, 0, c.n_embd as u32, c.eps);
                rec.barrier();
            }
            if li + 1 < self.layers.len() {
                rec.add_rms_norm(&self.x, 0, &self.o, 0, &self.layers[li + 1].attn_norm, 0, &self.xn, 0, c.n_embd as u32, c.eps);
                rec.barrier();
            } else if out.logits || out.hidden {
                rec.add_rms_norm(&self.x, 0, &self.o, 0, &self.out_norm, 0, &self.hidden_buf, 0, c.n_embd as u32, c.eps);
                rec.barrier();
            } else {
                rec.add(&self.x, 0, &self.o, 0, c.n_embd as u32);
                rec.barrier();
            }
        }

        if out.logits {
            let w = self.output.as_ref().unwrap_or(&self.embd);
            rec.qmv(w.ty, &w.buf, 0, &self.hidden_buf, 0, &self.logits_buf, 0, w.k, w.m);
            rec.barrier();
            if gpu_argmax {
                rec.argmax(&self.logits_buf, 0, &self.argmax_buf, c.n_vocab as u32);
                rec.barrier();
            }
        }
        // Reads recorded into the same submit (no separate download round trips).
        let mut idx = [0u32; 1];
        {
            let mut reads: Vec<(&Buffer, u64, &mut [u8])> = Vec::with_capacity(2);
            if out.hidden {
                reads.push((&self.hidden_buf, 0, bytemuck::cast_slice_mut(&mut self.hidden)));
            }
            if out.logits {
                if gpu_argmax {
                    reads.push((&self.argmax_buf, 0, bytemuck::cast_slice_mut(&mut idx)));
                } else {
                    reads.push((&self.logits_buf, 0, bytemuck::cast_slice_mut(&mut self.logits)));
                }
            }
            rec.finish_and_read(&mut reads).map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        if out.logits && gpu_argmax {
            return Ok(Some(idx[0]));
        }
        Ok(None)
    }

    /// Upload the page table unless pt_buf already holds exactly these entries (prefix
    /// comparison is not enough: entries are rewritten when sequences fork / reuse pages).
    fn upload_pt(&mut self, page_table: &[u32]) -> Result<(), EngineError> {
        if self.pt_last.len() >= page_table.len() && self.pt_last[..page_table.len()] == *page_table {
            return Ok(());
        }
        self.ctx.upload(&self.pt_buf, 0, bytemuck::cast_slice(page_table)).map_err(|e| EngineError::Backend(e.to_string()))?;
        if self.pt_last.len() < page_table.len() {
            self.pt_last.resize(page_table.len(), 0);
        }
        self.pt_last[..page_table.len()].copy_from_slice(page_table);
        Ok(())
    }

    fn ensure_rec_snaps(&mut self) -> Result<(), EngineError> {
        if self.rec_snaps.is_none() {
            let mut snaps = Vec::new();
            for l in &self.layers {
                match &l.mixer {
                    VkMixer::ShortConv { state, .. } => {
                        let b = self.ctx.create_device_buffer(state.size()).map_err(|e| EngineError::Backend(e.to_string()))?;
                        snaps.push((b, None));
                    }
                    VkMixer::DeltaNet { conv_state, ssm_state, .. } => {
                        // MAX_VERIFY - 1 rollback slots (after each verified token but the last).
                        let n = (MAX_VERIFY - 1) as u64;
                        let b1 = self.ctx.create_device_buffer(n * conv_state.size()).map_err(|e| EngineError::Backend(e.to_string()))?;
                        let b2 = self.ctx.create_device_buffer(n * ssm_state.size()).map_err(|e| EngineError::Backend(e.to_string()))?;
                        snaps.push((b1, Some(b2)));
                    }
                    VkMixer::FullAttn { .. } => {}
                }
            }
            self.rec_snaps = Some(snaps);
        }
        Ok(())
    }

    fn forward_batch_chunk(
        &mut self,
        chunk: &[u32],
        pos0: usize,
        out: Out,
        mut collect_hiddens: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<(), EngineError> {
        let c = &self.cfg;
        let ne = c.n_embd as u32;
        let n_layer = self.layers.len();
        let mut l0 = 0usize;
        // Small batches (decode-time verify / commit) stay in one submit: each extra submit costs a
        // CPU-GPU round trip, and their GPU time is far from the TDR limit.
        let split = chunk.len() >= 64;
        while split && n_layer - l0 > LAYERS_PER_SUBMIT {
            let mut rec = self.ctx.begin();
            rec.set_defer_barriers(true);
            rec.set_attn_split(false);
            self.record_batch_layers(&mut rec, chunk, pos0, false, l0..l0 + LAYERS_PER_SUBMIT);
            rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;
            l0 += LAYERS_PER_SUBMIT;
        }
        let mut rec = self.ctx.begin();
        rec.set_defer_barriers(true);
        rec.set_attn_split(false);
        self.record_batch_layers(&mut rec, chunk, pos0, false, l0..n_layer);

        let last_off = ((chunk.len() - 1) * c.n_embd) as u64;
        if collect_hiddens.is_some() {
            rec.rms_norm_batch(&self.x, 0, &self.out_norm, 0, &self.xn, 0, ne, c.eps, chunk.len() as u32);
            rec.barrier();
            if out.logits {
                let w = self.output.as_ref().unwrap_or(&self.embd);
                rec.qmv(w.ty, &w.buf, 0, &self.xn, last_off, &self.logits_buf, 0, w.k, w.m);
                rec.barrier();
            }
        } else {
            if out.logits || out.hidden {
                rec.rms_norm(&self.x, last_off, &self.out_norm, 0, &self.hidden_buf, 0, ne, c.eps);
                rec.barrier();
            }
            if out.logits {
                let w = self.output.as_ref().unwrap_or(&self.embd);
                rec.qmv(w.ty, &w.buf, 0, &self.hidden_buf, 0, &self.logits_buf, 0, w.k, w.m);
                rec.barrier();
            }
        }
        rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;

        if let Some(h_out) = collect_hiddens.as_mut() {
            let mut batch_h = vec![0f32; chunk.len() * c.n_embd];
            self.ctx
                .download(&self.xn, 0, bytemuck::cast_slice_mut(&mut batch_h))
                .map_err(|e| EngineError::Backend(e.to_string()))?;
            for b in 0..chunk.len() {
                h_out.push(batch_h[b * c.n_embd..(b + 1) * c.n_embd].to_vec());
            }
            if out.hidden {
                let last = (chunk.len() - 1) * c.n_embd;
                self.hidden.copy_from_slice(&batch_h[last..last + c.n_embd]);
            }
        } else if out.hidden {
            self.ctx.download(&self.hidden_buf, 0, bytemuck::cast_slice_mut(&mut self.hidden)).map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        if out.logits {
            self.ctx.download(&self.logits_buf, 0, bytemuck::cast_slice_mut(&mut self.logits)).map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        Ok(())
    }

    /// Record layers `layers` of a batched forward pass (token embedding included when the
    /// range starts at layer 0). Activations live in GPU buffers, so consecutive ranges may be
    /// recorded into separate submits.
    fn record_batch_layers(&self, rec: &mut ocak_vk::Recorder<'_>, chunk: &[u32], pos0: usize, snap_after_t0: bool, layers: std::ops::Range<usize>) {
        let c = &self.cfg;
        let batch = chunk.len() as u32;
        let ne = c.n_embd as u32;
        let qd = (c.n_head * c.head_dim) as u32;
        // Recurrent-state snapshot slots are numbered over the non-attention layers.
        let mut rec_snap_idx = self.layers[..layers.start].iter().filter(|l| !matches!(l.mixer, VkMixer::FullAttn { .. })).count();

        if layers.start == 0 {
            for (b, &tok) in chunk.iter().enumerate() {
                rec.embed(self.embd.ty, &self.embd.buf, tok, &self.x, (b * c.n_embd) as u64, ne);
            }
            rec.barrier();
        }

        // The residual add closing each layer is fused with the next layer's attn norm
        // (add_rms_norm_batch: bit-identical to the decode path's add_rms_norm).
        let layer_list = &self.layers[layers];
        for (li, l) in layer_list.iter().enumerate() {
            if li == 0 {
                rec.rms_norm_batch(&self.x, 0, &l.attn_norm, 0, &self.xn, 0, ne, c.eps, batch);
                rec.barrier();
            }
            match &l.mixer {
                VkMixer::FullAttn { wq, wk, wv, wo, bq, bk, bv, q_norm, k_norm, gated_q, n_head_kv, kv_layer_idx, sliding_window, rope_base, use_rope_factors, head_gate_w } => {
                    let kv_dim = (*n_head_kv as usize) * c.head_dim;
                    if *gated_q {
                        rec.qmm(wq.ty, &wq.buf, 0, &self.xn, 0, &self.qg, 0, wq.k, wq.m, batch);
                    } else {
                        rec.qmm(wq.ty, &wq.buf, 0, &self.xn, 0, &self.q, 0, wq.k, wq.m, batch);
                    }
                    rec.qmm(wk.ty, &wk.buf, 0, &self.xn, 0, &self.k, 0, wk.k, wk.m, batch);
                    rec.qmm(wv.ty, &wv.buf, 0, &self.xn, 0, &self.v, 0, wv.k, wv.m, batch);
                    if let Some(hg) = head_gate_w {
                        rec.qmm(hg.ty, &hg.buf, 0, &self.xn, 0, &self.head_gate_buf, 0, hg.k, hg.m, batch);
                    }
                    rec.barrier();
                    if *gated_q {
                        rec.split_qg_batch(&self.qg, &self.q, &self.q_gate, c.n_head as u32, c.head_dim as u32, batch);
                        rec.barrier();
                    }
                    if bq.is_some() || bk.is_some() || bv.is_some() {
                        for b in 0..chunk.len() {
                            if let Some(bias) = bq { rec.add(&self.q, (b * qd as usize) as u64, bias, 0, qd); }
                            if let Some(bias) = bk { rec.add(&self.k, (b * kv_dim) as u64, bias, 0, wk.m); }
                            if let Some(bias) = bv { rec.add(&self.v, (b * kv_dim) as u64, bias, 0, wv.m); }
                        }
                        rec.barrier();
                    }
                    if q_norm.is_some() || k_norm.is_some() {
                        if let Some(qn) = q_norm { rec.rms_norm_heads_batch(&self.q, 0, qn, c.n_head as u32, c.head_dim as u32, c.eps, batch); }
                        if let Some(kn) = k_norm { rec.rms_norm_heads_batch(&self.k, 0, kn, *n_head_kv, c.head_dim as u32, c.eps, batch); }
                        rec.barrier();
                    }
                    rec.rope_ext_batch(&self.q, 0, &self.k, 0, &self.rope_factors, *use_rope_factors, c.n_head as u32, *n_head_kv, c.head_dim as u32, c.n_rot as u32, pos0 as u32, *rope_base, c.rope_mode, batch);
                    rec.barrier();
                    let kv_off = *kv_layer_idx as u64 * self.layer_stride;
                    rec.kv_write_batch(&self.k, 0, &self.v, 0, &self.kv, kv_off, &self.pt_buf, 0, pos0 as u32, kv_dim as u32, batch);
                    rec.barrier();
                    if batch >= 16 {
                        rec.attention_prefill(&self.q, 0, &self.kv, kv_off, &self.pt_buf, 0, &self.attn, 0, c.n_head as u32, *n_head_kv, c.head_dim as u32, pos0 as u32, *sliding_window, batch);
                    } else {
                        // Small batches (speculative verify): per-token decode attention keeps
                        // split-K parallelism over long contexts.
                        for b in 0..chunk.len() {
                            rec.attention_swa(&self.q, (b * qd as usize) as u64, &self.kv, kv_off, &self.pt_buf, 0, &self.attn, (b * qd as usize) as u64, c.n_head as u32, *n_head_kv, c.head_dim as u32, (pos0 + b) as u32, *sliding_window);
                        }
                    }
                    rec.barrier();
                    if *gated_q {
                        rec.mul_sigmoid(&self.attn, &self.q_gate, batch * qd);
                        rec.barrier();
                    }
                    if head_gate_w.is_some() {
                        rec.mul_head_sigmoid_batch(&self.attn, &self.head_gate_buf, c.n_head as u32, c.head_dim as u32, batch);
                        rec.barrier();
                    }
                    rec.qmm(wo.ty, &wo.buf, 0, &self.attn, 0, &self.o, 0, wo.k, wo.m, batch);
                    rec.barrier();
                }
                VkMixer::ShortConv { in_proj, out_proj, conv_w, state, l_cache } => {
                    rec.qmm(in_proj.ty, &in_proj.buf, 0, &self.xn, 0, &self.bcx, 0, in_proj.k, in_proj.m, batch);
                    rec.barrier();
                    if snap_after_t0 && batch == 2 {
                        let snaps = self.rec_snaps.as_ref().unwrap();
                        rec.shortconv_step_batch(&self.bcx, conv_w, state, &self.xn, ne, *l_cache, 1);
                        rec.barrier();
                        rec.copy_buf_bytes(state, &snaps[rec_snap_idx].0, state.size());
                        rec.copy(&self.xn, 0, &self.o, 0, ne);
                        rec.copy(&self.bcx, (3 * ne) as u64, &self.bcx, 0, 3 * ne);
                        rec.barrier();
                        rec.shortconv_step_batch(&self.bcx, conv_w, state, &self.xn, ne, *l_cache, 1);
                        rec.barrier();
                        rec.copy(&self.xn, 0, &self.xn, ne as u64, ne);
                        rec.barrier();
                        rec.copy(&self.o, 0, &self.xn, 0, ne);
                        rec.barrier();
                    } else {
                        rec.shortconv_step_batch(&self.bcx, conv_w, state, &self.xn, ne, *l_cache, batch);
                        rec.barrier();
                    }
                    rec_snap_idx += 1;
                    rec.qmm(out_proj.ty, &out_proj.buf, 0, &self.xn, 0, &self.o, 0, out_proj.k, out_proj.m, batch);
                    rec.barrier();
                }
                VkMixer::DeltaNet {
                    wqkv, wgate, walpha, wbeta, wout, conv_w, dt_bias, ssm_a, ssm_norm,
                    conv_state, ssm_state, n_k_heads, n_v_heads, state_dim, k_conv, conv_ch,
                } => {
                    rec.qmm(wqkv.ty, &wqkv.buf, 0, &self.xn, 0, &self.qkv_raw, 0, wqkv.k, wqkv.m, batch);
                    rec.qmm(wgate.ty, &wgate.buf, 0, &self.xn, 0, &self.z_buf, 0, wgate.k, wgate.m, batch);
                    rec.qmm(walpha.ty, &walpha.buf, 0, &self.xn, 0, &self.alpha_buf, 0, walpha.k, walpha.m, batch);
                    rec.qmm(wbeta.ty, &wbeta.buf, 0, &self.xn, 0, &self.beta_buf, 0, wbeta.k, wbeta.m, batch);
                    rec.barrier();
                    if snap_after_t0 && (2..=MAX_VERIFY as u32).contains(&batch) {
                        // All verified tokens in one dispatch per kernel; the kernels also write
                        // the exact state after each token but the last (rollback points for
                        // rejected drafts) to the snapshot slots from registers.
                        let snaps = self.rec_snaps.as_ref().unwrap();
                        let n = batch - 1;
                        rec.ssm_conv_silu_batch_snap(&self.qkv_raw, conv_w, conv_state, &self.qkv_conv, *conv_ch, *k_conv, batch, Some((&snaps[rec_snap_idx].0, n)));
                        rec.barrier();
                        rec.deltanet_step_batch_snap(&self.qkv_conv, &self.alpha_buf, &self.beta_buf, dt_bias, ssm_a, ssm_state, &self.dn_out, *n_k_heads, *n_v_heads, *state_dim, c.eps, batch, Some((snaps[rec_snap_idx].1.as_ref().unwrap(), n)));
                        rec.barrier();
                        rec.gated_rms_norm_batch(&self.dn_out, ssm_norm, &self.z_buf, &self.dn_norm, *n_v_heads, *state_dim, c.eps, batch);
                        rec.barrier();
                    } else {
                        rec.ssm_conv_silu_batch(&self.qkv_raw, conv_w, conv_state, &self.qkv_conv, *conv_ch, *k_conv, batch);
                        rec.barrier();
                        rec.deltanet_step_batch(&self.qkv_conv, &self.alpha_buf, &self.beta_buf, dt_bias, ssm_a, ssm_state, &self.dn_out, *n_k_heads, *n_v_heads, *state_dim, c.eps, batch);
                        rec.barrier();
                        rec.gated_rms_norm_batch(&self.dn_out, ssm_norm, &self.z_buf, &self.dn_norm, *n_v_heads, *state_dim, c.eps, batch);
                        rec.barrier();
                    }
                    rec_snap_idx += 1;
                    rec.qmm(wout.ty, &wout.buf, 0, &self.dn_norm, 0, &self.o, 0, wout.k, wout.m, batch);
                    rec.barrier();
                }
            }
            if let Some(pan) = &l.post_attn_norm {
                rec.rms_norm_batch(&self.o, 0, pan, 0, &self.o, 0, ne, c.eps, batch);
                rec.barrier();
            }
            rec.add_rms_norm_batch(&self.x, 0, &self.o, 0, &l.ffn_norm, 0, &self.xn, 0, ne, c.eps, batch);
            rec.barrier();
            rec.qmm(l.gate.ty, &l.gate.buf, 0, &self.xn, 0, &self.gate, 0, l.gate.k, l.gate.m, batch);
            rec.qmm(l.up.ty, &l.up.buf, 0, &self.xn, 0, &self.up, 0, l.up.k, l.up.m, batch);
            rec.barrier();
            rec.swiglu(&self.gate, 0, &self.up, 0, &self.act, 0, batch * c.n_ff as u32);
            rec.barrier();
            rec.qmm(l.down.ty, &l.down.buf, 0, &self.act, 0, &self.o, 0, l.down.k, l.down.m, batch);
            rec.barrier();
            if let Some(pfn) = &l.post_ffw_norm {
                rec.rms_norm_batch(&self.o, 0, pfn, 0, &self.o, 0, ne, c.eps, batch);
                rec.barrier();
            }
            if let Some(next) = layer_list.get(li + 1) {
                rec.add_rms_norm_batch(&self.x, 0, &self.o, 0, &next.attn_norm, 0, &self.xn, 0, ne, c.eps, batch);
            } else {
                rec.add(&self.x, 0, &self.o, 0, batch * ne);
            }
            rec.barrier();
        }
    }
}
