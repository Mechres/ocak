use crate::config::{LayerKind, ModelConfig};
use crate::exec::{EngineError, Executor, Out};
use ocak_gguf::{GgufFile, TensorInfo};
use ocak_vk::{Buffer, VkContext, WType};
use std::sync::Arc;

struct WeightBuf {
    buf: Buffer,
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
    mixer: VkMixer,
    gate: WeightBuf,
    up: WeightBuf,
    down: WeightBuf,
}

pub struct VkExec {
    ctx: Arc<VkContext>,
    cfg: ModelConfig,
    layers: Vec<VkLayer>,
    embd: WeightBuf,
    out_norm: Buffer,
    output: Option<WeightBuf>,
    kv: Buffer,
    layer_stride: u64,
    pt_buf: Buffer,
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
    ctx.upload(&buf, 0, raw).map_err(|e| EngineError::Backend(e.to_string()))?;
    Ok(WeightBuf { buf, ty, k: ti.n_cols() as u32, m: ti.n_rows() as u32 })
}

fn req_tensor<'a>(g: &'a GgufFile, name: &str) -> Result<&'a TensorInfo, EngineError> {
    g.tensor(name).ok_or_else(|| EngineError::Unsupported(format!("missing {name}")))
}

impl VkExec {
    pub fn new(g: Arc<GgufFile>, cfg: ModelConfig, n_pages: usize) -> Result<Self, EngineError> {
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
                LayerKind::FullAttn { n_head_kv, gated_q, qk_norm, kv_layer_idx } => {
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
                mixer,
                gate: upload_weight(&ctx, &g, req_tensor(&g, &p("ffn_gate.weight"))?)?,
                up: upload_weight(&ctx, &g, req_tensor(&g, &p("ffn_up.weight"))?)?,
                down: upload_weight(&ctx, &g, req_tensor(&g, &p("ffn_down.weight"))?)?,
            });
        }
        let embd = upload_weight(&ctx, &g, req_tensor(&g, "token_embd.weight")?)?;
        let out_norm = upload_f32(&ctx, &g, &cfg.output_norm_name)?;
        let output = if cfg.tied_output { None } else { Some(upload_weight(&ctx, &g, req_tensor(&g, "output.weight")?)?) };

        let layer_stride = (n_pages * 2 * 16 * cfg.kv_dim()) as u64;
        let qd = cfg.n_head * cfg.head_dim;
        Ok(Self {
            kv: mk(layer_stride as usize * cfg.n_kv_layers)?,
            layer_stride,
            pt_buf: mk(n_pages.max(16))?,
            max_pages: n_pages.max(16),
            x: mk(cfg.n_embd)?,
            xn: mk(cfg.n_embd)?,
            qg: mk(2 * qd)?,
            q: mk(qd)?,
            q_gate: mk(qd)?,
            k: mk(cfg.kv_dim())?,
            v: mk(cfg.kv_dim())?,
            attn: mk(qd)?,
            o: mk(cfg.n_embd)?,
            bcx: mk(3 * cfg.n_embd)?,
            qkv_raw: mk(max_conv_ch)?,
            qkv_conv: mk(max_conv_ch)?,
            z_buf: mk(max_v_dim)?,
            alpha_buf: mk(max_v_heads)?,
            beta_buf: mk(max_v_heads)?,
            dn_out: mk(max_v_dim)?,
            dn_norm: mk(max_v_dim)?,
            gate: mk(cfg.n_ff)?,
            up: mk(cfg.n_ff)?,
            act: mk(cfg.n_ff)?,
            hidden_buf: mk(cfg.n_embd)?,
            logits_buf: mk(cfg.n_vocab)?,
            logits: vec![0.0; cfg.n_vocab],
            hidden: vec![0.0; cfg.n_embd],
            layers,
            embd,
            out_norm,
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
        if page_table.len() > self.max_pages {
            return Err(EngineError::Context("page table exceeds GPU capacity".into()));
        }
        self.ctx.upload(&self.pt_buf, 0, bytemuck::cast_slice(page_table)).map_err(|e| EngineError::Backend(e.to_string()))?;

        let c = &self.cfg;
        let mut rec = self.ctx.begin();
        rec.embed(self.embd.ty, &self.embd.buf, token, &self.x, 0, c.n_embd as u32);

        for l in &self.layers {
            rec.rms_norm(&self.x, 0, &l.attn_norm, 0, &self.xn, 0, c.n_embd as u32, c.eps);
            match &l.mixer {
                VkMixer::FullAttn { wq, wk, wv, wo, bq, bk, bv, q_norm, k_norm, gated_q, n_head_kv, kv_layer_idx } => {
                    if *gated_q {
                        rec.qmv(wq.ty, &wq.buf, 0, &self.xn, 0, &self.qg, 0, wq.k, wq.m);
                        rec.split_qg(&self.qg, &self.q, &self.q_gate, c.n_head as u32, c.head_dim as u32);
                    } else {
                        rec.qmv(wq.ty, &wq.buf, 0, &self.xn, 0, &self.q, 0, wq.k, wq.m);
                    }
                    rec.qmv(wk.ty, &wk.buf, 0, &self.xn, 0, &self.k, 0, wk.k, wk.m);
                    rec.qmv(wv.ty, &wv.buf, 0, &self.xn, 0, &self.v, 0, wv.k, wv.m);
                    if let Some(b) = bq { rec.add(&self.q, 0, b, 0, (c.n_head * c.head_dim) as u32); }
                    if let Some(b) = bk { rec.add(&self.k, 0, b, 0, wk.m); }
                    if let Some(b) = bv { rec.add(&self.v, 0, b, 0, wv.m); }
                    if let Some(qn) = q_norm { rec.rms_norm_heads(&self.q, 0, qn, c.n_head as u32, c.head_dim as u32, c.eps); }
                    if let Some(kn) = k_norm { rec.rms_norm_heads(&self.k, 0, kn, *n_head_kv, c.head_dim as u32, c.eps); }
                    rec.rope(&self.q, 0, &self.k, 0, c.n_head as u32, *n_head_kv, c.head_dim as u32, c.n_rot as u32, pos as u32, c.rope_base, c.rope_mode);
                    let kv_off = *kv_layer_idx as u64 * self.layer_stride;
                    rec.kv_write(&self.k, 0, &self.v, 0, &self.kv, kv_off, &self.pt_buf, 0, pos as u32, *n_head_kv * c.head_dim as u32);
                    rec.attention(&self.q, 0, &self.kv, kv_off, &self.pt_buf, 0, &self.attn, 0, c.n_head as u32, *n_head_kv, c.head_dim as u32, pos as u32);
                    if *gated_q {
                        rec.mul_sigmoid(&self.attn, &self.q_gate, (c.n_head * c.head_dim) as u32);
                    }
                    rec.qmv(wo.ty, &wo.buf, 0, &self.attn, 0, &self.o, 0, wo.k, wo.m);
                }
                VkMixer::ShortConv { in_proj, out_proj, conv_w, state, l_cache } => {
                    rec.qmv(in_proj.ty, &in_proj.buf, 0, &self.xn, 0, &self.bcx, 0, in_proj.k, in_proj.m);
                    rec.shortconv_step(&self.bcx, conv_w, state, &self.xn, c.n_embd as u32, *l_cache);
                    rec.qmv(out_proj.ty, &out_proj.buf, 0, &self.xn, 0, &self.o, 0, out_proj.k, out_proj.m);
                }
                VkMixer::DeltaNet {
                    wqkv, wgate, walpha, wbeta, wout, conv_w, dt_bias, ssm_a, ssm_norm,
                    conv_state, ssm_state, n_k_heads, n_v_heads, state_dim, k_conv, conv_ch,
                } => {
                    rec.qmv(wqkv.ty, &wqkv.buf, 0, &self.xn, 0, &self.qkv_raw, 0, wqkv.k, wqkv.m);
                    rec.qmv(wgate.ty, &wgate.buf, 0, &self.xn, 0, &self.z_buf, 0, wgate.k, wgate.m);
                    rec.qmv(walpha.ty, &walpha.buf, 0, &self.xn, 0, &self.alpha_buf, 0, walpha.k, walpha.m);
                    rec.qmv(wbeta.ty, &wbeta.buf, 0, &self.xn, 0, &self.beta_buf, 0, wbeta.k, wbeta.m);
                    rec.ssm_conv_silu(&self.qkv_raw, conv_w, conv_state, &self.qkv_conv, *conv_ch, *k_conv);
                    rec.deltanet_step(&self.qkv_conv, &self.alpha_buf, &self.beta_buf, dt_bias, ssm_a, ssm_state, &self.dn_out, *n_k_heads, *n_v_heads, *state_dim, c.eps);
                    rec.gated_rms_norm(&self.dn_out, ssm_norm, &self.z_buf, &self.dn_norm, *n_v_heads, *state_dim, c.eps);
                    rec.qmv(wout.ty, &wout.buf, 0, &self.dn_norm, 0, &self.o, 0, wout.k, wout.m);
                }
            }
            rec.add(&self.x, 0, &self.o, 0, c.n_embd as u32);
            rec.rms_norm(&self.x, 0, &l.ffn_norm, 0, &self.xn, 0, c.n_embd as u32, c.eps);
            rec.qmv(l.gate.ty, &l.gate.buf, 0, &self.xn, 0, &self.gate, 0, l.gate.k, l.gate.m);
            rec.qmv(l.up.ty, &l.up.buf, 0, &self.xn, 0, &self.up, 0, l.up.k, l.up.m);
            rec.swiglu(&self.gate, 0, &self.up, 0, &self.act, 0, c.n_ff as u32);
            rec.qmv(l.down.ty, &l.down.buf, 0, &self.act, 0, &self.o, 0, l.down.k, l.down.m);
            rec.add(&self.x, 0, &self.o, 0, c.n_embd as u32);
        }

        if out.logits || out.hidden {
            rec.rms_norm(&self.x, 0, &self.out_norm, 0, &self.hidden_buf, 0, c.n_embd as u32, c.eps);
        }
        if out.logits {
            let w = self.output.as_ref().unwrap_or(&self.embd);
            rec.qmv(w.ty, &w.buf, 0, &self.hidden_buf, 0, &self.logits_buf, 0, w.k, w.m);
        }
        rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;

        if out.hidden {
            self.ctx.download(&self.hidden_buf, 0, bytemuck::cast_slice_mut(&mut self.hidden)).map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        if out.logits {
            self.ctx.download(&self.logits_buf, 0, bytemuck::cast_slice_mut(&mut self.logits)).map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        Ok(())
    }

    fn logits(&self) -> &[f32] { &self.logits }
    fn hidden(&self) -> &[f32] { &self.hidden }
}
