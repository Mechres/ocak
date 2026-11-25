use crate::config::ModelConfig;
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

struct VkLayer {
    attn_norm: Buffer,
    ffn_norm: Buffer,
    wq: WeightBuf,
    wk: WeightBuf,
    wv: WeightBuf,
    wo: WeightBuf,
    gate: WeightBuf,
    up: WeightBuf,
    down: WeightBuf,
    bq: Option<Buffer>,
    bk: Option<Buffer>,
    bv: Option<Buffer>,
}

pub struct VkExec {
    ctx: Arc<VkContext>,
    cfg: ModelConfig,
    layers: Vec<VkLayer>,
    embd: WeightBuf,
    out_norm: Buffer,
    output: Option<WeightBuf>, // None when tied to embd
    kv: Buffer,
    layer_stride: u64,
    pt_buf: Buffer,
    max_pages: usize,
    x: Buffer,
    xn: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    o: Buffer,
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
    let buf = ctx
        .create_device_buffer((data.len() * 4) as u64)
        .map_err(|e| EngineError::Backend(e.to_string()))?;
    ctx.upload(&buf, 0, bytemuck::cast_slice(&data))
        .map_err(|e| EngineError::Backend(e.to_string()))?;
    Ok(buf)
}

fn upload_weight(ctx: &Arc<VkContext>, g: &GgufFile, ti: &TensorInfo) -> Result<WeightBuf, EngineError> {
    let ty = WType::from_ggml(ti.ty).map_err(|e| EngineError::Unsupported(e.to_string()))?;
    let raw = g.tensor_data(ti);
    let buf = ctx
        .create_device_buffer(raw.len() as u64)
        .map_err(|e| EngineError::Backend(e.to_string()))?;
    ctx.upload(&buf, 0, raw)
        .map_err(|e| EngineError::Backend(e.to_string()))?;
    Ok(WeightBuf { buf, ty, k: ti.n_cols() as u32, m: ti.n_rows() as u32 })
}

fn req_tensor<'a>(g: &'a GgufFile, name: &str) -> Result<&'a TensorInfo, EngineError> {
    g.tensor(name).ok_or_else(|| EngineError::Unsupported(format!("missing {name}")))
}

impl VkExec {
    pub fn new(g: Arc<GgufFile>, cfg: ModelConfig, n_pages: usize) -> Result<Self, EngineError> {
        let ctx = VkContext::new().map_err(|e| EngineError::Backend(e.to_string()))?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let b = |s: &str| -> Result<Option<Buffer>, EngineError> {
                if cfg.qkv_bias {
                    Ok(Some(upload_f32(&ctx, &g, &p(s))?))
                } else {
                    Ok(None)
                }
            };
            layers.push(VkLayer {
                attn_norm: upload_f32(&ctx, &g, &p("attn_norm.weight"))?,
                ffn_norm: upload_f32(&ctx, &g, &p("ffn_norm.weight"))?,
                wq: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_q.weight"))?)?,
                wk: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_k.weight"))?)?,
                wv: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_v.weight"))?)?,
                wo: upload_weight(&ctx, &g, req_tensor(&g, &p("attn_output.weight"))?)?,
                gate: upload_weight(&ctx, &g, req_tensor(&g, &p("ffn_gate.weight"))?)?,
                up: upload_weight(&ctx, &g, req_tensor(&g, &p("ffn_up.weight"))?)?,
                down: upload_weight(&ctx, &g, req_tensor(&g, &p("ffn_down.weight"))?)?,
                bq: b("attn_q.bias")?,
                bk: b("attn_k.bias")?,
                bv: b("attn_v.bias")?,
            });
        }
        let embd = upload_weight(&ctx, &g, req_tensor(&g, "token_embd.weight")?)?;
        let out_norm = upload_f32(&ctx, &g, "output_norm.weight")?;
        let output = if cfg.tied_output {
            None
        } else {
            Some(upload_weight(&ctx, &g, req_tensor(&g, "output.weight")?)?)
        };

        let mk = |elems: usize| -> Result<Buffer, EngineError> {
            ctx.create_device_buffer((elems * 4) as u64)
                .map_err(|e| EngineError::Backend(e.to_string()))
        };
        let layer_stride = (n_pages * 2 * 16 * cfg.kv_dim()) as u64;
        let qd = cfg.n_head * cfg.head_dim;
        Ok(Self {
            kv: mk(layer_stride as usize * cfg.n_layer)?,
            layer_stride,
            pt_buf: mk(n_pages.max(16))?,
            max_pages: n_pages.max(16),
            x: mk(cfg.n_embd)?,
            xn: mk(cfg.n_embd)?,
            q: mk(qd)?,
            k: mk(cfg.kv_dim())?,
            v: mk(cfg.kv_dim())?,
            attn: mk(qd)?,
            o: mk(cfg.n_embd)?,
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
    fn name(&self) -> String {
        format!("vulkan ({})", self.ctx.device_name())
    }

    fn forward(&mut self, token: u32, pos: usize, page_table: &[u32], out: Out) -> Result<(), EngineError> {
        if page_table.len() > self.max_pages {
            return Err(EngineError::Context("page table exceeds GPU capacity".into()));
        }
        self.ctx
            .upload(&self.pt_buf, 0, bytemuck::cast_slice(page_table))
            .map_err(|e| EngineError::Backend(e.to_string()))?;

        let c = &self.cfg;
        let mut rec = self.ctx.begin();
        rec.embed(self.embd.ty, &self.embd.buf, token, &self.x, 0, c.n_embd as u32);
        for (li, l) in self.layers.iter().enumerate() {
            rec.rms_norm(&self.x, 0, &l.attn_norm, 0, &self.xn, 0, c.n_embd as u32, c.eps);
            rec.qmv(l.wq.ty, &l.wq.buf, 0, &self.xn, 0, &self.q, 0, l.wq.k, l.wq.m);
            rec.qmv(l.wk.ty, &l.wk.buf, 0, &self.xn, 0, &self.k, 0, l.wk.k, l.wk.m);
            rec.qmv(l.wv.ty, &l.wv.buf, 0, &self.xn, 0, &self.v, 0, l.wv.k, l.wv.m);
            if let Some(b) = &l.bq { rec.add(&self.q, 0, b, 0, l.wq.m); }
            if let Some(b) = &l.bk { rec.add(&self.k, 0, b, 0, l.wk.m); }
            if let Some(b) = &l.bv { rec.add(&self.v, 0, b, 0, l.wv.m); }
            rec.rope(&self.q, 0, &self.k, 0, c.n_head as u32, c.n_head_kv as u32, c.head_dim as u32, c.n_rot as u32, pos as u32, c.rope_base, c.rope_mode);
            let kv_off = li as u64 * self.layer_stride;
            rec.kv_write(&self.k, 0, &self.v, 0, &self.kv, kv_off, &self.pt_buf, 0, pos as u32, c.kv_dim() as u32);
            rec.attention(&self.q, 0, &self.kv, kv_off, &self.pt_buf, 0, &self.attn, 0, c.n_head as u32, c.n_head_kv as u32, c.head_dim as u32, pos as u32);
            rec.qmv(l.wo.ty, &l.wo.buf, 0, &self.attn, 0, &self.o, 0, l.wo.k, l.wo.m);
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
            self.ctx
                .download(&self.hidden_buf, 0, bytemuck::cast_slice_mut(&mut self.hidden))
                .map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        if out.logits {
            self.ctx
                .download(&self.logits_buf, 0, bytemuck::cast_slice_mut(&mut self.logits))
                .map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        Ok(())
    }

    fn logits(&self) -> &[f32] { &self.logits }
    fn hidden(&self) -> &[f32] { &self.hidden }
}
