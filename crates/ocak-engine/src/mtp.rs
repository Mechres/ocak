//! Native Qwen3.5 / RVN NextN Multi-Token Prediction (MTP) self-speculative draft head (`blk.{n_layer}.*`).
use crate::config::ModelConfig;
use crate::exec::EngineError;
use ocak_cpu::{gemv, ops};
use ocak_gguf::quant::dequantize_row;
use ocak_gguf::{GgufFile, TensorInfo};
#[cfg(feature = "vulkan")]
use ocak_vk::{Buffer, VkContext, WType};
use std::sync::Arc;

#[cfg(feature = "vulkan")]
struct MtpGpuMat {
    buf: Buffer,
    ty: WType,
    k: u32,
    m: u32,
}

#[cfg(feature = "vulkan")]
struct MtpGpu {
    ctx: Arc<VkContext>,
    head_buf: Arc<Buffer>,
    head_ty: WType,
    head_k: u32,
    head_m: u32,
    eh_proj: MtpGpuMat,
    wq: MtpGpuMat,
    wk: MtpGpuMat,
    wv: MtpGpuMat,
    wo: MtpGpuMat,
    ffn_gate: MtpGpuMat,
    ffn_up: MtpGpuMat,
    ffn_down: MtpGpuMat,
    enorm: Buffer,
    hnorm: Buffer,
    attn_norm: Buffer,
    post_attn_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    head_norm: Buffer,
    // Scratch & KV buffers
    kv: Buffer,
    pt_buf: Buffer,
    tok_e: Buffer,
    h_in: Buffer,
    concat: Buffer,
    x: Buffer,
    xn: Buffer,
    qg: Buffer,
    q: Buffer,
    q_gate: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    o: Buffer,
    gf: Buffer,
    uf: Buffer,
    af: Buffer,
    h_out: Buffer,
    logits: Buffer,
}

pub struct MtpDrafter {
    g: Arc<GgufFile>,
    cfg: ModelConfig,
    embd: TensorInfo,
    eh_proj: TensorInfo,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    attn_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    wq: TensorInfo,
    wk: TensorInfo,
    wv: TensorInfo,
    wo: TensorInfo,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_gate: TensorInfo,
    ffn_up: TensorInfo,
    ffn_down: TensorInfo,
    head_norm: Vec<f32>,
    head_w: TensorInfo,
    kv: Vec<f32>,
    max_ctx: usize,
    #[cfg(feature = "vulkan")]
    gpu: Option<MtpGpu>,
}

fn req(g: &GgufFile, name: &str) -> Result<TensorInfo, EngineError> {
    g.tensor(name).cloned().ok_or_else(|| EngineError::Unsupported(format!("MTP missing {name}")))
}

fn f32v(g: &GgufFile, name: &str) -> Result<Vec<f32>, EngineError> {
    Ok(g.tensor_to_f32(&req(g, name)?))
}

impl MtpDrafter {
    pub fn try_load(g: Arc<GgufFile>, cfg: ModelConfig, max_ctx: usize) -> Result<Option<Self>, EngineError> {
        let nextn = g.arch_u64("nextn_predict_layers").unwrap_or(0);
        if nextn == 0 || cfg.arch != "qwen35" {
            return Ok(None);
        }
        let il = cfg.n_layer;
        let p = |s: &str| format!("blk.{il}.{s}");
        if g.tensor(&p("nextn.eh_proj.weight")).is_none() {
            return Ok(None);
        }
        let embd = g
            .tensor(&p("nextn.embed_tokens.weight"))
            .or_else(|| g.tensor("token_embd.weight"))
            .cloned()
            .ok_or_else(|| EngineError::Unsupported("MTP missing embed_tokens".into()))?;
        let head_norm = if g.tensor(&p("nextn.shared_head_norm.weight")).is_some() {
            f32v(&g, &p("nextn.shared_head_norm.weight"))?
        } else {
            f32v(&g, &cfg.output_norm_name)?
        };
        let head_w = g
            .tensor(&p("nextn.shared_head_head.weight"))
            .or_else(|| g.tensor("output.weight"))
            .cloned()
            .unwrap_or_else(|| embd.clone());

        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        Ok(Some(Self {
            eh_proj: req(&g, &p("nextn.eh_proj.weight"))?,
            enorm: f32v(&g, &p("nextn.enorm.weight"))?,
            hnorm: f32v(&g, &p("nextn.hnorm.weight"))?,
            attn_norm: f32v(&g, &p("attn_norm.weight"))?,
            post_attn_norm: f32v(&g, &p("post_attention_norm.weight"))?,
            wq: req(&g, &p("attn_q.weight"))?,
            wk: req(&g, &p("attn_k.weight"))?,
            wv: req(&g, &p("attn_v.weight"))?,
            wo: req(&g, &p("attn_output.weight"))?,
            q_norm: f32v(&g, &p("attn_q_norm.weight"))?,
            k_norm: f32v(&g, &p("attn_k_norm.weight"))?,
            ffn_gate: req(&g, &p("ffn_gate.weight"))?,
            ffn_up: req(&g, &p("ffn_up.weight"))?,
            ffn_down: req(&g, &p("ffn_down.weight"))?,
            head_norm,
            head_w,
            embd,
            kv: vec![0f32; 2 * max_ctx * kv_dim],
            max_ctx,
            cfg,
            g,
            #[cfg(feature = "vulkan")]
            gpu: None,
        }))
    }

    /// Attaches a shared Vulkan context + `lm_head` weight so MTP drafting runs in 1 GPU submit (~3-5 ms).
    #[cfg(feature = "vulkan")]
    pub fn attach_gpu(
        &mut self,
        ctx: Arc<VkContext>,
        head_buf: Arc<Buffer>,
        head_ty: WType,
        head_k: u32,
        head_m: u32,
    ) -> Result<(), EngineError> {
        let up_mat = |ti: &TensorInfo| -> Result<MtpGpuMat, EngineError> {
            let ty = WType::from_ggml(ti.ty).map_err(|e| EngineError::Backend(e.to_string()))?;
            let raw = self.g.tensor_data(ti);
            let buf = if let Ok(b) = ctx.create_device_buffer(raw.len() as u64) {
                ctx.upload(&b, 0, raw).map_err(|e| EngineError::Backend(e.to_string()))?;
                b
            } else {
                ctx.create_gtt_buffer_from_slice(raw).map_err(|e| EngineError::Backend(e.to_string()))?
            };
            Ok(MtpGpuMat { buf, ty, k: ti.n_cols() as u32, m: ti.n_rows() as u32 })
        };
        let up_f32 = |v: &[f32]| -> Result<Buffer, EngineError> {
            let bytes = bytemuck::cast_slice(v);
            let b = ctx.create_device_buffer(bytes.len() as u64).map_err(|e| EngineError::Backend(e.to_string()))?;
            ctx.upload(&b, 0, bytes).map_err(|e| EngineError::Backend(e.to_string()))?;
            Ok(b)
        };
        let mk = |elems: usize| -> Result<Buffer, EngineError> {
            ctx.create_device_buffer((elems.max(4) * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))
        };
        let c = &self.cfg;
        let qd = c.n_head * c.head_dim;
        let kv_dim = c.n_head_kv * c.head_dim;
        let n_pages = self.max_ctx.div_ceil(16) + 1;
        let pt: Vec<u32> = (0..n_pages as u32).collect();
        let pt_buf = ctx.create_device_buffer((pt.len() * 4) as u64).map_err(|e| EngineError::Backend(e.to_string()))?;
        ctx.upload(&pt_buf, 0, bytemuck::cast_slice(&pt)).map_err(|e| EngineError::Backend(e.to_string()))?;
        let kv = mk(n_pages * 2 * 16 * (kv_dim / 2))?;

        self.gpu = Some(MtpGpu {
            eh_proj: up_mat(&self.eh_proj)?,
            wq: up_mat(&self.wq)?,
            wk: up_mat(&self.wk)?,
            wv: up_mat(&self.wv)?,
            wo: up_mat(&self.wo)?,
            ffn_gate: up_mat(&self.ffn_gate)?,
            ffn_up: up_mat(&self.ffn_up)?,
            ffn_down: up_mat(&self.ffn_down)?,
            enorm: up_f32(&self.enorm)?,
            hnorm: up_f32(&self.hnorm)?,
            attn_norm: up_f32(&self.attn_norm)?,
            post_attn_norm: up_f32(&self.post_attn_norm)?,
            q_norm: up_f32(&self.q_norm)?,
            k_norm: up_f32(&self.k_norm)?,
            head_norm: up_f32(&self.head_norm)?,
            kv,
            pt_buf,
            tok_e: mk(c.n_embd)?,
            h_in: mk(c.n_embd)?,
            concat: mk(2 * c.n_embd)?,
            x: mk(c.n_embd)?,
            xn: mk(c.n_embd)?,
            qg: mk(2 * qd)?,
            q: mk(qd)?,
            q_gate: mk(qd)?,
            k: mk(kv_dim)?,
            v: mk(kv_dim)?,
            attn: mk(qd)?,
            o: mk(c.n_embd)?,
            gf: mk(c.n_ff)?,
            uf: mk(c.n_ff)?,
            af: mk(c.n_ff)?,
            h_out: mk(c.n_embd)?,
            logits: mk(c.n_vocab)?,
            ctx,
            head_buf,
            head_ty,
            head_k,
            head_m,
        });
        Ok(())
    }

    pub fn has_gpu(&self) -> bool {
        #[cfg(feature = "vulkan")]
        { self.gpu.is_some() }
        #[cfg(not(feature = "vulkan"))]
        { false }
    }

    pub fn reset(&mut self) {
        self.kv.fill(0.0);
    }

    fn mv(&self, ti: &TensorInfo, x: &[f32], y: &mut [f32]) {
        gemv(ti.ty, self.g.tensor_data(ti), x, y, ti.n_cols(), ti.n_rows());
    }

    /// Predicts draft logits for the token AFTER `token`, given the main trunk's post-norm hidden `h_main`.
    pub fn draft_logits(&mut self, h_main: &[f32], token: u32, pos: usize) -> Result<Vec<f32>, EngineError> {
        if pos >= self.max_ctx {
            return Err(EngineError::Context("MTP pos >= max_ctx".into()));
        }
        let c = &self.cfg;
        let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_head_kv, c.head_dim);
        let kv_dim = nkv * hd;

        let mut tok_e = vec![0f32; ne];
        let rb = self.embd.row_bytes();
        let r = token as usize;
        dequantize_row(self.embd.ty, &self.g.tensor_data(&self.embd)[r * rb..(r + 1) * rb], &mut tok_e, ne);

        #[cfg(feature = "vulkan")]
        if let Some(gp) = &self.gpu {
            gp.ctx.upload(&gp.tok_e, 0, bytemuck::cast_slice(&tok_e)).map_err(|e| EngineError::Backend(e.to_string()))?;
            gp.ctx.upload(&gp.h_in, 0, bytemuck::cast_slice(h_main)).map_err(|e| EngineError::Backend(e.to_string()))?;
            let mut rec = gp.ctx.begin();
            rec.set_defer_barriers(true);
            rec.rms_norm(&gp.tok_e, 0, &gp.enorm, 0, &gp.concat, 0, ne as u32, c.eps);
            rec.rms_norm(&gp.h_in, 0, &gp.hnorm, 0, &gp.concat, ne as u64, ne as u32, c.eps);
            rec.barrier();
            rec.qmv(gp.eh_proj.ty, &gp.eh_proj.buf, 0, &gp.concat, 0, &gp.x, 0, gp.eh_proj.k, gp.eh_proj.m);
            rec.barrier();
            rec.rms_norm(&gp.x, 0, &gp.attn_norm, 0, &gp.xn, 0, ne as u32, c.eps);
            rec.barrier();
            rec.qmv(gp.wq.ty, &gp.wq.buf, 0, &gp.xn, 0, &gp.qg, 0, gp.wq.k, gp.wq.m);
            rec.qmv(gp.wk.ty, &gp.wk.buf, 0, &gp.xn, 0, &gp.k, 0, gp.wk.k, gp.wk.m);
            rec.qmv(gp.wv.ty, &gp.wv.buf, 0, &gp.xn, 0, &gp.v, 0, gp.wv.k, gp.wv.m);
            rec.barrier();
            rec.split_qg(&gp.qg, &gp.q, &gp.q_gate, nh as u32, hd as u32);
            rec.barrier();
            rec.rms_norm_heads(&gp.q, 0, &gp.q_norm, nh as u32, hd as u32, c.eps);
            rec.rms_norm_heads(&gp.k, 0, &gp.k_norm, nkv as u32, hd as u32, c.eps);
            rec.barrier();
            rec.rope(&gp.q, 0, &gp.k, 0, nh as u32, nkv as u32, hd as u32, c.n_rot as u32, pos as u32, c.rope_base, 1);
            rec.barrier();
            rec.kv_write(&gp.k, 0, &gp.v, 0, &gp.kv, 0, &gp.pt_buf, 0, pos as u32, kv_dim as u32);
            rec.barrier();
            rec.attention_swa(&gp.q, 0, &gp.kv, 0, &gp.pt_buf, 0, &gp.attn, 0, nh as u32, nkv as u32, hd as u32, pos as u32, 0);
            rec.barrier();
            rec.mul_sigmoid(&gp.attn, &gp.q_gate, (nh * hd) as u32);
            rec.barrier();
            rec.qmv(gp.wo.ty, &gp.wo.buf, 0, &gp.attn, 0, &gp.o, 0, gp.wo.k, gp.wo.m);
            rec.barrier();
            rec.add(&gp.x, 0, &gp.o, 0, ne as u32);
            rec.barrier();
            rec.rms_norm(&gp.x, 0, &gp.post_attn_norm, 0, &gp.xn, 0, ne as u32, c.eps);
            rec.barrier();
            rec.qmv(gp.ffn_gate.ty, &gp.ffn_gate.buf, 0, &gp.xn, 0, &gp.gf, 0, gp.ffn_gate.k, gp.ffn_gate.m);
            rec.qmv(gp.ffn_up.ty, &gp.ffn_up.buf, 0, &gp.xn, 0, &gp.uf, 0, gp.ffn_up.k, gp.ffn_up.m);
            rec.barrier();
            rec.swiglu(&gp.gf, 0, &gp.uf, 0, &gp.af, 0, c.n_ff as u32);
            rec.barrier();
            rec.qmv(gp.ffn_down.ty, &gp.ffn_down.buf, 0, &gp.af, 0, &gp.o, 0, gp.ffn_down.k, gp.ffn_down.m);
            rec.barrier();
            rec.add(&gp.x, 0, &gp.o, 0, ne as u32);
            rec.barrier();
            rec.rms_norm(&gp.x, 0, &gp.head_norm, 0, &gp.h_out, 0, ne as u32, c.eps);
            rec.barrier();
            rec.qmv(gp.head_ty, &gp.head_buf, 0, &gp.h_out, 0, &gp.logits, 0, gp.head_k, gp.head_m);
            rec.barrier();
            rec.finish_and_wait().map_err(|e| EngineError::Backend(e.to_string()))?;
            let mut logits = vec![0f32; c.n_vocab];
            gp.ctx.download(&gp.logits, 0, bytemuck::cast_slice_mut(&mut logits)).map_err(|e| EngineError::Backend(e.to_string()))?;
            return Ok(logits);
        }

        let mut e_norm = vec![0f32; ne];
        let mut h_norm = vec![0f32; ne];
        ops::rms_norm(&tok_e, &self.enorm, &mut e_norm, c.eps);
        ops::rms_norm(h_main, &self.hnorm, &mut h_norm, c.eps);

        let mut concat = Vec::with_capacity(2 * ne);
        concat.extend_from_slice(&e_norm);
        concat.extend_from_slice(&h_norm);
        let mut x = vec![0f32; ne];
        self.mv(&self.eh_proj, &concat, &mut x);

        let mut xn = vec![0f32; ne];
        ops::rms_norm(&x, &self.attn_norm, &mut xn, c.eps);

        let mut qg = vec![0f32; 2 * nh * hd];
        self.mv(&self.wq, &xn, &mut qg);
        let mut q = vec![0f32; nh * hd];
        let mut q_gate = vec![0f32; nh * hd];
        for h in 0..nh {
            q[h * hd..(h + 1) * hd].copy_from_slice(&qg[h * 2 * hd..h * 2 * hd + hd]);
            q_gate[h * hd..(h + 1) * hd].copy_from_slice(&qg[(h * 2 + 1) * hd..(h * 2 + 2) * hd]);
        }
        let mut k = vec![0f32; kv_dim];
        let mut v = vec![0f32; kv_dim];
        self.mv(&self.wk, &xn, &mut k);
        self.mv(&self.wv, &xn, &mut v);
        ops::rms_norm_heads(&mut q, &self.q_norm, nh, hd, c.eps);
        ops::rms_norm_heads(&mut k, &self.k_norm, nkv, hd, c.eps);
        ops::rope(&mut q, nh, hd, c.n_rot, pos, c.rope_base, 1);
        ops::rope(&mut k, nkv, hd, c.n_rot, pos, c.rope_base, 1);

        // Store K, V in contiguous MTP KV buffer
        let k_base = pos * kv_dim;
        let v_base = self.max_ctx * kv_dim + pos * kv_dim;
        self.kv[k_base..k_base + kv_dim].copy_from_slice(&k);
        self.kv[v_base..v_base + kv_dim].copy_from_slice(&v);

        let group = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn = vec![0f32; nh * hd];
        let mut scores = vec![0f32; pos + 1];
        for h in 0..nh {
            let kvh = h / group;
            let qh = &q[h * hd..(h + 1) * hd];
            let mut mx = f32::NEG_INFINITY;
            for t in 0..=pos {
                let kk = &self.kv[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                let s: f32 = qh.iter().zip(kk).map(|(a, b)| a * b).sum::<f32>() * scale;
                scores[t] = s;
                mx = mx.max(s);
            }
            let mut sum = 0f32;
            for s in &mut scores { *s = (*s - mx).exp(); sum += *s; }
            let oh = &mut attn[h * hd..(h + 1) * hd];
            for t in 0..=pos {
                let w = scores[t] / sum;
                let vv = &self.kv[self.max_ctx * kv_dim + t * kv_dim + kvh * hd..self.max_ctx * kv_dim + t * kv_dim + (kvh + 1) * hd];
                for d in 0..hd { oh[d] += w * vv[d]; }
            }
        }
        ops::mul_sigmoid_inplace(&mut attn, &q_gate);
        let mut o = vec![0f32; ne];
        self.mv(&self.wo, &attn, &mut o);
        ops::add(&mut x, &o);

        ops::rms_norm(&x, &self.post_attn_norm, &mut xn, c.eps);
        let mut gf = vec![0f32; c.n_ff];
        let mut uf = vec![0f32; c.n_ff];
        let mut af = vec![0f32; c.n_ff];
        self.mv(&self.ffn_gate, &xn, &mut gf);
        self.mv(&self.ffn_up, &xn, &mut uf);
        ops::swiglu(&gf, &uf, &mut af);
        self.mv(&self.ffn_down, &af, &mut o);
        ops::add(&mut x, &o);

        let mut h_out = vec![0f32; ne];
        ops::rms_norm(&x, &self.head_norm, &mut h_out, c.eps);
        let mut logits = vec![0f32; c.n_vocab];
        self.mv(&self.head_w, &h_out, &mut logits);
        Ok(logits)
    }
}
