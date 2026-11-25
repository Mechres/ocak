//! Native Qwen3.5 / RVN NextN Multi-Token Prediction (MTP) self-speculative draft head (`blk.{n_layer}.*`).
use crate::config::ModelConfig;
use crate::exec::EngineError;
use ocak_cpu::{gemv, ops};
use ocak_gguf::quant::dequantize_row;
use ocak_gguf::{GgufFile, TensorInfo};
use std::sync::Arc;

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
        }))
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
