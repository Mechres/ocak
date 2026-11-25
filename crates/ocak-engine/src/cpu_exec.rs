use crate::config::{LayerKind, ModelConfig};
use crate::exec::{EngineError, Executor, Out};
use ocak_cpu::{gemv, ops};
use ocak_gguf::quant::dequantize_row;
use ocak_gguf::{GgufFile, TensorInfo};
use std::sync::Arc;

enum CpuMixer {
    FullAttn {
        wq: TensorInfo,
        wk: TensorInfo,
        wv: TensorInfo,
        wo: TensorInfo,
        bq: Option<Vec<f32>>,
        bk: Option<Vec<f32>>,
        bv: Option<Vec<f32>>,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
        gated_q: bool,
        n_head_kv: usize,
        kv_layer_idx: usize,
    },
    ShortConv {
        in_proj: TensorInfo,
        out_proj: TensorInfo,
        conv_w: Vec<f32>,
        l_cache: usize,
        state: Vec<f32>,
    },
    DeltaNet {
        wqkv: TensorInfo,
        wgate: TensorInfo,
        walpha: TensorInfo,
        wbeta: TensorInfo,
        wout: TensorInfo,
        conv_w: Vec<f32>,
        dt_bias: Vec<f32>,
        ssm_a: Vec<f32>,
        ssm_norm: Vec<f32>,
        conv_state: Vec<f32>,
        ssm_state: Vec<f32>,
        n_k_heads: usize,
        n_v_heads: usize,
        state_dim: usize,
        k_conv: usize,
    },
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    mixer: CpuMixer,
    gate: TensorInfo,
    up: TensorInfo,
    down: TensorInfo,
}

pub struct CpuExec {
    g: Arc<GgufFile>,
    cfg: ModelConfig,
    layers: Vec<Layer>,
    embd: TensorInfo,
    out_norm: Vec<f32>,
    output: TensorInfo,
    kv: Vec<f32>,
    layer_stride: usize,
    logits: Vec<f32>,
    hidden: Vec<f32>,
}

fn t(g: &GgufFile, name: &str) -> Result<TensorInfo, EngineError> {
    g.tensor(name).cloned().ok_or_else(|| EngineError::Unsupported(format!("missing tensor {name}")))
}

fn f32t(g: &GgufFile, name: &str) -> Result<Vec<f32>, EngineError> {
    let ti = t(g, name)?;
    Ok(g.tensor_to_f32(&ti))
}

impl CpuExec {
    pub fn new(g: Arc<GgufFile>, cfg: ModelConfig, n_pages: usize) -> Result<Self, EngineError> {
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for (i, &kind) in cfg.layer_kinds.iter().enumerate() {
            let p = |s: &str| format!("blk.{i}.{s}");
            let mixer = match kind {
                LayerKind::FullAttn { n_head_kv, gated_q, qk_norm, kv_layer_idx } => {
                    let bias = |s: &str| if cfg.qkv_bias { f32t(&g, &p(s)).map(Some) } else { Ok(None) };
                    CpuMixer::FullAttn {
                        wq: t(&g, &p("attn_q.weight"))?,
                        wk: t(&g, &p("attn_k.weight"))?,
                        wv: t(&g, &p("attn_v.weight"))?,
                        wo: t(&g, &p("attn_output.weight"))?,
                        bq: bias("attn_q.bias")?,
                        bk: bias("attn_k.bias")?,
                        bv: bias("attn_v.bias")?,
                        q_norm: if qk_norm { Some(f32t(&g, &p("attn_q_norm.weight"))?) } else { None },
                        k_norm: if qk_norm { Some(f32t(&g, &p("attn_k_norm.weight"))?) } else { None },
                        gated_q,
                        n_head_kv,
                        kv_layer_idx,
                    }
                }
                LayerKind::ShortConv { l_cache } => CpuMixer::ShortConv {
                    in_proj: t(&g, &p("shortconv.in_proj.weight"))?,
                    out_proj: t(&g, &p("shortconv.out_proj.weight"))?,
                    conv_w: f32t(&g, &p("shortconv.conv.weight"))?,
                    l_cache,
                    state: vec![0f32; cfg.n_embd * (l_cache - 1)],
                },
                LayerKind::DeltaNet { n_k_heads, n_v_heads, state_dim, k_conv } => {
                    let conv_ch = (2 * n_k_heads + n_v_heads) * state_dim;
                    CpuMixer::DeltaNet {
                        wqkv: t(&g, &p("attn_qkv.weight"))?,
                        wgate: t(&g, &p("attn_gate.weight"))?,
                        walpha: t(&g, &p("ssm_alpha.weight"))?,
                        wbeta: t(&g, &p("ssm_beta.weight"))?,
                        wout: t(&g, &p("ssm_out.weight"))?,
                        conv_w: f32t(&g, &p("ssm_conv1d.weight"))?,
                        dt_bias: f32t(&g, &p("ssm_dt.bias"))?,
                        ssm_a: f32t(&g, &p("ssm_a"))?,
                        ssm_norm: f32t(&g, &p("ssm_norm.weight"))?,
                        conv_state: vec![0f32; conv_ch * (k_conv - 1)],
                        ssm_state: vec![0f32; n_v_heads * state_dim * state_dim],
                        n_k_heads,
                        n_v_heads,
                        state_dim,
                        k_conv,
                    }
                }
            };
            layers.push(Layer {
                attn_norm: f32t(&g, &p("attn_norm.weight"))?,
                ffn_norm: f32t(&g, &p(&cfg.ffn_norm_suffix))?,
                mixer,
                gate: t(&g, &p("ffn_gate.weight"))?,
                up: t(&g, &p("ffn_up.weight"))?,
                down: t(&g, &p("ffn_down.weight"))?,
            });
        }
        let embd = t(&g, "token_embd.weight")?;
        let output = if cfg.tied_output { embd.clone() } else { t(&g, "output.weight")? };
        let layer_stride = n_pages * 2 * ops::PAGE * cfg.kv_dim();
        Ok(Self {
            out_norm: f32t(&g, &cfg.output_norm_name)?,
            kv: vec![0f32; layer_stride * cfg.n_kv_layers],
            layer_stride,
            logits: vec![0.0; cfg.n_vocab],
            hidden: vec![0.0; cfg.n_embd],
            layers,
            embd,
            output,
            cfg,
            g,
        })
    }

    fn mv(&self, ti: &TensorInfo, x: &[f32], y: &mut [f32]) {
        gemv(ti.ty, self.g.tensor_data(ti), x, y, ti.n_cols(), ti.n_rows());
    }
}

impl Executor for CpuExec {
    fn name(&self) -> String { "cpu-avx2".into() }

    fn reset_recurrent(&mut self) -> Result<(), EngineError> {
        for l in &mut self.layers {
            match &mut l.mixer {
                CpuMixer::ShortConv { state, .. } => state.fill(0.0),
                CpuMixer::DeltaNet { conv_state, ssm_state, .. } => {
                    conv_state.fill(0.0);
                    ssm_state.fill(0.0);
                }
                CpuMixer::FullAttn { .. } => {}
            }
        }
        Ok(())
    }

    fn forward(&mut self, token: u32, pos: usize, page_table: &[u32], out: Out) -> Result<(), EngineError> {
        let c = &self.cfg;
        let mut x = vec![0f32; c.n_embd];
        {
            let rb = self.embd.row_bytes();
            let data = self.g.tensor_data(&self.embd);
            let r = token as usize;
            if r >= self.embd.n_rows() {
                return Err(EngineError::Backend(format!("token {token} out of range")));
            }
            dequantize_row(self.embd.ty, &data[r * rb..(r + 1) * rb], &mut x, c.n_embd);
        }
        let mut xn = vec![0f32; c.n_embd];
        let mut o = vec![0f32; c.n_embd];
        let mut gate_ff = vec![0f32; c.n_ff];
        let mut up_ff = vec![0f32; c.n_ff];
        let mut act_ff = vec![0f32; c.n_ff];

        for li in 0..c.n_layer {
            ops::rms_norm(&x, &self.layers[li].attn_norm, &mut xn, c.eps);
            match &mut self.layers[li].mixer {
                CpuMixer::FullAttn { wq, wk, wv, wo, bq, bk, bv, q_norm, k_norm, gated_q, n_head_kv, kv_layer_idx } => {
                    let (hd, n_h, n_kv) = (c.head_dim, c.n_head, *n_head_kv);
                    let kv_dim = n_kv * hd;
                    let mut q = vec![0f32; n_h * hd];
                    let mut q_gate = if *gated_q { vec![0f32; n_h * hd] } else { Vec::new() };
                    if *gated_q {
                        let mut qg = vec![0f32; 2 * n_h * hd];
                        gemv(wq.ty, self.g.tensor_data(wq), &xn, &mut qg, wq.n_cols(), wq.n_rows());
                        for h in 0..n_h {
                            q[h * hd..(h + 1) * hd].copy_from_slice(&qg[h * 2 * hd..h * 2 * hd + hd]);
                            q_gate[h * hd..(h + 1) * hd].copy_from_slice(&qg[(h * 2 + 1) * hd..(h * 2 + 2) * hd]);
                        }
                    } else {
                        gemv(wq.ty, self.g.tensor_data(wq), &xn, &mut q, wq.n_cols(), wq.n_rows());
                    }
                    let mut k = vec![0f32; kv_dim];
                    let mut v = vec![0f32; kv_dim];
                    gemv(wk.ty, self.g.tensor_data(wk), &xn, &mut k, wk.n_cols(), wk.n_rows());
                    gemv(wv.ty, self.g.tensor_data(wv), &xn, &mut v, wv.n_cols(), wv.n_rows());
                    if let Some(b) = bq { ops::add(&mut q, b); }
                    if let Some(b) = bk { ops::add(&mut k, b); }
                    if let Some(b) = bv { ops::add(&mut v, b); }
                    if let Some(qn) = q_norm { ops::rms_norm_heads(&mut q, qn, n_h, hd, c.eps); }
                    if let Some(kn) = k_norm { ops::rms_norm_heads(&mut k, kn, n_kv, hd, c.eps); }
                    ops::rope(&mut q, n_h, hd, c.n_rot, pos, c.rope_base, c.rope_mode);
                    ops::rope(&mut k, n_kv, hd, c.n_rot, pos, c.rope_base, c.rope_mode);
                    let off = *kv_layer_idx * self.layer_stride;
                    ops::kv_write(&k, &v, &mut self.kv, off, page_table, pos, kv_dim);
                    let mut attn = vec![0f32; n_h * hd];
                    ops::attention(&q, &self.kv, off, page_table, &mut attn, n_h, n_kv, hd, pos);
                    if *gated_q {
                        ops::mul_sigmoid_inplace(&mut attn, &q_gate);
                    }
                    gemv(wo.ty, self.g.tensor_data(wo), &attn, &mut o, wo.n_cols(), wo.n_rows());
                }
                CpuMixer::ShortConv { in_proj, out_proj, conv_w, l_cache, state } => {
                    let mut bcx = vec![0f32; 3 * c.n_embd];
                    gemv(in_proj.ty, self.g.tensor_data(in_proj), &xn, &mut bcx, in_proj.n_cols(), in_proj.n_rows());
                    let mut sc = vec![0f32; c.n_embd];
                    ops::shortconv_step(&bcx, conv_w, state, &mut sc, c.n_embd, *l_cache);
                    gemv(out_proj.ty, self.g.tensor_data(out_proj), &sc, &mut o, out_proj.n_cols(), out_proj.n_rows());
                }
                CpuMixer::DeltaNet {
                    wqkv, wgate, walpha, wbeta, wout, conv_w, dt_bias, ssm_a, ssm_norm,
                    conv_state, ssm_state, n_k_heads, n_v_heads, state_dim, k_conv,
                } => {
                    let conv_ch = (2 * *n_k_heads + *n_v_heads) * *state_dim;
                    let v_dim = *n_v_heads * *state_dim;
                    let mut qkv_raw = vec![0f32; conv_ch];
                    let mut z = vec![0f32; v_dim];
                    let mut alpha = vec![0f32; *n_v_heads];
                    let mut beta = vec![0f32; *n_v_heads];
                    gemv(wqkv.ty, self.g.tensor_data(wqkv), &xn, &mut qkv_raw, wqkv.n_cols(), wqkv.n_rows());
                    gemv(wgate.ty, self.g.tensor_data(wgate), &xn, &mut z, wgate.n_cols(), wgate.n_rows());
                    gemv(walpha.ty, self.g.tensor_data(walpha), &xn, &mut alpha, walpha.n_cols(), walpha.n_rows());
                    gemv(wbeta.ty, self.g.tensor_data(wbeta), &xn, &mut beta, wbeta.n_cols(), wbeta.n_rows());
                    let mut qkv_conv = vec![0f32; conv_ch];
                    ops::ssm_conv_silu(&qkv_raw, conv_w, conv_state, &mut qkv_conv, conv_ch, *k_conv);
                    let mut dn_out = vec![0f32; v_dim];
                    ops::deltanet_step(&qkv_conv, &alpha, &beta, dt_bias, ssm_a, ssm_state, &mut dn_out, *n_k_heads, *n_v_heads, *state_dim, c.eps);
                    let mut dn_norm = vec![0f32; v_dim];
                    ops::gated_rms_norm_heads(&dn_out, ssm_norm, &z, &mut dn_norm, *n_v_heads, *state_dim, c.eps);
                    gemv(wout.ty, self.g.tensor_data(wout), &dn_norm, &mut o, wout.n_cols(), wout.n_rows());
                }
            }
            ops::add(&mut x, &o);
            ops::rms_norm(&x, &self.layers[li].ffn_norm, &mut xn, c.eps);
            self.mv(&self.layers[li].gate, &xn, &mut gate_ff);
            self.mv(&self.layers[li].up, &xn, &mut up_ff);
            ops::swiglu(&gate_ff, &up_ff, &mut act_ff);
            self.mv(&self.layers[li].down, &act_ff, &mut o);
            ops::add(&mut x, &o);
        }
        if out.logits || out.hidden {
            ops::rms_norm(&x, &self.out_norm, &mut self.hidden, c.eps);
        }
        if out.logits {
            let h = self.hidden.clone();
            let output = self.output.clone();
            let mut lg = std::mem::take(&mut self.logits);
            self.mv(&output, &h, &mut lg);
            self.logits = lg;
        }
        Ok(())
    }

    fn logits(&self) -> &[f32] { &self.logits }
    fn hidden(&self) -> &[f32] { &self.hidden }
}
