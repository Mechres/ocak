use crate::config::ModelConfig;
use crate::exec::{EngineError, Executor, Out};
use ocak_cpu::{gemv, ops};
use ocak_gguf::quant::dequantize_row;
use ocak_gguf::{GgufFile, TensorInfo};
use std::sync::Arc;

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    wq: TensorInfo,
    wk: TensorInfo,
    wv: TensorInfo,
    wo: TensorInfo,
    gate: TensorInfo,
    up: TensorInfo,
    down: TensorInfo,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
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
    x: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    o: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
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
        let mut layers = Vec::new();
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let bias = |s: &str| if cfg.qkv_bias { f32t(&g, &p(s)).map(Some) } else { Ok(None) };
            layers.push(Layer {
                attn_norm: f32t(&g, &p("attn_norm.weight"))?,
                ffn_norm: f32t(&g, &p("ffn_norm.weight"))?,
                wq: t(&g, &p("attn_q.weight"))?,
                wk: t(&g, &p("attn_k.weight"))?,
                wv: t(&g, &p("attn_v.weight"))?,
                wo: t(&g, &p("attn_output.weight"))?,
                gate: t(&g, &p("ffn_gate.weight"))?,
                up: t(&g, &p("ffn_up.weight"))?,
                down: t(&g, &p("ffn_down.weight"))?,
                bq: bias("attn_q.bias")?,
                bk: bias("attn_k.bias")?,
                bv: bias("attn_v.bias")?,
            });
        }
        let embd = t(&g, "token_embd.weight")?;
        let output = if cfg.tied_output { embd.clone() } else { t(&g, "output.weight")? };
        let layer_stride = n_pages * 2 * ops::PAGE * cfg.kv_dim();
        let qd = cfg.n_head * cfg.head_dim;
        Ok(Self {
            out_norm: f32t(&g, "output_norm.weight")?,
            kv: vec![0f32; layer_stride * cfg.n_layer],
            layer_stride,
            x: vec![0.0; cfg.n_embd],
            xn: vec![0.0; cfg.n_embd],
            q: vec![0.0; qd],
            k: vec![0.0; cfg.kv_dim()],
            v: vec![0.0; cfg.kv_dim()],
            attn: vec![0.0; qd],
            o: vec![0.0; cfg.n_embd],
            gate: vec![0.0; cfg.n_ff],
            up: vec![0.0; cfg.n_ff],
            act: vec![0.0; cfg.n_ff],
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
    fn name(&self) -> String {
        "cpu-avx2".into()
    }

    fn forward(&mut self, token: u32, pos: usize, page_table: &[u32], out: Out) -> Result<(), EngineError> {
        let c = self.cfg.clone();
        let (hd, kv_dim) = (c.head_dim, c.kv_dim());
        // embedding row
        {
            let rb = self.embd.row_bytes();
            let data = self.g.tensor_data(&self.embd);
            let r = token as usize;
            if r >= self.embd.n_rows() {
                return Err(EngineError::Backend(format!("token {token} out of range")));
            }
            dequantize_row(self.embd.ty, &data[r * rb..(r + 1) * rb], &mut self.x, c.n_embd);
        }
        for li in 0..c.n_layer {
            let (wq, wk, wv, wo) = (self.layers[li].wq.clone(), self.layers[li].wk.clone(), self.layers[li].wv.clone(), self.layers[li].wo.clone());
            ops::rms_norm(&self.x, &self.layers[li].attn_norm, &mut self.xn, c.eps);
            let xn = self.xn.clone();
            let (mut q, mut k, mut v) = (std::mem::take(&mut self.q), std::mem::take(&mut self.k), std::mem::take(&mut self.v));
            self.mv(&wq, &xn, &mut q);
            self.mv(&wk, &xn, &mut k);
            self.mv(&wv, &xn, &mut v);
            if let Some(b) = &self.layers[li].bq { ops::add(&mut q, b); }
            if let Some(b) = &self.layers[li].bk { ops::add(&mut k, b); }
            if let Some(b) = &self.layers[li].bv { ops::add(&mut v, b); }
            ops::rope(&mut q, c.n_head, hd, c.n_rot, pos, c.rope_base, c.rope_mode);
            ops::rope(&mut k, c.n_head_kv, hd, c.n_rot, pos, c.rope_base, c.rope_mode);
            let off = li * self.layer_stride;
            ops::kv_write(&k, &v, &mut self.kv, off, page_table, pos, kv_dim);
            ops::attention(&q, &self.kv, off, page_table, &mut self.attn, c.n_head, c.n_head_kv, hd, pos);
            self.q = q;
            self.k = k;
            self.v = v;
            let attn = self.attn.clone();
            let mut o = std::mem::take(&mut self.o);
            self.mv(&wo, &attn, &mut o);
            ops::add(&mut self.x, &o);
            ops::rms_norm(&self.x, &self.layers[li].ffn_norm, &mut self.xn, c.eps);
            let xn = self.xn.clone();
            let (gt, up, dn) = (self.layers[li].gate.clone(), self.layers[li].up.clone(), self.layers[li].down.clone());
            let (mut gate, mut upv, mut act) = (std::mem::take(&mut self.gate), std::mem::take(&mut self.up), std::mem::take(&mut self.act));
            self.mv(&gt, &xn, &mut gate);
            self.mv(&up, &xn, &mut upv);
            ops::swiglu(&gate, &upv, &mut act);
            self.mv(&dn, &act, &mut o);
            ops::add(&mut self.x, &o);
            self.gate = gate;
            self.up = upv;
            self.act = act;
            self.o = o;
        }
        if out.logits || out.hidden {
            ops::rms_norm(&self.x, &self.out_norm, &mut self.hidden, c.eps);
        }
        if out.logits {
            let h = self.hidden.clone();
            let mut lg = std::mem::take(&mut self.logits);
            let output = self.output.clone();
            self.mv(&output, &h, &mut lg);
            self.logits = lg;
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
