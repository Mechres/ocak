//! Transformer elementwise ops and paged GQA attention on the CPU.
pub const PAGE: usize = 16;

pub fn rms_norm(x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let s = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * s * w[i];
    }
}

pub fn add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..gate.len() {
        let g = gate[i];
        out[i] = g / (1.0 + (-g).exp()) * up[i];
    }
}

/// Rotates the first `n_rot` dims of each head. mode 0: adjacent pairs, mode 1: split halves.
pub fn rope(v: &mut [f32], n_heads: usize, head_dim: usize, n_rot: usize, pos: usize, freq_base: f32, mode: u32) {
    rope_ext(v, None, n_heads, head_dim, n_rot, pos, freq_base, mode);
}

pub fn rope_ext(v: &mut [f32], factors: Option<&[f32]>, n_heads: usize, head_dim: usize, n_rot: usize, pos: usize, freq_base: f32, mode: u32) {
    for h in 0..n_heads {
        let hv = &mut v[h * head_dim..(h + 1) * head_dim];
        for i in 0..n_rot / 2 {
            let inv_f = factors.map(|f| 1.0 / f[i]).unwrap_or(1.0);
            let theta = pos as f32 * freq_base.powf(-2.0 * i as f32 / n_rot as f32) * inv_f;
            let (s, c) = theta.sin_cos();
            let (a, b) = if mode == 0 { (2 * i, 2 * i + 1) } else { (i, i + n_rot / 2) };
            let (x0, x1) = (hv[a], hv[b]);
            hv[a] = x0 * c - x1 * s;
            hv[b] = x0 * s + x1 * c;
        }
    }
}

#[inline]
pub fn kv_index(page: usize, kv: usize, slot: usize, kv_dim: usize) -> usize {
    ((page * 2 + kv) * PAGE + slot) * kv_dim
}

/// Writes one token's K and V into the paged cache slot of `pos`. `layer_off` is in f32 elements.
pub fn kv_write(k: &[f32], v: &[f32], kv: &mut [f32], layer_off: usize, page_table: &[u32], pos: usize, kv_dim: usize) {
    let page = page_table[pos / PAGE] as usize;
    let slot = pos % PAGE;
    let ko = layer_off + kv_index(page, 0, slot, kv_dim);
    let vo = layer_off + kv_index(page, 1, slot, kv_dim);
    kv[ko..ko + kv_dim].copy_from_slice(&k[..kv_dim]);
    kv[vo..vo + kv_dim].copy_from_slice(&v[..kv_dim]);
}

/// Single-token GQA attention over keys 0..=pos.
pub fn attention(q: &[f32], kv: &[f32], layer_off: usize, page_table: &[u32], out: &mut [f32], n_heads: usize, n_kv_heads: usize, head_dim: usize, pos: usize) {
    attention_swa(q, kv, layer_off, page_table, out, n_heads, n_kv_heads, head_dim, pos, 0);
}

pub fn attention_swa(q: &[f32], kv: &[f32], layer_off: usize, page_table: &[u32], out: &mut [f32], n_heads: usize, n_kv_heads: usize, head_dim: usize, pos: usize, sliding_window: usize) {
    let kv_dim = n_kv_heads * head_dim;
    let group = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let t_start = if sliding_window > 0 && pos >= sliding_window { pos + 1 - sliding_window } else { 0 };
    let mut scores = vec![0f32; pos + 1];
    for h in 0..n_heads {
        let kvh = h / group;
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut mx = f32::NEG_INFINITY;
        for t in t_start..=pos {
            let page = page_table[t / PAGE] as usize;
            let o = layer_off + kv_index(page, 0, t % PAGE, kv_dim) + kvh * head_dim;
            let kk = &kv[o..o + head_dim];
            let s: f32 = qh.iter().zip(kk).map(|(a, b)| a * b).sum::<f32>() * scale;
            scores[t] = s;
            mx = mx.max(s);
        }
        let mut sum = 0f32;
        for s in scores[t_start..=pos].iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }
        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        oh.fill(0.0);
        for t in t_start..=pos {
            let page = page_table[t / PAGE] as usize;
            let o = layer_off + kv_index(page, 1, t % PAGE, kv_dim) + kvh * head_dim;
            let w = scores[t] / sum;
            for d in 0..head_dim {
                oh[d] += w * kv[o + d];
            }
        }
    }
}


pub fn rms_norm_heads(x: &mut [f32], w: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let sl = &mut x[h * head_dim..(h + 1) * head_dim];
        let ss: f32 = sl.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let sc = 1.0 / (ss + eps).sqrt();
        for d in 0..head_dim { sl[d] = sl[d] * sc * w[d]; }
    }
}

/// LFM2 ShortConv: bcx has [B (E), C (E), X (E)]; state has [E, d_conv] where d_conv = l_cache - 1;
/// conv_w has [E, l_cache] (ne[0] = l_cache contiguous per channel).
pub fn shortconv_step(bcx: &[f32], conv_w: &[f32], state: &mut [f32], out: &mut [f32], n_embd: usize, l_cache: usize) {
    let d_conv = l_cache - 1;
    let b = &bcx[0..n_embd];
    let c = &bcx[n_embd..2 * n_embd];
    let x = &bcx[2 * n_embd..3 * n_embd];
    for e in 0..n_embd {
        let bx_cur = b[e] * x[e];
        let st = &mut state[e * d_conv..(e + 1) * d_conv];
        let w = &conv_w[e * l_cache..(e + 1) * l_cache];
        let mut acc = 0f32;
        for t in 0..d_conv { acc += st[t] * w[t]; }
        acc += bx_cur * w[d_conv];
        // shift left, append bx_cur
        for t in 0..d_conv - 1 { st[t] = st[t + 1]; }
        st[d_conv - 1] = bx_cur;
        out[e] = c[e] * acc;
    }
}

/// Qwen3.5 SSM causal depthwise 1D conv + SiLU over `channels` channels, kernel size `k_conv` (4).
pub fn ssm_conv_silu(x: &[f32], conv_w: &[f32], state: &mut [f32], out: &mut [f32], channels: usize, k_conv: usize) {
    let d_conv = k_conv - 1;
    for ch in 0..channels {
        let st = &mut state[ch * d_conv..(ch + 1) * d_conv];
        let w = &conv_w[ch * k_conv..(ch + 1) * k_conv];
        let mut acc = 0f32;
        for t in 0..d_conv { acc += st[t] * w[t]; }
        acc += x[ch] * w[d_conv];
        for t in 0..d_conv - 1 { st[t] = st[t + 1]; }
        st[d_conv - 1] = x[ch];
        out[ch] = acc / (1.0 + (-acc).exp());
    }
}

/// Qwen3.5 Gated DeltaNet autoregressive step.
/// `qkv` = [Q (n_k_heads*dk), K (n_k_heads*dk), V (n_v_heads*dv)], dk == dv == head_dim.
/// `state` = [n_v_heads, dv, dk] (row j = v_dim, col i = k_dim).
pub fn deltanet_step(
    qkv: &[f32],
    alpha: &[f32],
    beta: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    n_k_heads: usize,
    n_v_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let k_total = n_k_heads * head_dim;
    let q_all = &qkv[0..k_total];
    let k_all = &qkv[k_total..2 * k_total];
    let v_all = &qkv[2 * k_total..2 * k_total + n_v_heads * head_dim];
    let q_scale = 1.0 / (head_dim as f32).sqrt();

    let mut q_norm = vec![0f32; head_dim];
    let mut k_norm = vec![0f32; head_dim];
    for h in 0..n_v_heads {
        let kh = h % n_k_heads;
        let qh = &q_all[kh * head_dim..(kh + 1) * head_dim];
        let kh_v = &k_all[kh * head_dim..(kh + 1) * head_dim];
        let q_inv = 1.0 / (qh.iter().map(|v| v * v).sum::<f32>() + eps).sqrt() * q_scale;
        let k_inv = 1.0 / (kh_v.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
        for i in 0..head_dim {
            q_norm[i] = qh[i] * q_inv;
            k_norm[i] = kh_v[i] * k_inv;
        }
        let ab = alpha[h] + dt_bias[h];
        let sp = if ab > 20.0 { ab } else { (1.0 + ab.exp()).ln() };
        let g = (sp * ssm_a[h]).exp();
        let b = 1.0 / (1.0 + (-beta[h]).exp());

        let sh = &mut state[h * head_dim * head_dim..(h + 1) * head_dim * head_dim];
        let vh = &v_all[h * head_dim..(h + 1) * head_dim];
        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        for j in 0..head_dim {
            let row = &mut sh[j * head_dim..(j + 1) * head_dim];
            let mut sk = 0f32;
            for i in 0..head_dim {
                row[i] *= g;
                sk += row[i] * k_norm[i];
            }
            let dj = (vh[j] - sk) * b;
            let mut o_j = 0f32;
            for i in 0..head_dim {
                row[i] += k_norm[i] * dj;
                o_j += row[i] * q_norm[i];
            }
            oh[j] = o_j;
        }
    }
}

pub fn gated_rms_norm_heads(x: &[f32], w: &[f32], z: &[f32], out: &mut [f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let base = h * head_dim;
        let ss: f32 = x[base..base + head_dim].iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let sc = 1.0 / (ss + eps).sqrt();
        for d in 0..head_dim {
            let zv = z[base + d];
            let silu = zv / (1.0 + (-zv).exp());
            out[base + d] = x[base + d] * sc * w[d] * silu;
        }
    }
}

pub fn mul_head_sigmoid_inplace(x: &mut [f32], gate: &[f32], n_heads: usize, head_dim: usize) {
    for h in 0..n_heads {
        let s = 1.0 / (1.0 + (-gate[h]).exp());
        for d in 0..head_dim { x[h * head_dim + d] *= s; }
    }
}

pub fn mul_sigmoid_inplace(x: &mut [f32], gate: &[f32]) {
    for (a, &g) in x.iter_mut().zip(gate) {
        *a *= 1.0 / (1.0 + (-g).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rms_norm_unit() {
        let x = [3.0f32, 4.0];
        let mut o = [0.0; 2];
        rms_norm(&x, &[1.0, 1.0], &mut o, 0.0);
        let r = (12.5f32).sqrt();
        assert!((o[0] - 3.0 / r).abs() < 1e-6 && (o[1] - 4.0 / r).abs() < 1e-6);
    }
    #[test]
    fn rope_pos0_identity_and_norm_preserved() {
        let mut v: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.5).collect();
        let orig = v.clone();
        rope(&mut v, 2, 8, 8, 0, 10000.0, 0);
        assert_eq!(v, orig);
        rope(&mut v, 2, 8, 6, 5, 10000.0, 1);
        let n0: f32 = orig.iter().map(|a| a * a).sum();
        let n1: f32 = v.iter().map(|a| a * a).sum();
        assert!((n0 - n1).abs() < 1e-4);
        assert_eq!(v[6], orig[6]); // dims beyond n_rot untouched
    }
    #[test]
    fn attention_single_key_returns_value() {
        let (nh, nkv, hd) = (4, 2, 4);
        let kv_dim = nkv * hd;
        let mut kv = vec![0f32; 2 * 2 * PAGE * kv_dim * 3];
        let k: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..kv_dim).map(|i| 10.0 + i as f32).collect();
        let pt = [2u32];
        kv_write(&k, &v, &mut kv, 0, &pt, 0, kv_dim);
        let q = vec![1.0f32; nh * hd];
        let mut out = vec![0f32; nh * hd];
        attention(&q, &kv, 0, &pt, &mut out, nh, nkv, hd, 0);
        for h in 0..nh {
            let kvh = h / (nh / nkv);
            for d in 0..hd {
                assert!((out[h * hd + d] - v[kvh * hd + d]).abs() < 1e-6);
            }
        }
    }
}
