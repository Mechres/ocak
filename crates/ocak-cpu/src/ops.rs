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
    for h in 0..n_heads {
        let hv = &mut v[h * head_dim..(h + 1) * head_dim];
        for i in 0..n_rot / 2 {
            let theta = pos as f32 * freq_base.powf(-2.0 * i as f32 / n_rot as f32);
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
    let kv_dim = n_kv_heads * head_dim;
    let group = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0f32; pos + 1];
    for h in 0..n_heads {
        let kvh = h / group;
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut mx = f32::NEG_INFINITY;
        for t in 0..=pos {
            let page = page_table[t / PAGE] as usize;
            let o = layer_off + kv_index(page, 0, t % PAGE, kv_dim) + kvh * head_dim;
            let kk = &kv[o..o + head_dim];
            let s: f32 = qh.iter().zip(kk).map(|(a, b)| a * b).sum::<f32>() * scale;
            scores[t] = s;
            mx = mx.max(s);
        }
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }
        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        oh.fill(0.0);
        for t in 0..=pos {
            let page = page_table[t / PAGE] as usize;
            let o = layer_off + kv_index(page, 1, t % PAGE, kv_dim) + kvh * head_dim;
            let w = scores[t] / sum;
            for d in 0..head_dim {
                oh[d] += w * kv[o + d];
            }
        }
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
