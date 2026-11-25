//! y[m] = W[m,k] . x[k] with W in GGML block layouts, f32 activations.
use half::{bf16, f16};
use ocak_gguf::quant::get_scale_min_k4;
use ocak_gguf::GgmlType;
use rayon::prelude::*;

#[inline(always)]
fn h(b: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

#[inline(always)]
fn dot_f32(w: &[u8], x: &[f32]) -> f32 {
    let mut a = [0f32; 8];
    for (c, xc) in w.chunks_exact(32).zip(x.chunks_exact(8)) {
        for i in 0..8 {
            a[i] += f32::from_le_bytes(c[4 * i..4 * i + 4].try_into().unwrap()) * xc[i];
        }
    }
    a.iter().sum()
}

#[inline(always)]
fn dot_f16(w: &[u8], x: &[f32]) -> f32 {
    let mut a = [0f32; 8];
    for (c, xc) in w.chunks_exact(16).zip(x.chunks_exact(8)) {
        for i in 0..8 {
            a[i] += f16::from_le_bytes([c[2 * i], c[2 * i + 1]]).to_f32() * xc[i];
        }
    }
    a.iter().sum()
}

#[inline(always)]
fn dot_bf16(w: &[u8], x: &[f32]) -> f32 {
    let mut a = [0f32; 8];
    for (c, xc) in w.chunks_exact(16).zip(x.chunks_exact(8)) {
        for i in 0..8 {
            a[i] += bf16::from_le_bytes([c[2 * i], c[2 * i + 1]]).to_f32() * xc[i];
        }
    }
    a.iter().sum()
}

#[inline(always)]
fn dot_q4_0(w: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (b, xb) in w.chunks_exact(18).zip(x.chunks_exact(32)) {
        let d = h(b, 0);
        let mut s = [0f32; 16];
        for j in 0..16 {
            let lo = (b[2 + j] & 0xF) as i32 - 8;
            let hi = (b[2 + j] >> 4) as i32 - 8;
            s[j] = lo as f32 * xb[j] + hi as f32 * xb[j + 16];
        }
        acc += d * s.iter().sum::<f32>();
    }
    acc
}

#[inline(always)]
fn dot_q8_0(w: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (b, xb) in w.chunks_exact(34).zip(x.chunks_exact(32)) {
        let d = h(b, 0);
        let mut s = [0f32; 32];
        for j in 0..32 {
            s[j] = (b[2 + j] as i8) as f32 * xb[j];
        }
        acc += d * s.iter().sum::<f32>();
    }
    acc
}

#[inline(always)]
fn dot_q2k(w: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (b, xb) in w.chunks_exact(84).zip(x.chunks_exact(256)) {
        let scales = &b[0..16];
        let qs = &b[16..80];
        let (d, dmin) = (h(b, 80), h(b, 82));
        let mut is = 0usize;
        let mut xi = 0usize;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            let mut shift = 0u8;
            for _ in 0..4 {
                let sc = scales[is]; is += 1;
                let (dl, ml) = (d * (sc & 0xF) as f32, dmin * (sc >> 4) as f32);
                let (mut s1, mut sx1) = (0f32, 0f32);
                for l in 0..16 { s1 += ((q[l] >> shift) & 3) as f32 * xb[xi + l]; sx1 += xb[xi + l]; }
                acc += dl * s1 - ml * sx1; xi += 16;
                let sc = scales[is]; is += 1;
                let (dl, ml) = (d * (sc & 0xF) as f32, dmin * (sc >> 4) as f32);
                let (mut s2, mut sx2) = (0f32, 0f32);
                for l in 0..16 { s2 += ((q[l + 16] >> shift) & 3) as f32 * xb[xi + l]; sx2 += xb[xi + l]; }
                acc += dl * s2 - ml * sx2; xi += 16;
                shift += 2;
            }
        }
    }
    acc
}

#[inline(always)]
fn dot_q3k(w: &[u8], x: &[f32]) -> f32 {
    const KM1: u32 = 0x03030303;
    const KM2: u32 = 0x0f0f0f0f;
    let mut acc = 0f32;
    for (b, xb) in w.chunks_exact(110).zip(x.chunks_exact(256)) {
        let hm = &b[0..32];
        let qs = &b[32..96];
        let mut aux = [
            u32::from_le_bytes(b[96..100].try_into().unwrap()),
            u32::from_le_bytes(b[100..104].try_into().unwrap()),
            u32::from_le_bytes(b[104..108].try_into().unwrap()),
            0u32,
        ];
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KM2) | (((tmp >> 4) & KM1) << 4);
        aux[3] = ((aux[1] >> 4) & KM2) | (((tmp >> 6) & KM1) << 4);
        aux[0] = (aux[0] & KM2) | ((tmp & KM1) << 4);
        aux[1] = (aux[1] & KM2) | (((tmp >> 2) & KM1) << 4);
        let mut scales = [0i8; 16];
        for k in 0..4 {
            let bytes = aux[k].to_le_bytes();
            for j in 0..4 { scales[k * 4 + j] = bytes[j] as i8; }
        }
        let d_all = h(b, 108);
        let mut is = 0usize;
        let mut m = 1u8;
        let mut xi = 0usize;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            let mut shift = 0u8;
            for _ in 0..4 {
                let dl = d_all * (scales[is] as i32 - 32) as f32; is += 1;
                let mut s1 = 0f32;
                for l in 0..16 {
                    let v = ((q[l] >> shift) & 3) as i32 - if hm[l] & m != 0 { 0 } else { 4 };
                    s1 += v as f32 * xb[xi + l];
                }
                acc += dl * s1; xi += 16;
                let dl = d_all * (scales[is] as i32 - 32) as f32; is += 1;
                let mut s2 = 0f32;
                for l in 0..16 {
                    let v = ((q[l + 16] >> shift) & 3) as i32 - if hm[l + 16] & m != 0 { 0 } else { 4 };
                    s2 += v as f32 * xb[xi + l];
                }
                acc += dl * s2; xi += 16;
                shift += 2;
                m <<= 1;
            }
        }
    }
    acc
}

#[inline(always)]
fn dot_q4k(w: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (b, xb) in w.chunks_exact(144).zip(x.chunks_exact(256)) {
        let (d, dmin) = (h(b, 0), h(b, 2));
        let scales = &b[4..16];
        let qs = &b[16..144];
        for g in 0..4 {
            let q = &qs[g * 32..g * 32 + 32];
            let xg = &xb[g * 64..g * 64 + 64];
            let (sc1, m1) = get_scale_min_k4(2 * g, scales);
            let (sc2, m2) = get_scale_min_k4(2 * g + 1, scales);
            let (mut lo, mut hi, mut sx1, mut sx2) = (0f32, 0f32, 0f32, 0f32);
            for l in 0..32 {
                lo += (q[l] & 0xF) as f32 * xg[l];
                hi += (q[l] >> 4) as f32 * xg[32 + l];
                sx1 += xg[l];
                sx2 += xg[32 + l];
            }
            acc += d * sc1 as f32 * lo - dmin * m1 as f32 * sx1;
            acc += d * sc2 as f32 * hi - dmin * m2 as f32 * sx2;
        }
    }
    acc
}

#[inline(always)]
fn dot_q5k(w: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (b, xb) in w.chunks_exact(176).zip(x.chunks_exact(256)) {
        let (d, dmin) = (h(b, 0), h(b, 2));
        let scales = &b[4..16];
        let qh = &b[16..48];
        let qs = &b[48..176];
        let (mut u1, mut u2) = (1u8, 2u8);
        for g in 0..4 {
            let q = &qs[g * 32..g * 32 + 32];
            let xg = &xb[g * 64..g * 64 + 64];
            let (sc1, m1) = get_scale_min_k4(2 * g, scales);
            let (sc2, m2) = get_scale_min_k4(2 * g + 1, scales);
            let (mut lo, mut hi, mut sx1, mut sx2) = (0f32, 0f32, 0f32, 0f32);
            for l in 0..32 {
                let a = (q[l] & 0xF) as i32 + if qh[l] & u1 != 0 { 16 } else { 0 };
                let c = (q[l] >> 4) as i32 + if qh[l] & u2 != 0 { 16 } else { 0 };
                lo += a as f32 * xg[l];
                hi += c as f32 * xg[32 + l];
                sx1 += xg[l];
                sx2 += xg[32 + l];
            }
            acc += d * sc1 as f32 * lo - dmin * m1 as f32 * sx1;
            acc += d * sc2 as f32 * hi - dmin * m2 as f32 * sx2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    acc
}

#[inline(always)]
fn dot_q6k(w: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (b, xb) in w.chunks_exact(210).zip(x.chunks_exact(256)) {
        let d = h(b, 208);
        for half in 0..2 {
            let ql = &b[half * 64..half * 64 + 64];
            let qh = &b[128 + half * 32..128 + half * 32 + 32];
            let sc = &b[192 + half * 8..192 + half * 8 + 8];
            let xh = &xb[half * 128..half * 128 + 128];
            let mut s = [0f32; 4];
            let mut t = [[0f32; 16]; 8];
            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32) - 32;
                let q2 = (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32) - 32;
                // accumulate per (scale index, quarter)
                t[is][l % 16] += q1 as f32 * xh[l];
                t[2 + is][l % 16] += q2 as f32 * xh[l + 32];
                t[4 + is][l % 16] += q3 as f32 * xh[l + 64];
                t[6 + is][l % 16] += q4 as f32 * xh[l + 96];
            }
            for i in 0..8 {
                s[i % 4] += (sc[i] as i8) as f32 * t[i].iter().sum::<f32>();
            }
            acc += d * s.iter().sum::<f32>();
        }
    }
    acc
}

#[inline(always)]
fn dot_iq_generic(ty: GgmlType, w: &[u8], x: &[f32]) -> f32 {
    let (be, bb) = ty.block().unwrap();
    let mut tmp = [0f32; 256];
    let mut acc = 0f32;
    for (b, xb) in w.chunks_exact(bb).zip(x.chunks_exact(be)) {
        ocak_gguf::quant::dequantize_row(ty, b, &mut tmp[..be], be);
        let mut s = 0f32;
        for i in 0..be { s += tmp[i] * xb[i]; }
        acc += s;
    }
    acc
}

macro_rules! dispatch_row {
    ($ty:expr, $w:expr, $x:expr) => {
        match $ty {
            GgmlType::F32 => dot_f32($w, $x),
            GgmlType::F16 => dot_f16($w, $x),
            GgmlType::BF16 => dot_bf16($w, $x),
            GgmlType::Q4_0 => dot_q4_0($w, $x),
            GgmlType::Q8_0 => dot_q8_0($w, $x),
            GgmlType::Q2K => dot_q2k($w, $x),
            GgmlType::Q3K => dot_q3k($w, $x),
            GgmlType::Q4K => dot_q4k($w, $x),
            GgmlType::Q5K => dot_q5k($w, $x),
            GgmlType::Q6K => dot_q6k($w, $x),
            GgmlType::IQ2XS | GgmlType::IQ3XXS | GgmlType::IQ3S | GgmlType::IQ4NL | GgmlType::IQ4XS => dot_iq_generic($ty, $w, $x),
            t => panic!("cpu gemv: unsupported type {}", t.name()),
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_dot_avx2(ty: GgmlType, w: &[u8], x: &[f32]) -> f32 {
    dispatch_row!(ty, w, x)
}

fn row_dot_scalar(ty: GgmlType, w: &[u8], x: &[f32]) -> f32 {
    dispatch_row!(ty, w, x)
}

pub fn supported(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::IQ2XS | GgmlType::IQ3XXS | GgmlType::IQ3S | GgmlType::IQ4NL | GgmlType::IQ4XS)
}

/// Dot product of one quantised row (`k` elements) with `x`.
pub fn row_dot(ty: GgmlType, w: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c") {
        return unsafe { row_dot_avx2(ty, w, x) };
    }
    row_dot_scalar(ty, w, x)
}

/// y[r] = dot(W row r, x) for r in 0..m. `w` holds m rows of `row_bytes(k)` bytes.
pub fn gemv(ty: GgmlType, w: &[u8], x: &[f32], y: &mut [f32], k: usize, m: usize) {
    let rb = ty.row_bytes(k).expect("k not a multiple of the block size");
    assert!(w.len() >= rb * m && y.len() >= m && x.len() >= k);
    let x = &x[..k];
    y[..m].par_chunks_mut(8).enumerate().for_each(|(ci, ys)| {
        for (i, yv) in ys.iter_mut().enumerate() {
            let r = ci * 8 + i;
            *yv = row_dot(ty, &w[r * rb..(r + 1) * rb], x);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use ocak_gguf::quant::dequantize_row;

    pub struct Lcg(pub u64);
    impl Lcg {
        pub fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        pub fn f(&mut self) -> f32 {
            (self.next() as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
        }
    }

    fn scale_offsets(ty: GgmlType) -> &'static [usize] {
        match ty {
            GgmlType::Q4_0 | GgmlType::Q8_0 => &[0],
            GgmlType::Q4K | GgmlType::Q5K => &[0, 2],
            GgmlType::Q2K => &[80, 82],
            GgmlType::Q3K => &[108],
            GgmlType::Q6K => &[208],
            GgmlType::IQ2XS | GgmlType::IQ3XXS | GgmlType::IQ3S | GgmlType::IQ4NL | GgmlType::IQ4XS => &[0],
            _ => &[],
        }
    }

    pub fn random_rows(ty: GgmlType, k: usize, m: usize, rng: &mut Lcg) -> Vec<u8> {
        let (be, bb) = ty.block().unwrap();
        let nblk = k / be * m;
        let mut v: Vec<u8> = (0..nblk * bb).map(|_| rng.next() as u8).collect();
        if matches!(ty, GgmlType::F32) {
            for c in v.chunks_exact_mut(4) {
                c.copy_from_slice(&rng.f().to_le_bytes());
            }
        } else if matches!(ty, GgmlType::F16) {
            for c in v.chunks_exact_mut(2) {
                c.copy_from_slice(&f16::from_f32(rng.f()).to_le_bytes());
            }
        } else if matches!(ty, GgmlType::BF16) {
            for c in v.chunks_exact_mut(2) {
                c.copy_from_slice(&half::bf16::from_f32(rng.f()).to_le_bytes());
            }
        } else {
            for blk in v.chunks_exact_mut(bb) {
                for &o in scale_offsets(ty) {
                    blk[o..o + 2].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
                }
            }
        }
        v
    }

    #[test]
    fn gemv_matches_dequant_reference_for_all_types() {
        let mut rng = Lcg(42);
        for ty in [GgmlType::F32, GgmlType::F16, GgmlType::BF16, GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q2K, GgmlType::Q3K, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K, GgmlType::IQ2XS, GgmlType::IQ3XXS, GgmlType::IQ3S, GgmlType::IQ4NL, GgmlType::IQ4XS] {
            let (k, m) = (512usize, 37usize);
            let w = random_rows(ty, k, m, &mut rng);
            let x: Vec<f32> = (0..k).map(|_| rng.f()).collect();
            let mut y = vec![0f32; m];
            gemv(ty, &w, &x, &mut y, k, m);
            let rb = ty.row_bytes(k).unwrap();
            for r in 0..m {
                let mut row = vec![0f32; k];
                dequantize_row(ty, &w[r * rb..(r + 1) * rb], &mut row, k);
                let refv: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
                let tol = 1e-3 * (1.0 + row.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum::<f32>());
                assert!((refv - y[r]).abs() < tol, "{:?} row {r}: ref {refv} got {}", ty, y[r]);
            }
            let scalar: Vec<f32> = (0..m).map(|r| row_dot_scalar(ty, &w[r * rb..(r + 1) * rb], &x)).collect();
            for r in 0..m {
                assert!((scalar[r] - y[r]).abs() < 1e-2 * (1.0 + y[r].abs()), "{:?} scalar vs avx2", ty);
            }
        }
    }
}
