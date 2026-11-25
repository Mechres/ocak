//! GGML tensor types and exact CPU reference dequantization.
use half::{bf16, f16};

pub const QK_K: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1M,
    BF16,
    Other(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        use GgmlType::*;
        match v {
            0 => F32, 1 => F16, 2 => Q4_0, 3 => Q4_1, 6 => Q5_0, 7 => Q5_1, 8 => Q8_0, 9 => Q8_1,
            10 => Q2K, 11 => Q3K, 12 => Q4K, 13 => Q5K, 14 => Q6K, 15 => Q8K,
            16 => IQ2XXS, 17 => IQ2XS, 18 => IQ3XXS, 19 => IQ1S, 20 => IQ4NL, 21 => IQ3S,
            22 => IQ2S, 23 => IQ4XS, 24 => I8, 25 => I16, 26 => I32, 27 => I64, 28 => F64,
            29 => IQ1M, 30 => BF16, o => Other(o),
        }
    }

    pub fn id(self) -> u32 {
        use GgmlType::*;
        match self {
            F32 => 0, F16 => 1, Q4_0 => 2, Q4_1 => 3, Q5_0 => 6, Q5_1 => 7, Q8_0 => 8, Q8_1 => 9,
            Q2K => 10, Q3K => 11, Q4K => 12, Q5K => 13, Q6K => 14, Q8K => 15,
            IQ2XXS => 16, IQ2XS => 17, IQ3XXS => 18, IQ1S => 19, IQ4NL => 20, IQ3S => 21,
            IQ2S => 22, IQ4XS => 23, I8 => 24, I16 => 25, I32 => 26, I64 => 27, F64 => 28,
            IQ1M => 29, BF16 => 30, Other(o) => o,
        }
    }

    pub fn name(self) -> String {
        use GgmlType::*;
        match self {
            Q4_0 => "Q4_0".into(), Q4_1 => "Q4_1".into(), Q5_0 => "Q5_0".into(), Q5_1 => "Q5_1".into(),
            Q8_0 => "Q8_0".into(), Q8_1 => "Q8_1".into(), Q2K => "Q2_K".into(), Q3K => "Q3_K".into(),
            Q4K => "Q4_K".into(), Q5K => "Q5_K".into(), Q6K => "Q6_K".into(), Q8K => "Q8_K".into(),
            IQ2XXS => "IQ2_XXS".into(), IQ2XS => "IQ2_XS".into(), IQ3XXS => "IQ3_XXS".into(),
            IQ1S => "IQ1_S".into(), IQ4NL => "IQ4_NL".into(), IQ3S => "IQ3_S".into(),
            IQ2S => "IQ2_S".into(), IQ4XS => "IQ4_XS".into(), IQ1M => "IQ1_M".into(),
            Other(o) => format!("type{o}"),
            o => format!("{o:?}"),
        }
    }

    /// (elements per block, bytes per block); None when unsupported here.
    pub fn block(self) -> Option<(usize, usize)> {
        use GgmlType::*;
        Some(match self {
            F32 => (1, 4),
            F16 | BF16 => (1, 2),
            F64 | I64 => (1, 8),
            I8 => (1, 1),
            I16 => (1, 2),
            I32 => (1, 4),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q8_1 => (32, 36),
            Q2K => (QK_K, 84),
            Q3K => (QK_K, 110),
            Q4K => (QK_K, 144),
            Q5K => (QK_K, 176),
            Q6K => (QK_K, 210),
            Q8K => (QK_K, 292),
            IQ2XXS => (QK_K, 66),
            IQ2XS => (QK_K, 74),
            IQ2S => (QK_K, 82),
            IQ3XXS => (QK_K, 98),
            IQ3S => (QK_K, 110),
            IQ1S => (QK_K, 50),
            IQ1M => (QK_K, 56),
            IQ4NL => (32, 18),
            IQ4XS => (QK_K, 136),
            Other(_) => return None,
        })
    }

    pub fn row_bytes(self, n_elems: usize) -> Option<usize> {
        let (be, bb) = self.block()?;
        if n_elems % be != 0 {
            return None;
        }
        Some(n_elems / be * bb)
    }

    /// Types `dequantize_row` implements.
    pub fn dequant_supported(self) -> bool {
        use GgmlType::*;
        matches!(self, F32 | F16 | BF16 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q2K | Q3K | Q4K | Q5K | Q6K | IQ2XS | IQ3XXS | IQ3S | IQ4NL | IQ4XS)
    }
}

#[inline]
fn f16_at(b: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Dequantizes `n` elements (a whole number of blocks) from `src` into `dst[..n]`.
pub fn dequantize_row(t: GgmlType, src: &[u8], dst: &mut [f32], n: usize) {
    use GgmlType::*;
    match t {
        F32 => {
            for i in 0..n {
                dst[i] = f32::from_le_bytes(src[4 * i..4 * i + 4].try_into().unwrap());
            }
        }
        F16 => {
            for i in 0..n {
                dst[i] = f16_at(src, 2 * i);
            }
        }
        BF16 => {
            for i in 0..n {
                dst[i] = bf16::from_le_bytes([src[2 * i], src[2 * i + 1]]).to_f32();
            }
        }
        Q4_0 => {
            for (bi, b) in src.chunks_exact(18).take(n / 32).enumerate() {
                let d = f16_at(b, 0);
                for j in 0..16 {
                    dst[bi * 32 + j] = ((b[2 + j] & 0xF) as i32 - 8) as f32 * d;
                    dst[bi * 32 + j + 16] = ((b[2 + j] >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        Q4_1 => {
            for (bi, b) in src.chunks_exact(20).take(n / 32).enumerate() {
                let (d, m) = (f16_at(b, 0), f16_at(b, 2));
                for j in 0..16 {
                    dst[bi * 32 + j] = (b[4 + j] & 0xF) as f32 * d + m;
                    dst[bi * 32 + j + 16] = (b[4 + j] >> 4) as f32 * d + m;
                }
            }
        }
        Q5_0 => {
            for (bi, b) in src.chunks_exact(22).take(n / 32).enumerate() {
                let d = f16_at(b, 0);
                let qh = u32::from_le_bytes(b[2..6].try_into().unwrap());
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    dst[bi * 32 + j] = (((b[6 + j] & 0xF) as u32 | xh0) as i32 - 16) as f32 * d;
                    dst[bi * 32 + j + 16] = (((b[6 + j] >> 4) as u32 | xh1) as i32 - 16) as f32 * d;
                }
            }
        }
        Q5_1 => {
            for (bi, b) in src.chunks_exact(24).take(n / 32).enumerate() {
                let (d, m) = (f16_at(b, 0), f16_at(b, 2));
                let qh = u32::from_le_bytes(b[4..8].try_into().unwrap());
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    dst[bi * 32 + j] = ((b[8 + j] & 0xF) as u32 | xh0) as f32 * d + m;
                    dst[bi * 32 + j + 16] = ((b[8 + j] >> 4) as u32 | xh1) as f32 * d + m;
                }
            }
        }
        Q8_0 => {
            for (bi, b) in src.chunks_exact(34).take(n / 32).enumerate() {
                let d = f16_at(b, 0);
                for j in 0..32 {
                    dst[bi * 32 + j] = (b[2 + j] as i8) as f32 * d;
                }
            }
        }
        Q2K => {
            for (bi, b) in src.chunks_exact(84).take(n / QK_K).enumerate() {
                let scales = &b[0..16];
                let qs = &b[16..80];
                let d = f16_at(b, 80);
                let dmin = f16_at(b, 82);
                let mut out = bi * QK_K;
                let mut is = 0usize;
                for half in 0..2 {
                    let q = &qs[half * 32..half * 32 + 32];
                    let mut shift = 0u8;
                    for _ in 0..4 {
                        let sc = scales[is]; is += 1;
                        let (dl, ml) = (d * (sc & 0xF) as f32, dmin * (sc >> 4) as f32);
                        for l in 0..16 { dst[out] = dl * ((q[l] >> shift) & 3) as f32 - ml; out += 1; }
                        let sc = scales[is]; is += 1;
                        let (dl, ml) = (d * (sc & 0xF) as f32, dmin * (sc >> 4) as f32);
                        for l in 0..16 { dst[out] = dl * ((q[l + 16] >> shift) & 3) as f32 - ml; out += 1; }
                        shift += 2;
                    }
                }
            }
        }
        Q3K => {
            const KM1: u32 = 0x03030303;
            const KM2: u32 = 0x0f0f0f0f;
            for (bi, b) in src.chunks_exact(110).take(n / QK_K).enumerate() {
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
                let d_all = f16_at(b, 108);
                let mut out = bi * QK_K;
                let mut is = 0usize;
                let mut m = 1u8;
                for half in 0..2 {
                    let q = &qs[half * 32..half * 32 + 32];
                    let mut shift = 0u8;
                    for _ in 0..4 {
                        let dl = d_all * (scales[is] as i32 - 32) as f32; is += 1;
                        for l in 0..16 {
                            let v = ((q[l] >> shift) & 3) as i32 - if hm[l] & m != 0 { 0 } else { 4 };
                            dst[out] = dl * v as f32; out += 1;
                        }
                        let dl = d_all * (scales[is] as i32 - 32) as f32; is += 1;
                        for l in 0..16 {
                            let v = ((q[l + 16] >> shift) & 3) as i32 - if hm[l + 16] & m != 0 { 0 } else { 4 };
                            dst[out] = dl * v as f32; out += 1;
                        }
                        shift += 2;
                        m <<= 1;
                    }
                }
            }
        }
        Q4K => {
            for (bi, b) in src.chunks_exact(144).take(n / QK_K).enumerate() {
                let d = f16_at(b, 0);
                let dmin = f16_at(b, 2);
                let scales = &b[4..16];
                let qs = &b[16..144];
                let mut out = bi * QK_K;
                let mut is = 0;
                for g in 0..4 {
                    let q = &qs[g * 32..g * 32 + 32];
                    let (sc, m) = get_scale_min_k4(is, scales);
                    let (d1, m1) = (d * sc as f32, dmin * m as f32);
                    let (sc, m) = get_scale_min_k4(is + 1, scales);
                    let (d2, m2) = (d * sc as f32, dmin * m as f32);
                    for l in 0..32 {
                        dst[out + l] = d1 * (q[l] & 0xF) as f32 - m1;
                    }
                    for l in 0..32 {
                        dst[out + 32 + l] = d2 * (q[l] >> 4) as f32 - m2;
                    }
                    out += 64;
                    is += 2;
                }
            }
        }
        Q5K => {
            for (bi, b) in src.chunks_exact(176).take(n / QK_K).enumerate() {
                let d = f16_at(b, 0);
                let dmin = f16_at(b, 2);
                let scales = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                let mut out = bi * QK_K;
                let mut is = 0;
                let (mut u1, mut u2) = (1u8, 2u8);
                for g in 0..4 {
                    let ql = &qs[g * 32..g * 32 + 32];
                    let (sc, m) = get_scale_min_k4(is, scales);
                    let (d1, m1) = (d * sc as f32, dmin * m as f32);
                    let (sc, m) = get_scale_min_k4(is + 1, scales);
                    let (d2, m2) = (d * sc as f32, dmin * m as f32);
                    for l in 0..32 {
                        let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                        dst[out + l] = d1 * ((ql[l] & 0xF) as i32 + hi) as f32 - m1;
                    }
                    for l in 0..32 {
                        let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                        dst[out + 32 + l] = d2 * ((ql[l] >> 4) as i32 + hi) as f32 - m2;
                    }
                    out += 64;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        Q6K => {
            for (bi, b) in src.chunks_exact(210).take(n / QK_K).enumerate() {
                let d = f16_at(b, 208);
                let mut out = bi * QK_K;
                for half in 0..2 {
                    let ql = &b[half * 64..half * 64 + 64];
                    let qh = &b[128 + half * 32..128 + half * 32 + 32];
                    let sc = &b[192 + half * 8..192 + half * 8 + 8];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = (((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32) - 32;
                        let q2 = (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32) - 32;
                        let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32) - 32;
                        let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32) - 32;
                        dst[out + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        dst[out + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        dst[out + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        dst[out + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                    out += 128;
                }
            }
        }
        IQ2XS => {
            use crate::iq_tables::{IQ2XS_GRID, KSIGNS_IQ2XS};
            for (bi, b) in src.chunks_exact(74).take(n / QK_K).enumerate() {
                let d = f16_at(b, 0);
                let qs = &b[2..66];
                let scales = &b[66..74];
                let mut out = bi * QK_K;
                for ib32 in 0..8 {
                    let db = [
                        d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25,
                        d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
                    ];
                    for l in 0..4 {
                        let q = u16::from_le_bytes([qs[2 * (4 * ib32 + l)], qs[2 * (4 * ib32 + l) + 1]]);
                        let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                        let dl = db[l / 2];
                        for j in 0..8 {
                            dst[out] = dl * grid[j] as f32 * if (signs & (1 << j)) != 0 { -1.0 } else { 1.0 };
                            out += 1;
                        }
                    }
                }
            }
        }
        IQ3XXS => {
            use crate::iq_tables::{IQ3XXS_GRID, KSIGNS_IQ2XS};
            for (bi, b) in src.chunks_exact(98).take(n / QK_K).enumerate() {
                let d = f16_at(b, 0);
                let qs = &b[2..66];
                let sas = &b[66..98];
                let mut out = bi * QK_K;
                for ib32 in 0..8 {
                    let aux32 = u32::from_le_bytes(sas[ib32 * 4..ib32 * 4 + 4].try_into().unwrap());
                    let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                    for l in 0..4 {
                        let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                        let g1 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l] as usize].to_le_bytes();
                        let g2 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l + 1] as usize].to_le_bytes();
                        for j in 0..4 {
                            dst[out + j] = db * g1[j] as f32 * if (signs & (1 << j)) != 0 { -1.0 } else { 1.0 };
                            dst[out + j + 4] = db * g2[j] as f32 * if (signs & (1 << (j + 4))) != 0 { -1.0 } else { 1.0 };
                        }
                        out += 8;
                    }
                }
            }
        }
        IQ3S => {
            use crate::iq_tables::IQ3S_GRID;
            for (bi, b) in src.chunks_exact(110).take(n / QK_K).enumerate() {
                let d = f16_at(b, 0);
                let mut qs = &b[2..66];
                let mut qh = &b[66..74];
                let mut signs = &b[74..106];
                let scales = &b[106..110];
                let mut out = bi * QK_K;
                for ib32 in (0..8).step_by(2) {
                    let db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xF) as i32) as f32;
                    let db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4) as i32) as f32;
                    for l in 0..4 {
                        let idx1 = (qs[2 * l] as usize) | (((qh[0] as usize) << (8 - 2 * l)) & 256);
                        let idx2 = (qs[2 * l + 1] as usize) | (((qh[0] as usize) << (7 - 2 * l)) & 256);
                        let g1 = IQ3S_GRID[idx1].to_le_bytes();
                        let g2 = IQ3S_GRID[idx2].to_le_bytes();
                        for j in 0..4 {
                            dst[out + j] = db1 * g1[j] as f32 * if (signs[l] & (1 << j)) != 0 { -1.0 } else { 1.0 };
                            dst[out + j + 4] = db1 * g2[j] as f32 * if (signs[l] & (1 << (j + 4))) != 0 { -1.0 } else { 1.0 };
                        }
                        out += 8;
                    }
                    qs = &qs[8..];
                    signs = &signs[4..];
                    for l in 0..4 {
                        let idx1 = (qs[2 * l] as usize) | (((qh[1] as usize) << (8 - 2 * l)) & 256);
                        let idx2 = (qs[2 * l + 1] as usize) | (((qh[1] as usize) << (7 - 2 * l)) & 256);
                        let g1 = IQ3S_GRID[idx1].to_le_bytes();
                        let g2 = IQ3S_GRID[idx2].to_le_bytes();
                        for j in 0..4 {
                            dst[out + j] = db2 * g1[j] as f32 * if (signs[l] & (1 << j)) != 0 { -1.0 } else { 1.0 };
                            dst[out + j + 4] = db2 * g2[j] as f32 * if (signs[l] & (1 << (j + 4))) != 0 { -1.0 } else { 1.0 };
                        }
                        out += 8;
                    }
                    qh = &qh[2..];
                    qs = &qs[8..];
                    signs = &signs[4..];
                }
            }
        }
        IQ4NL => {
            use crate::iq_tables::KVALUES_IQ4NL;
            for (bi, b) in src.chunks_exact(18).take(n / 32).enumerate() {
                let d = f16_at(b, 0);
                for j in 0..16 {
                    dst[bi * 32 + j] = d * KVALUES_IQ4NL[(b[2 + j] & 0xF) as usize] as f32;
                    dst[bi * 32 + j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
                }
            }
        }
        IQ4XS => {
            use crate::iq_tables::KVALUES_IQ4NL;
            for (bi, b) in src.chunks_exact(136).take(n / QK_K).enumerate() {
                let d = f16_at(b, 0);
                let scales_h = u16::from_le_bytes([b[2], b[3]]);
                let scales_l = &b[4..8];
                let qs = &b[8..136];
                for ib in 0..8 {
                    let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32)
                        | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
                    let dl = d * (ls - 32) as f32;
                    let q = &qs[ib * 16..ib * 16 + 16];
                    let out = bi * QK_K + ib * 32;
                    for j in 0..16 {
                        dst[out + j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
                        dst[out + j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
                    }
                }
            }
        }
        _ => panic!("dequantize_row: unsupported type {}", t.name()),
    }
}

/// Quantizes `x` (multiple of 32) into Q8_0 blocks (34 bytes each), matching ggml's reference.
pub fn quantize_q8_0(x: &[f32], out: &mut Vec<u8>) {
    for blk in x.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for v in blk {
            out.push((v * id).round() as i8 as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_ggml() {
        assert_eq!(GgmlType::Q4_0.block(), Some((32, 18)));
        assert_eq!(GgmlType::Q8_0.block(), Some((32, 34)));
        assert_eq!(GgmlType::Q4K.block(), Some((256, 144)));
        assert_eq!(GgmlType::Q5K.block(), Some((256, 176)));
        assert_eq!(GgmlType::Q6K.block(), Some((256, 210)));
        assert_eq!(GgmlType::from_u32(12), GgmlType::Q4K);
        assert_eq!(GgmlType::Q6K.id(), 14);
    }

    #[test]
    fn q4_0_roundtrip_layout() {
        let mut b = vec![0u8; 18];
        b[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        for j in 0..16 {
            b[2 + j] = ((((j as u8) + 1) & 0xF) << 4) | (j as u8);
        }
        let mut y = vec![0f32; 32];
        dequantize_row(GgmlType::Q4_0, &b, &mut y, 32);
        for j in 0..16 {
            assert_eq!(y[j], (j as i32 - 8) as f32 * 0.5);
            assert_eq!(y[j + 16], (((j as i32 + 1) & 0xF) - 8) as f32 * 0.5);
        }
    }

    #[test]
    fn q8_0_quantize_dequantize_close() {
        let x: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37).sin()).collect();
        let mut q = Vec::new();
        quantize_q8_0(&x, &mut q);
        assert_eq!(q.len(), 68);
        let mut y = vec![0f32; 64];
        dequantize_row(GgmlType::Q8_0, &q, &mut y, 64);
        for i in 0..64 {
            assert!((x[i] - y[i]).abs() < 0.01, "{} {}", x[i], y[i]);
        }
    }

    #[test]
    fn q4k_scale_unpack_matches_reference() {
        // scales packed per ggml: first 4 bytes low 6 bits sc[0..4], next 4 mins[0..4]
        let mut s = [0u8; 12];
        for i in 0..4 { s[i] = (i as u8 + 1) | (2 << 6); s[i + 4] = (i as u8 + 10) | (1 << 6); }
        for i in 0..4 { s[i + 8] = (i as u8 + 3) | (((i as u8 + 5) & 0xF) << 4); }
        assert_eq!(get_scale_min_k4(0, &s), (1, 10));
        let (sc, m) = get_scale_min_k4(4, &s);
        assert_eq!(sc, (s[8] & 0xF) | ((s[0] >> 6) << 4));
        assert_eq!(m, (s[8] >> 4) | ((s[4] >> 6) << 4));
    }
}
