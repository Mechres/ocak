//! GPU kernel tests (each compares a Vulkan kernel against a CPU / reference result).
use ocak_gguf::GgmlType;
use std::sync::Arc;
use crate::*;
use half::f16;
use ocak_gguf::quant::dequantize_row;
use std::sync::OnceLock;

static CTX: OnceLock<Arc<VkContext>> = OnceLock::new();
fn ctx() -> Arc<VkContext> {
    Arc::clone(CTX.get_or_init(|| VkContext::new().expect("vulkan init")))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f(&mut self) -> f32 { (self.next() as f32 / (1u32 << 31) as f32) * 2.0 - 1.0 }
}

fn rand_weights(ty: WType, k: usize, m: usize, rng: &mut Lcg) -> Vec<u8> {
    let g = ty.ggml();
    let (be, bb) = g.block().unwrap();
    let mut v: Vec<u8> = (0..(k / be * m * bb)).map(|_| rng.next() as u8).collect();
    match ty {
        WType::F32 => for c in v.chunks_exact_mut(4) { c.copy_from_slice(&rng.f().to_le_bytes()); },
        WType::F16 => for c in v.chunks_exact_mut(2) { c.copy_from_slice(&f16::from_f32(rng.f()).to_le_bytes()); },
        WType::BF16 => for c in v.chunks_exact_mut(2) { c.copy_from_slice(&half::bf16::from_f32(rng.f()).to_le_bytes()); },
        WType::Q4_0 | WType::Q8_0 => for blk in v.chunks_exact_mut(bb) { blk[0..2].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes()); },
        WType::Q4K | WType::Q5K => for blk in v.chunks_exact_mut(bb) {
            blk[0..2].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
            blk[2..4].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
        },
        WType::Q2K => for blk in v.chunks_exact_mut(bb) {
            blk[80..82].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
            blk[82..84].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
        },
        WType::Q3K => for blk in v.chunks_exact_mut(bb) {
            blk[108..110].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
        },
        WType::IQ2XXS | WType::IQ2XS | WType::IQ2S | WType::IQ3XXS | WType::IQ3S | WType::IQ1S | WType::IQ4NL | WType::IQ4XS => {
            for blk in v.chunks_exact_mut(bb) { blk[0..2].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes()); }
        }
        WType::IQ1M => for blk in v.chunks_exact_mut(bb) {
            let d = f16::from_f32(rng.f() * 0.05).to_bits();
            for s in 0..4 {
                let lo = (rng.next() as u16) & 0x0FFF;
                let hi = ((d >> (4 * s)) & 0xF) << 12;
                blk[48 + 2 * s..50 + 2 * s].copy_from_slice(&(lo | hi).to_le_bytes());
            }
        },
        WType::NVFP4 => for blk in v.chunks_exact_mut(bb) {
            for s in 0..4 { blk[s] = (rng.next() as u8) & 0x7E; }
        },
        WType::Q6K => for blk in v.chunks_exact_mut(bb) { blk[208..210].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes()); },
    }
    v
}

#[test]
fn vk_qmv_all_types_match_cpu() {
    let c = ctx();
    let mut rng = Lcg(1337);
    for ty in [
        WType::F32, WType::F16, WType::BF16, WType::Q8_0, WType::Q4_0,
        WType::Q2K, WType::Q3K, WType::Q4K, WType::Q5K, WType::Q6K,
        WType::IQ2XXS, WType::IQ2XS, WType::IQ2S, WType::IQ3XXS, WType::IQ3S,
        WType::IQ1S, WType::IQ1M, WType::IQ4NL, WType::IQ4XS, WType::NVFP4,
    ] {
        let (k, m) = (512usize, 37usize);
        let w = rand_weights(ty, k, m, &mut rng);
        let x: Vec<f32> = (0..k).map(|_| rng.f()).collect();
        let mut y_cpu = vec![0f32; m];
        ocak_cpu::gemv(ty.ggml(), &w, &x, &mut y_cpu, k, m);

        let bw = c.create_device_buffer(w.len() as u64).unwrap();
        let bx = c.create_device_buffer((k * 4) as u64).unwrap();
        let by = c.create_device_buffer((m * 4) as u64).unwrap();
        c.upload(&bw, 0, &w).unwrap();
        c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
        let mut rec = c.begin();
        rec.set_mmq(false);
        rec.qmv(ty, &bw, 0, &bx, 0, &by, 0, k as u32, m as u32);
        rec.finish_and_wait().unwrap();
        let mut y_gpu = vec![0f32; m];
        c.download(&by, 0, bytemuck::cast_slice_mut(&mut y_gpu)).unwrap();
        for r in 0..m {
            let err = (y_cpu[r] - y_gpu[r]).abs();
            let tol = 2e-3 * (1.0 + y_cpu[r].abs());
            assert!(err < tol, "{ty:?} row {r}: cpu={} gpu={} err={err}", y_cpu[r], y_gpu[r]);
        }
    }
}

#[test]
fn vk_elementwise_rope_embed_and_paged_attention() {
    let c = ctx();
    let mut rng = Lcg(999);
    // rms_norm in-place
    let n = 512usize;
    let x: Vec<f32> = (0..n).map(|_| rng.f()).collect();
    let w: Vec<f32> = (0..n).map(|_| rng.f()).collect();
    let mut out_cpu = vec![0f32; n];
    ocak_cpu::ops::rms_norm(&x, &w, &mut out_cpu, 1e-5);
    let bx = c.create_device_buffer((n * 4) as u64).unwrap();
    let bw = c.create_device_buffer((n * 4) as u64).unwrap();
    c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
    c.upload(&bw, 0, bytemuck::cast_slice(&w)).unwrap();
    let mut rec = c.begin();
    rec.rms_norm(&bx, 0, &bw, 0, &bx, 0, n as u32, 1e-5);
    rec.finish_and_wait().unwrap();
    let mut out_gpu = vec![0f32; n];
    c.download(&bx, 0, bytemuck::cast_slice_mut(&mut out_gpu)).unwrap();
    for i in 0..n { assert!((out_cpu[i] - out_gpu[i]).abs() < 1e-4); }

    // paged attention with non-identity page table [5, 2, 9] across 38 tokens (pos=37)
    let (nh, nkv, hd, pos) = (32usize, 4usize, 64usize, 37usize);
    let kv_dim = nkv * hd;
    let n_pages = 12usize;
    let pt = vec![5u32, 2, 9];
    let mut kv_cpu = vec![0f32; n_pages * 2 * 16 * kv_dim];
    let bkv = c.create_device_buffer((kv_cpu.len() * 4) as u64).unwrap();
    let bpt = c.create_device_buffer((pt.len() * 4) as u64).unwrap();
    c.upload(&bpt, 0, bytemuck::cast_slice(&pt)).unwrap();
    let bk = c.create_device_buffer((kv_dim * 4) as u64).unwrap();
    let bv = c.create_device_buffer((kv_dim * 4) as u64).unwrap();
    for t in 0..=pos {
        let k: Vec<f32> = (0..kv_dim).map(|_| rng.f()).collect();
        let v: Vec<f32> = (0..kv_dim).map(|_| rng.f()).collect();
        ocak_cpu::ops::kv_write(&k, &v, &mut kv_cpu, 0, &pt, t, kv_dim);
        c.upload(&bk, 0, bytemuck::cast_slice(&k)).unwrap();
        c.upload(&bv, 0, bytemuck::cast_slice(&v)).unwrap();
        let mut rec = c.begin();
        rec.kv_write(&bk, 0, &bv, 0, &bkv, 0, &bpt, 0, t as u32, kv_dim as u32);
        rec.finish_and_wait().unwrap();
    }
    let q: Vec<f32> = (0..nh * hd).map(|_| rng.f()).collect();
    let mut a_cpu = vec![0f32; nh * hd];
    ocak_cpu::ops::attention(&q, &kv_cpu, 0, &pt, &mut a_cpu, nh, nkv, hd, pos);
    let bq = c.create_device_buffer((q.len() * 4) as u64).unwrap();
    let bo = c.create_device_buffer((a_cpu.len() * 4) as u64).unwrap();
    c.upload(&bq, 0, bytemuck::cast_slice(&q)).unwrap();
    let mut rec = c.begin();
    rec.attention(&bq, 0, &bkv, 0, &bpt, 0, &bo, 0, nh as u32, nkv as u32, hd as u32, pos as u32);
    rec.finish_and_wait().unwrap();
    let mut a_gpu = vec![0f32; nh * hd];
    c.download(&bo, 0, bytemuck::cast_slice_mut(&mut a_gpu)).unwrap();
    for i in 0..a_cpu.len() {
        assert!((a_cpu[i] - a_gpu[i]).abs() < 2e-3, "attn[{i}]: cpu={} gpu={}", a_cpu[i], a_gpu[i]);
    }

    // embed Q4K check
    let w = rand_weights(WType::Q4K, 512, 5, &mut rng);
    let bw = c.create_device_buffer(w.len() as u64).unwrap();
    let bo = c.create_device_buffer(512 * 4).unwrap();
    c.upload(&bw, 0, &w).unwrap();
    let mut rec = c.begin();
    rec.embed(WType::Q4K, &bw, 3, &bo, 0, 512);
    rec.finish_and_wait().unwrap();
    let mut e_gpu = vec![0f32; 512];
    c.download(&bo, 0, bytemuck::cast_slice_mut(&mut e_gpu)).unwrap();
    let rb = WType::Q4K.row_bytes(512) as usize;
    let mut e_cpu = vec![0f32; 512];
    dequantize_row(GgmlType::Q4K, &w[3 * rb..4 * rb], &mut e_cpu, 512);
    for i in 0..512 { assert!((e_cpu[i] - e_gpu[i]).abs() < 1e-4); }
}


#[test]
fn vk_m4_hybrid_shortconv_and_deltanet_match_cpu() {
    let c = ctx();
    let mut rng = Lcg(4242);
    // 1. ShortConv (LFM2)
    let (ne, lc) = (256usize, 3usize);
    let bcx: Vec<f32> = (0..3 * ne).map(|_| rng.f()).collect();
    let cw: Vec<f32> = (0..ne * lc).map(|_| rng.f()).collect();
    let mut st_cpu = vec![0f32; ne * (lc - 1)];
    let mut o_cpu = vec![0f32; ne];
    let b_bcx = c.create_device_buffer((bcx.len() * 4) as u64).unwrap();
    let b_cw = c.create_device_buffer((cw.len() * 4) as u64).unwrap();
    let b_st = c.create_device_buffer((st_cpu.len() * 4) as u64).unwrap();
    let b_o = c.create_device_buffer((o_cpu.len() * 4) as u64).unwrap();
    c.upload(&b_bcx, 0, bytemuck::cast_slice(&bcx)).unwrap();
    c.upload(&b_cw, 0, bytemuck::cast_slice(&cw)).unwrap();
    c.zero_buffer(&b_st).unwrap();
    for _ in 0..4 {
        ocak_cpu::ops::shortconv_step(&bcx, &cw, &mut st_cpu, &mut o_cpu, ne, lc);
        let mut rec = c.begin();
        rec.shortconv_step(&b_bcx, &b_cw, &b_st, &b_o, ne as u32, lc as u32);
        rec.finish_and_wait().unwrap();
    }
    let mut o_gpu = vec![0f32; ne];
    c.download(&b_o, 0, bytemuck::cast_slice_mut(&mut o_gpu)).unwrap();
    for i in 0..ne { assert!((o_cpu[i] - o_gpu[i]).abs() < 1e-4, "shortconv[{i}]"); }

    // 2. Gated DeltaNet (Qwen3.5)
    let (nkh, nvh, hd) = (4usize, 8usize, 128usize);
    let qkv: Vec<f32> = (0..(2 * nkh + nvh) * hd).map(|_| rng.f()).collect();
    let alpha: Vec<f32> = (0..nvh).map(|_| rng.f()).collect();
    let beta: Vec<f32> = (0..nvh).map(|_| rng.f()).collect();
    let dt_bias: Vec<f32> = (0..nvh).map(|_| rng.f()).collect();
    let ssm_a: Vec<f32> = (0..nvh).map(|_| -rng.f().abs()).collect();
    let mut st_cpu = vec![0f32; nvh * hd * hd];
    let mut o_cpu = vec![0f32; nvh * hd];
    let b_qkv = c.create_device_buffer((qkv.len() * 4) as u64).unwrap();
    let b_al = c.create_device_buffer((alpha.len() * 4) as u64).unwrap();
    let b_bt = c.create_device_buffer((beta.len() * 4) as u64).unwrap();
    let b_dt = c.create_device_buffer((dt_bias.len() * 4) as u64).unwrap();
    let b_sa = c.create_device_buffer((ssm_a.len() * 4) as u64).unwrap();
    let b_st = c.create_device_buffer((st_cpu.len() * 4) as u64).unwrap();
    let b_o = c.create_device_buffer((o_cpu.len() * 4) as u64).unwrap();
    c.upload(&b_qkv, 0, bytemuck::cast_slice(&qkv)).unwrap();
    c.upload(&b_al, 0, bytemuck::cast_slice(&alpha)).unwrap();
    c.upload(&b_bt, 0, bytemuck::cast_slice(&beta)).unwrap();
    c.upload(&b_dt, 0, bytemuck::cast_slice(&dt_bias)).unwrap();
    c.upload(&b_sa, 0, bytemuck::cast_slice(&ssm_a)).unwrap();
    c.zero_buffer(&b_st).unwrap();
    for _ in 0..3 {
        ocak_cpu::ops::deltanet_step(&qkv, &alpha, &beta, &dt_bias, &ssm_a, &mut st_cpu, &mut o_cpu, nkh, nvh, hd, 1e-6);
        let mut rec = c.begin();
        rec.deltanet_step(&b_qkv, &b_al, &b_bt, &b_dt, &b_sa, &b_st, &b_o, nkh as u32, nvh as u32, hd as u32, 1e-6);
        rec.finish_and_wait().unwrap();
    }
    let mut o_gpu = vec![0f32; nvh * hd];
    c.download(&b_o, 0, bytemuck::cast_slice_mut(&mut o_gpu)).unwrap();
    for i in 0..o_cpu.len() { assert!((o_cpu[i] - o_gpu[i]).abs() < 1e-3, "deltanet[{i}]: cpu={} gpu={}", o_cpu[i], o_gpu[i]); }
}

#[test]
fn vk_recurrent_batch_matches_cpu() {
    // Multi-token (prefill) paths of the short convs and DeltaNet vs the CPU per-token ops,
    // with a chunk length that is not a multiple of the kernels' 16-token staging.
    let c = ctx();
    let mut rng = Lcg(9191);
    let nt = 37usize;
    let up = |v: &[f32]| {
        let b = c.create_device_buffer((v.len() * 4) as u64).unwrap();
        c.upload(&b, 0, bytemuck::cast_slice(v)).unwrap();
        b
    };
    // ShortConv
    let (ne, lc) = (300usize, 3usize);
    let bcx: Vec<f32> = (0..nt * 3 * ne).map(|_| rng.f()).collect();
    let cw: Vec<f32> = (0..ne * lc).map(|_| rng.f()).collect();
    let st0: Vec<f32> = (0..ne * (lc - 1)).map(|_| rng.f()).collect();
    let mut st = st0.clone();
    let mut o_cpu = vec![0f32; nt * ne];
    for t in 0..nt {
        ocak_cpu::ops::shortconv_step(&bcx[t * 3 * ne..(t + 1) * 3 * ne], &cw, &mut st, &mut o_cpu[t * ne..(t + 1) * ne], ne, lc);
    }
    let (b_bcx, b_cw, b_st) = (up(&bcx), up(&cw), up(&st0));
    let b_o = c.create_device_buffer((nt * ne * 4) as u64).unwrap();
    let mut rec = c.begin();
    rec.shortconv_step_batch(&b_bcx, &b_cw, &b_st, &b_o, ne as u32, lc as u32, nt as u32);
    rec.finish_and_wait().unwrap();
    let mut o_gpu = vec![0f32; nt * ne];
    c.download(&b_o, 0, bytemuck::cast_slice_mut(&mut o_gpu)).unwrap();
    for i in 0..o_cpu.len() { assert!((o_cpu[i] - o_gpu[i]).abs() < 1e-4, "shortconv batch[{i}]"); }
    let mut st_gpu = vec![0f32; st.len()];
    c.download(&b_st, 0, bytemuck::cast_slice_mut(&mut st_gpu)).unwrap();
    assert_eq!(st, st_gpu, "shortconv state");

    // Depthwise conv + SiLU
    let (ch, kc) = (300usize, 4usize);
    let x: Vec<f32> = (0..nt * ch).map(|_| rng.f()).collect();
    let cw: Vec<f32> = (0..ch * kc).map(|_| rng.f()).collect();
    let st0: Vec<f32> = (0..ch * (kc - 1)).map(|_| rng.f()).collect();
    let mut st = st0.clone();
    let mut o_cpu = vec![0f32; nt * ch];
    for t in 0..nt {
        ocak_cpu::ops::ssm_conv_silu(&x[t * ch..(t + 1) * ch], &cw, &mut st, &mut o_cpu[t * ch..(t + 1) * ch], ch, kc);
    }
    let (b_x, b_cw, b_st) = (up(&x), up(&cw), up(&st0));
    let b_o = c.create_device_buffer((nt * ch * 4) as u64).unwrap();
    let mut rec = c.begin();
    rec.ssm_conv_silu_batch(&b_x, &b_cw, &b_st, &b_o, ch as u32, kc as u32, nt as u32);
    rec.finish_and_wait().unwrap();
    let mut o_gpu = vec![0f32; nt * ch];
    c.download(&b_o, 0, bytemuck::cast_slice_mut(&mut o_gpu)).unwrap();
    for i in 0..o_cpu.len() { assert!((o_cpu[i] - o_gpu[i]).abs() < 1e-4, "ssm_conv batch[{i}]"); }

    // Gated DeltaNet
    let (nkh, nvh, hd) = (2usize, 4usize, 128usize);
    let cc = (2 * nkh + nvh) * hd;
    let qkv: Vec<f32> = (0..nt * cc).map(|_| rng.f()).collect();
    let alpha: Vec<f32> = (0..nt * nvh).map(|_| rng.f()).collect();
    let beta: Vec<f32> = (0..nt * nvh).map(|_| rng.f()).collect();
    let dt_bias: Vec<f32> = (0..nvh).map(|_| rng.f()).collect();
    let ssm_a: Vec<f32> = (0..nvh).map(|_| -rng.f().abs()).collect();
    let st0: Vec<f32> = (0..nvh * hd * hd).map(|_| rng.f() * 0.1).collect();
    let mut st = st0.clone();
    let mut o_cpu = vec![0f32; nt * nvh * hd];
    for t in 0..nt {
        ocak_cpu::ops::deltanet_step(
            &qkv[t * cc..(t + 1) * cc], &alpha[t * nvh..(t + 1) * nvh], &beta[t * nvh..(t + 1) * nvh], &dt_bias, &ssm_a,
            &mut st, &mut o_cpu[t * nvh * hd..(t + 1) * nvh * hd], nkh, nvh, hd, 1e-6,
        );
    }
    let (b_qkv, b_al, b_bt, b_dt, b_sa, b_st) = (up(&qkv), up(&alpha), up(&beta), up(&dt_bias), up(&ssm_a), up(&st0));
    let b_o = c.create_device_buffer((nt * nvh * hd * 4) as u64).unwrap();
    let mut rec = c.begin();
    rec.deltanet_step_batch(&b_qkv, &b_al, &b_bt, &b_dt, &b_sa, &b_st, &b_o, nkh as u32, nvh as u32, hd as u32, 1e-6, nt as u32);
    rec.finish_and_wait().unwrap();
    let mut o_gpu = vec![0f32; o_cpu.len()];
    c.download(&b_o, 0, bytemuck::cast_slice_mut(&mut o_gpu)).unwrap();
    for i in 0..o_cpu.len() { assert!((o_cpu[i] - o_gpu[i]).abs() < 1e-3, "deltanet batch[{i}]: cpu={} gpu={}", o_cpu[i], o_gpu[i]); }
    let mut st_gpu = vec![0f32; st.len()];
    c.download(&b_st, 0, bytemuck::cast_slice_mut(&mut st_gpu)).unwrap();
    for i in 0..st.len() { assert!((st[i] - st_gpu[i]).abs() < 1e-3, "deltanet state[{i}]"); }
}

#[test]
fn vk_argmax_first_max() {
    let c = ctx();
    let mut rng = Lcg(2024);
    for &n in &[7usize, 4096, 201_024, 248_320] {
        let mut x: Vec<f32> = (0..n).map(|_| rng.f() * 10.0).collect();
        // Tie: the same maximum at two indices -> the lower one wins.
        let (a, b) = (n / 3, (2 * n) / 3 + 1);
        x[a] = 50.0;
        x[b.min(n - 1)] = 50.0;
        let bx = c.create_device_buffer((n * 4) as u64).unwrap();
        c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
        let bo = c.create_device_buffer(16).unwrap();
        let mut rec = c.begin();
        rec.argmax(&bx, 0, &bo, n as u32);
        rec.finish_and_wait().unwrap();
        let mut out = [0u32; 1];
        c.download(&bo, 0, bytemuck::cast_slice_mut(&mut out)).unwrap();
        assert_eq!(out[0] as usize, a.min(b.min(n - 1)), "argmax n={n}");
    }
}

#[test]
fn vk_matvec_batch_matches_single_bitwise() {
    // Speculative decoding (MTP) verifies drafts with a multi-token matvec; it must reproduce
    // single-token decode exactly or greedy output changes when MTP is on. 3 tokens: the
    // 2..8-token path (all formats). 13 tokens: the 9..31-token int8 path (two groups).
    let c = ctx();
    let (k, m) = (2048usize, 600usize);
    let mut rng = Lcg(77);
    let x: Vec<f32> = (0..13 * k).map(|_| rng.f()).collect();
    let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
    c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
    let ya = c.create_device_buffer((13 * m * 4) as u64).unwrap();
    let yb = c.create_device_buffer((13 * m * 4) as u64).unwrap();
    for (wt, nb) in [WType::Q4K, WType::Q6K, WType::Q8_0, WType::IQ2S, WType::Q2K]
        .into_iter()
        .flat_map(|wt| [(wt, 3usize), (wt, 13)])
        .filter(|&(wt, nb)| nb <= 8 || wt != WType::Q8_0)
    {
        let raw: Vec<u8> = (0..m * wt.row_bytes(k as u32) as usize).map(|_| (rng.next() & 0x3F) as u8).collect();
        let bw = c.create_device_buffer(raw.len() as u64).unwrap();
        c.upload(&bw, 0, &raw).unwrap();
        let mut rec = c.begin();
        rec.qmm(wt, &bw, 0, &bx, 0, &ya, 0, k as u32, m as u32, nb as u32);
        rec.finish_and_wait().unwrap();
        for t in 0..nb {
            let mut rec = c.begin();
            rec.qmv(wt, &bw, 0, &bx, (t * k) as u64, &yb, (t * m) as u64, k as u32, m as u32);
            rec.finish_and_wait().unwrap();
        }
        let mut a = vec![0f32; nb * m];
        let mut b = vec![0f32; nb * m];
        c.download(&ya, 0, bytemuck::cast_slice_mut(&mut a)).unwrap();
        c.download(&yb, 0, bytemuck::cast_slice_mut(&mut b)).unwrap();
        for i in 0..nb * m {
            assert!(a[i].to_bits() == b[i].to_bits(), "{wt:?} batch {nb} idx {i}: batch {} vs single {}", a[i], b[i]);
        }
    }
}

#[test]
#[ignore = "benchmark"]
fn vk_context_init_bench() {
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let c = VkContext::new().expect("vulkan init");
        eprintln!("VkContext::new: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        drop(c);
    }
}

#[test]
#[ignore = "benchmark"]
fn vk_download_bench() {
    let c = ctx();
    let n = 248_320usize;
    let b = c.create_device_buffer((n * 4) as u64).unwrap();
    let mut out = vec![0f32; n];
    c.download(&b, 0, bytemuck::cast_slice_mut(&mut out)).unwrap();
    let t = std::time::Instant::now();
    for _ in 0..20 {
        c.download(&b, 0, bytemuck::cast_slice_mut(&mut out)).unwrap();
    }
    eprintln!("download {} KiB: {:.3} ms", n * 4 / 1024, t.elapsed().as_secs_f64() * 1e3 / 20.0);
}

#[test]
fn vk_context_create_drop_twice() {
    for _ in 0..2 {
        let tmp = VkContext::new().unwrap();
        assert!(!tmp.device_name().is_empty());
        drop(tmp);
    }
}

#[test]
fn vk_inkling_attn_prefill_matches_batch() {
    let c = ctx();
    let mut rng = Lcg(4242);
    let (nh, nkv, hd, nrel) = (24usize, 8usize, 128usize, 16usize);
    let (pos, n_tok) = (70usize, 50usize);
    for (is_swa, n_extent, n_swa, cap) in [(false, 40usize, 64usize, 0usize), (true, 64, 64, 64 + 16 + 64)] {
        let slots = if is_swa { cap } else { pos + n_tok };
        let mut f = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|_| rng.f() * s).collect() };
        let q = f(n_tok * nh * hd, 2.5);
        let r = f(n_tok * nh * nrel, 3.0);
        let sk = f(n_tok * nh, 1.0);
        let g = f(n_tok * nh * hd, 2.0);
        let qclip: Vec<f32> = f(nh * hd, 0.5).iter().map(|v| 1.0 + v).collect();
        let rclip: Vec<f32> = f(nh * nrel, 0.5).iter().map(|v| 1.0 + v).collect();
        let rbias = f(n_extent * nrel, 1.0);
        let kv_words = slots * nkv * hd / 2;
        let mk = |rng: &mut Lcg| -> Vec<u32> {
            (0..kv_words)
                .map(|_| {
                    let a = f16::from_f32(rng.f()).to_bits() as u32;
                    let b = f16::from_f32(rng.f()).to_bits() as u32;
                    a | (b << 16)
                })
                .collect()
        };
        let kc = mk(&mut rng);
        let vc = mk(&mut rng);
        let up = |v: &[f32]| {
            let b = c.create_device_buffer((v.len() * 4) as u64).unwrap();
            c.upload(&b, 0, bytemuck::cast_slice(v)).unwrap();
            b
        };
        let (bq, br, bs, bg) = (up(&q), up(&r), up(&sk), up(&g));
        let (bqc, brc, brb) = (up(&qclip), up(&rclip), up(&rbias));
        let bk = c.create_device_buffer((kv_words * 4) as u64).unwrap();
        c.upload(&bk, 0, bytemuck::cast_slice(&kc)).unwrap();
        let bv = c.create_device_buffer((kv_words * 4) as u64).unwrap();
        c.upload(&bv, 0, bytemuck::cast_slice(&vc)).unwrap();
        let n_out = n_tok * nh * hd;
        let bo_a = c.create_device_buffer((n_out * 4) as u64).unwrap();
        let bo_b = c.create_device_buffer((n_out * 4) as u64).unwrap();
        for (flash, bo) in [(true, &bo_a), (false, &bo_b)] {
            let mut rec = c.begin();
            rec.flash_attn = flash;
            rec.inkling_attn_batch(
                &bq, &br, &bs, &bg, &bqc, &brc, &brb, &bk, &bv, bo,
                (0, (nh * hd) as u32), (0, (nh * nrel) as u32), (0, nh as u32), (0, (nh * hd) as u32),
                nh as u32, nkv as u32, hd as u32, nrel as u32, n_extent as u32, n_swa as u32, is_swa,
                cap as u32, pos as u32, 0, 22.627417, 8.0, 0.1, 8.0, n_tok as u32,
            );
            rec.finish_and_wait().unwrap();
        }
        let mut oa = vec![0f32; n_out];
        let mut ob = vec![0f32; n_out];
        c.download(&bo_a, 0, bytemuck::cast_slice_mut(&mut oa)).unwrap();
        c.download(&bo_b, 0, bytemuck::cast_slice_mut(&mut ob)).unwrap();
        let scale = ob.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
        for i in 0..n_out {
            assert!((oa[i] - ob[i]).abs() <= 5e-3 * scale, "inkling flash swa={is_swa} idx={i}: {} vs {}", oa[i], ob[i]);
        }
    }
}

#[test]
fn vk_mla_attn_prefill_matches_split() {
    let c = ctx();
    let mut rng = Lcg(31337);
    let (nh, latent, n_rot, nope) = (32usize, 512usize, 64usize, 128usize);
    let key_dim = nope + n_rot;
    let half_hd = (latent + n_rot) / 2;
    for (pos, batch) in [(0usize, 40usize), (300, 24)] {
        let n_keys = pos + batch;
        let ql: Vec<f32> = (0..batch * nh * latent).map(|_| rng.f() * 2.0).collect();
        let qf: Vec<f32> = (0..batch * nh * key_dim).map(|_| rng.f() * 2.0).collect();
        let kv: Vec<u32> = (0..n_keys * half_hd)
            .map(|_| (f16::from_f32(rng.f()).to_bits() as u32) | ((f16::from_f32(rng.f()).to_bits() as u32) << 16))
            .collect();
        let up = |v: &[f32]| {
            let b = c.create_device_buffer((v.len() * 4) as u64).unwrap();
            c.upload(&b, 0, bytemuck::cast_slice(v)).unwrap();
            b
        };
        let (bql, bqf) = (up(&ql), up(&qf));
        let bkv = c.create_device_buffer((kv.len() * 4) as u64).unwrap();
        c.upload(&bkv, 0, bytemuck::cast_slice(&kv)).unwrap();
        let n_out = batch * nh * latent;
        let bo_a = c.create_device_buffer((n_out * 4) as u64).unwrap();
        let bo_b = c.create_device_buffer((n_out * 4) as u64).unwrap();
        for (flash, bo) in [(true, &bo_a), (false, &bo_b)] {
            let mut rec = c.begin();
            rec.flash_attn = flash;
            rec.mla_attn_batch(&bql, &bqf, &bkv, bo, nh as u32, key_dim as u32, nope as u32, n_rot as u32, latent as u32, pos as u32, 0, 0.07, batch as u32);
            rec.finish_and_wait().unwrap();
        }
        let mut oa = vec![0f32; n_out];
        let mut ob = vec![0f32; n_out];
        c.download(&bo_a, 0, bytemuck::cast_slice_mut(&mut oa)).unwrap();
        c.download(&bo_b, 0, bytemuck::cast_slice_mut(&mut ob)).unwrap();
        let scale = ob.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
        for i in 0..n_out {
            assert!((oa[i] - ob[i]).abs() <= 1e-3 * scale, "mla flash pos={pos} idx={i}: {} vs {}", oa[i], ob[i]);
        }
    }
}

#[test]
fn vk_inkling_shaders_match_cpu() {
    let c = ctx();
    let mut rng = Lcg(7777);
    // 1. moe_topk + qmv_nvfp4_id + situ_id + add_scaled_exp_id
    let (ne, nff, nexp, nused) = (256usize, 128usize, 16usize, 4usize);
    let logits: Vec<f32> = (0..nexp).map(|_| rng.f() * 3.0).collect();
    let bias: Vec<f32> = (0..nexp).map(|_| rng.f() * 0.5).collect();
    let b_lg = c.create_device_buffer((nexp * 4) as u64).unwrap();
    let b_bi = c.create_device_buffer((nexp * 4) as u64).unwrap();
    let b_ids = c.create_device_buffer((nused * 4) as u64).unwrap();
    let b_wts = c.create_device_buffer((nused * 4) as u64).unwrap();
    c.upload(&b_lg, 0, bytemuck::cast_slice(&logits)).unwrap();
    c.upload(&b_bi, 0, bytemuck::cast_slice(&bias)).unwrap();
    let mut rec = c.begin();
    rec.moe_topk(&b_lg, &b_bi, &b_ids, &b_wts, nexp as u32, nused as u32, true, true, 8.0);
    rec.finish_and_wait().unwrap();
    let mut ids_gpu = vec![0u32; nused];
    let mut wts_gpu = vec![0f32; nused];
    c.download(&b_ids, 0, bytemuck::cast_slice_mut(&mut ids_gpu)).unwrap();
    c.download(&b_wts, 0, bytemuck::cast_slice_mut(&mut wts_gpu)).unwrap();
    // CPU ref
    let probs: Vec<f32> = logits.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    let biased: Vec<f32> = probs.iter().zip(&bias).map(|(p, b)| p + b).collect();
    let mut idx: Vec<usize> = (0..nexp).collect();
    idx.sort_by(|&a, &b| biased[b].total_cmp(&biased[a]));
    idx.truncate(nused);
    let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>() + 1e-20;
    for k in 0..nused {
        assert_eq!(ids_gpu[k], idx[k] as u32, "moe_topk id[{k}]");
        assert!((wts_gpu[k] - probs[idx[k]] / wsum * 8.0).abs() < 1e-4, "moe_topk w[{k}]");
    }

    // 2. inkling_conv (dynamic + static)
    let (nch, nconv, nh_d) = (64usize, 4usize, 2usize);
    let inp: Vec<f32> = (0..nch).map(|_| rng.f()).collect();
    let wt: Vec<f32> = (0..nch * nconv).map(|_| rng.f()).collect();
    let dy: Vec<f32> = (0..nh_d * nconv).map(|_| rng.f()).collect();
    let mut hist_cpu = vec![0f32; nch * (nconv - 1)];
    let b_in = c.create_device_buffer((nch * 4) as u64).unwrap();
    let b_wt = c.create_device_buffer((wt.len() * 4) as u64).unwrap();
    let b_dy = c.create_device_buffer((dy.len() * 4) as u64).unwrap();
    let b_hi = c.create_device_buffer((hist_cpu.len() * 4) as u64).unwrap();
    let b_ou = c.create_device_buffer((nch * 4) as u64).unwrap();
    c.upload(&b_in, 0, bytemuck::cast_slice(&inp)).unwrap();
    c.upload(&b_wt, 0, bytemuck::cast_slice(&wt)).unwrap();
    c.upload(&b_dy, 0, bytemuck::cast_slice(&dy)).unwrap();
    c.zero_buffer(&b_hi).unwrap();
    let mut out_cpu = vec![0f32; nch];
    for _ in 0..3 {
        let nh = nconv - 1;
        let scale = 1.0 / ((2 * nconv + 1) as f32).sqrt();
        for ch in 0..nch {
            let ho = ch * nh;
            let wo = ch * nconv;
            let mut acc = 0f32;
            for k in 0..nh { acc += hist_cpu[ho + k] * wt[wo + k]; }
            acc += inp[ch] * wt[wo + nh];
            let h = ch / (nch / nh_d);
            acc += inp[ch] * dy[h * nconv];
            for lag in 1..nconv { acc += hist_cpu[ho + nh - lag] * dy[h * nconv + lag]; }
            for k in 0..nh - 1 { hist_cpu[ho + k] = hist_cpu[ho + k + 1]; }
            hist_cpu[ho + nh - 1] = inp[ch];
            out_cpu[ch] = (inp[ch] + acc) * scale;
        }
        let mut rec = c.begin();
        rec.inkling_conv(&b_in, 0, &b_wt, &b_dy, 0, &b_hi, 0, &b_ou, 0, nch as u32, nconv as u32, true, (nch / nh_d) as u32);
        rec.finish_and_wait().unwrap();
    }
    let mut out_gpu = vec![0f32; nch];
    c.download(&b_ou, 0, bytemuck::cast_slice_mut(&mut out_gpu)).unwrap();
    for i in 0..nch { assert!((out_cpu[i] - out_gpu[i]).abs() < 1e-4, "conv[{i}]: cpu={} gpu={}", out_cpu[i], out_gpu[i]); }
    let _ = (ne, nff);
}

#[test]
#[ignore = "benchmark"]
fn vk_qmm_tiled_bench() {
    let c = VkContext::new().expect("vulkan init");
    let (k, m, batch) = (5120usize, 17408usize, 256usize);
    let x: Vec<f32> = (0..batch * k).map(|i| ((i % 31) as f32 - 15.0) * 0.01).collect();
    let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
    c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
    let by = c.create_device_buffer((batch * m * 4) as u64).unwrap();
    for wt in [WType::F32, WType::F16, WType::Q8_0, WType::Q4K, WType::Q6K, WType::Q2K, WType::Q3K, WType::IQ3S, WType::IQ3XXS, WType::IQ2XS, WType::IQ2XXS, WType::IQ2S, WType::IQ1S] {
        let rb = wt.row_bytes(k as u32) as usize;
        let raw: Vec<u8> = (0..m * rb).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8 & 0x3F).collect();
        let bw = c.create_device_buffer(raw.len() as u64).unwrap();
        c.upload(&bw, 0, &raw).unwrap();
        let iters = 20;
        let run = |n: usize| {
            let mut rec = c.begin();
            for _ in 0..n {
                rec.qmm(wt, &bw, 0, &bx, 0, &by, 0, k as u32, m as u32, batch as u32);
            }
            rec.finish_and_wait().unwrap();
        };
        run(2);
        let t = std::time::Instant::now();
        run(iters);
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        let tflops = 2.0 * (k * m * batch) as f64 / dt / 1e12;
        eprintln!("{wt:?}: {:.3} ms  {:.2} TFLOPS", dt * 1e3, tflops);
    }
}

#[test]
#[ignore = "benchmark"]
fn vk_qmm_grouped_bench() {
    let c = VkContext::new().expect("vulkan init");
    let (n_exp, n_used, batch) = (64usize, 4usize, 256usize);
    let mut s = 0x2545_F491u64;
    let mut next = || -> u32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    let mut ids = Vec::new();
    for _ in 0..batch {
        let mut picked: Vec<u32> = Vec::new();
        while picked.len() < n_used {
            let e = next() % n_exp as u32;
            if !picked.contains(&e) { picked.push(e); }
        }
        ids.extend(picked);
    }
    let bids = c.create_device_buffer((ids.len() * 4) as u64).unwrap();
    c.upload(&bids, 0, bytemuck::cast_slice(&ids)).unwrap();
    let blist = c.create_device_buffer((n_exp * (batch + 1) * 4) as u64).unwrap();
    for (k, m) in [(3584usize, 1024usize), (1024, 3584)] {
        let x: Vec<f32> = (0..batch * n_used * k).map(|i| ((i % 31) as f32 - 15.0) * 0.01).collect();
        let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
        c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
        let by = c.create_device_buffer((batch * n_used * m * 4) as u64).unwrap();
        for wt in [WType::IQ3XXS, WType::IQ3S, WType::Q4K, WType::Q6K, WType::Q8_0, WType::IQ2S, WType::IQ4XS, WType::NVFP4] {
            let slice = m * wt.row_bytes(k as u32) as usize;
            let raw: Vec<u8> = (0..n_exp * slice).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8 & 0x3F).collect();
            let bw = c.create_device_buffer(raw.len() as u64).unwrap();
            c.upload(&bw, 0, &raw).unwrap();
            let run = |n: usize| {
                let mut rec = c.begin();
                rec.moe_group(&bids, &blist, (batch * n_used) as u32, n_exp as u32, batch as u32);
                rec.barrier();
                for _ in 0..n {
                    rec.qmm_grouped(wt, &bw, 0, slice as u64, &bx, 0, k as u32, if k > m { n_used as u32 } else { 1 }, if k > m { batch as u32 } else { (batch * n_used) as u32 }, &by, 0, k as u32, m as u32, n_exp as u32, Some(&blist), batch as u32);
                }
                rec.finish_and_wait().unwrap();
            };
            run(2);
            let t = std::time::Instant::now();
            run(20);
            let dt = t.elapsed().as_secs_f64() / 20.0;
            eprintln!("{wt:?} k={k} m={m}: {:.3} ms  {:.2} TFLOPS", dt * 1e3, 2.0 * (k * m * batch * n_used) as f64 / dt / 1e12);
        }
    }
}

#[test]
fn vk_attention_prefill_matches_batch() {
    let c = VkContext::new().expect("vulkan init");
    let mut s = 0x51ED_2701u64;
    let mut rnd = || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    // (n_heads, n_kv_heads, head_dim, pos0, n_tok, sliding_window)
    for &(nh, nkv, hd, pos0, ntok, sw) in &[(16u32, 4u32, 128u32, 37u32, 75u32, 0u32), (8, 8, 64, 0, 40, 0), (16, 2, 256, 300, 50, 0), (12, 4, 128, 500, 70, 64), (24, 4, 256, 100, 70, 0)] {
        let n_ctx = (pos0 + ntok) as usize;
        let n_pages = n_ctx.div_ceil(16);
        let half_kv_dim = (nkv * hd / 2) as usize;
        // Paged cache with a shuffled page table.
        let kvd: Vec<u32> = (0..n_pages * 2 * 16 * half_kv_dim).map(|_| half::f16::from_f32(rnd()).to_bits() as u32 | ((half::f16::from_f32(rnd()).to_bits() as u32) << 16)).collect();
        let pt: Vec<u32> = (0..n_pages as u32).rev().collect();
        let qv: Vec<f32> = (0..(ntok * nh * hd) as usize).map(|_| rnd() * 2.0).collect();
        let bkv = c.create_device_buffer((kvd.len() * 4) as u64).unwrap();
        c.upload(&bkv, 0, bytemuck::cast_slice(&kvd)).unwrap();
        let bpt = c.create_device_buffer((pt.len() * 4).max(16) as u64).unwrap();
        c.upload(&bpt, 0, bytemuck::cast_slice(&pt)).unwrap();
        let bq = c.create_device_buffer((qv.len() * 4) as u64).unwrap();
        c.upload(&bq, 0, bytemuck::cast_slice(&qv)).unwrap();
        let bo1 = c.create_device_buffer((qv.len() * 4) as u64).unwrap();
        let bo2 = c.create_device_buffer((qv.len() * 4) as u64).unwrap();
        let mut rec = c.begin();
        rec.attention_prefill(&bq, 0, &bkv, 0, &bpt, 0, &bo1, 0, nh, nkv, hd, pos0, sw, ntok);
        rec.attention_batch(&bq, 0, &bkv, 0, &bpt, 0, &bo2, 0, nh, nkv, hd, pos0, sw, ntok);
        rec.finish_and_wait().unwrap();
        let mut o1 = vec![0f32; qv.len()];
        let mut o2 = vec![0f32; qv.len()];
        c.download(&bo1, 0, bytemuck::cast_slice_mut(&mut o1)).unwrap();
        c.download(&bo2, 0, bytemuck::cast_slice_mut(&mut o2)).unwrap();
        let maxerr = o1.iter().zip(&o2).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(maxerr < 2e-3, "attention_prefill mismatch nh={nh} nkv={nkv} hd={hd} pos0={pos0} ntok={ntok} sw={sw}: max err {maxerr}");
    }
}

#[test]
#[ignore = "benchmark"]
fn vk_small_batch_bench() {
    // Decode-shaped matvecs (Qwen3.8 FFN: k=5120, m=17408). Six weight copies are rotated so
    // the weights come from VRAM rather than the 96 MB Infinity Cache.
    let c = VkContext::new().expect("vulkan init");
    let (k, m) = (5120usize, 17408usize);
    let x: Vec<f32> = (0..8 * k).map(|i| ((i % 31) as f32 - 15.0) * 0.01).collect();
    let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
    c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
    let by = c.create_device_buffer((8 * m * 4) as u64).unwrap();
    for wt in [WType::IQ2XXS, WType::IQ2S, WType::IQ2XS, WType::IQ3XXS, WType::IQ3S, WType::IQ1S, WType::Q2K, WType::Q3K, WType::Q4K, WType::Q6K, WType::Q8_0] {
        let rb = wt.row_bytes(k as u32) as usize;
        let raw: Vec<u8> = (0..m * rb).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8 & 0x3F).collect();
        let bws: Vec<_> = (0..6).map(|_| {
            let b = c.create_device_buffer(raw.len() as u64).unwrap();
            c.upload(&b, 0, &raw).unwrap();
            b
        }).collect();
        for batch in [1u32, 2] {
            let run = |n: usize| {
                let mut rec = c.begin();
                for i in 0..n {
                    rec.qmm(wt, &bws[i % 6], 0, &bx, 0, &by, 0, k as u32, m as u32, batch);
                }
                rec.finish_and_wait().unwrap();
            };
            run(30);
            let mut dt = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                run(60);
                dt = dt.min(t.elapsed().as_secs_f64() / 60.0);
            }
            eprintln!("{wt:?} batch={batch}: {:.1} us  ({:.0} GB/s weights)", dt * 1e6, raw.len() as f64 / dt / 1e9);
        }
    }
}

#[test]
fn vk_qmm_grouped_matches_qmv() {
    let c = VkContext::new().expect("vulkan init");
    let (k, m, n_exp, n_used, batch) = (512usize, 70usize, 6usize, 2usize, 45usize);
    let mut s = 0x9E37_79B9u64;
    let mut next = || -> u32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    // Routing: n_used distinct experts per token, skewed so some experts exceed one 32-column tile.
    let mut ids = Vec::new();
    for _ in 0..batch {
        let a = if next() % 3 == 0 { (next() % n_exp as u32) as usize } else { 0 };
        let mut b = (next() % n_exp as u32) as usize;
        if b == a { b = (a + 1) % n_exp; }
        ids.push(a as u32);
        ids.push(b as u32);
    }
    let x: Vec<f32> = (0..batch * k).map(|i| ((i % 37) as f32 - 18.0) * 0.03).collect();
    let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
    c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
    let bids = c.create_device_buffer((ids.len() * 4) as u64).unwrap();
    c.upload(&bids, 0, bytemuck::cast_slice(&ids)).unwrap();
    let blist = c.create_device_buffer((n_exp * (batch + 1) * 4) as u64).unwrap();
    let n_ent = batch * n_used;
    let by_g = c.create_device_buffer((n_ent * m * 4) as u64).unwrap();
    let by_r = c.create_device_buffer((n_ent * m * 4) as u64).unwrap();
    for (wt, mmq) in [WType::IQ3S, WType::Q4K, WType::IQ4NL, WType::Q8_0, WType::NVFP4, WType::Q6K, WType::IQ2S]
        .into_iter()
        .flat_map(|wt| [(wt, false), (wt, true)])
    {
        let rb = wt.row_bytes(k as u32) as usize;
        let slice = m * rb;
        let raw: Vec<u8> = (0..n_exp * slice).map(|_| (next() & 0x3F) as u8).collect();
        let bw = c.create_device_buffer(raw.len() as u64).unwrap();
        c.upload(&bw, 0, &raw).unwrap();
        // Expert mode: entry e = t * n_used + slot reads X row t.
        let mut rec = c.begin();
        rec.set_mmq(mmq);
        rec.moe_group(&bids, &blist, n_ent as u32, n_exp as u32, batch as u32);
        rec.barrier();
        assert!(rec.qmm_grouped(wt, &bw, 0, slice as u64, &bx, 0, k as u32, n_used as u32, batch as u32, &by_g, 0, k as u32, m as u32, n_exp as u32, Some(&blist), batch as u32));
        for (e, &ex) in ids.iter().enumerate() {
            rec.qmv_classic(wt, &bw, ex as u64 * slice as u64, &bx, ((e / n_used) * k) as u64, &by_r, (e * m) as u64, k as u32, m as u32);
        }
        rec.finish_and_wait().unwrap();
        let mut yg = vec![0f32; n_ent * m];
        let mut yr = vec![0f32; n_ent * m];
        c.download(&by_g, 0, bytemuck::cast_slice_mut(&mut yg)).unwrap();
        c.download(&by_r, 0, bytemuck::cast_slice_mut(&mut yr)).unwrap();
        let scale = yr.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
        // fp16 tiles: ~5e-4 of scale; int8 activations: ~2e-4 mean, up to ~1e-2 per element.
        let tol = if mmq && qmmq_index(wt).is_some() { 2e-2 } else { 3e-3 };
        for i in 0..n_ent * m {
            assert!((yg[i] - yr[i]).abs() <= tol * scale, "grouped(list) {wt:?} mmq={mmq} idx={i}: {} vs {}", yg[i], yr[i]);
        }
        // Head mode: group g = head, entry = t * n_groups + g, X rows strided.
        let (nh, stride) = (3usize, k + 64);
        let cols = 15usize;
        let xh: Vec<f32> = (0..cols * nh * stride).map(|i| ((i % 41) as f32 - 20.0) * 0.02).collect();
        let bxh = c.create_device_buffer((xh.len() * 4) as u64).unwrap();
        c.upload(&bxh, 0, bytemuck::cast_slice(&xh)).unwrap();
        let mut rec = c.begin();
        assert!(rec.qmm_grouped(wt, &bw, 0, slice as u64, &bxh, 0, stride as u32, 1, (cols * nh) as u32, &by_g, 0, k as u32, m as u32, nh as u32, None, cols as u32));
        for t in 0..cols {
            for h in 0..nh {
                let e = t * nh + h;
                rec.qmv_classic(wt, &bw, (h * slice) as u64, &bxh, (e * stride) as u64, &by_r, (e * m) as u64, k as u32, m as u32);
            }
        }
        rec.finish_and_wait().unwrap();
        let n = cols * nh * m;
        c.download(&by_g, 0, bytemuck::cast_slice_mut(&mut yg[..n])).unwrap();
        c.download(&by_r, 0, bytemuck::cast_slice_mut(&mut yr[..n])).unwrap();
        let scale = yr[..n].iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
        for i in 0..n {
            assert!((yg[i] - yr[i]).abs() <= 3e-3 * scale, "grouped(heads) {wt:?} idx={i}: {} vs {}", yg[i], yr[i]);
        }
    }
}

#[test]
fn vk_qmv_multi_matches_classic() {
    let c = VkContext::new().expect("vulkan init");
    let (k, m) = (512usize, 37usize);
    let mut s = 0xBADC_0FFEu64;
    let mut next_u8 = || -> u8 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 32) as u8
    };
    let x: Vec<f32> = (0..8 * k).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
    let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
    c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
    let by_a = c.create_device_buffer((8 * m * 4) as u64).unwrap();
    let by_b = c.create_device_buffer((8 * m * 4) as u64).unwrap();
    for wt in [
        WType::Q4K, WType::Q5K, WType::Q6K, WType::Q2K, WType::Q3K, WType::Q8_0, WType::Q4_0, WType::F16, WType::F32, WType::BF16,
        WType::IQ3S, WType::IQ3XXS, WType::IQ2XS, WType::IQ2S, WType::IQ2XXS, WType::IQ1S, WType::IQ1M, WType::IQ4XS, WType::IQ4NL, WType::NVFP4,
    ] {
        let rb = wt.row_bytes(k as u32) as usize;
        let raw: Vec<u8> = if wt == WType::F32 {
            let f: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.04).collect();
            bytemuck::cast_slice(&f).to_vec()
        } else {
            (0..m * rb).map(|_| next_u8() & 0x3F).collect()
        };
        let bw = c.create_device_buffer(raw.len() as u64).unwrap();
        c.upload(&bw, 0, &raw).unwrap();
        for batch in 1..=8u32 {
            let mut rec = c.begin();
        rec.set_mmq(false);
            rec.qmm(wt, &bw, 0, &bx, 0, &by_a, 0, k as u32, m as u32, batch);
            for b in 0..batch as usize {
                rec.qmv_classic(wt, &bw, 0, &bx, (b * k) as u64, &by_b, (b * m) as u64, k as u32, m as u32);
            }
            rec.finish_and_wait().unwrap();
            let n = batch as usize * m;
            let mut ya = vec![0f32; n];
            let mut yb = vec![0f32; n];
            c.download(&by_a, 0, bytemuck::cast_slice_mut(&mut ya)).unwrap();
            c.download(&by_b, 0, bytemuck::cast_slice_mut(&mut yb)).unwrap();
            for i in 0..n {
                let tol = 1e-3 + 1e-4 * yb[i].abs();
                assert!((ya[i] - yb[i]).abs() <= tol, "qmv_multi {wt:?} batch={batch} idx={i}: {} vs {}", ya[i], yb[i]);
            }
        }
    }
}

#[test]
fn vk_qmv_id_matches_classic() {
    let c = VkContext::new().expect("vulkan init");
    let (k, m, n_exp, count) = (512usize, 33usize, 5usize, 3usize);
    let mut s = 0x00DD_BA11_u64;
    let mut next_u8 = || -> u8 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 32) as u8
    };
    let ids: Vec<u32> = vec![9, 9, 4, 0, 2]; // slots 2..5 used (slot_base = 2)
    let bids = c.create_device_buffer((ids.len() * 4) as u64).unwrap();
    c.upload(&bids, 0, bytemuck::cast_slice(&ids)).unwrap();
    let x: Vec<f32> = (0..count * k).map(|i| ((i % 29) as f32 - 14.0) * 0.05).collect();
    let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
    c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
    let by_a = c.create_device_buffer((count * m * 4) as u64).unwrap();
    let by_b = c.create_device_buffer((count * m * 4) as u64).unwrap();
    for wt in [WType::IQ3S, WType::Q8_0, WType::Q4K, WType::Q6K, WType::IQ2XS, WType::NVFP4, WType::Q2K, WType::IQ1S] {
        let slice = m * wt.row_bytes(k as u32) as usize;
        let raw: Vec<u8> = (0..n_exp * slice).map(|_| next_u8() & 0x3F).collect();
        let bw = c.create_device_buffer(raw.len() as u64).unwrap();
        c.upload(&bw, 0, &raw).unwrap();
        let mut rec = c.begin();
        rec.qmv_id_batch(wt, &bw, &bids, 2, count as u32, &bx, 0, k as u32, &by_a, 0, k as u32, m as u32, slice as u64);
        for sl in 0..count {
            let e = ids[2 + sl] as u64;
            rec.qmv_classic(wt, &bw, e * slice as u64, &bx, (sl * k) as u64, &by_b, (sl * m) as u64, k as u32, m as u32);
        }
        rec.finish_and_wait().unwrap();
        let mut ya = vec![0f32; count * m];
        let mut yb = vec![0f32; count * m];
        c.download(&by_a, 0, bytemuck::cast_slice_mut(&mut ya)).unwrap();
        c.download(&by_b, 0, bytemuck::cast_slice_mut(&mut yb)).unwrap();
        for i in 0..count * m {
            assert!((ya[i] - yb[i]).abs() <= 1e-3 + 1e-4 * yb[i].abs(), "qmv_id {wt:?} idx={i}: {} vs {}", ya[i], yb[i]);
        }
    }
}

#[test]
fn vk_qmm_tiled_matches_qmv() {
    let c = VkContext::new().expect("vulkan init");
    // m and batch deliberately not multiples of the 64x128 tile; the second shape (small m,
    // large k) takes the split-K path.
    for (k, m, batch) in [(512usize, 70usize, 75usize), (4608, 24, 200)] {
        let mut s = 0x1234_5678u64;
        let mut next_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 32) as u8
        };
        let x: Vec<f32> = (0..batch * k).map(|i| ((i % 31) as f32 - 15.0) * 0.05).collect();
        let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
        c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
        let by_qmm = c.create_device_buffer((batch * m * 4) as u64).unwrap();
        let by_qmv = c.create_device_buffer((batch * m * 4) as u64).unwrap();
        for wt in [
            WType::Q4K, WType::Q5K, WType::Q6K, WType::Q2K, WType::Q3K, WType::Q8_0, WType::Q4_0, WType::F16, WType::F32,
            WType::IQ3S, WType::IQ3XXS, WType::IQ2XS, WType::IQ2S, WType::IQ2XXS, WType::IQ1S, WType::IQ1M, WType::IQ4XS,
            WType::IQ4NL, WType::BF16, WType::NVFP4,
        ] {
            let rb = wt.row_bytes(k as u32) as usize;
            let raw: Vec<u8> = if wt == WType::F32 {
                let f: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.04).collect();
                bytemuck::cast_slice(&f).to_vec()
            } else {
                (0..m * rb).map(|_| next_u8() & 0x3F).collect()
            };
            let bw = c.create_device_buffer(raw.len() as u64).unwrap();
            c.upload(&bw, 0, &raw).unwrap();
            let mut rec = c.begin();
            rec.set_mmq(false);
            rec.qmm(wt, &bw, 0, &bx, 0, &by_qmm, 0, k as u32, m as u32, batch as u32);
            for b in 0..batch {
                rec.qmv_classic(wt, &bw, 0, &bx, (b * k) as u64, &by_qmv, (b * m) as u64, k as u32, m as u32);
            }
            rec.finish_and_wait().unwrap();
            let mut y_qmm = vec![0f32; batch * m];
            let mut y_qmv = vec![0f32; batch * m];
            c.download(&by_qmm, 0, bytemuck::cast_slice_mut(&mut y_qmm)).unwrap();
            c.download(&by_qmv, 0, bytemuck::cast_slice_mut(&mut y_qmv)).unwrap();
            // The tiled kernel stages weights and activations as fp16 in shared memory (like
            // llama.cpp's mul_mm), so allow fp16-level error relative to the output scale. Indexing
            // bugs produce errors on the order of the scale itself.
            let scale = y_qmv.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
            for i in 0..batch * m {
                let tol = 3e-3 * scale;  // observed max ~5e-4
                assert!(
                    (y_qmm[i] - y_qmv[i]).abs() <= tol,
                    "tiled qmm vs qmv mismatch for {wt:?} k={k} m={m} idx={i}: qmm={} qmv={}",
                    y_qmm[i],
                    y_qmv[i]
                );
            }
        }
    }
}

#[test]
fn vk_qmm_int8_matches_qmv() {
    let c = VkContext::new().expect("vulkan init");
    // m and batch not multiples of the 64x128 tile; k spans two 256-blocks. The second shape
    // (small m, large k) takes the split-K path.
    for (k, m, batch) in [(512usize, 70usize, 150usize), (4608, 48, 200)] {
        let mut s = 0x0BAD_F00Du64;
        let mut next_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 32) as u8
        };
        // Activations with varying magnitude per 32-block, including all-zero blocks.
        let x: Vec<f32> = (0..batch * k)
            .map(|i| if (i / 32) % 17 == 5 { 0.0 } else { ((i * 7 % 31) as f32 - 15.0) * 0.05 * (1.0 + (i / 32 % 5) as f32) })
            .collect();
        let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
        c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
        let by_a = c.create_device_buffer((batch * m * 4) as u64).unwrap();
        let by_b = c.create_device_buffer((batch * m * 4) as u64).unwrap();
        let by_ref = c.create_device_buffer((batch * m * 4) as u64).unwrap();
        for wt in [WType::Q4K, WType::Q5K, WType::Q6K, WType::Q8_0, WType::Q2K, WType::Q3K, WType::IQ4NL, WType::IQ4XS, WType::IQ3S, WType::IQ3XXS, WType::IQ2XS, WType::IQ2S, WType::IQ2XXS, WType::IQ1S, WType::IQ1M, WType::NVFP4] {
            let rb = wt.row_bytes(k as u32) as usize;
            let raw: Vec<u8> = (0..m * rb).map(|_| next_u8() & 0x3F).collect();
            let bw = c.create_device_buffer(raw.len() as u64).unwrap();
            c.upload(&bw, 0, &raw).unwrap();
            let mut rec = c.begin();
            rec.set_defer_barriers(true);
            assert!(rec.qmm_int8(wt, &bw, 0, &bx, 0, &by_a, 0, k as u32, m as u32, batch as u32));
            // A second matmul on the same X reuses the quantized activations.
            assert!(rec.q8_cache.is_some());
            assert!(rec.qmm_int8(wt, &bw, 0, &bx, 0, &by_b, 0, k as u32, m as u32, batch as u32));
            rec.barrier();
            for b in 0..batch {
                rec.qmv_classic(wt, &bw, 0, &bx, (b * k) as u64, &by_ref, (b * m) as u64, k as u32, m as u32);
            }
            rec.barrier();
            rec.finish_and_wait().unwrap();
            let mut ya = vec![0f32; batch * m];
            let mut yb = vec![0f32; batch * m];
            let mut yr = vec![0f32; batch * m];
            c.download(&by_a, 0, bytemuck::cast_slice_mut(&mut ya)).unwrap();
            c.download(&by_b, 0, bytemuck::cast_slice_mut(&mut yb)).unwrap();
            c.download(&by_ref, 0, bytemuck::cast_slice_mut(&mut yr)).unwrap();
            assert_eq!(ya, yb, "{wt:?}: reused quantization differs");
            // int8 activations (like llama.cpp's q8_1 MMQ): error ~ amax/254 per element.
            let scale = yr.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
            let mut sum_err = 0f64;
            for i in 0..batch * m {
                let e = (ya[i] - yr[i]).abs();
                sum_err += e as f64;
                assert!(e <= 2e-2 * scale, "int8 qmm vs qmv mismatch for {wt:?} idx={i}: {} vs {}", ya[i], yr[i]);
            }
            let mean = sum_err / (batch * m) as f64;
            eprintln!("{wt:?}: int8 max-scale {scale} mean err {mean:.3e}");
            assert!(mean <= 3e-3 * scale as f64, "{wt:?}: mean error {mean} too high (scale {scale})");
        }
    }
}

#[test]
fn vk_qmv_int8_matches_classic() {
    let c = VkContext::new().expect("vulkan init");
    let (k, m) = (1024usize, 37usize);
    let mut s = 0x5EED_1234u64;
    let mut next_u8 = || -> u8 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 32) as u8
    };
    let x: Vec<f32> = (0..8 * k).map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.07 * (1.0 + (i / 32 % 3) as f32)).collect();
    let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
    c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
    let by_a = c.create_device_buffer((8 * m * 4) as u64).unwrap();
    let by_b = c.create_device_buffer((8 * m * 4) as u64).unwrap();
    for wt in [
        WType::Q4K, WType::Q5K, WType::Q6K, WType::Q8_0, WType::Q2K, WType::Q3K, WType::IQ4NL, WType::IQ4XS,
        WType::IQ3S, WType::IQ3XXS, WType::IQ2XS, WType::IQ2S, WType::IQ2XXS, WType::IQ1S, WType::IQ1M, WType::NVFP4,
    ] {
        let rb = wt.row_bytes(k as u32) as usize;
        let raw: Vec<u8> = (0..m * rb).map(|_| next_u8() & 0x3F).collect();
        let bw = c.create_device_buffer(raw.len() as u64).unwrap();
        c.upload(&bw, 0, &raw).unwrap();
        for batch in [1usize, 3, 8] {
            let mut rec = c.begin();
            assert!(rec.qmv_int8_dispatch(wt, &bw, 0, &bx, 0, &by_a, 0, k as u32, m as u32, batch as u32));
            for b in 0..batch {
                rec.qmv_classic(wt, &bw, 0, &bx, (b * k) as u64, &by_b, (b * m) as u64, k as u32, m as u32);
            }
            rec.finish_and_wait().unwrap();
            let mut ya = vec![0f32; batch * m];
            let mut yb = vec![0f32; batch * m];
            c.download(&by_a, 0, bytemuck::cast_slice_mut(&mut ya)).unwrap();
            c.download(&by_b, 0, bytemuck::cast_slice_mut(&mut yb)).unwrap();
            let scale = yb.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
            let mut sum = 0f64;
            for i in 0..batch * m {
                let e = (ya[i] - yb[i]).abs();
                sum += e as f64;
                assert!(e <= 2e-2 * scale, "int8 qmv {wt:?} batch={batch} idx={i}: {} vs {}", ya[i], yb[i]);
            }
            assert!(sum / ((batch * m) as f64) <= 3e-3 * scale as f64, "int8 qmv {wt:?} batch={batch}: mean error too high");
        }
    }
}

#[test]
fn vk_qmm_matches_qmv() {
    let c = VkContext::new().expect("vulkan init");
    let (k, m, batch) = (512usize, 13usize, 11usize);
    let mut s = 0xCAFE_BABEu64;
    let mut next_u8 = || -> u8 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 32) as u8
    };
    let x: Vec<f32> = (0..batch * k).map(|i| ((i % 29) as f32 - 14.0) * 0.05).collect();
    let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
    c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
    let by_qmm = c.create_device_buffer((batch * m * 4) as u64).unwrap();
    let by_qmv = c.create_device_buffer((batch * m * 4) as u64).unwrap();

    for wt in [WType::Q4K, WType::Q6K, WType::Q5K, WType::Q4_0, WType::Q8_0, WType::F16, WType::F32, WType::IQ2S, WType::IQ2XS, WType::IQ3S, WType::IQ3XXS] {
        let rb = wt.row_bytes(k as u32) as usize;
        let raw: Vec<u8> = if wt == WType::F32 {
            let f: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.04).collect();
            bytemuck::cast_slice(&f).to_vec()
        } else {
            // Keep f16 scales finite (avoid NaN/Inf exponents 0x7C..0x7F in low/high bytes)
            (0..m * rb).map(|_| next_u8() & 0x3F).collect()
        };
        let bw = c.create_device_buffer(raw.len() as u64).unwrap();
        c.upload(&bw, 0, &raw).unwrap();
        let mut rec = c.begin();
        rec.set_mmq(false);
        rec.qmm(wt, &bw, 0, &bx, 0, &by_qmm, 0, k as u32, m as u32, batch as u32);
        for b in 0..batch {
            rec.qmv_classic(wt, &bw, 0, &bx, (b * k) as u64, &by_qmv, (b * m) as u64, k as u32, m as u32);
        }
        rec.finish_and_wait().unwrap();
        let mut y_qmm = vec![0f32; batch * m];
        let mut y_qmv = vec![0f32; batch * m];
        c.download(&by_qmm, 0, bytemuck::cast_slice_mut(&mut y_qmm)).unwrap();
        c.download(&by_qmv, 0, bytemuck::cast_slice_mut(&mut y_qmv)).unwrap();
        for i in 0..batch * m {
            let tol = 1e-3 + 1e-4 * y_qmv[i].abs();
            assert!(
                (y_qmm[i] - y_qmv[i]).abs() <= tol,
                "qmm vs qmv mismatch for {wt:?} idx={i}: qmm={} qmv={}",
                y_qmm[i],
                y_qmv[i]
            );
        }
        // Also verify batch=2 (small-batch matvec path)
        let mut rec2 = c.begin();
        rec2.set_mmq(false);
        rec2.qmm(wt, &bw, 0, &bx, 0, &by_qmm, 0, k as u32, m as u32, 2);
        rec2.finish_and_wait().unwrap();
        let mut y_qmm2 = vec![0f32; 2 * m];
        c.download(&by_qmm, 0, bytemuck::cast_slice_mut(&mut y_qmm2)).unwrap();
        for i in 0..2 * m {
            let tol = 1e-3 + 1e-4 * y_qmv[i].abs();
            assert!(
                (y_qmm2[i] - y_qmv[i]).abs() <= tol,
                "qmm2 vs qmv mismatch for {wt:?} idx={i}: qmm2={} qmv={}",
                y_qmm2[i],
                y_qmv[i]
            );
        }
    }
}
