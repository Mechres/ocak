//! Dynamic `dlopen` AMD HIP runtime (`libamdhip64.so`) + `libhiprtc.so` JIT compiler.
//! Builds cleanly with zero ROCm SDK installed; lights up automatically when ROCm is present.
pub mod dense_ops;
pub mod ffi;
pub mod hybrid_ops;
pub mod kernels;
pub mod rtc;
pub mod runtime;

pub use dense_ops::HipDenseOps;
pub use ffi::{
    DeviceInfo, HipError, HipStatus, MemcpyKind, HIP_MEMCPY_D2D, HIP_MEMCPY_D2H, HIP_MEMCPY_H2D,
    HIP_MEMCPY_H2H,
};
pub use hybrid_ops::HipHybridOps;
pub use kernels::{HipKernels, HipWType};
pub use rtc::Hiprtc;
pub use runtime::{
    HipBuffer, HipContext, HipEvent, HipFunction, HipGraphExec, HipModule, HipPinnedBuffer,
    HipStream,
};

use std::ffi::c_int;

/// Lightweight probe entry point (kept compatible with earlier `ocak-hip` API).
pub struct HipRuntime {
    api: ffi::HipApi,
}

impl HipRuntime {
    pub fn load() -> Result<Self, HipError> {
        Ok(Self {
            api: ffi::HipApi::load()?,
        })
    }

    pub fn probe() -> HipStatus {
        match Self::load() {
            Ok(rt) => HipStatus {
                available: true,
                library: Some(rt.api.library_name.clone()),
                device_count: rt.device_count().unwrap_or(0),
            },
            Err(_) => HipStatus {
                available: false,
                library: None,
                device_count: 0,
            },
        }
    }

    pub fn device_count(&self) -> Result<i32, HipError> {
        let mut n: c_int = 0;
        self.api.check(unsafe { (self.api.get_device_count)(&mut n) })?;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use ocak_gguf::GgmlType;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn f(&mut self) -> f32 {
            (self.next() as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
        }
    }

    fn rand_weights(ty: GgmlType, k: usize, m: usize, rng: &mut Lcg) -> Vec<u8> {
        let (be, bb) = ty.block().unwrap();
        let mut v: Vec<u8> = (0..(k / be * m * bb)).map(|_| rng.next() as u8).collect();
        match ty {
            GgmlType::F32 => {
                for c in v.chunks_exact_mut(4) {
                    c.copy_from_slice(&(rng.f() * 0.5).to_le_bytes());
                }
            }
            GgmlType::F16 => {
                for c in v.chunks_exact_mut(2) {
                    c.copy_from_slice(&f16::from_f32(rng.f() * 0.5).to_le_bytes());
                }
            }
            GgmlType::Q4_0 | GgmlType::Q8_0 => {
                for blk in v.chunks_exact_mut(bb) {
                    blk[0..2].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
                }
            }
            GgmlType::Q4K | GgmlType::Q5K => {
                for blk in v.chunks_exact_mut(bb) {
                    blk[0..2].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
                    blk[2..4].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
                }
            }
            GgmlType::Q6K => {
                for blk in v.chunks_exact_mut(bb) {
                    blk[208..210].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
                }
            }
            GgmlType::Q2K => {
                for blk in v.chunks_exact_mut(bb) {
                    blk[80..82].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
                    blk[82..84].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
                }
            }
            GgmlType::Q3K => {
                for blk in v.chunks_exact_mut(bb) {
                    blk[108..110].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
        v
    }

    fn try_ctx() -> Option<std::sync::Arc<HipContext>> {
        match HipContext::new(0) {
            Ok(c) => Some(c),
            Err(HipError::Unavailable) => None,
            Err(e) => panic!("HIP init failed: {e}"),
        }
    }

    #[test]
    fn probe_reports_cleanly_without_rocm() {
        let st = HipRuntime::probe();
        assert!(st.device_count >= 0);
    }

    #[test]
    fn hip_device_info_and_memory_roundtrip() {
        let Some(ctx) = try_ctx() else { return };
        assert!(!ctx.info.name.is_empty());
        assert!(ctx.info.cu_count > 0);
        assert!(ctx.info.warp_size == 32 || ctx.info.warp_size == 64);
        assert!(ctx.info.total_vram_bytes > 0);

        let stream = ctx.create_stream().unwrap();
        let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.25).collect();
        let buf = ctx.buffer_from_slice(&data).unwrap();
        let mut out = vec![0f32; 1024];
        buf.download(bytemuck::cast_slice_mut(&mut out)).unwrap();
        assert_eq!(data, out);

        // Pinned async roundtrip
        let mut pinned_in = ctx.alloc_pinned(data.len() * 4).unwrap();
        pinned_in.as_mut_slice().copy_from_slice(bytemuck::cast_slice(&data));
        let mut pinned_out = ctx.alloc_pinned(data.len() * 4).unwrap();
        buf.zero_async(&stream).unwrap();
        buf.upload_pinned_async(0, &pinned_in, data.len() * 4, &stream).unwrap();
        buf.download_pinned_async(0, &mut pinned_out, data.len() * 4, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(pinned_out.as_slice(), bytemuck::cast_slice(&data));
    }

    #[test]
    fn hip_rms_norm_matches_cpu() {
        let Some(ctx) = try_ctx() else { return };
        let kernels = HipKernels::compile(&ctx).unwrap();
        let stream = ctx.create_stream().unwrap();
        let mut rng = Lcg(999);
        let n = 512usize;
        let x: Vec<f32> = (0..n).map(|_| rng.f()).collect();
        let w: Vec<f32> = (0..n).map(|_| rng.f()).collect();
        let mut y_cpu = vec![0f32; n];
        ocak_cpu::ops::rms_norm(&x, &w, &mut y_cpu, 1e-5);

        let bx = ctx.buffer_from_slice(&x).unwrap();
        let bw = ctx.buffer_from_slice(&w).unwrap();
        let by = ctx.alloc_buffer(n * 4).unwrap();
        kernels.rms_norm(&stream, &bx, &bw, &by, n as u32, 1, 1e-5).unwrap();
        stream.synchronize().unwrap();

        let mut y_gpu = vec![0f32; n];
        by.download(bytemuck::cast_slice_mut(&mut y_gpu)).unwrap();
        for i in 0..n {
            assert!((y_cpu[i] - y_gpu[i]).abs() < 1e-4, "rms[{i}]: cpu={} gpu={}", y_cpu[i], y_gpu[i]);
        }
    }

    #[test]
    fn hip_qmv_and_qmm_match_cpu() {
        let Some(ctx) = try_ctx() else { return };
        let kernels = HipKernels::compile(&ctx).unwrap();
        let stream = ctx.create_stream().unwrap();
        let mut rng = Lcg(1337);

        for (ty, hty) in [
            (GgmlType::F32, HipWType::F32),
            (GgmlType::F16, HipWType::F16),
            (GgmlType::Q4_0, HipWType::Q4_0),
            (GgmlType::Q8_0, HipWType::Q8_0),
            (GgmlType::Q4K, HipWType::Q4K),
            (GgmlType::Q5K, HipWType::Q5K),
            (GgmlType::Q6K, HipWType::Q6K),
            (GgmlType::Q2K, HipWType::Q2K),
            (GgmlType::Q3K, HipWType::Q3K),
        ] {
            let (k, m, batch) = (512usize, 70usize, 37usize);
            let w = rand_weights(ty, k, m, &mut rng);
            let x: Vec<f32> = (0..batch * k).map(|_| rng.f()).collect();

            let mut y_cpu = vec![0f32; batch * m];
            for b in 0..batch {
                ocak_cpu::gemv(ty, &w, &x[b * k..(b + 1) * k], &mut y_cpu[b * m..(b + 1) * m], k, m);
            }

            let bw = ctx.buffer_from_slice(&w).unwrap();
            let bx = ctx.buffer_from_slice(&x).unwrap();
            let n_blocks = (batch * k / 32) as u32;
            let bxq = ctx.alloc_buffer(batch * k).unwrap();
            let bxsc = ctx.alloc_buffer(n_blocks as usize * 16).unwrap();
            let by = ctx.alloc_buffer(batch * m * 4).unwrap();

            if hty.uses_q8_act() {
                kernels.quant_q8(&stream, &bx, &bxq, &bxsc, n_blocks).unwrap();
            }

            // 1) Decode qmv for batch = 3
            let b_dec = 3u32;
            kernels.qmv(hty, &stream, &bw, &bx, &bxq, &bxsc, &by, k as u32, m as u32, b_dec).unwrap();
            stream.synchronize().unwrap();
            let mut y_dec = vec![0f32; b_dec as usize * m];
            by.download_at(0, bytemuck::cast_slice_mut(&mut y_dec)).unwrap();
            let scale = y_cpu.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
            let mut sum_err = 0f64;
            for i in 0..y_dec.len() {
                let err = (y_cpu[i] - y_dec[i]).abs();
                sum_err += err as f64;
                assert!(err <= 2e-2 * scale, "qmv {ty:?}[{i}]: cpu={} gpu={} err={err} scale={scale}", y_cpu[i], y_dec[i]);
            }
            let mean = sum_err / y_dec.len() as f64;
            assert!(mean <= 3e-3 * scale as f64, "qmv {ty:?}: mean err {mean} (scale {scale})");

            // 2) Prefill qmm for full batch = 37
            kernels.qmm(hty, &stream, &bw, &bx, &bxq, &bxsc, &by, k as u32, m as u32, batch as u32).unwrap();
            stream.synchronize().unwrap();
            let mut y_mm = vec![0f32; batch * m];
            by.download(bytemuck::cast_slice_mut(&mut y_mm)).unwrap();
            let mut sum_err_mm = 0f64;
            for i in 0..y_mm.len() {
                let err = (y_cpu[i] - y_mm[i]).abs();
                sum_err_mm += err as f64;
                assert!(err <= 2e-2 * scale, "qmm {ty:?}[{i}]: cpu={} gpu={} err={err} scale={scale}", y_cpu[i], y_mm[i]);
            }
            let mean_mm = sum_err_mm / y_mm.len() as f64;
            assert!(mean_mm <= 3e-3 * scale as f64, "qmm {ty:?}: mean err {mean_mm} (scale {scale})");
            // qmv and qmm use the exact same activations -> should agree tightly on batch 0..3
            for i in 0..y_dec.len() {
                let err = (y_dec[i] - y_mm[i]).abs();
                assert!(err < 1e-4 * scale, "qmv vs qmm {ty:?}[{i}]: {} vs {}", y_dec[i], y_mm[i]);
            }

            // 3) Opt-in GROUP256 Q4_K (rescale once per 256)
            if hty == HipWType::Q4K {
                kernels.quant_q8_g256(&stream, &bx, &bxq, &bxsc, n_blocks).unwrap();
                kernels.qmm_q4_k_g256(&stream, &bw, &bxq, &bxsc, &by, k as u32, m as u32, batch as u32).unwrap();
                stream.synchronize().unwrap();
                let mut y_g256 = vec![0f32; batch * m];
                by.download(bytemuck::cast_slice_mut(&mut y_g256)).unwrap();
                let mut sum_err_g256 = 0f64;
                for i in 0..y_g256.len() {
                    let err = (y_cpu[i] - y_g256[i]).abs();
                    sum_err_g256 += err as f64;
                    assert!(err <= 3e-2 * scale, "qmm_g256 Q4K[{i}]: cpu={} gpu={} err={err} scale={scale}", y_cpu[i], y_g256[i]);
                }
                let mean_g256 = sum_err_g256 / y_g256.len() as f64;
                assert!(mean_g256 <= 4e-3 * scale as f64, "qmm_g256 Q4K: mean err {mean_g256} (scale {scale})");
            }
        }
    }

    #[test]
    fn hip_dense_ops_match_cpu() {
        let Some(ctx) = try_ctx() else { return };
        let ops = HipDenseOps::compile(&ctx).unwrap();
        let stream = ctx.create_stream().unwrap();
        let mut rng = Lcg(777);

        // 1) Embed Q4K + Q6K
        for (ty, hty) in [(GgmlType::Q4K, HipWType::Q4K), (GgmlType::Q6K, HipWType::Q6K)] {
            let w = rand_weights(ty, 512, 5, &mut rng);
            let bw = ctx.buffer_from_slice(&w).unwrap();
            let bo = ctx.alloc_buffer(512 * 4).unwrap();
            ops.embed_token(&stream, &bw, hty, &bo, 3, 512).unwrap();
            stream.synchronize().unwrap();
            let mut e_gpu = vec![0f32; 512];
            bo.download(bytemuck::cast_slice_mut(&mut e_gpu)).unwrap();
            let rb = hty.row_bytes(512) as usize;
            let mut e_cpu = vec![0f32; 512];
            ocak_gguf::quant::dequantize_row(ty, &w[3 * rb..4 * rb], &mut e_cpu, 512);
            for i in 0..512 {
                assert!((e_cpu[i] - e_gpu[i]).abs() < 1e-4, "embed {ty:?}[{i}]");
            }
        }

        // 2) Paged FP16 kv_write + attention (non-identity page table [5, 2, 9], pos=37)
        let (nh, nkv, hd, pos) = (16usize, 4usize, 64usize, 37usize);
        let kv_dim = nkv * hd;
        let n_pages = 12usize;
        let pt = vec![5u32, 2, 9];
        let mut kv_cpu = vec![0f32; n_pages * 2 * 16 * kv_dim];
        let bkv = ctx.alloc_buffer(n_pages * 2 * 16 * kv_dim * 2).unwrap();
        let bpt = ctx.buffer_from_slice(&pt).unwrap();
        for t in 0..=pos {
            let k: Vec<f32> = (0..kv_dim).map(|_| rng.f()).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| rng.f()).collect();
            ocak_cpu::ops::kv_write(&k, &v, &mut kv_cpu, 0, &pt, t, kv_dim);
            let bk = ctx.buffer_from_slice(&k).unwrap();
            let bv = ctx.buffer_from_slice(&v).unwrap();
            ops.kv_write(&stream, &bk, &bv, &bkv, 0, &bpt, t as u32, kv_dim as u32, 1).unwrap();
            stream.synchronize().unwrap();
        }
        let q: Vec<f32> = (0..nh * hd).map(|_| rng.f()).collect();
        let mut a_cpu = vec![0f32; nh * hd];
        ocak_cpu::ops::attention(&q, &kv_cpu, 0, &pt, &mut a_cpu, nh, nkv, hd, pos);
        let bq = ctx.buffer_from_slice(&q).unwrap();
        let bo = ctx.alloc_buffer(nh * hd * 4).unwrap();
        ops.attention(&stream, &bq, &bkv, 0, &bpt, &bo, nh as u32, nkv as u32, hd as u32, pos as u32, 0, 1).unwrap();
        stream.synchronize().unwrap();
        let mut a_gpu = vec![0f32; nh * hd];
        bo.download(bytemuck::cast_slice_mut(&mut a_gpu)).unwrap();
        for i in 0..a_cpu.len() {
            assert!((a_cpu[i] - a_gpu[i]).abs() < 2e-3, "attn[{i}]: cpu={} gpu={}", a_cpu[i], a_gpu[i]);
        }

        // 2b) Split-K decode attention (device-meta) vs CPU
        let meta = [0u32, pos as u32];
        let bmeta = ctx.buffer_from_slice(&meta).unwrap();
        let bpout = ctx.alloc_buffer(nh * 8 * hd * 4).unwrap();
        let bpml = ctx.alloc_buffer(nh * 8 * 2 * 4).unwrap();
        ops.attention_decode_split_meta(
            &stream, &bq, &bkv, 0, &bpt, &bmeta, &bpout, &bpml, &bo,
            nh as u32, nkv as u32, hd as u32, 0, 8,
        ).unwrap();
        stream.synchronize().unwrap();
        bo.download(bytemuck::cast_slice_mut(&mut a_gpu)).unwrap();
        for i in 0..a_cpu.len() {
            assert!((a_cpu[i] - a_gpu[i]).abs() < 2e-3, "attn_split[{i}]: cpu={} gpu={}", a_cpu[i], a_gpu[i]);
        }

        // 2c) Flash-tiled prefill attention (batch=12 ending at pos=37) vs CPU on final token
        let b_pf = 12usize;
        let mut q_pf: Vec<f32> = (0..(b_pf - 1) * nh * hd).map(|_| rng.f()).collect();
        q_pf.extend_from_slice(&q);
        let bq_pf = ctx.buffer_from_slice(&q_pf).unwrap();
        let bo_pf = ctx.alloc_buffer(b_pf * nh * hd * 4).unwrap();
        ops.attention_prefill(
            &stream, &bq_pf, &bkv, 0, &bpt, &bo_pf,
            nh as u32, nkv as u32, hd as u32, (pos + 1 - b_pf) as u32, 0, b_pf as u32,
        ).unwrap();
        stream.synchronize().unwrap();
        bo_pf.download_at((b_pf - 1) * nh * hd * 4, bytemuck::cast_slice_mut(&mut a_gpu)).unwrap();
        for i in 0..a_cpu.len() {
            assert!((a_cpu[i] - a_gpu[i]).abs() < 2e-3, "attn_prefill[{i}]: cpu={} gpu={}", a_cpu[i], a_gpu[i]);
        }

        // 3) GPU argmax
        let mut logits: Vec<f32> = (0..32000).map(|_| rng.f()).collect();
        logits[17321] = 42.5;
        let bl = ctx.buffer_from_slice(&logits).unwrap();
        let bpv = ctx.alloc_buffer(256 * 4).unwrap();
        let bpi = ctx.alloc_buffer(256 * 4).unwrap();
        let bi = ctx.alloc_buffer(4).unwrap();
        ops.argmax(&stream, &bl, &bpv, &bpi, &bi, logits.len() as u32).unwrap();
        stream.synchronize().unwrap();
        let mut idx = [0u32; 1];
        bi.download(bytemuck::cast_slice_mut(&mut idx)).unwrap();
        assert_eq!(idx[0], 17321);
    }

    #[test]
    fn hip_hybrid_shortconv_and_deltanet_match_cpu() {
        let Some(ctx) = try_ctx() else { return };
        let hops = HipHybridOps::compile(&ctx).unwrap();
        let stream = ctx.create_stream().unwrap();
        let mut rng = Lcg(4242);

        // 1) ShortConv (LFM2/LFM3)
        let (ne, lc) = (256usize, 3usize);
        let bcx: Vec<f32> = (0..3 * ne).map(|_| rng.f()).collect();
        let cw: Vec<f32> = (0..ne * lc).map(|_| rng.f()).collect();
        let mut st_cpu = vec![0f32; ne * (lc - 1)];
        let mut o_cpu = vec![0f32; ne];
        let b_bcx = ctx.buffer_from_slice(&bcx).unwrap();
        let b_cw = ctx.buffer_from_slice(&cw).unwrap();
        let b_st = ctx.alloc_buffer(st_cpu.len() * 4).unwrap();
        let b_o = ctx.alloc_buffer(o_cpu.len() * 4).unwrap();
        b_st.zero().unwrap();
        for _ in 0..4 {
            ocak_cpu::ops::shortconv_step(&bcx, &cw, &mut st_cpu, &mut o_cpu, ne, lc);
            hops.shortconv_step(&stream, &b_bcx, &b_cw, &b_st, &b_o, ne as u32, lc as u32, 1).unwrap();
        }
        stream.synchronize().unwrap();
        let mut o_gpu = vec![0f32; ne];
        b_o.download(bytemuck::cast_slice_mut(&mut o_gpu)).unwrap();
        for i in 0..ne {
            assert!((o_cpu[i] - o_gpu[i]).abs() < 1e-4, "shortconv[{i}]: cpu={} gpu={}", o_cpu[i], o_gpu[i]);
        }

        // 2) Gated DeltaNet (Qwen3.5)
        let (nkh, nvh, hd) = (4usize, 8usize, 128usize);
        let qkv: Vec<f32> = (0..(2 * nkh + nvh) * hd).map(|_| rng.f()).collect();
        let alpha: Vec<f32> = (0..nvh).map(|_| rng.f()).collect();
        let beta: Vec<f32> = (0..nvh).map(|_| rng.f()).collect();
        let dt_bias: Vec<f32> = (0..nvh).map(|_| rng.f()).collect();
        let ssm_a: Vec<f32> = (0..nvh).map(|_| -rng.f().abs()).collect();
        let mut st_cpu = vec![0f32; nvh * hd * hd];
        let mut o_cpu = vec![0f32; nvh * hd];
        let b_qkv = ctx.buffer_from_slice(&qkv).unwrap();
        let b_al = ctx.buffer_from_slice(&alpha).unwrap();
        let b_bt = ctx.buffer_from_slice(&beta).unwrap();
        let b_dt = ctx.buffer_from_slice(&dt_bias).unwrap();
        let b_sa = ctx.buffer_from_slice(&ssm_a).unwrap();
        let b_st = ctx.alloc_buffer(st_cpu.len() * 4).unwrap();
        let b_o = ctx.alloc_buffer(o_cpu.len() * 4).unwrap();
        b_st.zero().unwrap();
        for _ in 0..3 {
            ocak_cpu::ops::deltanet_step(&qkv, &alpha, &beta, &dt_bias, &ssm_a, &mut st_cpu, &mut o_cpu, nkh, nvh, hd, 1e-6);
            hops.deltanet_step(&stream, &b_qkv, &b_al, &b_bt, &b_dt, &b_sa, &b_st, &b_o, nkh as u32, nvh as u32, hd as u32, 1e-6, 1).unwrap();
        }
        stream.synchronize().unwrap();
        let mut o_gpu = vec![0f32; nvh * hd];
        b_o.download(bytemuck::cast_slice_mut(&mut o_gpu)).unwrap();
        for i in 0..o_cpu.len() {
            assert!((o_cpu[i] - o_gpu[i]).abs() < 1e-3, "deltanet[{i}]: cpu={} gpu={}", o_cpu[i], o_gpu[i]);
        }
    }

    #[test]
    fn hip_graph_capture_and_replay() {
        let Some(ctx) = try_ctx() else { return };
        let kernels = HipKernels::compile(&ctx).unwrap();
        let stream = ctx.create_stream().unwrap();
        let n = 512usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 2.0).collect();
        let w = vec![1.0f32; n];
        let bx = ctx.buffer_from_slice(&x).unwrap();
        let bw = ctx.buffer_from_slice(&w).unwrap();
        let by = ctx.alloc_buffer(n * 4).unwrap();

        let graph = stream
            .capture_graph(|s| kernels.rms_norm(s, &bx, &bw, &by, n as u32, 1, 1e-5))
            .unwrap();
        graph.launch(&stream).unwrap();
        stream.synchronize().unwrap();

        let mut y_gpu = vec![0f32; n];
        by.download(bytemuck::cast_slice_mut(&mut y_gpu)).unwrap();
        let mut y_cpu = vec![0f32; n];
        ocak_cpu::ops::rms_norm(&x, &w, &mut y_cpu, 1e-5);
        for i in 0..n {
            assert!((y_cpu[i] - y_gpu[i]).abs() < 1e-4);
        }
    }

    #[test]
    #[ignore = "benchmark"]
    fn hip_qmm_bench() {
        // Matches ocak-vk::tests::vk_qmm_tiled_bench (k=5120, m=17408, batch=256).
        let ctx = HipContext::new(0).expect("HIP init");
        eprintln!(
            "HIP device: {} ({}, {} CUs, wave={}, {:.1} GiB VRAM)",
            ctx.info.name,
            ctx.info.gcn_arch_name,
            ctx.info.cu_count,
            ctx.info.warp_size,
            ctx.info.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        let kernels = HipKernels::compile(&ctx).unwrap();
        let stream = ctx.create_stream().unwrap();
        let (k, m, batch) = (5120usize, 17408usize, 256usize);
        let x: Vec<f32> = (0..batch * k).map(|i| ((i % 31) as f32 - 15.0) * 0.01).collect();
        let bx = ctx.buffer_from_slice(&x).unwrap();
        let n_blocks = (batch * k / 32) as u32;
        let bxq = ctx.alloc_buffer(batch * k).unwrap();
        let bxsc = ctx.alloc_buffer(n_blocks as usize * 16).unwrap();
        let by = ctx.alloc_buffer(batch * m * 4).unwrap();
        let ev0 = ctx.create_event(true).unwrap();
        let ev1 = ctx.create_event(true).unwrap();

        for (label, ty, mode) in [
            ("Q8_0", GgmlType::Q8_0, 0u8),
            ("Q4K", GgmlType::Q4K, 1u8),
            ("Q4K_g256", GgmlType::Q4K, 2u8),
        ] {
            let (be, bb) = ty.block().unwrap();
            let rb = (k / be) * bb;
            let raw: Vec<u8> = (0..m * rb).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8 & 0x3F).collect();
            let bw = ctx.buffer_from_slice(&raw).unwrap();
            let iters = 20;
            let run_once = || {
                match mode {
                    0 => {
                        kernels.quant_q8(&stream, &bx, &bxq, &bxsc, n_blocks).unwrap();
                        kernels.qmm_q8_0(&stream, &bw, &bxq, &bxsc, &by, k as u32, m as u32, batch as u32).unwrap();
                    }
                    1 => {
                        kernels.quant_q8(&stream, &bx, &bxq, &bxsc, n_blocks).unwrap();
                        kernels.qmm_q4_k(&stream, &bw, &bxq, &bxsc, &by, k as u32, m as u32, batch as u32).unwrap();
                    }
                    _ => {
                        kernels.quant_q8_g256(&stream, &bx, &bxq, &bxsc, n_blocks).unwrap();
                        kernels.qmm_q4_k_g256(&stream, &bw, &bxq, &bxsc, &by, k as u32, m as u32, batch as u32).unwrap();
                    }
                }
            };
            for _ in 0..2 { run_once(); }
            stream.synchronize().unwrap();
            ev0.record(&stream).unwrap();
            for _ in 0..iters { run_once(); }
            ev1.record(&stream).unwrap();
            ev1.synchronize().unwrap();
            let dt = (ev1.elapsed_ms(&ev0).unwrap() as f64 / 1e3) / iters as f64;
            let tflops = 2.0 * (k * m * batch) as f64 / dt / 1e12;
            eprintln!("HIP {label}: {:.3} ms  {:.2} TFLOPS", dt * 1e3, tflops);
        }
    }

    #[test]
    #[ignore = "benchmark"]
    fn hip_small_batch_bench() {
        // Matches ocak-vk::tests::vk_small_batch_bench (k=5120, m=17408, rotating 6 weight copies).
        let ctx = HipContext::new(0).expect("HIP init");
        let kernels = HipKernels::compile(&ctx).unwrap();
        let stream = ctx.create_stream().unwrap();
        let (k, m) = (5120usize, 17408usize);
        let x: Vec<f32> = (0..8 * k).map(|i| ((i % 31) as f32 - 15.0) * 0.01).collect();
        let bx = ctx.buffer_from_slice(&x).unwrap();
        let bxq = ctx.alloc_buffer(8 * k).unwrap();
        let bxsc = ctx.alloc_buffer((8 * k / 32) * 16).unwrap();
        let by = ctx.alloc_buffer(8 * m * 4).unwrap();
        let ev0 = ctx.create_event(true).unwrap();
        let ev1 = ctx.create_event(true).unwrap();

        for (label, ty, is_q4k) in [("Q8_0", GgmlType::Q8_0, false), ("Q4K", GgmlType::Q4K, true)] {
            let (be, bb) = ty.block().unwrap();
            let rb = (k / be) * bb;
            let raw: Vec<u8> = (0..m * rb).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8 & 0x3F).collect();
            let bws: Vec<_> = (0..6).map(|_| ctx.buffer_from_slice(&raw).unwrap()).collect();

            for batch in [1u32, 2] {
                let nb = (batch as usize * k / 32) as u32;
                let run = |n: usize| {
                    for i in 0..n {
                        kernels.quant_q8(&stream, &bx, &bxq, &bxsc, nb).unwrap();
                        if is_q4k {
                            kernels.qmv_q4_k(&stream, &bws[i % 6], &bxq, &bxsc, &by, k as u32, m as u32, batch).unwrap();
                        } else {
                            kernels.qmv_q8_0(&stream, &bws[i % 6], &bxq, &bxsc, &by, k as u32, m as u32, batch).unwrap();
                        }
                    }
                };
                run(30);
                stream.synchronize().unwrap();
                let mut best_dt = f64::MAX;
                for _ in 0..5 {
                    ev0.record(&stream).unwrap();
                    run(60);
                    ev1.record(&stream).unwrap();
                    ev1.synchronize().unwrap();
                    let dt = (ev1.elapsed_ms(&ev0).unwrap() as f64 / 1e3) / 60.0;
                    best_dt = best_dt.min(dt);
                }
                eprintln!(
                    "HIP {label} batch={batch}: {:.1} us  ({:.0} GB/s weights)",
                    best_dt * 1e6,
                    raw.len() as f64 / best_dt / 1e9
                );
            }
        }
    }
}
