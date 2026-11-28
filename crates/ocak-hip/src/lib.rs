//! Dynamic `dlopen` AMD HIP runtime (`libamdhip64.so`) + `libhiprtc.so` JIT compiler.
//! Builds cleanly with zero ROCm SDK installed; lights up automatically when ROCm is present.
pub mod ffi;
pub mod kernels;
pub mod rtc;
pub mod runtime;

pub use ffi::{
    DeviceInfo, HipError, HipStatus, MemcpyKind, HIP_MEMCPY_D2D, HIP_MEMCPY_D2H, HIP_MEMCPY_H2D,
    HIP_MEMCPY_H2H,
};
pub use kernels::HipKernels;
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
            GgmlType::Q8_0 => {
                for blk in v.chunks_exact_mut(bb) {
                    blk[0..2].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
                }
            }
            GgmlType::Q4K => {
                for blk in v.chunks_exact_mut(bb) {
                    blk[0..2].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
                    blk[2..4].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
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

        for (ty, is_q4k) in [(GgmlType::Q8_0, false), (GgmlType::Q4K, true)] {
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

            kernels.quant_q8(&stream, &bx, &bxq, &bxsc, n_blocks).unwrap();

            // 1) Decode qmv for batch = 3
            let b_dec = 3u32;
            if is_q4k {
                kernels.qmv_q4_k(&stream, &bw, &bxq, &bxsc, &by, k as u32, m as u32, b_dec).unwrap();
            } else {
                kernels.qmv_q8_0(&stream, &bw, &bxq, &bxsc, &by, k as u32, m as u32, b_dec).unwrap();
            }
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
            if is_q4k {
                kernels.qmm_q4_k(&stream, &bw, &bxq, &bxsc, &by, k as u32, m as u32, batch as u32).unwrap();
            } else {
                kernels.qmm_q8_0(&stream, &bw, &bxq, &bxsc, &by, k as u32, m as u32, batch as u32).unwrap();
            }
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
            // qmv and qmm use the exact same int8 activations -> should agree tightly on batch 0..3
            for i in 0..y_dec.len() {
                let err = (y_dec[i] - y_mm[i]).abs();
                assert!(err < 1e-4 * scale, "qmv vs qmm {ty:?}[{i}]: {} vs {}", y_dec[i], y_mm[i]);
            }
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

        for (label, ty, is_q4k) in [("Q8_0", GgmlType::Q8_0, false), ("Q4K", GgmlType::Q4K, true)] {
            let (be, bb) = ty.block().unwrap();
            let rb = (k / be) * bb;
            let raw: Vec<u8> = (0..m * rb).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8 & 0x3F).collect();
            let bw = ctx.buffer_from_slice(&raw).unwrap();
            let iters = 20;
            let run_once = || {
                kernels.quant_q8(&stream, &bx, &bxq, &bxsc, n_blocks).unwrap();
                if is_q4k {
                    kernels.qmm_q4_k(&stream, &bw, &bxq, &bxsc, &by, k as u32, m as u32, batch as u32).unwrap();
                } else {
                    kernels.qmm_q8_0(&stream, &bw, &bxq, &bxsc, &by, k as u32, m as u32, batch as u32).unwrap();
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
