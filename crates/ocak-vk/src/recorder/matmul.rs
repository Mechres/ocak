//! Dense matvec / matmul dispatch (float, int8 and tiled kernels, split-K).
use crate::*;

impl<'a> Recorder<'a> {
    pub fn qmv(&mut self, ty: WType, w: &Buffer, w_off: u64, x: &Buffer, x_off: u64, y: &Buffer, y_off: u64, k: u32, m: u32) {
        if self.qmv_int8(ty, w, w_off, x, x_off, y, y_off, k, m, 1) {
            return;
        }
        // Same kernel as the 2..8-token path (qmm) so MTP verification reproduces single-token
        // decode bit for bit (speculative decoding stays lossless). Large matrices already take
        // the int8 path above; this only affects small / Q8_0 ones.
        if k.is_multiple_of(32) {
            let pc = [off_u32(w_off), off_u32(x_off), off_u32(y_off), k, m, ty.row_bytes(k), 1];
            let p = &self.ctx.pipelines.qmvm[qmmt_index(ty)];
            self.dispatch_raw(p, &[w.buf, x.buf, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), m.div_ceil(4), 1, 1);
            return;
        }
        self.qmv_classic(ty, w, w_off, x, x_off, y, y_off, k, m);
    }

    /// The hand-written per-format single-token matvec shaders (reference implementation).
    pub fn qmv_classic(&mut self, ty: WType, w: &Buffer, w_off: u64, x: &Buffer, x_off: u64, y: &Buffer, y_off: u64, k: u32, m: u32) {
        let pc = [off_u32(w_off), off_u32(x_off), off_u32(y_off), k, m, ty.row_bytes(k)];
        let gx = match ty {
            WType::IQ2XS => m.div_ceil(16),
            _ => m.div_ceil(4),
        };
        let (pipe, needs_iq) = match ty {
            WType::F32 => (&self.ctx.pipelines.qmv_f32, false),
            WType::F16 => (&self.ctx.pipelines.qmv_f16, false),
            WType::BF16 => (&self.ctx.pipelines.qmv_bf16, false),
            WType::Q8_0 => (&self.ctx.pipelines.qmv_q8_0, false),
            WType::Q4_0 => (&self.ctx.pipelines.qmv_q4_0, false),
            WType::Q4K => (&self.ctx.pipelines.qmv_q4_k, false),
            WType::Q5K => (&self.ctx.pipelines.qmv_q5_k, false),
            WType::Q6K => (&self.ctx.pipelines.qmv_q6_k, false),
            WType::Q2K => (&self.ctx.pipelines.qmv_q2_k, false),
            WType::Q3K => (&self.ctx.pipelines.qmv_q3_k, false),
            WType::IQ2XXS => (&self.ctx.pipelines.qmv_iq2_xxs, true),
            WType::IQ2XS => (&self.ctx.pipelines.qmv_iq2_xs, true),
            WType::IQ2S => (&self.ctx.pipelines.qmv_iq2_s, true),
            WType::IQ3XXS => (&self.ctx.pipelines.qmv_iq3_xxs, true),
            WType::IQ3S => (&self.ctx.pipelines.qmv_iq3_s, true),
            WType::IQ1S => (&self.ctx.pipelines.qmv_iq1_s, true),
            WType::IQ1M => (&self.ctx.pipelines.qmv_iq1_m, true),
            WType::IQ4NL => (&self.ctx.pipelines.qmv_iq4_nl, true),
            WType::IQ4XS => (&self.ctx.pipelines.qmv_iq4_xs, true),
            WType::NVFP4 => (&self.ctx.pipelines.qmv_nvfp4, false),
        };
        if needs_iq {
            self.dispatch_raw(pipe, &[w.buf, x.buf, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), gx, 1, 1);
        } else {
            self.dispatch_raw(pipe, &[w.buf, x.buf, y.buf], bytemuck::cast_slice(&pc), gx, 1, 1);
        }
    }

    /// Quantize `rows` activation rows of k floats (contiguous at x_off) to int8 blocks in the
    /// q8 scratch, or reuse the previous quantization of the same rows. Returns the scratch
    /// byte offset of the quantized words (scales follow at + rows * k), or None if unsupported.
    pub(crate) fn quantize_q8(&mut self, x: &Buffer, x_off: u64, k: u32, rows: u32) -> Option<u64> {
        self.quantize_q8_mode(x, x_off, k, rows, false)
    }

    /// `g256`: one scale per 256 values (quant_q8.comp -DGROUP256; k % 256 == 0). The cache
    /// key marks it with k's top bit so the two quantizations of one input never alias.
    pub(crate) fn quantize_q8_mode(&mut self, x: &Buffer, x_off: u64, k: u32, rows: u32, g256: bool) -> Option<u64> {
        let n = rows as u64 * k as u64;
        let bytes = (n + n / 2).next_multiple_of(256);
        if !self.mmq || !x_off.is_multiple_of(4) || !k.is_multiple_of(32) || bytes > Q8_SCRATCH_BYTES {
            return None;
        }
        let key = (x.buf, x_off, k | ((g256 as u32) << 31), rows);
        if let Some((b, o, kk, rr, q_off)) = self.q8_cache {
            if (b, o, kk, rr) == key {
                return Some(q_off);
            }
        }
        if self.q8_cursor + bytes > Q8_SCRATCH_BYTES {
            // Earlier matmuls may still read the region being reused.
            self.barrier();
            self.q8_cursor = 0;
        }
        let q_off = self.q8_cursor;
        self.q8_cursor += bytes;
        let n_blocks = off_u32(n / 32);
        let pc = [off_u32(x_off / 4), off_u32(q_off / 4), off_u32((q_off + n) / 16), n_blocks];
        let q8 = self.ctx.q8_scratch;
        let qp = if g256 { &self.ctx.pipelines.quant_q8_256 } else { &self.ctx.pipelines.quant_q8 };
        self.dispatch_raw(qp, &[x.buf, q8, q8], bytemuck::cast_slice(&pc), n_blocks.div_ceil(32), 1, 1);
        if self.defer_barriers {
            self.barrier();
        }
        Some(q_off)
    }

    /// Integer-dot matvec for 1..8 tokens (decode / MTP verify): activations quantized to int8
    /// blocks (shared with other matvecs on the same input), weights unpacked to int8.
    pub(crate) fn qmv_int8(&mut self, ty: WType, w: &Buffer, w_off: u64, x: &Buffer, x_off: u64, y: &Buffer, y_off: u64, k: u32, m: u32, batch: u32) -> bool {
        // Tiny matvecs (e.g. Xing's per-head absorbed projections, ~1500 per token) would pay an
        // activation-quantize dispatch and barrier each, serializing work that otherwise overlaps.
        // Q8_0 weights gain little from int8 activations (the float kernel is already cheap per
        // weight), so they skip the extra quantize dispatch.
        if !qmv_int8_enabled() || (k as u64) * (m as u64) < (1 << 20) || ty == WType::Q8_0 {
            return false;
        }
        self.qmv_int8_dispatch(ty, w, w_off, x, x_off, y, y_off, k, m, batch)
    }

    /// qmv_int8 without the size / format heuristics (also used directly by tests).
    pub(crate) fn qmv_int8_dispatch(&mut self, ty: WType, w: &Buffer, w_off: u64, x: &Buffer, x_off: u64, y: &Buffer, y_off: u64, k: u32, m: u32, batch: u32) -> bool {
        let Some(qi) = qmmq_index(ty) else { return false };
        if batch == 0 || batch > 8 || !w_off.is_multiple_of(4) {
            return false;
        }
        let Some(q_off) = self.quantize_q8(x, x_off, k, batch) else { return false };
        let n = batch as u64 * k as u64;
        let pc = [off_u32(w_off), off_u32(q_off / 8), off_u32((q_off + n) / 16), off_u32(y_off), k, m, ty.row_bytes(k), batch];
        let q8 = self.ctx.q8_scratch;
        // Rows per workgroup (qmv_q8.comp RPW): 16 for formats with a shared-memory lookup table.
        let rpw = if matches!(ty, WType::IQ1S | WType::IQ1M | WType::IQ2XS | WType::IQ2XXS | WType::IQ2S | WType::IQ3XXS | WType::IQ3S | WType::IQ4NL | WType::IQ4XS) { 16 } else { 4 };
        self.dispatch_raw(&self.ctx.pipelines.qmvq[qi], &[w.buf, q8, q8, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), m.div_ceil(rpw), 1, 1);
        self.q8_cache = Some((x.buf, x_off, k, batch, q_off));
        true
    }

    /// Integer-dot prefill matmul: quantizes X to int8 blocks (reusing the previous
    /// quantization when the same X feeds consecutive matmuls), then runs qmm_q8.comp.
    /// Returns false when the format / shape is not supported.
    pub(crate) fn qmm_int8(&mut self, ty: WType, w: &Buffer, w_off: u64, x: &Buffer, x_off: u64, y: &Buffer, y_off: u64, k: u32, m: u32, batch: u32) -> bool {
        let Some(qi) = qmmq_index(ty) else { return false };
        if !w_off.is_multiple_of(4) {
            return false;
        }
        let g256 = prefill_q256() && k.is_multiple_of(256);
        let Some(q_off) = self.quantize_q8_mode(x, x_off, k, batch, g256) else { return false };
        let pipe = if g256 { &self.ctx.pipelines.qmmq256[qi] } else { &self.ctx.pipelines.qmmq[qi] };
        let n = batch as u64 * k as u64;
        let q8 = self.ctx.q8_scratch;
        let (gx, gy) = (m.div_ceil(QMMT_BM), batch.div_ceil(QMMT_BN));
        let kblocks = k / 32;
        // Small m (e.g. 48-row gate projections, 1024-row KV projections) gives too few
        // workgroups: split the K blocks across workgroup z and sum the partials.
        let splits = if gx * gy < 80 { (160 / (gx * gy)).min(kblocks / 32).min(32) } else { 1 };
        let part = batch as u64 * m as u64 * 4;
        if splits >= 2 && part * splits as u64 <= ATTN_SCRATCH_BYTES {
            let bytes = part * splits as u64;
            if self.scratch_cursor + bytes > ATTN_SCRATCH_BYTES {
                self.barrier();
                self.scratch_cursor = 0;
            }
            let s_off = self.scratch_cursor;
            self.scratch_cursor += bytes;
            let kbchunk = kblocks.div_ceil(splits);
            let splits = kblocks.div_ceil(kbchunk);
            let scratch = self.ctx.attn_scratch;
            let pcs = [off_u32(w_off), off_u32(q_off / 16), off_u32((q_off + n) / 16), off_u32(s_off / 4), k, m, ty.row_bytes(k), batch, kbchunk, batch * m];
            self.dispatch_raw(pipe, &[w.buf, q8, q8, scratch, self.ctx.iq_buf], bytemuck::cast_slice(&pcs), gx, gy, splits);
            self.dep_barrier();
            let nn = batch * m;
            let pcr = [off_u32(s_off / 4), off_u32(y_off), nn, splits];
            self.dispatch_raw(&self.ctx.pipelines.sum_splits, &[scratch, y.buf], bytemuck::cast_slice(&pcr), nn.div_ceil(256), 1, 1);
        } else {
            let pc = [off_u32(w_off), off_u32(q_off / 16), off_u32((q_off + n) / 16), off_u32(y_off), k, m, ty.row_bytes(k), batch, kblocks, 0];
            self.dispatch_raw(pipe, &[w.buf, q8, q8, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), gx, gy, 1);
        }
        self.q8_cache = Some((x.buf, x_off, k | ((g256 as u32) << 31), batch, q_off));
        true
    }

    pub fn qmm(&mut self, ty: WType, w: &Buffer, w_off: u64, x: &Buffer, x_off: u64, y: &Buffer, y_off: u64, k: u32, m: u32, batch: u32) {
        if batch <= 1 {
            self.qmv(ty, w, w_off, x, x_off, y, y_off, k, m);
            return;
        }
        let pc = [off_u32(w_off), off_u32(x_off), off_u32(y_off), k, m, ty.row_bytes(k), batch];
        let gx = m.div_ceil(4);
        if batch <= 8 && self.qmv_int8(ty, w, w_off, x, x_off, y, y_off, k, m, batch) {
            return;
        }
        // 9..31 tokens (short prompts, below the tiled-matmul threshold): int8 matvec in groups of
        // 8 tokens (one weight read per group) instead of the old 8-token float kernels.
        if batch > 8 && batch < qmmt_min_batch() && x_off.is_multiple_of(4) && self.qmv_int8(ty, w, w_off, x, x_off, y, y_off, k, m, 8) {
            let mut t = 8u32;
            while t < batch {
                let nb = (batch - t).min(8);
                let ok = self.qmv_int8(ty, w, w_off, x, x_off + (t * k) as u64, y, y_off + (t * m) as u64, k, m, nb);
                debug_assert!(ok, "int8 matvec accepted the first group but not a later one");
                t += nb;
            }
            return;
        }
        if batch <= 8 && k.is_multiple_of(32) {
            // Small batch: one weight read for all tokens.
            let p = &self.ctx.pipelines.qmvm[qmmt_index(ty)];
            self.dispatch_raw(p, &[w.buf, x.buf, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), gx, 1, 1);
            return;
        }
        if batch == 2 {
            // Only reached when k is not a multiple of 32 (no block-quantized format).
            self.qmv(ty, w, w_off, x, x_off, y, y_off, k, m);
            self.qmv(ty, w, w_off, x, x_off + k as u64, y, y_off + m as u64, k, m);
            return;
        }
        if batch >= qmmt_min_batch() && k.is_multiple_of(32) {
            if self.qmm_int8(ty, w, w_off, x, x_off, y, y_off, k, m, batch) {
                return;
            }
            let p = &self.ctx.pipelines.qmmt[qmmt_index(ty)];
            let (gx, gy) = (m.div_ceil(QMMT_BM), batch.div_ceil(QMMT_BN));
            // Small m with large k (e.g. 24-row hyper-connection / router weights) gives only a
            // few workgroups: split K across workgroup z and sum the partials.
            let splits = if gx * gy < 40 { (160 / (gx * gy)).min(k / 512).min(32) } else { 1 };
            let part = batch as u64 * m as u64 * 4;
            if splits >= 2 && self.scratch_cursor.is_multiple_of(4) && part * splits as u64 <= ATTN_SCRATCH_BYTES {
                let bytes = part * splits as u64;
                if self.scratch_cursor + bytes > ATTN_SCRATCH_BYTES {
                    self.barrier();
                    self.scratch_cursor = 0;
                }
                let s_off = self.scratch_cursor;
                self.scratch_cursor += bytes;
                let kchunk = k.div_ceil(splits).next_multiple_of(32);
                let splits = k.div_ceil(kchunk);
                let scratch = self.ctx.attn_scratch;
                let pcs = [off_u32(w_off), off_u32(x_off), off_u32(s_off / 4), k, m, ty.row_bytes(k), batch, kchunk, batch * m];
                self.dispatch_raw(p, &[w.buf, x.buf, scratch, self.ctx.iq_buf], bytemuck::cast_slice(&pcs), gx, gy, splits);
                self.dep_barrier();
                let n = batch * m;
                let pcr = [off_u32(s_off / 4), off_u32(y_off), n, splits];
                self.dispatch_raw(&self.ctx.pipelines.sum_splits, &[scratch, y.buf], bytemuck::cast_slice(&pcr), n.div_ceil(256), 1, 1);
                return;
            }
            let pc = [off_u32(w_off), off_u32(x_off), off_u32(y_off), k, m, ty.row_bytes(k), batch, k, 0];
            self.dispatch_raw(p, &[w.buf, x.buf, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), gx, gy, 1);
            return;
        }
        let gy = batch.div_ceil(8);
        let gx8 = m.div_ceil(8);
        match ty {
            WType::Q4K => self.dispatch_raw(&self.ctx.pipelines.qmm_q4_k, &[w.buf, x.buf, y.buf], bytemuck::cast_slice(&pc), gx8, gy, 1),
            WType::Q6K => self.dispatch_raw(&self.ctx.pipelines.qmm_q6_k, &[w.buf, x.buf, y.buf], bytemuck::cast_slice(&pc), gx8, gy, 1),
            WType::Q5K => self.dispatch_raw(&self.ctx.pipelines.qmm_q5_k, &[w.buf, x.buf, y.buf], bytemuck::cast_slice(&pc), gx8, gy, 1),
            WType::Q4_0 => self.dispatch_raw(&self.ctx.pipelines.qmm_q4_0, &[w.buf, x.buf, y.buf], bytemuck::cast_slice(&pc), gx, gy, 1),
            WType::Q8_0 => self.dispatch_raw(&self.ctx.pipelines.qmm_q8_0, &[w.buf, x.buf, y.buf], bytemuck::cast_slice(&pc), gx8, gy, 1),
            WType::F16 => self.dispatch_raw(&self.ctx.pipelines.qmm_f16, &[w.buf, x.buf, y.buf], bytemuck::cast_slice(&pc), gx, gy, 1),
            WType::F32 => self.dispatch_raw(&self.ctx.pipelines.qmm_f32, &[w.buf, x.buf, y.buf], bytemuck::cast_slice(&pc), gx, gy, 1),
            WType::IQ2S => self.dispatch_raw(&self.ctx.pipelines.qmm_iq2_s, &[w.buf, x.buf, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), gx8, gy, 1),
            WType::IQ2XS => self.dispatch_raw(&self.ctx.pipelines.qmm_iq2_xs, &[w.buf, x.buf, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), gx8, gy, 1),
            WType::IQ3XXS => self.dispatch_raw(&self.ctx.pipelines.qmm_iq3_xxs, &[w.buf, x.buf, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), gx8, gy, 1),
            WType::IQ3S => self.dispatch_raw(&self.ctx.pipelines.qmm_iq3_s, &[w.buf, x.buf, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), gx8, gy, 1),
            _ => {
                for b in 0..batch {
                    self.qmv(ty, w, w_off, x, x_off + (b * k) as u64, y, y_off + (b * m) as u64, k, m);
                }
            }
        }
    }
}
