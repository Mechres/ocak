//! Mixture-of-experts routing and expert matmuls.
use crate::*;

impl<'a> Recorder<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk(&mut self, logits: &Buffer, bias: &Buffer, ids: &Buffer, weights: &Buffer, n_exp: u32, n_used: u32, has_bias: bool, norm_w: bool, scale: f32) {
        self.moe_topk_batch(logits, bias, ids, weights, n_exp, n_used, has_bias, norm_w, scale, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk_batch(&mut self, logits: &Buffer, bias: &Buffer, ids: &Buffer, weights: &Buffer, n_exp: u32, n_used: u32, has_bias: bool, norm_w: bool, scale: f32, batch: u32) {
        assert!(n_exp <= 256 && n_used <= n_exp, "moe_topk supports up to 256 experts");
        let pc = [n_exp, n_used, has_bias as u32, norm_w as u32, scale.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.moe_topk, &[logits.buf, bias.buf, ids.buf, weights.buf], bytemuck::cast_slice(&pc), 1, batch, 1);
    }

    pub fn qmv_id(&mut self, ty: WType, w: &Buffer, ids: &Buffer, slot: u32, x: &Buffer, x_off: u64, y: &Buffer, y_off: u64, k: u32, m: u32, slice_bytes: u64) {
        self.qmv_id_batch(ty, w, ids, slot, 1, x, x_off, 0, y, y_off, k, m, slice_bytes);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qmv_id_batch(&mut self, ty: WType, w: &Buffer, ids: &Buffer, slot_base: u32, count: u32, x: &Buffer, x_off: u64, x_stride: u32, y: &Buffer, y_off: u64, k: u32, m: u32, slice_bytes: u64) {
        let pc = [slot_base, off_u32(x_off), off_u32(y_off), k, m, ty.row_bytes(k), off_u32(slice_bytes), x_stride];
        let gx = m.div_ceil(4);
        // Hand-written id shaders exist for Q4_K and NVFP4, where they beat the generic kernel.
        let classic = matches!(ty, WType::NVFP4 | WType::Q4K);
        if !classic || qmvm_beats_qmv(ty) {
            let p = &self.ctx.pipelines.qmvi[qmmt_index(ty)];
            self.dispatch_raw(p, &[w.buf, x.buf, y.buf, self.ctx.iq_buf, ids.buf], bytemuck::cast_slice(&pc), gx, count, 1);
            return;
        }
        match ty {
            WType::NVFP4 => self.dispatch_raw(&self.ctx.pipelines.qmv_nvfp4_id, &[w.buf, x.buf, y.buf, ids.buf], bytemuck::cast_slice(&pc), gx, count, 1),
            WType::Q4K => self.dispatch_raw(&self.ctx.pipelines.qmv_q4_k_id, &[w.buf, x.buf, y.buf, ids.buf], bytemuck::cast_slice(&pc), gx, count, 1),
            _ => panic!("qmv_id: unsupported type {ty:?}"),
        }
    }

    pub fn add_scaled_id(&mut self, a: &Buffer, a_off: u64, b: &Buffer, b_off: u64, weights: &Buffer, n: u32, slot: u32) {
        self.add_scaled_id_batch(a, a_off, b, b_off, weights, n, slot, 1, false);
    }

    #[allow(clippy::too_many_arguments)]
    /// Bucket `n_entries` MoE routing entries (ids[i] = expert of entry i) by expert into `list`
    /// (layout documented in moe_group.comp). `list` needs n_groups * (cap + 1) u32s.
    pub fn moe_group(&mut self, ids: &Buffer, list: &Buffer, n_entries: u32, n_groups: u32, cap: u32) {
        assert!(n_groups <= 256, "moe_group supports <= 256 experts");
        let pc = [n_entries, n_groups, cap];
        self.dispatch_raw(&self.ctx.pipelines.moe_group, &[ids.buf, list.buf], bytemuck::cast_slice(&pc), 1, 1, 1);
    }

    /// Grouped tiled matmul. For each group g (expert or head), multiplies weight slice
    /// `w_off + g * slice_bytes` ([m, k]) with the group's columns:
    /// - `list = Some(buf)`: columns are the entries moe_group wrote for g (at most `cols` = cap);
    /// - `list = None`: columns are entries `j * n_groups + g` for j < `cols`.
    ///
    /// Entry e reads X row `(e / x_div) * x_stride` (in floats from x_off) and writes Y row
    /// `e * m`. Returns false (nothing recorded) when the weight format has no tiled kernel.
    /// `x_rows`: number of X rows the entries address (max entry / x_div + 1), used to size the
    /// int8 quantization of X.
    pub fn qmm_grouped(&mut self, ty: WType, w: &Buffer, w_off: u64, slice_bytes: u64, x: &Buffer, x_off: u64, x_stride: u32, x_div: u32, x_rows: u32, y: &Buffer, y_off: u64, k: u32, m: u32, n_groups: u32, list: Option<&Buffer>, cols: u32) -> bool {
        let i = qmmt_index(ty);
        if !k.is_multiple_of(32) {
            return false;
        }
        let use_list = list.is_some() as u32;
        let lbuf = list.map(|b| b.buf).unwrap_or(self.ctx.iq_buf);
        if let Some(qi) = qmmq_index(ty) {
            if x_stride == k && w_off.is_multiple_of(4) && slice_bytes.is_multiple_of(4) {
                if let Some(q_off) = self.quantize_q8(x, x_off, k, x_rows) {
                    let n = x_rows as u64 * k as u64;
                    let pc = [
                        off_u32(w_off), off_u32(q_off / 16), off_u32((q_off + n) / 16), off_u32(y_off), k, m, ty.row_bytes(k),
                        x_div, off_u32(slice_bytes), n_groups, cols, use_list, cols,
                    ];
                    let q8 = self.ctx.q8_scratch;
                    self.dispatch_raw(&self.ctx.pipelines.qmmqg[qi], &[w.buf, q8, q8, y.buf, self.ctx.iq_buf, lbuf], bytemuck::cast_slice(&pc), m.div_ceil(QMMT_BM), n_groups, cols.div_ceil(QMMQG_BN));
                    self.q8_cache = Some((x.buf, x_off, k, x_rows, q_off));
                    return true;
                }
            }
        }
        let pc = [off_u32(w_off), off_u32(x_off), off_u32(y_off), k, m, ty.row_bytes(k), x_stride, x_div, off_u32(slice_bytes), n_groups, cols, use_list, cols];
        let p = &self.ctx.pipelines.qmmg[i];
        self.dispatch_raw(p, &[w.buf, x.buf, y.buf, self.ctx.iq_buf, lbuf], bytemuck::cast_slice(&pc), m.div_ceil(QMMT_BM), n_groups, cols.div_ceil(QMMG_BN));
        true
    }

    /// `add_scaled_id_batch` for `batch` consecutive tokens (token t: a row t, b rows
    /// t*count.., weights slot_base + t*count..).
    pub fn add_scaled_id_tokens(&mut self, a: &Buffer, a_off: u64, b: &Buffer, b_off: u64, weights: &Buffer, n: u32, slot_base: u32, count: u32, overwrite: bool, batch: u32) {
        let pc = [off_u32(a_off), off_u32(b_off), n, slot_base, count, overwrite as u32];
        self.dispatch_raw(&self.ctx.pipelines.add_scaled_id, &[a.buf, b.buf, weights.buf], bytemuck::cast_slice(&pc), n.div_ceil(256), batch, 1);
    }

    pub fn add_scaled_id_batch(&mut self, a: &Buffer, a_off: u64, b: &Buffer, b_off: u64, weights: &Buffer, n: u32, slot_base: u32, count: u32, overwrite: bool) {
        let pc = [off_u32(a_off), off_u32(b_off), n, slot_base, count, overwrite as u32];
        self.dispatch_raw(&self.ctx.pipelines.add_scaled_id, &[a.buf, b.buf, weights.buf], bytemuck::cast_slice(&pc), n.div_ceil(256), 1, 1);
    }

    pub fn qmv_nvfp4_id(&mut self, w: &Buffer, ids: &Buffer, slot: u32, x: &Buffer, x_off: u64, y: &Buffer, y_off: u64, k: u32, m: u32, slice_bytes: u64) {
        self.qmv_id_batch(WType::NVFP4, w, ids, slot, 1, x, x_off, 0, y, y_off, k, m, slice_bytes);
    }

    pub fn situ_id(&mut self, gu: &Buffer, out: &Buffer, ids: &Buffer, scales: &Buffer, n_ff: u32, slot: u32, beta: f32, lin_beta: f32) {
        self.situ_id_batch(gu, out, ids, scales, n_ff, slot, 1, beta, lin_beta);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn situ_id_batch(&mut self, gu: &Buffer, out: &Buffer, ids: &Buffer, scales: &Buffer, n_ff: u32, slot_base: u32, count: u32, beta: f32, lin_beta: f32) {
        let pc = [n_ff, slot_base, beta.to_bits(), lin_beta.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.situ_id, &[gu.buf, out.buf, ids.buf, scales.buf], bytemuck::cast_slice(&pc), n_ff.div_ceil(256), count, 1);
    }

    pub fn add_scaled_exp_id(&mut self, a: &Buffer, b: &Buffer, weights: &Buffer, ids: &Buffer, dn_scales: &Buffer, n: u32, slot: u32) {
        self.add_scaled_exp_id_batch(a, b, weights, ids, dn_scales, n, slot, 1, false);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_scaled_exp_id_batch(&mut self, a: &Buffer, b: &Buffer, weights: &Buffer, ids: &Buffer, dn_scales: &Buffer, n: u32, slot_base: u32, count: u32, overwrite: bool) {
        self.add_scaled_exp_id_tokens(a, b, weights, ids, dn_scales, n, slot_base, count, overwrite, 1);
    }
    /// Token-batched `add_scaled_exp_id_batch`: token t accumulates into a row t from b rows
    /// t*count.. with weights/ids at slot_base + t*count..
    pub fn add_scaled_exp_id_tokens(&mut self, a: &Buffer, b: &Buffer, weights: &Buffer, ids: &Buffer, dn_scales: &Buffer, n: u32, slot_base: u32, count: u32, overwrite: bool, batch: u32) {
        let pc = [n, slot_base, count, overwrite as u32];
        self.dispatch_raw(&self.ctx.pipelines.add_scaled_exp_id, &[a.buf, b.buf, weights.buf, ids.buf, dn_scales.buf], bytemuck::cast_slice(&pc), n.div_ceil(256), batch, 1);
    }
}
