//! Element-wise ops, norms, RoPE, convolutions, recurrent (DeltaNet) steps, copies and argmax.
use ash::vk;
use crate::*;

impl<'a> Recorder<'a> {
    pub fn rms_norm(&mut self, x: &Buffer, x_off: u64, w: &Buffer, w_off: u64, out: &Buffer, out_off: u64, n: u32, eps: f32) {
        self.rms_norm_batch(x, x_off, w, w_off, out, out_off, n, eps, 1);
    }
    /// RMS-normalize `batch` consecutive rows of `n` floats in one dispatch.
    pub fn rms_norm_batch(&mut self, x: &Buffer, x_off: u64, w: &Buffer, w_off: u64, out: &Buffer, out_off: u64, n: u32, eps: f32, batch: u32) {
        self.rms_norm_batch_strided(x, x_off, w, w_off, out, out_off, n, n, eps, batch);
    }
    /// Like `rms_norm_batch`, but rows are `stride` floats apart (normalizes the first n of each).
    pub fn rms_norm_batch_strided(&mut self, x: &Buffer, x_off: u64, w: &Buffer, w_off: u64, out: &Buffer, out_off: u64, n: u32, stride: u32, eps: f32, batch: u32) {
        self.rms_norm_batch_io(x, x_off, stride, w, w_off, out, out_off, stride, n, eps, batch);
    }
    /// Batched RMS norm with independent input / output row strides (in floats).
    pub fn rms_norm_batch_io(&mut self, x: &Buffer, x_off: u64, x_stride: u32, w: &Buffer, w_off: u64, out: &Buffer, out_off: u64, out_stride: u32, n: u32, eps: f32, batch: u32) {
        let pc = [off_u32(x_off), off_u32(w_off), off_u32(out_off), n, eps.to_bits(), x_stride, out_stride];
        self.dispatch_raw(&self.ctx.pipelines.rms_norm, &[x.buf, w.buf, out.buf], bytemuck::cast_slice(&pc), 1, batch, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_rms_norm(&mut self, x: &Buffer, x_off: u64, r: &Buffer, r_off: u64, w: &Buffer, w_off: u64, out: &Buffer, out_off: u64, n: u32, eps: f32) {
        self.add_rms_norm_batch(x, x_off, r, r_off, w, w_off, out, out_off, n, eps, 1);
    }

    /// `batch` rows of `n` floats (back to back in x, r and out): x += r, out = rms_norm(x) * w,
    /// bit-identical per row to `add_rms_norm` (one workgroup per row, same reduction).
    #[allow(clippy::too_many_arguments)]
    pub fn add_rms_norm_batch(&mut self, x: &Buffer, x_off: u64, r: &Buffer, r_off: u64, w: &Buffer, w_off: u64, out: &Buffer, out_off: u64, n: u32, eps: f32, batch: u32) {
        let pc = [off_u32(x_off), off_u32(r_off), off_u32(w_off), off_u32(out_off), n, eps.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.add_rms_norm, &[x.buf, r.buf, w.buf, out.buf], bytemuck::cast_slice(&pc), 1, batch, 1);
    }

    pub fn argmax(&mut self, x: &Buffer, x_off: u64, out: &Buffer, n: u32) {
        self.argmax_at(x, x_off, out, 0, n);
    }

    /// Argmax of x[x_off .. x_off + n] written to out word `out_idx`.
    pub fn argmax_at(&mut self, x: &Buffer, x_off: u64, out: &Buffer, out_idx: u32, n: u32) {
        // Two passes: partial maxima over up to 128 slices (in the attention scratch), then one
        // workgroup reduces them (a single workgroup scanning a 200k+ vocabulary took ~250 us).
        let parts = n.div_ceil(2048).clamp(1, 128);
        let bytes = parts as u64 * 8;
        if self.scratch_cursor + bytes > ATTN_SCRATCH_BYTES {
            self.barrier();
            self.scratch_cursor = 0;
        }
        let p_off = off_u32(self.scratch_cursor / 4);
        self.scratch_cursor += bytes.next_multiple_of(256);
        let scratch = self.ctx.attn_scratch;
        let pc0 = [off_u32(x_off), n, 0, p_off, parts, out_idx];
        self.dispatch_raw(&self.ctx.pipelines.argmax, &[x.buf, out.buf, scratch], bytemuck::cast_slice(&pc0), parts, 1, 1);
        self.dep_barrier();
        let pc1 = [off_u32(x_off), n, 1, p_off, parts, out_idx];
        self.dispatch_raw(&self.ctx.pipelines.argmax, &[x.buf, out.buf, scratch], bytemuck::cast_slice(&pc1), 1, 1, 1);
    }

    pub fn add(&mut self, a: &Buffer, a_off: u64, b: &Buffer, b_off: u64, n: u32) {
        let pc = [off_u32(a_off), off_u32(b_off), n];
        self.dispatch_raw(&self.ctx.pipelines.add, &[a.buf, b.buf], bytemuck::cast_slice(&pc), n.div_ceil(256), 1, 1);
    }

    pub fn add_scaled(&mut self, a: &Buffer, a_off: u64, b: &Buffer, b_off: u64, n: u32, scale: f32) {
        let pc = [off_u32(a_off), off_u32(b_off), n, scale.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.add_scaled, &[a.buf, b.buf], bytemuck::cast_slice(&pc), n.div_ceil(256), 1, 1);
    }

    pub fn copy_buf_bytes(&mut self, src: &Buffer, dst: &Buffer, bytes: u64) {
        self.copy_buf_region(src, 0, dst, 0, bytes);
    }

    /// Transfer-queue copy of `bytes` from src[src_off..] to dst[dst_off..] (byte offsets), with
    /// compute <-> transfer barriers around it.
    pub fn copy_buf_region(&mut self, src: &Buffer, src_off: u64, dst: &Buffer, dst_off: u64, bytes: u64) {
        self.q8_cache = None;
        unsafe {
            let pre = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)];
            self.ctx.device.cmd_pipeline_barrier(
                self.exec.cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &pre,
                &[],
                &[],
            );
            let reg = [vk::BufferCopy::default().src_offset(src_off).dst_offset(dst_off).size(bytes)];
            self.ctx.device.cmd_copy_buffer(self.exec.cmd_buf, src.buf, dst.buf, &reg);
            let post = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            self.ctx.device.cmd_pipeline_barrier(
                self.exec.cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &post,
                &[],
                &[],
            );
        }
    }

    pub fn zero_buf(&mut self, dst: &Buffer) {
        self.q8_cache = None;
        unsafe {
            let pre = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
            self.ctx.device.cmd_pipeline_barrier(
                self.exec.cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &pre,
                &[],
                &[],
            );
            self.ctx.device.cmd_fill_buffer(self.exec.cmd_buf, dst.buf, 0, vk::WHOLE_SIZE, 0);
            let bar = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            self.ctx.device.cmd_pipeline_barrier(
                self.exec.cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &bar,
                &[],
                &[],
            );
        }
    }

    pub fn copy(&mut self, src: &Buffer, s_off: u64, dst: &Buffer, d_off: u64, n: u32) {
        let pc = [off_u32(s_off), off_u32(d_off), n];
        self.dispatch_raw(&self.ctx.pipelines.copy, &[src.buf, dst.buf], bytemuck::cast_slice(&pc), n.div_ceil(256), 1, 1);
    }

    pub fn swiglu(&mut self, gate: &Buffer, g_off: u64, up: &Buffer, u_off: u64, out: &Buffer, o_off: u64, n: u32) {
        let pc = [off_u32(g_off), off_u32(u_off), off_u32(o_off), n];
        self.dispatch_raw(&self.ctx.pipelines.swiglu, &[gate.buf, up.buf, out.buf], bytemuck::cast_slice(&pc), n.div_ceil(256), 1, 1);
    }

    pub fn situ(&mut self, gu: &Buffer, gu_off: u64, out: &Buffer, o_off: u64, n_ff: u32, scale: f32, beta: f32, lin_beta: f32) {
        let pc = [off_u32(gu_off), off_u32(o_off), n_ff, scale.to_bits(), beta.to_bits(), lin_beta.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.situ, &[gu.buf, out.buf], bytemuck::cast_slice(&pc), n_ff.div_ceil(256), 1, 1);
    }

    pub fn hc_sinkhorn(&mut self, mixes: &Buffer, params: &Buffer, out_hc: &Buffer, iters: u32, hc_eps: f32) {
        self.hc_sinkhorn_batch(mixes, params, out_hc, iters, hc_eps, 1);
    }

    pub fn hc_sinkhorn_batch(&mut self, mixes: &Buffer, params: &Buffer, out_hc: &Buffer, iters: u32, hc_eps: f32, batch: u32) {
        let pc = [iters, hc_eps.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.hc_sinkhorn, &[mixes.buf, params.buf, out_hc.buf], bytemuck::cast_slice(&pc), 1, batch, 1);
    }

    pub fn hc_mix(&mut self, streams: &Buffer, blk: &Buffer, hc: &Buffer, out: &Buffer, ne: u32, mode: u32) {
        self.hc_mix_batch(streams, blk, hc, out, ne, mode, 1);
    }

    pub fn hc_mix_batch(&mut self, streams: &Buffer, blk: &Buffer, hc: &Buffer, out: &Buffer, ne: u32, mode: u32, batch: u32) {
        let pc = [ne, mode];
        self.dispatch_raw(&self.ctx.pipelines.hc_mix, &[streams.buf, blk.buf, hc.buf, out.buf], bytemuck::cast_slice(&pc), ne.div_ceil(256), batch, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_yarn_mla(
        &mut self,
        q_full: &Buffer,
        kv_raw: &Buffer,
        kv_cache: &Buffer,
        nh: u32,
        key_dim: u32,
        nope: u32,
        n_rot: u32,
        latent: u32,
        pos: u32,
        kv_layer_off: u32,
        rope_base: f32,
        factor: f32,
        low: f32,
        high: f32,
        mscale: f32,
    ) {
        self.rope_yarn_mla_batch(q_full, kv_raw, kv_cache, nh, key_dim, nope, n_rot, latent, pos, kv_layer_off, rope_base, factor, low, high, mscale, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_yarn_mla_batch(
        &mut self,
        q_full: &Buffer,
        kv_raw: &Buffer,
        kv_cache: &Buffer,
        nh: u32,
        key_dim: u32,
        nope: u32,
        n_rot: u32,
        latent: u32,
        pos: u32,
        kv_layer_off: u32,
        rope_base: f32,
        factor: f32,
        low: f32,
        high: f32,
        mscale: f32,
        batch: u32,
    ) {
        let pc = [
            nh, key_dim, nope, n_rot, latent, pos, kv_layer_off,
            rope_base.to_bits(), factor.to_bits(), low.to_bits(), high.to_bits(), mscale.to_bits(),
        ];
        self.dispatch_raw(&self.ctx.pipelines.rope_yarn_mla, &[q_full.buf, kv_raw.buf, kv_cache.buf], bytemuck::cast_slice(&pc), nh + 1, batch, 1);
    }

    pub fn rope(&mut self, q: &Buffer, q_off: u64, k: &Buffer, k_off: u64, n_heads: u32, n_kv_heads: u32, head_dim: u32, n_rot: u32, pos: u32, freq_base: f32, mode: u32) {
        let pc = [off_u32(q_off), off_u32(k_off), n_heads, n_kv_heads, head_dim, n_rot, pos, mode, freq_base.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.rope, &[q.buf, k.buf], bytemuck::cast_slice(&pc), n_heads + n_kv_heads, 1, 1);
    }

    pub fn embed(&mut self, ty: WType, table: &Buffer, token: u32, out: &Buffer, out_off: u64, k: u32) {
        let pc = [ty.ggml().id(), token, off_u32(out_off), k, ty.row_bytes(k)];
        self.dispatch_raw(&self.ctx.pipelines.embed, &[table.buf, out.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), k.div_ceil(256), 1, 1);
    }

    pub fn rms_norm_heads(&mut self, x: &Buffer, x_off: u64, w: &Buffer, n_heads: u32, head_dim: u32, eps: f32) {
        self.rms_norm_heads_batch(x, x_off, w, n_heads, head_dim, eps, 1);
    }
    pub fn rms_norm_heads_batch(&mut self, x: &Buffer, x_off: u64, w: &Buffer, n_heads: u32, head_dim: u32, eps: f32, batch: u32) {
        let pc = [off_u32(x_off), n_heads, head_dim, eps.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.rms_norm_heads, &[x.buf, w.buf], bytemuck::cast_slice(&pc), n_heads, batch, 1);
    }

    pub fn shortconv_step(&mut self, bcx: &Buffer, cw: &Buffer, state: &Buffer, out: &Buffer, n_embd: u32, l_cache: u32) {
        self.shortconv_step_batch(bcx, cw, state, out, n_embd, l_cache, 1);
    }

    pub fn shortconv_step_batch(&mut self, bcx: &Buffer, cw: &Buffer, state: &Buffer, out: &Buffer, n_embd: u32, l_cache: u32, batch: u32) {
        assert!((2..=9).contains(&l_cache), "conv kernels keep at most 8 history taps (l_cache = {l_cache})");
        let pc = [n_embd, l_cache, batch];
        self.dispatch_raw(&self.ctx.pipelines.shortconv_step, &[bcx.buf, cw.buf, state.buf, out.buf], bytemuck::cast_slice(&pc), n_embd.div_ceil(256), 1, 1);
    }

    pub fn ssm_conv_silu(&mut self, x: &Buffer, cw: &Buffer, state: &Buffer, out: &Buffer, channels: u32, k_conv: u32) {
        self.ssm_conv_silu_batch(x, cw, state, out, channels, k_conv, 1);
    }

    pub fn ssm_conv_silu_batch(&mut self, x: &Buffer, cw: &Buffer, state: &Buffer, out: &Buffer, channels: u32, k_conv: u32, batch: u32) {
        self.ssm_conv_silu_batch_snap(x, cw, state, out, channels, k_conv, batch, None);
    }

    /// `snap = Some((buf, n))`: also write the conv history after each token t < n to slot t of
    /// `buf` (MTP rollback points, slot = channels * (k_conv - 1) floats), without copies.
    #[allow(clippy::too_many_arguments)]
    pub fn ssm_conv_silu_batch_snap(&mut self, x: &Buffer, cw: &Buffer, state: &Buffer, out: &Buffer, channels: u32, k_conv: u32, batch: u32, snap: Option<(&Buffer, u32)>) {
        assert!((2..=9).contains(&k_conv), "conv kernels keep at most 8 history taps (k_conv = {k_conv})");
        let (sb, sn) = snap.map_or((out.buf, 0), |(b, n)| (b.buf, n));
        let pc = [channels, k_conv, batch, sn];
        self.dispatch_raw(&self.ctx.pipelines.ssm_conv_silu, &[x.buf, cw.buf, state.buf, out.buf, sb], bytemuck::cast_slice(&pc), channels.div_ceil(256), 1, 1);
    }

    pub fn deltanet_step(&mut self, qkv: &Buffer, alpha: &Buffer, beta: &Buffer, dt_bias: &Buffer, ssm_a: &Buffer, state: &Buffer, out: &Buffer, n_k_heads: u32, n_v_heads: u32, head_dim: u32, eps: f32) {
        self.deltanet_step_batch(qkv, alpha, beta, dt_bias, ssm_a, state, out, n_k_heads, n_v_heads, head_dim, eps, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_step_batch(&mut self, qkv: &Buffer, alpha: &Buffer, beta: &Buffer, dt_bias: &Buffer, ssm_a: &Buffer, state: &Buffer, out: &Buffer, n_k_heads: u32, n_v_heads: u32, head_dim: u32, eps: f32, batch: u32) {
        self.deltanet_step_batch_snap(qkv, alpha, beta, dt_bias, ssm_a, state, out, n_k_heads, n_v_heads, head_dim, eps, batch, None);
    }

    /// `snap = Some((buf, n))` (batches < 16 only): also write the recurrent state after each token
    /// t < n to slot t of `buf` (MTP rollback points, slot = n_v_heads * head_dim^2 floats).
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_step_batch_snap(&mut self, qkv: &Buffer, alpha: &Buffer, beta: &Buffer, dt_bias: &Buffer, ssm_a: &Buffer, state: &Buffer, out: &Buffer, n_k_heads: u32, n_v_heads: u32, head_dim: u32, eps: f32, batch: u32, snap: Option<(&Buffer, u32)>) {
        assert!(head_dim <= 128, "deltanet_step supports head_dim <= 128 (got {head_dim})");
        if batch < 16 || !head_dim.is_multiple_of(32) {
            // Decode / MTP verify: 8 rows (32-lane clusters) per workgroup, no token staging.
            let (sb, sn) = snap.map_or((out.buf, 0), |(b, n)| (b.buf, n));
            let pc = [n_k_heads, n_v_heads, head_dim, eps.to_bits(), batch, sn];
            let bufs = [qkv.buf, alpha.buf, beta.buf, dt_bias.buf, ssm_a.buf, state.buf, out.buf, sb];
            self.dispatch_raw(&self.ctx.pipelines.deltanet_step_tok, &bufs, bytemuck::cast_slice(&pc), (n_v_heads * head_dim).div_ceil(8), 1, 1);
            return;
        }
        assert!(snap.is_none(), "deltanet snapshots are only supported for batches < 16");
        let pc = [n_k_heads, n_v_heads, head_dim, eps.to_bits(), batch];
        let bufs = [qkv.buf, alpha.buf, beta.buf, dt_bias.buf, ssm_a.buf, state.buf, out.buf];
        // 32 state rows (8-lane clusters) per 256-thread workgroup.
        self.dispatch_raw(&self.ctx.pipelines.deltanet_step, &bufs, bytemuck::cast_slice(&pc), (n_v_heads * head_dim).div_ceil(32), 1, 1);
    }

    pub fn gated_rms_norm(&mut self, x: &Buffer, w: &Buffer, z: &Buffer, out: &Buffer, n_heads: u32, head_dim: u32, eps: f32) {
        self.gated_rms_norm_batch(x, w, z, out, n_heads, head_dim, eps, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gated_rms_norm_batch(&mut self, x: &Buffer, w: &Buffer, z: &Buffer, out: &Buffer, n_heads: u32, head_dim: u32, eps: f32, batch: u32) {
        let pc = [n_heads, head_dim, eps.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.gated_rms_norm, &[x.buf, w.buf, z.buf, out.buf], bytemuck::cast_slice(&pc), n_heads, batch, 1);
    }

    pub fn split_qg(&mut self, qg: &Buffer, q: &Buffer, g: &Buffer, n_heads: u32, head_dim: u32) {
        self.split_qg_batch(qg, q, g, n_heads, head_dim, 1);
    }

    pub fn split_qg_batch(&mut self, qg: &Buffer, q: &Buffer, g: &Buffer, n_heads: u32, head_dim: u32, batch: u32) {
        let pc = [n_heads, head_dim];
        let total = n_heads * head_dim;
        self.dispatch_raw(&self.ctx.pipelines.split_qg, &[qg.buf, q.buf, g.buf], bytemuck::cast_slice(&pc), total.div_ceil(256), batch, 1);
    }

    pub fn mul_sigmoid(&mut self, x: &Buffer, g: &Buffer, n: u32) {
        let pc = [n];
        self.dispatch_raw(&self.ctx.pipelines.mul_sigmoid, &[x.buf, g.buf], bytemuck::cast_slice(&pc), n.div_ceil(256), 1, 1);
    }

    pub fn mul_head_sigmoid(&mut self, x: &Buffer, g: &Buffer, n_heads: u32, head_dim: u32) {
        self.mul_head_sigmoid_batch(x, g, n_heads, head_dim, 1);
    }

    pub fn mul_head_sigmoid_batch(&mut self, x: &Buffer, g: &Buffer, n_heads: u32, head_dim: u32, batch: u32) {
        let pc = [n_heads, head_dim];
        self.dispatch_raw(&self.ctx.pipelines.mul_head_sigmoid, &[x.buf, g.buf], bytemuck::cast_slice(&pc), (n_heads * head_dim).div_ceil(256), batch, 1);
    }

    pub fn rope_ext(&mut self, q: &Buffer, q_off: u64, k: &Buffer, k_off: u64, factors: &Buffer, use_factors: bool, n_heads: u32, n_kv_heads: u32, head_dim: u32, n_rot: u32, pos: u32, freq_base: f32, mode: u32) {
        self.rope_ext_batch(q, q_off, k, k_off, factors, use_factors, n_heads, n_kv_heads, head_dim, n_rot, pos, freq_base, mode, 1);
    }
    /// RoPE for tokens pos..pos+batch; q rows are n_heads*head_dim, k rows n_kv_heads*head_dim.
    pub fn rope_ext_batch(&mut self, q: &Buffer, q_off: u64, k: &Buffer, k_off: u64, factors: &Buffer, use_factors: bool, n_heads: u32, n_kv_heads: u32, head_dim: u32, n_rot: u32, pos: u32, freq_base: f32, mode: u32, batch: u32) {
        let pc = [off_u32(q_off), off_u32(k_off), n_heads, n_kv_heads, head_dim, n_rot, pos, mode, freq_base.to_bits(), use_factors as u32];
        self.dispatch_raw(&self.ctx.pipelines.rope_ext, &[q.buf, k.buf, factors.buf], bytemuck::cast_slice(&pc), n_heads + n_kv_heads, batch, 1);
    }
}
