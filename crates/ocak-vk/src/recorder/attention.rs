//! Attention kernels: paged / split-K / flash prefill, SWA, MLA and Inkling attention, KV writes.
use crate::*;

impl<'a> Recorder<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn mla_attn(
        &mut self,
        q_lat: &Buffer,
        q_full: &Buffer,
        kv_cache: &Buffer,
        out: &Buffer,
        nh: u32,
        key_dim: u32,
        nope: u32,
        n_rot: u32,
        latent: u32,
        pos: u32,
        kv_layer_off: u32,
        kq_scale: f32,
    ) {
        self.mla_attn_batch(q_lat, q_full, kv_cache, out, nh, key_dim, nope, n_rot, latent, pos, kv_layer_off, kq_scale, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_attn_batch(
        &mut self,
        q_lat: &Buffer,
        q_full: &Buffer,
        kv_cache: &Buffer,
        out: &Buffer,
        nh: u32,
        key_dim: u32,
        nope: u32,
        n_rot: u32,
        latent: u32,
        pos: u32,
        kv_layer_off: u32,
        kq_scale: f32,
        batch: u32,
    ) {
        assert!(latent <= 512 && n_rot <= 128, "mla_attn supports latent <= 512, n_rot <= 128");
        assert!(nh.is_multiple_of(8), "mla_attn processes heads in groups of 8 (n_heads = {nh})");
        if self.flash_attn && batch >= 16 && latent == 512 && n_rot == 64 && nh.is_multiple_of(32) {
            // Prefill: flash-style tiles (32 heads of one token share each staged key tile).
            let pc = [key_dim, nope, pos, kv_layer_off, kq_scale.to_bits(), batch];
            let p = self.ctx.pipelines.mla_attn_prefill_f2.as_ref().unwrap_or(&self.ctx.pipelines.mla_attn_prefill);
            self.dispatch_raw(p, &[q_lat.buf, q_full.buf, kv_cache.buf, out.buf], bytemuck::cast_slice(&pc), nh / 32, batch.div_ceil(MLA_PREFILL_TOKS), 1);
            return;
        }
        let max_nt = pos + batch; // longest context in the batch (token b attends pos + b + 1)
        // Smaller chunks for short contexts keep enough workgroups in flight (each covers 8 heads).
        let mla_chunk: u32 = if max_nt <= 1024 { 16 } else { 64 };
        let scratch = self.ctx.attn_scratch;
        let gx = nh / 8;
        if max_nt <= mla_chunk {
            let pc = [key_dim, nope, n_rot, latent, pos, kv_layer_off, kq_scale.to_bits(), max_nt, 0];
            self.dispatch_raw(&self.ctx.pipelines.mla_attn, &[q_lat.buf, q_full.buf, kv_cache.buf, out.buf, scratch], bytemuck::cast_slice(&pc), gx, batch, 1);
            return;
        }
        // Split-K over context chunks (each workgroup: 8 heads x one chunk), then merge per (token, head).
        let per_split = nh as u64 * batch as u64 * (latent as u64 + 2) * 4;
        let mut n_splits = max_nt.div_ceil(mla_chunk);
        while n_splits > 1 && per_split * n_splits as u64 > ATTN_SCRATCH_BYTES {
            n_splits = n_splits.div_ceil(2);
        }
        if n_splits <= 1 {
            // Large batch: (heads/8 x batch) workgroups already fill the GPU, and a single split
            // writes normalized output directly (the merge pass would read unwritten scratch).
            let pc = [key_dim, nope, n_rot, latent, pos, kv_layer_off, kq_scale.to_bits(), max_nt, 0];
            self.dispatch_raw(&self.ctx.pipelines.mla_attn, &[q_lat.buf, q_full.buf, kv_cache.buf, out.buf, scratch], bytemuck::cast_slice(&pc), gx, batch, 1);
            return;
        }
        let chunk = max_nt.div_ceil(n_splits);
        let bytes = per_split * n_splits as u64;
        if self.scratch_cursor + bytes > ATTN_SCRATCH_BYTES {
            self.barrier();
            self.scratch_cursor = 0;
        }
        let scratch_off = off_u32(self.scratch_cursor / 4);
        self.scratch_cursor += bytes;
        let pc = [key_dim, nope, n_rot, latent, pos, kv_layer_off, kq_scale.to_bits(), chunk, scratch_off];
        self.dispatch_raw(&self.ctx.pipelines.mla_attn, &[q_lat.buf, q_full.buf, kv_cache.buf, out.buf, scratch], bytemuck::cast_slice(&pc), gx, batch, n_splits);
        self.dep_barrier();
        assert!(n_splits <= 1024, "attention_merge supports <= 1024 splits");
        let pc2 = [scratch_off, 0, latent, n_splits];
        self.dispatch_raw(&self.ctx.pipelines.attention_merge, &[scratch, out.buf], bytemuck::cast_slice(&pc2), nh * batch, latent.div_ceil(128), 1);
    }

    pub fn inkling_mix(&mut self, blocks: &Buffer, cur: &Buffer, query: &Buffer, out: &Buffer, ne: u32, blk_start: u32, n_blk: u32, n_q: u32) {
        self.inkling_mix_batch(blocks, cur, query, out, ne, blk_start, n_blk, n_q, ne, ne, 1);
    }
    /// Token-batched depth mix: token t's rows sit at t*ne in `cur`, block b at b*blk_stride + t*ne,
    /// and query q's output at q*out_q_stride + t*ne.
    pub fn inkling_mix_batch(&mut self, blocks: &Buffer, cur: &Buffer, query: &Buffer, out: &Buffer, ne: u32, blk_start: u32, n_blk: u32, n_q: u32, blk_stride: u32, out_q_stride: u32, batch: u32) {
        let pc = [ne, blk_start, n_blk, n_q, blk_stride, ne, out_q_stride];
        self.dispatch_raw(&self.ctx.pipelines.inkling_mix, &[blocks.buf, cur.buf, query.buf, out.buf], bytemuck::cast_slice(&pc), n_q, batch, 1);
    }

    pub fn inkling_ctrl(&mut self, cur: &Buffer, cur_off: u64, msg: &Buffer, msg_off: u64, gf: &Buffer, gf_off: u64, ne: u32) {
        let pc = [off_u32(cur_off), off_u32(msg_off), off_u32(gf_off), ne];
        self.dispatch_raw(&self.ctx.pipelines.inkling_ctrl, &[cur.buf, msg.buf, gf.buf], bytemuck::cast_slice(&pc), ne.div_ceil(256), 1, 1);
    }

    pub fn inkling_conv(&mut self, inp: &Buffer, inp_off: u64, wt: &Buffer, dy: &Buffer, dy_off: u64, hist: &Buffer, hist_off: u64, out: &Buffer, out_off: u64, n_ch: u32, n_conv: u32, has_dyn: bool, head_ch: u32) {
        self.inkling_conv_batch(inp, inp_off, n_ch, wt, dy, dy_off, 0, hist, hist_off, out, out_off, n_ch, n_ch, n_conv, has_dyn, head_ch, 1);
    }
    /// Short convolution over `n_tok` consecutive tokens (rows inp_stride / dy_stride / out_stride
    /// apart), updating the history once at the end.
    pub fn inkling_conv_batch(&mut self, inp: &Buffer, inp_off: u64, inp_stride: u32, wt: &Buffer, dy: &Buffer, dy_off: u64, dy_stride: u32, hist: &Buffer, hist_off: u64, out: &Buffer, out_off: u64, out_stride: u32, n_ch: u32, n_conv: u32, has_dyn: bool, head_ch: u32, n_tok: u32) {
        assert!(n_conv <= 9, "inkling_conv supports n_conv <= 9");
        let pc = [off_u32(inp_off), off_u32(dy_off), off_u32(hist_off), off_u32(out_off), n_ch, n_conv, has_dyn as u32, head_ch, n_tok, inp_stride, dy_stride, out_stride];
        self.dispatch_raw(&self.ctx.pipelines.inkling_conv, &[inp.buf, wt.buf, dy.buf, hist.buf, out.buf], bytemuck::cast_slice(&pc), n_ch.div_ceil(32), 1, 1);
    }

    /// Batched Inkling prefill pass 1: clip/normalize and store K/V for tokens pos..pos+n_tok.
    pub fn inkling_kv_store(&mut self, k: &Buffer, k_off: u64, v: &Buffer, v_off: u64, k_cache: &Buffer, v_cache: &Buffer, kv_dim: u32, n_head_kv: u32, hd: u32, is_swa: bool, cap: u32, pos: u32, kv_slot_off: u32, qk_clip: f32, eps: f32, n_tok: u32) {
        let pc = [off_u32(k_off), off_u32(v_off), kv_dim, n_head_kv, hd, is_swa as u32, cap, pos, kv_slot_off, qk_clip.to_bits(), eps.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.inkling_kv_store, &[k.buf, v.buf, k_cache.buf, v_cache.buf], bytemuck::cast_slice(&pc), n_head_kv, n_tok, 1);
    }

    /// Batched Inkling prefill pass 2: attention for query tokens pos..pos+n_tok (K/V already
    /// stored). `q/r/s/g` are (offset, row stride) pairs in floats; output rows are n_head*hd.
    pub fn inkling_attn_batch(
        &mut self,
        q: &Buffer, r: &Buffer, sinks: &Buffer, gate: &Buffer,
        q_clip: &Buffer, r_clip: &Buffer, r_bias: &Buffer, k_cache: &Buffer, v_cache: &Buffer, out: &Buffer,
        qs: (u32, u32), rs: (u32, u32), ss: (u32, u32), gs: (u32, u32),
        n_head: u32, n_head_kv: u32, hd: u32, n_rel: u32, n_extent: u32, n_swa: u32, is_swa: bool,
        cap: u32, pos: u32, kv_slot_off: u32, qk_clip: f32, rel_clip: f32, log_alpha: f32, log_floor: f32, n_tok: u32,
    ) {
        // Flash-style kernel: head_dim 128, n_rel <= 16, and 64 / group query tokens per tile
        // small enough that the staged relative-bias rows cover the tile (group >= 2).
        let group = n_head / n_head_kv.max(1);
        if self.flash_attn && hd == 128 && n_rel <= 16 && n_head.is_multiple_of(n_head_kv) && (2..=64).contains(&group) {
            let pc = [
                qs.0, qs.1, rs.0, rs.1, ss.0, ss.1, gs.0, gs.1,
                n_head, n_head_kv, hd, n_rel, n_extent, n_swa, is_swa as u32, cap, pos, kv_slot_off,
                qk_clip.to_bits(), rel_clip.to_bits(), log_alpha.to_bits(), log_floor.to_bits(), n_tok,
            ];
            let p = self.ctx.pipelines.inkling_attn_prefill_f2.as_ref().unwrap_or(&self.ctx.pipelines.inkling_attn_prefill);
            self.dispatch_raw(
                p,
                &[q.buf, r.buf, sinks.buf, gate.buf, q_clip.buf, r_clip.buf, r_bias.buf, k_cache.buf, v_cache.buf, out.buf],
                bytemuck::cast_slice(&pc), n_head_kv, n_tok.div_ceil(64 / group), 1,
            );
            return;
        }
        let pc = [
            qs.0, qs.1, rs.0, rs.1, ss.0, ss.1, gs.0, gs.1,
            n_head, n_head_kv, hd, n_rel, n_extent, n_swa, is_swa as u32, cap, pos, kv_slot_off,
            qk_clip.to_bits(), rel_clip.to_bits(), log_alpha.to_bits(), log_floor.to_bits(),
        ];
        self.dispatch_raw(
            &self.ctx.pipelines.inkling_attn_batch,
            &[q.buf, r.buf, sinks.buf, gate.buf, q_clip.buf, r_clip.buf, r_bias.buf, k_cache.buf, v_cache.buf, out.buf],
            bytemuck::cast_slice(&pc), n_head, n_tok, 1,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inkling_attn(
        &mut self,
        q: &Buffer, k_cur: &Buffer, v_cur: &Buffer, r: &Buffer, sinks: &Buffer, gate: &Buffer,
        q_clip: &Buffer, r_clip: &Buffer, r_bias: &Buffer, k_cache: &Buffer, v_cache: &Buffer, out: &Buffer,
        q_off: u32, r_off: u32, s_off: u32, g_off: u32,
        n_head: u32, n_head_kv: u32, hd: u32, n_rel: u32, n_extent: u32, n_swa: u32, is_swa: bool,
        cap: u32, pos: u32, kv_slot_off: u32, qk_clip: f32, rel_clip: f32, tau: f32, eps: f32,
    ) {
        // Split-K path: store K/V (inkling_kv_store), partial attention per (kv head, key range)
        // for all query heads of the group, merge with sink + gate. The single-kernel version
        // below runs one workgroup per head over the whole context (decode halved at 5.7k tokens).
        let group = n_head / n_head_kv.max(1);
        if hd == 128 && n_rel <= 16 && n_head.is_multiple_of(n_head_kv) && (1..=4).contains(&group) && !inkling_attn_old() {
            let t_start = if is_swa && pos + 1 > n_swa { pos + 1 - n_swa } else { 0 };
            let t_len = pos + 1 - t_start;
            let splits = t_len.div_ceil(256).clamp(1, 32);
            let chunk = t_len.div_ceil(splits);
            let splits = t_len.div_ceil(chunk);
            let part_floats = (splits * n_head * (hd + 2)) as u64;
            if part_floats * 4 <= ATTN_SCRATCH_BYTES {
                let bytes = part_floats * 4;
                if self.scratch_cursor + bytes > ATTN_SCRATCH_BYTES {
                    self.barrier();
                    self.scratch_cursor = 0;
                }
                let part_off = off_u32(self.scratch_cursor / 4);
                self.scratch_cursor += bytes;
                let scratch = self.ctx.attn_scratch;
                self.inkling_kv_store(k_cur, 0, v_cur, 0, k_cache, v_cache, n_head_kv * hd, n_head_kv, hd, is_swa, cap, pos, kv_slot_off, qk_clip, eps, 1);
                self.dep_barrier();
                let pc = [
                    q_off, r_off, s_off, n_head, n_head_kv, n_rel, n_extent, is_swa as u32, cap, pos, kv_slot_off,
                    qk_clip.to_bits(), rel_clip.to_bits(), tau.to_bits(), t_start, chunk, part_off,
                ];
                self.dispatch_raw(
                    &self.ctx.pipelines.inkling_attn_split,
                    &[q.buf, r.buf, sinks.buf, q_clip.buf, r_clip.buf, r_bias.buf, k_cache.buf, v_cache.buf, scratch],
                    bytemuck::cast_slice(&pc), n_head_kv, splits, 1,
                );
                self.dep_barrier();
                let pcm = [g_off, n_head, splits, part_off];
                self.dispatch_raw(&self.ctx.pipelines.inkling_attn_merge, &[scratch, gate.buf, out.buf], bytemuck::cast_slice(&pcm), n_head, 1, 1);
                return;
            }
        }
        let pc = [
            q_off, r_off, s_off, g_off, n_head, n_head_kv, hd, n_rel, n_extent, n_swa, is_swa as u32,
            cap, pos, kv_slot_off, qk_clip.to_bits(), rel_clip.to_bits(), tau.to_bits(), eps.to_bits(),
        ];
        self.dispatch_raw(
            &self.ctx.pipelines.inkling_attn,
            &[q.buf, k_cur.buf, v_cur.buf, r.buf, sinks.buf, gate.buf, q_clip.buf, r_clip.buf, r_bias.buf, k_cache.buf, v_cache.buf, out.buf],
            bytemuck::cast_slice(&pc),
            n_head, 1, 1,
        );
    }

    pub fn kv_write(&mut self, k: &Buffer, k_off: u64, v: &Buffer, v_off: u64, kv: &Buffer, layer_kv_off: u64, page_table: &Buffer, pt_off: u64, pos: u32, kv_dim: u32) {
        self.kv_write_batch(k, k_off, v, v_off, kv, layer_kv_off, page_table, pt_off, pos, kv_dim, 1);
    }
    /// Write K/V rows for tokens pos..pos+batch (rows of kv_dim floats) into the paged cache.
    pub fn kv_write_batch(&mut self, k: &Buffer, k_off: u64, v: &Buffer, v_off: u64, kv: &Buffer, layer_kv_off: u64, page_table: &Buffer, pt_off: u64, pos: u32, kv_dim: u32, batch: u32) {
        let pc = [off_u32(k_off), off_u32(v_off), off_u32(layer_kv_off), off_u32(pt_off), pos, kv_dim];
        self.dispatch_raw(&self.ctx.pipelines.kv_write, &[k.buf, v.buf, kv.buf, page_table.buf], bytemuck::cast_slice(&pc), (kv_dim / 2).div_ceil(256), batch, 1);
    }

    pub fn attention(&mut self, q: &Buffer, q_off: u64, kv: &Buffer, layer_kv_off: u64, page_table: &Buffer, pt_off: u64, out: &Buffer, out_off: u64, n_heads: u32, n_kv_heads: u32, head_dim: u32, pos: u32) {
        self.attention_swa(q, q_off, kv, layer_kv_off, page_table, pt_off, out, out_off, n_heads, n_kv_heads, head_dim, pos, 0);
    }
    /// Flash-style causal prefill attention (attention_prefill.comp) for queries at pos..pos+batch:
    /// each workgroup stages K/V tiles once for group * (64 / group) query rows of one KV head.
    /// Falls back to `attention_batch` when head_dim > 256 or the GQA group exceeds 64.
    pub fn attention_prefill(&mut self, q: &Buffer, q_off: u64, kv: &Buffer, layer_kv_off: u64, page_table: &Buffer, pt_off: u64, out: &Buffer, out_off: u64, n_heads: u32, n_kv_heads: u32, head_dim: u32, pos: u32, sliding_window: u32, batch: u32) {
        let group = n_heads / n_kv_heads.max(1);
        let pipe = match head_dim {
            0..=128 => self.ctx.pipelines.attention_prefill_128_f2.as_ref().unwrap_or(&self.ctx.pipelines.attention_prefill_128),
            129..=256 => self.ctx.pipelines.attention_prefill_256_f2.as_ref().unwrap_or(&self.ctx.pipelines.attention_prefill_256),
            _ => {
                self.attention_batch(q, q_off, kv, layer_kv_off, page_table, pt_off, out, out_off, n_heads, n_kv_heads, head_dim, pos, sliding_window, batch);
                return;
            }
        };
        if group == 0 || group > 64 || !n_heads.is_multiple_of(n_kv_heads) || !head_dim.is_multiple_of(2) {
            self.attention_batch(q, q_off, kv, layer_kv_off, page_table, pt_off, out, out_off, n_heads, n_kv_heads, head_dim, pos, sliding_window, batch);
            return;
        }
        let pc = [off_u32(q_off), off_u32(layer_kv_off), off_u32(pt_off), off_u32(out_off), n_heads, n_kv_heads, head_dim, pos, sliding_window, batch];
        self.dispatch_raw(pipe, &[q.buf, kv.buf, page_table.buf, out.buf], bytemuck::cast_slice(&pc), n_kv_heads, batch.div_ceil(64 / group), 1);
    }

    /// Causal prefill attention for queries at pos..pos+batch in one single-pass dispatch.
    /// q/out rows are n_heads*head_dim floats; K/V for all those positions must already be written.
    pub fn attention_batch(&mut self, q: &Buffer, q_off: u64, kv: &Buffer, layer_kv_off: u64, page_table: &Buffer, pt_off: u64, out: &Buffer, out_off: u64, n_heads: u32, n_kv_heads: u32, head_dim: u32, pos: u32, sliding_window: u32, batch: u32) {
        let pc = [off_u32(q_off), off_u32(layer_kv_off), off_u32(pt_off), off_u32(out_off), n_heads, n_kv_heads, head_dim, pos, sliding_window];
        self.dispatch_raw(&self.ctx.pipelines.attention, &[q.buf, kv.buf, page_table.buf, out.buf], bytemuck::cast_slice(&pc), n_heads, batch, 1);
    }
    pub fn attention_swa(&mut self, q: &Buffer, q_off: u64, kv: &Buffer, layer_kv_off: u64, page_table: &Buffer, pt_off: u64, out: &Buffer, out_off: u64, n_heads: u32, n_kv_heads: u32, head_dim: u32, pos: u32, sliding_window: u32) {
        let t_start = if sliding_window > 0 && pos >= sliding_window { pos + 1 - sliding_window } else { 0 };
        let n_t = pos + 1 - t_start;
        if !self.attn_split || n_t <= attn_split_min_tokens() {
            let pc = [off_u32(q_off), off_u32(layer_kv_off), off_u32(pt_off), off_u32(out_off), n_heads, n_kv_heads, head_dim, pos, sliding_window];
            self.dispatch_raw(&self.ctx.pipelines.attention, &[q.buf, kv.buf, page_table.buf, out.buf], bytemuck::cast_slice(&pc), n_heads, 1, 1);
            return;
        }
        // Split-K: (n_heads x n_splits) partial passes, then a per-head merge.
        let n_splits = n_t.div_ceil(ATTN_SPLIT_CHUNK);
        let bytes = n_heads as u64 * n_splits as u64 * (head_dim as u64 + 2) * 4;
        assert!(bytes <= ATTN_SCRATCH_BYTES, "attention scratch too small: need {bytes} bytes");
        if self.scratch_cursor + bytes > ATTN_SCRATCH_BYTES {
            // Reusing scratch: make sure every earlier split/merge in this recording has finished.
            self.barrier();
            self.scratch_cursor = 0;
        }
        let scratch_off = off_u32(self.scratch_cursor / 4);
        self.scratch_cursor += bytes;
        let scratch = self.ctx.attn_scratch;
        let pc1 = [off_u32(q_off), off_u32(layer_kv_off), off_u32(pt_off), scratch_off, n_heads, n_kv_heads, head_dim, pos, t_start, ATTN_SPLIT_CHUNK];
        self.dispatch_raw(&self.ctx.pipelines.attention_split, &[q.buf, kv.buf, page_table.buf, scratch], bytemuck::cast_slice(&pc1), n_heads, n_splits, 1);
        self.dep_barrier();
        assert!(n_splits <= 1024, "attention_merge supports <= 1024 splits");
        let pc2 = [scratch_off, off_u32(out_off), head_dim, n_splits];
        self.dispatch_raw(&self.ctx.pipelines.attention_merge, &[scratch, out.buf], bytemuck::cast_slice(&pc2), n_heads, head_dim.div_ceil(128), 1);
    }
}
