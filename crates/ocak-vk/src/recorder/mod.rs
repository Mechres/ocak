//! Command recording: `Recorder` batches dispatches into one command buffer and submits it.

mod attention;
mod matmul;
mod moe;
mod ops;
use ash::vk;
use crate::*;

pub struct Recorder<'a> {
    pub(crate) ctx: &'a VkContext,
    pub(crate) exec: std::sync::MutexGuard<'a, ExecState>,
    pub(crate) defer_barriers: bool,
    pub(crate) scratch_cursor: u64,
    pub(crate) attn_split: bool,
    pub(crate) q8_cursor: u64,
    /// Last int8-quantized activation block, reusable by back-to-back integer-dot matmuls on the
    /// same input: (x buffer, x_off, k, batch, q8 scratch byte offset). Any dispatch other than
    /// such a matmul (or any copy / fill) clears it, since it might rewrite x.
    pub(crate) q8_cache: Option<(vk::Buffer, u64, u32, u32, u64)>,
    pub(crate) mmq: bool,
    pub(crate) flash_attn: bool,
}

#[inline]
pub(crate) fn off_u32(v: u64) -> u32 {
    u32::try_from(v).unwrap_or_else(|_| panic!("vulkan buffer offset {v} exceeds u32::MAX (4 GiB)"))
}


impl<'a> Recorder<'a> {
    /// Enable / disable the integer-dot prefill matmul for this recording (default: on unless
    /// OCAK_MMQ=0).
    pub fn set_mmq(&mut self, enable: bool) {
        self.mmq = enable;
    }

    pub fn set_defer_barriers(&mut self, defer: bool) {
        self.defer_barriers = defer;
    }

    /// Allow split-K attention for long contexts (default). Batched prefill turns this off:
    /// it already runs many tokens' attention in parallel, and the split/merge barrier would
    /// serialize them.
    pub fn set_attn_split(&mut self, enable: bool) {
        self.attn_split = enable;
    }

    /// Barrier for a dependency right after a `dispatch_raw`, which already placed one unless
    /// barriers are deferred (or profiling serializes dispatches): avoids back-to-back barriers.
    pub(crate) fn dep_barrier(&mut self) {
        if self.defer_barriers && self.exec.qpool == vk::QueryPool::null() {
            self.barrier();
        }
    }

    pub fn barrier(&mut self) {
        unsafe {
            let bar = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            self.ctx.device.cmd_pipeline_barrier(
                self.exec.cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &bar,
                &[],
                &[],
            );
        }
    }

    pub(crate) fn dispatch_raw(
        &mut self,
        pipe: &PipelineBundle,
        bufs: &[vk::Buffer],
        pc: &[u8],
        gx: u32,
        gy: u32,
        gz: u32,
    ) {
        debug_assert!(bufs.len() <= 16);
        self.q8_cache = None;
        unsafe {
            let mut infos = [vk::DescriptorBufferInfo::default(); 16];
            for (i, &b) in bufs.iter().enumerate() {
                infos[i] = vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(vk::WHOLE_SIZE);
            }
            self.ctx.device.cmd_bind_pipeline(self.exec.cmd_buf, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            if let Some(ref pd) = self.ctx.push_desc {
                let mut writes = [vk::WriteDescriptorSet::default(); 16];
                for i in 0..bufs.len() {
                    writes[i] = vk::WriteDescriptorSet::default()
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[i]));
                }
                pd.cmd_push_descriptor_set(
                    self.exec.cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    pipe.layout,
                    0,
                    &writes[..bufs.len()],
                );
            } else {
                let layouts = [pipe.ds_layout];
                let ds = self.ctx.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.exec.desc_pool)
                        .set_layouts(&layouts),
                ).expect("descriptor set alloc")[0];
                let mut writes = [vk::WriteDescriptorSet::default(); 16];
                for i in 0..bufs.len() {
                    writes[i] = vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[i]));
                }
                self.ctx.device.update_descriptor_sets(&writes[..bufs.len()], &[]);
                self.ctx.device.cmd_bind_descriptor_sets(
                    self.exec.cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    pipe.layout,
                    0,
                    &[ds],
                    &[],
                );
            }
            self.ctx.device.cmd_push_constants(
                self.exec.cmd_buf,
                pipe.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc,
            );
            let qpool = self.exec.qpool;
            let qi = 2 * self.exec.q_names.len() as u32;
            let timed = qpool != vk::QueryPool::null() && qi + 2 <= PROF_QUERIES;
            if timed {
                // Profiling serializes dispatches (barrier after each) so a dispatch's
                // timestamps bracket only its own work.
                self.ctx.device.cmd_write_timestamp(self.exec.cmd_buf, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qpool, qi);
            }
            self.ctx.device.cmd_dispatch(self.exec.cmd_buf, gx, gy, gz);
            if timed {
                self.ctx.device.cmd_write_timestamp(self.exec.cmd_buf, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qpool, qi + 1);
                self.exec.q_names.push(pipe.name);
            }
            if !self.defer_barriers || qpool != vk::QueryPool::null() {
                self.barrier();
            }
        }
    }

    /// Small host -> device write recorded in this command buffer (vkCmdUpdateBuffer, 64 KB
    /// chunks) instead of a separate upload submit + fence wait. Offsets / sizes multiple of 4.
    pub fn update(&mut self, dst: &Buffer, dst_off: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        assert!(dst_off.is_multiple_of(4) && data.len().is_multiple_of(4) && dst_off + data.len() as u64 <= dst.size);
        self.q8_cache = None;
        unsafe {
            let pre = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
            self.ctx.device.cmd_pipeline_barrier(self.exec.cmd_buf, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &pre, &[], &[]);
            for (i, ch) in data.chunks(65536).enumerate() {
                self.ctx.device.cmd_update_buffer(self.exec.cmd_buf, dst.buf, dst_off + (i * 65536) as u64, ch);
            }
            let post = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            self.ctx.device.cmd_pipeline_barrier(self.exec.cmd_buf, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &post, &[], &[]);
        }
    }

    /// finish_and_wait plus device -> host reads recorded at the end of the same command buffer
    /// (one submit instead of one per `download`). Falls back to separate downloads when the
    /// reads do not fit the staging buffer.
    pub fn finish_and_read(self, reads: &mut [(&Buffer, u64, &mut [u8])]) -> Result<(), VkError> {
        let total: u64 = reads.iter().map(|r| (r.2.len() as u64).next_multiple_of(16)).sum();
        let ctx = self.ctx;
        if total > self.exec.staging_cap {
            self.finish_and_wait()?;
            for r in reads.iter_mut() {
                ctx.download(r.0, r.1, r.2)?;
            }
            return Ok(());
        }
        let mut offs = Vec::with_capacity(reads.len());
        unsafe {
            // Covers compute writes and recorded transfers (copies, `update`) before the readback.
            let pre = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
            ctx.device.cmd_pipeline_barrier(self.exec.cmd_buf, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &pre, &[], &[]);
            let mut so = 0u64;
            for r in reads.iter() {
                if !r.2.is_empty() {
                    assert!(r.1 + r.2.len() as u64 <= r.0.size, "vk read out of bounds");
                    let region = [vk::BufferCopy::default().src_offset(r.1).dst_offset(so).size(r.2.len() as u64)];
                    ctx.device.cmd_copy_buffer(self.exec.cmd_buf, r.0.buf, self.exec.staging_buf, &region);
                }
                offs.push(so);
                so += (r.2.len() as u64).next_multiple_of(16);
            }
            let host = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)];
            ctx.device.cmd_pipeline_barrier(self.exec.cmd_buf, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &host, &[], &[]);
        }
        let staging = self.exec.staging_ptr as usize;
        self.end_submit()?;
        for (r, &o) in reads.iter_mut().zip(&offs) {
            unsafe { std::ptr::copy_nonoverlapping((staging + o as usize) as *const u8, r.2.as_mut_ptr(), r.2.len()) };
        }
        Ok(())
    }

    pub fn finish_and_wait(self) -> Result<(), VkError> {
        self.end_submit()
    }

    pub(crate) fn end_submit(&self) -> Result<(), VkError> {
        unsafe {
            self.ctx.device.end_command_buffer(self.exec.cmd_buf)?;
            self.ctx.submit_wait(&self.exec)?;
            let n = self.exec.q_names.len();
            if n > 0 {
                let mut ts = vec![0u64; 2 * n];
                self.ctx.device.get_query_pool_results(
                    self.exec.qpool,
                    0,
                    &mut ts,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
                let mut prof = GPU_PROF.lock().unwrap();
                let prof = prof.get_or_insert_with(Default::default);
                for (i, name) in self.exec.q_names.iter().enumerate() {
                    let e = prof.entry(name).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 += ts[2 * i + 1].saturating_sub(ts[2 * i]) as f64 * self.ctx.ts_period_ns;
                }
            }
        }
        Ok(())
    }
}
