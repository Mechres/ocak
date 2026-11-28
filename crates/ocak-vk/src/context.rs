//! Device setup (`VkContext`), uploads / downloads and the shared execution state.
use ash::{vk, Device, Entry, Instance};
use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use crate::*;

pub(crate) struct ExecState {
    pub(crate) cmd_pool: vk::CommandPool,
    pub(crate) cmd_buf: vk::CommandBuffer,
    pub(crate) desc_pool: vk::DescriptorPool,
    pub(crate) fence: vk::Fence,
    pub(crate) staging_buf: vk::Buffer,
    pub(crate) staging_mem: vk::DeviceMemory,
    pub(crate) staging_ptr: *mut u8,
    pub(crate) staging_cap: u64,
    /// GPU profiling (OCAK_GPU_PROFILE=1): timestamp pool and the pipeline name of each
    /// timed dispatch in the current recording.
    pub(crate) qpool: vk::QueryPool,
    pub(crate) q_names: Vec<&'static str>,
    /// Pipelined weight uploads (`upload_async`): two command buffers / fences, each owning half
    /// of the staging buffer, so the CPU copy of one chunk overlaps the DMA of the previous one.
    /// Every other use of this state drains them first (`drain_async`).
    pub(crate) up_cmds: [vk::CommandBuffer; 2],
    pub(crate) up_fences: [vk::Fence; 2],
    pub(crate) up_pending: [bool; 2],
    pub(crate) up_next: usize,
}

unsafe impl Send for ExecState {}

pub struct VkContext {
    pub(crate) _entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) _phys: vk::PhysicalDevice,
    pub(crate) device: Device,
    pub(crate) push_desc: Option<ash::khr::push_descriptor::Device>,
    pub(crate) queue: vk::Queue,
    pub(crate) mem_props: vk::PhysicalDeviceMemoryProperties,
    pub(crate) device_name: String,
    pub(crate) subgroup_size: u32,
    pub(crate) vram_bytes: u64,
    pub(crate) pipelines: Pipelines,
    pub(crate) iq_buf: vk::Buffer,
    pub(crate) iq_mem: vk::DeviceMemory,
    /// Device-local scratch for split-K attention partials (bump-allocated per recording).
    pub(crate) attn_scratch: vk::Buffer,
    pub(crate) attn_scratch_mem: vk::DeviceMemory,
    pub(crate) q8_scratch: vk::Buffer,
    pub(crate) q8_scratch_mem: vk::DeviceMemory,
    pub(crate) exec: Mutex<ExecState>,
    /// Set while `upload_async` copies may be in flight, so `Buffer::drop` knows to wait for
    /// them (cleared by `drain_async`). Never set while a `Recorder` holds `exec`.
    pub(crate) async_pending: AtomicBool,
    pub(crate) ts_period_ns: f64,
}


pub(crate) const MIXED_DOT_EXT: &CStr = c"VK_VALVE_shader_mixed_float_dot_product";

/// VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE (not in ash 0.38).
#[repr(C)]
pub(crate) struct MixedDotFeatures {
    pub(crate) s_type: vk::StructureType,
    pub(crate) p_next: *mut std::ffi::c_void,
    pub(crate) f16_acc32: vk::Bool32,
    pub(crate) f16_acc16: vk::Bool32,
    pub(crate) bf16_acc: vk::Bool32,
    pub(crate) f8_acc32: vk::Bool32,
}

impl Default for MixedDotFeatures {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1_000_673_000),
            p_next: std::ptr::null_mut(),
            f16_acc32: 0,
            f16_acc16: 0,
            bf16_acc: 0,
            f8_acc32: 0,
        }
    }
}

pub(crate) const STAGING_SIZE: u64 = 64 * 1024 * 1024;
/// Split-K attention scratch (partial softmax states); wraps with a barrier when exhausted.
pub(crate) const ATTN_SCRATCH_BYTES: u64 = 32 * 1024 * 1024;
/// Contexts at or below this many attended tokens use the single-pass attention kernel
/// (override with `OCAK_ATTN_SPLIT_MIN` for tuning / A-B testing).
pub(crate) fn attn_split_min_tokens() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OCAK_ATTN_SPLIT_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(512))
}
/// Tokens per split-K chunk.
pub(crate) const ATTN_SPLIT_CHUNK: u32 = 256;
pub(crate) const MAX_DESC_SETS: u32 = 65536;


impl VkContext {
    pub fn new() -> Result<Arc<Self>, VkError> {
        unsafe {
            let entry = Entry::load().map_err(|e| VkError::Load(e.to_string()))?;
            let app_name = CString::new("ocak").unwrap();
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .api_version(vk::make_api_version(0, 1, 3, 0));
            let val_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
            let mut layers = Vec::new();
            if std::env::var("OCAK_VK_VALIDATION").ok().as_deref() == Some("1") {
                layers.push(val_layer.as_ptr());
            }
            let inst_ci = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers);
            let instance = entry.create_instance(&inst_ci, None)?;

            let devs = instance.enumerate_physical_devices()?;
            let mut best: Option<(vk::PhysicalDevice, u32, String, u32, u64, i32)> = None;
            for pd in devs {
                let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
                let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg);
                instance.get_physical_device_properties2(pd, &mut props2);
                let props = props2.properties;
                if props.api_version < vk::make_api_version(0, 1, 3, 0) || sg.subgroup_size < 32 {
                    continue;
                }
                let mut avail12 = vk::PhysicalDeviceVulkan12Features::default();
                let mut avail2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut avail12);
                instance.get_physical_device_features2(pd, &mut avail2);
                if avail12.storage_buffer8_bit_access == 0 || avail12.shader_int8 == 0 || avail12.shader_float16 == 0 {
                    continue;
                }
                let qfs = instance.get_physical_device_queue_family_properties(pd);
                let qf = match qfs.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)) {
                    Some(i) => i as u32,
                    None => continue,
                };
                let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();
                let mp = instance.get_physical_device_memory_properties(pd);
                let vram: u64 = mp.memory_heaps[..mp.memory_heap_count as usize]
                    .iter()
                    .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                    .map(|h| h.size)
                    .max()
                    .unwrap_or(0);
                let mut score = 0i32;
                if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU { score += 100; }
                if props.vendor_id == 0x1002 { score += 50; }
                if best.as_ref().is_none_or(|b| score > b.5) {
                    best = Some((pd, qf, name, sg.subgroup_size, vram, score));
                }
            }
            let (phys, qf, device_name, subgroup_size, vram_bytes, _) = best.ok_or(VkError::NoDevice)?;
            let mem_props = instance.get_physical_device_memory_properties(phys);

            let exts = instance.enumerate_device_extension_properties(phys)?;
            let has_push_desc_ext = exts.iter().any(|e| {
                CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::push_descriptor::NAME
            });
            let use_push_desc = if has_push_desc_ext {
                let mut pd_props = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
                let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut pd_props);
                instance.get_physical_device_properties2(phys, &mut props2);
                pd_props.max_push_descriptors >= 16
            } else {
                false
            };
            let mut dev_exts: Vec<*const std::ffi::c_char> = Vec::new();
            if use_push_desc {
                dev_exts.push(ash::khr::push_descriptor::NAME.as_ptr());
            }
            // VK_VALVE_shader_mixed_float_dot_product: fp16 x fp16 dot with fp32 accumulate
            // (v_dot2_f32_f16 on RDNA), used by the prefill attention kernels when present.
            let has_mixed_dot = exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == MIXED_DOT_EXT)
                && std::env::var("OCAK_NO_FDOT2").is_err()
                && {
                    let mut q = MixedDotFeatures::default();
                    let mut f2 = vk::PhysicalDeviceFeatures2 { p_next: &mut q as *mut MixedDotFeatures as *mut std::ffi::c_void, ..Default::default() };
                    instance.get_physical_device_features2(phys, &mut f2);
                    q.f16_acc32 != 0
                };
            if has_mixed_dot {
                dev_exts.push(MIXED_DOT_EXT.as_ptr());
            }

            let prio = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
            let mut feat12 = vk::PhysicalDeviceVulkan12Features::default()
                .storage_buffer8_bit_access(true)
                .shader_int8(true)
                .shader_float16(true);
            // shaderIntegerDotProduct is a required Vulkan 1.3 feature (int8 dot matmul).
            let mut feat13 = vk::PhysicalDeviceVulkan13Features::default().shader_integer_dot_product(true);
            let mut feat2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut feat12).push_next(&mut feat13);
            let mut mixed = MixedDotFeatures { f16_acc32: 1, ..Default::default() };
            if has_mixed_dot {
                mixed.p_next = feat2.p_next;
                feat2.p_next = &mut mixed as *mut MixedDotFeatures as *mut std::ffi::c_void;
            }
            let dci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&qci)
                .enabled_extension_names(&dev_exts)
                .push_next(&mut feat2);
            let device = instance.create_device(phys, &dci, None)?;
            let push_desc = if use_push_desc {
                Some(ash::khr::push_descriptor::Device::new(&instance, &device))
            } else {
                None
            };
            let queue = device.get_device_queue(qf, 0);

            let pipelines = Pipelines::create(&device, use_push_desc, has_mixed_dot)?;

            let cmd_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qf)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmd_buf = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            // With push descriptors the pool is never used: keep a minimal one.
            let n_sets = if use_push_desc { 1 } else { MAX_DESC_SETS };
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(n_sets * 12)];
            let desc_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(n_sets)
                    .pool_sizes(&pool_sizes),
                None,
            )?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let up_cmd_vec = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(2),
            )?;
            let up_cmds = [up_cmd_vec[0], up_cmd_vec[1]];
            let up_fences = [device.create_fence(&vk::FenceCreateInfo::default(), None)?, device.create_fence(&vk::FenceCreateInfo::default(), None)?];

            // Host-cached staging when available: downloads (e.g. 1 MB of logits per sampled token)
            // read it back on the CPU, and uncached (write-combined) memory reads at ~200 MB/s.
            let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            let host_cc = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let (staging_buf, staging_mem) = alloc_raw_buffer(&device, &mem_props, STAGING_SIZE, staging_usage, host_cc | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|_| alloc_raw_buffer(&device, &mem_props, STAGING_SIZE, staging_usage, host_cc))?;
            let staging_ptr = device.map_memory(staging_mem, 0, STAGING_SIZE, vk::MemoryMapFlags::empty())? as *mut u8;
            let iq_data = ocak_gguf::iq_tables::packed_gpu_table();
            let iq_bytes = bytemuck::cast_slice::<u32, u8>(&iq_data);
            let (iq_buf, iq_mem) = alloc_raw_buffer(
                &device,
                &mem_props,
                iq_bytes.len() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            std::ptr::copy_nonoverlapping(iq_bytes.as_ptr(), staging_ptr, iq_bytes.len());
            device.begin_command_buffer(cmd_buf, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let reg = [vk::BufferCopy::default().src_offset(0).dst_offset(0).size(iq_bytes.len() as u64)];
            device.cmd_copy_buffer(cmd_buf, staging_buf, iq_buf, &reg);
            device.end_command_buffer(cmd_buf)?;
            let bufs = [cmd_buf];
            device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&bufs)], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.reset_fences(&[fence])?;
            let (attn_scratch, attn_scratch_mem) = alloc_raw_buffer(
                &device,
                &mem_props,
                ATTN_SCRATCH_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            let (q8_scratch, q8_scratch_mem) = alloc_raw_buffer(
                &device,
                &mem_props,
                Q8_SCRATCH_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let ts_period_ns = instance.get_physical_device_properties(phys).limits.timestamp_period as f64;
            let qpool = if gpu_profile() {
                device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(PROF_QUERIES),
                    None,
                )?
            } else {
                vk::QueryPool::null()
            };
            Ok(Arc::new(Self {
                _entry: entry,
                instance,
                _phys: phys,
                device,
                push_desc,
                queue,
                mem_props,
                device_name,
                subgroup_size,
                vram_bytes,
                pipelines,
                iq_buf,
                iq_mem,
                attn_scratch,
                attn_scratch_mem,
                q8_scratch,
                q8_scratch_mem,
                async_pending: AtomicBool::new(false),
                exec: Mutex::new(ExecState {
                    cmd_pool,
                    cmd_buf,
                    desc_pool,
                    fence,
                    staging_buf,
                    staging_mem,
                    staging_ptr,
                    staging_cap: STAGING_SIZE,
                    qpool,
                    q_names: Vec::new(),
                    up_cmds,
                    up_fences,
                    up_pending: [false; 2],
                    up_next: 0,
                }),
                ts_period_ns,
            }))
        }
    }

    pub fn device_name(&self) -> String { self.device_name.clone() }
    pub fn subgroup_size(&self) -> u32 { self.subgroup_size }
    pub fn vram_bytes(&self) -> u64 { self.vram_bytes }

    pub fn create_device_buffer(self: &Arc<Self>, bytes: u64) -> Result<Buffer, VkError> {
        let size = bytes.max(16);
        let (buf, mem) = unsafe {
            alloc_raw_buffer(
                &self.device,
                &self.mem_props,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?
        };
        Ok(Buffer { ctx: Arc::clone(self), buf, mem, size })
    }

    /// Allocates a GPU-readable GTT (`HOST_VISIBLE | HOST_COHERENT`) storage buffer and copies `data` directly
    /// from host memory (no staging copy). Ideal for sparse MoE weights that spill beyond 12 GiB VRAM.
    pub fn create_gtt_buffer_from_slice(self: &Arc<Self>, data: &[u8]) -> Result<Buffer, VkError> {
        let size = (data.len() as u64).max(16);
        unsafe {
            let (buf, mem) = alloc_raw_buffer(
                &self.device,
                &self.mem_props,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let ptr = self.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.device.unmap_memory(mem);
            Ok(Buffer { ctx: Arc::clone(self), buf, mem, size })
        }
    }


    /// Wait for pipelined uploads still in flight (they use the staging buffer).
    pub(crate) fn drain_async(&self, exec: &mut ExecState) {
        for i in 0..2 {
            if exec.up_pending[i] {
                unsafe {
                    let _ = self.device.wait_for_fences(&[exec.up_fences[i]], true, u64::MAX);
                    let _ = self.device.reset_fences(&[exec.up_fences[i]]);
                }
                exec.up_pending[i] = false;
            }
        }
        self.async_pending.store(false, Ordering::Release);
    }

    /// Like `upload`, but returns once the copy is submitted: the next call's CPU copy (into the
    /// other half of the staging buffer) overlaps this DMA. Any later operation on the context
    /// waits for it, so callers need no explicit flush; `upload_flush` makes it explicit.
    pub fn upload_async(&self, dst: &Buffer, dst_off: u64, data: &[u8]) -> Result<(), VkError> {
        if data.is_empty() {
            return Ok(());
        }
        assert!(dst_off.saturating_add(data.len() as u64) <= dst.size, "vk upload out of bounds");
        let mut exec = self.exec.lock().unwrap();
        let half = (exec.staging_cap / 2) as usize;
        let mut off = 0usize;
        while off < data.len() {
            let chunk = (data.len() - off).min(half);
            let slot = exec.up_next;
            exec.up_next ^= 1;
            unsafe {
                if exec.up_pending[slot] {
                    self.device.wait_for_fences(&[exec.up_fences[slot]], true, u64::MAX)?;
                    self.device.reset_fences(&[exec.up_fences[slot]])?;
                    exec.up_pending[slot] = false;
                }
                std::ptr::copy_nonoverlapping(data.as_ptr().add(off), exec.staging_ptr.add(slot * half), chunk);
                let cb = exec.up_cmds[slot];
                self.device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
                self.device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
                let region = [vk::BufferCopy::default().src_offset((slot * half) as u64).dst_offset(dst_off + off as u64).size(chunk as u64)];
                self.device.cmd_copy_buffer(cb, exec.staging_buf, dst.buf, &region);
                let bar = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                self.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &bar, &[], &[]);
                self.device.end_command_buffer(cb)?;
                let bufs = [cb];
                let si = [vk::SubmitInfo::default().command_buffers(&bufs)];
                self.device.queue_submit(self.queue, &si, exec.up_fences[slot])?;
                exec.up_pending[slot] = true;
                self.async_pending.store(true, Ordering::Release);
            }
            off += chunk;
        }
        Ok(())
    }

    /// Wait for all `upload_async` copies.
    pub fn upload_flush(&self) {
        let mut exec = self.exec.lock().unwrap();
        self.drain_async(&mut exec);
    }

    pub fn upload(&self, dst: &Buffer, dst_off: u64, data: &[u8]) -> Result<(), VkError> {
        if data.is_empty() { return Ok(()); }
        assert!(dst_off.saturating_add(data.len() as u64) <= dst.size, "vk upload out of bounds");
        let mut exec = self.exec.lock().unwrap();
        self.drain_async(&mut exec);
        let mut off = 0usize;
        while off < data.len() {
            let chunk = (data.len() - off).min(exec.staging_cap as usize);
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr().add(off), exec.staging_ptr, chunk);
                self.device.reset_command_buffer(exec.cmd_buf, vk::CommandBufferResetFlags::empty())?;
                self.device.begin_command_buffer(
                    exec.cmd_buf,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                let region = [vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(dst_off + off as u64)
                    .size(chunk as u64)];
                self.device.cmd_copy_buffer(exec.cmd_buf, exec.staging_buf, dst.buf, &region);
                let compute_bar = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                self.device.cmd_pipeline_barrier(
                    exec.cmd_buf,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &compute_bar,
                    &[],
                    &[],
                );
                self.device.end_command_buffer(exec.cmd_buf)?;
                self.submit_wait(&exec)?;
            }
            off += chunk;
        }
        Ok(())
    }

    pub fn download(&self, src: &Buffer, src_off: u64, out: &mut [u8]) -> Result<(), VkError> {
        if out.is_empty() { return Ok(()); }
        assert!(src_off.saturating_add(out.len() as u64) <= src.size, "vk download out of bounds");
        let mut exec = self.exec.lock().unwrap();
        self.drain_async(&mut exec);
        let mut off = 0usize;
        while off < out.len() {
            let chunk = (out.len() - off).min(exec.staging_cap as usize);
            unsafe {
                self.device.reset_command_buffer(exec.cmd_buf, vk::CommandBufferResetFlags::empty())?;
                self.device.begin_command_buffer(
                    exec.cmd_buf,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                let pre_bar = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
                self.device.cmd_pipeline_barrier(
                    exec.cmd_buf,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &pre_bar,
                    &[],
                    &[],
                );
                let region = [vk::BufferCopy::default()
                    .src_offset(src_off + off as u64)
                    .dst_offset(0)
                    .size(chunk as u64)];
                self.device.cmd_copy_buffer(exec.cmd_buf, src.buf, exec.staging_buf, &region);
                let host_bar = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                self.device.cmd_pipeline_barrier(
                    exec.cmd_buf,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &host_bar,
                    &[],
                    &[],
                );
                self.device.end_command_buffer(exec.cmd_buf)?;
                self.submit_wait(&exec)?;
                std::ptr::copy_nonoverlapping(exec.staging_ptr, out.as_mut_ptr().add(off), chunk);
            }
            off += chunk;
        }
        Ok(())
    }

    pub(crate) unsafe fn submit_wait(&self, exec: &ExecState) -> Result<(), VkError> {
        self.device.reset_fences(&[exec.fence])?;
        let bufs = [exec.cmd_buf];
        let si = [vk::SubmitInfo::default().command_buffers(&bufs)];
        self.device.queue_submit(self.queue, &si, exec.fence)?;
        self.device.wait_for_fences(&[exec.fence], true, u64::MAX)?;
        Ok(())
    }

    pub fn zero_buffer(&self, dst: &Buffer) -> Result<(), VkError> {
        let mut exec = self.exec.lock().unwrap();
        self.drain_async(&mut exec);
        unsafe {
            self.device.reset_command_buffer(exec.cmd_buf, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                exec.cmd_buf,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.device.cmd_fill_buffer(exec.cmd_buf, dst.buf, 0, vk::WHOLE_SIZE, 0);
            let bar = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            self.device.cmd_pipeline_barrier(exec.cmd_buf, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &bar, &[], &[]);
            self.device.end_command_buffer(exec.cmd_buf)?;
            self.submit_wait(&exec)?;
        }
        Ok(())
    }

    pub fn begin(&self) -> Recorder<'_> {
        let mut exec = self.exec.lock().unwrap();
        self.drain_async(&mut exec);
        unsafe {
            if self.push_desc.is_none() {
                self.device.reset_descriptor_pool(exec.desc_pool, vk::DescriptorPoolResetFlags::empty()).unwrap();
            }
            self.device.reset_command_buffer(exec.cmd_buf, vk::CommandBufferResetFlags::empty()).unwrap();
            self.device.begin_command_buffer(
                exec.cmd_buf,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ).unwrap();
        }
        let mut exec = exec;
        if exec.qpool != vk::QueryPool::null() {
            unsafe { self.device.cmd_reset_query_pool(exec.cmd_buf, exec.qpool, 0, PROF_QUERIES) };
            exec.q_names.clear();
        }
        Recorder { ctx: self, exec, defer_barriers: false, scratch_cursor: 0, attn_split: true, q8_cursor: 0, q8_cache: None, mmq: mmq_enabled(), flash_attn: true }
    }
}


impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            {
                let exec = self.exec.get_mut().unwrap_or_else(|e| e.into_inner());
                if exec.qpool != vk::QueryPool::null() {
                    self.device.destroy_query_pool(exec.qpool, None);
                }
                self.device.unmap_memory(exec.staging_mem);
                self.device.destroy_buffer(exec.staging_buf, None);
                self.device.free_memory(exec.staging_mem, None);
                self.device.destroy_fence(exec.fence, None);
                for f in exec.up_fences {
                    self.device.destroy_fence(f, None);
                }
                self.device.destroy_descriptor_pool(exec.desc_pool, None);
                self.device.destroy_command_pool(exec.cmd_pool, None);
            }
            self.device.destroy_buffer(self.iq_buf, None);
            self.device.free_memory(self.iq_mem, None);
            self.device.destroy_buffer(self.attn_scratch, None);
            self.device.free_memory(self.attn_scratch_mem, None);
            self.device.destroy_buffer(self.q8_scratch, None);
            self.device.free_memory(self.q8_scratch_mem, None);
            self.pipelines.destroy(&self.device);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
