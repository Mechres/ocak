//! Ground-up Vulkan 1.3/1.4 compute backend (`ash`) targeting RDNA 2 (`gfx1031`).
use ash::{vk, Device, Entry, Instance};
use ocak_gguf::GgmlType;
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum VkError {
    #[error("vulkan load error: {0}")]
    Load(String),
    #[error("vulkan error: {0:?}")]
    Vk(vk::Result),
    #[error("no suitable Vulkan compute device found")]
    NoDevice,
    #[error("no suitable memory type")]
    NoMemoryType,
    #[error("unsupported weight type: {0}")]
    UnsupportedType(String),
}

impl From<vk::Result> for VkError {
    fn from(r: vk::Result) -> Self {
        VkError::Vk(r)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WType {
    F32,
    F16,
    Q8_0,
    Q4_0,
    Q4K,
    Q5K,
    Q6K,
}

impl WType {
    pub fn from_ggml(t: GgmlType) -> Result<Self, VkError> {
        Ok(match t {
            GgmlType::F32 => WType::F32,
            GgmlType::F16 => WType::F16,
            GgmlType::Q8_0 => WType::Q8_0,
            GgmlType::Q4_0 => WType::Q4_0,
            GgmlType::Q4K => WType::Q4K,
            GgmlType::Q5K => WType::Q5K,
            GgmlType::Q6K => WType::Q6K,
            o => return Err(VkError::UnsupportedType(o.name())),
        })
    }

    pub fn ggml(self) -> GgmlType {
        match self {
            WType::F32 => GgmlType::F32,
            WType::F16 => GgmlType::F16,
            WType::Q8_0 => GgmlType::Q8_0,
            WType::Q4_0 => GgmlType::Q4_0,
            WType::Q4K => GgmlType::Q4K,
            WType::Q5K => GgmlType::Q5K,
            WType::Q6K => GgmlType::Q6K,
        }
    }

    pub fn row_bytes(self, k: u32) -> u32 {
        self.ggml().row_bytes(k as usize).expect("k not aligned to block size") as u32
    }
}

struct PipelineBundle {
    layout: vk::PipelineLayout,
    ds_layout: vk::DescriptorSetLayout,
    pipeline: vk::Pipeline,
}

struct Pipelines {
    rms_norm: PipelineBundle,
    add: PipelineBundle,
    copy: PipelineBundle,
    swiglu: PipelineBundle,
    rope: PipelineBundle,
    kv_write: PipelineBundle,
    attention: PipelineBundle,
    embed: PipelineBundle,
    qmv_f32: PipelineBundle,
    qmv_f16: PipelineBundle,
    qmv_q8_0: PipelineBundle,
    qmv_q4_0: PipelineBundle,
    qmv_q4_k: PipelineBundle,
    qmv_q5_k: PipelineBundle,
    qmv_q6_k: PipelineBundle,
}

impl Pipelines {
    unsafe fn destroy(&self, dev: &Device) {
        for p in [
            &self.rms_norm, &self.add, &self.copy, &self.swiglu, &self.rope,
            &self.kv_write, &self.attention, &self.embed,
            &self.qmv_f32, &self.qmv_f16, &self.qmv_q8_0, &self.qmv_q4_0,
            &self.qmv_q4_k, &self.qmv_q5_k, &self.qmv_q6_k,
        ] {
            dev.destroy_pipeline(p.pipeline, None);
            dev.destroy_pipeline_layout(p.layout, None);
            dev.destroy_descriptor_set_layout(p.ds_layout, None);
        }
    }
}

struct ExecState {
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    desc_pool: vk::DescriptorPool,
    fence: vk::Fence,
    staging_buf: vk::Buffer,
    staging_mem: vk::DeviceMemory,
    staging_ptr: *mut u8,
    staging_cap: u64,
}

unsafe impl Send for ExecState {}

pub struct VkContext {
    _entry: Entry,
    instance: Instance,
    _phys: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    device_name: String,
    subgroup_size: u32,
    vram_bytes: u64,
    pipelines: Pipelines,
    exec: Mutex<ExecState>,
}

const STAGING_SIZE: u64 = 64 * 1024 * 1024;
const MAX_DESC_SETS: u32 = 4096;

fn build_pipeline(
    dev: &Device,
    spv_bytes: &[u8],
    n_bindings: u32,
    push_size: u32,
) -> Result<PipelineBundle, VkError> {
    assert_eq!(spv_bytes.len() % 4, 0);
    let code: Vec<u32> = spv_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    unsafe {
        let sm = dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_bindings)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let ds_layout = dev.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let pc_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_size)];
        let set_layouts = [ds_layout];
        let layout = dev.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&pc_range),
            None,
        )?;
        let entry = CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(sm)
            .name(&entry);
        let ci = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
        let pipeline = dev
            .create_compute_pipelines(vk::PipelineCache::null(), &ci, None)
            .map_err(|(_, e)| VkError::Vk(e))?[0];
        dev.destroy_shader_module(sm, None);
        Ok(PipelineBundle { layout, ds_layout, pipeline })
    }
}

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
                if best.as_ref().map_or(true, |b| score > b.5) {
                    best = Some((pd, qf, name, sg.subgroup_size, vram, score));
                }
            }
            let (phys, qf, device_name, subgroup_size, vram_bytes, _) = best.ok_or(VkError::NoDevice)?;
            let mem_props = instance.get_physical_device_memory_properties(phys);

            let prio = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
            let mut feat12 = vk::PhysicalDeviceVulkan12Features::default()
                .storage_buffer8_bit_access(true)
                .shader_int8(true);
            let mut feat2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut feat12);
            let dci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&qci)
                .push_next(&mut feat2);
            let device = instance.create_device(phys, &dci, None)?;
            let queue = device.get_device_queue(qf, 0);

            let pipelines = Pipelines {
                rms_norm: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.spv")), 3, 20)?,
                add: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/add.spv")), 2, 12)?,
                copy: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/copy.spv")), 2, 12)?,
                swiglu: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/swiglu.spv")), 3, 16)?,
                rope: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv")), 2, 36)?,
                kv_write: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/kv_write.spv")), 4, 24)?,
                attention: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv")), 4, 32)?,
                embed: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/embed.spv")), 2, 20)?,
                qmv_f32: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/qmv_f32.spv")), 3, 24)?,
                qmv_f16: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/qmv_f16.spv")), 3, 24)?,
                qmv_q8_0: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q8_0.spv")), 3, 24)?,
                qmv_q4_0: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q4_0.spv")), 3, 24)?,
                qmv_q4_k: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q4_k.spv")), 3, 24)?,
                qmv_q5_k: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q5_k.spv")), 3, 24)?,
                qmv_q6_k: build_pipeline(&device, include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q6_k.spv")), 3, 24)?,
            };

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
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(MAX_DESC_SETS * 4)];
            let desc_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(MAX_DESC_SETS)
                    .pool_sizes(&pool_sizes),
                None,
            )?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

            let (staging_buf, staging_mem) = alloc_raw_buffer(
                &device,
                &mem_props,
                STAGING_SIZE,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let staging_ptr = device.map_memory(staging_mem, 0, STAGING_SIZE, vk::MemoryMapFlags::empty())? as *mut u8;

            Ok(Arc::new(Self {
                _entry: entry,
                instance,
                _phys: phys,
                device,
                queue,
                mem_props,
                device_name,
                subgroup_size,
                vram_bytes,
                pipelines,
                exec: Mutex::new(ExecState {
                    cmd_pool,
                    cmd_buf,
                    desc_pool,
                    fence,
                    staging_buf,
                    staging_mem,
                    staging_ptr,
                    staging_cap: STAGING_SIZE,
                }),
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

    pub fn create_host_buffer(self: &Arc<Self>, bytes: u64) -> Result<HostBuffer, VkError> {
        let size = bytes.max(16);
        unsafe {
            let (buf, mem) = alloc_raw_buffer(
                &self.device,
                &self.mem_props,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let ptr = self.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty())? as *mut u8;
            Ok(HostBuffer { ctx: Arc::clone(self), buf, mem, ptr, size })
        }
    }

    pub fn upload(&self, dst: &Buffer, dst_off: u64, data: &[u8]) -> Result<(), VkError> {
        if data.is_empty() { return Ok(()); }
        let exec = self.exec.lock().unwrap();
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
                self.device.end_command_buffer(exec.cmd_buf)?;
                self.submit_wait(&exec)?;
            }
            off += chunk;
        }
        Ok(())
    }

    pub fn download(&self, src: &Buffer, src_off: u64, out: &mut [u8]) -> Result<(), VkError> {
        if out.is_empty() { return Ok(()); }
        let exec = self.exec.lock().unwrap();
        let mut off = 0usize;
        while off < out.len() {
            let chunk = (out.len() - off).min(exec.staging_cap as usize);
            unsafe {
                self.device.reset_command_buffer(exec.cmd_buf, vk::CommandBufferResetFlags::empty())?;
                self.device.begin_command_buffer(
                    exec.cmd_buf,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                let region = [vk::BufferCopy::default()
                    .src_offset(src_off + off as u64)
                    .dst_offset(0)
                    .size(chunk as u64)];
                self.device.cmd_copy_buffer(exec.cmd_buf, src.buf, exec.staging_buf, &region);
                self.device.end_command_buffer(exec.cmd_buf)?;
                self.submit_wait(&exec)?;
                std::ptr::copy_nonoverlapping(exec.staging_ptr, out.as_mut_ptr().add(off), chunk);
            }
            off += chunk;
        }
        Ok(())
    }

    unsafe fn submit_wait(&self, exec: &ExecState) -> Result<(), VkError> {
        self.device.reset_fences(&[exec.fence])?;
        let bufs = [exec.cmd_buf];
        let si = [vk::SubmitInfo::default().command_buffers(&bufs)];
        self.device.queue_submit(self.queue, &si, exec.fence)?;
        self.device.wait_for_fences(&[exec.fence], true, u64::MAX)?;
        Ok(())
    }

    pub fn begin(&self) -> Recorder<'_> {
        let exec = self.exec.lock().unwrap();
        unsafe {
            self.device.reset_descriptor_pool(exec.desc_pool, vk::DescriptorPoolResetFlags::empty()).unwrap();
            self.device.reset_command_buffer(exec.cmd_buf, vk::CommandBufferResetFlags::empty()).unwrap();
            self.device.begin_command_buffer(
                exec.cmd_buf,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ).unwrap();
        }
        Recorder { ctx: self, exec }
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Ok(exec) = self.exec.get_mut() {
                self.device.unmap_memory(exec.staging_mem);
                self.device.destroy_buffer(exec.staging_buf, None);
                self.device.free_memory(exec.staging_mem, None);
                self.device.destroy_fence(exec.fence, None);
                self.device.destroy_descriptor_pool(exec.desc_pool, None);
                self.device.destroy_command_pool(exec.cmd_pool, None);
            }
            self.pipelines.destroy(&self.device);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn find_mem_type(
    mp: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mp.memory_type_count {
        if (type_bits & (1 << i)) != 0 && mp.memory_types[i as usize].property_flags.contains(flags) {
            return Some(i);
        }
    }
    None
}

unsafe fn alloc_raw_buffer(
    dev: &Device,
    mp: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    flags: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), VkError> {
    let buf = dev.create_buffer(
        &vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
    )?;
    let req = dev.get_buffer_memory_requirements(buf);
    let mem_type = find_mem_type(mp, req.memory_type_bits, flags).ok_or(VkError::NoMemoryType)?;
    let mem = dev.allocate_memory(
        &vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type),
        None,
    )?;
    dev.bind_buffer_memory(buf, mem, 0)?;
    Ok((buf, mem))
}

pub struct Buffer {
    ctx: Arc<VkContext>,
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    size: u64,
}

impl Buffer {
    pub fn size(&self) -> u64 { self.size }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_buffer(self.buf, None);
            self.ctx.device.free_memory(self.mem, None);
        }
    }
}

pub struct HostBuffer {
    ctx: Arc<VkContext>,
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    ptr: *mut u8,
    size: u64,
}

unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl HostBuffer {
    pub fn size(&self) -> u64 { self.size }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size as usize) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.unmap_memory(self.mem);
            self.ctx.device.destroy_buffer(self.buf, None);
            self.ctx.device.free_memory(self.mem, None);
        }
    }
}

pub struct Recorder<'a> {
    ctx: &'a VkContext,
    exec: std::sync::MutexGuard<'a, ExecState>,
}

impl<'a> Recorder<'a> {
    fn dispatch_raw(
        &mut self,
        pipe: &PipelineBundle,
        bufs: &[vk::Buffer],
        pc: &[u8],
        gx: u32,
        gy: u32,
        gz: u32,
    ) {
        unsafe {
            let layouts = [pipe.ds_layout];
            let ds = self.ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.exec.desc_pool)
                    .set_layouts(&layouts),
            ).expect("descriptor set alloc")[0];
            let infos: Vec<vk::DescriptorBufferInfo> = bufs
                .iter()
                .map(|&b| vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(vk::WHOLE_SIZE))
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            self.ctx.device.update_descriptor_sets(&writes, &[]);
            self.ctx.device.cmd_bind_pipeline(self.exec.cmd_buf, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            self.ctx.device.cmd_bind_descriptor_sets(
                self.exec.cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &[ds],
                &[],
            );
            self.ctx.device.cmd_push_constants(
                self.exec.cmd_buf,
                pipe.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc,
            );
            self.ctx.device.cmd_dispatch(self.exec.cmd_buf, gx, gy, gz);
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

    pub fn qmv(&mut self, ty: WType, w: &Buffer, w_off: u64, x: &Buffer, x_off: u64, y: &Buffer, y_off: u64, k: u32, m: u32) {
        let pipe = match ty {
            WType::F32 => &self.ctx.pipelines.qmv_f32,
            WType::F16 => &self.ctx.pipelines.qmv_f16,
            WType::Q8_0 => &self.ctx.pipelines.qmv_q8_0,
            WType::Q4_0 => &self.ctx.pipelines.qmv_q4_0,
            WType::Q4K => &self.ctx.pipelines.qmv_q4_k,
            WType::Q5K => &self.ctx.pipelines.qmv_q5_k,
            WType::Q6K => &self.ctx.pipelines.qmv_q6_k,
        };
        let pc = [w_off as u32, x_off as u32, y_off as u32, k, m, ty.row_bytes(k)];
        let gx = (m + 3) / 4;
        self.dispatch_raw(pipe, &[w.buf, x.buf, y.buf], bytemuck::cast_slice(&pc), gx, 1, 1);
    }

    pub fn rms_norm(&mut self, x: &Buffer, x_off: u64, w: &Buffer, w_off: u64, out: &Buffer, out_off: u64, n: u32, eps: f32) {
        let pc = [x_off as u32, w_off as u32, out_off as u32, n, eps.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.rms_norm, &[x.buf, w.buf, out.buf], bytemuck::cast_slice(&pc), 1, 1, 1);
    }

    pub fn add(&mut self, a: &Buffer, a_off: u64, b: &Buffer, b_off: u64, n: u32) {
        let pc = [a_off as u32, b_off as u32, n];
        self.dispatch_raw(&self.ctx.pipelines.add, &[a.buf, b.buf], bytemuck::cast_slice(&pc), (n + 255) / 256, 1, 1);
    }

    pub fn copy(&mut self, src: &Buffer, s_off: u64, dst: &Buffer, d_off: u64, n: u32) {
        let pc = [s_off as u32, d_off as u32, n];
        self.dispatch_raw(&self.ctx.pipelines.copy, &[src.buf, dst.buf], bytemuck::cast_slice(&pc), (n + 255) / 256, 1, 1);
    }

    pub fn swiglu(&mut self, gate: &Buffer, g_off: u64, up: &Buffer, u_off: u64, out: &Buffer, o_off: u64, n: u32) {
        let pc = [g_off as u32, u_off as u32, o_off as u32, n];
        self.dispatch_raw(&self.ctx.pipelines.swiglu, &[gate.buf, up.buf, out.buf], bytemuck::cast_slice(&pc), (n + 255) / 256, 1, 1);
    }

    pub fn rope(&mut self, q: &Buffer, q_off: u64, k: &Buffer, k_off: u64, n_heads: u32, n_kv_heads: u32, head_dim: u32, n_rot: u32, pos: u32, freq_base: f32, mode: u32) {
        let pc = [q_off as u32, k_off as u32, n_heads, n_kv_heads, head_dim, n_rot, pos, mode, freq_base.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.rope, &[q.buf, k.buf], bytemuck::cast_slice(&pc), n_heads + n_kv_heads, 1, 1);
    }

    pub fn kv_write(&mut self, k: &Buffer, k_off: u64, v: &Buffer, v_off: u64, kv: &Buffer, layer_kv_off: u64, page_table: &Buffer, pt_off: u64, pos: u32, kv_dim: u32) {
        let pc = [k_off as u32, v_off as u32, layer_kv_off as u32, pt_off as u32, pos, kv_dim];
        self.dispatch_raw(&self.ctx.pipelines.kv_write, &[k.buf, v.buf, kv.buf, page_table.buf], bytemuck::cast_slice(&pc), (kv_dim + 255) / 256, 1, 1);
    }

    pub fn attention(&mut self, q: &Buffer, q_off: u64, kv: &Buffer, layer_kv_off: u64, page_table: &Buffer, pt_off: u64, out: &Buffer, out_off: u64, n_heads: u32, n_kv_heads: u32, head_dim: u32, pos: u32) {
        let pc = [q_off as u32, layer_kv_off as u32, pt_off as u32, out_off as u32, n_heads, n_kv_heads, head_dim, pos];
        self.dispatch_raw(&self.ctx.pipelines.attention, &[q.buf, kv.buf, page_table.buf, out.buf], bytemuck::cast_slice(&pc), n_heads, 1, 1);
    }

    pub fn embed(&mut self, ty: WType, table: &Buffer, token: u32, out: &Buffer, out_off: u64, k: u32) {
        let pc = [ty.ggml().id(), token, out_off as u32, k, ty.row_bytes(k)];
        self.dispatch_raw(&self.ctx.pipelines.embed, &[table.buf, out.buf], bytemuck::cast_slice(&pc), (k + 255) / 256, 1, 1);
    }

    pub fn finish_and_wait(self) -> Result<(), VkError> {
        unsafe {
            self.ctx.device.end_command_buffer(self.exec.cmd_buf)?;
            self.ctx.submit_wait(&self.exec)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            WType::Q4_0 | WType::Q8_0 => for blk in v.chunks_exact_mut(bb) { blk[0..2].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes()); },
            WType::Q4K | WType::Q5K => for blk in v.chunks_exact_mut(bb) {
                blk[0..2].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
                blk[2..4].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes());
            },
            WType::Q6K => for blk in v.chunks_exact_mut(bb) { blk[208..210].copy_from_slice(&f16::from_f32(rng.f() * 0.05).to_le_bytes()); },
        }
        v
    }

    #[test]
    fn vk_qmv_all_types_match_cpu() {
        let c = ctx();
        let mut rng = Lcg(1337);
        for ty in [WType::F32, WType::F16, WType::Q8_0, WType::Q4_0, WType::Q4K, WType::Q5K, WType::Q6K] {
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
    fn vk_context_create_drop_twice() {
        for _ in 0..2 {
            let tmp = VkContext::new().unwrap();
            assert!(!tmp.device_name().is_empty());
            drop(tmp);
        }
    }
}
