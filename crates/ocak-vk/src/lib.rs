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
    BF16,
    Q8_0,
    Q4_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ1S,
    IQ1M,
    IQ4NL,
    IQ4XS,
    NVFP4,
}

impl WType {
    pub fn from_ggml(t: GgmlType) -> Result<Self, VkError> {
        Ok(match t {
            GgmlType::F32 => WType::F32,
            GgmlType::F16 => WType::F16,
            GgmlType::BF16 => WType::BF16,
            GgmlType::Q8_0 => WType::Q8_0,
            GgmlType::Q4_0 => WType::Q4_0,
            GgmlType::Q2K => WType::Q2K,
            GgmlType::Q3K => WType::Q3K,
            GgmlType::Q4K => WType::Q4K,
            GgmlType::Q5K => WType::Q5K,
            GgmlType::Q6K => WType::Q6K,
            GgmlType::IQ2XXS => WType::IQ2XXS,
            GgmlType::IQ2XS => WType::IQ2XS,
            GgmlType::IQ2S => WType::IQ2S,
            GgmlType::IQ3XXS => WType::IQ3XXS,
            GgmlType::IQ3S => WType::IQ3S,
            GgmlType::IQ1S => WType::IQ1S,
            GgmlType::IQ1M => WType::IQ1M,
            GgmlType::IQ4NL => WType::IQ4NL,
            GgmlType::IQ4XS => WType::IQ4XS,
            GgmlType::NVFP4 => WType::NVFP4,
            o => return Err(VkError::UnsupportedType(o.name())),
        })
    }

    pub fn ggml(self) -> GgmlType {
        match self {
            WType::F32 => GgmlType::F32,
            WType::F16 => GgmlType::F16,
            WType::BF16 => GgmlType::BF16,
            WType::Q8_0 => GgmlType::Q8_0,
            WType::Q4_0 => GgmlType::Q4_0,
            WType::Q2K => GgmlType::Q2K,
            WType::Q3K => GgmlType::Q3K,
            WType::Q4K => GgmlType::Q4K,
            WType::Q5K => GgmlType::Q5K,
            WType::Q6K => GgmlType::Q6K,
            WType::IQ2XXS => GgmlType::IQ2XXS,
            WType::IQ2XS => GgmlType::IQ2XS,
            WType::IQ2S => GgmlType::IQ2S,
            WType::IQ3XXS => GgmlType::IQ3XXS,
            WType::IQ3S => GgmlType::IQ3S,
            WType::IQ1S => GgmlType::IQ1S,
            WType::IQ1M => GgmlType::IQ1M,
            WType::IQ4NL => GgmlType::IQ4NL,
            WType::IQ4XS => GgmlType::IQ4XS,
            WType::NVFP4 => GgmlType::NVFP4,
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
    add_rms_norm: PipelineBundle,
    add: PipelineBundle,
    add_scaled: PipelineBundle,
    copy: PipelineBundle,
    swiglu: PipelineBundle,
    situ: PipelineBundle,
    rope: PipelineBundle,
    kv_write: PipelineBundle,
    attention: PipelineBundle,
    attention_split: PipelineBundle,
    attention_merge: PipelineBundle,
    embed: PipelineBundle,
    qmv_f32: PipelineBundle,
    qmv_f16: PipelineBundle,
    qmv_bf16: PipelineBundle,
    qmv_q8_0: PipelineBundle,
    qmv_q4_0: PipelineBundle,
    qmv_q4_k: PipelineBundle,
    qmv_q5_k: PipelineBundle,
    qmv_q6_k: PipelineBundle,
    qmv_q2_k: PipelineBundle,
    qmv_q3_k: PipelineBundle,
    rms_norm_heads: PipelineBundle,
    shortconv_step: PipelineBundle,
    ssm_conv_silu: PipelineBundle,
    deltanet_step: PipelineBundle,
    gated_rms_norm: PipelineBundle,
    split_qg: PipelineBundle,
    mul_sigmoid: PipelineBundle,
    mul_head_sigmoid: PipelineBundle,
    rope_ext: PipelineBundle,
    qmv_iq2_xxs: PipelineBundle,
    qmv_iq2_xs: PipelineBundle,
    qmv_iq2_s: PipelineBundle,
    qmv_iq3_xxs: PipelineBundle,
    qmv_iq3_s: PipelineBundle,
    qmv_iq1_s: PipelineBundle,
    qmv_iq1_m: PipelineBundle,
    qmv_iq4_nl: PipelineBundle,
    qmv_iq4_xs: PipelineBundle,
    qmv_nvfp4: PipelineBundle,
    hc_sinkhorn: PipelineBundle,
    hc_mix: PipelineBundle,
    rope_yarn_mla: PipelineBundle,
    mla_attn: PipelineBundle,
    moe_topk: PipelineBundle,
    qmv_iq3_xxs_id: PipelineBundle,
    qmv_iq3_s_id: PipelineBundle,
    qmv_nvfp4_id: PipelineBundle,
    qmv_q8_0_id: PipelineBundle,
    qmv_q4_k_id: PipelineBundle,
    situ_id: PipelineBundle,
    add_scaled_id: PipelineBundle,
    add_scaled_exp_id: PipelineBundle,
    inkling_mix: PipelineBundle,
    inkling_ctrl: PipelineBundle,
    inkling_conv: PipelineBundle,
    inkling_attn: PipelineBundle,
    qmm_q4_k: PipelineBundle,
    qmm_q6_k: PipelineBundle,
    qmm_q5_k: PipelineBundle,
    qmm_q4_0: PipelineBundle,
    qmm_q8_0: PipelineBundle,
    qmm_f16: PipelineBundle,
    qmm_f32: PipelineBundle,
    qmm_iq2_s: PipelineBundle,
    qmm_iq2_xs: PipelineBundle,
    qmm_iq3_xxs: PipelineBundle,
    qmm_iq3_s: PipelineBundle,
    qmm2_iq2_xs: PipelineBundle,
    qmm2_q4_k: PipelineBundle,
    qmm2_q5_k: PipelineBundle,
    qmm2_iq3_s: PipelineBundle,
    /// Tiled prefill matmul, indexed by `qmmt_index`.
    qmmt: Vec<PipelineBundle>,
    /// Grouped (MoE expert / per-head) variant of `qmmt`, same indexing.
    qmmg: Vec<PipelineBundle>,
    moe_group: PipelineBundle,
    inkling_kv_store: PipelineBundle,
    inkling_attn_batch: PipelineBundle,
    attention_prefill_128: PipelineBundle,
    attention_prefill_256: PipelineBundle,
    argmax: PipelineBundle,
}

/// Weight formats with a tiled prefill matmul (`qmm_tiled.comp`), in `Pipelines::qmmt` order.
fn qmmt_index(ty: WType) -> Option<usize> {
    Some(match ty {
        WType::Q4K => 0,
        WType::Q5K => 1,
        WType::Q6K => 2,
        WType::Q2K => 3,
        WType::Q3K => 4,
        WType::Q8_0 => 5,
        WType::Q4_0 => 6,
        WType::F16 => 7,
        WType::F32 => 8,
        WType::IQ3S => 9,
        WType::IQ3XXS => 10,
        WType::IQ2XS => 11,
        WType::IQ2S => 12,
        WType::IQ2XXS => 13,
        WType::IQ1S => 14,
        WType::IQ1M => 15,
        WType::IQ4XS => 16,
        WType::IQ4NL => 17,
        WType::BF16 => 18,
        WType::NVFP4 => 19,
    })
}

/// Whether batched (prefill) matmuls of this weight format use a tiled kernel. Formats without
/// one fall back to a per-token matvec loop, so callers should keep prefill chunks small.
pub fn has_tiled_qmm(ty: WType) -> bool {
    qmmt_index(ty).is_some()
}

/// Tile shape of qmm_tiled.comp (16*TM rows x 16*TN tokens); keep in sync with the shader defaults.
const QMMT_BM: u32 = 64;
const QMMT_BN: u32 = 128;
/// Column tile of the grouped variant (16 * TN with -DGROUPED).
const QMMG_BN: u32 = 32;

/// Smallest batch that uses the tiled prefill matmul (env `OCAK_QMMT_MIN`, default 32).
fn qmmt_min_batch() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OCAK_QMMT_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(32))
}

impl Pipelines {
    unsafe fn destroy(&self, dev: &Device) {
        for p in [
            &self.rms_norm, &self.add_rms_norm, &self.add, &self.add_scaled, &self.copy, &self.swiglu, &self.situ, &self.rope,
            &self.kv_write, &self.attention, &self.attention_split, &self.attention_merge, &self.embed,
            &self.qmv_f32, &self.qmv_f16, &self.qmv_bf16, &self.qmv_q8_0, &self.qmv_q4_0,
            &self.qmv_q4_k, &self.qmv_q5_k, &self.qmv_q6_k, &self.qmv_q2_k, &self.qmv_q3_k,
            &self.rms_norm_heads, &self.shortconv_step, &self.ssm_conv_silu, &self.deltanet_step,
            &self.gated_rms_norm, &self.split_qg, &self.mul_sigmoid, &self.mul_head_sigmoid, &self.rope_ext,
            &self.qmv_iq2_xxs, &self.qmv_iq2_xs, &self.qmv_iq2_s, &self.qmv_iq3_xxs, &self.qmv_iq3_s,
            &self.qmv_iq1_s, &self.qmv_iq1_m, &self.qmv_iq4_nl, &self.qmv_iq4_xs, &self.qmv_nvfp4,
            &self.hc_sinkhorn, &self.hc_mix, &self.rope_yarn_mla, &self.mla_attn, &self.moe_topk,
            &self.qmv_iq3_xxs_id, &self.qmv_iq3_s_id, &self.qmv_nvfp4_id, &self.qmv_q8_0_id, &self.qmv_q4_k_id, &self.situ_id,
            &self.add_scaled_id, &self.add_scaled_exp_id,
            &self.inkling_mix, &self.inkling_ctrl, &self.inkling_conv, &self.inkling_attn,
            &self.qmm_q4_k, &self.qmm_q6_k, &self.qmm_q5_k, &self.qmm_q4_0, &self.qmm_q8_0,
            &self.qmm_f16, &self.qmm_f32, &self.qmm_iq2_s, &self.qmm_iq2_xs, &self.qmm_iq3_xxs, &self.qmm_iq3_s,
            &self.qmm2_iq2_xs, &self.qmm2_q4_k, &self.qmm2_q5_k, &self.qmm2_iq3_s,
            &self.argmax,
        ]
        .into_iter()
        .chain(self.qmmt.iter())
        .chain(self.qmmg.iter())
        .chain([&self.moe_group, &self.inkling_kv_store, &self.inkling_attn_batch, &self.attention_prefill_128, &self.attention_prefill_256])
        {
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
    push_desc: Option<ash::khr::push_descriptor::Device>,
    queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    device_name: String,
    subgroup_size: u32,
    vram_bytes: u64,
    pipelines: Pipelines,
    iq_buf: vk::Buffer,
    iq_mem: vk::DeviceMemory,
    /// Device-local scratch for split-K attention partials (bump-allocated per recording).
    attn_scratch: vk::Buffer,
    attn_scratch_mem: vk::DeviceMemory,
    exec: Mutex<ExecState>,
}

const STAGING_SIZE: u64 = 64 * 1024 * 1024;
/// Split-K attention scratch (partial softmax states); wraps with a barrier when exhausted.
const ATTN_SCRATCH_BYTES: u64 = 32 * 1024 * 1024;
/// Contexts at or below this many attended tokens use the single-pass attention kernel
/// (override with `OCAK_ATTN_SPLIT_MIN` for tuning / A-B testing).
fn attn_split_min_tokens() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OCAK_ATTN_SPLIT_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(512))
}
/// Tokens per split-K chunk.
const ATTN_SPLIT_CHUNK: u32 = 256;
const MAX_DESC_SETS: u32 = 65536;

fn build_pipeline(
    dev: &Device,
    spv_bytes: &[u8],
    n_bindings: u32,
    push_size: u32,
    use_push_desc: bool,
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
        let ds_flags = if use_push_desc {
            vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR
        } else {
            vk::DescriptorSetLayoutCreateFlags::empty()
        };
        let ds_layout = dev.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default()
                .flags(ds_flags)
                .bindings(&bindings),
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

            let prio = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
            let mut feat12 = vk::PhysicalDeviceVulkan12Features::default()
                .storage_buffer8_bit_access(true)
                .shader_int8(true)
                .shader_float16(true);
            let mut feat2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut feat12);
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

            let bp = |spv: &[u8], nb: u32, pc: u32| build_pipeline(&device, spv, nb, pc, use_push_desc);
            let pipelines = Pipelines {
                rms_norm: bp(include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.spv")), 3, 28)?,
                add_rms_norm: bp(include_bytes!(concat!(env!("OUT_DIR"), "/add_rms_norm.spv")), 4, 24)?,
                add: bp(include_bytes!(concat!(env!("OUT_DIR"), "/add.spv")), 2, 12)?,
                add_scaled: bp(include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled.spv")), 2, 16)?,
                copy: bp(include_bytes!(concat!(env!("OUT_DIR"), "/copy.spv")), 2, 12)?,
                swiglu: bp(include_bytes!(concat!(env!("OUT_DIR"), "/swiglu.spv")), 3, 16)?,
                situ: bp(include_bytes!(concat!(env!("OUT_DIR"), "/situ.spv")), 2, 24)?,
                rope: bp(include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv")), 2, 36)?,
                kv_write: bp(include_bytes!(concat!(env!("OUT_DIR"), "/kv_write.spv")), 4, 24)?,
                attention: bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv")), 4, 36)?,
                attention_split: bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention_split.spv")), 4, 40)?,
                attention_merge: bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention_merge.spv")), 2, 16)?,
                embed: bp(include_bytes!(concat!(env!("OUT_DIR"), "/embed.spv")), 3, 20)?,
                qmv_f32: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_f32.spv")), 3, 24)?,
                qmv_f16: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_f16.spv")), 3, 24)?,
                qmv_bf16: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_bf16.spv")), 3, 24)?,
                qmv_q8_0: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q8_0.spv")), 3, 24)?,
                qmv_q4_0: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q4_0.spv")), 3, 24)?,
                qmv_q4_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q4_k.spv")), 3, 24)?,
                qmv_q5_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q5_k.spv")), 3, 24)?,
                qmv_q6_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q6_k.spv")), 3, 24)?,
                qmv_q2_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q2_k.spv")), 3, 24)?,
                qmv_q3_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q3_k.spv")), 3, 24)?,
                rms_norm_heads: bp(include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm_heads.spv")), 2, 16)?,
                shortconv_step: bp(include_bytes!(concat!(env!("OUT_DIR"), "/shortconv_step.spv")), 4, 12)?,
                ssm_conv_silu: bp(include_bytes!(concat!(env!("OUT_DIR"), "/ssm_conv_silu.spv")), 4, 12)?,
                deltanet_step: bp(include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_step.spv")), 7, 20)?,
                gated_rms_norm: bp(include_bytes!(concat!(env!("OUT_DIR"), "/gated_rms_norm.spv")), 4, 12)?,
                split_qg: bp(include_bytes!(concat!(env!("OUT_DIR"), "/split_qg.spv")), 3, 8)?,
                mul_sigmoid: bp(include_bytes!(concat!(env!("OUT_DIR"), "/mul_sigmoid.spv")), 2, 4)?,
                mul_head_sigmoid: bp(include_bytes!(concat!(env!("OUT_DIR"), "/mul_head_sigmoid.spv")), 2, 8)?,
                rope_ext: bp(include_bytes!(concat!(env!("OUT_DIR"), "/rope_ext.spv")), 3, 40)?,
                qmv_iq2_xxs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq2_xxs.spv")), 4, 24)?,
                qmv_iq2_xs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq2_xs.spv")), 4, 24)?,
                qmv_iq2_s: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq2_s.spv")), 4, 24)?,
                qmv_iq3_xxs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq3_xxs.spv")), 4, 24)?,
                qmv_iq3_s: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq3_s.spv")), 4, 24)?,
                qmv_iq1_s: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq1_s.spv")), 4, 24)?,
                qmv_iq1_m: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq1_m.spv")), 4, 24)?,
                qmv_iq4_nl: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq4_nl.spv")), 4, 24)?,
                qmv_iq4_xs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq4_xs.spv")), 4, 24)?,
                qmv_nvfp4: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_nvfp4.spv")), 3, 24)?,
                hc_sinkhorn: bp(include_bytes!(concat!(env!("OUT_DIR"), "/hc_sinkhorn.spv")), 3, 8)?,
                hc_mix: bp(include_bytes!(concat!(env!("OUT_DIR"), "/hc_mix.spv")), 4, 8)?,
                rope_yarn_mla: bp(include_bytes!(concat!(env!("OUT_DIR"), "/rope_yarn_mla.spv")), 3, 48)?,
                mla_attn: bp(include_bytes!(concat!(env!("OUT_DIR"), "/mla_attn.spv")), 5, 36)?,
                moe_topk: bp(include_bytes!(concat!(env!("OUT_DIR"), "/moe_topk.spv")), 4, 20)?,
                qmv_iq3_xxs_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq3_xxs_id.spv")), 5, 32)?,
                qmv_iq3_s_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq3_s_id.spv")), 5, 32)?,
                qmv_nvfp4_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_nvfp4_id.spv")), 4, 32)?,
                qmv_q8_0_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q8_0_id.spv")), 4, 32)?,
                qmv_q4_k_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q4_k_id.spv")), 4, 32)?,
                situ_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/situ_id.spv")), 4, 16)?,
                add_scaled_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled_id.spv")), 3, 24)?,
                add_scaled_exp_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled_exp_id.spv")), 5, 16)?,
                inkling_mix: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_mix.spv")), 4, 28)?,
                inkling_ctrl: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_ctrl.spv")), 3, 16)?,
                inkling_conv: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_conv.spv")), 5, 48)?,
                inkling_attn: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_attn.spv")), 12, 72)?,
                qmm_q4_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_q4_k.spv")), 3, 28)?,
                qmm_q6_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_q6_k.spv")), 3, 28)?,
                qmm_q5_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_q5_k.spv")), 3, 28)?,
                qmm_q4_0: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_q4_0.spv")), 3, 28)?,
                qmm_q8_0: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_q8_0.spv")), 3, 28)?,
                qmm_f16: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_f16.spv")), 3, 28)?,
                qmm_f32: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_f32.spv")), 3, 28)?,
                qmm_iq2_s: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_iq2_s.spv")), 4, 28)?,
                qmm_iq2_xs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_iq2_xs.spv")), 4, 28)?,
                qmm_iq3_xxs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_iq3_xxs.spv")), 4, 28)?,
                qmm_iq3_s: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_iq3_s.spv")), 4, 28)?,
                qmm2_iq2_xs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm2_iq2_xs.spv")), 4, 28)?,
                qmm2_q4_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm2_q4_k.spv")), 3, 28)?,
                qmm2_q5_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm2_q5_k.spv")), 3, 28)?,
                qmm2_iq3_s: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm2_iq3_s.spv")), 4, 28)?,
                qmmt: vec![
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q4_k.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q5_k.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q6_k.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q2_k.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q3_k.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q8_0.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q4_0.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_f16.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_f32.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq3_s.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq3_xxs.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq2_xs.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq2_s.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq2_xxs.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq1_s.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq1_m.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq4_xs.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq4_nl.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_bf16.spv")), 4, 28)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_nvfp4.spv")), 4, 28)?,
                ],
                qmmg: vec![
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q4_k.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q5_k.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q6_k.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q2_k.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q3_k.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q8_0.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q4_0.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_f16.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_f32.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq3_s.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq3_xxs.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq2_xs.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq2_s.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq2_xxs.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq1_s.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq1_m.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq4_xs.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq4_nl.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_bf16.spv")), 5, 52)?,
                    bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_nvfp4.spv")), 5, 52)?,
                ],
                moe_group: bp(include_bytes!(concat!(env!("OUT_DIR"), "/moe_group.spv")), 2, 12)?,
                inkling_kv_store: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_kv_store.spv")), 4, 44)?,
                inkling_attn_batch: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_attn_batch.spv")), 10, 88)?,
                attention_prefill_128: bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention_prefill_128.spv")), 4, 40)?,
                attention_prefill_256: bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention_prefill_256.spv")), 4, 40)?,
                argmax: bp(include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv")), 2, 8)?,
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
                .descriptor_count(MAX_DESC_SETS * 12)];
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
        assert!(dst_off.saturating_add(data.len() as u64) <= dst.size, "vk upload out of bounds");
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

    unsafe fn submit_wait(&self, exec: &ExecState) -> Result<(), VkError> {
        self.device.reset_fences(&[exec.fence])?;
        let bufs = [exec.cmd_buf];
        let si = [vk::SubmitInfo::default().command_buffers(&bufs)];
        self.device.queue_submit(self.queue, &si, exec.fence)?;
        self.device.wait_for_fences(&[exec.fence], true, u64::MAX)?;
        Ok(())
    }

    pub fn zero_buffer(&self, dst: &Buffer) -> Result<(), VkError> {
        let exec = self.exec.lock().unwrap();
        unsafe {
            self.device.reset_command_buffer(exec.cmd_buf, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                exec.cmd_buf,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.device.cmd_fill_buffer(exec.cmd_buf, dst.buf, 0, vk::WHOLE_SIZE, 0);
            self.device.end_command_buffer(exec.cmd_buf)?;
            self.submit_wait(&exec)?;
        }
        Ok(())
    }

    pub fn begin(&self) -> Recorder<'_> {
        let exec = self.exec.lock().unwrap();
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
        Recorder { ctx: self, exec, defer_barriers: false, scratch_cursor: 0, attn_split: true }
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
            self.device.destroy_buffer(self.iq_buf, None);
            self.device.free_memory(self.iq_mem, None);
            self.device.destroy_buffer(self.attn_scratch, None);
            self.device.free_memory(self.attn_scratch_mem, None);
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
    (0..mp.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0 && mp.memory_types[i as usize].property_flags.contains(flags)
    })
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
    defer_barriers: bool,
    scratch_cursor: u64,
    attn_split: bool,
}

#[inline]
fn off_u32(v: u64) -> u32 {
    u32::try_from(v).unwrap_or_else(|_| panic!("vulkan buffer offset {v} exceeds u32::MAX (4 GiB)"))
}

impl<'a> Recorder<'a> {
    pub fn set_defer_barriers(&mut self, defer: bool) {
        self.defer_barriers = defer;
    }

    /// Allow split-K attention for long contexts (default). Batched prefill turns this off:
    /// it already runs many tokens' attention in parallel, and the split/merge barrier would
    /// serialize them.
    pub fn set_attn_split(&mut self, enable: bool) {
        self.attn_split = enable;
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

    fn dispatch_raw(
        &mut self,
        pipe: &PipelineBundle,
        bufs: &[vk::Buffer],
        pc: &[u8],
        gx: u32,
        gy: u32,
        gz: u32,
    ) {
        debug_assert!(bufs.len() <= 16);
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
            self.ctx.device.cmd_dispatch(self.exec.cmd_buf, gx, gy, gz);
            if !self.defer_barriers {
                self.barrier();
            }
        }
    }

    pub fn qmv(&mut self, ty: WType, w: &Buffer, w_off: u64, x: &Buffer, x_off: u64, y: &Buffer, y_off: u64, k: u32, m: u32) {
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

    #[allow(clippy::too_many_arguments)]
    pub fn qmm(&mut self, ty: WType, w: &Buffer, w_off: u64, x: &Buffer, x_off: u64, y: &Buffer, y_off: u64, k: u32, m: u32, batch: u32) {
        if batch <= 1 {
            self.qmv(ty, w, w_off, x, x_off, y, y_off, k, m);
            return;
        }
        let pc = [off_u32(w_off), off_u32(x_off), off_u32(y_off), k, m, ty.row_bytes(k), batch];
        let gx = m.div_ceil(4);
        if batch == 2 {
            match ty {
                WType::IQ2XS => {
                    self.dispatch_raw(&self.ctx.pipelines.qmm2_iq2_xs, &[w.buf, x.buf, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), m.div_ceil(16), 1, 1);
                    return;
                }
                WType::Q4K => {
                    self.dispatch_raw(&self.ctx.pipelines.qmm2_q4_k, &[w.buf, x.buf, y.buf], bytemuck::cast_slice(&pc), gx, 1, 1);
                    return;
                }
                WType::Q5K => {
                    self.dispatch_raw(&self.ctx.pipelines.qmm2_q5_k, &[w.buf, x.buf, y.buf], bytemuck::cast_slice(&pc), gx, 1, 1);
                    return;
                }
                WType::IQ3S => {
                    self.dispatch_raw(&self.ctx.pipelines.qmm2_iq3_s, &[w.buf, x.buf, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), gx, 1, 1);
                    return;
                }
                _ => {
                    self.qmv(ty, w, w_off, x, x_off, y, y_off, k, m);
                    self.qmv(ty, w, w_off, x, x_off + k as u64, y, y_off + m as u64, k, m);
                    return;
                }
            }
        }
        if batch >= qmmt_min_batch() && k.is_multiple_of(32) {
            if let Some(i) = qmmt_index(ty) {
                let p = &self.ctx.pipelines.qmmt[i];
                self.dispatch_raw(p, &[w.buf, x.buf, y.buf, self.ctx.iq_buf], bytemuck::cast_slice(&pc), m.div_ceil(QMMT_BM), batch.div_ceil(QMMT_BN), 1);
                return;
            }
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
        let pc = [off_u32(x_off), off_u32(r_off), off_u32(w_off), off_u32(out_off), n, eps.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.add_rms_norm, &[x.buf, r.buf, w.buf, out.buf], bytemuck::cast_slice(&pc), 1, 1, 1);
    }

    pub fn argmax(&mut self, x: &Buffer, x_off: u64, out: &Buffer, n: u32) {
        let pc = [off_u32(x_off), n];
        self.dispatch_raw(&self.ctx.pipelines.argmax, &[x.buf, out.buf], bytemuck::cast_slice(&pc), 1, 1, 1);
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
            let reg = [vk::BufferCopy::default().src_offset(0).dst_offset(0).size(bytes)];
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
        self.barrier();
        assert!(n_splits <= 1024, "attention_merge supports <= 1024 splits");
        let pc2 = [scratch_off, 0, latent, n_splits];
        self.dispatch_raw(&self.ctx.pipelines.attention_merge, &[scratch, out.buf], bytemuck::cast_slice(&pc2), nh * batch, latent.div_ceil(128), 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk(&mut self, logits: &Buffer, bias: &Buffer, ids: &Buffer, weights: &Buffer, n_exp: u32, n_used: u32, has_bias: bool, norm_w: bool, scale: f32) {
        self.moe_topk_batch(logits, bias, ids, weights, n_exp, n_used, has_bias, norm_w, scale, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk_batch(&mut self, logits: &Buffer, bias: &Buffer, ids: &Buffer, weights: &Buffer, n_exp: u32, n_used: u32, has_bias: bool, norm_w: bool, scale: f32, batch: u32) {
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
        match ty {
            WType::IQ3XXS => self.dispatch_raw(&self.ctx.pipelines.qmv_iq3_xxs_id, &[w.buf, x.buf, y.buf, self.ctx.iq_buf, ids.buf], bytemuck::cast_slice(&pc), gx, count, 1),
            WType::IQ3S => self.dispatch_raw(&self.ctx.pipelines.qmv_iq3_s_id, &[w.buf, x.buf, y.buf, self.ctx.iq_buf, ids.buf], bytemuck::cast_slice(&pc), gx, count, 1),
            WType::NVFP4 => self.dispatch_raw(&self.ctx.pipelines.qmv_nvfp4_id, &[w.buf, x.buf, y.buf, ids.buf], bytemuck::cast_slice(&pc), gx, count, 1),
            WType::Q8_0 => self.dispatch_raw(&self.ctx.pipelines.qmv_q8_0_id, &[w.buf, x.buf, y.buf, ids.buf], bytemuck::cast_slice(&pc), gx, count, 1),
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
    pub fn qmm_grouped(&mut self, ty: WType, w: &Buffer, w_off: u64, slice_bytes: u64, x: &Buffer, x_off: u64, x_stride: u32, x_div: u32, y: &Buffer, y_off: u64, k: u32, m: u32, n_groups: u32, list: Option<&Buffer>, cols: u32) -> bool {
        let Some(i) = qmmt_index(ty) else { return false };
        if !k.is_multiple_of(32) {
            return false;
        }
        let use_list = list.is_some() as u32;
        let pc = [off_u32(w_off), off_u32(x_off), off_u32(y_off), k, m, ty.row_bytes(k), x_stride, x_div, off_u32(slice_bytes), n_groups, cols, use_list, cols];
        let lbuf = list.map(|b| b.buf).unwrap_or(self.ctx.iq_buf);
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
        self.dispatch_raw(&self.ctx.pipelines.inkling_conv, &[inp.buf, wt.buf, dy.buf, hist.buf, out.buf], bytemuck::cast_slice(&pc), n_ch.div_ceil(256), 1, 1);
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

    pub fn rope(&mut self, q: &Buffer, q_off: u64, k: &Buffer, k_off: u64, n_heads: u32, n_kv_heads: u32, head_dim: u32, n_rot: u32, pos: u32, freq_base: f32, mode: u32) {
        let pc = [off_u32(q_off), off_u32(k_off), n_heads, n_kv_heads, head_dim, n_rot, pos, mode, freq_base.to_bits()];
        self.dispatch_raw(&self.ctx.pipelines.rope, &[q.buf, k.buf], bytemuck::cast_slice(&pc), n_heads + n_kv_heads, 1, 1);
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
    /// each workgroup stages K/V tiles once for 64 query rows of one KV head. Falls back to
    /// `attention_batch` when head_dim > 256 or the GQA group size does not divide 64.
    pub fn attention_prefill(&mut self, q: &Buffer, q_off: u64, kv: &Buffer, layer_kv_off: u64, page_table: &Buffer, pt_off: u64, out: &Buffer, out_off: u64, n_heads: u32, n_kv_heads: u32, head_dim: u32, pos: u32, sliding_window: u32, batch: u32) {
        let group = n_heads / n_kv_heads.max(1);
        let pipe = match head_dim {
            0..=128 => &self.ctx.pipelines.attention_prefill_128,
            129..=256 => &self.ctx.pipelines.attention_prefill_256,
            _ => {
                self.attention_batch(q, q_off, kv, layer_kv_off, page_table, pt_off, out, out_off, n_heads, n_kv_heads, head_dim, pos, sliding_window, batch);
                return;
            }
        };
        if group == 0 || !64u32.is_multiple_of(group) || !head_dim.is_multiple_of(2) {
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
        self.barrier();
        assert!(n_splits <= 1024, "attention_merge supports <= 1024 splits");
        let pc2 = [scratch_off, off_u32(out_off), head_dim, n_splits];
        self.dispatch_raw(&self.ctx.pipelines.attention_merge, &[scratch, out.buf], bytemuck::cast_slice(&pc2), n_heads, head_dim.div_ceil(128), 1);
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
        let pc = [n_embd, l_cache, batch];
        self.dispatch_raw(&self.ctx.pipelines.shortconv_step, &[bcx.buf, cw.buf, state.buf, out.buf], bytemuck::cast_slice(&pc), n_embd.div_ceil(256), 1, 1);
    }

    pub fn ssm_conv_silu(&mut self, x: &Buffer, cw: &Buffer, state: &Buffer, out: &Buffer, channels: u32, k_conv: u32) {
        self.ssm_conv_silu_batch(x, cw, state, out, channels, k_conv, 1);
    }

    pub fn ssm_conv_silu_batch(&mut self, x: &Buffer, cw: &Buffer, state: &Buffer, out: &Buffer, channels: u32, k_conv: u32, batch: u32) {
        let pc = [channels, k_conv, batch];
        self.dispatch_raw(&self.ctx.pipelines.ssm_conv_silu, &[x.buf, cw.buf, state.buf, out.buf], bytemuck::cast_slice(&pc), channels.div_ceil(256), 1, 1);
    }

    pub fn deltanet_step(&mut self, qkv: &Buffer, alpha: &Buffer, beta: &Buffer, dt_bias: &Buffer, ssm_a: &Buffer, state: &Buffer, out: &Buffer, n_k_heads: u32, n_v_heads: u32, head_dim: u32, eps: f32) {
        self.deltanet_step_batch(qkv, alpha, beta, dt_bias, ssm_a, state, out, n_k_heads, n_v_heads, head_dim, eps, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_step_batch(&mut self, qkv: &Buffer, alpha: &Buffer, beta: &Buffer, dt_bias: &Buffer, ssm_a: &Buffer, state: &Buffer, out: &Buffer, n_k_heads: u32, n_v_heads: u32, head_dim: u32, eps: f32, batch: u32) {
        assert!(head_dim <= 128, "deltanet_step supports head_dim <= 128 (got {head_dim})");
        let pc = [n_k_heads, n_v_heads, head_dim, eps.to_bits(), batch];
        // 8 state rows (32-lane clusters) per 256-thread workgroup.
        self.dispatch_raw(&self.ctx.pipelines.deltanet_step, &[qkv.buf, alpha.buf, beta.buf, dt_bias.buf, ssm_a.buf, state.buf, out.buf], bytemuck::cast_slice(&pc), (n_v_heads * head_dim).div_ceil(8), 1, 1);
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
    fn vk_context_create_drop_twice() {
        for _ in 0..2 {
            let tmp = VkContext::new().unwrap();
            assert!(!tmp.device_name().is_empty());
            drop(tmp);
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
        let (k, m, batch) = (2560usize, 9216usize, 256usize);
        let x: Vec<f32> = (0..batch * k).map(|i| ((i % 31) as f32 - 15.0) * 0.01).collect();
        let bx = c.create_device_buffer((x.len() * 4) as u64).unwrap();
        c.upload(&bx, 0, bytemuck::cast_slice(&x)).unwrap();
        let by = c.create_device_buffer((batch * m * 4) as u64).unwrap();
        for wt in [WType::F32, WType::F16, WType::Q8_0, WType::Q4K, WType::Q6K, WType::IQ3S, WType::IQ2XS] {
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
            for wt in [WType::IQ3XXS, WType::IQ3S] {
                let slice = m * wt.row_bytes(k as u32) as usize;
                let raw: Vec<u8> = (0..n_exp * slice).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8 & 0x3F).collect();
                let bw = c.create_device_buffer(raw.len() as u64).unwrap();
                c.upload(&bw, 0, &raw).unwrap();
                let run = |n: usize| {
                    let mut rec = c.begin();
                    rec.moe_group(&bids, &blist, (batch * n_used) as u32, n_exp as u32, batch as u32);
                    rec.barrier();
                    for _ in 0..n {
                        rec.qmm_grouped(wt, &bw, 0, slice as u64, &bx, 0, k as u32, if k > m { n_used as u32 } else { 1 }, &by, 0, k as u32, m as u32, n_exp as u32, Some(&blist), batch as u32);
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
        for &(nh, nkv, hd, pos0, ntok, sw) in &[(16u32, 4u32, 128u32, 37u32, 75u32, 0u32), (8, 8, 64, 0, 40, 0), (16, 2, 256, 300, 50, 0), (12, 4, 128, 500, 70, 64)] {
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
        for wt in [WType::IQ3S, WType::Q4K, WType::IQ4NL, WType::Q8_0, WType::NVFP4] {
            let rb = wt.row_bytes(k as u32) as usize;
            let slice = m * rb;
            let raw: Vec<u8> = (0..n_exp * slice).map(|_| (next() & 0x3F) as u8).collect();
            let bw = c.create_device_buffer(raw.len() as u64).unwrap();
            c.upload(&bw, 0, &raw).unwrap();
            // Expert mode: entry e = t * n_used + slot reads X row t.
            let mut rec = c.begin();
            rec.moe_group(&bids, &blist, n_ent as u32, n_exp as u32, batch as u32);
            rec.barrier();
            assert!(rec.qmm_grouped(wt, &bw, 0, slice as u64, &bx, 0, k as u32, n_used as u32, &by_g, 0, k as u32, m as u32, n_exp as u32, Some(&blist), batch as u32));
            for (e, &ex) in ids.iter().enumerate() {
                rec.qmv(wt, &bw, ex as u64 * slice as u64, &bx, ((e / n_used) * k) as u64, &by_r, (e * m) as u64, k as u32, m as u32);
            }
            rec.finish_and_wait().unwrap();
            let mut yg = vec![0f32; n_ent * m];
            let mut yr = vec![0f32; n_ent * m];
            c.download(&by_g, 0, bytemuck::cast_slice_mut(&mut yg)).unwrap();
            c.download(&by_r, 0, bytemuck::cast_slice_mut(&mut yr)).unwrap();
            let scale = yr.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
            for i in 0..n_ent * m {
                assert!((yg[i] - yr[i]).abs() <= 3e-3 * scale, "grouped(list) {wt:?} idx={i}: {} vs {}", yg[i], yr[i]);
            }
            // Head mode: group g = head, entry = t * n_groups + g, X rows strided.
            let (nh, stride) = (3usize, k + 64);
            let cols = 15usize;
            let xh: Vec<f32> = (0..cols * nh * stride).map(|i| ((i % 41) as f32 - 20.0) * 0.02).collect();
            let bxh = c.create_device_buffer((xh.len() * 4) as u64).unwrap();
            c.upload(&bxh, 0, bytemuck::cast_slice(&xh)).unwrap();
            let mut rec = c.begin();
            assert!(rec.qmm_grouped(wt, &bw, 0, slice as u64, &bxh, 0, stride as u32, 1, &by_g, 0, k as u32, m as u32, nh as u32, None, cols as u32));
            for t in 0..cols {
                for h in 0..nh {
                    let e = t * nh + h;
                    rec.qmv(wt, &bw, (h * slice) as u64, &bxh, (e * stride) as u64, &by_r, (e * m) as u64, k as u32, m as u32);
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
    fn vk_qmm_tiled_matches_qmv() {
        let c = VkContext::new().expect("vulkan init");
        // m and batch deliberately not multiples of the 64x64 tile.
        let (k, m, batch) = (512usize, 70usize, 75usize);
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
            assert!(qmmt_index(wt).is_some());
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
            rec.qmm(wt, &bw, 0, &bx, 0, &by_qmm, 0, k as u32, m as u32, batch as u32);
            for b in 0..batch {
                rec.qmv(wt, &bw, 0, &bx, (b * k) as u64, &by_qmv, (b * m) as u64, k as u32, m as u32);
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
                    "tiled qmm vs qmv mismatch for {wt:?} idx={i}: qmm={} qmv={}",
                    y_qmm[i],
                    y_qmv[i]
                );
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
            rec.qmm(wt, &bw, 0, &bx, 0, &by_qmm, 0, k as u32, m as u32, batch as u32);
            for b in 0..batch {
                rec.qmv(wt, &bw, 0, &bx, (b * k) as u64, &by_qmv, (b * m) as u64, k as u32, m as u32);
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
            // Also verify batch=2 (qmm2_* fast path)
            let mut rec2 = c.begin();
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
}
