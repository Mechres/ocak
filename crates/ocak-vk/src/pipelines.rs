//! Compute pipeline objects: creation from the build-time SPIR-V and teardown.
use ash::{vk, Device};
use std::ffi::CString;
use crate::*;

pub(crate) struct PipelineBundle {
    pub(crate) name: &'static str,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) ds_layout: vk::DescriptorSetLayout,
    pub(crate) pipeline: vk::Pipeline,
}

pub(crate) struct Pipelines {
    pub(crate) rms_norm: PipelineBundle,
    pub(crate) add_rms_norm: PipelineBundle,
    pub(crate) add: PipelineBundle,
    pub(crate) add_scaled: PipelineBundle,
    pub(crate) copy: PipelineBundle,
    pub(crate) swiglu: PipelineBundle,
    pub(crate) situ: PipelineBundle,
    pub(crate) rope: PipelineBundle,
    pub(crate) kv_write: PipelineBundle,
    pub(crate) attention: PipelineBundle,
    pub(crate) attention_split: PipelineBundle,
    pub(crate) attention_merge: PipelineBundle,
    pub(crate) embed: PipelineBundle,
    pub(crate) qmv_f32: PipelineBundle,
    pub(crate) qmv_f16: PipelineBundle,
    pub(crate) qmv_bf16: PipelineBundle,
    pub(crate) qmv_q8_0: PipelineBundle,
    pub(crate) qmv_q4_0: PipelineBundle,
    pub(crate) qmv_q4_k: PipelineBundle,
    pub(crate) qmv_q5_k: PipelineBundle,
    pub(crate) qmv_q6_k: PipelineBundle,
    pub(crate) qmv_q2_k: PipelineBundle,
    pub(crate) qmv_q3_k: PipelineBundle,
    pub(crate) rms_norm_heads: PipelineBundle,
    pub(crate) shortconv_step: PipelineBundle,
    pub(crate) ssm_conv_silu: PipelineBundle,
    pub(crate) deltanet_step: PipelineBundle,
    pub(crate) deltanet_step_tok: PipelineBundle,
    pub(crate) gated_rms_norm: PipelineBundle,
    pub(crate) split_qg: PipelineBundle,
    pub(crate) mul_sigmoid: PipelineBundle,
    pub(crate) mul_head_sigmoid: PipelineBundle,
    pub(crate) rope_ext: PipelineBundle,
    pub(crate) qmv_iq2_xxs: PipelineBundle,
    pub(crate) qmv_iq2_xs: PipelineBundle,
    pub(crate) qmv_iq2_s: PipelineBundle,
    pub(crate) qmv_iq3_xxs: PipelineBundle,
    pub(crate) qmv_iq3_s: PipelineBundle,
    pub(crate) qmv_iq1_s: PipelineBundle,
    pub(crate) qmv_iq1_m: PipelineBundle,
    pub(crate) qmv_iq4_nl: PipelineBundle,
    pub(crate) qmv_iq4_xs: PipelineBundle,
    pub(crate) qmv_nvfp4: PipelineBundle,
    pub(crate) hc_sinkhorn: PipelineBundle,
    pub(crate) hc_mix: PipelineBundle,
    pub(crate) rope_yarn_mla: PipelineBundle,
    pub(crate) mla_attn: PipelineBundle,
    pub(crate) moe_topk: PipelineBundle,
    pub(crate) qmv_nvfp4_id: PipelineBundle,
    pub(crate) qmv_q4_k_id: PipelineBundle,
    pub(crate) situ_id: PipelineBundle,
    pub(crate) add_scaled_id: PipelineBundle,
    pub(crate) add_scaled_exp_id: PipelineBundle,
    pub(crate) inkling_mix: PipelineBundle,
    pub(crate) inkling_ctrl: PipelineBundle,
    pub(crate) inkling_conv: PipelineBundle,
    pub(crate) inkling_attn: PipelineBundle,
    pub(crate) qmm_q4_k: PipelineBundle,
    pub(crate) qmm_q6_k: PipelineBundle,
    pub(crate) qmm_q5_k: PipelineBundle,
    pub(crate) qmm_q4_0: PipelineBundle,
    pub(crate) qmm_q8_0: PipelineBundle,
    pub(crate) qmm_f16: PipelineBundle,
    pub(crate) qmm_f32: PipelineBundle,
    pub(crate) qmm_iq2_s: PipelineBundle,
    pub(crate) qmm_iq2_xs: PipelineBundle,
    pub(crate) qmm_iq3_xxs: PipelineBundle,
    pub(crate) qmm_iq3_s: PipelineBundle,
    /// Tiled prefill matmul, indexed by `qmmt_index`.
    pub(crate) qmmt: Vec<PipelineBundle>,
    /// Grouped (MoE expert / per-head) variant of `qmmt`, same indexing.
    pub(crate) qmmg: Vec<PipelineBundle>,
    /// Small-batch (2..8 token) matvec, same indexing as `qmmt`.
    pub(crate) qmvm: Vec<PipelineBundle>,
    /// MoE expert matvec for any format (qmv_id_multi.comp), same indexing as `qmmt`.
    pub(crate) qmvi: Vec<PipelineBundle>,
    pub(crate) moe_group: PipelineBundle,
    pub(crate) inkling_kv_store: PipelineBundle,
    pub(crate) inkling_attn_batch: PipelineBundle,
    pub(crate) inkling_attn_prefill: PipelineBundle,
    pub(crate) inkling_attn_prefill_f2: Option<PipelineBundle>,
    pub(crate) mla_attn_prefill: PipelineBundle,
    pub(crate) inkling_attn_split: PipelineBundle,
    pub(crate) inkling_attn_merge: PipelineBundle,
    /// mla_attn_prefill.comp -DFDOT2 (when VK_VALVE_shader_mixed_float_dot_product is present).
    pub(crate) mla_attn_prefill_f2: Option<PipelineBundle>,
    pub(crate) attention_prefill_128: PipelineBundle,
    pub(crate) attention_prefill_256: PipelineBundle,
    pub(crate) attention_prefill_128_f2: Option<PipelineBundle>,
    pub(crate) attention_prefill_256_f2: Option<PipelineBundle>,
    pub(crate) argmax: PipelineBundle,
    pub(crate) sum_splits: PipelineBundle,
    pub(crate) quant_q8: PipelineBundle,
    /// Integer-dot tiled matmuls, indexed by qmmq_index.
    pub(crate) qmmq: Vec<PipelineBundle>,
    /// qmm_q8 -DACT256 (same order as qmmq) and its 256-group quantizer (OCAK_PREFILL_Q256).
    pub(crate) qmmq256: Vec<PipelineBundle>,
    pub(crate) quant_q8_256: PipelineBundle,
    /// Integer-dot small-batch matvecs, indexed by qmmq_index.
    pub(crate) qmvq: Vec<PipelineBundle>,
    /// Grouped (MoE) integer-dot matmuls, indexed by qmmq_index.
    pub(crate) qmmqg: Vec<PipelineBundle>,
}


impl Pipelines {
    pub(crate) unsafe fn destroy(&self, dev: &Device) {
        for p in [
            &self.rms_norm, &self.add_rms_norm, &self.add, &self.add_scaled, &self.copy, &self.swiglu, &self.situ, &self.rope,
            &self.kv_write, &self.attention, &self.attention_split, &self.attention_merge, &self.embed,
            &self.qmv_f32, &self.qmv_f16, &self.qmv_bf16, &self.qmv_q8_0, &self.qmv_q4_0,
            &self.qmv_q4_k, &self.qmv_q5_k, &self.qmv_q6_k, &self.qmv_q2_k, &self.qmv_q3_k,
            &self.rms_norm_heads, &self.shortconv_step, &self.ssm_conv_silu, &self.deltanet_step, &self.deltanet_step_tok,
            &self.gated_rms_norm, &self.split_qg, &self.mul_sigmoid, &self.mul_head_sigmoid, &self.rope_ext,
            &self.qmv_iq2_xxs, &self.qmv_iq2_xs, &self.qmv_iq2_s, &self.qmv_iq3_xxs, &self.qmv_iq3_s,
            &self.qmv_iq1_s, &self.qmv_iq1_m, &self.qmv_iq4_nl, &self.qmv_iq4_xs, &self.qmv_nvfp4,
            &self.hc_sinkhorn, &self.hc_mix, &self.rope_yarn_mla, &self.mla_attn, &self.moe_topk,
            &self.qmv_nvfp4_id, &self.qmv_q4_k_id, &self.situ_id,
            &self.add_scaled_id, &self.add_scaled_exp_id,
            &self.inkling_mix, &self.inkling_ctrl, &self.inkling_conv, &self.inkling_attn,
            &self.qmm_q4_k, &self.qmm_q6_k, &self.qmm_q5_k, &self.qmm_q4_0, &self.qmm_q8_0,
            &self.qmm_f16, &self.qmm_f32, &self.qmm_iq2_s, &self.qmm_iq2_xs, &self.qmm_iq3_xxs, &self.qmm_iq3_s,
            
            &self.argmax,
        ]
        .into_iter()
        .chain(self.qmmt.iter())
        .chain(self.qmmg.iter())
        .chain(self.qmvm.iter())
        .chain(self.qmvi.iter())
        .chain(self.qmmq.iter())
        .chain(self.qmmq256.iter())
        .chain(self.qmvq.iter())
        .chain(self.attention_prefill_128_f2.iter())
        .chain(self.mla_attn_prefill_f2.iter())
        .chain(self.inkling_attn_prefill_f2.iter())
        .chain(self.attention_prefill_256_f2.iter())
        .chain(self.qmmqg.iter())
        .chain([
            &self.moe_group, &self.inkling_kv_store, &self.inkling_attn_batch, &self.inkling_attn_prefill, &self.mla_attn_prefill, &self.inkling_attn_split, &self.inkling_attn_merge,
            &self.attention_prefill_128, &self.attention_prefill_256, &self.quant_q8, &self.quant_q8_256, &self.sum_splits,
        ])
        {
            dev.destroy_pipeline(p.pipeline, None);
            dev.destroy_pipeline_layout(p.layout, None);
            dev.destroy_descriptor_set_layout(p.ds_layout, None);
        }
    }
}


pub(crate) fn build_pipeline(
    dev: &Device,
    spv_bytes: &[u8],
    name: &'static str,
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
        Ok(PipelineBundle { name, layout, ds_layout, pipeline })
    }
}


impl Pipelines {
    /// Builds every compute pipeline (the `_f2` variants only with the VALVE mixed-float dot).
    pub(crate) unsafe fn create(device: &Device, use_push_desc: bool, has_mixed_dot: bool) -> Result<Self, VkError> {
        let bp = |spv: &[u8], name: &'static str, nb: u32, pc: u32| build_pipeline(device, spv, name, nb, pc, use_push_desc);
        Ok(Pipelines {
            rms_norm: bp(include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.spv")), "rms_norm", 3, 28)?,
            add_rms_norm: bp(include_bytes!(concat!(env!("OUT_DIR"), "/add_rms_norm.spv")), "add_rms_norm", 4, 24)?,
            add: bp(include_bytes!(concat!(env!("OUT_DIR"), "/add.spv")), "add", 2, 12)?,
            add_scaled: bp(include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled.spv")), "add_scaled", 2, 16)?,
            copy: bp(include_bytes!(concat!(env!("OUT_DIR"), "/copy.spv")), "copy", 2, 12)?,
            swiglu: bp(include_bytes!(concat!(env!("OUT_DIR"), "/swiglu.spv")), "swiglu", 3, 16)?,
            situ: bp(include_bytes!(concat!(env!("OUT_DIR"), "/situ.spv")), "situ", 2, 24)?,
            rope: bp(include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv")), "rope", 2, 36)?,
            kv_write: bp(include_bytes!(concat!(env!("OUT_DIR"), "/kv_write.spv")), "kv_write", 4, 24)?,
            attention: bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv")), "attention", 4, 36)?,
            attention_split: bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention_split.spv")), "attention_split", 4, 40)?,
            attention_merge: bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention_merge.spv")), "attention_merge", 2, 16)?,
            embed: bp(include_bytes!(concat!(env!("OUT_DIR"), "/embed.spv")), "embed", 3, 20)?,
            qmv_f32: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_f32.spv")), "qmv_f32", 3, 24)?,
            qmv_f16: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_f16.spv")), "qmv_f16", 3, 24)?,
            qmv_bf16: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_bf16.spv")), "qmv_bf16", 3, 24)?,
            qmv_q8_0: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q8_0.spv")), "qmv_q8_0", 3, 24)?,
            qmv_q4_0: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q4_0.spv")), "qmv_q4_0", 3, 24)?,
            qmv_q4_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q4_k.spv")), "qmv_q4_k", 3, 24)?,
            qmv_q5_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q5_k.spv")), "qmv_q5_k", 3, 24)?,
            qmv_q6_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q6_k.spv")), "qmv_q6_k", 3, 24)?,
            qmv_q2_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q2_k.spv")), "qmv_q2_k", 3, 24)?,
            qmv_q3_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q3_k.spv")), "qmv_q3_k", 3, 24)?,
            rms_norm_heads: bp(include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm_heads.spv")), "rms_norm_heads", 2, 16)?,
            shortconv_step: bp(include_bytes!(concat!(env!("OUT_DIR"), "/shortconv_step.spv")), "shortconv_step", 4, 12)?,
            ssm_conv_silu: bp(include_bytes!(concat!(env!("OUT_DIR"), "/ssm_conv_silu.spv")), "ssm_conv_silu", 5, 16)?,
            deltanet_step: bp(include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_step.spv")), "deltanet_step", 7, 20)?,
            deltanet_step_tok: bp(include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_step_tok.spv")), "deltanet_step_tok", 8, 24)?,
            gated_rms_norm: bp(include_bytes!(concat!(env!("OUT_DIR"), "/gated_rms_norm.spv")), "gated_rms_norm", 4, 12)?,
            split_qg: bp(include_bytes!(concat!(env!("OUT_DIR"), "/split_qg.spv")), "split_qg", 3, 8)?,
            mul_sigmoid: bp(include_bytes!(concat!(env!("OUT_DIR"), "/mul_sigmoid.spv")), "mul_sigmoid", 2, 4)?,
            mul_head_sigmoid: bp(include_bytes!(concat!(env!("OUT_DIR"), "/mul_head_sigmoid.spv")), "mul_head_sigmoid", 2, 8)?,
            rope_ext: bp(include_bytes!(concat!(env!("OUT_DIR"), "/rope_ext.spv")), "rope_ext", 3, 40)?,
            qmv_iq2_xxs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq2_xxs.spv")), "qmv_iq2_xxs", 4, 24)?,
            qmv_iq2_xs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq2_xs.spv")), "qmv_iq2_xs", 4, 24)?,
            qmv_iq2_s: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq2_s.spv")), "qmv_iq2_s", 4, 24)?,
            qmv_iq3_xxs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq3_xxs.spv")), "qmv_iq3_xxs", 4, 24)?,
            qmv_iq3_s: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq3_s.spv")), "qmv_iq3_s", 4, 24)?,
            qmv_iq1_s: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq1_s.spv")), "qmv_iq1_s", 4, 24)?,
            qmv_iq1_m: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq1_m.spv")), "qmv_iq1_m", 4, 24)?,
            qmv_iq4_nl: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq4_nl.spv")), "qmv_iq4_nl", 4, 24)?,
            qmv_iq4_xs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_iq4_xs.spv")), "qmv_iq4_xs", 4, 24)?,
            qmv_nvfp4: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_nvfp4.spv")), "qmv_nvfp4", 3, 24)?,
            hc_sinkhorn: bp(include_bytes!(concat!(env!("OUT_DIR"), "/hc_sinkhorn.spv")), "hc_sinkhorn", 3, 8)?,
            hc_mix: bp(include_bytes!(concat!(env!("OUT_DIR"), "/hc_mix.spv")), "hc_mix", 4, 8)?,
            rope_yarn_mla: bp(include_bytes!(concat!(env!("OUT_DIR"), "/rope_yarn_mla.spv")), "rope_yarn_mla", 3, 48)?,
            mla_attn: bp(include_bytes!(concat!(env!("OUT_DIR"), "/mla_attn.spv")), "mla_attn", 5, 36)?,
            moe_topk: bp(include_bytes!(concat!(env!("OUT_DIR"), "/moe_topk.spv")), "moe_topk", 4, 20)?,
            qmv_nvfp4_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_nvfp4_id.spv")), "qmv_nvfp4_id", 4, 32)?,
            qmv_q4_k_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmv_q4_k_id.spv")), "qmv_q4_k_id", 4, 32)?,
            situ_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/situ_id.spv")), "situ_id", 4, 16)?,
            add_scaled_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled_id.spv")), "add_scaled_id", 3, 24)?,
            add_scaled_exp_id: bp(include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled_exp_id.spv")), "add_scaled_exp_id", 5, 16)?,
            inkling_mix: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_mix.spv")), "inkling_mix", 4, 28)?,
            inkling_ctrl: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_ctrl.spv")), "inkling_ctrl", 3, 16)?,
            inkling_conv: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_conv.spv")), "inkling_conv", 5, 48)?,
            inkling_attn: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_attn.spv")), "inkling_attn", 12, 72)?,
            qmm_q4_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_q4_k.spv")), "qmm_q4_k", 3, 28)?,
            qmm_q6_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_q6_k.spv")), "qmm_q6_k", 3, 28)?,
            qmm_q5_k: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_q5_k.spv")), "qmm_q5_k", 3, 28)?,
            qmm_q4_0: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_q4_0.spv")), "qmm_q4_0", 3, 28)?,
            qmm_q8_0: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_q8_0.spv")), "qmm_q8_0", 3, 28)?,
            qmm_f16: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_f16.spv")), "qmm_f16", 3, 28)?,
            qmm_f32: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_f32.spv")), "qmm_f32", 3, 28)?,
            qmm_iq2_s: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_iq2_s.spv")), "qmm_iq2_s", 4, 28)?,
            qmm_iq2_xs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_iq2_xs.spv")), "qmm_iq2_xs", 4, 28)?,
            qmm_iq3_xxs: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_iq3_xxs.spv")), "qmm_iq3_xxs", 4, 28)?,
            qmm_iq3_s: bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmm_iq3_s.spv")), "qmm_iq3_s", 4, 28)?,
            qmmt: vec![
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q4_k.spv")), "qmmt_q4_k", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q5_k.spv")), "qmmt_q5_k", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q6_k.spv")), "qmmt_q6_k", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q2_k.spv")), "qmmt_q2_k", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q3_k.spv")), "qmmt_q3_k", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q8_0.spv")), "qmmt_q8_0", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_q4_0.spv")), "qmmt_q4_0", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_f16.spv")), "qmmt_f16", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_f32.spv")), "qmmt_f32", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq3_s.spv")), "qmmt_iq3_s", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq3_xxs.spv")), "qmmt_iq3_xxs", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq2_xs.spv")), "qmmt_iq2_xs", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq2_s.spv")), "qmmt_iq2_s", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq2_xxs.spv")), "qmmt_iq2_xxs", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq1_s.spv")), "qmmt_iq1_s", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq1_m.spv")), "qmmt_iq1_m", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq4_xs.spv")), "qmmt_iq4_xs", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_iq4_nl.spv")), "qmmt_iq4_nl", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_bf16.spv")), "qmmt_bf16", 4, 36)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmt_nvfp4.spv")), "qmmt_nvfp4", 4, 36)?,
            ],
            qmmg: vec![
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q4_k.spv")), "qmmg_q4_k", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q5_k.spv")), "qmmg_q5_k", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q6_k.spv")), "qmmg_q6_k", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q2_k.spv")), "qmmg_q2_k", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q3_k.spv")), "qmmg_q3_k", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q8_0.spv")), "qmmg_q8_0", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_q4_0.spv")), "qmmg_q4_0", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_f16.spv")), "qmmg_f16", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_f32.spv")), "qmmg_f32", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq3_s.spv")), "qmmg_iq3_s", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq3_xxs.spv")), "qmmg_iq3_xxs", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq2_xs.spv")), "qmmg_iq2_xs", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq2_s.spv")), "qmmg_iq2_s", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq2_xxs.spv")), "qmmg_iq2_xxs", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq1_s.spv")), "qmmg_iq1_s", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq1_m.spv")), "qmmg_iq1_m", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq4_xs.spv")), "qmmg_iq4_xs", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_iq4_nl.spv")), "qmmg_iq4_nl", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_bf16.spv")), "qmmg_bf16", 5, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmg_nvfp4.spv")), "qmmg_nvfp4", 5, 52)?,
            ],
            qmvi: vec![
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_q4_k.spv")), "qmvi_q4_k", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_q5_k.spv")), "qmvi_q5_k", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_q6_k.spv")), "qmvi_q6_k", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_q2_k.spv")), "qmvi_q2_k", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_q3_k.spv")), "qmvi_q3_k", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_q8_0.spv")), "qmvi_q8_0", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_q4_0.spv")), "qmvi_q4_0", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_f16.spv")), "qmvi_f16", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_f32.spv")), "qmvi_f32", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_iq3_s.spv")), "qmvi_iq3_s", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_iq3_xxs.spv")), "qmvi_iq3_xxs", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_iq2_xs.spv")), "qmvi_iq2_xs", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_iq2_s.spv")), "qmvi_iq2_s", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_iq2_xxs.spv")), "qmvi_iq2_xxs", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_iq1_s.spv")), "qmvi_iq1_s", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_iq1_m.spv")), "qmvi_iq1_m", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_iq4_xs.spv")), "qmvi_iq4_xs", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_iq4_nl.spv")), "qmvi_iq4_nl", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_bf16.spv")), "qmvi_bf16", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvi_nvfp4.spv")), "qmvi_nvfp4", 5, 32)?,
            ],
            qmvm: vec![
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_q4_k.spv")), "qmvm_q4_k", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_q5_k.spv")), "qmvm_q5_k", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_q6_k.spv")), "qmvm_q6_k", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_q2_k.spv")), "qmvm_q2_k", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_q3_k.spv")), "qmvm_q3_k", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_q8_0.spv")), "qmvm_q8_0", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_q4_0.spv")), "qmvm_q4_0", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_f16.spv")), "qmvm_f16", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_f32.spv")), "qmvm_f32", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_iq3_s.spv")), "qmvm_iq3_s", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_iq3_xxs.spv")), "qmvm_iq3_xxs", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_iq2_xs.spv")), "qmvm_iq2_xs", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_iq2_s.spv")), "qmvm_iq2_s", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_iq2_xxs.spv")), "qmvm_iq2_xxs", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_iq1_s.spv")), "qmvm_iq1_s", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_iq1_m.spv")), "qmvm_iq1_m", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_iq4_xs.spv")), "qmvm_iq4_xs", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_iq4_nl.spv")), "qmvm_iq4_nl", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_bf16.spv")), "qmvm_bf16", 4, 28)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvm_nvfp4.spv")), "qmvm_nvfp4", 4, 28)?,
            ],
            moe_group: bp(include_bytes!(concat!(env!("OUT_DIR"), "/moe_group.spv")), "moe_group", 2, 12)?,
            inkling_kv_store: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_kv_store.spv")), "inkling_kv_store", 4, 44)?,
            inkling_attn_batch: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_attn_batch.spv")), "inkling_attn_batch", 10, 88)?,
            mla_attn_prefill: bp(include_bytes!(concat!(env!("OUT_DIR"), "/mla_attn_prefill.spv")), "mla_attn_prefill", 4, 24)?,
            inkling_attn_split: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_attn_split.spv")), "inkling_attn_split", 9, 68)?,
            inkling_attn_merge: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_attn_merge.spv")), "inkling_attn_merge", 3, 16)?,
            mla_attn_prefill_f2: if has_mixed_dot { Some(bp(include_bytes!(concat!(env!("OUT_DIR"), "/mla_attn_prefill_f2.spv")), "mla_attn_prefill_f2", 4, 24)?) } else { None },
            inkling_attn_prefill: bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_attn_prefill.spv")), "inkling_attn_prefill", 10, 92)?,
            inkling_attn_prefill_f2: if has_mixed_dot { Some(bp(include_bytes!(concat!(env!("OUT_DIR"), "/inkling_attn_prefill_f2.spv")), "inkling_attn_prefill_f2", 10, 92)?) } else { None },
            attention_prefill_128: bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention_prefill_128.spv")), "attention_prefill_128", 4, 40)?,
            attention_prefill_128_f2: if has_mixed_dot { Some(bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention_prefill_128_f2.spv")), "attention_prefill_128_f2", 4, 40)?) } else { None },
            attention_prefill_256_f2: if has_mixed_dot { Some(bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention_prefill_256_f2.spv")), "attention_prefill_256_f2", 4, 40)?) } else { None },
            attention_prefill_256: bp(include_bytes!(concat!(env!("OUT_DIR"), "/attention_prefill_256.spv")), "attention_prefill_256", 4, 40)?,
            argmax: bp(include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv")), "argmax", 3, 24)?,
            sum_splits: bp(include_bytes!(concat!(env!("OUT_DIR"), "/sum_splits.spv")), "sum_splits", 2, 16)?,
            quant_q8: bp(include_bytes!(concat!(env!("OUT_DIR"), "/quant_q8.spv")), "quant_q8", 3, 16)?,
            quant_q8_256: bp(include_bytes!(concat!(env!("OUT_DIR"), "/quant_q8_256.spv")), "quant_q8_256", 3, 16)?,
            qmmq256: vec![
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_q4_k.spv")), "qmmq256_q4_k", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_q5_k.spv")), "qmmq256_q5_k", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_q6_k.spv")), "qmmq256_q6_k", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_q8_0.spv")), "qmmq256_q8_0", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_q2_k.spv")), "qmmq256_q2_k", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_q3_k.spv")), "qmmq256_q3_k", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_iq4_nl.spv")), "qmmq256_iq4_nl", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_iq4_xs.spv")), "qmmq256_iq4_xs", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_iq3_s.spv")), "qmmq256_iq3_s", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_iq3_xxs.spv")), "qmmq256_iq3_xxs", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_iq2_xs.spv")), "qmmq256_iq2_xs", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_iq2_s.spv")), "qmmq256_iq2_s", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_iq2_xxs.spv")), "qmmq256_iq2_xxs", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_iq1_s.spv")), "qmmq256_iq1_s", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_iq1_m.spv")), "qmmq256_iq1_m", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq256_nvfp4.spv")), "qmmq256_nvfp4", 5, 40)?,
            ],
            qmmq: vec![
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_q4_k.spv")), "qmmq_q4_k", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_q5_k.spv")), "qmmq_q5_k", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_q6_k.spv")), "qmmq_q6_k", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_q8_0.spv")), "qmmq_q8_0", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_q2_k.spv")), "qmmq_q2_k", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_q3_k.spv")), "qmmq_q3_k", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_iq4_nl.spv")), "qmmq_iq4_nl", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_iq4_xs.spv")), "qmmq_iq4_xs", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_iq3_s.spv")), "qmmq_iq3_s", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_iq3_xxs.spv")), "qmmq_iq3_xxs", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_iq2_xs.spv")), "qmmq_iq2_xs", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_iq2_s.spv")), "qmmq_iq2_s", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_iq2_xxs.spv")), "qmmq_iq2_xxs", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_iq1_s.spv")), "qmmq_iq1_s", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_iq1_m.spv")), "qmmq_iq1_m", 5, 40)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmq_nvfp4.spv")), "qmmq_nvfp4", 5, 40)?,
            ],
            qmvq: vec![
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_q4_k.spv")), "qmvq_q4_k", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_q5_k.spv")), "qmvq_q5_k", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_q6_k.spv")), "qmvq_q6_k", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_q8_0.spv")), "qmvq_q8_0", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_q2_k.spv")), "qmvq_q2_k", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_q3_k.spv")), "qmvq_q3_k", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_iq4_nl.spv")), "qmvq_iq4_nl", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_iq4_xs.spv")), "qmvq_iq4_xs", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_iq3_s.spv")), "qmvq_iq3_s", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_iq3_xxs.spv")), "qmvq_iq3_xxs", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_iq2_xs.spv")), "qmvq_iq2_xs", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_iq2_s.spv")), "qmvq_iq2_s", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_iq2_xxs.spv")), "qmvq_iq2_xxs", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_iq1_s.spv")), "qmvq_iq1_s", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_iq1_m.spv")), "qmvq_iq1_m", 5, 32)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmvq_nvfp4.spv")), "qmvq_nvfp4", 5, 32)?,
            ],
            qmmqg: vec![
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_q4_k.spv")), "qmmqg_q4_k", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_q5_k.spv")), "qmmqg_q5_k", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_q6_k.spv")), "qmmqg_q6_k", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_q8_0.spv")), "qmmqg_q8_0", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_q2_k.spv")), "qmmqg_q2_k", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_q3_k.spv")), "qmmqg_q3_k", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_iq4_nl.spv")), "qmmqg_iq4_nl", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_iq4_xs.spv")), "qmmqg_iq4_xs", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_iq3_s.spv")), "qmmqg_iq3_s", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_iq3_xxs.spv")), "qmmqg_iq3_xxs", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_iq2_xs.spv")), "qmmqg_iq2_xs", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_iq2_s.spv")), "qmmqg_iq2_s", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_iq2_xxs.spv")), "qmmqg_iq2_xxs", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_iq1_s.spv")), "qmmqg_iq1_s", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_iq1_m.spv")), "qmmqg_iq1_m", 6, 52)?,
                bp(include_bytes!(concat!(env!("OUT_DIR"), "/qmmqg_nvfp4.spv")), "qmmqg_nvfp4", 6, 52)?,
            ],
        })
    }
}
