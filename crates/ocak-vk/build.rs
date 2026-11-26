use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shaders = [
        "rms_norm",
        "add_rms_norm",
        "add",
        "add_scaled",
        "copy",
        "swiglu",
        "situ",
        "rope",
        "kv_write",
        "attention",
        "embed",
        "qmv_f32",
        "qmv_f16",
        "qmv_bf16",
        "qmv_q8_0",
        "qmv_q4_0",
        "qmv_q4_k",
        "qmv_q5_k",
        "qmv_q6_k",
        "qmv_q2_k",
        "qmv_q3_k",
        "rms_norm_heads",
        "shortconv_step",
        "ssm_conv_silu",
        "deltanet_step",
        "gated_rms_norm",
        "split_qg",
        "mul_sigmoid",
        "mul_head_sigmoid",
        "rope_ext",
        "qmv_iq2_xxs",
        "qmv_iq2_xs",
        "qmv_iq2_s",
        "qmv_iq3_xxs",
        "qmv_iq3_s",
        "qmv_iq1_s",
        "qmv_iq1_m",
        "qmv_iq4_nl",
        "qmv_iq4_xs",
        "qmv_nvfp4",
        "hc_sinkhorn",
        "hc_mix",
        "rope_yarn_mla",
        "mla_attn",
        "moe_topk",
        "qmv_iq3_xxs_id",
        "qmv_iq3_s_id",
        "qmv_nvfp4_id",
        "qmv_q8_0_id",
        "qmv_q4_k_id",
        "situ_id",
        "add_scaled_id",
        "add_scaled_exp_id",
        "inkling_mix",
        "inkling_ctrl",
        "inkling_conv",
        "inkling_attn",
        "qmm_q4_k",
        "qmm_q6_k",
        "qmm_q5_k",
        "qmm_q4_0",
        "qmm_q8_0",
        "qmm_f16",
        "qmm_f32",
        "qmm_iq2_s",
        "qmm_iq2_xs",
        "qmm_iq3_xxs",
        "qmm_iq3_s",
        "qmm2_iq2_xs",
        "qmm2_q4_k",
        "qmm2_q5_k",
        "qmm2_iq3_s",
        "argmax",
    ];
    let has_glslc = Command::new("glslc").arg("--version").output().is_ok();
    let has_usr_glslc = !has_glslc && Command::new("/usr/bin/glslc").arg("--version").output().is_ok();
    let has_glslang = !has_glslc
        && !has_usr_glslc
        && Command::new("glslangValidator").arg("--version").output().is_ok();
    if !has_glslc && !has_usr_glslc && !has_glslang {
        panic!("neither `glslc` (shaderc) nor `glslangValidator` (glslang-tools) found in PATH");
    }
    for name in shaders {
        let src = format!("shaders/{name}.comp");
        println!("cargo:rerun-if-changed={src}");
        let dst = out_dir.join(format!("{name}.spv"));
        let status = if has_glslc || has_usr_glslc {
            let bin = if has_glslc { "glslc" } else { "/usr/bin/glslc" };
            Command::new(bin)
                .args(["--target-env=vulkan1.3", "-O", &src, "-o"])
                .arg(&dst)
                .status()
                .expect("failed to invoke glslc")
        } else {
            Command::new("glslangValidator")
                .args(["-V", "--target-env", "vulkan1.3", "-o"])
                .arg(&dst)
                .arg(&src)
                .status()
                .expect("failed to invoke glslangValidator")
        };
        assert!(status.success(), "shader compilation failed on {src}");
        let _ = fs::metadata(&dst).expect("spv missing");
    }
}
