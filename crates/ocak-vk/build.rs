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
        "attention_split",
        "attention_merge",
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
        "moe_group",
        "inkling_kv_store",
        "inkling_attn_batch",
        "inkling_attn_prefill",
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
    // Sources compiled several times with different defines: (output name, source, defines).
    println!("cargo:rerun-if-changed=shaders/dequant_i8.glsl");
    let variants: [(&str, &str, &[&str]); 35] = [
        ("attention_prefill_128", "shaders/attention_prefill.comp", &["-DHDMAX=128"]),
        ("attention_prefill_256", "shaders/attention_prefill.comp", &["-DHDMAX=256"]),
        ("quant_q8", "shaders/quant_q8.comp", &[]),
        ("qmmq_q4_k", "shaders/qmm_q8.comp", &["-DQ4_K"]),
        ("qmmq_q5_k", "shaders/qmm_q8.comp", &["-DQ5_K"]),
        ("qmmq_q6_k", "shaders/qmm_q8.comp", &["-DQ6_K"]),
        ("qmmq_q8_0", "shaders/qmm_q8.comp", &["-DQ8_0"]),
        ("qmmq_q2_k", "shaders/qmm_q8.comp", &["-DQ2_K"]),
        ("qmmq_q3_k", "shaders/qmm_q8.comp", &["-DQ3_K"]),
        ("qmmq_iq4_nl", "shaders/qmm_q8.comp", &["-DIQ4_NL"]),
        ("qmmq_iq4_xs", "shaders/qmm_q8.comp", &["-DIQ4_XS"]),
        ("qmmq_iq3_s", "shaders/qmm_q8.comp", &["-DIQ3_S"]),
        ("qmmq_iq3_xxs", "shaders/qmm_q8.comp", &["-DIQ3_XXS"]),
        ("qmmq_iq2_xs", "shaders/qmm_q8.comp", &["-DIQ2_XS"]),
        ("qmmq_iq2_s", "shaders/qmm_q8.comp", &["-DIQ2_S"]),
        ("qmmq_iq2_xxs", "shaders/qmm_q8.comp", &["-DIQ2_XXS"]),
        ("qmmq_iq1_s", "shaders/qmm_q8.comp", &["-DIQ1_S"]),
        ("qmmq_iq1_m", "shaders/qmm_q8.comp", &["-DIQ1_M"]),
        ("qmmq_nvfp4", "shaders/qmm_q8.comp", &["-DNVFP4"]),
        ("qmmqg_q4_k", "shaders/qmm_q8.comp", &["-DQ4_K", "-DGROUPED"]),
        ("qmmqg_q5_k", "shaders/qmm_q8.comp", &["-DQ5_K", "-DGROUPED"]),
        ("qmmqg_q6_k", "shaders/qmm_q8.comp", &["-DQ6_K", "-DGROUPED"]),
        ("qmmqg_q8_0", "shaders/qmm_q8.comp", &["-DQ8_0", "-DGROUPED"]),
        ("qmmqg_q2_k", "shaders/qmm_q8.comp", &["-DQ2_K", "-DGROUPED"]),
        ("qmmqg_q3_k", "shaders/qmm_q8.comp", &["-DQ3_K", "-DGROUPED"]),
        ("qmmqg_iq4_nl", "shaders/qmm_q8.comp", &["-DIQ4_NL", "-DGROUPED"]),
        ("qmmqg_iq4_xs", "shaders/qmm_q8.comp", &["-DIQ4_XS", "-DGROUPED"]),
        ("qmmqg_iq3_s", "shaders/qmm_q8.comp", &["-DIQ3_S", "-DGROUPED"]),
        ("qmmqg_iq3_xxs", "shaders/qmm_q8.comp", &["-DIQ3_XXS", "-DGROUPED"]),
        ("qmmqg_iq2_xs", "shaders/qmm_q8.comp", &["-DIQ2_XS", "-DGROUPED"]),
        ("qmmqg_iq2_s", "shaders/qmm_q8.comp", &["-DIQ2_S", "-DGROUPED"]),
        ("qmmqg_iq2_xxs", "shaders/qmm_q8.comp", &["-DIQ2_XXS", "-DGROUPED"]),
        ("qmmqg_iq1_s", "shaders/qmm_q8.comp", &["-DIQ1_S", "-DGROUPED"]),
        ("qmmqg_iq1_m", "shaders/qmm_q8.comp", &["-DIQ1_M", "-DGROUPED"]),
        ("qmmqg_nvfp4", "shaders/qmm_q8.comp", &["-DNVFP4", "-DGROUPED"]),
    ];
    for (name, src, defs) in variants {
        println!("cargo:rerun-if-changed={src}");
        let dst = out_dir.join(format!("{name}.spv"));
        let status = if has_glslc || has_usr_glslc {
            let bin = if has_glslc { "glslc" } else { "/usr/bin/glslc" };
            Command::new(bin).args(["--target-env=vulkan1.3", "-O"]).args(defs).args([src, "-o"]).arg(&dst).status().expect("failed to invoke glslc")
        } else {
            Command::new("glslangValidator").args(["-V", "--target-env", "vulkan1.3"]).args(defs).arg("-o").arg(&dst).arg(src).status().expect("failed to invoke glslangValidator")
        };
        assert!(status.success(), "shader compilation failed on {src} ({name})");
    }
    // One tiled-prefill matmul source, specialised per weight format.
    let tiled = [
        ("q4_k", "Q4_K"), ("q5_k", "Q5_K"), ("q6_k", "Q6_K"), ("q2_k", "Q2_K"), ("q3_k", "Q3_K"),
        ("q8_0", "Q8_0"), ("q4_0", "Q4_0"), ("f16", "F16"), ("f32", "F32"),
        ("iq3_s", "IQ3_S"), ("iq3_xxs", "IQ3_XXS"), ("iq2_xs", "IQ2_XS"), ("iq2_s", "IQ2_S"),
        ("iq2_xxs", "IQ2_XXS"), ("iq1_s", "IQ1_S"), ("iq1_m", "IQ1_M"), ("iq4_xs", "IQ4_XS"), ("iq4_nl", "IQ4_NL"),
        ("bf16", "BF16"), ("nvfp4", "NVFP4"),
    ];
    let src = "shaders/qmm_tiled.comp";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-changed=shaders/dequant8.glsl");
    println!("cargo:rerun-if-changed=shaders/qmv_multi.comp");
    println!("cargo:rerun-if-changed=shaders/qmv_id_multi.comp");
    // kind 0 = tiled, 1 = grouped tiled, 2 = small-batch matvec, 3 = MoE id matvec
    for (name, def, kind) in tiled.iter().flat_map(|&(n, d)| [(n, d, 0), (n, d, 1), (n, d, 2), (n, d, 3)]) {
        let prefix = ["qmmt", "qmmg", "qmvm", "qmvi"][kind];
        let src = match kind {
            2 => "shaders/qmv_multi.comp",
            3 => "shaders/qmv_id_multi.comp",
            _ => src,
        };
        let dst = out_dir.join(format!("{prefix}_{name}.spv"));
        let define = format!("-D{def}");
        let gdef = if kind == 1 { "-DGROUPED" } else { "-DUNGROUPED" };
        let status = if has_glslc || has_usr_glslc {
            let bin = if has_glslc { "glslc" } else { "/usr/bin/glslc" };
            Command::new(bin)
                .args(["--target-env=vulkan1.3", "-O", &define, gdef, src, "-o"])
                .arg(&dst)
                .status()
                .expect("failed to invoke glslc")
        } else {
            Command::new("glslangValidator")
                .args(["-V", "--target-env", "vulkan1.3", &define, gdef, "-o"])
                .arg(&dst)
                .arg(src)
                .status()
                .expect("failed to invoke glslangValidator")
        };
        assert!(status.success(), "shader compilation failed on {src} ({def})");
    }
}
