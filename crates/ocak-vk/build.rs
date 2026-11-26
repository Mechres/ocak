use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shaders = [
        "rms_norm",
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
    ];
    for name in shaders {
        let src = format!("shaders/{name}.comp");
        println!("cargo:rerun-if-changed={src}");
        let dst = out_dir.join(format!("{name}.spv"));
        let status = Command::new("/usr/bin/glslc")
            .args(["--target-env=vulkan1.3", "-O", &src, "-o"])
            .arg(&dst)
            .status()
            .expect("failed to invoke /usr/bin/glslc");
        assert!(status.success(), "glslc failed on {src}");
        let _ = fs::metadata(&dst).expect("spv missing");
    }
}
