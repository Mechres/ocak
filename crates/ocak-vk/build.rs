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
        "copy",
        "swiglu",
        "rope",
        "kv_write",
        "attention",
        "embed",
        "qmv_f32",
        "qmv_f16",
        "qmv_q8_0",
        "qmv_q4_0",
        "qmv_q4_k",
        "qmv_q5_k",
        "qmv_q6_k",
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
