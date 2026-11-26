use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=csrc/ocak_llama_oracle.c");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    let llama_root = env::var("LLAMA_CPP_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| env::var("HOME").ok().map(|h| PathBuf::from(h).join("llama.cpp")));
    let Some(root) = llama_root else {
        println!("cargo:rustc-env=OCAK_ORACLE_SO=");
        return;
    };
    let llama_inc = root.join("include");
    let ggml_inc = root.join("ggml/include");
    let llama_bin = root.join("build/bin");
    let out_dir = env::var("OUT_DIR").unwrap();
    let so_path = format!("{out_dir}/libocak_llama_oracle.so");
    if llama_inc.join("llama.h").exists() && llama_bin.join("libllama.so").exists() {
        let status = Command::new("gcc")
            .args([
                "-O2", "-shared", "-fPIC",
                "csrc/ocak_llama_oracle.c",
                &format!("-I{}", llama_inc.display()),
                &format!("-I{}", ggml_inc.display()),
                &format!("-L{}", llama_bin.display()),
                &format!("-Wl,-rpath,{}", llama_bin.display()),
                "-lllama", "-lggml",
                "-o", &so_path,
            ])
            .status();
        if let Ok(st) = status {
            if st.success() {
                println!("cargo:rustc-env=OCAK_ORACLE_SO={so_path}");
                return;
            }
        }
    }
    println!("cargo:rustc-env=OCAK_ORACLE_SO=");
}
