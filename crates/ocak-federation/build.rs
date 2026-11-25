use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=csrc/ocak_llama_oracle.c");
    let llama_inc = Path::new("/home/mechres/llama.cpp/include");
    let _ggml_inc = Path::new("/home/mechres/llama.cpp/ggml/include");
    let llama_bin = Path::new("/home/mechres/llama.cpp/build/bin");
    let out_dir = env::var("OUT_DIR").unwrap();
    let so_path = format!("{out_dir}/libocak_llama_oracle.so");
    if llama_inc.join("llama.h").exists() && llama_bin.join("libllama.so").exists() {
        let status = Command::new("/usr/bin/gcc")
            .args([
                "-O2", "-shared", "-fPIC",
                "csrc/ocak_llama_oracle.c",
                "-I/home/mechres/llama.cpp/include",
                "-I/home/mechres/llama.cpp/ggml/include",
                "-L/home/mechres/llama.cpp/build/bin",
                "-Wl,-rpath,/home/mechres/llama.cpp/build/bin",
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
