use libloading::{Library, Symbol};
use std::ffi::CString;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum OracleError {
    #[error("libllama oracle shared library not built")]
    Unavailable,
    #[error("dlopen error: {0}")]
    Load(String),
    #[error("oracle eval returned error code {0}")]
    EvalFailed(i32),
}

type EvalFn = unsafe extern "C" fn(*const std::ffi::c_char, *const i32, i32, *mut f32, i32) -> i32;

pub fn eval_with_llama_oracle(
    model_path: &Path,
    tokens: &[u32],
    n_vocab: usize,
) -> Result<Vec<f32>, OracleError> {
    let so = env!("OCAK_ORACLE_SO");
    if so.is_empty() || !Path::new(so).exists() {
        return Err(OracleError::Unavailable);
    }
    let cpath = CString::new(model_path.to_string_lossy().as_bytes())
        .map_err(|e| OracleError::Load(e.to_string()))?;
    let toks: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let mut logits = vec![0f32; n_vocab];
    unsafe {
        let lib = Library::new(so).map_err(|e| OracleError::Load(e.to_string()))?;
        let f: Symbol<EvalFn> = lib.get(b"ocak_oracle_eval").map_err(|e| OracleError::Load(e.to_string()))?;
        let rc = f(cpath.as_ptr(), toks.as_ptr(), toks.len() as i32, logits.as_mut_ptr(), n_vocab as i32);
        if rc < 0 {
            return Err(OracleError::EvalFailed(rc));
        }
        logits.truncate((rc as usize).min(n_vocab));
    }
    Ok(logits)
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}
