//! Model discovery across model directories.
use crate::digest::cached_sha256;
use crate::gguf::GgufFile;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct CatalogEntry {
    pub id: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub arch: String,
    pub quant_summary: String,
    pub n_layers: u64,
    pub n_ctx_train: u64,
    pub sha256: Option<String>,
    pub is_mmproj: bool,
}

fn quant_summary(g: &GgufFile) -> String {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for t in &g.tensors {
        if t.n_dims < 2 { continue; }
        let n = t.ty.name();
        match counts.iter_mut().find(|(k, _)| *k == n) {
            Some(c) => c.1 += 1,
            None => counts.push((n, 1)),
        }
    }
    counts.sort_by_key(|c| std::cmp::Reverse(c.1));
    counts.iter().map(|(k, c)| format!("{k}x{c}")).collect::<Vec<_>>().join(",")
}

pub fn describe(path: &Path) -> Option<CatalogEntry> {
    let g = GgufFile::open(path).ok()?;
    let id = path.file_stem()?.to_string_lossy().to_string();
    let is_mmproj = id.starts_with("mmproj") || g.architecture() == Some("clip");
    Some(CatalogEntry {
        id,
        path: path.to_path_buf(),
        size_bytes: g.file_len(),
        arch: g.architecture().unwrap_or("?").to_string(),
        quant_summary: quant_summary(&g),
        n_layers: g.arch_u64("block_count").unwrap_or(0),
        n_ctx_train: g.arch_u64("context_length").unwrap_or(0),
        sha256: cached_sha256(path),
        is_mmproj,
    })
}

/// Scans `dirs` for `*.gguf` (top level plus one subdirectory level, e.g. `Qwen3-ASR-1.7B/`).
pub fn scan(dirs: &[PathBuf]) -> Vec<CatalogEntry> {
    let mut out = Vec::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(d) else { continue };
        let mut paths: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        paths.sort();
        for p in paths {
            let mut files = Vec::new();
            if p.is_dir() {
                if let Ok(sub) = std::fs::read_dir(&p) {
                    let mut sp: Vec<_> = sub.filter_map(|e| e.ok()).map(|e| e.path()).collect();
                    sp.sort();
                    files.extend(sp);
                }
            } else {
                files.push(p);
            }
            for f in files {
                if f.extension().is_some_and(|e| e == "gguf") {
                    if let Some(e) = describe(&f) {
                        out.push(e);
                    }
                }
            }
        }
    }
    out
}
