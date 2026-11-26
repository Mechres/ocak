use ocak_gguf::preset::Presets;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[derive(Debug, Clone, Serialize)]
pub struct HipProbe {
    pub available: bool,
    pub library_path: Option<String>,
    pub detail: String,
}

pub fn hip_probe() -> HipProbe {
    for cand in [
        "libamdhip64.so.7",
        "libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so",
    ] {
        if unsafe { libloading::Library::new(cand) }.is_ok() {
            return HipProbe {
                available: true,
                library_path: Some(cand.to_string()),
                detail: format!("loaded {cand} dynamically"),
            };
        }
    }
    HipProbe {
        available: false,
        library_path: None,
        detail: "libamdhip64.so not installed; using native Vulkan 1.4 RADV".into(),
    }
}

pub struct LlamaBridgeHandle {
    pub model_id: String,
    pub port: u16,
    child: Child,
}

impl LlamaBridgeHandle {
    pub fn base_addr(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for LlamaBridgeHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

pub struct ManagedLlamaBridge {
    pub bin_path: PathBuf,
    pub lib_dir: PathBuf,
}

impl Default for ManagedLlamaBridge {
    fn default() -> Self {
        Self {
            bin_path: PathBuf::from("/home/mechres/llama.cpp/build/bin/llama-server"),
            lib_dir: PathBuf::from("/home/mechres/llama.cpp/build/bin"),
        }
    }
}

impl ManagedLlamaBridge {
    pub fn available(&self) -> bool {
        self.bin_path.exists()
    }

    /// Spawns a dedicated loopback `llama-server` configured from `models-preset.ini`
    /// and waits until `/health` responds.
    pub async fn spawn_model(
        &self,
        model_id: &str,
        model_path: &Path,
        mmproj_path: Option<&Path>,
        presets: &Presets,
        port: u16,
    ) -> Result<LlamaBridgeHandle, String> {
        if !self.available() {
            return Err(format!("llama-server not found at {}", self.bin_path.display()));
        }
        let resolved = presets.resolve(model_id);
        let mut cmd = Command::new(&self.bin_path);
        cmd.env("LD_LIBRARY_PATH", &self.lib_dir)
            .arg("-m")
            .arg(model_path)
            .args(["--host", "127.0.0.1", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if let Some(mm) = mmproj_path {
            cmd.arg("--mmproj").arg(mm);
        }
        if let Some(c) = resolved.get("c") { cmd.args(["-c", c]); }
        if let Some(k) = resolved.get("cache-type-k") { cmd.args(["--cache-type-k", k]); }
        if let Some(v) = resolved.get("cache-type-v") { cmd.args(["--cache-type-v", v]); }
        if resolved.get("fa").map(|v| v == "1" || v == "on").unwrap_or(false) {
            cmd.args(["-fa", "on"]);
        }
        if let Some(ngl) = resolved.get("n-gpu-layers") { cmd.args(["-ngl", ngl]); }
        if let Some(moe) = resolved.get("n-cpu-moe") { cmd.args(["--n-cpu-moe", moe]); }
        if let Some(ub) = resolved.get("ub") { cmd.args(["-ub", ub]); }

        let mut child = cmd.spawn().map_err(|e| format!("spawn llama-server: {e}"))?;
        let deadline = Instant::now() + Duration::from_secs(60);
        let addr = format!("127.0.0.1:{port}");
        while Instant::now() < deadline {
            if let Ok(Some(st)) = child.try_wait() {
                return Err(format!("llama-server exited early: {st}"));
            }
            if let Ok((status, _)) = http_request(&addr, "GET", "/health", "application/json", &[]).await {
                if status == 200 {
                    return Ok(LlamaBridgeHandle { model_id: model_id.to_string(), port, child });
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        let _ = child.kill();
        let _ = child.wait();
        Err("llama-server health timeout".into())
    }
}

#[derive(Debug, Clone)]
pub struct MergenBridge {
    pub endpoint: String,
}

impl Default for MergenBridge {
    fn default() -> Self {
        let ep = std::env::var("MERGEN_SERVER_URL").unwrap_or_else(|_| "127.0.0.1:8088".into());
        Self { endpoint: ep.trim_start_matches("http://").to_string() }
    }
}

impl MergenBridge {
    pub async fn forward_json(&self, path: &str, body: &[u8]) -> Result<(u16, Vec<u8>), String> {
        http_request(&self.endpoint, "POST", path, "application/json", body).await
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StrataBridge {
    pub root: PathBuf,
    pub binary_present: bool,
    pub hip: HipProbe,
}

impl StrataBridge {
    pub fn probe() -> Self {
        let root = PathBuf::from("/home/mechres/Strata");
        Self {
            binary_present: root.join("engine/strata").exists(),
            root,
            hip: hip_probe(),
        }
    }
}

/// Minimal async HTTP/1.1 client over loopback `TcpStream` with 30s timeout (no TLS/reqwest bloat).
pub async fn http_request(
    addr: &str,
    method: &str,
    path: &str,
    content_type: &str,
    body: &[u8],
) -> Result<(u16, Vec<u8>), String> {
    let io_timeout = Duration::from_secs(30);
    let mut stream = tokio::time::timeout(io_timeout, TcpStream::connect(addr))
        .await
        .map_err(|_| format!("connect {addr}: timed out"))?
        .map_err(|e| format!("connect {addr}: {e}"))?;
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    tokio::time::timeout(io_timeout, async {
        stream.write_all(req.as_bytes()).await?;
        if !body.is_empty() {
            stream.write_all(body).await?;
        }
        let mut resp = Vec::new();
        stream.read_to_end(&mut resp).await?;
        Ok::<Vec<u8>, std::io::Error>(resp)
    })
    .await
    .map_err(|_| format!("http {method} {addr}{path}: timed out"))?
    .map_err(|e| e.to_string())
    .and_then(|resp| {
        let split = resp
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| "malformed HTTP response".to_string())?;
        let head = String::from_utf8_lossy(&resp[..split]);
        let status = head
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(500);
        Ok((status, resp[split + 4..].to_vec()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hip_probe_does_not_panic() {
        let p = hip_probe();
        assert!(!p.detail.is_empty());
    }
}
