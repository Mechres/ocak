use anyhow::Result;
use clap::{Parser, Subcommand};
use ocak_engine::{BackendKind, EngineOptions, GenRequest, NativeEngine, SamplerParams};
use ocak_federation::{cosine_similarity, eval_with_llama_oracle, hip_probe, StrataBridge};
use ocak_gguf::catalog::scan;
use ocak_gguf::preset::Presets;
use ocak_sysfs::{cpu_info, discover_amd_gpu, host_memory};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "ocak", about = "Ocak ground-up LLM CLI & TUI")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print live GPU/CPU/Vulkan/HIP hardware telemetry
    Hw,
    /// Inspect local GGUF models, .digest SHA-256 cache, and models-preset.ini
    Inspect {
        #[arg(long, default_value = "/home/mechres/models")]
        models_dir: PathBuf,
        #[arg(long, default_value = "/home/mechres/Projeler/mergen/models")]
        extra_dir: PathBuf,
    },
    /// Run native ground-up generation directly on Vulkan or CPU AVX2
    Run {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long, default_value = "vk")]
        backend: String,
        #[arg(long, default_value_t = 0.0)]
        temp: f32,
    },
    /// Compare native Vulkan & CPU AVX2 logits against libllama.so C oracle
    ParityTest {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 16)]
        steps: usize,
    },
    /// Zero-dependency ANSI live dashboard polling ocakd
    Top {
        #[arg(long, default_value = "127.0.0.1:8090")]
        addr: String,
    },
}

fn top_k_str(logits: &[f32], tok: &ocak_tok::Tokenizer, k: usize) -> String {
    let mut idx: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    idx.truncate(k);
    idx.into_iter()
        .map(|(i, v)| format!("{i}({:?}:{v:.2})", tok.token_text(i as u32).unwrap_or("?")))
        .collect::<Vec<_>>()
        .join(", ")
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Hw => {
            let cpu = cpu_info();
            let mem = host_memory();
            println!("CPU: {} logical CPUs | AVX2={} FMA={} F16C={}", cpu.logical_cpus, cpu.avx2, cpu.fma, cpu.f16c);
            println!("RAM: {:.2} GiB available / {:.2} GiB total", mem.available_bytes as f64 / (1u64 << 30) as f64, mem.total_bytes as f64 / (1u64 << 30) as f64);
            if let Some(g) = discover_amd_gpu() {
                let t = g.telemetry();
                println!("AMD GPU ({} {:04x}:{:04x}):", g.card, g.vendor_id, g.device_id);
                println!("  VRAM: {:.2} / {:.2} GiB", t.vram_used_bytes as f64 / (1u64 << 30) as f64, t.vram_total_bytes as f64 / (1u64 << 30) as f64);
                println!("  GTT : {:.2} / {:.2} GiB", t.gtt_used_bytes as f64 / (1u64 << 30) as f64, t.gtt_total_bytes as f64 / (1u64 << 30) as f64);
                println!("  Power: {:.1} W | Temp: {:.1} °C | Busy: {}%", t.power_watts, t.temp_celsius, t.gpu_busy_percent);
            }
            match ocak_vk::VkContext::new() {
                Ok(vk) => println!("Vulkan 1.3+: {} (subgroup={}, vram={:.2} GiB)", vk.device_name(), vk.subgroup_size(), vk.vram_bytes() as f64 / (1u64 << 30) as f64),
                Err(e) => println!("Vulkan: unavailable ({e})"),
            }
            let hip = hip_probe();
            println!("HIP runtime: available={} ({})", hip.available, hip.detail);
            let strata = StrataBridge::probe();
            println!("Strata: binary_present={} ({})", strata.binary_present, strata.root.display());
        }
        Cmd::Inspect { models_dir, extra_dir } => {
            let presets = Presets::load(&models_dir.join("models-preset.ini"));
            let entries = scan(&[models_dir, extra_dir]);
            println!("{:<34} {:<10} {:>8} {:<14} {:<10} {}", "MODEL", "ARCH", "SIZE", "BACKEND", "SHA256", "QUANTS");
            for e in entries {
                let eng = if matches!(e.arch.as_str(), "llama" | "qwen2" | "lfm2" | "lfm3" | "qwen35" | "xing4_0") && !["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_S"].iter().any(|u| e.quant_summary.contains(u)) { "native-vk" } else { "bridge-llama" };
                let sha = e.sha256.as_deref().map(|s| &s[..8]).unwrap_or("none");
                let has_p = if presets.models.contains_key(&e.id) { "*" } else { " " };
                println!(
                    "{}{:<33} {:<10} {:>6.2}GB {:<14} {:<10} {}",
                    has_p,
                    e.id,
                    e.arch,
                    e.size_bytes as f64 / (1u64 << 30) as f64,
                    eng,
                    sha,
                    e.quant_summary
                );
            }
        }
        Cmd::Run { model, prompt, max_tokens, backend, temp } => {
            let bk = if backend == "cpu" { BackendKind::CpuAvx2 } else { BackendKind::Vulkan };
            let t0 = Instant::now();
            let mut eng = NativeEngine::load(&model, EngineOptions { backend: bk, max_ctx: 2048, kv_pages: 128 })?;
            eprintln!("Loaded {} on {} in {:.0} ms", model.display(), eng.backend_name(), t0.elapsed().as_secs_f64() * 1000.0);
            let req = GenRequest {
                prompt,
                parse_special: true,
                max_tokens,
                sampler: SamplerParams { temperature: temp, ..Default::default() },
                stop: Vec::new(),
                json_mode: false,
                thinking: None,
            };
            let res = eng.generate(&req, |_, piece| {
                print!("{piece}");
                let _ = std::io::stdout().flush();
                true
            })?;
            println!();
            eprintln!(
                "prompt: {} tok (cache={}) {:.1} tok/s | decode: {} tok {:.1} tok/s",
                res.timings.prompt_n, res.timings.cache_n, res.timings.prompt_per_second,
                res.timings.predicted_n, res.timings.predicted_per_second
            );
        }
        Cmd::ParityTest { model, prompt, steps } => {
            let mut eng_vk = NativeEngine::load(&model, EngineOptions { backend: BackendKind::Vulkan, max_ctx: 512, kv_pages: 64 })?;
            let mut eng_cpu = NativeEngine::load(&model, EngineOptions { backend: BackendKind::CpuAvx2, max_ctx: 512, kv_pages: 64 })?;
            let tokens = eng_vk.tokenizer.encode(&prompt, true, false);
            println!("Prompt: {:?} -> {} tokens: {:?}", prompt, tokens.len(), tokens);

            let l_vk = eng_vk.eval_logits(&tokens)?;
            let l_cpu = eng_cpu.eval_logits(&tokens)?;
            let cos_vk_cpu = cosine_similarity(&l_vk, &l_cpu);
            println!("Vulkan vs CPU AVX2 cosine_sim = {cos_vk_cpu:.6}");
            println!("  VK  top-3: {}", top_k_str(&l_vk, &eng_vk.tokenizer, 3));
            println!("  CPU top-3: {}", top_k_str(&l_cpu, &eng_vk.tokenizer, 3));
            anyhow::ensure!(cos_vk_cpu > 0.999, "VK vs CPU cosine similarity too low: {cos_vk_cpu}");

            match eval_with_llama_oracle(&model, &tokens, eng_vk.cfg.n_vocab) {
                Ok(l_ref) => {
                    let cos_vk_ref = cosine_similarity(&l_vk, &l_ref);
                    let cos_cpu_ref = cosine_similarity(&l_cpu, &l_ref);
                    println!("Vulkan vs libllama.so cosine_sim = {cos_vk_ref:.6}");
                    println!("CPU    vs libllama.so cosine_sim = {cos_cpu_ref:.6}");
                    println!("  REF top-3: {}", top_k_str(&l_ref, &eng_vk.tokenizer, 3));
                    anyhow::ensure!(cos_vk_ref > 0.999, "VK vs libllama cosine similarity too low: {cos_vk_ref}");
                }
                Err(e) => println!("libllama oracle skipped: {e}"),
            }

            // Greedy multi-step decode benchmark on RX 6700 XT
            let res = eng_vk.generate(
                &GenRequest {
                    prompt,
                    parse_special: false,
                    max_tokens: steps,
                    sampler: SamplerParams::greedy(),
                    stop: Vec::new(),
                    json_mode: false,
                    thinking: None,
                },
                |_, _| true,
            )?;
            println!(
                "Greedy {}-step decode on {}: {:?}\n  prefill: {:.1} tok/s | decode: {:.1} tok/s",
                steps, eng_vk.backend_name(), res.text, res.timings.prompt_per_second, res.timings.predicted_per_second
            );
        }
        Cmd::Top { addr } => {
            let clean = addr.trim_start_matches("http://").to_string();
            loop {
                print!("\x1b[2J\x1b[H");
                match ocak_federation::bridges::http_request(&clean, "GET", "/v1/ocak/status", "application/json", &[]).await {
                    Ok((200, body)) => {
                        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body) {
                            println!("=== OCAK TOP ({clean}) ===  [Ctrl+C to exit]");
                            if let Some(g) = v.get("gpu").and_then(|x| x.as_object()) {
                                let vu = g["vram_used_bytes"].as_f64().unwrap_or(0.0) / (1u64 << 30) as f64;
                                let vt = g["vram_total_bytes"].as_f64().unwrap_or(1.0) / (1u64 << 30) as f64;
                                let gu = g["gtt_used_bytes"].as_f64().unwrap_or(0.0) / (1u64 << 30) as f64;
                                let gt = g["gtt_total_bytes"].as_f64().unwrap_or(1.0) / (1u64 << 30) as f64;
                                println!("VRAM: {vu:.2}/{vt:.2} GiB | GTT: {gu:.2}/{gt:.2} GiB | Power: {:.1}W | Temp: {:.0}C | Busy: {}%",
                                    g["power_watts"].as_f64().unwrap_or(0.0),
                                    g["temp_celsius"].as_f64().unwrap_or(0.0),
                                    g["gpu_busy_percent"].as_u64().unwrap_or(0));
                            }
                            println!("Tokens: {} | Requests: {} | Radix KV hits: {}",
                                v["tokens_generated_total"], v["requests_total"], v["kv_radix_hits_total"]);
                            println!("\n{:<34} {:<10} {:<14} {}", "MODEL", "ARCH", "ENGINE", "STATE");
                            if let Some(ms) = v.get("models").and_then(|x| x.as_array()) {
                                for m in ms {
                                    println!("{:<34} {:<10} {:<14} {}",
                                        m["id"].as_str().unwrap_or(""),
                                        m["arch"].as_str().unwrap_or(""),
                                        m["engine"].as_str().unwrap_or(""),
                                        m["residency"].as_str().unwrap_or(""));
                                }
                            }
                        }
                    }
                    Err(e) => println!("Waiting for ocakd at {clean}: {e}"),
                    Ok((c, _)) => println!("HTTP {c} from {clean}"),
                }
                let _ = std::io::stdout().flush();
                tokio::time::sleep(std::time::Duration::from_millis(750)).await;
            }
        }
    }
    Ok(())
}
