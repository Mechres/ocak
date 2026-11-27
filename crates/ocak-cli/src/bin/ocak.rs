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

fn default_models_dir() -> PathBuf {
    if let Ok(d) = std::env::var("OCAK_MODELS_DIR") {
        if !d.is_empty() {
            return PathBuf::from(d);
        }
    }
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join("models"))
        .unwrap_or_else(|_| PathBuf::from("./models"))
}

#[derive(Subcommand)]
enum Cmd {
    /// Print live GPU/CPU/Vulkan/HIP hardware telemetry
    Hw,
    /// Inspect local GGUF models, .digest SHA-256 cache, and models-preset.ini
    Inspect {
        #[arg(long, env = "OCAK_MODELS_DIR")]
        models_dir: Option<PathBuf>,
        #[arg(long)]
        extra_dir: Option<PathBuf>,
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
        /// Context window in tokens (KV cache is sized to fit it)
        #[arg(long, default_value_t = 2048)]
        max_ctx: usize,
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
    /// Verify native Qwen3.5/RVN NextN MTP draft head predictions against main trunk
    MtpTest {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
    },
    /// Transcribe a WAV file natively using Qwen3-ASR (qwen3a mmproj + qwen3vl Vulkan decoder)
    Asr {
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        mmproj: Option<PathBuf>,
        #[arg(long)]
        wav: PathBuf,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
    },
    /// Long-context counting parity test ("Counting: 1, 2, ..., N," -> expects N+1) across native architectures
    CountTest {
        #[arg(long, env = "OCAK_MODELS_DIR")]
        models_dir: Option<PathBuf>,
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// Benchmark prefill and decode throughput across local native GGUF models and emit a Markdown table
    Bench {
        #[arg(long, env = "OCAK_MODELS_DIR")]
        models_dir: Option<PathBuf>,
        #[arg(long)]
        filter: Option<String>,
        #[arg(long, default_value_t = 128)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 64)]
        decode_tokens: usize,
        #[arg(long, default_value = "vk")]
        backend: String,
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
            let models_dir = models_dir.unwrap_or_else(default_models_dir);
            let presets = Presets::load(&models_dir.join("models-preset.ini"));
            let mut dirs = vec![models_dir];
            if let Some(ed) = extra_dir {
                dirs.push(ed);
            }
            let entries = scan(&dirs);
            println!("{:<34} {:<10} {:>8} {:<14} {:<10} QUANTS", "MODEL", "ARCH", "SIZE", "BACKEND", "SHA256");
            for e in entries {
                let eng = if matches!(e.arch.as_str(), "llama" | "qwen2" | "qwen3vl" | "lfm2" | "lfm3" | "qwen35" | "xing4_0" | "inkling2_loop") { "native-vk" } else { "bridge-llama" };
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
        Cmd::Run { model, prompt, max_tokens, backend, temp, max_ctx } => {
            let bk = if backend == "cpu" { BackendKind::CpuAvx2 } else { BackendKind::Vulkan };
            let t0 = Instant::now();
            let mut eng = NativeEngine::load(&model, EngineOptions { backend: bk, max_ctx, kv_pages: 128 })?;
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
            if res.timings.mtp_drafted > 0 {
                let rate = 100.0 * res.timings.mtp_accepted as f64 / res.timings.mtp_drafted as f64;
                eprintln!(
                    "prompt: {} tok (cache={}) {:.1} tok/s | decode: {} tok {:.1} tok/s | mtp: {}/{} ({:.0}%)",
                    res.timings.prompt_n, res.timings.cache_n, res.timings.prompt_per_second,
                    res.timings.predicted_n, res.timings.predicted_per_second,
                    res.timings.mtp_accepted, res.timings.mtp_drafted, rate
                );
            } else {
                eprintln!(
                    "prompt: {} tok (cache={}) {:.1} tok/s | decode: {} tok {:.1} tok/s",
                    res.timings.prompt_n, res.timings.cache_n, res.timings.prompt_per_second,
                    res.timings.predicted_n, res.timings.predicted_per_second
                );
            }
            if let Some(report) = ocak_vk::take_gpu_profile_report() {
                eprintln!("{report}");
            }
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
        Cmd::MtpTest { model, prompt } => {
            let gf = std::sync::Arc::new(ocak_gguf::GgufFile::open(&model)?);
            let mut eng = NativeEngine::load(&model, EngineOptions { backend: BackendKind::Vulkan, max_ctx: 256, kv_pages: 32 })?;
            let mut mtp = ocak_engine::mtp::MtpDrafter::try_load(gf, eng.cfg.clone(), 256)?
                .ok_or_else(|| anyhow::anyhow!("model has no Qwen3.5 NextN MTP head"))?;
            let tokens = eng.tokenizer.encode(&prompt, true, false);
            let (logits_main, h_main) = eng.eval_logits_and_hidden(&tokens)?;
            let next1 = logits_main.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
            let draft_logits = mtp.draft_logits(&h_main, next1, tokens.len() - 1)?;
            let draft2 = draft_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
            let mut ext = tokens.clone();
            ext.push(next1);
            let logits_verify = eng.eval_logits(&ext)?;
            let true2 = logits_verify.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
            println!("Prompt: {prompt:?}");
            println!("  Main trunk next token : {next1} ({:?})", eng.tokenizer.token_text(next1).unwrap_or("?"));
            println!("  MTP draft +2 token    : {draft2} ({:?})", eng.tokenizer.token_text(draft2).unwrap_or("?"));
            println!("  Main trunk +2 verify  : {true2} ({:?}) | match={}", eng.tokenizer.token_text(true2).unwrap_or("?"), draft2 == true2);
        }
        Cmd::Asr { model, mmproj, wav, max_tokens } => {
            let base = default_models_dir().join("Qwen3-ASR-1.7B");
            let model = model.unwrap_or_else(|| base.join("Qwen3-ASR-1.7B-Q8_0.gguf"));
            let mmproj = mmproj.unwrap_or_else(|| base.join("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"));
            let t0 = std::time::Instant::now();
            let mut asr = ocak_engine::asr::Qwen3AsrEngine::load(&model, &mmproj, true)?;
            eprintln!("Loaded Qwen3-ASR native (qwen3a + {}) in {} ms", asr.decoder.backend_name(), t0.elapsed().as_millis());
            let bytes = std::fs::read(&wav)?;
            let t1 = std::time::Instant::now();
            let text = asr.transcribe_wav(&bytes, max_tokens)?;
            println!("{text}");
            eprintln!("Transcribed {} bytes in {} ms", bytes.len(), t1.elapsed().as_millis());
        }
        Cmd::CountTest { models_dir, model } => {
            let models_dir = models_dir.unwrap_or_else(default_models_dir);
            let targets: Vec<PathBuf> = if let Some(m) = model {
                vec![m]
            } else {
                [
                    "Qwen3.5-4B-Q4_K_M.gguf",
                    "LFM3-1.7B-AD-Q6_K.gguf",
                    "LFM2.5-2.6B-Q4_K_M.gguf",
                    "minicpm5-2b-abliterated-Q4_K_M.gguf",
                    "Qwopus3.5-9B-coder-Exp-Q4_K_M.gguf",
                    "RVN-IQ2_S-mtp.gguf",
                    "Xing4.0-29B-A4B-IQ3_XXS.gguf",
                    "Inkling-2-Mini-UD-Q4_K_XL.gguf",
                ]
                .iter()
                .map(|f| models_dir.join(f))
                .collect()
            };
            let make_prompt = |n: usize| -> String {
                let nums: Vec<String> = (1..=n).map(|i| i.to_string()).collect();
                format!("Counting: {},", nums.join(", "))
            };
            println!("{:<36} {:<16} {:<16}", "MODEL", "N=20 (exp 21)", "N=120 (exp 121)");
            let mut failed = 0usize;
            for path in targets {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
                if !path.exists() {
                    println!("{:<36} {:<16} {:<16}", name, "SKIP", "SKIP");
                    continue;
                }
                let mut eng = NativeEngine::load(
                    &path,
                    EngineOptions { backend: BackendKind::Vulkan, max_ctx: 1024, kv_pages: 128 },
                )?;
                let mut row = Vec::new();
                for n in [20usize, 120usize] {
                    let prompt = make_prompt(n);
                    let exp = (n + 1).to_string();
                    let gr = eng.generate(
                        &GenRequest {
                            prompt,
                            parse_special: false,
                            max_tokens: 6,
                            sampler: SamplerParams { temperature: 0.0, ..Default::default() },
                            stop: Vec::new(),
                            json_mode: false,
                            thinking: None,
                        },
                        |_, _| true,
                    )?;
                    let got = gr.text.trim_start().to_string();
                    let ok = got.starts_with(&exp);
                    if !ok {
                        failed += 1;
                    }
                    let preview: String = got.replace('\n', "\\n").chars().take(10).collect();
                    row.push(format!("{} ({preview:?})", if ok { "PASS" } else { "FAIL" }));
                }
                println!("{:<36} {:<16} {:<16}", name, row[0], row[1]);
                drop(eng);
            }
            if failed > 0 {
                anyhow::bail!("{failed} counting parity check(s) failed");
            }
        }
        Cmd::Bench { models_dir, filter, prompt_tokens, decode_tokens, backend } => {
            let models_dir = models_dir.unwrap_or_else(default_models_dir);
            let bk = if backend == "cpu" { BackendKind::CpuAvx2 } else { BackendKind::Vulkan };
            let mut entries: Vec<_> = scan(&[models_dir])
                .into_iter()
                .filter(|e| {
                    !e.is_mmproj
                        && matches!(
                            e.arch.as_str(),
                            "llama" | "qwen2" | "lfm2" | "lfm3" | "qwen35" | "xing4_0" | "inkling2_loop"
                        )
                        && filter
                            .as_deref()
                            .is_none_or(|f| e.id.to_lowercase().contains(&f.to_lowercase()))
                })
                .collect();
            // The scan follows symlinks (e.g. a stages/ directory linking to the same files):
            // bench each underlying file once.
            let mut seen = std::collections::HashSet::new();
            entries.retain(|e| seen.insert(std::fs::canonicalize(&e.path).unwrap_or_else(|_| e.path.clone())));
            entries.sort_by_key(|e| e.size_bytes);
            if entries.is_empty() {
                anyhow::bail!("no matching native GGUF models found");
            }
            // Deterministic prompt (~prompt_tokens tokens)
            let count_n = (prompt_tokens / 3).max(16);
            let nums: Vec<String> = (1..=count_n).map(|i| i.to_string()).collect();
            let prompt = format!("Counting: {},", nums.join(", "));

            println!("| Model | Arch | Size | Load (ms) | Prefill (tok/s) | Decode (tok/s) | MTP |");
            println!("|---|---|---:|---:|---:|---:|---|");
            let _ = std::io::stdout().flush();

            for e in entries {
                let t0 = Instant::now();
                let kv_pages = if e.size_bytes > 8 * 1024 * 1024 * 1024 { 128 } else { 256 };
                let eng_res = NativeEngine::load(
                    &e.path,
                    EngineOptions { backend: bk, max_ctx: 2048, kv_pages },
                );
                let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let mut eng = match eng_res {
                    Ok(eng) => eng,
                    Err(err) => {
                        println!(
                            "| {} | {} | {:.2} GiB | ERR | - | - | {} |",
                            e.id,
                            e.arch,
                            e.size_bytes as f64 / (1u64 << 30) as f64,
                            err
                        );
                        continue;
                    }
                };
                let gr = eng.generate(
                    &GenRequest {
                        prompt: prompt.clone(),
                        parse_special: false,
                        max_tokens: decode_tokens,
                        sampler: SamplerParams::greedy(),
                        stop: Vec::new(),
                        json_mode: false,
                        thinking: None,
                    },
                    |_, _| true,
                )?;
                let mtp_col = if gr.timings.mtp_drafted > 0 {
                    format!(
                        "{}/{} ({:.0}%)",
                        gr.timings.mtp_accepted,
                        gr.timings.mtp_drafted,
                        100.0 * gr.timings.mtp_accepted as f64 / gr.timings.mtp_drafted as f64
                    )
                } else {
                    "-".to_string()
                };
                println!(
                    "| {} | {} | {:.2} GiB | {:.0} | {:.1} | {:.1} | {} |",
                    e.id,
                    e.arch,
                    e.size_bytes as f64 / (1u64 << 30) as f64,
                    load_ms,
                    gr.timings.prompt_per_second,
                    gr.timings.predicted_per_second,
                    mtp_col
                );
                let _ = std::io::stdout().flush();
                drop(eng);
                if let Some(report) = ocak_vk::take_gpu_profile_report() {
                    eprintln!("{report}");
                }
            }
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
                            println!("\n{:<34} {:<10} {:<14} STATE", "MODEL", "ARCH", "ENGINE");
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
