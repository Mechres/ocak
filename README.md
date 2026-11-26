# Ocak

**Ocak** (*hearth / forge*) is a ground-up Rust LLM & ASR inference engine, GPU VRAM/power orchestrator, and multi-protocol API server built directly on **Vulkan 1.3/1.4 Compute (`ash` + 53 custom GLSL/SPIR-V shaders with Wave32 `subgroupClusteredAdd`, 1-submit-per-token full-GPU pipelines, & GTT sparse paging)** and **x86-64 AVX2/FMA** (plus an optional runtime-loaded `ocak-hip` HSACO launcher) — with zero CUDA or ROCm SDK required to build or run.

Designed and verified on a consumer Linux workstation (**AMD Ryzen 5 5600 + Radeon RX 6700 XT 12 GiB VRAM, Mesa RADV Vulkan 1.4**), Ocak runs dense, hybrid-recurrent, looped MoE, and hybrid GPU+CPU Mixture-of-Experts GGUF models natively from 1B up to 29B parameters while automatically curing AMD RDNA 2's 54W idle compute clock-lock bug.

---

## Highlights

- **Ground-up GPU compute engine (`ocak-vk`)**: Raw `ash` Vulkan 1.3/1.4 compute pipeline with 53 hand-written GLSL shaders compiled at build time via `glslc` — zero graphics/render middleware, hardware `subgroupClusteredAdd(32)` reductions, `HOST_VISIBLE` GTT sparse paging for >12 GiB MoE models, single `vkQueueSubmit` forward pass per token across dense, hybrid-recurrent, and MoE architectures, and full `VkDevice` teardown on idle sleep.
- **Zero-copy GGUF v2/v3 & 20 quantization formats (`ocak-gguf`, `ocak-cpu`, `ocak-vk`)**: Direct `mmap` loading with stat-verified `.digest` SHA-256 caching and native `models-preset.ini` parsing.
  - Supported formats: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `NVFP4`.
- **Native architectures (`ocak-engine`)**:
  - **Llama / MiniCPM / Qwen2** (dense GQA + NeoX / adjacent RoPE)
  - **Liquid LFM2 / LFM2.5** (causal `ShortConv` + QK-norm GQA)
  - **Liquid LFM3** (`ShortConv` + per-head sigmoid attention gate + sliding-window attention + `rope_freqs` + post-attention/FFW norms)
  - **Qwen3.5 / Qwen3.8 / RVN** (3:1 hybrid **Gated DeltaNet** linear recurrence + **Gated Full Attention** + partial NeoX RoPE + native **NextN MTP** draft head `MtpDrafter`)
  - **Xing4.0 MoE (29B-A4B)** (4-stream Sinkhorn **Hyper-Connections** + **YaRN MLA** latent KV cache + 64-expert / 4-active MoE + shared expert with fused GPU MoE dispatch, parallel MLA, and automatic hybrid GPU+CPU expert offloading `n_cpu_moe`)
  - **Inkling-2-Mini (26.4B Looped MoE)** (24 physical layers with middle 12 looped 3× = 48 logical layers, depth-wise Attention Residuals, cross-pass gain/forget controllers, dynamic `ShortConv`, relative-position bias + attention sinks, GPU `situ.comp` activation, 128-expert / 8-active `NVFP4` MoE on VRAM + GTT sparse paging)
  - **Qwen3-ASR-1.7B (Native Audio)** (zero-dep WAV/16 kHz resampler + 128-bin Slaney log-mel spectrogram + `qwen3a` 3×Conv2D + 24-layer Transformer audio encoder + `qwen3vl` Vulkan decoder for `/v1/audio/transcriptions` and `ocak asr`)
- **Radix Prefix Paged KV Cache & Q8 Compression (`ocak-kv`)**: 16-token physical KV pages organized in a Copy-on-Write Radix prefix tree with LRU zero-ref leaf eviction and `CompressedKvStore` (3.55× Q8 host-RAM spill) for instant multi-turn prefix reuse.
- **RX 6700 XT 12 GiB VRAM & `WarmSleep` Power Governor (`ocak-sysfs`)**: Auto-discovers `amdgpu` DRM/hwmon sensors (VRAM, GTT, wattage, temperature, busy %) and destroys active Vulkan compute contexts after `sleep-idle-seconds`, dropping GPU idle draw back to desktop idle (~29 W) while waking from Linux page cache in ~300 ms.
- **Multi-Protocol Server (`ocak-server`)**:
  - **OpenAI API**: `GET /v1/models`, `POST /v1/chat/completions` (sync + streaming SSE + JSON grammar mode + `<think>` token budget), `POST /v1/completions`, `POST /v1/embeddings` (L2-normalized), `POST /v1/audio/transcriptions`
  - **Anthropic Messages API**: `POST /v1/messages` (streaming SSE `message_start` / `content_block_delta` / `message_stop`) & `POST /v1/messages/count_tokens` — compatible with Claude Code (`ANTHROPIC_BASE_URL=http://127.0.0.1:8090`)
  - **Observability & UI**: Prometheus `GET /metrics`, live SSE `GET /v1/ocak/events`, embedded dark-mode Web dashboard at `GET /ui`, and zero-dependency ANSI terminal monitor (`ocak top`).
- **Tiered Federation & Parity Harness (`ocak-federation`)**: Optional `libllama.so` C logit parity oracle (`ocak parity-test`), managed `llama-server` fallback bridge (for exotic/multimodal audio models like `Qwen3-ASR-1.7B`), `mergen` distributed fleet bridge, and runtime `dlopen` HIP probe.

---

## Architecture

```mermaid
flowchart TD
    Clients["Clients (OpenAI SDK, Claude Code, Yagent, ocak top, Web /ui)"] --> Server["ocak-server (axum HTTP/SSE)"]
    Server --> Gov["ocak-sysfs (amdgpu VRAM/GTT/Wattage + WarmSleep Governor)"]
    Server --> Engine["ocak-engine (Llama, Qwen2, LFM2/3, Qwen3.5/3.8, Xing4.0 MoE)"]
    Server --> Fed["ocak-federation (libllama parity/ASR bridge, Mergen fleet, HIP probe)"]
    Engine --> KV["ocak-kv (16-tok Paged Radix Prefix Tree)"]
    Engine --> Tok["ocak-tok (SentencePiece, Byte-BPE, Templates, Thinking Cap, JSON Grammar)"]
    Engine --> VK["ocak-vk (Vulkan 1.4 Compute — 53 GLSL/SPIR-V Shaders)"]
    Engine --> HIP["ocak-hip (Dynamic dlopen libamdhip64 HSACO Runtime)"]
    Engine --> CPU["ocak-cpu (x86-64 AVX2/FMA + Hybrid CPU MoE)"]
    Engine --> GGUF["ocak-gguf (Zero-Copy mmap GGUF v2/v3, Presets, .digest)"]
```

---

## Verified Benchmarks (Ryzen 5 5600 + RX 6700 XT, Mesa RADV Vulkan 1.4)

Measured natively on `ocak-vk` and cross-checked against `libllama.so` via `ocak parity-test`:

| Model | Architecture | Size | Vulkan vs CPU `cos_sim` | Vulkan vs `libllama.so` `cos_sim` | Native Decode |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `tinyllama-1.1b-chat-v1.0.Q4_0` | `llama` | 0.59 GB | `1.000000` | `0.999894` | **184.1 tok/s** |
| `tinyllama-1.1b-chat-v1.0.Q4_K_M` | `llama` | 0.62 GB | `1.000000` | `0.999802` | **143.7 tok/s** |
| `LFM3-1.7B-AD-Q6_K` | `lfm3` (ShortConv + Gated SWA) | 1.48 GB | `1.000000` | *native-only* | **73.3 tok/s** |
| `minicpm5-2b-abliterated-Q4_K_M` | `llama` | 1.45 GB | `1.000000` | `0.999706` | **62.6 tok/s** |
| `LFM2.5-2.6B-Q4_K_M` | `lfm2` (ShortConv + GQA) | 1.56 GB | `1.000000` | `0.999716` (1-tok) | **52.3 tok/s** |
| `Qwen3.5-4B-Q4_K_M` | `qwen35` (Gated DeltaNet + Attn) | 2.55 GB | `1.000000` | `0.999335` | **34.8 tok/s** |
| `Qwopus3.5-9B-coder-Exp-Q4_K_M` | `qwen35` (9B Hybrid) | 5.24 GB | `1.000000` | `0.999584` | **19.9 tok/s** |
| `Qwen3.8-27B-UD-IQ2_S` | `qwen35` (27B Hybrid, `IQ1_S`/`IQ1_M`/`IQ2_XXS`/`IQ2_S`) | 7.80 GB | — | greedy verified | **7.8 tok/s** |
| `RVN-IQ2_S-mtp` | `qwen35` (27B Hybrid + NextN MTP) | 9.14 GB | — | greedy + MTP 7/7 (100%) | **9.7 tok/s** |
| `Xing4.0-29B-A4B-IQ3_XXS` | `xing4_0` (29B MoE, 1-submit batched 2D GPU VRAM+GTT) | 10.79 GB | — | greedy verified | **26.7 tok/s** |
| `Inkling-2-Mini-UD-Q4_K_XL` | `inkling2_loop` (26.4B 3× Looped `NVFP4`/`Q8_0` MoE, core-VRAM priority) | 16.42 GB | — | greedy verified | **17.5 tok/s** |
| `Qwen3-ASR-1.7B` | `qwen3a` + `qwen3vl` (Native Audio) | 2.35 GB | — | end-to-end WAV verified | **1.49 s / clip** |

- **Native NextN MTP draft verification (`ocak mtp-test`)**: Verified on 27B `RVN-IQ2_S-mtp` (`"The capital of France is"` -> trunk `+1`: `" Paris"`, MTP draft `+2`: `"."`, trunk verify `+2`: `"."`, `match=true`).
- **1-submit-per-token batched 2D full-GPU MoE (`XingExec` & `InklingExec`)**: Zero per-layer CPU↔GPU fence waits, 2D `top_k`-batched MoE dispatches, deferred parallel barriers, and 3×-looped-core VRAM priority — `Xing4.0-29B-A4B` reaches **26.7 tok/s** decode / **26.6 tok/s** prefill (15.7× over initial hybrid path), and `Inkling-2-Mini` reaches **17.5 tok/s** decode / **17.4 tok/s** prefill (17.5× over initial path, outpacing `llama.cpp` on the same 12 GiB RX 6700 XT).
- **`WarmSleep` idle power check**: Active VRAM `1.79 GiB` -> `1.07 GiB` on teardown, RX 6700 XT power returns to **`29.0 W`** idle, automatic wake on next request in **`307 ms`**.
- **Radix KV prefix cache**: Verified 100% prefix hit (`cache_n=48` on 48-token conversation prefix).

---

## Prerequisites & Build

- **Rust** (`cargo`, `rustc` 1.80+)
- **Vulkan 1.3+ runtime & driver** (e.g., Mesa RADV on AMD, NVIDIA Vulkan, Intel ANV)
- **Shader compiler**: `glslc` (`shaderc` package on Arch/Ubuntu/Fedora)

```bash
git clone https://github.com/Mechres/ocak.git
cd ocak
cargo build --release --workspace
```

Run the full test suite (including Vulkan GPU shader parity tests):

```bash
cargo test --release --workspace
```

---

## Quickstart

### 1. Inspect Hardware & Model Catalog
```bash
./target/release/ocak hw
./target/release/ocak inspect --models-dir ~/models
```

### 2. Run Native Inference Directly from CLI
```bash
./target/release/ocak run \
  --model ~/models/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt "<|im_start|>user\nExplain paged attention in two sentences.<|im_end|>\n<|im_start|>assistant\n" \
  --max-tokens 96
```

### 3. Start the Server (`ocakd`) + Web Dashboard & TUI
```bash
./target/release/ocakd \
  --host 127.0.0.1 \
  --port 8090 \
  --compat-port 11434 \
  --models-dir ~/models \
  --sleep-idle-seconds 120
```

- **Web Dashboard**: `http://127.0.0.1:8090/ui`
- **Live Terminal Monitor**: `./target/release/ocak top --addr 127.0.0.1:8090`
- **Claude Code / Anthropic clients**:
  ```bash
  ANTHROPIC_BASE_URL=http://127.0.0.1:8090 ANTHROPIC_API_KEY=local claude
  ```
- **Systemd user service**:
  ```bash
  mkdir -p ~/.config/systemd/user
  cp packaging/ocakd.service ~/.config/systemd/user/
  systemctl --user daemon-reload && systemctl --user enable --now ocakd.service
  ```

---

## Workspace Layout

| Crate | Purpose |
| :--- | :--- |
| `crates/ocak-sysfs` | Linux `amdgpu` DRM/hwmon telemetry + 12 GiB VRAM/power governor |
| `crates/ocak-gguf` | Zero-copy `mmap` GGUF v2/v3 parser, quant reference math, IQ lookup tables, presets, `.digest` |
| `crates/ocak-kv` | 16-token paged KV allocator + Copy-on-Write Radix prefix tree |
| `crates/ocak-tok` | SentencePiece & byte-level BPE tokenizers, chat templates, `<think>` budget, JSON grammar |
| `crates/ocak-cpu` | x86-64 AVX2/FMA quantized GEMV and hybrid transformer ops |
| `crates/ocak-vk` | Raw `ash` Vulkan 1.4 compute runtime + 53 GLSL/SPIR-V shaders (20 weight quant formats) |
| `crates/ocak-hip` | Dynamic `dlopen("libamdhip64.so")` runtime + HSACO module/kernel launcher |
| `crates/ocak-engine` | Native executors (`VkExec`, `CpuExec`, `XingExec`, `InklingExec`), `MtpDrafter`, sampler & generator |
| `crates/ocak-federation` | `libllama.so` parity oracle, managed bridge, `mergen` fleet bridge, HIP `dlopen` probe |
| `crates/ocak-server` | `axum` OpenAI + Anthropic + Embeddings + Audio + Prometheus + embedded `/ui` |
| `crates/ocak-cli` | `ocakd` daemon & `ocak` CLI (`hw`, `inspect`, `run`, `parity-test`, `mtp-test`, `top`) |

---

## Roadmap

- [x] **M1–M3**: Ground-up GGUF/tokenizer/Radix KV + Q8 spill store, Vulkan 1.4 + AVX2 Llama/Qwen2 engine, OpenAI + Anthropic + Embeddings + Web UI/TUI + RX 6700 XT `WarmSleep`.
- [x] **M4 / M4b**: Native `LFM2`, `LFM3`, `Qwen3.5`/`Qwen3.8` Gated DeltaNet & Gated Attention, `Q2_K`/`Q3_K` & all `IQ1_S`/`IQ1_M`/`IQ2_XXS`/`IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S`/`IQ4_NL`/`IQ4_XS` shaders.
- [x] **M5 / M5b**: Native hybrid GPU+CPU MoE for `Xing4.0-29B-A4B` (fused GPU MoE + parallel MLA) and `Inkling-2-Mini` (`inkling2_loop` 3× looped MoE + `NVFP4` & `BF16` shaders).
- [x] **M6**: Native `Qwen3.5`/`RVN` NextN MTP draft head (`MtpDrafter` wired into `generate()` + `ocak mtp-test`, 100% draft match), native `Qwen3-ASR-1.7B` audio frontend (`qwen3a` + `qwen3vl` on Vulkan for `/v1/audio/transcriptions` and `ocak asr`), and dynamic `ocak-hip` HSACO runtime (`dlopen("libamdhip64.so")`).
- [ ] **Future**: 2-token batched MTP verify pass, tiled `qmm` prefill shaders, and continuous multi-slot request batching.

---

## License

MIT — see [LICENSE](LICENSE).
