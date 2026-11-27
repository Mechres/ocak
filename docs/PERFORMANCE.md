# Performance notes

Working log of GPU performance work on ocak (RX 6700 XT / RDNA2, Mesa RADV). It records what was
measured, what helped, and — just as important — what was tried and **did not** help, so the same
experiments are not repeated. Keep it up to date when you try something new: add a line under
"Tried and rejected" even (especially) when an idea fails.

## How to measure

- **Per-kernel GPU time:** `OCAK_GPU_PROFILE=1 ocak run ...` or `... ocak bench ...` prints a
  table of pipelines by total GPU time (dispatches are serialized for timing, so per-call numbers
  are exact but the total can exceed wall time when work normally overlaps).
- **End-to-end:** `ocak bench --prompt-tokens 512 --decode-tokens 64` (prefill / decode tok/s per
  model), or `ocak run --model M --prompt P --max-tokens 128 --temp 0` (prints prompt / decode
  tok/s). Run twice: the GPU's clock / thermal state moves results by ~5-10% between runs.
- **Never benchmark while the GPU CI job is running** (`gh run list --workflow gpu.yml`): the
  self-hosted runner shares the GPU and both measurements become garbage. After pushing, wait for
  the run (`gh run watch`) before measuring again.
- **Shader statistics / ISA:** `MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaders,shaderstats`
  (VGPRs, "Subgroups per SIMD" = occupancy, instruction counts, full disassembly). Counting the
  instructions in a hot loop explained more than any amount of guessing (see "decode is
  instruction-bound" below).
- **Micro-benchmarks:** `cargo test --release -p ocak-vk vk_qmm_tiled_bench -- --ignored --nocapture`
  (2560x9216x256 matmul per format), `vk_qmm_grouped_bench`, `vk_small_batch_bench`.
- **Correctness bar for kernel changes:** `cargo test --release --workspace` (GPU kernels vs
  reference kernels / CPU), `ocak count-test` (long-context counting, 8 models), greedy output
  vs the previous build on a fixed prompt, and for numerics changes `ocak parity-test` (logit
  cosine vs libllama). int8 activations make greedy output differ in wording on some models;
  that is expected — check coherence and parity cosine, not bit equality.

## Current design (what is fast and why)

### Prefill (batched)
- **int8 dot-product matmul** (`qmm_q8.comp`): activations quantized to int8 blocks of 32
  (`quant_q8.comp`, reused across matmuls on the same input), weights unpacked to int8
  (`dequant_i8.glsl`), `dotPacked4x8EXT`. ~1.8x the fp16-tile kernel. Instruction-issue bound:
  ~620 instructions per 32-wide K block, 256 of them dot4, ~128 the float scale fix-up.
- Scales live in **scalar shared arrays**; formats without a block min skip the min term.
- **Split-K** (`sum_splits.comp`) whenever the grid has < 40-80 workgroups (small-m matrices
  like 24/48-row projections or 1024-row KV projections).
- **Grouped (MoE) int8** with 64-column tiles; MoE prefill chunks sized to give experts many
  tokens per weight pass (Inkling: 512-2048 tokens chosen from free VRAM; Xing 1024).
- **Flash-style attention** (`attention_prefill.comp`, `inkling_attn_prefill.comp`,
  `mla_attn_prefill.comp`): K/V tiles staged once per workgroup for many query rows,
  **padded shared rows** (64-word rows put every lane on one bank), register prefetch of the next
  tile, 2 query rows per thread; QK uses `OpFDot2MixAcc32VALVE` (fp16 dot, fp32 accumulate) when
  `VK_VALVE_shader_mixed_float_dot_product` is available.
- Recurrent / conv layers: conv history in registers; Inkling conv token-parallel; DeltaNet
  stages q/k/v + per-token scalars in shared memory with 8-lane rows and two reductions per step
  (decode keeps the old 32-lane kernel, faster for 1-4 tokens).

### Decode (1-8 tokens)
- **int8 matvec** (`qmv_q8.comp`) for matrices >= 1M weights (not Q8_0): block-level unpack
  (`dqi32`, one 32-weight block per lane: one address / scale computation, a few wide loads).
  Decode was **instruction-bound** (~85 instructions per 8 weights before), not bandwidth-bound.
- IQ lookup tables staged in shared memory (16 rows per workgroup); IQ signs applied packed
  (grid bytes are nonzero, so a byte-wise negate never carries).
- Parallel MoE top-k (rank = number of higher-scored experts), two-pass argmax, 512/1024-thread
  norms with subgroup reductions.

## What worked (with numbers)

| Change | Effect |
|---|---|
| int8 prefill matmul | Q4_K micro-bench 1.36 -> 0.75 ms; dense prefill +50-60% |
| Register prefetch in int8 matmul | 1.36 -> 0.77 ms (was 1.12 ms without prefetch) |
| Scalar scale arrays + skip min term | Q6_K 0.98 -> 0.73 ms, Q8_0 0.72 -> 0.63 ms |
| Q2_K sub-scale folded into int8 weights (q*sc <= 45) | Q2_K matmul -6% |
| Split-K (fp and int8) | Xing 24-row F16 matmul 1437 -> 149 us; Qwen3.8 48-row projections |
| Flash attention + padded rows + prefetch | attention per call 2-2.3x faster |
| VALVE mixed fp16 dot for QK | attention -5-10% |
| MoE chunk size (Inkling VRAM-sized, Xing 1024) + 64-col grouped tiles | Inkling prefill ~2-3x on long prompts |
| int8 decode matvec + block-level unpack | Qwen3.8 IQ2_S decode 15 -> 35 tok/s |
| IQ1 grid in shared memory | IQ1_S matvec 258 -> 121 us |
| Parallel MoE top-k | 235 -> 8 us per call (Inkling decode +40% with the int8 work) |
| Two-pass argmax, wide norms | argmax 252 -> 5 us, add_rms_norm 11 -> 3.7 us |

## Tried and rejected (do not repeat without a new idea)

| Idea | Result | Why (as far as known) |
|---|---|---|
| Register prefetch in the **fp16** tiled matmul | slower (1.41 -> 1.49 ms) | fp16 kernel is compute-bound, prefetch only adds registers |
| int8 matmul with **TM = 8** (128-row tiles) | slower (Q4_K 0.73 -> 0.96 ms) | VGPR pressure / occupancy |
| int8 matmul **TN = 8 (128-col) grouped tiles** | slower than 64-col | too many empty columns per expert |
| Grouped int8 for per-16-scale formats *before* the scalar-scale change | slower than fp16 | fix-up cost; fine after the scale rework |
| Removing the fix-up entirely (experiment) | 0.75 -> 0.54 ms | shows the fix-up is ~28%; per-256 activation scales would cut it but cost precision — not done |
| MLA attention: K tile staged as fp32 in LDS | 26 -> 58 ms per call | 2x LDS -> 1 workgroup per CU + bank conflicts |
| MLA attention: 16-byte LDS reads | 26 -> 29 ms | not LDS-instruction bound |
| MLA attention: fp16-typed sources / mixed fp16 dot | no change / slower (37 ms) | — |
| fp16 P with transposed V (mixed dot for P.V) | no gain | transposed V staging cost and bank conflicts |
| DeltaNet: 5 independent reductions per step | slower (688 -> 842 us) | it was not latency-bound; more reduction work costs more |
| DeltaNet new kernel for decode | 2x slower per call than old kernel | staging only pays off over many tokens -> kept old kernel for < 16 tokens |
| Larger dense prefill chunks (512/1024) | no change | dense matmuls already efficient at 256 |
| Decode matvec: unroll 4 steps with all loads first | slower | not memory-latency bound (instruction-bound) |
| Decode matvec: stage activations in shared memory | ~25% slower | per-workgroup copy + LDS size cost more than the saved loads |
| Decode matvec: 32 weights per lane via 4x the 8-weight unpack (relying on CSE) | no gain / Q4_K worse | needed a real block-level decoder (`dqi32`), which did work |
| int8 decode for tiny matrices (< 1M weights) | Xing 37.9 -> 30.5 tok/s | each needs a quantize dispatch + barrier; ~1500 per token on Xing |
| int8 decode for Q8_0 | no gain on Inkling | float Q8_0 matvec is already cheap per weight |
| int8 MoE **expert** matvec (decode) | ~30% less kernel time, no end-to-end change | Inkling decode is bound by PCIe reads of host-resident experts; removed |
| More Inkling experts in VRAM at the cost of prefill buffers | est. +5% decode, -40% long-prompt prefill | not done; 12 GB card |
| Driver note | RADV/ACO (Mesa 26.2.6) segfaulted compiling a qmm_q8 variant with a vec4 shared array whose components went unused | scalar shared arrays avoid it |

## Open ideas (not tried yet)

- Decode: two rows per lane group so each activation load serves two rows (activation loads
  outnumber weight loads per block for 2-bit formats).
- Decode: fuse activation quantization into the preceding norm (saves a dispatch + barrier per
  input).
- Decode: block-level `dqi32` for Q8_0 / NVFP4 / IQ4_NL (Inkling's formats).
- Prefill int8 matmul: per-256 activation scales with integer sub-scale accumulation (cuts the
  fix-up; precision trade-off).
- MLA prefill attention is still ~55% of Xing's long-prompt prefill.
