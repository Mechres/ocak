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
  (2560x9216x256 matmul per format), `vk_qmm_grouped_bench`, `vk_small_batch_bench` (decode
  matvec at Qwen3.8's FFN shape, rotating 6 weight copies). A single small weight matrix stays in
  the 96 MB Infinity Cache across repeats and reports impossible bandwidth — rotate buffers that
  together exceed ~100 MB.
- **Comparing sampled runs:** `SamplerParams::seed = 0` means a random seed per run; set
  `OCAK_SEED=<n>` to fix it. With a fixed seed, sampled output must be identical with MTP on and off.
- **MTP must stay lossless:** with greedy sampling, `OCAK_DISABLE_MTP=1` and the default must
  produce identical text. The verify step (`forward_verify2`, batched layer code) and single-token
  decode use different call paths, so every kernel pair must give bit-identical results: same
  matvec kernel for 1 and 2..8 tokens, same reduction order in `add_rms_norm` and `rms_norm`.
  `vk_matvec_batch_matches_single_bitwise` checks the matvecs; check the end-to-end equality
  after touching norms, matvec routing or anything in the decode path.
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
| IQ1 grid pre-transformed to 8 (g + 1) per byte (new 4096-word table at [8592, 12688)), so the unpack is one `sub_bytes` per word instead of the borrow-free g + 1 formula; that table in shared memory (the block decoder had been reading the global copy) | IQ1_S matvec 75.6 -> 56.5 us (225 -> 308 GB/s); batch 2 68 us |
| Sign byte-mask table for IQ2_XS / IQ2_S block decoders (shared memory, per KSIGNS index / raw sign byte; negate as (g ^ M) + ((M >> 7) & 0x01010101) instead of deriving the mask from 4 sign bits) | IQ2_XS 77.8 -> 75.5 us, IQ2_S 84.4 -> 82.8 us; Qwen3.8 IQ2_S decode 36.2 -> 36.3 tok/s |
| Opt-in `OCAK_PREFILL_Q256=1`: prefill activations with one int8 scale per 256 values (quant_q8 -DGROUP256, same layout); qmm_q8 -DACT256 keeps acc in units of the current activation scale and rescales once per 256, so the fix-up drops the d_w * d_x multiply (lossy: 1.6-1.9x the int8 rounding error in vk_qmm_int8_matches_qmv; Qwen3.8 greedy text identical for ~70 tokens of a summary, then a different wording; count-test passes) | matmul +1-7% (IQ2_XS 2.97 -> 2.78 ms, IQ2_S 2.96 -> 2.80); Qwen3.8 IQ2_S prefill 300.5 -> 313.6 tok/s |
| Q3_K block decoder: the 12 scale bytes as three word loads instead of four byte loads per block (timing experiments: byte loads ~28% of the kernel, unaligned-word funnels ~11%), and decode from aligned words then realign each result once (110-byte superblocks sit at offset 0 or 2 mod 4) | Q3_K matvec 155 -> 110 us (247 -> 347 GB/s); Qwen3.8 Q2_K decode 27.0 -> 29.9 tok/s, with MTP 41.9 -> 45.8; output identical |
| Same word-load treatment elsewhere: Q4_K / Q5_K scales as three aligned words, Q6_K realign-once + one 16-bit scale load, IQ1_S / IQ1_M / Q2_K adjacent scale bytes as one load | Q4_K 146.6 -> 140-142 us, Q6_K 198.5 -> 195.7 us; IQ1_S / Q2_K unchanged (already one or two loads); output identical |
| JSON mode (`json_mode`): token bytes decoded once per tokenizer (the GPT-2 char->byte map was rebuilt per call), one grammar scratch state per token instead of a clone per byte, and the vocabulary mask cached as a bitset keyed by the exact grammar state (strings / numbers repeat it) | mask per generated token 990 ms -> 0.24 ms on a cache hit, ~11 ms on a state change (248k vocab) |
| JSON mask on a state change (cache miss): per-state table of accepted first bytes, skip leading whitespace that is a no-op in the current state, accept plain string bytes without stepping inside strings, end-of-generation ids handled once, flat token-byte table | miss 11 ms -> 1.0-1.3 ms between values, 2.3-2.9 ms in strings / numbers (same allowed sets) |
| Sampler: streaming top-k scan for top_k <= 1024 (no full-vocabulary copy / partition); plain temperature computes each exp once | top_k=40: 0.66 -> 0.33 ms per draw; plain T: 1.60 -> 1.19 ms |
| Batched path (prefill, MTP verify): `add` + `rms_norm_batch` fused into `add_rms_norm_batch` (one workgroup per row, bit-identical to decode's add_rms_norm), including the layer-closing add with the next layer's attn norm | Qwen3.8 IQ2_S MTP decode 50.1 -> 50.5 tok/s; output identical |
| Page-table upload skipped when unchanged (it changes once per 16-token page; each upload was a submit + fence wait) | Qwen3.8 IQ2_S decode 36.2-36.3 -> 36.4-36.5 tok/s (small model: no measurable change) |
| Batched MTP prompt warmup (`warmup_kv_batch`: two uploads + one submit per 128 rows with batched norms / qmm / rope / kv_write, instead of two uploads + a submit per token) | 500-token prompt with MTP on: 308.2 -> 317 tok/s (MTP off: 319.8); same acceptance (75/115 on an 833-token summary) |
| Readbacks recorded into the compute submit (`Recorder::finish_and_read`) for decode / verify / drafter, and the drafter's inputs written with `vkCmdUpdateBuffer` (`Recorder::update`) instead of separate upload / download submits | Qwen3.8 IQ2_S MTP 51.7 -> ~51.9 tok/s (+0.5%, near noise); plain decode unchanged; output identical (greedy and seeded sampled) |
| Model load: background sequential read of the file into the page cache when it is not already resident (mincore sample), and pipelined weight uploads (`upload_async`: two staging halves / fences, CPU copy of one tensor overlaps the DMA of the previous) | Qwen3.8 IQ2_S (8.4 GB) cold load 5.2-5.8 -> 4.3 s (disk reads 2.5 GB/s: ~3.4 s floor), warm 1.47 -> 1.15 s |
| Xing MLA prefill attention: new `mla_attn_prefill_f2.comp` (scores with VK_VALVE mixed-float dot on packed-fp16 Q; 16 lanes per head, each lane computes two consecutive tokens so every shared-memory read feeds two rows; one key tile per token pair); the fallback kernel also shares tiles across 2 tokens | 15.7k-token prompt 481 -> 532.7 tok/s (fallback 488) |
| Parallel MoE top-k | 235 -> 8 us per call (Inkling decode +40% with the int8 work) |
| Two-pass argmax, wide norms | argmax 252 -> 5 us, add_rms_norm 11 -> 3.7 us |
| int8 matvec in 8-token groups for 9..31-token batches (short prompts; used to fall back to the old 8-token float kernels at ~1.4 ms per call) | Qwen3.8 IQ2_S 12-token prompt 22 -> 115 tok/s; Qwen3.5-4B 219 -> 337 |
| MTP verify with device-side argmax for plain greedy sampling (no 2 x vocab logits download / CPU sampling) | verify host overhead 6.7 -> ~1.4 ms; Qwen3.8 IQ2_S (borrowed head) 40.4 -> 46.8 tok/s, Q2_K 32 -> 38 |
| MTP verify: DeltaNet / conv kernels write the post-token-0 rollback state from registers (was: per-token dispatches + copies with transfer barriers in all 48 layers) | verify 33.6 -> 32.7 ms; IQ2_S 47.8 tok/s |
| Host-cached staging buffer (was write-combined: CPU reads ~200 MB/s) | 1 MB logits download 4.5 -> 0.2 ms; sampled decode Qwen3.8 IQ2_S 26.5 -> 33.7 tok/s |
| MTP for sampled decoding by default (sample from the target logits, accept the draft iff it equals the sample: exact, same draw count per token) | Qwen3.8 Q2_K at T=0.7: 24.5 -> 37.9 tok/s |
| Sampler without a full-vocabulary sort when top_k = 0 (O(n) for plain temperature; growing top-M prefix for top_p / min_p) | top_k = 0 sampling 8.0 -> 1.8 ms per token (248k vocab) |
| Drafter KV gap fill: feed the accepted (hidden, draft) pair to the MTP layer's KV (it skipped one position per accepted draft) | acceptance +1-2 points, ~+1% |
| Multi-draft greedy MTP: chained drafts (drafter's own output hidden), one 2..4-token verify, per-token rollback snapshots written by the DeltaNet/conv kernels, adaptive 1 vs 2 drafts (EMA of second-draft acceptance) | Qwen3.8 Q2_K story 39 -> 44.4, code 48.7; IQ2_S (borrowed head) code 50.4 -> 54.8 tok/s; output identical |
| Multi-draft MTP for sampled decoding (logits per verified position; sample each in turn, keep drafts while they equal the samples) | Qwen3.8 Q2_K at T=0.7: 34.8 -> 36.6 (story), 38.9 -> 41.7 (code); seeded output identical to MTP off |
| Borrowed MTP head (`OCAK_MTP_FROM=<gguf>`): Qwen3.8 IQ2_S has no NextN layer, the Q2_K file of the same model does | IQ2_S decode 34.7 -> 40.4 tok/s, 74% acceptance, output identical |

Decode matvec bandwidth at Qwen3.8's FFN shape (5120 x 17408, batch 1, from VRAM): IQ2_XXS 321,
IQ2_S 324, IQ2_XS 330, IQ3_XXS 349, IQ3_S 351, IQ1_S 308, Q2_K 351, Q3_K 347, Q4_K 336, Q6_K 368,
Q8_0 371 GB/s (RX 6700 XT nominal 384 GB/s). Large matrices are near the limit.

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
| Decode matvec: two rows per 32-lane group (share activation loads) | no change per kernel | activation loads are not the limit (cached, overlapped) |
| MTP draft head over only the first N vocab rows (BPE: low ids are common) | N=65536: ~+2-5% (within noise), acceptance 72 -> 69% | marginal, and acceptance would drop for text with high token ids (non-English, code) |
| Fixed 3 MTP drafts per step | slower than 2 (Q2_K 41.0 vs 44.7) | third-draft acceptance too low for its drafter pass + verify row |
| Fixed 2 drafts on low-acceptance text | IQ2_S story 46.7 vs 47.7 with 1 draft | replaced by the adaptive choice |
| Decode Q3_K strided decoder (lane = 8 byte columns -> 8 weights of each of the 4 sub-blocks; each qs / hmask byte loaded once) | slower (155 -> 205 us at 5120x17408) | activation loads quadruple (4 segments x (uvec2 + scale)) and the per-lane scale decode diverges; the block decoder's redundant weight loads hit cache |
| IQ1 decode with 32 rows per workgroup (amortize the 16 KB table copy) | no change (60 us either way) | the copy is not the bottleneck |
| Sign byte-mask table for IQ2_XXS / IQ3_XXS / IQ3_S | IQ2_XXS slower (72.3 -> 73-75 us), IQ3 unchanged | IQ2_XXS / IQ3_XXS replace one KSIGNS read per 8 weights with two mask reads; IQ3 is already near bandwidth |
| int8 path for DeltaNet's 5120x48 Q8_0 `ssm_alpha` / `ssm_beta` when the q8 cache already holds the input | no change (10.6 vs 11.1 us serialized, 36.2 tok/s) | latency-bound: 12 workgroups each walk a serial load chain; skipping both matvecs entirely only gives +1.2% (36.1 -> 36.5), so the ceiling is small |
| Drafting the second MTP token without the CPU round trip (GPU embedding lookup) | not built: estimated < 0.5% | profile: draft passes are 12% of MTP decode (~3.2 ms each, dominated by the 248k-vocab head matvec); a submit + wait + 4-byte read is ~0.1-0.2 ms per chained draft |
| Prefill int8 matmul with double-buffered shared tiles (one barrier per K block instead of two) | no change (Q8_0 2.42 vs 2.45 ms, IQ2_S 2.95 vs 3.01 ms at 5120x17408x256) | barriers are not the limit; the loop is instruction-issue bound (see below) |
| Fuse decode activation quantization into `add_rms_norm` / `rms_norm` (norm kernel also writes the int8 copy; bit-identical, MTP stays exact) | no change (Qwen3.8 IQ2_S 36.2-36.3 tok/s; standalone quant_q8 dispatches halved) | the separate quant dispatch + barrier was not on the critical path |
| Q3_K: fold the -4 offset into the min term (unsigned 0..7 weights, dot(q - 4, x) = dot(q, x) - 4 sum(x)) | no change (154 vs 155 us; 110 us either way after the scale fix) | the kernel was load-bound (byte loads), not ALU-bound; dropped to keep the old bits |
| int8 decode path for Q8_0 matvecs (Inkling's dense Q8_0 tensors, ~290 calls / token on the float kernel) | slower (Inkling decode 24.5 -> 23.8 tok/s) | the float Q8_0 kernel is already cheap per weight; the int8 path adds activation quantization |
| Load: posix_fadvise(WILLNEED) readahead thread | no change (5.5 s cold) | real sequential reads were needed; on a warm cache a read-ahead pass costs ~110 ms, hence the residency check |
| Lazy pipeline creation | not needed | VkContext::new (all ~240 pipelines) takes 16-17 ms warm thanks to Mesa's shader cache |
| MLA prefill: 2 tokens per workgroup alone (halves latent-cache tile loads) | +1% (481 -> 487 tok/s at 15.7k) | not memory-bound: ACO already emits one v_fma_mix per f16 MAC (1088 per key per head), so only fewer MACs or 2-MAC instructions help |
| MLA prefill 2-rows-per-lane layout without FDOT2 (vec2 Q) | slower (485 -> 349 tok/s) | register spills; kept only for the FDOT2 variant (packed fp16 Q) |
| Unabsorbed MLA prefill (decompress K/V per layer: 192 + 128 dims per key instead of 576 + 512) | not built | needs a transposed copy of wk_b (IQ4_NL rows of 128; +160 MB as f16, but Xing already keeps 8-12 expert layers in GTT) and per-context decompressed V buffers (~134 MB at 16k) with key-range tiling; estimated ~2x long-prompt prefill if VRAM allowed |
| Pitfall: changing `add_rms_norm` / `rms_norm` thread counts independently | broke lossless MTP (text changed with MTP on) | different reduction order in decode vs verify; int8 quantization amplifies the last-bit difference |
| Driver note | RADV/ACO (Mesa 26.2.6) segfaulted compiling a qmm_q8 variant with a vec4 shared array whose components went unused | scalar shared arrays avoid it |

### Prefill matmul ISA (RADV_DEBUG=shaders, qmm_q8 non-SCALE16 main loop, Nov 2026)

614 instructions per 32-deep K block per thread: 256 `v_dot4_i32_i8` (42%), 131 for the float
scale fix-up (32 `v_cvt_f32_i32`, 33 `v_mul`, 66 FMA), 64 `ds_read_b64`, 41 `s_waitcnt`, the rest
addressing / loader unpack; 128 VGPRs. Removing the fix-up entirely (wrong results, timing only)
gives 18.9 -> 22.4 TOPS on Q8_0, so the kernel is issue-bound, not memory- or barrier-bound;
SCALE16 formats (IQ2_XS / IQ2_S / IQ1_M / Q3_K / Q6_K) pay the fix-up twice per block. The remaining
big lever is fewer fix-ups (coarser activation scales: lossy). Reference: llama.cpp Vulkan
(build d1e573195) pp512 on Qwen3.8 IQ2_S is 246 tok/s; ocak 296.

## Open ideas (not tried yet)

- MTP prompt warmup still downloads all prompt hidden states (one ~10 MB read) before the
  batched warmup; reading the last 255 rows from the GPU buffer directly would save a few ms.
- Inkling attention (`inkling_attn.comp`: shared-memory tree reductions, strided K reads, no
  split-K) is ~3% of decode at short context; matters only for long contexts.

- Prefill: the dqi8 unpackers (qmm_q8 loader) still use single-byte scale loads (Q3_K, Q4_K,
  IQ1_M); the prefill loop is issue-bound, so the gain is likely small.
- Decode: block-level `dqi32` for Q8_0 / NVFP4 / IQ4_NL (Inkling's formats).
- Prefill: the per-256 opt-in still converts every block's int32 dot to float; accumulating
  ia * integer weight sub-scale in int32 needs a 24-bit multiply ACO only emits when it can prove
  the operand ranges (otherwise quarter-rate `v_mul_lo_u32`).
- MLA prefill attention is still the largest share of Xing's long-prompt prefill (66% at 15.7k
  tokens before the FDOT2 kernel); the value accumulation (2 v_fma_mix per pair per row) is now
  the bigger half.
