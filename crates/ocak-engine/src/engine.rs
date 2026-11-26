use crate::config::ModelConfig;
use crate::cpu_exec::CpuExec;
use crate::exec::{EngineError, Executor, Out, SlotStep};
use crate::sampler::{Sampler, SamplerParams};
#[cfg(feature = "vulkan")]
use crate::vk_exec::VkExec;
use ocak_gguf::GgufFile;
use ocak_kv::{PagerStats, RadixPager, PAGE_TOKENS};
use ocak_tok::{IncrementalDetokenizer, JsonGrammar, ThinkingGovernor, Tokenizer};
use serde::Serialize;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Vulkan,
    CpuAvx2,
}

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub backend: BackendKind,
    pub max_ctx: usize,
    pub kv_pages: usize,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self { backend: BackendKind::Vulkan, max_ctx: 2048, kv_pages: 256 }
    }
}

#[derive(Debug, Clone)]
pub struct ThinkingCfg {
    pub max_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct GenRequest {
    pub prompt: String,
    pub parse_special: bool,
    pub max_tokens: usize,
    pub sampler: SamplerParams,
    pub stop: Vec<String>,
    pub json_mode: bool,
    pub thinking: Option<ThinkingCfg>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Timings {
    pub prompt_n: usize,
    pub cache_n: usize,
    pub prompt_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub predicted_per_second: f64,
    pub mtp_drafted: usize,
    pub mtp_accepted: usize,
}

#[derive(Debug, Clone)]
pub struct GenResult {
    pub text: String,
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    pub timings: Timings,
}

pub struct NativeEngine {
    pub cfg: ModelConfig,
    pub tokenizer: Tokenizer,
    exec: Box<dyn Executor>,
    mtp: Option<crate::mtp::MtpDrafter>,
    pager: RadixPager,
    max_ctx: usize,
    backend_name: String,
    vram_bytes: u64,
}

impl NativeEngine {
    pub fn load(path: impl AsRef<Path>, opts: EngineOptions) -> Result<Self, EngineError> {
        let g = Arc::new(GgufFile::open(path).map_err(|e| EngineError::Gguf(e.to_string()))?);
        let cfg = ModelConfig::from_gguf(&g)?;
        let tokenizer = Tokenizer::from_gguf(&g).map_err(|e| EngineError::Gguf(e.to_string()))?;
        let mut max_ctx = opts.max_ctx.min(cfg.n_ctx_train.max(2048));
        let page_bytes = (cfg.n_kv_layers * 2 * PAGE_TOKENS * cfg.kv_dim() * 2) as u64;
        let mut n_pages = opts.kv_pages.max(max_ctx.div_ceil(PAGE_TOKENS) + 16);
        // Size the paged KV cache to the VRAM that is actually free (dense / hybrid VkExec models;
        // Xing and Inkling manage their own placement). Shrinks the context rather than overcommit.
        if opts.backend == BackendKind::Vulkan && cfg.arch != "xing4_0" && cfg.arch != "inkling2_loop" && page_bytes > 0 {
            if let Some(budget) = vram_kv_budget(g.file_len()) {
                let fit_pages = (budget / page_bytes) as usize;
                let min_pages = max_ctx.div_ceil(PAGE_TOKENS) + 16;
                if fit_pages < min_pages {
                    let fit_ctx = (fit_pages.saturating_sub(16) * PAGE_TOKENS) / 256 * 256;
                    let new_ctx = fit_ctx.max(2048).min(max_ctx);
                    if new_ctx < max_ctx {
                        eprintln!("ocak: requested max_ctx {max_ctx} does not fit in free VRAM; using {new_ctx}");
                        max_ctx = new_ctx;
                    }
                }
                let min_pages = max_ctx.div_ceil(PAGE_TOKENS) + 16;
                n_pages = n_pages.min(fit_pages).max(min_pages);
            }
        }
        let kv_bytes = n_pages as u64 * page_bytes;
        let vram_bytes = g.file_len() + kv_bytes + 32 * 1024 * 1024;
        let exec: Box<dyn Executor> = if cfg.arch == "xing4_0" {
            let n_cpu_moe = if g.file_len() > 11 * (1u64 << 30) { 12 } else { 8 };
            Box::new(crate::xing_exec::XingExec::new(Arc::clone(&g), opts.backend == BackendKind::Vulkan, max_ctx, n_cpu_moe)?)
        } else if cfg.arch == "inkling2_loop" {
            // 24 physical MoE layers (~14.8 GB NVFP4): keep 12 on RX 6700 XT VRAM (~7.4 GB) + 12 mmap CPU
            Box::new(crate::inkling_exec::InklingExec::new(Arc::clone(&g), opts.backend == BackendKind::Vulkan, max_ctx, 12)?)
        } else { match opts.backend {
            #[cfg(feature = "vulkan")]
            BackendKind::Vulkan => Box::new(VkExec::new(Arc::clone(&g), cfg.clone(), n_pages)?),
            #[cfg(not(feature = "vulkan"))]
            BackendKind::Vulkan => return Err(EngineError::Backend("vulkan feature disabled".into())),
            BackendKind::CpuAvx2 => Box::new(CpuExec::new(Arc::clone(&g), cfg.clone(), n_pages)?),
        } };
        let backend_name = exec.name();
        let mut mtp = crate::mtp::MtpDrafter::try_load(Arc::clone(&g), cfg.clone(), max_ctx).ok().flatten();
        #[cfg(feature = "vulkan")]
        if let (Some(m), Some((ctx, hb, hty, hk, hm))) = (mtp.as_mut(), exec.mtp_gpu_shared()) {
            let _ = m.attach_gpu(ctx, hb, hty, hk, hm);
        }
        Ok(Self {
            cfg,
            tokenizer,
            exec,
            mtp,
            pager: RadixPager::new(n_pages),
            max_ctx,
            backend_name,
            vram_bytes,
        })
    }

    pub fn has_mtp(&self) -> bool { self.mtp.is_some() }
    pub fn supports_multi_slot(&self) -> bool { !self.cfg.is_hybrid }

    pub fn backend_name(&self) -> &str { &self.backend_name }
    pub fn estimated_vram_bytes(&self) -> u64 { self.vram_bytes }
    pub fn pager_stats(&self) -> PagerStats { self.pager.stats() }
    pub fn clear_kv(&mut self) { self.pager.clear(); }

    /// Evaluates `tokens` and returns `(logits, final_hidden)` (used by MTP drafter verification).
    pub fn eval_logits_and_hidden(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
        self.exec.reset_recurrent()?;
        let need_pages = tokens.len().div_ceil(PAGE_TOKENS);
        let mut pages = Vec::with_capacity(need_pages);
        for _ in 0..need_pages {
            pages.push(self.pager.alloc_page().ok_or(EngineError::KvExhausted)?);
        }
        let mut res = Ok(());
        for (pos, &tok) in tokens.iter().enumerate() {
            let is_last = pos + 1 == tokens.len();
            if let Err(e) = self.exec.forward(tok, pos, &pages, Out { logits: is_last, hidden: is_last }) {
                res = Err(e);
                break;
            }
        }
        let out = (self.exec.logits().to_vec(), self.exec.hidden().to_vec());
        self.pager.release(&pages);
        res?;
        Ok(out)
    }

    /// Evaluates `tokens` from scratch and returns the final token's logits (used by parity tests).
    pub fn eval_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::Context("empty token sequence".into()));
        }
        self.exec.reset_recurrent()?;
        let need_pages = tokens.len().div_ceil(PAGE_TOKENS);
        let mut pages = Vec::with_capacity(need_pages);
        for _ in 0..need_pages {
            pages.push(self.pager.alloc_page().ok_or(EngineError::KvExhausted)?);
        }
        let res = self.exec.forward_batch(tokens, 0, &pages, Out { logits: true, hidden: false });
        let logits = self.exec.logits().to_vec();
        self.pager.release(&pages);
        res?;
        Ok(logits)
    }

    /// Computes an L2-normalized mean-pooled embedding vector for `text`.
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>, EngineError> {
        let mut tokens = self.tokenizer.encode(text, true, false);
        if tokens.is_empty() {
            if let Some(b) = self.tokenizer.bos() { tokens.push(b); } else { tokens.push(0); }
        }
        if tokens.len() > self.max_ctx { tokens.truncate(self.max_ctx); }
        self.exec.reset_recurrent()?;
        let need_pages = tokens.len().div_ceil(PAGE_TOKENS);
        let mut pages = Vec::with_capacity(need_pages);
        for _ in 0..need_pages {
            pages.push(self.pager.alloc_page().ok_or(EngineError::KvExhausted)?);
        }
        let mut sum = vec![0f32; self.cfg.n_embd];
        let mut res = Ok(());
        for (pos, &tok) in tokens.iter().enumerate() {
            if let Err(e) = self.exec.forward(tok, pos, &pages, Out { logits: false, hidden: true }) {
                res = Err(e);
                break;
            }
            for (s, &h) in sum.iter_mut().zip(self.exec.hidden()) {
                *s += h;
            }
        }
        self.pager.release(&pages);
        res?;
        let n = tokens.len() as f32;
        for s in &mut sum { *s /= n; }
        let norm = sum.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for s in &mut sum { *s /= norm; }
        Ok(sum)
    }

    /// Generates completion tokens with Radix KV prefix reuse and calls `on_chunk` for each decoded UTF-8 piece.
    /// Returning `false` from `on_chunk` cancels generation cleanly.
    pub fn generate<F>(&mut self, req: &GenRequest, mut on_chunk: F) -> Result<GenResult, EngineError>
    where
        F: FnMut(u32, &str) -> bool,
    {
        let mut tokens = self.tokenizer.encode(&req.prompt, true, req.parse_special);
        if tokens.is_empty() {
            if let Some(b) = self.tokenizer.bos() { tokens.push(b); } else { tokens.push(0); }
        }
        if tokens.len() >= self.max_ctx {
            return Err(EngineError::Context(format!("prompt {} >= max_ctx {}", tokens.len(), self.max_ctx)));
        }

        // Prefix match via RadixPager: never match the very last token so we always get fresh logits for it
        let (cache_n, mut pages) = if self.cfg.is_hybrid {
            self.exec.reset_recurrent()?;
            (0usize, Vec::new())
        } else {
            let match_slice = if tokens.len() > 1 { &tokens[..tokens.len() - 1] } else { &[] };
            let pm = self.pager.match_prefix(match_slice);
            (pm.matched_tokens, pm.pages)
        };

        let ensure_page = |pager: &mut RadixPager, pages: &mut Vec<u32>, pos: usize| -> Result<(), EngineError> {
            let need = pos / PAGE_TOKENS + 1;
            while pages.len() < need {
                pages.push(pager.alloc_page().ok_or(EngineError::KvExhausted)?);
            }
            Ok(())
        };

        let use_mtp = self.mtp.is_some()
            && !req.json_mode
            && req.thinking.is_none()
            && (req.sampler.temperature <= 0.0 || std::env::var("OCAK_MTP_VERIFY").is_ok())
            && std::env::var("OCAK_DISABLE_MTP").is_err();
        if let Some(m) = &mut self.mtp {
            m.reset();
        }
        let t_prefill = Instant::now();
        let prompt_len = tokens.len();
        if prompt_len > 0 {
            if let Err(e) = ensure_page(&mut self.pager, &mut pages, prompt_len - 1) {
                self.pager.release(&pages);
                return Err(e);
            }
        }
        let mut mtp_pos = 0usize;
        if cache_n < prompt_len {
            if use_mtp && cache_n == 0 && prompt_len >= 2 {
                let hiddens = match self.exec.forward_batch_collect_hiddens(
                    &tokens[cache_n..prompt_len],
                    cache_n,
                    &pages,
                    Out { logits: true, hidden: true },
                ) {
                    Ok(h) => h,
                    Err(e) => {
                        self.pager.release(&pages);
                        return Err(e);
                    }
                };
                if hiddens.len() == prompt_len {
                    if let Some(m) = self.mtp.as_mut() {
                        let start = prompt_len.saturating_sub(256);
                        for i in start..prompt_len - 1 {
                            if m.warmup_kv(&hiddens[i], tokens[i + 1], mtp_pos).is_err() {
                                break;
                            }
                            mtp_pos += 1;
                        }
                    }
                }
            } else if let Err(e) = self.exec.forward_batch(
                &tokens[cache_n..prompt_len],
                cache_n,
                &pages,
                Out { logits: true, hidden: use_mtp },
            ) {
                self.pager.release(&pages);
                return Err(e);
            }
        }
        let mut eval_len = prompt_len;
        let prompt_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let prefill_eval_n = prompt_len - cache_n;

        // Register full prompt pages in Radix tree right away
        self.pager.insert_prefix(&tokens, &pages);

        let mut sampler = Sampler::new(req.sampler.clone());
        for &t in &tokens {
            sampler.accept(t);
        }
        let mut detok = IncrementalDetokenizer::new();
        let mut json_g = req.json_mode.then(JsonGrammar::new);
        let mut think_gov = req.thinking.as_ref().and_then(|tc| {
            let open = self.tokenizer.token_id("<think>")?;
            let close = self.tokenizer.token_id("</think>")?;
            Some(ThinkingGovernor::new(open, close, tc.max_tokens))
        });

        let mut gen_tokens = Vec::new();
        let mut full_text = String::new();
        let mut finish = FinishReason::Length;
        let t_dec = Instant::now();

        let mut logits_buf = self.exec.logits().to_vec();
        let mut hidden_buf = if use_mtp { self.exec.hidden().to_vec() } else { Vec::new() };
        let mut mtp_drafted = 0usize;
        let mut mtp_accepted = 0usize;
        let mut forced_tok: Option<u32> = None;
        let pure_greedy = req.sampler.temperature <= 0.0
            && (req.sampler.repeat_penalty - 1.0).abs() < 1e-6
            && json_g.is_none()
            && think_gov.is_none()
            && !use_mtp;
        let prof_mtp = std::env::var("OCAK_MTP_PROFILE").is_ok();
        let (mut ms_draft, mut ms_v2, mut ms_rej) = (0.0f64, 0.0f64, 0.0f64);

        while gen_tokens.len() < req.max_tokens && tokens.len() < self.max_ctx {
            let next_tok = if let Some(ft) = forced_tok.take() {
                ft
            } else if let Some(forced) = think_gov.as_mut().and_then(|g| g.forced_next()) {
                forced
            } else {
                if let Some(jg) = &json_g {
                    jg.mask_logits(&self.tokenizer, &mut logits_buf);
                }
                sampler.sample(&mut logits_buf)
            };

            if self.tokenizer.is_eog(next_tok) {
                finish = FinishReason::Stop;
                break;
            }
            sampler.accept(next_tok);
            if let Some(tg) = &mut think_gov {
                tg.observe(next_tok);
            }
            let bytes = self.tokenizer.token_bytes(next_tok);
            if let Some(jg) = &mut json_g {
                for &b in &bytes {
                    jg.feed_byte(b);
                }
            }
            let piece = detok.push(&bytes);
            gen_tokens.push(next_tok);
            tokens.push(next_tok);
            full_text.push_str(&piece);

            if req.stop.iter().any(|s| !s.is_empty() && full_text.ends_with(s)) {
                finish = FinishReason::Stop;
                break;
            }
            if !on_chunk(next_tok, &piece) {
                finish = FinishReason::Cancelled;
                break;
            }
            if gen_tokens.len() >= req.max_tokens || tokens.len() >= self.max_ctx {
                break;
            }

            let pos = tokens.len() - 1;
            // Real 2-token batched speculative MTP verify when greedy + MTP drafter available
            if use_mtp && gen_tokens.len() + 1 < req.max_tokens && tokens.len() + 1 < self.max_ctx {
                let t0 = Instant::now();
                let draft_tok = self.mtp.as_mut().and_then(|m| {
                    let dt = m.draft_token(&hidden_buf, next_tok, mtp_pos).ok()?;
                    mtp_pos += 1;
                    Some(dt)
                });
                ms_draft += t0.elapsed().as_secs_f64() * 1000.0;
                if let Some(dt) = draft_tok {
                    mtp_drafted += 1;
                    if let Err(e) = ensure_page(&mut self.pager, &mut pages, pos + 1) {
                        self.pager.release(&pages);
                        return Err(e);
                    }
                    let t2 = Instant::now();
                    match self.exec.forward_verify2([next_tok, dt], pos, &pages) {
                        Ok(Some(([mut l0, l1], [h0, h1]))) => {
                            ms_v2 += t2.elapsed().as_secs_f64() * 1000.0;
                            let v1 = sampler.sample(&mut l0);
                            if v1 == dt {
                                // ACCEPTED! Emit draft token immediately and carry forward (l1, h1)!
                                mtp_accepted += 1;
                                eval_len = pos + 2;
                                if self.tokenizer.is_eog(dt) {
                                    finish = FinishReason::Stop;
                                    break;
                                }
                                sampler.accept(dt);
                                let b2 = self.tokenizer.token_bytes(dt);
                                let p2 = detok.push(&b2);
                                gen_tokens.push(dt);
                                tokens.push(dt);
                                full_text.push_str(&p2);
                                if req.stop.iter().any(|s| !s.is_empty() && full_text.ends_with(s)) {
                                    finish = FinishReason::Stop;
                                    break;
                                }
                                if !on_chunk(dt, &p2) {
                                    finish = FinishReason::Cancelled;
                                    break;
                                }
                                logits_buf = l1;
                                hidden_buf = h1;
                                continue;
                            } else {
                                // REJECTED: restore exact post-next_tok recurrent state (zero re-forward!)
                                eval_len = pos + 1;
                                let t3 = Instant::now();
                                if let Err(e) = self.exec.restore_recurrent() {
                                    self.pager.release(&pages);
                                    return Err(e);
                                }
                                ms_rej += t3.elapsed().as_secs_f64() * 1000.0;
                                forced_tok = Some(v1);
                                hidden_buf = h0;
                                continue;
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            self.pager.release(&pages);
                            return Err(e);
                        }
                    }
                }
            }

            if let Err(e) = ensure_page(&mut self.pager, &mut pages, pos) {
                self.pager.release(&pages);
                return Err(e);
            }
            if pure_greedy {
                if let Some(tok) = self.exec.forward_argmax(next_tok, pos, &pages).inspect_err(|_e| {
                    self.pager.release(&pages);
                })? {
                    eval_len = pos + 1;
                    forced_tok = Some(tok);
                    continue;
                }
            }
            if let Err(e) = self.exec.forward(next_tok, pos, &pages, Out { logits: true, hidden: use_mtp }) {
                self.pager.release(&pages);
                return Err(e);
            }
            eval_len = pos + 1;
            logits_buf.copy_from_slice(self.exec.logits());
            if use_mtp {
                hidden_buf.copy_from_slice(self.exec.hidden());
            }
        }

        let tail = detok.flush();
        if !tail.is_empty() {
            full_text.push_str(&tail);
            let _ = on_chunk(0, &tail);
        }

        // Only cache tokens whose forward pass actually populated the KV cache!
        self.pager.insert_prefix(&tokens[..eval_len.min(tokens.len())], &pages);
        self.pager.release(&pages);

        let dec_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
        if prof_mtp {
            let gpu_ok = self.mtp.as_ref().map(|m| m.has_gpu()).unwrap_or(false);
            eprintln!(
                "[mtp-profile] gpu={gpu_ok} drafted={mtp_drafted} accepted={mtp_accepted} draft={ms_draft:.1}ms v2={ms_v2:.1}ms rej={ms_rej:.1}ms total={dec_ms:.1}ms"
            );
        }
        let timings = Timings {
            prompt_n: prompt_len,
            cache_n,
            prompt_ms,
            prompt_per_second: if prompt_ms > 0.0 { prefill_eval_n as f64 / (prompt_ms / 1000.0) } else { 0.0 },
            predicted_n: gen_tokens.len(),
            predicted_ms: dec_ms,
            predicted_per_second: if dec_ms > 0.0 { gen_tokens.len() as f64 / (dec_ms / 1000.0) } else { 0.0 },
            mtp_drafted,
            mtp_accepted,
        };

        Ok(GenResult { text: full_text, tokens: gen_tokens, finish_reason: finish, timings })
    }

    /// Runs multimodal greedy decode: `prefix_tokens` -> `embds` (audio/vision) -> `suffix_tokens` -> greedy decode.
    pub fn generate_multimodal_greedy(
        &mut self,
        prefix_tokens: &[u32],
        embds: &[Vec<f32>],
        suffix_tokens: &[u32],
        max_tokens: usize,
    ) -> Result<String, EngineError> {
        self.exec.reset_recurrent()?;
        let total_in = prefix_tokens.len() + embds.len() + suffix_tokens.len();
        if total_in >= self.max_ctx {
            return Err(EngineError::Context(format!("multimodal prompt {total_in} >= max_ctx {}", self.max_ctx)));
        }
        let mut pages = Vec::new();
        let ensure_page = |pager: &mut RadixPager, pages: &mut Vec<u32>, pos: usize| -> Result<(), EngineError> {
            let need = pos / PAGE_TOKENS + 1;
            while pages.len() < need {
                pages.push(pager.alloc_page().ok_or(EngineError::KvExhausted)?);
            }
            Ok(())
        };
        let mut pos = 0usize;
        for &tok in prefix_tokens {
            ensure_page(&mut self.pager, &mut pages, pos)?;
            self.exec.forward(tok, pos, &pages, Out { logits: false, hidden: false })?;
            pos += 1;
        }
        for emb in embds {
            ensure_page(&mut self.pager, &mut pages, pos)?;
            self.exec.forward_embd(emb, pos, &pages, Out { logits: false, hidden: false })?;
            pos += 1;
        }
        for (i, &tok) in suffix_tokens.iter().enumerate() {
            ensure_page(&mut self.pager, &mut pages, pos)?;
            let is_last = i + 1 == suffix_tokens.len();
            self.exec.forward(tok, pos, &pages, Out { logits: is_last, hidden: false })?;
            pos += 1;
        }
        let mut detok = IncrementalDetokenizer::new();
        let mut text = String::new();
        for step in 0..max_tokens {
            let next_tok = self.exec.logits()
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            if self.tokenizer.is_eog(next_tok) {
                break;
            }
            let bytes = self.tokenizer.token_bytes(next_tok);
            text.push_str(&detok.push(&bytes));
            if step + 1 == max_tokens || pos >= self.max_ctx {
                break;
            }
            ensure_page(&mut self.pager, &mut pages, pos)?;
            self.exec.forward(next_tok, pos, &pages, Out { logits: true, hidden: false })?;
            pos += 1;
        }
        text.push_str(&detok.flush());
        self.pager.release(&pages);
        Ok(text)
    }

    /// Opens a continuous-batching decode slot: matches Radix KV prefix, prefills uncached prompt tokens,
    /// and returns an `ActiveSlot` ready for interleaved multi-slot `step_slots` calls.
    pub fn open_slot(&mut self, req: GenRequest) -> Result<ActiveSlot, EngineError> {
        let mut tokens = self.tokenizer.encode(&req.prompt, true, req.parse_special);
        if tokens.is_empty() {
            if let Some(b) = self.tokenizer.bos() { tokens.push(b); } else { tokens.push(0); }
        }
        if tokens.len() >= self.max_ctx {
            return Err(EngineError::Context(format!("prompt {} >= max_ctx {}", tokens.len(), self.max_ctx)));
        }
        let match_slice = if tokens.len() > 1 { &tokens[..tokens.len() - 1] } else { &[] };
        let pm = self.pager.match_prefix(match_slice);
        let cache_n = pm.matched_tokens;
        let mut pages = pm.pages;

        let prompt_len = tokens.len();
        let need_pages = (prompt_len - 1) / PAGE_TOKENS + 1;
        while pages.len() < need_pages {
            match self.pager.alloc_page() {
                Some(p) => pages.push(p),
                None => {
                    self.pager.release(&pages);
                    return Err(EngineError::KvExhausted);
                }
            }
        }
        let t_prefill = Instant::now();
        if cache_n < prompt_len {
            if let Err(e) = self.exec.forward_batch(&tokens[cache_n..prompt_len], cache_n, &pages, Out { logits: true, hidden: false }) {
                self.pager.release(&pages);
                return Err(e);
            }
        }
        let prompt_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let prefill_eval_n = prompt_len - cache_n;
        self.pager.insert_prefix(&tokens, &pages);

        let mut sampler = Sampler::new(req.sampler.clone());
        for &t in &tokens {
            sampler.accept(t);
        }
        let json_g = req.json_mode.then(JsonGrammar::new);
        let think_gov = req.thinking.as_ref().and_then(|tc| {
            let open = self.tokenizer.token_id("<think>")?;
            let close = self.tokenizer.token_id("</think>")?;
            Some(ThinkingGovernor::new(open, close, tc.max_tokens))
        });
        Ok(ActiveSlot {
            max_tokens: req.max_tokens,
            stop: req.stop,
            tokens,
            eval_len: prompt_len,
            pages,
            logits: self.exec.logits().to_vec(),
            sampler,
            detok: IncrementalDetokenizer::new(),
            json_g,
            think_gov,
            gen_tokens: Vec::new(),
            full_text: String::new(),
            finish: FinishReason::Length,
            prompt_len,
            cache_n,
            prefill_eval_n,
            prompt_ms,
            t_dec: Instant::now(),
            done: false,
        })
    }

    /// Advances all active slots by 1 token, executing the forward pass for all remaining slots
    /// in a single batched `forward_multi_slot` GPU submit! Returns `done[i]` for each slot.
    pub fn step_slots<F>(&mut self, slots: &mut [ActiveSlot], mut on_token: F) -> Result<Vec<bool>, EngineError>
    where
        F: FnMut(usize, u32, &str) -> bool,
    {
        let mut need_fwd: Vec<usize> = Vec::with_capacity(slots.len());
        for (i, s) in slots.iter_mut().enumerate() {
            if s.done {
                continue;
            }
            if s.gen_tokens.len() >= s.max_tokens || s.tokens.len() >= self.max_ctx {
                s.done = true;
                continue;
            }
            let next_tok = if let Some(forced) = s.think_gov.as_mut().and_then(|g| g.forced_next()) {
                forced
            } else {
                if let Some(jg) = &s.json_g {
                    jg.mask_logits(&self.tokenizer, &mut s.logits);
                }
                s.sampler.sample(&mut s.logits)
            };
            if self.tokenizer.is_eog(next_tok) {
                s.finish = FinishReason::Stop;
                s.done = true;
                continue;
            }
            s.sampler.accept(next_tok);
            if let Some(tg) = &mut s.think_gov {
                tg.observe(next_tok);
            }
            let bytes = self.tokenizer.token_bytes(next_tok);
            if let Some(jg) = &mut s.json_g {
                for &b in &bytes {
                    jg.feed_byte(b);
                }
            }
            let piece = s.detok.push(&bytes);
            s.gen_tokens.push(next_tok);
            s.tokens.push(next_tok);
            s.full_text.push_str(&piece);
            if s.stop.iter().any(|st| !st.is_empty() && s.full_text.ends_with(st)) {
                s.finish = FinishReason::Stop;
                s.done = true;
                let _ = on_token(i, next_tok, &piece);
                continue;
            }
            if !on_token(i, next_tok, &piece) {
                s.finish = FinishReason::Cancelled;
                s.done = true;
                continue;
            }
            if s.gen_tokens.len() >= s.max_tokens || s.tokens.len() >= self.max_ctx {
                s.done = true;
                continue;
            }
            let pos = s.tokens.len() - 1;
            let need = pos / PAGE_TOKENS + 1;
            while s.pages.len() < need {
                s.pages.push(self.pager.alloc_page().ok_or(EngineError::KvExhausted)?);
            }
            need_fwd.push(i);
        }

        if need_fwd.len() >= 2 {
            let steps: Vec<SlotStep<'_>> = need_fwd
                .iter()
                .map(|&idx| {
                    let s = &slots[idx];
                    SlotStep {
                        token: *s.tokens.last().unwrap(),
                        pos: s.tokens.len() - 1,
                        page_table: &s.pages,
                    }
                })
                .collect();
            if let Some(all_logits) = self.exec.forward_multi_slot(&steps)? {
                for (&idx, l) in need_fwd.iter().zip(all_logits) {
                    slots[idx].logits = l;
                    slots[idx].eval_len = slots[idx].tokens.len();
                }
            } else {
                for &idx in &need_fwd {
                    let s = &mut slots[idx];
                    self.exec.forward(*s.tokens.last().unwrap(), s.tokens.len() - 1, &s.pages, Out { logits: true, hidden: false })?;
                    s.logits.copy_from_slice(self.exec.logits());
                    s.eval_len = s.tokens.len();
                }
            }
        } else if let Some(&idx) = need_fwd.first() {
            let s = &mut slots[idx];
            self.exec.forward(*s.tokens.last().unwrap(), s.tokens.len() - 1, &s.pages, Out { logits: true, hidden: false })?;
            s.logits.copy_from_slice(self.exec.logits());
            s.eval_len = s.tokens.len();
        }

        Ok(slots.iter().map(|s| s.done).collect())
    }

    /// Finalizes an `ActiveSlot`, registering its evaluated pages in the Radix KV tree and releasing active refs.
    pub fn close_slot(&mut self, mut slot: ActiveSlot) -> GenResult {
        let tail = slot.detok.flush();
        if !tail.is_empty() {
            slot.full_text.push_str(&tail);
        }
        self.pager.insert_prefix(&slot.tokens[..slot.eval_len.min(slot.tokens.len())], &slot.pages);
        self.pager.release(&slot.pages);
        let dec_ms = slot.t_dec.elapsed().as_secs_f64() * 1000.0;
        let timings = Timings {
            prompt_n: slot.prompt_len,
            cache_n: slot.cache_n,
            prompt_ms: slot.prompt_ms,
            prompt_per_second: if slot.prompt_ms > 0.0 { slot.prefill_eval_n as f64 / (slot.prompt_ms / 1000.0) } else { 0.0 },
            predicted_n: slot.gen_tokens.len(),
            predicted_ms: dec_ms,
            predicted_per_second: if dec_ms > 0.0 { slot.gen_tokens.len() as f64 / (dec_ms / 1000.0) } else { 0.0 },
            mtp_drafted: 0,
            mtp_accepted: 0,
        };
        GenResult { text: slot.full_text, tokens: slot.gen_tokens, finish_reason: slot.finish, timings }
    }
}

pub struct ActiveSlot {
    max_tokens: usize,
    stop: Vec<String>,
    tokens: Vec<u32>,
    eval_len: usize,
    pages: Vec<u32>,
    logits: Vec<f32>,
    sampler: Sampler,
    detok: IncrementalDetokenizer,
    json_g: Option<JsonGrammar>,
    think_gov: Option<ThinkingGovernor>,
    gen_tokens: Vec<u32>,
    full_text: String,
    finish: FinishReason,
    prompt_len: usize,
    cache_n: usize,
    prefill_eval_n: usize,
    prompt_ms: f64,
    t_dec: Instant,
    done: bool,
}

/// Free VRAM (bytes) left for the KV cache after loading `weight_bytes` of weights, from the
/// amdgpu sysfs counters. Keeps a margin for activations, staging, scratch and other apps.
/// Returns None when no AMD GPU telemetry is available (no clamping is applied then).
fn vram_kv_budget(weight_bytes: u64) -> Option<u64> {
    const MARGIN: u64 = 768 * 1024 * 1024;
    let tel = ocak_sysfs::amdgpu::discover_amd_gpu()?.telemetry();
    if tel.vram_total_bytes == 0 {
        return None;
    }
    let free = tel.vram_total_bytes.saturating_sub(tel.vram_used_bytes);
    Some(free.saturating_sub(weight_bytes).saturating_sub(MARGIN))
}
