use crate::config::ModelConfig;
use crate::cpu_exec::CpuExec;
use crate::exec::{EngineError, Executor, Out};
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
        let max_ctx = opts.max_ctx.min(cfg.n_ctx_train.max(2048));
        let n_pages = opts.kv_pages.max((max_ctx + PAGE_TOKENS - 1) / PAGE_TOKENS + 16);
        let kv_bytes = (cfg.n_layer * n_pages * 2 * PAGE_TOKENS * cfg.kv_dim() * 4) as u64;
        let vram_bytes = g.file_len() + kv_bytes + 32 * 1024 * 1024;
        let exec: Box<dyn Executor> = match opts.backend {
            #[cfg(feature = "vulkan")]
            BackendKind::Vulkan => Box::new(VkExec::new(Arc::clone(&g), cfg.clone(), n_pages)?),
            #[cfg(not(feature = "vulkan"))]
            BackendKind::Vulkan => return Err(EngineError::Backend("vulkan feature disabled".into())),
            BackendKind::CpuAvx2 => Box::new(CpuExec::new(Arc::clone(&g), cfg.clone(), n_pages)?),
        };
        let backend_name = exec.name();
        Ok(Self {
            cfg,
            tokenizer,
            exec,
            pager: RadixPager::new(n_pages),
            max_ctx,
            backend_name,
            vram_bytes,
        })
    }

    pub fn backend_name(&self) -> &str { &self.backend_name }
    pub fn estimated_vram_bytes(&self) -> u64 { self.vram_bytes }
    pub fn pager_stats(&self) -> PagerStats { self.pager.stats() }
    pub fn clear_kv(&mut self) { self.pager.clear(); }

    /// Evaluates `tokens` from scratch and returns the final token's logits (used by parity tests).
    pub fn eval_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::Context("empty token sequence".into()));
        }
        let need_pages = (tokens.len() + PAGE_TOKENS - 1) / PAGE_TOKENS;
        let mut pages = Vec::with_capacity(need_pages);
        for _ in 0..need_pages {
            pages.push(self.pager.alloc_page().ok_or(EngineError::KvExhausted)?);
        }
        let mut res = Ok(());
        for (pos, &tok) in tokens.iter().enumerate() {
            let is_last = pos + 1 == tokens.len();
            if let Err(e) = self.exec.forward(tok, pos, &pages, Out { logits: is_last, hidden: false }) {
                res = Err(e);
                break;
            }
        }
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
        let need_pages = (tokens.len() + PAGE_TOKENS - 1) / PAGE_TOKENS;
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
        let match_slice = if tokens.len() > 1 { &tokens[..tokens.len() - 1] } else { &[] };
        let pm = self.pager.match_prefix(match_slice);
        let cache_n = pm.matched_tokens;
        let mut pages = pm.pages;

        let ensure_page = |pager: &mut RadixPager, pages: &mut Vec<u32>, pos: usize| -> Result<(), EngineError> {
            let need = pos / PAGE_TOKENS + 1;
            while pages.len() < need {
                pages.push(pager.alloc_page().ok_or(EngineError::KvExhausted)?);
            }
            Ok(())
        };

        let t_prefill = Instant::now();
        let prompt_len = tokens.len();
        for pos in cache_n..prompt_len {
            ensure_page(&mut self.pager, &mut pages, pos)?;
            let is_last = pos + 1 == prompt_len;
            if let Err(e) = self.exec.forward(tokens[pos], pos, &pages, Out { logits: is_last, hidden: false }) {
                self.pager.release(&pages);
                return Err(e);
            }
        }
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
        for step in 0..req.max_tokens {
            let next_tok = if let Some(forced) = think_gov.as_mut().and_then(|g| g.forced_next()) {
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

            // Check stop strings
            if req.stop.iter().any(|s| !s.is_empty() && full_text.ends_with(s)) {
                finish = FinishReason::Stop;
                break;
            }
            if !on_chunk(next_tok, &piece) {
                finish = FinishReason::Cancelled;
                break;
            }
            if step + 1 == req.max_tokens || tokens.len() >= self.max_ctx {
                break;
            }
            let pos = tokens.len() - 1;
            ensure_page(&mut self.pager, &mut pages, pos)?;
            if let Err(e) = self.exec.forward(next_tok, pos, &pages, Out { logits: true, hidden: false }) {
                self.pager.release(&pages);
                return Err(e);
            }
            logits_buf.copy_from_slice(self.exec.logits());
        }

        let tail = detok.flush();
        if !tail.is_empty() {
            full_text.push_str(&tail);
            let _ = on_chunk(0, &tail);
        }

        // Cache generated full pages too, then release active references
        self.pager.insert_prefix(&tokens, &pages);
        self.pager.release(&pages);

        let dec_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
        let timings = Timings {
            prompt_n: prompt_len,
            cache_n,
            prompt_ms,
            prompt_per_second: if prompt_ms > 0.0 { prefill_eval_n as f64 / (prompt_ms / 1000.0) } else { 0.0 },
            predicted_n: gen_tokens.len(),
            predicted_ms: dec_ms,
            predicted_per_second: if dec_ms > 0.0 { gen_tokens.len() as f64 / (dec_ms / 1000.0) } else { 0.0 },
        };

        Ok(GenResult { text: full_text, tokens: gen_tokens, finish_reason: finish, timings })
    }
}
