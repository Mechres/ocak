#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("unsupported model: {0}")]
    Unsupported(String),
    #[error("gguf: {0}")]
    Gguf(String),
    #[error("backend: {0}")]
    Backend(String),
    #[error("context overflow: {0}")]
    Context(String),
    #[error("KV cache exhausted")]
    KvExhausted,
}

#[derive(Clone, Copy, Debug)]
pub struct Out {
    pub logits: bool,
    pub hidden: bool,
}

/// One-token-at-a-time transformer/hybrid forward pass with paged KV + recurrent states.
pub trait Executor: Send {
    fn name(&self) -> String;
    /// Resets any recurrent (ShortConv / DeltaNet) layer states before a fresh sequence.
    fn reset_recurrent(&mut self) -> Result<(), EngineError> {
        Ok(())
    }
    /// Processes `token` at position `pos`; `page_table[p]` maps logical page p to a physical page.
    fn forward(&mut self, token: u32, pos: usize, page_table: &[u32], out: Out) -> Result<(), EngineError>;
    /// Processes a contiguous batch of `tokens` starting at `pos0`, returning `out` for the final token.
    fn forward_batch(&mut self, tokens: &[u32], pos0: usize, page_table: &[u32], out: Out) -> Result<(), EngineError> {
        for (i, &tok) in tokens.iter().enumerate() {
            let is_last = i + 1 == tokens.len();
            self.forward(
                tok,
                pos0 + i,
                page_table,
                Out { logits: is_last && out.logits, hidden: is_last && out.hidden },
            )?;
        }
        Ok(())
    }
    /// Processes a precomputed input embedding vector (`[n_embd]`) at position `pos` (for multimodal audio/vision).
    fn forward_embd(&mut self, _embd: &[f32], _pos: usize, _page_table: &[u32], _out: Out) -> Result<(), EngineError> {
        Err(EngineError::Unsupported("forward_embd not implemented for this executor".into()))
    }
    fn logits(&self) -> &[f32];
    /// Final-norm hidden state of the last forward (when requested).
    fn hidden(&self) -> &[f32];
}
