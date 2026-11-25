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
    fn logits(&self) -> &[f32];
    /// Final-norm hidden state of the last forward (when requested).
    fn hidden(&self) -> &[f32];
}
