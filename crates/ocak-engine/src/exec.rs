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

/// One active sequence's decode step for continuous multi-slot batching.
#[derive(Clone, Copy, Debug)]
pub struct SlotStep<'a> {
    pub token: u32,
    pub pos: usize,
    pub page_table: &'a [u32],
}

#[cfg(feature = "hip")]
pub struct HipMtpShared {
    pub ctx: std::sync::Arc<ocak_hip::HipContext>,
    pub kernels: std::sync::Arc<ocak_hip::HipKernels>,
    pub iq_kernels: Option<std::sync::Arc<ocak_hip::HipIqKernels>>,
    pub iq_tbl: Option<std::sync::Arc<ocak_hip::HipBuffer>>,
    pub ops: std::sync::Arc<ocak_hip::HipDenseOps>,
    pub hops: std::sync::Arc<ocak_hip::HipHybridOps>,
    pub head_buf: std::sync::Arc<ocak_hip::HipBuffer>,
    pub head_ty: ocak_hip::HipWType,
    pub head_k: u32,
    pub head_m: u32,
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
    /// Fast greedy forward pass that computes argmax on-device and downloads only 4 bytes instead of full logits.
    fn forward_argmax(&mut self, _token: u32, _pos: usize, _page_table: &[u32]) -> Result<Option<u32>, EngineError> {
        Ok(None)
    }
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
    /// Like `forward_batch`, optionally returning the final-norm hidden state for every token in the batch
    /// (used to warm up MTP self-speculative drafters with prompt context).
    fn forward_batch_collect_hiddens(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        page_table: &[u32],
        out: Out,
    ) -> Result<Vec<Vec<f32>>, EngineError> {
        self.forward_batch(tokens, pos0, page_table, out)?;
        Ok(Vec::new())
    }
    /// Runs a 2-token speculative verify batch `[t0, t1]` at `pos0, pos0 + 1`, returning
    /// `Some(([logits0, logits1], [hidden0, hidden1]))` when supported.
    fn forward_verify2(
        &mut self,
        _tokens: [u32; 2],
        _pos0: usize,
        _page_table: &[u32],
    ) -> Result<Option<([Vec<f32>; 2], [Vec<f32>; 2])>, EngineError> {
        Ok(None)
    }
    /// Greedy variant of `forward_verify2`: returns the argmax token of each position (computed
    /// on the device) instead of full logits, plus the hidden states. None if unsupported.
    fn forward_verify2_argmax(
        &mut self,
        _tokens: [u32; 2],
        _pos0: usize,
        _page_table: &[u32],
    ) -> Result<Option<([u32; 2], [Vec<f32>; 2])>, EngineError> {
        Ok(None)
    }
    /// Greedy verification of 2..=4 tokens (current token + up to 3 drafts): argmax and hidden
    /// state after each position; recurrent-state rollback points are kept after every token
    /// but the last (see `restore_recurrent_to`). None if unsupported.
    fn forward_verify_argmax(
        &mut self,
        _tokens: &[u32],
        _pos0: usize,
        _page_table: &[u32],
    ) -> Result<Option<(Vec<u32>, Vec<Vec<f32>>)>, EngineError> {
        Ok(None)
    }
    /// Like `forward_verify_argmax` but returns the full logits of every position (for sampled
    /// decoding). None if unsupported.
    fn forward_verify_logits(
        &mut self,
        _tokens: &[u32],
        _pos0: usize,
        _page_table: &[u32],
    ) -> Result<Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>, EngineError> {
        Ok(None)
    }
    /// Restore the recurrent state saved after token `k` of the last verification.
    fn restore_recurrent_to(&mut self, k: usize) -> Result<(), EngineError> {
        if k == 0 {
            self.restore_recurrent()
        } else {
            Err(EngineError::Unsupported("multi-token rollback".into()))
        }
    }
    /// Runs a continuous multi-slot decode batch across independent sequences (each with its own `pos` and `page_table`),
    /// returning one logits vector per slot when supported.
    fn forward_multi_slot(&mut self, _slots: &[SlotStep<'_>]) -> Result<Option<Vec<Vec<f32>>>, EngineError> {
        Ok(None)
    }
    /// Snapshots recurrent (DeltaNet / ShortConv) states for speculative rollback.
    fn snapshot_recurrent(&mut self) -> Result<(), EngineError> {
        Ok(())
    }
    /// Restores previously snapshotted recurrent states after a speculative rejection.
    fn restore_recurrent(&mut self) -> Result<(), EngineError> {
        Ok(())
    }
    /// Optional shared Vulkan context + lm_head weight for GPU-accelerated MTP drafting.
    #[cfg(feature = "vulkan")]
    fn mtp_gpu_shared(&self) -> Option<(std::sync::Arc<ocak_vk::VkContext>, std::sync::Arc<ocak_vk::Buffer>, ocak_vk::WType, u32, u32)> {
        None
    }
    /// Optional shared HIP context, compiled kernels, and lm_head weight for HIP-accelerated MTP drafting.
    #[cfg(feature = "hip")]
    fn mtp_hip_shared(&self) -> Option<HipMtpShared> {
        None
    }
    /// VRAM the executor allocates lazily on the first MTP verify (state snapshots, logits).
    #[cfg(feature = "hip")]
    fn mtp_hip_verify_bytes(&self) -> u64 {
        0
    }
    /// Processes a precomputed input embedding vector (`[n_embd]`) at position `pos` (for multimodal audio/vision).
    fn forward_embd(&mut self, _embd: &[f32], _pos: usize, _page_table: &[u32], _out: Out) -> Result<(), EngineError> {
        Err(EngineError::Unsupported("forward_embd not implemented for this executor".into()))
    }
    fn logits(&self) -> &[f32];
    /// Final-norm hidden state of the last forward (when requested).
    fn hidden(&self) -> &[f32];
}
