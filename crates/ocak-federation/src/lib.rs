//! Federation bridges: libllama parity oracle, managed llama-server bridge (Day-1 exotic/ASR),
//! Mergen phone-cluster bridge, Strata staged-RAM bridge, and dynamic HIP runtime probe.
pub mod bridges;
pub mod oracle;

pub use bridges::{
    hip_probe, HipProbe, LlamaBridgeHandle, ManagedLlamaBridge, MergenBridge, StrataBridge,
};
pub use oracle::{cosine_similarity, eval_with_llama_oracle, OracleError};
