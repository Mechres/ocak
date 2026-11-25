//! GGUF tokenizer (SentencePiece + byte-level BPE), chat templates, thinking budget, and JSON grammar.
pub mod grammar;
pub mod template;
pub mod tokenizer;

pub use grammar::{JsonGrammar, ThinkingGovernor};
pub use template::{ChatMessage, TemplateFamily};
pub use tokenizer::{IncrementalDetokenizer, TokError, Tokenizer};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn incremental_detok_multibyte() {
        let mut d = IncrementalDetokenizer::new();
        let s = "Merhaba dünya 🚀!";
        let bytes = s.as_bytes();
        let mut out = String::new();
        for &b in bytes {
            out.push_str(&d.push(&[b]));
        }
        out.push_str(&d.flush());
        assert_eq!(out, s);
    }

    #[test]
    fn thinking_governor_forces_close() {
        let mut g = ThinkingGovernor::new(10, 11, 3);
        g.observe(10);
        assert!(g.in_thinking());
        g.observe(1); assert_eq!(g.forced_next(), None);
        g.observe(2); assert_eq!(g.forced_next(), None);
        g.observe(3); assert_eq!(g.forced_next(), Some(11));
        assert!(!g.in_thinking());
    }

    #[test]
    fn json_grammar_valid_and_invalid() {
        for ok in [
            r#"{"a": 1, "b": [true, false, null, -3.14e+2, "hi\n\"\\"]}"#,
            r#"{}"#,
            r#"[]"#,
            r#"{"nested": {"k": "v"}}"#,
        ] {
            let mut g = JsonGrammar::new();
            assert!(ok.bytes().all(|b| g.feed_byte(b)), "failed on {ok}");
            assert!(g.is_complete(), "not complete on {ok}");
        }
        for bad in [r#"{"a":}"#, r#"[1,]"#, r#"{"a":1,}"#, r#"{"a":1} extra"#] {
            let mut g = JsonGrammar::new();
            let all_ok = bad.bytes().all(|b| g.feed_byte(b));
            assert!(!all_ok || !g.is_complete(), "should reject {bad}");
        }
    }

    #[test]
    fn tinyllama_real_gguf_tokenizer() {
        let p = Path::new("/home/mechres/Projeler/mergen/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
        if !p.exists() { return; }
        let gf = ocak_gguf::GgufFile::open(p).unwrap();
        let tok = Tokenizer::from_gguf(&gf).unwrap();
        assert_eq!(tok.encode("Hello world", true, false), vec![1, 15043, 3186]);
        let rt = "The capital of France is Paris! 🚀";
        let ids = tok.encode(rt, true, false);
        assert_eq!(tok.decode(&ids), rt);
        let fam = template::detect_family(tok.chat_template());
        assert_eq!(fam, TemplateFamily::Zephyr);
        let rendered = template::render(fam, &[ChatMessage { role: "user".into(), content: "Hi".into() }], true, true);
        assert!(rendered.contains("<|user|>\nHi</s>\n<|assistant|>\n"));
    }

    #[test]
    fn qwen35_real_gguf_tokenizer() {
        let p = Path::new("/home/mechres/models/Qwen3.5-4B-Q4_K_M.gguf");
        if !p.exists() { return; }
        let gf = ocak_gguf::GgufFile::open(p).unwrap();
        let tok = Tokenizer::from_gguf(&gf).unwrap();
        let s = "Hello 123, dünya!\nNext line.";
        let ids = tok.encode(s, false, false);
        assert_eq!(tok.decode(&ids), s);
        let sp = "<|im_start|>user\nhi<|im_end|>";
        let sp_ids = tok.encode(sp, false, true);
        assert_eq!(sp_ids.first().copied(), tok.token_id("<|im_start|>"));
        assert_eq!(sp_ids.last().copied(), tok.token_id("<|im_end|>"));
    }
}
