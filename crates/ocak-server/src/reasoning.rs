//! Reasoning vs final-answer splitting for OpenAI (`reasoning_content`) and Anthropic (`thinking` blocks).
//! Handles `<think>...</think>` (Qwen3.5, Xing) and `<|content_thinking|>...<|content_text|>` (Inkling).

/// Splits a completed model output into `(reasoning, content)`.
/// `started_in_think` is true when the prompt itself opened `<think>` (e.g. Xing with thinking enabled).
pub fn split_reasoning(raw: &str, started_in_think: bool) -> (Option<String>, String) {
    // 1. Inkling markers: <|content_thinking|> ... <|content_text|> ...
    if raw.contains("<|content_thinking|>") || raw.contains("<|content_text|>") {
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut in_think = false;
        let mut rest = raw;
        while !rest.is_empty() {
            let next_t = rest.find("<|content_thinking|>");
            let next_c = rest.find("<|content_text|>");
            match (next_t, next_c) {
                (Some(ti), Some(ci)) if ti < ci => {
                    let seg = &rest[..ti];
                    if in_think { reasoning.push_str(seg); } else { content.push_str(seg); }
                    in_think = true;
                    rest = &rest[ti + "<|content_thinking|>".len()..];
                }
                (_, Some(ci)) => {
                    let seg = &rest[..ci];
                    if in_think { reasoning.push_str(seg); } else { content.push_str(seg); }
                    in_think = false;
                    rest = &rest[ci + "<|content_text|>".len()..];
                }
                (Some(ti), None) => {
                    let seg = &rest[..ti];
                    if in_think { reasoning.push_str(seg); } else { content.push_str(seg); }
                    in_think = true;
                    rest = &rest[ti + "<|content_thinking|>".len()..];
                }
                (None, None) => {
                    if in_think { reasoning.push_str(rest); } else { content.push_str(rest); }
                    break;
                }
            }
        }
        let r = reasoning.trim();
        let r_opt = if r.is_empty() { None } else { Some(r.to_string()) };
        return (r_opt, content.trim_start().to_string());
    }

    // 2. Standard <think> ... </think> (Qwen3.5 & Xing)
    if let Some(end_idx) = raw.find("</think>") {
        let before = &raw[..end_idx];
        let after = &raw[end_idx + "</think>".len()..];
        let before_clean = before.trim_start().strip_prefix("<think>").unwrap_or(before).trim();
        let r_opt = if before_clean.is_empty() { None } else { Some(before_clean.to_string()) };
        return (r_opt, after.trim_start().to_string());
    }

    if started_in_think {
        let r = raw.trim_start().strip_prefix("<think>").unwrap_or(raw).trim();
        let r_opt = if r.is_empty() { None } else { Some(r.to_string()) };
        return (r_opt, String::new());
    }

    if let Some(after_open) = raw.trim_start().strip_prefix("<think>") {
        let r = after_open.trim();
        let r_opt = if r.is_empty() { None } else { Some(r.to_string()) };
        return (r_opt, String::new());
    }

    (None, raw.to_string())
}

const TAGS: &[&str] = &[
    "<think>",
    "</think>",
    "<|content_thinking|>",
    "<|content_text|>",
];

/// Incremental streaming splitter emitting `(reasoning_delta, content_delta)` chunks.
pub struct StreamReasoningSplitter {
    in_think: bool,
    buf: String,
    trim_leading_content_ws: bool,
}

impl StreamReasoningSplitter {
    pub fn new(started_in_think: bool) -> Self {
        Self {
            in_think: started_in_think,
            buf: String::new(),
            trim_leading_content_ws: false,
        }
    }

    pub fn push(&mut self, piece: &str) -> (String, String) {
        self.buf.push_str(piece);
        self.drain(false)
    }

    pub fn flush(&mut self) -> (String, String) {
        self.drain(true)
    }

    fn drain(&mut self, eof: bool) -> (String, String) {
        let mut out_r = String::new();
        let mut out_c = String::new();
        loop {
            // Find earliest complete tag in self.buf
            let mut best: Option<(usize, &str)> = None;
            for &tag in TAGS {
                if let Some(pos) = self.buf.find(tag) {
                    if best.is_none_or(|(bp, _)| pos < bp) {
                        best = Some((pos, tag));
                    }
                }
            }
            if let Some((pos, tag)) = best {
                let seg: String = self.buf.drain(..pos).collect();
                self.emit_seg(&seg, &mut out_r, &mut out_c);
                match tag {
                    "<think>" | "<|content_thinking|>" => {
                        self.in_think = true;
                    }
                    "</think>" | "<|content_text|>" => {
                        self.in_think = false;
                        self.trim_leading_content_ws = true;
                    }
                    _ => {}
                }
                self.buf.drain(..tag.len());
                continue;
            }
            if eof {
                let rem = std::mem::take(&mut self.buf);
                self.emit_seg(&rem, &mut out_r, &mut out_c);
                break;
            }
            // Hold back any suffix that could be a prefix of a tag
            let keep = self.partial_tag_suffix_len();
            let emit_len = self.buf.len().saturating_sub(keep);
            if emit_len > 0 {
                // Ensure UTF-8 char boundary
                let mut safe = emit_len;
                while safe > 0 && !self.buf.is_char_boundary(safe) {
                    safe -= 1;
                }
                if safe > 0 {
                    let seg: String = self.buf.drain(..safe).collect();
                    self.emit_seg(&seg, &mut out_r, &mut out_c);
                }
            }
            break;
        }
        (out_r, out_c)
    }

    fn emit_seg(&mut self, seg: &str, out_r: &mut String, out_c: &mut String) {
        if seg.is_empty() {
            return;
        }
        if self.in_think {
            out_r.push_str(seg);
        } else if self.trim_leading_content_ws {
            let trimmed = seg.trim_start();
            if !trimmed.is_empty() {
                self.trim_leading_content_ws = false;
                out_c.push_str(trimmed);
            }
        } else {
            out_c.push_str(seg);
        }
    }

    fn partial_tag_suffix_len(&self) -> usize {
        let mut max_keep = 0usize;
        for &tag in TAGS {
            let limit = (tag.len() - 1).min(self.buf.len());
            for len in 1..=limit {
                if self.buf.ends_with(&tag[..len]) {
                    max_keep = max_keep.max(len);
                }
            }
        }
        max_keep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_qwen_xing_and_inkling_reasoning() {
        // Qwen3.5 empty think leak
        assert_eq!(split_reasoning("<think>\n\n</think>\n\nParis", false), (None, "Paris".into()));
        // Qwen3.5 real think
        assert_eq!(
            split_reasoning("<think>\nFrance -> Paris\n</think>\n\nParis", false),
            (Some("France -> Paris".into()), "Paris".into())
        );
        // Xing (started inside <think>)
        assert_eq!(
            split_reasoning("Thinking...\n</think>\nParis", true),
            (Some("Thinking...".into()), "Paris".into())
        );
        // Inkling
        assert_eq!(
            split_reasoning("<|content_thinking|>Need one word.<|content_text|>Paris", false),
            (Some("Need one word.".into()), "Paris".into())
        );
        assert_eq!(split_reasoning("<|content_text|>Paris", false), (None, "Paris".into()));

        // Streaming splitter across arbitrary chunk boundaries
        let mut sp = StreamReasoningSplitter::new(false);
        let mut r_all = String::new();
        let mut c_all = String::new();
        for ch in ["<|content_", "thinking|>Step 1", "<|content_te", "xt|>\nParis"] {
            let (r, c) = sp.push(ch);
            r_all.push_str(&r);
            c_all.push_str(&c);
        }
        let (r, c) = sp.flush();
        r_all.push_str(&r);
        c_all.push_str(&c);
        assert_eq!(r_all, "Step 1");
        assert_eq!(c_all, "Paris");
    }
}
