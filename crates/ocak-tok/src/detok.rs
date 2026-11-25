/// Turns a stream of raw token bytes into text, emitting only complete UTF-8 sequences.
#[derive(Debug, Default, Clone)]
pub struct IncrementalDetokenizer {
    pending: Vec<u8>,
}

impl IncrementalDetokenizer {
    pub fn new() -> Self { Self::default() }

    /// Appends bytes and returns every complete character that is now available.
    /// A trailing incomplete sequence is buffered; invalid bytes become U+FFFD.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut consumed = 0;
        loop {
            match std::str::from_utf8(&self.pending[consumed..]) {
                Ok(s) => {
                    out.push_str(s);
                    consumed = self.pending.len();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // SAFETY-free: the prefix is valid by contract of valid_up_to.
                    out.push_str(std::str::from_utf8(&self.pending[consumed..consumed + valid]).unwrap());
                    consumed += valid;
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{FFFD}');
                            consumed += n;
                        }
                        None => break, // incomplete tail: keep for next push
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        out
    }

    /// Emits whatever is buffered (lossily) and resets.
    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }

    pub fn has_pending(&self) -> bool { !self.pending.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_multibyte_across_pushes() {
        let text = "aé😀日";
        let bytes = text.as_bytes();
        let mut d = IncrementalDetokenizer::new();
        let mut out = String::new();
        for b in bytes {
            let s = d.push(&[*b]);
            out.push_str(&s);
        }
        assert!(!d.has_pending());
        assert_eq!(out, text);
        assert_eq!(d.flush(), "");
    }

    #[test]
    fn buffers_partial_and_flushes_lossy() {
        let mut d = IncrementalDetokenizer::new();
        assert_eq!(d.push(&[0xF0, 0x9F]), "");
        assert!(d.has_pending());
        assert_eq!(d.push(&[0x98]), "");
        assert_eq!(d.push(&[0x80, b'x']), "😀x");
        assert_eq!(d.push(&[0xE6, 0x97]), "");
        assert_eq!(d.flush(), "\u{FFFD}");
    }

    #[test]
    fn invalid_bytes_become_replacement() {
        let mut d = IncrementalDetokenizer::new();
        assert_eq!(d.push(&[b'a', 0xFF, b'b']), "a\u{FFFD}b");
        // lead byte followed by a non-continuation
        assert_eq!(d.push(&[0xC3]), "");
        assert_eq!(d.push(&[b'z']), "\u{FFFD}z");
    }
}
