use crate::tokenizer::Tokenizer;

/// Enforces a token budget inside `<think>...</think>` blocks.
pub struct ThinkingGovernor {
    think_open: u32,
    think_close: u32,
    max_tokens: usize,
    in_thinking: bool,
    used: usize,
    forced_done: bool,
}

impl ThinkingGovernor {
    pub fn new(think_open: u32, think_close: u32, max_reasoning_tokens: usize) -> Self {
        Self {
            think_open,
            think_close,
            max_tokens: max_reasoning_tokens,
            in_thinking: false,
            used: 0,
            forced_done: false,
        }
    }

    pub fn observe(&mut self, tok: u32) {
        if tok == self.think_open {
            self.in_thinking = true;
            self.used = 0;
            self.forced_done = false;
        } else if tok == self.think_close {
            self.in_thinking = false;
        } else if self.in_thinking {
            self.used += 1;
        }
    }

    pub fn forced_next(&mut self) -> Option<u32> {
        if self.in_thinking && self.max_tokens > 0 && self.used >= self.max_tokens && !self.forced_done {
            self.forced_done = true;
            self.in_thinking = false;
            return Some(self.think_close);
        }
        None
    }

    pub fn in_thinking(&self) -> bool {
        self.in_thinking
    }
}

/// Pushdown byte automaton validating JSON syntax (RFC 8259).
#[derive(Debug, Clone)]
pub struct JsonGrammar {
    stack: Vec<Frame>,
    state: State,
    /// Last vocabulary mask (bit set = token allowed) and the exact grammar state it was built
    /// for: inside strings / numbers consecutive tokens usually see the same state, so masking is
    /// a bitset apply instead of stepping every token's bytes.
    mask_cache: std::cell::RefCell<Option<MaskCache>>,
}

#[derive(Debug, Clone)]
struct MaskCache {
    stack: Vec<Frame>,
    state: State,
    bits: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Frame {
    ObjectKeyOrEnd,
    ObjectKeyOnly,
    ObjectColon,
    ObjectCommaOrEnd,
    ArrayValueOrEnd,
    ArrayCommaOrEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    ExpectValue,
    /// `cont`: UTF-8 continuation bytes still expected, the next one in `lo..=hi` (rejects
    /// malformed, overlong and surrogate sequences).
    InString { for_key: bool, escape: bool, hex_rem: u8, cont: u8, lo: u8, hi: u8 },
    /// `lead_zero`: the integer part is exactly "0" so far (no further digit allowed).
    InNumber { need_digit: bool, seen_dot: bool, seen_exp: bool, exp_sign_ok: bool, lead_zero: bool },
    InLiteral(&'static [u8], usize),
    AfterValue,
}

impl Default for JsonGrammar {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGrammar {
    pub fn new() -> Self {
        Self { stack: Vec::new(), state: State::ExpectValue, mask_cache: std::cell::RefCell::new(None) }
    }

    pub fn is_complete(&self) -> bool {
        if !self.stack.is_empty() {
            return false;
        }
        matches!(
            self.state,
            State::AfterValue | State::InNumber { need_digit: false, .. }
        )
    }

    pub fn feed_byte(&mut self, b: u8) -> bool {
        // Roll back on rejection without cloning the mask cache.
        let (stack, state) = (self.stack.clone(), self.state);
        if self.step(b) {
            true
        } else {
            self.stack = stack;
            self.state = state;
            false
        }
    }

    fn step(&mut self, b: u8) -> bool {
        match self.state {
            State::InString { for_key, escape, hex_rem, cont, lo, hi } => {
                if cont > 0 {
                    if b < lo || b > hi {
                        return false;
                    }
                    self.state = State::InString { for_key, escape: false, hex_rem: 0, cont: cont - 1, lo: 0x80, hi: 0xBF };
                    return true;
                }
                if hex_rem > 0 {
                    if !b.is_ascii_hexdigit() {
                        return false;
                    }
                    self.state = State::InString { for_key, escape: false, hex_rem: hex_rem - 1, cont: 0, lo: 0x80, hi: 0xBF };
                    return true;
                }
                if escape {
                    if b == b'u' {
                        self.state = State::InString { for_key, escape: false, hex_rem: 4, cont: 0, lo: 0x80, hi: 0xBF };
                        return true;
                    }
                    if matches!(b, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') {
                        self.state = State::InString { for_key, escape: false, hex_rem: 0, cont: 0, lo: 0x80, hi: 0xBF };
                        return true;
                    }
                    return false;
                }
                if b == b'\\' {
                    self.state = State::InString { for_key, escape: true, hex_rem: 0, cont: 0, lo: 0x80, hi: 0xBF };
                    return true;
                }
                if b == b'"' {
                    if for_key {
                        self.stack.push(Frame::ObjectColon);
                        self.state = State::AfterValue;
                    } else {
                        self.state = State::AfterValue;
                    }
                    return true;
                }
                if b < 0x80 {
                    return b >= 0x20;
                }
                // UTF-8 lead byte: number of continuation bytes and the range of the first one.
                let (n, lo, hi) = match b {
                    0xC2..=0xDF => (1, 0x80, 0xBF),
                    0xE0 => (2, 0xA0, 0xBF),
                    0xED => (2, 0x80, 0x9F),
                    0xE1..=0xEF => (2, 0x80, 0xBF),
                    0xF0 => (3, 0x90, 0xBF),
                    0xF1..=0xF3 => (3, 0x80, 0xBF),
                    0xF4 => (3, 0x80, 0x8F),
                    _ => return false,
                };
                self.state = State::InString { for_key, escape: false, hex_rem: 0, cont: n, lo, hi };
                true
            }
            State::InLiteral(lit, idx) => {
                if b != lit[idx] {
                    return false;
                }
                if idx + 1 == lit.len() {
                    self.state = State::AfterValue;
                } else {
                    self.state = State::InLiteral(lit, idx + 1);
                }
                true
            }
            State::InNumber { need_digit, seen_dot, seen_exp, exp_sign_ok, lead_zero } => {
                if b.is_ascii_digit() {
                    if lead_zero {
                        return false; // no leading zeros ("01")
                    }
                    // First integer digit (after '-'): a '0' must stand alone.
                    let first_int = need_digit && !seen_dot && !seen_exp;
                    self.state = State::InNumber { need_digit: false, seen_dot, seen_exp, exp_sign_ok: false, lead_zero: first_int && b == b'0' };
                    return true;
                }
                if !need_digit && !seen_dot && !seen_exp && b == b'.' {
                    self.state = State::InNumber { need_digit: true, seen_dot: true, seen_exp: false, exp_sign_ok: false, lead_zero: false };
                    return true;
                }
                if !need_digit && !seen_exp && (b == b'e' || b == b'E') {
                    self.state = State::InNumber { need_digit: true, seen_dot, seen_exp: true, exp_sign_ok: true, lead_zero: false };
                    return true;
                }
                if exp_sign_ok && (b == b'+' || b == b'-') {
                    self.state = State::InNumber { need_digit: true, seen_dot, seen_exp, exp_sign_ok: false, lead_zero: false };
                    return true;
                }
                if need_digit {
                    return false;
                }
                // Number ended; transition to AfterValue and re-process `b`
                self.state = State::AfterValue;
                self.step(b)
            }
            State::ExpectValue => {
                if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                    return true;
                }
                if let Some(&Frame::ArrayValueOrEnd) = self.stack.last() {
                    if b == b']' {
                        self.stack.pop();
                        self.state = State::AfterValue;
                        return true;
                    }
                    *self.stack.last_mut().unwrap() = Frame::ArrayCommaOrEnd;
                }
                match b {
                    b'{' => {
                        self.stack.push(Frame::ObjectKeyOrEnd);
                        self.state = State::AfterValue;
                        true
                    }
                    b'[' => {
                        self.stack.push(Frame::ArrayValueOrEnd);
                        self.state = State::ExpectValue;
                        true
                    }
                    b'"' => {
                        self.state = State::InString { for_key: false, escape: false, hex_rem: 0, cont: 0, lo: 0x80, hi: 0xBF };
                        true
                    }
                    b'-' => {
                        self.state = State::InNumber { need_digit: true, seen_dot: false, seen_exp: false, exp_sign_ok: false, lead_zero: false };
                        true
                    }
                    b'0'..=b'9' => {
                        self.state = State::InNumber { need_digit: false, seen_dot: false, seen_exp: false, exp_sign_ok: false, lead_zero: b == b'0' };
                        true
                    }
                    b't' => { self.state = State::InLiteral(b"true", 1); true }
                    b'f' => { self.state = State::InLiteral(b"false", 1); true }
                    b'n' => { self.state = State::InLiteral(b"null", 1); true }
                    _ => false,
                }
            }
            State::AfterValue => {
                if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                    return true;
                }
                match self.stack.last().copied() {
                    None => false, // trailing non-whitespace garbage
                    Some(Frame::ObjectKeyOrEnd) => {
                        if b == b'}' {
                            self.stack.pop();
                            self.state = State::AfterValue;
                            true
                        } else if b == b'"' {
                            *self.stack.last_mut().unwrap() = Frame::ObjectCommaOrEnd;
                            self.state = State::InString { for_key: true, escape: false, hex_rem: 0, cont: 0, lo: 0x80, hi: 0xBF };
                            true
                        } else {
                            false
                        }
                    }
                    Some(Frame::ObjectKeyOnly) => {
                        if b == b'"' {
                            *self.stack.last_mut().unwrap() = Frame::ObjectCommaOrEnd;
                            self.state = State::InString { for_key: true, escape: false, hex_rem: 0, cont: 0, lo: 0x80, hi: 0xBF };
                            true
                        } else {
                            false
                        }
                    }
                    Some(Frame::ObjectColon) => {
                        if b == b':' {
                            self.stack.pop();
                            self.state = State::ExpectValue;
                            true
                        } else {
                            false
                        }
                    }
                    Some(Frame::ObjectCommaOrEnd) => {
                        if b == b'}' {
                            self.stack.pop();
                            self.state = State::AfterValue;
                            true
                        } else if b == b',' {
                            *self.stack.last_mut().unwrap() = Frame::ObjectKeyOnly;
                            self.state = State::AfterValue;
                            true
                        } else {
                            false
                        }
                    }
                    Some(Frame::ArrayCommaOrEnd) => {
                        if b == b']' {
                            self.stack.pop();
                            self.state = State::AfterValue;
                            true
                        } else if b == b',' {
                            self.state = State::ExpectValue;
                            true
                        } else {
                            false
                        }
                    }
                    Some(Frame::ArrayValueOrEnd) => false,
                }
            }
        }
    }

    pub fn mask_logits(&self, tok: &Tokenizer, logits: &mut [f32]) {
        let n = logits.len();
        let mut cache = self.mask_cache.borrow_mut();
        let hit = matches!(&*cache, Some(c) if c.state == self.state && c.stack == self.stack && c.bits.len() == n.div_ceil(64));
        if !hit {
            *cache = Some(MaskCache { stack: self.stack.clone(), state: self.state, bits: self.build_mask(tok, n) });
        }
        let bits = &cache.as_ref().unwrap().bits;
        for (i, l) in logits.iter_mut().enumerate() {
            if (bits[i >> 6] >> (i & 63)) & 1 == 0 {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// Allowed-token bitset for the current state over a vocabulary of `n` ids.
    pub(crate) fn build_mask(&self, tok: &Tokenizer, n: usize) -> Vec<u64> {
        let mut bits = vec![0u64; n.div_ceil(64)];
        let complete = self.is_complete();
        let mut g = JsonGrammar { stack: Vec::with_capacity(self.stack.len() + 8), state: self.state, mask_cache: Default::default() };
        let reset = |g: &mut JsonGrammar| {
            g.stack.clear();
            g.stack.extend_from_slice(&self.stack);
            g.state = self.state;
        };
        // First bytes the current state accepts: most tokens fail on their first byte outside
        // strings, which is then one table lookup instead of a state reset + step.
        let mut first_ok = [false; 256];
        for (b, ok) in first_ok.iter_mut().enumerate() {
            reset(&mut g);
            *ok = g.step(b as u8);
        }
        // Inside a string (no pending escape or UTF-8 sequence) an ASCII byte >= 0x20 other than '"'
        // and '\\' is accepted and leaves the state unchanged, so such tokens need no stepping.
        let plain_string = matches!(self.state, State::InString { escape: false, hex_rem: 0, cont: 0, .. });
        // JSON whitespace that leaves the state unchanged here (between values): skip a token's
        // leading whitespace and test its first other byte (~40% of BPE tokens start with ' ').
        let mut ws_noop = [false; 256];
        for b in *b" \t\n\r" {
            reset(&mut g);
            ws_noop[b as usize] = g.step(b) && g.state == self.state && g.stack == self.stack;
        }
        let (data, offs) = tok.token_bytes_table();
        let n_tab = n.min(offs.len().saturating_sub(1));
        for i in 0..n_tab {
            let ok = {
                let full = &data[offs[i] as usize..offs[i + 1] as usize];
                let lead = full.iter().take_while(|&&b| ws_noop[b as usize]).count();
                let bytes = &full[lead..];
                if full.is_empty() {
                    false
                } else if bytes.is_empty() {
                    true // whitespace only, all no-ops here
                } else if !first_ok[bytes[0] as usize] {
                    false
                } else if plain_string && bytes.iter().all(|&b| (0x20..0x80).contains(&b) && b != b'"' && b != b'\\') {
                    true
                } else {
                    // One reset of a scratch copy per token (no allocation); `step` mutates it in
                    // place (a rejected token's state is dropped, so no per-byte rollback).
                    reset(&mut g);
                    bytes.iter().all(|&b| g.step(b))
                }
            };
            if ok {
                bits[i >> 6] |= 1u64 << (i & 63);
            }
        }
        // End-of-generation tokens: allowed exactly when the document is complete (set once here
        // instead of a hash lookup per vocabulary entry).
        for id in tok.eog_ids() {
            let i = id as usize;
            if i < n {
                if complete {
                    bits[i >> 6] |= 1u64 << (i & 63);
                } else {
                    bits[i >> 6] &= !(1u64 << (i & 63));
                }
            }
        }
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::JsonGrammar;

    fn accepts(doc: &[u8]) -> bool {
        let mut g = JsonGrammar::new();
        doc.iter().all(|&b| g.feed_byte(b)) && g.is_complete()
    }

    #[test]
    fn json_valid_documents() {
        for d in [
            &b"{\"a\": [1, -0.5, 0, -0, 10, 1e5, 2E-3, true, false, null], \"b\": {}}"[..],
            b"\"\\u00e9\\n\"",
            "\"caf\u{e9} \u{65e5}\u{672c} \u{1f600}\"".as_bytes(),
            b" \t\r\n[ ]",
        ] {
            assert!(accepts(d), "rejected {:?}", String::from_utf8_lossy(d));
        }
    }

    #[test]
    fn json_invalid_documents() {
        for d in [
            &b"01"[..],
            b"-01",
            b"[00]",
            b"\x0c{}",
            b"[1,\x0c2]",
            b"\"\xff\"",
            b"\"\xc0\xaf\"",         // overlong '/'
            b"\"\xed\xa0\x80\"",     // UTF-16 surrogate
            b"\"\xe2\x82\"",         // truncated sequence
            b"\"\x80\"",             // stray continuation byte
        ] {
            assert!(!accepts(d), "accepted {:?}", d);
        }
    }
}
