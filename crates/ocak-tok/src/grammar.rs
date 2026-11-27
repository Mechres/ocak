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
    InString { for_key: bool, escape: bool, hex_rem: u8 },
    InNumber { need_digit: bool, seen_dot: bool, seen_exp: bool, exp_sign_ok: bool },
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
            State::InString { for_key, escape, hex_rem } => {
                if hex_rem > 0 {
                    if !b.is_ascii_hexdigit() {
                        return false;
                    }
                    self.state = State::InString { for_key, escape: false, hex_rem: hex_rem - 1 };
                    return true;
                }
                if escape {
                    if b == b'u' {
                        self.state = State::InString { for_key, escape: false, hex_rem: 4 };
                        return true;
                    }
                    if matches!(b, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') {
                        self.state = State::InString { for_key, escape: false, hex_rem: 0 };
                        return true;
                    }
                    return false;
                }
                if b == b'\\' {
                    self.state = State::InString { for_key, escape: true, hex_rem: 0 };
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
                b >= 0x20
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
            State::InNumber { need_digit, seen_dot, seen_exp, exp_sign_ok } => {
                if b.is_ascii_digit() {
                    self.state = State::InNumber { need_digit: false, seen_dot, seen_exp, exp_sign_ok: false };
                    return true;
                }
                if !need_digit && !seen_dot && !seen_exp && b == b'.' {
                    self.state = State::InNumber { need_digit: true, seen_dot: true, seen_exp: false, exp_sign_ok: false };
                    return true;
                }
                if !need_digit && !seen_exp && (b == b'e' || b == b'E') {
                    self.state = State::InNumber { need_digit: true, seen_dot, seen_exp: true, exp_sign_ok: true };
                    return true;
                }
                if exp_sign_ok && (b == b'+' || b == b'-') {
                    self.state = State::InNumber { need_digit: true, seen_dot, seen_exp, exp_sign_ok: false };
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
                if b.is_ascii_whitespace() {
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
                        self.state = State::InString { for_key: false, escape: false, hex_rem: 0 };
                        true
                    }
                    b'-' => {
                        self.state = State::InNumber { need_digit: true, seen_dot: false, seen_exp: false, exp_sign_ok: false };
                        true
                    }
                    b'0'..=b'9' => {
                        self.state = State::InNumber { need_digit: false, seen_dot: false, seen_exp: false, exp_sign_ok: false };
                        true
                    }
                    b't' => { self.state = State::InLiteral(b"true", 1); true }
                    b'f' => { self.state = State::InLiteral(b"false", 1); true }
                    b'n' => { self.state = State::InLiteral(b"null", 1); true }
                    _ => false,
                }
            }
            State::AfterValue => {
                if b.is_ascii_whitespace() {
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
                            self.state = State::InString { for_key: true, escape: false, hex_rem: 0 };
                            true
                        } else {
                            false
                        }
                    }
                    Some(Frame::ObjectKeyOnly) => {
                        if b == b'"' {
                            *self.stack.last_mut().unwrap() = Frame::ObjectCommaOrEnd;
                            self.state = State::InString { for_key: true, escape: false, hex_rem: 0 };
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
    fn build_mask(&self, tok: &Tokenizer, n: usize) -> Vec<u64> {
        let mut bits = vec![0u64; n.div_ceil(64)];
        let complete = self.is_complete();
        let mut g = JsonGrammar { stack: Vec::with_capacity(self.stack.len() + 8), state: self.state, mask_cache: Default::default() };
        for i in 0..n {
            let id = i as u32;
            let ok = if tok.is_eog(id) {
                complete
            } else {
                let bytes = tok.token_bytes_ref(id);
                // One reset of a scratch copy per token (no allocation); `step` mutates it in place
                // (a rejected token's state is dropped, so no per-byte rollback is needed).
                g.stack.clear();
                g.stack.extend_from_slice(&self.stack);
                g.state = self.state;
                !bytes.is_empty() && bytes.iter().all(|&b| g.step(b))
            };
            if ok {
                bits[i >> 6] |= 1u64 << (i & 63);
            }
        }
        bits
    }
}
