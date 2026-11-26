use ocak_gguf::GgufFile;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Debug, thiserror::Error)]
pub enum TokError {
    #[error("missing GGUF key: {0}")]
    Missing(String),
    #[error("unsupported tokenizer model: {0}")]
    Unsupported(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VocabKind {
    Spm,
    Bpe,
}

pub struct Tokenizer {
    kind: VocabKind,
    tokens: Vec<String>,
    scores: Vec<f32>,
    token_types: Vec<i32>,
    tok_to_id: FxHashMap<String, u32>,
    merges: FxHashMap<(String, String), usize>,
    specials: Vec<(String, u32)>,
    byte_tokens: [Option<u32>; 256],
    bos: Option<u32>,
    eos: Option<u32>,
    eog: FxHashSet<u32>,
    add_bos: bool,
    add_space_prefix: bool,
    llama3_pre: bool,
    chat_template: Option<String>,
}

fn gpt2_byte_to_char() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..256u32 {
        let printable = (b >= b'!' as u32 && b <= b'~' as u32)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b);
        let cp = if printable { b } else { let v = next; next += 1; v };
        map[b as usize] = char::from_u32(cp).unwrap();
    }
    map
}

fn gpt2_char_to_byte() -> FxHashMap<char, u8> {
    let m = gpt2_byte_to_char();
    let mut rev = FxHashMap::default();
    for (b, c) in m.iter().enumerate() {
        rev.insert(*c, b as u8);
    }
    rev
}

impl Tokenizer {
    pub fn from_gguf(g: &GgufFile) -> Result<Self, TokError> {
        let model = g.get_str("tokenizer.ggml.model").unwrap_or("llama");
        let kind = match model {
            "llama" | "spm" => VocabKind::Spm,
            "gpt2" | "bpe" => VocabKind::Bpe,
            other => return Err(TokError::Unsupported(other.to_string())),
        };
        let tokens = g
            .strings("tokenizer.ggml.tokens")
            .ok_or_else(|| TokError::Missing("tokenizer.ggml.tokens".into()))?
            .clone();
        let n = tokens.len();
        let scores: Vec<f32> = g
            .floats("tokenizer.ggml.scores")
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .unwrap_or_else(|| vec![0.0; n]);
        let token_types: Vec<i32> = g
            .ints("tokenizer.ggml.token_type")
            .map(|v| v.iter().map(|&x| x as i32).collect())
            .unwrap_or_else(|| vec![1; n]);

        let mut tok_to_id = FxHashMap::default();
        let mut specials = Vec::new();
        let mut byte_tokens = [None; 256];
        for (i, tok) in tokens.iter().enumerate() {
            let id = i as u32;
            tok_to_id.entry(tok.clone()).or_insert(id);
            let tt = token_types.get(i).copied().unwrap_or(1);
            if (tt == 3 || tt == 4) && !tok.is_empty() {
                specials.push((tok.clone(), id));
            }
            if tt == 6 && tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                    byte_tokens[b as usize] = Some(id);
                }
            }
        }
        // Longest-first for greedy special-token matching
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));

        let mut merges = FxHashMap::default();
        if let Some(ms) = g.strings("tokenizer.ggml.merges") {
            for (rank, m) in ms.iter().enumerate() {
                if let Some((a, b)) = m.split_once(' ') {
                    merges.insert((a.to_string(), b.to_string()), rank);
                }
            }
        }

        let bos = g.get_u64("tokenizer.ggml.bos_token_id").map(|v| v as u32);
        let eos = g.get_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32);
        let mut eog = FxHashSet::default();
        if let Some(id) = eos {
            eog.insert(id);
        }
        for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
            if let Some(id) = g.get_u64(key) {
                eog.insert(id as u32);
            }
        }
        for txt in ["</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"] {
            if let Some(&id) = tok_to_id.get(txt) {
                eog.insert(id);
            }
        }

        let add_bos = g
            .get_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(kind == VocabKind::Spm);
        let add_space_prefix = g
            .get_bool("tokenizer.ggml.add_space_prefix")
            .unwrap_or(kind == VocabKind::Spm);
        let pre = g.get_str("tokenizer.ggml.pre").unwrap_or("");
        let llama3_pre = pre.contains("llama3")
            || pre.contains("llama-bpe")
            || pre.contains("minicpm")
            || pre.contains("inkling")
            || pre.contains("lfm");
        let chat_template = g.get_str("tokenizer.chat_template").map(|s| s.to_string());

        Ok(Self {
            kind,
            tokens,
            scores,
            token_types,
            tok_to_id,
            merges,
            specials,
            byte_tokens,
            bos,
            eos,
            eog,
            add_bos,
            add_space_prefix,
            llama3_pre,
            chat_template,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }
    pub fn bos(&self) -> Option<u32> {
        self.bos
    }
    pub fn eos(&self) -> Option<u32> {
        self.eos
    }
    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }
    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.tok_to_id.get(text).copied()
    }

    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        if text.is_empty() {
            return out;
        }
        if !parse_special || self.specials.is_empty() {
            self.encode_fragment(text, true, &mut out);
            return out;
        }
        // Split on control / user_defined tokens longest-first
        let mut pos = 0;
        let mut seg_start = 0;
        let mut first_fragment = true;
        while pos < text.len() {
            let rest = &text[pos..];
            let mut matched: Option<(usize, u32)> = None;
            for (sp, id) in &self.specials {
                if rest.starts_with(sp.as_str()) {
                    matched = Some((sp.len(), *id));
                    break;
                }
            }
            if let Some((len, id)) = matched {
                if pos > seg_start {
                    self.encode_fragment(&text[seg_start..pos], first_fragment, &mut out);
                }
                out.push(id);
                pos += len;
                seg_start = pos;
                first_fragment = false;
            } else {
                let ch_len = rest.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                pos += ch_len;
            }
        }
        if seg_start < text.len() {
            self.encode_fragment(&text[seg_start..], first_fragment, &mut out);
        }
        out
    }

    fn encode_fragment(&self, text: &str, is_first: bool, out: &mut Vec<u32>) {
        match self.kind {
            VocabKind::Spm => self.encode_spm(text, is_first, out),
            VocabKind::Bpe => self.encode_bpe(text, out),
        }
    }

    fn encode_spm(&self, text: &str, is_first: bool, out: &mut Vec<u32>) {
        let mut norm = String::with_capacity(text.len() + 3);
        if is_first && self.add_space_prefix {
            norm.push('\u{2581}');
        }
        for c in text.chars() {
            if c == ' ' {
                norm.push('\u{2581}');
            } else {
                norm.push(c);
            }
        }
        #[derive(Clone, Copy)]
        struct SpmCand {
            score: f32,
            left: usize,
            right: usize,
            gen_l: u32,
            gen_r: u32,
        }
        impl PartialEq for SpmCand {
            fn eq(&self, other: &Self) -> bool {
                self.score.to_bits() == other.score.to_bits() && self.left == other.left
            }
        }
        impl Eq for SpmCand {}
        impl PartialOrd for SpmCand {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for SpmCand {
            fn cmp(&self, other: &Self) -> Ordering {
                self.score
                    .total_cmp(&other.score)
                    .then_with(|| other.left.cmp(&self.left))
            }
        }
        struct Node {
            sym: String,
            prev: usize,
            next: usize,
            alive: bool,
            gen: u32,
        }
        const NONE: usize = usize::MAX;
        let mut nodes: Vec<Node> = norm
            .chars()
            .enumerate()
            .map(|(i, c)| Node {
                sym: c.to_string(),
                prev: if i == 0 { NONE } else { i - 1 },
                next: i + 1,
                alive: true,
                gen: 0,
            })
            .collect();
        if let Some(last) = nodes.last_mut() {
            last.next = NONE;
        }
        if nodes.len() >= 2 {
            let mut heap: BinaryHeap<SpmCand> = BinaryHeap::with_capacity(nodes.len());
            let mut cand = String::new();
            let try_push = |l: usize, r: usize, nodes: &[Node], cand: &mut String, heap: &mut BinaryHeap<SpmCand>| {
                if l == NONE || r == NONE || !nodes[l].alive || !nodes[r].alive {
                    return;
                }
                cand.clear();
                cand.push_str(&nodes[l].sym);
                cand.push_str(&nodes[r].sym);
                if let Some(&id) = self.tok_to_id.get(cand.as_str()) {
                    let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                    heap.push(SpmCand {
                        score,
                        left: l,
                        right: r,
                        gen_l: nodes[l].gen,
                        gen_r: nodes[r].gen,
                    });
                }
            };
            for i in 0..nodes.len() - 1 {
                try_push(i, i + 1, &nodes, &mut cand, &mut heap);
            }
            while let Some(c) = heap.pop() {
                let (l, r) = (c.left, c.right);
                if !nodes[l].alive
                    || !nodes[r].alive
                    || nodes[l].next != r
                    || nodes[l].gen != c.gen_l
                    || nodes[r].gen != c.gen_r
                {
                    continue;
                }
                let r_sym = std::mem::take(&mut nodes[r].sym);
                nodes[l].sym.push_str(&r_sym);
                nodes[l].gen += 1;
                nodes[r].alive = false;
                let rn = nodes[r].next;
                nodes[l].next = rn;
                if rn != NONE {
                    nodes[rn].prev = l;
                }
                try_push(nodes[l].prev, l, &nodes, &mut cand, &mut heap);
                try_push(l, rn, &nodes, &mut cand, &mut heap);
            }
        }
        let mut cur = if nodes.is_empty() { NONE } else { 0 };
        while cur != NONE {
            let p = &nodes[cur].sym;
            if let Some(&id) = self.tok_to_id.get(p) {
                out.push(id);
            } else {
                for b in p.as_bytes() {
                    if let Some(id) = self.byte_tokens[*b as usize] {
                        out.push(id);
                    }
                }
            }
            cur = nodes[cur].next;
        }
    }

    fn encode_bpe(&self, text: &str, out: &mut Vec<u32>) {
        struct Node {
            sym: String,
            prev: usize,
            next: usize,
            alive: bool,
            gen: u32,
        }
        const NONE: usize = usize::MAX;
        let b2c = gpt2_byte_to_char();
        for (s, e) in pre_tokenize_bpe(text, self.llama3_pre) {
            let bytes = &text.as_bytes()[s..e];
            let mut nodes: Vec<Node> = bytes
                .iter()
                .enumerate()
                .map(|(i, &b)| Node {
                    sym: b2c[b as usize].to_string(),
                    prev: if i == 0 { NONE } else { i - 1 },
                    next: i + 1,
                    alive: true,
                    gen: 0,
                })
                .collect();
            if let Some(last) = nodes.last_mut() {
                last.next = NONE;
            }
            if !self.merges.is_empty() && nodes.len() >= 2 {
                // O(n log n) min-heap BPE merge with exact leftmost tie-breaking
                let mut heap: BinaryHeap<Reverse<(usize, usize, usize, u32, u32)>> =
                    BinaryHeap::with_capacity(nodes.len());
                let try_push = |l: usize,
                                r: usize,
                                nodes: &[Node],
                                heap: &mut BinaryHeap<Reverse<(usize, usize, usize, u32, u32)>>| {
                    if l == NONE || r == NONE || !nodes[l].alive || !nodes[r].alive {
                        return;
                    }
                    if let Some(&rank) = self.merges.get(&(nodes[l].sym.clone(), nodes[r].sym.clone())) {
                        heap.push(Reverse((rank, l, r, nodes[l].gen, nodes[r].gen)));
                    }
                };
                for i in 0..nodes.len() - 1 {
                    try_push(i, i + 1, &nodes, &mut heap);
                }
                while let Some(Reverse((_, l, r, gl, gr))) = heap.pop() {
                    if !nodes[l].alive
                        || !nodes[r].alive
                        || nodes[l].next != r
                        || nodes[l].gen != gl
                        || nodes[r].gen != gr
                    {
                        continue;
                    }
                    let r_sym = std::mem::take(&mut nodes[r].sym);
                    nodes[l].sym.push_str(&r_sym);
                    nodes[l].gen += 1;
                    nodes[r].alive = false;
                    let rn = nodes[r].next;
                    nodes[l].next = rn;
                    if rn != NONE {
                        nodes[rn].prev = l;
                    }
                    try_push(nodes[l].prev, l, &nodes, &mut heap);
                    try_push(l, rn, &nodes, &mut heap);
                }
            }
            let mut cur = if nodes.is_empty() { NONE } else { 0 };
            while cur != NONE {
                let sym = &nodes[cur].sym;
                if let Some(&id) = self.tok_to_id.get(sym) {
                    out.push(id);
                } else {
                    // Fallback byte-by-byte
                    for ch in sym.chars() {
                        let s1 = ch.to_string();
                        if let Some(&id) = self.tok_to_id.get(&s1) {
                            out.push(id);
                        }
                    }
                }
                cur = nodes[cur].next;
            }
        }
    }

    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(tok) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        let tt = self.token_types.get(id as usize).copied().unwrap_or(1);
        if tt == 3 {
            // Preserve reasoning-mode control markers so reasoning can be separated from answer text
            if matches!(tok.as_str(), "<think>" | "</think>" | "<|content_thinking|>" | "<|content_text|>") {
                return tok.as_bytes().to_vec();
            }
            // Control token (e.g. <s>, </s>, <|im_end|>)
            return Vec::new();
        }
        if tt == 6 && tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
            if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                return vec![b];
            }
        }
        match self.kind {
            VocabKind::Spm => tok.replace('\u{2581}', " ").into_bytes(),
            VocabKind::Bpe => {
                let rev = gpt2_char_to_byte();
                let mut bytes = Vec::with_capacity(tok.len());
                for ch in tok.chars() {
                    if let Some(&b) = rev.get(&ch) {
                        bytes.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
                bytes
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut raw = Vec::new();
        for (i, &id) in ids.iter().enumerate() {
            let b = self.token_bytes(id);
            let strip_lead = self.kind == VocabKind::Spm
                && self.add_space_prefix
                && b.starts_with(b" ")
                && (i == 0 || (i == 1 && self.bos == Some(ids[0])));
            if strip_lead {
                raw.extend_from_slice(&b[1..]);
            } else {
                raw.extend_from_slice(&b);
            }
        }
        String::from_utf8_lossy(&raw).into_owned()
    }
}

/// Hand-written pre-tokenizer matching Qwen2 / Llama-3 byte-level BPE regexes.
fn pre_tokenize_bpe(text: &str, llama3: bool) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte_end = |idx: usize| if idx < n { chars[idx].0 } else { text.len() };
    let is_l = |i: usize| i < n && chars[i].1.is_alphabetic();
    let is_n = |i: usize| i < n && chars[i].1.is_numeric();
    let is_ws = |i: usize| i < n && chars[i].1.is_whitespace();
    let is_nl = |i: usize| i < n && (chars[i].1 == '\r' || chars[i].1 == '\n');

    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let start = i;
        // 1. Contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if chars[i].1 == '\'' && i + 1 < n {
            let c1 = chars[i + 1].1.to_ascii_lowercase();
            let c2 = if i + 2 < n { Some(chars[i + 2].1.to_ascii_lowercase()) } else { None };
            if matches!(c1, 's' | 't' | 'm' | 'd') {
                out.push((byte_end(start), byte_end(i + 2)));
                i += 2;
                continue;
            }
            if matches!((c1, c2), ('r', Some('e')) | ('v', Some('e')) | ('l', Some('l'))) {
                out.push((byte_end(start), byte_end(i + 3)));
                i += 3;
                continue;
            }
        }
        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        if is_l(i) {
            i += 1;
            while is_l(i) { i += 1; }
            out.push((byte_end(start), byte_end(i)));
            continue;
        }
        if !is_nl(i) && !is_n(i) && i + 1 < n && is_l(i + 1) {
            i += 2;
            while is_l(i) { i += 1; }
            out.push((byte_end(start), byte_end(i)));
            continue;
        }
        // 3. \p{N} (Qwen2) or \p{N}{1,3} (Llama3)
        if is_n(i) {
            let lim = if llama3 { 3 } else { 1 };
            let mut k = 1;
            while k < lim && is_n(i + k) { k += 1; }
            i += k;
            out.push((byte_end(start), byte_end(i)));
            continue;
        }
        // 4. ' '?[^\s\p{L}\p{N}]+[\r\n]*
        if (chars[i].1 == ' ' && i + 1 < n && !is_ws(i + 1) && !is_l(i + 1) && !is_n(i + 1))
            || (!is_ws(i) && !is_l(i) && !is_n(i))
        {
            if chars[i].1 == ' ' { i += 1; }
            while i < n && !is_ws(i) && !is_l(i) && !is_n(i) { i += 1; }
            while is_nl(i) { i += 1; }
            out.push((byte_end(start), byte_end(i)));
            continue;
        }
        // 5. \s*[\r\n]+ | \s+(?!\S) | \s+
        if is_ws(i) {
            let mut j = i;
            let mut last_nl = None;
            while j < n && is_ws(j) {
                if is_nl(j) { last_nl = Some(j); }
                j += 1;
            }
            if let Some(nl) = last_nl {
                i = nl + 1;
            } else if j < n && j - start > 1 {
                // \s+(?!\S): leave last space for the next token
                i = j - 1;
            } else {
                i = j;
            }
            out.push((byte_end(start), byte_end(i)));
            continue;
        }
        i += 1;
        out.push((byte_end(start), byte_end(i)));
    }
    out
}

#[derive(Default)]
pub struct IncrementalDetokenizer {
    buf: Vec<u8>,
}

impl IncrementalDetokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.buf.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        out.push_str(std::str::from_utf8(&self.buf[..valid]).unwrap());
                        self.buf.drain(..valid);
                    }
                    if let Some(err_len) = e.error_len() {
                        out.push('\u{FFFD}');
                        self.buf.drain(..err_len);
                    } else {
                        // Incomplete multi-byte sequence at end; wait for next chunk
                        break;
                    }
                }
            }
        }
        out
    }

    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        s
    }
}
