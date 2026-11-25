//! Hand-written equivalents of the Qwen2 / Qwen3.5 / Llama-3 pre-tokenizer regexes:
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)
//! |[^\r\n\p{L}\p{N}]?\p{L}+        (Qwen3.5: [\p{L}\p{M}]+)
//! |\p{N}                            (Llama-3: \p{N}{1,3})
//! | ?[^\s\p{L}\p{N}]+[\r\n]*
//! |\s*[\r\n]+
//! |\s+(?!\S)
//! |\s+
//! ```

fn is_letter(c: char) -> bool { c.is_alphabetic() && !c.is_numeric() }

fn is_mark(c: char) -> bool {
    matches!(c as u32,
        0x0300..=0x036F | 0x0483..=0x0489 | 0x0591..=0x05BD | 0x0610..=0x061A | 0x064B..=0x065F
        | 0x0900..=0x0903 | 0x093A..=0x094F | 0x0E31 | 0x0E34..=0x0E3A | 0x0E47..=0x0E4E
        | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0x3099..=0x309A | 0xFE00..=0xFE0F | 0xFE20..=0xFE2F)
}

#[derive(Clone, Copy, Debug)]
pub struct PreStyle {
    /// Group digits in runs of up to three (Llama-3) instead of one at a time (Qwen).
    pub digits3: bool,
    /// Treat combining marks as letters (Qwen3.5).
    pub marks: bool,
}

pub fn split(text: &str, style: PreStyle) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let off = |i: usize| if i < n { cs[i].0 } else { text.len() };
    let letter = |c: char| is_letter(c) || (style.marks && is_mark(c));
    let num = |c: char| c.is_numeric();
    let punct = |c: char| !c.is_whitespace() && !letter(c) && !num(c);
    let nl = |c: char| c == '\r' || c == '\n';

    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = cs[i].1;
        let end = (|| {
            // 1. contractions
            if c == '\'' && i + 1 < n {
                let c1 = cs[i + 1].1.to_ascii_lowercase();
                if matches!(c1, 's' | 't' | 'm' | 'd') {
                    return Some(i + 2);
                }
                if i + 2 < n {
                    let c2 = cs[i + 2].1.to_ascii_lowercase();
                    if matches!((c1, c2), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                        return Some(i + 3);
                    }
                }
            }
            // 2. optional single prefix char + letters
            let start_letters = if letter(c) {
                Some(i)
            } else if !nl(c) && !is_letter(c) && !c.is_numeric() && i + 1 < n && letter(cs[i + 1].1) {
                Some(i + 1)
            } else {
                None
            };
            if let Some(s) = start_letters {
                let mut j = s;
                while j < n && letter(cs[j].1) { j += 1; }
                return Some(j);
            }
            // 3. digits
            if num(c) {
                let mut j = i + 1;
                if style.digits3 {
                    while j < n && j - i < 3 && num(cs[j].1) { j += 1; }
                }
                return Some(j);
            }
            // 4. optional space + punctuation run + newlines
            let k = if c == ' ' { i + 1 } else { i };
            if k < n && punct(cs[k].1) {
                let mut j = k;
                while j < n && punct(cs[j].1) { j += 1; }
                while j < n && nl(cs[j].1) { j += 1; }
                return Some(j);
            }
            // 5-7. whitespace handling
            if c.is_whitespace() {
                let mut j = i;
                while j < n && cs[j].1.is_whitespace() { j += 1; }
                if let Some(last) = (i..j).rev().find(|&k| nl(cs[k].1)) {
                    return Some(last + 1);
                }
                if j == n { return Some(j); }
                if j - 1 > i { return Some(j - 1); }
                return Some(j);
            }
            None
        })();
        let end = end.unwrap_or(i + 1).max(i + 1);
        out.push(&text[off(i)..off(end)]);
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(s: &str) -> Vec<&str> { split(s, PreStyle { digits3: false, marks: false }) }
    fn l(s: &str) -> Vec<&str> { split(s, PreStyle { digits3: true, marks: false }) }

    #[test]
    fn words_and_punct() {
        assert_eq!(q("Hello, world!"), vec!["Hello", ",", " world", "!"]);
        assert_eq!(q("It's I'M they're"), vec!["It", "'s", " I", "'M", " they", "'re"]);
        assert_eq!(q("foo.bar"), vec!["foo", ".bar"]);
    }

    #[test]
    fn digits() {
        assert_eq!(q("a 12345"), vec!["a", " ", "1", "2", "3", "4", "5"]);
        assert_eq!(l("a 12345"), vec!["a", " ", "123", "45"]);
    }

    #[test]
    fn whitespace() {
        assert_eq!(q("x\n\n\n  y"), vec!["x", "\n\n\n", " ", " y"]);
        assert_eq!(q("a\r\nb\t\tc"), vec!["a", "\r\n", "b", "\t", "\tc"]);
        assert_eq!(q("end  "), vec!["end", "  "]);
        assert_eq!(q("a! \nb"), vec!["a", "!", " \n", "b"]);
        assert_eq!(q("x!\n\ny"), vec!["x", "!\n\n", "y"]);
    }

    #[test]
    fn concatenation_is_lossless() {
        for s in ["", " ", "  \n \t x'y 日本語 😀!!?\r\n\r\n12 ", "'", "'s'", "é\u{0301}x"] {
            assert_eq!(q(s).concat(), s);
            assert_eq!(l(s).concat(), s);
        }
    }
}
