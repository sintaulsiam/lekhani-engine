//! Zero-Allocation High-Speed Phonetic Parsing Engine

use std::sync::Arc;
use super::compiler::CompiledLayout;
use super::rule::{ConditionScope, MatchType};
use super::tables::*;

#[derive(Debug, Clone)]
pub struct LekhaniParser {
    pub layout: Arc<CompiledLayout>,
}

impl LekhaniParser {
    pub fn new(layout: Arc<CompiledLayout>) -> Self {
        Self { layout }
    }

    /// Convert input string into Unicode Bengali, writing directly into `output`.
    /// Zero heap allocations on the hot path.
    pub fn convert_into(&self, input: &str, output: &mut String) {
        if input.is_empty() {
            return;
        }

        // Handle Latin Chandra Bindu pre-normalization: e.g. "c^ad" -> "ca^d"
        let norm_storage: String;
        let effective_input = if input.as_bytes().contains(&b'^') {
            norm_storage = normalize_chandra_latin(input);
            norm_storage.as_str()
        } else {
            input
        };

        let has_combining = if effective_input.is_ascii() {
            self.convert_ascii_into(effective_input, output)
        } else {
            self.convert_mixed_into(effective_input, output)
        };

        // Post-process to ensure canonical Unicode Combining Class sequence
        if has_combining {
            canonicalize_combining_marks(output);
        }
    }

    /// Convert input string using pure raw grammar rules without extra diacritic reordering.
    /// Exactly matches the raw Avro grammar specification.
    pub fn convert_raw_into(&self, input: &str, output: &mut String) {
        if input.is_empty() {
            return;
        }

        if input.is_ascii() {
            self.convert_ascii_into(input, output);
        } else {
            self.convert_mixed_into(input, output);
        }
    }

    /// Convert input string into Unicode Bengali, returning a newly allocated String.
    pub fn convert(&self, input: &str) -> String {
        let mut out = String::with_capacity(input.len() * 3);
        self.convert_into(input, &mut out);
        out
    }

    /// Convert input string using raw rules, returning a newly allocated String.
    pub fn convert_raw(&self, input: &str) -> String {
        let mut out = String::with_capacity(input.len() * 3);
        self.convert_raw_into(input, &mut out);
        out
    }

    /// Fast path for pure ASCII input. Returns whether any combining marks were emitted.
    fn convert_ascii_into(&self, input: &str, output: &mut String) -> bool {
        let in_bytes = input.as_bytes();
        let len = in_bytes.len();

        // Stack-allocated case-normalized buffer for up to 128 bytes only if normalization needed
        let mut stack_buf = [0u8; 128];
        let heap_buf: Vec<u8>;
        let fixed: &[u8] = if in_bytes.iter().any(|&b| needs_avro_norm(b)) {
            if len <= 128 {
                for i in 0..len {
                    stack_buf[i] = normalize_avro_byte(in_bytes[i]);
                }
                &stack_buf[..len]
            } else {
                heap_buf = in_bytes.iter().map(|&b| normalize_avro_byte(b)).collect();
                &heap_buf
            }
        } else {
            in_bytes
        };

        let mut has_combining = false;
        let mut cur = 0;
        while cur < len {
            if let Some((pat_idx, match_len)) = self.layout.trie.longest_match(fixed, cur) {
                let pat = &self.layout.patterns[pat_idx as usize];
                if pat.may_have_combining {
                    has_combining = true;
                }
                let start = cur as isize;
                let end = (cur + match_len) as isize;
                let mut matched_rule = false;

                for rule in &pat.rules {
                    let mut conditions_met = true;

                    for cond in &rule.conditions {
                        let is_neg = cond.is_negative;
                        let chk = match cond.match_type {
                            MatchType::Prefix => start - 1,
                            MatchType::Suffix => end,
                        };

                        let cond_result = match cond.scope {
                            ConditionScope::Punctuation => {
                                if (cond.match_type == MatchType::Prefix && chk < 0)
                                    || (cond.match_type == MatchType::Suffix && chk >= len as isize)
                                {
                                    true
                                } else if chk >= 0 && (chk as usize) < len {
                                    is_punctuation_byte(fixed[chk as usize])
                                } else {
                                    false
                                }
                            }
                            ConditionScope::Vowel => {
                                if chk >= 0 && (chk as usize) < len {
                                    is_vowel_byte(fixed[chk as usize])
                                } else {
                                    false
                                }
                            }
                            ConditionScope::Consonant => {
                                if chk >= 0 && (chk as usize) < len {
                                    is_consonant_byte(fixed[chk as usize])
                                } else {
                                    false
                                }
                            }
                            ConditionScope::Number => {
                                if chk >= 0 && (chk as usize) < len {
                                    is_number_byte(fixed[chk as usize])
                                } else {
                                    false
                                }
                            }
                            ConditionScope::Exact => {
                                let needle = cond.exact_value.as_bytes();
                                let n_len = needle.len() as isize;
                                let (s, e) = match cond.match_type {
                                    MatchType::Prefix => (start - n_len, start),
                                    MatchType::Suffix => (end, end + n_len),
                                };

                                if s >= 0 && e < len as isize {
                                    &fixed[s as usize..e as usize] == needle
                                } else {
                                    false
                                }

                            }
                        };

                        if cond_result == is_neg {
                            conditions_met = false;
                            break;
                        }
                    }

                    if conditions_met {
                        output.push_str(&rule.replace);
                        cur += match_len;
                        matched_rule = true;
                        break;
                    }
                }

                if !matched_rule {
                    output.push_str(&pat.default_replace);
                    cur += match_len;
                }
            } else {
                output.push(fixed[cur] as char);
                cur += 1;
            }
        }
        has_combining
    }

    /// Mixed ASCII / non-ASCII path: segments ASCII chunks and passes Unicode directly
    fn convert_mixed_into(&self, input: &str, output: &mut String) -> bool {
        let mut ascii_chunk = String::with_capacity(input.len());
        let mut has_combining = false;

        for ch in input.chars() {
            if ch.is_ascii() {
                ascii_chunk.push(ch);
            } else {
                if !ascii_chunk.is_empty() {
                    if self.convert_ascii_into(&ascii_chunk, output) {
                        has_combining = true;
                    }
                    ascii_chunk.clear();
                }
                output.push(ch);
            }
        }

        if !ascii_chunk.is_empty() {
            if self.convert_ascii_into(&ascii_chunk, output) {
                has_combining = true;
            }
        }
        has_combining
    }
}

/// Normalizes Latin caret (^) placement: e.g. "c^ad" -> "ca^d"
fn normalize_chandra_latin(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '^' && i + 1 < chars.len() {
            let next = chars[i + 1];
            if is_latin_vowel(next) {
                let mut v_end = i + 1;
                while v_end < chars.len() && is_latin_vowel(chars[v_end]) {
                    v_end += 1;
                }
                for v in &chars[i + 1..v_end] {
                    out.push(*v);
                }
                out.push('^');
                i = v_end;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

#[inline(always)]
fn is_latin_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U' | 'y' | 'Y' | '`'
    )
}

/// Canonicalize Bengali Unicode combining sequences (Nukta composition & Chandra Bindu ordering)
fn canonicalize_combining_marks(text: &mut String) {
    if text.contains('\u{09BC}') {
        let replaced = text
            .replace("\u{09AF}\u{09BC}", "য়")
            .replace("\u{09A1}\u{09BC}", "ড়")
            .replace("\u{09A2}\u{09BC}", "ঢ়");
        *text = replaced;
    }

    if text.contains('ঁ') {
        let replaced = text
            .replace("ঁআ", "াঁ")
            .replace("ঁা", "াঁ")
            .replace("ঁএ", "েঁ")
            .replace("ঁে", "েঁ")
            .replace("ঁই", "িঁ")
            .replace("ঁি", "িঁ")
            .replace("ঁউ", "ুঁ")
            .replace("ঁু", "ুঁ")
            .replace("ঁও", "োঁ")
            .replace("ঁো", "োঁ")
            .replace("ঁঐ", "ৈঁ")
            .replace("ঁৈ", "ৈঁ")
            .replace("ঁঔ", "ৌঁ")
            .replace("ঁৌ", "ৌঁ")
            .replace("ঁৃ", "ৃঁ");
        *text = replaced;
    }
}
