//! Pragyan Phase 1 Step 3: Grapheme-Level Tokenization and Translation Unit Inventory
//!
//! Separates Bengali text and Roman input into fine-grained, linguistically sound tokens:
//!
//! Bengali tokens (in logical Unicode NFC order):
//! - `Consonant`: single consonant (ক..হ), atomic nukta consonants (ড়, ঢ়, য়), or Khanda Ta (ৎ).
//! - `Conjunct`: full atomic conjunct cluster (C + ্ + C [+ ্ + C]*). Never splits a hasanta!
//! - `VowelKar`: dependent vowel sign (া, ি, ী, ু, ূ, ৃ, ে, ৈ, ো, ৌ).
//! - `IndependentVowel`: independent vowel (অ, আ, ই, ঈ, উ, ঊ, ঋ, এ, ঐ, ও, ঔ).
//! - `Modifier`: ঁ (chandrabindu), ং (anusvara), ঃ (visarga).
//!
//! Roman tokens (greedy longest-match):
//! - Consonant digraphs: kh, gh, ch, chh, th, dh, ph, bh, sh, ng, nj
//! - Vowel digraphs: aa, ee, oo, oi, ou, ai, au
//! - Doubled consonants: tt, ss, kk, dd, bb, mm, ll, nn, pp, cc, rr, gg, jj
//! - Single ASCII alphabetic characters
//! - Collapses 3+ repeated letters to 2.
//!
//! Unit rules:
//! - 1:1 default, or ε on either side.
//! - 1:2 whitelist only (e.g. x -> ক্স, a -> অ্যা, ri -> ৃ).
//! - 2:1 for doubled consonants only.
//! - Hard constraints: no unit begins with a KAR unless Roman side is a vowel; conjuncts are atomic.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BengaliTokenKind {
    Consonant,
    Conjunct,
    VowelKar,
    IndependentVowel,
    Modifier,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BengaliToken {
    pub kind: BengaliTokenKind,
    pub text: String,
}

impl BengaliToken {
    #[inline]
    pub fn new(kind: BengaliTokenKind, text: impl Into<String>) -> Self {
        Self {
            kind,
            text: text.into(),
        }
    }

    #[inline]
    pub fn is_vowel_kar(&self) -> bool {
        self.kind == BengaliTokenKind::VowelKar
    }

    #[inline]
    pub fn is_independent_vowel(&self) -> bool {
        self.kind == BengaliTokenKind::IndependentVowel
    }

    #[inline]
    pub fn is_consonant(&self) -> bool {
        self.kind == BengaliTokenKind::Consonant
    }

    #[inline]
    pub fn is_conjunct(&self) -> bool {
        self.kind == BengaliTokenKind::Conjunct
    }

    #[inline]
    pub fn is_modifier(&self) -> bool {
        self.kind == BengaliTokenKind::Modifier
    }
}

#[inline]
fn is_bengali_consonant_char(c: char) -> bool {
    ('\u{0995}'..='\u{09B9}').contains(&c)
        || matches!(c, '\u{09CE}' | '\u{09DC}' | '\u{09DD}' | '\u{09DF}')
}

#[inline]
fn is_bengali_kar(c: char) -> bool {
    matches!(
        c,
        'া' | 'ি' | 'ী' | 'ু' | 'ূ' | 'ৃ' | 'ে' | 'ৈ' | 'ো' | 'ৌ'
    )
}

#[inline]
fn is_bengali_independent_vowel(c: char) -> bool {
    matches!(
        c,
        'অ' | 'আ' | 'ই' | 'ঈ' | 'উ' | 'ঊ' | 'ঋ' | 'এ' | 'ঐ' | 'ও' | 'ঔ'
    )
}

#[inline]
fn is_bengali_modifier(c: char) -> bool {
    matches!(c, 'ঁ' | 'ং' | 'ঃ')
}

/// Tokenizes canonical Bengali text into fine-grained atomic linguistic tokens.
/// Guarantees that concatenating all token texts reproduces the input string identically.
pub fn tokenize_bengali(text: &str) -> Vec<BengaliToken> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut tokens = Vec::with_capacity(n);
    let mut i = 0;

    while i < n {
        let c = chars[i];

        if is_bengali_independent_vowel(c) {
            tokens.push(BengaliToken::new(BengaliTokenKind::IndependentVowel, c.to_string()));
            i += 1;
        } else if is_bengali_kar(c) {
            tokens.push(BengaliToken::new(BengaliTokenKind::VowelKar, c.to_string()));
            i += 1;
        } else if is_bengali_modifier(c) {
            tokens.push(BengaliToken::new(BengaliTokenKind::Modifier, c.to_string()));
            i += 1;
        } else if c == 'ৎ' {
            tokens.push(BengaliToken::new(BengaliTokenKind::Consonant, c.to_string()));
            i += 1;
        } else if is_bengali_consonant_char(c) {
            // Base consonant cluster (includes nukta if decomposed and hasanta chains)
            let mut cluster = String::with_capacity(8);
            cluster.push(c);
            i += 1;
            if i < n && chars[i] == '\u{09BC}' {
                cluster.push(chars[i]);
                i += 1;
            }

            let mut has_conjunct = false;
            while i + 1 < n && chars[i] == '\u{09CD}' && is_bengali_consonant_char(chars[i + 1]) {
                has_conjunct = true;
                cluster.push(chars[i]); // hasanta
                cluster.push(chars[i + 1]); // next consonant
                i += 2;
                if i < n && chars[i] == '\u{09BC}' {
                    cluster.push(chars[i]);
                    i += 1;
                }
            }

            // Trailing hasanta (if any halant occurs at end of word or before non-consonant)
            if i < n && chars[i] == '\u{09CD}' {
                cluster.push(chars[i]);
                i += 1;
            }

            let kind = if has_conjunct {
                BengaliTokenKind::Conjunct
            } else {
                BengaliTokenKind::Consonant
            };
            tokens.push(BengaliToken::new(kind, cluster));
        } else {
            // Punctuation, whitespace, or other
            tokens.push(BengaliToken::new(BengaliTokenKind::Other, c.to_string()));
            i += 1;
        }
    }

    tokens
}

const CONSONANT_DIGRAPHS: &[&str] = &[
    "chh", "kh", "gh", "ch", "th", "dh", "ph", "bh", "sh", "ng", "nj",
];

const VOWEL_DIGRAPHS: &[&str] = &[
    "aa", "ee", "oo", "oi", "ou", "ai", "au",
];

/// Normalizes a Roman string by lowercasing and collapsing 3+ repeated letters to 2.
pub fn normalize_roman(text: &str) -> String {
    let mut res = String::with_capacity(text.len());
    for c in text.chars().map(|c| c.to_ascii_lowercase()) {
        let len = res.len();
        if len >= 2 {
            let bytes = res.as_bytes();
            if bytes[len - 1] as char == c && bytes[len - 2] as char == c {
                continue;
            }
        }
        res.push(c);
    }
    res
}

/// Tokenizes a Roman input string using greedy longest-match.
/// Guarantees that concatenating all tokens reproduces `normalize_roman(text)`.
pub fn tokenize_roman(text: &str) -> Vec<String> {
    let norm = normalize_roman(text);
    let chars: Vec<char> = norm.chars().collect();
    let n = chars.len();
    let mut tokens = Vec::with_capacity(n);
    let mut i = 0;

    while i < n {
        // 1. Try 3-char digraph ("chh")
        if i + 3 <= n {
            let sub: String = chars[i..i + 3].iter().collect();
            if CONSONANT_DIGRAPHS.contains(&sub.as_str()) {
                tokens.push(sub);
                i += 3;
                continue;
            }
        }

        // 2. Try 2-char digraphs or doubled consonants
        if i + 2 <= n {
            let sub: String = chars[i..i + 2].iter().collect();
            let is_digraph = CONSONANT_DIGRAPHS.contains(&sub.as_str())
                || VOWEL_DIGRAPHS.contains(&sub.as_str());
            let is_doubled = chars[i] == chars[i + 1] && chars[i].is_ascii_alphabetic();

            if is_digraph || is_doubled {
                tokens.push(sub);
                i += 2;
                continue;
            }
        }

        // 3. Fallback: single character
        tokens.push(chars[i].to_string());
        i += 1;
    }

    tokens
}

#[inline]
pub fn is_roman_vowel_token(tok: &str) -> bool {
    matches!(
        tok,
        "a" | "e" | "i" | "o" | "u" | "aa" | "ee" | "oo" | "oi" | "ou" | "ai" | "au" | "w" | "y"
    )
}

#[inline]
pub fn is_roman_consonant_token(tok: &str) -> bool {
    !is_roman_vowel_token(tok) && tok.chars().all(|c| c.is_ascii_alphabetic())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TranslationUnit {
    pub roman: String,
    pub bengali: String,
}

impl TranslationUnit {
    #[inline]
    pub fn new(roman: impl Into<String>, bengali: impl Into<String>) -> Self {
        Self {
            roman: roman.into(),
            bengali: bengali.into(),
        }
    }
}

/// Whitelist of authorized 1:2 and 2:1 mappings
pub fn is_authorized_multi_token_unit(roman: &str, bengali_tokens: &[BengaliToken]) -> bool {
    // 1:2 whitelist
    if bengali_tokens.len() == 2 {
        let b1 = &bengali_tokens[0];
        let b2 = &bengali_tokens[1];
        return match roman {
            "x" => (b1.text == "ক" && b2.text == "স") || (b1.text == "ক্স"),
            "a" => (b1.text == "অ" && b2.text == "্যা") || (b1.text == "অ" && b2.text == "য"),
            "ri" => b1.text == "ঋ" || b1.text == "ৃ",
            _ => false,
        };
    }

    // 2:1 doubled consonant check
    if bengali_tokens.len() == 1 {
        let b = &bengali_tokens[0];
        if roman.len() == 2 && roman.as_bytes()[0] == roman.as_bytes()[1] {
            return b.is_consonant() || b.is_conjunct();
        }
    }

    false
}

/// Enforces the hard constraints on a candidate translation unit:
/// 1. No unit begins with a KAR unless the Roman side is a vowel token.
/// 2. A conjunct (CONJ) is an atomic token; it is never split.
/// 3. Default ratio is 1:1, 1:0 (deletion), 0:1 (insertion). Multi-token units must match whitelist.
pub fn is_valid_translation_unit(
    roman_tokens: &[String],
    bengali_tokens: &[BengaliToken],
) -> bool {
    // Null-null mapping is invalid
    if roman_tokens.is_empty() && bengali_tokens.is_empty() {
        return false;
    }

    // Hard Constraint 1: No unit begins with a KAR on the Bengali side unless the Roman side is a vowel token
    if let Some(first_b) = bengali_tokens.first() {
        if first_b.is_vowel_kar() {
            let has_vowel_roman = roman_tokens.iter().any(|r| is_roman_vowel_token(r));
            if !has_vowel_roman {
                return false;
            }
        }
    }

    // Hard Constraint 2: Ratio checks (1:1, 1:0, 0:1, or authorized whitelist)
    let r_len = roman_tokens.len();
    let b_len = bengali_tokens.len();

    if r_len <= 1 && b_len <= 1 {
        return true;
    }

    let roman_joined = roman_tokens.join("");
    is_authorized_multi_token_unit(&roman_joined, bengali_tokens)
}
