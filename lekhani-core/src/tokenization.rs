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

use crate::session::ShiftOrigin;
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

        // Composite independent vowel: অ/এ + ্ + য [+ া] (e.g. অ্যা in অ্যাসিড, অ্যাসোসিয়েশন)
        if (c == 'অ' || c == 'এ') && i + 2 < n && chars[i + 1] == '\u{09CD}' && chars[i + 2] == 'য' {
            let mut vow = String::with_capacity(8);
            vow.push(c);
            vow.push(chars[i + 1]);
            vow.push(chars[i + 2]);
            i += 3;
            if i < n && chars[i] == 'া' {
                vow.push(chars[i]);
                i += 1;
            }
            tokens.push(BengaliToken::new(BengaliTokenKind::IndependentVowel, vow));
            continue;
        }

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

/// Normalizes a Roman string while strictly preserving character case distinctions
/// (e.g. distinguishing Avro retroflex 'T' from dental 't', 'D' from 'd', 'S' from 's', 'O' from 'o'),
/// while collapsing 3+ repeated letters to 2.
pub fn normalize_roman_cased(text: &str) -> String {
    let mut res = String::with_capacity(text.len());
    for c in text.chars() {
        let len = res.len();
        if len >= 2 {
            let bytes = res.as_bytes();
            let c_lower = c.to_ascii_lowercase();
            let b1_lower = (bytes[len - 1] as char).to_ascii_lowercase();
            let b2_lower = (bytes[len - 2] as char).to_ascii_lowercase();
            if b1_lower == c_lower && b2_lower == c_lower {
                continue;
            }
        }
        res.push(c);
    }
    res
}

/// Tokenizes a Roman input string using greedy longest-match while preserving case distinctions.
/// Digraphs match case-insensitively (e.g. "th", "Th", "TH", "kkh", "Kkh", "Chh"),
/// preserving the exact casing of the underlying characters.
pub fn tokenize_roman_cased(text: &str) -> Vec<String> {
    let norm = normalize_roman_cased(text);
    let chars: Vec<char> = norm.chars().collect();
    let n = chars.len();
    let mut tokens = Vec::with_capacity(n);
    let mut i = 0;

    while i < n {
        // 1. Try 3-char digraph ("chh", "Chh", etc.)
        if i + 3 <= n {
            let sub_lower: String = chars[i..i + 3].iter().map(|c| c.to_ascii_lowercase()).collect();
            if CONSONANT_DIGRAPHS.contains(&sub_lower.as_str()) {
                let token: String = chars[i..i + 3].iter().collect();
                tokens.push(token);
                i += 3;
                continue;
            }
        }

        // 2. Try 2-char digraphs or doubled consonants
        if i + 2 <= n {
            let sub_lower: String = chars[i..i + 2].iter().map(|c| c.to_ascii_lowercase()).collect();
            let is_digraph = CONSONANT_DIGRAPHS.contains(&sub_lower.as_str())
                || VOWEL_DIGRAPHS.contains(&sub_lower.as_str());
            let is_doubled = chars[i].to_ascii_lowercase() == chars[i + 1].to_ascii_lowercase()
                && chars[i].is_ascii_alphabetic();

            if is_digraph || is_doubled {
                let token: String = chars[i..i + 2].iter().collect();
                tokens.push(token);
                i += 2;
                continue;
            }
        }

        // 3. Fallback: single character (cased)
        tokens.push(chars[i].to_string());
        i += 1;
    }

    tokens
}

/// Tokenizes a Roman input string with awareness of the shift origin.
/// If `origin == ShiftOrigin::AutoCapitalize` and the first character is uppercase while all subsequent
/// characters are lowercase (e.g. sentence-initial "Taka" or "Tumi"), the initial character is treated
/// as uncapitalized ("taka" / "tumi") to prevent auto-capitalization from triggering accidental retroflex Avro mapping.
/// If `origin == ShiftOrigin::Manual` or mid-word uppercase is present (e.g. "daKTar"), full case is preserved.
pub fn tokenize_roman_with_origin(text: &str, origin: ShiftOrigin) -> Vec<String> {
    let text = match origin {
        ShiftOrigin::AutoCapitalize => {
            let chars: Vec<char> = text.chars().collect();
            if chars.first().is_some_and(|c| c.is_ascii_uppercase())
                && chars.iter().skip(1).all(|c| !c.is_ascii_uppercase())
            {
                // TitleCase produced by system auto-capitalization: uncapitalize initial char
                let mut uncap = String::with_capacity(text.len());
                if let Some(first) = chars.first() {
                    uncap.push(first.to_ascii_lowercase());
                }
                for &c in chars.iter().skip(1) {
                    uncap.push(c);
                }
                uncap
            } else {
                text.to_string()
            }
        }
        ShiftOrigin::Manual => text.to_string(),
    };
    tokenize_roman_cased(&text)
}

/// Returns all valid segmentation paths for a normalized Roman string.
/// Unlike greedy longest-match, this generates a lattice of valid segmentations,
/// allowing digraphs (e.g. "ng", "kh", "th") and doubled consonants ("tt", "bb")
/// to be evaluated either as atomic units or individual single letters by the aligner.
pub fn roman_segmentation_lattice(text: &str) -> Vec<Vec<String>> {
    let norm = normalize_roman(text);
    let chars: Vec<char> = norm.chars().collect();
    let n = chars.len();
    if n == 0 {
        return vec![Vec::new()];
    }

    fn recurse(
        chars: &[char],
        start: usize,
        memo: &mut hashbrown::HashMap<usize, Vec<Vec<String>>>,
    ) -> Vec<Vec<String>> {
        if start == chars.len() {
            return vec![Vec::new()];
        }
        if let Some(cached) = memo.get(&start) {
            return cached.clone();
        }

        let n = chars.len();
        let mut paths = Vec::new();

        // 1. Try 3-char "chh" (indivisible)
        if start + 3 <= n {
            let sub_str: String = chars[start..start + 3].iter().collect();
            if sub_str == "chh" {
                for sub in recurse(chars, start + 3, memo) {
                    let mut p = Vec::with_capacity(sub.len() + 1);
                    p.push(sub_str.clone());
                    p.extend(sub);
                    paths.push(p);
                }
                memo.insert(start, paths.clone());
                return paths;
            }
        }

        // 2. Try 2-char tokens
        if start + 2 <= n {
            let sub_str: String = chars[start..start + 2].iter().collect();
            let is_doubled = chars[start] == chars[start + 1] && chars[start].is_ascii_alphabetic();

            // Known ambiguous digraphs and doubled consonants that allow lattice alternatives:
            // "ng" (ng vs n|g), "sh" (sh vs s|h), and geminates (tt vs t|t)
            if sub_str == "ng" || sub_str == "sh" || is_doubled {
                // Option A: Take as single atomic token
                for sub in recurse(chars, start + 2, memo) {
                    let mut p = Vec::with_capacity(sub.len() + 1);
                    p.push(sub_str.clone());
                    p.extend(sub);
                    paths.push(p);
                }
                // Option B: Take as single character and continue
                let s1 = chars[start].to_string();
                for sub in recurse(chars, start + 1, memo) {
                    let mut p = Vec::with_capacity(sub.len() + 1);
                    p.push(s1.clone());
                    p.extend(sub);
                    paths.push(p);
                }
                memo.insert(start, paths.clone());
                return paths;
            }

            // All other digraphs ("kh", "gh", "ch", "th", "dh", "ph", "bh", "nj", "aa", "ee", "oo", "oi", "ou", "ai", "au")
            // are INDIVISIBLE ATOMIC units; alternative decomposition is strictly forbidden.
            let is_indivisible = CONSONANT_DIGRAPHS.contains(&sub_str.as_str())
                || VOWEL_DIGRAPHS.contains(&sub_str.as_str());

            if is_indivisible {
                for sub in recurse(chars, start + 2, memo) {
                    let mut p = Vec::with_capacity(sub.len() + 1);
                    p.push(sub_str.clone());
                    p.extend(sub);
                    paths.push(p);
                }
                memo.insert(start, paths.clone());
                return paths;
            }
        }

        // 3. Fallback: single character
        let s1 = chars[start].to_string();
        for sub in recurse(chars, start + 1, memo) {
            let mut p = Vec::with_capacity(sub.len() + 1);
            p.push(s1.clone());
            p.extend(sub);
            paths.push(p);
        }

        memo.insert(start, paths.clone());
        paths
    }

    let mut memo = hashbrown::HashMap::new();
    recurse(&chars, 0, &mut memo)
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
/// 2. Consonant deletion to ε is forbidden (only Roman vowel tokens may delete to ε).
/// 3. A conjunct (CONJ) is an atomic token; it is never split.
/// 4. Default ratio is 1:1, 1:0 (vowel deletion only), 0:1 (insertion). Multi-token units must match whitelist.
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

    // Hard Constraint 2: Consonant deletion to ε is forbidden.
    // Roman deletion (bengali_tokens is empty): ONLY vowels are allowed to delete to ε.
    if bengali_tokens.is_empty() {
        return roman_tokens.iter().all(|r| is_roman_vowel_token(r));
    }

    // Bengali insertion (roman_tokens is empty): max 1 Bengali token (e.g. independent vowel or modifier)
    if roman_tokens.is_empty() {
        return bengali_tokens.len() == 1;
    }

    // Hard Constraint 3: Ratio checks (1:1, or authorized whitelist)
    let r_len = roman_tokens.len();
    let b_len = bengali_tokens.len();

    if r_len <= 1 && b_len <= 1 {
        return true;
    }

    let roman_joined = roman_tokens.join("");
    is_authorized_multi_token_unit(&roman_joined, bengali_tokens)
}
