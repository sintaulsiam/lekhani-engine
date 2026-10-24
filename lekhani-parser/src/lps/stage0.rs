//! LPS v1.0 Stage 0: Tokenizer, Operators, Escapes and Diagnostics
//!
//! Natural mode reads lowercase a–z case-insensitively.
//! Reserved control characters: ' + - ^ \
//! Multi-letter tokens (longest first): chh ksh kh gh ng ch jh ny th dh ph bh sh rh ri oi ou ae ii uu oo gg.

use std::fmt;

/// LPS Warning Diagnostics as defined in Section 6.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LpsWarning {
    /// W101: Token has no marked variant for lock '
    W101InvalidLock(String),
    /// W102: Join (+) at word start with no preceding consonant
    W102JoinAtWordStart,
    /// W103: Chandrabindu (^) with no valid host
    W103NoChandrabinduHost,
    /// W104: Unknown escape character
    W104UnknownEscape(char),
    /// W105: Invalid or out-of-range Unicode hex escape
    W105InvalidUnicodeEscape(String),
    /// W106: Empty \e English lock run
    W106EmptyEnglishRun,
}

impl fmt::Display for LpsWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::W101InvalidLock(tok) => write!(f, "LPS-W101: Token '{}' has no marked lock variant", tok),
            Self::W102JoinAtWordStart => write!(f, "LPS-W102: Join operator '+' at word start is invalid"),
            Self::W103NoChandrabinduHost => write!(f, "LPS-W103: Chandrabindu '^' has no valid host"),
            Self::W104UnknownEscape(c) => write!(f, "LPS-W104: Unknown escape sequence '\\{}'", c),
            Self::W105InvalidUnicodeEscape(s) => write!(f, "LPS-W105: Invalid unicode escape '\\u{{{}}}'", s),
            Self::W106EmptyEnglishRun => write!(f, "LPS-W106: Empty English lock '\\e' run"),
        }
    }
}

/// Recognized phonetic token kinds in Stage 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    /// Multi-character digraphs or monographs: "ksh", "kh", "ng", "ri", "a", "k", etc.
    Phonetic(String),
    /// Literal character sequence (from escape or fallback)
    Literal(String),
    /// Explicit code point escape: \u{hex}
    ExactChar(char),
    /// Zero Width Joiner \w (U+200D)
    ZWJ,
    /// Zero Width Non-Joiner \z (U+200C)
    ZWNJ,
    /// Explicit English locked text run from \e
    EnglishRun(String),
}

/// Token with associated lock level, join, split, or modifier marks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage0Token {
    pub kind: TokenKind,
    /// 0 = no lock, 1 = single lock ('), 2 = double lock ('')
    pub lock_count: u8,
    /// Whether this token is followed by a hard join (+)
    pub has_join: bool,
    /// Whether a split (-) preceded this token (blocks cluster/digraph merge across boundary)
    pub has_split_before: bool,
    /// Whether chandrabindu (^) was applied to this unit
    pub has_chandrabindu: bool,
}

impl Stage0Token {
    pub fn new(kind: TokenKind) -> Self {
        Self {
            kind,
            lock_count: 0,
            has_join: false,
            has_split_before: false,
            has_chandrabindu: false,
        }
    }
}

/// Known multi-letter tokens sorted strictly longest-match first.
pub const MULTI_LETTER_TOKENS: &[&str] = &[
    "chh", "ksh", // 3 letters
    "kh", "gh", "ng", "ch", "jh", "ny", "th", "dh",
    "ph", "bh", "sh", "rh", "ri", "oi", "ou", "ae",
    "ii", "uu", "oo", "gg", // 2 letters
];

/// Stage 0 Tokenizer and Segmentation Result.
#[derive(Debug, Clone, Default)]
pub struct Stage0Result {
    pub primary_tokens: Vec<Stage0Token>,
    pub alternative_segmentations: Vec<Vec<Stage0Token>>,
    pub diagnostics: Vec<LpsWarning>,
}

pub fn tokenize_stage0(input: &str) -> Stage0Result {
    // Section 13: Latin token detection (emails, URLs, web addresses)
    if input.contains('@') || input.contains("://") || input.starts_with("www.") {
        return Stage0Result {
            primary_tokens: vec![Stage0Token::new(TokenKind::EnglishRun(input.to_string()))],
            alternative_segmentations: Vec::new(),
            diagnostics: Vec::new(),
        };
    }

    let mut tokens: Vec<Stage0Token> = Vec::new();
    let mut diagnostics: Vec<LpsWarning> = Vec::new();

    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut pending_split = false;

    while i < len {
        let b = bytes[i];

        // 1. Escapes (\)
        if b == b'\\' {
            i += 1;
            if i >= len {
                tokens.push(Stage0Token::new(TokenKind::Literal("\\".to_string())));
                diagnostics.push(LpsWarning::W104UnknownEscape('\\'));
                break;
            }

            match bytes[i] {
                b'w' => {
                    tokens.push(Stage0Token::new(TokenKind::ZWJ));
                    i += 1;
                }
                b'z' => {
                    tokens.push(Stage0Token::new(TokenKind::ZWNJ));
                    i += 1;
                }
                b'e' => {
                    // English lock \e: continues until space or a hyphen followed by a letter
                    i += 1;
                    let start_e = i;
                    while i < len {
                        if bytes[i] == b' ' {
                            break;
                        }
                        if bytes[i] == b'-' && i + 1 < len && bytes[i + 1].is_ascii_alphabetic() {
                            break;
                        }
                        i += 1;
                    }
                    if i > start_e {
                        let eng_text = String::from_utf8_lossy(&bytes[start_e..i]).to_string();
                        tokens.push(Stage0Token::new(TokenKind::EnglishRun(eng_text)));
                    } else {
                        tokens.push(Stage0Token::new(TokenKind::Literal("\\e".to_string())));
                        diagnostics.push(LpsWarning::W106EmptyEnglishRun);
                    }
                }
                b'u' if i + 1 < len && bytes[i + 1] == b'{' => {
                    // \u{hex}
                    i += 2;
                    let hex_start = i;
                    while i < len && bytes[i] != b'}' {
                        i += 1;
                    }
                    if i < len && bytes[i] == b'}' {
                        let hex_str = String::from_utf8_lossy(&bytes[hex_start..i]);
                        if let Ok(cp) = u32::from_str_radix(&hex_str, 16) {
                            if let Some(ch) = char::from_u32(cp) {
                                tokens.push(Stage0Token::new(TokenKind::ExactChar(ch)));
                            } else {
                                tokens.push(Stage0Token::new(TokenKind::Literal(format!("\\u{{{}}}", hex_str))));
                                diagnostics.push(LpsWarning::W105InvalidUnicodeEscape(hex_str.to_string()));
                            }
                        } else {
                            tokens.push(Stage0Token::new(TokenKind::Literal(format!("\\u{{{}}}", hex_str))));
                            diagnostics.push(LpsWarning::W105InvalidUnicodeEscape(hex_str.to_string()));
                        }
                        i += 1; // consume '}'
                    } else {
                        let rem = String::from_utf8_lossy(&bytes[hex_start..]).to_string();
                        tokens.push(Stage0Token::new(TokenKind::Literal(format!("\\u{{{}}}", rem))));
                        diagnostics.push(LpsWarning::W105InvalidUnicodeEscape(rem));
                    }
                }
                b'\'' | b'+' | b'-' | b'^' | b'\\' => {
                    tokens.push(Stage0Token::new(TokenKind::Literal((bytes[i] as char).to_string())));
                    i += 1;
                }
                other => {
                    tokens.push(Stage0Token::new(TokenKind::Literal(format!("\\{}", other as char))));
                    diagnostics.push(LpsWarning::W104UnknownEscape(other as char));
                    i += 1;
                }
            }
            continue;
        }

        // 2. Lock operator (')
        if b == b'\'' {
            let mut invalid_lock_target = None;
            let mut append_literal = false;

            if let Some(last) = tokens.last_mut() {
                if let TokenKind::Phonetic(ref p) = last.kind {
                    if is_lockable_token(p) {
                        if last.lock_count < 2 {
                            last.lock_count += 1;
                        } else {
                            append_literal = true;
                        }
                    } else {
                        invalid_lock_target = Some(p.clone());
                        append_literal = true;
                    }
                } else {
                    append_literal = true;
                }
            } else {
                append_literal = true;
            }

            if let Some(tok_name) = invalid_lock_target {
                diagnostics.push(LpsWarning::W101InvalidLock(tok_name));
            }
            if append_literal {
                tokens.push(Stage0Token::new(TokenKind::Literal("'".to_string())));
            }
            i += 1;
            continue;
        }

        // 3. Join operator (+)
        if b == b'+' {
            if let Some(last) = tokens.last_mut() {
                last.has_join = true;
            } else {
                tokens.push(Stage0Token::new(TokenKind::Literal("+".to_string())));
                diagnostics.push(LpsWarning::W102JoinAtWordStart);
            }
            i += 1;
            continue;
        }

        // 4. Split operator (-)
        if b == b'-' {
            if i + 1 < len && bytes[i + 1] == b'-' {
                // -- yields literal hyphen
                tokens.push(Stage0Token::new(TokenKind::Literal("-".to_string())));
                i += 2;
                continue;
            }
            // Consumed between phonetic tokens; marks split
            pending_split = true;
            i += 1;
            continue;
        }

        // 5. Chandrabindu (^)
        if b == b'^' {
            if let Some(last) = tokens.last_mut() {
                last.has_chandrabindu = true;
            } else {
                tokens.push(Stage0Token::new(TokenKind::Literal("^".to_string())));
                diagnostics.push(LpsWarning::W103NoChandrabinduHost);
            }
            i += 1;
            continue;
        }

        // 6. Whitespace or punctuation (preserves as literal or finishes word)
        if !b.is_ascii_alphabetic() {
            tokens.push(Stage0Token::new(TokenKind::Literal((b as char).to_string())));
            i += 1;
            continue;
        }

        // 7. Phonetic Tokens (Longest-match first, case-insensitive)
        let rem_str = std::str::from_utf8(&bytes[i..]).unwrap_or("");
        let rem_lower = rem_str.to_ascii_lowercase();

        let mut matched = false;
        // Check multi-letter tokens
        for &ml in MULTI_LETTER_TOKENS {
            if rem_lower.starts_with(ml) {
                let mut tok = Stage0Token::new(TokenKind::Phonetic(ml.to_string()));
                if pending_split {
                    tok.has_split_before = true;
                    pending_split = false;
                }
                tokens.push(tok);
                i += ml.len();
                matched = true;
                break;
            }
        }

        if !matched {
            // Single ASCII character
            let ch = rem_lower.chars().next().unwrap();
            let mut tok = Stage0Token::new(TokenKind::Phonetic(ch.to_string()));
            if pending_split {
                tok.has_split_before = true;
                pending_split = false;
            }
            tokens.push(tok);
            i += ch.len_utf8();
        }
    }

    // Alternative segmentations (e.g. "sh" -> "s" + "h", "ng" -> "n" + "g")
    let alternative_segmentations = generate_alternative_segmentations(&tokens);

    Stage0Result {
        primary_tokens: tokens,
        alternative_segmentations,
        diagnostics,
    }
}

/// Checks whether a phonetic token supports the ' lock operator.
pub fn is_lockable_token(tok: &str) -> bool {
    matches!(
        tok,
        "ng" | "j" | "t" | "th" | "d" | "dh" | "n" | "r" | "s" | "sh" | "h" | "y" | "o" | "oi" | "ou" | "ri"
    )
}

/// Generates alternative decompositions (e.g., sh -> s + h, ng -> n + g) for Stage 1 exploration.
fn generate_alternative_segmentations(tokens: &[Stage0Token]) -> Vec<Vec<Stage0Token>> {
    let mut alternatives = Vec::new();

    for (idx, tok) in tokens.iter().enumerate() {
        if let TokenKind::Phonetic(ref p) = tok.kind {
            let split_parts: Option<(&str, &str)> = match p.as_str() {
                "sh" => Some(("s", "h")),
                "ng" => Some(("n", "g")),
                "ksh" => Some(("k", "sh")),
                "ch" => Some(("c", "h")),
                "th" => Some(("t", "h")),
                "dh" => Some(("d", "h")),
                "ph" => Some(("p", "h")),
                "bh" => Some(("b", "h")),
                "kh" => Some(("k", "h")),
                "gh" => Some(("g", "h")),
                "jh" => Some(("j", "h")),
                _ => None,
            };

            if let Some((p1, p2)) = split_parts {
                let mut alt = tokens.to_vec();
                let mut t1 = Stage0Token::new(TokenKind::Phonetic(p1.to_string()));
                t1.has_split_before = tok.has_split_before;
                let mut t2 = Stage0Token::new(TokenKind::Phonetic(p2.to_string()));
                t2.lock_count = tok.lock_count;
                t2.has_join = tok.has_join;
                t2.has_chandrabindu = tok.has_chandrabindu;

                alt.remove(idx);
                alt.insert(idx, t2);
                alt.insert(idx, t1);
                alternatives.push(alt);
            }
        }
    }

    alternatives
}
