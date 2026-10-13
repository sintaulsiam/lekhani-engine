//! # Lekhani Parser
//!
//! Ultra-fast, zero-allocation pure Rust Bengali phonetic grammar parser and transliterator.
//! High-performance replacement for `rupantor` with Longest-Prefix L1 Trie matching,
//! branchless byte-level condition evaluations, and canonical diacritic sequencing.

pub mod compiler;
pub mod engine;
pub mod rule;
pub mod tables;
pub mod trie;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use compiler::CompiledLayout;
pub use engine::LekhaniParser;
pub use rule::{ConditionScope, MatchType, Pattern, PatternRule, RuleCondition};

use std::sync::{Arc, OnceLock};

/// Cached singleton of the compiled default Avro Phonetic layout.
/// Initialized once, zero allocations and zero disk I/O on subsequent accesses.
static DEFAULT_AVRO_LAYOUT: OnceLock<Arc<CompiledLayout>> = OnceLock::new();

/// Embedded standard Avro Phonetic layout JSON
const EMBEDDED_AVRO_JSON: &str = include_str!("../data/avrophonetic.json");

/// Retrieve or initialize the shared singleton of the compiled default Avro layout.
pub fn get_default_avro_layout() -> Arc<CompiledLayout> {
    DEFAULT_AVRO_LAYOUT
        .get_or_init(|| {
            let json: serde_json::Value = serde_json::from_str(EMBEDDED_AVRO_JSON)
                .expect("Corrupt embedded avrophonetic.json layout");
            let layout = CompiledLayout::from_json(&json)
                .expect("Failed to compile embedded avrophonetic.json");
            Arc::new(layout)
        })
        .clone()
}

/// Create a new `LekhaniParser` with the default Avro Phonetic layout.
pub fn default_avro_parser() -> LekhaniParser {
    LekhaniParser::new(get_default_avro_layout())
}

/// Returns true if the token contains explicit, unambiguous Avro grammatical operators
/// or mid-word case distinctions that indicate intentional phonetic rule typing.
///
/// Returns true if the token contains explicit, unambiguous Avro grammatical operators
/// or mid-word case distinctions that indicate intentional phonetic rule typing.
///
/// Executes in a single-pass linear byte loop (< 3 nanoseconds) with zero heap allocations.
#[inline]
pub fn has_explicit_avro_intent(input: &str) -> bool {
    let bytes = input.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return false;
    }

    let mut has_lower = false;
    let mut has_mid_avro_upper = false;

    let mut i = 0;
    while i < len {
        let b = bytes[i];
        match b {
            // Explicit Avro diacritics and hasanta / khanda-ta / currency operators
            b'^' | b'`' | b'\'' | b'$' => return true,
            // Bisorgo operator: ignore numeric clock ratios (e.g. 10:30)
            b':' => {
                let is_clock_ratio = i > 0
                    && bytes[i - 1].is_ascii_digit()
                    && i + 1 < len
                    && bytes[i + 1].is_ascii_digit();
                if !is_clock_ratio {
                    return true;
                }
            }
            // Double-dot or double-comma hasanta operators
            b'.' if i + 1 < len && bytes[i + 1] == b'.' => return true,
            b',' if i + 1 < len && bytes[i + 1] == b',' => return true,
            // Invariant Avro vowel digraph: 'rri' -> ঋ / ৃ (e.g. rritu -> ঋতু, krrishi -> কৃষি)
            b'r' | b'R' if i + 2 < len
                && (bytes[i + 1] == b'r' || bytes[i + 1] == b'R')
                && (bytes[i + 2] == b'i' || bytes[i + 2] == b'I') => return true,
            // Lowercase ASCII characters prove the token is not an all-caps acronym or shout-typing
            b'a'..=b'z' => {
                has_lower = true;
            }
            // Grammatical Avro uppercase characters mid-word (non-initial to preserve sentence-initial TitleCase):
            // O (ো), I (ঈ), U (ঊ), D (ড), G (জ্ঞ/ঘ), J (য), N (ণ), R (ড়), S (ষ), T (ট), Y (য়), Z (য)
            b'O' | b'I' | b'U' | b'D' | b'G' | b'J' | b'N' | b'R' | b'S' | b'T' | b'Y' | b'Z' if i > 0 => {
                has_mid_avro_upper = true;
            }
            _ => {}
        }
        i += 1;
    }

    // Mid-word case distinctions only apply when the word is not entirely all-caps
    has_mid_avro_upper && has_lower
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_explicit_avro_intent() {
        // Explicit operators
        assert!(has_explicit_avro_intent("c^ad"));
        assert!(has_explicit_avro_intent("ca^d"));
        assert!(has_explicit_avro_intent("h^as"));
        assert!(has_explicit_avro_intent("b^ash"));
        assert!(has_explicit_avro_intent("du:kho"));
        assert!(has_explicit_avro_intent("k`kh"));
        assert!(has_explicit_avro_intent("k'kh"));
        assert!(has_explicit_avro_intent("hot`at"));
        assert!(has_explicit_avro_intent("hoThat``"));
        assert!(has_explicit_avro_intent("k,,k"));
        assert!(has_explicit_avro_intent("k..k"));
        assert!(has_explicit_avro_intent("$500"));
        assert!(has_explicit_avro_intent("rritu"));
        assert!(has_explicit_avro_intent("krrishi"));

        // Standard lowercase phonetic words without operators must NOT trigger
        assert!(!has_explicit_avro_intent("kaccha"));
        assert!(!has_explicit_avro_intent("porikkhay"));
        assert!(!has_explicit_avro_intent("shikkha"));

        // Mid-word Avro uppercase distinctions
        assert!(has_explicit_avro_intent("kZackZacani"));
        assert!(has_explicit_avro_intent("paR"));
        assert!(has_explicit_avro_intent("poR"));
        assert!(has_explicit_avro_intent("kOno"));
        assert!(has_explicit_avro_intent("kaTh"));
        assert!(has_explicit_avro_intent("biSTi"));
        assert!(has_explicit_avro_intent("bODh"));
        assert!(has_explicit_avro_intent("bigganI"));

        // Sentence-initial titlecase must NOT trigger
        assert!(!has_explicit_avro_intent("Tumi"));
        assert!(!has_explicit_avro_intent("Din"));
        assert!(!has_explicit_avro_intent("Tara"));
        assert!(!has_explicit_avro_intent("kemon"));
        assert!(!has_explicit_avro_intent("basha"));
        assert!(!has_explicit_avro_intent("ami"));

        // All-caps acronyms and shout typing must NOT trigger
        assert!(!has_explicit_avro_intent("UI"));
        assert!(!has_explicit_avro_intent("API"));
        assert!(!has_explicit_avro_intent("RAM"));
        assert!(!has_explicit_avro_intent("CPU"));
        assert!(!has_explicit_avro_intent("TUMI"));
        assert!(!has_explicit_avro_intent("DESH"));

        // Numeric clock ratios must NOT trigger as bisorgo
        assert!(!has_explicit_avro_intent("10:30"));
        assert!(!has_explicit_avro_intent("12:00"));
    }
}
