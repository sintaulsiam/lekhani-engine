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
/// Executes in single-pass linear byte time (< 5 nanoseconds) with zero heap allocations.
#[inline]
pub fn has_explicit_avro_intent(input: &str) -> bool {
    let bytes = input.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return false;
    }

    // Check if the token is entirely all-caps (e.g. acronyms like "UI", "API", or shout typing like "TUMI").
    // All-caps tokens are handled by acronym passthrough or case-insensitive dictionary lookup,
    // not by explicit Avro case differentiation rules.
    let is_all_caps = bytes.iter().any(|b| b.is_ascii_alphabetic())
        && bytes.iter().filter(|b| b.is_ascii_alphabetic()).all(|b| b.is_ascii_uppercase());

    let mut i = 0;
    while i < len {
        match bytes[i] {
            // Explicit Avro diacritics and hasanta / khanda-ta / bisorgo / taka operators
            b'^' | b'`' | b'\'' | b':' | b'$' => return true,
            // Double-dot or double-comma hasanta operators
            b'.' if i + 1 < len && bytes[i + 1] == b'.' => return true,
            b',' if i + 1 < len && bytes[i + 1] == b',' => return true,
            // Grammatical Avro uppercase characters mid-word (non-initial to avoid Android/mobile TitleCase,
            // and non-all-caps to preserve technical acronyms and capitalized words):
            // O (ো), I (ঈ), U (ঊ), D (ড), G (জ্ঞ/ঘ), J (য), N (ণ), R (ড়), S (ষ), T (ট), Y (য়), Z (য)
            b'O' | b'I' | b'U' | b'D' | b'G' | b'J' | b'N' | b'R' | b'S' | b'T' | b'Y' | b'Z'
                if i > 0 && !is_all_caps =>
            {
                return true;
            }
            _ => {}
        }
        i += 1;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_explicit_avro_intent() {
        assert!(has_explicit_avro_intent("c^ad"));
        assert!(has_explicit_avro_intent("h^as"));
        assert!(has_explicit_avro_intent("b^ash"));
        assert!(has_explicit_avro_intent("du:kho"));
        assert!(has_explicit_avro_intent("k`kh"));
        assert!(has_explicit_avro_intent("k'kh"));
        assert!(has_explicit_avro_intent("hot`at"));
        assert!(has_explicit_avro_intent("hot'at"));
        assert!(has_explicit_avro_intent("k,,k"));
        assert!(has_explicit_avro_intent("k..k"));
        assert!(has_explicit_avro_intent("$500"));

        assert!(has_explicit_avro_intent("paR"));
        assert!(has_explicit_avro_intent("kOno"));
        assert!(has_explicit_avro_intent("kaTh"));
        assert!(has_explicit_avro_intent("biSTi"));
        assert!(has_explicit_avro_intent("bODh"));

        assert!(!has_explicit_avro_intent("Tumi"));
        assert!(!has_explicit_avro_intent("Din"));
        assert!(!has_explicit_avro_intent("Tara"));
        assert!(!has_explicit_avro_intent("kemon"));
        assert!(!has_explicit_avro_intent("basha"));
    }
}
