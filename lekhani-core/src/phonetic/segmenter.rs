//! Concatenated Word Lattice Segmenter
//!
//! Evaluates unspaced Latin input strings (e.g. "kemonaso", "dhonnobadbhai", "kihoise")
//! and finds optimal split points into multi-word Bengali phrases.

use super::database::PhoneticDatabase;
use super::fuzzy::cuts_layout_digraph;

/// Candidate segmented phrase hypothesis
#[derive(Debug, Clone)]
pub struct SegmentedPhrase {
    pub text: String,
    pub score: i32,
}

/// Attempt to segment a long unspaced token into a 2-word phrase
pub fn segment_concatenated_token(
    input: &str,
    database: &PhoneticDatabase,
    convert_fn: impl Fn(&str) -> String,
) -> Vec<SegmentedPhrase> {
    let lower = input.to_ascii_lowercase();
    let len = lower.len();

    // Minimum length for a 2-word compound is 6 letters (e.g. "kiholo", "kemonaso", "dhonnobadbhai")
    if len < 6 || len > 28 || lower.contains(' ') {
        return Vec::new();
    }

    const VALID_SHORT_WORDS: &[&str] = &[
        "am", "ba", "ei", "ek", "ha", "ja", "je", "ke", "ki", "ma", "na", "oi", "se", "ta",
    ];

    let resolve_part = |part: &str| -> Option<(String, u32)> {
        if part.len() < 3 && !VALID_SHORT_WORDS.contains(&part) {
            return None;
        }

        // 1. Check autocorrect / Banglish shorthand
        if let Some(ac) = database.get_autocorrect_raw(part) {
            if is_isolated_consonant(&ac) {
                return None;
            }
            let freq = database.get_frequency(&ac).max(1500);
            return Some((ac, freq));
        }

        // 2. Direct phonetic transliteration
        let conv = convert_fn(part);
        if !conv.is_empty() && database.is_exact_dictionary_word(&conv) {
            if is_isolated_consonant(&conv) {
                return None;
            }
            let freq = database.get_frequency(&conv);
            return Some((conv, freq));
        }

        None
    };

    let mut hypotheses = Vec::new();

    // Try split point from 2 to len - 2
    for i in 2..=(len - 2) {
        if cuts_layout_digraph(&lower, i) {
            continue;
        }

        let part1 = &lower[..i];
        let part2 = &lower[i..];

        if let (Some((bn1, freq1)), Some((bn2, freq2))) = (resolve_part(part1), resolve_part(part2)) {
            if is_isolated_consonant(&bn1) || is_isolated_consonant(&bn2) {
                continue;
            }
            let combined = format!("{} {}", bn1, bn2);
            let score = (freq1.min(5000) as i32) + (freq2.min(5000) as i32);
            hypotheses.push(SegmentedPhrase {
                text: combined,
                score,
            });
        }
    }

    hypotheses.sort_by(|a, b| b.score.cmp(&a.score));
    hypotheses.dedup_by(|a, b| a.text == b.text);
    hypotheses.truncate(3);
    hypotheses
}

fn is_isolated_consonant(s: &str) -> bool {
    let mut chars = s.chars();
    if let Some(c) = chars.next() {
        if chars.next().is_none() {
            return matches!(c, '\u{0995}'..='\u{09B9}' | '\u{09DC}'..='\u{09DF}' | '\u{09CE}');
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segmentation_logic() {
        let db = PhoneticDatabase::new();
        let convert = |s: &str| -> String {
            match s {
                "kemon" => "কেমন".to_string(),
                "aso" => "আছো".to_string(),
                "dhonnobad" => "ধন্যবাদ".to_string(),
                "bhai" => "ভাই".to_string(),
                "ki" => "কী".to_string(),
                "holo" => "হলো".to_string(),
                _ => s.to_string(),
            }
        };

        let res = segment_concatenated_token("kemonaso", &db, convert);
        assert!(!res.is_empty());
        assert_eq!(res[0].text, "কেমন আছো");

        let res2 = segment_concatenated_token("dhonnobadbhai", &db, convert);
        assert!(!res2.is_empty());
        assert_eq!(res2[0].text, "ধন্যবাদ ভাই");
    }
}
