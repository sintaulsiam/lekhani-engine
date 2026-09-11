//! Bengali Morphological Stemmer & Suffix Analyzer
//!
//! Provides rule-based decomposition of inflected Bengali words to extract
//! underlying root stems (e.g. "কুয়েটে" ➔ "কুয়েট", "কম্পিউটারগুলো" ➔ "কম্পিউটার").

/// Common Bengali nominal and verbal inflectional suffixes
pub const BENGALI_INFLECTIONAL_SUFFIXES: &[&str] = &[
    "গুলোকে", "গুলিকে", "গুলোর", "গুলির", "গুলো", "গুলি",
    "দেরকে", "দের", "েদের",
    "খানা", "খানি", "টিকে", "টাকে", "টির", "টার", "টি", "টা",
    "ভাবে", "পূর্বক", "শীল", "মুখী", "ময়", "তম", "তর",
    "ছেন", "ছেনি", "ছিলাম", "ছিলেন", "ছিলা", "ছিল",
    "বেন", "বা", "বে", "ছি", "ছে", "ছিস",
    "লাম", "লে", "লেন", "তাম", "তেন",
    "ের", "েতে", "তে", "কে", "ায়", "য়ে", "ে", "র", "য়", "এ",
];

/// Attempt to extract the underlying base root of an inflected Bengali word
pub fn extract_root_stem(word: &str) -> Option<String> {
    let word = word.trim();
    if word.chars().count() <= 2 {
        return None;
    }

    for &suffix in BENGALI_INFLECTIONAL_SUFFIXES {
        if word.ends_with(suffix) && word.len() > suffix.len() {
            let base = &word[..word.len() - suffix.len()];
            if base.chars().count() >= 2 {
                // Handle Sandhi reversals:
                // If base ends with glide 'য়' when suffix was a kar, strip 'য়'
                let mut clean_base = base.to_string();
                if clean_base.ends_with('য়') && clean_base.chars().count() >= 3 {
                    clean_base.pop();
                }
                return Some(clean_base);
            }
        }
    }

    None
}

/// Analyze a word and return both the full word and candidate extracted stem
pub fn analyze_morphemes(word: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    let word_trimmed = word.trim().to_string();
    if !word_trimmed.is_empty() {
        candidates.push(word_trimmed.clone());
        if let Some(stem) = extract_root_stem(&word_trimmed) {
            if stem != word_trimmed && !candidates.contains(&stem) {
                candidates.push(stem);
            }
        }
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_root_stem() {
        assert_eq!(extract_root_stem("কুয়েটে"), Some("কুয়েট".to_string()));
        assert_eq!(extract_root_stem("ঢাবিতে"), Some("ঢাবি".to_string()));
        assert_eq!(extract_root_stem("কম্পিউটারগুলো"), Some("কম্পিউটার".to_string()));
        assert_eq!(extract_root_stem("শিক্ষার্থীদের"), Some("শিক্ষার্থী".to_string()));
        assert_eq!(extract_root_stem("বইটির"), Some("বই".to_string()));
    }
}
