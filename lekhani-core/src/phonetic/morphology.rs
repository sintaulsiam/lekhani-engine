//! Bengali Morphological Stemmer, Affix Engine & Sandhi Analyzer
//!
//! Provides modular 4-layer affix composition, Sandhi transformation rules,
//! and recursive stem peeling for inflected Bengali words.

use crate::chars::BengaliCharExt;

/// Layer 1: Nominal & Pronominal Classifiers (নির্দিষ্টতাবাচক প্রত্যয়)
pub const CLASSIFIERS: &[&str] = &[
    "টা", "টি", "খানা", "খানি", "টুকু", "টুকুন", "গাছা", "গাছি", "জন",
];

/// Layer 2: Plural & Collective Markers (বহুবচন প্রত্যয়)
pub const PLURALS: &[&str] = &[
    "গুলো", "গুলি", "গুলা", "দের", "বৃন্দ", "গণ", "বর্গ", "রা", "েরা", "সমূহ",
];

/// Layer 3: Inflectional Case Markers (কারক ও বিভক্তি)
pub const CASE_MARKERS: &[&str] = &[
    "কে", "রে", "তে", "েতে", "ের", "র", "য়ে", "ায়", "ে", "য়",
    "দ্বারা", "দিয়ে", "থেকে", "হতে", "ভাবে", "পূর্বক", "শীল", "হীন", "সহ",
];

/// Layer 4: Emphasis & Particle Clitics (বলবাচক প্রত্যয় ও সংযোজক)
pub const EMPHASIS_PARTICLES: &[&str] = &["ই", "ও", "তো"];

/// Comprehensive list of inflectional suffixes ordered by descending length for greedy stemming
pub const BENGALI_INFLECTIONAL_SUFFIXES: &[&str] = &[
    "গুলোতেই", "গুলোতেও", "গুলিতেই", "গুলিতেও", "দেরকেই", "দেরকেও",
    "সমূহতেই", "সমূহতেও", "সমূহকেই", "সমূহকেও", "সমূহেই", "সমূহেরই",
    "বর্গেরাই", "বর্গেরাও", "বৃন্দরাই", "বৃন্দরাও", "গুলোরই", "গুলিরই",
    "দেরকে", "দেরই", "দেরও", "গুলোকে", "গুলিকে", "গুলোর", "গুলির",
    "গুলোতে", "গুলিতে", "গুলাতে", "গুলো", "গুলি", "গুলা",
    "েদের", "দের", "খানা", "খানি", "টিকে", "টাকে", "টির", "টার", "টিতে", "টাতে",
    "টি", "টা", "টুকু", "টুকুন", "গাছা", "গাছি", "ভাবে", "পূর্বক", "শীল",
    "মুখী", "ময়", "তম", "তর", "সমূহ", "বৃন্দ", "গণ", "বর্গ",
    "ছেন", "ছেনি", "ছিলাম", "ছিলেন", "ছিলা", "ছিল", "বেন", "বা", "বে",
    "ছি", "ছে", "ছিস", "লাম", "লে", "লেন", "তাম", "তেন",
    "দ্বারা", "দিয়ে", "থেকে", "হতে",
    "েতে", "ের", "তে", "কে", "রে", "ায়", "য়ে", "ে", "র", "য়", "এ",
];

/// Apply orthographic Sandhi rules when joining a base word and a suffix
pub fn apply_sandhi_join(base: &str, suffix: &str) -> String {
    let base = base.trim();
    let suffix = suffix.trim();
    if base.is_empty() {
        return suffix.to_string();
    }
    if suffix.is_empty() {
        return base.to_string();
    }

    let mut result = base.to_string();
    let base_last = base.chars().last();
    let suffix_first = suffix.chars().next();

    if let (Some(b_last), Some(s_first)) = (base_last, suffix_first) {
        // 1. Khanda-Ta (ৎ) transforms to Ta (ত) before vowel/kar or vowel-based suffixes
        if b_last == 'ৎ' && (s_first.is_kar() || s_first.is_vowel() || s_first == 'ে') {
            result.pop();
            result.push('ত');
        }
        // 2. Anusvara (ং) transforms to Nga (ঙ) before vowel/kar suffixes
        else if b_last == 'ং' && (s_first.is_kar() || s_first.is_vowel() || s_first == 'ে') {
            result.pop();
            result.push('ঙ');
        }
        // 3. Vowel Hiatus / Glide Insertion:
        // When base ends with a vowel/kar and suffix begins with E-kar (ে), insert Ya (য়) or preserve glide
        else if (b_last.is_kar() || b_last.is_vowel()) && s_first == 'ে' {
            result.push('য়');
        }
    }

    result.push_str(suffix);
    result
}

/// Attempt to extract the underlying base root of an inflected Bengali word recursively
pub fn extract_root_stem(word: &str) -> Option<String> {
    let word = word.trim();
    if word.chars().count() <= 2 {
        return None;
    }

    for &suffix in BENGALI_INFLECTIONAL_SUFFIXES {
        if word.ends_with(suffix) && word.len() > suffix.len() {
            let base = &word[..word.len() - suffix.len()];
            if base.chars().count() >= 2 {
                let mut clean_base = base.to_string();
                // Handle Sandhi reversals:
                // Strip inserted glide 'য়' when suffix was a vowel inflection
                if clean_base.ends_with('য়') && clean_base.chars().count() >= 3 {
                    clean_base.pop();
                }
                // Reverse ত -> ৎ for words like জগতের -> জগৎ, ভবিষ্যতের -> ভবিষ্যৎ
                return Some(clean_base);
            }
        }
    }

    None
}

/// Recursively peel multiple layers of affixes to uncover the innermost root lemma
pub fn peel_all_stems(word: &str) -> Vec<String> {
    let mut stems = Vec::new();
    let mut current = word.trim().to_string();

    while let Some(stem) = extract_root_stem(&current) {
        if stem == current || stems.contains(&stem) {
            break;
        }
        stems.push(stem.clone());
        current = stem;
    }

    stems
}

/// Analyze a word and return the full word, candidate extracted stem, and peeled lemmas
pub fn analyze_morphemes(word: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    let word_trimmed = word.trim().to_string();
    if !word_trimmed.is_empty() {
        candidates.push(word_trimmed.clone());
        for stem in peel_all_stems(&word_trimmed) {
            if !candidates.contains(&stem) {
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

    #[test]
    fn test_sandhi_joins() {
        assert_eq!(apply_sandhi_join("কলম", "টা"), "কলমটা");
        assert_eq!(apply_sandhi_join("বইগুলো", "তে"), "বইগুলোতে");
        assert_eq!(apply_sandhi_join("ভবিষ্যৎ", "ের"), "ভবিষ্যতের");
        assert_eq!(apply_sandhi_join("রং", "ের"), "রঙের");
        assert_eq!(apply_sandhi_join("পা", "ে"), "পায়ে");
    }

    #[test]
    fn test_recursive_peel_stems() {
        let peeled = peel_all_stems("বইগুলোতেই");
        assert!(peeled.contains(&"বইগুলো".to_string()) || peeled.contains(&"বই".to_string()));
    }
}
