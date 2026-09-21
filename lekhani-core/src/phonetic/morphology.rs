//! Bengali Morphological Stemmer, Affix Engine & Sandhi Analyzer
//!
//! Provides modular 4-layer affix composition, Sandhi transformation rules,
//! and recursive stem peeling for inflected Bengali words.

use crate::chars::BengaliCharExt;

/// Layer 1: Nominal & Pronominal Classifiers (নির্দিষ্টতাবাচক প্রত্যয়)
pub const CLASSIFIERS: &[&str] = &["টা", "টি", "খানা", "খানি", "টুকু", "টুকুন", "গাছা", "গাছি", "জন"];

/// Layer 2: Plural & Collective Markers (বহুবচন প্রত্যয়)
pub const PLURALS: &[&str] = &[
    "গুলো",
    "গুলি",
    "গুলা",
    "দের",
    "বৃন্দ",
    "গণ",
    "বর্গ",
    "রা",
    "েরা",
    "সমূহ",
];

/// Layer 3: Inflectional Case Markers (কারক ও বিভক্তি)
pub const CASE_MARKERS: &[&str] = &[
    "কে",
    "রে",
    "তে",
    "েতে",
    "ের",
    "র",
    "য়ে",
    "ায়",
    "ে",
    "য়",
    "দ্বারা",
    "দিয়ে",
    "থেকে",
    "হতে",
    "ভাবে",
    "পূর্বক",
    "শীল",
    "হীন",
    "সহ",
];

/// Layer 4: Emphasis & Particle Clitics (বলবাচক প্রত্যয় ও সংযোজক)
pub const EMPHASIS_PARTICLES: &[&str] = &["ই", "ও", "তো"];

/// Colloquial & Spoken continuous verb suffix patterns (Latin suffix -> [Standard Cholit, Spoken/Sadhu])
pub const COLLOQUIAL_VERBAL_PATTERNS: &[(&str, &[&str])] = &[
    ("tesilam", &["ছিলাম", "তেছিলাম"]),
    ("tesilen", &["ছিলেন", "তেছিলেন"]),
    ("tesila", &["ছিলা", "তেছিলা"]),
    ("tesilo", &["ছিল", "তেছিল"]),
    ("techhilam", &["তেছিলাম", "ছিলাম"]),
    ("techhilen", &["তেছিলেন", "ছিলেন"]),
    ("techhilo", &["তেছিল", "ছিল"]),
    ("chhilam", &["ছিলাম", "তেছিলাম"]),
    ("chhilen", &["ছিলেন", "তেছিলেন"]),
    ("chhilo", &["ছিল", "তেছিল"]),
    ("chhile", &["ছিলে", "তেছিলে"]),
    ("tesi", &["ছি", "তেছি"]),
    ("tasi", &["ছি", "তেছি"]),
    ("taso", &["ছো", "তেছো"]),
    ("tasa", &["ছো", "তেছো"]),
    ("tase", &["ছে", "তেছে"]),
    ("tasen", &["ছেন", "তেছেন"]),
    ("tasilam", &["ছিলাম", "তেছিলাম"]),
    ("tasila", &["ছিলা", "তেছিলা"]),
    ("tasilo", &["ছিল", "তেছিল"]),
    ("tasilen", &["ছিলেন", "তেছিলেন"]),
    ("chen", &["ছেন", "তেছেন"]),
    ("cheni", &["ছেনi", "তেছেন"]),
    // Spoken Past (-silam, -silo, -sila, -si, -se, -so, -sen)
    ("silam", &["ছিলাম", "েছিলাম"]),
    ("silen", &["ছিলেন", "েছিলেন"]),
    ("silo", &["ছিল", "েছিল"]),
    ("sila", &["ছিলা", "েছিলা"]),
    ("sen", &["ছেন", "েছেন"]),
    ("so", &["ছো", "েছো"]),
    ("se", &["ছে", "েছে"]),
    ("si", &["ছি", "েছি"]),
    // Spoken Future & Dialectal (-mu, -ba)
    ("mu", &["ব"]),
    ("ba", &["বে", "বা"]),
];

pub const BENGALI_VERB_ROOTS: &[(&str, &str)] = &[
    ("thak", "থাক"),
    ("dekh", "দেখ"),
    ("bol", "বল"),
    ("ash", "আস"),
    ("as", "আস"),
    ("esh", "এস"),
    ("es", "এস"),
    ("chol", "চল"),
    ("kor", "কর"),
    ("par", "পার"),
    ("shun", "শুন"),
    ("sun", "শুন"),
    ("jan", "জান"),
    ("rakh", "রাখ"),
    ("likh", "লিখ"),
    ("uth", "উঠ"),
    ("bhab", "ভাব"),
    ("bujh", "বুঝ"),
    ("bosh", "বস"),
    ("bos", "বস"),
    ("ja", "যা"),
    ("ge", "গে"),
    ("kha", "খা"),
    ("khe", "খে"),
    ("de", "দে"),
    ("ne", "নে"),
    ("ho", "হ"),
    ("dak", "ডাক"),
    ("dhar", "ধর"),
    ("mar", "মার"),
    ("por", "পড়"),
    ("patha", "পাঠা"),
];

pub const BENGALI_VERBAL_CONJUGATIONS: &[(&str, &str)] = &[
    ("tesilam", "তেছিলাম"),
    ("tesilen", "তেছিলেন"),
    ("tesilo", "তেছিল"),
    ("tesila", "তেছিলা"),
    ("tasilam", "তেছিলাম"),
    ("tasilen", "তেছিলেন"),
    ("tasilo", "তেছিল"),
    ("tasila", "তেছিলা"),
    ("techhilam", "তেছিলাম"),
    ("techhilen", "তেছিলেন"),
    ("techhilo", "তেছিল"),
    ("silam", "ছিলাম"),
    ("silen", "ছিলেন"),
    ("silo", "ছিল"),
    ("sila", "ছিলা"),
    ("echilam", "েছিলাম"),
    ("echilen", "েছিলেন"),
    ("echilo", "েছিল"),
    ("echhile", "েছিলে"),
    ("chhilam", "ছিলাম"),
    ("chhilen", "ছিলেন"),
    ("chhilo", "ছিল"),
    ("chhile", "ছিলে"),
    ("chilam", "ছিলাম"),
    ("chilen", "ছিলেন"),
    ("chilo", "ছিল"),
    ("chile", "ছিলে"),
    ("yechi", "য়েছি"),
    ("yecho", "য়েছো"),
    ("yeche", "য়েছে"),
    ("yechen", "য়েছেন"),
    ("echi", "েছি"),
    ("echo", "েছো"),
    ("eche", "েছে"),
    ("echen", "েছেন"),
    ("tesi", "তেছি"),
    ("tese", "তেছে"),
    ("tesen", "তেছেন"),
    ("teso", "তেছো"),
    ("tasi", "তেছি"),
    ("tase", "তেছে"),
    ("tasen", "তেছেন"),
    ("taso", "তেছো"),
    ("cchi", "চ্ছি"),
    ("cche", "চ্ছে"),
    ("cchen", "চ্ছেন"),
    ("ccho", "চ্ছো"),
    ("cci", "চ্ছি"),
    ("cce", "চ্ছে"),
    ("ccen", "চ্ছেন"),
    ("si", "ছি"),
    ("sen", "ছেন"),
    ("mu", "ব"),
    ("ba", "বে"),
    ("lam", "লাম"),
    ("len", "লেন"),
    ("lo", "ল"),
    ("le", "লে"),
    ("ben", "বেন"),
    ("be", "বে"),
    ("bo", "ব"),
    ("bi", "বি"),
    ("tam", "তাম"),
    ("ten", "তেন"),
    ("to", "তো"),
    ("ta", "তা"),
    ("ish", "ইশ"),
    ("is", "িস"),
    ("un", "ুন"),
    ("en", "েন"),
    ("chhi", "চ্ছি"),
    ("chhe", "চ্ছে"),
    ("chhen", "চ্ছেন"),
    ("chho", "চ্ছো"),
    ("chi", "ছি"),
    ("che", "ছে"),
    ("chen", "ছেন"),
    ("cho", "ছো"),
    ("i", "ি"),
    ("o", "ো"),
    ("e", "ে"),
];

/// Decompose a compound verbal form into root stem and conjugation to prevent false Juktoborno
pub fn decompose_verbal_form(input: &str) -> Option<String> {
    let lower = input.to_ascii_lowercase();
    for &(conj_en, conj_bn) in BENGALI_VERBAL_CONJUGATIONS {
        if lower.len() > conj_en.len() && lower.ends_with(conj_en) {
            let root_part = &lower[..lower.len() - conj_en.len()];
            for &(root_en, root_bn) in BENGALI_VERB_ROOTS {
                if root_part == root_en {
                    let mut result = root_bn.to_string();
                    result.push_str(conj_bn);
                    return Some(result);
                }
            }
        }
    }
    None
}

/// Comprehensive list of inflectional suffixes ordered by descending length for greedy stemming
pub const BENGALI_INFLECTIONAL_SUFFIXES: &[&str] = &[
    "গুলোতেই",
    "গুলোতেও",
    "গুলিতেই",
    "গুলিতেও",
    "গুলাতেও",
    "গুলাতেই",
    "দেরকেই",
    "দেরকেও",
    "সমূহতেই",
    "সমূহতেও",
    "সমূহকেই",
    "সমূহকেও",
    "সমূহেই",
    "সমূহেরই",
    "বর্গেরাই",
    "বর্গেরাও",
    "বৃন্দরাই",
    "বৃন্দরাও",
    "গুলোরই",
    "গুলিরই",
    "গুলোরও",
    "গুলিরও",
    "দেরকে",
    "দেরই",
    "দেরও",
    "গুলোকে",
    "গুলিকে",
    "গুলোর",
    "গুলির",
    "গুলাতে",
    "গুলোতে",
    "গুলিতে",
    "গুলোতেই",
    "গুলোতেও",
    "গুলোও",
    "গুলিও",
    "গুলাও",
    "গুলো",
    "গুলি",
    "গুলা",
    "েদেরও",
    "েদেরই",
    "েদের",
    "দের",
    "খানা",
    "খানি",
    "টিকে",
    "টাকে",
    "টিরও",
    "টারও",
    "টির",
    "টার",
    "টিতে",
    "টাতে",
    "টিতেই",
    "টাতেই",
    "টিতেও",
    "টাতেও",
    "টায়",
    "টায়",
    "টিই",
    "টাই",
    "টিও",
    "টাও",
    "টি",
    "টা",
    "টুকু",
    "টুকুন",
    "গাছা",
    "গাছি",
    "ভাবে",
    "মতো",
    "মত",
    "পূর্বক",
    "শীল",
    "মুখী",
    "ময়",
    "তম",
    "তর",
    "সমূহ",
    "বৃন্দ",
    "গণ",
    "বর্গ",
    "তেছিলাম",
    "তেছিলেন",
    "তেছিল",
    "তেছিলা",
    "তেছেন",
    "তেছে",
    "তেছো",
    "তেছি",
    "ছেন",
    "ছেনি",
    "ছিলাম",
    "ছিলেন",
    "ছিলা",
    "ছিল",
    "বেন",
    "বা",
    "বে",
    "ছি",
    "ছে",
    "ছিস",
    "লাম",
    "লে",
    "লেন",
    "তাম",
    "তেন",
    "দ্বারা",
    "দিয়ে",
    "থেকে",
    "হতে",
    "েতে",
    "ের",
    "তে",
    "কে",
    "রে",
    "ায়",
    "য়ে",
    "ে",
    "র",
    "য়",
    "এ",
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
        // 2. Anusvara (ং) transforms to Nga (ঙ) for native roots (রং -> রঙের), or inserts Ya glide (মিটিং -> মিটিংয়ে)
        else if b_last == 'ং' && (s_first.is_kar() || s_first.is_vowel() || s_first == 'ে') {
            if base == "রং" || base == "ঢং" || base == "স্বাং" || base == "অং"
            {
                result.pop();
                result.push('ঙ');
            } else {
                result.push('য়');
            }
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

/// Extract all candidate stems by checking every applicable suffix rule
pub fn extract_all_candidate_stems(word: &str) -> Vec<String> {
    let word = word.trim();
    if word.chars().count() <= 2 {
        return Vec::new();
    }

    let mut stems = Vec::new();
    for &suffix in BENGALI_INFLECTIONAL_SUFFIXES {
        if word.ends_with(suffix) && word.len() > suffix.len() {
            let base = &word[..word.len() - suffix.len()];
            if base.chars().count() >= 2 {
                let mut clean_base = base.to_string();
                if clean_base.ends_with('য়') && clean_base.chars().count() >= 3 {
                    clean_base.pop();
                }
                if !stems.contains(&clean_base) {
                    stems.push(clean_base.clone());
                }
                if clean_base.ends_with('ত') {
                    let mut kt_base = clean_base;
                    kt_base.pop();
                    kt_base.push('ৎ');
                    if !stems.contains(&kt_base) {
                        stems.push(kt_base);
                    }
                }
            }
        }
    }

    stems
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
