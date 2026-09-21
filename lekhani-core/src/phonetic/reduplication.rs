//! Bengali Reduplicated Words (দ্বিরুক্ত শব্দ) Predictor
//!
//! Recognizes canonical Bengali reduplicative roots (adverbs, emphasis,
//! adjectives, onomatopoeia) and instantly predicts the matching pair upon committing.

/// Canonical Bengali words that are predominantly used in reduplicated pairs (দ্বিরুক্ত শব্দ)
pub const REDUPLICATIVE_WORDS: &[&str] = &[
    // Adverbs & Movement
    "ধীরে", "আস্তে", "মাঝে", "বার", "ঘন", "জলদি", "দেরি", "শীঘ্র",
    // Adjectives & Descriptions
    "ছোট", "বড়", "ভালো", "নতুন", "পুরাতন", "পুরনো", "গরম", "ঠান্ডা",
    "লাল", "নীল", "কালো", "সাদা", "কাঁচা", "পাকা", "লম্বা", "মোটা",
    "মিষ্টি", "ঝাল", "তিক্ত", "ভারী", "হালকা",
    // Onomatopoeia (ধ্বনাত্মক ও অনুকার দ্বিরুক্তি)
    "টিপ", "টুপ", "ঝিরি", "শন", "ভন", "ঝন", "কল", "টল", "খট", "ফট",
    "পট", "ঠক", "চিক", "ঝিক", "মিট", "টিম", "ধুক", "চুপ", "টুক",
    "ঝম", "গম", "রম", "ঝল", "টলমল", "ঝলমল", "হুহু", "ধুধু",
    // Psychological & Body States
    "কেমন", "এমন", "ভয়ে", "লাজে", "মনে", "চোখে", "হাতে", "মুখে", "বুকে",
    "ঘরে", "দেশে", "পথে", "ঘাটে", "বনে",
];

/// Check if a committed Bengali word is a candidate for instant reduplication
pub fn is_reduplicative_word(word: &str) -> bool {
    let clean = word.trim_matches(|c: char| {
        c.is_ascii_punctuation()
            || c == '।'
            || c == '—'
            || c == '‘'
            || c == '’'
            || c == '“'
            || c == '”'
            || c == '\''
            || c == '"'
            || c == ','
    });

    REDUPLICATIVE_WORDS.contains(&clean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduplicative_words() {
        assert!(is_reduplicative_word("ধীরে"));
        assert!(is_reduplicative_word("মাঝে"));
        assert!(is_reduplicative_word("ছোট"));
        assert!(is_reduplicative_word("টিপ"));
        assert!(is_reduplicative_word("গরম"));

        // Non-reduplicative normal nouns/pronouns
        assert!(!is_reduplicative_word("বাংলাদেশ"));
        assert!(!is_reduplicative_word("কম্পিউটার"));
    }
}
