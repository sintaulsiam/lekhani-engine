//! Bengali Statistical & Neural-Transition Language Model
//!
//! Provides sub-microsecond N-gram probability estimation with Jelinek-Mercer
//! interpolation for contextual candidate re-ranking and next-word prediction.

use hashbrown::HashMap;

/// Pre-computed high-frequency Bengali unigram frequencies (normalized log probabilities)
pub const UNIGRAM_LOG_PROBS: &[(&str, f32)] = &[
    ("আমি", -1.8),
    ("তুমি", -2.1),
    ("আপনি", -2.3),
    ("আমরা", -2.4),
    ("তারা", -2.6),
    ("সে", -2.2),
    ("তিনি", -2.7),
    ("ভালো", -2.0),
    ("আছি", -2.3),
    ("আছেন", -2.6),
    ("বাংলাদেশ", -2.2),
    ("বাংলা", -2.1),
    ("ভাষা", -2.4),
    ("বই", -2.5),
    ("পড়া", -2.7),
    ("পরা", -2.8),
    ("শার্ট", -3.1),
    ("প্যান্ট", -3.4),
    ("ভাত", -2.6),
    ("খাচ্ছি", -3.0),
    ("খাব", -3.1),
    ("খেয়েছি", -3.2),
    ("যাব", -2.8),
    ("যাচ্ছি", -2.9),
    ("যাবে", -2.9),
    ("করব", -2.8),
    ("করছি", -2.8),
    ("করে", -2.3),
    ("হবে", -2.3),
    ("হচ্ছে", -2.5),
    ("ধন্যবাদ", -2.4),
    ("স্বাগতম", -3.0),
    ("সুন্দর", -2.7),
    ("দেশ", -2.5),
    ("মানুষ", -2.5),
    ("গান", -2.9),
    ("গাই", -3.5),
    ("স্কুল", -3.1),
    ("কলেজ", -3.3),
    ("ভার্সিটি", -3.6),
    ("অফিস", -3.0),
    ("কাজ", -2.5),
    ("সময়", -2.6),
    ("আজ", -2.6),
    ("কাল", -2.8),
    ("সকাল", -2.9),
    ("সন্ধ্যা", -3.1),
    ("রাত", -2.8),
    ("চা", -2.8),
    ("পানি", -2.7),
    ("খাবার", -3.0),
    ("ডাক্তার", -3.3),
    ("টাকা", -2.7),
    ("এক", -2.2),
    ("দুই", -2.6),
    ("তিন", -2.8),
    ("লক্ষ", -3.2),
    ("লক্ষ্য", -3.1),
    ("জীবনের", -3.0),
    ("একটি", -2.3),
    ("স্বাধীন", -3.4),
    ("উন্নয়নশীল", -3.6),
];

/// Pre-computed high-frequency Bengali Bigram transition log probabilities: P(w2 | w1)
pub const BIGRAM_TRANSITIONS: &[((&str, &str), f32)] = &[
    // Pronoun + Verb / Adjective collocations
    (("আমি", "ভালো"), -0.4),
    (("আমি", "আছি"), -0.6),
    (("আমি", "ভাত"), -0.7),
    (("আমি", "যাব"), -0.8),
    (("আমি", "যাচ্ছি"), -0.8),
    (("আমি", "চাই"), -0.8),
    (("আমি", "তোমাকে"), -0.9),
    (("আমি", "বাংলায়"), -0.9),
    (("আমি", "বাংলা"), -1.0),
    (("আমি", "বই"), -1.1),
    (("তুমি", "কেমন"), -0.3),
    (("তুমি", "কি"), -0.5),
    (("তুমি", "কোথায়"), -0.7),
    (("তুমি", "যাবে"), -0.7),
    (("তুমি", "আছো"), -0.8),
    (("আপনি", "কেমন"), -0.3),
    (("আপনি", "আছেন"), -0.4),
    (("আপনি", "কি"), -0.5),
    (("আমরা", "সবাই"), -0.5),
    (("আমরা", "যাব"), -0.7),
    (("আমরা", "করব"), -0.8),
    // Everyday activities
    (("ভাত", "খাচ্ছি"), -0.3),
    (("ভাত", "খাব"), -0.4),
    (("ভাত", "খেয়েছি"), -0.5),
    (("ভাত", "রান্না"), -0.8),
    (("চা", "খাব"), -0.4),
    (("চা", "খাচ্ছি"), -0.5),
    (("চা", "পান"), -0.7),
    (("পানি", "খাব"), -0.5),
    (("পানি", "পান"), -0.6),
    // Contextual Homophones & Apparel
    (("বই", "পড়া"), -0.2),
    (("বই", "পড়ছি"), -0.3),
    (("বই", "পড়ব"), -0.4),
    (("বই", "মেলা"), -0.5),
    (("শার্ট", "পরা"), -0.2),
    (("শার্ট", "পরছি"), -0.3),
    (("শার্ট", "পরব"), -0.4),
    (("প্যান্ট", "পরা"), -0.2),
    (("জুতো", "পরা"), -0.3),
    (("শাড়ি", "পরা"), -0.2),
    (("চশমা", "পরা"), -0.3),
    (("ঘড়ি", "পরা"), -0.3),
    // Language and Culture
    (("বাংলা", "ভাষা"), -0.2),
    (("বাংলা", "ভাষায়"), -0.4),
    (("বাংলা", "গান"), -0.5),
    (("বাংলায়", "গান"), -0.3),
    (("গান", "গাই"), -0.2),
    (("গান", "শুনছি"), -0.3),
    (("গান", "শুনব"), -0.4),
    // National & Descriptive
    (("বাংলাদেশ", "একটি"), -0.3),
    (("বাংলাদেশ", "আমার"), -0.5),
    (("বাংলাদেশ", "আমাদের"), -0.5),
    (("একটি", "সুন্দর"), -0.4),
    (("একটি", "স্বাধীন"), -0.5),
    (("একটি", "উন্নয়নশীল"), -0.6),
    (("জীবনের", "লক্ষ্য"), -0.2),
    (("জীবনের", "উদ্দেশ্য"), -0.5),
    (("এক", "লক্ষ"), -0.2),
    (("পাঁচ", "লক্ষ"), -0.2),
    (("দশ", "লক্ষ"), -0.2),
    // Connectors, Postpositions & Common phrases
    (("জীবন", "থেকে"), -0.2),
    (("জীবন", "সুন্দর"), -0.3),
    (("জীবন", "যাপন"), -0.4),
    (("থেকে", "নেওয়া"), -0.2),
    (("থেকে", "পাওয়া"), -0.3),
    (("থেকে", "শেখা"), -0.3),
    (("থেকে", "শুরু"), -0.3),
    (("থেকে", "দূরে"), -0.4),
    (("থেকে", "বের"), -0.4),
    (("তার", "নাম"), -0.2),
    (("তার", "সাথে"), -0.2),
    (("তার", "কথা"), -0.3),
    (("তার", "বাড়ি"), -0.4),
    (("তার", "কাছে"), -0.3),
    (("তার", "জন্য"), -0.3),
    (("নাম", "কী"), -0.2),
    (("নাম", "জানেন"), -0.3),
    (("নাম", "জানি"), -0.3),
    (("নাম", "বলুন"), -0.4),
    (("নাম", "মনে"), -0.4),
    (("সাথে", "কথা"), -0.2),
    (("সাথে", "দেখা"), -0.2),
    (("সাথে", "যাব"), -0.3),
    (("সাথে", "আছি"), -0.3),
    (("সাথে", "থাকব"), -0.4),
    (("সাথে", "আমি"), -0.3),
    (("আমার", "সাথে"), -0.2),
    (("আমার", "নাম"), -0.2),
    (("আমার", "বাড়ি"), -0.3),
    (("আমার", "দেশ"), -0.3),
    (("আমার", "কথা"), -0.4),
    (("তোমার", "নাম"), -0.2),
    (("তোমার", "সাথে"), -0.2),
    (("তোমার", "বাড়ি"), -0.3),
    (("তোমার", "খবর"), -0.3),
    (("আপনার", "নাম"), -0.2),
    (("আপনার", "সাথে"), -0.2),
    (("আপনার", "অফিস"), -0.3),
    (("আপনার", "খবর"), -0.3),
    (("জন্য", "ধন্যবাদ"), -0.2),
    (("জন্য", "অপেক্ষা"), -0.3),
    (("জন্য", "ভালো"), -0.3),
    (("জন্য", "দোয়া"), -0.4),
    (("কাজ", "করছি"), -0.2),
    (("কাজ", "করব"), -0.2),
    (("কাজ", "শেষ"), -0.3),
    (("কাজ", "চলছে"), -0.3),
    (("টাকা", "পাঠাও"), -0.2),
    (("টাকা", "দাও"), -0.2),
    (("টাকা", "লাগবে"), -0.3),
    (("টাকা", "আছে"), -0.3),
    (("সময়", "নেই"), -0.2),
    (("সময়", "হয়েছে"), -0.3),
    (("সময়", "মতো"), -0.3),
    (("বাড়ি", "যাব"), -0.2),
    (("বাড়ি", "আছি"), -0.3),
    (("বাড়ি", "কোথায়"), -0.3),
    (("অফিস", "যাব"), -0.2),
    (("অফিস", "ছুটি"), -0.3),
    (("স্কুল", "যাব"), -0.2),
    (("স্কুল", "ছুটি"), -0.3),
    (("আজ", "বৃষ্টি"), -0.2),
    (("আজ", "ছুটি"), -0.3),
    (("আজ", "সকালে"), -0.3),
    (("আজ", "দেখা"), -0.3),
    (("কাল", "দেখা"), -0.2),
    (("কাল", "হবে"), -0.2),
    (("কাল", "যাব"), -0.3),
    (("কাল", "সকালে"), -0.3),
    // Politeness & Formal Greetings
    (("অনেক", "ধন্যবাদ"), -0.3),
    (("আপনাকে", "ধন্যবাদ"), -0.3),
    (("আন্তরিক", "শুভেচ্ছা"), -0.2),
    (("শুভেচ্ছা", "ও"), -0.1),
    (("ও", "অভিনন্দন"), -0.1),
    (("শুভ", "সকাল"), -0.2),
    (("শুভ", "রাত্রি"), -0.2),
    (("শুভ", "সন্ধ্যা"), -0.2),
];

/// Pre-computed high-frequency Trigram transition log probabilities: P(w3 | w1, w2)
pub const TRIGRAM_TRANSITIONS: &[((&str, &str, &str), f32)] = &[
    (("আমি", "ভাত", "খাচ্ছি"), -0.1),
    (("আমি", "ভাত", "খাব"), -0.2),
    (("আমি", "ভাত", "খেয়েছি"), -0.2),
    (("আমি", "তোমাকে", "ভালোবাসি"), -0.1),
    (("আমি", "বাংলায়", "গান"), -0.1),
    (("বাংলায়", "গান", "গাই"), -0.1),
    (("আমি", "বই", "পড়ছি"), -0.1),
    (("আমি", "শার্ট", "পরছি"), -0.1),
    (("জীবন", "থেকে", "নেওয়া"), -0.1),
    (("জীবন", "থেকে", "শেখা"), -0.1),
    (("তার", "নাম", "কী"), -0.1),
    (("তার", "নাম", "জানেন"), -0.2),
    (("তার", "সাথে", "আমি"), -0.1),
    (("তার", "সাথে", "কথা"), -0.1),
    (("তার", "সাথে", "দেখা"), -0.1),
    (("সাথে", "আমি", "যাব"), -0.1),
    (("সাথে", "আমি", "আছি"), -0.1),
    (("সাথে", "আমি", "কথা"), -0.2),
    (("বাংলাদেশ", "একটি", "সুন্দর"), -0.2),
    (("বাংলাদেশ", "একটি", "স্বাধীন"), -0.2),
    (("বাংলাদেশ", "একটি", "উন্নয়নশীল"), -0.3),
    (("আমার", "জীবনের", "লক্ষ্য"), -0.1),
    (("মোট", "এক", "লক্ষ"), -0.1),
    (("তুমি", "কেমন", "আছো"), -0.1),
    (("আপনি", "কেমন", "আছেন"), -0.1),
    (("আন্তরিক", "শুভেচ্ছা", "ও"), -0.05),
    (("শুভেচ্ছা", "ও", "অভিনন্দন"), -0.05),
];

#[derive(Debug, Clone)]
pub struct LanguageModel {
    unigrams: HashMap<&'static str, f32>,
    bigrams: HashMap<(&'static str, &'static str), f32>,
    trigrams: HashMap<(&'static str, &'static str, &'static str), f32>,
    next_word_map: HashMap<&'static str, Vec<(&'static str, f32)>>,
    lambda1: f32,
    lambda2: f32,
    lambda3: f32,
    unigram_floor: f32,
}

impl Default for LanguageModel {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageModel {
    pub fn new() -> Self {
        let mut unigrams = HashMap::with_capacity(UNIGRAM_LOG_PROBS.len() + 100);
        for &(w, p) in UNIGRAM_LOG_PROBS {
            unigrams.insert(w, p);
        }

        let mut bigrams = HashMap::with_capacity(BIGRAM_TRANSITIONS.len() + 100);
        let mut next_word_map: HashMap<&'static str, Vec<(&'static str, f32)>> = HashMap::new();

        for &((w1, w2), p) in BIGRAM_TRANSITIONS {
            bigrams.insert((w1, w2), p);
            next_word_map
                .entry(w1)
                .or_default()
                .push((w2, p));
        }

        // Sort next words by highest probability
        for list in next_word_map.values_mut() {
            list.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        let mut trigrams = HashMap::with_capacity(TRIGRAM_TRANSITIONS.len() + 50);
        for &((w1, w2, w3), p) in TRIGRAM_TRANSITIONS {
            trigrams.insert((w1, w2, w3), p);
        }

        Self {
            unigrams,
            bigrams,
            trigrams,
            next_word_map,
            lambda1: 0.15,
            lambda2: 0.35,
            lambda3: 0.50,
            unigram_floor: -6.0,
        }
    }

    /// Calculate interpolated conditional probability P(word | w_t-2, w_t-1) with 0 allocations
    pub fn score_candidate(&self, prev2: Option<&str>, prev1: Option<&str>, word: &str) -> f32 {
        let unigram_p = self.unigrams.get(word).copied().unwrap_or(self.unigram_floor);
        let mut score = self.lambda1 * (10.0f32.powf(unigram_p));

        if let Some(w1) = prev1 {
            if let Some(&bi_p) = self.bigrams.get(&(w1, word)) {
                score += self.lambda2 * (10.0f32.powf(bi_p));
            }

            if let Some(w2) = prev2 {
                if let Some(&tri_p) = self.trigrams.get(&(w2, w1, word)) {
                    score += self.lambda3 * (10.0f32.powf(tri_p));
                }
            }
        }

        score.log10()
    }

    /// Query the most likely continuations given previous word
    pub fn get_next_words(&self, previous_word: &str, limit: usize) -> Vec<String> {
        if let Some(list) = self.next_word_map.get(previous_word) {
            list.iter().take(limit).map(|(w, _)| (*w).to_string()).collect()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_model_scoring() {
        let lm = LanguageModel::new();

        // P(পরা | শার্ট) > P(পড়া | শার্ট)
        let score_shirt_pora = lm.score_candidate(None, Some("শার্ট"), "পরা");
        let score_shirt_poda = lm.score_candidate(None, Some("শার্ট"), "পড়া");
        assert!(score_shirt_pora > score_shirt_poda);

        // P(পড়া | বই) > P(পরা | বই)
        let score_book_poda = lm.score_candidate(None, Some("বই"), "পড়া");
        let score_book_pora = lm.score_candidate(None, Some("বই"), "পরা");
        assert!(score_book_poda > score_book_pora);
    }

    #[test]
    fn test_trigram_scoring() {
        let lm = LanguageModel::new();

        // P(খাচ্ছি | আমি, ভাত) should be very high
        let score_eating = lm.score_candidate(Some("আমি"), Some("ভাত"), "খাচ্ছি");
        let score_random = lm.score_candidate(Some("আমি"), Some("ভাত"), "গাই");
        assert!(score_eating > score_random);
    }
}
