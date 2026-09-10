//! N-Gram Context-Aware Next-Word Prediction Engine

use hashbrown::HashMap;

#[derive(Debug, Clone, Default)]
pub struct NGramPredictor {
    bigram_map: HashMap<String, Vec<String>>,
}

impl NGramPredictor {
    pub fn new() -> Self {
        let mut bigram_map = HashMap::new();

        // Common Bengali Bigram Frequencies
        bigram_map.insert("আমি".to_string(), vec!["তোমাকে".to_string(), "মনে".to_string(), "বাংলাদেশকে".to_string(), "ভালো".to_string(), "ভাত".to_string(), "এখন".to_string()]);
        bigram_map.insert("তুমি".to_string(), vec!["কেমন".to_string(), "কোথায়".to_string(), "কী".to_string(), "কখন".to_string(), "আমাকে".to_string()]);
        bigram_map.insert("আমরা".to_string(), vec!["সবাই".to_string(), "বাংলাদেশী".to_string(), "একসাথে".to_string(), "যাব".to_string()]);
        bigram_map.insert("বাংলাদেশ".to_string(), vec!["আমাদের".to_string(), "একটি".to_string(), "আমার".to_string(), "সরকার".to_string()]);
        bigram_map.insert("আমাদের".to_string(), vec!["দেশ".to_string(), "মাতৃভাষা".to_string(), "জাতীয়".to_string(), "সংস্কৃতি".to_string()]);
        bigram_map.insert("কেমন".to_string(), vec!["আছো".to_string(), "আছেন".to_string(), "হলো".to_string()]);
        bigram_map.insert("অনেক".to_string(), vec!["ধন্যবাদ".to_string(), "ভালো".to_string(), "সুন্দর".to_string(), "দিন".to_string()]);
        bigram_map.insert("শুভ".to_string(), vec!["সকাল".to_string(), "রাত্রি".to_string(), "জন্মদিন".to_string(), "নববর্ষ".to_string(), "কামনা".to_string()]);
        bigram_map.insert("জাতীয়".to_string(), vec!["পতাকা".to_string(), "সঙ্গীত".to_string(), "সংসদ".to_string(), "স্মৃতিসৌধ".to_string()]);
        bigram_map.insert("বাংলা".to_string(), vec!["ভাষা".to_string(), "একাডেমি".to_string(), "সাহিত্য".to_string(), "কীবোর্ড".to_string()]);

        Self { bigram_map }
    }

    /// Predict next words given the previous committed word
    pub fn predict_next_words(&self, previous_word: &str) -> Vec<String> {
        self.bigram_map.get(previous_word).cloned().unwrap_or_default()
    }

    /// Record new bigram transition from user typing habits
    pub fn learn_bigram(&mut self, prev_word: String, next_word: String) {
        let entry = self.bigram_map.entry(prev_word).or_default();
        if !entry.contains(&next_word) {
            entry.insert(0, next_word);
            entry.truncate(8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_prediction() {
        let predictor = NGramPredictor::new();
        let next_words = predictor.predict_next_words("শুভ");
        assert!(next_words.contains(&"সকাল".to_string()));
        assert!(next_words.contains(&"জন্মদিন".to_string()));
    }
}
