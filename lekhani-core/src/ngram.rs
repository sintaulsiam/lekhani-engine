//! N-Gram Context-Aware Next-Word Prediction & User Statistics Engine

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserStats {
    pub total_keystrokes: u64,
    pub total_words_typed: u64,
    pub keystrokes_saved: u64,
    pub top_words: HashMap<String, u64>,
}

impl Default for UserStats {
    fn default() -> Self {
        Self::new()
    }
}

impl UserStats {
    pub fn new() -> Self {
        Self {
            total_keystrokes: 0,
            total_words_typed: 0,
            keystrokes_saved: 0,
            top_words: HashMap::new(),
        }
    }

    /// Record a completed word commit
    pub fn record_commit(&mut self, typed_len: usize, committed_text: &str) {
        if committed_text.is_empty() {
            return;
        }
        self.total_keystrokes += typed_len as u64;
        self.total_words_typed += 1;

        let committed_char_count = committed_text.chars().count();
        if committed_char_count > typed_len {
            self.keystrokes_saved += (committed_char_count - typed_len) as u64;
        }

        *self.top_words.entry(committed_text.to_string()).or_insert(0) += 1;
    }

    /// Calculate percentage of keystrokes saved via suggestions and smart snippets
    pub fn savings_percentage(&self) -> f64 {
        let total_virtual = self.total_keystrokes + self.keystrokes_saved;
        if total_virtual == 0 {
            0.0
        } else {
            (self.keystrokes_saved as f64 / total_virtual as f64) * 100.0
        }
    }

    /// Get sorted list of most frequent words
    pub fn get_top_words(&self, limit: usize) -> Vec<(String, u64)> {
        let mut list: Vec<(String, u64)> = self
            .top_words
            .iter()
            .map(|(w, c)| (w.clone(), *c))
            .collect();
        list.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        list.truncate(limit);
        list
    }

    /// Save statistics to a JSON file
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let data = serde_json::to_string_pretty(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
        std::fs::write(path, data)
    }

    /// Load statistics from a JSON file
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(stats) = serde_json::from_str::<UserStats>(&content) {
                    return stats;
                }
            }
        }
        Self::new()
    }
}

#[derive(Debug, Clone, Default)]
pub struct NGramPredictor {
    bigram_map: HashMap<String, Vec<String>>,
    stats: UserStats,
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

        Self {
            bigram_map,
            stats: UserStats::new(),
        }
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

    pub fn stats_mut(&mut self) -> &mut UserStats {
        &mut self.stats
    }

    pub fn stats(&self) -> &UserStats {
        &self.stats
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

    #[test]
    fn test_user_stats() {
        let mut stats = UserStats::new();
        stats.record_commit(3, "আমি");
        stats.record_commit(10, "আন্তর্জাতিক শুভেচ্ছা ও অভিনন্দন");
        assert_eq!(stats.total_words_typed, 2);
        assert!(stats.keystrokes_saved > 0);
        let top = stats.get_top_words(5);
        assert_eq!(top.len(), 2);
    }
}
