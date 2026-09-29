//! User Typing Telemetry & Statistics Engine

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UserStats {
    pub total_keystrokes: u64,
    pub total_words_typed: u64,
    pub keystrokes_saved: u64,
    pub top_words: HashMap<String, u64>,
    pub char_frequencies: HashMap<char, u64>,
    pub word_last_seen: HashMap<String, u64>,
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
            char_frequencies: HashMap::new(),
            word_last_seen: HashMap::new(),
        }
    }

    /// Record a completed word commit using current system time
    pub fn record_commit(&mut self, typed_len: usize, committed_text: &str) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.record_commit_with_timestamp(typed_len, committed_text, now);
    }

    /// Record a completed word commit with explicit Unix timestamp in seconds
    pub fn record_commit_with_timestamp(
        &mut self,
        typed_len: usize,
        committed_text: &str,
        current_unix_ts: u64,
    ) {
        if committed_text.is_empty() {
            return;
        }
        self.total_keystrokes += typed_len as u64;
        self.total_words_typed += 1;

        let committed_char_count = committed_text.chars().count();
        if committed_char_count > typed_len {
            self.keystrokes_saved += (committed_char_count - typed_len) as u64;
        }

        *self
            .top_words
            .entry(committed_text.to_string())
            .or_insert(0) += 1;

        self.word_last_seen
            .insert(committed_text.to_string(), current_unix_ts);

        for ch in committed_text.chars() {
            if !ch.is_whitespace() && !ch.is_ascii_punctuation() {
                *self.char_frequencies.entry(ch).or_insert(0) += 1;
            }
        }
    }

    /// Compute Ebbinghaus recency-decay-weighted frequency score for a word.
    /// EffectiveFreq(w, t) = BaseFreq(w) * e^(-lambda * delta_days)
    /// with 3x recency boost if typed within the last 24 hours.
    pub fn decay_weighted_score(&self, word: &str, current_unix_ts: u64) -> f32 {
        let base_freq = match self.top_words.get(word) {
            Some(&freq) if freq > 0 => freq as f32,
            _ => return 0.0,
        };

        let last_ts = self.word_last_seen.get(word).copied().unwrap_or(current_unix_ts);
        let delta_secs = current_unix_ts.saturating_sub(last_ts);
        let delta_days = (delta_secs as f32) / 86400.0;

        const LAMBDA: f32 = 0.05;
        let decay = (-LAMBDA * delta_days).exp();

        let recency_multiplier = if delta_secs <= 86400 {
            3.0
        } else {
            1.0
        };

        base_freq * decay * recency_multiplier
    }

    /// Populate a PersonalScoreOverlay from the current UserStats
    pub fn populate_overlay(&self, overlay: &mut lekhani_ai::PersonalScoreOverlay, current_unix_ts: u64) {
        for (word, _) in &self.top_words {
            let score = self.decay_weighted_score(word, current_unix_ts);
            // Scale score to [0.0, 5.0] boost range
            let boost = (score.ln_1p() * 0.5).min(5.0);
            overlay.set_score(word, boost);
        }
    }

    /// Get sorted list of most frequent characters with count and percentage
    pub fn get_top_characters(&self, limit: usize) -> Vec<(char, u64, f64)> {
        let total_chars: u64 = self.char_frequencies.values().sum();
        let mut list: Vec<(char, u64)> = self
            .char_frequencies
            .iter()
            .map(|(c, v)| (*c, *v))
            .collect();
        list.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        list.truncate(limit);
        list.into_iter()
            .map(|(c, count)| {
                let pct = if total_chars == 0 {
                    0.0
                } else {
                    (count as f64 / total_chars as f64) * 100.0
                };
                (c, count, pct)
            })
            .collect()
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

    /// Merge another UserStats snapshot additively
    pub fn merge(&mut self, other: &UserStats) {
        self.total_keystrokes += other.total_keystrokes;
        self.total_words_typed += other.total_words_typed;
        self.keystrokes_saved += other.keystrokes_saved;
        for (w, c) in &other.top_words {
            let entry = self.top_words.entry(w.clone()).or_insert(0);
            *entry += *c;
        }
        for (ch, c) in &other.char_frequencies {
            let entry = self.char_frequencies.entry(*ch).or_insert(0);
            *entry += *c;
        }
        for (w, ts) in &other.word_last_seen {
            let entry = self.word_last_seen.entry(w.clone()).or_insert(0);
            *entry = (*entry).max(*ts);
        }
    }

    /// Save statistics to a JSON file safely using atomic rename
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let path = path.as_ref();
        let data = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        crate::fs::atomic_write_secure(path, data.as_bytes())
    }

    /// Load statistics from a JSON file
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(mut stats) = serde_json::from_str::<UserStats>(&content) {
                    if stats.char_frequencies.is_empty() && !stats.top_words.is_empty() {
                        for (word, count) in &stats.top_words {
                            for ch in word.chars() {
                                if !ch.is_whitespace() && !ch.is_ascii_punctuation() {
                                    *stats.char_frequencies.entry(ch).or_insert(0) += *count;
                                }
                            }
                        }
                    }
                    return stats;
                }
            }
        }
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_stats() {
        let mut stats = UserStats::new();
        stats.record_commit(3, "আমি");
        stats.record_commit(10, "আন্তরিক শুভেচ্ছা ও অভিনন্দন");
        assert_eq!(stats.total_words_typed, 2);
        assert!(stats.keystrokes_saved > 0);
        let top = stats.get_top_words(5);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_user_stats_merge() {
        let mut s1 = UserStats::new();
        s1.record_commit(5, "বাংলা");
        assert_eq!(s1.total_words_typed, 1);

        let mut s2 = UserStats::new();
        s2.record_commit(10, "বাংলাদেশ");
        s2.record_commit(4, "ভাষা");
        assert_eq!(s2.total_words_typed, 2);

        s1.merge(&s2);
        assert_eq!(s1.total_words_typed, 3);
        assert!(s1.top_words.contains_key("বাংলাদেশ"));
        assert!(s1.top_words.contains_key("বাংলা"));
    }

    #[test]
    fn test_decay_weighted_score() {
        let mut stats = UserStats::new();
        let base_ts = 1_000_000;
        stats.record_commit_with_timestamp(3, "লেখনী", base_ts);

        // Within 24h (3600 seconds later): should have 3x recency boost
        let score_recent = stats.decay_weighted_score("লেখনী", base_ts + 3600);
        assert!(score_recent >= 2.9);

        // 30 days later (2,592,000 seconds later): 1x multiplier and decay
        let score_old = stats.decay_weighted_score("লেখনী", base_ts + 2_592_000);
        assert!(score_old < score_recent);
        assert!(score_old > 0.0);
    }

    #[test]
    fn test_populate_overlay() {
        let mut stats = UserStats::new();
        let now = 1_000_000;
        stats.record_commit_with_timestamp(3, "বিশেষ", now);

        let mut overlay = lekhani_ai::PersonalScoreOverlay::new();
        stats.populate_overlay(&mut overlay, now);
        assert!(overlay.boost_for("বিশেষ") > 0.0);
    }
}
