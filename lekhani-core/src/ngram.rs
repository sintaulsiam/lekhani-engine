//! User Typing Telemetry & Statistics Engine

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
}
