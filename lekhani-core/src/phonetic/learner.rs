//! Autonomous Unsupervised Dictionary Learner Engine
//!
//! Automatically discovers new vocabulary, proper nouns, and technical jargon
//! from the user's typing stream and indexes them into the active trie.

use hashbrown::{Equivalent, HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::morphology::analyze_morphemes;
use crate::trie::PrefixTrie;

#[derive(Hash, PartialEq, Eq)]
pub struct BigramKey<'a>(pub &'a str, pub &'a str);

impl<'a> Equivalent<(String, String)> for BigramKey<'a> {
    fn equivalent(&self, key: &(String, String)) -> bool {
        self.0 == key.0.as_str() && self.1 == key.1.as_str()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousLearner {
    pub observed_counts: HashMap<String, u32>,
    pub learned_words: HashSet<String>,
    #[serde(default = "default_threshold")]
    pub auto_learn_threshold: u32,
    #[serde(default)]
    pub user_bigrams: HashMap<(String, String), u32>,
    #[serde(default)]
    pub last_committed_word: Option<String>,
}

fn default_threshold() -> u32 {
    1
}

impl Default for AutonomousLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl AutonomousLearner {
    pub fn new() -> Self {
        Self {
            observed_counts: HashMap::new(),
            learned_words: HashSet::new(),
            auto_learn_threshold: 1,
            user_bigrams: HashMap::new(),
            last_committed_word: None,
        }
    }

    /// Record a committed word pair transition to personalize future ranking
    pub fn observe_committed_pair(&mut self, prev_word: &str, current_word: &str) {
        let clean_prev = prev_word.trim();
        let clean_curr = current_word.trim();
        if clean_prev.is_empty() || clean_curr.is_empty() || clean_prev == clean_curr {
            return;
        }
        let count = self
            .user_bigrams
            .entry((clean_prev.to_string(), clean_curr.to_string()))
            .or_insert(0);
        *count = (*count + 1).min(1000);
        self.last_committed_word = Some(clean_curr.to_string());
    }

    /// Retrieve the personalized candidate score boost for a word following a previous word
    pub fn get_user_bigram_boost(&self, prev_word: &str, candidate: &str) -> i32 {
        let clean_prev = prev_word.trim();
        let clean_cand = candidate.trim();
        if clean_prev.is_empty() || clean_cand.is_empty() {
            return 0;
        }
        if let Some(&count) = self.user_bigrams.get(&BigramKey(clean_prev, clean_cand)) {
            (1500 + count.min(6) as i32 * 500).min(4500)
        } else {
            0
        }
    }

    /// Process a committed word, extract potential root stems, and auto-learn new vocabulary
    pub fn observe_and_learn(
        &mut self,
        committed_word: &str,
        trie: &mut PrefixTrie,
    ) -> Vec<String> {
        let morphemes = analyze_morphemes(committed_word);
        let mut newly_learned = Vec::new();

        for word in morphemes {
            // Only learn valid Bengali non-trivial terms
            if word.chars().count() < 2 || word.is_ascii() {
                continue;
            }

            // If already known in trie or learned list, boost weight
            if self.learned_words.contains(&word) {
                trie.insert_weighted(word.clone(), 9500);
                continue;
            }

            let count = self.observed_counts.entry(word.clone()).or_insert(0);
            *count += 1;

            if *count >= self.auto_learn_threshold && !trie.contains_exact(&word) {
                self.learned_words.insert(word.clone());
                trie.insert_weighted(word.clone(), 9200);
                newly_learned.push(word);
            }
        }

        newly_learned
    }

    /// Save learned dictionary to a JSON file
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let data = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        std::fs::write(path, data)
    }

    /// Load learned dictionary from a JSON file
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(learner) = serde_json::from_str::<AutonomousLearner>(&content) {
                    return learner;
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
    fn test_autonomous_learning() {
        let mut learner = AutonomousLearner::new();
        let mut trie = PrefixTrie::new();

        assert!(!trie.contains_exact("কুয়েট"));

        let learned = learner.observe_and_learn("কুয়েটে", &mut trie);
        assert!(learned.contains(&"কুয়েট".to_string()) || learned.contains(&"কুয়েটে".to_string()));
        assert!(trie.contains_exact("কুয়েট") || trie.contains_exact("কুয়েটে"));
    }

    #[test]
    fn test_user_bigram_adaptation() {
        let mut learner = AutonomousLearner::new();
        assert_eq!(learner.get_user_bigram_boost("আমি", "খাব"), 0);

        learner.observe_committed_pair("আমি", "খাব");
        let boost1 = learner.get_user_bigram_boost("আমি", "খাব");
        assert!(boost1 >= 1500);

        learner.observe_committed_pair("আমি", "খাব");
        let boost2 = learner.get_user_bigram_boost("আমি", "খাব");
        assert!(boost2 > boost1);
    }
}
