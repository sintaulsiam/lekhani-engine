//! Autonomous Unsupervised Dictionary Learner Engine
//!
//! Automatically discovers new vocabulary, proper nouns, and technical jargon
//! from the user's typing stream and indexes them into the active trie.

use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::morphology::analyze_morphemes;
use crate::trie::PrefixTrie;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousLearner {
    pub observed_counts: HashMap<String, u32>,
    pub learned_words: HashSet<String>,
    #[serde(default = "default_threshold")]
    pub auto_learn_threshold: u32,
    #[serde(default)]
    pub user_bigrams: HashMap<String, u32>,
    #[serde(default)]
    pub last_committed_word: Option<String>,
    #[serde(default)]
    pub candidate_memory: HashMap<String, String>,
    #[serde(skip)]
    pub dirty: bool,
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
    pub const BINARY_MAGIC: &'static [u8; 4] = b"LLRN";
    pub const BINARY_VERSION: u32 = 1;

    pub fn new() -> Self {
        Self {
            observed_counts: HashMap::new(),
            learned_words: HashSet::new(),
            auto_learn_threshold: 1,
            user_bigrams: HashMap::new(),
            last_committed_word: None,
            candidate_memory: HashMap::new(),
            dirty: false,
        }
    }

    /// Record a committed word pair transition to personalize future ranking
    pub fn observe_committed_pair(&mut self, prev_word: &str, current_word: &str) {
        let clean_prev = prev_word.trim();
        let clean_curr = current_word.trim();
        if clean_prev.is_empty() || clean_curr.is_empty() || clean_prev == clean_curr {
            return;
        }
        let key = format!("{}\t{}", clean_prev, clean_curr);
        let count = self.user_bigrams.entry(key).or_insert(0);
        *count = (*count + 1).min(1000);
        self.last_committed_word = Some(clean_curr.to_string());
        self.dirty = true;
        if self.user_bigrams.len() > 8500 {
            self.prune_if_needed();
        }
    }

    /// Retrieve the personalized candidate score boost for a word following a previous word
    pub fn get_user_bigram_boost(&self, prev_word: &str, candidate: &str) -> i32 {
        let clean_prev = prev_word.trim();
        let clean_cand = candidate.trim();
        if clean_prev.is_empty() || clean_cand.is_empty() {
            return 0;
        }
        let key = format!("{}\t{}", clean_prev, clean_cand);
        if let Some(&count) = self.user_bigrams.get(&key) {
            (1500 + count.min(6) as i32 * 500).min(4500)
        } else {
            0
        }
    }

    /// Record user's candidate selection override permanently
    pub fn record_candidate_selection(&mut self, buffer: &str, candidate: &str) {
        let clean_buf = buffer.trim();
        let clean_cand = candidate.trim();
        if clean_buf.is_empty() || clean_cand.is_empty() {
            return;
        }
        self.candidate_memory
            .insert(clean_buf.to_string(), clean_cand.to_string());
        self.dirty = true;
        if self.candidate_memory.len() > 2200 {
            self.prune_if_needed();
        }
    }

    /// Retrieve top user-learned continuations following `prev_word` for next-word prediction
    pub fn get_top_user_continuations(&self, prev_word: &str, limit: usize) -> Vec<String> {
        let clean_prev = prev_word.trim();
        if clean_prev.is_empty() {
            return Vec::new();
        }
        let prefix = format!("{}\t", clean_prev);
        let mut matches: Vec<(&str, u32)> = Vec::new();
        for (k, &count) in &self.user_bigrams {
            if let Some(next_word) = k.strip_prefix(&prefix) {
                if !next_word.is_empty() {
                    matches.push((next_word, count));
                }
            }
        }
        matches.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
        matches
            .into_iter()
            .take(limit)
            .map(|(w, _)| w.to_string())
            .collect()
    }

    /// Prune low-frequency and excess entries to maintain bounded memory and fast serialization
    pub fn prune_if_needed(&mut self) {
        const MAX_BIGRAMS: usize = 8000;
        const MAX_OBSERVED: usize = 4000;
        const MAX_CANDIDATES: usize = 2000;

        if self.user_bigrams.len() > MAX_BIGRAMS {
            self.user_bigrams.retain(|_, &mut count| count > 1);
            if self.user_bigrams.len() > MAX_BIGRAMS {
                let mut entries: Vec<(String, u32)> = self.user_bigrams.drain().collect();
                entries.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
                entries.truncate(MAX_BIGRAMS - 1000);
                self.user_bigrams = entries.into_iter().collect();
            }
        }

        if self.observed_counts.len() > MAX_OBSERVED {
            self.observed_counts.retain(|_, &mut count| count > 1);
            if self.observed_counts.len() > MAX_OBSERVED {
                let mut entries: Vec<(String, u32)> = self.observed_counts.drain().collect();
                entries.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
                entries.truncate(MAX_OBSERVED - 500);
                self.observed_counts = entries.into_iter().collect();
            }
        }

        if self.candidate_memory.len() > MAX_CANDIDATES {
            let mut entries: Vec<(String, String)> = self.candidate_memory.drain().collect();
            entries.truncate(MAX_CANDIDATES - 200);
            self.candidate_memory = entries.into_iter().collect();
        }
    }

    /// Clear all user-learned data and reset baseline
    pub fn clear_user_data(&mut self) {
        self.observed_counts.clear();
        self.learned_words.clear();
        self.candidate_memory.clear();
        self.user_bigrams.clear();
        self.last_committed_word = None;
        self.dirty = true;
        self.pretrain_baseline();
    }

    /// Process a committed word, extract potential root stems, and auto-learn new vocabulary
    pub fn observe_and_learn(
        &mut self,
        committed_word: &str,
        trie: &PrefixTrie,
    ) -> Vec<String> {
        let morphemes = analyze_morphemes(committed_word);
        let mut newly_learned = Vec::new();

        for word in morphemes {
            // Only learn valid Bengali non-trivial terms
            if word.chars().count() < 2 || word.is_ascii() {
                continue;
            }

            // If already known in learned list, increment count
            if self.learned_words.contains(&word) {
                let count = self.observed_counts.entry(word).or_insert(0);
                *count = (*count + 1).min(10000);
                self.dirty = true;
                continue;
            }

            let count = self.observed_counts.entry(word.clone()).or_insert(0);
            *count += 1;
            self.dirty = true;

            if *count >= self.auto_learn_threshold && !trie.contains_exact(&word) {
                self.learned_words.insert(word.clone());
                self.dirty = true;
                newly_learned.push(word);
            }
        }

        if self.observed_counts.len() > 4200 {
            self.prune_if_needed();
        }

        newly_learned
    }

    /// Pre-trained high-frequency conversational word pairs and idioms for instant warm-start
    pub fn pretrain_baseline(&mut self) {
        for &(w1, w2, count) in PRETRAINED_CONVERSATIONAL_BIGRAMS {
            let key = format!("{}\t{}", w1, w2);
            self.user_bigrams.insert(key, count);
            self.learned_words.insert(w1.to_string());
            self.learned_words.insert(w2.to_string());
            *self.observed_counts.entry(w1.to_string()).or_insert(0) += count;
            *self.observed_counts.entry(w2.to_string()).or_insert(0) += count;
        }
        self.dirty = true;
    }

    /// Ingest a raw Bengali text corpus to train both vocabulary and word pair transitions
    pub fn train_text(&mut self, text: &str) {
        let mut prev_word: Option<String> = None;
        for token in text.split_whitespace() {
            let clean: String = token
                .chars()
                .filter(|c| {
                    !c.is_ascii_punctuation()
                        && *c != '।'
                        && *c != '—'
                        && *c != ','
                        && *c != '"'
                        && *c != '\''
                        && *c != '‘'
                        && *c != '’'
                        && *c != '“'
                        && *c != '”'
                })
                .collect();
            let clean = clean.trim();
            if clean.chars().count() >= 2 && !clean.is_ascii() {
                self.learned_words.insert(clean.to_string());
                *self.observed_counts.entry(clean.to_string()).or_insert(0) += 1;
                self.dirty = true;
                if let Some(ref prev) = prev_word {
                    self.observe_committed_pair(prev, clean);
                }
                prev_word = Some(clean.to_string());
            }
            if token.contains('।') || token.contains('?') || token.contains('!') {
                prev_word = None;
            }
        }
    }

    /// Merge another learner instance into this one
    pub fn merge(&mut self, other: &AutonomousLearner) {
        for word in &other.learned_words {
            self.learned_words.insert(word.clone());
        }
        for (word, &count) in &other.observed_counts {
            let entry = self.observed_counts.entry(word.clone()).or_insert(0);
            *entry = (*entry).max(count).min(10000);
        }
        for (key, &count) in &other.user_bigrams {
            let entry = self.user_bigrams.entry(key.clone()).or_insert(0);
            *entry = (*entry).max(count).min(1000);
        }
        for (k, v) in &other.candidate_memory {
            self.candidate_memory.insert(k.clone(), v.clone());
        }
        self.dirty = true;
        self.prune_if_needed();
    }

    /// Export learned dictionary as formatted JSON for inspection or backup
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Import learned dictionary from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Save learned dictionary to a compact binary file safely using atomic rename.
    /// Skips disk I/O immediately if dirty flag is false.
    pub fn save_to_path<P: AsRef<Path>>(&mut self, path: P) -> Result<(), std::io::Error> {
        if !self.dirty {
            return Ok(());
        }
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let payload = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let mut buf = Vec::with_capacity(8 + payload.len());
        buf.extend_from_slice(Self::BINARY_MAGIC);
        buf.extend_from_slice(&Self::BINARY_VERSION.to_le_bytes());
        buf.extend_from_slice(&payload);

        let tmp_path = path.with_extension("tmp");
        let res = if std::fs::write(&tmp_path, &buf).is_ok() && std::fs::rename(&tmp_path, path).is_ok() {
            Ok(())
        } else {
            std::fs::write(path, &buf)
        };

        if res.is_ok() {
            self.dirty = false;
        }
        res
    }

    /// Load learned dictionary from a compact binary file, automatically seeding baseline if file doesn't exist
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();
        if path.exists() {
            if let Ok(bytes) = std::fs::read(path) {
                if bytes.len() >= 8 && &bytes[0..4] == Self::BINARY_MAGIC {
                    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap_or([0; 4]));
                    if version == Self::BINARY_VERSION {
                        if let Ok(mut learner) = bincode::deserialize::<AutonomousLearner>(&bytes[8..]) {
                            learner.dirty = false;
                            return learner;
                        }
                    }
                }
            }
        }
        let mut learner = Self::new();
        learner.pretrain_baseline();
        learner.dirty = true;
        let _ = learner.save_to_path(path);
        learner
    }
}

/// Pre-trained high-frequency Bengali conversational word pairs
pub const PRETRAINED_CONVERSATIONAL_BIGRAMS: &[(&str, &str, u32)] = &[
    ("কেমন", "আছো", 20),
    ("কেমন", "আছেন", 20),
    ("ভালো", "আছি", 20),
    ("ভালো", "আছেন", 18),
    ("ভালো", "থেকো", 15),
    ("অনেক", "ধন্যবাদ", 20),
    ("অসংখ্য", "ধন্যবাদ", 16),
    ("শুভ", "সকাল", 15),
    ("শুভ", "রাত্রি", 15),
    ("শুভ", "কামনা", 14),
    ("শুভ", "জন্মদিন", 18),
    ("চা", "খাব", 16),
    ("চা", "খাচ্ছি", 14),
    ("ভাত", "খাব", 16),
    ("ভাত", "খাচ্ছি", 16),
    ("পানি", "খাব", 14),
    ("বাসায়", "যাব", 18),
    ("বাসায়", "যাচ্ছি", 18),
    ("বাসায়", "আছি", 16),
    ("অফিসে", "আছি", 16),
    ("অফিসে", "যাব", 16),
    ("দেরি", "হবে", 18),
    ("দেরি", "হচ্ছে", 16),
    ("দেখা", "হবে", 18),
    ("কথা", "বলব", 16),
    ("কথা", "বলছি", 16),
    ("ফোন", "দাও", 15),
    ("ফোন", "দিচ্ছি", 16),
    ("ফোন", "করেছি", 14),
    ("সমস্যা", "নাই", 20),
    ("সমস্যা", "নেই", 18),
    ("প্যারা", "নাই", 20),
    ("কোথায়", "যাচ্ছ", 16),
    ("কোথায়", "আছো", 18),
    ("কোথায়", "তুমি", 16),
    ("কী", "খবর", 18),
    ("কী", "করছ", 18),
    ("কী", "করছেন", 16),
    ("সব", "ঠিক", 18),
    ("ঠিক", "আছে", 20),
    ("একটু", "পরে", 18),
    ("একটু", "দেরি", 16),
    ("জরুরি", "কাজ", 16),
    ("জরুরি", "কথা", 16),
    ("আজকে", "বৃষ্টি", 15),
    ("আজকে", "যাব", 15),
    ("কালকে", "দেখা", 16),
    ("বই", "পড়া", 20),
    ("শার্ট", "পরা", 20),
    ("জুতা", "পরা", 18),
    ("গান", "শুনব", 16),
    ("গান", "শুনছি", 16),
    ("ছবি", "দেখব", 16),
    ("ছবি", "দেখছি", 16),
    ("খুব", "সুন্দর", 18),
    ("দারুণ", "হয়েছে", 18),
    ("অভিনন্দন", "জানাই", 16),
    ("খোদা", "হাফেজ", 18),
    ("আল্লাহ", "হাফেজ", 18),
    ("ইনশাআল্লাহ", "হবে", 18),
    ("আলহামদুলিল্লাহ", "ভালো", 20),
    ("মাশাল্লাহ", "সুন্দর", 16),
    ("ঘুম", "পাচ্ছে", 16),
    ("ঘুমাব", "এখন", 16),
    ("বাইরে", "যাচ্ছি", 16),
    ("দ্রুত", "আসো", 16),
    ("তাড়াতাড়ি", "করো", 16),
    ("কোথাও", "যাব", 14),
    ("কিছু", "বলব", 14),
    ("মনে", "পড়ছে", 16),
    ("মনে", "হচ্ছে", 18),
    ("মনে", "রাখব", 16),
    ("কাজ", "করছি", 16),
    ("কাজ", "শেষ", 18),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autonomous_learning() {
        let mut learner = AutonomousLearner::new();
        let trie = PrefixTrie::new();

        assert!(!trie.contains_exact("কুয়েট"));

        let learned = learner.observe_and_learn("কুয়েটে", &trie);
        assert!(learned.contains(&"কুয়েট".to_string()) || learned.contains(&"কুয়েটে".to_string()));
        assert!(learner.learned_words.contains("কুয়েট") || learner.learned_words.contains("কুয়েটে"));
    }

    #[test]
    fn test_binary_roundtrip_and_dirty_flag() {
        let mut learner = AutonomousLearner::new();
        learner.observe_committed_pair("বাংলা", "ভাষা");
        learner.record_candidate_selection("amr", "আমার");
        assert!(learner.dirty);

        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("lekhani_test_learner_bin_roundtrip.bin");

        learner.save_to_path(&temp_path).expect("Binary save must succeed");
        assert!(!learner.dirty);

        let loaded = AutonomousLearner::load_from_path(&temp_path);
        assert_eq!(loaded.get_user_bigram_boost("বাংলা", "ভাষা"), learner.get_user_bigram_boost("বাংলা", "ভাষা"));
        assert_eq!(loaded.candidate_memory.get("amr"), Some(&"আমার".to_string()));
        assert!(!loaded.dirty);

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_merge() {
        let mut l1 = AutonomousLearner::new();
        l1.observe_committed_pair("আমি", "ভাত");
        l1.record_candidate_selection("tui", "তুই");

        let mut l2 = AutonomousLearner::new();
        l2.observe_committed_pair("আমি", "চা");
        l2.record_candidate_selection("apni", "আপনি");

        l1.merge(&l2);
        assert!(l1.user_bigrams.contains_key("আমি\tভাত"));
        assert!(l1.user_bigrams.contains_key("আমি\tচা"));
        assert_eq!(l1.candidate_memory.get("tui"), Some(&"তুই".to_string()));
        assert_eq!(l1.candidate_memory.get("apni"), Some(&"আপনি".to_string()));
        assert!(l1.dirty);
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

    #[test]
    fn test_pretrain_baseline_and_serialization() {
        let mut learner = AutonomousLearner::new();
        learner.pretrain_baseline();

        assert!(learner.learned_words.len() > 50);
        assert!(learner.user_bigrams.len() > 50);
        assert!(learner.get_user_bigram_boost("কেমন", "আছো") >= 1500);
        assert!(learner.get_user_bigram_boost("প্যারা", "নাই") >= 1500);

        // Verify JSON serialization and deserialization
        let json = serde_json::to_string(&learner).expect("JSON serialization must succeed");
        let deserialized: AutonomousLearner =
            serde_json::from_str(&json).expect("JSON deserialization must succeed");

        assert_eq!(
            deserialized.learned_words.len(),
            learner.learned_words.len()
        );
        assert_eq!(deserialized.user_bigrams.len(), learner.user_bigrams.len());
    }

    #[test]
    fn test_candidate_selection_memory() {
        let mut learner = AutonomousLearner::new();
        learner.record_candidate_selection("kormo", "কর্ম");
        assert_eq!(learner.candidate_memory.get("kormo"), Some(&"কর্ম".to_string()));

        let json = serde_json::to_string(&learner).expect("JSON serialization must succeed");
        let deserialized: AutonomousLearner = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.candidate_memory.get("kormo"), Some(&"কর্ম".to_string()));
    }

    #[test]
    fn test_top_user_continuations() {
        let mut learner = AutonomousLearner::new();
        learner.observe_committed_pair("আমি", "ভাত");
        learner.observe_committed_pair("আমি", "ভাত");
        learner.observe_committed_pair("আমি", "চা");

        let continuations = learner.get_top_user_continuations("আমি", 2);
        assert_eq!(continuations, vec!["ভাত".to_string(), "চা".to_string()]);
    }

    #[test]
    fn test_clear_user_data() {
        let mut learner = AutonomousLearner::new();
        learner.observe_committed_pair("কাস্টম", "শব্দ");
        learner.record_candidate_selection("test", "টেস্ট");
        learner.clear_user_data();

        assert_eq!(learner.candidate_memory.len(), 0);
        assert!(!learner.user_bigrams.is_empty()); // baseline re-seeded
        assert!(learner.get_user_bigram_boost("কেমন", "আছো") >= 1500);
    }
}
