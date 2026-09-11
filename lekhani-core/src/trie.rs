//! High-Performance Frequency-Weighted Prefix Store & Dictionary Engine

#[derive(Debug, Clone, Default)]
pub struct PrefixTrie {
    entries: Vec<(String, u32)>,
    is_sorted: bool,
}

impl PrefixTrie {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            is_sorted: true,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            is_sorted: true,
        }
    }

    /// Insert a word with default frequency weight (100)
    pub fn insert(&mut self, _key: &str, candidate: String) {
        self.insert_weighted(candidate, 100);
    }

    /// Insert a word with an explicit frequency weight maintaining sorted order
    pub fn insert_weighted(&mut self, candidate: String, frequency: u32) {
        if !self.is_sorted {
            self.ensure_sorted();
        }
        match self.entries.binary_search_by(|e| e.0.as_str().cmp(candidate.as_str())) {
            Ok(idx) => {
                if frequency > self.entries[idx].1 {
                    self.entries[idx].1 = frequency;
                }
            }
            Err(idx) => {
                self.entries.insert(idx, (candidate, frequency));
            }
        }
    }

    /// Bulk insert words with default frequencies
    pub fn insert_bulk(&mut self, new_words: Vec<String>) {
        let weighted: Vec<(String, u32)> = new_words.into_iter().map(|w| (w, 100)).collect();
        self.insert_bulk_weighted(weighted);
    }

    /// Bulk insert words with specific frequency weights
    pub fn insert_bulk_weighted(&mut self, mut new_entries: Vec<(String, u32)>) {
        self.entries.append(&mut new_entries);
        self.is_sorted = false;
        self.ensure_sorted();
    }

    /// Ensure internal word list is sorted and deduplicated (keeping highest frequency)
    pub fn ensure_sorted(&mut self) {
        if !self.is_sorted {
            self.entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
            self.entries.dedup_by(|a, b| a.0 == b.0);
            self.is_sorted = true;
        }
    }

    /// Exact lookup for a key
    pub fn get_exact(&self, key: &str) -> Option<&(String, u32)> {
        if let Ok(idx) = self.entries.binary_search_by(|e| e.0.as_str().cmp(key)) {
            Some(&self.entries[idx])
        } else {
            None
        }
    }

    /// Get frequency weight of a word (defaults to 10 if unindexed)
    pub fn get_frequency(&self, key: &str) -> u32 {
        if let Ok(idx) = self.entries.binary_search_by(|e| e.0.as_str().cmp(key)) {
            self.entries[idx].1
        } else {
            10
        }
    }

    /// Check if dictionary contains exact word
    pub fn contains_exact(&self, key: &str) -> bool {
        self.entries.binary_search_by(|e| e.0.as_str().cmp(key)).is_ok()
    }

    /// Find all candidates whose key starts with the given prefix, prioritized by frequency and length
    pub fn find_prefix_matches(&self, prefix: &str, limit: usize) -> Vec<String> {
        if prefix.is_empty() || self.entries.is_empty() {
            return Vec::new();
        }

        // Find starting index using binary search
        let start_idx = match self.entries.binary_search_by(|e| e.0.as_str().cmp(prefix)) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        let mut matched: Vec<&(String, u32)> = Vec::with_capacity(limit * 4);
        for entry in &self.entries[start_idx..] {
            if entry.0.starts_with(prefix) {
                matched.push(entry);
                if matched.len() >= limit * 8 {
                    break;
                }
            } else {
                break;
            }
        }

        // Sort matches: High frequency first -> Shorter length -> Alphabetical
        matched.sort_unstable_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| a.0.len().cmp(&b.0.len()))
                .then_with(|| a.0.cmp(&b.0))
        });

        matched.into_iter().take(limit).map(|e| e.0.clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_trie() {
        let mut trie = PrefixTrie::new();
        trie.insert_weighted("বাংলা".to_string(), 9000);
        trie.insert_weighted("বাংলাদেশ".to_string(), 9500);
        trie.insert_weighted("বাংলাদেশী".to_string(), 2000);
        trie.insert_weighted("বাংলানিউজ".to_string(), 500);
        trie.insert_weighted("বই".to_string(), 8000);
        trie.ensure_sorted();

        assert!(trie.get_exact("বাংলা").is_some());
        assert_eq!(trie.get_exact("unknown"), None);
        assert_eq!(trie.get_frequency("বাংলাদেশ"), 9500);

        let matches = trie.find_prefix_matches("বাং", 10);
        assert_eq!(matches.len(), 4);
        // "বাংলাদেশ" has highest frequency (9500), then "বাংলা" (9000)
        assert_eq!(matches[0], "বাংলাদেশ");
        assert_eq!(matches[1], "বাংলা");
    }
}
