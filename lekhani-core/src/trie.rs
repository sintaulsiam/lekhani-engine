//! High-Performance Flat Prefix Store & Dictionary Engine

#[derive(Debug, Clone, Default)]
pub struct PrefixTrie {
    words: Vec<String>,
    is_sorted: bool,
}

impl PrefixTrie {
    pub fn new() -> Self {
        Self {
            words: Vec::new(),
            is_sorted: true,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            words: Vec::with_capacity(capacity),
            is_sorted: true,
        }
    }

    /// Insert a key/candidate representation
    pub fn insert(&mut self, _key: &str, candidate: String) {
        self.words.push(candidate);
        self.is_sorted = false;
    }

    /// Bulk insert words and sort in a single fast pass
    pub fn insert_bulk(&mut self, mut new_words: Vec<String>) {
        self.words.append(&mut new_words);
        self.is_sorted = false;
        self.ensure_sorted();
    }

    /// Ensure internal word list is sorted and deduplicated
    pub fn ensure_sorted(&mut self) {
        if !self.is_sorted {
            self.words.sort_unstable();
            self.words.dedup();
            self.is_sorted = true;
        }
    }

    /// Exact lookup for a key
    pub fn get_exact(&self, key: &str) -> Option<&[String]> {
        if let Ok(idx) = self.words.binary_search_by(|w| w.as_str().cmp(key)) {
            Some(&self.words[idx..=idx])
        } else {
            None
        }
    }

    /// Check if dictionary contains exact word
    pub fn contains_exact(&self, key: &str) -> bool {
        self.words.binary_search_by(|w| w.as_str().cmp(key)).is_ok()
    }

    /// Find all candidates whose key starts with the given prefix, prioritized by length
    pub fn find_prefix_matches(&self, prefix: &str, limit: usize) -> Vec<String> {
        if prefix.is_empty() || self.words.is_empty() {
            return Vec::new();
        }

        // Find starting index using binary search
        let start_idx = match self.words.binary_search_by(|w| w.as_str().cmp(prefix)) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        let mut matched: Vec<&String> = Vec::with_capacity(limit * 2);
        for w in &self.words[start_idx..] {
            if w.starts_with(prefix) {
                matched.push(w);
                if matched.len() >= limit * 4 {
                    break;
                }
            } else {
                break;
            }
        }

        // Sort matches by length (shorter root words first) and then alphabetical
        matched.sort_unstable_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        matched.into_iter().take(limit).cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_trie() {
        let mut trie = PrefixTrie::new();
        trie.insert("bangla", "বাংলা".to_string());
        trie.insert("bangladesh", "বাংলাদেশ".to_string());
        trie.insert("bangladeshi", "বাংলাদেশী".to_string());
        trie.insert("boi", "বই".to_string());
        trie.ensure_sorted();

        assert!(trie.get_exact("বাংলা").is_some());
        assert_eq!(trie.get_exact("unknown"), None);

        let matches = trie.find_prefix_matches("বাং", 10);
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0], "বাংলা"); // Shorter root word first!
        assert!(matches.contains(&"বাংলাদেশ".to_string()));
        assert!(matches.contains(&"বাংলাদেশী".to_string()));
    }
}
