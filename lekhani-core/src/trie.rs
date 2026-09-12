//! High-Performance Frequency-Weighted Compact Prefix Store & Dictionary Engine
//!
//! Uses a contiguous buffer and compact 12-byte entries for ultra-low memory
//! footprint and cache-locality friendly binary search lookups.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrieEntry {
    pub offset: u32,
    pub len: u16,
    pub freq: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrefixTrie {
    buffer: String,
    entries: Vec<TrieEntry>,
    is_sorted: bool,
}

impl PrefixTrie {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            entries: Vec::new(),
            is_sorted: true,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: String::with_capacity(capacity * 12),
            entries: Vec::with_capacity(capacity),
            is_sorted: true,
        }
    }

    pub fn with_capacities(entries_capacity: usize, buffer_capacity: usize) -> Self {
        Self {
            buffer: String::with_capacity(buffer_capacity),
            entries: Vec::with_capacity(entries_capacity),
            is_sorted: true,
        }
    }

    #[inline]
    pub fn word_at(&self, entry: &TrieEntry) -> &str {
        let start = entry.offset as usize;
        let end = start + entry.len as usize;
        &self.buffer[start..end]
    }

    /// Total memory footprint of raw dictionary data in bytes
    pub fn memory_usage(&self) -> usize {
        self.buffer.capacity() + (self.entries.capacity() * std::mem::size_of::<TrieEntry>())
    }

    /// Insert a word with default frequency weight (100)
    pub fn insert(&mut self, _key: &str, candidate: String) {
        self.insert_weighted(candidate, 100);
    }

    /// Insert a word with an explicit frequency weight maintaining sorted order
    pub fn insert_weighted(&mut self, candidate: String, frequency: u32) {
        if candidate.is_empty() {
            return;
        }
        if !self.is_sorted {
            self.ensure_sorted();
        }
        match self
            .entries
            .binary_search_by(|e| self.word_at(e).cmp(candidate.as_str()))
        {
            Ok(idx) => {
                if frequency > self.entries[idx].freq {
                    self.entries[idx].freq = frequency;
                }
            }
            Err(idx) => {
                let offset = self.buffer.len() as u32;
                let len = candidate.len() as u16;
                self.buffer.push_str(&candidate);
                self.entries.insert(
                    idx,
                    TrieEntry {
                        offset,
                        len,
                        freq: frequency,
                    },
                );
            }
        }
    }

    /// Bulk insert words with default frequencies
    pub fn insert_bulk(&mut self, new_words: Vec<String>) {
        let weighted: Vec<(String, u32)> = new_words.into_iter().map(|w| (w, 100)).collect();
        self.insert_bulk_weighted(weighted);
    }

    /// Bulk insert words with specific frequency weights
    pub fn insert_bulk_weighted(&mut self, new_entries: Vec<(String, u32)>) {
        self.entries.reserve(new_entries.len());
        let total_add_bytes: usize = new_entries.iter().map(|(s, _)| s.len()).sum();
        self.buffer.reserve(total_add_bytes);

        for (cand, freq) in new_entries {
            if cand.is_empty() {
                continue;
            }
            let offset = self.buffer.len() as u32;
            let len = cand.len() as u16;
            self.buffer.push_str(&cand);
            self.entries.push(TrieEntry {
                offset,
                len,
                freq,
            });
        }
        self.is_sorted = false;
        self.ensure_sorted();
    }

    /// Compact the string buffer to remove unused/deduplicated string segments
    pub fn compact(&mut self) {
        if self.entries.is_empty() {
            self.buffer.clear();
            return;
        }
        let total_len: usize = self.entries.iter().map(|e| e.len as usize).sum();
        let mut new_buf = String::with_capacity(total_len);
        let old_buf = &self.buffer;
        for entry in &mut self.entries {
            let start = entry.offset as usize;
            let end = start + entry.len as usize;
            let word = &old_buf[start..end];
            let new_offset = new_buf.len() as u32;
            new_buf.push_str(word);
            entry.offset = new_offset;
        }
        self.buffer = new_buf;
    }

    /// Ensure internal word list is sorted and deduplicated (keeping highest frequency)
    pub fn ensure_sorted(&mut self) {
        if !self.is_sorted {
            let buf = &self.buffer;
            self.entries.sort_unstable_by(|a, b| {
                let str_a = &buf[a.offset as usize..(a.offset as usize + a.len as usize)];
                let str_b = &buf[b.offset as usize..(b.offset as usize + b.len as usize)];
                str_a.cmp(str_b).then_with(|| b.freq.cmp(&a.freq))
            });
            self.entries.dedup_by(|a, b| {
                let str_a = &buf[a.offset as usize..(a.offset as usize + a.len as usize)];
                let str_b = &buf[b.offset as usize..(b.offset as usize + b.len as usize)];
                str_a == str_b
            });
            self.compact();
            self.is_sorted = true;
        }
    }

    /// Exact lookup for a key returning borrowed reference and frequency
    pub fn get_exact(&self, key: &str) -> Option<(&str, u32)> {
        if let Ok(idx) = self.entries.binary_search_by(|e| self.word_at(e).cmp(key)) {
            let entry = &self.entries[idx];
            Some((self.word_at(entry), entry.freq))
        } else {
            None
        }
    }

    /// Get frequency weight of a word (defaults to 10 if unindexed)
    pub fn get_frequency(&self, key: &str) -> u32 {
        if let Ok(idx) = self.entries.binary_search_by(|e| self.word_at(e).cmp(key)) {
            self.entries[idx].freq
        } else {
            10
        }
    }

    /// Check if dictionary contains exact word
    pub fn contains_exact(&self, key: &str) -> bool {
        self.entries
            .binary_search_by(|e| self.word_at(e).cmp(key))
            .is_ok()
    }

    /// Zero-allocation lookup returning borrowed string slices and frequency weights
    pub fn find_prefix_entries<'a>(&'a self, prefix: &str, limit: usize) -> Vec<(&'a str, u32)> {
        if prefix.is_empty() || self.entries.is_empty() {
            return Vec::new();
        }

        // Find starting index using binary search
        let start_idx = match self.entries.binary_search_by(|e| self.word_at(e).cmp(prefix)) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        let mut matched: Vec<(&'a str, u32)> = Vec::with_capacity(limit * 4);
        for entry in &self.entries[start_idx..] {
            let word = self.word_at(entry);
            if word.starts_with(prefix) {
                matched.push((word, entry.freq));
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
                .then_with(|| a.0.cmp(b.0))
        });

        matched.into_iter().take(limit).collect()
    }

    /// Zero-allocation lookup returning borrowed string slices
    pub fn find_prefix_matches_ref<'a>(&'a self, prefix: &str, limit: usize) -> Vec<&'a str> {
        self.find_prefix_entries(prefix, limit)
            .into_iter()
            .map(|(w, _)| w)
            .collect()
    }

    /// Find all candidates whose key starts with the given prefix, prioritized by frequency and length
    pub fn find_prefix_matches(&self, prefix: &str, limit: usize) -> Vec<String> {
        self.find_prefix_entries(prefix, limit)
            .into_iter()
            .map(|(w, _)| w.to_string())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all words and frequencies
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.entries.iter().map(move |e| (self.word_at(e), e.freq))
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

        // Zero-allocation slice references
        let matches_ref = trie.find_prefix_matches_ref("বাং", 2);
        assert_eq!(matches_ref, vec!["বাংলাদেশ", "বাংলা"]);

        // Memory usage and iteration
        assert!(trie.memory_usage() > 0);
        let items: Vec<(&str, u32)> = trie.iter().collect();
        assert_eq!(items.len(), 5);

        // Serialization roundtrip
        let json = serde_json::to_string(&trie).unwrap();
        let deserialized: PrefixTrie = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.get_frequency("বাংলাদেশ"), 9500);
        assert_eq!(deserialized.find_prefix_matches("বাং", 2), vec!["বাংলাদেশ", "বাংলা"]);
    }
}
