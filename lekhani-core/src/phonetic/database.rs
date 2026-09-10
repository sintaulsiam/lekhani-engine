//! Phonetic Database & Trie Dictionary Engine

use hashbrown::HashMap;
use std::path::Path;
use crate::emojis::EmojiMap;
use crate::snippets::SnippetManager;
use crate::trie::PrefixTrie;

#[derive(Debug, Clone)]
pub struct PhoneticDatabase {
    trie: PrefixTrie,
    suffix: HashMap<String, String>,
    autocorrect: HashMap<String, String>,
    user_autocorrect: HashMap<String, String>,
    emojis: EmojiMap,
    snippets: SnippetManager,
}

impl Default for PhoneticDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl PhoneticDatabase {
    pub fn new() -> Self {
        Self {
            trie: PrefixTrie::new(),
            suffix: HashMap::new(),
            autocorrect: HashMap::new(),
            user_autocorrect: HashMap::new(),
            emojis: EmojiMap::new(),
            snippets: SnippetManager::new(),
        }
    }

    /// Load database from a directory containing dictionary.json, suffix.json, autocorrect.json
    pub fn load_from_dir<P: AsRef<Path>>(&mut self, dir: P) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let dir = dir.as_ref();
        
        let dict_path = dir.join("dictionary.json");
        if dict_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&dict_path) {
                if let Ok(raw_map) = serde_json::from_str::<HashMap<String, Vec<String>>>(&content) {
                    for (k, v_list) in raw_map {
                        for v in v_list {
                            self.trie.insert(&k, v);
                        }
                    }
                }
            }
        }

        let suffix_path = dir.join("suffix.json");
        if suffix_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&suffix_path) {
                if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content) {
                    self.suffix = map;
                }
            }
        }

        let ac_path = dir.join("autocorrect.json");
        if ac_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&ac_path) {
                if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content) {
                    self.autocorrect = map;
                }
            }
        }

        Ok(())
    }

    /// Load user-specific autocorrect file
    pub fn load_user_autocorrect<P: AsRef<Path>>(&mut self, path: P) {
        let path = path.as_ref();
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content) {
                    self.user_autocorrect = map;
                }
            }
        }
    }

    /// Add custom user autocorrect entry
    pub fn insert_user_autocorrect(&mut self, trigger: String, replacement: String) {
        self.user_autocorrect.insert(trigger, replacement);
    }

    /// Remove custom user autocorrect entry
    pub fn remove_user_autocorrect(&mut self, trigger: &str) -> Option<String> {
        self.user_autocorrect.remove(trigger)
    }

    pub fn get_user_autocorrect(&self) -> &HashMap<String, String> {
        &self.user_autocorrect
    }

    pub fn get_system_autocorrect(&self) -> &HashMap<String, String> {
        &self.autocorrect
    }

    /// Search dictionary using fast prefix trie
    pub fn search_dictionary(&self, prefix: &str, limit: usize) -> Vec<String> {
        self.trie.find_prefix_matches(prefix, limit)
    }

    /// Find matching suffix
    pub fn find_suffix(&self, suffix_str: &str) -> Option<&str> {
        self.suffix.get(suffix_str).map(String::as_str)
    }

    /// Search for autocorrect, snippet, or emoji match
    pub fn search_special(&self, term: &str) -> Option<String> {
        // 1. Snippets / Macros
        if let Some(snip) = self.snippets.expand(term) {
            return Some(snip);
        }

        // 2. User Autocorrect
        if let Some(correct) = self.user_autocorrect.get(term) {
            return Some(correct.clone());
        }

        // 3. System Autocorrect
        if let Some(correct) = self.autocorrect.get(term) {
            return Some(correct.clone());
        }

        // 4. Emoji / Symbol Shortcode
        if let Some(emoji) = self.emojis.lookup(term) {
            return Some(emoji.clone());
        }

        None
    }
}
