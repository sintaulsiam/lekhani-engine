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
        
        // 1. Fast Flat Binary Dictionary (or JSON fallback)
        let dict_bin_path = dir.join("dictionary.bin");
        if dict_bin_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&dict_bin_path) {
                let words: Vec<String> = content.lines().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                self.trie.insert_bulk(words);
            }
        } else {
            let dict_path = dir.join("dictionary.json");
            if dict_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&dict_path) {
                    if let Ok(raw_map) = serde_json::from_str::<HashMap<String, Vec<String>>>(&content) {
                        let mut words = Vec::with_capacity(160000);
                        for (_k, v_list) in raw_map {
                            words.extend(v_list);
                        }
                        let lines = words.join("\n");
                        let _ = std::fs::write(&dict_bin_path, lines);
                        self.trie.insert_bulk(words);
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
                    self.autocorrect = map.into_iter().filter(|(k, v)| k != v).collect();
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

    /// Check if word is an exact valid dictionary word
    pub fn is_exact_dictionary_word(&self, word: &str) -> bool {
        self.trie.contains_exact(word)
    }

    /// Lookup contextual emojis by keyword
    pub fn lookup_emoji_keywords(&self, keyword: &str) -> Vec<String> {
        self.emojis.lookup_by_keyword(keyword)
    }

    /// Find matching suffix
    pub fn find_suffix(&self, suffix_str: &str) -> Option<&str> {
        self.suffix.get(suffix_str).map(String::as_str)
    }

    /// Check for common phonetic transliteration overrides (e.g. gari -> গাড়ি, valo -> ভালো)
    pub fn get_common_phonetic_override(term: &str) -> Option<&'static str> {
        match term {
            "ami" => Some("আমি"),
            "tumi" => Some("তুমি"),
            "apni" => Some("আপনি"),
            "amra" => Some("আমরা"),
            "tara" => Some("তারা"),
            "kemon" => Some("কেমন"),
            "cha" => Some("চা"),
            "chai" => Some("চাই"),
            "chao" => Some("চাও"),
            "chobi" => Some("ছবি"),
            "chele" => Some("ছেলে"),
            "meye" => Some("মেয়ে"),
            "gari" | "garii" => Some("গাড়ি"),
            "bari" | "barii" => Some("বাড়ি"),
            "shari" | "sharii" => Some("শাড়ি"),
            "pora" => Some("পড়া"),
            "pori" => Some("পড়ি"),
            "valo" | "bhalo" => Some("ভালো"),
            "sundor" | "shundor" => Some("সুন্দর"),
            "daktar" => Some("ডাক্তার"),
            "taka" => Some("টাকা"),
            "thik" => Some("ঠিক"),
            "biral" => Some("বিড়াল"),
            "boro" => Some("বড়"),
            "choto" => Some("ছোট"),
            "ekta" | "akta" => Some("একটা"),
            "gaan" => Some("গান"),
            "tomake" => Some("তোমাকে"),
            "tomar" => Some("তোমার"),
            "tomra" => Some("তোমরা"),
            "bhalobasha" | "valobasha" => Some("ভালোবাসা"),
            "bhalobashi" | "valobashi" => Some("ভালোবাসি"),
            "bhalobasbo" | "valobasbo" => Some("ভালোবাসবো"),
            "jabo" => Some("যাব"),
            "jabe" => Some("যাবে"),
            "jacchi" => Some("যাচ্ছি"),
            "kothay" => Some("কোথায়"),
            "chara" => Some("ছাড়া"),
            "oboshoy" | "oboshsho" => Some("অবশ্যই"),
            "dim" => Some("ডিম"),
            "shagotom" | "shwagotom" => Some("স্বাগতম"),
            "dhonnobad" => Some("ধন্যবাদ"),
            "shotti" => Some("সত্যি"),
            "shob" => Some("সব"),
            "shokal" => Some("সকাল"),
            "shondha" | "shondhya" => Some("সন্ধ্যা"),
            "shomoy" => Some("সময়"),
            _ => None,
        }
    }

    /// Get natural grammatical inflections and associations for common root words
    pub fn get_common_inflections(term: &str) -> Option<&'static [&'static str]> {
        match term {
            "ami" | "aami" => Some(&["আমাকে", "আমার", "আমায়", "আমাদের", "আমিই"]),
            "tumi" => Some(&["তোমাকে", "তোমার", "তোমায়", "তোমরা", "তোমাদের", "তুমিই"]),
            "apni" => Some(&["আপনাকে", "আপনার", "আপনারা", "আপনাদের", "আপনিই"]),
            "amra" => Some(&["আমাদের", "আমাদেরকে", "আমরাই"]),
            "tara" => Some(&["তাদের", "তাদেরকে", "তারাই"]),
            "she" | "se" => Some(&["তাকে", "তার", "সেটাই", "সেই"]),
            "tini" => Some(&["তাঁকে", "তাঁর", "তাঁরা", "তিনিই"]),
            "amar" => Some(&["আমাদের", "আমারই"]),
            "tomar" => Some(&["তোমাদের", "তোমারই"]),
            "apnar" => Some(&["আপনাদের", "আপনারই"]),
            "kemon" => Some(&["কেমনে", "কেমন আছো", "কেমন আছেন"]),
            "valo" | "bhalo" => Some(&["ভালোই", "ভালোভাবে", "ভালোবাসা", "ভালোমন্দ"]),
            "bhalobasha" | "valobasha" => Some(&["ভালোবাসি", "ভালোবাসব", "ভালোবাসবে", "ভালোবাসতাম"]),
            "bhalobashi" | "valobashi" => Some(&["ভালোবাসি", "ভালোবাসো", "ভালোবাসে"]),
            "gari" | "garii" => Some(&["গাড়িতে", "গাড়ির", "গাড়িটি", "গাড়িগুলো"]),
            "bari" | "barii" => Some(&["বাড়িতে", "বাড়ির", "বাড়িটি", "বাড়িগুলো"]),
            "taka" => Some(&["টাকায়", "টাকার", "টাকাটা", "টাকাগুলো"]),
            "shob" => Some(&["সবাই", "সবাইকে", "সবার", "সবকিছু"]),
            "cha" => Some(&["চাই", "চাও", "চায়", "চাচ্ছি"]),
            "gaan" => Some(&["গানে", "গানের", "গানটি", "গানগুলো"]),
            "kothay" => Some(&["কোথাও", "কোথাকার"]),
            "kichu" => Some(&["কিছুই", "কিছুটা", "কিছুতেই"]),
            "kotha" => Some(&["কথায়", "কথার", "কথাটি", "কথাবার্তা"]),
            "kaaj" | "kaj" => Some(&["কাজে", "কাজের", "কাজটি", "কাজকর্ম"]),
            "manush" => Some(&["মানুষের", "মানুষকে", "মানুষজন", "মানুষটি"]),
            "desh" => Some(&["দেশে", "দেশের", "দেশবাসী"]),
            "din" => Some(&["দিনে", "দিনের", "দিনকাল", "দিনরাত"]),
            "shomoy" => Some(&["সময়ে", "সময়ের", "সময়মতো"]),
            "bochor" => Some(&["বছরে", "বছরের", "বছরব্যাপী"]),
            "shondha" | "shondhya" => Some(&["সন্ধ্যায়", "সন্ধ্যাবেলা"]),
            "shokal" => Some(&["সকালে", "সকালবেলা"]),
            "raat" | "rat" => Some(&["রাতে", "রাতের", "রাতভর"]),
            _ => None,
        }
    }

    /// Get raw autocorrect replacement (user autocorrect -> system autocorrect -> common overrides)
    pub fn get_autocorrect_raw(&self, term: &str) -> Option<String> {
        if let Some(correct) = self.user_autocorrect.get(term) {
            return Some(correct.clone());
        }
        if let Some(correct) = self.autocorrect.get(term) {
            return Some(correct.clone());
        }
        if let Some(lower_match) = self.autocorrect.get(&term.to_lowercase()) {
            return Some(lower_match.clone());
        }
        Self::get_common_phonetic_override(term).map(|s| s.to_string())
    }

    /// Search for non-transliterated specials (snippets, math, currency, exact emoji/symbols)
    pub fn search_special_literals(&self, term: &str) -> Vec<String> {
        let mut results = Vec::new();

        // 1. Math / Unit / Dynamic Snippets
        let snips = self.snippets.expand_all(term);
        results.extend(snips);

        // 2. Exact Emoji / Symbol Shortcode / Emoticons
        if let Some(emoji) = self.emojis.lookup(term) {
            if !results.contains(emoji) {
                results.push(emoji.clone());
            }
        }

        results
    }

    /// Search for autocorrect, snippet, math, currency, or emoji matches returning all candidates
    pub fn search_special_all(&self, term: &str) -> Vec<String> {
        let mut results = self.search_special_literals(term);

        // User / System Autocorrect
        if let Some(correct) = self.user_autocorrect.get(term) {
            if !results.contains(correct) {
                results.push(correct.clone());
            }
        }
        if let Some(correct) = self.autocorrect.get(term) {
            if !results.contains(correct) {
                results.push(correct.clone());
            }
        }

        results
    }

    /// Search for live emoji and symbol autocomplete
    pub fn search_emojis_prefix(&self, query: &str, limit: usize) -> Vec<String> {
        self.emojis.search_prefix(query, limit)
    }

    /// Search for live snippet autocomplete
    pub fn search_snippets_prefix(&self, query: &str, limit: usize) -> Vec<String> {
        self.snippets.search_prefix(query, limit)
    }
}
