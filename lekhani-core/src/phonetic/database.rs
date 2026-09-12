//! Phonetic Database & Trie Dictionary Engine

use crate::emojis::EmojiMap;
use crate::phonetic::AutonomousLearner;
use crate::snippets::SnippetManager;
use crate::trie::PrefixTrie;
use hashbrown::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct PhoneticDatabase {
    pub trie: PrefixTrie,
    suffix: HashMap<String, String>,
    autocorrect: HashMap<String, String>,
    user_autocorrect: HashMap<String, String>,
    emojis: EmojiMap,
    snippets: SnippetManager,
    pub learner: AutonomousLearner,
}

impl Default for PhoneticDatabase {
    fn default() -> Self {
        Self::new()
    }
}

pub const CORE_SUFFIXES: &[(&str, &str)] = &[
    ("ta", "টা"),
    ("Ta", "টা"),
    ("ti", "টি"),
    ("Ti", "টি"),
    ("tai", "টাই"),
    ("Tai", "টাই"),
    ("gulo", "গুলো"),
    ("gula", "গুলা"),
    ("guli", "গুলি"),
    ("der", "দের"),
    ("ke", "কে"),
    ("re", "রে"),
    ("te", "তে"),
    ("e", "ে"),
    ("er", "ের"),
    ("r", "র"),
    ("y", "য়"),
    ("ye", "য়ে"),
    ("ra", "রা"),
    ("bhabe", "ভাবে"),
    ("khana", "খানা"),
    ("khani", "খানি"),
    ("shomuh", "সমূহ"),
    ("shil", "শীল"),
    ("hin", "হীন"),
    ("jon", "জন"),
    ("to", "তো"),
    ("i", "ই"),
    ("o", "ও"),
];

impl PhoneticDatabase {
    pub fn new() -> Self {
        let mut trie = PrefixTrie::new();
        for &(word, freq) in CORE_BENGALI_FREQUENCIES {
            trie.insert_weighted(word.to_string(), freq);
        }
        trie.ensure_sorted();

        let mut suffix = HashMap::new();
        for &(k, v) in CORE_SUFFIXES {
            suffix.insert(k.to_string(), v.to_string());
        }

        Self {
            trie,
            suffix,
            autocorrect: HashMap::new(),
            user_autocorrect: HashMap::new(),
            emojis: EmojiMap::new(),
            snippets: SnippetManager::new(),
            learner: AutonomousLearner::new(),
        }
    }

    /// Load database from a directory containing dictionary.json, suffix.json, autocorrect.json
    pub fn load_from_dir<P: AsRef<Path>>(
        &mut self,
        dir: P,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let dir = dir.as_ref();

        // 1. Fast Flat Binary Dictionary (or JSON fallback)
        let dict_bin_path = dir.join("dictionary.bin");
        if dict_bin_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&dict_bin_path) {
                let words: Vec<String> = content
                    .lines()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                self.trie.insert_bulk(words);
            }
        } else {
            let dict_path = dir.join("dictionary.json");
            if dict_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&dict_path) {
                    if let Ok(raw_map) =
                        serde_json::from_str::<HashMap<String, Vec<String>>>(&content)
                    {
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
                    self.suffix.extend(map);
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

        // Inject core high-frequency weights into trie
        for &(word, freq) in CORE_BENGALI_FREQUENCIES {
            self.trie.insert_weighted(word.to_string(), freq);
        }
        self.trie.ensure_sorted();

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

    /// Save user-specific autocorrect file
    pub fn save_user_autocorrect<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let json = serde_json::to_string_pretty(&self.user_autocorrect)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load user-learned vocabulary
    pub fn load_user_learned<P: AsRef<Path>>(&mut self, path: P) {
        self.learner = AutonomousLearner::load_from_path(path);
        for word in &self.learner.learned_words {
            self.trie.insert_weighted(word.clone(), 9500);
        }
    }

    /// Save user-learned vocabulary
    pub fn save_user_learned<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        self.learner.save_to_path(path)
    }

    /// Observe committed word and auto-learn new vocabulary / root stems
    pub fn observe_committed_word(&mut self, word: &str) -> Vec<String> {
        self.learner.observe_and_learn(word, &mut self.trie)
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

    /// Search dictionary with zero-allocation borrowed string slices
    pub fn search_dictionary_ref<'a>(&'a self, prefix: &str, limit: usize) -> Vec<&'a str> {
        self.trie.find_prefix_matches_ref(prefix, limit)
    }

    /// Search dictionary returning word slice and frequency weight
    pub fn search_dictionary_entries<'a>(&'a self, prefix: &str, limit: usize) -> Vec<(&'a str, u32)> {
        self.trie.find_prefix_entries(prefix, limit)
    }

    /// Get corpus frequency of a word
    pub fn get_frequency(&self, word: &str) -> u32 {
        self.trie.get_frequency(word)
    }

    /// Check if word is an exact valid dictionary word
    pub fn is_exact_dictionary_word(&self, word: &str) -> bool {
        self.trie.contains_exact(word)
    }

    /// Bilingual loanwords for natural code-mixing (English -> Transliteration, English Word)
    pub fn get_bilingual_loanword(term: &str) -> Option<(&'static str, &'static str)> {
        match term {
            "meeting" => Some(("মিটিং", "meeting")),
            "laptop" => Some(("ল্যাপটপ", "laptop")),
            "password" => Some(("পাসওয়ার্ড", "password")),
            "office" => Some(("অফিস", "office")),
            "message" | "msg" => Some(("মেসেজ", "message")),
            "link" => Some(("লিংক", "link")),
            "call" => Some(("কল", "call")),
            "problem" => Some(("প্রবলেম", "problem")),
            "doctor" => Some(("ডাক্তার", "doctor")),
            "phone" => Some(("ফোন", "phone")),
            "mobile" => Some(("মোবাইল", "mobile")),
            "group" => Some(("গ্রুপ", "group")),
            "class" => Some(("ক্লাস", "class")),
            "time" => Some(("টাইম", "time")),
            "code" => Some(("কোড", "code")),
            "school" => Some(("স্কুল", "school")),
            "college" => Some(("কলেজ", "college")),
            "university" | "varsity" => Some(("ভার্সিটি", "university")),
            "bus" => Some(("বাস", "bus")),
            "train" => Some(("ট্রেন", "train")),
            "ticket" => Some(("টিকিট", "ticket")),
            "post" => Some(("পোস্ট", "post")),
            "share" => Some(("শেয়ার", "share")),
            "comment" => Some(("কমেন্ট", "comment")),
            "shirt" => Some(("শার্ট", "shirt")),
            "t-shirt" | "tshirt" => Some(("টি-শার্ট", "t-shirt")),
            "pant" | "pants" => Some(("প্যান্ট", "pant")),
            "shoe" | "shoes" => Some(("জুতো", "shoes")),
            "video" => Some(("ভিডিও", "video")),
            "photo" => Some(("ছবি", "photo")),
            "online" => Some(("অনলাইন", "online")),
            "network" => Some(("নেটওয়ার্ক", "network")),
            "system" => Some(("সিস্টেম", "system")),
            "file" => Some(("ফাইল", "file")),
            "typing" => Some(("টাইপিং", "typing")),
            _ => None,
        }
    }

    /// Lookup contextual emojis by keyword
    pub fn lookup_emoji_keywords(&self, keyword: &str) -> Vec<String> {
        self.emojis.lookup_by_keyword(keyword)
    }

    /// Find matching suffix
    pub fn find_suffix(&self, suffix_str: &str) -> Option<&str> {
        self.suffix.get(suffix_str).map(String::as_str)
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

    /// Get raw autocorrect replacement (user autocorrect -> system autocorrect)
    pub fn get_autocorrect_raw(&self, term: &str) -> Option<String> {
        if let Some(correct) = self.user_autocorrect.get(term) {
            return Some(correct.clone());
        }
        if let Some(lower_match) = self.user_autocorrect.get(&term.to_lowercase()) {
            return Some(lower_match.clone());
        }
        if let Some(correct) = self.autocorrect.get(term) {
            return Some(correct.clone());
        }
        if let Some(lower_match) = self.autocorrect.get(&term.to_lowercase()) {
            return Some(lower_match.clone());
        }
        None
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

/// Core high-frequency Bengali unigrams with ranking weights
pub const CORE_BENGALI_FREQUENCIES: &[(&str, u32)] = &[
    // Pronouns & Demonstratives
    ("আমি", 10000),
    ("তুমি", 9500),
    ("আপনি", 9200),
    ("সে", 9400),
    ("তিনি", 8800),
    ("আমরা", 9100),
    ("তোমরা", 8700),
    ("আপনারা", 8600),
    ("তারা", 8900),
    ("তাঁরা", 8400),
    ("এটা", 9300),
    ("এটাই", 9200),
    ("সেটা", 9100),
    ("সেটাই", 9000),
    ("ওটা", 8500),
    ("ওটাই", 8800),
    ("এই", 9400),
    ("সেই", 8900),
    ("যা", 8600),
    ("তা", 8700),
    ("কে", 9000),
    ("কি", 9200),
    ("কী", 9000),
    ("কেন", 9100),
    ("কেমন", 8900),
    ("কোথায়", 8800),
    ("কখন", 8700),
    ("না", 10000),
    ("হ্যাঁ", 9600),
    ("তো", 9500),
    ("যে", 9600),
    ("আচ্ছা", 9400),
    ("হাই", 9000),
    ("ওকে", 9100),
    ("কিভাবে", 8600),
    ("কবে", 8500),
    // Pronoun Inflections
    ("আমাকে", 9300),
    ("আমার", 9500),
    ("আমাদের", 9400),
    ("তোমাকে", 9200),
    ("তোমার", 9400),
    ("তোমাদের", 9100),
    ("আপনাকে", 9100),
    ("আপনার", 9300),
    ("আপনাদের", 9000),
    ("তাকে", 9200),
    ("তার", 9400),
    ("তাদের", 9200),
    ("কাউকে", 8700),
    ("কারো", 8700),
    ("সবার", 8900),
    ("সবাইকে", 8900),
    // Common Verbs & Conjugations
    ("পড়া", 9000),
    ("পরা", 8800),
    ("পড়ছি", 8900),
    ("পড়ব", 8800),
    ("পড়াশোনা", 8900),
    ("করা", 9500),
    ("করব", 9200),
    ("করছি", 9100),
    ("করছে", 9100),
    ("করেছি", 9100),
    ("করেছে", 9200),
    ("করবেন", 9100),
    ("করলাম", 8900),
    ("বলা", 9200),
    ("বলি", 8900),
    ("বলছি", 9000),
    ("বলেছি", 9000),
    ("বলবেন", 8900),
    ("হওয়া", 9400),
    ("হলো", 9200),
    ("হবে", 9300),
    ("হচ্ছে", 9100),
    ("হয়েছে", 9200),
    ("যাওয়া", 9300),
    ("যাব", 9100),
    ("যাচ্ছি", 9000),
    ("যাবেন", 9000),
    ("গেছে", 9100),
    ("গেল", 9000),
    ("আসা", 9200),
    ("আসছি", 9000),
    ("আসবে", 9000),
    ("আসবেন", 8900),
    ("এসেছে", 9000),
    ("খাওয়া", 9100),
    ("খাব", 9000),
    ("খাচ্ছি", 8900),
    ("খেয়েছি", 8900),
    ("নেওয়া", 9000),
    ("নেব", 8800),
    ("নিচ্ছি", 8800),
    ("দেওয়া", 9200),
    ("দেখা", 9100),
    ("দেখছি", 8900),
    ("দেখব", 8900),
    ("শোনা", 8900),
    ("জানা", 9000),
    ("জানি", 9100),
    ("জানেন", 8900),
    ("বোঝা", 8800),
    ("বুঝতে", 8900),
    ("থাকা", 9100),
    ("আছি", 9300),
    ("আছো", 9200),
    ("আছেন", 9200),
    ("থাকব", 8900),
    ("রাখা", 8800),
    ("চাওয়া", 8900),
    ("চাই", 9200),
    ("চাও", 8900),
    ("চান", 8900),
    ("পারা", 9100),
    ("পারি", 9000),
    ("পারব", 8900),
    ("পারবেন", 8900),
    ("লাগা", 8800),
    ("লাগে", 9000),
    ("লাগল", 8800),
    ("চলা", 8700),
    ("লেখা", 8900),
    ("লিখছি", 8800),
    ("লিখব", 8800),
    ("শেখা", 8700),
    ("পাঠানো", 8600),
    ("পাঠিয়েছি", 8600),
    ("জানানো", 8500),
    ("বানানো", 8500),
    ("ভালোবাসা", 9200),
    ("ভালোবাসি", 9300),
    ("ভালোবাসব", 9000),
    ("গান", 9300),
    ("গাই", 9100),
    ("পড়া", 9300),
    ("পড়ছি", 9100),
    ("পড়ব", 9100),
    ("পরা", 9100),
    ("পরছি", 9000),
    ("পরব", 9000),
    // Common Nouns
    ("বাংলাদেশ", 9800),
    ("ঢাকা", 9500),
    ("বাংলা", 9600),
    ("মানুষ", 9400),
    ("দেশ", 9300),
    ("ভাষা", 9200),
    ("কথা", 9400),
    ("কাজ", 9300),
    ("সময়", 9300),
    ("দিন", 9200),
    ("রাত", 9000),
    ("বছর", 9100),
    ("মাস", 8900),
    ("সপ্তাহ", 8700),
    ("সকাল", 9000),
    ("দুপুর", 8700),
    ("সুন্দর", 9300),
    ("সবুজ", 9000),
    ("সুখ", 9100),
    ("সুখী", 9000),
    ("সুস্থ", 9100),
    ("অসুস্থ", 8900),
    ("সম্ভব", 9300),
    ("অসম্ভব", 9000),
    ("সম্ভাবনা", 9000),
    ("স্বার্থ", 9100),
    ("সার্থকতা", 8900),
    ("সাক্ষী", 9000),
    ("স্বাক্ষর", 9000),
    ("দৃষ্টি", 9200),
    ("দৃষ্টিকোণ", 8900),
    ("অন্যান্য", 9200),
    ("অন্যরকম", 9000),
    ("পরিবর্তন", 9300),
    ("সাথে", 9500),
    ("সংস্কৃতি", 9200),
    ("সংস্কার", 9000),
    ("সংস্থা", 9200),
    ("উৎসব", 9200),
    ("উৎসাহ", 9100),
    ("উৎপাদন", 9200),
    ("উৎকৃষ্ট", 8900),
    ("উৎসর্গ", 8900),
    ("উৎস", 9100),
    ("চিৎকার", 9000),
    ("বসন্ত", 9100),
    ("চোখ", 9200),
    ("চোখে", 9200),
    ("জুড়ানো", 9000),
    ("জোড়ানো", 8900),
    ("ফুল", 9200),
    ("ফুটে", 9000),
    ("শীতের", 9100),
    ("সকালে", 9200),
    ("কুয়াশা", 9000),
    ("ঘেরা", 9000),
    ("হাঁটতে", 9100),
    ("লিখতে", 9200),
    ("চাও", 9200),
    ("এখনই", 9300),
    ("অভ্র", 9300),
    ("ফোনেটিক", 9200),
    ("চেষ্টা", 9400),
    ("করো", 9200),
    ("এটি", 9400),
    ("সহজ", 9300),
    ("চমৎকার", 9300),
    ("একটি", 9600),
    ("মাধ্যম", 9200),
    ("যুক্তাক্ষর", 9200),
    ("গেলেও", 9100),
    ("কোনো", 9400),
    ("ঝামেলা", 9100),
    ("হয়", 9600),
    ("শুধু", 9400),
    ("পরপর", 9000),
    ("বর্ণ", 9200),
    ("গুলো", 9500),
    ("টিপে", 9000),
    ("গেলেই", 9200),
    ("হলো", 9500),
    ("যেমন", 9300),
    ("কষ্ট", 9300),
    ("টাইপিং", 9200),
    ("সাহায্য", 9300),
    ("বিকেল", 8800),
    ("সন্ধ্যা", 8800),
    ("বাড়ি", 9200),
    ("ঘর", 8900),
    ("রাস্তা", 8800),
    ("গাড়ি", 9000),
    ("বই", 9300),
    ("শার্ট", 8600),
    ("জামা", 8500),
    ("কাপড়", 8600),
    ("কলম", 8700),
    ("খাতা", 8600),
    ("কাগজ", 8500),
    ("পানি", 9200),
    ("জল", 8900),
    ("ভাত", 9100),
    ("রুটি", 8700),
    ("চা", 9200),
    ("কফি", 8600),
    ("দুধ", 8700),
    ("চিনি", 8600),
    ("মিষ্টি", 8800),
    ("ফল", 8800),
    ("মাছ", 9000),
    ("মাংস", 8800),
    ("ডিম", 8900),
    ("টাকা", 9300),
    ("পয়সা", 8400),
    ("ব্যাংক", 8800),
    ("অফিস", 9000),
    ("স্কুল", 9100),
    ("কলেজ", 8900),
    ("বিশ্ববিদ্যালয়", 9000),
    ("হাসপাতাল", 8800),
    ("ডাক্তার", 9000),
    ("শিক্ষক", 8900),
    ("বন্ধু", 9200),
    ("ভাই", 9100),
    ("বোন", 9000),
    ("মা", 9500),
    ("বাবা", 9400),
    ("ছেলে", 9200),
    ("মেয়ে", 9100),
    ("সন্তান", 8700),
    ("পরিবার", 9100),
    ("সমাজ", 8800),
    ("সরকার", 9000),
    ("পৃথিবী", 8900),
    ("আকাশ", 9000),
    ("বাতাস", 8800),
    ("নদী", 9000),
    ("সাগর", 8800),
    ("বৃষ্টি", 9000),
    ("রোদ", 8700),
    ("আলো", 8900),
    ("ফুল", 9000),
    ("গাছ", 8900),
    ("বিড়াল", 8700),
    ("কুকুর", 8600),
    ("পাখি", 8900),
    // Adjectives, Adverbs, Connectives & Particles
    ("ভালো", 9600),
    ("সুন্দর", 9400),
    ("বড়", 9300),
    ("ছোট", 9200),
    ("নতুন", 9100),
    ("পুরনো", 8700),
    ("খারাপ", 9000),
    ("সহজ", 9000),
    ("কঠিন", 8800),
    ("অনেক", 9400),
    ("অল্প", 8800),
    ("বেশি", 9300),
    ("কম", 9000),
    ("খুব", 9400),
    ("দারুণ", 8900),
    ("চমৎকার", 8800),
    ("প্রিয়", 8900),
    ("সব", 9300),
    ("সবাই", 9200),
    ("সবকিছু", 9100),
    ("কিছু", 9200),
    ("কোনো", 9100),
    ("একটু", 9200),
    ("একদম", 9000),
    ("অবশ্যই", 9100),
    ("সত্যি", 9100),
    ("ঠিক", 9300),
    ("ভুল", 9000),
    ("সাথে", 9400),
    ("সঙ্গে", 9100),
    ("ছাড়া", 9000),
    ("মতো", 9200),
    ("জন্য", 9500),
    ("কারণ", 9200),
    ("কিন্তু", 9400),
    ("এবং", 9400),
    ("অথবা", 9000),
    ("তবে", 9100),
    ("আর", 9500),
    ("তাই", 9300),
    ("যদি", 9200),
    ("ধন্যবাদ", 9200),
    ("স্বাগতম", 8900),
    ("শুভেচ্ছা", 9000),
    ("অভিনন্দন", 8800),
    // High-Frequency Conjuncts (যুক্তবর্ণ)
    ("কষ্ট", 9500),
    ("নষ্ট", 9300),
    ("বিজ্ঞান", 9400),
    ("বিজ্ঞানী", 9300),
    ("সৃষ্টি", 9400),
    ("বৃষ্টি", 9400),
    ("দৃষ্টি", 9300),
    ("মিষ্টি", 9400),
    ("দুষ্টু", 9100),
    ("অনুষ্ঠান", 9300),
    ("ব্যবস্থা", 9400),
    ("অবস্থা", 9400),
    ("স্পষ্ট", 9200),
    ("শ্রেষ্ঠ", 9200),
    ("সন্তুষ্ট", 9100),
    ("সত্য", 9500),
    ("সত্যি", 9500),
    ("মিথ্যা", 9400),
    ("তথ্য", 9400),
    ("পদ্ধতি", 9300),
    ("বুদ্ধি", 9300),
    ("শুদ্ধ", 9200),
    ("ইচ্ছা", 9500),
    ("শিক্ষা", 9500),
    ("পরীক্ষা", 9400),
    ("অক্ষর", 9200),
    ("দক্ষ", 9200),
    ("দক্ষতা", 9300),
    ("লক্ষ্য", 9400),
    ("লক্ষ্মী", 9200),
    ("ক্ষমা", 9300),
    ("ক্ষতি", 9300),
    ("স্মৃতি", 9300),
    ("স্মরণীয়", 9200),
    ("চিহ্ন", 9200),
    ("কৃষ্ণ", 9200),
    ("তৃষ্ণা", 9100),
    ("স্বাস্থ্য", 9400),
    ("স্বাধীনতা", 9400),
    ("স্বাধীন", 9300),
    ("বিশ্বাস", 9500),
    ("উজ্জ্বল", 9200),
    ("লজ্জা", 9200),
    ("শত্রু", 9200),
    ("প্রশ্ন", 9400),
    ("শ্রদ্ধা", 9300),
    ("শান্তি", 9400),
    ("আনন্দ", 9500),
    ("শব্দ", 9400),
    ("শব", 8500),
    ("স্বপ্ন", 9400),
    ("সম্পর্ক", 9500),
    ("সম্পূর্ণ", 9400),
    ("জ্ঞান", 9500),
    ("জ্ঞানী", 9300),
    ("অজ্ঞান", 9100),
    ("বিজ্ঞাপন", 9300),
    ("জিজ্ঞাসা", 9300),
    ("পরিষ্কার", 9400),
    ("পুরস্কার", 9400),
    ("আবিষ্কার", 9300),
    // Chandrabindu Words (ঁ)
    ("চাঁদ", 9400),
    ("দাঁত", 9300),
    ("ফাঁকা", 9100),
    ("বাঁধ", 9100),
    ("হাঁস", 9000),
    ("বাঁশ", 9000),
    ("পাঁচ", 9300),
    ("ঠোঁট", 9100),
    ("খুঁজি", 9100),
    ("ঝাঁঝ", 8900),
    ("গাঁজা", 8900),
    ("কাঁচ", 9000),
    // Khanda-Ta & Hasanta (ৎ)
    ("হঠাৎ", 9400),
    ("উৎসব", 9300),
    ("উৎসাহ", 9200),
    ("উৎপন্ন", 9100),
    ("উৎপাদন", 9200),
    ("উৎকৃষ্ট", 9100),
    ("বিখ্যাত", 9200),
    ("সৎ", 9100),
    ("তৎপর", 9000),
    // Bengali Glide & Frequent Verb Endings
    ("খাওয়া", 9400),
    ("দেওয়া", 9400),
    ("নেওয়া", 9300),
    ("যাওয়া", 9400),
    ("হাওয়া", 9200),
    ("পাওয়া", 9400),
    ("চাওয়া", 9200),
    ("হওয়া", 9500),
    ("শোনা", 9300),
    ("বোঝা", 9200),
    ("শোওয়া", 9000),
    ("প্লিজ", 9200),
    ("ভাই", 9500),
    ("আপু", 9300),
    ("ভাবি", 9100),
    ("শুভ", 9400),
    // Reph & Motion Verbs
    ("বর্তমান", 9500),
    ("অর্থনৈতিক", 9400),
    ("কার্যক্রম", 9300),
    ("প্রদর্শিত", 9200),
    ("দুর্নীতি", 9400),
    ("পরিবর্তন", 9400),
    ("পরিবর্তিত", 9300),
    ("সূর্য", 9300),
    ("আশীর্বাদ", 9200),
    ("আশীর্বাদে", 9200),
    ("কর্ম", 9300),
    ("ধর্ম", 9400),
    ("তর্ক", 9100),
    ("সতর্ক", 9200),
    ("যাব", 9500),
    ("যাচ্ছি", 9400),
    ("যাচ্ছ", 9300),
    ("যাচ্ছেন", 9400),
    ("যাচ্ছে", 9400),
    ("যেতে", 9400),
    ("করছেন", 9500),
    ("করছে", 9500),
    ("করছো", 9400),
    ("করছ", 9300),
    ("বলছেন", 9400),
    ("বলছে", 9400),
    ("বলছো", 9300),
    ("দেখছেন", 9300),
    ("দেখছে", 9300),
    ("নিচ্ছেন", 9400),
    ("নিচ্ছে", 9400),
    ("দিচ্ছেন", 9400),
    ("দিচ্ছে", 9400),
    ("যুদ্ধ", 9500),
    ("যুদ্ধের", 9400),
    ("মুক্তিযুদ্ধ", 9500),
    ("মুক্তিযুদ্ধের", 9500),
    ("যন্ত্র", 9300),
    ("যোগাযোগ", 9400),
    ("প্রযুক্তি", 9500),
    ("প্রযুক্তির", 9400),
    ("অনুযায়ী", 9400),
    ("অনুযায়ী", 9400),
    ("যোগ্য", 9300),
    // Geminates & Ja-fala
    ("বিদ্যালয়", 9400),
    ("বিদ্যালয়", 9400),
    ("বিদ্যালয়ে", 9300),
    ("বিদ্যালয়ে", 9300),
    ("বিদ্বান", 9200),
    ("বিদ্যুৎ", 9400),
    ("তত্ত্ব", 9300),
    ("জন্য", 9600),
    ("জন্যে", 9500),
    ("ক্ষেত্র", 9400),
    // Science, Environment & Law
    ("কৃত্রিম", 9300),
    ("মাতৃভূমি", 9300),
    ("প্রাকৃতিক", 9300),
    ("বৃদ্ধি", 9400),
    ("বুদ্ধিজীবী", 9200),
    ("দূষণ", 9300),
    ("বায়ুমণ্ডলীয়", 9200),
    ("বায়ুমণ্ডলীয়", 9200),
    ("বায়ুমণ্ডল", 9200),
    ("পরিবেশ", 9400),
    ("ঘটছে", 9300),
    ("ঘটনা", 9400),
    ("একটি", 9600),
    ("একটা", 9600),
    ("উঠে", 9400),
    ("উঠেছে", 9300),
    ("উঠব", 9400),
    ("কঠিন", 9300),
    ("ছোট", 9400),
    ("ছোট্ট", 9300),
    ("ছোটটা", 9200),
    ("ভর্তা", 9200),
    ("উল্টো", 9200),
    ("ঘণ্টা", 9300),
    ("ঘন্টা", 9300),
    // Sibilant Standard (Bangla Academy প্রমিত স)
    ("সরকার", 9600),
    ("সরকারি", 9500),
    ("সরকারী", 9300),
    ("সবাই", 9600),
    ("সবাইকে", 9400),
    ("সংবিধান", 9400),
    ("সংবিধানের", 9400),
    ("সংখ্যাগরিষ্ঠ", 9300),
    ("সদস্য", 9400),
    ("সদস্যদের", 9300),
    ("সিদ্ধান্ত", 9400),
    ("প্রশাসন", 9400),
    ("প্রশাসনিক", 9300),
    ("সচেতনতা", 9300),
    ("সচেতন", 9300),
    ("সাশ্রয়", 9200),
    ("সাশ্রয়", 9200),
    ("সাশ্রয়ী", 9200),
    ("সাশ্রয়ী", 9200),
    ("সাংবাদিক", 9300),
    ("সাধারণ", 9500),
    ("সন্ধ্যা", 9300),
    ("শান্ত", 9300),
    ("শান্তিতে", 9300),
    ("আত্মত্যাগ", 9300),
    ("শহীদ", 9400),
    ("শহীদের", 9300),
    ("শহীদদের", 9300),
    ("অনুরোধ", 9400),
    ("মনোযোগ", 9300),
    ("অবলোকন", 9100),
    ("বিশেষজ্ঞ", 9300),
    ("বিশেষজ্ঞরা", 9300),
    // Suffix-inflected forms
    ("বইটা", 9300),
    ("বইটি", 9300),
    ("কাজটা", 9400),
    ("গাছটা", 9300),
    ("মানুষগুলো", 9400),
    ("মানুষজন", 9400),
    ("বইগুলো", 9300),
    ("দেশটা", 9300),
    ("কথাটা", 9400),
    ("সবগুলো", 9400),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf8_autocorrect_loading() {
        let mut db = PhoneticDatabase::new();
        let ac_candidates = [
            "../../data",
            "data",
            "../data",
        ];
        for dir in ac_candidates {
            let p = std::path::Path::new(dir).join("dictionaries");
            if p.exists() {
                let _ = db.load_from_dir(&p);
                break;
            }
        }

        if let Some(res) = db.get_autocorrect_raw("account") {
            assert_eq!(res, "অ্যাকাউন্ট");
        }
        if let Some(res) = db.get_autocorrect_raw("birthday") {
            assert_eq!(res, "বার্থডে");
        }
    }
}
