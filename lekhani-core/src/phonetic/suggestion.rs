//! Phonetic Suggestion Generation Engine

use edit_distance::edit_distance;
use hashbrown::HashMap;
use rupantor::parser::PhoneticParser;
use serde_json::Value;

use super::database::PhoneticDatabase;
use crate::chars::BengaliCharExt;

use std::sync::Arc;

#[derive(Clone)]
pub struct PhoneticSuggestion {
    pub database: PhoneticDatabase,
    phonetic_parser: Option<Arc<PhoneticParser>>,
    cache: HashMap<String, Vec<String>>,
}

impl std::fmt::Debug for PhoneticSuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhoneticSuggestion")
            .field("database", &self.database)
            .field("has_parser", &self.phonetic_parser.is_some())
            .field("cache", &self.cache)
            .finish()
    }
}

impl Default for PhoneticSuggestion {
    fn default() -> Self {
        Self::new()
    }
}

impl PhoneticSuggestion {
    pub fn new() -> Self {
        Self {
            database: PhoneticDatabase::new(),
            phonetic_parser: None,
            cache: HashMap::new(),
        }
    }

    pub fn with_layout(layout_json: &Value) -> Self {
        let layout_obj = if let Some(l) = layout_json.get("layout") {
            l
        } else {
            layout_json
        };
        let parser = Arc::new(PhoneticParser::new(layout_obj));
        Self {
            database: PhoneticDatabase::new(),
            phonetic_parser: Some(parser),
            cache: HashMap::new(),
        }
    }

    pub fn set_layout(&mut self, layout_json: &Value) {
        let layout_obj = if let Some(l) = layout_json.get("layout") {
            l
        } else {
            layout_json
        };
        self.phonetic_parser = Some(Arc::new(PhoneticParser::new(layout_obj)));
        self.cache.clear();
    }

    /// Transliterate directly using Avro phonetic rules
    pub fn convert_phonetic(&self, text: &str) -> String {
        if let Some(ref parser) = self.phonetic_parser {
            parser.convert(text)
        } else {
            text.to_string()
        }
    }

    /// Generate ranked candidates for typed term
    pub fn suggest(
        &mut self,
        term: &str,
        include_english: bool,
        use_dictionary: bool,
        candidate_memory: &HashMap<String, String>,
    ) -> (Vec<String>, usize) {
        if term.is_empty() {
            return (Vec::new(), 0);
        }

        let (pre, middle, post) = split_word_punct(term);

        if !use_dictionary || middle.is_empty() {
            let lonely = format!("{}{}{}", self.convert_phonetic(pre), self.convert_phonetic(middle), self.convert_phonetic(post));
            return (vec![lonely], 0);
        }

        let phonetic = self.convert_phonetic(middle);
        let mut candidates: Vec<String> = Vec::with_capacity(10);

        // 1. Special Matches (Emojis, Autocorrect, Snippets)
        if let Some(special) = self.database.search_special(term) {
            candidates.push(special);
        } else if let Some(special) = self.database.search_special(middle) {
            candidates.push(special);
        }

        // 2. Dictionary Search & Suffix Mutation
        if !self.cache.contains_key(middle) {
            let dict_matches = self.database.search_dictionary(middle, 20);
            self.cache.insert(middle.to_string(), dict_matches);
        }

        let mut suffixed_matches = self.add_suffixes(middle);
        suffixed_matches.sort_unstable_by(|a, b| edit_distance(&phonetic, a).cmp(&edit_distance(&phonetic, b)));

        for item in suffixed_matches {
            if !candidates.iter().any(|c| c == &item) {
                candidates.push(item);
            }
        }

        // 3. Fallback direct phonetic
        if !candidates.iter().any(|c| c == &phonetic) {
            candidates.push(phonetic.clone());
        }

        // 4. Original English
        if include_english && !candidates.iter().any(|c| c == term) {
            candidates.push(term.to_string());
        }

        // Apply pre and post punctuation
        let mut final_candidates = Vec::with_capacity(candidates.len());
        let pre_converted = self.convert_phonetic(pre);
        let post_converted = self.convert_phonetic(post);

        for cand in candidates {
            final_candidates.push(format!("{}{}{}", pre_converted, cand, post_converted));
        }

        // Determine previously selected candidate index
        let selected_index = if let Some(fav) = candidate_memory.get(term).or_else(|| candidate_memory.get(middle)) {
            final_candidates.iter().position(|c| c == fav || c.contains(fav)).unwrap_or(0)
        } else {
            0
        };

        (final_candidates, selected_index)
    }

    fn add_suffixes(&self, middle: &str) -> Vec<String> {
        let mut list = self.cache.get(middle).cloned().unwrap_or_default();

        if middle.len() > 2 {
            for i in 1..middle.len() {
                let suffix_key = &middle[i..];
                if let Some(suffix) = self.database.find_suffix(suffix_key) {
                    let base_key = &middle[..i];
                    if let Some(base_cache) = self.cache.get(base_key) {
                        for base in base_cache {
                            let mut word = base.clone();
                            if let (Some(base_rmc), Some(suffix_lmc)) = (base.chars().last(), suffix.chars().next()) {
                                if base_rmc.is_vowel() && suffix_lmc.is_kar() {
                                    word.push('য়');
                                } else if base_rmc == 'ৎ' {
                                    word.pop();
                                    word.push('ত');
                                } else if base_rmc == 'ং' {
                                    word.pop();
                                    word.push('ঙ');
                                }
                            }
                            word.push_str(suffix);
                            if !list.iter().any(|item| item == &word) {
                                list.push(word);
                            }
                        }
                    }
                }
            }
        }

        list
    }
}

/// Helper to split leading/trailing punctuation from core word
pub fn split_word_punct(input: &str) -> (&str, &str, &str) {
    let mut start = 0;
    let mut end = input.len();

    let bytes = input.as_bytes();
    while start < bytes.len() && is_punct_byte(bytes[start]) {
        start += 1;
    }
    while end > start && is_punct_byte(bytes[end - 1]) {
        end -= 1;
    }

    (&input[..start], &input[start..end], &input[end..])
}

fn is_punct_byte(b: u8) -> bool {
    b.is_ascii_punctuation() && b != b'`'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_word_punct() {
        assert_eq!(split_word_punct("(kotha)"), ("(", "kotha", ")"));
        assert_eq!(split_word_punct("!ami,"), ("!", "ami", ","));
        assert_eq!(split_word_punct("bangla"), ("", "bangla", ""));
    }
}
