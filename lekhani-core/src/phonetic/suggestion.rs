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
        if text.is_empty() {
            return String::new();
        }
        // If string already contains non-ASCII characters (e.g. Bengali Unicode, emojis, symbols), return as-is
        if !text.is_ascii() {
            return text.to_string();
        }
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

        // Fast Memoization Cache check
        let cache_key = format!("{}:{}:{}", term, include_english, use_dictionary);
        if let Some(cached) = self.cache.get(&cache_key) {
            let selected_index = if let Some(fav) = candidate_memory
                .get(term)
                .or_else(|| candidate_memory.get(&cache_key))
            {
                cached.iter().position(|c| c == fav || c.contains(fav)).unwrap_or(0)
            } else {
                0
            };
            return (cached.clone(), selected_index);
        }

        // 1. Smart Typography & Punctuation
        match term {
            ".." => {
                let mut cands = vec!["।".to_string()];
                if include_english {
                    cands.push("..".to_string());
                }
                return (cands, 0);
            }
            "--" => {
                let mut cands = vec!["—".to_string()];
                if include_english {
                    cands.push("--".to_string());
                }
                return (cands, 0);
            }
            "..." => {
                let mut cands = vec!["…".to_string()];
                if include_english {
                    cands.push("...".to_string());
                }
                return (cands, 0);
            }
            "* " => {
                let mut cands = vec!["• ".to_string()];
                if include_english {
                    cands.push("* ".to_string());
                }
                return (cands, 0);
            }
            _ => {}
        }

        // 2. Direct whole-term special matches (Math `=125*8`, Currency `#usd50`, Snippets `!shubhechha`, Exact Emojis `:smile:`, `:)`, `$$`, `*taka*`)
        let literals = self.database.search_special_literals(term);
        if !literals.is_empty() {
            let mut cands = literals;
            if include_english && !cands.iter().any(|c| c == term) {
                cands.push(term.to_string());
            }
            return (cands, 0);
        }

        // 3. Live Snippet / Emoji / Symbol prefix autocompletion (e.g. "!sh", ":sm", "*t", ":bhalo")
        if (term.starts_with(':') || term.starts_with('*')) && term.len() >= 2 {
            let prefix_emojis = self.database.search_emojis_prefix(term, 8);
            if !prefix_emojis.is_empty() {
                let mut cands = prefix_emojis;
                if include_english && !cands.iter().any(|c| c == term) {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
        }
        if term.starts_with('!') && term.len() >= 2 {
            let prefix_snippets = self.database.search_snippets_prefix(term, 8);
            if !prefix_snippets.is_empty() {
                let mut cands = prefix_snippets;
                if include_english && !cands.iter().any(|c| c == term) {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
        }

        let (pre, middle, post) = split_word_punct(term);

        if middle.is_empty() {
            let lonely = format!(
                "{}{}{}",
                self.convert_phonetic(pre),
                self.convert_phonetic(middle),
                self.convert_phonetic(post)
            );
            return (vec![lonely], 0);
        }

        let phonetic = self.convert_phonetic(middle);
        let mut candidates: Vec<String> = Vec::with_capacity(12);

        // 4. Special literal matches on middle portion (e.g. (:smile:), (=12+5))
        let middle_literals = self.database.search_special_literals(middle);
        for special in middle_literals {
            if !candidates.contains(&special) {
                candidates.push(special);
            }
        }

        // 5. Autocorrect / Common Overrides
        let preferred_word = if let Some(raw_ac) = self.database.get_autocorrect_raw(middle) {
            let converted_ac = self.convert_phonetic(&raw_ac);
            if !converted_ac.is_empty() {
                if !candidates.contains(&converted_ac) {
                    candidates.push(converted_ac.clone());
                }
                Some(converted_ac)
            } else {
                None
            }
        } else {
            None
        };

        let primary = preferred_word.as_deref().unwrap_or(&phonetic);
        let is_primary_in_dict = self.database.is_exact_dictionary_word(primary);

        // 6. Direct Transliterations
        if !candidates.contains(&primary.to_string()) {
            candidates.push(primary.to_string());
        }
        if primary != &phonetic && !candidates.contains(&phonetic) {
            candidates.push(phonetic.clone());
        }

        if use_dictionary {
            // 7. Natural Linguistic & Grammatical Inflections (for frequent words - take top 3)
            if let Some(inflections) = PhoneticDatabase::get_common_inflections(middle) {
                for &inf in inflections.iter().take(3) {
                    let inf_str = inf.to_string();
                    if !candidates.contains(&inf_str) {
                        candidates.push(inf_str);
                    }
                }
            }

            // 8. Contextual Emoji / Symbol Keywords (e.g. bhalobasha -> ❤️, cha -> ☕, gari -> 🚗, taka -> ৳)
            let kw_emojis = self.database.lookup_emoji_keywords(middle);
            for em in kw_emojis {
                if !candidates.contains(&em) {
                    candidates.push(em);
                }
            }

            // 9. Remaining inflections if any
            if let Some(inflections) = PhoneticDatabase::get_common_inflections(middle) {
                for &inf in inflections.iter().skip(3) {
                    let inf_str = inf.to_string();
                    if !candidates.contains(&inf_str) {
                        candidates.push(inf_str);
                    }
                }
            }

            // 10. Suffix-decomposed forms (when Latin string contains stem + suffix)
            let suffixed_matches = self.add_suffixes(middle, primary);
            for item in suffixed_matches {
                if !candidates.contains(&item) {
                    candidates.push(item);
                }
            }

            // 11. Exact Fuzzy Spelling Variants (Homophones / Orthographic variants)
            let fuzzy_variants = super::fuzzy::generate_phonetic_variants(middle);
            let mut exact_fuzzy_matches: Vec<String> = Vec::new();

            for variant in &fuzzy_variants {
                let var_phonetic = self.convert_phonetic(variant);
                if self.database.is_exact_dictionary_word(&var_phonetic) {
                    let len_diff = (var_phonetic.chars().count() as isize - phonetic.chars().count() as isize).abs();
                    if len_diff <= 1
                        && !exact_fuzzy_matches.contains(&var_phonetic)
                        && &var_phonetic != primary
                        && var_phonetic != phonetic
                    {
                        exact_fuzzy_matches.push(var_phonetic);
                    }
                }
            }

            exact_fuzzy_matches.sort_unstable_by(|a, b| {
                edit_distance(&phonetic, a).cmp(&edit_distance(&phonetic, b))
            });

            for exact in exact_fuzzy_matches {
                if !candidates.contains(&exact) {
                    candidates.push(exact);
                }
            }

            // 12. Dictionary Autocomplete (ONLY if word is incomplete OR space permits)
            if !is_primary_in_dict {
                let prefix_exts = self.database.search_dictionary(primary, 4);
                for ext in prefix_exts {
                    if !candidates.contains(&ext) {
                        candidates.push(ext);
                    }
                }
            } else if candidates.len() < 4 {
                let prefix_exts = self.database.search_dictionary(primary, 2);
                for ext in prefix_exts {
                    if !candidates.contains(&ext) {
                        candidates.push(ext);
                    }
                }
            }
        } else {
            // Contextual Emoji / Symbol Keywords when dictionary is off
            let kw_emojis = self.database.lookup_emoji_keywords(middle);
            for em in kw_emojis {
                if !candidates.contains(&em) {
                    candidates.push(em);
                }
            }
        }

        // 13. Original English Fallback
        if include_english && !candidates.contains(&term.to_string()) {
            candidates.push(term.to_string());
        }

        // Limit candidate list to top 8 most relevant options
        if candidates.len() > 8 {
            let mut trimmed: Vec<String> = Vec::with_capacity(8);
            let has_english = include_english && candidates.iter().any(|c| c == term);
            let target_len = if has_english { 7 } else { 8 };
            for c in candidates {
                if c == term {
                    continue;
                }
                if trimmed.len() < target_len {
                    trimmed.push(c);
                }
            }
            if has_english {
                trimmed.push(term.to_string());
            }
            candidates = trimmed;
        }

        // Apply pre and post punctuation to candidates
        let mut final_candidates = Vec::with_capacity(candidates.len());
        let pre_converted = self.convert_phonetic(pre);
        let post_converted = self.convert_phonetic(post);

        for cand in candidates {
            if pre.is_empty() && post.is_empty() {
                final_candidates.push(cand);
            } else {
                final_candidates.push(format!("{}{}{}", pre_converted, cand, post_converted));
            }
        }

        // Cache the computed candidates for instant sub-millisecond retrieval
        if self.cache.len() > 1000 {
            self.cache.clear();
        }
        self.cache.insert(cache_key, final_candidates.clone());

        // Determine previously selected candidate index
        let selected_index = if let Some(fav) = candidate_memory
            .get(term)
            .or_else(|| candidate_memory.get(middle))
            .or_else(|| candidate_memory.get(&phonetic))
        {
            final_candidates
                .iter()
                .position(|c| c == fav || c.contains(fav))
                .unwrap_or(0)
        } else {
            0
        };

        (final_candidates, selected_index)
    }

    /// Transliterate a full phrase or sentence, resolving phonetics, snippets, symbols, math, macros, and punctuation
    pub fn transliterate_phrase_or_sentence(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        let mut result = String::with_capacity(text.len() * 2);
        let mut current_token = String::new();
        let empty_memory = HashMap::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current_token.is_empty() {
                    let (cands, _) = self.suggest(&current_token, false, true, &empty_memory);
                    if let Some(first) = cands.first() {
                        result.push_str(first);
                    } else {
                        result.push_str(&self.convert_phonetic(&current_token));
                    }
                    current_token.clear();
                }
                result.push(ch);
            } else {
                current_token.push(ch);
            }
        }

        if !current_token.is_empty() {
            let (cands, _) = self.suggest(&current_token, false, true, &empty_memory);
            if let Some(first) = cands.first() {
                result.push_str(first);
            } else {
                result.push_str(&self.convert_phonetic(&current_token));
            }
        }

        result
    }

    fn add_suffixes(&self, middle: &str, phonetic: &str) -> Vec<String> {
        let mut list = Vec::new();

        if middle.len() > 2 {
            for i in 1..middle.len() {
                let suffix_key = &middle[i..];
                if let Some(suffix) = self.database.find_suffix(suffix_key) {
                    let base_key = &middle[..i];
                    let mut base_candidates = Vec::new();
                    if let Some(ac) = self.database.get_autocorrect_raw(base_key) {
                        let conv = self.convert_phonetic(&ac);
                        if !conv.is_empty() {
                            base_candidates.push(conv);
                        }
                    }
                    let base_phonetic = self.convert_phonetic(base_key);
                    if !base_candidates.contains(&base_phonetic) {
                        base_candidates.push(base_phonetic);
                    }

                    for base in &base_candidates {
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
                        if self.database.is_exact_dictionary_word(&word) {
                            if !list.iter().any(|item| item == &word) && word != phonetic {
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
    if input.is_empty() {
        return ("", "", "");
    }

    // Preserve Avro escape sequences like ".`", ":`", ",`", etc.
    if input.ends_with('`') && input.len() == 2 && is_punct_byte(input.as_bytes()[0]) {
        return (input, "", "");
    }

    let bytes = input.as_bytes();
    let mut start = 0;
    let mut end = input.len();

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

    #[test]
    fn test_emoji_and_symbol_suggestions() {
        let mut sugg = PhoneticSuggestion::new();
        let layout_candidates = [
            std::path::Path::new("../../data/layouts/avrophonetic.json"),
            std::path::Path::new("data/layouts/avrophonetic.json"),
            std::path::Path::new("../data/layouts/avrophonetic.json"),
        ];
        for p in layout_candidates {
            if p.exists() {
                if let Ok(content) = std::fs::read_to_string(p) {
                    if let Ok(json) = serde_json::from_str(&content) {
                        sugg.set_layout(&json);
                        break;
                    }
                }
            }
        }
        let empty_memory = HashMap::new();

        // Emoji shortcode :smile:
        let (cands, _) = sugg.suggest(":smile:", true, true, &empty_memory);
        assert!(!cands.is_empty());
        assert_eq!(cands[0], "😊");

        // Bengali emoji :bhalobasha:
        let (cands, _) = sugg.suggest(":bhalobasha:", true, true, &empty_memory);
        assert!(!cands.is_empty());
        assert_eq!(cands[0], "❤️");

        // Taka symbol *taka*
        let (cands, _) = sugg.suggest("*taka*", true, true, &empty_memory);
        assert!(!cands.is_empty());
        assert_eq!(cands[0], "৳");

        // Smiley :)
        let (cands, _) = sugg.suggest(":)", true, true, &empty_memory);
        assert!(!cands.is_empty());
        assert_eq!(cands[0], "😊");

        // Snippet !shubhechha
        let (cands, _) = sugg.suggest("!shubhechha", true, true, &empty_memory);
        assert!(!cands.is_empty());
        assert_eq!(cands[0], "আন্তরিক শুভেচ্ছা ও অভিনন্দন");

        // Prefix emoji search :sm
        let (cands_prefix, _) = sugg.suggest(":sm", true, true, &empty_memory);
        assert!(!cands_prefix.is_empty());
        assert!(cands_prefix.contains(&"😊".to_string()) || cands_prefix.contains(&"😃".to_string()));

        // Prefix symbol search *t
        let (cands_sym, _) = sugg.suggest("*t", true, true, &empty_memory);
        assert!(!cands_sym.is_empty());
        assert_eq!(cands_sym[0], "৳");

        // Inline Math Formula =125*8
        let (cands_math, _) = sugg.suggest("=125*8", true, true, &empty_memory);
        assert!(!cands_math.is_empty());
        assert_eq!(cands_math[0], "১,০০০");
        assert_eq!(cands_math[1], "1,000");

        // Currency Conversion #usd50
        let (cands_curr, _) = sugg.suggest("#usd50", true, true, &empty_memory);
        assert!(!cands_curr.is_empty());
        assert!(cands_curr[0].contains("৳"));

        // Snippet prefix search !sh
        let (cands_snip_prefix, _) = sugg.suggest("!sh", true, true, &empty_memory);
        assert!(!cands_snip_prefix.is_empty());
        assert!(cands_snip_prefix.contains(&"আন্তরিক শুভেচ্ছা ও অভিনন্দন".to_string()));
        assert!(cands_snip_prefix.contains(&"স্বাগতম".to_string()));

        // Bengali symbol aliases *chandrabindu*, *khandata*, $$
        let (cands_chandra, _) = sugg.suggest("*chandrabindu*", true, true, &empty_memory);
        assert_eq!(cands_chandra[0], "ঁ");

        let (cands_khanda, _) = sugg.suggest("*khandata*", true, true, &empty_memory);
        assert_eq!(cands_khanda[0], "ৎ");

        let (cands_dollar, _) = sugg.suggest("$$", true, true, &empty_memory);
        assert_eq!(cands_dollar[0], "৳");

        // Transliterate phrase or sentence
        let res_sentence = sugg.transliterate_phrase_or_sentence("!shubhechha *taka* =125*8");
        assert!(res_sentence.contains("আন্তরিক শুভেচ্ছা ও অভিনন্দন"));
        assert!(res_sentence.contains('৳'));
        assert!(res_sentence.contains("১,০০০"));

        // Common Bengali words & candidate ranking
        let (cands_gari, _) = sugg.suggest("gari", true, true, &empty_memory);
        assert_eq!(cands_gari[0], "গাড়ি");

        let (cands_valo, _) = sugg.suggest("valo", true, true, &empty_memory);
        assert_eq!(cands_valo[0], "ভালো");

        let (cands_sundor, _) = sugg.suggest("sundor", true, true, &empty_memory);
        assert_eq!(cands_sundor[0], "সুন্দর");

        let (cands_tomake, _) = sugg.suggest("tomake", true, true, &empty_memory);
        assert_eq!(cands_tomake[0], "তোমাকে");

        let (cands_bhalo, _) = sugg.suggest("bhalobashi", true, true, &empty_memory);
        assert_eq!(cands_bhalo[0], "ভালোবাসি");

        // Transliterate full natural sentences with emojis
        let res_love = sugg.transliterate_phrase_or_sentence("ami tomake bhalobashi :heart:");
        assert_eq!(res_love, "আমি তোমাকে ভালোবাসি ❤️");

        let res_car = sugg.transliterate_phrase_or_sentence("gari diye bari jabo");
        assert_eq!(res_car, "গাড়ি দিয়ে বাড়ি যাব");
    }

    #[test]
    fn test_candidate_relevance() {
        let mut sugg = PhoneticSuggestion::new();
        let layout_candidates = [
            std::path::Path::new("../../data/layouts/avrophonetic.json"),
            std::path::Path::new("data/layouts/avrophonetic.json"),
            std::path::Path::new("../data/layouts/avrophonetic.json"),
        ];
        for p in layout_candidates {
            if p.exists() {
                if let Ok(content) = std::fs::read_to_string(p) {
                    if let Ok(json) = serde_json::from_str(&content) {
                        sugg.set_layout(&json);
                        break;
                    }
                }
            }
        }
        let empty_memory = HashMap::new();

        // 1. "ami" should produce natural inflections and not dictionary words like "আমিষ" / "আমিন"
        let (cands_ami, _) = sugg.suggest("ami", true, true, &empty_memory);
        assert_eq!(cands_ami[0], "আমি");
        assert!(cands_ami.contains(&"আমাকে".to_string()));
        assert!(cands_ami.contains(&"আমার".to_string()));
        assert!(cands_ami.contains(&"আমাদের".to_string()));
        assert!(!cands_ami.contains(&"আমিষ".to_string()));
        assert!(!cands_ami.contains(&"আমিন".to_string()));

        // 2. "gari" should produce vehicle emoji and inflections
        let (cands_gari, _) = sugg.suggest("gari", true, true, &empty_memory);
        assert_eq!(cands_gari[0], "গাড়ি");
        assert!(cands_gari.contains(&"🚗".to_string()));
        assert!(cands_gari.contains(&"গাড়িতে".to_string()));

        // 3. "taka" should produce Taka symbol
        let (cands_taka, _) = sugg.suggest("taka", true, true, &empty_memory);
        assert_eq!(cands_taka[0], "টাকা");
        assert!(cands_taka.contains(&"৳".to_string()));

        // 4. "shob" should produce "সব", "শব", "সবাই"
        let (cands_shob, _) = sugg.suggest("shob", true, true, &empty_memory);
        assert_eq!(cands_shob[0], "সব");
        assert!(cands_shob.contains(&"শব".to_string()));
        assert!(cands_shob.contains(&"সবাই".to_string()));
    }
}
