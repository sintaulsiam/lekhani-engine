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
    pub ai_context: lekhani_ai::ContextScorer,
    pub ai_predictor: lekhani_ai::NextWordPredictor,
    pub ai_decoder: lekhani_ai::BeamSearchDecoder,
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
            ai_context: lekhani_ai::ContextScorer::new(),
            ai_predictor: lekhani_ai::NextWordPredictor::new(),
            ai_decoder: lekhani_ai::BeamSearchDecoder::new(),
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
            ai_context: lekhani_ai::ContextScorer::new(),
            ai_predictor: lekhani_ai::NextWordPredictor::new(),
            ai_decoder: lekhani_ai::BeamSearchDecoder::new(),
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
        if text.is_ascii() {
            if text.contains('-') && text.len() >= 3 && text.split('-').all(|p| p.len() <= 2) {
                let parts: Vec<String> = text
                    .split('-')
                    .map(|part| match part {
                        "o" => "ও".to_string(),
                        "a" => "আ".to_string(),
                        "i" => "ই".to_string(),
                        "e" => "এ".to_string(),
                        "u" => "উ".to_string(),
                        _ if !part.is_empty() => {
                            if let Some(ref parser) = self.phonetic_parser {
                                parser.convert(part)
                            } else {
                                part.to_string()
                            }
                        }
                        _ => String::new(),
                    })
                    .collect();
                return parts.join("-");
            }
            if let Some(ref parser) = self.phonetic_parser {
                return parser.convert(text);
            }
            return text.to_string();
        }

        // If string contains mixed ASCII and non-ASCII (e.g. "geleই"),
        // segment and convert ASCII chunks while preserving non-ASCII parts
        let mut result = String::with_capacity(text.len() * 2);
        let mut ascii_chunk = String::new();

        for ch in text.chars() {
            if ch.is_ascii() {
                ascii_chunk.push(ch);
            } else {
                if !ascii_chunk.is_empty() {
                    if let Some(ref parser) = self.phonetic_parser {
                        result.push_str(&parser.convert(&ascii_chunk));
                    } else {
                        result.push_str(&ascii_chunk);
                    }
                    ascii_chunk.clear();
                }
                result.push(ch);
            }
        }
        if !ascii_chunk.is_empty() {
            if let Some(ref parser) = self.phonetic_parser {
                result.push_str(&parser.convert(&ascii_chunk));
            } else {
                result.push_str(&ascii_chunk);
            }
        }

        result
    }

    /// Generate ranked candidates for typed term with multi-token preceding context
    pub fn suggest_with_multi_context(
        &mut self,
        term: &str,
        context: &[&str],
        include_english: bool,
        use_dictionary: bool,
        candidate_memory: &HashMap<String, String>,
    ) -> (Vec<String>, usize) {
        if term.is_empty() {
            return (Vec::new(), 0);
        }

        // Fast Memoization Cache check
        let cache_key = format!(
            "{}:{}:{}:{}",
            term,
            context.join(" "),
            include_english,
            use_dictionary
        );
        if let Some(cached) = self.cache.get(&cache_key) {
            let selected_index = if let Some(fav) = candidate_memory
                .get(term)
                .or_else(|| candidate_memory.get(&cache_key))
            {
                cached
                    .iter()
                    .position(|c| c == fav || c.contains(fav))
                    .unwrap_or(0)
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
            "o" | "O" => {
                let mut cands = vec!["ও".to_string(), "অ".to_string()];
                if include_english {
                    cands.push(term.to_string());
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

        // 4. Bilingual Loanword Code-Mixing (e.g. "meeting" -> "মিটিং", "meeting")
        let loan_opt = PhoneticDatabase::get_bilingual_loanword(middle).or_else(|| {
            if middle.contains('-') && middle.len() >= 3 && !middle.split('-').all(|p| p.len() <= 1)
            {
                let unhyphenated = middle.replace('-', "");
                PhoneticDatabase::get_bilingual_loanword(&unhyphenated)
            } else {
                None
            }
        });
        if let Some((bn_loan, en_loan)) = loan_opt {
            candidates.push(bn_loan.to_string());
            if include_english {
                candidates.push(en_loan.to_string());
            }
        }

        // 5. Special literal matches on middle portion (e.g. (:smile:), (=12+5))
        let middle_literals = self.database.search_special_literals(middle);
        for special in middle_literals {
            if !candidates.contains(&special) {
                candidates.push(special);
            }
        }

        // 6. Autocorrect / Common Overrides / Elongation Collapse
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
            let mut found_collapsed = None;
            for collapsed in super::fuzzy::collapse_elongated_runs(middle) {
                if let Some(raw_ac) = self.database.get_autocorrect_raw(&collapsed) {
                    let converted_ac = self.convert_phonetic(&raw_ac);
                    if !converted_ac.is_empty() {
                        if !candidates.contains(&converted_ac) {
                            candidates.push(converted_ac.clone());
                        }
                        found_collapsed = Some(converted_ac);
                        break;
                    }
                }
                let direct_collapsed = self.convert_phonetic(&collapsed);
                if self.database.is_exact_dictionary_word(&direct_collapsed) {
                    if !candidates.contains(&direct_collapsed) {
                        candidates.push(direct_collapsed.clone());
                    }
                    found_collapsed = Some(direct_collapsed);
                    break;
                }
            }
            found_collapsed
        };

        let primary = preferred_word.as_deref().unwrap_or(&phonetic);
        let is_primary_in_dict = self.database.is_exact_dictionary_word(primary);

        // 7. Direct Transliterations
        if !candidates.contains(&primary.to_string()) {
            candidates.push(primary.to_string());
        }
        if primary != phonetic && !candidates.contains(&phonetic) {
            candidates.push(phonetic.clone());
        }

        if use_dictionary {
            // 8. Natural Linguistic & Grammatical Inflections (for frequent words - take top 3)
            if let Some(inflections) = PhoneticDatabase::get_common_inflections(middle) {
                for &inf in inflections.iter().take(3) {
                    let inf_str = inf.to_string();
                    if !candidates.contains(&inf_str) {
                        candidates.push(inf_str);
                    }
                }
            }

            // 9. Contextual Emoji / Symbol Keywords (e.g. bhalobasha -> ❤️, cha -> ☕, gari -> 🚗, taka -> ৳)
            let kw_emojis = self.database.lookup_emoji_keywords(middle);
            for em in kw_emojis {
                if !candidates.contains(&em) {
                    candidates.push(em);
                }
            }

            // 10. Remaining inflections if any
            if let Some(inflections) = PhoneticDatabase::get_common_inflections(middle) {
                for &inf in inflections.iter().skip(3) {
                    let inf_str = inf.to_string();
                    if !candidates.contains(&inf_str) {
                        candidates.push(inf_str);
                    }
                }
            }

            // 11. Suffix-decomposed forms (when Latin string contains stem + suffix)
            let suffixed_matches = self.add_suffixes(middle, primary);
            for item in suffixed_matches {
                if !candidates.contains(&item) {
                    candidates.push(item);
                }
            }

            // 12. Exact Fuzzy Spelling Variants (Homophones / Orthographic variants & Juktoborno)
            let fuzzy_variants = super::fuzzy::generate_phonetic_variants(middle);
            let mut exact_fuzzy_matches: Vec<String> = Vec::new();

            for variant in &fuzzy_variants {
                let var_phonetic = self.convert_phonetic(variant);
                if self.database.is_exact_dictionary_word(&var_phonetic) {
                    let len_diff = (var_phonetic.chars().count() as isize
                        - phonetic.chars().count() as isize)
                        .abs();
                    if len_diff <= 3
                        && !exact_fuzzy_matches.contains(&var_phonetic)
                        && var_phonetic != primary
                        && var_phonetic != phonetic
                    {
                        exact_fuzzy_matches.push(var_phonetic);
                    }
                }
            }

            if middle.contains('-') && middle.len() >= 3 && !middle.split('-').all(|p| p.len() <= 1)
            {
                let unhyphenated = middle.replace('-', "");
                let unhyphenated_conv = self.convert_phonetic(&unhyphenated);
                if self.database.is_exact_dictionary_word(&unhyphenated_conv)
                    && !exact_fuzzy_matches.contains(&unhyphenated_conv)
                {
                    exact_fuzzy_matches.push(unhyphenated_conv);
                }
                for fz in super::fuzzy::generate_phonetic_variants(&unhyphenated) {
                    let fz_conv = self.convert_phonetic(&fz);
                    if self.database.is_exact_dictionary_word(&fz_conv)
                        && !exact_fuzzy_matches.contains(&fz_conv)
                    {
                        exact_fuzzy_matches.push(fz_conv);
                    }
                }
            }

            exact_fuzzy_matches.sort_unstable_by(|a, b| {
                let freq_a = self.database.get_frequency(a);
                let freq_b = self.database.get_frequency(b);
                if freq_a != freq_b {
                    freq_b.cmp(&freq_a)
                } else {
                    edit_distance(&phonetic, a).cmp(&edit_distance(&phonetic, b))
                }
            });

            for exact in exact_fuzzy_matches {
                if !candidates.contains(&exact) {
                    candidates.push(exact);
                }
            }

            // 13. Dictionary Autocomplete (ONLY if word is incomplete OR space permits)
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

            // 14. Single-Edit Typo Tolerance Fallback (for fat-finger typos when 0 dictionary matches found)
            if !is_primary_in_dict && candidates.len() <= 2 && middle.chars().count() >= 4 {
                let mut typo_matches: Vec<String> = Vec::new();
                for (i, ch) in middle.char_indices() {
                    let mut del = String::with_capacity(middle.len());
                    del.push_str(&middle[..i]);
                    del.push_str(&middle[i + ch.len_utf8()..]);
                    let del_conv = self.convert_phonetic(&del);
                    if self.database.is_exact_dictionary_word(&del_conv)
                        && !candidates.contains(&del_conv)
                    {
                        typo_matches.push(del_conv);
                    }
                }
                typo_matches
                    .sort_unstable_by_key(|w| std::cmp::Reverse(self.database.get_frequency(w)));
                for tm in typo_matches.into_iter().take(2) {
                    if !candidates.contains(&tm) {
                        candidates.push(tm);
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

        // 14. Original English Fallback
        if include_english && !candidates.contains(&term.to_string()) {
            candidates.push(term.to_string());
        }

        // 15. Contextual Homophone & Deep AI Multi-Token Re-Ranking
        if !context.is_empty() {
            let prev = context.last().copied();
            // Check if any candidate has a strong homophone boost
            let mut best_boost = 0;
            let mut best_idx = None;
            for (idx, cand) in candidates.iter().enumerate() {
                let boost = Self::score_contextual_homophone(prev, cand);
                if boost > best_boost {
                    best_boost = boost;
                    best_idx = Some(idx);
                }
            }
            if let Some(idx) = best_idx {
                if idx > 0 && best_boost >= 500 {
                    let boosted = candidates.remove(idx);
                    candidates.insert(0, boosted);
                }
            } else if use_dictionary && candidates.len() > 1 {
                candidates = self.ai_context.rank_candidates(context, &candidates);
            }
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

    /// Backwards compatible suggest calling suggest_with_context without previous word
    pub fn suggest(
        &mut self,
        term: &str,
        include_english: bool,
        use_dictionary: bool,
        candidate_memory: &HashMap<String, String>,
    ) -> (Vec<String>, usize) {
        self.suggest_with_multi_context(
            term,
            &[],
            include_english,
            use_dictionary,
            candidate_memory,
        )
    }

    /// Convenience forwarder for single previous word context
    pub fn suggest_with_context(
        &mut self,
        term: &str,
        previous_word: Option<&str>,
        include_english: bool,
        use_dictionary: bool,
        candidate_memory: &HashMap<String, String>,
    ) -> (Vec<String>, usize) {
        if let Some(p) = previous_word {
            self.suggest_with_multi_context(
                term,
                &[p],
                include_english,
                use_dictionary,
                candidate_memory,
            )
        } else {
            self.suggest_with_multi_context(
                term,
                &[],
                include_english,
                use_dictionary,
                candidate_memory,
            )
        }
    }

    /// Score homophone pairs based on semantic preceding context
    pub fn score_contextual_homophone(previous_word: Option<&str>, candidate: &str) -> i32 {
        let prev = match previous_word {
            Some(p) if !p.trim().is_empty() => p.trim(),
            _ => return 0,
        };

        let rule_score = match candidate {
            "পড়া" | "পড়ছি" | "পড়ব" | "পড়াশোনা" => {
                if matches!(
                    prev,
                    "বই" | "বইটি"
                        | "বইয়ের"
                        | "বইগুলো"
                        | "পত্রিকা"
                        | "লেখা"
                        | "ক্লাস"
                        | "স্কুল"
                        | "কলেজ"
                        | "পরীক্ষা"
                        | "পাঠ"
                        | "মন"
                        | "নোট"
                ) {
                    1000
                } else {
                    0
                }
            }
            "পরা" | "পরছি" | "পরব" => {
                if matches!(
                    prev,
                    "শার্ট"
                        | "প্যান্ট"
                        | "জামা"
                        | "কাপড়"
                        | "জুতো"
                        | "জুতা"
                        | "ঘড়ি"
                        | "চশমা"
                        | "পোশাক"
                        | "শাল"
                        | "শাড়ি"
                ) {
                    1000
                } else {
                    0
                }
            }
            "ভাষা" | "ভাষায়" | "ভাষার" => {
                if matches!(
                    prev,
                    "বাংলা"
                        | "ইংরেজি"
                        | "মাতৃভাষা"
                        | "কথ্য"
                        | "রাষ্ট্র"
                        | "আমাদের"
                        | "সুন্দর"
                        | "আন্তর্জাতিক"
                ) {
                    1000
                } else {
                    0
                }
            }
            "ভাসা" | "ভাসছে" => {
                if matches!(
                    prev,
                    "পানিতে"
                        | "জলে"
                        | "নদীতে"
                        | "সাগরে"
                        | "ভেসে"
                        | "রক্তে"
                        | "চোখের"
                ) {
                    1000
                } else {
                    0
                }
            }
            "জাতি" | "জাতির" => {
                if matches!(
                    prev,
                    "বাঙালি"
                        | "মুসলিম"
                        | "হিন্দু"
                        | "উন্নত"
                        | "মানব"
                        | "বিশ্ব"
                        | "পুরো"
                ) {
                    1000
                } else {
                    0
                }
            }
            "কুল" => {
                if matches!(prev, "বংশ" | "উচ্চ" | "মান" | "মর্যাদা")
                {
                    1000
                } else {
                    0
                }
            }
            "কূল" => {
                if matches!(prev, "নদী" | "নদীর" | "সাগর" | "সাগরের" | "উপকূল" | "তীর")
                {
                    1000
                } else {
                    0
                }
            }
            "লক্ষ্য" => {
                if matches!(
                    prev,
                    "জীবনের"
                        | "মূল"
                        | "প্রধান"
                        | "উদ্দেশ্য"
                        | "আমাদের"
                        | "চূড়ান্ত"
                ) {
                    1000
                } else {
                    0
                }
            }
            "লক্ষ" => {
                if matches!(
                    prev,
                    "এক" | "দুই"
                        | "তিন"
                        | "চার"
                        | "পাঁচ"
                        | "দশ"
                        | "কোটি"
                        | "টাকা"
                        | "মানুষ"
                ) {
                    1000
                } else {
                    0
                }
            }
            "কাঁচা" => {
                if matches!(prev, "আম" | "ফল" | "মরিচ" | "রাস্তা" | "টাকা" | "বয়স")
                {
                    1000
                } else {
                    0
                }
            }
            "কাচা" => {
                if matches!(prev, "কাপড়" | "জামা" | "ধোয়া") {
                    1000
                } else {
                    0
                }
            }
            "স্বত্ব" => {
                if matches!(prev, "কপিরাইট" | "মালিকানা" | "গ্রন্থ" | "প্রকাশক")
                {
                    1000
                } else {
                    0
                }
            }
            "সত্য" => {
                if matches!(prev, "বলা" | "চিরন্তন" | "কথা" | "সবসময়" | "পরম" | "প্রকৃত")
                {
                    1000
                } else {
                    0
                }
            }
            "দিন" => {
                if matches!(
                    prev,
                    "আজকের"
                        | "শুভ"
                        | "প্রতি"
                        | "সারাদিন"
                        | "কয়েক"
                        | "ভালো"
                        | "খারাপ"
                ) {
                    500
                } else if matches!(
                    prev,
                    "আমাকে"
                        | "তাকে"
                        | "একটু"
                        | "দয়া"
                        | "টাকা"
                        | "বইটি"
                        | "করে"
                ) {
                    800
                } else {
                    0
                }
            }
            _ => 0,
        };

        if rule_score > 0 {
            rule_score
        } else {
            let score =
                lekhani_ai::LanguageModel::new().score_candidate(None, Some(prev), candidate);
            if score > -1.0 {
                1000
            } else if score > -2.0 {
                500
            } else {
                0
            }
        }
    }

    /// Predict next probable Bengali words for zero-preedit state
    pub fn suggest_next_words(&self, previous_word: &str) -> Vec<String> {
        let prev = previous_word.trim();
        if prev.is_empty() {
            return Vec::new();
        }

        let words: &[&str] = match prev {
            "আমি" => &["ভালো", "তোমাকে", "যাব", "করব", "চাই", "আছি", "এখন", "বলছি"],
            "তুমি" => &["কেমন", "কোথায়", "কী", "কবে", "যাবে", "খাবে", "আছো", "বলো"],
            "আপনি" => &["কেমন", "কোথায়", "কী", "কবে", "যাবেন", "আছেন", "বলুন"],
            "আমরা" => &["সবাই", "একসাথে", "যাব", "করব", "চাই", "আছি", "বাংলাদেশী"],
            "ধন্যবাদ" => &["ভাই", "আপনাকে", "তোমাকে", "অনেক", "স্যার", "জানাই"],
            "শুভ" => &["সকাল", "রাত্রি", "সন্ধ্যা", "কামনা", "নববর্ষ", "জন্মদিন", "বিকেল"],
            "অনেক" => &["ধন্যবাদ", "ভালো", "সুন্দর", "দিন", "টাকা", "মানুষ", "কষ্ট"],
            "খুব" => &["ভালো", "সুন্দর", "খারাপ", "কঠিন", "সহজ", "তাড়াতাড়ি", "দ্রুত"],
            "কেমন" => &["আছো", "আছেন", "হলো", "লাগল", "লাগে", "চলছে"],
            "ভালো" => &["আছি", "থাকবেন", "থাকো", "বাসি", "লাগে", "লাগল", "হবে"],
            "বাংলাদেশ" => &["আমার", "একটি", "জিন্দাবাদ", "ক্রিকেট", "সরকার"],
            "বই" => &["পড়া", "পড়ছি", "পড়ব", "মেলা", "কিনেছি"],
            "চা" => &["খাবেন", "খাব", "বানাও", "পান", "গরম"],
            "কি" | "কী" => &["খবর", "করছ", "করছেন", "হয়েছে", "হলো", "চাও", "চান"],
            "কোথায়" => &["আছো", "আছেন", "যাবে", "যাবেন", "গেলে"],
            "কেন" => &["এমন", "করছ", "করছেন", "হলো", "গেলে"],
            "ইনশাআল্লাহ" => &["হবে", "যাব", "দেখা", "ভালো"],
            "মাশাল্লাহ" => &["অনেক", "সুন্দর", "খুব"],
            "আলহামদুলিল্লাহ" => &["ভালো", "আমি", "সব"],
            "আমার" => &["দেশ", "সোনার", "বন্ধু", "নাম", "জীবন", "মন"],
            "তোমার" => &["নাম", "বাড়ি", "খবর", "কথা", "মন"],
            "আপনার" => &["নাম", "অফিস", "খবর", "দয়া", "কথা"],
            "আজ" => &["সকালে", "রাতে", "বৃষ্টি", "ছুটি", "কেমন"],
            "কাল" => &["দেখা", "হবে", "যাব", "আসবে"],
            _ => &[],
        };

        let mut list: Vec<String> = words.iter().map(|s| s.to_string()).collect();

        // Query AI neural transition language model for additional predictions
        for pred in self.ai_predictor.predict_next(&[prev], 6) {
            if !list.contains(&pred) {
                list.push(pred);
            }
        }

        list
    }

    /// Transliterate a full phrase or sentence using global AI Beam Search sequence decoding
    pub fn transliterate_phrase_or_sentence(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        // Tokenize by word while preserving delimiters
        let mut tokens = Vec::new();
        let mut cur_word = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !cur_word.is_empty() {
                    tokens.push((true, cur_word.clone()));
                    cur_word.clear();
                }
                tokens.push((false, ch.to_string()));
            } else {
                cur_word.push(ch);
            }
        }
        if !cur_word.is_empty() {
            tokens.push((true, cur_word));
        }

        let empty_memory = HashMap::new();
        let mut word_cands: Vec<Vec<String>> = Vec::new();
        let mut word_indices: Vec<usize> = Vec::new();

        for (idx, (is_word, tok)) in tokens.iter().enumerate() {
            if *is_word {
                let (cands, _) =
                    self.suggest_with_multi_context(tok, &[], false, true, &empty_memory);
                if !cands.is_empty() {
                    word_cands.push(cands);
                } else {
                    word_cands.push(vec![self.convert_phonetic(tok)]);
                }
                word_indices.push(idx);
            }
        }

        let optimal_path = if !word_cands.is_empty() {
            self.ai_decoder.decode(&word_cands)
        } else {
            Vec::new()
        };

        let mut result = String::with_capacity(text.len() * 2);
        let mut opt_idx = 0;

        for (is_word, tok) in tokens {
            if is_word {
                if let Some(decoded) = optimal_path.get(opt_idx) {
                    result.push_str(decoded);
                } else {
                    result.push_str(&self.convert_phonetic(&tok));
                }
                opt_idx += 1;
            } else {
                result.push_str(&tok);
            }
        }

        result
    }

    fn add_suffixes(&self, middle: &str, phonetic: &str) -> Vec<String> {
        let mut list = Vec::new();

        if middle.chars().count() > 2 {
            for (i, _) in middle.char_indices().skip(1) {
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
                        if let (Some(base_rmc), Some(suffix_lmc)) =
                            (base.chars().last(), suffix.chars().next())
                        {
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
                        if (self.database.is_exact_dictionary_word(&word)
                            || self.database.is_exact_dictionary_word(base))
                            && !list.iter().any(|item| item == &word)
                            && word != phonetic
                        {
                            list.push(word);
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
        assert!(
            cands_prefix.contains(&"😊".to_string()) || cands_prefix.contains(&"😃".to_string())
        );

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
        assert!(cands_gari.contains(&"গাড়ি".to_string()));
        let (cands_ga_ri, _) = sugg.suggest("gaRi", true, true, &empty_memory);
        assert_eq!(cands_ga_ri[0], "গাড়ি");

        let (cands_valo, _) = sugg.suggest("valo", true, true, &empty_memory);
        assert!(cands_valo.contains(&"ভালো".to_string()));
        let (cands_val_o, _) = sugg.suggest("valO", true, true, &empty_memory);
        assert_eq!(cands_val_o[0], "ভালো");

        let (cands_sundor, _) = sugg.suggest("sundor", true, true, &empty_memory);
        assert_eq!(cands_sundor[0], "সুন্দর");

        let (cands_tomake, _) = sugg.suggest("tomake", true, true, &empty_memory);
        assert!(cands_tomake.contains(&"তোমাকে".to_string()));
        let (cands_t_o_make, _) = sugg.suggest("tOmake", true, true, &empty_memory);
        assert_eq!(cands_t_o_make[0], "তোমাকে");

        let (cands_bhalo, _) = sugg.suggest("bhalobashi", true, true, &empty_memory);
        assert!(cands_bhalo.contains(&"ভালোবাসি".to_string()));
        let (cands_bhal_o, _) = sugg.suggest("bhalObasi", true, true, &empty_memory);
        assert_eq!(cands_bhal_o[0], "ভালোবাসি");

        // Transliterate full natural sentences with emojis
        let res_love = sugg.transliterate_phrase_or_sentence("ami tomake bhalobashi :heart:");
        assert_eq!(res_love, "আমি তোমাকে ভালোবাসি ❤️");

        let res_car = sugg.transliterate_phrase_or_sentence("gaRi diye bari jabo");
        assert!(res_car.contains("গাড়ি"));
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
        assert!(cands_gari.contains(&"গাড়ি".to_string()));
        assert!(cands_gari.contains(&"🚗".to_string()));
        assert!(
            cands_gari.contains(&"গাড়িতে".to_string()) || cands_gari.contains(&"গাড়ির".to_string())
        );

        // 3. "taka" should suggest Taka symbol and "টাকা"
        let (cands_taka, _) = sugg.suggest("taka", true, true, &empty_memory);
        assert!(cands_taka.contains(&"টাকা".to_string()));
        assert!(cands_taka.contains(&"৳".to_string()));
        let (cands_ta_ka, _) = sugg.suggest("Taka", true, true, &empty_memory);
        assert_eq!(cands_ta_ka[0], "টাকা");

        // 4. "shob" should produce "সব", "শব", "সবাই"
        let (cands_shob, _) = sugg.suggest("shob", true, true, &empty_memory);
        assert!(cands_shob.contains(&"সব".to_string()));
        assert!(cands_shob.contains(&"শব".to_string()));
        assert!(cands_shob.contains(&"সবাই".to_string()));
    }

    #[test]
    fn test_contextual_homophones() {
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

        // When previous word is "বই", "pora" should rank "পড়া" first
        let (cands_book, _) =
            sugg.suggest_with_context("pora", Some("বই"), true, true, &empty_memory);
        assert_eq!(cands_book[0], "পড়া");

        // When previous word is "শার্ট", "pora" should rank "পরা" first
        let (cands_shirt, _) =
            sugg.suggest_with_context("pora", Some("শার্ট"), true, true, &empty_memory);
        assert_eq!(cands_shirt[0], "পরা");

        // When previous word is "বাংলা", "bhasha" ranks "ভাষা"
        let (cands_lang, _) =
            sugg.suggest_with_context("bhasha", Some("বাংলা"), true, true, &empty_memory);
        assert_eq!(cands_lang[0], "ভাষা");
    }

    #[test]
    fn test_next_word_predictions() {
        let sugg = PhoneticSuggestion::new();
        let next_ami = sugg.suggest_next_words("আমি");
        assert!(!next_ami.is_empty());
        assert!(next_ami.contains(&"ভালো".to_string()));
        assert!(next_ami.contains(&"তোমাকে".to_string()));

        let next_thanks = sugg.suggest_next_words("ধন্যবাদ");
        assert!(!next_thanks.is_empty());
        assert!(
            next_thanks.contains(&"ভাই".to_string()) || next_thanks.contains(&"আপনাকে".to_string())
        );
    }

    #[test]
    fn test_bilingual_loanwords() {
        let mut sugg = PhoneticSuggestion::new();
        let empty_memory = HashMap::new();

        let (cands_meet, _) = sugg.suggest("meeting", true, true, &empty_memory);
        assert!(!cands_meet.is_empty());
        assert!(cands_meet.contains(&"মিটিং".to_string()));
        assert!(cands_meet.contains(&"meeting".to_string()));
    }

    #[test]
    fn test_multi_token_context_and_beam_transliteration() {
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

        // 1. Multi-token context suggestion
        let (cands_multi, _) =
            sugg.suggest_with_multi_context("khacchi", &["আমি", "ভাত"], true, true, &empty_memory);
        assert!(!cands_multi.is_empty());
        assert_eq!(cands_multi[0], "খাচ্ছি");

        // 2. Full sentence beam search decoding
        let sentence = sugg.transliterate_phrase_or_sentence("ami banglay gaan gai");
        assert!(sentence.contains("বাংলা") || sentence.contains("বাংলায়"));
        assert!(sentence.contains("গান"));
        assert!(sentence.contains("গাই"));
    }

    #[test]
    fn test_juktoborno_casual_typing() {
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

        // 1. "kosto" produces "কস্ত" as literal and suggests "কষ্ট"
        let (cands_kosto, _) = sugg.suggest("kosto", true, true, &empty_memory);
        assert!(!cands_kosto.is_empty());
        assert_eq!(cands_kosto[0], "কস্ত");
        assert!(cands_kosto.contains(&"কষ্ট".to_string()));

        let (cands_koshto, _) = sugg.suggest("koshto", true, true, &empty_memory);
        assert!(cands_koshto.contains(&"কষ্ট".to_string()));
        let (cands_ko_sh_to, _) = sugg.suggest("koShTo", true, true, &empty_memory);
        assert_eq!(cands_ko_sh_to[0], "কষ্ট");

        // 2. "nosto" produces "নস্ত" as literal and suggests "নষ্ট"
        let (cands_nosto, _) = sugg.suggest("nosto", true, true, &empty_memory);
        assert!(!cands_nosto.is_empty());
        assert_eq!(cands_nosto[0], "নস্ত");
        assert!(cands_nosto.contains(&"নষ্ট".to_string()));

        let (cands_noshto, _) = sugg.suggest("noshto", true, true, &empty_memory);
        assert!(cands_noshto.contains(&"নষ্ট".to_string()));
        let (cands_no_sh_to, _) = sugg.suggest("noShTo", true, true, &empty_memory);
        assert_eq!(cands_no_sh_to[0], "নষ্ট");

        // 3. "biganni" and "biggani" suggest "বিজ্ঞানী"
        let (cands_biganni, _) = sugg.suggest("biganni", true, true, &empty_memory);
        assert!(cands_biganni.contains(&"বিজ্ঞানী".to_string()));

        let (cands_biggani, _) = sugg.suggest("biggani", true, true, &empty_memory);
        assert!(cands_biggani.contains(&"বিজ্ঞানী".to_string()));

        // 4. "biggan" suggests "বিজ্ঞান"
        let (cands_biggan, _) = sugg.suggest("biggan", true, true, &empty_memory);
        assert!(cands_biggan.contains(&"বিজ্ঞান".to_string()));

        // 5. "sristi" suggests "সৃষ্টি"
        let (cands_sristi, _) = sugg.suggest("sristi", true, true, &empty_memory);
        assert!(cands_sristi.contains(&"সৃষ্টি".to_string()));

        let (cands_srish_ti, _) = sugg.suggest("srriShTi", true, true, &empty_memory);
        assert_eq!(cands_srish_ti[0], "সৃষ্টি");

        // 6. "bristi" suggests "বৃষ্টি"
        let (cands_bristi, _) = sugg.suggest("bristi", true, true, &empty_memory);
        assert!(cands_bristi.contains(&"বৃষ্টি".to_string()));

        // 7. "bebostha" suggests "ব্যবস্থা"
        let (cands_bebostha, _) = sugg.suggest("bebostha", true, true, &empty_memory);
        assert!(cands_bebostha.contains(&"ব্যবস্থা".to_string()));

        // 8. "onusthan" suggests "অনুষ্ঠান"
        let (cands_onusthan, _) = sugg.suggest("onusthan", true, true, &empty_memory);
        assert!(cands_onusthan.contains(&"অনুষ্ঠান".to_string()));
    }

    #[test]
    fn test_natural_mistake_tolerance() {
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

        // 1. Elongation collapse suggestions
        let (cands_thik, _) = sugg.suggest("thiiik", true, true, &empty_memory);
        assert!(!cands_thik.is_empty());
        assert!(cands_thik.contains(&"ঠিক".to_string()));

        let (cands_bhalo, _) = sugg.suggest("bhalooo", true, true, &empty_memory);
        assert!(!cands_bhalo.is_empty());
        assert!(cands_bhalo.contains(&"ভালো".to_string()));

        let (cands_na, _) = sugg.suggest("naaa", true, true, &empty_memory);
        assert!(!cands_na.is_empty());
        assert!(cands_na.contains(&"না".to_string()));

        // 2. Vai literal transliteration
        let (cands_vai, _) = sugg.suggest("vai", true, true, &empty_memory);
        assert_eq!(cands_vai[0], "ভাই");

        // 3. Chandrabindu natural spelling suggestions
        let (cands_chad, _) = sugg.suggest("chad", true, true, &empty_memory);
        assert_eq!(cands_chad[0], "ছাদ");
        assert!(cands_chad.contains(&"চাঁদ".to_string()));

        let (cands_cad, _) = sugg.suggest("cad", true, true, &empty_memory);
        assert_eq!(cands_cad[0], "চাদ");
        assert!(cands_cad.contains(&"চাঁদ".to_string()));

        let (cands_c_ad, _) = sugg.suggest("ca^d", true, true, &empty_memory);
        assert_eq!(cands_c_ad[0], "চাঁদ");

        let (cands_dat, _) = sugg.suggest("dat", true, true, &empty_memory);
        assert_eq!(cands_dat[0], "দাত");
        assert!(cands_dat.contains(&"দাঁত".to_string()));

        let (cands_has, _) = sugg.suggest("has", true, true, &empty_memory);
        assert_eq!(cands_has[0], "হাস");
        assert!(cands_has.contains(&"হাঁস".to_string()));

        let (cands_bas, _) = sugg.suggest("bas", true, true, &empty_memory);
        assert_eq!(cands_bas[0], "বাস");
        assert!(cands_bas.contains(&"বাঁশ".to_string()));

        let (cands_pach, _) = sugg.suggest("pach", true, true, &empty_memory);
        assert!(cands_pach.contains(&"পাঁচ".to_string()));

        // 4. Khanda-Ta suggestions
        let (cands_hothat, _) = sugg.suggest("hothat", true, true, &empty_memory);
        assert!(cands_hothat.contains(&"হঠাৎ".to_string()));

        let (cands_utsob, _) = sugg.suggest("utsob", true, true, &empty_memory);
        assert!(cands_utsob.contains(&"উৎসব".to_string()));

        let (cands_utsaho, _) = sugg.suggest("utsaho", true, true, &empty_memory);
        assert!(cands_utsaho.contains(&"উৎসাহ".to_string()));

        // 5. Bengali Glide Verbs
        let (cands_khawa, _) = sugg.suggest("khawa", true, true, &empty_memory);
        assert!(cands_khawa.contains(&"খাওয়া".to_string()));

        let (cands_dewa, _) = sugg.suggest("dewa", true, true, &empty_memory);
        assert!(cands_dewa.contains(&"দেওয়া".to_string()));

        let (cands_jawa, _) = sugg.suggest("jawa", true, true, &empty_memory);
        assert!(cands_jawa.contains(&"যাওয়া".to_string()));

        let (cands_pawa, _) = sugg.suggest("pawa", true, true, &empty_memory);
        assert!(cands_pawa.contains(&"পাওয়া".to_string()));
    }

    #[test]
    fn test_complex_word_ergonomics_and_standards() {
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

        // 1. Reph Consonants
        let (cands_bortoman, _) = sugg.suggest("bortoman", true, true, &empty_memory);
        assert!(cands_bortoman.contains(&"বর্তমান".to_string()));

        let (cands_orthonoitik, _) = sugg.suggest("orthonoitik", true, true, &empty_memory);
        assert!(cands_orthonoitik.contains(&"অর্থনৈতিক".to_string()));

        let (cands_karjokrom, _) = sugg.suggest("karjokrom", true, true, &empty_memory);
        assert!(cands_karjokrom.contains(&"কার্যক্রম".to_string()));

        let (cands_durniti, _) = sugg.suggest("durniti", true, true, &empty_memory);
        assert!(cands_durniti.contains(&"দুর্নীতি".to_string()));

        // 2. Conjunction 'o'
        let (cands_o, _) = sugg.suggest("o", true, true, &empty_memory);
        assert_eq!(cands_o[0], "ও");

        // 3. Ja-fala & Geminates
        let (cands_tottho, _) = sugg.suggest("tottho", true, true, &empty_memory);
        assert!(cands_tottho.contains(&"তথ্য".to_string()));

        let (cands_biddaloy, _) = sugg.suggest("biddaloy", true, true, &empty_memory);
        assert!(
            cands_biddaloy.contains(&"বিদ্যালয়".to_string())
                || cands_biddaloy.contains(&"বিদ্যালয়".to_string())
        );

        let (cands_jonne, _) = sugg.suggest("jonne", true, true, &empty_memory);
        assert!(
            cands_jonne.contains(&"জন্য".to_string()) || cands_jonne.contains(&"জন্যে".to_string())
        );

        // 4. Motion Verbs & Antastha-Ja
        let (cands_jacchi, _) = sugg.suggest("jacchi", true, true, &empty_memory);
        assert!(cands_jacchi.contains(&"যাচ্ছি".to_string()));

        let (cands_juddho, _) = sugg.suggest("juddho", true, true, &empty_memory);
        assert!(cands_juddho.contains(&"যুদ্ধ".to_string()));

        let (cands_projukti, _) = sugg.suggest("projukti", true, true, &empty_memory);
        assert!(cands_projukti.contains(&"প্রযুক্তি".to_string()));

        // 5. Verb Inflections
        let (cands_korchhen, _) = sugg.suggest("korchhen", true, true, &empty_memory);
        assert!(
            cands_korchhen.contains(&"করছেন".to_string())
                || cands_korchhen.contains(&"করছেন".to_string())
        );

        // 6. Sibilants
        let (cands_shadhinota, _) = sugg.suggest("shadhinota", true, true, &empty_memory);
        assert!(cands_shadhinota.contains(&"স্বাধীনতা".to_string()));

        let (cands_shorkari, _) = sugg.suggest("shorkari", true, true, &empty_memory);
        assert!(cands_shorkari.contains(&"সরকারি".to_string()));

        let (cands_shobai, _) = sugg.suggest("shobai", true, true, &empty_memory);
        assert!(cands_shobai.contains(&"সবাই".to_string()));

        // 7. Ri-kar & Long Vowels
        let (cands_kritrim, _) = sugg.suggest("kritrim", true, true, &empty_memory);
        assert!(cands_kritrim.contains(&"কৃত্রিম".to_string()));

        let (cands_matribhumi, _) = sugg.suggest("matribhumi", true, true, &empty_memory);
        assert!(cands_matribhumi.contains(&"মাতৃভূমি".to_string()));

        // 8. Morphological Suffixes
        let (cands_boita, _) = sugg.suggest("boita", true, true, &empty_memory);
        assert!(cands_boita.contains(&"বইটা".to_string()));

        let (cands_kajta, _) = sugg.suggest("kajta", true, true, &empty_memory);
        assert!(cands_kajta.contains(&"কাজটা".to_string()));
    }
}
