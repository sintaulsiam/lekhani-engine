//! Phonetic Suggestion Generation Engine

use edit_distance::edit_distance;
use hashbrown::HashMap;
use rupantor::parser::PhoneticParser;
use serde_json::Value;

use super::database::PhoneticDatabase;
use crate::chars::BengaliCharExt;

use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateSource {
    Autocorrect,
    Loanword,
    ExactDictionary,
    FuzzySoundLaw,
    MorphologicalInflection,
    TriePrefixAutocomplete,
    DirectTransliteration,
    TypoFallback,
    EmojiKeyword,
}

#[derive(Debug, Clone)]
pub struct CandidateHypothesis {
    pub text: String,
    pub source: CandidateSource,
    pub initial_boost: i32,
}

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
        let mut raw_candidates: Vec<CandidateHypothesis> = Vec::with_capacity(32);
        let mut seen = hashbrown::HashSet::new();

        let add_cand = |text: String,
                        source: CandidateSource,
                        initial_boost: i32,
                        raw_candidates: &mut Vec<CandidateHypothesis>,
                        seen: &mut hashbrown::HashSet<String>| {
            if !text.is_empty() && seen.insert(text.clone()) {
                raw_candidates.push(CandidateHypothesis {
                    text,
                    source,
                    initial_boost,
                });
            }
        };

        // 4. Bilingual Loanword Code-Mixing (e.g. "meeting" -> "মিটিং", "meeting")
        let mut preferred_loanword = None;
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
            add_cand(
                bn_loan.to_string(),
                CandidateSource::Loanword,
                6500,
                &mut raw_candidates,
                &mut seen,
            );
            preferred_loanword = Some(bn_loan.to_string());
            if include_english {
                add_cand(
                    en_loan.to_string(),
                    CandidateSource::Loanword,
                    -1500,
                    &mut raw_candidates,
                    &mut seen,
                );
            }
        } else {
            // Suffix decomposition on loanwords (e.g. "computere" -> "কম্পিউটার" + "ে" -> "কম্পিউটারে", "fileti" -> "ফাইলটি")
            const LOANWORD_SUFFIXES: &[(&str, &str)] = &[
                ("developerder", "দের"),
                ("developerra", "রা"),
                ("notificationgulo", "গুলো"),
                ("guloteo", "গুলোতেও"),
                ("gulateo", "গুলাতেও"),
                ("gulotei", "গুলোতেই"),
                ("gulatei", "গুলাতেই"),
                ("derkeo", "দেরকেও"),
                ("derkei", "দেরকেই"),
                ("gulote", "গুলোতে"),
                ("gulate", "গুলাতে"),
                ("gulor", "গুলোর"),
                ("gular", "গুলার"),
                ("derke", "দেরকে"),
                ("titeo", "টিতেও"),
                ("tateo", "টাতেও"),
                ("titei", "টিতেই"),
                ("tatei", "টাতেই"),
                ("tiro", "টিরও"),
                ("taro", "টারও"),
                ("gulo", "গুলো"),
                ("gula", "গুলা"),
                ("guli", "গুলি"),
                ("deri", "দেরই"),
                ("dero", "দেরও"),
                ("der", "দের"),
                ("tai", "টাই"),
                ("tao", "টাও"),
                ("tio", "টিও"),
                ("tii", "টিই"),
                ("tay", "টায়"),
                ("ete", "েতে"),
                ("ero", "েরও"),
                ("eri", "েরই"),
                ("tei", "তেই"),
                ("rei", "রেই"),
                ("ta", "টা"),
                ("ti", "টি"),
                ("te", "ে"),
                ("er", "ের"),
                ("re", "রে"),
                ("ke", "কে"),
                ("ra", "রা"),
                ("ei", "েই"),
                ("eo", "েও"),
                ("ey", "েই"),
                ("ye", "য়ে"),
                ("ay", "ায়"),
                ("e", "ে"),
                ("r", "র"),
            ];

            let clean_middle = if middle.contains('-') && middle.len() >= 3 {
                middle.replace('-', "")
            } else {
                middle.to_string()
            };

            for &(suf_en, suf_bn) in LOANWORD_SUFFIXES {
                if clean_middle.len() > suf_en.len() && clean_middle.ends_with(suf_en) {
                    let base_en = &clean_middle[..clean_middle.len() - suf_en.len()];
                    if let Some((bn_loan, en_loan)) = PhoneticDatabase::get_bilingual_loanword(base_en) {
                        let actual_suffix = if suf_bn == "র" && !bn_loan.chars().last().map_or(false, |c| c.is_kar() || c.is_vowel()) {
                            "ের"
                        } else {
                            suf_bn
                        };
                        let combined_bn = super::morphology::apply_sandhi_join(bn_loan, actual_suffix);
                        add_cand(
                            combined_bn.clone(),
                            CandidateSource::Loanword,
                            6000,
                            &mut raw_candidates,
                            &mut seen,
                        );
                        preferred_loanword = Some(combined_bn);
                        if include_english {
                            add_cand(
                                format!("{}{}", en_loan, suf_en),
                                CandidateSource::Loanword,
                                -1600,
                                &mut raw_candidates,
                                &mut seen,
                            );
                        }
                        break;
                    }
                }
            }
        }

        // 5. Special literal matches on middle portion (e.g. (:smile:), (=12+5))
        let middle_literals = self.database.search_special_literals(middle);
        for special in middle_literals {
            add_cand(
                special,
                CandidateSource::Autocorrect,
                6000,
                &mut raw_candidates,
                &mut seen,
            );
        }

        // 6. Autocorrect / Common Overrides / Loanwords / Verbal Decomposition / Elongation Collapse
        let resolve_ac = |raw: &str| -> String {
            if raw.chars().any(|c| c.is_bengali()) {
                raw.to_string()
            } else {
                self.convert_phonetic(raw)
            }
        };

        let preferred_word = if let Some(raw_ac) = self.database.get_autocorrect_raw(middle) {
            let converted_ac = resolve_ac(&raw_ac);
            if !converted_ac.is_empty() {
                add_cand(
                    converted_ac.clone(),
                    CandidateSource::Autocorrect,
                    6500,
                    &mut raw_candidates,
                    &mut seen,
                );
                Some(converted_ac)
            } else {
                None
            }
        } else if let Some(loan_pref) = preferred_loanword {
            Some(loan_pref)
        } else if let Some(verbal_word) = super::morphology::decompose_verbal_form(middle) {
            add_cand(
                verbal_word.clone(),
                CandidateSource::ExactDictionary,
                5500,
                &mut raw_candidates,
                &mut seen,
            );
            Some(verbal_word)
        } else {
            let mut found_collapsed = None;
            for collapsed in super::fuzzy::collapse_elongated_runs(middle) {
                if let Some(raw_ac) = self.database.get_autocorrect_raw(&collapsed) {
                    let converted_ac = resolve_ac(&raw_ac);
                    if !converted_ac.is_empty() {
                        add_cand(
                            converted_ac.clone(),
                            CandidateSource::Autocorrect,
                            6000,
                            &mut raw_candidates,
                            &mut seen,
                        );
                        found_collapsed = Some(converted_ac);
                        break;
                    }
                }
                let direct_collapsed = self.convert_phonetic(&collapsed);
                if self.database.is_exact_dictionary_word(&direct_collapsed) {
                    add_cand(
                        direct_collapsed.clone(),
                        CandidateSource::ExactDictionary,
                        3800,
                        &mut raw_candidates,
                        &mut seen,
                    );
                    found_collapsed = Some(direct_collapsed);
                    break;
                }
            }
            found_collapsed
        };

        let primary = preferred_word.as_deref().unwrap_or(&phonetic);
        let is_primary_in_dict = self.database.is_exact_dictionary_word(primary);

        // 7. Direct Transliterations
        if is_primary_in_dict {
            add_cand(
                primary.to_string(),
                CandidateSource::ExactDictionary,
                2800,
                &mut raw_candidates,
                &mut seen,
            );
        } else {
            add_cand(
                primary.to_string(),
                CandidateSource::DirectTransliteration,
                600,
                &mut raw_candidates,
                &mut seen,
            );
        }

        if primary != phonetic {
            if self.database.is_exact_dictionary_word(&phonetic) {
                add_cand(
                    phonetic.clone(),
                    CandidateSource::ExactDictionary,
                    2600,
                    &mut raw_candidates,
                    &mut seen,
                );
            } else {
                add_cand(
                    phonetic.clone(),
                    CandidateSource::DirectTransliteration,
                    500,
                    &mut raw_candidates,
                    &mut seen,
                );
            }
        }

        if use_dictionary {
            // 8. Natural Linguistic & Grammatical Inflections
            if let Some(inflections) = PhoneticDatabase::get_common_inflections(middle) {
                for &inf in inflections {
                    add_cand(
                        inf.to_string(),
                        CandidateSource::MorphologicalInflection,
                        2400,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }

            // 9. Suffix-decomposed forms (when Latin string contains stem + suffix)
            let suffixed_matches = self.add_suffixes(middle, primary);
            for item in suffixed_matches {
                add_cand(
                    item,
                    CandidateSource::MorphologicalInflection,
                    2000,
                    &mut raw_candidates,
                    &mut seen,
                );
            }

            // 9b. Colloquial & Spoken continuous verb forms (e.g. kortesi -> করছি, করতেছি)
            let colloquial_matches = self.add_colloquial_verbs(middle);
            for item in colloquial_matches {
                add_cand(
                    item,
                    CandidateSource::MorphologicalInflection,
                    3300,
                    &mut raw_candidates,
                    &mut seen,
                );
            }

            // 10. Exact Fuzzy Spelling Variants & Sound Laws (Homophones / Orthographic variants & Juktoborno)
            let fuzzy_variants = super::fuzzy::generate_phonetic_variants(middle);
            let mut valid_fuzzy_stems = Vec::new();

            for variant in &fuzzy_variants {
                let var_phonetic = self.convert_phonetic(variant);
                if self.database.is_exact_dictionary_word(&var_phonetic) {
                    let len_diff = (var_phonetic.chars().count() as isize
                        - phonetic.chars().count() as isize)
                        .abs();
                    if len_diff <= 3 {
                        if !valid_fuzzy_stems.contains(&var_phonetic) {
                            valid_fuzzy_stems.push(var_phonetic.clone());
                        }
                        add_cand(
                            var_phonetic,
                            CandidateSource::FuzzySoundLaw,
                            3200,
                            &mut raw_candidates,
                            &mut seen,
                        );
                    }
                }
            }

            if middle.contains('-') && middle.len() >= 3 && !middle.split('-').all(|p| p.len() <= 1)
            {
                let unhyphenated = middle.replace('-', "");
                let unhyphenated_conv = self.convert_phonetic(&unhyphenated);
                if self.database.is_exact_dictionary_word(&unhyphenated_conv) {
                    if !valid_fuzzy_stems.contains(&unhyphenated_conv) {
                        valid_fuzzy_stems.push(unhyphenated_conv.clone());
                    }
                    add_cand(
                        unhyphenated_conv,
                        CandidateSource::FuzzySoundLaw,
                        3200,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
                for fz in super::fuzzy::generate_phonetic_variants(&unhyphenated) {
                    let fz_conv = self.convert_phonetic(&fz);
                    if self.database.is_exact_dictionary_word(&fz_conv) {
                        if !valid_fuzzy_stems.contains(&fz_conv) {
                            valid_fuzzy_stems.push(fz_conv.clone());
                        }
                        add_cand(
                            fz_conv,
                            CandidateSource::FuzzySoundLaw,
                            3200,
                            &mut raw_candidates,
                            &mut seen,
                        );
                    }
                }
            }

            // 11. Sound-Law Aware Trie Prefix Autocompletion
            let mut prefix_search_roots = vec![primary.to_string()];
            for stem in valid_fuzzy_stems.iter().take(3) {
                if !prefix_search_roots.contains(stem) {
                    prefix_search_roots.push(stem.clone());
                }
            }

            for root in &prefix_search_roots {
                let limit = if is_primary_in_dict { 3 } else { 5 };
                let prefix_exts = self.database.search_dictionary(root, limit);
                for ext in prefix_exts {
                    if ext != *root {
                        add_cand(
                            ext,
                            CandidateSource::TriePrefixAutocomplete,
                            1600,
                            &mut raw_candidates,
                            &mut seen,
                        );
                    }
                }
            }

            // 12. Single-Edit Typo Tolerance Fallback
            let has_dict_cand = raw_candidates
                .iter()
                .any(|c| self.database.is_exact_dictionary_word(&c.text));
            if !has_dict_cand && middle.chars().count() >= 4 {
                let mut typo_matches: Vec<String> = Vec::new();
                for (i, ch) in middle.char_indices() {
                    let mut del = String::with_capacity(middle.len());
                    del.push_str(&middle[..i]);
                    del.push_str(&middle[i + ch.len_utf8()..]);
                    let del_conv = self.convert_phonetic(&del);
                    if self.database.is_exact_dictionary_word(&del_conv) {
                        typo_matches.push(del_conv);
                    }
                }
                typo_matches
                    .sort_unstable_by_key(|w| std::cmp::Reverse(self.database.get_frequency(w)));
                for tm in typo_matches.into_iter().take(3) {
                    add_cand(
                        tm,
                        CandidateSource::TypoFallback,
                        1000,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
        }

        // 13. Contextual Emoji / Symbol Keywords (Demoted so they sit trailing)
        let mut kw_emojis = self.database.lookup_emoji_keywords(middle);
        for em in self.database.lookup_emoji_keywords(primary) {
            if !kw_emojis.contains(&em) {
                kw_emojis.push(em);
            }
        }
        if phonetic != *primary {
            for em in self.database.lookup_emoji_keywords(&phonetic) {
                if !kw_emojis.contains(&em) {
                    kw_emojis.push(em);
                }
            }
        }
        for em in &kw_emojis {
            add_cand(
                em.clone(),
                CandidateSource::EmojiKeyword,
                -2000,
                &mut raw_candidates,
                &mut seen,
            );
        }

        // 14. Original English Fallback
        if include_english {
            add_cand(
                term.to_string(),
                CandidateSource::DirectTransliteration,
                -3000,
                &mut raw_candidates,
                &mut seen,
            );
        }

        // 15. Global Multi-Factor Probabilistic Scoring & Ranker
        let has_valid_dict_candidates = raw_candidates.iter().any(|c| {
            (c.source == CandidateSource::FuzzySoundLaw || c.source == CandidateSource::Autocorrect)
                && self.database.is_exact_dictionary_word(&c.text)
        });

        let mut scored_candidates: Vec<(String, i32)> = Vec::with_capacity(raw_candidates.len());

        for cand in raw_candidates {
            let is_in_dict = self.database.is_exact_dictionary_word(&cand.text);
            let freq = self.database.get_frequency(&cand.text);
            let mut score = cand.initial_boost;

            // 1. Exact Dictionary Status & Promotion Rule
            if is_in_dict {
                score += 2200;
            } else if cand.source == CandidateSource::DirectTransliteration {
                // If raw transliteration is NOT in the dictionary, but valid sound-law alternatives exist, penalize it!
                if has_valid_dict_candidates {
                    score -= 2800;
                }
            }

            // 2. Frequency bonus (logarithmic scaling)
            if freq > 0 {
                let freq_f = freq as f64;
                let log_freq = freq_f.log2();
                score += (log_freq * 130.0) as i32;
            }

            // 3. Edit distance & length difference penalty from raw phonetic form
            if cand.source != CandidateSource::EmojiKeyword {
                let dist = edit_distance(&phonetic, &cand.text);
                if cand.text == phonetic || cand.text == *primary {
                    if is_in_dict {
                        score += 3500;
                    } else {
                        score += 1500;
                    }
                } else {
                    score -= (dist as i32) * 200;
                    let len_diff = (cand.text.chars().count() as isize
                        - phonetic.chars().count() as isize)
                        .abs() as i32;
                    score -= len_diff * 400;
                }
            }

            // 4. Contextual Homophone & AI Language Model Scoring
            if !context.is_empty() && cand.source != CandidateSource::EmojiKeyword {
                let prev = context.last().copied();
                let homophone_boost = Self::score_contextual_homophone(prev, &cand.text);
                if homophone_boost > 0 {
                    score += homophone_boost * 4;
                } else if use_dictionary {
                    let lm_score = lekhani_ai::LanguageModel::new().score_candidate(
                        if context.len() >= 2 {
                            context.get(context.len() - 2).copied()
                        } else {
                            None
                        },
                        prev,
                        &cand.text,
                    );
                    if lm_score > -1.0 {
                        score += 1800;
                    } else if lm_score > -2.2 {
                        score += 900;
                    } else if lm_score > -3.5 {
                        score += 300;
                    }
                }
            }

            // 5. Candidate Memory & Autonomous Learner Boost
            if let Some(fav) = candidate_memory
                .get(term)
                .or_else(|| candidate_memory.get(middle))
                .or_else(|| candidate_memory.get(&phonetic))
            {
                if &cand.text == fav || cand.text.contains(fav) {
                    score += 5000;
                }
            }
            if self.database.learner.learned_words.contains(&cand.text) {
                score += 3500;
            }

            scored_candidates.push((cand.text, score));
        }

        // Sort candidates by descending total score
        scored_candidates.sort_by_key(|a| std::cmp::Reverse(a.1));

        let mut candidates: Vec<String> = Vec::with_capacity(8);
        let has_english = include_english && scored_candidates.iter().any(|(c, _)| c == term);
        let has_emoji = scored_candidates.iter().any(|(c, _)| kw_emojis.contains(c));
        let first_emoji = scored_candidates
            .iter()
            .find(|(c, _)| kw_emojis.contains(c))
            .map(|(c, _)| c.clone());

        let text_target_len = match (has_english, has_emoji) {
            (true, true) => 6,
            (true, false) | (false, true) => 7,
            (false, false) => 8,
        };

        for (c, _) in &scored_candidates {
            if c == term || kw_emojis.contains(c) {
                continue;
            }
            if candidates.len() < text_target_len {
                candidates.push(c.clone());
            }
        }

        if let Some(em) = first_emoji {
            if !candidates.contains(&em) {
                candidates.push(em);
            }
        }

        if has_english {
            candidates.push(term.to_string());
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

    /// Score homophone pairs based on semantic preceding context using the statistical language model
    pub fn score_contextual_homophone(previous_word: Option<&str>, candidate: &str) -> i32 {
        let prev = match previous_word {
            Some(p) if !p.trim().is_empty() => p.trim(),
            _ => return 0,
        };

        let score = lekhani_ai::LanguageModel::new().score_candidate(None, Some(prev), candidate);
        if score > -0.5 {
            1200
        } else if score > -1.0 {
            1000
        } else if score > -2.0 {
            600
        } else if score > -3.0 {
            250
        } else {
            0
        }
    }

    /// Predict next probable Bengali words for zero-preedit state using the statistical language model
    pub fn suggest_next_words(&self, previous_word: &str) -> Vec<String> {
        let prev = previous_word.trim();
        if prev.is_empty() {
            return Vec::new();
        }

        self.ai_predictor.predict_next(&[prev], 8)
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

    fn add_suffixes(&self, middle: &str, _phonetic: &str) -> Vec<String> {
        let mut list = Vec::new();

        if middle.chars().count() > 2 {
            for (i, _) in middle.char_indices().skip(1) {
                let suffix_key = &middle[i..];
                if let Some(suffix) = self.database.find_suffix(suffix_key) {
                    let base_key = &middle[..i];
                    let mut base_candidates = Vec::new();
                    if let Some(ac) = self.database.get_autocorrect_raw(base_key) {
                        let conv = if ac.chars().any(|c| c.is_bengali()) {
                            ac
                        } else {
                            self.convert_phonetic(&ac)
                        };
                        if !conv.is_empty() && !base_candidates.contains(&conv) {
                            base_candidates.push(conv);
                        }
                    }
                    let base_phonetic = self.convert_phonetic(base_key);
                    if !base_candidates.contains(&base_phonetic) {
                        base_candidates.push(base_phonetic);
                    }

                    // Apply stem-level sound-laws (e.g. "manush" in "manusher" -> "মানুষ")
                    for fz in super::fuzzy::generate_phonetic_variants(base_key) {
                        let conv = self.convert_phonetic(&fz);
                        if self.database.is_exact_dictionary_word(&conv) && !base_candidates.contains(&conv) {
                            base_candidates.push(conv);
                        }
                    }

                    for base in &base_candidates {
                        let word = super::morphology::apply_sandhi_join(base, suffix);
                        if (self.database.is_exact_dictionary_word(&word)
                            || self.database.is_exact_dictionary_word(base))
                            && !list.iter().any(|item| item == &word)
                        {
                            list.push(word);
                        }
                    }
                }
            }
        }

        list
    }

    fn add_colloquial_verbs(&self, middle: &str) -> Vec<String> {
        let mut list = Vec::new();
        for &(suffix_key, expansions) in super::morphology::COLLOQUIAL_VERBAL_PATTERNS {
            if middle.ends_with(suffix_key) && middle.len() > suffix_key.len() {
                let stem_latin = &middle[..middle.len() - suffix_key.len()];
                if stem_latin.chars().count() >= 2 {
                    let stem_bn = self.convert_phonetic(stem_latin);
                    for &exp in expansions {
                        let combined = match stem_latin {
                            "jai" | "jawa" => match exp {
                                "ছি" => "যাচ্ছি".to_string(),
                                "ছো" => "যাচ্ছ".to_string(),
                                "ছে" => "যাচ্ছে".to_string(),
                                "ছেন" => "যাচ্ছেন".to_string(),
                                "ছিলাম" => "যাচ্ছিলাম".to_string(),
                                "ছিলা" => "যাচ্ছিলা".to_string(),
                                "ছিল" => "যাচ্ছিল".to_string(),
                                "ছিলেন" => "যাচ্ছিলেন".to_string(),
                                _ => format!("যাই{}", exp),
                            },
                            "khai" | "khawa" => match exp {
                                "ছি" => "খাচ্ছি".to_string(),
                                "ছো" => "খাচ্ছ".to_string(),
                                "ছে" => "খাচ্ছে".to_string(),
                                "ছেন" => "খাচ্ছেন".to_string(),
                                "ছিলাম" => "খাচ্ছিলাম".to_string(),
                                "ছিলা" => "খাচ্ছিলা".to_string(),
                                "ছিল" => "খাচ্ছিল".to_string(),
                                "ছিলেন" => "খাচ্ছিলেন".to_string(),
                                _ => format!("খাই{}", exp),
                            },
                            "de" | "di" => match exp {
                                "ছি" => "দিচ্ছি".to_string(),
                                "ছো" => "দিচ্ছ".to_string(),
                                "ছে" => "দিচ্ছে".to_string(),
                                "ছেন" => "দিচ্ছেন".to_string(),
                                "ছিলাম" => "দিচ্ছিলাম".to_string(),
                                "ছিল" => "দিচ্ছিল".to_string(),
                                _ => format!("দি{}", exp),
                            },
                            "ne" | "ni" => match exp {
                                "ছি" => "নিচ্ছি".to_string(),
                                "ছো" => "নিচ্ছ".to_string(),
                                "ছে" => "নিচ্ছে".to_string(),
                                "ছেন" => "নিচ্ছেন".to_string(),
                                "ছিলাম" => "নিচ্ছিলাম".to_string(),
                                "ছিল" => "নিচ্ছিল".to_string(),
                                _ => format!("নি{}", exp),
                            },
                            "ash" | "as" => format!("আস{}", exp),
                            "bol" => format!("বল{}", exp),
                            "kor" => format!("কর{}", exp),
                            "dekh" | "dek" => format!("দেখ{}", exp),
                            _ => format!("{}{}", stem_bn, exp),
                        };
                        if !list.contains(&combined) {
                            list.push(combined);
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
    fn test_bilingual_keyword_emoji_suggestions() {
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

        // 1. Bengali word "cha" should suggest "চা" at top, and "☕" in trailing candidates
        let (cands_cha, _) = sugg.suggest("cha", true, true, &empty_memory);
        assert_eq!(cands_cha[0], "চা");
        assert!(cands_cha.contains(&"☕".to_string()));

        // 2. Bengali word "bhalobasha" should suggest "ভালোবাসা" at top, and "❤️" in trailing candidates
        let (cands_love, _) = sugg.suggest("bhalobasha", true, true, &empty_memory);
        assert_eq!(cands_love[0], "ভালোবাসা");
        assert!(cands_love.contains(&"❤️".to_string()));

        // 3. English word "coffee" / "tea" should suggest "☕"
        let (cands_coffee, _) = sugg.suggest("coffee", true, true, &empty_memory);
        assert!(cands_coffee.contains(&"☕".to_string()));

        // 4. English word "dog" / "cat" should suggest emojis
        let (cands_dog, _) = sugg.suggest("dog", true, true, &empty_memory);
        assert!(cands_dog.contains(&"🐶".to_string()));

        let (cands_cat, _) = sugg.suggest("cat", true, true, &empty_memory);
        assert!(cands_cat.contains(&"🐱".to_string()));
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

        // 1. "kosto" accurately suggests "কষ্ট" at Rank #1
        let (cands_kosto, _) = sugg.suggest("kosto", true, true, &empty_memory);
        assert!(!cands_kosto.is_empty());
        assert_eq!(cands_kosto[0], "কষ্ট");

        let (cands_koshto, _) = sugg.suggest("koshto", true, true, &empty_memory);
        assert_eq!(cands_koshto[0], "কষ্ট");
        let (cands_ko_sh_to, _) = sugg.suggest("koShTo", true, true, &empty_memory);
        assert_eq!(cands_ko_sh_to[0], "কষ্ট");

        // 2. "nosto" accurately suggests "নষ্ট" at Rank #1
        let (cands_nosto, _) = sugg.suggest("nosto", true, true, &empty_memory);
        assert!(!cands_nosto.is_empty());
        assert_eq!(cands_nosto[0], "নষ্ট");

        let (cands_noshto, _) = sugg.suggest("noshto", true, true, &empty_memory);
        assert_eq!(cands_noshto[0], "নষ্ট");
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

        // 5. "sristi" suggests "সৃষ্টি" at Rank #1
        let (cands_sristi, _) = sugg.suggest("sristi", true, true, &empty_memory);
        assert_eq!(cands_sristi[0], "সৃষ্টি");

        let (cands_srish_ti, _) = sugg.suggest("srriShTi", true, true, &empty_memory);
        assert_eq!(cands_srish_ti[0], "সৃষ্টি");

        // 6. "bristi" suggests "বৃষ্টি" at Rank #1
        let (cands_bristi, _) = sugg.suggest("bristi", true, true, &empty_memory);
        assert_eq!(cands_bristi[0], "বৃষ্টি");

        // 7. "bebostha" suggests "ব্যবস্থা" at Rank #1
        let (cands_bebostha, _) = sugg.suggest("bebostha", true, true, &empty_memory);
        assert_eq!(cands_bebostha[0], "ব্যবস্থা");

        // 8. "onusthan" suggests "অনুষ্ঠান" at Rank #1
        let (cands_onusthan, _) = sugg.suggest("onusthan", true, true, &empty_memory);
        assert_eq!(cands_onusthan[0], "অনুষ্ঠান");
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
        assert!(cands_chad.contains(&"ছাদ".to_string()) && cands_chad.contains(&"চাঁদ".to_string()));

        let (cands_cad, _) = sugg.suggest("cad", true, true, &empty_memory);
        assert!(cands_cad.contains(&"চাঁদ".to_string()));

        let (cands_c_ad, _) = sugg.suggest("ca^d", true, true, &empty_memory);
        assert_eq!(cands_c_ad[0], "চাঁদ");

        let (cands_dat, _) = sugg.suggest("dat", true, true, &empty_memory);
        assert!(cands_dat.contains(&"দাঁত".to_string()));

        let (cands_has, _) = sugg.suggest("has", true, true, &empty_memory);
        assert!(cands_has.contains(&"হাঁস".to_string()));

        let (cands_bas, _) = sugg.suggest("bas", true, true, &empty_memory);
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

    #[test]
    fn test_unified_rank1_probabilistic_promotion() {
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

        // 1. Invalid naive transliterations are demoted in favor of real dictionary words at Rank #1
        let (cands_kosto, _) = sugg.suggest("kosto", true, true, &empty_memory);
        assert_eq!(cands_kosto[0], "কষ্ট");

        let (cands_nosto, _) = sugg.suggest("nosto", true, true, &empty_memory);
        assert_eq!(cands_nosto[0], "নষ্ট");

        let (cands_bortoman, _) = sugg.suggest("bortoman", true, true, &empty_memory);
        assert_eq!(cands_bortoman[0], "বর্তমান");

        let (cands_chesta, _) = sugg.suggest("chesta", true, true, &empty_memory);
        assert_eq!(cands_chesta[0], "চেষ্টা");

        let (cands_sristi, _) = sugg.suggest("sristi", true, true, &empty_memory);
        assert_eq!(cands_sristi[0], "সৃষ্টি");

        let (cands_shundor, _) = sugg.suggest("shundor", true, true, &empty_memory);
        assert_eq!(cands_shundor[0], "সুন্দর");

        let (cands_shadhinota, _) = sugg.suggest("shadhinota", true, true, &empty_memory);
        assert_eq!(cands_shadhinota[0], "স্বাধীনতা");
    }

    #[test]
    fn test_sound_law_prefix_autocomplete_and_trailing_emojis() {
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

        // 1. Sound-law prefix autocompletion for "shadhin" suggests "স্বাধীনতা"
        let (cands_shadhin, _) = sugg.suggest("shadhin", true, true, &empty_memory);
        assert_eq!(cands_shadhin[0], "স্বাধীন");
        assert!(cands_shadhin.contains(&"স্বাধীনতা".to_string()));

        // 2. Trailing emoji demotion: emojis should not displace top grammatical words
        let (cands_love, _) = sugg.suggest("bhalobasha", true, true, &empty_memory);
        assert_eq!(cands_love[0], "ভালোবাসা");
        if let Some(pos) = cands_love.iter().position(|c| c == "❤️") {
            assert!(pos >= 3, "Emoji should not displace top candidate words");
        }

        let (cands_tea, _) = sugg.suggest("cha", true, true, &empty_memory);
        assert_ne!(cands_tea[0], "☕");
    }

    #[test]
    fn test_colloquial_verb_suggestions() {
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

        let (cands_kor, _) = sugg.suggest("kortesi", true, true, &empty_memory);
        assert!(cands_kor.contains(&"করছি".to_string()) || cands_kor.contains(&"করতেছি".to_string()));

        let (cands_jai, _) = sugg.suggest("jaitasi", true, true, &empty_memory);
        assert!(cands_jai.contains(&"যাচ্ছি".to_string()) || cands_jai.contains(&"যাইতেছি".to_string()));
    }

    #[test]
    fn test_inflection_and_ranking_regressions() {
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

        // 1. Inflected Nouns (should rank exact inflections at #1, not longer compound autocompletions)
        let (cands_deshe, _) = sugg.suggest("deshe", true, true, &empty_memory);
        assert_eq!(cands_deshe[0], "দেশে");

        let (cands_desher, _) = sugg.suggest("desher", true, true, &empty_memory);
        assert_eq!(cands_desher[0], "দেশের");

        let (cands_dine, _) = sugg.suggest("dine", true, true, &empty_memory);
        assert_eq!(cands_dine[0], "দিনে");

        let (cands_ekhane, _) = sugg.suggest("ekhane", true, true, &empty_memory);
        assert_eq!(cands_ekhane[0], "এখানে");

        let (cands_bhabe, _) = sugg.suggest("bhabe", true, true, &empty_memory);
        assert_eq!(cands_bhabe[0], "ভাবে");

        // 2. Everyday common words outranking obscure dictionary entries
        let (cands_tomra, _) = sugg.suggest("tomra", true, true, &empty_memory);
        assert_eq!(cands_tomra[0], "তোমরা");

        let (cands_gari, _) = sugg.suggest("gari", true, true, &empty_memory);
        assert_eq!(cands_gari[0], "গাড়ি");

        let (cands_shob, _) = sugg.suggest("shob", true, true, &empty_memory);
        assert_eq!(cands_shob[0], "সব");

        let (cands_jabo, _) = sugg.suggest("jabo", true, true, &empty_memory);
        assert_eq!(cands_jabo[0], "যাব");

        let (cands_kono, _) = sugg.suggest("kono", true, true, &empty_memory);
        assert_eq!(cands_kono[0], "কোনো");

        // 3. Suffixes and Classifiers
        let (cands_boita, _) = sugg.suggest("boita", true, true, &empty_memory);
        assert_eq!(cands_boita[0], "বইটা");

        let (cands_garite, _) = sugg.suggest("garite", true, true, &empty_memory);
        assert_eq!(cands_garite[0], "গাড়িতে");

        // 4. Sound-law Stem Propagation on Inflected Words
        let (cands_manusher, _) = sugg.suggest("manusher", true, true, &empty_memory);
        assert_eq!(cands_manusher[0], "মানুষের");

        let (cands_porikkhay, _) = sugg.suggest("porikkhay", true, true, &empty_memory);
        assert!(cands_porikkhay[0] == "পরীক্ষায়" || cands_porikkhay[0] == "পরীক্ষায়");

        // 5. Casual Phonetic Typing
        let (cands_ektu, _) = sugg.suggest("ektu", true, true, &empty_memory);
        assert_eq!(cands_ektu[0], "একটু");

        let (cands_bisshash, _) = sugg.suggest("bisshash", true, true, &empty_memory);
        assert_eq!(cands_bisshash[0], "বিশ্বাস");

        let (cands_bissho, _) = sugg.suggest("bissho", true, true, &empty_memory);
        assert_eq!(cands_bissho[0], "বিশ্ব");

        let (cands_ditiyo, _) = sugg.suggest("ditiyo", true, true, &empty_memory);
        assert_eq!(cands_ditiyo[0], "দ্বিতীয়");

        let (cands_shikhok, _) = sugg.suggest("shikhok", true, true, &empty_memory);
        assert_eq!(cands_shikhok[0], "শিক্ষক");

        let (cands_shartho, _) = sugg.suggest("shartho", true, true, &empty_memory);
        assert_eq!(cands_shartho[0], "স্বার্থ");

        // 6. Conversational 2nd-person Verbs (acho, korcho, esho, dekho, cholo, paro)
        let (cands_acho, _) = sugg.suggest("acho", true, true, &empty_memory);
        assert_eq!(cands_acho[0], "আছো");

        let (cands_korcho, _) = sugg.suggest("korcho", true, true, &empty_memory);
        assert_eq!(cands_korcho[0], "করছো");

        let (cands_esho, _) = sugg.suggest("esho", true, true, &empty_memory);
        assert_eq!(cands_esho[0], "এসো");

        let (cands_dekho, _) = sugg.suggest("dekho", true, true, &empty_memory);
        assert_eq!(cands_dekho[0], "দেখো");

        let (cands_cholo, _) = sugg.suggest("cholo", true, true, &empty_memory);
        assert_eq!(cands_cholo[0], "চলো");

        let (cands_paro, _) = sugg.suggest("paro", true, true, &empty_memory);
        assert_eq!(cands_paro[0], "পারো");

        // 7. Complex Sanskrit / Ha-Conjuncts & Clitics
        let (cands_ahban, _) = sugg.suggest("ahban", true, true, &empty_memory);
        assert_eq!(cands_ahban[0], "আহ্বান");

        let (cands_chinho, _) = sugg.suggest("chinho", true, true, &empty_memory);
        assert_eq!(cands_chinho[0], "চিহ্ন");

        let (cands_apranho, _) = sugg.suggest("apranho", true, true, &empty_memory);
        assert_eq!(cands_apranho[0], "অপরাহ্ন");

        let (cands_hrit, _) = sugg.suggest("hritpindo", true, true, &empty_memory);
        assert_eq!(cands_hrit[0], "হৃৎপিণ্ড");

        let (cands_chhatro, _) = sugg.suggest("chhatrochhatriderkeo", true, true, &empty_memory);
        assert_eq!(cands_chhatro[0], "ছাত্রছাত্রীদেরকেও");

        let (cands_dhai, _) = sugg.suggest("dhai", true, true, &empty_memory);
        assert_eq!(cands_dhai[0], "আড়াই");

        // 8. Inflected Loanwords (computere -> কম্পিউটারে, fileti -> ফাইলটি)
        let (cands_comp, _) = sugg.suggest("computere", true, true, &empty_memory);
        assert!(cands_comp[0] == "কম্পিউটারে" || cands_comp.contains(&"কম্পিউটারে".to_string()));

        let (cands_file, _) = sugg.suggest("fileti", true, true, &empty_memory);
        assert!(cands_file[0] == "ফাইলটি" || cands_file.contains(&"ফাইলটি".to_string()));

        let (cands_phone, _) = sugg.suggest("smartphonete", true, true, &empty_memory);
        assert!(cands_phone[0] == "স্মার্টফোনে" || cands_phone.contains(&"স্মার্টফোনে".to_string()));
    }
}
