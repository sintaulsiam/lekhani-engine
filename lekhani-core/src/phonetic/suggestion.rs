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
    CodeShield,
    SegmentedLattice,
}

#[derive(Debug, Clone)]
pub struct CandidateHypothesis {
    pub text: String,
    pub source: CandidateSource,
    pub initial_boost: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiProfile {
    Balanced,
    Lite,
    Off,
}

impl Default for AiProfile {
    fn default() -> Self {
        Self::Balanced
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PhoneticSuggestionConfig {
    pub ai_profile: AiProfile,
    pub use_dictionary: bool,
    pub include_english: bool,
    pub enable_code_shield: bool,
    pub enable_word_segmentation: bool,
    pub enable_colloquial_dialects: bool,
    pub enable_banglish_shorthand: bool,
    pub enable_reduplication: bool,
    pub enable_phrase_prediction: bool,
    pub enable_dynamic_macros: bool,
    pub auto_dari: bool,
}

impl Default for PhoneticSuggestionConfig {
    fn default() -> Self {
        Self {
            ai_profile: AiProfile::default(),
            use_dictionary: true,
            include_english: true,
            enable_code_shield: false,
            enable_word_segmentation: false,
            enable_colloquial_dialects: true,
            enable_banglish_shorthand: true,
            enable_reduplication: true,
            enable_phrase_prediction: true,
            enable_dynamic_macros: true,
            auto_dari: true,
        }
    }
}

#[derive(Clone)]
pub struct PhoneticSuggestion {
    pub database: PhoneticDatabase,
    phonetic_parser: Option<Arc<PhoneticParser>>,
    cache: HashMap<String, Vec<String>>,
    pub ai_context: lekhani_ai::ContextScorer,
    pub ai_predictor: lekhani_ai::NextWordPredictor,
    pub ai_decoder: lekhani_ai::BeamSearchDecoder,
    pub config: PhoneticSuggestionConfig,
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

/// Normalize caret position in Latin phonetic input (e.g. "ch^ad" -> "cha^d", "k^ada" -> "ka^da")
pub fn normalize_chandra_latin(input: &str) -> String {
    if !input.contains('^') {
        return input.to_string();
    }
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '^' && i + 1 < chars.len() {
            let next = chars[i + 1];
            if is_latin_vowel(next) {
                let mut v_end = i + 1;
                while v_end < chars.len() && is_latin_vowel(chars[v_end]) {
                    v_end += 1;
                }
                for v in &chars[i + 1..v_end] {
                    out.push(*v);
                }
                out.push('^');
                i = v_end;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

#[inline]
fn is_latin_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U' | 'y' | 'Y' | '`'
    )
}

/// Ensure canonical Unicode ordering for Bengali Chandra Bindu and Nukta glyphs
pub fn fix_chandra_bindu_position(text: &str) -> String {
    let mut res = text.to_string();
    if res.contains('\u{09BC}') {
        res = res.replace("\u{09AF}\u{09BC}", "য়");
        res = res.replace("\u{09A1}\u{09BC}", "ড়");
        res = res.replace("\u{09A2}\u{09BC}", "ঢ়");
    }
    if !res.contains('ঁ') {
        return res;
    }
    res = res.replace("ঁআ", "াঁ");
    res = res.replace("ঁা", "াঁ");
    res = res.replace("ঁএ", "েঁ");
    res = res.replace("ঁে", "েঁ");
    res = res.replace("ঁই", "িঁ");
    res = res.replace("ঁি", "িঁ");
    res = res.replace("ঁউ", "ুঁ");
    res = res.replace("ঁু", "ুঁ");
    res = res.replace("ঁও", "োঁ");
    res = res.replace("ঁো", "োঁ");
    res = res.replace("ঁঈ", "ীঁ");
    res = res.replace("ঁী", "ীঁ");
    res = res.replace("ঁঊ", "ূঁ");
    res = res.replace("ঁূ", "ূঁ");
    res = res.replace("ঁঋ", "ৃঁ");
    res = res.replace("ঁৃ", "ৃঁ");
    res = res.replace("ঁঐ", "ৈঁ");
    res = res.replace("ঁৈ", "ৈঁ");
    res = res.replace("ঁঔ", "ৌঁ");
    res = res.replace("ঁৌ", "ৌঁ");
    res
}

impl Default for PhoneticSuggestion {
    fn default() -> Self {
        Self::new()
    }
}

impl PhoneticSuggestion {
    pub fn new() -> Self {
        let lm = lekhani_ai::LanguageModel::new();
        Self {
            database: PhoneticDatabase::new(),
            phonetic_parser: None,
            cache: HashMap::new(),
            ai_context: lekhani_ai::ContextScorer::with_language_model(lm.clone()),
            ai_predictor: lekhani_ai::NextWordPredictor::with_language_model(lm.clone()),
            ai_decoder: lekhani_ai::BeamSearchDecoder::with_language_model(lm, 4),
            config: PhoneticSuggestionConfig::default(),
        }
    }

    /// Check if a layout JSON structure contains valid phonetic patterns
    pub fn is_valid_phonetic_layout(layout_json: &Value) -> bool {
        let layout_obj = if let Some(l) = layout_json.get("layout") {
            l
        } else {
            layout_json
        };
        layout_obj.is_object() && layout_obj.get("patterns").map_or(false, |p| p.is_array())
    }

    pub fn with_layout(layout_json: &Value) -> Self {
        let layout_obj = if let Some(l) = layout_json.get("layout") {
            l
        } else {
            layout_json
        };
        let parser = if Self::is_valid_phonetic_layout(layout_json) {
            std::panic::catch_unwind(|| Arc::new(PhoneticParser::new(layout_obj))).ok()
        } else {
            None
        };
        let lm = lekhani_ai::LanguageModel::new();
        Self {
            database: PhoneticDatabase::new(),
            phonetic_parser: parser,
            cache: HashMap::new(),
            ai_context: lekhani_ai::ContextScorer::with_language_model(lm.clone()),
            ai_predictor: lekhani_ai::NextWordPredictor::with_language_model(lm.clone()),
            ai_decoder: lekhani_ai::BeamSearchDecoder::with_language_model(lm, 4),
            config: PhoneticSuggestionConfig::default(),
        }
    }

    pub fn set_layout(&mut self, layout_json: &Value) -> bool {
        let layout_obj = if let Some(l) = layout_json.get("layout") {
            l
        } else {
            layout_json
        };
        if Self::is_valid_phonetic_layout(layout_json) {
            if let Ok(parser) = std::panic::catch_unwind(|| Arc::new(PhoneticParser::new(layout_obj))) {
                self.phonetic_parser = Some(parser);
                self.cache.clear();
                return true;
            }
        }
        false
    }

    pub fn update_config(&mut self, config: PhoneticSuggestionConfig) {
        self.config = config;
        self.cache.clear();
    }

    /// Clear cached candidate suggestions
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Transliterate directly using Avro phonetic rules
    pub fn convert_phonetic(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        let norm_text = if text.contains('^') {
            normalize_chandra_latin(text)
        } else {
            text.to_string()
        };
        let text = norm_text.as_str();

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
                                fix_chandra_bindu_position(&parser.convert(part))
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
                return fix_chandra_bindu_position(&parser.convert(text));
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
                        result.push_str(&fix_chandra_bindu_position(&parser.convert(&ascii_chunk)));
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
                result.push_str(&fix_chandra_bindu_position(&parser.convert(&ascii_chunk)));
            } else {
                result.push_str(&ascii_chunk);
            }
        }

        fix_chandra_bindu_position(&result)
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

        // 1. Instant Fast-Path: Smart Typography, Punctuation & Vowel Signs (Zero Allocations)
        match term {
            ".." if self.config.auto_dari => {
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
            "a" | "A" => {
                let mut cands = vec!["আ".to_string(), "া".to_string()];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            "i" => {
                let mut cands = vec![
                    "ই".to_string(),
                    "ি".to_string(),
                    "ঈ".to_string(),
                    "ী".to_string(),
                ];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            "I" => {
                let mut cands = vec![
                    "ঈ".to_string(),
                    "ী".to_string(),
                    "ই".to_string(),
                    "ি".to_string(),
                ];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            "u" => {
                let mut cands = vec![
                    "উ".to_string(),
                    "ু".to_string(),
                    "ঊ".to_string(),
                    "ূ".to_string(),
                ];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            "U" => {
                let mut cands = vec![
                    "ঊ".to_string(),
                    "ূ".to_string(),
                    "উ".to_string(),
                    "ু".to_string(),
                ];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            "e" | "E" => {
                let mut cands = vec!["এ".to_string(), "ে".to_string()];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            "o" | "O" => {
                let mut cands = vec!["ও".to_string(), "অ".to_string(), "ো".to_string()];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            "rri" | "RRI" => {
                let mut cands = vec!["ঋ".to_string(), "ৃ".to_string()];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            "oi" => {
                let mut cands = vec!["ওই".to_string(), "ঐ".to_string(), "ৈ".to_string()];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            "OI" => {
                let mut cands = vec!["ঐ".to_string(), "ৈ".to_string(), "ওই".to_string()];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            "ou" => {
                let mut cands = vec!["ঔ".to_string(), "ৌ".to_string(), "ওউ".to_string()];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            "OU" => {
                let mut cands = vec!["ঔ".to_string(), "ৌ".to_string()];
                if include_english {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
            _ => {}
        }

        // Fast Memoization Cache check (avoid context.join allocation when context is empty)
        let cfg_mask = (self.config.enable_code_shield as u32)
            | ((self.config.enable_word_segmentation as u32) << 1)
            | ((self.config.enable_colloquial_dialects as u32) << 2)
            | ((self.config.enable_banglish_shorthand as u32) << 3)
            | ((self.config.enable_dynamic_macros as u32) << 4);

        let cache_key = if context.is_empty() {
            format!("{}:{}:{}:{}", term, include_english, use_dictionary, cfg_mask)
        } else {
            format!(
                "{}:{}:{}:{}:{}",
                term,
                context.join(" "),
                include_english,
                use_dictionary,
                cfg_mask
            )
        };
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

        // 2. Direct whole-term special matches (Math `=125*8`, Currency `#usd50`, Snippets `!shubhechha`, Exact Emojis `:smile:`, `:)`, `$$`, `*taka*`)
        if self.config.enable_dynamic_macros || !term.starts_with('!') {
            let literals = self.database.search_special_literals(term);
            if !literals.is_empty() {
                let mut cands = literals;
                if include_english && !cands.iter().any(|c| c == term) {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
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
        if self.config.enable_dynamic_macros && term.starts_with('!') && term.len() >= 2 {
            let prefix_snippets = self.database.search_snippets_prefix(term, 8);
            if !prefix_snippets.is_empty() {
                let mut cands = prefix_snippets;
                if include_english && !cands.iter().any(|c| c == term) {
                    cands.push(term.to_string());
                }
                return (cands, 0);
            }
        }

        let (pre, middle_raw, post) = split_word_punct(term);
        let middle_norm = if middle_raw.contains('^') {
            normalize_chandra_latin(middle_raw)
        } else {
            middle_raw.to_string()
        };
        let middle = middle_norm.as_str();

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

        let add_cand = |mut text: String,
                        source: CandidateSource,
                        initial_boost: i32,
                        raw_candidates: &mut Vec<CandidateHypothesis>,
                        seen: &mut hashbrown::HashSet<String>| {
            text = fix_chandra_bindu_position(&text);
            if !text.is_empty() {
                if seen.insert(text.clone()) {
                    raw_candidates.push(CandidateHypothesis {
                        text,
                        source,
                        initial_boost,
                    });
                } else if source == CandidateSource::MorphologicalInflection
                    || source == CandidateSource::ExactDictionary
                {
                    if let Some(existing) = raw_candidates.iter_mut().find(|c| c.text == text) {
                        if existing.source == CandidateSource::DirectTransliteration
                            || existing.source == CandidateSource::TypoFallback
                        {
                            existing.source = source;
                            existing.initial_boost = existing.initial_boost.max(initial_boost);
                        }
                    }
                }
            }
        };

        // 3b. Developer Code-Mixing Shield (e.g. "onClick", "user_id", "--help", "const", "https://...")
        let is_code = if self.config.enable_code_shield {
            (super::code_shield::is_code_token(term)
                || super::code_shield::is_code_token(middle))
                && !self.database.is_exact_dictionary_word(&phonetic)
                && !is_atomic_cv_syllable(&phonetic)
        } else {
            false
        };
        if is_code {
            add_cand(
                middle.to_string(),
                CandidateSource::CodeShield,
                30000,
                &mut raw_candidates,
                &mut seen,
            );
        }

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
                    if let Some((bn_loan, en_loan)) =
                        PhoneticDatabase::get_bilingual_loanword(base_en)
                    {
                        let actual_suffix = if suf_bn == "র"
                            && !bn_loan
                                .chars()
                                .last()
                                .is_some_and(|c| c.is_kar() || c.is_vowel())
                        {
                            "ের"
                        } else {
                            suf_bn
                        };
                        let combined_bn =
                            super::morphology::apply_sandhi_join(bn_loan, actual_suffix);
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

        let preferred_word = if let Some(raw_ac) = self.database.get_autocorrect_raw_filtered(middle, self.config.enable_banglish_shorthand) {
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
        } else if let Some(ref loan_pref) = preferred_loanword {
            Some(loan_pref.clone())
        } else if let Some(verbal_word) = if self.config.enable_colloquial_dialects {
            super::morphology::decompose_verbal_form(middle)
        } else {
            None
        } {
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
                if let Some(raw_ac) = self.database.get_autocorrect_raw_filtered(&collapsed, self.config.enable_banglish_shorthand) {
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
                4200,
                &mut raw_candidates,
                &mut seen,
            );
        } else {
            add_cand(
                primary.to_string(),
                CandidateSource::DirectTransliteration,
                1500,
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
                    1200,
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
                    3500,
                    &mut raw_candidates,
                    &mut seen,
                );
            }

            // 9b. Colloquial & Spoken continuous verb forms (e.g. kortesi -> করছি, করতেছি)
            if self.config.enable_colloquial_dialects {
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
                } else if middle.chars().count() <= 3
                    && !var_phonetic.is_empty()
                    && var_phonetic.chars().all(|c| c.is_bengali())
                {
                    add_cand(
                        var_phonetic,
                        CandidateSource::FuzzySoundLaw,
                        1800,
                        &mut raw_candidates,
                        &mut seen,
                    );
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

            // 12. Physical QWERTY Typo & Transposition Fallback
            let has_dict_cand = raw_candidates
                .iter()
                .any(|c| self.database.is_exact_dictionary_word(&c.text));
            if !has_dict_cand && middle.chars().count() >= 4 {
                let mut typo_matches: Vec<String> = Vec::new();

                // 12a. Physical QWERTY key adjacency slips and adjacent transpositions
                let qwerty_variants = super::fuzzy::generate_qwerty_typo_variants(middle);
                for qv in &qwerty_variants {
                    let conv = self.convert_phonetic(qv);
                    if self.database.is_exact_dictionary_word(&conv)
                        && !typo_matches.contains(&conv)
                    {
                        typo_matches.push(conv);
                    }
                    for fz in super::fuzzy::generate_phonetic_variants(qv) {
                        let fz_conv = self.convert_phonetic(&fz);
                        if self.database.is_exact_dictionary_word(&fz_conv)
                            && !typo_matches.contains(&fz_conv)
                        {
                            typo_matches.push(fz_conv);
                        }
                    }
                }

                // 12b. Single deletion typo fallback
                if typo_matches.is_empty() {
                    for (i, ch) in middle.char_indices() {
                        let mut del = String::with_capacity(middle.len());
                        del.push_str(&middle[..i]);
                        del.push_str(&middle[i + ch.len_utf8()..]);
                        let del_conv = self.convert_phonetic(&del);
                        if self.database.is_exact_dictionary_word(&del_conv)
                            && !typo_matches.contains(&del_conv)
                        {
                            typo_matches.push(del_conv);
                        }
                    }
                }

                typo_matches.sort_by(|a, b| {
                    let dist_a = super::fuzzy::damerau_levenshtein(middle, a);
                    let dist_b = super::fuzzy::damerau_levenshtein(middle, b);
                    dist_a.cmp(&dist_b).then_with(|| {
                        self.database
                            .get_frequency(b)
                            .cmp(&self.database.get_frequency(a))
                    })
                });
                for tm in typo_matches.into_iter().take(4) {
                    add_cand(
                        tm,
                        CandidateSource::TypoFallback,
                        900,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
        }

        // 12c. Concatenated Word Lattice Segmentation (e.g. "kemonaso" -> "কেমন আছো", "dhonnobadbhai" -> "ধন্যবাদ ভাই")
        if self.config.enable_word_segmentation && middle.len() >= 6 && !middle.contains(' ') {
            let segmented = super::segmenter::segment_concatenated_token(
                middle,
                &self.database,
                |s| self.convert_phonetic(s),
            );
            for seg in segmented {
                let boost = if is_primary_in_dict { 3000 } else { 12000 };
                add_cand(
                    seg.text,
                    CandidateSource::SegmentedLattice,
                    boost + (seg.score.min(2000)),
                    &mut raw_candidates,
                    &mut seen,
                );
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
        let has_backtick = middle.contains('`') || term.contains('`');
        let has_explicit_casing = middle.chars().any(|c| c.is_ascii_uppercase() && c != 'K');
        let has_explicit_rri_digraph = middle.contains("rri") || term.contains("rri");
        let is_atomic_syllable = is_atomic_cv_syllable(&phonetic);
        let is_short_token = middle.chars().count() <= 3;

        let has_dominant_homophone = raw_candidates.iter().any(|c| {
            c.source != CandidateSource::DirectTransliteration
                && c.source != CandidateSource::SegmentedLattice
                && c.source != CandidateSource::TypoFallback
                && c.source != CandidateSource::TriePrefixAutocomplete
                && c.text != phonetic
                && (c.source == CandidateSource::Autocorrect
                    || c.source == CandidateSource::Loanword
                    || self.database.get_frequency(&c.text) >= 1000)
        });

        let is_clitic_o_word = |w: &str| -> bool {
            w.ends_with('ও')
                || w == "এখনো"
                || w == "তখনো"
                || w == "কখনো"
                || w == "এমনো"
                || w == "কোনো"
        };

        let has_clitic_o_candidate = middle.ends_with('o')
            && !(middle.ends_with("yo") && raw_candidates.iter().any(|c| c.text.ends_with("্য")))
            && raw_candidates.iter().any(|c| {
                is_clitic_o_word(&c.text)
                    && self.database.is_exact_dictionary_word(&c.text)
                    && c.source != CandidateSource::TriePrefixAutocomplete
                    && c.source != CandidateSource::TypoFallback
            });

        let has_clitic_i_candidate = !has_explicit_rri_digraph
            && middle.ends_with('i')
            && raw_candidates.iter().any(|c| {
                c.text.ends_with('ই')
                    && self.database.is_exact_dictionary_word(&c.text)
                    && c.source != CandidateSource::TriePrefixAutocomplete
                    && c.source != CandidateSource::TypoFallback
            });

        let clitic_o_targets: Vec<String> = if has_clitic_o_candidate {
            raw_candidates
                .iter()
                .filter(|c| is_clitic_o_word(&c.text))
                .map(|c| c.text.clone())
                .collect()
        } else {
            Vec::new()
        };

        let clitic_i_targets: Vec<String> = if has_clitic_i_candidate {
            raw_candidates
                .iter()
                .filter(|c| c.text.ends_with('ই'))
                .map(|c| c.text.clone())
                .collect()
        } else {
            Vec::new()
        };

        let mut scored_candidates: Vec<(String, i32)> = Vec::with_capacity(raw_candidates.len());
        let learner_guard = self.database.learner.read().ok();

        for cand in raw_candidates {
            if cand.source == CandidateSource::CodeShield {
                scored_candidates.push((cand.text, 60000));
                continue;
            }
            if cand.source == CandidateSource::SegmentedLattice {
                let boost = if is_primary_in_dict || has_dominant_homophone || preferred_loanword.is_some() {
                    cand.initial_boost.min(3000)
                } else {
                    cand.initial_boost + 2000
                };
                scored_candidates.push((cand.text, boost));
                continue;
            }

            let is_in_dict = self.database.is_exact_dictionary_word(&cand.text);
            let freq = self.database.get_frequency(&cand.text);
            let mut score = cand.initial_boost;

            // 1. Exact Dictionary Status & Promotion Rule
            if is_in_dict || cand.source == CandidateSource::MorphologicalInflection || cand.source == CandidateSource::Autocorrect || cand.source == CandidateSource::Loanword {
                score += 2200;
            } else if cand.source == CandidateSource::DirectTransliteration {
                // If raw transliteration is NOT in the dictionary, but valid sound-law alternatives exist,
                // penalize full words so casual homophones (e.g. মানুষ for manus, চা for cha) win.
                // But NEVER penalize when explicit casing, backticks, or explicit rri digraph are present!
                if has_dominant_homophone
                    && !has_explicit_casing
                    && !has_backtick
                    && !has_explicit_rri_digraph
                    && (!is_atomic_syllable || !is_primary_in_dict)
                    && !is_short_token
                {
                    score -= 2800;
                }
            }

            // 2. Frequency bonus (logarithmic scaling)
            if freq > 0 {
                let freq_f = freq as f64;
                let log_freq = freq_f.log2();
                score += (log_freq * 130.0) as i32;
            }
            if freq >= 8000 {
                score += 2000;
            }
            if self.database.trie.contains_exact(&cand.text) {
                score += 1000;
            }

            // 3. Edit distance & length difference penalty from raw phonetic form
            if cand.source != CandidateSource::EmojiKeyword {
                let dist = edit_distance(&phonetic, &cand.text);
                if cand.text == phonetic || cand.text == *primary {
                    if is_in_dict || cand.source == CandidateSource::MorphologicalInflection || cand.source == CandidateSource::Autocorrect || cand.source == CandidateSource::Loanword {
                        score += 3500;
                    } else {
                        score += 1500;
                    }
                    if has_backtick {
                        score += 5000;
                    } else if has_explicit_casing || has_explicit_rri_digraph {
                        score += 3500;
                    } else if is_atomic_syllable && is_primary_in_dict {
                        score += 3000;
                    } else if is_short_token {
                        score += 1200;
                    }
                } else if cand.source == CandidateSource::TypoFallback {
                    // Typo fallback candidates alter the user's physical keypresses;
                    // apply motor slip penalty so intentional words aren't hijacked
                    score -= 4200;
                    score -= (dist.min(3) as i32) * 200;
                } else {
                    score -= (dist as i32) * 200;
                    let len_diff = (cand.text.chars().count() as isize
                        - phonetic.chars().count() as isize)
                        .abs() as i32;
                    score -= len_diff * 400;

                    if (is_primary_in_dict || has_explicit_rri_digraph)
                        && is_atomic_syllable
                        && dist > 0
                    {
                        score -= 3500;
                    }

                    if cand.source == CandidateSource::FuzzySoundLaw {
                        if has_backtick {
                            score -= 4000;
                        } else if has_explicit_casing {
                            score -= 2500;
                        } else if middle.chars().count() <= 2 && dist > 0 {
                            score -= 3000;
                        }
                    }

                    if cand.source == CandidateSource::Autocorrect {
                        if has_backtick {
                            score -= 5000;
                        } else if has_explicit_casing {
                            score -= 3500;
                        }
                    }
                }
            }

            // 3b. Clitic preservation vs root-drop resolution (prevent high-frequency roots from suppressing clitics)
            if has_clitic_o_candidate {
                let is_clitic_o = is_clitic_o_word(&cand.text);
                if is_clitic_o {
                    score += 3500;
                } else if !cand.text.ends_with("্য")
                    && clitic_o_targets.iter().any(|t| {
                        t.strip_prefix(&cand.text).map_or(false, |s| {
                            s == "ও"
                                || (s == "ো"
                                    && (t == "এখনো"
                                        || t == "তখনো"
                                        || t == "কখনো"
                                        || t == "এমনো"
                                        || t == "কোনো"))
                        })
                    })
                {
                    score -= 3000;
                }
            }

            if has_clitic_i_candidate {
                if cand.text.ends_with('ই') {
                    score += 3500;
                } else if clitic_i_targets.iter().any(|t| {
                    t.strip_prefix(&cand.text).map_or(false, |s| s == "ই")
                }) {
                    score -= 3000;
                }
            }

            // 4. Contextual Homophone & AI Language Model Scoring (Zero Allocations)
            if self.config.ai_profile != AiProfile::Off && !context.is_empty() && cand.source != CandidateSource::EmojiKeyword {
                let prev = context.last().copied();
                let prev_prev = if self.config.ai_profile == AiProfile::Balanced && context.len() >= 2 {
                    context.get(context.len() - 2).copied()
                } else {
                    None
                };

                let lm_score = self
                    .ai_context
                    .lm()
                    .score_candidate(prev_prev, prev, &cand.text);
                if lm_score > -0.5 {
                    score += 4800;
                } else if lm_score > -1.0 {
                    score += 4000;
                } else if lm_score > -2.0 {
                    score += 2400;
                } else if lm_score > -3.0 {
                    score += 1000;
                }

                // 5. Dynamic User Bigram Personalization Boost
                if let Some(p) = prev {
                    if let Some(ref l) = learner_guard {
                        let user_boost = l.get_user_bigram_boost(p, &cand.text);
                        score += user_boost;
                    }
                }
            }

            // 6. Candidate Memory & Autonomous Learner Boost
            if let Some(fav) = candidate_memory
                .get(term)
                .or_else(|| candidate_memory.get(middle))
                .or_else(|| candidate_memory.get(&phonetic))
                .or_else(|| learner_guard.as_ref().and_then(|l| l.candidate_memory.get(term)))
                .or_else(|| learner_guard.as_ref().and_then(|l| l.candidate_memory.get(middle)))
                .or_else(|| learner_guard.as_ref().and_then(|l| l.candidate_memory.get(&phonetic)))
            {
                if &cand.text == fav {
                    score += 5000;
                }
            }
            if let Some(ref l) = learner_guard {
                if l.learned_words.contains(&cand.text) {
                    score += 3500;
                }
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
            if (c == term && !is_code) || kw_emojis.contains(c) {
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

        if has_english && !candidates.contains(&term.to_string()) {
            candidates.push(term.to_string());
        }

        // Apply pre and post punctuation to candidates
        let mut final_candidates = Vec::with_capacity(candidates.len());
        let pre_converted = self.convert_phonetic(pre);
        let post_converted = self.convert_phonetic(post);

        for cand in candidates {
            let norm_cand = fix_chandra_bindu_position(&cand);
            if pre.is_empty() && post.is_empty() {
                final_candidates.push(norm_cand);
            } else {
                final_candidates.push(format!("{}{}{}", pre_converted, norm_cand, post_converted));
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
                .position(|c| c == fav)
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

        static DEFAULT_LM: std::sync::OnceLock<lekhani_ai::LanguageModel> =
            std::sync::OnceLock::new();
        let lm = DEFAULT_LM.get_or_init(lekhani_ai::LanguageModel::new);

        let score = lm.score_candidate(None, Some(prev), candidate);
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

    /// Observe a committed word and preceding context to learn personalization and new vocabulary
    pub fn observe_committed(
        &mut self,
        previous_word: Option<&str>,
        committed_word: &str,
    ) -> Vec<String> {
        if let Some(prev) = previous_word {
            if let Ok(mut l) = self.database.learner.write() {
                l.observe_committed_pair(prev, committed_word);
            }
        }
        self.database.observe_committed_word(committed_word)
    }

    /// Predict next probable Bengali words for zero-preedit state using the statistical language model
    pub fn suggest_next_words(&self, previous_word: &str) -> Vec<String> {
        let prev = previous_word.trim();
        if prev.is_empty() {
            return Vec::new();
        }

        self.suggest_next_words_with_context(&[prev])
    }

    /// Predict next words given multi-word sentence context (up to trigrams)
    pub fn suggest_next_words_with_context(&self, context: &[&str]) -> Vec<String> {
        if context.is_empty()
            || self.config.ai_profile == AiProfile::Off
            || !self.config.enable_phrase_prediction
        {
            return Vec::new();
        }
        let mut predictions = self.ai_predictor.predict_next_with_options(
            context,
            8,
            self.config.enable_phrase_prediction,
        );

        // Personalized user-learned continuations promotion
        if let Some(&last_word) = context.last() {
            let clean_last = last_word.trim_matches(|c: char| {
                c.is_ascii_punctuation() || c == '।' || c == '—' || c == ','
            });
            let user_conts = if let Ok(l) = self.database.learner.read() {
                l.get_top_user_continuations(clean_last, 4)
            } else {
                Vec::new()
            };
            for cont in user_conts.into_iter().rev() {
                if let Some(pos) = predictions.iter().position(|p| p == &cont) {
                    predictions.remove(pos);
                }
                predictions.insert(0, cont);
            }
        }

        // Instant Bengali Reduplication (দ্বিরুক্ত শব্দ) promotion
        if self.config.enable_reduplication {
            if let Some(&last_word) = context.last() {
                let clean_last = last_word.trim_matches(|c: char| {
                    c.is_ascii_punctuation() || c == '।' || c == '—' || c == ','
                });
                if super::reduplication::is_reduplicative_word(clean_last) {
                    let redup_str = clean_last.to_string();
                    if let Some(pos) = predictions.iter().position(|p| p == &redup_str) {
                        predictions.remove(pos);
                    }
                    predictions.insert(0, redup_str);
                }
            }
        }

        predictions.truncate(8);
        predictions
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
                // Suffix split point cannot cut strictly inside an atomic layout digraph
                if super::fuzzy::cuts_layout_digraph(middle, i) {
                    continue;
                }
                let suffix_key = &middle[i..];
                if let Some(suffix) = self.database.find_suffix(suffix_key) {
                    let base_key = &middle[..i];
                    let mut base_candidates = Vec::new();
                    let mut has_verified_base = false;

                    if let Some(ac) = self.database.get_autocorrect_raw_filtered(base_key, self.config.enable_banglish_shorthand) {
                        let conv = if ac.chars().any(|c| c.is_bengali()) {
                            ac
                        } else {
                            self.convert_phonetic(&ac)
                        };
                        if !conv.is_empty() && !base_candidates.contains(&conv) {
                            base_candidates.push(conv);
                            has_verified_base = true;
                        }
                    }

                    let base_phonetic = self.convert_phonetic(base_key);
                    let is_base_dict = self.database.is_exact_dictionary_word(&base_phonetic);
                    if is_base_dict {
                        has_verified_base = true;
                    }
                    if !base_candidates.contains(&base_phonetic) {
                        base_candidates.push(base_phonetic);
                    }

                    // Apply stem-level sound-laws (e.g. "manush" in "manusher" -> "মানুষ")
                    // Only permit fuzzy expansion if base is verified or a multi-syllable stem (>= 5 chars)
                    if has_verified_base || base_key.chars().count() >= 5 {
                        for fz in super::fuzzy::generate_phonetic_variants(base_key) {
                            let conv = self.convert_phonetic(&fz);
                            if self.database.is_exact_dictionary_word(&conv)
                                && !base_candidates.contains(&conv)
                            {
                                base_candidates.push(conv);
                            }
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
                            "jai" | "jawa" | "ja" => match exp {
                                "ছি" => "যাচ্ছি".to_string(),
                                "ছো" => "যাচ্ছ".to_string(),
                                "ছে" => "যাচ্ছে".to_string(),
                                "ছেন" => "যাচ্ছেন".to_string(),
                                "ছিলাম" => "যাচ্ছিলাম".to_string(),
                                "ছিলা" => "যাচ্ছিলা".to_string(),
                                "ছিল" => "যাচ্ছিল".to_string(),
                                "ছিলেন" => "যাচ্ছিলেন".to_string(),
                                "ব" => "যাব".to_string(),
                                "বা" => "যাবা".to_string(),
                                _ => format!("যা{}", exp),
                            },
                            "khai" | "khawa" | "kha" => match exp {
                                "ছি" => "খাচ্ছি".to_string(),
                                "ছো" => "খাচ্ছ".to_string(),
                                "ছে" => "খাচ্ছে".to_string(),
                                "ছেন" => "খাচ্ছেন".to_string(),
                                "ছিলাম" => "খাচ্ছিলাম".to_string(),
                                "ছিলা" => "খাচ্ছিলা".to_string(),
                                "ছিল" => "খাচ্ছিল".to_string(),
                                "ছিলেন" => "খাচ্ছিলেন".to_string(),
                                "ব" => "খাব".to_string(),
                                "বা" => "খাবা".to_string(),
                                _ => format!("খা{}", exp),
                            },
                            "de" | "di" => match exp {
                                "ছি" => "দিচ্ছি".to_string(),
                                "ছো" => "দিচ্ছ".to_string(),
                                "ছে" => "দিচ্ছে".to_string(),
                                "ছেন" => "দিচ্ছেন".to_string(),
                                "ছিলাম" => "দিচ্ছিলাম".to_string(),
                                "ছিল" => "দিচ্ছিল".to_string(),
                                "ব" => "দেব".to_string(),
                                _ => format!("দি{}", exp),
                            },
                            "ne" | "ni" => match exp {
                                "ছি" => "নিচ্ছি".to_string(),
                                "ছো" => "নিচ্ছ".to_string(),
                                "ছে" => "নিচ্ছে".to_string(),
                                "ছেন" => "নিচ্ছেন".to_string(),
                                "ছিলাম" => "নিচ্ছিলাম".to_string(),
                                "ছিল" => "নিচ্ছিল".to_string(),
                                "ব" => "নেব".to_string(),
                                _ => format!("নি{}", exp),
                            },
                            "ge" => match exp {
                                "ছি" => "গেছি".to_string(),
                                "েছি" => "গিয়েছি".to_string(),
                                "ছিলাম" => "গেছিলাম".to_string(),
                                "েছিলাম" => "গিয়েছিলাম".to_string(),
                                "ছিল" => "গেছিল".to_string(),
                                "েছিল" => "গিয়েছিল".to_string(),
                                _ => format!("গে{}", exp),
                            },
                            "bhab" | "vab" => match exp {
                                "ছি" | "চ্ছি" => "ভাবছি".to_string(),
                                "েছি" => "ভেবেছি".to_string(),
                                "ছিলাম" | "চ্ছিলাম" => "ভাবছিলাম".to_string(),
                                "েছিলাম" => "ভেবেছিলাম".to_string(),
                                "ব" => "ভাবব".to_string(),
                                _ => format!("ভাব{}", exp),
                            },
                            "bujh" => match exp {
                                "ছি" | "চ্ছি" => "বুঝছি".to_string(),
                                "েছি" => "বুঝেছি".to_string(),
                                "ছিলাম" | "চ্ছিলাম" => "বুঝছিলাম".to_string(),
                                "েছিলাম" => "বুঝেছিলাম".to_string(),
                                "ব" => "বুঝব".to_string(),
                                _ => format!("বুঝ{}", exp),
                            },
                            "ash" | "as" => match exp {
                                "ছি" | "চ্ছি" => "আসছি".to_string(),
                                "েছি" => "এসেছি".to_string(),
                                "ছিলাম" | "চ্ছিলাম" => "আসছিলাম".to_string(),
                                "েছিলাম" => "এসেছিলাম".to_string(),
                                "ব" => "আসব".to_string(),
                                _ => format!("আস{}", exp),
                            },
                            "bol" => match exp {
                                "ছি" | "চ্ছি" => "বলছি".to_string(),
                                "ছে" | "চ্ছে" => "বলছে".to_string(),
                                "ছো" | "চ্ছো" => "বলছো".to_string(),
                                "েছি" => "বলেছি".to_string(),
                                "ছিলাম" | "চ্ছিলাম" => "বলছিলাম".to_string(),
                                "েছিলাম" => "বলেছিলাম".to_string(),
                                "ছিল" | "চ্ছিল" => "বলছিল".to_string(),
                                "েছিল" => "বলেছিল".to_string(),
                                "ছেন" | "চ্ছেন" => "বলছেন".to_string(),
                                "েছেন" => "বলেছেন".to_string(),
                                "ব" => "বলব".to_string(),
                                _ => format!("বল{}", exp),
                            },
                            "kor" => match exp {
                                "ছি" | "চ্ছি" => "করছি".to_string(),
                                "ছে" | "চ্ছে" => "করছে".to_string(),
                                "ছো" | "চ্ছো" => "করছো".to_string(),
                                "েছি" => "করেছি".to_string(),
                                "ছিলাম" | "চ্ছিলাম" => "করছিলাম".to_string(),
                                "েছিলাম" => "করেছিলাম".to_string(),
                                "ছিল" | "চ্ছিল" => "করছিল".to_string(),
                                "েছিল" => "করেছিল".to_string(),
                                "ছিলা" | "চ্ছিলা" => "করছিলা".to_string(),
                                "েছিলা" => "করেছিলা".to_string(),
                                "ছেন" | "চ্ছেন" => "করছেন".to_string(),
                                "েছেন" => "করেছেন".to_string(),
                                "ব" => "করব".to_string(),
                                "বা" => "করবা".to_string(),
                                _ => format!("কর{}", exp),
                            },
                            "dekh" | "dek" => match exp {
                                "ছি" => "দেখছি".to_string(),
                                "েছি" => "দেখেছি".to_string(),
                                "ছিলাম" => "দেখছিলাম".to_string(),
                                "েছিলাম" => "দেখেছিলাম".to_string(),
                                "ব" => "দেখব".to_string(),
                                _ => format!("দেখ{}", exp),
                            },
                            "shun" | "sun" => match exp {
                                "ছি" => "শুনছি".to_string(),
                                "েছি" => "শুনেছি".to_string(),
                                "ছিলাম" => "শুনছিলাম".to_string(),
                                "েছিলাম" => "শুনেছিলাম".to_string(),
                                "ব" => "শুনব".to_string(),
                                _ => format!("শুন{}", exp),
                            },
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

/// Check if a Bengali string represents an atomic CV syllable (1 consonant + 1 kar) or a single independent vowel/consonant
pub fn is_atomic_cv_syllable(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return false;
    }
    if chars.len() == 1 {
        return chars[0].is_vowel() || chars[0].is_consonant();
    }
    if chars.len() == 2 {
        return chars[0].is_consonant() && chars[1].is_kar();
    }
    false
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
        println!("cands_book: {:?}", cands_book);
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
        assert!(
            next_ami.contains(&"মনে".to_string())
                || next_ami.contains(&"জানি".to_string())
                || next_ami.contains(&"আমার".to_string())
                || next_ami.contains(&"ভালো".to_string())
                || next_ami.contains(&"যাচ্ছি".to_string())
                || next_ami.contains(&"তোমাকে".to_string())
        );

        let next_thanks = sugg.suggest_next_words("ধন্যবাদ");
        assert!(!next_thanks.is_empty());
        assert!(
            next_thanks.contains(&"জানান".to_string())
                || next_thanks.contains(&"জানাই".to_string())
                || next_thanks.contains(&"তোমাকে".to_string())
                || next_thanks.contains(&"ভাই".to_string())
                || next_thanks.contains(&"আপনাকে".to_string())
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
        assert!(
            cands_kor.contains(&"করছি".to_string()) || cands_kor.contains(&"করতেছি".to_string())
        );

        let (cands_jai, _) = sugg.suggest("jaitasi", true, true, &empty_memory);
        assert!(
            cands_jai.contains(&"যাচ্ছি".to_string()) || cands_jai.contains(&"যাইতেছি".to_string())
        );
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
        assert_eq!(cands_dhai[0], "\u{0986}\u{09DC}\u{09BE}\u{0987}");

        // 8. Inflected Loanwords (computere -> কম্পিউটারে, fileti -> ফাইলটি)
        let (cands_comp, _) = sugg.suggest("computere", true, true, &empty_memory);
        assert!(cands_comp[0] == "কম্পিউটারে" || cands_comp.contains(&"কম্পিউটারে".to_string()));

        let (cands_file, _) = sugg.suggest("fileti", true, true, &empty_memory);
        assert!(cands_file[0] == "ফাইলটি" || cands_file.contains(&"ফাইলটি".to_string()));

        let (cands_phone, _) = sugg.suggest("smartphonete", true, true, &empty_memory);
        assert!(cands_phone[0] == "স্মার্টফোনে" || cands_phone.contains(&"স্মার্টফোনে".to_string()));
    }

    #[test]
    fn test_auto_dari_toggle() {
        let mut sugg = PhoneticSuggestion::new();
        let empty_memory = HashMap::new();

        // Default: auto_dari = true
        let (cands, _) = sugg.suggest("..", true, true, &empty_memory);
        assert_eq!(cands[0], "।");

        // Disabled: auto_dari = false
        sugg.config.auto_dari = false;
        let (cands_disabled, _) = sugg.suggest("..", true, true, &empty_memory);
        assert_ne!(cands_disabled.first().map(|s| s.as_str()), Some("।"));
    }

    #[test]
    fn test_malformed_layout_json_handling() {
        let mut sugg = PhoneticSuggestion::new();

        // 1. Null layout
        assert!(!sugg.set_layout(&serde_json::Value::Null));

        // 2. String layout
        assert!(!sugg.set_layout(&serde_json::json!("invalid layout string")));

        // 3. Object without patterns
        assert!(!sugg.set_layout(&serde_json::json!({"name": "broken"})));

        // 4. Object with non-array patterns
        assert!(!sugg.set_layout(&serde_json::json!({"patterns": 123})));
    }
}
