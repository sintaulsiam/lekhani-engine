//! Phonetic Suggestion Generation Engine

use hashbrown::HashMap;
use lekhani_parser::{CompiledLayout, LekhaniParser};
use serde_json::Value;

use super::database::PhoneticDatabase;
use super::ranking::{extract_candidate_features, CandidateContext, RankFeatures};
use crate::chars::BengaliCharExt;

use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateSource {
    PhoneticOverride,
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
    PragyanBeam,
    LpsLattice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneticGrammar {
    #[default]
    AvroClassic,
    Lps,
}

#[derive(Debug, Clone)]
pub struct CandidateHypothesis {
    pub text: String,
    pub source: CandidateSource,
    pub initial_boost: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiProfile {
    #[default]
    Balanced,
    Lite,
    Off,
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
    pub enable_bilingual_clitic: bool,
    pub strict_avro_mode: bool,
    pub enable_pragyan: bool,
    pub phonetic_grammar: PhoneticGrammar,
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
            enable_bilingual_clitic: false,
            strict_avro_mode: false,
            enable_pragyan: true,
            phonetic_grammar: PhoneticGrammar::default(),
        }
    }
}

pub type CachedSuggestion = (
    Vec<String>,
    Vec<RankFeatures>,
    Vec<lekhani_parser::lps::LpsLetterChip>,
);

#[derive(Clone)]
pub struct PhoneticSuggestion {
    pub database: PhoneticDatabase,
    phonetic_parser: Option<Arc<LekhaniParser>>,
    cache: HashMap<String, CachedSuggestion>,
    pub ai_context: lekhani_ai::ContextScorer,
    pub ai_predictor: lekhani_ai::NextWordPredictor,
    pub ai_decoder: lekhani_ai::BeamSearchDecoder,
    pub config: PhoneticSuggestionConfig,
    pub neural_predictor: Option<Arc<lekhani_neural::NeuralContextPredictor>>,
    pub pragyan_engine: Option<Arc<crate::phonetic::pragyan::PragyanEngine>>,
    /// Seq2Seq character-level transliterator safety net
    pub seq2seq: Option<Arc<lekhani_neural::Seq2SeqTransliterator>>,
    /// Pre-allocated inference buffers for seq2seq (avoids per-call heap allocation)
    seq2seq_scratch: Option<lekhani_neural::InferenceScratch>,
    /// Generation number mirroring database.generation. When they diverge the
    /// local cache is stale and must be cleared before the next lookup.
    cache_generation: u64,
    /// Features computed for the most recently ranked candidates
    last_computed_features: Vec<RankFeatures>,
    /// Candidates computed for the most recently ranked candidates
    last_computed_candidates: Vec<String>,
    /// Letter chips computed from the most recent LPS parse
    pub last_lps_chips: Vec<lekhani_parser::lps::LpsLetterChip>,
}

fn probe_neural_predictor() -> Option<Arc<lekhani_neural::NeuralContextPredictor>> {
    let mut candidate_dirs: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(env_dir) = std::env::var("LEKHANI_MODELS_DIR") {
        candidate_dirs.push(std::path::PathBuf::from(env_dir));
    }
    candidate_dirs.extend([
        std::path::PathBuf::from("data/dictionaries"),
        std::path::PathBuf::from("../data/dictionaries"),
        std::path::PathBuf::from("../../data/dictionaries"),
        std::path::PathBuf::from("/usr/local/share/lekhani/dictionaries"),
        std::path::PathBuf::from("/usr/share/lekhani/dictionaries"),
    ]);

    for dir in &candidate_dirs {
        let p_model = dir.join("bengali_gru_v2.bin");
        let p_vocab = dir.join("bengali_vocab_v2.json");
        if p_model.exists() && p_vocab.exists() {
            if let (Ok(model), Ok(vocab)) = (
                lekhani_neural::MicroGruModel::load_binary(&p_model),
                lekhani_neural::BpeVocabulary::load_json(&p_vocab),
            ) {
                if model.vocab_size() == vocab.len() {
                    if let Ok(pred) = lekhani_neural::NeuralContextPredictor::try_new(
                        Arc::new(model),
                        Arc::new(vocab),
                    ) {
                        return Some(Arc::new(pred));
                    }
                }
            }
        }
    }
    None
}

fn probe_pragyan_engine() -> Option<Arc<crate::phonetic::pragyan::PragyanEngine>> {
    crate::phonetic::pragyan::PragyanEngine::probe()
}

fn probe_seq2seq() -> Option<(Arc<lekhani_neural::Seq2SeqTransliterator>, lekhani_neural::InferenceScratch)> {
    let mut candidate_dirs: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(env_dir) = std::env::var("LEKHANI_MODELS_DIR") {
        candidate_dirs.push(std::path::PathBuf::from(env_dir));
    }
    candidate_dirs.extend([
        std::path::PathBuf::from("data/dictionaries"),
        std::path::PathBuf::from("../data/dictionaries"),
        std::path::PathBuf::from("../../data/dictionaries"),
        std::path::PathBuf::from("/usr/local/share/lekhani/dictionaries"),
        std::path::PathBuf::from("/usr/share/lekhani/dictionaries"),
    ]);
    for dir in &candidate_dirs {
        let p = dir.join("seq2seq_transliterator.bin");
        if p.exists() {
            if let Ok(model) = lekhani_neural::Seq2SeqTransliterator::load(&p) {
                let scratch = lekhani_neural::InferenceScratch::for_model(&model);
                return Some((Arc::new(model), scratch));
            }
        }
    }
    None
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
        let mut sugg = Self {
            database: PhoneticDatabase::new(),
            phonetic_parser: Some(Arc::new(lekhani_parser::default_avro_parser())),
            cache: HashMap::new(),
            ai_context: lekhani_ai::ContextScorer::with_language_model(lm.clone()),
            ai_predictor: lekhani_ai::NextWordPredictor::with_language_model(lm.clone()),
            ai_decoder: lekhani_ai::BeamSearchDecoder::with_language_model(lm, 4),
            config: PhoneticSuggestionConfig::default(),
            neural_predictor: probe_neural_predictor(),
            pragyan_engine: probe_pragyan_engine(),
            seq2seq: None,
            seq2seq_scratch: None,
            cache_generation: 0,
            last_computed_features: Vec::new(),
            last_computed_candidates: Vec::new(),
            last_lps_chips: Vec::new(),
        };
        // Lazy-probe seq2seq after struct is built (avoids startup cost if model absent)
        if let Some((s2s, scratch)) = probe_seq2seq() {
            sugg.seq2seq = Some(s2s);
            sugg.seq2seq_scratch = Some(scratch);
        }
        sugg.sync_pragyan_dictionary();
        sugg
    }

    pub fn set_neural_predictor(&mut self, predictor: Arc<lekhani_neural::NeuralContextPredictor>) {
        self.neural_predictor = Some(predictor);
        self.cache.clear();
    }

    pub fn with_neural_predictor(
        mut self,
        predictor: Arc<lekhani_neural::NeuralContextPredictor>,
    ) -> Self {
        self.neural_predictor = Some(predictor);
        self
    }

    pub fn set_pragyan_engine(&mut self, engine: Arc<crate::phonetic::pragyan::PragyanEngine>) {
        self.pragyan_engine = Some(engine);
        self.sync_pragyan_dictionary();
        self.cache.clear();
    }

    pub fn with_pragyan_engine(
        mut self,
        engine: Arc<crate::phonetic::pragyan::PragyanEngine>,
    ) -> Self {
        self.pragyan_engine = Some(engine);
        self.sync_pragyan_dictionary();
        self
    }

    pub fn sync_pragyan_dictionary(&mut self) {
        if let Some(ref mut engine) = self.pragyan_engine {
            let mut new_engine = (**engine).clone();
            new_engine.sync_dictionary(&self.database.trie);
            *engine = Arc::new(new_engine);
        }
    }

    pub fn load_from_dir<P: AsRef<std::path::Path>>(
        &mut self,
        dir: P,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let res = self.database.load_from_dir(dir);
        self.sync_pragyan_dictionary();
        self.cache.clear();
        res
    }

    pub fn load_from_dir_with_dict<P: AsRef<std::path::Path>>(
        &mut self,
        dir: P,
        dict_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let res = self.database.load_from_dir_with_dict(dir, dict_name);
        self.sync_pragyan_dictionary();
        self.cache.clear();
        res
    }

    /// Check if a layout JSON structure contains valid phonetic patterns
    pub fn is_valid_phonetic_layout(layout_json: &Value) -> bool {
        let layout_obj = if let Some(l) = layout_json.get("layout") {
            l
        } else {
            layout_json
        };
        layout_obj.is_object() && layout_obj.get("patterns").is_some_and(|p| p.is_array())
    }

    pub fn with_layout(layout_json: &Value) -> Self {
        let layout_obj = if let Some(l) = layout_json.get("layout") {
            l
        } else {
            layout_json
        };
        let parser = if Self::is_valid_phonetic_layout(layout_json) {
            CompiledLayout::from_json(layout_obj)
                .ok()
                .map(|c| Arc::new(LekhaniParser::new(Arc::new(c))))
        } else {
            None
        };
        let lm = lekhani_ai::LanguageModel::new();
        let mut sugg = Self {
            database: PhoneticDatabase::new(),
            phonetic_parser: parser,
            cache: HashMap::new(),
            ai_context: lekhani_ai::ContextScorer::with_language_model(lm.clone()),
            ai_predictor: lekhani_ai::NextWordPredictor::with_language_model(lm.clone()),
            ai_decoder: lekhani_ai::BeamSearchDecoder::with_language_model(lm, 4),
            config: PhoneticSuggestionConfig::default(),
            neural_predictor: probe_neural_predictor(),
            pragyan_engine: probe_pragyan_engine(),
            seq2seq: None,
            seq2seq_scratch: None,
            cache_generation: 0,
            last_computed_features: Vec::new(),
            last_computed_candidates: Vec::new(),
            last_lps_chips: Vec::new(),
        };
        if let Some((s2s, scratch)) = probe_seq2seq() {
            sugg.seq2seq = Some(s2s);
            sugg.seq2seq_scratch = Some(scratch);
        }
        sugg.sync_pragyan_dictionary();
        sugg
    }

    pub fn set_layout(&mut self, layout_json: &Value) -> bool {
        let layout_obj = if let Some(l) = layout_json.get("layout") {
            l
        } else {
            layout_json
        };
        if Self::is_valid_phonetic_layout(layout_json) {
            if let Ok(compiled) = CompiledLayout::from_json(layout_obj) {
                self.phonetic_parser = Some(Arc::new(LekhaniParser::new(Arc::new(compiled))));
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

    pub fn get_config(&self) -> PhoneticSuggestionConfig {
        self.config
    }

    /// Clear cached candidate suggestions
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    pub fn get_last_computed_features(&self) -> &[RankFeatures] {
        &self.last_computed_features
    }

    pub fn get_last_computed_candidates(&self) -> &[String] {
        &self.last_computed_candidates
    }

    /// Returns the letter-level disambiguation chips computed from the most recent LPS parse.
    pub fn last_lps_chips(&self) -> &[lekhani_parser::lps::LpsLetterChip] {
        &self.last_lps_chips
    }

    /// Generates canonical exact LPS Latin form for given Bengali text (Section 12 exactness guarantee).
    pub fn exact_form(&self, text: &str) -> String {
        lekhani_parser::lps::exact(text)
    }

    /// Parses text directly through the LPS v1.0 engine, returning candidates and diagnostics.
    pub fn parse_lps(
        &self,
        text: &str,
    ) -> (Vec<lekhani_parser::lps::LpsCandidate>, Vec<lekhani_parser::lps::LpsWarning>) {
        lekhani_parser::lps::LpsEngine::new().parse(text)
    }

    pub fn dictionary_path(&self) -> Option<&std::path::Path> {
        self.database.dictionary_path.as_deref()
    }

    pub fn dictionary_hash(&self) -> Option<&str> {
        self.database.dictionary_hash.as_deref()
    }

    /// Transliterate directly using Avro phonetic rules
    pub fn convert_phonetic(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

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

        if self.config.phonetic_grammar == PhoneticGrammar::Lps {
            let (cands, _) = lekhani_parser::lps::LpsEngine::new().parse(text);
            if let Some(first) = cands.first() {
                return first.text.clone();
            }
        }

        if let Some(ref parser) = self.phonetic_parser {
            return parser.convert(text);
        }
        text.to_string()
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
        self.suggest_with_multi_context_bidirectional(
            term,
            context,
            None,
            include_english,
            use_dictionary,
            candidate_memory,
        )
    }

    /// Generate ranked candidates for typed term with bidirectional context (preceding + optional following word)
    pub fn suggest_with_multi_context_bidirectional(
        &mut self,
        term: &str,
        context: &[&str],
        right_word: Option<&str>,
        include_english: bool,
        use_dictionary: bool,
        candidate_memory: &HashMap<String, String>,
    ) -> (Vec<String>, usize) {
        if term.is_empty() {
            return (Vec::new(), 0);
        }

        self.last_lps_chips.clear();

        // Invalidate stale cache if the database was mutated since our last call.
        // One relaxed atomic load per keystroke — no mutex, no contention.
        let db_gen = self
            .database
            .generation
            .load(std::sync::atomic::Ordering::Relaxed);
        if db_gen != self.cache_generation {
            self.cache.clear();
            self.cache_generation = db_gen;
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
            | ((self.config.enable_dynamic_macros as u32) << 4)
            | ((self.config.strict_avro_mode as u32) << 5)
            | (((self.config.phonetic_grammar as u32) & 0x3) << 6);

        let cache_key = if context.is_empty() {
            format!(
                "{}:{}:{}:{}",
                term, include_english, use_dictionary, cfg_mask
            )
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
        if let Some((cached_cands, cached_feats, cached_chips)) = self.cache.get(&cache_key) {
            self.last_computed_features = cached_feats.clone();
            self.last_computed_candidates = cached_cands.clone();
            self.last_lps_chips = cached_chips.clone();
            let selected_index = if let Some(fav) = candidate_memory
                .get(term)
                .or_else(|| candidate_memory.get(&cache_key))
            {
                cached_cands
                    .iter()
                    .position(|c| c == fav || c.contains(fav))
                    .unwrap_or(0)
            } else {
                0
            };
            return (cached_cands.clone(), selected_index);
        }

        // 2. Direct whole-term special matches (Math `=125*8`, Currency `#usd50`, Snippets `!shubhechha`, Exact Emojis `:smile:`, `:)`, `$$`, `*taka*`)
        if self.config.enable_dynamic_macros || !term.starts_with('!') {
            let literals = self.database.search_special_literals(term);
            if !literals.is_empty() {
                self.last_computed_features.clear();
                self.last_computed_candidates.clear();
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
                self.last_computed_features.clear();
                self.last_computed_candidates.clear();
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
                self.last_computed_features.clear();
                self.last_computed_candidates.clear();
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
            self.last_computed_features.clear();
            self.last_computed_candidates.clear();
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
                } else if source == CandidateSource::Autocorrect
                    || source == CandidateSource::PhoneticOverride
                    || source == CandidateSource::MorphologicalInflection
                    || source == CandidateSource::ExactDictionary
                    || source == CandidateSource::PragyanBeam
                {
                    if let Some(existing) = raw_candidates.iter_mut().find(|c| c.text == text) {
                        let can_upgrade = match source {
                            CandidateSource::Autocorrect | CandidateSource::PhoneticOverride => true,
                            _ => matches!(
                                existing.source,
                                CandidateSource::DirectTransliteration
                                    | CandidateSource::TypoFallback
                                    | CandidateSource::PragyanBeam
                            ),
                        };
                        if can_upgrade {
                            if matches!(source, CandidateSource::Autocorrect | CandidateSource::PhoneticOverride) {
                                existing.source = source;
                                existing.initial_boost = existing.initial_boost.max(initial_boost);
                            } else if initial_boost > existing.initial_boost {
                                existing.source = source;
                                existing.initial_boost = initial_boost;
                            }
                        }
                    }
                }
            }
        };

        // 3b. Developer Code-Mixing Shield (e.g. "onClick", "user_id", "--help", "const", "https://...", "CCA")
        let is_all_caps_acronym = term.len() >= 2
            && term.chars().all(|c| c.is_ascii_uppercase())
            && !lekhani_parser::has_explicit_avro_intent(term);

        let is_code = if self.config.enable_code_shield || is_all_caps_acronym {
            (super::code_shield::is_code_token(term)
                || super::code_shield::is_code_token(middle)
                || is_all_caps_acronym)
                && !self.database.is_exact_dictionary_word(&phonetic)
                && !is_atomic_cv_syllable(&phonetic)
        } else {
            false
        };
        if is_code {
            add_cand(
                term.to_string(),
                CandidateSource::CodeShield,
                30000,
                &mut raw_candidates,
                &mut seen,
            );
        }

        // 3a. Pragyan Statistical Phonetic Decoding (Mode 2: Smart Adaptive)
        if !self.config.strict_avro_mode && self.config.enable_pragyan && !is_code {
            if let Some(ref engine) = self.pragyan_engine {
                let pragyan_cands = engine.suggest(middle, crate::session::ShiftOrigin::AutoCapitalize);
                for (idx, cand) in pragyan_cands.into_iter().take(4).enumerate() {
                    let boost = 5500 - (idx as i32 * 300);
                    add_cand(
                        cand,
                        CandidateSource::PragyanBeam,
                        boost,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
        }

        // 3a-2. LPS Grammar Engine (Lekhani Phonetic Standard v1.0)
        if self.config.phonetic_grammar == PhoneticGrammar::Lps && !is_code {
            let (lps_cands, _) = lekhani_parser::lps::LpsEngine::new().parse(middle);
            for (idx, cand) in lps_cands.into_iter().enumerate() {
                if self.last_lps_chips.is_empty() && !cand.chips.is_empty() {
                    self.last_lps_chips = cand.chips.clone();
                }
                let boost = 6000 - (cand.rule_cost as i32 * 500) - (idx as i32 * 150);
                add_cand(
                    cand.text,
                    CandidateSource::LpsLattice,
                    boost,
                    &mut raw_candidates,
                    &mut seen,
                );
            }
        }

        // 3c. Supervised Phonetic Overrides (e.g. colloquialisms, texting slang, proper nouns)
        let is_shorthand = self.database.shorthand.contains_key(middle)
            || self.database.shorthand.contains_key(&middle.to_lowercase());
        let allow_override = if is_shorthand {
            self.config.enable_banglish_shorthand && !post.contains('`') && !term.contains('`')
        } else {
            !post.contains('`') && !term.contains('`')
        };

        let mut preferred_override = None;
        let mut is_contextual_override = false;
        if allow_override {
            let ov_opt = self.database.lookup_override(middle).or_else(|| {
                let lower = middle.to_lowercase();
                if lower != middle {
                    let is_title_case = middle.len() >= 3
                        && middle.chars().next().is_some_and(|c| c.is_uppercase())
                        && middle
                            .chars()
                            .skip(1)
                            .all(|c| c.is_lowercase() || !c.is_alphabetic());
                    if is_title_case {
                        self.database.lookup_override(&lower)
                    } else {
                        None
                    }
                } else {
                    None
                }
            });
            if let Some(overrides) = ov_opt {
                if let Some((top_ov, top_conf)) = overrides.first() {
                    let is_valid_override = top_conf >= 0.70
                        || (top_conf >= 0.50
                            && (self.database.is_exact_dictionary_word(top_ov)
                                || self.database.trie.contains_exact(top_ov))
                            && !self.database.is_exact_dictionary_word(&phonetic));
                    if is_valid_override {
                        preferred_override = Some(top_ov.to_string());
                    }
                }
                for (ov_cand, conf) in overrides {
                    let boost = (7000.0 * conf) as i32;
                    add_cand(
                        ov_cand.to_string(),
                        CandidateSource::PhoneticOverride,
                        boost,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
        }

        // 3d. Project "অন্বয়" (Anvaya) — Dynamic Concordant Verb Synthesis
        let anvaya_state = lekhani_ai::AnvayaAgreementEngine::extract_state(context);
        let mut concordant_verb_str: Option<String> = None;
        if anvaya_state.honorific_tier != lekhani_ai::HonorificRegister::Neutral {
            if let Some(concordant_verb) =
                lekhani_ai::AnvayaAgreementEngine::synthesize_concordant_verb(
                    middle,
                    anvaya_state.honorific_tier,
                )
            {
                concordant_verb_str = Some(concordant_verb.to_string());
                add_cand(
                    concordant_verb.to_string(),
                    CandidateSource::ExactDictionary,
                    8500,
                    &mut raw_candidates,
                    &mut seen,
                );
            }
        }

        // 3e. Project "রূপমূল" (Rupmul) — Multi-Tier Agglutinative DAG Verification
        if preferred_override.is_none() {
            let rupmul_synth = crate::morpheme::RupmulGraphSynthesizer::new();
            if let Some(decomp) = rupmul_synth.decompose_dag(&phonetic) {
                if decomp.root.chars().count() >= 2
                    && (self.database.is_exact_dictionary_word(&decomp.root)
                        || self.database.trie.contains_exact(&decomp.root))
                {
                    add_cand(
                        phonetic.clone(),
                        CandidateSource::MorphologicalInflection,
                        4000,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
        }

        // 4. Bilingual Loanword Code-Mixing (e.g. "meeting" -> "মিটিং", "meeting")
        let mut preferred_loanword = None;
        let mut is_tech_collocation = false;
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
            is_tech_collocation = if let Some(&last_ctx) = context.last() {
                crate::bilingual::is_english_tech_collocation(last_ctx, middle)
            } else {
                false
            };

            if is_tech_collocation {
                // Add English form as CodeShield-level candidate so it wins slot 0 and passes the
                // term-exclusion filter in the final assembly (line: `c == term && !is_code`).
                add_cand(
                    en_loan.to_string(),
                    CandidateSource::CodeShield,
                    60000,
                    &mut raw_candidates,
                    &mut seen,
                );
                add_cand(
                    bn_loan.to_string(),
                    CandidateSource::Loanword,
                    7000,
                    &mut raw_candidates,
                    &mut seen,
                );
                preferred_loanword = Some(en_loan.to_string());
                preferred_override = Some(en_loan.to_string());
                is_contextual_override = true;
            } else {
                let is_bengali_tech_collocation = if let Some(&last_ctx) = context.last() {
                    crate::bilingual::is_bengali_tech_collocation(last_ctx, middle)
                } else {
                    false
                };
                let boost = if is_bengali_tech_collocation { 15000 } else { 6500 };
                add_cand(
                    bn_loan.to_string(),
                    CandidateSource::Loanword,
                    boost,
                    &mut raw_candidates,
                    &mut seen,
                );
                preferred_loanword = Some(bn_loan.to_string());
                if is_bengali_tech_collocation {
                    preferred_override = Some(bn_loan.to_string());
                    is_contextual_override = true;
                }
                if include_english {
                    add_cand(
                        en_loan.to_string(),
                        CandidateSource::Loanword,
                        -1500,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
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

        // 4b. Generalized Bilingual Light-Verb & State Agreement (LVC Syntax: Foreign Noun + Light Verb)
        if let Some(&last_ctx) = context.last() {
            let is_foreign_or_loan = (last_ctx.len() >= 2 && last_ctx.chars().all(|c| c.is_ascii_alphabetic()))
                || PhoneticDatabase::get_bilingual_loanword(last_ctx).is_some();
            if is_foreign_or_loan {
                let middle_lower = middle.to_lowercase();
                if let Some(light_verb) = match middle_lower.as_str() {
                    "koro" => Some("করো"),
                    "koren" => Some("করেন"),
                    "koris" => Some("করিস"),
                    "pathaw" | "pathao" => Some("পাঠাও"),
                    "dao" => Some("দাও"),
                    "den" => Some("দেন"),
                    "nen" => Some("নেন"),
                    "on" => Some("অন"),
                    "off" => Some("অফ"),
                    _ => None,
                } {
                    add_cand(
                        light_verb.to_string(),
                        CandidateSource::ExactDictionary,
                        9500,
                        &mut raw_candidates,
                        &mut seen,
                    );
                    preferred_override = Some(light_verb.to_string());
                    is_contextual_override = true;
                }
            }
        }
        if middle.eq_ignore_ascii_case("pathaw") {
            add_cand(
                "পাঠাও".to_string(),
                CandidateSource::PhoneticOverride,
                8800,
                &mut raw_candidates,
                &mut seen,
            );
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

        // 5b. Bilingual Digit-Units, Hyphenated Clitics & Unhyphenated Clitics
        if let Some((digit_primary, digit_cands)) =
            crate::bilingual::resolve_digit_unit_candidates(middle)
        {
            add_cand(
                digit_primary,
                CandidateSource::PhoneticOverride,
                9500,
                &mut raw_candidates,
                &mut seen,
            );
            for c in digit_cands {
                add_cand(
                    c,
                    CandidateSource::PhoneticOverride,
                    8500,
                    &mut raw_candidates,
                    &mut seen,
                );
            }
        }
        if self.config.enable_bilingual_clitic {
            if let Some((hyphen_primary, hyphen_cands)) =
                crate::bilingual::resolve_bilingual_hyphen_clitic(middle)
            {
                add_cand(
                    hyphen_primary,
                    CandidateSource::PhoneticOverride,
                    9500,
                    &mut raw_candidates,
                    &mut seen,
                );
                for c in hyphen_cands {
                    add_cand(
                        c,
                        CandidateSource::PhoneticOverride,
                        8500,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
            if let Some((unhyphen_primary, unhyphen_cands)) =
                crate::bilingual::resolve_unhyphenated_bilingual_clitic(middle)
            {
                add_cand(
                    unhyphen_primary,
                    CandidateSource::PhoneticOverride,
                    9500,
                    &mut raw_candidates,
                    &mut seen,
                );
                for c in unhyphen_cands {
                    add_cand(
                        c,
                        CandidateSource::PhoneticOverride,
                        8500,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
        }

        // 6. Autocorrect / Common Overrides / Loanwords / Verbal Decomposition / Elongation Collapse
        let resolve_ac = |raw: &str| -> String {
            if raw.chars().any(|c| c.is_bengali()) {
                raw.to_string()
            } else {
                self.convert_phonetic(raw)
            }
        };

        let learner_guard = self.database.learner.try_read().ok();
        let preferred_word = if let Some(fav) = candidate_memory
            .get(term)
            .or_else(|| candidate_memory.get(middle))
            .or_else(|| {
                learner_guard
                    .as_ref()
                    .and_then(|l| l.candidate_memory.get(term))
            })
            .or_else(|| {
                learner_guard
                    .as_ref()
                    .and_then(|l| l.candidate_memory.get(middle))
            }) {
            Some(fav.clone())
        } else if is_tech_collocation
            && preferred_loanword
                .as_ref()
                .map(|l| !l.chars().any(|c| c.is_bengali()))
                .unwrap_or(false)
        {
            Some(preferred_loanword.as_ref().unwrap().clone())
        } else if let Some(raw_ac) = self
            .database
            .get_autocorrect_raw_filtered(middle, self.config.enable_banglish_shorthand)
        {
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
        } else if let Some(ref ov_pref) = preferred_override {
            if !self.database.is_exact_dictionary_word(&phonetic) {
                Some(ov_pref.clone())
            } else {
                None
            }
        } else if let Some(ref loan_pref) = preferred_loanword {
            Some(loan_pref.clone())
        } else if let Some(morph_pref) =
            if use_dictionary && !self.database.is_exact_dictionary_word(&phonetic) {
                self.decompose_stem_suffix_override(middle)
            } else {
                None
            }
        {
            add_cand(
                morph_pref.clone(),
                CandidateSource::ExactDictionary,
                6400,
                &mut raw_candidates,
                &mut seen,
            );
            Some(morph_pref)
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
        } else if let Some((dialect_hyp, compound_hyp)) = if self.config.enable_colloquial_dialects {
            let resolve_stem = |stem: &str| -> Option<String> {
                let conv = if let Some(ref p) = self.phonetic_parser {
                    p.convert(stem)
                } else {
                    stem.to_string()
                };
                if !conv.is_empty()
                    && (self.database.is_exact_dictionary_word(&conv)
                        || self.database.trie.contains_exact(&conv)
                        || self.database.lookup_override(stem).is_some()
                        || self.database.get_frequency(&conv) > 0)
                {
                    Some(conv)
                } else {
                    None
                }
            };
            let d = super::morphology::decompose_dialect_verbal_token(middle, resolve_stem);
            let c = super::morphology::decompose_compound_token(middle, resolve_stem);
            if d.is_some() || c.is_some() {
                Some((d, c))
            } else {
                None
            }
        } else {
            None
        } {
            let mut top = None;
            if let Some(d) = dialect_hyp {
                add_cand(
                    d.dialect.clone(),
                    CandidateSource::ExactDictionary,
                    6500,
                    &mut raw_candidates,
                    &mut seen,
                );
                if d.standard != d.dialect {
                    add_cand(
                        d.standard,
                        CandidateSource::ExactDictionary,
                        5500,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
                top = Some(d.dialect);
            }
            if let Some(c) = compound_hyp {
                add_cand(
                    c.unified.clone(),
                    CandidateSource::ExactDictionary,
                    6500,
                    &mut raw_candidates,
                    &mut seen,
                );
                if let Some(hyphenated) = c.hyphenated {
                    if hyphenated != c.unified {
                        add_cand(
                            hyphenated,
                            CandidateSource::ExactDictionary,
                            6000,
                            &mut raw_candidates,
                            &mut seen,
                        );
                    }
                }
                if top.is_none() {
                    top = Some(c.unified);
                }
            }
            top
        } else {
            let mut found_collapsed = None;
            for collapsed in super::fuzzy::collapse_elongated_runs(middle) {
                if let Some(raw_ac) = self
                    .database
                    .get_autocorrect_raw_filtered(&collapsed, self.config.enable_banglish_shorthand)
                {
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
                } else if let Some(ov) = self.database.lookup_override(&collapsed) {
                    if let Some((top_ov, _)) = ov.first() {
                        let top_ov_str = top_ov.to_string();
                        add_cand(
                            top_ov_str.clone(),
                            CandidateSource::PhoneticOverride,
                            6000,
                            &mut raw_candidates,
                            &mut seen,
                        );
                        found_collapsed = Some(top_ov_str);
                        break;
                    }
                }
            }
            if found_collapsed.is_none() {
                for collapsed in super::fuzzy::collapse_elongated_runs(middle) {
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
            }
            found_collapsed
        };

        let primary_raw = preferred_word.as_deref().unwrap_or(&phonetic);
        let primary_norm = fix_chandra_bindu_position(primary_raw);
        let primary = primary_norm.as_str();
        let is_primary_in_dict = self.database.is_exact_dictionary_word(primary)
            || preferred_word.is_some();

        // 7. Direct Transliterations
        let has_double_consonant = middle.contains("cc")
            || middle.contains("kk")
            || middle.contains("tt")
            || middle.contains("dd")
            || middle.contains("bb")
            || middle.contains("pp")
            || middle.contains("mm")
            || middle.contains("nn")
            || middle.contains("ll")
            || middle.contains("ss")
            || middle.contains("jj")
            || middle.contains("gg");
        let is_explicit_geminate = has_double_consonant && primary.contains('্');

        if is_primary_in_dict || is_explicit_geminate {
            add_cand(
                primary.to_string(),
                if is_primary_in_dict {
                    CandidateSource::ExactDictionary
                } else {
                    CandidateSource::DirectTransliteration
                },
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

        // 7b. Initial Vowel 'a' -> 'অ' Ambiguity Resolution (Natural Banglish Digraph Invariant)
        // Natural romanization frequently uses 'a' for both 'আ' and 'অ' (e.g. anurodh -> অনুরোধ,
        // anchale -> অঞ্চলে, akkhar -> অক্ষর, artho -> অর্থ).
        let is_a_initial = middle.len() >= 2
            && (middle.starts_with('a') || middle.starts_with('A'))
            && !middle.starts_with("aa")
            && !middle.starts_with("Aa")
            && !middle.starts_with("aA")
            && !middle.starts_with("AA");

        if is_a_initial && !self.config.strict_avro_mode {
            let o_latin = format!("o{}", &middle[1..]);
            let o_phonetic = self.convert_phonetic(&o_latin);
            let is_o_phon_dict = self.database.is_exact_dictionary_word(&o_phonetic)
                || self.database.trie.contains_exact(&o_phonetic);
            if is_o_phon_dict {
                add_cand(
                    o_phonetic.clone(),
                    CandidateSource::ExactDictionary,
                    3900,
                    &mut raw_candidates,
                    &mut seen,
                );
            }

            let has_a_or_e = phonetic.starts_with('আ') || phonetic.starts_with('এ');
            if has_a_or_e {
                let p_len = if phonetic.starts_with('আ') {
                    'আ'.len_utf8()
                } else {
                    'এ'.len_utf8()
                };
                let o_cand = format!("অ{}", &phonetic[p_len..]);
                if (self.database.is_exact_dictionary_word(&o_cand)
                    || self.database.trie.contains_exact(&o_cand))
                    && o_cand != o_phonetic
                {
                    add_cand(
                        o_cand,
                        CandidateSource::ExactDictionary,
                        3900,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
        }

        // 7b-ii. Initial Vowel 'o' -> 'অ' Ambiguity Resolution (Natural Banglish Romanization)
        // Natural romanization overwhelmingly uses 'o' for 'অ' (e.g. onek -> অনেক, onno -> অন্য,
        // okhondo -> অখণ্ড, onurodh -> অনুরোধ, otithi -> অতিথি, odbhut -> অদ্ভুত, ongkush -> অঙ্কুশ).
        let is_o_initial = middle.len() >= 2
            && (middle.starts_with('o') || middle.starts_with('O'))
            && !middle.starts_with("oo")
            && !middle.starts_with("Oo")
            && !middle.starts_with("oO")
            && !middle.starts_with("OO")
            && !middle.starts_with("oi")
            && !middle.starts_with("ou");

        if is_o_initial && !self.config.strict_avro_mode {
            let a_latin = format!("a{}", &middle[1..]);
            let a_phonetic = self.convert_phonetic(&a_latin);
            let is_a_phon_dict = self.database.is_exact_dictionary_word(&a_phonetic)
                || self.database.trie.contains_exact(&a_phonetic);
            if is_a_phon_dict {
                add_cand(
                    a_phonetic.clone(),
                    CandidateSource::ExactDictionary,
                    3900,
                    &mut raw_candidates,
                    &mut seen,
                );
            }

            if phonetic.starts_with('ও') {
                let o_tail = &phonetic['ও'.len_utf8()..];
                let o_to_a_cand = format!("অ{}", o_tail);
                if (self.database.is_exact_dictionary_word(&o_to_a_cand)
                    || self.database.trie.contains_exact(&o_to_a_cand))
                    && o_to_a_cand != a_phonetic
                {
                    add_cand(
                        o_to_a_cand,
                        CandidateSource::ExactDictionary,
                        3900,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
            if a_phonetic.starts_with('আ') || a_phonetic.starts_with('এ') {
                let p_len = if a_phonetic.starts_with('আ') {
                    'আ'.len_utf8()
                } else {
                    'এ'.len_utf8()
                };
                let a_tail = &a_phonetic[p_len..];
                let a_to_o_cand = format!("অ{}", a_tail);
                if (self.database.is_exact_dictionary_word(&a_to_o_cand)
                    || self.database.trie.contains_exact(&a_to_o_cand))
                    && a_to_o_cand != a_phonetic
                {
                    add_cand(
                        a_to_o_cand,
                        CandidateSource::ExactDictionary,
                        3900,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
        }

        // 7c. Natural Consonant Inherent-Vowel & Digraph Sound Law
        // In natural Bengali transliteration, typists write 'a' for the inherent vowel 'অ'
        // (Hunterian/ITRANS: chalaman -> চলমান, badan -> বদন, upanibeshbad -> উপনিবেশবাদ,
        // ekkhando -> একখণ্ড, khunna -> ক্ষুণ্ণ, durgata -> দুর্গত, parishodhita -> পরিশোধিত).
        if !self.config.strict_avro_mode && middle.len() >= 3 && use_dictionary {
            let mut inherent_variants = Vec::with_capacity(16);

            let has_aa = middle.contains("aa");
            let a_indices: Vec<usize> = middle
                .match_indices('a')
                .map(|(i, _)| i)
                .filter(|&i| {
                    if has_aa {
                        (i == 0 || middle.as_bytes()[i - 1] != b'a')
                            && (i + 1 == middle.len() || middle.as_bytes()[i + 1] != b'a')
                    } else {
                        true
                    }
                })
                .collect();

            if !a_indices.is_empty() && a_indices.len() <= 4 {
                // Single replacement for each 'a'
                for &idx in &a_indices {
                    let mut s = middle.to_string();
                    s.replace_range(idx..idx + 1, "o");
                    inherent_variants.push(s);
                }
                if a_indices.len() > 1 {
                    // All 'a' -> 'o'
                    let mut s = middle.to_string();
                    for &idx in a_indices.iter().rev() {
                        s.replace_range(idx..idx + 1, "o");
                    }
                    inherent_variants.push(s);

                    // All non-final 'a' -> 'o'
                    let mut s_non_final = middle.to_string();
                    for &idx in a_indices[..a_indices.len() - 1].iter().rev() {
                        s_non_final.replace_range(idx..idx + 1, "o");
                    }
                    inherent_variants.push(s_non_final);
                }
            }

            // Collapse accidental/phonetic doubled 'aa' after consonants (kaaraabhog -> karabhog, daakotaa -> dakota)
            if has_aa {
                inherent_variants.push(middle.replace("aa", "a"));
            }

            // Digraph combinations: 'ch' -> 'c', 'kh' -> 'kkh', 'sh' -> 's', 'st' -> 'sT', 'rj' -> 'ry'
            let mut expanded = Vec::with_capacity(inherent_variants.len() * 3 + 12);
            for k in &inherent_variants {
                if k.contains("ch") {
                    expanded.push(k.replace("ch", "c"));
                }
                if k.contains("kh") && !k.contains("kkh") {
                    expanded.push(k.replace("kh", "kkh"));
                }
                if k.contains("sh") {
                    expanded.push(k.replace("sh", "s"));
                }
                if k.contains("st") {
                    expanded.push(k.replace("st", "sT"));
                }
                if k.contains("rj") {
                    expanded.push(k.replace("rj", "ry"));
                }
            }
            if middle.contains("ch") {
                expanded.push(middle.replace("ch", "c"));
            }
            if middle.contains("kh") && !middle.contains("kkh") {
                expanded.push(middle.replace("kh", "kkh"));
            }
            if middle.contains("sh") {
                expanded.push(middle.replace("sh", "s"));
            }
            if middle.contains("st") {
                expanded.push(middle.replace("st", "sT"));
            }
            if middle.contains("rj") {
                expanded.push(middle.replace("rj", "ry"));
            }
            if is_a_initial {
                let o_init = format!("o{}", &middle[1..]);
                expanded.push(o_init.clone());
                if o_init.contains("ch") {
                    expanded.push(o_init.replace("ch", "c"));
                }
                if o_init.contains("kh") && !o_init.contains("kkh") {
                    expanded.push(o_init.replace("kh", "kkh"));
                }
                if o_init.contains("sh") {
                    expanded.push(o_init.replace("sh", "s"));
                }
                if o_init.contains("st") {
                    expanded.push(o_init.replace("st", "sT"));
                }
            }

            // Trailing inherent vowel peeling (Hunterian/ITRANS inherent 'o' or Tatsama '-ita'/'-ta'/'-pto')
            if middle.len() >= 4 && (middle.ends_with('o') || middle.ends_with('O')) {
                let peeled = &middle[..middle.len() - 1];
                expanded.push(peeled.to_string());
                if is_o_initial {
                    expanded.push(format!("a{}", &peeled[1..]));
                }
            }
            if middle.len() >= 5 && (middle.ends_with("ita") || middle.ends_with("pta") || middle.ends_with("kta") || middle.ends_with("dha")) {
                let peeled = &middle[..middle.len() - 1];
                expanded.push(peeled.to_string());
            }

            inherent_variants.extend(expanded);

            for key in inherent_variants {
                let cand = self.convert_phonetic(&key);
                if self.database.is_exact_dictionary_word(&cand)
                    || self.database.trie.contains_exact(&cand)
                {
                    let score = if is_primary_in_dict { 3700 } else { 4100 };
                    add_cand(
                        cand,
                        CandidateSource::FuzzySoundLaw,
                        score,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
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
            let variant_limit = if is_primary_in_dict { 16 } else { 32 };

            for variant in fuzzy_variants.iter().take(variant_limit) {
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

            // 10b. Project "ধ্বনি-সেতু" (Dhvani-Setu) — Phonological Acoustic Recovery
            if self.config.enable_banglish_shorthand {
                for ac_variant in super::fuzzy::DhvaniSetuEngine::generate_acoustic_variants(middle)
                {
                    let ac_phonetic = self.convert_phonetic(&ac_variant);
                    if self.database.is_exact_dictionary_word(&ac_phonetic) {
                        add_cand(
                            ac_phonetic,
                            CandidateSource::FuzzySoundLaw,
                            3100,
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

            // 10c. Curated Phonetic Equivalences & Sound Laws (Reph doubling, retroflex stops, geminates)
            for eq_cand in self.database.suggest_phonetic_equivalences(
                middle,
                |k| self.convert_phonetic(k),
                8,
            ) {
                if !valid_fuzzy_stems.contains(&eq_cand) {
                    valid_fuzzy_stems.push(eq_cand.clone());
                }
                add_cand(
                    eq_cand,
                    CandidateSource::FuzzySoundLaw,
                    3800,
                    &mut raw_candidates,
                    &mut seen,
                );
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
            let is_direct_dict = self.database.is_exact_dictionary_word(&phonetic)
                && self.database.get_frequency(&phonetic) >= 1000;
            if !is_direct_dict && middle.chars().count() >= 3 && middle.chars().count() <= 12 {
                let mut typo_matches: Vec<String> = Vec::new();

                // 12a. Physical QWERTY key adjacency slips and adjacent transpositions
                let qwerty_variants = super::fuzzy::generate_qwerty_typo_variants(middle);
                for qv in qwerty_variants.iter().take(16) {
                    if let Some(ov) = self.database.lookup_override(qv) {
                        for (w, _) in ov {
                            let w_str = w.to_string();
                            if !typo_matches.contains(&w_str) {
                                typo_matches.push(w_str);
                            }
                        }
                    } else if let Some(ac) = self.database.get_autocorrect_raw_filtered(
                        qv,
                        self.config.enable_banglish_shorthand,
                    ) {
                        let ac_str = if ac.chars().any(|c| c.is_bengali()) {
                            ac
                        } else {
                            self.convert_phonetic(&ac)
                        };
                        if !typo_matches.contains(&ac_str) {
                            typo_matches.push(ac_str);
                        }
                    } else {
                        let conv = self.convert_phonetic(qv);
                        if self.database.is_exact_dictionary_word(&conv)
                            && !typo_matches.contains(&conv)
                        {
                            typo_matches.push(conv);
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
                    let freq = self.database.get_frequency(&tm);
                    let boost = if freq >= 1000 { 3800 } else { 2000 };
                    add_cand(
                        tm,
                        CandidateSource::TypoFallback,
                        boost,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
        }

        // 12c. Concatenated Word Lattice Segmentation (e.g. "kemonaso" -> "কেমন আছো", "dhonnobadbhai" -> "ধন্যবাদ ভাই")
        if self.config.enable_word_segmentation && middle.len() >= 6 && !middle.contains(' ') {
            let has_exact_single_word = is_primary_in_dict
                || raw_candidates.iter().any(|c| {
                    c.source != CandidateSource::SegmentedLattice
                        && !c.text.contains(' ')
                        && (self.database.trie.contains_exact(&c.text)
                            || self.database.is_exact_dictionary_word(&c.text)
                            || c.source == CandidateSource::Loanword
                            || c.source == CandidateSource::Autocorrect
                            || c.source == CandidateSource::PhoneticOverride)
                })
                || preferred_loanword.is_some()
                || PhoneticDatabase::get_bilingual_loanword(middle).is_some()
                || self.database.get_autocorrect_raw(middle).is_some();
            let segmented =
                super::segmenter::segment_concatenated_token(middle, &self.database, |s| {
                    self.convert_phonetic(s)
                });
            for seg in segmented {
                let boost = if has_exact_single_word { 2000 } else { 6000 };
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
        let has_explicit_intent = lekhani_parser::has_explicit_avro_intent(middle)
            || lekhani_parser::has_explicit_avro_intent(term);
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
                    || c.source == CandidateSource::PhoneticOverride
                    || self.database.get_frequency(&c.text) >= 1000)
        });

        // A word is an emphatic clitic-ও form when:
        //   (a) it ends in the emphatic particle ও (excluding basic verbs like খাও, যাও, দাও, পাও, হও), OR
        //   (b) it is one of the Bengali adverbial clitic-o forms (এখনো, তখনো, কখনো, এমনো, কোনো).
        let is_clitic_o_word = |w: &str| -> bool {
            (w.ends_with('ও')
                && w != "খাও"
                && w != "যাও"
                && w != "দাও"
                && w != "পাও"
                && w != "হও"
                && w != "নেও"
                && w != "দেও")
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
            && !raw_candidates
                .iter()
                .any(|c| c.text.ends_with('ি') && self.database.is_exact_dictionary_word(&c.text))
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

        // Phase 3: Formal Vocabulary Segmented Beam Expansion for Long Words (>= 8 chars)
        // For inputs >= 8 characters where candidate coverage is thin, test compound/sandhi splits
        // to discover formal words composed of prefix/root or root/suffix (e.g. চিকিৎসা+বিজ্ঞান, পরিবর্তন+শীলতা).
        if middle.len() >= 8 && raw_candidates.len() < 5 {
            let min_split = 3;
            let max_split = middle.len().saturating_sub(3);
            for split_idx in min_split..=max_split {
                let part1 = &middle[..split_idx];
                let part2 = &middle[split_idx..];
                let conv1 = self.convert_phonetic(part1);
                let conv2 = self.convert_phonetic(part2);

                let composite = format!("{}{}", conv1, conv2);
                if self.database.is_exact_dictionary_word(&composite) {
                    add_cand(
                        composite,
                        CandidateSource::ExactDictionary,
                        3000,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }

                let sandhi_joined = crate::phonetic::morphology::apply_sandhi_join(&conv1, &conv2);
                if sandhi_joined != format!("{}{}", conv1, conv2)
                    && self.database.is_exact_dictionary_word(&sandhi_joined)
                {
                    add_cand(
                        sandhi_joined,
                        CandidateSource::MorphologicalInflection,
                        3200,
                        &mut raw_candidates,
                        &mut seen,
                    );
                }
            }
        }

        let mut scored_candidates: Vec<(String, i32, RankFeatures)> =
            Vec::with_capacity(raw_candidates.len());
        let learner_guard = self.database.learner.try_read().ok();
        let weights = learner_guard
            .as_ref()
            .map(|l| l.rank_weights)
            .unwrap_or_default();

        let flat_context: Vec<&str> = context
            .iter()
            .flat_map(|s| s.split_whitespace())
            .collect();

        let prev = if self.config.ai_profile != AiProfile::Off && !flat_context.is_empty() {
            flat_context.last().copied()
        } else {
            None
        };
        let prev_prev = if self.config.ai_profile == AiProfile::Balanced && flat_context.len() >= 2 {
            flat_context.get(flat_context.len() - 2).copied()
        } else {
            None
        };
        let prev_3 = if self.config.ai_profile == AiProfile::Balanced && flat_context.len() >= 3 {
            flat_context.get(flat_context.len() - 3).copied()
        } else {
            None
        };

        // Extended context fields ────────────────────────────────────────────
        //
        // is_sentence_initial: true when context is empty (start of text) OR the
        // last committed word ends with a sentence boundary marker (।, !, ?).
        let is_sentence_initial = context.is_empty()
            || context.last().copied().is_some_and(|w: &str| {
                let w = w.trim_end();
                w.ends_with('।') || w.ends_with('!') || w.ends_with('?')
            });

        // is_formal_register_context: true when the last committed word is a known
        // Bengali formal/academic discourse connector.
        const FORMAL_DISCOURSE_HEADS: &[&str] = &[
            "প্রসঙ্গত",
            "উল্লেখ্য",
            "উল্লেখযোগ্য",
            "বিবেচনায়",
            "বিবেচনা",
            "পরিপ্রেক্ষিতে",
            "প্রেক্ষাপটে",
            "সংক্ষেপে",
            "তদুপরি",
            "অন্যদিকে",
            "পাশাপাশি",
            "তবে",
            "যদিও",
            "তথাপি",
            "অর্থাৎ",
            "যেমন",
            "এতদসত্ত্বেও",
            "তাছাড়া",
            "এমনকি",
            "যেহেতু",
            "সেহেতু",
        ];
        let is_formal_register_context = context
            .last()
            .copied()
            .is_some_and(|w: &str| FORMAL_DISCOURSE_HEADS.contains(&w.trim_matches('।')));

        const CASUAL_DISCOURSE_HEADS: &[&str] = &[
            "রে",
            "মামা",
            "ব্রো",
            "দোস্ত",
            "কিরে",
            "নাকি",
            "আরে",
            "ভাই",
            "vai",
            "bro",
            "boss",
            "বস",
        ];
        let is_casual_register_context = context.iter().any(|&w| {
            let clean = w.trim_matches(|c: char| c.is_ascii_punctuation() || c == '।' || c == ',');
            CASUAL_DISCOURSE_HEADS.contains(&clean)
        });

        // neural_alpha: adaptive blending weight based on N-gram LM confidence and context.
        // Computed once here so it can be used both in ranking and as a feature.
        let ngram_top_log = prev.map(|p| self.ai_context.lm().score_candidate(prev_prev, None, p));
        let neural_alpha = if self.config.ai_profile == AiProfile::Balanced {
            lekhani_neural::compute_neural_alpha(ngram_top_log.unwrap_or(-6.0), true, context.len())
        } else {
            0.0
        };

        // Predict semantic candidates once for the given context using neural model
        let neural_predictions = if let Some(ref np) = self.neural_predictor {
            if !context.is_empty() && self.config.ai_profile == AiProfile::Balanced {
                let joined = context.join(" ");
                np.predict_candidates(&joined, 16)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        // ─────────────────────────────────────────────────────────────────────

        let candidate_ctx = CandidateContext {
            phonetic: &phonetic,
            primary,
            middle,
            term,
            has_dominant_homophone,
            has_explicit_intent,
            has_explicit_casing,
            has_backtick,
            has_explicit_rri_digraph,
            is_atomic_syllable,
            is_primary_in_dict,
            is_short_token,
            has_clitic_o_candidate,
            has_clitic_i_candidate,
            clitic_o_targets: &clitic_o_targets,
            clitic_i_targets: &clitic_i_targets,
            lm_score: None,
            user_bigram_boost: 0,
            is_user_favored: false,
            is_user_learned: false,
            // Extended
            context_lm_score: None,
            gru_log_prob: None,
            neural_alpha,
            is_sentence_initial,
            is_formal_register_context,
            is_casual_register_context,
        };

        let has_exact_single_word = is_primary_in_dict
            || raw_candidates.iter().any(|c| {
                c.source != CandidateSource::SegmentedLattice
                    && !c.text.contains(' ')
                    && self.database.trie.contains_exact(&c.text)
            });

        for cand in &raw_candidates {
            if cand.source == CandidateSource::CodeShield {
                scored_candidates.push((cand.text.clone(), 60000, RankFeatures::default()));
                continue;
            }
            if cand.source == CandidateSource::SegmentedLattice {
                let boost = if has_exact_single_word
                    || has_dominant_homophone
                    || preferred_loanword.is_some()
                {
                    cand.initial_boost.min(2500)
                } else {
                    cand.initial_boost + 6000
                };
                let feats = RankFeatures {
                    is_in_dict: 1.0,
                    ..Default::default()
                };
                scored_candidates.push((cand.text.clone(), boost, feats));
                continue;
            }

            let morph_analyzer = crate::morpheme::BengaliMorphAnalyzer::new();
            let is_in_dict = self.database.is_exact_dictionary_word(&cand.text);
            let mut freq = self.database.get_frequency(&cand.text);
            let trie_contains = self.database.trie.contains_exact(&cand.text);
            let is_clitic_o = is_clitic_o_word(&cand.text);

            let decomp = morph_analyzer.decompose(&cand.text);

            if freq == 0 {
                if let Some(ref d) = decomp {
                    let root_freq = self.database.get_frequency(d.root);
                    if root_freq > 0 {
                        freq = (root_freq * 3 / 4).max(50);
                    }
                }
            }

            let lm_score = if prev.is_some() && cand.source != CandidateSource::EmojiKeyword {
                let direct_score = self
                    .ai_context
                    .lm()
                    .score_candidate(prev_prev, prev, &cand.text);
                if let Some(ref d) = decomp {
                    let root_score = self
                        .ai_context
                        .lm()
                        .score_candidate(prev_prev, prev, d.root);
                    let inflected_score =
                        morph_analyzer.estimate_inflected_score(root_score, d.category);
                    Some(direct_score.max(inflected_score))
                } else {
                    Some(direct_score)
                }
            } else {
                None
            };

            // Context-conditioned LM score using 3-word context (fourgram) for extended feature.
            let context_lm_score = if prev.is_some()
                && self.config.ai_profile == AiProfile::Balanced
                && cand.source != CandidateSource::EmojiKeyword
            {
                let s = self
                    .ai_context
                    .lm()
                    .score_candidate_fourgram(prev_3, prev_prev, prev, &cand.text);
                Some(s)
            } else {
                None
            };

            let user_bigram_boost = if let (Some(p), Some(l)) = (prev, learner_guard.as_ref()) {
                l.get_user_bigram_boost(p, &cand.text)
            } else {
                0
            };

            let is_user_favored = candidate_memory
                .get(term)
                .or_else(|| candidate_memory.get(middle))
                .or_else(|| candidate_memory.get(&phonetic))
                .or_else(|| {
                    learner_guard
                        .as_ref()
                        .and_then(|l| l.candidate_memory.get(term))
                })
                .or_else(|| {
                    learner_guard
                        .as_ref()
                        .and_then(|l| l.candidate_memory.get(middle))
                })
                .or_else(|| {
                    learner_guard
                        .as_ref()
                        .and_then(|l| l.candidate_memory.get(&phonetic))
                })
                == Some(&cand.text);

            let is_user_learned = learner_guard
                .as_ref()
                .is_some_and(|l| l.learned_words.contains(&cand.text));

            let cand_gru_log_prob = if !neural_predictions.is_empty() {
                neural_predictions
                    .iter()
                    .find(|nc| nc.word == cand.text)
                    .map(|nc| nc.log_prob)
            } else {
                None
            };

            let per_cand_ctx = CandidateContext {
                lm_score,
                user_bigram_boost,
                is_user_favored,
                is_user_learned,
                context_lm_score,
                gru_log_prob: cand_gru_log_prob,
                ..candidate_ctx
            };

            let features = extract_candidate_features(
                cand,
                is_in_dict,
                freq,
                trie_contains,
                &per_cand_ctx,
                is_clitic_o,
            );
            let score = cand.initial_boost + weights.compute_score(&features);

            scored_candidates.push((cand.text.clone(), score, features));


        }

        // Sort candidates by descending total score
        scored_candidates.sort_by_key(|a| std::cmp::Reverse(a.1));
        // When explicit Avro syntax/casing is detected, strict_avro_mode is active,
        // or an explicit geminate was collapsed by dictionary/fuzzy,
        // ensure the direct transliteration candidate (phonetic) is locked at Slot 0 and protected against LM demotion.
        let should_lock_direct = !is_tech_collocation
            && (has_explicit_intent
                || self.config.strict_avro_mode
                || (is_explicit_geminate
                    && scored_candidates
                        .first()
                        .map(|(t, _, _)| !t.contains('্'))
                        .unwrap_or(false)));

        if should_lock_direct {
            if let Some(pos) = scored_candidates
                .iter()
                .position(|(t, _, _)| t == &phonetic)
            {
                if pos > 0 {
                    let direct = scored_candidates.remove(pos);
                    scored_candidates.insert(0, direct);
                }
            }
        } else if let Some(ref pref) = preferred_word {
            // Supervised override, autocorrect, or remembered candidate takes Slot 0
            if let Some(pos) = scored_candidates.iter().position(|(t, _, _)| t == pref) {
                if pos > 0 {
                    let fav = scored_candidates.remove(pos);
                    scored_candidates.insert(0, fav);
                }
            }
        } else {
            // Explicit verbal inflection fidelity: If the user explicitly typed an inflection ending in
            // 'bo', 'cho', 'ro', 'lo', 'no', 'ko', 'go', or 'wo', prioritize the 'ো'-inflected form
            // over un-inflected dictionary stems (e.g. khabo->খাবো, korcho->করছো, koro->করো, bolo->বলো).
            let is_explicit_o_inflection = (middle.ends_with("bo") || middle.ends_with("cho"))
                && !middle.contains('`');
            if is_explicit_o_inflection {
                if let Some(pos) = scored_candidates.iter().position(|(t, _, _)| t.ends_with('ো')) {
                    if pos > 0 {
                        let o_cand = scored_candidates.remove(pos);
                        scored_candidates.insert(0, o_cand);
                    }
                }
            }
        }

        // Honorific Agreement Invariant: Concordant verbs sit at Slot 1 (index 1) when there is an honorific clash,
        // NEVER overriding Slot 0. If the typed input is already concordant or an orthographic variant of the concordant
        // verb (e.g. khaben -> খাবেন, jaben -> যাবেন), concordant verb stays at Slot 0.
        if let Some(ref conc) = concordant_verb_str {
            let typed_verb = preferred_override.as_deref().unwrap_or(&phonetic);
            if typed_verb == conc {
                // Typed verb IS the concordant verb: ensure it sits at Slot 0.
                if let Some(pos) = scored_candidates.iter().position(|(t, _, _)| t == conc) {
                    if pos != 0 {
                        let item = scored_candidates.remove(pos);
                        scored_candidates.insert(0, item);
                    }
                }
            } else if lekhani_ai::AnvayaAgreementEngine::is_honorific_shift(typed_verb, conc) {
                // Genuine honorific register clash (e.g. 'aso' [Familiar] after 'আপনি' [Formal]):
                // Slot 0 strictly retains typed verb; concordant verb sits at Slot 1.
                if let Some(pos) = scored_candidates.iter().position(|(t, _, _)| t == typed_verb) {
                    if pos != 0 {
                        let item = scored_candidates.remove(pos);
                        scored_candidates.insert(0, item);
                    }
                }
                if let Some(pos) = scored_candidates.iter().position(|(t, _, _)| t == conc) {
                    if pos != 1 {
                        let item = scored_candidates.remove(pos);
                        scored_candidates.insert(1.min(scored_candidates.len()), item);
                    }
                }
            } else {
                // Not an honorific shift (e.g. Banglish digraph 'jaben' -> 'যাবেন' vs 'জাবেন').
                // If concordant verb is in dictionary and typed_verb is not, promote concordant verb to Slot 0.
                let typed_in_dict = self.database.is_exact_dictionary_word(typed_verb);
                let conc_in_dict = self.database.is_exact_dictionary_word(conc);
                if conc_in_dict && !typed_in_dict {
                    if let Some(pos) = scored_candidates.iter().position(|(t, _, _)| t == conc) {
                        if pos != 0 {
                            let item = scored_candidates.remove(pos);
                            scored_candidates.insert(0, item);
                        }
                    }
                }
            }
        }

        // Phase 1a: Initial Vowel Statistical Reranker (A2)
        // When top candidate starts with 'আ' and user typed ambiguous single 'a',
        // check whether an 'অ'-prefixed form exists in candidates or dictionary.
        let is_a_ambiguous = middle.len() >= 2
            && (middle.starts_with('a') || middle.starts_with('A'))
            && !middle.starts_with("aa")
            && !middle.starts_with("Aa")
            && !middle.starts_with("aA")
            && !middle.starts_with("AA")
            && !should_lock_direct
            && !self.config.strict_avro_mode
            && preferred_override.is_none()
            && !scored_candidates.is_empty();

        let top_cand_vowel_prefix = if scored_candidates[0].0.starts_with('আ') {
            Some('আ'.len_utf8())
        } else if scored_candidates[0].0.starts_with('এ') {
            Some('এ'.len_utf8())
        } else {
            None
        };

        if is_a_ambiguous {
            if let Some(prefix_len) = top_cand_vowel_prefix {
                let top_cand = scored_candidates[0].0.clone();
                let tail = &top_cand[prefix_len..];
            let o_equivalent = format!("অ{}", tail);

            let o_pos = scored_candidates.iter().position(|(t, _, _)| t == &o_equivalent);
            let o_in_dict = self.database.is_exact_dictionary_word(&o_equivalent)
                || self.database.trie.contains_exact(&o_equivalent);
            let top_in_dict = self.database.is_exact_dictionary_word(&top_cand)
                || self.database.trie.contains_exact(&top_cand);

            let should_promote_o = if !top_in_dict && o_in_dict {
                true
            } else if top_in_dict && o_in_dict {
                let prev1 = context.last().copied();
                let prev2 = if context.len() >= 2 { Some(context[context.len() - 2]) } else { None };
                let prev3 = if context.len() >= 3 { Some(context[context.len() - 3]) } else { None };

                let score_top = self.ai_context.lm().score_candidate_fourgram(prev3, prev2, prev1, &top_cand);
                let score_o = self.ai_context.lm().score_candidate_fourgram(prev3, prev2, prev1, &o_equivalent);

                if (score_o - score_top).abs() < 1e-4 {
                    let freq_o = self.database.get_frequency(&o_equivalent);
                    let freq_top = self.database.get_frequency(&top_cand);
                    freq_o >= (freq_top as f32 * 1.5) as u32 || (freq_top == 0 && freq_o > 0)
                } else {
                    score_o > score_top
                }
            } else {
                false
            };

            if should_promote_o {
                if let Some(pos) = o_pos {
                    if pos > 0 {
                        let item = scored_candidates.remove(pos);
                        scored_candidates.insert(0, item);
                    }
                } else if o_in_dict {
                    let freq = self.database.get_frequency(&o_equivalent);
                    let per_cand_ctx = CandidateContext {
                        lm_score: None,
                        user_bigram_boost: 0,
                        is_user_favored: false,
                        is_user_learned: false,
                        context_lm_score: None,
                        gru_log_prob: None,
                        ..candidate_ctx
                    };
                    let features = extract_candidate_features(
                        &CandidateHypothesis {
                            text: o_equivalent.clone(),
                            source: CandidateSource::ExactDictionary,
                            initial_boost: 4200,
                        },
                        true,
                        freq,
                        true,
                        &per_cand_ctx,
                        false,
                    );
                    let score = 4200 + weights.compute_score(&features);
                    scored_candidates.insert(0, (o_equivalent, score, features));
                }
            } else if !top_in_dict {
                // If top candidate is unverified non-dictionary text, see if any candidate starting with 'অ' is in dict
                if let Some(pos) = scored_candidates.iter().position(|(t, _, _)| {
                    t.starts_with('অ') && (self.database.is_exact_dictionary_word(t) || self.database.trie.contains_exact(t))
                }) {
                    if pos > 0 {
                        let item = scored_candidates.remove(pos);
                        scored_candidates.insert(0, item);
                    }
                }
            }
        }
    }

        // Phase 1: Context-Gated Bidirectional LM Re-sort Pass
        //
        // If we have preceding context AND there are ≥2 dictionary candidates competing
        // (i.e. potential homophones), run a final pass through ContextScorer which applies
        // a richer bidirectional LM model + personal overlay — both of which are NOT
        // captured by the per-candidate `lm_score` feature baked into the initial sort.
        //
        // This re-sort is zero-allocation (in-place) and runs only when it can make
        // a meaningful difference: ≥2 non-trivial candidates with context available.
        //
        // Note: Explicit Avro intent and Anvaya-governed concordant pairs completely bypass this pass.
        let has_english_tech_passthrough = preferred_loanword
            .as_deref()
            .map(|l| !l.chars().any(|c| c.is_bengali()))
            .unwrap_or(false);

        if self.config.ai_profile != AiProfile::Off
            && !has_explicit_intent
            && !self.config.strict_avro_mode
            && concordant_verb_str.is_none()
            && !is_explicit_geminate
            && !context.is_empty()
            && scored_candidates.len() >= 2
            && !is_contextual_override
            && !has_english_tech_passthrough
        {
            let top_is_exact = scored_candidates[0].2.is_exact_phonetic > 0.5;
            let rerank_limit = scored_candidates.len().min(8);
            let mut top_texts: Vec<String> = scored_candidates[..rerank_limit]
                .iter()
                .filter(|(_, _, f)| {
                    !top_is_exact
                        || f.is_exact_phonetic > 0.5
                        || (f.is_in_dict > 0.5 && f.phonetic_similarity >= -1.0)
                })
                .map(|(t, _, _)| t.clone())
                .collect();

            if top_texts.len() >= 2 {
                self.ai_context.rank_candidates_in_place_bidirectional(
                    context,
                    right_word,
                    &mut top_texts,
                );

                let mut reordered = Vec::with_capacity(top_texts.len());
                for text in &top_texts {
                    if let Some(pos) = scored_candidates.iter().position(|(t, _, _)| t == text) {
                        reordered.push(scored_candidates.remove(pos));
                    }
                }
                for item in reordered.into_iter().rev() {
                    scored_candidates.insert(0, item);
                }
            }
        }

        let mut candidates: Vec<(String, RankFeatures)> = Vec::with_capacity(8);
        let has_english = include_english && scored_candidates.iter().any(|(c, _, _)| c == term);
        let has_emoji = scored_candidates
            .iter()
            .any(|(c, _, _)| kw_emojis.contains(c));
        let first_emoji = scored_candidates
            .iter()
            .find(|(c, _, _)| kw_emojis.contains(c))
            .map(|(c, _, f)| (c.clone(), *f));

        let text_target_len = match (has_english, has_emoji) {
            (true, true) => 6,
            (true, false) | (false, true) => 7,
            (false, false) => 8,
        };

        let mut text_cands = Vec::with_capacity(8);
        for (c, _, f) in &scored_candidates {
            if (c == term && !is_code && (!is_tech_collocation || preferred_word.as_deref() != Some(c.as_str()))) || kw_emojis.contains(c) {
                continue;
            }
            if text_cands.len() < text_target_len {
                text_cands.push((c.clone(), *f));
            }
        }

        if let Some((em, em_f)) = first_emoji {
            let insert_pos = 3.min(text_cands.len());
            for (idx, (c, f)) in text_cands.into_iter().enumerate() {
                if idx == insert_pos {
                    candidates.push((em.clone(), em_f));
                }
                candidates.push((c, f));
            }
            if candidates.len() <= insert_pos && !candidates.iter().any(|(c, _)| c == &em) {
                candidates.push((em, em_f));
            }
        } else {
            candidates.extend(text_cands);
        }

        if has_english && !candidates.iter().any(|(c, _)| c == term) {
            candidates.push((term.to_string(), RankFeatures::default()));
        }

        // Apply pre and post punctuation to candidates
        let mut final_candidates = Vec::with_capacity(candidates.len());
        let mut final_features = Vec::with_capacity(candidates.len());
        let pre_converted = self.convert_phonetic(pre);
        let post_converted = self.convert_phonetic(post);

        for (cand, feat) in candidates {
            let norm_cand = fix_chandra_bindu_position(&cand);
            if (pre.is_empty() && post.is_empty()) || norm_cand == term {
                final_candidates.push(norm_cand);
            } else {
                final_candidates.push(format!("{}{}{}", pre_converted, norm_cand, post_converted));
            }
            final_features.push(feat);
        }

        self.last_computed_features = final_features;

        // ── Seq2Seq Safety-Net ─────────────────────────────────────────────────
        // Append-only mode: seq2seq candidates are added at the end of the list
        // and NEVER promoted to top-1 until the model achieves ≥30% top-1 accuracy.
        // Length gate (≥5 chars): avoids latency cost on short common words where
        // the beam decoder is almost always correct.
        if let (Some(s2s), Some(scratch)) = (
            self.seq2seq.as_ref().map(Arc::clone),
            self.seq2seq_scratch.as_mut(),
        ) {
            let top_has_dict = self
                .last_computed_features
                .iter()
                .take(2)
                .any(|f| f.is_in_dict > 0.5);

            if final_candidates.is_empty() && middle.len() >= 4 {
                let s2s_cands = s2s.transliterate(middle, 2, scratch);
                for sc in s2s_cands {
                    if sc.is_empty() || final_candidates.contains(&sc) {
                        continue;
                    }
                    final_candidates.push(sc);
                }
                final_candidates.truncate(8);
            } else if !top_has_dict && !is_code && !is_tech_collocation && middle.len() >= 4 && !self.config.strict_avro_mode {
                let s2s_cands = s2s.transliterate(middle, 1, scratch);
                if let Some(top_s2s) = s2s_cands.first() {
                    if !top_s2s.is_empty() {
                        final_candidates.retain(|c| c != top_s2s);
                        final_candidates.insert(0, top_s2s.clone());
                    }
                }
                for sc in s2s_cands {
                    if sc.is_empty() || final_candidates.contains(&sc) {
                        continue;
                    }
                    final_candidates.push(sc);
                }
                final_candidates.truncate(8);
            }
        }
        // ──────────────────────────────────────────────────────────────────────


        self.last_computed_candidates = final_candidates.clone();

        // Cache the computed candidates for instant sub-millisecond retrieval (bounded to 128 entries)
        if self.cache.len() >= 128 {
            self.cache.clear();
        }
        self.cache.insert(
            cache_key,
            (
                final_candidates.clone(),
                self.last_computed_features.clone(),
                self.last_lps_chips.clone(),
            ),
        );

        // Determine previously selected candidate index
        let selected_index = if let Some(fav) = candidate_memory
            .get(term)
            .or_else(|| candidate_memory.get(middle))
            .or_else(|| candidate_memory.get(&phonetic))
        {
            final_candidates.iter().position(|c| c == fav).unwrap_or(0)
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
        self.record_commit_feedback(previous_word, committed_word, None)
    }

    /// Record full commit feedback with a single write lock acquisition
    pub fn record_commit_feedback(
        &mut self,
        previous_word: Option<&str>,
        committed_word: &str,
        selection_override: Option<(&str, Option<RankFeatures>, Option<RankFeatures>)>,
    ) -> Vec<String> {
        if let Ok(mut l) = self.database.learner.write() {
            if let Some((buffer, chosen_f, rejected_f)) = selection_override {
                l.record_candidate_selection_direct(buffer, committed_word);
                if let (Some(chosen), Some(rejected)) = (chosen_f, rejected_f) {
                    l.update_rank_weights(&chosen, &rejected);
                }
            }
            if let Some(prev) = previous_word {
                l.observe_committed_pair(prev, committed_word);
            }
            l.observe_and_learn(committed_word, &self.database.trie)
        } else {
            Vec::new()
        }
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
            let user_conts = if let Ok(l) = self.database.learner.try_read() {
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

                    if let Some(ac) = self.database.get_autocorrect_raw_filtered(
                        base_key,
                        self.config.enable_banglish_shorthand,
                    ) {
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
                    let is_base_aa = base_phonetic.starts_with('আ');
                    let is_base_oo = base_phonetic.starts_with('ও');
                    let base_aa_tail = if is_base_aa {
                        base_phonetic['আ'.len_utf8()..].to_string()
                    } else {
                        String::new()
                    };
                    let base_oo_tail = if is_base_oo {
                        base_phonetic['ও'.len_utf8()..].to_string()
                    } else {
                        String::new()
                    };
                    if !base_candidates.contains(&base_phonetic) {
                        base_candidates.push(base_phonetic);
                    }

                    // Initial 'a' -> 'o' stem support for suffixed words (e.g. "anchal" in "anchale" -> "অঞ্চল" -> "অঞ্চলে")
                    let is_base_a_initial = base_key.len() >= 2
                        && (base_key.starts_with('a') || base_key.starts_with('A'))
                        && !base_key.starts_with("aa")
                        && !base_key.starts_with("Aa")
                        && !base_key.starts_with("aA")
                        && !base_key.starts_with("AA");

                    if is_base_a_initial {
                        let base_o_latin = format!("o{}", &base_key[1..]);
                        let conv_o = self.convert_phonetic(&base_o_latin);
                        if (self.database.is_exact_dictionary_word(&conv_o)
                            || self.database.trie.contains_exact(&conv_o))
                            && !base_candidates.contains(&conv_o)
                        {
                            base_candidates.push(conv_o);
                            has_verified_base = true;
                        }
                        if is_base_aa {
                            let conv_o_cand = format!("অ{}", base_aa_tail);
                            if (self.database.is_exact_dictionary_word(&conv_o_cand)
                                || self.database.trie.contains_exact(&conv_o_cand))
                                && !base_candidates.contains(&conv_o_cand)
                            {
                                base_candidates.push(conv_o_cand);
                                has_verified_base = true;
                            }
                        }
                    }

                    // Initial 'o' -> 'অ' stem support for suffixed words (e.g. "onchol" in "oncholke" -> "অঞ্চল" -> "অঞ্চলকে", "okkhorke" -> "অক্ষরকে")
                    let is_base_o_initial = base_key.len() >= 2
                        && (base_key.starts_with('o') || base_key.starts_with('O'))
                        && !base_key.starts_with("oo")
                        && !base_key.starts_with("Oo")
                        && !base_key.starts_with("oO")
                        && !base_key.starts_with("OO")
                        && !base_key.starts_with("oi")
                        && !base_key.starts_with("ou");

                    if is_base_o_initial {
                        let base_a_latin = format!("a{}", &base_key[1..]);
                        let conv_a = self.convert_phonetic(&base_a_latin);
                        if (self.database.is_exact_dictionary_word(&conv_a)
                            || self.database.trie.contains_exact(&conv_a))
                            && !base_candidates.contains(&conv_a)
                        {
                            base_candidates.push(conv_a.clone());
                            has_verified_base = true;
                        }
                        if conv_a.starts_with('আ') || conv_a.starts_with('এ') {
                            let p_len = if conv_a.starts_with('আ') {
                                'আ'.len_utf8()
                            } else {
                                'এ'.len_utf8()
                            };
                            let conv_a_o_cand = format!("অ{}", &conv_a[p_len..]);
                            if (self.database.is_exact_dictionary_word(&conv_a_o_cand)
                                || self.database.trie.contains_exact(&conv_a_o_cand))
                                && !base_candidates.contains(&conv_a_o_cand)
                            {
                                base_candidates.push(conv_a_o_cand);
                                has_verified_base = true;
                            }
                        }
                        if is_base_oo {
                            let conv_o_to_a = format!("অ{}", base_oo_tail);
                            if (self.database.is_exact_dictionary_word(&conv_o_to_a)
                                || self.database.trie.contains_exact(&conv_o_to_a))
                                && !base_candidates.contains(&conv_o_to_a)
                            {
                                base_candidates.push(conv_o_to_a);
                                has_verified_base = true;
                            }
                        }
                    }

                    // Supervised phonetic override stem support (e.g. "bristi" in "bristite" -> "বৃষ্টি")
                    if let Some(ov) = self.database.lookup_override(base_key) {
                        for (stem, _) in ov {
                            if !base_candidates.iter().any(|c| c == stem) {
                                base_candidates.push(stem.to_string());
                                has_verified_base = true;
                            }
                        }
                    }

                    // Apply stem-level sound-laws (e.g. "manush" in "manusher" -> "মানুষ")
                    // Permit fuzzy expansion if base is verified or >= 4 chars (covers common
                    // 4-letter Bengali stems: valo, bhai, maro, gelo, dhoro, etc.)
                    if has_verified_base || base_key.chars().count() >= 4 {
                        for fz in super::fuzzy::generate_phonetic_variants(base_key) {
                            let conv = self.convert_phonetic(&fz);
                            if self.database.is_exact_dictionary_word(&conv)
                                && !base_candidates.contains(&conv)
                            {
                                base_candidates.push(conv);
                            }
                        }
                    }

                    let mut suffixes_to_try = vec![suffix];
                    if suffix_key == "ta" {
                        suffixes_to_try.push("তা");
                    } else if suffix_key == "ti" {
                        suffixes_to_try.push("তি");
                    }

                    for base in &base_candidates {
                        for sfx in &suffixes_to_try {
                            let word = super::morphology::apply_sandhi_join(base, sfx);
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
        }

        list
    }

    /// Decompose an inflected input into stem with supervised override + morphological suffix
    fn decompose_stem_suffix_override(&self, middle: &str) -> Option<String> {
        if middle.chars().count() <= 3 {
            return None;
        }
        for (i, _) in middle.char_indices().skip(2) {
            if super::fuzzy::cuts_layout_digraph(middle, i) {
                continue;
            }
            let suffix_key = &middle[i..];
            if let Some(suffix) = self.database.find_suffix(suffix_key) {
                let base_key = &middle[..i];
                if let Some(ov) = self.database.lookup_override(base_key) {
                    for (stem, conf) in ov {
                        if conf >= 0.70 {
                            let joined = super::morphology::apply_sandhi_join(stem, suffix);
                            if self.database.is_exact_dictionary_word(&joined) {
                                return Some(joined);
                            }
                        }
                    }
                }
                if let Some(ac) = self.database.get_autocorrect_raw_filtered(
                    base_key,
                    self.config.enable_banglish_shorthand,
                ) {
                    let base_stem = if ac.chars().any(|c| c.is_bengali()) {
                        ac
                    } else {
                        self.convert_phonetic(&ac)
                    };
                    let joined = super::morphology::apply_sandhi_join(&base_stem, suffix);
                    if self.database.is_exact_dictionary_word(&joined) {
                        return Some(joined);
                    }
                }
                let base_bn = self.convert_phonetic(base_key);
                if self.database.is_exact_dictionary_word(&base_bn) {
                    let joined = super::morphology::apply_sandhi_join(&base_bn, suffix);
                    if self.database.is_exact_dictionary_word(&joined) {
                        return Some(joined);
                    }
                }
            }
        }
        None
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

        // NOTE: Strict rank-1 ordering for contextual homophones requires the
        // corpus-trained LM (Fix 4B: `lekhani dev train`). With only the 383-word
        // static table, পরা’s +3500 exact-match bonus (Avro directly maps "pora"→পরা)
        // narrowly outweighs the bigram LM edge for পড়া. We assert the weaker but
        // still linguistically correct property: পড়া must appear in candidates,
        // and the শার্ট context must still produce পরা at rank-1 (that one is robust).
        // When previous word is "বই", "পড়া" must be rank 1
        let (cands_book, _) =
            sugg.suggest_with_context("pora", Some("বই"), true, true, &empty_memory);
        println!("cands_book: {:?}", cands_book);
        assert_eq!(
            cands_book[0], "পড়া",
            "With context বই, পড়া must be ranked #1; got {:?}",
            cands_book
        );

        // When previous word is "শার্ট", "pora" should rank "পরা" first (robust even without trained LM)
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
        assert!(sentence.contains("বাংলা") || sentence.contains("বাংলায়"));
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
                || cands_biddaloy.contains(&"বিদ্যালয়".to_string())
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
        assert!(cands_porikkhay[0] == "পরীক্ষায়" || cands_porikkhay[0] == "পরীক্ষায়");

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

        let (cands_chair, _) = sugg.suggest("chaire", false, true, &empty_memory);
        assert_eq!(cands_chair[0], "চেয়ারে");
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

    #[test]
    fn test_strict_avro_mode_toggle() {
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

        // 1. Default (strict_avro_mode = false): smart adaptive ranking promotes dictionary word 'চেষ্টা' over naive 'ছেস্তা'
        assert!(!sugg.config.strict_avro_mode);
        let (cands_adaptive, _) = sugg.suggest("chesta", true, true, &empty_memory);
        assert_eq!(cands_adaptive[0], "চেষ্টা");

        // 2. Strict Avro Mode enabled: locks direct naive transliteration 'ছেস্তা' at Slot 0
        sugg.config.strict_avro_mode = true;
        let (cands_strict, _) = sugg.suggest("chesta", true, true, &empty_memory);
        assert_eq!(cands_strict[0], "ছেস্তা");
    }

    #[test]
    fn test_inherent_vowel_sound_law() {
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
        let dict_candidates = [
            std::path::Path::new("../../data/dictionaries"),
            std::path::Path::new("data/dictionaries"),
            std::path::Path::new("../data/dictionaries"),
        ];
        for d in dict_candidates {
            if d.join("dictionary.json").exists() {
                let _ = sugg.database.load_from_dir_with_dict(d, "dictionary.json");
                break;
            }
        }
        let empty_memory = HashMap::new();
        // Inherent vowel words that users type with 'a' (e.g. upanibeshbad, chalaman, badan)
        let (cands, _) = sugg.suggest("upanibeshbad", true, true, &empty_memory);
        assert!(cands.contains(&"উপনিবেশবাদ".to_string()), "Candidates {:?} should contain উপনিবেশবাদ", cands);

        let (cands2, _) = sugg.suggest("chalaman", true, true, &empty_memory);
        assert!(cands2.contains(&"চলমান".to_string()), "Candidates {:?} should contain চলমান", cands2);

        let (cands3, _) = sugg.suggest("anchale", true, true, &empty_memory);
        assert!(cands3.contains(&"অঞ্চলে".to_string()), "Candidates {:?} should contain অঞ্চলে", cands3);

        let (cands4, _) = sugg.suggest("shadharon", true, true, &empty_memory);
        assert!(cands4.contains(&"সাধারণ".to_string()), "Candidates {:?} should contain সাধারণ", cands4);
    }
}
