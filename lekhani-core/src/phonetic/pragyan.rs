//! Lekhani Pragyan (প্রজ্ঞান) Dynamic Phonetic Statistical Engine
//!
//! Production-grade implementation of the M2M-aligned Translation Unit N-gram
//! Language Model and Lexicon-Constrained Beam Search Decoder developed in Pragyan Phase 1.
//!
//! Architecture:
//! - 2:1 Constrained M2M Translation Units (Roman digraphs/monographs -> Bengali clusters)
//! - Compact Joint Unit Language Model (Witten-Bell smoothing, Order <= 4)
//! - Trie-constrained beam search with explicit Avro intent protection
//! - Sub-millisecond inference with zero-allocation scoring hot-path

use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

use crate::session::ShiftOrigin;
use crate::tokenization::{normalize_roman, tokenize_roman_with_origin};
use lekhani_parser::{default_avro_parser, has_explicit_avro_intent, LekhaniParser};

pub type UnitId = u16;
pub const BOS_TOKEN: UnitId = 0;
pub const EOS_TOKEN: UnitId = 1;
pub const UNK_TOKEN: UnitId = 2;

/// Translation unit vocabulary mapping joint (roman, bengali) pairs to integer IDs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnitVocab {
    pub id_to_unit: Vec<(String, String)>,
    pub unit_to_id: HashMap<(String, String), UnitId>,
    pub roman_to_units: HashMap<String, Vec<UnitId>>,
}

impl UnitVocab {
    pub fn build(sequences: &[Vec<(String, String)>]) -> Self {
        let mut id_to_unit = vec![
            ("<BOS>".to_string(), "<BOS>".to_string()),
            ("<EOS>".to_string(), "<EOS>".to_string()),
            ("<UNK>".to_string(), "<UNK>".to_string()),
        ];
        let mut unit_to_id = HashMap::new();
        unit_to_id.insert(("<BOS>".to_string(), "<BOS>".to_string()), BOS_TOKEN);
        unit_to_id.insert(("<EOS>".to_string(), "<EOS>".to_string()), EOS_TOKEN);
        unit_to_id.insert(("<UNK>".to_string(), "<UNK>".to_string()), UNK_TOKEN);

        for seq in sequences {
            for unit in seq {
                if !unit_to_id.contains_key(unit) {
                    let id = id_to_unit.len() as UnitId;
                    id_to_unit.push(unit.clone());
                    unit_to_id.insert(unit.clone(), id);
                }
            }
        }

        let mut roman_to_units: HashMap<String, Vec<UnitId>> = HashMap::new();
        for (id_idx, (r, _)) in id_to_unit.iter().enumerate() {
            let u = id_idx as UnitId;
            if u != BOS_TOKEN && u != EOS_TOKEN && u != UNK_TOKEN {
                roman_to_units.entry(r.clone()).or_default().push(u);
            }
        }

        UnitVocab {
            id_to_unit,
            unit_to_id,
            roman_to_units,
        }
    }
}

/// Compact node storing continuations and totals for a specific n-gram history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextNode {
    pub c_total: f32,
    pub t_unique: u16,
    pub transitions: Vec<(UnitId, f32)>,
}

/// Compact Joint Unit Language Model with Witten-Bell backoff smoothing.
///
/// Designed to fit in < 3.0 MB on disk (Gate G6 compliant) and execute scoring
/// with zero heap allocations on the hot path.
#[derive(Clone, Serialize, Deserialize)]
pub struct JointUnitLM {
    pub order: usize,
    pub vocab: UnitVocab,
    /// Level-specific context maps. contexts[0] = unigrams (empty ctx), contexts[1] = bigrams, etc.
    pub contexts: Vec<HashMap<Vec<UnitId>, ContextNode>>,
    pub unigram_probs: Vec<f32>,
}

impl JointUnitLM {
    pub fn train(
        sequences: &[Vec<(String, String)>],
        vocab: UnitVocab,
        order: usize,
        min_count: f32,
    ) -> Self {
        let mut raw_ngrams: Vec<HashMap<Vec<UnitId>, f32>> = vec![HashMap::new(); order];

        for seq in sequences {
            let mut ids = Vec::with_capacity(seq.len() + order);
            ids.extend(std::iter::repeat_n(BOS_TOKEN, order - 1));
            for unit in seq {
                let id = vocab.unit_to_id.get(unit).copied().unwrap_or(UNK_TOKEN);
                ids.push(id);
            }
            ids.push(EOS_TOKEN);

            for n in 1..=order {
                for window in ids.windows(n) {
                    *raw_ngrams[n - 1].entry(window.to_vec()).or_default() += 1.0;
                }
            }
        }

        // Apply pruning: unigrams (n=1) are always kept; higher-order n-grams pruned by min_count
        let mut filtered_ngrams: Vec<HashMap<Vec<UnitId>, f32>> = vec![HashMap::new(); order];
        for n in 1..=order {
            for (ng, count) in raw_ngrams[n - 1].drain() {
                if n == 1 || count >= min_count {
                    filtered_ngrams[n - 1].insert(ng, count);
                }
            }
        }

        // Build compact contexts for n in 2..=order
        let mut contexts: Vec<HashMap<Vec<UnitId>, ContextNode>> = vec![HashMap::new(); order];

        for n in 2..=order {
            // Group transitions by context (length n - 1)
            let mut ctx_groups: HashMap<Vec<UnitId>, Vec<(UnitId, f32)>> = HashMap::new();
            for (ng, count) in &filtered_ngrams[n - 1] {
                let ctx = ng[..n - 1].to_vec();
                let next_u = ng[n - 1];
                ctx_groups.entry(ctx).or_default().push((next_u, *count));
            }

            for (ctx, mut trans) in ctx_groups {
                trans.sort_by_key(|t| t.0);
                let c_total: f32 = trans.iter().map(|t| t.1).sum();
                let t_unique = trans.len() as u16;
                contexts[n - 1].insert(
                    ctx,
                    ContextNode {
                        c_total,
                        t_unique,
                        transitions: trans,
                    },
                );
            }
        }

        let unigram_total: f32 = filtered_ngrams[0].values().sum();
        let vocab_size = vocab.id_to_unit.len();
        let mut unigram_probs = vec![1e-7f32; vocab_size];

        for (ng, count) in &filtered_ngrams[0] {
            let id = ng[0] as usize;
            if id < vocab_size {
                unigram_probs[id] = (count / unigram_total).max(1e-7);
            }
        }

        JointUnitLM {
            order,
            vocab,
            contexts,
            unigram_probs,
        }
    }

    #[inline]
    pub fn score_log(&self, unit: UnitId, history: &[UnitId]) -> f64 {
        let prob = self.score_unit_witten_bell(unit, history, self.order);
        (prob as f64).ln()
    }

    fn score_unit_witten_bell(&self, unit: UnitId, history: &[UnitId], n: usize) -> f32 {
        if n == 1 || history.is_empty() {
            return self
                .unigram_probs
                .get(unit as usize)
                .copied()
                .unwrap_or(1e-7);
        }

        let ctx_len = n - 1;
        let ctx = if history.len() >= ctx_len {
            &history[history.len() - ctx_len..]
        } else {
            history
        };

        let effective_n = ctx.len() + 1;
        if let Some(node) = self.contexts[effective_n - 1].get(ctx) {
            let count = node
                .transitions
                .binary_search_by_key(&unit, |t| t.0)
                .ok()
                .map(|idx| node.transitions[idx].1)
                .unwrap_or(0.0);

            let t_unique = node.t_unique as f32;
            let c_total = node.c_total;
            let lambda = 1.0 - (t_unique / (c_total + t_unique));
            let p_higher = count / c_total;
            let p_backoff = self.score_unit_witten_bell(unit, &ctx[1..], effective_n - 1);

            lambda * p_higher + (1.0 - lambda) * p_backoff
        } else {
            self.score_unit_witten_bell(unit, &ctx[1..], effective_n - 1)
        }
    }
}

/// Beam search state candidate.
#[derive(Clone, Debug)]
pub struct BeamCandidate {
    pub pos: usize,
    pub history: Vec<UnitId>,
    pub bengali: String,
    pub score: f64,
    pub consecutive_eps: usize,
}

/// Decodes Roman input tokens into ranked Bengali candidates using beam search.
pub fn decode_beam_ngram(
    roman_tokens: &[String],
    lm: &JointUnitLM,
    beam_width: usize,
) -> Vec<(String, f64)> {
    let m = roman_tokens.len();
    if m == 0 {
        return Vec::new();
    }

    // Precompute matching units for all (pos, kr) pairs up to 3 tokens (zero allocations in hot loop)
    let mut step_matches: Vec<Vec<(usize, &[UnitId])>> = Vec::with_capacity(m);
    let mut key_buf = String::with_capacity(8);
    for pos in 0..m {
        let mut matches_at_pos = Vec::with_capacity(3);
        for kr in 1..=3.min(m - pos) {
            key_buf.clear();
            for tok in &roman_tokens[pos..pos + kr] {
                key_buf.push_str(tok);
            }
            if let Some(matching) = lm.vocab.roman_to_units.get(&key_buf) {
                matches_at_pos.push((kr, matching.as_slice()));
            }
        }
        step_matches.push(matches_at_pos);
    }
    let empty_r_units = lm
        .vocab
        .roman_to_units
        .get("")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let initial_hist = vec![BOS_TOKEN; lm.order - 1];
    let mut beam = vec![BeamCandidate {
        pos: 0,
        history: initial_hist,
        bengali: String::new(),
        score: 0.0,
        consecutive_eps: 0,
    }];

    for _step in 0..=(m * 2) {
        let mut next_beam = Vec::with_capacity(beam.len() * 8);

        for cand in &beam {
            if cand.pos == m {
                next_beam.push(cand.clone());
                continue;
            }

            for &(kr, matching) in &step_matches[cand.pos] {
                for &u in matching {
                    let (_, ref b) = lm.vocab.id_to_unit[u as usize];
                    let trans_log = lm.score_log(u, &cand.history);
                    let mut new_hist = cand.history.clone();
                    new_hist.push(u);
                    let mut new_bn = cand.bengali.clone();
                    new_bn.push_str(b);

                    next_beam.push(BeamCandidate {
                        pos: cand.pos + kr,
                        history: new_hist,
                        bengali: new_bn,
                        score: cand.score + trans_log,
                        consecutive_eps: 0,
                    });
                }
            }

            if cand.consecutive_eps < 1 {
                for &u in empty_r_units {
                    let (_, ref b) = lm.vocab.id_to_unit[u as usize];
                    let trans_log = lm.score_log(u, &cand.history);
                    let mut new_hist = cand.history.clone();
                    new_hist.push(u);
                    let mut new_bn = cand.bengali.clone();
                    new_bn.push_str(b);

                    next_beam.push(BeamCandidate {
                        pos: cand.pos,
                        history: new_hist,
                        bengali: new_bn,
                        score: cand.score + trans_log,
                        consecutive_eps: cand.consecutive_eps + 1,
                    });
                }
            }
        }

        next_beam.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.bengali.cmp(&b.bengali))
        });

        let mut dedup_seen = hashbrown::HashSet::with_capacity(beam_width);
        let mut sorted_cands = Vec::with_capacity(beam_width);
        for cand in next_beam {
            if dedup_seen.insert((cand.pos, cand.bengali.clone())) {
                sorted_cands.push(cand);
                if sorted_cands.len() == beam_width {
                    break;
                }
            }
        }
        beam = sorted_cands;

        if beam.iter().all(|c| c.pos == m) {
            break;
        }
    }

    let mut cands: Vec<(String, f64)> = Vec::new();
    let mut seen = HashSet::new();
    for cand in &beam {
        if cand.pos == m {
            let eos_log = lm.score_log(EOS_TOKEN, &cand.history);
            if seen.insert(cand.bengali.clone()) {
                cands.push((cand.bengali.clone(), cand.score + eos_log));
            }
        }
    }
    cands.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    cands
}

/// Computes phonetic equivalence key for fuzzy homophone normalization.
pub fn phonetic_equivalence_key(word: &str) -> String {
    let mut key = String::with_capacity(word.len());
    for ch in word.chars() {
        match ch {
            'শ' | 'ষ' | 'স' => key.push('স'),
            'ণ' | 'ন' => key.push('ন'),
            'ই' | 'ঈ' | 'ি' | 'ী' => key.push('ি'),
            'উ' | 'ঊ' | 'ু' | 'ূ' => key.push('ু'),
            'ড়' | 'ঢ়' => key.push('র'),
            'য' | 'জ' => key.push('জ'),
            _ => key.push(ch),
        }
    }
    key
}

/// Checks whether a Bengali word exists in the dictionary or is phonetically equivalent.
pub fn is_valid_bengali_word(
    word: &str,
    dict_words: &HashSet<String>,
    dict_keys: &HashSet<String>,
) -> bool {
    if dict_words.contains(word) {
        return true;
    }
    let key = phonetic_equivalence_key(word);
    dict_keys.contains(&key)
}

/// Checks if token has explicit Avro casing.
pub fn is_avro_specific_token(raw_roman: &str) -> bool {
    raw_roman.chars().any(|c| c.is_ascii_uppercase())
}

/// Pragyan deployable candidate blending policy (Gate G2, G4, G5 verified).
#[allow(clippy::too_many_arguments)]
pub fn run_deployable_policy(
    raw_roman: &str,
    ngram_cands: &[(String, f64)],
    avro_cand: &str,
    avro_uncap_cand: Option<&str>,
    eng_cands: &[String],
    dict_words: &HashSet<String>,
    dict_keys: &HashSet<String>,
    l_dict: f64,
    l_avro: f64,
    l_eng: f64,
) -> Vec<(String, f64)> {
    let top_s = ngram_cands.first().map(|c| c.1).unwrap_or(0.0);
    let min_s = ngram_cands.last().map(|c| c.1).unwrap_or(-50.0);
    let floor_s = min_s.max(top_s - 3.5);

    let has_intent = has_explicit_avro_intent(raw_roman);
    let is_avro_cased = is_avro_specific_token(raw_roman);
    let avro_in_dict = is_valid_bengali_word(avro_cand, dict_words, dict_keys);

    let mut cand_map: HashMap<String, f64> = HashMap::with_capacity(ngram_cands.len() + 4);
    for (w, s) in ngram_cands {
        cand_map.insert(w.clone(), *s);
    }

    if avro_in_dict || has_intent || is_avro_cased || eng_cands.contains(&avro_cand.to_string()) {
        cand_map.entry(avro_cand.to_string()).or_insert(floor_s);
    }
    if let Some(u) = avro_uncap_cand {
        if is_valid_bengali_word(u, dict_words, dict_keys) {
            cand_map.entry(u.to_string()).or_insert(floor_s);
        }
    }
    if let Some(top_e) = eng_cands.first() {
        cand_map.entry(top_e.clone()).or_insert(floor_s);
    }

    let mut scored: Vec<(String, f64)> = cand_map
        .into_iter()
        .map(|(w, base_s)| {
            let mut s = base_s;
            if dict_words.contains(&w) {
                s += l_dict;
            }
            if eng_cands.contains(&w) {
                s += l_eng;
            }
            let is_avro_raw = w == avro_cand;
            let is_avro_uncap = avro_uncap_cand.is_some_and(|u| w == u);
            let is_avro = is_avro_raw || is_avro_uncap;

            if is_avro {
                let in_beam = ngram_cands.iter().any(|c| c.0 == w);
                let is_dict = is_valid_bengali_word(&w, dict_words, dict_keys);
                if has_intent || is_avro_cased {
                    s += l_avro;
                } else if is_dict {
                    s += l_avro * 0.85;
                } else if in_beam {
                    s += l_avro * 0.40;
                }
            }

            (w, s)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored
}

/// Configuration for the Pragyan Phonetic Engine.
#[derive(Clone, Debug)]
pub struct PragyanConfig {
    pub beam_width: usize,
    pub lambda_dict: f64,
    pub lambda_avro: f64,
    pub lambda_engine: f64,
    pub enabled: bool,
}

impl Default for PragyanConfig {
    fn default() -> Self {
        Self {
            beam_width: 8,
            lambda_dict: 6.0,
            lambda_avro: 8.0,
            lambda_engine: 2.0,
            enabled: true,
        }
    }
}

/// Production Pragyan Statistical Phonetic Engine.
#[derive(Clone)]
pub struct PragyanEngine {
    pub lm: Arc<JointUnitLM>,
    pub avro_parser: Arc<LekhaniParser>,
    pub dict_words: Arc<HashSet<String>>,
    pub dict_keys: Arc<HashSet<String>>,
    pub config: PragyanConfig,
}

impl PragyanEngine {
    /// Loads a precompiled Pragyan language model and dictionary into memory.
    pub fn new(
        lm: Arc<JointUnitLM>,
        dict_words: Arc<HashSet<String>>,
        dict_keys: Arc<HashSet<String>>,
    ) -> Self {
        Self {
            lm,
            avro_parser: Arc::new(default_avro_parser()),
            dict_words,
            dict_keys,
            config: PragyanConfig::default(),
        }
    }

    /// Attempts to load the trained model binary from disk.
    pub fn load_from_file<P: AsRef<Path>>(
        model_path: P,
        dict_words: Arc<HashSet<String>>,
        dict_keys: Arc<HashSet<String>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(model_path)?;
        let lm: JointUnitLM = bincode::deserialize(&bytes)?;
        Ok(Self::new(Arc::new(lm), dict_words, dict_keys))
    }

    /// Automatically probe standard search paths for a deployed Pragyan model binary and lexicon.
    pub fn probe() -> Option<Arc<Self>> {
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
            let p_model = dir.join("pragyan_unit_lm.bin");
            if p_model.exists() {
                if let Ok(bytes) = std::fs::read(&p_model) {
                    if let Ok(lm) = bincode::deserialize::<JointUnitLM>(&bytes) {
                        let mut dict_words = hashbrown::HashSet::new();
                        let p_dict_bin = dir.join("dictionary.bin");
                        let p_dict_json = dir.join("dictionary.json");
                        if p_dict_bin.exists() {
                            if let Ok(trie) = crate::trie::PrefixTrie::load_binary(&p_dict_bin) {
                                for (w, _) in trie.iter() {
                                    dict_words.insert(w.to_string());
                                }
                            }
                        } else if p_dict_json.exists() {
                            if let Ok(file) = std::fs::File::open(&p_dict_json) {
                                let reader = std::io::BufReader::new(file);
                                if let Ok(raw_map) = serde_json::from_reader::<
                                    _,
                                    hashbrown::HashMap<String, serde_json::Value>,
                                >(reader)
                                {
                                    for val in raw_map.values() {
                                        if let Some(list) = val.as_array() {
                                            for item in list {
                                                if let Some(s) = item.as_str() {
                                                    dict_words
                                                        .insert(crate::chars::canonicalize(s));
                                                } else if let Some(arr) = item.as_array() {
                                                    if let Some(s) =
                                                        arr.first().and_then(|v| v.as_str())
                                                    {
                                                        dict_words
                                                            .insert(crate::chars::canonicalize(s));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        for &(word, _) in crate::phonetic::database::CORE_BENGALI_FREQUENCIES {
                            dict_words.insert(word.to_string());
                        }
                        let dict_keys: hashbrown::HashSet<String> = dict_words
                            .iter()
                            .map(|w| phonetic_equivalence_key(w))
                            .collect();
                        let engine =
                            Self::new(Arc::new(lm), Arc::new(dict_words), Arc::new(dict_keys));
                        return Some(Arc::new(engine));
                    }
                }
            }
        }
        None
    }

    /// Synchronizes the active dictionary and equivalence keys from the runtime PrefixTrie.
    pub fn sync_dictionary(&mut self, trie: &crate::trie::PrefixTrie) {
        if !trie.is_empty() {
            let mut dict_words = hashbrown::HashSet::with_capacity(trie.len() + 100);
            for (w, _) in trie.iter() {
                dict_words.insert(w.to_string());
            }
            for &(word, _) in crate::phonetic::database::CORE_BENGALI_FREQUENCIES {
                dict_words.insert(word.to_string());
            }
            let dict_keys: hashbrown::HashSet<String> = dict_words
                .iter()
                .map(|w| phonetic_equivalence_key(w))
                .collect();
            self.dict_words = Arc::new(dict_words);
            self.dict_keys = Arc::new(dict_keys);
        }
    }

    /// Transliterates typed Roman term into ranked Bengali candidates.
    pub fn suggest(&self, term: &str, origin: ShiftOrigin) -> Vec<String> {
        if term.is_empty() {
            return Vec::new();
        }

        let avro_cand = self.avro_parser.convert(term);
        let has_capital = term.chars().any(|c| c.is_ascii_uppercase());
        let avro_uncap_cand = if has_capital {
            Some(self.avro_parser.convert(&term.to_lowercase()))
        } else {
            None
        };

        // If Pragyan statistical decoding is disabled or strict avro intent is active:
        if !self.config.enabled || has_explicit_avro_intent(term) {
            let mut list = vec![avro_cand];
            if let Some(uncap) = avro_uncap_cand {
                if !list.contains(&uncap) {
                    list.push(uncap);
                }
            }
            return list;
        }

        let lat_norm = normalize_roman(term);
        let roman_tokens = tokenize_roman_with_origin(&lat_norm, origin);
        let ngram_cands = decode_beam_ngram(&roman_tokens, &self.lm, self.config.beam_width);

        let ranked = run_deployable_policy(
            term,
            &ngram_cands,
            &avro_cand,
            avro_uncap_cand.as_deref(),
            &[],
            &self.dict_words,
            &self.dict_keys,
            self.config.lambda_dict,
            self.config.lambda_avro,
            self.config.lambda_engine,
        );

        ranked.into_iter().map(|(w, _)| w).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phonetic_equivalence_key() {
        assert_eq!(
            phonetic_equivalence_key("শান্তি"),
            phonetic_equivalence_key("সান্তি")
        );
        assert_eq!(
            phonetic_equivalence_key("পানি"),
            phonetic_equivalence_key("পাণী")
        );
    }

    #[test]
    fn test_pragyan_production_model_suggest() {
        let model_candidates = [
            Path::new("data/dictionaries/pragyan_unit_lm.bin"),
            Path::new("../data/dictionaries/pragyan_unit_lm.bin"),
            Path::new("../../data/dictionaries/pragyan_unit_lm.bin"),
        ];
        let model_path = match model_candidates.iter().find(|p| p.exists()) {
            Some(p) => *p,
            None => {
                eprintln!(
                    "Skipping test_pragyan_production_model_suggest: pragyan_unit_lm.bin not found"
                );
                return;
            }
        };

        let dict_words = Arc::new(HashSet::new());
        let dict_keys = Arc::new(HashSet::new());

        let engine = PragyanEngine::load_from_file(model_path, dict_words, dict_keys)
            .expect("Failed to load production pragyan_unit_lm.bin");

        let cands_amar = engine.suggest("amar", ShiftOrigin::AutoCapitalize);
        assert!(!cands_amar.is_empty(), "amar should produce candidates");
        assert_eq!(cands_amar[0], "আমার", "amar should produce আমার as Top-1");

        let cands_kemon = engine.suggest("kemon", ShiftOrigin::AutoCapitalize);
        assert!(!cands_kemon.is_empty(), "kemon should produce candidates");
        assert_eq!(cands_kemon[0], "কেমন", "kemon should produce কেমন as Top-1");

        let cands_tumi = engine.suggest("tumi", ShiftOrigin::AutoCapitalize);
        assert!(!cands_tumi.is_empty(), "tumi should produce candidates");
        assert_eq!(cands_tumi[0], "তুমি", "tumi should produce তুমি as Top-1");

        let cands_shonge = engine.suggest("shonge", ShiftOrigin::AutoCapitalize);
        assert!(!cands_shonge.is_empty(), "shonge should produce candidates");
        assert!(
            cands_shonge.contains(&"সঙ্গে".to_string()),
            "shonge should contain সঙ্গে"
        );

        let cands_kosto = engine.suggest("kosto", ShiftOrigin::AutoCapitalize);
        assert!(!cands_kosto.is_empty(), "kosto should produce candidates");
        assert!(
            cands_kosto.contains(&"কষ্ট".to_string()),
            "kosto should contain কষ্ট"
        );
    }

    #[test]
    fn test_explicit_avro_intent_preservation() {
        let model_path = Path::new("data/dictionaries/pragyan_unit_lm.bin");
        if !model_path.exists() {
            return;
        }

        let dict_words = Arc::new(HashSet::new());
        let dict_keys = Arc::new(HashSet::new());

        let engine = PragyanEngine::load_from_file(model_path, dict_words, dict_keys)
            .expect("Failed to load production pragyan_unit_lm.bin");

        // When explicit backtick or explicit Avro casing is present, direct Avro is preserved
        let cands = engine.suggest("`cesta", ShiftOrigin::Manual);
        assert!(!cands.is_empty());
        assert_eq!(cands[0], "চেষ্টা");
    }
}
