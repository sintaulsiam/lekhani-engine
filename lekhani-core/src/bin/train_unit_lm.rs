//! Pragyan Phase 1 Step 7: Joint Translation Unit N-gram Language Model
//!
//! Architectural Directives & Work Order Specifications:
//! 1. Estimate joint transition probabilities over aligned unit sequences from Step 4.
//! 2. Sentence boundary symbols at both ends: <s> and </s>.
//! 3. Orders 3, 4, 5 with Witten-Bell backoff smoothing.
//! 4. Two weighting schemes:
//!    - Row-level (Dakshina counts used as frequency)
//!    - Type-level (each pair weighted equally as 1.0)
//! 5. Count pruning at 3 levels per order:
//!    - Level 0: count >= 1 (no pruning)
//!    - Level 1: count >= 2 for order >= 2
//!    - Level 2: count >= 5 for order >= 2
//! 6. Flat, compact serialization format with size measurement in KB.
//! 7. Evaluation on held-out DEV (N = 9,279):
//!    - Top-1 accuracy, Top-3 accuracy, Reachability (N-gram alone, no lexicon, no Avro).
//! 8. Gate G1 Verification:
//!    - N-gram alone must beat raw Avro on dev (Raw Avro = 12.37%).

use bincode;
use hashbrown::{HashMap, HashSet};
use lekhani_core::chars::canonicalize;
use lekhani_core::tokenization::{
    is_valid_translation_unit, normalize_roman, tokenize_bengali, tokenize_roman, BengaliToken,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_EM_ITERATIONS: usize = 6;
const NULL_PENALTY: f64 = 0.002;

#[derive(Clone, Debug)]
pub struct AlignerPair {
    pub roman_tokens: Vec<String>,
    pub roman_norm: String,
    pub bengali_tokens: Vec<BengaliToken>,
    pub raw_roman: String,
    pub raw_bengali: String,
    pub count: f64,
}

/// Loads a dataset with Filters OFF (100% inclusion policy).
pub fn load_dataset(path: &Path) -> (Vec<AlignerPair>, usize) {
    let file = File::open(path).unwrap_or_else(|_| panic!("Failed to open dataset at {:?}", path));
    let reader = BufReader::new(file);
    let mut pairs = Vec::with_capacity(100_000);
    let mut dropped = 0;

    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split('\t').collect();
        if parts.len() < 2 {
            dropped += 1;
            continue;
        }

        let raw_bn = parts[0].trim();
        let raw_lat = parts[1].trim();
        let count: f64 = if parts.len() >= 3 {
            parts[2].trim().parse().unwrap_or(1.0)
        } else {
            1.0
        };

        let bn = canonicalize(raw_bn);
        let lat_norm = normalize_roman(raw_lat);

        if bn.is_empty() || lat_norm.is_empty() {
            dropped += 1;
            continue;
        }

        let bn_tokens = tokenize_bengali(&bn);
        let r_tokens = tokenize_roman(&lat_norm);

        pairs.push(AlignerPair {
            roman_tokens: r_tokens,
            roman_norm: lat_norm,
            bengali_tokens: bn_tokens,
            raw_roman: raw_lat.to_string(),
            raw_bengali: bn,
            count,
        });
    }

    (pairs, dropped)
}

/// Seeds Avro soft prior with weight α into the transition counts.
pub fn seed_avro_prior(counts: &mut HashMap<(String, String), f64>, alpha: f64) {
    if alpha <= 0.0 {
        return;
    }

    let priors: &[(&str, &str)] = &[
        // Single Consonants
        ("k", "ক"), ("kh", "খ"), ("g", "গ"), ("gh", "ঘ"), ("ng", "ঙ"),
        ("c", "চ"), ("ch", "ছ"), ("j", "জ"), ("jh", "ঝ"),
        ("t", "ত"), ("t", "ট"), ("th", "থ"), ("th", "ঠ"),
        ("d", "দ"), ("d", "ড"), ("dh", "ধ"), ("dh", "ঢ"),
        ("n", "ন"), ("n", "ণ"),
        ("p", "প"), ("ph", "ফ"), ("f", "ফ"), ("b", "ব"), ("bh", "ভ"), ("v", "ভ"), ("m", "ম"),
        ("z", "য"), ("r", "র"), ("l", "ল"),
        ("sh", "শ"), ("sh", "ষ"), ("s", "স"), ("s", "শ"), ("h", "হ"),
        ("y", "য়"), ("w", "ও"), ("q", "ক"), ("x", "ক্স"),
        // Vowels & Kars
        ("a", "া"), ("a", "আ"), ("a", "অ"), ("aa", "আ"), ("aa", "া"),
        ("i", "ি"), ("i", "ই"), ("ee", "ী"), ("ee", "ি"),
        ("u", "ু"), ("u", "উ"), ("oo", "ূ"), ("oo", "ু"),
        ("e", "ে"), ("e", "এ"), ("oi", "ঐ"), ("ou", "ঔ"),
        ("o", "ও"), ("o", "ো"), ("o", "অ"),
        // Rule-Based Conjuncts
        ("kkh", "ক্ষ"), ("ngk", "ঙ্ক"), ("ngkh", "ঙ্খ"), ("ngg", "ঙ্গ"), ("nggh", "ঙ্ঘ"),
        ("cch", "চ্ছ"), ("jj", "জ্জ"), ("jjh", "জ্ঝ"), ("nc", "ঞ্চ"), ("nch", "ঞ্ছ"), ("nj", "ঞ্জ"),
        ("tt", "ট্ট"), ("nth", "ণ্ঠ"), ("nd", "ণ্ড"), ("kt", "ক্ত"), ("pt", "প্ত"), ("bd", "ব্দ"),
        ("st", "স্ত"), ("sth", "স্থ"), ("sn", "স্ন"), ("sp", "স্প"), ("sph", "স্ফ"),
        ("nt", "ন্ত"), ("nth", "ন্থ"), ("nd", "ন্দ"), ("ndh", "ন্ধ"),
        ("mp", "ম্প"), ("mb", "ম্ব"), ("mbh", "ম্ভ"),
        ("rk", "র্ক"), ("rg", "র্গ"), ("rt", "র্ত"), ("rt", "র্ড"), ("rd", "র্দ"),
        ("rn", "র্ণ"), ("rp", "র্প"), ("rb", "র্ব"), ("rm", "র্ম"), ("rsh", "র্ষ"), ("rs", "র্স"),
        // Common Affixes & Morphological Suffixes
        ("er", "ের"), ("ar", "ার"), ("te", "তে"), ("ke", "কে"), ("der", "দের"),
        // Deletion / Inherent Vowel Syncope
        ("a", ""), ("o", ""), ("e", ""),
    ];

    for &(r, b) in priors {
        *counts.entry((r.to_string(), b.to_string())).or_default() += alpha;
    }
}

/// Initializes the joint translation model from TRAIN pairs only (Token-Level).
pub fn initialize_model(
    pairs: &[AlignerPair],
    max_r: usize,
    max_b: usize,
    alpha: f64,
) -> HashMap<(String, String), f64> {
    let mut counts: HashMap<(String, String), f64> = HashMap::new();
    seed_avro_prior(&mut counts, alpha);

    for pair in pairs {
        let m = pair.roman_tokens.len();
        let n = pair.bengali_tokens.len();
        let pair_weight = pair.count / ((m * n) as f64).max(1.0);

        for i in 0..m {
            for kr in 1..=max_r.min(m - i) {
                let r_slice = &pair.roman_tokens[i..i + kr];
                if is_valid_translation_unit(r_slice, &[]) {
                    let r = r_slice.join("");
                    *counts.entry((r, String::new())).or_default() += pair_weight * NULL_PENALTY;
                }
            }
        }

        for i in 0..m {
            for kr in 1..=max_r.min(m - i) {
                let r_slice = &pair.roman_tokens[i..i + kr];
                let r = r_slice.join("");

                for j in 0..n {
                    for kb in 1..=max_b.min(n - j) {
                        let b_slice = &pair.bengali_tokens[j..j + kb];
                        if !is_valid_translation_unit(r_slice, b_slice) {
                            continue;
                        }
                        let b: String = b_slice.iter().map(|tok| tok.text.as_str()).collect();
                        *counts.entry((r.clone(), b)).or_default() += pair_weight;
                    }
                }
            }
        }

        for j in 0..n {
            for kb in 1..=max_b.min(n - j) {
                let b_slice = &pair.bengali_tokens[j..j + kb];
                if is_valid_translation_unit(&[], b_slice) {
                    let b: String = b_slice.iter().map(|tok| tok.text.as_str()).collect();
                    *counts.entry((String::new(), b)).or_default() += pair_weight * NULL_PENALTY;
                }
            }
        }
    }

    let total: f64 = counts.values().sum();
    for v in counts.values_mut() {
        *v /= total;
    }
    counts
}

/// Executes one EM iteration across all pairs in parallel.
pub fn em_iteration(
    pairs: &[AlignerPair],
    model: &HashMap<(String, String), f64>,
    max_r: usize,
    max_b: usize,
) -> (HashMap<(String, String), f64>, f64) {
    let (expected_counts, total_log_likelihood) = pairs
        .par_chunks(2048)
        .map(|chunk| {
            let mut local_counts: HashMap<(String, String), f64> = HashMap::new();
            let mut local_ll = 0.0;

            for pair in chunk {
                let m = pair.roman_tokens.len();
                let n = pair.bengali_tokens.len();

                let mut alpha = vec![vec![0.0f64; n + 1]; m + 1];
                alpha[0][0] = 1.0;

                for i in 0..=m {
                    for j in 0..=n {
                        let a_ij = alpha[i][j];
                        if a_ij == 0.0 {
                            continue;
                        }

                        for kr in 0..=max_r.min(m - i) {
                            for kb in 0..=max_b.min(n - j) {
                                if kr == 0 && kb == 0 {
                                    continue;
                                }

                                let r_slice = if kr == 0 { &[] } else { &pair.roman_tokens[i..i + kr] };
                                let b_slice = if kb == 0 { &[] } else { &pair.bengali_tokens[j..j + kb] };

                                if !is_valid_translation_unit(r_slice, b_slice) {
                                    continue;
                                }

                                let r: String = r_slice.join("");
                                let b: String = if kb == 0 {
                                    String::new()
                                } else {
                                    b_slice.iter().map(|t| t.text.as_str()).collect()
                                };

                                if let Some(&prob) = model.get(&(r, b)) {
                                    alpha[i + kr][j + kb] += a_ij * prob;
                                }
                            }
                        }
                    }
                }

                let z = alpha[m][n];
                if z <= 0.0 || z.is_nan() {
                    continue;
                }
                local_ll += pair.count * z.ln();

                let mut beta = vec![vec![0.0f64; n + 1]; m + 1];
                beta[m][n] = 1.0;

                for i in (0..=m).rev() {
                    for j in (0..=n).rev() {
                        let b_ij = beta[i][j];
                        if b_ij == 0.0 {
                            continue;
                        }

                        for kr in 0..=max_r.min(i) {
                            for kb in 0..=max_b.min(j) {
                                if kr == 0 && kb == 0 {
                                    continue;
                                }

                                let r_slice = if kr == 0 { &[] } else { &pair.roman_tokens[i - kr..i] };
                                let b_slice = if kb == 0 { &[] } else { &pair.bengali_tokens[j - kb..j] };

                                if !is_valid_translation_unit(r_slice, b_slice) {
                                    continue;
                                }

                                let r: String = r_slice.join("");
                                let b: String = if kb == 0 {
                                    String::new()
                                } else {
                                    b_slice.iter().map(|t| t.text.as_str()).collect()
                                };

                                if let Some(&prob) = model.get(&(r, b)) {
                                    beta[i - kr][j - kb] += b_ij * prob;
                                }
                            }
                        }
                    }
                }

                let inv_z = 1.0 / z;
                for i in 0..=m {
                    for j in 0..=n {
                        let a_ij = alpha[i][j];
                        if a_ij == 0.0 {
                            continue;
                        }

                        for kr in 0..=max_r.min(m - i) {
                            for kb in 0..=max_b.min(n - j) {
                                if kr == 0 && kb == 0 {
                                    continue;
                                }
                                let b_next = beta[i + kr][j + kb];
                                if b_next == 0.0 {
                                    continue;
                                }

                                let r_slice = if kr == 0 { &[] } else { &pair.roman_tokens[i..i + kr] };
                                let b_slice = if kb == 0 { &[] } else { &pair.bengali_tokens[j..j + kb] };

                                if !is_valid_translation_unit(r_slice, b_slice) {
                                    continue;
                                }

                                let r: String = r_slice.join("");
                                let b: String = if kb == 0 {
                                    String::new()
                                } else {
                                    b_slice.iter().map(|t| t.text.as_str()).collect()
                                };

                                if let Some(&prob) = model.get(&(r.clone(), b.clone())) {
                                    let gamma = (a_ij * prob * b_next) * inv_z;
                                    *local_counts.entry((r, b)).or_default() += pair.count * gamma;
                                }
                            }
                        }
                    }
                }
            }

            (local_counts, local_ll)
        })
        .reduce(
            || (HashMap::new(), 0.0),
            |(mut c1, ll1), (c2, ll2)| {
                for (k, v) in c2 {
                    *c1.entry(k).or_default() += v;
                }
                (c1, ll1 + ll2)
            },
        );

    let total_count: f64 = expected_counts.values().sum();
    let mut new_model = HashMap::with_capacity(expected_counts.len());
    for (k, v) in expected_counts {
        new_model.insert(k, v / total_count);
    }

    (new_model, total_log_likelihood)
}

/// Trains the M2M alignment model using Expectation-Maximization.
pub fn train_m2m_aligner(
    pairs: &[AlignerPair],
    iterations: usize,
    alpha_prior: f64,
    max_r: usize,
    max_b: usize,
) -> HashMap<(String, String), f64> {
    let mut model = initialize_model(pairs, max_r, max_b, alpha_prior);
    for _iter in 1..=iterations {
        let (new_model, _) = em_iteration(pairs, &model, max_r, max_b);
        model = new_model;
    }
    model
}

/// Computes the optimal Viterbi forced alignment path: Vec<(r, b)>.
pub fn viterbi_forced_alignment(
    pair: &AlignerPair,
    model: &HashMap<(String, String), f64>,
    max_r: usize,
    max_b: usize,
) -> (f64, Vec<(String, String)>) {
    let m = pair.roman_tokens.len();
    let n = pair.bengali_tokens.len();

    let mut viterbi = vec![vec![f64::NEG_INFINITY; n + 1]; m + 1];
    let mut backpointer = vec![vec![(0usize, 0usize, String::new(), String::new()); n + 1]; m + 1];

    viterbi[0][0] = 0.0;

    for i in 0..=m {
        for j in 0..=n {
            let score = viterbi[i][j];
            if score == f64::NEG_INFINITY {
                continue;
            }

            for kr in 0..=max_r.min(m - i) {
                for kb in 0..=max_b.min(n - j) {
                    if kr == 0 && kb == 0 {
                        continue;
                    }

                    let r_slice = if kr == 0 { &[] } else { &pair.roman_tokens[i..i + kr] };
                    let b_slice = if kb == 0 { &[] } else { &pair.bengali_tokens[j..j + kb] };

                    if !is_valid_translation_unit(r_slice, b_slice) {
                        continue;
                    }

                    let r: String = r_slice.join("");
                    let b: String = if kb == 0 {
                        String::new()
                    } else {
                        b_slice.iter().map(|t| t.text.as_str()).collect()
                    };

                    if let Some(&prob) = model.get(&(r.clone(), b.clone())) {
                        if prob > 0.0 {
                            let trans_score = score + prob.ln();
                            if trans_score > viterbi[i + kr][j + kb] {
                                viterbi[i + kr][j + kb] = trans_score;
                                backpointer[i + kr][j + kb] = (kr, kb, r, b);
                            }
                        }
                    }
                }
            }
        }
    }

    if viterbi[m][n] == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, Vec::new());
    }

    let mut path = Vec::new();
    let mut curr_i = m;
    let mut curr_j = n;

    while curr_i > 0 || curr_j > 0 {
        let (kr, kb, ref r, ref b) = backpointer[curr_i][curr_j];
        if kr == 0 && kb == 0 {
            break;
        }
        path.push((r.clone(), b.clone()));
        curr_i -= kr;
        curr_j -= kb;
    }

    path.reverse();
    (viterbi[m][n], path)
}

pub type UnitId = u16;
pub const BOS_TOKEN: UnitId = 0;
pub const EOS_TOKEN: UnitId = 1;

/// Vocabulary mapping translation units to compact integer IDs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnitVocabulary {
    pub id_to_unit: Vec<(String, String)>,
    pub unit_to_id: HashMap<(String, String), UnitId>,
    pub roman_to_units: HashMap<String, Vec<UnitId>>,
}

impl UnitVocabulary {
    pub fn new(units: Vec<(String, String)>) -> Self {
        let mut id_to_unit = Vec::with_capacity(units.len() + 2);
        let mut unit_to_id = HashMap::with_capacity(units.len() + 2);
        let mut roman_to_units: HashMap<String, Vec<UnitId>> = HashMap::new();

        // 0: <s>
        id_to_unit.push(("<s>".to_string(), "<s>".to_string()));
        unit_to_id.insert(("<s>".to_string(), "<s>".to_string()), BOS_TOKEN);

        // 1: </s>
        id_to_unit.push(("</s>".to_string(), "</s>".to_string()));
        unit_to_id.insert(("</s>".to_string(), "</s>".to_string()), EOS_TOKEN);

        for (r, b) in units {
            if (r.as_str(), b.as_str()) == ("<s>", "<s>") || (r.as_str(), b.as_str()) == ("</s>", "</s>") {
                continue;
            }
            if unit_to_id.contains_key(&(r.clone(), b.clone())) {
                continue;
            }
            let id = id_to_unit.len() as UnitId;
            id_to_unit.push((r.clone(), b.clone()));
            unit_to_id.insert((r.clone(), b.clone()), id);
            roman_to_units.entry(r).or_default().push(id);
        }

        Self {
            id_to_unit,
            unit_to_id,
            roman_to_units,
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.id_to_unit.len()
    }

    #[inline]
    pub fn get_id(&self, r: &str, b: &str) -> Option<UnitId> {
        self.unit_to_id.get(&(r.to_string(), b.to_string())).copied()
    }
}

/// N-Gram Language Model with Witten-Bell Smoothing.
#[derive(Clone, Serialize, Deserialize)]
pub struct JointUnitLM {
    pub order: usize,
    pub vocab: UnitVocabulary,
    /// Maps n-gram prefix history -> (C(h), T(h), continuation_counts: HashMap<UnitId, f64>)
    pub histories: HashMap<Vec<UnitId>, (f64, usize, HashMap<UnitId, f64>)>,
    pub unigram_counts: HashMap<UnitId, f64>,
    pub total_unigrams: f64,
}

impl JointUnitLM {
    /// Trains a joint translation unit N-gram model from aligned unit sequences.
    pub fn train(
        sequences: &[(Vec<UnitId>, f64)],
        vocab: UnitVocabulary,
        order: usize,
        prune_min_count: f64,
    ) -> Self {
        let mut ngram_counts: HashMap<Vec<UnitId>, f64> = HashMap::new();
        let mut unigram_counts: HashMap<UnitId, f64> = HashMap::new();
        let mut total_unigrams = 0.0;

        for (seq, weight) in sequences {
            let mut full_seq = Vec::with_capacity(seq.len() + 2);
            full_seq.push(BOS_TOKEN);
            full_seq.extend_from_slice(seq);
            full_seq.push(EOS_TOKEN);

            let len = full_seq.len();
            for i in 0..len {
                let u = full_seq[i];
                *unigram_counts.entry(u).or_default() += weight;
                total_unigrams += weight;

                for k in 2..=order.min(i + 1) {
                    let start = i + 1 - k;
                    let ng = full_seq[start..=i].to_vec();
                    *ngram_counts.entry(ng).or_default() += weight;
                }
            }
        }

        // Apply count pruning on order >= 2
        if prune_min_count > 1.0 {
            ngram_counts.retain(|_, count| *count >= prune_min_count);
        }

        // Build Witten-Bell history tables:
        // history -> (C(h), T(h), Map<u, C(h, u)>)
        let mut histories: HashMap<Vec<UnitId>, HashMap<UnitId, f64>> = HashMap::new();
        for (ng, count) in ngram_counts {
            let hist = ng[0..ng.len() - 1].to_vec();
            let u = ng[ng.len() - 1];
            histories.entry(hist).or_default().insert(u, count);
        }

        let mut compiled_histories = HashMap::with_capacity(histories.len());
        for (hist, continuations) in histories {
            let c_h: f64 = continuations.values().sum();
            let t_h = continuations.len();
            compiled_histories.insert(hist, (c_h, t_h, continuations));
        }

        Self {
            order,
            vocab,
            histories: compiled_histories,
            unigram_counts,
            total_unigrams,
        }
    }

    /// Evaluates log-probability of unit `u` given context `history` using Witten-Bell smoothing.
    pub fn score_log(&self, u: UnitId, history: &[UnitId]) -> f64 {
        let prob = self.score_prob(u, history);
        if prob <= 0.0 {
            -50.0 // Bounded floor for zero probability
        } else {
            prob.ln()
        }
    }

    fn score_prob(&self, u: UnitId, history: &[UnitId]) -> f64 {
        let max_hist_len = self.order - 1;
        let start = if history.len() > max_hist_len {
            history.len() - max_hist_len
        } else {
            0
        };
        let curr_hist = &history[start..];

        for h_len in (1..=curr_hist.len()).rev() {
            let h_sub = &curr_hist[curr_hist.len() - h_len..];
            if let Some((c_h, t_h, continuations)) = self.histories.get(h_sub) {
                let denominator = *c_h + *t_h as f64;
                if denominator > 0.0 {
                    let lambda = *t_h as f64 / denominator;
                    let direct_prob = continuations.get(&u).copied().unwrap_or(0.0) / denominator;
                    let backoff_prob = self.score_prob(u, &h_sub[1..]);
                    return direct_prob + lambda * backoff_prob;
                }
            }
        }

        // Base case: unigram
        let c_u = self.unigram_counts.get(&u).copied().unwrap_or(0.0);
        let v = self.vocab.size() as f64;
        (c_u + 1.0) / (self.total_unigrams + v)
    }

    /// Serialized size in bytes.
    pub fn serialized_size_bytes(&self) -> usize {
        bincode::serialize(self).map(|bytes| bytes.len()).unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
struct BeamCandidate {
    pos: usize,
    history: Vec<UnitId>,
    bengali: String,
    score: f64,
    consecutive_eps: u8,
}

/// N-gram Beam Search Decoder (Pure Sequence Model: No Lexicon, No Avro).
pub fn decode_beam_ngram(
    roman_tokens: &[String],
    lm: &JointUnitLM,
    beam_width: usize,
) -> Vec<(String, f64)> {
    let m = roman_tokens.len();
    if m == 0 {
        return Vec::new();
    }

    let mut beam = vec![BeamCandidate {
        pos: 0,
        history: vec![BOS_TOKEN],
        bengali: String::new(),
        score: 0.0,
        consecutive_eps: 0,
    }];

    for _step in 0..m * 2 + 4 {
        let mut next_beam: Vec<BeamCandidate> = Vec::with_capacity(beam.len() * 8);

        for cand in &beam {
            if cand.pos == m {
                // Already fully consumed Roman; can stay in next beam
                next_beam.push(cand.clone());
                continue;
            }

            // 1. Try consuming matching Roman tokens (kr = 1 or 2)
            for kr in 1..=2.min(m - cand.pos) {
                let r_slice = &roman_tokens[cand.pos..cand.pos + kr];
                let r_str = r_slice.join("");

                if let Some(matching_units) = lm.vocab.roman_to_units.get(&r_str) {
                    for &u in matching_units {
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
            }

            // 2. Try epsilon Roman insertion (r = ""): max 1 consecutive to avoid infinite cycles
            if cand.consecutive_eps < 1 {
                if let Some(empty_r_units) = lm.vocab.roman_to_units.get("") {
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
        }

        // Deduplicate hypotheses by (pos, bengali) keeping highest score
        let mut dedup_map: HashMap<(usize, String), BeamCandidate> = HashMap::with_capacity(next_beam.len());
        for cand in next_beam {
            let key = (cand.pos, cand.bengali.clone());
            if let Some(existing) = dedup_map.get_mut(&key) {
                if cand.score > existing.score {
                    *existing = cand;
                }
            } else {
                dedup_map.insert(key, cand);
            }
        }

        let mut sorted_cands: Vec<BeamCandidate> = dedup_map.into_values().collect();
        sorted_cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        sorted_cands.truncate(beam_width);

        beam = sorted_cands;
        if beam.iter().all(|c| c.pos == m) {
            break;
        }
    }

    // Add EOS transition for completed candidates
    let mut final_cands: Vec<(String, f64)> = Vec::new();
    let mut seen_words: HashSet<String> = HashSet::new();

    for cand in beam {
        if cand.pos == m {
            let eos_log = lm.score_log(EOS_TOKEN, &cand.history);
            let final_score = cand.score + eos_log;
            if seen_words.insert(cand.bengali.clone()) {
                final_cands.push((cand.bengali, final_score));
            }
        }
    }

    final_cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    final_cands
}

#[derive(Debug, Clone)]
pub struct DevEvaluationMetrics {
    pub order: usize,
    pub prune_level: usize,
    pub prune_desc: String,
    pub weighting: String,
    pub size_kb: f64,
    pub top1_pct: f64,
    pub top3_pct: f64,
    pub reach_pct: f64,
    pub p95_latency_us: f64,
}

pub fn evaluate_on_dev(
    dev_pairs: &[AlignerPair],
    lm: &JointUnitLM,
    order: usize,
    prune_level: usize,
    prune_desc: &str,
    weighting: &str,
    beam_width: usize,
) -> DevEvaluationMetrics {
    let size_bytes = lm.serialized_size_bytes();
    let size_kb = size_bytes as f64 / 1024.0;

    let total = dev_pairs.len();
    let mut latencies: Vec<f64> = Vec::with_capacity(total);

    let results: Vec<(bool, bool, bool, f64)> = dev_pairs
        .par_iter()
        .map(|pair| {
            let t0 = Instant::now();
            let cands = decode_beam_ngram(&pair.roman_tokens, lm, beam_width);
            let elapsed_us = t0.elapsed().as_micros() as f64;

            let target = &pair.raw_bengali;
            let is_top1 = cands.first().map(|c| &c.0 == target).unwrap_or(false);
            let is_top3 = cands.iter().take(3).any(|c| &c.0 == target);
            let is_reach = cands.iter().any(|c| &c.0 == target);

            (is_top1, is_top3, is_reach, elapsed_us)
        })
        .collect();

    let mut top1_count = 0;
    let mut top3_count = 0;
    let mut reach_count = 0;

    for (t1, t3, reach, lat) in results {
        if t1 {
            top1_count += 1;
        }
        if t3 {
            top3_count += 1;
        }
        if reach {
            reach_count += 1;
        }
        latencies.push(lat);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95_latency_us = latencies[(latencies.len() as f64 * 0.95) as usize];

    DevEvaluationMetrics {
        order,
        prune_level,
        prune_desc: prune_desc.to_string(),
        weighting: weighting.to_string(),
        size_kb,
        top1_pct: (top1_count as f64 / total as f64) * 100.0,
        top3_pct: (top3_count as f64 / total as f64) * 100.0,
        reach_pct: (reach_count as f64 / total as f64) * 100.0,
        p95_latency_us,
    }
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════════════════════");
    println!("   🔬 PRAGYAN PHASE 1 STEP 7: JOINT TRANSLATION UNIT N-GRAM LANGUAGE MODEL");
    println!("══════════════════════════════════════════════════════════════════════════════════");

    let train_path = PathBuf::from("data/eval/bn.translit.sampled.train.tsv");
    let dev_path = PathBuf::from("data/eval/bn.translit.sampled.dev.tsv");

    println!("\n▶️  Loading Datasets (Filters OFF Policy)...");
    let (train_pairs, train_dropped) = load_dataset(&train_path);
    let (dev_pairs, dev_dropped) = load_dataset(&dev_path);

    println!(
        "   - Train Pairs: {} (Dropped: {}, {:.4}%)",
        train_pairs.len(),
        train_dropped,
        (train_dropped as f64 / (train_pairs.len() + train_dropped) as f64) * 100.0
    );
    println!(
        "   - Dev Pairs:   {} (Dropped: {}, {:.4}%)",
        dev_pairs.len(),
        dev_dropped,
        (dev_dropped as f64 / (dev_pairs.len() + dev_dropped) as f64) * 100.0
    );

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 7.1: EM Alignment & Viterbi Sequence Extraction
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n▶️  Training Step 4 M2M 2:1 Aligner on 94,546 Train Pairs (α = 50)...");
    let t_em = Instant::now();
    let aligner_model = train_m2m_aligner(&train_pairs, DEFAULT_EM_ITERATIONS, 50.0, 2, 1);
    println!("   ✓ Aligner training complete in {:.2?}", t_em.elapsed());

    println!("\n▶️  Extracting Viterbi Unit Sequences for TRAIN...");
    let t_vit = Instant::now();

    let alignments_tsv_path = PathBuf::from("data/eval/train_alignments.tsv");
    let mut alignments_tsv = File::create(&alignments_tsv_path).expect("Create train_alignments.tsv");
    writeln!(alignments_tsv, "id\tbengali\troman\tcount\tunit_alignment").unwrap();

    let mut raw_train_sequences: Vec<(Vec<(String, String)>, f64)> = Vec::with_capacity(train_pairs.len());
    let mut all_unique_units: HashSet<(String, String)> = HashSet::new();

    use lekhani_core::tokenization::roman_segmentation_lattice;
    for (i, pair) in train_pairs.iter().enumerate() {
        let lattice_paths = roman_segmentation_lattice(&pair.roman_norm);
        let mut best_score = f64::NEG_INFINITY;
        let mut align = Vec::new();

        for path in &lattice_paths {
            let mut temp_pair = pair.clone();
            temp_pair.roman_tokens = path.clone();
            let (score, a) = viterbi_forced_alignment(&temp_pair, &aligner_model, 2, 1);
            if score > best_score {
                best_score = score;
                align = a;
            }
        }

        if best_score == f64::NEG_INFINITY {
            let (score, a) = viterbi_forced_alignment(pair, &aligner_model, 2, 1);
            if score > best_score {
                align = a;
            }
        }

        if align.is_empty() {
            continue;
        }

        for (r, b) in &align {
            all_unique_units.insert((r.clone(), b.clone()));
        }

        let align_str: String = align
            .iter()
            .map(|(r, b)| format!("{}→{}", if r.is_empty() { "ε" } else { r }, if b.is_empty() { "ε" } else { b }))
            .collect::<Vec<_>>()
            .join(" | ");

        writeln!(
            alignments_tsv,
            "{}\t{}\t{}\t{}\t{}",
            i + 1,
            pair.raw_bengali,
            pair.raw_roman,
            pair.count,
            align_str
        )
        .unwrap();

        raw_train_sequences.push((align, pair.count));
    }

    println!(
        "   ✓ Extracted {} aligned sequences ({} unique translation units) in {:.2?}",
        raw_train_sequences.len(),
        all_unique_units.len(),
        t_vit.elapsed()
    );
    println!("   💾 Saved Viterbi alignments to {:?}", alignments_tsv_path);

    // Build vocabulary
    let mut sorted_units: Vec<(String, String)> = all_unique_units.into_iter().collect();
    sorted_units.sort();
    let vocab = UnitVocabulary::new(sorted_units);
    println!("   ✓ Compiled Unit Vocabulary: {} tokens (including <s>, </s>)", vocab.size());

    // Encode sequences to UnitIds
    let mut row_level_sequences: Vec<(Vec<UnitId>, f64)> = Vec::with_capacity(raw_train_sequences.len());
    let mut type_level_sequences: Vec<(Vec<UnitId>, f64)> = Vec::with_capacity(raw_train_sequences.len());

    for (seq, count) in &raw_train_sequences {
        let id_seq: Vec<UnitId> = seq
            .iter()
            .map(|(r, b)| vocab.get_id(r, b).unwrap())
            .collect();
        row_level_sequences.push((id_seq.clone(), *count));
        type_level_sequences.push((id_seq, 1.0));
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 7.2: Weighting Scheme Comparison on DEV (Order = 3, Prune = 0)
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   🔬 WEIGHTING SCHEME COMPARISON: Row-Level (Dakshina Freq) vs Type-Level (1.0)");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let lm_row_3 = JointUnitLM::train(&row_level_sequences, vocab.clone(), 3, 1.0);
    let lm_type_3 = JointUnitLM::train(&type_level_sequences, vocab.clone(), 3, 1.0);

    let m_row = evaluate_on_dev(&dev_pairs, &lm_row_3, 3, 0, "No Pruning", "Row-Level", 16);
    let m_type = evaluate_on_dev(&dev_pairs, &lm_type_3, 3, 0, "No Pruning", "Type-Level", 16);

    println!("   ├─ Row-Level (Dakshina counts): Top-1 = {:.2}%, Top-3 = {:.2}%, Reach = {:.2}%, Size = {:.1} KB",
        m_row.top1_pct, m_row.top3_pct, m_row.reach_pct, m_row.size_kb);
    println!("   ├─ Type-Level (Uniform 1.0)   : Top-1 = {:.2}%, Top-3 = {:.2}%, Reach = {:.2}%, Size = {:.1} KB",
        m_type.top1_pct, m_type.top3_pct, m_type.reach_pct, m_type.size_kb);

    let (chosen_weighting, chosen_seqs) = if m_row.top1_pct >= m_type.top1_pct {
        println!("   👉 Decision on DEV: Selected ROW-LEVEL weighting (Top-1: {:.2}% >= {:.2}%)", m_row.top1_pct, m_type.top1_pct);
        ("Row-Level", &row_level_sequences)
    } else {
        println!("   👉 Decision on DEV: Selected TYPE-LEVEL weighting (Top-1: {:.2}% > {:.2}%)", m_type.top1_pct, m_row.top1_pct);
        ("Type-Level", &type_level_sequences)
    };

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 7.3: Full Sweeps Across Orders (3, 4, 5) and Prune Levels (0, 1, 2)
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   🔬 FULL GRID SWEEP: Order (3, 4, 5) × Prune Level (0, 1, 2) on Held-Out DEV");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let orders = [3, 4, 5];
    let prune_configs = [
        (0, "Count >= 1 (Unpruned)", 1.0),
        (1, "Count >= 2 (Pruned singletons)", 2.0),
        (2, "Count >= 5 (Aggressive prune)", 5.0),
    ];

    let mut sweep_results = Vec::new();

    for &order in &orders {
        for &(p_lvl, p_desc, min_cnt) in &prune_configs {
            print!("▶️  Evaluating Order = {}, Prune Level = {} ({})... ", order, p_lvl, p_desc);
            let t0 = Instant::now();
            let lm = JointUnitLM::train(chosen_seqs, vocab.clone(), order, min_cnt);
            let metrics = evaluate_on_dev(&dev_pairs, &lm, order, p_lvl, p_desc, chosen_weighting, 16);
            println!("Done in {:.2?}: Top-1 = {:.2}%, Reach = {:.2}%, Size = {:.1} KB",
                t0.elapsed(), metrics.top1_pct, metrics.reach_pct, metrics.size_kb);
            sweep_results.push(metrics);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 7.4: Print Standard Work Order Markdown Table
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════════════════════════");
    println!("   📊 WORK ORDER STEP 7 REPORT TABLE: Order × Prune Level → DEV Performance");
    println!("══════════════════════════════════════════════════════════════════════════════════");
    println!("| Order | Prune Level | Description | Weighting | File Size (KB) | DEV Top-1 (%) | DEV Top-3 (%) | Reachability (%) | p95 Latency (µs) |");
    println!("|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|");
    for m in &sweep_results {
        println!(
            "| **{}** | Level {} | {} | {} | **{:.1}** | **{:.2}%** | {:.2}% | {:.2}% | {:.1} µs |",
            m.order, m.prune_level, m.prune_desc, m.weighting, m.size_kb, m.top1_pct, m.top3_pct, m.reach_pct, m.p95_latency_us
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 7.5: Gate G1 Verification
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   🛡️ GATE G1 VERIFICATION: N-Gram Alone > Raw Avro on DEV");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let raw_avro_dev_top1 = 12.37; // from baseline.json
    let best_top1 = sweep_results.iter().map(|m| m.top1_pct).fold(0.0f64, f64::max);
    let best_model_entry = sweep_results.iter().find(|m| (m.top1_pct - best_top1).abs() < 1e-4).unwrap();

    println!("   - Raw Avro DEV Top-1 (baseline.json): {:.2}%", raw_avro_dev_top1);
    println!("   - Best N-Gram Alone DEV Top-1       : {:.2}% (Order {}, Prune Level {})",
        best_top1, best_model_entry.order, best_model_entry.prune_level);

    if best_top1 > raw_avro_dev_top1 {
        let gain = best_top1 - raw_avro_dev_top1;
        println!("   🎉 GATE G1 PASSED! N-gram alone ({:.2}%) outperforms Raw Avro ({:.2}%) by +{:.2}% points!",
            best_top1, raw_avro_dev_top1, gain);
    } else {
        println!("   ❌ GATE G1 FAILED: N-gram alone ({:.2}%) did not beat Raw Avro ({:.2}%)",
            best_top1, raw_avro_dev_top1);
    }

    // Export the best trained model artifact to data/eval/joint_unit_lm_best.bin
    let best_lm = JointUnitLM::train(
        chosen_seqs,
        vocab.clone(),
        best_model_entry.order,
        if best_model_entry.prune_level == 0 { 1.0 } else if best_model_entry.prune_level == 1 { 2.0 } else { 5.0 },
    );
    let out_model_path = PathBuf::from("data/eval/joint_unit_lm_best.bin");
    let serialized_bytes = bincode::serialize(&best_lm).expect("Serialize LM");
    let mut out_file = File::create(&out_model_path).expect("Create joint_unit_lm_best.bin");
    out_file.write_all(&serialized_bytes).expect("Write serialized LM");
    println!("💾 Exported best model artifact to {:?} ({:.1} KB)", out_model_path, serialized_bytes.len() as f64 / 1024.0);
}
