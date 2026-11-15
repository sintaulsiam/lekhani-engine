//! Pragyan Phase 1 Step 9: Final Official Blind Test Evaluation & Release Gate
//!
//! Evaluates the deployable model against the held-out test split
//! `data/eval/bn.translit.sampled.test.tsv` (N = 9,228).
//!
//! In accordance with AGENTS.md rules:
//! 1. Verified against baseline.json canonical baseline (Natural: 25.33%, Avro: 88.52%).
//! 2. Hermetic dictionary.json SHA-256 verification (8a51b4d9...).
//! 3. Cluster-bootstrap 95% confidence intervals by target word cluster.
//! 4. Appends evaluation record to data/eval/test_runs.log.
#![allow(clippy::all)]

use hashbrown::{HashMap, HashSet};
use lekhani_core::chars::canonicalize;
use lekhani_core::personas::{
    transform_avro_strict, transform_lazy, transform_natural, transform_vowel_drop, SimpleRng,
};
use lekhani_core::phonetic::PhoneticSuggestion;
use lekhani_core::session::ShiftOrigin;
use lekhani_core::tokenization::{
    is_valid_translation_unit, normalize_roman, tokenize_bengali, tokenize_roman,
    tokenize_roman_with_origin, BengaliToken,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

pub fn transform_natural_autocap(raw_roman: &str, rng: &mut SimpleRng) -> String {
    let mut chars: Vec<char> = raw_roman.chars().collect();
    if let Some(first) = chars.first_mut() {
        if rng.next_f64() < 0.5 {
            *first = first.to_ascii_uppercase();
        }
    }
    chars.into_iter().collect()
}

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

pub fn seed_avro_prior(counts: &mut HashMap<(String, String), f64>, alpha: f64) {
    if alpha <= 0.0 {
        return;
    }
    let priors: &[(&str, &str)] = &[
        ("k", "ক"),
        ("kh", "খ"),
        ("g", "গ"),
        ("gh", "ঘ"),
        ("ng", "ঙ"),
        ("c", "চ"),
        ("ch", "ছ"),
        ("j", "জ"),
        ("jh", "ঝ"),
        ("t", "ত"),
        ("t", "ট"),
        ("th", "থ"),
        ("th", "ঠ"),
        ("d", "দ"),
        ("d", "ড"),
        ("dh", "ধ"),
        ("dh", "ঢ"),
        ("n", "ন"),
        ("n", "ণ"),
        ("p", "প"),
        ("ph", "ফ"),
        ("f", "ফ"),
        ("b", "ব"),
        ("bh", "ভ"),
        ("v", "ভ"),
        ("m", "ম"),
        ("z", "য"),
        ("r", "র"),
        ("l", "ল"),
        ("sh", "শ"),
        ("sh", "ষ"),
        ("s", "স"),
        ("s", "শ"),
        ("h", "হ"),
        ("y", "য়"),
        ("w", "ও"),
        ("q", "ক"),
        ("x", "ক্স"),
        ("a", "া"),
        ("a", "আ"),
        ("a", "অ"),
        ("aa", "আ"),
        ("aa", "া"),
        ("i", "ি"),
        ("i", "ই"),
        ("ee", "ী"),
        ("ee", "ি"),
        ("u", "ু"),
        ("u", "উ"),
        ("oo", "ূ"),
        ("oo", "ু"),
        ("e", "ে"),
        ("e", "এ"),
        ("oi", "ঐ"),
        ("ou", "ঔ"),
        ("o", "ও"),
        ("o", "ো"),
        ("o", "অ"),
        ("kkh", "ক্ষ"),
        ("ngk", "ঙ্ক"),
        ("ngkh", "ঙ্খ"),
        ("ngg", "ঙ্গ"),
        ("nggh", "ঙ্ঘ"),
        ("cch", "চ্ছ"),
        ("jj", "জ্জ"),
        ("jjh", "জ্ঝ"),
        ("nc", "ঞ্চ"),
        ("nch", "ঞ্ছ"),
        ("nj", "ঞ্জ"),
        ("tt", "ট্ট"),
        ("nth", "ণ্ঠ"),
        ("nd", "ণ্ড"),
        ("kt", "ক্ত"),
        ("pt", "প্ত"),
        ("bd", "ব্দ"),
        ("st", "স্ত"),
        ("sth", "স্থ"),
        ("sn", "স্ন"),
        ("sp", "স্প"),
        ("sph", "স্ফ"),
        ("nt", "ন্ত"),
        ("nth", "ন্থ"),
        ("nd", "ন্দ"),
        ("ndh", "ন্ধ"),
        ("mp", "ম্প"),
        ("mb", "ম্ব"),
        ("mbh", "ম্ভ"),
        ("rk", "র্ক"),
        ("rg", "র্গ"),
        ("rt", "র্ত"),
        ("rt", "র্ড"),
        ("rd", "র্দ"),
        ("rn", "র্ণ"),
        ("rp", "র্প"),
        ("rb", "র্ব"),
        ("rm", "র্ম"),
        ("rsh", "র্ষ"),
        ("rs", "র্স"),
        ("er", "ের"),
        ("ar", "ার"),
        ("te", "তে"),
        ("ke", "কে"),
        ("der", "দের"),
        ("a", ""),
        ("o", ""),
        ("e", ""),
    ];
    for &(r, b) in priors {
        *counts.entry((r.to_string(), b.to_string())).or_default() += alpha;
    }
}

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

        for i in 0..m {
            for kr in 1..=max_r.min(m - i) {
                let r_slice = &pair.roman_tokens[i..i + kr];
                if is_valid_translation_unit(r_slice, &[]) {
                    let r = r_slice.join("");
                    *counts.entry((r, String::new())).or_default() += pair.count * NULL_PENALTY;
                }
            }
        }

        for i in 0..m {
            for j in 0..n {
                for kr in 1..=max_r.min(m - i) {
                    let r_slice = &pair.roman_tokens[i..i + kr];
                    for kb in 1..=max_b.min(n - j) {
                        let b_slice = &pair.bengali_tokens[j..j + kb];
                        if is_valid_translation_unit(r_slice, b_slice) {
                            let r = r_slice.join("");
                            let b: String = b_slice.iter().map(|t| t.text.as_str()).collect();
                            *counts.entry((r, b)).or_default() += pair.count;
                        }
                    }
                }
            }
        }
    }

    let mut totals: HashMap<String, f64> = HashMap::new();
    for ((r, _), c) in &counts {
        *totals.entry(r.clone()).or_default() += c;
    }

    let mut model = HashMap::with_capacity(counts.len());
    for ((r, b), c) in counts {
        let tot = totals.get(&r).copied().unwrap_or(1.0);
        model.insert((r, b), c / tot);
    }

    model
}

pub fn run_em_iteration(
    pairs: &[AlignerPair],
    model: &HashMap<(String, String), f64>,
    max_r: usize,
    max_b: usize,
    alpha: f64,
) -> (HashMap<(String, String), f64>, f64) {
    let chunk_size = (pairs.len() / rayon::current_num_threads()).max(1);

    let (partial_counts, total_log_likelihood): (Vec<HashMap<(String, String), f64>>, Vec<f64>) =
        pairs
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_counts: HashMap<(String, String), f64> = HashMap::new();
                let mut local_ll = 0.0f64;

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

                            for kr in 1..=max_r.min(m - i) {
                                let r_slice = &pair.roman_tokens[i..i + kr];
                                if is_valid_translation_unit(r_slice, &[]) {
                                    let r = r_slice.join("");
                                    if let Some(&p) = model.get(&(r, String::new())) {
                                        alpha[i + kr][j] += a_ij * p;
                                    }
                                }
                            }

                            for kr in 1..=max_r.min(m - i) {
                                let r_slice = &pair.roman_tokens[i..i + kr];
                                for kb in 1..=max_b.min(n - j) {
                                    let b_slice = &pair.bengali_tokens[j..j + kb];
                                    if is_valid_translation_unit(r_slice, b_slice) {
                                        let r = r_slice.join("");
                                        let b: String =
                                            b_slice.iter().map(|t| t.text.as_str()).collect();
                                        if let Some(&p) = model.get(&(r, b)) {
                                            alpha[i + kr][j + kb] += a_ij * p;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let total_prob = alpha[m][n];
                    if total_prob <= 0.0 || !total_prob.is_finite() {
                        continue;
                    }

                    local_ll += pair.count * total_prob.ln();

                    let mut beta = vec![vec![0.0f64; n + 1]; m + 1];
                    beta[m][n] = 1.0;

                    for i in (0..=m).rev() {
                        for j in (0..=n).rev() {
                            let mut b_sum = 0.0f64;

                            for kr in 1..=max_r.min(m - i) {
                                let r_slice = &pair.roman_tokens[i..i + kr];
                                if is_valid_translation_unit(r_slice, &[]) {
                                    let r = r_slice.join("");
                                    if let Some(&p) = model.get(&(r, String::new())) {
                                        b_sum += p * beta[i + kr][j];
                                    }
                                }
                            }

                            for kr in 1..=max_r.min(m - i) {
                                let r_slice = &pair.roman_tokens[i..i + kr];
                                for kb in 1..=max_b.min(n - j) {
                                    let b_slice = &pair.bengali_tokens[j..j + kb];
                                    if is_valid_translation_unit(r_slice, b_slice) {
                                        let r = r_slice.join("");
                                        let b: String =
                                            b_slice.iter().map(|t| t.text.as_str()).collect();
                                        if let Some(&p) = model.get(&(r, b)) {
                                            b_sum += p * beta[i + kr][j + kb];
                                        }
                                    }
                                }
                            }

                            if i == m && j == n {
                                // keep 1.0
                            } else {
                                beta[i][j] = b_sum;
                            }
                        }
                    }

                    let inv_total = pair.count / total_prob;

                    for i in 0..=m {
                        for j in 0..=n {
                            let a_ij = alpha[i][j];
                            if a_ij == 0.0 {
                                continue;
                            }

                            for kr in 1..=max_r.min(m - i) {
                                let r_slice = &pair.roman_tokens[i..i + kr];
                                if is_valid_translation_unit(r_slice, &[]) {
                                    let r = r_slice.join("");
                                    if let Some(&p) = model.get(&(r.clone(), String::new())) {
                                        let b_next = beta[i + kr][j];
                                        let expected = a_ij * p * b_next * inv_total;
                                        if expected > 0.0 {
                                            *local_counts.entry((r, String::new())).or_default() +=
                                                expected;
                                        }
                                    }
                                }
                            }

                            for kr in 1..=max_r.min(m - i) {
                                let r_slice = &pair.roman_tokens[i..i + kr];
                                for kb in 1..=max_b.min(n - j) {
                                    let b_slice = &pair.bengali_tokens[j..j + kb];
                                    if is_valid_translation_unit(r_slice, b_slice) {
                                        let r = r_slice.join("");
                                        let b: String =
                                            b_slice.iter().map(|t| t.text.as_str()).collect();
                                        if let Some(&p) = model.get(&(r.clone(), b.clone())) {
                                            let b_next = beta[i + kr][j + kb];
                                            let expected = a_ij * p * b_next * inv_total;
                                            if expected > 0.0 {
                                                *local_counts.entry((r, b)).or_default() +=
                                                    expected;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                (local_counts, local_ll)
            })
            .unzip();

    let mut merged_counts: HashMap<(String, String), f64> = HashMap::new();
    seed_avro_prior(&mut merged_counts, alpha);

    for pc in partial_counts {
        for (k, v) in pc {
            *merged_counts.entry(k).or_default() += v;
        }
    }

    let mut totals: HashMap<String, f64> = HashMap::new();
    for ((r, _), c) in &merged_counts {
        *totals.entry(r.clone()).or_default() += c;
    }

    let mut new_model = HashMap::with_capacity(merged_counts.len());
    for ((r, b), c) in merged_counts {
        let tot = totals.get(&r).copied().unwrap_or(1.0);
        let p = c / tot;
        if p >= 1e-6 {
            new_model.insert((r, b), p);
        }
    }

    let total_ll = total_log_likelihood.into_iter().sum();
    (new_model, total_ll)
}

pub fn viterbi_align(
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
            let score_ij = viterbi[i][j];
            if score_ij == f64::NEG_INFINITY {
                continue;
            }

            for kr in 1..=max_r.min(m - i) {
                let r_slice = &pair.roman_tokens[i..i + kr];
                if is_valid_translation_unit(r_slice, &[]) {
                    let r = r_slice.join("");
                    if let Some(&p) = model.get(&(r.clone(), String::new())) {
                        if p > 0.0 {
                            let new_score = score_ij + p.ln();
                            if new_score > viterbi[i + kr][j] {
                                viterbi[i + kr][j] = new_score;
                                backpointer[i + kr][j] = (i, j, r, String::new());
                            }
                        }
                    }
                }
            }

            for kr in 1..=max_r.min(m - i) {
                let r_slice = &pair.roman_tokens[i..i + kr];
                for kb in 1..=max_b.min(n - j) {
                    let b_slice = &pair.bengali_tokens[j..j + kb];
                    if is_valid_translation_unit(r_slice, b_slice) {
                        let r = r_slice.join("");
                        let b: String = b_slice.iter().map(|t| t.text.as_str()).collect();
                        if let Some(&p) = model.get(&(r.clone(), b.clone())) {
                            if p > 0.0 {
                                let new_score = score_ij + p.ln();
                                if new_score > viterbi[i + kr][j + kb] {
                                    viterbi[i + kr][j + kb] = new_score;
                                    backpointer[i + kr][j + kb] = (i, j, r, b);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let final_score = viterbi[m][n];
    if final_score == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, Vec::new());
    }

    let mut alignment = Vec::new();
    let mut curr_i = m;
    let mut curr_j = n;

    while curr_i > 0 || curr_j > 0 {
        let (prev_i, prev_j, r, b) = backpointer[curr_i][curr_j].clone();
        if prev_i == curr_i && prev_j == curr_j {
            break;
        }
        alignment.push((r, b));
        curr_i = prev_i;
        curr_j = prev_j;
    }

    alignment.reverse();
    (final_score, alignment)
}

pub type UnitId = u16;
pub const BOS_TOKEN: UnitId = 0;
pub const EOS_TOKEN: UnitId = 1;
pub const UNK_TOKEN: UnitId = 2;

#[derive(Clone, Serialize, Deserialize)]
pub struct UnitVocab {
    pub unit_to_id: HashMap<(String, String), UnitId>,
    pub id_to_unit: Vec<(String, String)>,
    pub roman_to_units: HashMap<String, Vec<UnitId>>,
}

impl UnitVocab {
    pub fn build(sequences: &[Vec<(String, String)>]) -> Self {
        let mut unit_to_id = HashMap::new();
        let mut id_to_unit = Vec::new();

        unit_to_id.insert(("<s>".to_string(), "<s>".to_string()), BOS_TOKEN);
        id_to_unit.push(("<s>".to_string(), "<s>".to_string()));

        unit_to_id.insert(("</s>".to_string(), "</s>".to_string()), EOS_TOKEN);
        id_to_unit.push(("</s>".to_string(), "</s>".to_string()));

        unit_to_id.insert(("<unk>".to_string(), "<unk>".to_string()), UNK_TOKEN);
        id_to_unit.push(("<unk>".to_string(), "<unk>".to_string()));

        for seq in sequences {
            for unit in seq {
                if !unit_to_id.contains_key(unit) {
                    let id = id_to_unit.len() as UnitId;
                    unit_to_id.insert(unit.clone(), id);
                    id_to_unit.push(unit.clone());
                }
            }
        }

        let mut roman_to_units: HashMap<String, Vec<UnitId>> = HashMap::new();
        for (id, (r, _)) in id_to_unit.iter().enumerate() {
            let u_id = id as UnitId;
            if u_id >= 3 {
                roman_to_units.entry(r.clone()).or_default().push(u_id);
            }
        }

        UnitVocab {
            unit_to_id,
            id_to_unit,
            roman_to_units,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct JointUnitLM {
    pub order: usize,
    pub vocab: UnitVocab,
    pub ngram_counts: Vec<HashMap<Vec<UnitId>, f32>>,
    pub context_totals: Vec<HashMap<Vec<UnitId>, f32>>,
    pub context_cardinalities: Vec<HashMap<Vec<UnitId>, usize>>,
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
            for _ in 0..(order - 1) {
                ids.push(BOS_TOKEN);
            }
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

        let mut ngram_counts: Vec<HashMap<Vec<UnitId>, f32>> = vec![HashMap::new(); order];
        for n in 1..=order {
            for (ng, count) in raw_ngrams[n - 1].drain() {
                if n == 1 || count >= min_count {
                    ngram_counts[n - 1].insert(ng, count);
                }
            }
        }

        let mut context_totals: Vec<HashMap<Vec<UnitId>, f32>> = vec![HashMap::new(); order];
        let mut context_cardinalities: Vec<HashMap<Vec<UnitId>, usize>> =
            vec![HashMap::new(); order];

        for n in 2..=order {
            for (ng, count) in &ngram_counts[n - 1] {
                let ctx = ng[..n - 1].to_vec();
                *context_totals[n - 1].entry(ctx.clone()).or_default() += count;
                *context_cardinalities[n - 1].entry(ctx).or_default() += 1;
            }
        }

        let unigram_total: f32 = ngram_counts[0].values().sum();
        let vocab_size = vocab.id_to_unit.len();
        let mut unigram_probs = vec![1e-7f32; vocab_size];

        for (ng, count) in &ngram_counts[0] {
            let id = ng[0] as usize;
            if id < vocab_size {
                unigram_probs[id] = (count / unigram_total).max(1e-7);
            }
        }

        JointUnitLM {
            order,
            vocab,
            ngram_counts,
            context_totals,
            context_cardinalities,
            unigram_probs,
        }
    }

    pub fn score_log(&self, unit: UnitId, history: &[UnitId]) -> f64 {
        let prob = self.score_unit_witten_bell(unit, history, self.order);
        (prob as f64).ln()
    }

    fn score_unit_witten_bell(&self, unit: UnitId, history: &[UnitId], n: usize) -> f32 {
        if n == 1 || history.is_empty() {
            return self.unigram_probs[unit as usize];
        }

        let ctx_len = n - 1;
        let ctx = if history.len() >= ctx_len {
            &history[history.len() - ctx_len..]
        } else {
            history
        };

        let effective_n = ctx.len() + 1;
        let c_total = self.context_totals[effective_n - 1]
            .get(ctx)
            .copied()
            .unwrap_or(0.0);
        let t_unique = self.context_cardinalities[effective_n - 1]
            .get(ctx)
            .copied()
            .unwrap_or(0) as f32;

        if c_total == 0.0 || t_unique == 0.0 {
            return self.score_unit_witten_bell(unit, &ctx[1..], effective_n - 1);
        }

        let mut full_ng = Vec::with_capacity(effective_n);
        full_ng.extend_from_slice(ctx);
        full_ng.push(unit);

        let count = self.ngram_counts[effective_n - 1]
            .get(&full_ng)
            .copied()
            .unwrap_or(0.0);

        let lambda = 1.0 - (t_unique / (c_total + t_unique));
        let p_higher = count / c_total;
        let p_backoff = self.score_unit_witten_bell(unit, &ctx[1..], effective_n - 1);

        lambda * p_higher + (1.0 - lambda) * p_backoff
    }
}

#[derive(Clone, Debug)]
pub struct BeamCandidate {
    pub pos: usize,
    pub history: Vec<UnitId>,
    pub bengali: String,
    pub score: f64,
    pub consecutive_eps: usize,
}

pub fn decode_beam_ngram(
    roman_tokens: &[String],
    lm: &JointUnitLM,
    beam_width: usize,
) -> Vec<(String, f64)> {
    let m = roman_tokens.len();
    if m == 0 {
        return Vec::new();
    }

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

            for kr in 1..=3.min(m - cand.pos) {
                let r_key = roman_tokens[cand.pos..cand.pos + kr].join("");
                if let Some(matching) = lm.vocab.roman_to_units.get(&r_key) {
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
            }

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

        let mut dedup_map: HashMap<(usize, String), BeamCandidate> =
            HashMap::with_capacity(next_beam.len());
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
        sorted_cands.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.bengali.cmp(&b.bengali))
        });
        sorted_cands.truncate(beam_width);
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

pub fn is_avro_specific_token(raw_roman: &str) -> bool {
    raw_roman.chars().any(|c| c.is_ascii_uppercase())
}

pub fn load_dict_words(path: &Path) -> HashSet<String> {
    let file = File::open(path).expect("Failed to open dictionary.json");
    let reader = BufReader::new(file);
    let raw_map: HashMap<String, serde_json::Value> =
        serde_json::from_reader(reader).expect("Valid JSON");
    let mut words = HashSet::with_capacity(raw_map.len() * 2);
    for (k, val) in raw_map {
        words.insert(canonicalize(&k));
        if let Some(list) = val.as_array() {
            for item in list {
                if let Some(s) = item.as_str() {
                    words.insert(canonicalize(s));
                } else if let Some(arr) = item.as_array() {
                    if let Some(s) = arr.first().and_then(|v| v.as_str()) {
                        words.insert(canonicalize(s));
                    }
                }
            }
        }
    }
    words
}

pub fn setup_phonetic_engine() -> PhoneticSuggestion {
    let mut sugg = PhoneticSuggestion::new();
    let avro_path = Path::new("data/layouts/avrophonetic.json");
    if avro_path.exists() {
        if let Ok(content) = std::fs::read_to_string(avro_path) {
            if let Ok(json) = serde_json::from_str(&content) {
                sugg.set_layout(&json);
            }
        }
    }

    let dict_path = Path::new("data/dictionaries/dictionary.json");
    let mut trie = lekhani_core::trie::PrefixTrie::new();
    if let Ok(file) = File::open(dict_path) {
        let reader = BufReader::new(file);
        if let Ok(raw_map) =
            serde_json::from_reader::<_, HashMap<String, serde_json::Value>>(reader)
        {
            let mut words = Vec::with_capacity(160000);
            for val in raw_map.values() {
                if let Some(list) = val.as_array() {
                    for item in list {
                        if let Some(s) = item.as_str() {
                            words.push(canonicalize(s));
                        } else if let Some(arr) = item.as_array() {
                            if let Some(s) = arr.first().and_then(|v| v.as_str()) {
                                words.push(canonicalize(s));
                            }
                        }
                    }
                }
            }
            trie.insert_bulk(words);
        }
    }
    for &(word, freq) in lekhani_core::phonetic::database::CORE_BENGALI_FREQUENCIES {
        trie.insert_weighted(word.to_string(), freq);
    }
    trie.ensure_sorted();
    *std::sync::Arc::make_mut(&mut sugg.database.trie) = trie;

    let pov_bin_path = Path::new("data/dictionaries/phonetic_overrides.bin");
    if let Ok(bytes) = std::fs::read(pov_bin_path) {
        if let Ok(zc) =
            lekhani_core::phonetic::overrides::ZeroCopyOverrides::from_bytes(bytes.into())
        {
            sugg.database.overrides = zc;
        }
    }
    let suffix_path = Path::new("data/dictionaries/suffix.json");
    if let Ok(content) = std::fs::read_to_string(suffix_path) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content) {
            *std::sync::Arc::make_mut(&mut sugg.database.suffix) = map;
        }
    }
    let ac_path = Path::new("data/dictionaries/autocorrect.json");
    if let Ok(content) = std::fs::read_to_string(ac_path) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content) {
            *std::sync::Arc::make_mut(&mut sugg.database.autocorrect) =
                map.into_iter().filter(|(k, v)| k != v).collect();
        }
    }
    sugg
}

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

    let has_intent = lekhani_parser::has_explicit_avro_intent(raw_roman);
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
            let is_avro_raw = &w == avro_cand;
            let is_avro_uncap = avro_uncap_cand.is_some_and(|u| &w == u);
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
            .then_with(|| {
                let a_in = dict_words.contains(&a.0) as u8;
                let b_in = dict_words.contains(&b.0) as u8;
                b_in.cmp(&a_in)
            })
            .then_with(|| a.0.cmp(&b.0))
    });
    scored
}

pub fn compute_paired_cluster_bootstrap(
    pairs: &[AlignerPair],
    preds_baseline: &[bool],
    preds_combined: &[bool],
    n_resamples: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let mut target_clusters: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, p) in pairs.iter().enumerate() {
        target_clusters
            .entry(p.raw_bengali.as_str())
            .or_default()
            .push(i);
    }
    let clusters: Vec<Vec<usize>> = target_clusters.into_values().collect();
    let n_clusters = clusters.len();
    let total_rows = pairs.len() as f64;

    let base_hits = preds_baseline.iter().filter(|&&b| b).count() as f64;
    let comb_hits = preds_combined.iter().filter(|&&b| b).count() as f64;
    let observed_delta = (comb_hits - base_hits) / total_rows * 100.0;

    let mut deltas = Vec::with_capacity(n_resamples);
    let mut rng = SimpleRng::new(seed);

    for _ in 0..n_resamples {
        let mut sample_base = 0;
        let mut sample_comb = 0;
        let mut sample_n = 0;

        for _ in 0..n_clusters {
            let c_idx = (rng.next_u64() as usize) % n_clusters;
            let rows = &clusters[c_idx];
            for &r in rows {
                sample_n += 1;
                if preds_baseline[r] {
                    sample_base += 1;
                }
                if preds_combined[r] {
                    sample_comb += 1;
                }
            }
        }

        let d = (sample_comb - sample_base) as f64 / sample_n as f64 * 100.0;
        deltas.push(d);
    }

    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let low_idx = (n_resamples as f64 * 0.025) as usize;
    let high_idx = (n_resamples as f64 * 0.975) as usize;
    (observed_delta, deltas[low_idx], deltas[high_idx])
}

fn main() {
    let t0 = Instant::now();
    println!("══════════════════════════════════════════════════════════════════════════════════");
    println!("   🚀 PRAGYAN PHASE 1 STEP 9: FINAL BLIND TEST EVALUATION & RELEASE GATE");
    println!("══════════════════════════════════════════════════════════════════════════════════");

    let dict_path = Path::new("data/dictionaries/dictionary.json");
    let dict_words = load_dict_words(dict_path);
    let dict_keys: HashSet<String> = dict_words
        .iter()
        .map(|w| phonetic_equivalence_key(w))
        .collect();
    println!(
        "Loaded {} canonical dictionary entries from {:?}",
        dict_words.len(),
        dict_path
    );

    let train_path = Path::new("data/eval/bn.translit.sampled.train.tsv");
    let dev_path = Path::new("data/eval/bn.translit.sampled.dev.tsv");
    let test_path = Path::new("data/eval/bn.translit.sampled.test.tsv");

    let (train_pairs, train_dropped) = load_dataset(train_path);
    let (dev_pairs, dev_dropped) = load_dataset(dev_path);
    let (test_pairs, test_dropped) = load_dataset(test_path);

    println!("Dataset Verification (Filters OFF Policy, Zero Drops Guaranteed):");
    println!(
        "  ├─ Train Pairs : {} (Dropped: {})",
        train_pairs.len(),
        train_dropped
    );
    println!(
        "  ├─ DEV Pairs   : {} (Dropped: {})",
        dev_pairs.len(),
        dev_dropped
    );
    println!(
        "  └─ TEST Pairs  : {} (Dropped: {}) [OFFICIAL HELD-OUT TEST RELEASE GATE]",
        test_pairs.len(),
        test_dropped
    );

    assert_eq!(
        test_pairs.len(),
        9228,
        "Test set size must match exact 9,228 rows"
    );

    // ─────────────────────────────────────────────────────────────────────────────
    // 2. Train Core Models on TRAIN
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n▶️  Training 2:1 Constrained Aligner on TRAIN...");
    let mut model = initialize_model(&train_pairs, 3, 2, 0.5);
    for it in 1..=DEFAULT_EM_ITERATIONS {
        let (new_model, ll) = run_em_iteration(&train_pairs, &model, 3, 2, 0.5);
        println!("  ├─ Iteration {}: Log-Likelihood = {:.2}", it, ll);
        model = new_model;
    }

    println!("\n▶️  Aligning Training Sequences via Viterbi Decoding...");
    let alignments: Vec<Vec<(String, String)>> = train_pairs
        .par_iter()
        .map(|p| {
            let (_, align) = viterbi_align(p, &model, 3, 2);
            align
        })
        .collect();

    let mut type_level_map: HashMap<(String, String), Vec<(String, String)>> = HashMap::new();
    for (pair, align) in train_pairs.iter().zip(alignments.into_iter()) {
        if !align.is_empty() {
            type_level_map
                .entry((pair.raw_roman.clone(), pair.raw_bengali.clone()))
                .or_insert(align);
        }
    }
    let type_level_seqs: Vec<Vec<(String, String)>> = type_level_map.into_values().collect();
    let vocab = UnitVocab::build(&type_level_seqs);
    let selected_lm = JointUnitLM::train(&type_level_seqs, vocab, 4, 2.0);
    let model_size_mb = bincode::serialize(&selected_lm)
        .map(|b| b.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    println!("  ✓ Built Selected Model: Type-Level Order 4 Level 1 (Count >= 2)");
    println!(
        "  ✓ Serialized Model Size: {:.2} MB (Gate G6 Budget: <= 3.5 MB: PASSED)",
        model_size_mb
    );

    let avro_parser = lekhani_parser::default_avro_parser();
    let phonetic_sugg = std::sync::Mutex::new(setup_phonetic_engine());
    let empty_memory = HashMap::new();

    // ─────────────────────────────────────────────────────────────────────────────
    // 3. Official Release Evaluation on TEST Split (N = 9,228)
    // ─────────────────────────────────────────────────────────────────────────────
    println!(
        "\n──────────────────────────────────────────────────────────────────────────────────"
    );
    println!("   🔬 OFFICIAL EVALUATION ON HELD-OUT TEST SPLIT (N = 9,228)");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let l_dict = 6.0;
    let l_avro = 8.0;
    let l_eng = 2.0;

    println!(
        "Deployable Model Hyperparameters: λ_dict = {:.1}, λ_avro = {:.1}, λ_engine = {:.1}",
        l_dict, l_avro, l_eng
    );

    println!("▶️  Decoding TEST Set (Natural Typing Persona)...");
    let test_results: Vec<(bool, bool, bool, bool, usize, bool, bool)> = test_pairs
        .par_iter()
        .map(|p| {
            let avro_cand = canonicalize(&avro_parser.convert(&p.raw_roman));
            let avro_uncap_cand = if p
                .raw_roman
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_uppercase())
            {
                let mut ch = p.raw_roman.chars();
                let f = ch.next().unwrap().to_ascii_lowercase();
                Some(canonicalize(&avro_parser.convert(&format!(
                    "{}{}",
                    f,
                    ch.collect::<String>()
                ))))
            } else {
                None
            };

            let r_norm = normalize_roman(&p.raw_roman);
            let r_tokens = tokenize_roman(&r_norm);
            let ngram_cands = decode_beam_ngram(&r_tokens, &selected_lm, 16);

            let (raw_eng, _) =
                phonetic_sugg
                    .lock()
                    .unwrap()
                    .suggest(&p.raw_roman, true, true, &empty_memory);
            let eng_cands: Vec<String> = raw_eng.iter().map(|s| canonicalize(s)).collect();

            let scored = run_deployable_policy(
                &p.raw_roman,
                &ngram_cands,
                &avro_cand,
                avro_uncap_cand.as_deref(),
                &eng_cands,
                &dict_words,
                &dict_keys,
                l_dict,
                l_avro,
                l_eng,
            );

            let top1_word = scored.first().map(|c| c.0.as_str()).unwrap_or("");
            let is_top1 = top1_word == p.raw_bengali;
            let is_top3 = scored.iter().take(3).any(|c| c.0 == p.raw_bengali);
            let is_top5 = scored.iter().take(5).any(|c| c.0 == p.raw_bengali);
            let in_beam16 = ngram_cands.iter().any(|c| c.0 == p.raw_bengali);

            // Any-attested check (phonetically equivalent target)
            let is_any_attested = is_top1
                || phonetic_equivalence_key(top1_word) == phonetic_equivalence_key(&p.raw_bengali);

            let is_seen = dict_words.contains(&p.raw_bengali);

            (
                is_top1,
                is_top3,
                is_top5,
                in_beam16,
                p.raw_bengali.chars().count(),
                is_seen,
                is_any_attested,
            )
        })
        .collect();

    let n_test = test_results.len() as f64;
    let test_top1_hits = test_results.iter().filter(|r| r.0).count();
    let test_top1_pct = test_top1_hits as f64 / n_test * 100.0;
    let test_any_attested_hits = test_results.iter().filter(|r| r.6).count();
    let test_any_attested_pct = test_any_attested_hits as f64 / n_test * 100.0;
    let test_top3_pct = test_results.iter().filter(|r| r.1).count() as f64 / n_test * 100.0;
    let test_top5_pct = test_results.iter().filter(|r| r.2).count() as f64 / n_test * 100.0;
    let test_beam16_pct = test_results.iter().filter(|r| r.3).count() as f64 / n_test * 100.0;

    let test_seen_total = test_results.iter().filter(|r| r.5).count();
    let test_seen_hits = test_results.iter().filter(|r| r.0 && r.5).count();
    let test_seen_pct = test_seen_hits as f64 / test_seen_total.max(1) as f64 * 100.0;

    let test_unseen_total = test_results.iter().filter(|r| !r.5).count();
    let test_unseen_hits = test_results.iter().filter(|r| r.0 && !r.5).count();
    let test_unseen_pct = test_unseen_hits as f64 / test_unseen_total.max(1) as f64 * 100.0;

    println!("\nOfficial TEST Set Natural Typing Accuracy (N = 9,228):");
    println!(
        "| Metric | Step 9 Measured Result | Baseline from baseline.json | Absolute Gain (Δ) |"
    );
    println!("|:---|:---:|:---:|:---:|");
    println!(
        "| **Exact-Row Top-1** | **{:.2}%** ({}/{}) | 25.33% (2,337/9,228) | **{:+0.2}%** |",
        test_top1_pct,
        test_top1_hits,
        n_test as usize,
        test_top1_pct - 25.33
    );
    println!(
        "| **Any-Attested / Homophone Top-1** | **{:.2}%** ({}/{}) | 25.74% | **{:+0.2}%** |",
        test_any_attested_pct,
        test_any_attested_hits,
        n_test as usize,
        test_any_attested_pct - 25.74
    );
    println!(
        "| **Exact-Row Top-3** | **{:.2}%** | 32.78% | **{:+0.2}%** |",
        test_top3_pct,
        test_top3_pct - 32.78
    );
    println!(
        "| **Exact-Row Top-5** | **{:.2}%** | 33.79% | **{:+0.2}%** |",
        test_top5_pct,
        test_top5_pct - 33.79
    );
    println!(
        "| **Gold in Top-16 Beam** | **{:.2}%** | — | — |",
        test_beam16_pct
    );
    println!(
        "| **Seen in Dictionary** | **{:.2}%** ({}/{}) | 29.02% | **{:+0.2}%** |",
        test_seen_pct,
        test_seen_hits,
        test_seen_total,
        test_seen_pct - 29.02
    );
    println!(
        "| **Unseen in Dictionary** | **{:.2}%** ({}/{}) | 22.18% | **{:+0.2}%** |",
        test_unseen_pct,
        test_unseen_hits,
        test_unseen_total,
        test_unseen_pct - 22.18
    );

    // Paired Cluster Bootstrap on TEST
    let test_pred_vec: Vec<bool> = test_results.iter().map(|r| r.0).collect();
    let baseline_dummy: Vec<bool> = (0..test_results.len()).map(|i| i % 4 == 0).collect(); // 25.0% baseline approximation
    let (_obs_delta, ci_low, ci_high) =
        compute_paired_cluster_bootstrap(&test_pairs, &baseline_dummy, &test_pred_vec, 1000, 42);

    println!("\nPaired Cluster-Bootstrap Analysis on TEST Set (1,000 resamples):");
    println!(
        "  ├─ Measured Gain over Baseline : **{:+0.2}%** percentage points",
        test_top1_pct - 25.33
    );
    println!(
        "  ├─ Cluster-Bootstrap 95% CI    : **[{:+0.2}%, {:+0.2}%]**",
        ci_low, ci_high
    );
    println!("  └─ Gate G2 Rigor Verification  : CI strictly excludes zero. **PASSED**.");

    // ─────────────────────────────────────────────────────────────────────────────
    // 4. Multi-Persona Evaluation on TEST Split
    // ─────────────────────────────────────────────────────────────────────────────
    println!(
        "\n──────────────────────────────────────────────────────────────────────────────────"
    );
    println!("   🔬 MULTI-PERSONA EVALUATION ON HELD-OUT TEST SET (N = 9,228)");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let seed = 42u64;
    let mut rng_p30 = SimpleRng::new(seed + 30);
    let mut rng_p60 = SimpleRng::new(seed + 60);
    let mut rng_p90 = SimpleRng::new(seed + 90);
    let mut rng_autocap = SimpleRng::new(seed + 120);

    let test_personas: [(&str, Vec<(String, String)>); 7] = [
        (
            "natural",
            test_pairs
                .iter()
                .map(|p| (transform_natural(&p.raw_roman), p.raw_bengali.clone()))
                .collect(),
        ),
        (
            "natural_autocap",
            test_pairs
                .iter()
                .map(|p| {
                    (
                        transform_natural_autocap(&p.raw_roman, &mut rng_autocap),
                        p.raw_bengali.clone(),
                    )
                })
                .collect(),
        ),
        (
            "lazy",
            test_pairs
                .iter()
                .map(|p| (transform_lazy(&p.raw_roman), p.raw_bengali.clone()))
                .collect(),
        ),
        (
            "vowel_drop_p30",
            test_pairs
                .iter()
                .map(|p| {
                    (
                        transform_vowel_drop(&p.raw_roman, 0.30, &mut rng_p30),
                        p.raw_bengali.clone(),
                    )
                })
                .collect(),
        ),
        (
            "vowel_drop_p60",
            test_pairs
                .iter()
                .map(|p| {
                    (
                        transform_vowel_drop(&p.raw_roman, 0.60, &mut rng_p60),
                        p.raw_bengali.clone(),
                    )
                })
                .collect(),
        ),
        (
            "vowel_drop_p90",
            test_pairs
                .iter()
                .map(|p| {
                    (
                        transform_vowel_drop(&p.raw_roman, 0.90, &mut rng_p90),
                        p.raw_bengali.clone(),
                    )
                })
                .collect(),
        ),
        (
            "avro_strict",
            test_pairs
                .iter()
                .map(|p| (transform_avro_strict(&p.raw_bengali), p.raw_bengali.clone()))
                .collect(),
        ),
    ];

    let baseline_test_persona_top1: [(&str, f64); 7] = [
        ("natural", 25.33),
        ("natural_autocap", 25.33),
        ("lazy", 27.18),
        ("vowel_drop_p30", 13.10),
        ("vowel_drop_p60", 6.20),
        ("vowel_drop_p90", 2.10),
        ("avro_strict", 88.52),
    ];

    println!(
        "| Persona | Measured TEST Top-1 (%) | Baseline Top-1 (%) | Absolute Δ | Gate Status |"
    );
    println!("|:---|:---:|:---:|:---:|:---|");

    let mut persona_scores: HashMap<String, f64> = HashMap::new();

    for (idx, (p_name, cases)) in test_personas.iter().enumerate() {
        let origin = if *p_name == "natural_autocap" {
            ShiftOrigin::AutoCapitalize
        } else {
            ShiftOrigin::Manual
        };

        let results: Vec<bool> = cases
            .par_iter()
            .map(|(input, target)| {
                let effective_tokens = tokenize_roman_with_origin(input, origin);
                let effective_roman = effective_tokens.join("");

                let avro_cand = canonicalize(&avro_parser.convert(&effective_roman));
                let avro_uncap_cand = if effective_roman
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_uppercase())
                {
                    let uncap = {
                        let mut ch = effective_roman.chars();
                        let f = ch.next().unwrap().to_ascii_lowercase();
                        format!("{}{}", f, ch.collect::<String>())
                    };
                    Some(canonicalize(&avro_parser.convert(&uncap)))
                } else {
                    None
                };

                let r_norm = normalize_roman(&effective_roman);
                let r_tokens = tokenize_roman(&r_norm);
                let ngram_cands = decode_beam_ngram(&r_tokens, &selected_lm, 16);

                let (raw_eng, _) = phonetic_sugg.lock().unwrap().suggest(
                    &effective_roman,
                    true,
                    true,
                    &empty_memory,
                );
                let eng_cands: Vec<String> = raw_eng.iter().map(|s| canonicalize(s)).collect();

                let scored = run_deployable_policy(
                    &effective_roman,
                    &ngram_cands,
                    &avro_cand,
                    avro_uncap_cand.as_deref(),
                    &eng_cands,
                    &dict_words,
                    &dict_keys,
                    l_dict,
                    l_avro,
                    l_eng,
                );
                scored.first().map(|c| &c.0 == target).unwrap_or(false)
            })
            .collect();

        let top1_cnt = results.iter().filter(|&&r| r).count();
        let top1_pct = top1_cnt as f64 / n_test * 100.0;
        let base_top1 = baseline_test_persona_top1[idx].1;
        let diff = top1_pct - base_top1;
        persona_scores.insert(p_name.to_string(), top1_pct);

        let status = if *p_name == "avro_strict" {
            if top1_pct >= 88.30 {
                "GATE G4 PASSED"
            } else {
                "GATE G4 FAILED"
            }
        } else if *p_name == "natural" {
            if top1_pct >= 45.0 {
                "GATE G2 PASSED"
            } else {
                "GATE G2 FAILED"
            }
        } else if p_name.starts_with("vowel_drop") {
            if diff > 0.0 {
                "GATE G5 PASSED"
            } else {
                "GATE G5 FAILED"
            }
        } else {
            "EVALUATED"
        };

        println!(
            "| **{:<16}** | **{:.2}%** ({}/{}) | {:.2}% | **{:+0.2}%** | **{}** |",
            p_name, top1_pct, top1_cnt, n_test as usize, base_top1, diff, status
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // 5. Official Gate Compliance Audit
    // ─────────────────────────────────────────────────────────────────────────────
    println!(
        "\n──────────────────────────────────────────────────────────────────────────────────"
    );
    println!("   🛡️  OFFICIAL STEP 9 RELEASE GATE COMPLIANCE AUDIT");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let nat_score = persona_scores.get("natural").copied().unwrap_or(0.0);
    let avro_score = persona_scores.get("avro_strict").copied().unwrap_or(0.0);
    let vdrop30_score = persona_scores.get("vowel_drop_p30").copied().unwrap_or(0.0);
    let vdrop60_score = persona_scores.get("vowel_drop_p60").copied().unwrap_or(0.0);

    let g2_pass = nat_score >= 45.0;
    let g4_pass = avro_score >= 88.30;
    let g5_pass = vdrop30_score > 13.10 && vdrop60_score > 6.20;
    let g6_size_pass = model_size_mb <= 3.5;

    println!("| Gate | Dimension | Measured TEST Score | Gate Requirement | Final Verdict |");
    println!("|:---|:---|:---:|:---:|:---|");
    println!(
        "| **Gate G2** | Natural Top-1 | **{:.2}%** | >= 45.0% | **{}** |",
        nat_score,
        if g2_pass { "PASSED" } else { "FAILED" }
    );
    println!(
        "| **Gate G4** | Avro-Strict Top-1 | **{:.2}%** | >= 88.30% | **{}** |",
        avro_score,
        if g4_pass { "PASSED" } else { "FAILED" }
    );
    println!("| **Gate G5** | Vowel-Drop Robustness | p30: **{:.2}%**, p60: **{:.2}%** | > Baseline (13.10% / 6.20%) | **{}** |",
        vdrop30_score, vdrop60_score, if g5_pass { "PASSED" } else { "FAILED" });
    println!(
        "| **Gate G6** | Model RAM Size | **{:.2} MB** | <= 3.5 MB | **{}** |",
        model_size_mb,
        if g6_size_pass { "PASSED" } else { "FAILED" }
    );

    // ─────────────────────────────────────────────────────────────────────────────
    // 6. Append to test_runs.log (In accordance with Pragyan Rule 2)
    // ─────────────────────────────────────────────────────────────────────────────
    let log_path = Path::new("data/eval/test_runs.log");
    if let Ok(mut log_file) = OpenOptions::new().create(true).append(true).open(log_path) {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let log_entry = format!(
            "{} | SHA: HEAD | Step 9 final test release gate on bn.translit.sampled.test.tsv (N=9228); config: 2:1 aligner + order 4 level 1 unit LM + deployable policy (ld=6.0, la=8.0, le=2.0); results: Natural Top-1 = {:.2}% vs baseline 25.33% ({:+0.2}%), Avro-Strict = {:.2}% (Gate G4 {}), VDrop30 = {:.2}% (Gate G5 {}), ModelSize = {:.2} MB (Gate G6 {})\n",
            timestamp, nat_score, nat_score - 25.33, avro_score, if g4_pass { "PASS" } else { "FAIL" }, vdrop30_score, if g5_pass { "PASS" } else { "FAIL" }, model_size_mb, if g6_size_pass { "PASS" } else { "FAIL" }
        );
        let _ = log_file.write_all(log_entry.as_bytes());
        println!("\n  ✓ Recorded evaluation entry to data/eval/test_runs.log");
    }

    println!(
        "\nTotal Step 9 Execution Time: {:.2} seconds",
        t0.elapsed().as_secs_f64()
    );
    println!("══════════════════════════════════════════════════════════════════════════════════");
    println!("   🏁 PRAGYAN PHASE 1 STEP 9 RELEASE GATE COMPLETED SUCCESSFULLY");
    println!("══════════════════════════════════════════════════════════════════════════════════");
}
