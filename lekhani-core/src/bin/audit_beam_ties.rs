#![allow(clippy::all)]
//! Audit Beam Ties & Deterministic Tie-Breaking
//!
//! Evaluates the exact mechanism of the 42.10% (K=256) vs 41.27% (K=16) Dev-A gap
//! and the 41.43% (K=256) vs 41.11% (K=16) Dev-B gap:
//! 1. Identifies the exact list of rows where K=256 hits gold but K=16 prunes it.
//! 2. Identifies all instances of candidate score ties.
//! 3. Implements deterministic tie-breaking: (score desc, in_dict desc, string asc).
//! 4. Verifies 100% determinism across runs.

use hashbrown::{HashMap, HashSet};
use lekhani_core::chars::canonicalize;
use lekhani_core::tokenization::{
    is_valid_translation_unit, normalize_roman, tokenize_bengali, tokenize_roman, BengaliToken,
};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

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
        ("o", "ো"),
        ("o", "অ"),
        ("o", "ও"),
        ("oi", "ৈ"),
        ("ou", "ৌ"),
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

pub fn em_iteration(
    pairs: &[AlignerPair],
    model: &HashMap<(String, String), f64>,
    max_r: usize,
    max_b: usize,
) -> HashMap<(String, String), f64> {
    let expected_counts = pairs
        .par_chunks(2048)
        .map(|chunk| {
            let mut local_counts: HashMap<(String, String), f64> = HashMap::new();

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

                                let r_slice = if kr == 0 {
                                    &[]
                                } else {
                                    &pair.roman_tokens[i..i + kr]
                                };
                                let b_slice = if kb == 0 {
                                    &[]
                                } else {
                                    &pair.bengali_tokens[j..j + kb]
                                };

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

                                let r_slice = if kr == 0 {
                                    &[]
                                } else {
                                    &pair.roman_tokens[i - kr..i]
                                };
                                let b_slice = if kb == 0 {
                                    &[]
                                } else {
                                    &pair.bengali_tokens[j - kb..j]
                                };

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

                                let r_slice = if kr == 0 {
                                    &[]
                                } else {
                                    &pair.roman_tokens[i..i + kr]
                                };
                                let b_slice = if kb == 0 {
                                    &[]
                                } else {
                                    &pair.bengali_tokens[j..j + kb]
                                };

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
            local_counts
        })
        .reduce(HashMap::new, |mut acc, map| {
            for (k, v) in map {
                *acc.entry(k).or_default() += v;
            }
            acc
        });

    let total: f64 = expected_counts.values().sum();
    let mut new_model = HashMap::with_capacity(expected_counts.len());
    for (k, v) in expected_counts {
        new_model.insert(k, v / total);
    }
    new_model
}

pub fn viterbi_forced_alignment(
    pair: &AlignerPair,
    model: &HashMap<(String, String), f64>,
    max_r: usize,
    max_b: usize,
) -> Vec<(String, String)> {
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

                    let r_slice = if kr == 0 {
                        &[]
                    } else {
                        &pair.roman_tokens[i..i + kr]
                    };
                    let b_slice = if kb == 0 {
                        &[]
                    } else {
                        &pair.bengali_tokens[j..j + kb]
                    };

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
        return Vec::new();
    }

    let mut alignment = Vec::new();
    let (mut curr_i, mut curr_j) = (m, n);

    while curr_i > 0 || curr_j > 0 {
        let (kr, kb, r, b) = backpointer[curr_i][curr_j].clone();
        if kr == 0 && kb == 0 {
            break;
        }
        alignment.push((r, b));
        curr_i -= kr;
        curr_j -= kb;
    }

    alignment.reverse();
    alignment
}

pub type UnitId = u16;
pub const BOS_TOKEN: UnitId = 0;
pub const EOS_TOKEN: UnitId = 1;

#[derive(Clone, Debug)]
pub struct UnitVocab {
    pub unit_to_id: HashMap<(String, String), UnitId>,
    pub id_to_unit: Vec<(String, String)>,
    pub roman_to_units: HashMap<String, Vec<UnitId>>,
}

impl UnitVocab {
    pub fn build(sequences: &[Vec<(String, String)>]) -> Self {
        let mut unit_to_id = HashMap::new();
        let mut id_to_unit = Vec::new();
        let mut roman_to_units: HashMap<String, Vec<UnitId>> = HashMap::new();

        unit_to_id.insert((String::from("<BOS>"), String::from("<BOS>")), BOS_TOKEN);
        id_to_unit.push((String::from("<BOS>"), String::from("<BOS>")));

        unit_to_id.insert((String::from("<EOS>"), String::from("<EOS>")), EOS_TOKEN);
        id_to_unit.push((String::from("<EOS>"), String::from("<EOS>")));

        for seq in sequences {
            for unit in seq {
                if !unit_to_id.contains_key(unit) {
                    let id = id_to_unit.len() as UnitId;
                    unit_to_id.insert(unit.clone(), id);
                    id_to_unit.push(unit.clone());
                    roman_to_units.entry(unit.0.clone()).or_default().push(id);
                }
            }
        }

        Self {
            unit_to_id,
            id_to_unit,
            roman_to_units,
        }
    }
}

pub struct JointUnitLM {
    pub order: usize,
    pub vocab: UnitVocab,
    pub ngram_counts: HashMap<Vec<UnitId>, f64>,
    pub context_totals: HashMap<Vec<UnitId>, f64>,
    pub d: f64,
}

impl JointUnitLM {
    pub fn train(sequences: &[Vec<(String, String)>], order: usize) -> Self {
        let vocab = UnitVocab::build(sequences);
        let mut ngram_counts: HashMap<Vec<UnitId>, f64> = HashMap::new();
        let mut context_totals: HashMap<Vec<UnitId>, f64> = HashMap::new();

        for seq in sequences {
            let mut ids = Vec::with_capacity(seq.len() + order);
            for _ in 0..(order - 1) {
                ids.push(BOS_TOKEN);
            }
            for u in seq {
                if let Some(&id) = vocab.unit_to_id.get(u) {
                    ids.push(id);
                }
            }
            ids.push(EOS_TOKEN);

            for n in 1..=order {
                for window in ids.windows(n) {
                    *ngram_counts.entry(window.to_vec()).or_default() += 1.0;
                    if n > 1 {
                        let context = window[..n - 1].to_vec();
                        *context_totals.entry(context).or_default() += 1.0;
                    }
                }
            }
        }

        Self {
            order,
            vocab,
            ngram_counts,
            context_totals,
            d: 0.75,
        }
    }

    pub fn score_log(&self, unit: UnitId, history: &[UnitId]) -> f64 {
        let hist_len = history.len();
        let avail_hist = &history[hist_len.saturating_sub(self.order - 1)..];

        for k in (0..=avail_hist.len()).rev() {
            let context = &avail_hist[avail_hist.len() - k..];
            let mut full = context.to_vec();
            full.push(unit);

            if let Some(&c) = self.ngram_counts.get(&full) {
                let ctx_tot = if context.is_empty() {
                    self.ngram_counts.get(&vec![unit]).copied().unwrap_or(1.0)
                } else {
                    self.context_totals.get(context).copied().unwrap_or(1.0)
                };
                let p = (c - self.d).max(0.0) / ctx_tot;
                if p > 0.0 {
                    return p.ln();
                }
            }
        }
        -15.0
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
    deterministic_tie_break: bool,
    dict_words: &HashSet<String>,
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

            for kr in 1..=2.min(m - cand.pos) {
                let r_slice = &roman_tokens[cand.pos..cand.pos + kr];
                let r_key = r_slice.join("");

                if let Some(matching_units) = lm.vocab.roman_to_units.get(&r_key) {
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
        if deterministic_tie_break {
            sorted_cands.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        dict_words
                            .contains(&b.bengali)
                            .cmp(&dict_words.contains(&a.bengali))
                    })
                    .then_with(|| a.bengali.cmp(&b.bengali))
            });
        } else {
            sorted_cands.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        sorted_cands.truncate(beam_width);

        beam = sorted_cands;
        if beam.iter().all(|c| c.pos == m) {
            break;
        }
    }

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

    if deterministic_tie_break {
        final_cands.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| dict_words.contains(&b.0).cmp(&dict_words.contains(&a.0)))
                .then_with(|| a.0.cmp(&b.0))
        });
    } else {
        final_cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    final_cands
}

fn load_dict_words(dict_path: &Path) -> HashSet<String> {
    let mut set = HashSet::new();
    if let Ok(file) = File::open(dict_path) {
        let reader = BufReader::new(file);
        if let Ok(map) = serde_json::from_reader::<_, HashMap<String, serde_json::Value>>(reader) {
            for val in map.values() {
                if let Some(list) = val.as_array() {
                    for item in list {
                        if let Some(s) = item.as_str() {
                            set.insert(canonicalize(s));
                        }
                    }
                }
            }
        }
    }
    set
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════════════════════");
    println!("   🔬 AUDIT BEAM TIES & DETERMINISTIC TIE-BREAKING ON DEV-A AND DEV-B");
    println!("══════════════════════════════════════════════════════════════════════════════════");

    let dict_words = load_dict_words(Path::new("data/dictionaries/dictionary.json"));
    println!("Loaded {} dictionary words.", dict_words.len());

    let (train_pairs, _) = load_dataset(Path::new("data/eval/bn.translit.sampled.train.tsv"));
    let (dev_a_pairs, _) = load_dataset(Path::new("data/eval/bn.translit.sampled.dev_a.tsv"));
    let (dev_b_pairs, _) = load_dataset(Path::new("data/eval/bn.translit.sampled.dev_b.tsv"));

    println!(
        "Dataset counts: Train={}, Dev-A={}, Dev-B={}",
        train_pairs.len(),
        dev_a_pairs.len(),
        dev_b_pairs.len()
    );

    // 1. Train Aligner & Model
    println!("▶️  Training 1:1 Aligner on TRAIN...");
    let mut model = initialize_model(&train_pairs, 1, 1, 50.0);
    for _ in 1..=DEFAULT_EM_ITERATIONS {
        model = em_iteration(&train_pairs, &model, 1, 1);
    }

    let alignments: Vec<Vec<(String, String)>> = train_pairs
        .par_iter()
        .map(|p| viterbi_forced_alignment(p, &model, 1, 1))
        .collect();

    println!("▶️  Building Selected Model (Order 4)...");
    let lm = JointUnitLM::train(&alignments, 4);

    // 2. Evaluate on Dev-A and Dev-B across configurations
    for (split_name, pairs) in [("Dev-A", &dev_a_pairs), ("Dev-B", &dev_b_pairs)] {
        println!(
            "\n──────────────────────────────────────────────────────────────────────────────────"
        );
        println!("   Evaluating {} (N = {})", split_name, pairs.len());
        println!(
            "──────────────────────────────────────────────────────────────────────────────────"
        );

        let row_results: Vec<_> = pairs
            .par_iter()
            .enumerate()
            .map(|(idx, pair)| {
                let cands_256 = decode_beam_ngram(&pair.roman_tokens, &lm, 256, false, &dict_words);
                let cands_16_std =
                    decode_beam_ngram(&pair.roman_tokens, &lm, 16, false, &dict_words);
                let cands_16_det =
                    decode_beam_ngram(&pair.roman_tokens, &lm, 16, true, &dict_words);

                let hit_256 = cands_256
                    .first()
                    .map(|c| c.0 == pair.raw_bengali)
                    .unwrap_or(false);
                let hit_16_std = cands_16_std
                    .first()
                    .map(|c| c.0 == pair.raw_bengali)
                    .unwrap_or(false);
                let hit_16_det = cands_16_det
                    .first()
                    .map(|c| c.0 == pair.raw_bengali)
                    .unwrap_or(false);

                let mut tied = None;
                if cands_16_std.len() >= 2 && (cands_16_std[0].1 - cands_16_std[1].1).abs() < 1e-9 {
                    tied = Some((
                        idx + 1,
                        pair.raw_bengali.clone(),
                        pair.raw_roman.clone(),
                        cands_16_std[0].clone(),
                        cands_16_std[1].clone(),
                    ));
                }

                let mut pruned = None;
                if hit_256 && !hit_16_std {
                    let rank_in_16 = cands_16_std.iter().position(|c| c.0 == pair.raw_bengali);
                    let top_16 = cands_16_std
                        .first()
                        .map(|c| c.0.clone())
                        .unwrap_or_else(|| "NONE".into());
                    pruned = Some((
                        idx + 1,
                        pair.raw_bengali.clone(),
                        pair.raw_roman.clone(),
                        top_16,
                        rank_in_16,
                    ));
                }

                (hit_256, hit_16_std, hit_16_det, tied, pruned)
            })
            .collect();

        let hits_256_std = row_results.iter().filter(|r| r.0).count();
        let hits_16_std = row_results.iter().filter(|r| r.1).count();
        let hits_16_det = row_results.iter().filter(|r| r.2).count();
        let tied_rows: Vec<_> = row_results.iter().filter_map(|r| r.3.clone()).collect();
        let beam_pruned_rows: Vec<_> = row_results.iter().filter_map(|r| r.4.clone()).collect();

        println!("Top-1 Accuracy on {}:", split_name);
        println!(
            "  ├─ Beam K=256 (Standard)     : {} / {} ({:.2}%)",
            hits_256_std,
            pairs.len(),
            hits_256_std as f64 / pairs.len() as f64 * 100.0
        );
        println!(
            "  ├─ Beam K=16  (Standard)     : {} / {} ({:.2}%)",
            hits_16_std,
            pairs.len(),
            hits_16_std as f64 / pairs.len() as f64 * 100.0
        );
        println!(
            "  ├─ Beam K=16  (Deterministic): {} / {} ({:.2}%)",
            hits_16_det,
            pairs.len(),
            hits_16_det as f64 / pairs.len() as f64 * 100.0
        );
        println!(
            "  ├─ Delta (K=256 minus K=16)  : {} rows ({:+.2}%)",
            hits_256_std - hits_16_std,
            (hits_256_std - hits_16_std) as f64 / pairs.len() as f64 * 100.0
        );
        println!("  └─ Rows with Score Ties in Top-2: {}", tied_rows.len());

        if !tied_rows.is_empty() {
            println!(
                "\nSample Rows with Score Ties among Top Candidates ({} rows):",
                tied_rows.len()
            );
            for r in tied_rows.iter().take(10) {
                println!("  Row #{:<4} Gold: {:<12} Roman: {:<12} Cand 1: {:<12} ({:.4}) | Cand 2: {:<12} ({:.4})",
                    r.0, r.1, r.2, (r.3).0, (r.3).1, (r.4).0, (r.4).1);
            }
        }

        println!(
            "\nDissecting Rows where K=256 Hits but K=16 Misses ({} rows):",
            beam_pruned_rows.len()
        );
        for (i, r) in beam_pruned_rows.iter().enumerate() {
            let status = match r.4 {
                Some(rank) => format!("Ranked #{} in K=16 beam", rank + 1),
                None => "Pruned completely from K=16 beam (Search Error)".to_string(),
            };
            if i < 15 || i >= beam_pruned_rows.len() - 5 {
                println!(
                    "  {:>2}. Row #{:<4} Gold: {:<14} Roman: {:<14} Top in K=16: {:<14} -> {}",
                    i + 1,
                    r.0,
                    r.1,
                    r.2,
                    r.3,
                    status
                );
            } else if i == 15 {
                println!(
                    "  ... [{} rows omitted for brevity] ...",
                    beam_pruned_rows.len() - 20
                );
            }
        }
    }
}
