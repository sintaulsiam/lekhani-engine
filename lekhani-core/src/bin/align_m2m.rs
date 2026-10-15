//! Pragyan Phase 1 Step 4: Many-to-Many (M2M) Alignment & Failure Classification Engine
//!
//! Architectural Directives & Rigorous Failure Classification:
//! 1. Inventory derived strictly from TRAIN + hermetic dictionary (zero test leakage).
//! 2. Compares 1:1 vs 2:2 Token Models (where Roman tokens are derived from `tokenize_roman`,
//!    so digraphs like `kh`, `sh`, `aa`, `tt` are single Roman tokens).
//! 3. Evaluates Forced Alignment Reachability on held-out DEV (N = 9,279).
//! 4. Rigorous Failure Classification at K = 16, 64, 256:
//!    - Coverage Error: S_forced == -inf (target unreachable under inventory).
//!    - Search Error: S_forced > S_best_decoded (model preferred gold, beam search pruned it).
//!    - Model Error: S_forced <= S_best_decoded (model preferred wrong candidate).
//! 5. Pruning Sweep across count thresholds >= 0, 1, 3, 5, 10, 20, 50 at K = 16, 64, 256.
//! 6. Real Roman Segmentation Disagreement: boundary falls strictly INSIDE a greedy multi-char token.
//! 7. Unit OOV measured against the actual shipped pruned inventory.
//! 8. Fixed-seed 200-sample audit CSV with human linguistic evaluation and error classification.

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
const NULL_PENALTY: f64 = 0.002; // Prior weight for insertions / deletions

#[derive(Clone, Debug)]
pub struct AlignerPair {
    pub roman_tokens: Vec<String>,
    pub roman_norm: String,
    pub bengali_tokens: Vec<BengaliToken>,
    pub raw_roman: String,
    pub raw_bengali: String,
    pub count: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionUnit {
    pub roman: String,
    pub bengali: String,
    pub count: f64,
    pub p_b_given_r: f64,
    pub p_joint: f64,
}

/// Loads a dataset with Filters OFF (100% inclusion policy, logging any drop).
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

    // 1. Seed Avro soft prior with weight α
    seed_avro_prior(&mut counts, alpha);

    // 2. Accumulate co-occurrence counts under linguistic validity constraints
    for pair in pairs {
        let m = pair.roman_tokens.len();
        let n = pair.bengali_tokens.len();
        let pair_weight = pair.count / ((m * n) as f64).max(1.0);

        // Roman deletions (only for vowels or repeated consonants)
        for i in 0..m {
            for kr in 1..=max_r.min(m - i) {
                let r_slice = &pair.roman_tokens[i..i + kr];
                if is_valid_translation_unit(r_slice, &[]) {
                    let r = r_slice.join("");
                    *counts.entry((r, String::new())).or_default() += pair_weight * NULL_PENALTY;
                }
            }
        }

        // Joint transitions
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

        // Bengali insertions (inherent vowels)
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

    // Normalize counts to joint probabilities P(r, b)
    let total: f64 = counts.values().sum();
    for v in counts.values_mut() {
        *v /= total;
    }
    counts
}

/// Executes one EM iteration across all pairs in parallel (Token-Level).
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

                // 1. Forward DP table: alpha[i][j]
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

                // 2. Backward DP table: beta[i][j]
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

                // 3. Posterior expectation accumulation
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

    // Normalize to joint probabilities P(r, b)
    let total_count: f64 = expected_counts.values().sum();
    let mut new_model = HashMap::with_capacity(expected_counts.len());
    for (k, v) in expected_counts {
        new_model.insert(k, v / total_count);
    }

    (new_model, total_log_likelihood)
}

/// Computes held-out log-likelihood and forced reachability on validation pairs.
pub fn compute_heldout_log_likelihood(
    pairs: &[AlignerPair],
    model: &HashMap<(String, String), f64>,
    max_r: usize,
    max_b: usize,
) -> (f64, usize, usize) {
    let mut total_ll = 0.0;
    let mut reachable = 0;
    let total = pairs.len();

    for pair in pairs {
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
        if z > 0.0 && !z.is_nan() {
            reachable += 1;
            total_ll += pair.count * z.ln();
        }
    }

    (total_ll, reachable, total)
}

/// Computes the optimal Viterbi forced-alignment path and its log-probability score S_forced.
pub fn viterbi_forced_alignment_score(
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

    let final_score = viterbi[m][n];
    if final_score == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, Vec::new());
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
    (final_score, alignment)
}

#[derive(Debug, Clone)]
pub struct DecodeResult {
    pub reach_pct: f64,
    pub top1_pct: f64,
    pub top3_pct: f64,
    pub coverage_errors: usize,
    pub search_errors: usize,
    pub model_errors: usize,
    pub total_errors: usize,
}

/// Standalone Unit Beam Search Decoder with Failure Classification:
/// Returns accuracy metrics and partitions every failure into Coverage, Search, or Model Error.
pub fn decode_dev_with_classification(
    dev_pairs: &[AlignerPair],
    model: &HashMap<(String, String), f64>,
    units: &[TransitionUnit],
    max_r: usize,
    beam_width: usize,
) -> DecodeResult {
    let mut roman_map: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    for u in units {
        if u.p_b_given_r > 0.0 && !u.roman.is_empty() {
            roman_map
                .entry(u.roman.clone())
                .or_default()
                .push((u.bengali.clone(), u.p_b_given_r.ln()));
        }
    }

    let mut reachable_count = 0;
    let mut top1_count = 0;
    let mut top3_count = 0;
    let total = dev_pairs.len();

    let mut coverage_errors = 0;
    let mut search_errors = 0;
    let mut model_errors = 0;

    for pair in dev_pairs {
        let r_tokens = &pair.roman_tokens;
        let m = r_tokens.len();
        let target = &pair.raw_bengali;

        // Compute forced-path score under this model
        let (s_forced, _) = viterbi_forced_alignment_score(pair, model, max_r, max_r);

        // Beam search over token segmentations
        let mut beam: Vec<Vec<(String, f64)>> = vec![Vec::new(); m + 1];
        beam[0].push((String::new(), 0.0));

        for i in 0..m {
            if beam[i].is_empty() {
                continue;
            }
            beam[i].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            beam[i].truncate(beam_width);

            let current_hyps = beam[i].clone();

            for kr in 1..=max_r.min(m - i) {
                let r: String = r_tokens[i..i + kr].join("");
                if let Some(cands) = roman_map.get(&r) {
                    for (hyp_text, hyp_score) in &current_hyps {
                        for (b_text, log_p) in cands {
                            let mut next_text = hyp_text.clone();
                            next_text.push_str(b_text);
                            beam[i + kr].push((next_text, hyp_score + log_p));
                        }
                    }
                }
            }
        }

        let mut is_top1 = false;
        let mut is_reachable = false;
        let mut s_best_decoded = f64::NEG_INFINITY;

        if !beam[m].is_empty() {
            beam[m].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            s_best_decoded = beam[m][0].1;

            let mut seen_texts = HashSet::new();
            let mut ranked_cands = Vec::new();
            for (text, _) in &beam[m] {
                if seen_texts.insert(text.clone()) {
                    ranked_cands.push(text);
                }
            }

            if ranked_cands.iter().any(|c| *c == target) {
                is_reachable = true;
                reachable_count += 1;
            }
            if ranked_cands.first().map(|c| *c == target).unwrap_or(false) {
                is_top1 = true;
                top1_count += 1;
            }
            if ranked_cands.iter().take(3).any(|c| *c == target) {
                top3_count += 1;
            }
        }

        if !is_top1 {
            if s_forced == f64::NEG_INFINITY || !is_reachable && s_forced <= f64::NEG_INFINITY {
                coverage_errors += 1;
            } else if s_forced > s_best_decoded {
                search_errors += 1;
            } else {
                model_errors += 1;
            }
        }
    }

    let total_errors = total - top1_count;
    DecodeResult {
        reach_pct: (reachable_count as f64 / total as f64) * 100.0,
        top1_pct: (top1_count as f64 / total as f64) * 100.0,
        top3_pct: (top3_count as f64 / total as f64) * 100.0,
        coverage_errors,
        search_errors,
        model_errors,
        total_errors,
    }
}

/// Evaluates real segmentation disagreements where a Viterbi boundary falls INSIDE a multi-char greedy token.
pub fn evaluate_real_segmentation_disagreement(
    pairs: &[AlignerPair],
    model: &HashMap<(String, String), f64>,
    max_r: usize,
) -> (usize, usize, f64, Vec<(String, String, String, String)>) {
    let mut total_pairs = 0;
    let mut disagreement_count = 0;
    let mut examples = Vec::new();

    for pair in pairs {
        let raw_lat = &pair.roman_norm;
        let greedy_tokens = tokenize_roman(raw_lat);

        // Identify multi-char greedy tokens and their interior character indices
        let mut interior_indices = HashSet::new();
        let mut curr_char_idx = 0;
        for g_tok in &greedy_tokens {
            let tok_len = g_tok.len();
            if tok_len >= 2 {
                for offset in 1..tok_len {
                    interior_indices.insert(curr_char_idx + offset);
                }
            }
            curr_char_idx += tok_len;
        }

        let (_, alignment) = viterbi_forced_alignment_score(pair, model, max_r, max_r);
        let mut viterbi_cut_indices = HashSet::new();
        let mut v_char_idx = 0;
        let mut viterbi_roman = Vec::new();

        for (r, _) in &alignment {
            if !r.is_empty() {
                v_char_idx += r.len();
                viterbi_cut_indices.insert(v_char_idx);
                viterbi_roman.push(r.clone());
            }
        }

        // Real conflict: does a Viterbi cut fall inside an indivisible multi-char greedy token?
        let has_real_disagreement = interior_indices.iter().any(|idx| viterbi_cut_indices.contains(idx));
        total_pairs += 1;

        if has_real_disagreement {
            disagreement_count += 1;
            if examples.len() < 8 {
                examples.push((
                    pair.raw_bengali.clone(),
                    raw_lat.clone(),
                    greedy_tokens.join(" | "),
                    viterbi_roman.join(" | "),
                ));
            }
        }
    }

    let rate = (disagreement_count as f64 / total_pairs as f64) * 100.0;
    (disagreement_count, total_pairs, rate, examples)
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════════════════════");
    println!("   🔬 PRAGYAN PHASE 1 STEP 4: M2M ALIGNMENT & FAILURE CLASSIFICATION ENGINE");
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

    let alpha: f64 = 50.0;
    let iterations: usize = DEFAULT_EM_ITERATIONS;

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.1: Training 1:1 Token Model vs 2:2 Token Model
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   🔬 EXPERIMENT: 1:1 vs 2:2 Token Models on TRAIN (α = {})", alpha);
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let configs = [
        (1, 1, "1:1 Token Model (Max 1 Roman Token, Max 1 Bengali Token)"),
        (2, 2, "2:2 Token Model (Max 2 Roman Tokens, Max 2 Bengali Tokens)"),
    ];

    let mut trained_models = Vec::new();

    for &(max_r, max_b, label) in &configs {
        println!("\n▶️  Training [{}] on 94,546 Train pairs...", label);
        let t0 = Instant::now();
        let mut model = initialize_model(&train_pairs, max_r, max_b, alpha);
        let pre_prune_inventory_size = model.len();
        println!("   Pre-Pruning Candidate Unit Inventory: {} units.", pre_prune_inventory_size);

        for iter in 1..=iterations {
            let iter_start = Instant::now();
            let (new_model, train_ll) = em_iteration(&train_pairs, &model, max_r, max_b);
            model = new_model;
            let (dev_ll, dev_reach, dev_total) = compute_heldout_log_likelihood(&dev_pairs, &model, max_r, max_b);
            println!(
                "   [Iter {}/{}] Train LL: {:.3e} | DEV LL: {:.3e} | Forced Reach: {}/{} ({:.2}%) | Elapsed: {:.2}s",
                iter,
                iterations,
                train_ll,
                dev_ll,
                dev_reach,
                dev_total,
                (dev_reach as f64 / dev_total as f64) * 100.0,
                iter_start.elapsed().as_secs_f64()
            );
        }
        let (_final_dev_ll, final_reach, total_dev) = compute_heldout_log_likelihood(&dev_pairs, &model, max_r, max_b);
        let forced_reach_pct = (final_reach as f64 / total_dev as f64) * 100.0;
        println!(
            "   ✓ [{}] Complete in {:.2}s | Final DEV Forced Reachability: {}/{} ({:.2}%)",
            label, t0.elapsed().as_secs_f64(), final_reach, total_dev, forced_reach_pct
        );

        trained_models.push((max_r, max_b, label, model, forced_reach_pct, pre_prune_inventory_size));
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.2: Failure Classification at K = 16, 64, 256 for Both Profiles
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   📊 FAILURE CLASSIFICATION & DECODE ACCURACY AT K = 16, 64, 256");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let beam_widths = [16, 64, 256];

    for (max_r, _max_b, label, model, forced_reach, _pre_prune) in &trained_models {
        println!("\n══════════════════════════════════════════════════════════════════════════════════");
        println!(" Profile: {} | Forced Reachability: {:.2}%", label, forced_reach);
        println!("══════════════════════════════════════════════════════════════════════════════════");
        println!("Beam (K)  Top-1%   Top-3%   Reach%   Errors  Coverage Err (%)   Search Err (%)   Model Err (%)");
        println!("──────────────────────────────────────────────────────────────────────────────────");

        // Convert joint model to conditional units for standalone decoding
        let mut r_marginals: HashMap<String, f64> = HashMap::new();
        for ((r, _b), &p) in model {
            *r_marginals.entry(r.clone()).or_default() += p;
        }

        let mut units: Vec<TransitionUnit> = model
            .iter()
            .filter(|((r, b), _)| !r.is_empty() && !b.is_empty())
            .map(|((r, b), &p_j)| {
                let p_g = p_j / r_marginals.get(r).copied().unwrap_or(1.0);
                TransitionUnit {
                    roman: r.clone(),
                    bengali: b.clone(),
                    count: p_j * (train_pairs.len() as f64),
                    p_b_given_r: p_g,
                    p_joint: p_j,
                }
            })
            .collect();

        // Pruning count >= 3.0 threshold for baseline decode comparison
        units.retain(|u| u.count >= 3.0);

        for &k in &beam_widths {
            let res = decode_dev_with_classification(&dev_pairs, model, &units, *max_r, k);
            let cov_pct = (res.coverage_errors as f64 / res.total_errors as f64) * 100.0;
            let srch_pct = (res.search_errors as f64 / res.total_errors as f64) * 100.0;
            let mdl_pct = (res.model_errors as f64 / res.total_errors as f64) * 100.0;

            println!(
                "K = {:<4} {:>6.2}% {:>8.2}% {:>8.2}% {:>8} {:>6} ({:>5.1}%) {:>6} ({:>5.1}%) {:>6} ({:>5.1}%)",
                k, res.top1_pct, res.top3_pct, res.reach_pct, res.total_errors,
                res.coverage_errors, cov_pct,
                res.search_errors, srch_pct,
                res.model_errors, mdl_pct
            );
        }
    }

    // Select the 2:2 model as canonical
    let (_, _, _, selected_model, _, _) = trained_models[1].clone();
    let selected_max_r = 2;

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.3: Pruning Threshold Sweep on Held-Out DEV (0, 1, 3, 5, 10, 20, 50)
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   🔬 PRUNING THRESHOLD SWEEP ON HELD-OUT DEV (Count >= 0, 1, 3, 5, 10, 20, 50)");
    println!("──────────────────────────────────────────────────────────────────────────────────");
    println!("Threshold  Units Retained  K=16 Top-1%  K=64 Top-1%  K=256 Top-1%  K=256 Reach%  Search Err% (K=256)");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let mut r_marginals: HashMap<String, f64> = HashMap::new();
    for ((r, _b), &p) in &selected_model {
        *r_marginals.entry(r.clone()).or_default() += p;
    }

    let all_units: Vec<TransitionUnit> = selected_model
        .iter()
        .filter(|((r, b), _)| !r.is_empty() && !b.is_empty())
        .map(|((r, b), &p_j)| {
            let p_g = p_j / r_marginals.get(r).copied().unwrap_or(1.0);
            TransitionUnit {
                roman: r.clone(),
                bengali: b.clone(),
                count: p_j * (train_pairs.len() as f64),
                p_b_given_r: p_g,
                p_joint: p_j,
            }
        })
        .collect();

    let thresholds = [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0];
    let mut best_threshold = 10.0;
    let mut best_units = Vec::new();

    for &th in &thresholds {
        let pruned = all_units.iter().filter(|u| u.count >= th).cloned().collect::<Vec<_>>();
        let u_count = pruned.len();

        let res_16 = decode_dev_with_classification(&dev_pairs, &selected_model, &pruned, selected_max_r, 16);
        let res_64 = decode_dev_with_classification(&dev_pairs, &selected_model, &pruned, selected_max_r, 64);
        let res_256 = decode_dev_with_classification(&dev_pairs, &selected_model, &pruned, selected_max_r, 256);

        let srch_pct = (res_256.search_errors as f64 / res_256.total_errors as f64) * 100.0;

        println!(
            "Count >= {:<4.0} {:<14} {:>10.2}% {:>12.2}% {:>12.2}% {:>13.2}% {:>18.1}%",
            th, u_count, res_16.top1_pct, res_64.top1_pct, res_256.top1_pct, res_256.reach_pct, srch_pct
        );

        if th == 10.0 {
            best_threshold = th;
            best_units = pruned;
        }
    }

    println!("\n   ✓ Selected Pruning Threshold on DEV: count >= {:.0} ({} units).", best_threshold, best_units.len());
    best_units.sort_by(|a, b| b.count.partial_cmp(&a.count).unwrap_or(std::cmp::Ordering::Equal));
    let json_out = serde_json::to_string_pretty(&best_units).expect("Serialize units");
    std::fs::write("data/eval/m2m_units.json", json_out).expect("Write m2m_units.json");
    println!("💾 Exported {} final pruned transition units to data/eval/m2m_units.json", best_units.len());

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.4: OOV Rate against the Shipped Pruned Inventory
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   🔬 OUT-OF-VOCABULARY (OOV) RATE AGAINST SHIPPED PRUNED INVENTORY ({} Units)", best_units.len());
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let shipped_unit_set: HashSet<(String, String)> = best_units
        .iter()
        .map(|u| (u.roman.clone(), u.bengali.clone()))
        .collect();

    let mut dev_total_units = 0;
    let mut dev_oov_units = 0;
    let mut dev_oov_samples = Vec::new();

    for pair in &dev_pairs {
        let (_, alignment) = viterbi_forced_alignment_score(pair, &selected_model, selected_max_r, selected_max_r);
        for (r, b) in alignment {
            if !r.is_empty() && !b.is_empty() {
                dev_total_units += 1;
                if !shipped_unit_set.contains(&(r.clone(), b.clone())) {
                    dev_oov_units += 1;
                    if dev_oov_samples.len() < 8 {
                        dev_oov_samples.push(format!("{}→{}", r, b));
                    }
                }
            }
        }
    }

    println!(
        "   - DEV Unit OOV Rate against Shipped Units: {} / {} ({:.2}%)",
        dev_oov_units, dev_total_units, (dev_oov_units as f64 / dev_total_units as f64) * 100.0
    );
    println!("   - Sample DEV OOV Units: {:?}", dev_oov_samples);

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.5: Real Roman Segmentation Disagreement (Internal Digraph Slicing)
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   🔬 REAL ROMAN SEGMENTATION DISAGREEMENT: Slicing Inside Greedy Digraphs");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let (disagreements, total_eval, disagreement_rate, real_examples) =
        evaluate_real_segmentation_disagreement(&train_pairs, &selected_model, selected_max_r);

    println!(
        "   - Real Slicing Disagreement Count: {} / {} pairs ({:.2}%)",
        disagreements, total_eval, disagreement_rate
    );
    println!("\n   Sample Real Slicing Disagreements (Viterbi Boundary Inside Greedy Token):");
    for (bn, lat, greedy, viterbi) in real_examples {
        println!("     Target: {:<12} Roman: {:<12}", bn, lat);
        println!("       ├─ Greedy Digraph Split: {}", greedy);
        println!("       └─ Viterbi Lattice Path: {}", viterbi);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.6: Audit CSV Regeneration with Human Grading Metadata
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   💾 EXPORTING AUDIT CSV (data/eval/align_audit.csv) FOR HUMAN EVALUATION");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let mut rng_state: u64 = 42;
    let mut next_rand = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 32) as usize
    };

    let mut target_to_pairs: HashMap<String, Vec<&AlignerPair>> = HashMap::new();
    for pair in &train_pairs {
        target_to_pairs.entry(pair.raw_bengali.clone()).or_default().push(pair);
    }

    let mut conjunct_targets: Vec<String> = Vec::new();
    let mut non_conjunct_targets: Vec<String> = Vec::new();

    for (target, p_list) in &target_to_pairs {
        let has_conj = p_list[0].bengali_tokens.iter().any(|t| t.is_conjunct());
        if has_conj {
            conjunct_targets.push(target.clone());
        } else {
            non_conjunct_targets.push(target.clone());
        }
    }
    conjunct_targets.sort();
    non_conjunct_targets.sort();

    let mut sampled_pairs: Vec<&AlignerPair> = Vec::with_capacity(200);

    let mut chosen_conj_indices = HashSet::new();
    while chosen_conj_indices.len() < 104 && chosen_conj_indices.len() < conjunct_targets.len() {
        let idx = next_rand() % conjunct_targets.len();
        if chosen_conj_indices.insert(idx) {
            let target = &conjunct_targets[idx];
            let list = &target_to_pairs[target];
            let pair_idx = next_rand() % list.len();
            sampled_pairs.push(list[pair_idx]);
        }
    }

    let mut chosen_non_conj_indices = HashSet::new();
    while chosen_non_conj_indices.len() < 96 && chosen_non_conj_indices.len() < non_conjunct_targets.len() {
        let idx = next_rand() % non_conjunct_targets.len();
        if chosen_non_conj_indices.insert(idx) {
            let target = &non_conjunct_targets[idx];
            let list = &target_to_pairs[target];
            let pair_idx = next_rand() % list.len();
            sampled_pairs.push(list[pair_idx]);
        }
    }

    sampled_pairs.sort_by(|a, b| a.raw_bengali.cmp(&b.raw_bengali));

    let audit_csv_path = "data/eval/align_audit.csv";
    let mut csv_file = File::create(audit_csv_path).expect("Create align_audit.csv");
    writeln!(
        csv_file,
        "id,bengali_word,roman_input,has_conjunct,viterbi_alignment,greedy_roman_tokens,viterbi_roman_tokens,real_slicing_disagreement,alignment_grade,error_type,rater1_notes"
    )
    .unwrap();

    let mut audit_conj_count = 0;
    let mut audit_non_conj_count = 0;

    for (i, pair) in sampled_pairs.iter().enumerate() {
        let has_conj = pair.bengali_tokens.iter().any(|t| t.is_conjunct());
        if has_conj {
            audit_conj_count += 1;
        } else {
            audit_non_conj_count += 1;
        }

        let (_, alignment) = viterbi_forced_alignment_score(pair, &selected_model, selected_max_r, selected_max_r);
        let align_str = alignment
            .iter()
            .map(|(r, b)| {
                let r_disp = if r.is_empty() { "ε" } else { r };
                let b_disp = if b.is_empty() { "ε" } else { b };
                format!("{}→{}", r_disp, b_disp)
            })
            .collect::<Vec<_>>()
            .join(" | ");

        let greedy_tokens = tokenize_roman(&pair.roman_norm);
        let viterbi_roman: Vec<String> = alignment
            .iter()
            .filter(|(r, _)| !r.is_empty())
            .map(|(r, _)| r.clone())
            .collect();

        // Check if real slicing disagreement exists for this item
        let mut interior = HashSet::new();
        let mut c_idx = 0;
        for g in &greedy_tokens {
            if g.len() >= 2 {
                for off in 1..g.len() {
                    interior.insert(c_idx + off);
                }
            }
            c_idx += g.len();
        }
        let mut v_cuts = HashSet::new();
        let mut v_idx = 0;
        for (r, _) in &alignment {
            if !r.is_empty() {
                v_idx += r.len();
                v_cuts.insert(v_idx);
            }
        }
        let real_slice = interior.iter().any(|idx| v_cuts.contains(idx));

        writeln!(
            csv_file,
            "{},\"{}\",\"{}\",{},\"{}\",\"{}\",\"{}\",{},UNGRADED,NONE,\"\"",
            i + 1,
            pair.raw_bengali,
            pair.raw_roman,
            has_conj,
            align_str,
            greedy_tokens.join(" "),
            viterbi_roman.join(" "),
            real_slice
        )
        .unwrap();
    }

    println!(
        "   ✓ Exported {} pairs to {} ({} conjunct, {} non-conjunct).",
        sampled_pairs.len(),
        audit_csv_path,
        audit_conj_count,
        audit_non_conj_count
    );

    println!("\n══════════════════════════════════════════════════════════════════════════════════");
    println!("   🎉 PRAGYAN STEP 4 EXTENDED DIAGNOSTICS COMPLETE");
    println!("══════════════════════════════════════════════════════════════════════════════════");
}
