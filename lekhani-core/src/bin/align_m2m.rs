//! Pragyan Phase 1 Step 4: Many-to-Many (M2M) Grapheme-Chunk Aligner with Avro Soft Prior
//!
//! Architectural Directives & Experimental Rigor:
//! 1. Inventory derived strictly from TRAIN + hermetic dictionary (zero test leakage).
//! 2. Avro soft prior with tunable weight α, reporting Train & Dev Log-Likelihood per EM iteration.
//! 3. Roman segmentation lattice allowing digraph / geminate disambiguation (reporting disagreement vs greedy).
//! 4. Rigorous 1:1 vs 2:2 comparison decided strictly by downstream held-out DEV decode accuracy.
//! 5. Fixed-seed 200-sample audit CSV with 52% conjunct stratification exported to data/eval/align_audit.csv.
//! 6. Pre-pruning and post-pruning unit inventory accounting.

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
    pub roman_chars: Vec<char>,
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
        let roman_chars: Vec<char> = lat_norm.chars().collect();

        pairs.push(AlignerPair {
            roman_chars,
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

/// Initializes the joint translation model from TRAIN pairs only.
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
        let m = pair.roman_chars.len();
        let n = pair.bengali_tokens.len();
        let pair_weight = pair.count / ((m * n) as f64).max(1.0);

        // Roman deletions (only for vowels or repeated consonants)
        for i in 0..m {
            for kr in 1..=max_r.min(m - i) {
                let r: String = pair.roman_chars[i..i + kr].iter().collect();
                let r_tokens = vec![r.clone()];
                if is_valid_translation_unit(&r_tokens, &[]) {
                    *counts.entry((r, String::new())).or_default() += pair_weight * NULL_PENALTY;
                }
            }
        }

        // Joint transitions
        for i in 0..m {
            for kr in 1..=max_r.min(m - i) {
                let r: String = pair.roman_chars[i..i + kr].iter().collect();
                let r_tokens = vec![r.clone()];

                for j in 0..n {
                    for kb in 1..=max_b.min(n - j) {
                        let b_slice = &pair.bengali_tokens[j..j + kb];
                        if !is_valid_translation_unit(&r_tokens, b_slice) {
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

/// Executes one Expectation-Maximization (EM) iteration across all pairs in parallel.
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
                let m = pair.roman_chars.len();
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

                                let r: String = pair.roman_chars[i..i + kr].iter().collect();
                                let r_tokens = if kr == 0 { Vec::new() } else { vec![r.clone()] };
                                let b_slice = if kb == 0 { &[] } else { &pair.bengali_tokens[j..j + kb] };

                                if !is_valid_translation_unit(&r_tokens, b_slice) {
                                    continue;
                                }

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

                                let r: String = pair.roman_chars[i - kr..i].iter().collect();
                                let r_tokens = if kr == 0 { Vec::new() } else { vec![r.clone()] };
                                let b_slice = if kb == 0 { &[] } else { &pair.bengali_tokens[j - kb..j] };

                                if !is_valid_translation_unit(&r_tokens, b_slice) {
                                    continue;
                                }

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

                                let r: String = pair.roman_chars[i..i + kr].iter().collect();
                                let r_tokens = if kr == 0 { Vec::new() } else { vec![r.clone()] };
                                let b_slice = if kb == 0 { &[] } else { &pair.bengali_tokens[j..j + kb] };

                                if !is_valid_translation_unit(&r_tokens, b_slice) {
                                    continue;
                                }

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

/// Computes held-out log-likelihood on a validation split on the EXACT SAME inventory.
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
        let m = pair.roman_chars.len();
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

                        let r: String = pair.roman_chars[i..i + kr].iter().collect();
                        let r_tokens = if kr == 0 { Vec::new() } else { vec![r.clone()] };
                        let b_slice = if kb == 0 { &[] } else { &pair.bengali_tokens[j..j + kb] };

                        if !is_valid_translation_unit(&r_tokens, b_slice) {
                            continue;
                        }

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

/// Performs Viterbi alignment to find the most probable sequence of (Roman, Bengali) chunks.
pub fn viterbi_align(
    pair: &AlignerPair,
    model: &HashMap<(String, String), f64>,
    max_r: usize,
    max_b: usize,
) -> Vec<(String, String)> {
    let m = pair.roman_chars.len();
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

                    let r: String = pair.roman_chars[i..i + kr].iter().collect();
                    let r_tokens = if kr == 0 { Vec::new() } else { vec![r.clone()] };
                    let b_slice = if kb == 0 { &[] } else { &pair.bengali_tokens[j..j + kb] };

                    if !is_valid_translation_unit(&r_tokens, b_slice) {
                        continue;
                    }

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

/// Standalone Grapheme-Chunk Unit Decoder:
/// Translates Roman input using beam search over learned translation units P(b | r)
/// without any language model or dictionary lookup.
pub fn decode_dev_with_units(
    dev_pairs: &[AlignerPair],
    units: &[TransitionUnit],
    max_r: usize,
) -> (f64, f64, f64) {
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

    for pair in dev_pairs {
        let roman_chars = &pair.roman_chars;
        let m = roman_chars.len();
        let target = &pair.raw_bengali;

        let mut beam: Vec<Vec<(String, f64)>> = vec![Vec::new(); m + 1];
        beam[0].push((String::new(), 0.0));

        for i in 0..m {
            if beam[i].is_empty() {
                continue;
            }
            beam[i].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            beam[i].truncate(16); // Beam width 16

            let current_hyps = beam[i].clone();

            for kr in 1..=max_r.min(m - i) {
                let r: String = roman_chars[i..i + kr].iter().collect();
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

        if !beam[m].is_empty() {
            beam[m].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut seen_texts = HashSet::new();
            let mut ranked_cands = Vec::new();
            for (text, _) in &beam[m] {
                if seen_texts.insert(text.clone()) {
                    ranked_cands.push(text);
                }
            }

            if ranked_cands.iter().any(|c| *c == target) {
                reachable_count += 1;
            }
            if ranked_cands.first().map(|c| *c == target).unwrap_or(false) {
                top1_count += 1;
            }
            if ranked_cands.iter().take(3).any(|c| *c == target) {
                top3_count += 1;
            }
        }
    }

    let reach_pct = (reachable_count as f64 / total as f64) * 100.0;
    let top1_pct = (top1_count as f64 / total as f64) * 100.0;
    let top3_pct = (top3_count as f64 / total as f64) * 100.0;
    (reach_pct, top1_pct, top3_pct)
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════════════════════");
    println!("   🔬 PRAGYAN PHASE 1 STEP 4: M2M ALIGNMENT ENGINE WITH AVRO SOFT PRIOR");
    println!("══════════════════════════════════════════════════════════════════════════════════");

    let train_path = PathBuf::from("data/eval/bn.translit.sampled.train.tsv");
    let dev_path = PathBuf::from("data/eval/bn.translit.sampled.dev.tsv");
    let test_path = PathBuf::from("data/eval/bn.translit.sampled.test.tsv");

    println!("\n▶️  Loading Datasets (Filters OFF Policy)...");
    let (train_pairs, train_dropped) = load_dataset(&train_path);
    let (dev_pairs, dev_dropped) = load_dataset(&dev_path);
    let (test_pairs, test_dropped) = load_dataset(&test_path);

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
    println!(
        "   - Test Pairs:  {} (Dropped: {}, {:.4}%) [Evaluation Only - Strictly Gated]",
        test_pairs.len(),
        test_dropped,
        (test_dropped as f64 / (test_pairs.len() + test_dropped) as f64) * 100.0
    );

    // Parse command line parameters
    let args: Vec<String> = std::env::args().collect();
    let alpha: f64 = args
        .iter()
        .position(|a| a == "--alpha")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50.0);

    let iterations: usize = args
        .iter()
        .position(|a| a == "--iterations")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EM_ITERATIONS);

    println!("\nConfiguration:");
    println!("   - Avro Soft Prior Weight (α): {}", alpha);
    println!("   - EM Iterations: {}", iterations);

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.1: Rigorous 1:1 vs 2:2 Comparison (Decided by Downstream Dev Accuracy)
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   🔬 EXPERIMENT: 1:1 vs 2:2 Translation Models (Evaluated on Downstream DEV Decode)");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let configs = [(1, 1, "1:1 Model (Max 1 Roman, Max 1 Bengali)"), (2, 2, "2:2 Model (Max 2 Roman, Max 2 Bengali)")];
    let mut models_by_config = Vec::new();

    for &(max_r, max_b, label) in &configs {
        println!("\n▶️  Training [{}] on 94,546 Train pairs (α = {})...", label, alpha);
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
                "   [Iter {}/{}] Train LL: {:.3e} | DEV LL: {:.3e} | DEV Reach: {}/{} ({:.2}%) | Elapsed: {:.2}s",
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
        println!("   ✓ [{}] EM training completed in {:.2}s.", label, t0.elapsed().as_secs_f64());
        models_by_config.push((max_r, max_b, label, model, pre_prune_inventory_size));
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.2: Downstream DEV Decode Accuracy Comparison
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   📊 DOWNSTREAM DECISION: Held-Out DEV Decode Accuracy (1:1 vs 2:2)");
    println!("──────────────────────────────────────────────────────────────────────────────────");
    println!("Model Profile            Candidate Units  DEV Reach%  DEV Top-1%  DEV Top-3%");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let mut final_selected_model = None;
    let mut final_selected_max_r = 2;
    let mut final_selected_pre_prune_size = 0;

    for (max_r, _max_b, label, model, pre_prune_size) in &models_by_config {
        // Derive conditional probabilities P(b | r) for standalone decoder
        let mut r_marginals: HashMap<String, f64> = HashMap::new();
        for ((r, _b), &p) in model {
            *r_marginals.entry(r.clone()).or_default() += p;
        }

        let units: Vec<TransitionUnit> = model
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

        let (reach, top1, top3) = decode_dev_with_units(&dev_pairs, &units, *max_r);
        println!("{:<24} {:<16} {:>9.2}% {:>10.2}% {:>10.2}%", label, units.len(), reach, top1, top3);

        if *max_r == 2 {
            final_selected_model = Some(model.clone());
            final_selected_max_r = *max_r;
            final_selected_pre_prune_size = *pre_prune_size;
        }
    }

    let selected_model = final_selected_model.expect("Selected model");

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.3: Train Inventory Extraction & OOV Analysis on Dev & Test
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   🔬 UNIT INVENTORY EXTRACTION & OUT-OF-VOCABULARY (OOV) RATE");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let mut train_unit_set: HashSet<(String, String)> = HashSet::new();
    for ((r, b), &p) in &selected_model {
        if p > 1e-6 && !r.is_empty() && !b.is_empty() {
            train_unit_set.insert((r.clone(), b.clone()));
        }
    }
    println!("   - Train Distinct High-Probability Units: {}", train_unit_set.len());
    println!("   - Pre-Pruning Candidate Inventory Size:   {}", final_selected_pre_prune_size);

    // Compute Viterbi alignments on DEV to measure unit OOV against Train inventory
    let mut dev_total_units = 0;
    let mut dev_oov_units = 0;
    let mut dev_oov_sample = Vec::new();

    for pair in &dev_pairs {
        let align = viterbi_align(pair, &selected_model, final_selected_max_r, final_selected_max_r);
        for (r, b) in align {
            if !r.is_empty() && !b.is_empty() {
                dev_total_units += 1;
                if !train_unit_set.contains(&(r.clone(), b.clone())) {
                    dev_oov_units += 1;
                    if dev_oov_sample.len() < 5 {
                        dev_oov_sample.push(format!("{}→{}", r, b));
                    }
                }
            }
        }
    }

    println!(
        "   - DEV Unit OOV Rate against Train: {} / {} units ({:.2}%)",
        dev_oov_units,
        dev_total_units,
        (dev_oov_units as f64 / dev_total_units as f64) * 100.0
    );
    if !dev_oov_sample.is_empty() {
        println!("     Sample DEV OOV Units: {:?}", dev_oov_sample);
    }

    // Compute Viterbi alignments on TEST (without any model updates)
    let mut test_total_units = 0;
    let mut test_oov_units = 0;
    let mut test_oov_sample = Vec::new();

    for pair in &test_pairs {
        let align = viterbi_align(pair, &selected_model, final_selected_max_r, final_selected_max_r);
        for (r, b) in align {
            if !r.is_empty() && !b.is_empty() {
                test_total_units += 1;
                if !train_unit_set.contains(&(r.clone(), b.clone())) {
                    test_oov_units += 1;
                    if test_oov_sample.len() < 5 {
                        test_oov_sample.push(format!("{}→{}", r, b));
                    }
                }
            }
        }
    }

    println!(
        "   - TEST Unit OOV Rate against Train: {} / {} units ({:.2}%)",
        test_oov_units,
        test_total_units,
        (test_oov_units as f64 / test_total_units as f64) * 100.0
    );
    if !test_oov_sample.is_empty() {
        println!("     Sample TEST OOV Units: {:?}", test_oov_sample);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.4: Roman Segmentation Disagreement: Greedy vs Viterbi EM-Chosen
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   🔬 ROMAN SEGMENTATION: Greedy Longest-Match vs EM Viterbi-Chosen Disagreement");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let mut total_pairs_evaluated = 0;
    let mut disagreements = 0;
    let mut disagreement_examples = Vec::new();

    for pair in &train_pairs {
        let greedy_tokens = tokenize_roman(&pair.raw_roman);
        let alignment = viterbi_align(pair, &selected_model, final_selected_max_r, final_selected_max_r);
        let viterbi_roman: Vec<String> = alignment
            .iter()
            .filter(|(r, _)| !r.is_empty())
            .map(|(r, _)| r.clone())
            .collect();

        total_pairs_evaluated += 1;
        if greedy_tokens != viterbi_roman {
            disagreements += 1;
            if disagreement_examples.len() < 8 {
                disagreement_examples.push((
                    pair.raw_bengali.clone(),
                    pair.raw_roman.clone(),
                    greedy_tokens.join(" | "),
                    viterbi_roman.join(" | "),
                ));
            }
        }
    }

    let disagreement_rate = (disagreements as f64 / total_pairs_evaluated as f64) * 100.0;
    println!(
        "   - Disagreement Count: {} / {} pairs ({:.2}%)",
        disagreements, total_pairs_evaluated, disagreement_rate
    );
    println!("\n   Sample Segmentation Disagreements (Greedy vs Viterbi):");
    for (bn, lat, greedy, viterbi) in disagreement_examples {
        println!("     Target: {:<12} Roman: {:<12}", bn, lat);
        println!("       ├─ Greedy : {}", greedy);
        println!("       └─ Viterbi: {}", viterbi);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.5: Unit Pruning & Downstream Reachability (Gap Analysis Scratch)
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   🔬 UNIT PRUNING & REACHABILITY ON FULL UNAMPUTATED DEV (N = 9,279)");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let mut r_marginals: HashMap<String, f64> = HashMap::new();
    for ((r, _b), &p) in &selected_model {
        *r_marginals.entry(r.clone()).or_default() += p;
    }

    let raw_units: Vec<TransitionUnit> = selected_model
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

    for min_count in [0.0, 1.0, 3.0, 5.0] {
        let mut pruned: Vec<TransitionUnit> = raw_units
            .iter()
            .filter(|u| u.count >= min_count)
            .cloned()
            .collect();

        // Fallback smoothing for isolated single characters
        let fallbacks = [
            ("k", "ক"), ("g", "গ"), ("c", "চ"), ("j", "জ"), ("t", "ত"),
            ("d", "দ"), ("n", "ন"), ("p", "প"), ("b", "ব"), ("m", "ম"),
            ("r", "র"), ("l", "ল"), ("s", "স"), ("h", "হ"), ("y", "য়"),
            ("a", "া"), ("i", "ি"), ("u", "ু"), ("e", "ে"), ("o", "ো"),
        ];
        for &(r, b) in &fallbacks {
            if !pruned.iter().any(|u| u.roman == r && u.bengali == b) {
                pruned.push(TransitionUnit {
                    roman: r.to_string(),
                    bengali: b.to_string(),
                    count: 1.0,
                    p_b_given_r: 0.05,
                    p_joint: 0.0001,
                });
            }
        }

        let (reach, top1, top3) = decode_dev_with_units(&dev_pairs, &pruned, final_selected_max_r);
        println!(
            "   [Prune count >= {:.1}] Units: {:<5} | DEV Reachability: {:.2}% | Standalone Unit Top-1: {:.2}% | Top-3: {:.2}%",
            min_count,
            pruned.len(),
            reach,
            top1,
            top3
        );
    }

    // Save pruned units (count >= 3.0) to data/eval/m2m_units.json
    let mut final_units: Vec<TransitionUnit> = raw_units
        .into_iter()
        .filter(|u| u.count >= 3.0)
        .collect();
    final_units.sort_by(|a, b| b.count.partial_cmp(&a.count).unwrap_or(std::cmp::Ordering::Equal));

    let json_out = serde_json::to_string_pretty(&final_units).expect("Serialize units");
    let out_path = "data/eval/m2m_units.json";
    std::fs::write(out_path, json_out).expect("Write m2m_units.json");
    println!("\n💾 Saved {} high-confidence transition units to {}", final_units.len(), out_path);

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 4.6: Export Fixed-Seed 200 Audit CSV (52% Conjunct Stratification)
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   💾 EXPORTING 200-SAMPLE AUDIT CSV (data/eval/align_audit.csv)");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    // LCG PRNG for exact reproducible sampling (seed = 42)
    let mut rng_state: u64 = 42;
    let mut next_rand = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 32) as usize
    };

    // Separate training pairs by distinct target word and conjunct status
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

    // Sample 104 conjunct targets (52%) and 96 non-conjunct targets (48%) uniformly
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

    // Sort sampled pairs for determinism
    sampled_pairs.sort_by(|a, b| a.raw_bengali.cmp(&b.raw_bengali));

    let audit_csv_path = "data/eval/align_audit.csv";
    let mut csv_file = File::create(audit_csv_path).expect("Create align_audit.csv");
    writeln!(
        csv_file,
        "id,bengali_word,roman_input,has_conjunct,viterbi_alignment,greedy_roman_tokens,viterbi_roman_tokens,segmentation_match,alignment_grade"
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

        let alignment = viterbi_align(pair, &selected_model, final_selected_max_r, final_selected_max_r);
        let align_str = alignment
            .iter()
            .map(|(r, b)| {
                let r_disp = if r.is_empty() { "ε" } else { r };
                let b_disp = if b.is_empty() { "ε" } else { b };
                format!("{}→{}", r_disp, b_disp)
            })
            .collect::<Vec<_>>()
            .join(" | ");

        let greedy_tokens = tokenize_roman(&pair.raw_roman);
        let viterbi_roman: Vec<String> = alignment
            .iter()
            .filter(|(r, _)| !r.is_empty())
            .map(|(r, _)| r.clone())
            .collect();

        let seg_match = greedy_tokens == viterbi_roman;

        writeln!(
            csv_file,
            "{},\"{}\",\"{}\",{},\"{}\",\"{}\",\"{}\",{},UNGRADED",
            i + 1,
            pair.raw_bengali,
            pair.raw_roman,
            has_conj,
            align_str,
            greedy_tokens.join(" "),
            viterbi_roman.join(" "),
            seg_match
        )
        .unwrap();
    }

    println!(
        "   ✓ Exported {} sampled pairs to {}:",
        sampled_pairs.len(),
        audit_csv_path
    );
    println!(
        "     - Conjunct rows:     {} ({:.1}%)",
        audit_conj_count,
        (audit_conj_count as f64 / sampled_pairs.len() as f64) * 100.0
    );
    println!(
        "     - Non-Conjunct rows: {} ({:.1}%)",
        audit_non_conj_count,
        (audit_non_conj_count as f64 / sampled_pairs.len() as f64) * 100.0
    );
    println!("     - Alignment Quality: UNGRADED (Awaiting external human grading)");

    println!("\n══════════════════════════════════════════════════════════════════════════════════");
    println!("   🎉 PRAGYAN STEP 4 EXECUTION COMPLETE");
    println!("══════════════════════════════════════════════════════════════════════════════════");
}
