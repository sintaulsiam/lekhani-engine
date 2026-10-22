//! Measure Digraph Disagreement & Consonant Deletions in Viterbi Alignments
//!
//! Re-measures the disagreement properly across all 94,546 training pairs:
//! 1. Identifies Viterbi paths with a unit boundary inside a multi-letter Roman token.
//! 2. Categorizes splits by token type (consonant digraphs like sh, bh, th, dh; vowel digraphs; doubled consonants).
//! 3. Counts Roman consonant letters mapped to ε (deletions like h→ε, yo→ε, r→ε).
//! 4. Outputs exact counts, percentages, and frequent failure patterns before any changes.

use hashbrown::{HashMap, HashSet};
use lekhani_core::chars::canonicalize;
use lekhani_core::tokenization::{
    is_valid_translation_unit, normalize_roman, tokenize_bengali, tokenize_roman, BengaliToken,
};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
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
        let r_tokens: Vec<String> = lat_norm.chars().map(|c| c.to_string()).collect();

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
        ("k", "ক"), ("kh", "খ"), ("g", "গ"), ("gh", "ঘ"), ("ng", "ঙ"),
        ("c", "চ"), ("ch", "ছ"), ("j", "জ"), ("jh", "ঝ"),
        ("t", "ত"), ("t", "ট"), ("th", "থ"), ("th", "ঠ"),
        ("d", "দ"), ("d", "ড"), ("dh", "ধ"), ("dh", "ঢ"),
        ("n", "ন"), ("n", "ণ"),
        ("p", "প"), ("ph", "ফ"), ("f", "ফ"), ("b", "ব"), ("bh", "ভ"), ("v", "ভ"), ("m", "ম"),
        ("z", "য"), ("r", "র"), ("l", "ল"),
        ("sh", "শ"), ("sh", "ষ"), ("s", "স"), ("s", "শ"), ("h", "হ"),
        ("y", "য়"), ("w", "ও"), ("q", "ক"), ("x", "ক্স"),
        ("a", "া"), ("a", "আ"), ("a", "অ"), ("aa", "আ"), ("aa", "া"),
        ("i", "ি"), ("i", "ই"), ("ee", "ী"), ("ee", "ি"),
        ("u", "ু"), ("u", "উ"), ("oo", "ূ"), ("oo", "ু"),
        ("e", "ে"), ("e", "এ"), ("o", "ো"), ("o", "অ"), ("o", "ও"),
        ("oi", "ৈ"), ("ou", "ৌ"),
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

fn is_consonant_char(c: char) -> bool {
    matches!(c, 'b'|'c'|'d'|'f'|'g'|'h'|'j'|'k'|'l'|'m'|'n'|'p'|'q'|'r'|'s'|'t'|'v'|'w'|'x'|'y'|'z')
        && !matches!(c, 'a'|'e'|'i'|'o'|'u')
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════════════════════");
    println!("   🔬 MEASURING DIGRAPH SPLITS & CONSONANT DELETIONS ON TRAIN (N = 94,546)");
    println!("══════════════════════════════════════════════════════════════════════════════════");

    let train_path = Path::new("data/eval/bn.translit.sampled.train.tsv");
    let (train_pairs, dropped) = load_dataset(train_path);
    println!("Loaded {} pairs from TRAIN (Dropped: {})", train_pairs.len(), dropped);

    let t0 = Instant::now();
    println!("▶️  Training Baseline 1:1 Aligner on TRAIN...");
    let mut model_1_1 = initialize_model(&train_pairs, 1, 1, 50.0);
    for _ in 1..=DEFAULT_EM_ITERATIONS {
        model_1_1 = em_iteration(&train_pairs, &model_1_1, 1, 1);
    }
    println!("   1:1 Aligner training complete in {:.2}s", t0.elapsed().as_secs_f64());

    let t1 = Instant::now();
    println!("▶️  Training Baseline 2:2 Aligner on TRAIN...");
    let mut model_2_2 = initialize_model(&train_pairs, 2, 2, 50.0);
    for _ in 1..=DEFAULT_EM_ITERATIONS {
        model_2_2 = em_iteration(&train_pairs, &model_2_2, 2, 2);
    }
    println!("   2:2 Aligner training complete in {:.2}s", t1.elapsed().as_secs_f64());

    for (label, model, max_r, max_b) in [
        ("1:1 Baseline Model", &model_1_1, 1, 1),
        ("2:2 Baseline Model", &model_2_2, 2, 2),
    ] {
        println!("\n──────────────────────────────────────────────────────────────────────────────────");
        println!("   Auditing: {}", label);
        println!("──────────────────────────────────────────────────────────────────────────────────");

        let results: Vec<(bool, bool, Vec<String>, Vec<String>)> = train_pairs
            .par_iter()
            .map(|pair| {
                let (_, alignment) = viterbi_forced_alignment(pair, model, max_r, max_b);
                let greedy_tokens = tokenize_roman(&pair.roman_norm);

                // 1. Identify interior character boundaries of multi-char greedy tokens
                let mut interior_char_indices = HashSet::new();
                let mut curr_char = 0;
                for tok in &greedy_tokens {
                    let len = tok.len();
                    if len >= 2 {
                        for off in 1..len {
                            interior_char_indices.insert(curr_char + off);
                        }
                    }
                    curr_char += len;
                }

                // 2. Identify Viterbi unit cuts along Roman characters
                let mut viterbi_cuts = HashSet::new();
                let mut v_char = 0;
                let mut split_digraphs = Vec::new();
                let mut consonant_deletions = Vec::new();

                for (r, b) in &alignment {
                    if !r.is_empty() {
                        v_char += r.len();
                        viterbi_cuts.insert(v_char);
                    }
                    // Check for consonant letter mapped to ε
                    if !r.is_empty() && b.is_empty() {
                        if r.chars().any(is_consonant_char) {
                            consonant_deletions.push(format!("{}→ε", r));
                        }
                    }
                }

                let has_split = interior_char_indices.iter().any(|idx| viterbi_cuts.contains(idx));
                if has_split {
                    // Collect which tokens were split
                    let mut tok_start = 0;
                    for tok in &greedy_tokens {
                        let tok_end = tok_start + tok.len();
                        if tok.len() >= 2 {
                            let was_cut = (tok_start + 1..tok_end).any(|c| viterbi_cuts.contains(&c));
                            if was_cut {
                                split_digraphs.push(tok.clone());
                            }
                        }
                        tok_start = tok_end;
                    }
                }

                (has_split, !consonant_deletions.is_empty(), split_digraphs, consonant_deletions)
            })
            .collect();

        let total = results.len();
        let split_count = results.iter().filter(|r| r.0).count();
        let deletion_count = results.iter().filter(|r| r.1).count();
        let either_defect = results.iter().filter(|r| r.0 || r.1).count();

        println!("Audit Results on All {} Pairs:", total);
        println!("  ├─ Pairs with Internal Digraph Split : {:>6} / {} ({:>6.2}%)", split_count, total, split_count as f64 / total as f64 * 100.0);
        println!("  ├─ Pairs with Consonant Deleted (→ε) : {:>6} / {} ({:>6.2}%)", deletion_count, total, deletion_count as f64 / total as f64 * 100.0);
        println!("  └─ Pairs with EITHER Defect          : {:>6} / {} ({:>6.2}%)", either_defect, total, either_defect as f64 / total as f64 * 100.0);

        let mut digraph_freq: HashMap<String, usize> = HashMap::new();
        for r in &results {
            for d in &r.2 {
                *digraph_freq.entry(d.clone()).or_default() += 1;
            }
        }
        let mut sorted_digraphs: Vec<_> = digraph_freq.into_iter().collect();
        sorted_digraphs.sort_by(|a, b| b.1.cmp(&a.1));

        println!("\nTop Split Digraphs / Tokens:");
        for (tok, cnt) in sorted_digraphs.iter().take(15) {
            println!("    {:>6} : {:>5} occurrences ({:.2}% of split pairs)", tok, cnt, *cnt as f64 / split_count.max(1) as f64 * 100.0);
        }

        let mut del_freq: HashMap<String, usize> = HashMap::new();
        for r in &results {
            for d in &r.3 {
                *del_freq.entry(d.clone()).or_default() += 1;
            }
        }
        let mut sorted_dels: Vec<_> = del_freq.into_iter().collect();
        sorted_dels.sort_by(|a, b| b.1.cmp(&a.1));

        println!("\nTop Consonant Deletions (→ε):");
        for (del, cnt) in sorted_dels.iter().take(15) {
            println!("    {:>10} : {:>5} occurrences", del, cnt);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Constrained Model: Aligned strictly on Roman tokens with lattice options
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════════════════════════");
    println!("   🛡️  TRAINING CONSTRAINED ATOMIC TOKEN ALIGNER (LATTICE OPTIONS ONLY)");
    println!("══════════════════════════════════════════════════════════════════════════════════");
    
    // Convert pairs to atomic Roman tokens (greedy tokens by default)
    let token_pairs: Vec<AlignerPair> = train_pairs
        .iter()
        .map(|p| {
            let mut cp = p.clone();
            cp.roman_tokens = tokenize_roman(&p.roman_norm);
            cp
        })
        .collect();

    let t2 = Instant::now();
    println!("▶️  Training Constrained 1:1 Aligner on TRAIN...");
    let mut model_constrained = initialize_model(&token_pairs, 1, 1, 50.0);
    for _ in 1..=DEFAULT_EM_ITERATIONS {
        model_constrained = em_iteration(&token_pairs, &model_constrained, 1, 1);
    }
    println!("   Constrained Aligner training complete in {:.2}s", t2.elapsed().as_secs_f64());

    println!("\n──────────────────────────────────────────────────────────────────────────────────");
    println!("   Auditing: Constrained Atomic Token Model (with Lattice Options)");
    println!("──────────────────────────────────────────────────────────────────────────────────");

    let results_constrained: Vec<(bool, bool, Vec<String>, Vec<String>)> = train_pairs
        .par_iter()
        .map(|pair| {
            use lekhani_core::tokenization::roman_segmentation_lattice;
            let lattice_paths = roman_segmentation_lattice(&pair.roman_norm);
            
            let mut best_score = f64::NEG_INFINITY;
            let mut best_alignment = Vec::new();

            for path in &lattice_paths {
                let mut temp_pair = pair.clone();
                temp_pair.roman_tokens = path.clone();
                let (score, alignment) = viterbi_forced_alignment(&temp_pair, &model_constrained, 1, 1);
                if score > best_score {
                    best_score = score;
                    best_alignment = alignment;
                }
            }

            // Fallback if none reachable
            if best_score == f64::NEG_INFINITY {
                let (score, alignment) = viterbi_forced_alignment(pair, &model_constrained, 1, 1);
                if score > best_score {
                    best_alignment = alignment;
                }
            }

            let greedy_tokens = tokenize_roman(&pair.roman_norm);

            let mut interior_char_indices = HashSet::new();
            let mut curr_char = 0;
            for tok in &greedy_tokens {
                let len = tok.len();
                if len >= 2 {
                    for off in 1..len {
                        interior_char_indices.insert(curr_char + off);
                    }
                }
                curr_char += len;
            }

            let mut viterbi_cuts = HashSet::new();
            let mut v_char = 0;
            let mut split_digraphs = Vec::new();
            let mut consonant_deletions = Vec::new();

            for (r, b) in &best_alignment {
                if !r.is_empty() {
                    v_char += r.len();
                    viterbi_cuts.insert(v_char);
                }
                if !r.is_empty() && b.is_empty() {
                    if r.chars().any(is_consonant_char) {
                        consonant_deletions.push(format!("{}→ε", r));
                    }
                }
            }

            let has_split = interior_char_indices.iter().any(|idx| viterbi_cuts.contains(idx));
            if has_split {
                let mut tok_start = 0;
                for tok in &greedy_tokens {
                    let tok_end = tok_start + tok.len();
                    if tok.len() >= 2 {
                        let was_cut = (tok_start + 1..tok_end).any(|c| viterbi_cuts.contains(&c));
                        if was_cut {
                            split_digraphs.push(tok.clone());
                        }
                    }
                    tok_start = tok_end;
                }
            }

            (has_split, !consonant_deletions.is_empty(), split_digraphs, consonant_deletions)
        })
        .collect();

    let total_c = results_constrained.len();
    let split_count_c = results_constrained.iter().filter(|r| r.0).count();
    let deletion_count_c = results_constrained.iter().filter(|r| r.1).count();
    let either_defect_c = results_constrained.iter().filter(|r| r.0 || r.1).count();

    println!("Audit Results on All {} Pairs:", total_c);
    println!("  ├─ Pairs with Internal Digraph Split : {:>6} / {} ({:>6.2}%)", split_count_c, total_c, split_count_c as f64 / total_c as f64 * 100.0);
    println!("  ├─ Pairs with Consonant Deleted (→ε) : {:>6} / {} ({:>6.2}%)", deletion_count_c, total_c, deletion_count_c as f64 / total_c as f64 * 100.0);
    println!("  └─ Pairs with EITHER Defect          : {:>6} / {} ({:>6.2}%)", either_defect_c, total_c, either_defect_c as f64 / total_c as f64 * 100.0);

    let mut digraph_freq_c: HashMap<String, usize> = HashMap::new();
    for r in &results_constrained {
        for d in &r.2 {
            *digraph_freq_c.entry(d.clone()).or_default() += 1;
        }
    }
    let mut sorted_digraphs_c: Vec<_> = digraph_freq_c.into_iter().collect();
    sorted_digraphs_c.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\nTop Split Digraphs / Tokens (Constrained Model):");
    for (tok, cnt) in sorted_digraphs_c.iter().take(15) {
        println!("    {:>6} : {:>5} occurrences ({:.2}% of split pairs)", tok, cnt, *cnt as f64 / split_count_c.max(1) as f64 * 100.0);
    }

    let mut del_freq_c: HashMap<String, usize> = HashMap::new();
    for r in &results_constrained {
        for d in &r.3 {
            *del_freq_c.entry(d.clone()).or_default() += 1;
        }
    }
    let mut sorted_dels_c: Vec<_> = del_freq_c.into_iter().collect();
    sorted_dels_c.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\nTop Consonant Deletions (→ε) (Constrained Model):");
    for (del, cnt) in sorted_dels_c.iter().take(15) {
        println!("    {:>10} : {:>5} occurrences", del, cnt);
    }
}
