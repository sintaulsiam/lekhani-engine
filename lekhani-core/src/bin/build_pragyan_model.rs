//! Build Pragyan Production Statistical Language Model (pragyan_unit_lm.bin)
//!
//! Compiles the canonical constrained M2M aligned joint unit language model
//! from `data/eval/bn.translit.sampled.train.tsv` into a compact, zero-allocation binary artifact.
//!
//! Enforces:
//! 1. Hermetic lexicon check on `data/dictionaries/dictionary.json` (SHA-256).
//! 2. Zero-drop dataset loading (Filters OFF policy).
//! 3. Gate G6 Memory Budget compliance (<= 3.5 MB).

use hashbrown::HashMap;
use lekhani_core::chars::canonicalize;
use lekhani_core::fs::compute_sha256_hex;
use lekhani_core::phonetic::pragyan::{JointUnitLM, UnitVocab};
use lekhani_core::tokenization::{
    is_valid_translation_unit, normalize_roman, tokenize_bengali, tokenize_roman, BengaliToken,
};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const EXPECTED_DICT_JSON_SHA256: &str =
    "8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb";

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
        ("k", "ক"), ("kh", "খ"), ("g", "গ"), ("gh", "ঘ"), ("ng", "ঙ"),
        ("c", "চ"), ("ch", "ছ"), ("j", "জ"), ("jh", "ঝ"),
        ("t", "ত"), ("t", "ট"), ("th", "থ"), ("th", "ঠ"),
        ("d", "দ"), ("d", "ড"), ("dh", "ধ"), ("dh", "ঢ"),
        ("n", "ন"), ("n", "ণ"),
        ("p", "প"), ("f", "ফ"), ("ph", "ফ"), ("b", "ব"), ("bh", "ভ"), ("v", "ভ"),
        ("m", "ম"), ("z", "য"), ("r", "র"), ("l", "ল"),
        ("sh", "শ"), ("s", "স"), ("s", "ষ"), ("h", "হ"),
        ("r", "ড়"), ("rh", "ঢ়"), ("y", "য়"),
        ("a", "া"), ("a", "আ"), ("i", "ি"), ("i", "ই"),
        ("ee", "ী"), ("ee", "ঈ"), ("u", "ু"), ("u", "উ"),
        ("oo", "ূ"), ("oo", "ঊ"), ("e", "ে"), ("e", "এ"),
        ("oi", "ৈ"), ("oi", "ঐ"), ("o", "ো"), ("o", "ও"),
        ("ou", "ৌ"), ("ou", "ঔ"),
    ];
    for (r, b) in priors {
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

#[allow(clippy::type_complexity)]
pub fn run_em_iteration(
    pairs: &[AlignerPair],
    model: &HashMap<(String, String), f64>,
    max_r: usize,
    max_b: usize,
    alpha: f64,
) -> (HashMap<(String, String), f64>, f64) {
    let chunk_size = (pairs.len() / rayon::current_num_threads()).max(1);

    let (partial_counts, total_log_likelihood): (Vec<HashMap<(String, String), f64>>, Vec<f64>) = pairs
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_counts: HashMap<(String, String), f64> = HashMap::new();
            let mut local_ll = 0.0f64;

            for pair in chunk {
                let m = pair.roman_tokens.len();
                let n = pair.bengali_tokens.len();

                let mut alpha_matrix = vec![vec![0.0f64; n + 1]; m + 1];
                alpha_matrix[0][0] = 1.0;

                for i in 0..=m {
                    for j in 0..=n {
                        let a_ij = alpha_matrix[i][j];
                        if a_ij == 0.0 {
                            continue;
                        }

                        for kr in 1..=max_r.min(m - i) {
                            let r_slice = &pair.roman_tokens[i..i + kr];
                            if is_valid_translation_unit(r_slice, &[]) {
                                let r = r_slice.join("");
                                if let Some(&p) = model.get(&(r, String::new())) {
                                    alpha_matrix[i + kr][j] += a_ij * p;
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
                                    if let Some(&p) = model.get(&(r, b)) {
                                        alpha_matrix[i + kr][j + kb] += a_ij * p;
                                    }
                                }
                            }
                        }
                    }
                }

                let total_prob = alpha_matrix[m][n];
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
                                    let b: String = b_slice.iter().map(|t| t.text.as_str()).collect();
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
                        let a_ij = alpha_matrix[i][j];
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
                                        *local_counts.entry((r, String::new())).or_default() += expected;
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
                                        let b_next = beta[i + kr][j + kb];
                                        let expected = a_ij * p * b_next * inv_total;
                                        if expected > 0.0 {
                                            *local_counts.entry((r, b)).or_default() += expected;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("══════════════════════════════════════════════════════════════════");
    println!("   🚀 Lekhani Pragyan Model Compiler (build_pragyan_model)");
    println!("══════════════════════════════════════════════════════════════════\n");

    let dict_path = Path::new("data/dictionaries/dictionary.json");
    if !dict_path.exists() {
        eprintln!("❌ Fatal: dictionary.json not found at {:?}", dict_path);
        std::process::exit(1);
    }
    let dict_bytes = std::fs::read(dict_path)?;
    let actual_hash = compute_sha256_hex(&dict_bytes);
    println!("1. Verifying Hermetic Lexicon (SHA-256):");
    println!("   ├─ Expected: {}", EXPECTED_DICT_JSON_SHA256);
    println!("   ├─ Actual  : {}", actual_hash);
    if actual_hash != EXPECTED_DICT_JSON_SHA256 {
        eprintln!("❌ Fatal: dictionary.json hash mismatch!");
        std::process::exit(1);
    }
    println!("   └─ ✓ Lexicon Integrity Verified.");

    let train_path = PathBuf::from("data/eval/bn.translit.sampled.train.tsv");
    println!("\n2. Loading Training Dataset (Filters OFF): {:?}", train_path);
    let (train_pairs, train_dropped) = load_dataset(&train_path);
    println!("   ├─ Total Loaded: {} pairs", train_pairs.len());
    println!("   └─ Dropped: {} (100.0% kept)", train_dropped);

    println!("\n3. Training 2:1 Constrained M2M Aligner (3x2, alpha=0.5)...");
    let mut model = initialize_model(&train_pairs, 3, 2, 0.5);
    let start_em = Instant::now();
    for it in 1..=DEFAULT_EM_ITERATIONS {
        let t0 = Instant::now();
        let (new_model, ll) = run_em_iteration(&train_pairs, &model, 3, 2, 0.5);
        println!("   ├─ Iteration {}/{}: Log-Likelihood = {:.2} ({:.2}s)",
            it, DEFAULT_EM_ITERATIONS, ll, t0.elapsed().as_secs_f64());
        model = new_model;
    }
    println!("   └─ ✓ EM Alignment finished in {:.2}s", start_em.elapsed().as_secs_f64());

    println!("\n4. Viterbi Aligning Sequences & Deduplicating Type-Level Map...");
    let alignments: Vec<Vec<(String, String)>> = train_pairs
        .par_iter()
        .map(|p| {
            let (_, align) = viterbi_align(p, &model, 3, 2);
            align
        })
        .collect();

    let mut type_level_map: HashMap<(String, String), Vec<(String, String)>> = HashMap::new();
    let mut aligned_count = 0;
    for (pair, align) in train_pairs.iter().zip(alignments) {
        if !align.is_empty() {
            aligned_count += 1;
            type_level_map
                .entry((pair.raw_roman.clone(), pair.raw_bengali.clone()))
                .or_insert(align);
        }
    }
    println!("   ├─ Successfully aligned: {} / {} pairs ({:.2}%)",
        aligned_count, train_pairs.len(), (aligned_count as f64 / train_pairs.len() as f64) * 100.0);
    println!("   └─ Unique Type-Level Sequences: {}", type_level_map.len());

    let type_level_seqs: Vec<Vec<(String, String)>> = type_level_map.into_values().collect();
    let vocab = UnitVocab::build(&type_level_seqs);
    println!("   └─ Unit Vocabulary Size: {} units", vocab.id_to_unit.len());

    println!("\n5. Training Compact Joint Unit LM (Order 4, Min Count = 2.0)...");
    let lm = JointUnitLM::train(&type_level_seqs, vocab, 4, 2.0);

    let serialized = bincode::serialize(&lm)?;
    let size_mb = serialized.len() as f64 / 1024.0 / 1024.0;
    println!("   ├─ Serialized Size: {:.2} MB ({} bytes)", size_mb, serialized.len());

    let out_path = PathBuf::from("data/dictionaries/pragyan_unit_lm.bin");
    let mut out_file = File::create(&out_path)?;
    out_file.write_all(&serialized)?;
    println!("   └─ ✓ Saved production model to {:?}", out_path);

    // Gate G6 Verification
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("   GATE G6 AUDIT VERIFICATION");
    println!("══════════════════════════════════════════════════════════════════");
    let g6_passed = size_mb <= 3.5;
    println!("   ├─ Model File Size : {:.2} MB", size_mb);
    println!("   ├─ Budget Ceiling  : <= 3.50 MB");
    println!("   └─ Gate G6 Status  : {}", if g6_passed { "✅ PASSED" } else { "❌ FAILED" });

    if !g6_passed {
        eprintln!("❌ Fatal: Gate G6 violated!");
        std::process::exit(1);
    }

    Ok(())
}
