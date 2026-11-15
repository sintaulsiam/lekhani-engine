#![allow(clippy::all)]
//! Lekhani Multi-Persona Transliteration Evaluation Harness & Regression Gate
//!
//! Evaluates the phonetic transliteration engine across 6 standardized personas:
//! 1. natural: Raw human-annotated romanizations from held-out Dakshina splits
//! 2. lazy: Lowercase, no Shift distinctions, chat digraph folds (aa->a, ee->i, w->o)
//! 3. vowel_drop_p30: Non-initial vowels dropped with probability p = 0.30
//! 4. vowel_drop_p60: Non-initial vowels dropped with probability p = 0.60
//! 5. vowel_drop_p90: Non-initial vowels dropped with probability p = 0.90
//! 6. avro_strict: Canonical Avro forward-mapped sanity check ceiling
//!
//! Evaluates both DEV and TEST splits using strictly dictionary.json (zero Wikipedia unigrams).
//! Computes:
//! - Raw Avro-only Top-1 baseline (parser only, no lexicon/ranking)
//! - Top-1/Top-3/Top-5 accuracy
//! - Cluster bootstrap 95% confidence intervals (clustered by target word)
//! - Dual scoring: "any attested" vs "most frequent attested" vs "exact row"
//! - True candidate reachability vs "Not in Top-5"
//! - KSPC (Keystrokes Per Character): effective typing vs conditional on hits
//! - Latency percentiles (p50, p95, p99)

use hashbrown::{HashMap, HashSet};
use lekhani_core::chars::canonicalize;
use lekhani_core::fs::file_sha256_hex;
use lekhani_core::personas::{
    transform_avro_strict, transform_lazy, transform_natural, transform_vowel_drop, SimpleRng,
};
use lekhani_core::phonetic::PhoneticSuggestion;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    pub git_sha: String,
    pub dirty_tree: bool,
    pub seed: u64,
    pub canonical: String,
    pub dictionary_file: String,
    pub dictionary_hash: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitMetrics {
    pub n: usize,
    pub top1: f64,
    pub top3: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthMetrics {
    pub n: usize,
    pub top1: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyBreakdown {
    pub exact_row: f64,
    pub any_attested: f64,
    pub most_frequent_attested: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KspcMetrics {
    pub effective: f64,
    pub conditional_hits: f64,
    pub accept_at_rank: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaMetrics {
    pub n: usize,
    pub top1: f64,
    pub top1_ci95_cluster: [f64; 2],
    pub top3: f64,
    pub top5: f64,
    pub mrr: f64,
    pub not_in_top5: f64,
    pub unreachable: f64,
    pub scoring: AccuracyBreakdown,
    pub by_seen: HashMap<String, SplitMetrics>,
    pub by_length: HashMap<String, LengthMetrics>,
    pub kspc: KspcMetrics,
    pub latency_us: LatencyMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorstCaseItem {
    pub split: String,
    pub persona: String,
    pub input: String,
    pub expected: String,
    pub got: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitReport {
    pub n: usize,
    pub unique_targets: usize,
    pub raw_avro_top1: f64,
    pub personas: HashMap<String, PersonaMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPersonaReport {
    pub run: RunMeta,
    pub raw_avro_top1: HashMap<String, f64>,
    pub dev: SplitReport,
    pub test: SplitReport,
    pub personas: HashMap<String, PersonaMetrics>,
    pub worst_cases: Vec<WorstCaseItem>,
}

pub struct DakshinaGoldIndex {
    pub roman_to_attested: HashMap<String, HashSet<String>>,
    pub roman_to_most_frequent: HashMap<String, String>,
}

/// Compute 1000-iteration cluster bootstrap 95% confidence interval
/// Resamples clusters (target Bengali words) with replacement to account for correlation
fn compute_cluster_bootstrap_ci95(clusters: &[Vec<bool>], seed: u64) -> [f64; 2] {
    let num_clusters = clusters.len();
    if num_clusters == 0 {
        return [0.0, 0.0];
    }
    let mut rng = SimpleRng::new(seed);
    let iterations = 1000;
    let mut sampled_accs = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let mut total_hits = 0;
        let mut total_rows = 0;
        for _ in 0..num_clusters {
            let idx = (rng.next_u64() as usize) % num_clusters;
            let cluster = &clusters[idx];
            total_rows += cluster.len();
            for &hit in cluster {
                if hit {
                    total_hits += 1;
                }
            }
        }
        if total_rows > 0 {
            sampled_accs.push(total_hits as f64 / total_rows as f64 * 100.0);
        }
    }

    sampled_accs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let low_idx = (iterations as f64 * 0.025) as usize;
    let high_idx = (iterations as f64 * 0.975) as usize;
    [
        (sampled_accs[low_idx] * 100.0).round() / 100.0,
        (sampled_accs[high_idx] * 100.0).round() / 100.0,
    ]
}

fn get_git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().chars().take(8).collect())
        .unwrap_or_else(|| "unknown".to_string())
}

fn is_tree_dirty() -> bool {
    std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| {
            let relevant_lines = s
                .lines()
                .filter(|line| {
                    !line.contains("baseline.json")
                        && !line.contains("data/eval/reports")
                        && !line.contains("data/eval/test_runs.log")
                })
                .count();
            relevant_lines > 0
        })
        .unwrap_or(false)
}

fn setup_engine(dict_filename: &str) -> (PhoneticSuggestion, PathBuf, String) {
    let mut sugg = PhoneticSuggestion::new();
    let layout_candidates = [
        PathBuf::from("data/layouts/avrophonetic.json"),
        PathBuf::from("../data/layouts/avrophonetic.json"),
        PathBuf::from("../../data/layouts/avrophonetic.json"),
    ];
    for p in &layout_candidates {
        if p.exists() {
            if let Ok(content) = std::fs::read_to_string(p) {
                if let Ok(json) = serde_json::from_str(&content) {
                    sugg.set_layout(&json);
                    break;
                }
            }
        }
    }

    let dict_dir_candidates = [
        PathBuf::from("data/dictionaries"),
        PathBuf::from("../data/dictionaries"),
        PathBuf::from("../../data/dictionaries"),
    ];
    let dict_dir = dict_dir_candidates
        .iter()
        .find(|p| p.join(dict_filename).exists())
        .cloned()
        .expect("Directory containing dictionary file not found!");

    let dict_full_path = dict_dir.join(dict_filename);
    let dict_hash = file_sha256_hex(&dict_full_path)
        .expect("Failed to compute SHA-256 hash of dictionary.json");

    sugg.database
        .load_from_dir_with_dict(&dict_dir, dict_filename)
        .expect("Failed to load database with dictionary");

    (sugg, dict_full_path, dict_hash)
}

#[derive(Clone)]
struct RawRow {
    bn_target: String,
    latin: String,
    _count: usize,
}

fn load_split_data(split_name: &str) -> (Vec<RawRow>, DakshinaGoldIndex, PathBuf) {
    let filename = format!("bn.translit.sampled.{}.tsv", split_name);
    let candidates = [
        PathBuf::from(format!("data/eval/{}", filename)),
        PathBuf::from(format!("../data/eval/{}", filename)),
        PathBuf::from(format!("../../data/eval/{}", filename)),
        PathBuf::from(format!(
            "/home/smsiam/.gemini/antigravity-ide/brain/6e8f5951-b148-4374-87c2-2e54e03df641/scratch/dakshina/dakshina_dataset_v1.0/bn/lexicons/{}",
            filename
        )),
    ];
    let path = candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .unwrap_or_else(|| panic!("Dakshina {} file not found!", filename));

    let file = File::open(&path).unwrap_or_else(|_| panic!("Failed to open {:?}", path));
    let reader = BufReader::new(file);
    let mut rows = Vec::with_capacity(10000);
    let mut roman_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();

    for l in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = l.split('\t').collect();
        if parts.len() >= 2 {
            let bn_target = canonicalize(parts[0].trim());
            let latin = parts[1].trim().to_string();
            let count: usize = if parts.len() >= 3 {
                parts[2].trim().parse().unwrap_or(1)
            } else {
                1
            };
            if !bn_target.is_empty() && !latin.is_empty() {
                rows.push(RawRow {
                    bn_target: bn_target.clone(),
                    latin: latin.clone(),
                    _count: count,
                });

                roman_counts
                    .entry(latin)
                    .or_default()
                    .entry(bn_target)
                    .and_modify(|c| *c += count)
                    .or_insert(count);
            }
        }
    }

    let mut roman_to_attested = HashMap::new();
    let mut roman_to_most_frequent = HashMap::new();

    for (roman, bn_map) in roman_counts {
        let mut attested_set = HashSet::new();
        let mut max_count = 0;
        let mut best_bn = String::new();

        for (bn, count) in bn_map {
            if count > max_count {
                max_count = count;
                best_bn = bn.clone();
            }
            attested_set.insert(bn);
        }
        roman_to_attested.insert(roman.clone(), attested_set);
        roman_to_most_frequent.insert(roman, best_bn);
    }

    (
        rows,
        DakshinaGoldIndex {
            roman_to_attested,
            roman_to_most_frequent,
        },
        path,
    )
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
                        } else if let Some(arr) = item.as_array() {
                            if let Some(s) = arr.first().and_then(|v| v.as_str()) {
                                set.insert(canonicalize(s));
                            }
                        }
                    }
                }
            }
        }
    }
    set
}

fn compute_raw_avro_top1(cases: &[(String, String)]) -> f64 {
    let raw_parser = lekhani_parser::default_avro_parser();
    let mut hits = 0usize;
    for (bn_target, latin) in cases {
        let conv = canonicalize(&raw_parser.convert(latin));
        if &conv == bn_target {
            hits += 1;
        }
    }
    (hits as f64 / cases.len().max(1) as f64) * 100.0
}

#[allow(clippy::too_many_arguments)]
fn evaluate_persona(
    split_name: &str,
    persona_name: &str,
    cases: &[(String, String)], // (input_keystrokes, target_bn)
    sugg: &mut PhoneticSuggestion,
    dict_words: &HashSet<String>,
    gold_index: Option<&DakshinaGoldIndex>,
    seed: u64,
    worst_cases_out: &mut Vec<WorstCaseItem>,
) -> PersonaMetrics {
    let empty_memory = HashMap::new();
    let n = cases.len();
    let mut exact_hits = Vec::with_capacity(n);
    let mut any_attested_hits = 0usize;
    let mut most_freq_hits = 0usize;
    let mut top3_hits = 0usize;
    let mut top5_hits = 0usize;
    let mut mrr_sum = 0.0;
    let mut unreachable_count = 0usize;

    let mut latencies_us = Vec::with_capacity(n);
    let mut total_keystrokes_eff = 0usize;
    let mut hits_keystrokes = 0usize;
    let mut hits_target_chars = 0usize;
    let mut total_target_chars = 0usize;
    let mut rank_counts = [0usize; 3];

    let mut seen_n = 0;
    let mut seen_top1 = 0;
    let mut seen_top3 = 0;
    let mut unseen_n = 0;
    let mut unseen_top1 = 0;
    let mut unseen_top3 = 0;

    let mut len_1_3_n = 0;
    let mut len_1_3_top1 = 0;
    let mut len_4_6_n = 0;
    let mut len_4_6_top1 = 0;
    let mut len_7_plus_n = 0;
    let mut len_7_plus_top1 = 0;

    for (input, target_bn) in cases {
        let is_seen = dict_words.contains(target_bn);
        if is_seen {
            seen_n += 1;
        } else {
            unseen_n += 1;
        }

        let input_len = input.chars().count();
        let target_len = target_bn.chars().count();
        total_target_chars += target_len;

        if input_len <= 3 {
            len_1_3_n += 1;
        } else if input_len <= 6 {
            len_4_6_n += 1;
        } else {
            len_7_plus_n += 1;
        }

        let t0 = Instant::now();
        let (raw_candidates, _) = sugg.suggest(input, true, true, &empty_memory);
        let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
        latencies_us.push(elapsed_us);

        let candidates: Vec<String> = raw_candidates.iter().map(|s| canonicalize(s)).collect();
        let top_cand = candidates.first().cloned().unwrap_or_default();

        let mut rank = None;
        for (i, cand) in candidates.iter().take(5).enumerate() {
            if cand == target_bn {
                rank = Some(i);
                break;
            }
        }

        let is_top1 = rank == Some(0);
        exact_hits.push(is_top1);

        // Dual scoring
        if let Some(idx) = gold_index {
            if idx
                .roman_to_attested
                .get(input)
                .is_some_and(|set| set.contains(&top_cand))
            {
                any_attested_hits += 1;
            }
            if idx.roman_to_most_frequent.get(input) == Some(&top_cand) {
                most_freq_hits += 1;
            }
        } else {
            if is_top1 {
                any_attested_hits += 1;
                most_freq_hits += 1;
            }
        }

        if is_top1 {
            if is_seen {
                seen_top1 += 1;
            } else {
                unseen_top1 += 1;
            }
            if input_len <= 3 {
                len_1_3_top1 += 1;
            } else if input_len <= 6 {
                len_4_6_top1 += 1;
            } else {
                len_7_plus_top1 += 1;
            }
        }

        // True reachability: was target in the candidate list at all?
        let in_full_candidates = candidates.iter().any(|c| c == target_bn);
        if !in_full_candidates {
            unreachable_count += 1;
        }

        if let Some(r) = rank {
            if r < 3 {
                top3_hits += 1;
                if is_seen {
                    seen_top3 += 1;
                } else {
                    unseen_top3 += 1;
                }
            }
            if r < 5 {
                top5_hits += 1;
            }
            if r < 3 {
                rank_counts[r] += 1;
            }
            mrr_sum += 1.0 / (r as f64 + 1.0);

            // KSPC: selection cost (1 for space or tap, 2 for rank 3/4)
            let selection_cost = match r {
                0 => 1,
                1 | 2 => 1,
                _ => 2,
            };
            total_keystrokes_eff += input_len + selection_cost;
            hits_keystrokes += input_len + selection_cost;
            hits_target_chars += target_len;
        } else {
            // Miss penalty: backspace whole word + re-type target + space
            total_keystrokes_eff += input_len + target_len + 1;

            if worst_cases_out.len() < 50 {
                worst_cases_out.push(WorstCaseItem {
                    split: split_name.to_string(),
                    persona: persona_name.to_string(),
                    input: input.clone(),
                    expected: target_bn.clone(),
                    got: candidates.iter().take(3).cloned().collect(),
                });
            }
        }
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = latencies_us[latencies_us.len() * 50 / 100];
    let p95 = latencies_us[latencies_us.len() * 95 / 100];
    let p99 = latencies_us[latencies_us.len() * 99 / 100];

    let top1_pct = (exact_hits.iter().filter(|&&h| h).count() as f64 / n as f64) * 100.0;

    // Cluster bootstrap by target Bengali word
    let mut target_clusters: HashMap<&str, Vec<bool>> = HashMap::new();
    for ((_, target_bn), &hit) in cases.iter().zip(exact_hits.iter()) {
        target_clusters
            .entry(target_bn.as_str())
            .or_default()
            .push(hit);
    }
    let clusters: Vec<Vec<bool>> = target_clusters.into_values().collect();
    let top1_ci95_cluster = compute_cluster_bootstrap_ci95(&clusters, seed);

    let top3_pct = (top3_hits as f64 / n as f64) * 100.0;
    let top5_pct = (top5_hits as f64 / n as f64) * 100.0;
    let mrr = mrr_sum / n as f64;
    let not_in_top5 = 100.0 - top5_pct;
    let unreachable = (unreachable_count as f64 / n as f64) * 100.0;

    let effective_kspc = total_keystrokes_eff as f64 / total_target_chars.max(1) as f64;
    let conditional_kspc = hits_keystrokes as f64 / hits_target_chars.max(1) as f64;

    let any_attested_pct = (any_attested_hits as f64 / n as f64) * 100.0;
    let most_freq_pct = (most_freq_hits as f64 / n as f64) * 100.0;

    let mut by_seen = HashMap::new();
    by_seen.insert(
        "seen".to_string(),
        SplitMetrics {
            n: seen_n,
            top1: (seen_top1 as f64 / seen_n.max(1) as f64) * 100.0,
            top3: (seen_top3 as f64 / seen_n.max(1) as f64) * 100.0,
        },
    );
    by_seen.insert(
        "unseen".to_string(),
        SplitMetrics {
            n: unseen_n,
            top1: (unseen_top1 as f64 / unseen_n.max(1) as f64) * 100.0,
            top3: (unseen_top3 as f64 / unseen_n.max(1) as f64) * 100.0,
        },
    );

    let mut by_length = HashMap::new();
    by_length.insert(
        "1-3".to_string(),
        LengthMetrics {
            n: len_1_3_n,
            top1: (len_1_3_top1 as f64 / len_1_3_n.max(1) as f64) * 100.0,
        },
    );
    by_length.insert(
        "4-6".to_string(),
        LengthMetrics {
            n: len_4_6_n,
            top1: (len_4_6_top1 as f64 / len_4_6_n.max(1) as f64) * 100.0,
        },
    );
    by_length.insert(
        "7+".to_string(),
        LengthMetrics {
            n: len_7_plus_n,
            top1: (len_7_plus_top1 as f64 / len_7_plus_n.max(1) as f64) * 100.0,
        },
    );

    let rank_total = rank_counts[0] + rank_counts[1] + rank_counts[2];
    let accept_at_rank = if rank_total > 0 {
        [
            rank_counts[0] as f64 / rank_total as f64,
            rank_counts[1] as f64 / rank_total as f64,
            rank_counts[2] as f64 / rank_total as f64,
        ]
    } else {
        [0.0, 0.0, 0.0]
    };

    PersonaMetrics {
        n,
        top1: (top1_pct * 100.0).round() / 100.0,
        top1_ci95_cluster,
        top3: (top3_pct * 100.0).round() / 100.0,
        top5: (top5_pct * 100.0).round() / 100.0,
        mrr: (mrr * 10000.0).round() / 10000.0,
        not_in_top5: (not_in_top5 * 100.0).round() / 100.0,
        unreachable: (unreachable * 100.0).round() / 100.0,
        scoring: AccuracyBreakdown {
            exact_row: (top1_pct * 100.0).round() / 100.0,
            any_attested: (any_attested_pct * 100.0).round() / 100.0,
            most_frequent_attested: (most_freq_pct * 100.0).round() / 100.0,
        },
        by_seen,
        by_length,
        kspc: KspcMetrics {
            effective: (effective_kspc * 100.0).round() / 100.0,
            conditional_hits: (conditional_kspc * 100.0).round() / 100.0,
            accept_at_rank,
        },
        latency_us: LatencyMetrics {
            p50: (p50 * 10.0).round() / 10.0,
            p95: (p95 * 10.0).round() / 10.0,
            p99: (p99 * 10.0).round() / 10.0,
        },
    }
}

fn evaluate_split(
    split_name: &str,
    raw_pairs: &[(String, String)],
    gold_index: &DakshinaGoldIndex,
    dict_words: &HashSet<String>,
    sugg: &mut PhoneticSuggestion,
    seed: u64,
    worst_cases_out: &mut Vec<WorstCaseItem>,
) -> SplitReport {
    let raw_avro_top1 = compute_raw_avro_top1(raw_pairs);
    let mut report_personas = HashMap::new();

    let mut persona_order = [0, 1, 2, 3, 4, 5];
    let mut rng_order = SimpleRng::new(seed);
    for i in (1..6).rev() {
        let j = (rng_order.next_u64() as usize) % (i + 1);
        persona_order.swap(i, j);
    }

    for &p_idx in &persona_order {
        match p_idx {
            0 => {
                // 1. Natural
                println!("  ▶️  Evaluating [{}] - [Natural]...", split_name);
                let natural_cases: Vec<(String, String)> = raw_pairs
                    .iter()
                    .map(|(bn, latin)| (transform_natural(latin), bn.clone()))
                    .collect();
                let natural_metrics = evaluate_persona(
                    split_name,
                    "natural",
                    &natural_cases,
                    sugg,
                    dict_words,
                    Some(gold_index),
                    seed + 1,
                    worst_cases_out,
                );
                report_personas.insert("natural".to_string(), natural_metrics);
            }
            1 => {
                // 2. Lazy
                println!("  ▶️  Evaluating [{}] - [Lazy]...", split_name);
                let lazy_cases: Vec<(String, String)> = raw_pairs
                    .iter()
                    .map(|(bn, latin)| (transform_lazy(latin), bn.clone()))
                    .collect();
                let lazy_metrics = evaluate_persona(
                    split_name,
                    "lazy",
                    &lazy_cases,
                    sugg,
                    dict_words,
                    Some(gold_index),
                    seed + 2,
                    worst_cases_out,
                );
                report_personas.insert("lazy".to_string(), lazy_metrics);
            }
            2 => {
                // 3. Vowel-Dropped p=0.30
                println!(
                    "  ▶️  Evaluating [{}] - [Vowel-Dropped p=0.30]...",
                    split_name
                );
                let mut rng_p30 = SimpleRng::new(seed + 30);
                let vdrop_p30_cases: Vec<(String, String)> = raw_pairs
                    .iter()
                    .map(|(bn, latin)| {
                        (transform_vowel_drop(latin, 0.30, &mut rng_p30), bn.clone())
                    })
                    .collect();
                let vdrop_p30_metrics = evaluate_persona(
                    split_name,
                    "vowel_drop_p30",
                    &vdrop_p30_cases,
                    sugg,
                    dict_words,
                    None,
                    seed + 3,
                    worst_cases_out,
                );
                report_personas.insert("vowel_drop_p30".to_string(), vdrop_p30_metrics);
            }
            3 => {
                // 4. Vowel-Dropped p=0.60
                println!(
                    "  ▶️  Evaluating [{}] - [Vowel-Dropped p=0.60]...",
                    split_name
                );
                let mut rng_p60 = SimpleRng::new(seed + 60);
                let vdrop_p60_cases: Vec<(String, String)> = raw_pairs
                    .iter()
                    .map(|(bn, latin)| {
                        (transform_vowel_drop(latin, 0.60, &mut rng_p60), bn.clone())
                    })
                    .collect();
                let vdrop_p60_metrics = evaluate_persona(
                    split_name,
                    "vowel_drop_p60",
                    &vdrop_p60_cases,
                    sugg,
                    dict_words,
                    None,
                    seed + 4,
                    worst_cases_out,
                );
                report_personas.insert("vowel_drop_p60".to_string(), vdrop_p60_metrics);
            }
            4 => {
                // 5. Vowel-Dropped p=0.90
                println!(
                    "  ▶️  Evaluating [{}] - [Vowel-Dropped p=0.90]...",
                    split_name
                );
                let mut rng_p90 = SimpleRng::new(seed + 90);
                let vdrop_p90_cases: Vec<(String, String)> = raw_pairs
                    .iter()
                    .map(|(bn, latin)| {
                        (transform_vowel_drop(latin, 0.90, &mut rng_p90), bn.clone())
                    })
                    .collect();
                let vdrop_p90_metrics = evaluate_persona(
                    split_name,
                    "vowel_drop_p90",
                    &vdrop_p90_cases,
                    sugg,
                    dict_words,
                    None,
                    seed + 5,
                    worst_cases_out,
                );
                report_personas.insert("vowel_drop_p90".to_string(), vdrop_p90_metrics);
            }
            5 => {
                // 6. Avro-Strict
                println!("  ▶️  Evaluating [{}] - [Avro-Strict]...", split_name);
                let avro_cases: Vec<(String, String)> = raw_pairs
                    .iter()
                    .map(|(bn, _)| (transform_avro_strict(bn), bn.clone()))
                    .collect();
                let avro_metrics = evaluate_persona(
                    split_name,
                    "avro_strict",
                    &avro_cases,
                    sugg,
                    dict_words,
                    None,
                    seed + 6,
                    worst_cases_out,
                );
                report_personas.insert("avro_strict".to_string(), avro_metrics);
            }
            _ => unreachable!(),
        }
    }

    let unique_targets = raw_pairs
        .iter()
        .map(|(bn, _)| bn)
        .collect::<HashSet<_>>()
        .len();

    SplitReport {
        n: raw_pairs.len(),
        unique_targets,
        raw_avro_top1: (raw_avro_top1 * 100.0).round() / 100.0,
        personas: report_personas,
    }
}

fn print_split_table(title: &str, report: &SplitReport) {
    println!(
        "\n📊 {} (N = {}, Unique Targets = {}, Raw Avro Top-1 = {:.2}%):",
        title, report.n, report.unique_targets, report.raw_avro_top1
    );
    println!("───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────");
    println!(
        "{:<17} {:>6} {:>17} {:>10} {:>10} {:>8} {:>9} {:>9} {:>13} {:>11}",
        "Persona",
        "N",
        "Top-1 (Exact)",
        "Any Attest",
        "Top-3",
        "MRR",
        "NotInTop5",
        "Unreach%",
        "KSPC (Eff/Hit)",
        "Lat p95"
    );
    println!("───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────");

    let persona_order = [
        "natural",
        "lazy",
        "vowel_drop_p30",
        "vowel_drop_p60",
        "vowel_drop_p90",
        "avro_strict",
    ];
    for name in &persona_order {
        if let Some(m) = report.personas.get(*name) {
            let ci_str = format!(
                "{:.1}% [{:.1}-{:.1}]",
                m.top1, m.top1_ci95_cluster[0], m.top1_ci95_cluster[1]
            );
            let kspc_str = format!("{:.2} / {:.2}", m.kspc.effective, m.kspc.conditional_hits);
            println!(
                "{:<17} {:>6} {:>17} {:>9.1}% {:>9.1}% {:>8.4} {:>8.1}% {:>8.1}% {:>13} {:>9.0}µs",
                name,
                m.n,
                ci_str,
                m.scoring.any_attested,
                m.top3,
                m.mrr,
                m.not_in_top5,
                m.unreachable,
                kspc_str,
                m.latency_us.p95
            );
        }
    }
    println!("───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────");
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("  🚀 Lekhani Multi-Persona Transliteration Evaluation Harness & Regression Gate");
    println!("══════════════════════════════════════════════════════════════════════════════════════════════════\n");

    let args: Vec<String> = std::env::args().collect();
    let dict_filename = if args.iter().any(|a| a == "--dict-bin") {
        "dictionary.bin"
    } else {
        "dictionary.json"
    };
    let (mut sugg, dict_path, dict_hash) = setup_engine(dict_filename);

    let git_sha = get_git_sha();
    let dirty_tree = is_tree_dirty();
    let seed = 42u64;

    println!("Environment Metadata:");
    println!("  ➔ Git Commit SHA : {}", git_sha);
    println!("  ➔ Dirty Tree     : {}", dirty_tree);
    println!("  ➔ Dictionary File: {}", dict_path.display());
    println!("  ➔ Dictionary SHA : {}", dict_hash);
    println!("  ➔ Random Seed    : {}", seed);
    println!(
        "  ➔ Target NFC     : standard decomposed nuktas (U+09BC) + atomic Khanda Ta (U+09CE)"
    );
    println!("  ➔ Cluster Boot   : clustered by target Bengali word (1,000 iterations)\n");

    let json_path = dict_path.with_file_name("dictionary.json");
    let dict_words = load_dict_words(&json_path);
    println!(
        "Loaded {} canonical dictionary entries from {}.\n",
        dict_words.len(),
        json_path.display()
    );
    let benchmark_runs = args
        .iter()
        .position(|a| a == "--benchmark-latency")
        .and_then(|idx| args.get(idx + 1).and_then(|s| s.parse::<usize>().ok()))
        .or_else(|| {
            if args.iter().any(|a| a == "--benchmark-latency") {
                Some(5)
            } else {
                None
            }
        });

    if let Some(num_runs) = benchmark_runs {
        println!("══════════════════════════════════════════════════════════════════════════════════════════════════");
        println!(
            "   ⏱️  QUIET-MACHINE LATENCY BENCHMARK ({} RUNS ON HELD-OUT DEV)",
            num_runs
        );
        println!("══════════════════════════════════════════════════════════════════════════════════════════════════\n");

        let (dev_rows, dev_gold, _) = load_split_data("dev");
        let dev_pairs: Vec<(String, String)> = dev_rows
            .into_iter()
            .map(|r| (r.bn_target, r.latin))
            .collect();

        let personas = [
            "avro_strict",
            "lazy",
            "natural",
            "vowel_drop_p30",
            "vowel_drop_p60",
            "vowel_drop_p90",
        ];
        let mut runs_p50: HashMap<&str, Vec<f64>> = HashMap::new();
        let mut runs_p95: HashMap<&str, Vec<f64>> = HashMap::new();
        let mut runs_p99: HashMap<&str, Vec<f64>> = HashMap::new();

        println!("🔥 Executing warm-up pass (1,000 queries) to eliminate cold-start cache/page-fault overhead...");
        let warm_memory = HashMap::new();
        for (_, lat) in dev_pairs.iter().take(1000) {
            let _ = sugg.suggest(lat, true, true, &warm_memory);
        }

        for run_idx in 1..=num_runs {
            println!(
                "▶️  Executing Run {} / {} on idle system...",
                run_idx, num_runs
            );
            std::thread::sleep(std::time::Duration::from_millis(500));
            let mut wc = Vec::new();
            let report = evaluate_split(
                "dev",
                &dev_pairs,
                &dev_gold,
                &dict_words,
                &mut sugg,
                seed + run_idx as u64,
                &mut wc,
            );

            for &p in &personas {
                if let Some(m) = report.personas.get(p) {
                    runs_p50.entry(p).or_default().push(m.latency_us.p50);
                    runs_p95.entry(p).or_default().push(m.latency_us.p95);
                    runs_p99.entry(p).or_default().push(m.latency_us.p99);
                }
            }
        }

        println!("\n══════════════════════════════════════════════════════════════════════════════════════════════════");
        println!(
            "   📊 QUIET-MACHINE LATENCY DISTRIBUTIONS (N = {} Runs, DEV 9,279 rows)",
            num_runs
        );
        println!("══════════════════════════════════════════════════════════════════════════════════════════════════");
        println!("| Persona | Median p50 (µs) | Median p95 (µs) | Median p99 (µs) | p95 Spread [Min, Max] (µs) | All Run p95s (µs) |");
        println!("|:---|:---:|:---:|:---:|:---:|:---|");

        let mut medians: HashMap<String, LatencyMetrics> = HashMap::new();
        for &p in &personas {
            let mut p50s = runs_p50.get(p).cloned().unwrap_or_default();
            let mut p95s = runs_p95.get(p).cloned().unwrap_or_default();
            let mut p99s = runs_p99.get(p).cloned().unwrap_or_default();

            p50s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            p95s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            p99s.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let med_p50 = p50s[p50s.len() / 2];
            let med_p95 = p95s[p95s.len() / 2];
            let med_p99 = p99s[p99s.len() / 2];

            let min_p95 = *p95s.first().unwrap_or(&0.0);
            let max_p95 = *p95s.last().unwrap_or(&0.0);

            medians.insert(
                p.to_string(),
                LatencyMetrics {
                    p50: med_p50,
                    p95: med_p95,
                    p99: med_p99,
                },
            );

            let raw_p95_str: Vec<String> = runs_p95
                .get(p)
                .unwrap()
                .iter()
                .map(|v| format!("{:.0}", v))
                .collect();

            println!(
                "| **{:<14}** | {:>7.1} | {:>7.1} | {:>7.1} | [{:>6.1}, {:>6.1}] (Δ={:>5.1}) | {} |",
                p,
                med_p50,
                med_p95,
                med_p99,
                min_p95,
                max_p95,
                max_p95 - min_p95,
                raw_p95_str.join(", ")
            );
        }

        // Save raw timings JSON
        let raw_output = serde_json::json!({
            "num_runs": num_runs,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "p50_runs": runs_p50,
            "p95_runs": runs_p95,
            "p99_runs": runs_p99,
            "medians": medians,
        });
        let raw_json_path = Path::new("data/eval/quiet_latency_runs.json");
        std::fs::write(
            raw_json_path,
            serde_json::to_string_pretty(&raw_output).unwrap(),
        )
        .expect("Failed to write quiet_latency_runs.json");
        println!("\n💾 Saved raw multi-run timings to {:?}", raw_json_path);

        // Update baseline.json with restored median quiet-machine latencies
        let baseline_path = Path::new("data/eval/baseline.json");
        if baseline_path.exists() {
            if let Ok(content) = std::fs::read_to_string(baseline_path) {
                if let Ok(mut base_json) = serde_json::from_str::<serde_json::Value>(&content) {
                    for (p, m) in &medians {
                        if let Some(dev_persona) = base_json
                            .get_mut("dev")
                            .and_then(|d| d.get_mut("personas"))
                            .and_then(|ps| ps.get_mut(p))
                        {
                            dev_persona["latency_us"] = serde_json::json!({
                                "p50": m.p50,
                                "p95": m.p95,
                                "p99": m.p99,
                            });
                        }
                    }
                    std::fs::write(
                        baseline_path,
                        serde_json::to_string_pretty(&base_json).unwrap(),
                    )
                    .expect("Failed to update baseline.json with quiet-machine medians");
                    println!(
                        "💾 Restored quiet-machine median latencies in {:?}",
                        baseline_path
                    );
                }
            }
        }
        return;
    }

    let mut all_worst_cases = Vec::new();

    // ── DEV SPLIT EVALUATION ──────────────────────────────────────────────
    println!("▶️  Loading and Evaluating Held-Out DEV split...");
    let (dev_rows, dev_gold, dev_path) = load_split_data("dev");
    let dev_pairs: Vec<(String, String)> = dev_rows
        .into_iter()
        .map(|r| (r.bn_target, r.latin))
        .collect();
    println!("Loaded {} dev pairs from {:?}", dev_pairs.len(), dev_path);
    let dev_report = evaluate_split(
        "dev",
        &dev_pairs,
        &dev_gold,
        &dict_words,
        &mut sugg,
        seed,
        &mut all_worst_cases,
    );

    let dev_only = args.iter().any(|a| a == "--dev-only");
    let update_baseline = args.iter().any(|a| a == "--update-baseline");

    print_split_table("DEV SET EVALUATION SUMMARY", &dev_report);

    if !dev_only {
        // Log to test_runs.log per Standing Rule 5
        let log_path = Path::new("data/eval/test_runs.log");
        let now = chrono::Utc::now().to_rfc3339();
        let log_entry = format!(
            "{} | SHA: {} | Multi-persona test run; dict: {}\n",
            now,
            git_sha,
            dict_path.display()
        );
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(log_path) {
            use std::io::Write;
            let _ = f.write_all(log_entry.as_bytes());
        }

        // ── TEST SPLIT EVALUATION ─────────────────────────────────────────────
        println!("\n▶️  Loading and Evaluating Held-Out TEST split...");
        let (test_rows, test_gold, test_path) = load_split_data("test");
        let test_pairs: Vec<(String, String)> = test_rows
            .into_iter()
            .map(|r| (r.bn_target, r.latin))
            .collect();
        println!(
            "Loaded {} test pairs from {:?}",
            test_pairs.len(),
            test_path
        );
        let test_report = evaluate_split(
            "test",
            &test_pairs,
            &test_gold,
            &dict_words,
            &mut sugg,
            seed + 100,
            &mut all_worst_cases,
        );

        print_split_table("TEST SET EVALUATION SUMMARY", &test_report);

        if update_baseline {
            let mut raw_avro_top1 = HashMap::new();
            raw_avro_top1.insert("dev".to_string(), dev_report.raw_avro_top1);
            raw_avro_top1.insert("test".to_string(), test_report.raw_avro_top1);

            let report = MultiPersonaReport {
                run: RunMeta {
                    git_sha,
                    dirty_tree,
                    seed,
                    canonical: "nfc+khanda".to_string(),
                    dictionary_file: dict_path.display().to_string(),
                    dictionary_hash: dict_hash.clone(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                },
                raw_avro_top1,
                dev: dev_report,
                test: test_report.clone(),
                personas: test_report.personas, // Alias for backward compatibility
                worst_cases: all_worst_cases,
            };

            let baseline_path = Path::new("data/eval/baseline.json");
            if let Ok(json) = serde_json::to_string_pretty(&report) {
                let _ = std::fs::write(baseline_path, json);
                println!(
                    "\n💾 Saved updated multi-persona baseline report to {:?}",
                    baseline_path
                );
            }
        }
    }
}
