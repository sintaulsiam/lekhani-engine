//! Lekhani Multi-Persona Transliteration Evaluation Harness & Regression Gate
//!
//! Evaluates the phonetic transliteration engine across 5 standardized personas:
//! 1. natural: Raw human-annotated romanizations from held-out Dakshina test set
//! 2. lazy: Lowercase, no Shift distinctions, chat digraph folds (aa->a, ee->i, w->o)
//! 3. vowel_drop_p30: Non-initial vowels dropped with probability p = 0.30
//! 4. vowel_drop_p60: Non-initial vowels dropped with probability p = 0.60
//! 5. avro_strict: Canonical Avro forward-mapped sanity check ceiling
//!
//! Computes Top-1/Top-3/Top-5 accuracy, bootstrap 95% confidence intervals,
//! KSPC (Keystrokes Per Character), OOV rate, seen/unseen splits, length breakdowns,
//! and latency percentiles (p50, p95, p99).

use hashbrown::{HashMap, HashSet};
use lekhani_core::chars::canonicalize;
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
    pub seed: u64,
    pub split: String,
    pub canonical: String,
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
pub struct KspcMetrics {
    pub mean: f64,
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
    pub top1_ci95: [f64; 2],
    pub top3: f64,
    pub top5: f64,
    pub mrr: f64,
    pub oov_rate: f64,
    pub by_seen: HashMap<String, SplitMetrics>,
    pub by_length: HashMap<String, LengthMetrics>,
    pub kspc: KspcMetrics,
    pub latency_us: LatencyMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorstCaseItem {
    pub persona: String,
    pub input: String,
    pub expected: String,
    pub got: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPersonaReport {
    pub run: RunMeta,
    pub personas: HashMap<String, PersonaMetrics>,
    pub worst_cases: Vec<WorstCaseItem>,
}

/// Compute 1000-iteration bootstrap 95% confidence interval
fn compute_bootstrap_ci95(hits: &[bool], seed: u64) -> [f64; 2] {
    let n = hits.len();
    if n == 0 {
        return [0.0, 0.0];
    }
    let mut rng = SimpleRng::new(seed);
    let iterations = 1000;
    let mut sampled_accs = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let mut count = 0;
        for _ in 0..n {
            let idx = (rng.next_u64() as usize) % n;
            if hits[idx] {
                count += 1;
            }
        }
        sampled_accs.push(count as f64 / n as f64 * 100.0);
    }

    sampled_accs.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

fn setup_engine() -> PhoneticSuggestion {
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
    let dict_candidates = [
        PathBuf::from("data/dictionaries"),
        PathBuf::from("../data/dictionaries"),
        PathBuf::from("../../data/dictionaries"),
    ];
    for p in &dict_candidates {
        if p.exists() {
            let _ = sugg.database.load_from_dir(p);
            break;
        }
    }
    sugg
}

fn load_test_pairs() -> Vec<(String, String)> {
    let test_candidates = [
        PathBuf::from("/home/smsiam/.gemini/antigravity-ide/brain/6e8f5951-b148-4374-87c2-2e54e03df641/scratch/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.test.tsv"),
        PathBuf::from("data/eval/bn.translit.sampled.test.tsv"),
        PathBuf::from("../data/eval/bn.translit.sampled.test.tsv"),
    ];
    let test_path = test_candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .expect("Held-out Dakshina test.tsv not found!");

    let file = File::open(&test_path).expect("Failed to open test.tsv");
    let reader = BufReader::new(file);
    let mut pairs = Vec::with_capacity(10000);

    for line in reader.lines() {
        if let Ok(l) = line {
            let parts: Vec<&str> = l.split('\t').collect();
            if parts.len() >= 2 {
                let bn_target = canonicalize(parts[0].trim());
                let latin = parts[1].trim().to_string();
                if !bn_target.is_empty() && !latin.is_empty() {
                    pairs.push((bn_target, latin));
                }
            }
        }
    }

    pairs
}

fn load_dict_words() -> HashSet<String> {
    let dict_candidates = [
        PathBuf::from("data/dictionaries/dictionary.json"),
        PathBuf::from("../data/dictionaries/dictionary.json"),
    ];
    let mut set = HashSet::new();
    for p in &dict_candidates {
        if p.exists() {
            if let Ok(file) = File::open(p) {
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
                    break;
                }
            }
        }
    }
    set
}

fn evaluate_persona(
    persona_name: &str,
    cases: &[(String, String)], // (input_keystrokes, expected_bn)
    sugg: &mut PhoneticSuggestion,
    dict_words: &HashSet<String>,
    seed: u64,
    worst_cases_out: &mut Vec<WorstCaseItem>,
) -> PersonaMetrics {
    let empty_memory = HashMap::new();
    let n = cases.len();
    let mut top1_hits = Vec::with_capacity(n);
    let mut top3_hits = 0;
    let mut top5_hits = 0;
    let mut mrr_sum = 0.0;
    let mut oov_count = 0;

    let mut latencies_us = Vec::with_capacity(n);
    let mut total_keystrokes = 0usize;
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
        let elapsed_us = t0.elapsed().as_micros() as f64;
        latencies_us.push(elapsed_us);

        let candidates: Vec<String> = raw_candidates.iter().map(|s| canonicalize(s)).collect();

        let mut rank = None;
        for (i, cand) in candidates.iter().enumerate() {
            if cand == target_bn {
                rank = Some(i);
                break;
            }
        }

        let is_top1 = rank == Some(0);
        top1_hits.push(is_top1);

        if is_top1 {
            if is_seen { seen_top1 += 1; } else { unseen_top1 += 1; }
            if input_len <= 3 { len_1_3_top1 += 1; }
            else if input_len <= 6 { len_4_6_top1 += 1; }
            else { len_7_plus_top1 += 1; }
        }

        if let Some(r) = rank {
            if r < 3 {
                top3_hits += 1;
                if is_seen { seen_top3 += 1; } else { unseen_top3 += 1; }
            }
            if r < 5 {
                top5_hits += 1;
            }
            if r < 3 {
                rank_counts[r] += 1;
            }
            mrr_sum += 1.0 / (r as f64 + 1.0);

            // KSPC keystroke cost
            let selection_cost = match r {
                0 => 1, // Space
                1 | 2 => 1, // Candidate tap
                _ => 2, // Scroll & tap
            };
            total_keystrokes += input_len + selection_cost;
        } else {
            oov_count += 1;
            // Penalty: backspace whole word + re-type target + space
            total_keystrokes += input_len + target_len + 1;

            if worst_cases_out.len() < 50 {
                worst_cases_out.push(WorstCaseItem {
                    persona: persona_name.to_string(),
                    input: input.clone(),
                    expected: target_bn.clone(),
                    got: candidates.iter().take(3).cloned().collect(),
                });
            }
        }
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies_us[latencies_us.len() * 50 / 100];
    let p95 = latencies_us[latencies_us.len() * 95 / 100];
    let p99 = latencies_us[latencies_us.len() * 99 / 100];

    let top1_pct = (top1_hits.iter().filter(|&&h| h).count() as f64 / n as f64) * 100.0;
    let top1_ci95 = compute_bootstrap_ci95(&top1_hits, seed);
    let top3_pct = (top3_hits as f64 / n as f64) * 100.0;
    let top5_pct = (top5_hits as f64 / n as f64) * 100.0;
    let mrr = mrr_sum / n as f64;
    let oov_rate = (oov_count as f64 / n as f64) * 100.0;
    let mean_kspc = total_keystrokes as f64 / total_target_chars.max(1) as f64;

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

    PersonaMetrics {
        n,
        top1: (top1_pct * 100.0).round() / 100.0,
        top1_ci95,
        top3: (top3_pct * 100.0).round() / 100.0,
        top5: (top5_pct * 100.0).round() / 100.0,
        mrr: (mrr * 10000.0).round() / 10000.0,
        oov_rate: (oov_rate * 100.0).round() / 100.0,
        by_seen,
        by_length,
        kspc: KspcMetrics {
            mean: (mean_kspc * 1000.0).round() / 1000.0,
            accept_at_rank: [
                (rank_counts[0] as f64 / n as f64 * 100.0).round() / 100.0,
                (rank_counts[1] as f64 / n as f64 * 100.0).round() / 100.0,
                (rank_counts[2] as f64 / n as f64 * 100.0).round() / 100.0,
            ],
        },
        latency_us: LatencyMetrics {
            p50: (p50 * 10.0).round() / 10.0,
            p95: (p95 * 10.0).round() / 10.0,
            p99: (p99 * 10.0).round() / 10.0,
        },
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       🎯 Lekhani Multi-Persona Transliteration Evaluation        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let seed: u64 = 42;
    let git_sha = get_git_sha();
    println!("Git SHA: {} | Seed: {}", git_sha, seed);

    let raw_pairs = load_test_pairs();
    println!("Loaded {} held-out test pairs from Dakshina.", raw_pairs.len());

    let dict_words = load_dict_words();
    println!("Loaded {} canonical words from dictionary.json.", dict_words.len());

    let mut sugg = setup_engine();
    let mut report_personas = HashMap::new();
    let mut all_worst_cases = Vec::new();

    // 1. Natural Persona (Raw human romanization)
    println!("\n▶️  Evaluating Persona: [Natural]...");
    let natural_cases: Vec<(String, String)> = raw_pairs
        .iter()
        .map(|(bn, latin)| (transform_natural(latin), bn.clone()))
        .collect();
    let natural_metrics = evaluate_persona(
        "natural",
        &natural_cases,
        &mut sugg,
        &dict_words,
        seed + 1,
        &mut all_worst_cases,
    );
    report_personas.insert("natural".to_string(), natural_metrics);

    // 2. Lazy Persona (Lowercase, collapsed Shift, chat folds)
    println!("▶️  Evaluating Persona: [Lazy]...");
    let lazy_cases: Vec<(String, String)> = raw_pairs
        .iter()
        .map(|(bn, latin)| (transform_lazy(latin), bn.clone()))
        .collect();
    let lazy_metrics = evaluate_persona(
        "lazy",
        &lazy_cases,
        &mut sugg,
        &dict_words,
        seed + 2,
        &mut all_worst_cases,
    );
    report_personas.insert("lazy".to_string(), lazy_metrics);

    // 3. Vowel-Dropped Persona (p = 0.30)
    println!("▶️  Evaluating Persona: [Vowel-Dropped p=0.30]...");
    let mut rng_p30 = SimpleRng::new(seed + 30);
    let vdrop_p30_cases: Vec<(String, String)> = raw_pairs
        .iter()
        .map(|(bn, latin)| (transform_vowel_drop(latin, 0.30, &mut rng_p30), bn.clone()))
        .collect();
    let vdrop_p30_metrics = evaluate_persona(
        "vowel_drop_p30",
        &vdrop_p30_cases,
        &mut sugg,
        &dict_words,
        seed + 3,
        &mut all_worst_cases,
    );
    report_personas.insert("vowel_drop_p30".to_string(), vdrop_p30_metrics);

    // 4. Vowel-Dropped Persona (p = 0.60)
    println!("▶️  Evaluating Persona: [Vowel-Dropped p=0.60]...");
    let mut rng_p60 = SimpleRng::new(seed + 60);
    let vdrop_p60_cases: Vec<(String, String)> = raw_pairs
        .iter()
        .map(|(bn, latin)| (transform_vowel_drop(latin, 0.60, &mut rng_p60), bn.clone()))
        .collect();
    let vdrop_p60_metrics = evaluate_persona(
        "vowel_drop_p60",
        &vdrop_p60_cases,
        &mut sugg,
        &dict_words,
        seed + 4,
        &mut all_worst_cases,
    );
    report_personas.insert("vowel_drop_p60".to_string(), vdrop_p60_metrics);

    // 5. Avro-Strict Persona (Canonical forward-mapped ceiling)
    println!("▶️  Evaluating Persona: [Avro-Strict]...");
    let avro_cases: Vec<(String, String)> = raw_pairs
        .iter()
        .map(|(bn, _)| (transform_avro_strict(bn), bn.clone()))
        .collect();
    let avro_metrics = evaluate_persona(
        "avro_strict",
        &avro_cases,
        &mut sugg,
        &dict_words,
        seed + 5,
        &mut all_worst_cases,
    );
    report_personas.insert("avro_strict".to_string(), avro_metrics);

    // Display Summary Table
    println!("\n📊 Multi-Persona Evaluation Summary:");
    println!("─────────────────────────────────────────────────────────────────────────────────────────────");
    println!(
        "{:<18} {:>7} {:>16} {:>10} {:>8} {:>8} {:>8} {:>12}",
        "Persona", "N", "Top-1 (95% CI)", "Top-3", "MRR", "OOV %", "KSPC", "Latency p95"
    );
    println!("─────────────────────────────────────────────────────────────────────────────────────────────");

    let persona_order = ["natural", "lazy", "vowel_drop_p30", "vowel_drop_p60", "avro_strict"];
    for name in &persona_order {
        if let Some(m) = report_personas.get(*name) {
            let ci_str = format!("{:.1}% [{:.1}-{:.1}]", m.top1, m.top1_ci95[0], m.top1_ci95[1]);
            println!(
                "{:<18} {:>7} {:>16} {:>9.1}% {:>8.4} {:>7.1}% {:>8.2} {:>10.0}µs",
                name, m.n, ci_str, m.top3, m.mrr, m.oov_rate, m.kspc.mean, m.latency_us.p95
            );
        }
    }
    println!("─────────────────────────────────────────────────────────────────────────────────────────────");

    let report = MultiPersonaReport {
        run: RunMeta {
            git_sha,
            seed,
            split: "test".to_string(),
            canonical: "nfc+khanda".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        },
        personas: report_personas,
        worst_cases: all_worst_cases,
    };

    let baseline_path = Path::new("data/eval/baseline.json");
    if let Ok(json) = serde_json::to_string_pretty(&report) {
        let _ = std::fs::write(baseline_path, json);
        println!("\n💾 Saved multi-persona baseline report to {:?}", baseline_path);
    }
}
