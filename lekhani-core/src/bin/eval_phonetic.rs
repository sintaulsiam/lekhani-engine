//! Phonetic Transliteration Evaluation Harness
//!
//! Evaluates Lekhani's phonetic transliteration pipeline against a gold standard
//! dataset, reporting Top-1 and Top-3 accuracy across different linguistic categories.

use hashbrown::HashMap;
use lekhani_core::phonetic::PhoneticSuggestion;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryResult {
    pub total: usize,
    pub top1_correct: usize,
    pub top3_correct: usize,
    pub top1_accuracy: f64,
    pub top3_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub timestamp: String,
    pub total_cases: usize,
    pub top1_accuracy: f64,
    pub top3_accuracy: f64,
    pub eval_duration_ms: f64,
    pub avg_latency_us: f64,
    pub categories: HashMap<String, CategoryResult>,
    pub top_disagreements: Vec<DisagreementItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisagreementItem {
    pub input: String,
    pub expected: String,
    pub got_top1: String,
    pub category: String,
}

struct TestCase {
    input: String,
    expected: String,
    category: String,
}

fn load_gold_set<P: AsRef<Path>>(path: P) -> Result<Vec<TestCase>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    let mut cases = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split('\t').collect();
        if parts.len() >= 3 {
            cases.push(TestCase {
                input: parts[0].trim().to_string(),
                expected: parts[1].trim().to_string(),
                category: parts[2].trim().to_string(),
            });
        }
    }
    Ok(cases)
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

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       🎯 Lekhani Phonetic Transliteration Evaluation Harness     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let gold_candidates = [
        PathBuf::from("data/eval/phonetic_gold.tsv"),
        PathBuf::from("../data/eval/phonetic_gold.tsv"),
        PathBuf::from("../../data/eval/phonetic_gold.tsv"),
    ];
    let gold_path = gold_candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .unwrap_or_else(|| PathBuf::from("data/eval/phonetic_gold.tsv"));

    let test_cases = match load_gold_set(&gold_path) {
        Ok(cases) => cases,
        Err(e) => {
            eprintln!("❌ Failed to load gold set from {:?}: {}", gold_path, e);
            std::process::exit(1);
        }
    };

    println!("Loaded {} test pairs from {:?}", test_cases.len(), gold_path);
    let mut sugg = setup_engine();
    let empty_memory = HashMap::new();

    let mut cat_totals: HashMap<String, usize> = HashMap::new();
    let mut cat_top1: HashMap<String, usize> = HashMap::new();
    let mut cat_top3: HashMap<String, usize> = HashMap::new();
    let mut disagreements: Vec<DisagreementItem> = Vec::new();

    let total_cases = test_cases.len();
    let start_time = Instant::now();

    for case in &test_cases {
        *cat_totals.entry(case.category.clone()).or_insert(0) += 1;
        let (candidates, _) = sugg.suggest(&case.input, true, true, &empty_memory);

        let top1_match = candidates.first().map(|s| s == &case.expected).unwrap_or(false);
        let top3_match = candidates
            .iter()
            .take(3)
            .any(|s| s == &case.expected);

        if top1_match {
            *cat_top1.entry(case.category.clone()).or_insert(0) += 1;
        } else {
            let got_top1 = candidates.first().cloned().unwrap_or_else(|| "<EMPTY>".to_string());
            disagreements.push(DisagreementItem {
                input: case.input.clone(),
                expected: case.expected.clone(),
                got_top1,
                category: case.category.clone(),
            });
        }

        if top3_match {
            *cat_top3.entry(case.category.clone()).or_insert(0) += 1;
        }
    }

    let elapsed = start_time.elapsed();
    let total_top1: usize = cat_top1.values().sum();
    let total_top3: usize = cat_top3.values().sum();
    let overall_top1_pct = (total_top1 as f64 / total_cases as f64) * 100.0;
    let overall_top3_pct = (total_top3 as f64 / total_cases as f64) * 100.0;
    let avg_latency_us = (elapsed.as_micros() as f64) / (total_cases as f64);

    println!("\n📊 Evaluation Summary:");
    println!("──────────────────────────────────────────────────────────────────");
    println!("  Total Test Cases:    {}", total_cases);
    println!("  Overall Top-1 Acc:   {:.2}% ({}/{})", overall_top1_pct, total_top1, total_cases);
    println!("  Overall Top-3 Acc:   {:.2}% ({}/{})", overall_top3_pct, total_top3, total_cases);
    println!("  Evaluation Time:     {:.2} ms ({:.2} µs/term)", elapsed.as_secs_f64() * 1000.0, avg_latency_us);
    println!("──────────────────────────────────────────────────────────────────");

    println!("\n📋 Breakdown by Category:");
    println!("{:<22} {:>8} {:>14} {:>14}", "Category", "Pairs", "Top-1 Acc", "Top-3 Acc");
    println!("──────────────────────────────────────────────────────────────────");

    let mut sorted_cats: Vec<String> = cat_totals.keys().cloned().collect();
    sorted_cats.sort();

    let mut cat_results = HashMap::new();
    for cat in &sorted_cats {
        let count = cat_totals[cat];
        let t1 = cat_top1.get(cat).copied().unwrap_or(0);
        let t3 = cat_top3.get(cat).copied().unwrap_or(0);
        let t1_pct = (t1 as f64 / count as f64) * 100.0;
        let t3_pct = (t3 as f64 / count as f64) * 100.0;
        println!("{:<22} {:>8} {:>13.1}% {:>13.1}%", cat, count, t1_pct, t3_pct);

        cat_results.insert(
            cat.clone(),
            CategoryResult {
                total: count,
                top1_correct: t1,
                top3_correct: t3,
                top1_accuracy: t1_pct,
                top3_accuracy: t3_pct,
            },
        );
    }
    println!("──────────────────────────────────────────────────────────────────");

    if !disagreements.is_empty() {
        println!("\n🔍 Top Disagreements (Sample {} of {}):", disagreements.len().min(10), disagreements.len());
        for item in disagreements.iter().take(10) {
            println!(
                "  [{}] {:<14} ➔ Expected: {:<12} Got: {}",
                item.category, item.input, item.expected, item.got_top1
            );
        }
    }

    // Save baseline JSON
    let report = EvalReport {
        timestamp: chrono::Utc::now().to_rfc3339(),
        total_cases,
        top1_accuracy: overall_top1_pct,
        top3_accuracy: overall_top3_pct,
        eval_duration_ms: elapsed.as_secs_f64() * 1000.0,
        avg_latency_us,
        categories: cat_results,
        top_disagreements: disagreements,
    };

    let baseline_candidates = [
        PathBuf::from("data/eval/baseline.json"),
        PathBuf::from("../data/eval/baseline.json"),
    ];
    let baseline_path = baseline_candidates
        .iter()
        .find(|p| p.parent().map(|d| d.exists()).unwrap_or(false))
        .cloned()
        .unwrap_or_else(|| PathBuf::from("data/eval/baseline.json"));

    if let Ok(json) = serde_json::to_string_pretty(&report) {
        let _ = std::fs::write(&baseline_path, json);
        println!("\n💾 Baseline report written to {:?}", baseline_path);
    }
}
