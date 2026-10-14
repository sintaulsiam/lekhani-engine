use hashbrown::HashMap;
use lekhani_core::phonetic::PhoneticSuggestion;
use std::path::PathBuf;

#[test]
fn test_phonetic_eval_baseline_guard() {
    let gold_candidates = [
        PathBuf::from("data/eval/phonetic_gold.tsv"),
        PathBuf::from("../data/eval/phonetic_gold.tsv"),
        PathBuf::from("../../data/eval/phonetic_gold.tsv"),
    ];
    let gold_path = match gold_candidates.iter().find(|p| p.exists()) {
        Some(p) => p.clone(),
        None => return, // Skip if run from outside workspace
    };

    let content = std::fs::read_to_string(&gold_path).expect("Read phonetic_gold.tsv failed");
    let mut test_cases = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split('\t').collect();
        if parts.len() >= 3 {
            test_cases.push((
                parts[0].trim().to_string(),
                parts[1].trim().to_string(),
                parts[2].trim().to_string(),
            ));
        }
    }

    let mut sugg = PhoneticSuggestion::new();
    let layout_candidates = [
        PathBuf::from("data/layouts/avrophonetic.json"),
        PathBuf::from("../data/layouts/avrophonetic.json"),
        PathBuf::from("../../data/layouts/avrophonetic.json"),
    ];
    for p in &layout_candidates {
        if p.exists() {
            if let Ok(c) = std::fs::read_to_string(p) {
                if let Ok(json) = serde_json::from_str(&c) {
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

    let empty_memory = HashMap::new();
    let total = test_cases.len();
    let mut top1_correct = 0;
    let mut top3_correct = 0;

    for (inp, exp, _cat) in &test_cases {
        let (cands, _) = sugg.suggest(inp, true, true, &empty_memory);
        if cands.first().map(|s| s == exp).unwrap_or(false) {
            top1_correct += 1;
        } else {
            eprintln!(
                "MISMATCH: [{inp}] expected='{exp}', top_cands={:?}",
                cands.iter().take(3).collect::<Vec<_>>()
            );
        }
        if cands.iter().take(3).any(|s| s == exp) {
            top3_correct += 1;
        }
    }

    let top1_acc = (top1_correct as f64 / total as f64) * 100.0;
    let top3_acc = (top3_correct as f64 / total as f64) * 100.0;

    println!(
        "\nCI Phonetic Regression Guard: Top-1 = {:.2}% ({}/{}), Top-3 = {:.2}% ({}/{})",
        top1_acc, top1_correct, total, top3_acc, top3_correct, total
    );

    // Baseline guard: Top-1 must stay >= 83.0% and Top-3 >= 88.0%
    assert!(
        top1_acc >= 83.0,
        "Top-1 accuracy regressed to {:.2}% (below 83.0% baseline)",
        top1_acc
    );
    assert!(
        top3_acc >= 88.0,
        "Top-3 accuracy regressed to {:.2}% (below 88.0% baseline)",
        top3_acc
    );
}

#[derive(Debug)]
pub struct GateResult {
    pub top1: f64,
    pub p95_latency_us: f64,
}

pub fn check_dictionary_hash(actual: &str, expected: &str) -> Result<(), String> {
    if actual.trim().eq_ignore_ascii_case(expected.trim()) {
        Ok(())
    } else {
        Err(format!(
            "GATE FAILURE: Dictionary SHA-256 mismatch!\n  Expected: {}\n  Actual:   {}",
            expected, actual
        ))
    }
}

pub fn check_baseline_gate(
    actual: &GateResult,
    min_top1_threshold: f64,
    max_p95_lat_threshold: f64,
) -> Result<(), String> {
    if actual.top1 < min_top1_threshold {
        return Err(format!(
            "GATE FAILURE: Top-1 accuracy {:.2}% fell below baseline lower bound {:.2}%",
            actual.top1, min_top1_threshold
        ));
    }
    if actual.p95_latency_us > max_p95_lat_threshold {
        return Err(format!(
            "GATE FAILURE: p95 latency {:.1} µs exceeded baseline p95+20% budget {:.1} µs",
            actual.p95_latency_us, max_p95_lat_threshold
        ));
    }
    Ok(())
}

#[test]
fn test_regression_gate_fails_on_deliberate_degradation() {
    // Deliberate accuracy degradation: below 29.57%
    let degraded_acc = GateResult {
        top1: 27.50,
        p95_latency_us: 1200.0,
    };
    let res_acc = check_baseline_gate(&degraded_acc, 29.57, 1942.0);
    assert!(
        res_acc.is_err(),
        "Gate must fail when accuracy drops below baseline lower bound"
    );
    assert!(res_acc
        .unwrap_err()
        .contains("Top-1 accuracy 27.50% fell below baseline"));

    // Deliberate latency degradation: exceeding p95 + 20%
    let degraded_lat = GateResult {
        top1: 31.00,
        p95_latency_us: 2200.0,
    };
    let res_lat = check_baseline_gate(&degraded_lat, 29.57, 1942.0);
    assert!(
        res_lat.is_err(),
        "Gate must fail when p95 latency exceeds +20% threshold"
    );
    assert!(res_lat
        .unwrap_err()
        .contains("p95 latency 2200.0 µs exceeded"));
}

#[test]
fn test_regression_gate_fails_on_tampered_dictionary_hash() {
    let original = "8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb";
    let tampered = "8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdc"; // tampered last byte
    let res = check_dictionary_hash(tampered, original);
    assert!(
        res.is_err(),
        "Gate must fail when dictionary SHA-256 differs"
    );
    assert!(res.unwrap_err().contains("Dictionary SHA-256 mismatch"));
}

#[test]
fn test_dakshina_baseline_gate_against_json() {
    let baseline_candidates = [
        PathBuf::from("data/eval/baseline.json"),
        PathBuf::from("../data/eval/baseline.json"),
        PathBuf::from("../../data/eval/baseline.json"),
    ];
    let baseline_path = match baseline_candidates.iter().find(|p| p.exists()) {
        Some(p) => p.clone(),
        None => return,
    };

    let test_candidates = [
        PathBuf::from("data/eval/bn.translit.sampled.test.tsv"),
        PathBuf::from("../data/eval/bn.translit.sampled.test.tsv"),
        PathBuf::from("../../data/eval/bn.translit.sampled.test.tsv"),
        PathBuf::from("/home/smsiam/.gemini/antigravity-ide/brain/6e8f5951-b148-4374-87c2-2e54e03df641/scratch/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.test.tsv"),
    ];
    let test_path = match test_candidates.iter().find(|p| p.exists()) {
        Some(p) => p.clone(),
        None => return,
    };

    let baseline_str =
        std::fs::read_to_string(&baseline_path).expect("Failed to read baseline.json");
    let baseline_json: serde_json::Value = serde_json::from_str(&baseline_str).expect("Valid JSON");

    let natural = if baseline_json.get("test").is_some() {
        &baseline_json["test"]["personas"]["natural"]
    } else {
        &baseline_json["personas"]["natural"]
    };
    let min_top1 = natural["top1_ci95_cluster"][0]
        .as_f64()
        .expect("ci lower bound");
    let baseline_p95 = natural["latency_us"]["p95"].as_f64().expect("p95 latency");
    let max_p95_lat = if cfg!(debug_assertions) {
        baseline_p95 * 8.0 // unoptimized debug profile allowance
    } else {
        baseline_p95 * 1.20 // strict release budget (+20%)
    };

    let test_file = std::fs::File::open(&test_path).expect("Open test.tsv");
    let reader = std::io::BufReader::new(test_file);
    let mut pairs = Vec::new();
    for l in std::io::BufRead::lines(reader).map_while(Result::ok) {
        let parts: Vec<&str> = l.split('\t').collect();
        if parts.len() >= 2 {
            pairs.push((parts[0].trim().to_string(), parts[1].trim().to_string()));
        }
    }

    // In debug mode, sample 500 items for rapid CI; in release mode run full test set
    let eval_slice: &[(String, String)] = if cfg!(debug_assertions) && pairs.len() > 500 {
        &pairs[..500]
    } else {
        &pairs[..]
    };

    let mut sugg = PhoneticSuggestion::new();
    let layout_candidates = [
        PathBuf::from("data/layouts/avrophonetic.json"),
        PathBuf::from("../data/layouts/avrophonetic.json"),
        PathBuf::from("../../data/layouts/avrophonetic.json"),
    ];
    for p in &layout_candidates {
        if p.exists() {
            if let Ok(c) = std::fs::read_to_string(p) {
                if let Ok(json) = serde_json::from_str(&c) {
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

    // 1. Verify Dictionary SHA-256 Hash against baseline.json
    if let Some(expected_hash) = baseline_json.get("run").and_then(|r| r.get("dictionary_hash")).and_then(|h| h.as_str()) {
        let actual_hash = sugg.database.dictionary_hash.as_deref().unwrap_or("");
        let hash_check = check_dictionary_hash(actual_hash, expected_hash);
        assert!(
            hash_check.is_ok(),
            "Dakshina CI dictionary hash gate failed: {:?}",
            hash_check.err()
        );
    }

    let empty_mem = HashMap::new();
    let mut correct = 0;
    let mut latencies_us = Vec::with_capacity(eval_slice.len());

    for (target, input) in eval_slice {
        let t0 = std::time::Instant::now();
        let (cands, _) = sugg.suggest(input, true, true, &empty_mem);
        let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
        latencies_us.push(elapsed_us);

        let target_norm = lekhani_core::chars::canonicalize(target);
        if cands.first().map(|c| c == &target_norm).unwrap_or(false) {
            correct += 1;
        }
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95_idx = ((latencies_us.len() as f64) * 0.95) as usize;
    let actual_p95 = latencies_us.get(p95_idx).copied().unwrap_or(0.0);
    let actual_top1 = (correct as f64 / eval_slice.len() as f64) * 100.0;

    println!(
        "\nDakshina Baseline Gate Verification (N={}): Top-1 = {:.2}% (threshold >= {:.2}%), p95 = {:.1}µs (threshold <= {:.1}µs)",
        eval_slice.len(), actual_top1, min_top1, actual_p95, max_p95_lat
    );

    let actual_result = GateResult {
        top1: actual_top1,
        p95_latency_us: actual_p95,
    };
    let gate_eval = check_baseline_gate(&actual_result, min_top1, max_p95_lat);
    assert!(
        gate_eval.is_ok(),
        "Dakshina CI baseline gate failed: {:?}",
        gate_eval.err()
    );
}
