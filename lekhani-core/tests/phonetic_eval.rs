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
            test_cases.push((parts[0].trim().to_string(), parts[1].trim().to_string(), parts[2].trim().to_string()));
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
