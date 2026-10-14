//! Diagnostic and Ablation Analysis on Held-Out Test Set
//!
//! Investigates:
//! 1. Component Ablation: Full vs No-Overrides vs No-Dictionary vs Raw-Avro-Only
//! 2. Trace of 20 Seen-Word Failures: Where does the target sit?
//! 3. Avro-Strict Error Analysis: Why is Top-1 77.4% instead of 100%?

use hashbrown::{HashMap, HashSet};
use lekhani_core::chars::canonicalize;
use lekhani_core::personas::transform_avro_strict;
use lekhani_core::phonetic::PhoneticSuggestion;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

fn setup_engine() -> PhoneticSuggestion {
    let mut sugg = PhoneticSuggestion::new();
    let layout_candidates = [
        PathBuf::from("data/layouts/avrophonetic.json"),
        PathBuf::from("../data/layouts/avrophonetic.json"),
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
    let test_path = Path::new("/home/smsiam/.gemini/antigravity-ide/brain/6e8f5951-b148-4374-87c2-2e54e03df641/scratch/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.test.tsv");
    let file = File::open(test_path).expect("Failed to open test.tsv");
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for l in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = l.split('\t').collect();
        if parts.len() >= 2 {
            let bn = canonicalize(parts[0].trim());
            let latin = parts[1].trim().to_string();
            if !bn.is_empty() && !latin.is_empty() {
                pairs.push((bn, latin));
            }
        }
    }
    pairs
}

fn load_dict_words() -> HashSet<String> {
    let dict_path = Path::new("data/dictionaries/dictionary.json");
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

fn main() {
    let test_pairs = load_test_pairs();
    let dict_words = load_dict_words();
    let raw_parser = lekhani_parser::default_avro_parser();
    let empty_memory = HashMap::new();

    println!(
        "Loaded {} test pairs, {} dict words.\n",
        test_pairs.len(),
        dict_words.len()
    );

    // ────────────────────────────────────────────────────────────────
    // PART 1: Ablation on Natural Persona
    // ────────────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("   🔬 PART 1: ABLATION STUDY ON NATURAL PERSONA (Held-Out Test)");
    println!("══════════════════════════════════════════════════════════════════");

    // Condition A: Full Engine (Default)
    let mut sugg_full = setup_engine();
    let mut full_top1 = 0;
    let mut full_top3 = 0;
    let mut full_seen_top1 = 0;
    let mut full_unseen_top1 = 0;

    // Condition B: No Overrides (use_dictionary = true, overrides = empty)
    let mut sugg_no_ov = setup_engine();
    let empty_ov_map = hashbrown::HashMap::new();
    sugg_no_ov.database.overrides =
        lekhani_core::phonetic::overrides::ZeroCopyOverrides::from_hashbrown_map(&empty_ov_map);
    let mut no_ov_top1 = 0;
    let mut no_ov_top3 = 0;

    // Condition C: No Dictionary (use_dictionary = false, overrides active)
    let mut nodict_top1 = 0;
    let mut nodict_top3 = 0;

    // Condition D: No Overrides & No Dictionary
    let mut sugg_no_both = setup_engine();
    sugg_no_both.database.overrides =
        lekhani_core::phonetic::overrides::ZeroCopyOverrides::from_hashbrown_map(&empty_ov_map);
    let mut no_both_top1 = 0;
    let mut no_both_top3 = 0;

    // Condition E: Raw Avro Parser Only (convert_phonetic only, no engine)
    let mut raw_avro_top1 = 0;

    let total = test_pairs.len();
    let mut seen_total = 0;
    let mut unseen_total = 0;

    let mut len_seen_total = [0usize; 3];
    let mut len_seen_top1 = [0usize; 3];
    let mut len_unseen_total = [0usize; 3];
    let mut len_unseen_top1 = [0usize; 3];

    for (bn_target, latin) in &test_pairs {
        let is_seen = dict_words.contains(bn_target);
        if is_seen {
            seen_total += 1;
        } else {
            unseen_total += 1;
        }

        // Track length-stratified Seen vs Unseen
        let lat_len = latin.chars().count();
        let bucket_idx = if lat_len <= 3 {
            0
        } else if lat_len <= 6 {
            1
        } else {
            2
        };

        if is_seen {
            len_seen_total[bucket_idx] += 1;
        } else {
            len_unseen_total[bucket_idx] += 1;
        }

        // A. Full Engine
        let (cands_full, _) = sugg_full.suggest(latin, true, true, &empty_memory);
        let cands_full_canon: Vec<String> = cands_full.iter().map(|s| canonicalize(s)).collect();
        if cands_full_canon.first() == Some(bn_target) {
            full_top1 += 1;
            if is_seen {
                full_seen_top1 += 1;
                len_seen_top1[bucket_idx] += 1;
            } else {
                full_unseen_top1 += 1;
                len_unseen_top1[bucket_idx] += 1;
            }
        }
        if cands_full_canon.iter().take(3).any(|c| c == bn_target) {
            full_top3 += 1;
        }

        // B. No Overrides
        let (cands_no_ov, _) = sugg_no_ov.suggest(latin, true, true, &empty_memory);
        let cands_no_ov_canon: Vec<String> = cands_no_ov.iter().map(|s| canonicalize(s)).collect();
        if cands_no_ov_canon.first() == Some(bn_target) {
            no_ov_top1 += 1;
        }
        if cands_no_ov_canon.iter().take(3).any(|c| c == bn_target) {
            no_ov_top3 += 1;
        }

        // C. No Dictionary
        let (cands_nodict, _) = sugg_full.suggest(latin, true, false, &empty_memory);
        let cands_nodict_canon: Vec<String> =
            cands_nodict.iter().map(|s| canonicalize(s)).collect();
        if cands_nodict_canon.first() == Some(bn_target) {
            nodict_top1 += 1;
        }
        if cands_nodict_canon.iter().take(3).any(|c| c == bn_target) {
            nodict_top3 += 1;
        }

        // D. No Overrides & No Dictionary
        let (cands_no_both, _) = sugg_no_both.suggest(latin, true, false, &empty_memory);
        let cands_no_both_canon: Vec<String> =
            cands_no_both.iter().map(|s| canonicalize(s)).collect();
        if cands_no_both_canon.first() == Some(bn_target) {
            no_both_top1 += 1;
        }
        if cands_no_both_canon.iter().take(3).any(|c| c == bn_target) {
            no_both_top3 += 1;
        }

        // E. Raw Avro Parser Only
        let raw_conv = canonicalize(&raw_parser.convert(latin));
        if raw_conv == *bn_target {
            raw_avro_top1 += 1;
        }
    }

    println!(
        "{:<36} {:>10} {:>10}",
        "Ablation Configuration", "Top-1 Acc", "Top-3 Acc"
    );
    println!("──────────────────────────────────────────────────────────────────────");
    println!(
        "{:<36} {:>9.2}% {:>9.2}%",
        "1. Full Engine (Default)",
        full_top1 as f64 / total as f64 * 100.0,
        full_top3 as f64 / total as f64 * 100.0
    );
    println!(
        "{:<36} {:>9.2}% {:>9.2}%",
        "2. No Overrides (Dict + Raw)",
        no_ov_top1 as f64 / total as f64 * 100.0,
        no_ov_top3 as f64 / total as f64 * 100.0
    );
    println!(
        "{:<36} {:>9.2}% {:>9.2}%",
        "3. No Dictionary (Overrides + Raw)",
        nodict_top1 as f64 / total as f64 * 100.0,
        nodict_top3 as f64 / total as f64 * 100.0
    );
    println!(
        "{:<36} {:>9.2}% {:>9.2}%",
        "4. No Overrides & No Dictionary",
        no_both_top1 as f64 / total as f64 * 100.0,
        no_both_top3 as f64 / total as f64 * 100.0
    );
    println!(
        "{:<36} {:>9.2}% {:>10}",
        "5. Raw Avro Parser Only",
        raw_avro_top1 as f64 / total as f64 * 100.0,
        "N/A"
    );
    println!("──────────────────────────────────────────────────────────────────────");
    println!(
        "Overall Seen vs Unseen: Seen Top-1 = {:.2}% ({}/{}) | Unseen Top-1 = {:.2}% ({}/{})\n",
        full_seen_top1 as f64 / seen_total as f64 * 100.0,
        full_seen_top1,
        seen_total,
        full_unseen_top1 as f64 / unseen_total as f64 * 100.0,
        full_unseen_top1,
        unseen_total
    );

    println!("📊 Stratified Seen vs Unseen Top-1 by Latin Character Length:");
    println!("──────────────────────────────────────────────────────────────────────");
    println!(
        "{:<14} {:<24} {:<24}",
        "Length Bucket", "Seen Words Top-1", "Unseen Words Top-1"
    );
    println!("──────────────────────────────────────────────────────────────────────");
    let bucket_names = ["1-3 chars", "4-6 chars", "7+ chars"];
    for i in 0..3 {
        let s_acc = if len_seen_total[i] > 0 {
            len_seen_top1[i] as f64 / len_seen_total[i] as f64 * 100.0
        } else {
            0.0
        };
        let u_acc = if len_unseen_total[i] > 0 {
            len_unseen_top1[i] as f64 / len_unseen_total[i] as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "{:<14} {:>6.2}% ({:>4}/{:<4})      {:>6.2}% ({:>4}/{:<4})",
            bucket_names[i],
            s_acc,
            len_seen_top1[i],
            len_seen_total[i],
            u_acc,
            len_unseen_top1[i],
            len_unseen_total[i]
        );
    }
    println!("──────────────────────────────────────────────────────────────────────\n");

    // ────────────────────────────────────────────────────────────────
    // PART 2: Trace 20 Seen-Word Failures (Across 20 DISTINCT Targets)
    // ────────────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("   🔍 PART 2: TRACE OF 20 SEEN-WORD FAILURES (20 DISTINCT Targets)");
    println!("══════════════════════════════════════════════════════════════════");

    let mut traced_distinct_targets = HashSet::new();
    let mut traced = 0;
    for (bn_target, latin) in &test_pairs {
        if !dict_words.contains(bn_target) {
            continue;
        }
        if traced_distinct_targets.contains(bn_target) {
            continue;
        }
        let (cands, _) = sugg_full.suggest(latin, true, true, &empty_memory);
        let cands_canon: Vec<String> = cands.iter().map(|s| canonicalize(s)).collect();
        if cands_canon.first() == Some(bn_target) {
            continue; // Not a failure
        }

        traced_distinct_targets.insert(bn_target.clone());
        let rank = cands_canon.iter().position(|c| c == bn_target);
        traced += 1;
        println!(
            "Case #{:02}: Input: {:<16} Target: {:<14} (Seen: YES)",
            traced, latin, bn_target
        );
        match rank {
            Some(r) => println!(
                "  ➔ Target found at Rank {} (Slot {}) in candidate list!",
                r, r
            ),
            None => println!("  ➔ Target NOT in candidate list at any rank (REACHABILITY FAILURE)"),
        }
        println!(
            "  ➔ Returned Candidates: {:?}",
            cands_canon.iter().take(5).collect::<Vec<_>>()
        );
        println!("  ─────────────────────────────────────────────────────────────");

        if traced >= 20 {
            break;
        }
    }

    // ────────────────────────────────────────────────────────────────
    // PART 3: Avro-Strict Error Analysis
    // ────────────────────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("   🔍 PART 3: AVRO-STRICT ERROR ANALYSIS (Why is ceiling 77.4%?)");
    println!("══════════════════════════════════════════════════════════════════");

    let mut avro_strict_diffs = Vec::new();
    for (bn_target, _) in &test_pairs {
        let avro_keystrokes = transform_avro_strict(bn_target);
        let raw_conv = canonicalize(&raw_parser.convert(&avro_keystrokes));
        let (cands, _) = sugg_full.suggest(&avro_keystrokes, true, true, &empty_memory);
        let cands_canon: Vec<String> = cands.iter().map(|s| canonicalize(s)).collect();
        let top1_cand = cands_canon.first().cloned().unwrap_or_default();

        let raw_matches_target = raw_conv == *bn_target;
        let engine_matches_target = top1_cand == *bn_target;

        if !engine_matches_target {
            avro_strict_diffs.push((
                bn_target.clone(),
                avro_keystrokes,
                raw_conv,
                top1_cand,
                raw_matches_target,
            ));
        }
    }

    let mut mapper_flaws = 0;
    let mut engine_displacement = 0;

    for (target, keystrokes, raw_conv, engine_top, raw_matches) in avro_strict_diffs.iter().take(25)
    {
        if !raw_matches {
            mapper_flaws += 1;
            println!(
                "  [MAPPER FLAW] Target: {:<10} Keystrokes: {:<14} Parser Got: {:<10} (Mapper emitted wrong Avro string)",
                target, keystrokes, raw_conv
            );
        } else {
            engine_displacement += 1;
            println!(
                "  [DISPLACEMENT] Target: {:<10} Keystrokes: {:<14} Parser: {:<10} But Engine Got: {:<10} (Displaced!)",
                target, keystrokes, raw_conv, engine_top
            );
        }
    }
    println!(
        "  Summary (first 25 sampled): {} mapper flaws, {} engine displacements",
        mapper_flaws, engine_displacement
    );

    let total_avro_fails = avro_strict_diffs.len();
    let total_mapper = avro_strict_diffs.iter().filter(|d| !d.4).count();
    let total_displaced = avro_strict_diffs.iter().filter(|d| d.4).count();

    println!(
        "\nAvro-Strict Failure Triage Total ({} failures / {} items):",
        total_avro_fails, total
    );
    println!(
        "  - Mapper Gaps (Raw parser doesn't produce target from keystrokes): {} ({:.1}%)",
        total_mapper,
        total_mapper as f64 / total_avro_fails as f64 * 100.0
    );
    println!("  - Engine Displacement (Raw parser DOES produce target, but Engine Slot 0 overrides it): {} ({:.1}%)",
        total_displaced, total_displaced as f64 / total_avro_fails as f64 * 100.0);

    // ────────────────────────────────────────────────────────────────
    // PART 4: Extract 100 Failures of Natural Persona for Audit
    // ────────────────────────────────────────────────────────────────
    #[derive(serde::Serialize)]
    struct FailureCase {
        id: usize,
        input: String,
        target: String,
        candidates: Vec<String>,
        target_rank: Option<usize>,
        in_dictionary: bool,
    }

    let mut failures_by_target: HashMap<String, Vec<FailureCase>> = HashMap::new();
    let mut item_id = 0;
    for (bn_target, latin) in &test_pairs {
        let (cands, _) = sugg_full.suggest(latin, true, true, &empty_memory);
        let cands_canon: Vec<String> = cands.iter().map(|s| canonicalize(s)).collect();
        if cands_canon.first() != Some(bn_target) {
            item_id += 1;
            let rank = cands_canon.iter().position(|c| c == bn_target);
            failures_by_target
                .entry(bn_target.clone())
                .or_default()
                .push(FailureCase {
                    id: item_id,
                    input: latin.clone(),
                    target: bn_target.clone(),
                    candidates: cands_canon.into_iter().take(5).collect(),
                    target_rank: rank,
                    in_dictionary: dict_words.contains(bn_target),
                });
        }
    }

    let mut distinct_failed_targets: Vec<String> = failures_by_target.keys().cloned().collect();
    distinct_failed_targets.sort(); // Lexicographical sort to cover the entire alphabet
    let num_distinct = distinct_failed_targets.len();
    println!(
        "Found failures across {} distinct target Bengali words.",
        num_distinct
    );

    // Uniformly sample 100 distinct target words across the sorted vocabulary
    let sample_size = 100.min(num_distinct);
    let step = num_distinct as f64 / sample_size as f64;
    let mut sampled_failures = Vec::with_capacity(sample_size);

    for i in 0..sample_size {
        let target_idx = (i as f64 * step) as usize;
        let target_word = &distinct_failed_targets[target_idx];
        if let Some(cases) = failures_by_target.get(target_word) {
            if let Some(case) = cases.first() {
                let c = FailureCase {
                    id: i + 1,
                    input: case.input.clone(),
                    target: case.target.clone(),
                    candidates: case.candidates.clone(),
                    target_rank: case.target_rank,
                    in_dictionary: case.in_dictionary,
                };
                sampled_failures.push(c);
            }
        }
    }

    let json_out = serde_json::to_string_pretty(&sampled_failures).expect("Serialize failures");
    std::fs::write("data/eval/failures_100_stratified.json", json_out)
        .expect("Write failures_100_stratified.json");
    println!("✅ Dumped {} Natural persona failures across {} DISTINCT targets to data/eval/failures_100_stratified.json for hand audit.", sampled_failures.len(), sample_size);
}
