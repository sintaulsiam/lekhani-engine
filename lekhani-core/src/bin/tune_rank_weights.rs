#![allow(clippy::all)]
//! Offline Margin Perceptron / RankWeights tuner on Target-Disjoint Dev-A -> Dev-B.

use hashbrown::HashMap;
use lekhani_core::chars::canonicalize;
use lekhani_core::phonetic::ranking::{RankFeatures, RankWeights};
use lekhani_core::phonetic::PhoneticSuggestion;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

fn load_pairs(path: &str) -> Vec<(String, String)> {
    let file = File::open(path).expect("open tsv");
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for l in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = l.split('\t').collect();
        if parts.len() >= 2 {
            pairs.push((canonicalize(parts[0].trim()), parts[1].trim().to_string()));
        }
    }
    pairs
}

fn eval_dataset(sugg: &mut PhoneticSuggestion, pairs: &[(String, String)]) -> (usize, usize) {
    let empty_memory: HashMap<String, String> = HashMap::new();
    let mut hits = 0usize;
    for (target, input) in pairs {
        let (cands, _) = sugg.suggest(input, true, true, &empty_memory);
        let top1 = cands.first().map(|s| canonicalize(s)).unwrap_or_default();
        if &top1 == target {
            hits += 1;
        }
    }
    (hits, pairs.len())
}

fn update_all_weights(w: &mut RankWeights, chosen: &RankFeatures, rejected: &RankFeatures, lr: f32) {
    w.is_in_dict = (w.is_in_dict + lr * (chosen.is_in_dict - rejected.is_in_dict)).clamp(200.0, 3000.0);
    w.normalized_freq = (w.normalized_freq + lr * (chosen.normalized_freq - rejected.normalized_freq)).clamp(200.0, 2500.0);
    w.is_high_freq = (w.is_high_freq + lr * (chosen.is_high_freq - rejected.is_high_freq)).clamp(0.0, 2000.0);
    w.lm_score = (w.lm_score + lr * (chosen.lm_score - rejected.lm_score)).clamp(3000.0, 9000.0);
    w.is_exact_phonetic = (w.is_exact_phonetic + lr * (chosen.is_exact_phonetic - rejected.is_exact_phonetic)).clamp(3000.0, 9000.0);
    w.phonetic_similarity = (w.phonetic_similarity + lr * (chosen.phonetic_similarity - rejected.phonetic_similarity)).clamp(150.0, 1500.0);
    w.avro_native_anchor = (w.avro_native_anchor + lr * (chosen.avro_native_anchor - rejected.avro_native_anchor)).clamp(800.0, 3500.0);
    w.length_penalty = (w.length_penalty + lr * (chosen.length_penalty - rejected.length_penalty)).clamp(150.0, 1000.0);
    w.source_weight = (w.source_weight + lr * (chosen.source_weight - rejected.source_weight)).clamp(1500.0, 5000.0);
    w.clitic_alignment = (w.clitic_alignment + lr * (chosen.clitic_alignment - rejected.clitic_alignment)).clamp(2000.0, 5000.0);
    w.intent_modifier_boost = (w.intent_modifier_boost + lr * (chosen.intent_modifier_boost - rejected.intent_modifier_boost)).clamp(2500.0, 7000.0);
}

fn main() {
    let mut sugg = PhoneticSuggestion::new();
    let dict_dir = PathBuf::from("data/dictionaries");
    sugg.load_from_dir_with_dict(&dict_dir, "dictionary.json")
        .expect("load dict");
    if let Ok(content) = std::fs::read_to_string("data/layouts/avrophonetic.json") {
        if let Ok(json) = serde_json::from_str(&content) {
            sugg.set_layout(&json);
        }
    }

    let dev_a = load_pairs("data/eval/bn.translit.sampled.dev_a.tsv");
    let dev_b = load_pairs("data/eval/bn.translit.sampled.dev_b.tsv");
    let empty_memory: HashMap<String, String> = HashMap::new();

    // Collect (chosen_feats, rejected_feats) from dev_a where target is in cands
    let mut all_samples: Vec<(usize, Vec<RankFeatures>)> = Vec::with_capacity(dev_a.len());
    let mut base_a_hits = 0usize;
    for (target, input) in &dev_a {
        let (cands, _) = sugg.suggest(input, true, true, &empty_memory);
        let feats = sugg.get_last_computed_features().to_vec();
        if cands.first().map(|s| canonicalize(s)).as_ref() == Some(target) {
            base_a_hits += 1;
        }
        if feats.len() == cands.len() && cands.len() >= 2 {
            if let Some(pos) = cands.iter().position(|c| canonicalize(c) == *target) {
                all_samples.push((pos, feats));
            }
        }
    }
    let (base_b_hits, base_b_n) = eval_dataset(&mut sugg, &dev_b);
    println!(
        "Baseline Dev-A: {:.2}% ({}/{}) | Dev-B: {:.2}% ({}/{})",
        base_a_hits as f64 / dev_a.len() as f64 * 100.0,
        base_a_hits,
        dev_a.len(),
        base_b_hits as f64 / base_b_n as f64 * 100.0,
        base_b_hits,
        base_b_n
    );
    println!("Collected {} rankable Dev-A samples.", all_samples.len());

    let mut best_weights = RankWeights::default();
    let mut best_feature_hits = 0usize;
    let mut w = RankWeights::default();

    for epoch in 0..80 {
        let lr = 8.0 * (0.97_f32).powi(epoch);
        for (target_idx, feats) in &all_samples {
            let target_f = &feats[*target_idx];
            // Find highest-scoring candidate under current weights
            let mut best_i = 0usize;
            let mut best_s = i32::MIN;
            for (i, f) in feats.iter().enumerate() {
                let s = w.compute_score(f);
                if s > best_s {
                    best_s = s;
                    best_i = i;
                }
            }
            if best_i != *target_idx {
                update_all_weights(&mut w, target_f, &feats[best_i], lr);
            }
        }
        // Evaluate on Dev-A rankable samples
        let mut hits = 0usize;
        for (target_idx, feats) in &all_samples {
            let mut best_i = 0usize;
            let mut best_s = i32::MIN;
            for (i, f) in feats.iter().enumerate() {
                let s = w.compute_score(f);
                if s > best_s {
                    best_s = s;
                    best_i = i;
                }
            }
            if best_i == *target_idx {
                hits += 1;
            }
        }
        if hits > best_feature_hits {
            best_feature_hits = hits;
            best_weights = w;
        }
    }

    println!("Best feature-only Dev-A hits: {}/{}", best_feature_hits, all_samples.len());
    println!("Tuned RankWeights:\n{}", serde_json::to_string_pretty(&best_weights).unwrap());

    // Apply tuned weights end-to-end on Dev-A and Dev-B
    {
        let mut learner = sugg.database.learner.write().expect("lock learner");
        learner.rank_weights = best_weights;
    }
    sugg.clear_cache();
    let (new_a_hits, new_a_n) = eval_dataset(&mut sugg, &dev_a);
    let (new_b_hits, new_b_n) = eval_dataset(&mut sugg, &dev_b);
    println!(
        "Tuned End-to-End Dev-A: {:.2}% ({}/{}) | Dev-B: {:.2}% ({}/{})",
        new_a_hits as f64 / new_a_n as f64 * 100.0,
        new_a_hits,
        new_a_n,
        new_b_hits as f64 / new_b_n as f64 * 100.0,
        new_b_hits,
        new_b_n
    );
}
