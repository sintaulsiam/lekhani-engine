//! Offline perceptron pre-trainer for Lekhani ranking weights.
//! Usage: pretrain_rank_weights <path-to-phonetic_overrides.json>

use lekhani_core::phonetic::ranking::{RankFeatures, RankWeights};
use std::collections::HashMap;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: pretrain_rank_weights <overrides.json>");
    let data = std::fs::read_to_string(&path).expect("Failed to read overrides JSON");
    let overrides: HashMap<String, Vec<(String, f32)>> =
        serde_json::from_str(&data).expect("Invalid JSON");

    let mut weights = RankWeights::default();

    // Each entry: romanized -> [(bengali_form, confidence_score), ...]
    // We train the perceptron to rank the highest-confidence form above the lowest.
    let pairs: Vec<(RankFeatures, RankFeatures)> = overrides
        .values()
        .filter(|v| v.len() >= 2)
        .map(|v| {
            // chosen = first entry (highest confidence, already sorted in overrides)
            // rejected = last entry (lowest confidence)
            let chosen_score = v[0].1;
            let rejected_score = v[v.len() - 1].1;

            let chosen = RankFeatures {
                is_in_dict: 1.0,
                normalized_freq: (chosen_score * 16.0).min(16.0) / 16.0,
                is_high_freq: if chosen_score >= 0.9 { 1.0 } else { 0.0 },
                lm_score: chosen_score.clamp(0.0, 1.0),
                is_exact_phonetic: 1.0,
                phonetic_similarity: 0.0,
                length_penalty: 0.0,
                source_weight: 1.0,
                clitic_alignment: 0.0,
                intent_modifier_boost: if chosen_score >= 0.9 { 1.0 } else { 0.5 },
                user_bigram_prob: 0.0,
                user_favored: if chosen_score >= 0.95 { 1.0 } else { 0.5 },
                ..Default::default()
            };

            let rejected = RankFeatures {
                is_in_dict: if rejected_score > 0.5 { 1.0 } else { 0.0 },
                normalized_freq: (rejected_score * 16.0).min(16.0) / 16.0,
                is_high_freq: 0.0,
                lm_score: rejected_score.clamp(0.0, 1.0),
                is_exact_phonetic: 0.43, // secondary phonetic form
                phonetic_similarity: -1.0,
                length_penalty: -0.5,
                source_weight: 0.0,
                clitic_alignment: 0.0,
                intent_modifier_boost: 0.0,
                user_bigram_prob: 0.0,
                user_favored: 0.0,
                ..Default::default()
            };

            (chosen, rejected)
        })
        .collect();

    eprintln!("Loaded {} training pairs from {} override entries.", pairs.len(), overrides.len());

    // Train for 500 epochs
    for epoch in 0..500 {
        let mut mistakes = 0usize;
        for (chosen, rejected) in &pairs {
            let score_chosen = weights.compute_score(chosen);
            let score_rejected = weights.compute_score(rejected);
            if score_chosen <= score_rejected {
                weights.update_online(chosen, rejected, 35.0);
                mistakes += 1;
            }
        }
        if epoch % 100 == 0 {
            eprintln!("Epoch {}: {} mistakes / {} pairs", epoch, mistakes, pairs.len());
        }
        if mistakes == 0 {
            eprintln!("Converged at epoch {}.", epoch);
            break;
        }
    }

    let json = serde_json::to_string_pretty(&weights).expect("Serialization failed");
    println!("{}", json);
}
