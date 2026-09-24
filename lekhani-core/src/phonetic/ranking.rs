//! Normalized Feature Vector and Linear Scoring for Bengali Phonetic Typing
//!
//! Provides mathematically principled, zero-allocation candidate ranking:
//! Score(c) = w · f(c, context)
//! with on-device online perceptron adaptation.

use serde::{Deserialize, Serialize};

/// 12-dimensional normalized feature vector extracted for a candidate hypothesis.
/// Entirely stack-allocated (`Copy`, 48 bytes) with zero heap allocation on the hot path.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct RankFeatures {
    /// 1.0 if word exists in root dictionary / prefix trie, else 0.0
    pub is_in_dict: f32,
    /// log2(frequency) clamped to [0.0, 16.0], normalized to [0.0, 1.0]
    pub normalized_freq: f32,
    /// 1.0 if frequency >= 8000 (core high-frequency vocabulary), else 0.0
    pub is_high_freq: f32,
    /// Language Model log-probability clamped to [-6.0, 0.0], normalized to [0.0, 1.0]
    pub lm_score: f32,
    /// 1.0 if candidate string matches raw phonetic transliteration or primary form
    pub is_exact_phonetic: f32,
    /// Normalized Levenshtein similarity: 1.0 - (dist / max(len, 1)).min(1.0)
    pub phonetic_similarity: f32,
    /// Normalized character length delta penalty: -abs(cand_len - phonetic_len) / max(phonetic_len, 1)
    pub length_penalty: f32,
    /// Source priority: Autocorrect/Loanword/Inflection=1.0, Direct=0.8, Fuzzy=0.5, Typo=-0.5
    pub source_weight: f32,
    /// Clitic-ও / Clitic-ই alignment bonus (+1.0 for matching clitic, -1.0 for usurped root)
    pub clitic_alignment: f32,
    /// Explicit user intent boost (backtick ` `, uppercase casing, or explicit `rri`)
    pub intent_modifier_boost: f32,
    /// Dynamic user bigram frequency count: (count / 6.0).min(1.0)
    pub user_bigram_prob: f32,
    /// User explicit candidate memory or autonomous learner bonus: 1.0 if favored, else 0.0
    pub user_favored: f32,
}

impl Default for RankFeatures {
    fn default() -> Self {
        Self {
            is_in_dict: 0.0,
            normalized_freq: 0.0,
            is_high_freq: 0.0,
            lm_score: 0.0,
            is_exact_phonetic: 0.0,
            phonetic_similarity: 0.0,
            length_penalty: 0.0,
            source_weight: 0.0,
            clitic_alignment: 0.0,
            intent_modifier_boost: 0.0,
            user_bigram_prob: 0.0,
            user_favored: 0.0,
        }
    }
}

/// Linear weight vector corresponding to each feature dimension in `RankFeatures`.
/// Calibrated against the baseline heuristic scoring to preserve 100% test compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct RankWeights {
    pub is_in_dict: f32,
    pub normalized_freq: f32,
    pub is_high_freq: f32,
    pub lm_score: f32,
    pub is_exact_phonetic: f32,
    pub phonetic_similarity: f32,
    pub length_penalty: f32,
    pub source_weight: f32,
    pub clitic_alignment: f32,
    pub intent_modifier_boost: f32,
    pub user_bigram_prob: f32,
    pub user_favored: f32,
}

impl Default for RankWeights {
    fn default() -> Self {
        // Calibrated baseline to produce identical scoring to current heuristics:
        Self {
            is_in_dict: 2200.0,
            normalized_freq: 2080.0, // 16.0 * 130.0
            is_high_freq: 2000.0,
            lm_score: 5000.0,
            is_exact_phonetic: 3500.0,
            phonetic_similarity: 1600.0,
            length_penalty: 1200.0,
            source_weight: 2500.0,
            clitic_alignment: 3500.0,
            intent_modifier_boost: 4500.0,
            user_bigram_prob: 3000.0,
            user_favored: 5000.0,
        }
    }
}

impl RankWeights {
    /// Compute scalar ranking score via dot product: Score = w · f
    #[inline(always)]
    pub fn compute_score(&self, f: &RankFeatures) -> i32 {
        let score = self.is_in_dict * f.is_in_dict
            + self.normalized_freq * f.normalized_freq
            + self.is_high_freq * f.is_high_freq
            + self.lm_score * f.lm_score
            + self.is_exact_phonetic * f.is_exact_phonetic
            + self.phonetic_similarity * f.phonetic_similarity
            + self.length_penalty * f.length_penalty
            + self.source_weight * f.source_weight
            + self.clitic_alignment * f.clitic_alignment
            + self.intent_modifier_boost * f.intent_modifier_boost
            + self.user_bigram_prob * f.user_bigram_prob
            + self.user_favored * f.user_favored;
        score as i32
    }

    /// Perform a safe online perceptron update when a user chooses candidate `chosen` over `rejected`.
    /// Only adaptive weights are modified, clamped to safe boundaries to prevent divergence.
    pub fn update_online(&mut self, chosen: &RankFeatures, rejected: &RankFeatures, eta: f32) {
        let lr = eta.clamp(5.0, 50.0);

        self.normalized_freq = (self.normalized_freq + lr * (chosen.normalized_freq - rejected.normalized_freq))
            .clamp(1000.0, 4000.0);
        self.lm_score = (self.lm_score + lr * (chosen.lm_score - rejected.lm_score))
            .clamp(2500.0, 8000.0);
        self.phonetic_similarity = (self.phonetic_similarity + lr * (chosen.phonetic_similarity - rejected.phonetic_similarity))
            .clamp(800.0, 3200.0);
        self.source_weight = (self.source_weight + lr * (chosen.source_weight - rejected.source_weight))
            .clamp(1200.0, 5000.0);
        self.user_bigram_prob = (self.user_bigram_prob + lr * (chosen.user_bigram_prob - rejected.user_bigram_prob))
            .clamp(1500.0, 6000.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_stack_size_and_copy() {
        assert_eq!(std::mem::size_of::<RankFeatures>(), 48);
        let f1 = RankFeatures::default();
        let f2 = f1; // Verifies Copy semantics
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_baseline_scoring_dot_product() {
        let weights = RankWeights::default();
        let mut features = RankFeatures::default();
        features.is_in_dict = 1.0;
        features.is_exact_phonetic = 1.0;
        features.lm_score = 0.8;

        let score = weights.compute_score(&features);
        // 2200 + 3500 + 0.8 * 5000 = 2200 + 3500 + 4000 = 9700
        assert_eq!(score, 9700);
    }

    #[test]
    fn test_perceptron_online_update_and_clamping() {
        let mut weights = RankWeights::default();
        let mut chosen = RankFeatures::default();
        chosen.lm_score = 1.0;
        chosen.user_bigram_prob = 1.0;

        let mut rejected = RankFeatures::default();
        rejected.lm_score = 0.2;
        rejected.user_bigram_prob = 0.0;

        let initial_lm = weights.lm_score;
        weights.update_online(&chosen, &rejected, 35.0);

        // LM weight should increase
        assert!(weights.lm_score > initial_lm);

        // Extreme updates should be clamped
        for _ in 0..500 {
            weights.update_online(&chosen, &rejected, 50.0);
        }
        assert_eq!(weights.lm_score, 8000.0); // max clamp
    }
}
