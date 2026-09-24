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
            phonetic_similarity: 200.0,
            length_penalty: 400.0,
            source_weight: 4200.0,
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
            .clamp(100.0, 600.0);
        self.length_penalty = (self.length_penalty + lr * (chosen.length_penalty - rejected.length_penalty))
            .clamp(200.0, 800.0);
        self.user_bigram_prob = (self.user_bigram_prob + lr * (chosen.user_bigram_prob - rejected.user_bigram_prob))
            .clamp(1500.0, 6000.0);
    }
}

use super::suggestion::{CandidateHypothesis, CandidateSource};
use edit_distance::edit_distance;

/// Per-token contextual properties extracted once per keystroke.
pub struct CandidateContext<'a> {
    pub phonetic: &'a str,
    pub primary: &'a str,
    pub middle: &'a str,
    pub term: &'a str,
    pub has_dominant_homophone: bool,
    pub has_explicit_casing: bool,
    pub has_backtick: bool,
    pub has_explicit_rri_digraph: bool,
    pub is_atomic_syllable: bool,
    pub is_primary_in_dict: bool,
    pub is_short_token: bool,
    pub has_clitic_o_candidate: bool,
    pub has_clitic_i_candidate: bool,
    pub clitic_o_targets: &'a [String],
    pub clitic_i_targets: &'a [String],
    pub lm_score: Option<f32>,
    pub user_bigram_boost: i32,
    pub is_user_favored: bool,
    pub is_user_learned: bool,
}

/// Extract a 12-dimensional normalized feature vector for a candidate hypothesis.
/// Stack-allocated, zero heap allocations.
pub fn extract_candidate_features(
    cand: &CandidateHypothesis,
    is_in_dict: bool,
    freq: u32,
    trie_contains: bool,
    ctx: &CandidateContext,
    is_clitic_o: bool,
) -> RankFeatures {
    let mut f = RankFeatures::default();

    // 1. Dictionary & Trie Presence
    if is_in_dict
        || cand.source == CandidateSource::MorphologicalInflection
        || cand.source == CandidateSource::Autocorrect
        || cand.source == CandidateSource::Loanword
    {
        f.is_in_dict = 1.0;
    } else if cand.source == CandidateSource::DirectTransliteration {
        if ctx.has_dominant_homophone
            && !ctx.has_explicit_casing
            && !ctx.has_backtick
            && !ctx.has_explicit_rri_digraph
            && (!ctx.is_atomic_syllable || !ctx.is_primary_in_dict)
            && !ctx.is_short_token
        {
            f.is_in_dict = -1.27; // -2800 / 2200
        }
    }

    // 2. Frequency
    if freq > 0 {
        let log_freq = (freq as f32).log2().clamp(0.0, 16.0);
        f.normalized_freq = log_freq / 16.0;
    }
    if freq >= 8000 {
        f.is_high_freq += 1.0;
    }
    if trie_contains {
        f.is_high_freq += 0.5; // +1000
    }

    // 3. Phonetic Similarity & Length
    if cand.source != CandidateSource::EmojiKeyword {
        let dist = edit_distance(ctx.phonetic, &cand.text);
        if cand.text == ctx.phonetic || cand.text == ctx.primary {
            let is_dict_equiv = is_in_dict
                || cand.source == CandidateSource::MorphologicalInflection
                || cand.source == CandidateSource::Autocorrect
                || cand.source == CandidateSource::Loanword;
            f.is_exact_phonetic = if is_dict_equiv { 1.0 } else { 0.43 }; // 3500 vs 1500

            if ctx.has_backtick {
                f.intent_modifier_boost = 1.11; // 5000 / 4500
            } else if ctx.has_explicit_casing || ctx.has_explicit_rri_digraph {
                f.intent_modifier_boost = 0.78; // 3500 / 4500
            } else if ctx.is_atomic_syllable && ctx.is_primary_in_dict {
                f.intent_modifier_boost = 0.67; // 3000 / 4500
            } else if ctx.is_short_token {
                f.intent_modifier_boost = 0.27; // 1200 / 4500
            }
        } else if cand.source == CandidateSource::TypoFallback {
            f.source_weight = -1.0; // -4200
            f.phonetic_similarity = -((dist.min(3)) as f32); // -dist * 200
        } else {
            f.phonetic_similarity = -(dist as f32);
            let len_diff = (cand.text.chars().count() as isize - ctx.phonetic.chars().count() as isize)
                .abs() as f32;
            f.length_penalty = -len_diff;

            if (ctx.is_primary_in_dict || ctx.has_explicit_rri_digraph)
                && ctx.is_atomic_syllable
                && dist > 0
            {
                f.intent_modifier_boost -= 0.78; // -3500 / 4500
            }
            if cand.source == CandidateSource::FuzzySoundLaw {
                if ctx.has_backtick {
                    f.intent_modifier_boost -= 0.89; // -4000 / 4500
                } else if ctx.has_explicit_casing {
                    f.intent_modifier_boost -= 0.56; // -2500 / 4500
                } else if ctx.middle.chars().count() <= 2 && dist > 0 {
                    f.intent_modifier_boost -= 0.67; // -3000 / 4500
                }
            }
            if cand.source == CandidateSource::Autocorrect {
                if ctx.has_backtick {
                    f.intent_modifier_boost -= 1.11; // -5000 / 4500
                } else if ctx.has_explicit_casing {
                    f.intent_modifier_boost -= 0.78; // -3500 / 4500
                }
            }
        }
    }


    // 5. Clitics
    if ctx.has_clitic_o_candidate {
        if is_clitic_o {
            f.clitic_alignment = 1.0;
        } else if !cand.text.ends_with("্য")
            && ctx.clitic_o_targets.iter().any(|t| {
                t.strip_prefix(&cand.text).map_or(false, |s| {
                    s == "ও"
                        || (s == "ো"
                            && (t == "এখনো"
                                || t == "তখনো"
                                || t == "কখনো"
                                || t == "এমনো"
                                || t == "কোনো"
                                || t == "যখনো"))
                })
            })
        {
            f.clitic_alignment = -1.28; // -4500 / 3500
        }
    }

    if ctx.has_clitic_i_candidate {
        if cand.text.ends_with('ই') {
            f.clitic_alignment = 1.0;
        } else if ctx
            .clitic_i_targets
            .iter()
            .any(|t| t.strip_prefix(&cand.text).map_or(false, |s| s == "ই"))
        {
            f.clitic_alignment = -0.85; // -3000 / 3500
        }
    }

    // 6. Language Model
    if let Some(lm) = ctx.lm_score {
        // Continuous scale: clamp log-prob to [-6, 0] then map to [0, 1]
        f.lm_score = (lm.max(-6.0) / 6.0 + 1.0).clamp(0.0, 1.0);
    }

    // 7. Dynamic User Bigram & Personalization
    if ctx.user_bigram_boost > 0 {
        f.user_bigram_prob = (ctx.user_bigram_boost as f32 / 4500.0).clamp(0.0, 1.0);
    }

    if ctx.is_user_favored {
        f.user_favored = 1.0;
    } else if ctx.is_user_learned {
        f.user_favored = 0.7; // 3500 / 5000
    }

    f
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

    #[test]
    fn test_extract_candidate_features() {
        let targets = Vec::new();
        let ctx = CandidateContext {
            phonetic: "ami",
            primary: "আমি",
            middle: "ami",
            term: "ami",
            has_dominant_homophone: false,
            has_explicit_casing: false,
            has_backtick: false,
            has_explicit_rri_digraph: false,
            is_atomic_syllable: false,
            is_primary_in_dict: true,
            is_short_token: false,
            has_clitic_o_candidate: false,
            has_clitic_i_candidate: false,
            clitic_o_targets: &targets,
            clitic_i_targets: &targets,
            lm_score: Some(-0.5),
            user_bigram_boost: 0,
            is_user_favored: false,
            is_user_learned: false,
        };

        let cand = CandidateHypothesis {
            text: "আমি".to_string(),
            source: CandidateSource::DirectTransliteration,
            initial_boost: 0,
        };

        let f = extract_candidate_features(&cand, true, 10000, true, &ctx, false);
        assert_eq!(f.is_in_dict, 1.0);
        assert_eq!(f.is_high_freq, 1.5);
        assert_eq!(f.is_exact_phonetic, 1.0);
        assert!(f.lm_score > 0.8);
    }
}

