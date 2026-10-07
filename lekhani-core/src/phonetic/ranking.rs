//! Normalized Feature Vector and Linear Scoring for Bengali Phonetic Typing
//!
//! Provides mathematically principled, zero-allocation candidate ranking:
//! Score(c) = w · f(c, context)
//! with on-device online perceptron adaptation.

use serde::{Deserialize, Serialize};

/// 20-dimensional normalized feature vector extracted for a candidate hypothesis.
/// Entirely stack-allocated (`Copy`, 80 bytes) with zero heap allocation on the hot path.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct RankFeatures {
    // ── Core features (original 12D) ─────────────────────────────────────────
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
    // ── Extended features (new 8D) ────────────────────────────────────────────
    /// Context-conditioned LM score from fourgram with committed words: [-6,0] → [0,1]
    pub context_lm_score: f32,
    /// Log-probability of candidate from MicroGRU neural predictor: [-6,0] → [0,1]
    pub gru_log_prob: f32,
    /// Neural alpha blending weight at time of ranking (compute_neural_alpha output)
    pub neural_alpha: f32,
    /// Morpheme suffix chain depth: 0.0=root, 0.33=1 suffix, 0.67=2 suffixes, 1.0=3+
    pub morpheme_depth: f32,
    /// 1.0 if candidate hits the named entity dictionary
    pub is_named_entity: f32,
    /// 1.0 if candidate was sourced from colloquial verbal patterns
    pub is_colloquial: f32,
    /// 1.0 if this is the first token in the sentence (post-দাঁড়ি / sentence-start)
    pub sentence_initial: f32,
    /// 1.0 if context matches a formal register phrase head (BENGALI_FORMAL_PHRASES)
    pub register_match: f32,
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
            context_lm_score: 0.0,
            gru_log_prob: 0.0,
            neural_alpha: 0.0,
            morpheme_depth: 0.0,
            is_named_entity: 0.0,
            is_colloquial: 0.0,
            sentence_initial: 0.0,
            register_match: 0.0,
        }
    }
}

/// Linear weight vector corresponding to each feature dimension in `RankFeatures`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
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
    // Extended
    pub context_lm_score: f32,
    pub gru_log_prob: f32,
    pub neural_alpha: f32,
    pub morpheme_depth: f32,
    pub is_named_entity: f32,
    pub is_colloquial: f32,
    pub sentence_initial: f32,
    pub register_match: f32,
}

impl Default for RankWeights {
    fn default() -> Self {
        Self {
            is_in_dict: 2200.0,
            normalized_freq: 2080.0,
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
            // Extended weights — conservative but meaningful
            context_lm_score: 1800.0,
            gru_log_prob: 1200.0,
            neural_alpha: 0.0,      // meta-feature, zero weight for now
            morpheme_depth: -400.0, // deeper suffix chains less likely primary candidates
            is_named_entity: 1500.0,
            is_colloquial: 600.0,
            sentence_initial: 800.0,
            register_match: 1200.0,
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
            + self.user_favored * f.user_favored
            + self.context_lm_score * f.context_lm_score
            + self.gru_log_prob * f.gru_log_prob
            + self.neural_alpha * f.neural_alpha
            + self.morpheme_depth * f.morpheme_depth
            + self.is_named_entity * f.is_named_entity
            + self.is_colloquial * f.is_colloquial
            + self.sentence_initial * f.sentence_initial
            + self.register_match * f.register_match;
        score as i32
    }

    /// Safe online perceptron update. Only adaptive weights are modified.
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
        self.context_lm_score = (self.context_lm_score + lr * (chosen.context_lm_score - rejected.context_lm_score))
            .clamp(500.0, 4000.0);
        self.gru_log_prob = (self.gru_log_prob + lr * (chosen.gru_log_prob - rejected.gru_log_prob))
            .clamp(200.0, 3000.0);
    }
}

use super::suggestion::{CandidateHypothesis, CandidateSource};
use unicode_segmentation::UnicodeSegmentation;

/// Grapheme-cluster-aware Levenshtein edit distance to preserve Bengali conjuncts.
pub fn grapheme_edit_distance(a: &str, b: &str) -> usize {
    let a_g: Vec<&str> = a.graphemes(true).collect();
    let b_g: Vec<&str> = b.graphemes(true).collect();
    let (la, lb) = (a_g.len(), b_g.len());
    if la == 0 { return lb; }
    if lb == 0 { return la; }
    let mut prev: Vec<usize> = (0..=lb).collect();
    let mut curr: Vec<usize> = vec![0; lb + 1];
    for (i, &ga) in a_g.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &gb) in b_g.iter().enumerate() {
            let cost = if ga == gb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        prev.copy_from_slice(&curr);
    }
    prev[lb]
}

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
    // Extended context (new)
    /// LM score conditioned on last 2 committed words (fourgram). None if no prior context.
    pub context_lm_score: Option<f32>,
    /// MicroGRU log-probability for this candidate. None if GRU unavailable.
    pub gru_log_prob: Option<f32>,
    /// Neural alpha blending weight (output of compute_neural_alpha).
    pub neural_alpha: f32,
    /// True if this is the first word after a sentence boundary (।, ?, !)
    pub is_sentence_initial: bool,
    /// True if current context head matches a formal register phrase
    pub is_formal_register_context: bool,
}

impl<'a> CandidateContext<'a> {
    #[inline]
    pub fn has_extended_context(&self) -> bool {
        self.context_lm_score.is_some() || self.gru_log_prob.is_some()
    }
}

/// Extract a 20-dimensional normalized feature vector for a candidate hypothesis.
/// Stack-allocated, zero heap allocations on the hot path.
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
        || cand.source == CandidateSource::PhoneticOverride
    {
        f.is_in_dict = 1.0;
    } else if cand.source == CandidateSource::DirectTransliteration
        && ctx.has_dominant_homophone
        && !ctx.has_explicit_casing
        && !ctx.has_backtick
        && !ctx.has_explicit_rri_digraph
        && (!ctx.is_atomic_syllable || !ctx.is_primary_in_dict)
        && !ctx.is_short_token
    {
        f.is_in_dict = -1.27;
    }

    // 2. Frequency
    if freq > 0 {
        let log_freq = (freq as f32).log2().clamp(0.0, 16.0);
        f.normalized_freq = log_freq / 16.0;
    } else if (cand.source == CandidateSource::PhoneticOverride
        || cand.source == CandidateSource::Autocorrect)
        && cand.text.contains(' ')
    {
        f.normalized_freq = 0.80;
        f.is_high_freq += 0.5;
    }
    if freq >= 8000 {
        f.is_high_freq += 1.0;
    }
    if trie_contains {
        f.is_high_freq += 0.5;
    }

    // 3. Phonetic Similarity & Length
    if cand.source != CandidateSource::EmojiKeyword {
        let dist = grapheme_edit_distance(ctx.phonetic, &cand.text);
        if cand.text == ctx.phonetic || cand.text == ctx.primary {
            let is_dict_equiv = is_in_dict
                || cand.source == CandidateSource::MorphologicalInflection
                || cand.source == CandidateSource::Autocorrect
                || cand.source == CandidateSource::Loanword
                || cand.source == CandidateSource::PhoneticOverride;
            f.is_exact_phonetic = if is_dict_equiv { 1.0 } else { 0.43 };

            if ctx.has_backtick {
                f.intent_modifier_boost = 1.11;
            } else if ctx.has_explicit_casing || ctx.has_explicit_rri_digraph {
                f.intent_modifier_boost = 0.78;
            } else if ctx.is_atomic_syllable && ctx.is_primary_in_dict {
                f.intent_modifier_boost = 0.67;
            } else if ctx.is_short_token {
                f.intent_modifier_boost = 0.27;
            }
        } else if cand.source == CandidateSource::TypoFallback {
            f.source_weight = -0.43;
            f.phonetic_similarity = -((dist.min(3)) as f32);
        } else {
            f.phonetic_similarity = -(dist as f32);
            let len_diff = (cand.text.chars().count() as isize - ctx.phonetic.chars().count() as isize)
                .abs() as f32;
            f.length_penalty = -len_diff;

            if (ctx.is_primary_in_dict || ctx.has_explicit_rri_digraph)
                && ctx.is_atomic_syllable
                && dist > 0
            {
                f.intent_modifier_boost -= 0.78;
            }
            if cand.source == CandidateSource::FuzzySoundLaw {
                if ctx.has_backtick {
                    f.intent_modifier_boost -= 0.89;
                } else if ctx.has_explicit_casing {
                    f.intent_modifier_boost -= 0.56;
                } else if ctx.middle.chars().count() <= 2 && dist > 0 {
                    f.intent_modifier_boost -= 0.67;
                }
                if ctx.lm_score.is_some_and(|s| s > -1.5) && !ctx.has_explicit_casing && !ctx.has_backtick {
                    f.is_exact_phonetic = 0.9;
                }
            }
            if cand.source == CandidateSource::Autocorrect {
                if ctx.has_backtick {
                    f.intent_modifier_boost -= 1.11;
                } else if ctx.has_explicit_casing {
                    f.intent_modifier_boost -= 0.78;
                }
            }
        }
    }

    // 4. Clitics
    if ctx.has_clitic_o_candidate {
        if is_clitic_o {
            f.clitic_alignment = 1.0;
        } else if !cand.text.ends_with("্য")
            && ctx.clitic_o_targets.iter().any(|t| {
                t.strip_prefix(&cand.text).is_some_and(|s| {
                    s == "ও"
                        || (s == "ো"
                            && matches!(
                                t.as_str(),
                                "এখনো" | "তখনো" | "কখনো" | "এমনো" | "কোনো" | "যখনো"
                            ))
                })
            })
        {
            f.clitic_alignment = -1.28;
        }
    }

    if ctx.has_clitic_i_candidate {
        let is_valid_clitic = cand.text.ends_with('ই')
            && (ctx.primary == ctx.phonetic || cand.text == ctx.primary);
        if is_valid_clitic {
            f.clitic_alignment = 1.0;
        } else if ctx.clitic_i_targets.iter().any(|t| t.strip_prefix(&cand.text) == Some("ই")) {
            f.clitic_alignment = -0.85;
        }
    }

    // 5. Language Model (local)
    if let Some(lm) = ctx.lm_score {
        f.lm_score = (lm.max(-6.0) / 6.0 + 1.0).clamp(0.0, 1.0);
    }

    // 6. User Personalization
    if ctx.user_bigram_boost > 0 {
        f.user_bigram_prob = (ctx.user_bigram_boost as f32 / 4500.0).clamp(0.0, 1.0);
    }
    if ctx.is_user_favored {
        f.user_favored = 1.0;
    } else if ctx.is_user_learned {
        f.user_favored = 0.7;
    }

    // 7. Extended features

    // 7a. Context-conditioned LM (fourgram with committed words)
    if let Some(clm) = ctx.context_lm_score {
        f.context_lm_score = (clm.max(-6.0) / 6.0 + 1.0).clamp(0.0, 1.0);
    }

    // 7b. GRU neural log-probability
    if let Some(gp) = ctx.gru_log_prob {
        f.gru_log_prob = (gp.max(-6.0) / 6.0 + 1.0).clamp(0.0, 1.0);
    }

    // 7c. Neural alpha
    f.neural_alpha = ctx.neural_alpha.clamp(0.0, 1.0);

    // 7d. Morpheme depth
    f.morpheme_depth = if cand.source == CandidateSource::MorphologicalInflection {
        (count_suffix_depth(&cand.text) as f32 / 3.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // 7e. Named entity sentinel (initial_boost == i32::MAX)
    if cand.initial_boost == i32::MAX {
        f.is_named_entity = 1.0;
    }

    // 7f. Colloquial verbal form detection
    if cand.source == CandidateSource::MorphologicalInflection && is_colloquial_verbal_form(&cand.text) {
        f.is_colloquial = 1.0;
    }

    // 7g. Sentence-initial
    if ctx.is_sentence_initial {
        f.sentence_initial = 1.0;
    }

    // 7h. Formal register context
    if ctx.is_formal_register_context {
        f.register_match = 1.0;
    }

    f
}

/// Count approximate morpheme suffix depth for a Bengali word.
#[inline]
fn count_suffix_depth(text: &str) -> usize {
    const SUFFIX_MARKERS: &[&str] = &[
        "গুলোর", "গুলোকে", "গুলো", "দেরকে", "দের",
        "টিকে", "টাকে", "টির", "টার", "টিতে", "টাতে", "টি", "টা",
        "তেও", "রাও", "রা", "ের", "এর", "কে", "তে", "ও", "ই",
    ];
    let mut depth = 0usize;
    for &sfx in SUFFIX_MARKERS {
        if text.ends_with(sfx) {
            depth += 1;
            if let Some(stem) = text.strip_suffix(sfx) {
                for &sfx2 in SUFFIX_MARKERS {
                    if stem.ends_with(sfx2) { depth += 1; break; }
                }
            }
            break;
        }
    }
    depth
}

/// Returns true if the word ends with a recognizable colloquial Bengali verbal suffix.
#[inline]
fn is_colloquial_verbal_form(text: &str) -> bool {
    const COLLOQUIAL_ENDINGS: &[&str] = &[
        "চ্ছিলাম", "চ্ছিলে", "চ্ছিল",
        "চ্ছি", "চ্ছে",
        "ছিলাম", "ছিলে", "ছিল",
        "বো", "বে", "বেন",
    ];
    COLLOQUIAL_ENDINGS.iter().any(|&e| text.ends_with(e))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_stack_size_and_copy() {
        // 20 f32 fields × 4 bytes = 80 bytes
        assert_eq!(std::mem::size_of::<RankFeatures>(), 80);
        let f1 = RankFeatures::default();
        let f2 = f1;
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_baseline_scoring_dot_product() {
        let weights = RankWeights::default();
        let features = RankFeatures {
            is_in_dict: 1.0,
            is_exact_phonetic: 1.0,
            lm_score: 0.8,
            ..Default::default()
        };
        let score = weights.compute_score(&features);
        // 2200 + 3500 + 0.8 * 5000 = 9700
        assert_eq!(score, 9700);
    }

    #[test]
    fn test_perceptron_online_update_and_clamping() {
        let mut weights = RankWeights::default();
        let chosen = RankFeatures { lm_score: 1.0, user_bigram_prob: 1.0, ..Default::default() };
        let rejected = RankFeatures { lm_score: 0.2, user_bigram_prob: 0.0, ..Default::default() };
        let initial_lm = weights.lm_score;
        weights.update_online(&chosen, &rejected, 35.0);
        assert!(weights.lm_score > initial_lm);
        for _ in 0..500 {
            weights.update_online(&chosen, &rejected, 50.0);
        }
        assert_eq!(weights.lm_score, 8000.0);
    }

    fn make_ctx<'a>(
        phonetic: &'a str,
        primary: &'a str,
        targets: &'a Vec<String>,
    ) -> CandidateContext<'a> {
        CandidateContext {
            phonetic,
            primary,
            middle: phonetic,
            term: phonetic,
            has_dominant_homophone: false,
            has_explicit_casing: false,
            has_backtick: false,
            has_explicit_rri_digraph: false,
            is_atomic_syllable: false,
            is_primary_in_dict: true,
            is_short_token: false,
            has_clitic_o_candidate: false,
            has_clitic_i_candidate: false,
            clitic_o_targets: targets,
            clitic_i_targets: targets,
            lm_score: Some(-0.5),
            user_bigram_boost: 0,
            is_user_favored: false,
            is_user_learned: false,
            context_lm_score: Some(-1.2),
            gru_log_prob: Some(-2.0),
            neural_alpha: 0.35,
            is_sentence_initial: true,
            is_formal_register_context: false,
        }
    }

    #[test]
    fn test_extract_candidate_features_extended() {
        let targets = Vec::new();
        let ctx = make_ctx("ami", "আমি", &targets);
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
        assert!(f.context_lm_score > 0.0);
        assert!(f.gru_log_prob > 0.0);
        assert_eq!(f.neural_alpha, 0.35);
        assert_eq!(f.sentence_initial, 1.0);
    }

    #[test]
    fn test_grapheme_edit_distance_bengali_conjuncts() {
        assert_eq!(grapheme_edit_distance("ক্ষ", "ক"), 1);
        assert_eq!(grapheme_edit_distance("পড়া", "পরা"), 1);
        assert_eq!(grapheme_edit_distance("বাংলাদেশ", "বাংলাদেশ"), 0);
        assert_eq!(grapheme_edit_distance("", "আমি"), 2);
        assert_eq!(grapheme_edit_distance("তুমি", ""), 2);
    }

    #[test]
    fn test_named_entity_sentinel() {
        let targets = Vec::new();
        let ctx = CandidateContext {
            phonetic: "dhaka", primary: "ঢাকা", middle: "dhaka", term: "dhaka",
            has_dominant_homophone: false, has_explicit_casing: false, has_backtick: false,
            has_explicit_rri_digraph: false, is_atomic_syllable: false, is_primary_in_dict: true,
            is_short_token: false, has_clitic_o_candidate: false, has_clitic_i_candidate: false,
            clitic_o_targets: &targets, clitic_i_targets: &targets, lm_score: None,
            user_bigram_boost: 0, is_user_favored: false, is_user_learned: false,
            context_lm_score: None, gru_log_prob: None, neural_alpha: 0.1,
            is_sentence_initial: true, is_formal_register_context: false,
        };
        let cand = CandidateHypothesis {
            text: "ঢাকা".to_string(),
            source: CandidateSource::DirectTransliteration,
            initial_boost: i32::MAX,
        };
        let f = extract_candidate_features(&cand, true, 5000, true, &ctx, false);
        assert_eq!(f.is_named_entity, 1.0);
    }

    #[test]
    fn test_morpheme_depth_suffix() {
        assert_eq!(count_suffix_depth("মানুষের"), 1);
        assert_eq!(count_suffix_depth("মানুষ"), 0);
        assert_eq!(count_suffix_depth("মানুষগুলোর"), 1);
    }

    #[test]
    fn test_colloquial_verbal_form_detection() {
        assert!(is_colloquial_verbal_form("যাচ্ছি"));
        assert!(is_colloquial_verbal_form("করছিলাম"));  // ছিলাম
        assert!(!is_colloquial_verbal_form("করেছে"));
        assert!(is_colloquial_verbal_form("যাবো"));
    }
}
