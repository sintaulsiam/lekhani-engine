//! Asynchronous Micro-Neural Next-Word Predictor
//!
//! Provides intent-driven, semantically clustered word predictions.
//! Blends neural semantic candidates with Tier 1 N-gram candidates with an
//! adaptive weighting factor.

use std::sync::Arc;
use crate::model::MicroGruModel;
use crate::vocab::BpeVocabulary;

/// Candidate predicted by the neural engine
#[derive(Debug, Clone, PartialEq)]
pub struct NeuralCandidate {
    pub word: String,
    pub log_prob: f32,
}

/// Asynchronous Semantic Predictor wrapping GRU model and BPE vocabulary
#[derive(Clone)]
pub struct NeuralContextPredictor {
    model: Arc<MicroGruModel>,
    vocab: Arc<BpeVocabulary>,
}

impl NeuralContextPredictor {
    pub fn new(model: Arc<MicroGruModel>, vocab: Arc<BpeVocabulary>) -> Self {
        Self { model, vocab }
    }

    /// Predict top semantic candidates for a conversational context string.
    pub fn predict_candidates(&self, context: &str, top_k: usize) -> Vec<NeuralCandidate> {
        let token_ids = self.vocab.encode(context);
        if token_ids.is_empty() {
            return Vec::new();
        }

        // Limit context window to last 32 tokens
        let window = if token_ids.len() > 32 {
            &token_ids[token_ids.len() - 32..]
        } else {
            &token_ids[..]
        };

        let top_tokens = self.model.predict_top_k(window, top_k * 2);
        let mut candidates = Vec::with_capacity(top_k);

        for (id, log_prob) in top_tokens {
            if let Some(word) = self.vocab.get_token(id) {
                // Ignore special tokens
                if word == crate::vocab::PAD_TOKEN
                    || word == crate::vocab::UNK_TOKEN
                    || word == crate::vocab::BOS_TOKEN
                    || word == crate::vocab::EOS_TOKEN
                {
                    continue;
                }

                if !candidates.iter().any(|c: &NeuralCandidate| c.word == word) {
                    candidates.push(NeuralCandidate {
                        word: word.to_string(),
                        log_prob,
                    });
                    if candidates.len() >= top_k {
                        break;
                    }
                }
            }
        }

        candidates
    }

    /// Blend Tier 1 N-gram candidates with Tier 2 Neural semantic candidates.
    ///
    /// `alpha` represents the weight of the neural engine:
    /// - During active typing: alpha ~ 0.2 (Tier 1 N-gram dominates)
    /// - After spacebar or typing pause: alpha ~ 0.6 (Tier 2 Semantic dominates)
    pub fn blend_candidates(
        &self,
        ngram_candidates: &[String],
        neural_candidates: &[NeuralCandidate],
        alpha: f32,
    ) -> Vec<String> {
        let mut merged: Vec<String> = Vec::with_capacity(ngram_candidates.len() + neural_candidates.len());

        let alpha = alpha.clamp(0.0, 1.0);

        if alpha >= 0.5 {
            // Neural-dominant: place top neural candidates first, then fill with N-grams
            for nc in neural_candidates {
                if !merged.contains(&nc.word) {
                    merged.push(nc.word.clone());
                }
            }
            for ng in ngram_candidates {
                if !merged.contains(ng) {
                    merged.push(ng.clone());
                }
            }
        } else {
            // Ngram-dominant: preserve N-gram rank 1, interleave high-confidence neural
            for (idx, ng) in ngram_candidates.iter().enumerate() {
                if !merged.contains(ng) {
                    merged.push(ng.clone());
                }
                // Interleave neural candidate after top-1 N-gram
                if idx == 0 {
                    if let Some(top_neural) = neural_candidates.first() {
                        if !merged.contains(&top_neural.word) {
                            merged.push(top_neural.word.clone());
                        }
                    }
                }
            }
            for nc in neural_candidates {
                if !merged.contains(&nc.word) {
                    merged.push(nc.word.clone());
                }
            }
        }

        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_blending_alpha() {
        let vocab = Arc::new(BpeVocabulary::from_tokens(vec![
            "আমি".into(),
            "কাল".into(),
            "যাব".into(),
            "আসব".into(),
        ]));
        let model = Arc::new(MicroGruModel::new(vocab.len(), 8, 8));
        let predictor = NeuralContextPredictor::new(model, vocab);

        let ngrams = vec!["যাব".to_string(), "খাব".to_string()];
        let neurals = vec![
            NeuralCandidate { word: "আসব".into(), log_prob: -0.2 },
            NeuralCandidate { word: "যাব".into(), log_prob: -0.8 },
        ];

        // Active typing: N-gram dominant (alpha = 0.2)
        let blended_active = predictor.blend_candidates(&ngrams, &neurals, 0.2);
        assert_eq!(blended_active[0], "যাব");
        assert_eq!(blended_active[1], "আসব");

        // Spacebar pause: Neural dominant (alpha = 0.6)
        let blended_pause = predictor.blend_candidates(&ngrams, &neurals, 0.6);
        assert_eq!(blended_pause[0], "আসব");
        assert_eq!(blended_pause[1], "যাব");
    }
}
