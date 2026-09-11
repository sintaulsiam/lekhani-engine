//! Zero-Preedit Next-Word Prediction Engine

use crate::lm::LanguageModel;
use crate::trainer::TrainedLanguageModelData;

#[derive(Debug, Clone, Default)]
pub struct NextWordPredictor {
    lm: LanguageModel,
}

impl NextWordPredictor {
    pub fn new() -> Self {
        Self {
            lm: LanguageModel::new(),
        }
    }

    pub fn with_language_model(lm: LanguageModel) -> Self {
        Self { lm }
    }

    pub fn lm(&self) -> &LanguageModel {
        &self.lm
    }

    pub fn lm_mut(&mut self) -> &mut LanguageModel {
        &mut self.lm
    }

    /// Ingest raw text and train the underlying language model
    pub fn train_text(&mut self, text: &str) {
        self.lm.train_text(text);
    }

    /// Load pre-compiled dataset into the underlying language model
    pub fn load_trained_data(&mut self, data: &TrainedLanguageModelData) {
        self.lm.load_trained_data(data);
    }

    /// Predict the top-K probable next words given preceding sentence context
    pub fn predict_next(&self, context: &[&str], limit: usize) -> Vec<String> {
        if context.is_empty() {
            return vec![
                "আমি".to_string(),
                "আপনি".to_string(),
                "তুমি".to_string(),
                "আমরা".to_string(),
                "ধন্যবাদ".to_string(),
            ]
            .into_iter()
            .take(limit)
            .collect();
        }

        let pool_size = (limit * 3).max(12);
        let mut candidate_set: Vec<String> = Vec::with_capacity(pool_size);

        if context.len() >= 2 {
            let prev2 = context[context.len() - 2];
            let prev1 = context[context.len() - 1];

            // 1. Trigram continuations
            for w in self.lm.get_next_words_trigram(prev2, prev1, pool_size) {
                if !candidate_set.contains(&w) {
                    candidate_set.push(w);
                }
            }

            // 2. Bigram continuations
            for w in self.lm.get_next_words(prev1, pool_size) {
                if !candidate_set.contains(&w) {
                    candidate_set.push(w);
                }
            }

            // Score and sort candidates
            candidate_set.sort_by(|a, b| {
                let score_a = self.lm.score_candidate(Some(prev2), Some(prev1), a);
                let score_b = self.lm.score_candidate(Some(prev2), Some(prev1), b);
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            let prev1 = context[context.len() - 1];

            // Bigram continuations
            for w in self.lm.get_next_words(prev1, pool_size) {
                if !candidate_set.contains(&w) {
                    candidate_set.push(w);
                }
            }

            // Score and sort candidates
            candidate_set.sort_by(|a, b| {
                let score_a = self.lm.score_candidate(None, Some(prev1), a);
                let score_b = self.lm.score_candidate(None, Some(prev1), b);
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Fill remaining slots with high-frequency contextual fallbacks
        if candidate_set.len() < limit {
            let fallbacks = [
                "হবে",
                "আছে",
                "করব",
                "যাব",
                "ভালো",
                "চাই",
                "কথা",
                "দেখা",
                "ছিল",
                "করছি",
            ];
            for fb in fallbacks {
                if candidate_set.len() >= limit {
                    break;
                }
                if !candidate_set.contains(&fb.to_string()) && !context.contains(&fb) {
                    candidate_set.push(fb.to_string());
                }
            }
        }

        candidate_set.truncate(limit);
        candidate_set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_continuations() {
        let predictor = NextWordPredictor::new();

        let preds_ami = predictor.predict_next(&["আমি"], 5);
        assert!(!preds_ami.is_empty());
        assert!(preds_ami.contains(&"ভালো".to_string()));

        let preds_rice = predictor.predict_next(&["আমি", "ভাত"], 3);
        assert!(!preds_rice.is_empty());
        assert!(preds_rice.contains(&"খাচ্ছি".to_string()) || preds_rice.contains(&"খাব".to_string()));

        let preds_bd = predictor.predict_next(&["বাংলাদেশ", "একটি"], 3);
        assert!(!preds_bd.is_empty());
        assert!(preds_bd.contains(&"সুন্দর".to_string()) || preds_bd.contains(&"স্বাধীন".to_string()));
    }

    #[test]
    fn test_dynamic_corpus_training_prediction() {
        let mut predictor = NextWordPredictor::new();
        predictor.train_text("বাংলা আমার অহংকার। বাংলা আমার মাতৃভাষা।");

        let preds = predictor.predict_next(&["বাংলা", "আমার"], 3);
        assert!(!preds.is_empty());
        assert!(preds.contains(&"অহংকার".to_string()) || preds.contains(&"মাতৃভাষা".to_string()));
    }
}
