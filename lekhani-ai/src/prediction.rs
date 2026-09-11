//! Zero-Preedit Next-Word Prediction Engine

use crate::lm::LanguageModel;

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

        let last_word = context.last().unwrap_or(&"");
        let mut predictions = self.lm.get_next_words(last_word, limit);

        // If context has 2+ words, check if trigram continuations exist
        if context.len() >= 2 {
            let prev2 = context[context.len() - 2];
            let prev1 = context[context.len() - 1];

            // Rerank or boost trigram-matching predictions
            predictions.sort_by(|a, b| {
                let score_a = self.lm.score_candidate(Some(prev2), Some(prev1), a);
                let score_b = self.lm.score_candidate(Some(prev2), Some(prev1), b);
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        if predictions.len() < limit {
            let fallbacks = [
                "হবে", "আছে", "করব", "যাব", "ভালো", "চাই", "কথা", "দেখা", "ছিল", "করছি",
            ];
            for fb in fallbacks {
                if predictions.len() >= limit {
                    break;
                }
                if !predictions.contains(&fb.to_string()) && !context.contains(&fb) {
                    predictions.push(fb.to_string());
                }
            }
        }

        predictions
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
    }
}
