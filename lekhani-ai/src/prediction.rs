//! Zero-Preedit Next-Word Prediction Engine

use crate::lm::LanguageModel;
use crate::trainer::TrainedLanguageModelData;

#[derive(Debug, Clone, Default)]
pub struct NextWordPredictor {
    lm: LanguageModel,
}

pub const BENGALI_IDIOM_PHRASES: &[(&[&str], &[&str])] = &[
    (&["অনেক", "অনেক"], &["ধন্যবাদ", "শুভেচ্ছা ও অভিনন্দন"]),
    (&["অনেক"], &["ধন্যবাদ", "সুন্দর", "ভালো"]),
    (&["কেমন"], &["আছেন?", "আছো?", "হলো?"]),
    (&["শুভ"], &["সকাল", "সন্ধ্যা", "রাত্রি", "কামনা", "জন্মদিন", "নববর্ষ"]),
    (&["সব", "কিছু"], &["ঠিক আছে", "সুন্দর"]),
    (&["ঠিক"], &["আছে", "আছেন"]),
    (&["ইনশা"], &["আল্লাহ"]),
    (&["আলহামদু"], &["লিল্লাহ"]),
    (&["মাশা"], &["আল্লাহ"]),
    (&["খুব"], &["ভালো", "সুন্দর", "কষ্ট"]),
];

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
        self.predict_next_with_options(context, limit, true)
    }

    /// Predict the top-K probable next words with explicit scores given preceding sentence context
    pub fn predict_next_scored(&self, context: &[&str], limit: usize) -> Vec<(String, f32)> {
        self.predict_next_scored_with_options(context, limit, true)
    }

    /// Predict the top-K probable next words with scores given preceding sentence context with optional idiom phrases
    pub fn predict_next_scored_with_options(
        &self,
        context: &[&str],
        limit: usize,
        enable_idiom_phrases: bool,
    ) -> Vec<(String, f32)> {
        let context = crate::context::truncate_at_sentence_boundary(context);
        if context.is_empty() {
            let defaults = [
                ("আমি", -1.8),
                ("আপনি", -2.3),
                ("তুমি", -2.1),
                ("আমরা", -2.4),
                ("ধন্যবাদ", -2.5),
            ];
            return defaults
                .iter()
                .take(limit)
                .map(|(w, s)| (w.to_string(), *s))
                .collect();
        }

        // 0. Match high-confidence conversational idioms and phrases
        let mut idiom_matches: Vec<(String, f32)> = Vec::new();
        let mut seen: hashbrown::HashSet<String> = hashbrown::HashSet::new();

        if enable_idiom_phrases {
            for &(pattern, continuations) in BENGALI_IDIOM_PHRASES {
                if context.len() >= pattern.len() {
                    let tail = &context[context.len() - pattern.len()..];
                    if tail == pattern {
                        for (idx, &cont) in continuations.iter().enumerate() {
                            if seen.insert(cont.to_string()) {
                                idiom_matches.push((cont.to_string(), 0.5 - 0.1 * (idx as f32)));
                            }
                        }
                    }
                }
            }
        }

        let pool_size = (limit * 3).max(12);
        let mut raw_candidates: Vec<String> = Vec::with_capacity(pool_size);

        let (prev2, prev1) = if context.len() >= 2 {
            (Some(context[context.len() - 2]), Some(context[context.len() - 1]))
        } else {
            (None, Some(context[context.len() - 1]))
        };

        if let (Some(p2), Some(p1)) = (prev2, prev1) {
            // 1. Trigram continuations
            for w in self.lm.get_next_words_trigram(p2, p1, pool_size) {
                if seen.insert(w.clone()) {
                    raw_candidates.push(w);
                }
            }
        }

        if let Some(p1) = prev1 {
            // 2. Bigram continuations
            for w in self.lm.get_next_words(p1, pool_size) {
                if seen.insert(w.clone()) {
                    raw_candidates.push(w);
                }
            }
        }

        // Score candidates
        let mut scored_candidates: Vec<(String, f32)> = raw_candidates
            .into_iter()
            .map(|cand| {
                let score = self.lm.score_candidate(prev2, prev1, &cand);
                (cand, score)
            })
            .collect();

        scored_candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Fill remaining slots with high-frequency contextual fallbacks
        if idiom_matches.len() + scored_candidates.len() < limit {
            let fallbacks = [
                "হবে", "আছে", "করব", "যাব", "ভালো",
                "চাই", "কথা", "দেখা", "ছিল", "করছি",
            ];
            for fb in fallbacks {
                if idiom_matches.len() + scored_candidates.len() >= limit {
                    break;
                }
                if seen.insert(fb.to_string()) && !context.contains(&fb) {
                    let score = self.lm.score_candidate(prev2, prev1, fb);
                    scored_candidates.push((fb.to_string(), score));
                }
            }
        }

        let mut final_result = idiom_matches;
        final_result.extend(scored_candidates);
        final_result.truncate(limit);
        final_result
    }

    /// Predict the top-K probable next words given preceding sentence context with optional idiom phrases
    pub fn predict_next_with_options(
        &self,
        context: &[&str],
        limit: usize,
        enable_idiom_phrases: bool,
    ) -> Vec<String> {
        self.predict_next_scored_with_options(context, limit, enable_idiom_phrases)
            .into_iter()
            .map(|(w, _)| w)
            .collect()
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
        assert!(
            preds_ami.contains(&"মনে".to_string())
                || preds_ami.contains(&"জানি".to_string())
                || preds_ami.contains(&"আমার".to_string())
                || preds_ami.contains(&"ভালো".to_string())
                || preds_ami.contains(&"যাচ্ছি".to_string())
        );

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
