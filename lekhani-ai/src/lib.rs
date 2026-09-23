//! Lekhani AI: Pure Rust On-Device Context & Language Intelligence
//!
//! Provides lightweight, sub-microsecond neural-transition language modeling,
//! multi-token context scoring, homophone disambiguation, beam search decoding,
//! and next-word predictive typing for the Lekhani Bengali typing ecosystem.

pub mod beam;
pub mod context;
pub mod lm;
pub mod prediction;
pub mod trainer;

pub use beam::BeamSearchDecoder;
pub use context::ContextScorer;
pub use lm::LanguageModel;
pub use prediction::NextWordPredictor;
pub use trainer::{
    train_files_streaming, CorpusTrainer, TrainedLanguageModelData, TrainingConfig,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_ai_pipeline() {
        let lm = LanguageModel::new();
        let scorer = ContextScorer::new();
        let predictor = NextWordPredictor::new();
        let decoder = BeamSearchDecoder::new();

        // 1. Language Model Score
        let p = lm.score_candidate(Some("আমি"), Some("ভাত"), "খাচ্ছি");
        assert!(p > -2.0);

        // 2. Context Reranking
        let cands = vec!["পড়া".to_string(), "পরা".to_string()];
        let reranked = scorer.rank_candidates(&["শার্ট"], &cands);
        assert_eq!(reranked[0], "পরা");

        // 3. Next Word Prediction
        let next_words = predictor.predict_next(&["বাংলাদেশ", "একটি"], 3);
        assert!(!next_words.is_empty());

        // 4. Global Beam Decoding
        let seq = vec![
            vec!["আমি".to_string()],
            vec!["বই".to_string()],
            vec!["পরা".to_string(), "পড়া".to_string()],
        ];
        let path = decoder.decode(&seq);
        assert_eq!(path, vec!["আমি", "বই", "পড়া"]);
    }
}
