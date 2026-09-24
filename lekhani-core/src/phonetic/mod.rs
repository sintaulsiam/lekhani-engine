//! Avro Phonetic Engine Module

pub mod code_shield;
pub mod database;
pub mod fuzzy;
pub mod learner;
pub mod method;
pub mod morphology;
pub mod ranking;
pub mod reduplication;
pub mod segmenter;
pub mod suggestion;

pub use database::PhoneticDatabase;
pub use learner::AutonomousLearner;
pub use method::PhoneticMethod;
pub use ranking::{CandidateContext, RankFeatures, RankWeights, extract_candidate_features};
pub use suggestion::{PhoneticSuggestion, PhoneticSuggestionConfig};

