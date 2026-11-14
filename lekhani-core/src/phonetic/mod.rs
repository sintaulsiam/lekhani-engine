//! Avro Phonetic Engine Module

pub mod code_shield;
pub mod compound;
pub mod database;
pub mod fuzzy;
pub mod learner;
pub mod method;
pub mod morphology;
pub mod overrides;
pub mod pragyan;
pub mod ranking;
pub mod reduplication;
pub mod reranker;
pub mod segmenter;
pub mod suggestion;

pub use compound::CompoundSplitter;
pub use database::PhoneticDatabase;
pub use learner::AutonomousLearner;
pub use method::PhoneticMethod;
pub use overrides::{OverrideCandidate, OverrideEntry, ZeroCopyOverrides};
pub use pragyan::{JointUnitLM, PragyanConfig, PragyanEngine, UnitVocab};
pub use ranking::{extract_candidate_features, CandidateContext, RankFeatures, RankWeights};
pub use reranker::CandidateReranker;
pub use suggestion::{PhoneticGrammar, PhoneticSuggestion, PhoneticSuggestionConfig};
