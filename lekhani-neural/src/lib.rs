//! Lekhani Neural Language Model & Semantic Prediction Engine
//!
//! Provides asynchronous 2-layer INT8 GRU inference for long-range intent-driven
//! next-word prediction and bilingual Bengali-English code-mixing.

pub mod model;
pub mod predictor;
pub mod vocab;

pub use model::{GruLayerWeights, MicroGruModel};
pub use predictor::{compute_neural_alpha, NeuralCandidate, NeuralContextPredictor};
pub use vocab::{
    BpeVocabulary, BOS_TOKEN, BOS_TOKEN_ID, EOS_TOKEN, EOS_TOKEN_ID, PAD_TOKEN, PAD_TOKEN_ID,
    UNK_TOKEN, UNK_TOKEN_ID,
};
