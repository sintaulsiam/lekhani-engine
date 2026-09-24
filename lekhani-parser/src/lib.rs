//! # Lekhani Parser
//!
//! Ultra-fast, zero-allocation pure Rust Bengali phonetic grammar parser and transliterator.
//! High-performance replacement for `rupantor` with Longest-Prefix L1 Trie matching,
//! branchless byte-level condition evaluations, and canonical diacritic sequencing.

pub mod compiler;
pub mod engine;
pub mod rule;
pub mod tables;
pub mod trie;

pub use compiler::CompiledLayout;
pub use engine::LekhaniParser;
pub use rule::{ConditionScope, MatchType, Pattern, PatternRule, RuleCondition};

use std::sync::{Arc, OnceLock};

/// Cached singleton of the compiled default Avro Phonetic layout.
/// Initialized once, zero allocations and zero disk I/O on subsequent accesses.
static DEFAULT_AVRO_LAYOUT: OnceLock<Arc<CompiledLayout>> = OnceLock::new();

/// Embedded standard Avro Phonetic layout JSON
const EMBEDDED_AVRO_JSON: &str = include_str!("../data/avrophonetic.json");

/// Retrieve or initialize the shared singleton of the compiled default Avro layout.
pub fn get_default_avro_layout() -> Arc<CompiledLayout> {
    DEFAULT_AVRO_LAYOUT
        .get_or_init(|| {
            let json: serde_json::Value = serde_json::from_str(EMBEDDED_AVRO_JSON)
                .expect("Corrupt embedded avrophonetic.json layout");
            let layout = CompiledLayout::from_json(&json)
                .expect("Failed to compile embedded avrophonetic.json");
            Arc::new(layout)
        })
        .clone()
}

/// Create a new `LekhaniParser` with the default Avro Phonetic layout.
pub fn default_avro_parser() -> LekhaniParser {
    LekhaniParser::new(get_default_avro_layout())
}
