//! # Lekhani Phonetic Standard (LPS) v1.0 Grammar Engine
//!
//! Two-stage pronunciation and orthography lattice engine implementing the formal LPS specification.
//!
//! Pipeline:
//! Latin edits -> Stage 0 (tokens & operators) -> Stage 1 (pronunciation lattice) ->
//! Stage 2 (orthography & licit clusters) -> Canonical NFC candidates & letter chips.

pub mod exact;
pub mod stage0;
pub mod stage1;
pub mod stage2;

pub use exact::exact;
pub use stage0::{tokenize_stage0, LpsWarning, Stage0Result, Stage0Token, TokenKind};
pub use stage1::{expand_stage1, PronunciationUnit, Stage1Path};
pub use stage2::{is_licit_cluster, realize_candidates, realize_path, LpsCandidate, LpsLetterChip};

/// Headless, high-performance LPS parser.
#[derive(Debug, Default, Clone)]
pub struct LpsEngine;

impl LpsEngine {
    pub fn new() -> Self {
        Self
    }

    /// Transliterates Latin input into ordered Bengali candidates with letter-level chips and diagnostics.
    pub fn parse(&self, input: &str) -> (Vec<LpsCandidate>, Vec<LpsWarning>) {
        if input.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let stage0 = tokenize_stage0(input);
        let mut all_paths = expand_stage1(&stage0.primary_tokens);

        // Include alternative segmentations if available
        for alt_tokens in &stage0.alternative_segmentations {
            let alt_paths = expand_stage1(alt_tokens);
            for mut ap in alt_paths {
                ap.rule_cost += 1; // delta_split cost
                all_paths.push(ap);
            }
        }

        let candidates = realize_candidates(&all_paths);
        (candidates, stage0.diagnostics)
    }

    /// Convenience helper returning candidate text strings.
    pub fn convert_to_strings(&self, input: &str) -> Vec<String> {
        let (cands, _) = self.parse(input);
        cands.into_iter().map(|c| c.text).collect()
    }
}
