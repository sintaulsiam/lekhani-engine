//! Transliteration Pipeline Pattern for Lekhani Parser
//!
//! Organizes transliteration into composable, isolated pipeline stages:
//! 1. `PreNormalizeStage`: Latin input pre-normalization (e.g. Chandra Bindu shift)
//! 2. `GrammarTransliterateStage`: Pattern & Trie matching grammar emission
//! 3. `PostCanonicalizeStage`: Combining marks reordering and Unicode NFC normalization

use crate::engine::{canonicalize_combining_marks, normalize_chandra_latin, LekhaniParser};

/// A pipeline coordinating transliteration stages.
pub struct TransliterationPipeline<'a> {
    parser: &'a LekhaniParser,
}

impl<'a> TransliterationPipeline<'a> {
    #[inline]
    pub fn new(parser: &'a LekhaniParser) -> Self {
        Self { parser }
    }

    /// Run the full 3-stage transliteration pipeline with zero heap allocation on the hot path.
    #[inline]
    pub fn process(&self, input: &str, output: &mut String) {
        if input.is_empty() {
            return;
        }

        // Stage 1: Pre-normalization
        let norm_storage: String;
        let effective_input = if input.as_bytes().contains(&b'^') {
            norm_storage = normalize_chandra_latin(input);
            norm_storage.as_str()
        } else {
            input
        };

        // Stage 2: Grammar transliteration
        let has_combining = self.parser.transliterate_grammar(effective_input, output);

        // Stage 3: Unicode canonical combining mark sequencing
        if has_combining {
            canonicalize_combining_marks(output);
        }
    }

    /// Run the raw grammar pipeline without post-canonicalization.
    #[inline]
    pub fn process_raw(&self, input: &str, output: &mut String) {
        if input.is_empty() {
            return;
        }
        self.parser.transliterate_grammar(input, output);
    }
}
