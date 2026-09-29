//! Bengali Morphological Suffix Deconstruction & Subword Engine
//!
//! Provides root-suffix decomposition, grammatical inflection analysis,
//! and morpheme-aware probability estimation for unseen inflected Bengali words.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuffixCategory {
    Definitive,
    CaseMarker,
    VerbInflection,
    Particle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MorphemeDecomposition<'a> {
    pub root: &'a str,
    pub suffix: &'a str,
    pub category: SuffixCategory,
}

/// Definitive markers (নির্দিষ্টতাবাচক প্রত্যয়)
pub const DEFINITIVES: &[&str] = &[
    "গুলো", "গুলি", "গুলা", "খানা", "খানি", "টুকুন", "টুকু", "টা", "টি",
];

/// Nominal & pronominal case markers (কারক-বিভক্তি)
pub const CASE_MARKERS: &[&str] = &[
    "দেরকে", "দের", "ের", "কে", "রে", "তে", "েতে", "য়ে", "র", "ে",
];

/// Verb tense-aspect-person inflections (ক্রিয়া বিভক্তি)
pub const VERB_INFLECTIONS: &[&str] = &[
    "ছিলেন", "ছিলাম", "ছিলে", "ছিল",
    "ছেন", "ছি", "ছিস", "ছে", "ছো",
    "লাম", "লেন", "লে",
    "বেন", "বে", "বা", "ব",
];

/// Colloquial & emphatic clitic particles (অনুগামী প্রত্যয়)
pub const PARTICLES: &[&str] = &[
    "ই", "ও", "তো", "না",
];

/// Morpheme Analyzer for decomposing inflected Bengali words into root and suffix
#[derive(Debug, Clone, Default)]
pub struct BengaliMorphAnalyzer;

impl BengaliMorphAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Decompose an inflected Bengali word into its root stem and suffix.
    /// Returns `None` if no valid suffix pattern matches or if the root would be too short.
    pub fn decompose<'a>(&self, word: &'a str) -> Option<MorphemeDecomposition<'a>> {
        if word.chars().count() < 3 {
            return None;
        }

        // 1. Try compound definitives first (e.g. "-গুলোর", "-গুলির", "-টার", "-টির")
        for def in &["গুলোর", "গুলির", "গুলার", "টার", "টির", "খানার", "খানির"] {
            if word.ends_with(def) {
                let cut_idx = word.len() - def.len();
                let root = &word[..cut_idx];
                if root.chars().count() >= 2 {
                    return Some(MorphemeDecomposition {
                        root,
                        suffix: def,
                        category: SuffixCategory::Definitive,
                    });
                }
            }
        }

        // 2. Try standard definitives
        for &def in DEFINITIVES {
            if word.ends_with(def) {
                let cut_idx = word.len() - def.len();
                let root = &word[..cut_idx];
                if root.chars().count() >= 2 {
                    return Some(MorphemeDecomposition {
                        root,
                        suffix: def,
                        category: SuffixCategory::Definitive,
                    });
                }
            }
        }

        // 3. Try case markers (check longer suffixes first to avoid false short matches)
        for &cm in CASE_MARKERS {
            if word.ends_with(cm) {
                let cut_idx = word.len() - cm.len();
                let root = &word[..cut_idx];
                if root.chars().count() >= 2 {
                    return Some(MorphemeDecomposition {
                        root,
                        suffix: cm,
                        category: SuffixCategory::CaseMarker,
                    });
                }
            }
        }

        // 4. Try verb inflections
        for &vi in VERB_INFLECTIONS {
            if word.ends_with(vi) {
                let cut_idx = word.len() - vi.len();
                let root = &word[..cut_idx];
                if root.chars().count() >= 2 {
                    return Some(MorphemeDecomposition {
                        root,
                        suffix: vi,
                        category: SuffixCategory::VerbInflection,
                    });
                }
            }
        }

        // 5. Try emphatic particles (-ই, -ও, -তো, -না)
        for &pt in PARTICLES {
            if word.ends_with(pt) {
                let cut_idx = word.len() - pt.len();
                let root = &word[..cut_idx];
                if root.chars().count() >= 2 {
                    return Some(MorphemeDecomposition {
                        root,
                        suffix: pt,
                        category: SuffixCategory::Particle,
                    });
                }
            }
        }

        None
    }

    /// Estimate log-probability for an unseen inflected word given the root lemma's log-probability.
    /// P(word) ≈ P(root) * P(suffix|category)
    #[inline]
    pub fn estimate_inflected_score(&self, root_log_prob: f32, category: SuffixCategory) -> f32 {
        let suffix_penalty = match category {
            SuffixCategory::Definitive => -0.35,
            SuffixCategory::CaseMarker => -0.40,
            SuffixCategory::VerbInflection => -0.30,
            SuffixCategory::Particle => -0.25,
        };
        root_log_prob + suffix_penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morpheme_decomposition() {
        let analyzer = BengaliMorphAnalyzer::new();

        // Definitives
        let dec_boi_gulo = analyzer.decompose("বইগুলো").expect("Should decompose বইগুলো");
        assert_eq!(dec_boi_gulo.root, "বই");
        assert_eq!(dec_boi_gulo.suffix, "গুলো");
        assert_eq!(dec_boi_gulo.category, SuffixCategory::Definitive);

        let dec_boi_ti = analyzer.decompose("বইটি").expect("Should decompose বইটি");
        assert_eq!(dec_boi_ti.root, "বই");
        assert_eq!(dec_boi_ti.suffix, "টি");
        assert_eq!(dec_boi_ti.category, SuffixCategory::Definitive);

        // Case markers
        let dec_manush = analyzer.decompose("মানুষদের").expect("Should decompose মানুষদের");
        assert_eq!(dec_manush.root, "মানুষ");
        assert_eq!(dec_manush.suffix, "দের");
        assert_eq!(dec_manush.category, SuffixCategory::CaseMarker);

        let dec_kolom = analyzer.decompose("কলমকে").expect("Should decompose কলমকে");
        assert_eq!(dec_kolom.root, "কলম");
        assert_eq!(dec_kolom.suffix, "কে");
        assert_eq!(dec_kolom.category, SuffixCategory::CaseMarker);

        // Verb inflections
        let dec_korchi = analyzer.decompose("করছিলাম").expect("Should decompose করছিলাম");
        assert_eq!(dec_korchi.root, "কর");
        assert_eq!(dec_korchi.suffix, "ছিলাম");
        assert_eq!(dec_korchi.category, SuffixCategory::VerbInflection);

        // Subword probability estimation
        let est = analyzer.estimate_inflected_score(-2.5, SuffixCategory::Definitive);
        assert!((est - (-2.85)).abs() < 1e-5);
    }
}
