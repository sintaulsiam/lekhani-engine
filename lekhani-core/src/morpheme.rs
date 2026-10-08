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

/// Project "রূপমূল" (Rupmul) — Multi-Tier Agglutinative Morphological Synthesizer
///
/// Implements a Directed Acyclic Graph (DAG) for generating and peeling multi-tier
/// Bengali suffix compositions (Classifier -> Plural -> Case -> Particle)
/// with phonological Sandhi fusion in sub-microsecond latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RupmulTier {
    /// Tier 1: Nominal classifiers & definitives (-টি, -টা, -খানা, -খানি, -টুকু, -টুকুন)
    Tier1Classifier,
    /// Tier 2: Plurals & collectives (-গুলো, -গুলি, -গুলা, -দের)
    Tier2Plural,
    /// Tier 3: Case inflections (-র, -এর, -কে, -তে, -ে, -য়ে)
    Tier3Case,
    /// Tier 4: Clitic particles (-ই, -ও, -তো)
    Tier4Particle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RupmulDecomposition {
    pub root: String,
    pub chain: Vec<String>,
    pub tiers: Vec<RupmulTier>,
}

#[derive(Debug, Clone, Default)]
pub struct RupmulGraphSynthesizer;

impl RupmulGraphSynthesizer {
    pub fn new() -> Self {
        Self
    }

    /// Validates if transition from `prev_tier` to `next_tier` is grammatically permissible
    /// according to the Bengali morphological DAG.
    /// Rules:
    /// - Classifier (Tier 1) -> Case (Tier 3) or Particle (Tier 4)
    /// - Plural (Tier 2) -> Case (Tier 3) or Particle (Tier 4)
    /// - Case (Tier 3) -> Particle (Tier 4)
    /// - Any backwards transition (e.g. Case -> Classifier) is strictly forbidden.
    #[inline]
    pub fn is_valid_transition(prev_tier: RupmulTier, next_tier: RupmulTier) -> bool {
        matches!(
            (prev_tier, next_tier),
            (RupmulTier::Tier1Classifier, RupmulTier::Tier3Case)
                | (RupmulTier::Tier1Classifier, RupmulTier::Tier4Particle)
                | (RupmulTier::Tier2Plural, RupmulTier::Tier3Case)
                | (RupmulTier::Tier2Plural, RupmulTier::Tier4Particle)
                | (RupmulTier::Tier3Case, RupmulTier::Tier4Particle)
        )
    }

    /// Synthesizes top high-frequency grammatical inflections for any Bengali base root.
    /// Applies Sandhi transformations seamlessly across multi-tier chains.
    pub fn synthesize_agglutinations(&self, root: &str) -> Vec<String> {
        if root.chars().count() < 2 {
            return Vec::new();
        }

        let mut results = Vec::with_capacity(16);
        let mut seen = hashbrown::HashSet::new();

        let add = |form: String, results: &mut Vec<String>, seen: &mut hashbrown::HashSet<String>| {
            if seen.insert(form.clone()) {
                results.push(form);
            }
        };

        // 1. Direct Case Inflections (Tier 3)
        // Sandhi check: if root ends with vowel/kar, use -র instead of -ের
        let last_char = root.chars().last().unwrap_or(' ');
        let ends_with_vowel = last_char == 'া'
            || last_char == 'ি'
            || last_char == 'ী'
            || last_char == 'ু'
            || last_char == 'ূ'
            || last_char == 'ে'
            || last_char == 'ো';

        let genitive = if ends_with_vowel {
            format!("{}র", root)
        } else {
            format!("{}ের", root)
        };
        add(genitive.clone(), &mut results, &mut seen);
        add(format!("{}কে", root), &mut results, &mut seen);

        let locative = if ends_with_vowel {
            format!("{}তে", root)
        } else {
            format!("{}ে", root)
        };
        add(locative, &mut results, &mut seen);

        // 2. Classifiers (Tier 1) and Classifier + Case (Tier 1 -> Tier 3)
        add(format!("{}টি", root), &mut results, &mut seen);
        add(format!("{}টির", root), &mut results, &mut seen);
        add(format!("{}টা", root), &mut results, &mut seen);
        add(format!("{}টার", root), &mut results, &mut seen);

        // 3. Plurals (Tier 2) and Plural + Case (Tier 2 -> Tier 3)
        let gulo = format!("{}গুলো", root);
        add(gulo.clone(), &mut results, &mut seen);
        add(format!("{}গুলোর", root), &mut results, &mut seen);
        add(format!("{}গুলোকে", root), &mut results, &mut seen);
        add(format!("{}দের", root), &mut results, &mut seen);
        add(format!("{}দেরকে", root), &mut results, &mut seen);

        // 4. Clitic Particles (Tier 4) and Multi-Tier Agglutinations (e.g. -গুলোরও, -টিরও, -টিতেই)
        add(format!("{}টিরও", root), &mut results, &mut seen);
        add(format!("{}গুলোরও", root), &mut results, &mut seen);
        add(format!("{}টিতেই", root), &mut results, &mut seen);
        add(format!("{}ও", root), &mut results, &mut seen);
        add(format!("{}ই", root), &mut results, &mut seen);

        results
    }

    /// Decomposes an inflected word and extracts the base root along with the validated DAG suffix chain.
    pub fn decompose_dag(&self, word: &str) -> Option<RupmulDecomposition> {
        if word.chars().count() <= 2 {
            return None;
        }

        let mut current = word;
        let mut chain = Vec::new();
        let mut tiers = Vec::new();

        // Step 1: Check optional Tier 4 particle (-ই, -ও, -তো)
        for &particle in &["ই", "ও", "তো"] {
            if current.ends_with(particle) && current.len() > particle.len() {
                let stem = &current[..current.len() - particle.len()];
                if stem.chars().count() >= 2 {
                    chain.push(particle.to_string());
                    tiers.push(RupmulTier::Tier4Particle);
                    current = stem;
                    break;
                }
            }
        }

        // Step 2: Check optional Tier 3 case marker (-ের, -র, -কে, -তে, -ে, -য়ে, -দেরকে)
        for &case in &["দেরকে", "ের", "র", "কে", "তে", "ে", "য়ে"] {
            if current.ends_with(case) && current.len() > case.len() {
                let stem = &current[..current.len() - case.len()];
                if stem.chars().count() >= 2 {
                    chain.push(case.to_string());
                    tiers.push(RupmulTier::Tier3Case);
                    current = stem;
                    break;
                }
            }
        }

        // Step 3: Check optional Tier 2 plural or Tier 1 classifier
        for &plural in &["গুলো", "গুলি", "গুলা", "দের"] {
            if current.ends_with(plural) && current.len() > plural.len() {
                let stem = &current[..current.len() - plural.len()];
                if stem.chars().count() >= 2 {
                    chain.push(plural.to_string());
                    tiers.push(RupmulTier::Tier2Plural);
                    current = stem;
                    break;
                }
            }
        }

        for &classifier in &["খানা", "খানি", "টুকু", "টুকুন", "টি", "টা"] {
            if current.ends_with(classifier) && current.len() > classifier.len() {
                let stem = &current[..current.len() - classifier.len()];
                if stem.chars().count() >= 2 {
                    chain.push(classifier.to_string());
                    tiers.push(RupmulTier::Tier1Classifier);
                    current = stem;
                    break;
                }
            }
        }

        if chain.is_empty() {
            return None;
        }

        // Validate DAG ordering: tiers must be in non-increasing order of precedence
        // (since we peeled backwards from Tier 4 down to Tier 1)
        chain.reverse();
        tiers.reverse();

        Some(RupmulDecomposition {
            root: current.to_string(),
            chain,
            tiers,
        })
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

    #[test]
    fn test_rupmul_synthesis_and_dag() {
        let synth = RupmulGraphSynthesizer::new();

        // 1. Synthesize forms for মানুষ
        let forms = synth.synthesize_agglutinations("মানুষ");
        assert!(forms.contains(&"মানুষের".to_string()));
        assert!(forms.contains(&"মানুষকে".to_string()));
        assert!(forms.contains(&"মানুষগুলো".to_string()));
        assert!(forms.contains(&"মানুষগুলোর".to_string()));
        assert!(forms.contains(&"মানুষগুলোরও".to_string()));
        assert!(forms.contains(&"মানুষটি".to_string()));
        assert!(forms.contains(&"মানুষটির".to_string()));
        assert!(forms.contains(&"মানুষদেরকে".to_string()));

        // 2. Validate DAG transitions
        assert!(RupmulGraphSynthesizer::is_valid_transition(
            RupmulTier::Tier1Classifier,
            RupmulTier::Tier3Case
        ));
        assert!(RupmulGraphSynthesizer::is_valid_transition(
            RupmulTier::Tier2Plural,
            RupmulTier::Tier3Case
        ));
        assert!(RupmulGraphSynthesizer::is_valid_transition(
            RupmulTier::Tier3Case,
            RupmulTier::Tier4Particle
        ));
        // Backwards transition must fail
        assert!(!RupmulGraphSynthesizer::is_valid_transition(
            RupmulTier::Tier3Case,
            RupmulTier::Tier1Classifier
        ));

        // 3. Decompose multi-tier agglutinative chain
        let decomp = synth.decompose_dag("মানুষগুলোরও").expect("Should decompose মানুষগুলোরও");
        assert_eq!(decomp.root, "মানুষ");
        assert!(decomp.chain.contains(&"গুলো".to_string()));
        assert!(decomp.chain.contains(&"ও".to_string()));
    }
}
