//! LPS v1.0 Stage 2: Orthography & Licit Clusters
//!
//! Realizes pronunciation units into canonical NFC Unicode.
//! Enforces the licit-cluster table (Section 4), Reph, Phalas, ZWJ/ZWNJ shaping,
//! and variant sets for mobile chips and desktop Alt-navigation.

use super::stage1::{PronunciationUnit, Stage1Path};

/// A realized candidate word with its rule cost and letter-level variant chips.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LpsCandidate {
    pub text: String,
    pub rule_cost: u32,
    pub chips: Vec<LpsLetterChip>,
}

/// Letter-level ambiguous chip group for UI presentation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LpsLetterChip {
    pub char_index: usize,
    pub active: String,
    pub siblings: Vec<String>,
}

/// Checks if a 2-consonant or 3-consonant cluster is in the licit cluster table.
pub fn is_licit_cluster(c1: &str, c2: &str) -> bool {
    matches!(
        (c1, c2),
        ("ক", "ত") // ক্ত
            | ("ক", "ক") // ক্ক
            | ("ত", "ত") // ত্ত
            | ("ত", "ন") // ত্ন
            | ("প", "ত") // প্ত
            | ("ব", "দ") // ব্দ
            | ("ন", "ত") // ন্ত
            | ("ন", "দ") // ন্দ
            | ("ন", "ধ") // ন্ধ
            | ("ন", "ন") // ন্ন
            | ("ণ", "ট") // ণ্ট
            | ("ণ", "ঠ") // ণ্ঠ
            | ("ণ", "ড") // ণ্ড
            | ("ঙ", "ক") // ঙ্ক
            | ("ঙ", "খ") // ঙ্খ
            | ("ঙ", "গ") // ঙ্গ
            | ("ঙ", "ঘ") // ঙ্ঘ
            | ("ঞ", "চ") // ঞ্চ
            | ("ঞ", "ছ") // ঞ্ছ
            | ("ঞ", "জ") // ঞ্জ
            | ("ঞ", "ঝ") // ঞ্ঝ
            | ("ম", "প") // ম্প
            | ("ম", "ফ") // ম্ফ
            | ("ম", "ব") // ম্ব
            | ("ম", "ভ") // ম্ভ
            | ("ম", "ম") // ম্ম
            | ("স", "ত") // স্ত
            | ("স", "থ") // স্থ
            | ("স", "ক") // স্ক
            | ("স", "প") // স্প
            | ("স", "ফ") // স্ফ
            | ("স", "ম") // স্ম
            | ("ষ", "ট") // ষ্ট
            | ("ষ", "ঠ") // ষ্ঠ
            | ("ষ", "ণ") // ষ্ণ
            | ("ষ", "প") // ষ্প
            | ("ষ", "ফ") // ষ্ফ
            | ("ষ", "ম") // ষ্ম
            | ("শ", "চ") // শ্চ
            | ("শ", "ছ") // শ্ছ
            | ("শ", "ব") // শ্ব
            | ("শ", "ম") // শ্ম
            | ("হ", "ম") // হ্ম
            | ("হ", "ন") // হ্ন
            | ("হ", "ণ") // হ্ণ
            | ("হ", "ল") // হ্ল
            | ("ল", "প") // ল্প
            | ("ল", "ল") // ল্ল
            | ("ল", "ক") // ল্ক
            | ("ল", "গ") // ল্গ
            | ("গ", "গ") // গ্গ
            | ("জ", "ঞ") // জ্ঞ
            | ("ব", "ধ") // ব্ধ
            | ("ব", "ব") // ব্ব
            | ("দ", "দ") // দ্দ
            | ("দ", "ধ") // দ্ধ
            | ("দ", "ব") // দ্ব
            | ("ট", "ট") // ট্ট
            | ("ড", "ড") // ড্ড
    )
}

/// Realizes a Stage 1 pronunciation path into canonical NFC Bengali text.
pub fn realize_path(path: &Stage1Path) -> LpsCandidate {
    let mut out = String::with_capacity(path.units.len() * 3);
    let mut chips = Vec::new();

    let n = path.units.len();
    let mut i = 0;

    while i < n {
        let has_chandra = path.has_chandrabindu.get(i).copied().unwrap_or(false);

        match &path.units[i] {
            PronunciationUnit::ZWJ => {
                out.push('\u{200D}');
                i += 1;
            }
            PronunciationUnit::ZWNJ => {
                out.push('\u{200C}');
                i += 1;
            }
            PronunciationUnit::Literal(lit) => {
                out.push_str(lit);
                if has_chandra {
                    out.push('ঁ');
                }
                i += 1;
            }
            PronunciationUnit::EnglishRun(eng) => {
                out.push_str(eng);
                i += 1;
            }
            PronunciationUnit::INH | PronunciationUnit::SPLIT => {
                // Syllable boundary / split marker; emits no visual glyph
                i += 1;
            }
            PronunciationUnit::HAL => {
                // Explicit hasanta
                out.push('্');
                i += 1;
            }
            PronunciationUnit::Vowel {
                independent,
                dependent,
                active_independent,
            } => {
                let v_glyph = if *active_independent {
                    *independent
                } else {
                    *dependent
                };
                out.push_str(v_glyph);
                if has_chandra {
                    out.push('ঁ');
                }
                i += 1;
            }
            PronunciationUnit::Consonant {
                active,
                variants,
                ..
            } => {
                let c_start_idx = out.chars().count();

                // Check Reph: r + hasanta + Consonant (when r is followed by consonant without split)
                if active == "র" && i + 1 < n {
                    if let PronunciationUnit::Consonant { active: ref next_c, .. } = path.units[i + 1] {
                        if next_c != "র" && next_c != "ড়" && next_c != "ঢ়" {
                            // Render reph: র + ্ + C
                            out.push('র');
                            out.push('্');
                            i += 1;
                            continue;
                        }
                    }
                }

                // Check forced hasanta (HAL) directly following
                let forced_hasanta = i + 1 < n && matches!(path.units[i + 1], PronunciationUnit::HAL);

                // Check licit cluster with next consonant
                let mut form_cluster = false;
                if !forced_hasanta && i + 1 < n {
                    if let PronunciationUnit::Consonant { active: ref next_c, .. } = path.units[i + 1] {
                        if is_licit_cluster(active, next_c) {
                            form_cluster = true;
                        }
                    }
                }

                out.push_str(active);

                if variants.len() > 1 {
                    chips.push(LpsLetterChip {
                        char_index: c_start_idx,
                        active: active.clone(),
                        siblings: variants.iter().map(|s| s.to_string()).collect(),
                    });
                }

                if has_chandra {
                    out.push('ঁ');
                }

                if form_cluster {
                    out.push('্');
                }

                i += 1;
            }
        }
    }

    LpsCandidate {
        text: out,
        rule_cost: path.rule_cost,
        chips,
    }
}

/// Converts a set of Stage 1 paths into sorted, deduplicated Stage 2 candidates.
pub fn realize_candidates(paths: &[Stage1Path]) -> Vec<LpsCandidate> {
    let mut candidates = Vec::with_capacity(paths.len());
    let mut seen = std::collections::HashSet::new();

    for path in paths {
        let cand = realize_path(path);
        if !cand.text.is_empty() && seen.insert(cand.text.clone()) {
            candidates.push(cand);
        }
    }

    candidates.sort_by_key(|c| c.rule_cost);
    candidates
}
