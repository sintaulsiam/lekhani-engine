//! LPS v1.0 Stage 1: Pronunciation Lattice
//!
//! Turns Stage 0 tokens into a sequence of pronunciation units:
//! consonant classes, vowels, and structural markers (INH, SPLIT, HAL).
//! Encodes S1–S5 inherent vowel branching, diphthong splits, and context rules.

use super::stage0::{Stage0Token, TokenKind};

/// Structural marker in pronunciation stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PronunciationUnit {
    /// Consonant phoneme with canonical glyph, variant options, and lock state
    Consonant {
        primary: String,
        variants: Vec<String>,
        active: String,
        is_cluster_head: bool,
    },
    /// Vowel phoneme with independent and dependent forms
    Vowel {
        independent: &'static str,
        dependent: &'static str,
        active_independent: bool,
    },
    /// Inherent vowel (no visual sign, syllable boundary, prevents cluster)
    INH,
    /// Implied syllable boundary splitting a potential cluster
    SPLIT,
    /// Forced hasanta
    HAL,
    /// Exact raw literal character (e.g. from escape \u, punctuation, or fallback)
    Literal(String),
    /// Zero-Width Joiner (ZWJ, U+200D)
    ZWJ,
    /// Zero-Width Non-Joiner (ZWNJ, U+200C)
    ZWNJ,
    /// English text run from \e
    EnglishRun(String),
}

/// Represents a candidate pronunciation path through the Stage 1 lattice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage1Path {
    pub units: Vec<PronunciationUnit>,
    pub has_chandrabindu: Vec<bool>,
    /// Accumulated rule cost (0 = primary, higher = alternative branching)
    pub rule_cost: u32,
}

/// Expands a sequence of Stage 0 tokens into candidate Stage 1 pronunciation paths.
pub fn expand_stage1(tokens: &[Stage0Token]) -> Vec<Stage1Path> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut paths: Vec<Stage1Path> = vec![Stage1Path {
        units: Vec::with_capacity(tokens.len() * 2),
        has_chandrabindu: Vec::with_capacity(tokens.len() * 2),
        rule_cost: 0,
    }];

    for (idx, tok) in tokens.iter().enumerate() {
        let is_first = idx == 0;
        let is_last = idx + 1 == tokens.len();
        let prev_tok = if idx > 0 {
            Some(&tokens[idx - 1])
        } else {
            None
        };
        let next_tok = if idx + 1 < tokens.len() {
            Some(&tokens[idx + 1])
        } else {
            None
        };

        let mut next_paths = Vec::with_capacity(paths.len() * 2);

        for path in paths {
            match &tok.kind {
                TokenKind::ZWJ => {
                    let mut p = path.clone();
                    p.units.push(PronunciationUnit::ZWJ);
                    p.has_chandrabindu.push(false);
                    next_paths.push(p);
                }
                TokenKind::ZWNJ => {
                    let mut p = path.clone();
                    p.units.push(PronunciationUnit::ZWNJ);
                    p.has_chandrabindu.push(false);
                    next_paths.push(p);
                }
                TokenKind::ExactChar(ch) => {
                    let mut p = path.clone();
                    p.units.push(PronunciationUnit::Literal(ch.to_string()));
                    p.has_chandrabindu.push(tok.has_chandrabindu);
                    next_paths.push(p);
                }
                TokenKind::Literal(lit) => {
                    let mut p = path.clone();
                    p.units.push(PronunciationUnit::Literal(lit.clone()));
                    p.has_chandrabindu.push(tok.has_chandrabindu);
                    next_paths.push(p);
                }
                TokenKind::EnglishRun(eng) => {
                    let mut p = path.clone();
                    p.units.push(PronunciationUnit::EnglishRun(eng.clone()));
                    p.has_chandrabindu.push(false);
                    next_paths.push(p);
                }
                TokenKind::Phonetic(raw_p) => {
                    let branches = resolve_phonetic_token(
                        raw_p,
                        tok.lock_count,
                        tok.has_split_before,
                        is_first,
                        is_last,
                        prev_tok,
                        next_tok,
                        &path,
                    );

                    for (branch_units, branch_cost) in branches {
                        let mut p = path.clone();
                        p.rule_cost += branch_cost;
                        let n_units = branch_units.len();
                        p.units.extend(branch_units);
                        for b_idx in 0..n_units {
                            if b_idx + 1 == n_units {
                                p.has_chandrabindu.push(tok.has_chandrabindu);
                            } else {
                                p.has_chandrabindu.push(false);
                            }
                        }
                        if tok.has_join {
                            p.units.push(PronunciationUnit::HAL);
                            p.has_chandrabindu.push(false);
                        }
                        next_paths.push(p);
                    }
                }
            }
        }

        // Bounded beam pruning (limit path explosion per token step)
        if next_paths.len() > 32 {
            next_paths.sort_by_key(|p| p.rule_cost);
            next_paths.truncate(32);
        }

        paths = next_paths;
    }

    paths
}

/// Resolves a single phonetic token into candidate pronunciation unit sequences and their rule costs.
#[allow(clippy::too_many_arguments)]
fn resolve_phonetic_token(
    tok: &str,
    lock_count: u8,
    has_split_before: bool,
    is_first: bool,
    is_last: bool,
    _prev_tok: Option<&Stage0Token>,
    next_tok: Option<&Stage0Token>,
    current_path: &Stage1Path,
) -> Vec<(Vec<PronunciationUnit>, u32)> {
    let mut branches = Vec::new();

    let is_prev_r = current_path.units.last().is_some_and(|u| {
        if let PronunciationUnit::Consonant { ref active, .. } = u {
            active == "র"
        } else {
            false
        }
    });

    // Check if preceding unit is a consonant or requires independent vowel
    let precedes_independent = if has_split_before || is_first {
        if has_split_before && is_prev_r && tok == "i" {
            false // r-i forces the split of digraph ri into র + ি
        } else {
            true
        }
    } else if let Some(last_unit) = current_path.units.last() {
        matches!(
            last_unit,
            PronunciationUnit::Vowel { .. }
                | PronunciationUnit::INH
                | PronunciationUnit::SPLIT
                | PronunciationUnit::Literal(_)
                | PronunciationUnit::EnglishRun(_)
        )
    } else {
        true
    };

    // 1. VOWELS
    match tok {
        "a" => {
            branches.push((
                vec![PronunciationUnit::Vowel {
                    independent: "আ",
                    dependent: "া",
                    active_independent: precedes_independent,
                }],
                0,
            ));
            return branches;
        }
        "i" => {
            branches.push((
                vec![PronunciationUnit::Vowel {
                    independent: "ই",
                    dependent: "ি",
                    active_independent: precedes_independent,
                }],
                0,
            ));
            if has_split_before && is_prev_r {
                // Also provide independent vowel as secondary variant
                branches.push((
                    vec![PronunciationUnit::Vowel {
                        independent: "ই",
                        dependent: "ি",
                        active_independent: true,
                    }],
                    1,
                ));
            }
            return branches;
        }
        "ii" => {
            branches.push((
                vec![PronunciationUnit::Vowel {
                    independent: "ঈ",
                    dependent: "ী",
                    active_independent: precedes_independent,
                }],
                0,
            ));
            return branches;
        }
        "u" => {
            branches.push((
                vec![PronunciationUnit::Vowel {
                    independent: "উ",
                    dependent: "ু",
                    active_independent: precedes_independent,
                }],
                0,
            ));
            return branches;
        }
        "uu" => {
            branches.push((
                vec![PronunciationUnit::Vowel {
                    independent: "ঊ",
                    dependent: "ূ",
                    active_independent: precedes_independent,
                }],
                0,
            ));
            return branches;
        }
        "e" => {
            branches.push((
                vec![PronunciationUnit::Vowel {
                    independent: "এ",
                    dependent: "ে",
                    active_independent: precedes_independent,
                }],
                0,
            ));
            return branches;
        }
        "oo" => {
            branches.push((
                vec![PronunciationUnit::Vowel {
                    independent: "ও",
                    dependent: "ো",
                    active_independent: precedes_independent,
                }],
                0,
            ));
            return branches;
        }
        "ae" => {
            branches.push((
                vec![PronunciationUnit::Vowel {
                    independent: "অ্যা",
                    dependent: "্যা",
                    active_independent: precedes_independent,
                }],
                0,
            ));
            return branches;
        }
        "o" => {
            if lock_count == 1 {
                // o' -> bare অ, or INH after consonant
                if precedes_independent {
                    branches.push((vec![PronunciationUnit::Literal("অ".to_string())], 0));
                } else {
                    branches.push((vec![PronunciationUnit::INH], 0));
                }
            } else if precedes_independent {
                // S4. Word-initial o branches into অ or ও
                branches.push((vec![PronunciationUnit::Literal("অ".to_string())], 0));
                branches.push((vec![PronunciationUnit::Literal("ও".to_string())], 1));
            } else {
                // S1 & S3. o after consonant branches into INH (no sign) or ো
                branches.push((vec![PronunciationUnit::INH], 0));
                branches.push((
                    vec![PronunciationUnit::Vowel {
                        independent: "ও",
                        dependent: "ো",
                        active_independent: false,
                    }],
                    if is_last { 0 } else { 1 },
                ));
            }
            return branches;
        }
        "oi" => {
            if lock_count == 1 {
                branches.push((
                    vec![PronunciationUnit::Vowel {
                        independent: "ঐ",
                        dependent: "ৈ",
                        active_independent: precedes_independent,
                    }],
                    0,
                ));
            } else {
                // Unlocked oi branches into ঐ/ৈ and o + i (producing boi -> বই and বৈ)
                branches.push((
                    vec![
                        PronunciationUnit::INH,
                        PronunciationUnit::Vowel {
                            independent: "ই",
                            dependent: "ি",
                            active_independent: true,
                        },
                    ],
                    0,
                ));
                branches.push((
                    vec![PronunciationUnit::Vowel {
                        independent: "ঐ",
                        dependent: "ৈ",
                        active_independent: precedes_independent,
                    }],
                    1,
                ));
            }
            return branches;
        }
        "ou" => {
            if lock_count == 1 {
                branches.push((
                    vec![PronunciationUnit::Vowel {
                        independent: "ঔ",
                        dependent: "ৌ",
                        active_independent: precedes_independent,
                    }],
                    0,
                ));
            } else {
                // Unlocked ou branches into ঔ/ৌ and o + u (producing bou -> বউ and বৌ)
                branches.push((
                    vec![
                        PronunciationUnit::INH,
                        PronunciationUnit::Vowel {
                            independent: "উ",
                            dependent: "ু",
                            active_independent: true,
                        },
                    ],
                    0,
                ));
                branches.push((
                    vec![PronunciationUnit::Vowel {
                        independent: "ঔ",
                        dependent: "ৌ",
                        active_independent: precedes_independent,
                    }],
                    1,
                ));
            }
            return branches;
        }
        "ri" => {
            if lock_count == 1 {
                branches.push((
                    vec![PronunciationUnit::Vowel {
                        independent: "ঋ",
                        dependent: "ৃ",
                        active_independent: precedes_independent,
                    }],
                    0,
                ));
            } else if has_split_before {
                // r-i forces the split (র + ি)
                branches.push((
                    vec![
                        PronunciationUnit::Consonant {
                            primary: "র".to_string(),
                            variants: vec!["র".to_string(), "ড়".to_string()],
                            active: "র".to_string(),
                            is_cluster_head: false,
                        },
                        PronunciationUnit::Vowel {
                            independent: "ই",
                            dependent: "ি",
                            active_independent: false,
                        },
                    ],
                    0,
                ));
            } else {
                // ri branches into ঋ/ৃ and র + ি
                branches.push((
                    vec![PronunciationUnit::Vowel {
                        independent: "ঋ",
                        dependent: "ৃ",
                        active_independent: precedes_independent,
                    }],
                    0,
                ));
                branches.push((
                    vec![
                        PronunciationUnit::Consonant {
                            primary: "র".to_string(),
                            variants: vec!["র".to_string(), "ড়".to_string()],
                            active: "র".to_string(),
                            is_cluster_head: false,
                        },
                        PronunciationUnit::Vowel {
                            independent: "ই",
                            dependent: "ি",
                            active_independent: false,
                        },
                    ],
                    1,
                ));
            }
            return branches;
        }
        _ => {}
    }

    // 2. CONSONANTS
    let is_prev_consonant = current_path
        .units
        .last()
        .is_some_and(|u| matches!(u, PronunciationUnit::Consonant { .. }));

    // Check S2: Consonant directly following consonant without INH/SPLIT branches into CLUSTER or SPLIT

    let (c_active, c_primary, c_variants) = match tok {
        "k" => ("ক", "ক", &["ক"][..]),
        "kh" => ("খ", "খ", &["খ"][..]),
        "g" => ("গ", "গ", &["গ"][..]),
        "gh" => ("ঘ", "ঘ", &["ঘ"][..]),
        "c" | "ch" => ("চ", "চ", &["চ"][..]),
        "chh" => ("ছ", "ছ", &["ছ"][..]),
        "j" => {
            if lock_count == 1 {
                ("য", "জ", &["জ", "য"][..])
            } else {
                ("জ", "জ", &["জ", "য"][..])
            }
        }
        "jh" => ("ঝ", "ঝ", &["ঝ"][..]),
        "ny" => ("ঞ", "ঞ", &["ঞ"][..]),
        "t" => {
            if lock_count == 1 {
                ("ট", "ত", &["ত", "ট", "ৎ"][..])
            } else if lock_count == 2 {
                ("ৎ", "ত", &["ত", "ট", "ৎ"][..])
            } else {
                ("ত", "ত", &["ত", "ট", "ৎ"][..])
            }
        }
        "th" => {
            if lock_count == 1 {
                ("ঠ", "থ", &["থ", "ঠ"][..])
            } else {
                ("থ", "থ", &["থ", "ঠ"][..])
            }
        }
        "d" => {
            if lock_count == 1 {
                ("ড", "দ", &["দ", "ড"][..])
            } else {
                ("দ", "দ", &["দ", "ড"][..])
            }
        }
        "dh" => {
            if lock_count == 1 {
                ("ঢ", "ধ", &["ধ", "ঢ"][..])
            } else {
                ("ধ", "ধ", &["ধ", "ঢ"][..])
            }
        }
        "n" => {
            if lock_count == 1 {
                ("ণ", "ন", &["ন", "ণ"][..])
            } else {
                ("ন", "ন", &["ন", "ণ"][..])
            }
        }
        "p" => ("প", "প", &["প"][..]),
        "ph" | "f" => ("ফ", "ফ", &["ফ"][..]),
        "b" => ("ব", "ব", &["ব"][..]),
        "bh" => ("ভ", "ভ", &["ভ"][..]),
        "m" => ("ম", "ম", &["ম"][..]),
        "r" => {
            if lock_count == 1 {
                ("ড়", "র", &["র", "ড়"][..])
            } else {
                ("র", "র", &["র", "ড়"][..])
            }
        }
        "rh" => ("ঢ়", "ঢ়", &["ঢ়"][..]),
        "l" => ("ল", "ল", &["ল"][..]),
        "s" => {
            if lock_count == 1 {
                ("ষ", "স", &["স", "শ", "ষ"][..])
            } else {
                ("স", "স", &["স", "শ", "ষ"][..])
            }
        }
        "sh" => {
            if lock_count == 1 {
                ("ষ", "শ", &["শ", "স", "ষ"][..])
            } else {
                ("শ", "শ", &["শ", "স", "ষ"][..])
            }
        }
        "h" => {
            if lock_count == 1 {
                ("ঃ", "হ", &["হ", "ঃ"][..])
            } else {
                ("হ", "হ", &["হ", "ঃ"][..])
            }
        }
        "ng" => {
            if lock_count == 1 {
                ("ঙ", "ঙ", &["ঙ", "ং"][..])
            } else if lock_count == 2 {
                ("ং", "ং", &["ঙ", "ং"][..])
            } else {
                // ng: ঙ before a vowel or velar stop (k, kh, g, gh), ং elsewhere
                let is_velar_or_vowel = next_tok.is_some_and(|nt| {
                    if let TokenKind::Phonetic(ref p) = nt.kind {
                        matches!(
                            p.as_str(),
                            "a" | "i"
                                | "ii"
                                | "u"
                                | "uu"
                                | "e"
                                | "o"
                                | "oo"
                                | "oi"
                                | "ou"
                                | "ri"
                                | "ae"
                                | "k"
                                | "kh"
                                | "g"
                                | "gh"
                        )
                    } else {
                        false
                    }
                });
                if is_velar_or_vowel {
                    ("ঙ", "ঙ", &["ঙ", "ং"][..])
                } else {
                    ("ং", "ং", &["ং", "ঙ"][..])
                }
            }
        }
        "y" => {
            // y after vowel or INH defaults to য় (e.g. hoy -> হয়). Word-initially defaults to য. After consonant defaults to য-ফলা.
            if lock_count == 1 {
                ("য়", "য", &["য়", "য"][..])
            } else if is_first {
                ("য", "য", &["য", "য়"][..])
            } else if is_prev_consonant {
                ("্য", "য", &["্য", "য়", "য"][..])
            } else {
                ("য়", "য", &["য়", "য"][..])
            }
        }
        "w" => {
            // w word-initially or after vowel defaults to ও. After consonant defaults to ব-ফলা (sw -> স্ব)
            if is_prev_consonant {
                ("্ব", "ব", &["্ব", "ও"][..])
            } else {
                ("ও", "ও", &["ও"][..])
            }
        }
        "z" => ("জ", "জ", &["জ"][..]),
        "v" => ("ভ", "ভ", &["ভ"][..]),
        "q" => ("ক", "ক", &["ক"][..]),
        "x" => ("ক্স", "ক্স", &["ক্স"][..]),
        "ksh" => ("ক্ষ", "ক্ষ", &["ক্ষ"][..]),
        "gg" => {
            // D2: Generate both জ্ঞ and গ্গ
            ("জ্ঞ", "জ্ঞ", &["জ্ঞ", "গ্গ"][..])
        }
        other => (other, other, &[][..]),
    };

    // If context is intervocalic r without lock, offer ড় as high-ranked variant (e.g. bari -> বাড়ি, বারি)
    let is_intervocalic_r = tok == "r"
        && lock_count == 0
        && !is_first
        && !is_last
        && !is_prev_consonant
        && next_tok.is_some_and(|nt| {
            if let TokenKind::Phonetic(ref p) = nt.kind {
                matches!(
                    p.as_str(),
                    "a" | "i" | "ii" | "u" | "uu" | "e" | "o" | "oo" | "oi" | "ou" | "ri" | "ae"
                )
            } else {
                false
            }
        });

    let c_primary_str = c_primary.to_string();
    let c_variants_vec: Vec<String> = c_variants.iter().map(|s| s.to_string()).collect();

    let make_unit = |act: &str| PronunciationUnit::Consonant {
        primary: c_primary_str.clone(),
        variants: c_variants_vec.clone(),
        active: act.to_string(),
        is_cluster_head: false,
    };

    // Main consonant path
    if is_prev_consonant && has_split_before {
        // Explicit '-' blocks cluster: ONLY SPLIT branch
        branches.push((vec![PronunciationUnit::SPLIT, make_unit(c_active)], 0));
    } else if is_prev_consonant {
        // Direct adjacent consonant without explicit split: offer both CLUSTER and SPLIT
        // 1. CLUSTER branch (cost 0)
        branches.push((vec![make_unit(c_active)], 0));

        // 2. SPLIT branch (cost 1)
        branches.push((vec![PronunciationUnit::SPLIT, make_unit(c_active)], 1));
    } else {
        branches.push((vec![make_unit(c_active)], 0));
    }

    if is_intervocalic_r {
        branches.push((vec![make_unit("ড়")], 0)); // Equal ranking per spec vector
    }

    if tok == "j" && lock_count == 0 {
        branches.push((vec![make_unit("য")], 0));
    }

    if tok == "sh" && lock_count == 0 {
        branches.push((vec![make_unit("ষ")], 0));
        branches.push((vec![make_unit("স")], 1));
    }

    if tok == "s" && lock_count == 0 {
        branches.push((vec![make_unit("শ")], 1));
    }

    if tok == "ng" && lock_count == 0 {
        let alt_act = if c_active == "ং" { "ঙ" } else { "ং" };
        branches.push((vec![make_unit(alt_act)], 1));
    }

    if tok == "gg" {
        branches.push((vec![make_unit("গ্গ")], 1));
    }

    branches
}
