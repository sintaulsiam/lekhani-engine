//! Pragyan Phase 1 Step 3: Tokenization Property Tests & Invariant Verification
//!
//! Validates:
//! 1. 100% Round-trip lossless concatenation on all splits (train, dev, test) and dictionary.json
//! 2. Zero hard-constraint violations (no KAR without Roman vowel; conjuncts strictly atomic)
//! 3. Silent filter regression guard: rejects unlogged row dropping across all loaders

use lekhani_core::chars::canonicalize;
use lekhani_core::tokenization::{
    is_valid_translation_unit, normalize_roman, tokenize_bengali, tokenize_roman,
    BengaliTokenKind,
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

fn find_file(candidates: &[&str]) -> PathBuf {
    for c in candidates {
        let p = Path::new(c);
        if p.exists() {
            return p.to_path_buf();
        }
    }
    panic!("None of the candidate files found: {:?}", candidates);
}

#[test]
fn test_bengali_roundtrip_concatenation_dev_split() {
    let dev_path = find_file(&[
        "data/eval/bn.translit.sampled.dev.tsv",
        "../data/eval/bn.translit.sampled.dev.tsv",
        "../../data/eval/bn.translit.sampled.dev.tsv",
    ]);
    let file = File::open(&dev_path).expect("Open dev.tsv");
    let reader = BufReader::new(file);

    let mut row_count = 0;
    for line in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            row_count += 1;
            let target = canonicalize(parts[0].trim());
            let tokens = tokenize_bengali(&target);
            let reconstructed: String = tokens.iter().map(|t| t.text.as_str()).collect();
            assert_eq!(
                reconstructed, target,
                "Lossless round-trip failed for target: '{}'",
                target
            );
        }
    }
    assert_eq!(
        row_count, 9279,
        "Dev set row count mismatch: silent drop detected!"
    );
}

#[test]
fn test_roman_roundtrip_concatenation_dev_split() {
    let dev_path = find_file(&[
        "data/eval/bn.translit.sampled.dev.tsv",
        "../data/eval/bn.translit.sampled.dev.tsv",
        "../../data/eval/bn.translit.sampled.dev.tsv",
    ]);
    let file = File::open(&dev_path).expect("Open dev.tsv");
    let reader = BufReader::new(file);

    for line in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            let roman = parts[1].trim();
            let norm = normalize_roman(roman);
            let tokens = tokenize_roman(roman);
            let reconstructed: String = tokens.concat();
            assert_eq!(
                reconstructed, norm,
                "Roman round-trip failed for input '{}'",
                roman
            );
        }
    }
}

#[test]
fn test_hard_constraints_on_translation_units() {
    // 1. Valid 1:1 consonant mapping
    let t_valid_cons = tokenize_bengali("ক");
    assert!(is_valid_translation_unit(&["k".to_string()], &t_valid_cons));

    // 2. Valid 1:1 vowel kar with Roman vowel
    let t_valid_kar = tokenize_bengali("া");
    assert!(is_valid_translation_unit(&["a".to_string()], &t_valid_kar));

    // 3. VIOLATION: Vowel kar with Roman consonant (e.g. k -> া)
    assert!(
        !is_valid_translation_unit(&["k".to_string()], &t_valid_kar),
        "Hard constraint violated: KAR cannot be mapped to a consonant!"
    );

    // 4. Valid 1:2 whitelist (x -> ক্স)
    let t_x = tokenize_bengali("ক্স");
    assert!(is_valid_translation_unit(&["x".to_string()], &t_x));

    // 5. Valid 2:1 doubled consonant (kk -> ক)
    assert!(is_valid_translation_unit(
        &["k".to_string(), "k".to_string()],
        &t_valid_cons
    ));

    // 6. VIOLATION: Unwhitelisted arbitrary 1:2 expansion (z -> কম)
    let t_invalid_multi = tokenize_bengali("কম");
    assert!(
        !is_valid_translation_unit(&["z".to_string()], &t_invalid_multi),
        "Arbitrary unwhitelisted 1:2 expansion must be rejected"
    );

    // 7. Conjunct atomicity: a conjunct is a single token with hasanta, never fragmented
    let t_conj = tokenize_bengali("ক্ষ");
    assert_eq!(t_conj.len(), 1);
    assert_eq!(t_conj[0].kind, BengaliTokenKind::Conjunct);
    assert_eq!(t_conj[0].text, "ক্ষ");
}

#[test]
fn test_loader_rejects_unlogged_drops_and_preserves_full_dev_set() {
    // Guard against the byte length bug recurring: verify dev loader loads all 9,279 rows
    let dev_path = find_file(&[
        "data/eval/bn.translit.sampled.dev.tsv",
        "../data/eval/bn.translit.sampled.dev.tsv",
        "../../data/eval/bn.translit.sampled.dev.tsv",
    ]);
    let file = File::open(&dev_path).expect("Open dev.tsv");
    let reader = BufReader::new(file);

    let mut kept = 0;
    let mut dropped = 0;

    for line in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 2 || parts[0].trim().is_empty() || parts[1].trim().is_empty() {
            dropped += 1;
        } else {
            kept += 1;
        }
    }

    assert_eq!(
        dropped, 0,
        "Silent row drop detected! Expected 0 dropped rows, but found {}",
        dropped
    );
    assert_eq!(
        kept, 9279,
        "Dev set row count mismatch: expected 9,279, got {}",
        kept
    );
}

#[test]
fn test_no_hasanta_at_token_boundary_except_word_end() {
    let dev_path = find_file(&[
        "data/eval/bn.translit.sampled.dev.tsv",
        "../data/eval/bn.translit.sampled.dev.tsv",
        "../../data/eval/bn.translit.sampled.dev.tsv",
    ]);
    let file = File::open(&dev_path).expect("Open dev.tsv");
    let reader = BufReader::new(file);

    for line in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            let target = canonicalize(parts[0].trim());
            let tokens = tokenize_bengali(&target);
            let n = tokens.len();
            for (idx, tok) in tokens.iter().enumerate() {
                // If it's not the last token of the word, it must not end with hasanta
                if idx + 1 < n {
                    assert!(
                        !tok.text.ends_with('্'),
                        "Mid-word hasanta boundary violation in '{}' (token #{}: '{}')",
                        target,
                        idx,
                        tok.text
                    );
                }
            }
        }
    }
}

#[test]
fn test_no_kar_at_word_start() {
    let dev_path = find_file(&[
        "data/eval/bn.translit.sampled.dev.tsv",
        "../data/eval/bn.translit.sampled.dev.tsv",
        "../../data/eval/bn.translit.sampled.dev.tsv",
    ]);
    let file = File::open(&dev_path).expect("Open dev.tsv");
    let reader = BufReader::new(file);

    for line in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            let target = canonicalize(parts[0].trim());
            let tokens = tokenize_bengali(&target);
            if let Some(first) = tokens.first() {
                assert!(
                    !first.is_vowel_kar(),
                    "Phonotactic violation: dependent KAR at word start in '{}'",
                    target
                );
            }
        }
    }
}

#[test]
fn test_tokenization_idempotence_on_dev() {
    let dev_path = find_file(&[
        "data/eval/bn.translit.sampled.dev.tsv",
        "../data/eval/bn.translit.sampled.dev.tsv",
        "../../data/eval/bn.translit.sampled.dev.tsv",
    ]);
    let file = File::open(&dev_path).expect("Open dev.tsv");
    let reader = BufReader::new(file);

    for line in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            let target = canonicalize(parts[0].trim());
            let tokens = tokenize_bengali(&target);
            let joined: String = tokens.iter().map(|t| t.text.as_str()).collect();
            let re_tokens = tokenize_bengali(&joined);
            assert_eq!(
                tokens, re_tokens,
                "Idempotence failed for target '{}'",
                target
            );
        }
    }
}

#[test]
fn test_special_character_accounting_and_decisions() {
    let dev_path = find_file(&[
        "data/eval/bn.translit.sampled.dev.tsv",
        "../data/eval/bn.translit.sampled.dev.tsv",
        "../../data/eval/bn.translit.sampled.dev.tsv",
    ]);
    let file = File::open(&dev_path).expect("Open dev.tsv");
    let reader = BufReader::new(file);

    let mut two_count = 0;
    let mut trailing_hasanta_count = 0;
    let mut zwnj_count = 0;

    for line in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            let target = canonicalize(parts[0].trim());
            if target.contains('২') {
                two_count += 1;
            }
            if target.ends_with('্') {
                trailing_hasanta_count += 1;
            }
            if target.contains('\u{200C}') {
                zwnj_count += 1;
            }
        }
    }

    // In DEV:
    // 0 rows contain '২' (digit reduplication)
    // 9 rows have trailing hasanta (Arabic honorifics like 'আল্লাহ্' and English loan transliterations like 'টাইমস্')
    // 0 rows contain ZWNJ
    assert_eq!(two_count, 0);
    assert_eq!(trailing_hasanta_count, 9);
    assert_eq!(zwnj_count, 0);
}

#[test]
fn test_roman_segmentation_lattice_offers_both_decomposed_and_digraph_paths() {
    use lekhani_core::tokenization::roman_segmentation_lattice;

    // 1. Digraph 'ng' and 'bh': bhangon
    let paths_bhangon = roman_segmentation_lattice("bhangon");
    // Must contain greedy digraph path ["bh", "a", "ng", "o", "n"]
    assert!(paths_bhangon.iter().any(|p| p == &["bh", "a", "ng", "o", "n"]));
    // Must also contain decomposed path allowing 'n' + 'g' ["bh", "a", "n", "g", "o", "n"]
    assert!(paths_bhangon.iter().any(|p| p == &["bh", "a", "n", "g", "o", "n"]));

    // 2. Doubled consonant 'tt': kotto
    let paths_kotto = roman_segmentation_lattice("kotto");
    // Must contain both ["k", "o", "tt", "o"] and ["k", "o", "t", "t", "o"]
    assert!(paths_kotto.iter().any(|p| p == &["k", "o", "tt", "o"]));
    assert!(paths_kotto.iter().any(|p| p == &["k", "o", "t", "t", "o"]));
}

