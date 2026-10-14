//! Normalization Integrity Test Suite
//!
//! Ensures that all lexicons, override files, and evaluation gold standards
//! strictly conform to Lekhani's canonical representation (Unicode NFC with
//! decomposed nuktas and modern atomic Khanda Ta U+09CE).

use lekhani_core::chars::is_canonical;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[test]
fn test_phonetic_overrides_are_canonical() {
    let path = Path::new("data/dictionaries/phonetic_overrides.json");
    if !path.exists() {
        return;
    }
    let file = File::open(path).expect("Failed to open phonetic_overrides.json");
    let reader = BufReader::new(file);
    let map: hashbrown::HashMap<String, Vec<serde_json::Value>> =
        serde_json::from_reader(reader).expect("Failed to parse phonetic_overrides.json");

    for (k, v) in map {
        for item in v {
            let word = if let Some(s) = item.as_str() {
                s
            } else if let Some(arr) = item.as_array() {
                arr[0].as_str().unwrap()
            } else {
                continue;
            };
            assert!(
                is_canonical(word),
                "Non-canonical word found in phonetic_overrides for key '{}': '{}'",
                k,
                word
            );
        }
    }
}

#[test]
fn test_gold_tsv_is_canonical() {
    let path = Path::new("data/eval/phonetic_gold.tsv");
    if !path.exists() {
        return;
    }
    let file = File::open(path).expect("Failed to open phonetic_gold.tsv");
    let reader = BufReader::new(file);
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.expect("Failed to read line");
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split('\t').collect();
        if parts.len() >= 2 {
            let expected = parts[1].trim();
            assert!(
                is_canonical(expected),
                "Non-canonical expected word at line {}: '{}'",
                line_idx + 1,
                expected
            );
        }
    }
}

#[test]
fn test_dictionary_is_canonical() {
    let path = Path::new("data/dictionaries/dictionary.json");
    if !path.exists() {
        return;
    }
    let file = File::open(path).expect("Failed to open dictionary.json");
    let reader = BufReader::new(file);
    let map: hashbrown::HashMap<String, Vec<serde_json::Value>> =
        serde_json::from_reader(reader).expect("Failed to parse dictionary.json");

    for (k, v) in map {
        for item in v {
            let word = if let Some(s) = item.as_str() {
                s
            } else if let Some(arr) = item.as_array() {
                arr[0].as_str().unwrap()
            } else {
                continue;
            };
            assert!(
                is_canonical(word),
                "Non-canonical word found in dictionary for prefix '{}': '{}'",
                k,
                word
            );
        }
    }
}
