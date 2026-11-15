#![allow(clippy::all)]
//! Build Hermetic Bengali Binary Dictionary (dictionary.bin)
//!
//! Strictly compiles from the canonical, committed root lexicon:
//! `data/dictionaries/dictionary.json` (SHA-256: 8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb).
//!
//! Invariant: Zero external, uncommitted, or Wikipedia-derived unigram imports.
//! Outputs a compact, mmap-ready LDI3 binary trie for sub-millisecond IME startup.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use lekhani_core::chars::canonicalize;
use lekhani_core::fs::compute_sha256_hex;
use lekhani_core::trie::PrefixTrie;

pub const EXPECTED_DICT_JSON_SHA256: &str =
    "8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("══════════════════════════════════════════════════════════════════");
    println!("   📚 Lekhani Hermetic Dictionary Builder (build_dict)");
    println!("══════════════════════════════════════════════════════════════════\n");

    let args: Vec<String> = std::env::args().collect();
    let dict_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("data/dictionaries/dictionary.json")
    };
    let out_bin_path = if args.len() > 2 {
        PathBuf::from(&args[2])
    } else {
        PathBuf::from("data/dictionaries/dictionary.bin")
    };

    if !dict_path.exists() {
        eprintln!("❌ Fatal: dictionary.json not found at {:?}", dict_path);
        std::process::exit(1);
    }

    println!(
        "1. Reading and verifying SHA-256 of source lexicon: {:?}",
        dict_path
    );
    let raw_bytes = std::fs::read(&dict_path)?;
    let actual_hash = compute_sha256_hex(&raw_bytes);
    println!("   ├─ Expected SHA-256: {}", EXPECTED_DICT_JSON_SHA256);
    println!("   ├─ Actual SHA-256  : {}", actual_hash);

    if actual_hash != EXPECTED_DICT_JSON_SHA256 {
        eprintln!(
            "❌ Fatal: Dictionary hash mismatch! Expected {}, got {}",
            EXPECTED_DICT_JSON_SHA256, actual_hash
        );
        std::process::exit(1);
    }
    println!("   └─ ✓ SHA-256 Integrity Verified.");

    println!("\n2. Parsing canonical JSON word entries...");
    let file = File::open(dict_path)?;
    let reader = BufReader::new(file);
    let raw_map: HashMap<String, serde_json::Value> = serde_json::from_reader(reader)?;

    let mut words = Vec::with_capacity(165_000);
    let mut unique_words = HashSet::with_capacity(165_000);
    let mut dropped_empty = 0usize;

    for val in raw_map.values() {
        if let Some(list) = val.as_array() {
            for item in list {
                let w = if let Some(s) = item.as_str() {
                    Some(canonicalize(s))
                } else if let Some(arr) = item.as_array() {
                    arr.first().and_then(|v| v.as_str()).map(canonicalize)
                } else {
                    None
                };

                if let Some(word) = w {
                    if word.trim().is_empty() {
                        dropped_empty += 1;
                        continue;
                    }
                    if unique_words.insert(word.clone()) {
                        words.push(word);
                    }
                }
            }
        }
    }

    println!("   ├─ Total distinct JSON keys       : {}", raw_map.len());
    println!(
        "   ├─ Total unique canonical words  : {}",
        unique_words.len()
    );
    println!("   └─ Dropped empty/whitespace items : {}", dropped_empty);

    println!("\n3. Building and sorting PrefixTrie...");
    let mut trie = PrefixTrie::new();
    trie.insert_bulk(words);
    trie.ensure_sorted();
    println!("   └─ ✓ Trie compiled with {} sorted entries.", trie.len());

    println!("\n4. Serializing to LDI3 binary format: {:?}", out_bin_path);
    trie.save_binary(&out_bin_path)?;

    let bin_bytes = std::fs::read(&out_bin_path)?;
    let bin_hash = compute_sha256_hex(&bin_bytes);
    println!(
        "   ├─ Binary File Size : {} bytes ({:.2} MB)",
        bin_bytes.len(),
        bin_bytes.len() as f64 / 1_048_576.0
    );
    println!("   └─ Binary SHA-256   : {}", bin_hash);

    println!("\n5. Verifying roundtrip deserialization...");
    let loaded_trie = PrefixTrie::from_binary(&bin_bytes)?;
    assert_eq!(
        loaded_trie.len(),
        trie.len(),
        "Roundtrip entry count mismatch!"
    );
    println!(
        "   └─ ✓ Roundtrip verification successful ({} words).",
        loaded_trie.len()
    );

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("   ✓ HERMETIC LEXICON COMPILATION SUCCESSFUL");
    println!("══════════════════════════════════════════════════════════════════");

    Ok(())
}
