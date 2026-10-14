//! Pragyan Phase 1 Step 3: Token Inventory and Property Measurement
//!
//! Measures pre-pruning token inventories:
//! - Unique Bengali tokens across all splits and dictionary.json (by kind: CONS, CONJ, KAR, IVOW, MOD)
//! - Unique Roman tokens across all splits
//! - Verification of 100% round-trip lossless concatenation

use hashbrown::HashSet;
use lekhani_core::chars::canonicalize;
use lekhani_core::tokenization::{
    tokenize_bengali, tokenize_roman, BengaliTokenKind,
};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn load_tsv_pairs(path: &str) -> Vec<(String, String)> {
    let file = File::open(path).unwrap_or_else(|_| panic!("Failed to open {}", path));
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            pairs.push((canonicalize(parts[0].trim()), parts[1].trim().to_string()));
        }
    }
    pairs
}

fn load_dictionary_words(path: &str) -> Vec<String> {
    let file = File::open(path).unwrap_or_else(|_| panic!("Failed to open {}", path));
    let reader = BufReader::new(file);
    let json: serde_json::Value =
        serde_json::from_reader(reader).expect("Failed to parse dictionary.json");
    let mut words = Vec::new();
    if let Some(obj) = json.as_object() {
        for list in obj.values() {
            if let Some(arr) = list.as_array() {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        words.push(canonicalize(s));
                    }
                }
            }
        }
    }
    words
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════════════════════");
    println!("   🔬 PRAGYAN PHASE 1 STEP 3: LINGUISTIC TOKEN INVENTORY & PROPERTIES");
    println!("══════════════════════════════════════════════════════════════════════════════════\n");

    let train = load_tsv_pairs("data/eval/bn.translit.sampled.train.tsv");
    let dev = load_tsv_pairs("data/eval/bn.translit.sampled.dev.tsv");
    let test = load_tsv_pairs("data/eval/bn.translit.sampled.test.tsv");
    let dict_words = load_dictionary_words("data/dictionaries/dictionary.json");

    println!("Dataset Sizes:");
    println!("  - Train: {} pairs", train.len());
    println!("  - Dev:   {} pairs", dev.len());
    println!("  - Test:  {} pairs", test.len());
    println!("  - Dict:  {} words\n", dict_words.len());

    let mut bn_cons = HashSet::new();
    let mut bn_conj = HashSet::new();
    let mut bn_kar = HashSet::new();
    let mut bn_ivow = HashSet::new();
    let mut bn_mod = HashSet::new();
    let mut bn_other = HashSet::new();
    let mut roman_tokens = HashSet::new();

    let mut total_words_tested = 0;
    let mut total_bengali_roundtrip_ok = 0;
    let mut total_roman_roundtrip_ok = 0;

    // Process Train, Dev, Test
    for split in [&train, &dev, &test] {
        for (bn, lat) in split.iter() {
            total_words_tested += 1;
            let b_toks = tokenize_bengali(bn.as_str());
            let b_reconstructed: String = b_toks.iter().map(|t| t.text.as_str()).collect();
            if b_reconstructed == *bn {
                total_bengali_roundtrip_ok += 1;
            }

            for tok in b_toks {
                match tok.kind {
                    BengaliTokenKind::Consonant => { bn_cons.insert(tok.text); }
                    BengaliTokenKind::Conjunct => { bn_conj.insert(tok.text); }
                    BengaliTokenKind::VowelKar => { bn_kar.insert(tok.text); }
                    BengaliTokenKind::IndependentVowel => { bn_ivow.insert(tok.text); }
                    BengaliTokenKind::Modifier => { bn_mod.insert(tok.text); }
                    BengaliTokenKind::Other => { bn_other.insert(tok.text); }
                }
            }

            let r_toks = tokenize_roman(lat.as_str());
            let r_norm = lekhani_core::tokenization::normalize_roman(lat.as_str());
            let r_reconstructed: String = r_toks.concat();
            if r_reconstructed == r_norm {
                total_roman_roundtrip_ok += 1;
            }

            for tok in r_toks {
                roman_tokens.insert(tok);
            }
        }
    }

    // Process Dictionary
    let mut dict_roundtrip_ok = 0;
    for w in &dict_words {
        let b_toks = tokenize_bengali(w);
        let b_reconstructed: String = b_toks.iter().map(|t| t.text.as_str()).collect();
        if b_reconstructed == *w {
            dict_roundtrip_ok += 1;
        }

        for tok in b_toks {
            match tok.kind {
                BengaliTokenKind::Consonant => { bn_cons.insert(tok.text); }
                BengaliTokenKind::Conjunct => { bn_conj.insert(tok.text); }
                BengaliTokenKind::VowelKar => { bn_kar.insert(tok.text); }
                BengaliTokenKind::IndependentVowel => { bn_ivow.insert(tok.text); }
                BengaliTokenKind::Modifier => { bn_mod.insert(tok.text); }
                BengaliTokenKind::Other => { bn_other.insert(tok.text); }
            }
        }
    }

    println!("### 1. Bengali Token Inventory (Pre-Pruning)");
    println!("| Token Category | Unique Count | Sample Items |");
    println!("|:---|:---:|:---|");
    println!("| Consonants (CONS) | {} | {:?} |", bn_cons.len(), sample_items(&bn_cons, 5));
    println!("| Conjuncts (CONJ) | {} | {:?} |", bn_conj.len(), sample_items(&bn_conj, 5));
    println!("| Vowel Kars (KAR) | {} | {:?} |", bn_kar.len(), sample_items(&bn_kar, 5));
    println!("| Independent Vowels (IVOW) | {} | {:?} |", bn_ivow.len(), sample_items(&bn_ivow, 5));
    println!("| Modifiers (MOD) | {} | {:?} |", bn_mod.len(), sample_items(&bn_mod, 5));
    println!("| Other (Punctuation/Sym) | {} | {:?} |", bn_other.len(), sample_items(&bn_other, 5));
    let total_bn_tokens = bn_cons.len() + bn_conj.len() + bn_kar.len() + bn_ivow.len() + bn_mod.len() + bn_other.len();
    println!("| **Total Unique Bengali Tokens** | **{}** | Complete Script Coverage |\n", total_bn_tokens);

    println!("### 2. Roman Token Inventory (Pre-Pruning)");
    println!("| Category | Unique Count | Sample Items |");
    println!("|:---|:---:|:---|");
    println!("| **Total Unique Roman Tokens** | **{}** | {:?} |\n", roman_tokens.len(), sample_items(&roman_tokens, 10));

    println!("### 3. Property Test Round-Trip Concatenation Results");
    println!("| Evaluation Target | Items Tested | Round-Trip Success | Success Rate (%) |");
    println!("|:---|:---:|:---:|:---:|");
    println!("| Bengali Words (Train+Dev+Test) | {} | {} | **{:.4}%** |", total_words_tested, total_bengali_roundtrip_ok, (total_bengali_roundtrip_ok as f64 / total_words_tested as f64) * 100.0);
    println!("| Roman Inputs (Train+Dev+Test) | {} | {} | **{:.4}%** |", total_words_tested, total_roman_roundtrip_ok, (total_roman_roundtrip_ok as f64 / total_words_tested as f64) * 100.0);
    println!("| Dictionary Entries (`dictionary.json`) | {} | {} | **{:.4}%** |\n", dict_words.len(), dict_roundtrip_ok, (dict_roundtrip_ok as f64 / dict_words.len() as f64) * 100.0);
}

fn sample_items(set: &HashSet<String>, count: usize) -> Vec<String> {
    let mut vec: Vec<String> = set.iter().cloned().collect();
    vec.sort();
    vec.into_iter().take(count).collect()
}
