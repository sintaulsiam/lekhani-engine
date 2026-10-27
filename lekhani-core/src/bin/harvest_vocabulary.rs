//! Harvest Multi-Domain Bengali Vocabulary
//!
//! Merges:
//! 1. Canonical lemmas from `data/dictionaries/dictionary.json`
//! 2. High-frequency conversational unigrams from `data/corpus/bengali_corpus.txt` (count >= 3)
//! 3. Colloquial spoken verb inflections & case clitics
//! 4. Modern tech and social media loanwords
//!
//! Applies Orthographic Canonicalization Filter to discard corrupt internet typos.
//! Outputs `data/corpus/expanded_vocabulary.txt`.

use hashbrown::{HashMap, HashSet};
use lekhani_core::chars::canonicalize;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

const TECH_LOANWORDS: &[&str] = &[
    "হোয়াটসঅ্যাপ", "ইউটিউব", "ফেসবুক", "অনলাইন", "ডাউনলোড", "লগইন", "পাসওয়ার্ড",
    "আপডেট", "শেয়ার", "কমেন্ট", "স্ক্রিনশট", "ইনবক্স", "নোটিফিকেশন", "প্রোফাইল",
    "মেসেজ", "ইমেইল", "ক্লিক", "ব্রাউজার", "সার্ভার", "ইন্টারনেট", "ওয়াইফাই",
    "মোবাইল", "কম্পিউটার", "ল্যাপটপ", "সফটওয়্যার", "অ্যাপ", "অ্যাপ্লিকেশন",
    "পোস্ট", "ভিডিও", "অডিও", "ফাইল", "ফোল্ডার", "লিংক", "কোড", "ডাটা",
    "নেটওয়ার্ক", "কন্টেন্ট", "চ্যানেল", "সাবস্ক্রাইব", "লাইক", "ভিউ", "স্ক্রিন",
    "ডিসপ্লে", "ক্যামেরা", "ব্যাটারি", "চার্জার", "গুগল", "অ্যান্ড্রয়েড",
    "উইন্ডোজ", "আইফোন", "সেটিংস", "ইনস্টল", "সার্চ", "কপি", "পেস্ট", "কাট",
    "ডিলিট", "রিমুভ", "অ্যাড", "সেন্ড", "রিসিভ", "কল", "ব্লক", "আনব্লক",
    "রিপোর্ট", "গ্রুপ", "চ্যাট", "ভয়েস", "রেকর্ড", "প্লে", "পজ", "স্টপ",
    "মিউট", "আনমিউট", "স্ক্রোল", "রিফ্রেশ", "রিলোড", "ব্যাকআপ", "রিস্টোর",
    "অ্যাকাউন্ট", "ইউজার", "এডমিন", "লগআউট", "সাইনআপ", "কানেক্ট", "ডিসকানেক্ট",
];

const COLLOQUIAL_VERB_ROOTS: &[&str] = &[
    "কর", "যা", "খা", "দেখ", "বল", "শোন", "আস", "বস", "পড়", "লিখ",
    "চল", "জান", "নে", "দে", "হ", "মর", "ধর", "মার", "রাখ", "ডাক",
    "ভাব", "কাট", "ছাড়", "বাঁচ", "হার", "জিত", "খেল", "ঘুম", "হাস", "কাঁদ",
];

const COLLOQUIAL_SUFFIXES: &[&str] = &[
    // Spoken past
    "লাম", "লে", "ল", "লেন", "লি",
    "সিলাম", "সিলে", "সিল", "সিলেন", "সিলি",
    "সি", "স", "সে", "সেন",
    "ছিলাম", "ছিলে", "ছিল", "ছিলেন", "ছিলি",
    "ছি", "ছে", "ছেন",
    // Spoken present continuous
    "তেসি", "তেস", "তেসে", "তেসেন",
    "তাছি", "তাছস", "তাছে", "তাছেন",
    "তেছি", "তেছ", "তেছে", "তেছেন",
    // Spoken future
    "ব", "বে", "বেন", "বি",
    "বা", "বানে", "বেনে",
    // Spoken imperative & conditional
    "িস", "িও", "িয়েন", "ুক", "ুন",
    "লে", "লেহে", "লেতো",
];

const NOUN_CLITICS: &[&str] = &[
    "টা", "টি", "খানা", "খানি", "গুলো", "গুলা", "দের", "টারে", "টার",
    "গুলার", "গুলোতে", "গুলাতে", "খানাতে",
];

fn clean_bengali_token(raw: &str) -> Option<String> {
    let bn_only: String = raw
        .chars()
        .filter(|&c| ('\u{0980}'..='\u{09FF}').contains(&c) || c == '\u{200D}')
        .collect();

    if bn_only.chars().count() < 2 {
        return None;
    }

    let canon = canonicalize(&bn_only);
    if canon.chars().count() >= 2 {
        Some(canon)
    } else {
        None
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("══════════════════════════════════════════════════════════════════");
    println!("   🌾 Lekhani Multi-Domain Vocabulary Harvester");
    println!("══════════════════════════════════════════════════════════════════\n");

    let dict_path = Path::new("data/dictionaries/dictionary.json");
    let corpus_path = Path::new("data/corpus/bengali_corpus.txt");
    let out_path = Path::new("data/corpus/expanded_vocabulary.txt");

    let mut canonical_dict_set = HashSet::new();
    let mut expanded_vocab = HashSet::new();

    // 1. Load Canonical dictionary.json
    println!("1. Loading canonical dictionary.json...");
    if dict_path.exists() {
        let f = File::open(dict_path)?;
        let r = BufReader::new(f);
        let dict_map: HashMap<String, Vec<String>> = serde_json::from_reader(r)?;
        for (_, words) in dict_map {
            for w in words {
                let canon = canonicalize(&w);
                canonical_dict_set.insert(canon.clone());
                expanded_vocab.insert(canon);
            }
        }
        println!("   └─ Loaded {} canonical words.", canonical_dict_set.len());
    } else {
        println!("   └─ ⚠️ Warning: dictionary.json not found!");
    }

    // 2. Scan Real Corpus (bengali_corpus.txt)
    println!("\n2. Scanning multi-domain corpus ({:?})...", corpus_path);
    if corpus_path.exists() {
        let f = File::open(corpus_path)?;
        let r = BufReader::new(f);
        let mut unigram_counts: HashMap<String, u32> = HashMap::new();
        let mut line_count = 0usize;

        for line in r.lines().map_while(Result::ok) {
            line_count += 1;
            for token in line.split_whitespace() {
                if let Some(cleaned) = clean_bengali_token(token) {
                    *unigram_counts.entry(cleaned).or_insert(0) += 1;
                }
            }
            if line_count % 200_000 == 0 {
                println!("   ├─ Processed {} lines ({} unique candidates)...", line_count, unigram_counts.len());
            }
        }
        println!("   ├─ Total lines read: {}", line_count);
        println!("   ├─ Total unique word forms: {}", unigram_counts.len());

        let mut added_from_corpus = 0usize;
        let mut dropped_typos = 0usize;

        for (word, count) in unigram_counts {
            if count < 3 {
                continue; // Drop rare noise
            }

            if canonical_dict_set.contains(&word) {
                expanded_vocab.insert(word);
                added_from_corpus += 1;
                continue;
            }

            // Orthographic Harmonization Filter:
            // Check if this unigram is a common internet misspelling of an existing canonical word
            let mut is_corrupt_spelling = false;
            // Quick check on words of length 3..=10
            if word.chars().count() >= 3 && word.chars().count() <= 10 {
                // Check if simple letter swaps like ণ -> ন or শ -> স produce a canonical word
                let normalized_candidate = word
                    .replace('ণ', "ন")
                    .replace('ষ', "শ")
                    .replace('স', "শ")
                    .replace('ী', "ি")
                    .replace('ূ', "ু");
                if canonical_dict_set.contains(&normalized_candidate) {
                    // Canonical form takes precedence; do not ingest corrupted internet spelling
                    is_corrupt_spelling = true;
                    dropped_typos += 1;
                }
            }

            if !is_corrupt_spelling {
                expanded_vocab.insert(word);
                added_from_corpus += 1;
            }
        }
        println!("   ├─ Ingested {} valid conversational words from corpus.", added_from_corpus);
        println!("   └─ Filtered out {} corrupted internet misspellings.", dropped_typos);
    } else {
        println!("   └─ ⚠️ bengali_corpus.txt not found, skipping corpus pass.");
    }

    // 3. Morphological Colloquial Verb Inflections
    println!("\n3. Synthesizing spoken verb inflections...");
    let mut morph_count = 0usize;
    for &root in COLLOQUIAL_VERB_ROOTS {
        for &suffix in COLLOQUIAL_SUFFIXES {
            let inflected = format!("{}{}", root, suffix);
            let canon = canonicalize(&inflected);
            if expanded_vocab.insert(canon) {
                morph_count += 1;
            }
        }
        // Base root with noun clitics (for verbal nouns)
        for &clitic in NOUN_CLITICS {
            let inflected = format!("{}{}", root, clitic);
            let canon = canonicalize(&inflected);
            if expanded_vocab.insert(canon) {
                morph_count += 1;
            }
        }
    }
    println!("   └─ Added {} colloquial morphological forms.", morph_count);

    // 4. Ingest Tech Loanwords
    println!("\n4. Ingesting modern tech & social media loanwords...");
    let mut loan_count = 0usize;
    for &word in TECH_LOANWORDS {
        let canon = canonicalize(word);
        if expanded_vocab.insert(canon) {
            loan_count += 1;
        }
        for &clitic in NOUN_CLITICS {
            let inflected = format!("{}{}", word, clitic);
            let canon = canonicalize(&inflected);
            if expanded_vocab.insert(canon) {
                loan_count += 1;
            }
        }
    }
    println!("   └─ Added {} tech loanwords & clitic variants.", loan_count);

    // 5. Write Sorted Expanded Vocabulary
    println!("\n5. Writing expanded vocabulary to {:?}...", out_path);
    let mut sorted_vocab: Vec<String> = expanded_vocab.into_iter().collect();
    sorted_vocab.sort();

    let out_file = File::create(out_path)?;
    let mut writer = BufWriter::new(out_file);
    for w in &sorted_vocab {
        writeln!(writer, "{}", w)?;
    }
    writer.flush()?;

    println!("   └─ ✓ Successfully wrote {} total unique words.", sorted_vocab.len());
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("   🎉 Vocabulary Harvesting Complete!");
    println!("══════════════════════════════════════════════════════════════════");

    Ok(())
}
