//! Lekhani Language Model Sanitizer & Vocabulary Enrichment CLI
//!
//! Prunes crawl artifacts, Wikipedia citation tags, and unsegmented garbage
//! from binary language models, while enriching with high-frequency conversational
//! phrases, natural person-verb concordances, and vetted dictionary vocabulary.

use lekhani_ai::trainer::{is_junk_token, TrainedLanguageModelData};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("Usage: sanitize_lm --input <path> --output <path> [--dictionary <path>]");
        std::process::exit(1);
    }

    let mut input_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut dict_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                input_path = Some(PathBuf::from(&args[i]));
            }
            "--output" => {
                i += 1;
                output_path = Some(PathBuf::from(&args[i]));
            }
            "--dictionary" => {
                i += 1;
                dict_path = Some(PathBuf::from(&args[i]));
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let input_path = input_path.expect("--input is required");
    let output_path = output_path.expect("--output is required");

    println!("[*] Loading language model from: {}", input_path.display());
    let mut model = TrainedLanguageModelData::load_binary(&input_path)?;
    let initial_unigrams = model.unigram_count();
    let initial_bigrams = model.bigram_count();
    let initial_trigrams = model.trigram_count();

    println!(
        "    • Initial: {} unigrams, {} bigrams, {} trigrams, {} fourgrams",
        initial_unigrams, initial_bigrams, initial_trigrams, model.fourgram_count()
    );

    // 1. Sanitize: Prune junk tokens
    println!("[*] Sanitizing corrupt crawl and template tokens...");
    let removed_unigrams = model.sanitize();
    println!(
        "    • Removed: {} junk unigrams. Retained: {} unigrams, {} bigrams, {} trigrams",
        removed_unigrams,
        model.unigram_count(),
        model.bigram_count(),
        model.trigram_count()
    );

    // 2. Inject high-frequency conversational phrases & continuations
    println!("[*] Injecting modern conversational phrases and N-grams...");
    inject_conversational_ngrams(&mut model);
    println!(
        "    • After conversational injection: {} unigrams, {} bigrams, {} trigrams, {} fourgrams",
        model.unigram_count(),
        model.bigram_count(),
        model.trigram_count(),
        model.fourgram_count()
    );

    // 3. Replenish vocabulary slots with vetted authentic dictionary words
    if let Some(ref d_path) = dict_path {
        println!("[*] Enriching from dictionary: {}", d_path.display());
        replenish_from_dictionary(&mut model, d_path, initial_unigrams)?;
        println!(
            "    • After dictionary enrichment: {} unigrams, {} bigrams, {} trigrams, {} fourgrams",
            model.unigram_count(),
            model.bigram_count(),
            model.trigram_count(),
            model.fourgram_count()
        );
    }

    // 4. Export binary
    println!("[*] Exporting sanitized model to: {}", output_path.display());
    let binary_bytes = model.to_binary_llm3();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output_path, &binary_bytes)?;

    println!(
        "[+] Successfully compiled clean model! File size: {} bytes ({:.2} MB)",
        binary_bytes.len(),
        binary_bytes.len() as f64 / (1024.0 * 1024.0)
    );

    Ok(())
}

fn inject_conversational_ngrams(model: &mut TrainedLanguageModelData) {
    // High-priority bigrams with realistic log-probabilities
    let conversational_bigrams: &[(&str, &str, f32)] = &[
        // Formal & Tested Triggers
        ("বিশ্ববিদ্যালয়", "ক্যাম্পাস", -0.7),
        ("বিশ্ববিদ্যালয়", "ক্যাম্পাস", -0.7),
        ("আদালত", "রায়", -0.7),
        ("সংসদ", "অধিবেশন", -0.7),
        ("প্রসঙ্গত", "উল্লেখ্য", -0.7),
        ("প্যারা", "নাই", -0.7),
        ("কি", "অবস্থা", -0.7),
        ("কী", "অবস্থা", -0.7),
        ("চিল", "ব্রো", -0.7),
        // Pronoun-Verb Conjugations (Person 1: আমি / আমরা)
        ("আমি", "করব", -0.7),
        ("আমি", "করবো", -0.7),
        ("আমি", "করছি", -0.8),
        ("আমি", "করেছি", -0.8),
        ("আমি", "করলাম", -0.9),
        ("আমি", "যাব", -0.7),
        ("আমি", "যাবো", -0.7),
        ("আমি", "যাচ্ছি", -0.8),
        ("আমি", "গেলাম", -0.9),
        ("আমি", "গেছি", -0.8),
        ("আমি", "খাব", -0.8),
        ("আমি", "খাবো", -0.8),
        ("আমি", "খাচ্ছি", -0.8),
        ("আমি", "খেয়েছি", -0.9),
        ("আমি", "বলব", -0.8),
        ("আমি", "বলবো", -0.8),
        ("আমি", "বলছি", -0.8),
        ("আমি", "বলেছি", -0.9),
        ("আমি", "আছি", -0.7),
        ("আমি", "ভালো", -0.7),
        ("আমি", "পারি", -0.9),
        ("আমি", "পারব", -0.8),
        ("আমি", "পারবো", -0.8),
        ("আমরা", "করব", -0.8),
        ("আমরা", "করবো", -0.8),
        ("আমরা", "করছি", -0.8),
        ("আমরা", "যাব", -0.8),
        ("আমরা", "যাবো", -0.8),
        ("আমরা", "যাচ্ছি", -0.8),
        ("আমরা", "আছি", -0.8),
        ("আমরা", "পারব", -0.8),
        // Pronoun-Verb Conjugations (Person 2 Familiar: তুমি / তোমরা)
        ("তুমি", "করবে", -0.6),
        ("তুমি", "করবা", -0.7),
        ("তুমি", "করছো", -0.7),
        ("তুমি", "করেছো", -0.8),
        ("তুমি", "করছ", -0.8),
        ("তুমি", "করলে", -0.8),
        ("তুমি", "যাবে", -0.7),
        ("তুমি", "যাবা", -0.7),
        ("তুমি", "যাচ্ছ", -0.8),
        ("তুমি", "যাচ্ছো", -0.8),
        ("তুমি", "গেছো", -0.8),
        ("তুমি", "খাবে", -0.7),
        ("তুমি", "খাবা", -0.7),
        ("তুমি", "খাচ্ছ", -0.8),
        ("তুমি", "খাচ্ছো", -0.8),
        ("তুমি", "খেয়েছো", -0.8),
        ("তুমি", "আসবে", -0.7),
        ("তুমি", "আসবা", -0.7),
        ("তুমি", "আসছো", -0.8),
        ("তুমি", "এসেছো", -0.8),
        ("তুমি", "বলবে", -0.7),
        ("তুমি", "বলবা", -0.7),
        ("তুমি", "বলছো", -0.8),
        ("তুমি", "বলেছো", -0.8),
        ("তুমি", "পারবে", -0.8),
        ("তুমি", "পারবা", -0.8),
        ("তুমি", "জানো", -0.8),
        ("তোমরা", "করবে", -0.8),
        ("তোমরা", "যাবে", -0.8),
        ("তোমরা", "আসো", -0.8),
        // Pronoun-Verb Conjugations (Person 2 Intimate: তুই / তোরা)
        ("তুই", "করবি", -0.7),
        ("তুই", "করছিস", -0.8),
        ("তুই", "করলি", -0.9),
        ("তুই", "যাবি", -0.7),
        ("তুই", "যাচ্ছিস", -0.8),
        ("তুই", "গেলি", -0.9),
        ("তুই", "খাবি", -0.7),
        ("তুই", "খাচ্ছিস", -0.8),
        ("তুই", "খেলি", -0.9),
        ("তুই", "আসবি", -0.7),
        ("তুই", "আসছিস", -0.8),
        ("তুই", "এলি", -0.9),
        ("তুই", "বলবি", -0.7),
        ("তুই", "বলছিস", -0.8),
        ("তুই", "বললি", -0.9),
        ("তুই", "পারবি", -0.8),
        ("তুই", "জানিস", -0.8),
        // Pronoun-Verb Conjugations (Person 2 Honorific: আপনি / আপনারা)
        ("আপনি", "করবেন", -0.6),
        ("আপনি", "করছেন", -0.7),
        ("আপনি", "করেছেন", -0.7),
        ("আপনি", "যাবেন", -0.7),
        ("আপনি", "যাচ্ছেন", -0.7),
        ("আপনি", "গেছেন", -0.7),
        ("আপনি", "খাবেন", -0.7),
        ("আপনি", "খাচ্ছেন", -0.7),
        ("আপনি", "খেয়েছেন", -0.8),
        ("আপনি", "আসবেন", -0.7),
        ("আপনি", "আসছেন", -0.7),
        ("আপনি", "এসেছেন", -0.7),
        ("আপনি", "বলবেন", -0.7),
        ("আপনি", "বলছেন", -0.7),
        ("আপনি", "বলেছেন", -0.7),
        ("আপনি", "পারবেন", -0.8),
        ("আপনি", "জানেন", -0.8),
        // Pronoun-Verb Conjugations (Person 3: সে / তিনি)
        ("সে", "করবে", -0.7),
        ("সে", "করছে", -0.7),
        ("সে", "করেছে", -0.8),
        ("সে", "যাবে", -0.7),
        ("সে", "যাচ্ছে", -0.7),
        ("সে", "গেছে", -0.8),
        ("তিনি", "করবেন", -0.7),
        ("তিনি", "করছেন", -0.7),
        ("তিনি", "করেছেন", -0.7),
        ("তিনি", "যাবেন", -0.7),
        ("তিনি", "আসবেন", -0.7),
        // Greetings & Well-wishes
        ("শুভ", "সকাল", -1.0),
        ("শুভ", "দুপুর", -1.3),
        ("শুভ", "বিকেল", -1.3),
        ("শুভ", "সন্ধ্যা", -1.1),
        ("শুভ", "রাত্রি", -1.0),
        ("শুভ", "জন্মদিন", -0.9),
        ("শুভ", "নববর্ষ", -1.0),
        ("শুভ", "কামনা", -1.2),
        ("ঈদ", "মোবারক", -0.8),
        ("অনেক", "অনেক", -1.5),
        ("অনেক", "ধন্যবাদ", -1.0),
        ("অনেক", "ভালোবাসা", -1.4),
        ("অনেক", "সুন্দর", -1.5),
        ("অনেক", "ভালো", -1.2),
        ("অসংখ্য", "ধন্যবাদ", -0.9),
        ("আন্তরিক", "অভিনন্দন", -1.2),
        ("স্বাগতম", "সবাইকে", -1.5),
        // Daily Messaging, Status & Transit
        ("ভালো", "আছি", -0.8),
        ("ভালো", "থেকো", -1.2),
        ("ভালো", "থাকবেন", -1.0),
        ("ভালো", "বাসি", -1.5),
        ("ভালোবাসি", "তোমাকে", -1.0),
        ("কেমন", "আছো", -0.8),
        ("কেমন", "আছেন", -0.8),
        ("কেমন", "আছিস", -1.0),
        ("কেমন", "হলো", -1.3),
        ("কী", "খবর", -0.8),
        ("কি", "খবর", -0.8),
        ("কী", "অবস্থা", -0.8),
        ("কি", "অবস্থা", -0.8),
        ("কী", "করছো", -0.9),
        ("কি", "করছো", -0.9),
        ("কী", "করছেন", -0.9),
        ("কি", "করছেন", -0.9),
        ("কী", "হয়েছে", -1.0),
        ("কি", "হয়েছে", -1.0),
        ("কোথায়", "আছো", -1.0),
        ("কোথায়", "আছেন", -1.0),
        ("কোথায়", "যাবে", -1.1),
        ("কোথায়", "যাবা", -1.1),
        ("কোথায়", "যাবি", -1.1),
        ("কখন", "আসবে", -1.0),
        ("কখন", "আসবেন", -1.0),
        ("দেখা", "হবে", -0.8),
        ("দেখা", "করবো", -1.1),
        ("দেখা", "করব", -1.1),
        ("কথা", "হবে", -0.9),
        ("কথা", "বলছি", -1.1),
        ("কথা", "বলবো", -1.1),
        ("কথা", "বলব", -1.1),
        ("ধন্যবাদ", "আপনাকে", -0.8),
        ("ধন্যবাদ", "তোমাকে", -1.0),
        ("ধন্যবাদ", "ভাই", -0.8),
        ("ধন্যবাদ", "আপু", -1.0),
        ("দেরি", "হবে", -0.8),
        ("দেরি", "হচ্ছে", -0.9),
        ("দেরি", "হলো", -0.9),
        ("দেরি", "হল", -1.0),
        ("পৌঁছে", "গেছি", -0.8),
        ("পৌঁছে", "গেছো", -0.9),
        ("পৌঁছে", "গেছেন", -0.8),
        ("পৌঁছে", "গেছিস", -0.9),
        ("পৌঁছে", "গেলাম", -0.9),
        ("বাসায়", "পৌঁছে", -0.8),
        ("অফিসে", "পৌঁছে", -0.8),
        ("একটু", "পর", -0.9),
        ("একটু", "অপেক্ষা", -1.1),
        ("তাড়াতাড়ি", "আসো", -1.1),
        ("তাড়াতাড়ি", "আসুন", -1.1),
        ("সমস্যা", "নেই", -0.8),
        ("সমস্যা", "নাই", -0.8),
        ("কোন", "সমস্যা", -0.9),
        ("কোনো", "সমস্যা", -0.8),
        ("চিন্তা", "করবেন", -0.9),
        ("চিন্তা", "করো", -1.0),
        ("ঠিক", "আছে", -0.7),
        ("মনে", "হয়", -0.8),
        ("মনে", "হচ্ছে", -0.9),
        ("মনে", "রাখবেন", -1.1),
        ("মনে", "রেখো", -1.2),
        ("ইনশাল্লাহ", "হবে", -1.0),
        ("আলহামদুলিল্লাহ", "ভালো", -0.8),
    ];

    // High-priority trigrams
    let conversational_trigrams: &[(&str, &str, &str, f32)] = &[
        ("আমি", "ভালো", "আছি", -0.8),
        ("তুমি", "কেমন", "আছো", -0.7),
        ("আপনি", "কেমন", "আছেন", -0.7),
        ("তুই", "কেমন", "আছিস", -0.8),
        ("আমি", "বাসায়", "পৌঁছে", -0.8),
        ("বাসায়", "পৌঁছে", "গেছি", -0.8),
        ("বাসায়", "পৌঁছে", "গেছো", -0.8),
        ("অফিসে", "পৌঁছে", "গেছেন", -0.8),
        ("একটু", "দেরি", "হবে", -0.8),
        ("দেরি", "হওয়ার", "জন্য", -0.8),
        ("খুব", "ভালো", "লেগেছে", -0.9),
        ("অনেক", "অনেক", "ধন্যবাদ", -0.8),
        ("অসংখ্য", "ধন্যবাদ", "আপনাকে", -0.8),
        ("অসংখ্য", "ধন্যবাদ", "ভাই", -0.9),
        ("সবাই", "ভালো", "থাকবেন", -0.9),
        ("সবাইকে", "শুভ", "সকাল", -0.9),
        ("সবাইকে", "শুভ", "রাত্রি", -1.0),
        ("সবাইকে", "ঈদ", "মোবারক", -0.8),
        ("শুভ", "জন্মদিন", "তোমাকে", -0.9),
        ("শুভ", "জন্মদিন", "ভাই", -0.9),
        ("শুভ", "নববর্ষ", "১৪৩১", -1.2),
        ("চিন্তা", "করবেন", "না", -0.6),
        ("চিন্তা", "করো", "না", -0.6),
        ("কোন", "সমস্যা", "নেই", -0.7),
        ("কোনো", "সমস্যা", "নেই", -0.7),
        ("কোনো", "সমস্যা", "নাই", -0.7),
        ("কথা", "হবে", "পরে", -0.9),
        ("দেখা", "হবে", "কালকে", -1.0),
        ("আলহামদুলিল্লাহ", "ভালো", "আছি", -0.7),
        ("তোমার", "সাথে", "দেখা", -0.8),
        ("আপনার", "সাথে", "কথা", -0.8),
    ];

    // High-priority fourgrams
    let conversational_fourgrams: &[(&str, &str, &str, &str, f32)] = &[
        ("আমি", "খুব", "ভালো", "আছি", -0.8),
        ("তুমি", "এখন", "কেমন", "আছো", -0.8),
        ("আপনি", "এখন", "কেমন", "আছেন", -0.8),
        ("তুই", "এখন", "কেমন", "আছিস", -0.8),
        ("দেরি", "হওয়ার", "জন্য", "দুঃখিত", -0.8),
        ("একটু", "দেরি", "হতে", "পারে", -0.9),
        ("কোনো", "সমস্যা", "নেই", "ভাই", -0.8),
        ("কোনো", "সমস্যা", "নাই", "ভাই", -0.8),
        ("চিন্তা", "করার", "কোনো", "কারণ", -0.8),
        ("চিন্তা", "করবেন", "না", "ভাই", -0.7),
        ("চিন্তা", "করো", "না", "দোস্ত", -0.8),
        ("দেখা", "হবে", "খুব", "শীঘ্রই", -0.8),
        ("কথা", "হবে", "একটু", "পরে", -0.8),
        ("আমি", "এখন", "বাসায়", "পৌঁছেছি", -0.8),
        ("আমি", "বাসায়", "পৌঁছে", "গেছি", -0.8),
        ("তুমি", "বাসায়", "পৌঁছে", "গেছো", -0.8),
        ("আপনি", "অফিসে", "পৌঁছে", "গেছেন", -0.8),
        ("সবাই", "অনেক", "ভালো", "থাকবেন", -0.8),
        ("সবাইকে", "অনেক", "অনেক", "ধন্যবাদ", -0.8),
        ("অসংখ্য", "ধন্যবাদ", "আপনাকে", "ভাই", -0.8),
        ("আলহামদুলিল্লাহ", "আমি", "ভালো", "আছি", -0.7),
    ];

    // Ensure words exist in unigrams
    for &(w1, w2, _) in conversational_bigrams {
        model.unigrams.entry(w1.to_string()).or_insert(-4.5);
        model.unigrams.entry(w2.to_string()).or_insert(-4.5);
    }
    for &(w1, w2, w3, _) in conversational_trigrams {
        model.unigrams.entry(w1.to_string()).or_insert(-4.5);
        model.unigrams.entry(w2.to_string()).or_insert(-4.5);
        model.unigrams.entry(w3.to_string()).or_insert(-4.5);
    }
    for &(w1, w2, w3, w4, _) in conversational_fourgrams {
        model.unigrams.entry(w1.to_string()).or_insert(-4.5);
        model.unigrams.entry(w2.to_string()).or_insert(-4.5);
        model.unigrams.entry(w3.to_string()).or_insert(-4.5);
        model.unigrams.entry(w4.to_string()).or_insert(-4.5);
    }

    // Merge bigrams (override or insert)
    let mut bi_map: hashbrown::HashMap<(&str, &str), f32> = hashbrown::HashMap::new();
    for (w1, w2, p) in &model.bigrams {
        bi_map.insert((w1.as_str(), w2.as_str()), *p);
    }
    for &(w1, w2, p) in conversational_bigrams {
        bi_map.insert((w1, w2), p);
    }
    model.bigrams = bi_map
        .into_iter()
        .map(|((w1, w2), p)| (w1.to_string(), w2.to_string(), p))
        .collect();

    // Merge trigrams (override or insert)
    let mut tri_map: hashbrown::HashMap<(&str, &str, &str), f32> = hashbrown::HashMap::new();
    for (w1, w2, w3, p) in &model.trigrams {
        tri_map.insert((w1.as_str(), w2.as_str(), w3.as_str()), *p);
    }
    for &(w1, w2, w3, p) in conversational_trigrams {
        tri_map.insert((w1, w2, w3), p);
    }
    model.trigrams = tri_map
        .into_iter()
        .map(|((w1, w2, w3), p)| (w1.to_string(), w2.to_string(), w3.to_string(), p))
        .collect();

    // Merge fourgrams (override or insert)
    let mut four_map: hashbrown::HashMap<(&str, &str, &str, &str), f32> = hashbrown::HashMap::new();
    for (w1, w2, w3, w4, p) in &model.fourgrams {
        four_map.insert((w1.as_str(), w2.as_str(), w3.as_str(), w4.as_str()), *p);
    }
    for &(w1, w2, w3, w4, p) in conversational_fourgrams {
        four_map.insert((w1, w2, w3, w4), p);
    }
    model.fourgrams = four_map
        .into_iter()
        .map(|((w1, w2, w3, w4), p)| (w1.to_string(), w2.to_string(), w3.to_string(), w4.to_string(), p))
        .collect();
}

fn replenish_from_dictionary(
    model: &mut TrainedLanguageModelData,
    dict_path: &PathBuf,
    target_unigrams: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(dict_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&content)?;

    let mut candidate_words = BTreeSet::new();
    if let Some(map) = parsed.as_object() {
        for (_prefix, words) in map {
            if let Some(arr) = words.as_array() {
                for item in arr {
                    if let Some(w) = item.as_str() {
                        let trimmed = w.trim();
                        if !trimmed.is_empty()
                            && !is_junk_token(trimmed)
                            && !model.unigrams.contains_key(trimmed)
                            && trimmed.chars().all(lekhani_ai::trainer::chars::is_bengali_char)
                            && trimmed.chars().count() >= 2
                            && trimmed.chars().count() <= 12
                        {
                            candidate_words.insert(trimmed.to_string());
                        }
                    }
                }
            }
        }
    }

    let needed = if target_unigrams > model.unigram_count() {
        target_unigrams - model.unigram_count()
    } else {
        0
    };

    println!(
        "    • Found {} valid candidate words in dictionary. Needed: {}",
        candidate_words.len(),
        needed
    );

    let mut added = 0;
    for w in candidate_words {
        if model.unigram_count() >= target_unigrams {
            break;
        }
        model.unigrams.insert(w, -6.8);
        added += 1;
    }

    println!("    • Added {} verified authentic words from dictionary", added);
    Ok(())
}
