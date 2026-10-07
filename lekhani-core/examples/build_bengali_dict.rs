use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use lekhani_core::trie::PrefixTrie;
use lekhani_ai::trainer::is_junk_token;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Building Bengali PrefixTrie dictionary from dictionary.json...");
    let dict_path = "/mnt/data/lekhani-android/data/dictionaries/dictionary.json";
    let file = File::open(dict_path)?;
    let reader = BufReader::new(file);
    let raw_map: HashMap<String, Vec<String>> = serde_json::from_reader(reader)?;

    let mut word_freq_map: HashMap<String, u32> = HashMap::with_capacity(180000);
    let mut pruned_dict_junk = 0;
    for (_k, v_list) in raw_map {
        for word in v_list {
            if is_junk_token(&word) {
                pruned_dict_junk += 1;
                continue;
            }
            word_freq_map.insert(word, 100);
        }
    }
    println!("Loaded {} unique words from dictionary.json (pruned {} junk)", word_freq_map.len(), pruned_dict_junk);

    let lm_path = "/mnt/data/lekhani-engine/data/dictionaries/bengali_lm.bin";
    let mut mapped_lm_count = 0;
    let mut pruned_lm_junk = 0;
    if let Ok(model) = lekhani_ai::zero_copy::ZeroCopyLanguageModel::from_file(lm_path) {
        println!("Loading 74M-word language model from {}...", lm_path);
        for id in 0..model.vocab_count() as u32 {
            if let Some(word_str) = model.get_word(id) {
                if !word_str.is_empty() && word_str.chars().all(|c| ('\u{0980}'..='\u{09FF}').contains(&c)) {
                    if is_junk_token(word_str) {
                        pruned_lm_junk += 1;
                        continue;
                    }
                    let log_p = model.get_unigram_prob(id).unwrap_or(-7.0);
                    let norm = ((log_p - (-7.0)) / (-2.0 - (-7.0))).clamp(0.0, 1.0);
                    let freq = (1000.0 + norm * 8800.0) as u32;
                    word_freq_map.insert(word_str.to_string(), freq);
                    mapped_lm_count += 1;
                }
            }
        }
    }
    println!("Mapped authentic corpus frequencies for {} words from 74M-word LM (pruned {} junk). Total dictionary entries: {}", mapped_lm_count, pruned_lm_junk, word_freq_map.len());

    let weighted_words: Vec<(String, u32)> = word_freq_map.into_iter().collect();

    let mut trie = PrefixTrie::new();
    trie.insert_bulk_weighted(weighted_words);
    trie.ensure_sorted();

    let target_paths = [
        "/mnt/data/lekhani-android/data/dictionaries/dictionary.bin",
        "/mnt/data/lekhani-android/android/app/src/main/assets/dictionaries/dictionary.bin",
        "/mnt/data/lekhani/data/dictionaries/dictionary.bin",
        "/mnt/data/lekhani-engine/data/dictionaries/dictionary.bin",
    ];

    for path in target_paths {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if parent.exists() {
                trie.save_binary(path)?;
                let meta = std::fs::metadata(path)?;
                println!("Saved {} ({} bytes, {:.2} MB)", path, meta.len(), meta.len() as f64 / 1_048_576.0);
            }
        }
    }

    Ok(())
}
