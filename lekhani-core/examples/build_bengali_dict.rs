use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use lekhani_core::trie::PrefixTrie;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Building Bengali PrefixTrie dictionary from dictionary.json...");
    let dict_path = "/mnt/data/lekhani-android/data/dictionaries/dictionary.json";
    let file = File::open(dict_path)?;
    let reader = BufReader::new(file);
    let raw_map: HashMap<String, Vec<String>> = serde_json::from_reader(reader)?;

    let mut words = Vec::with_capacity(160000);
    for (_k, v_list) in raw_map {
        words.extend(v_list);
    }
    println!("Loaded {} words from dictionary.json", words.len());

    let lm_path = "/mnt/data/lekhani-engine/data/dictionaries/bengali_lm.bin";
    let lm = if std::path::Path::new(lm_path).exists() {
        println!("Loading 74M-word language model from {}...", lm_path);
        lekhani_ai::zero_copy::ZeroCopyLanguageModel::from_file(lm_path).ok()
    } else {
        None
    };

    let mut weighted_words = Vec::with_capacity(words.len());
    let mut mapped_lm_count = 0;
    for word in words {
        let mut freq = 100;
        if let Some(ref model) = lm {
            if let Some(w_id) = model.get_word_id(&word) {
                let log_p = model.get_unigram_prob(w_id).unwrap_or(-7.0);
                // log_p ranges roughly from -1.7 (super frequent) to -7.0 (rare)
                let norm = ((log_p - (-7.0)) / (-2.0 - (-7.0))).clamp(0.0, 1.0);
                freq = (1000.0 + norm * 8800.0) as u32;
                mapped_lm_count += 1;
            }
        }
        weighted_words.push((word, freq));
    }
    println!("Mapped authentic corpus frequencies for {} / {} words from 74M-word LM.", mapped_lm_count, weighted_words.len());

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
