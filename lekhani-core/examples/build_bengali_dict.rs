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

    let mut trie = PrefixTrie::new();
    trie.insert_bulk(words);
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
