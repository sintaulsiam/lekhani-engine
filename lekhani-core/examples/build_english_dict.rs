use std::fs::File;
use std::io::{BufRead, BufReader};
use lekhani_core::trie::PrefixTrie;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Building English PrefixTrie dictionary from top60k.txt...");
    let file = File::open("crates/lekhani-core/examples/top60k.txt")?;
    let reader = BufReader::new(file);

    let mut entries: Vec<(String, u32)> = Vec::with_capacity(65000);

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let word = parts[0].to_lowercase();
        let raw_freq: u64 = parts[1].parse().unwrap_or(100);

        // Filter valid English words
        if word.is_empty() || word.len() > 24 {
            continue;
        }
        if !word.chars().all(|c| c.is_ascii_lowercase() || c == '\'') {
            continue;
        }

        // Scale raw frequencies logarithmically to fit nicely in u32 (100 to 1,000,000)
        // ln(23135851162) ~ 23.86, ln(243000) ~ 12.4
        let scaled_freq = (((raw_freq as f64).ln() - 10.0).max(1.0) * 50000.0) as u32;
        entries.push((word, scaled_freq));
    }

    println!("Filtered {} valid English words.", entries.len());

    let mut trie = PrefixTrie::new();
    trie.insert_bulk_weighted(entries);
    trie.ensure_sorted();

    let output_path = "data/dictionaries/english_dict.bin";
    trie.save_binary(output_path)?;

    let metadata = std::fs::metadata(output_path)?;
    println!("Successfully saved {} ({} bytes, {:.2} MB)", output_path, metadata.len(), metadata.len() as f64 / 1_048_576.0);

    // Verify loading
    let bytes = std::fs::read(output_path)?;
    let loaded = PrefixTrie::from_binary(&bytes)?;
    println!("Loaded dictionary back: verified {} entries", loaded.memory_usage());

    // Test some common prefixes
    for prefix in ["th", "hel", "prog", "lekh"] {
        let matches = loaded.find_prefix_matches(prefix, 5);
        println!("Prefix '{}' -> {:?}", prefix, matches);
    }

    // Clean up temporary text file
    let _ = std::fs::remove_file("crates/lekhani-core/examples/top60k.txt");

    Ok(())
}
