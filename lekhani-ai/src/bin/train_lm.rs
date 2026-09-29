//! Lekhani Language Model Training & Compilation CLI
//!
//! Trains quantized LLM3 language models from raw corpora with 4-gram support
//! and merges with baseline dictionaries.

use lekhani_ai::trainer::{train_files_streaming, TrainedLanguageModelData, TrainingConfig};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: train_lm --output <path> --corpus <path1> [--corpus <path2>...] [--merge-baseline <path>] [--format <llm2|llm3>]");
        std::process::exit(1);
    }

    let mut output_path: Option<PathBuf> = None;
    let mut corpus_paths: Vec<PathBuf> = Vec::new();
    let mut baseline_path: Option<PathBuf> = None;
    let mut format = "llm3".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                i += 1;
                output_path = Some(PathBuf::from(&args[i]));
            }
            "--corpus" => {
                i += 1;
                corpus_paths.push(PathBuf::from(&args[i]));
            }
            "--merge-baseline" => {
                i += 1;
                baseline_path = Some(PathBuf::from(&args[i]));
            }
            "--format" => {
                i += 1;
                format = args[i].clone();
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let output_path = output_path.expect("--output is required");
    if corpus_paths.is_empty() {
        eprintln!("At least one --corpus is required");
        std::process::exit(1);
    }

    println!("[*] Training language model from {} corpus file(s)...", corpus_paths.len());
    for p in &corpus_paths {
        println!("    - {}", p.display());
    }

    let config = TrainingConfig::llm3_production();

    // Pass 1 & 2: Streaming training
    println!("[*] Running parallel streaming training...");
    let mut trained = train_files_streaming(&corpus_paths, &config)?;
    println!(
        "[*] Trained: {} unigrams, {} bigrams, {} trigrams, {} fourgrams",
        trained.unigram_count(),
        trained.bigram_count(),
        trained.trigram_count(),
        trained.fourgram_count(),
    );

    // Merge baseline if requested
    if let Some(ref base) = baseline_path {
        println!("[*] Merging with baseline model from {}...", base.display());
        let baseline_bytes = std::fs::read(base)?;
        let baseline = TrainedLanguageModelData::from_binary(&baseline_bytes)?;
        println!(
            "[*] Baseline: {} unigrams, {} bigrams, {} trigrams",
            baseline.unigram_count(),
            baseline.bigram_count(),
            baseline.trigram_count(),
        );
        trained.merge(&baseline);
        println!(
            "[*] Merged model: {} unigrams, {} bigrams, {} trigrams, {} fourgrams",
            trained.unigram_count(),
            trained.bigram_count(),
            trained.trigram_count(),
            trained.fourgram_count(),
        );
    }

    // Export to binary format
    println!("[*] Compiling to binary format ({format})...");
    let binary_bytes = match format.as_str() {
        "llm3" => trained.to_binary_llm3(),
        _ => trained.to_binary(),
    };

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output_path, &binary_bytes)?;

    let size_mb = binary_bytes.len() as f64 / (1024.0 * 1024.0);
    println!(
        "[+] Successfully generated {} ({} bytes, {:.2} MB)",
        output_path.display(),
        binary_bytes.len(),
        size_mb
    );

    // Verification check: roundtrip load
    println!("[*] Verifying binary integrity via zero-copy roundtrip...");
    let loaded = TrainedLanguageModelData::from_binary(&binary_bytes)?;
    assert_eq!(loaded.unigram_count(), trained.unigram_count());
    println!("[✓] Verification passed cleanly!");

    Ok(())
}
