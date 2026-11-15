#![allow(clippy::all)]
//! Tool: check_avro_lowercased
//! Measures the impact of lowercasing Avro-strict inputs on raw Avro parsing and the baseline engine.

use hashbrown::HashSet;
use lekhani_core::chars::canonicalize;
use lekhani_core::personas::transform_avro_strict;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let dev_path = "data/eval/bn.translit.sampled.dev.tsv";
    let file = File::open(dev_path).expect("Failed to open dev.tsv");
    let reader = BufReader::new(file);

    let mut dev_pairs = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = line.trim().split('\t').collect();
        if parts.len() >= 2 {
            let bn = canonicalize(parts[0].trim());
            let lat = parts[1].trim().to_string();
            if !bn.is_empty() && !lat.is_empty() {
                dev_pairs.push((bn, lat));
            }
        }
    }

    println!("Total DEV pairs: {}", dev_pairs.len());

    let parser = lekhani_parser::default_avro_parser();

    let mut uppercase_count = 0;
    let mut unique_targets = HashSet::new();
    for (bn, _) in &dev_pairs {
        unique_targets.insert(bn.clone());
    }

    let mut raw_avro_strict_hits = 0;
    let mut raw_avro_strict_lower_hits = 0;

    for bn in &unique_targets {
        let avro_str = transform_avro_strict(bn);
        if avro_str.chars().any(|c| c.is_ascii_uppercase()) {
            uppercase_count += 1;
        }

        let conv = canonicalize(&parser.convert(&avro_str));
        if &conv == bn {
            raw_avro_strict_hits += 1;
        }

        let lower = avro_str.to_lowercase();
        let conv_lower = canonicalize(&parser.convert(&lower));
        if &conv_lower == bn {
            raw_avro_strict_lower_hits += 1;
        }
    }

    let n_targets = unique_targets.len() as f64;
    println!("\nUnique targets: {}", unique_targets.len());
    println!(
        "Targets whose Avro-strict has uppercase letters: {} ({:.1}%)",
        uppercase_count,
        uppercase_count as f64 / n_targets * 100.0
    );
    println!(
        "Raw Avro parser on exact Avro-strict:      {}/{} ({:.2}%)",
        raw_avro_strict_hits,
        unique_targets.len(),
        raw_avro_strict_hits as f64 / n_targets * 100.0
    );
    println!(
        "Raw Avro parser on LOWERCASED Avro-strict:  {}/{} ({:.2}%)",
        raw_avro_strict_lower_hits,
        unique_targets.len(),
        raw_avro_strict_lower_hits as f64 / n_targets * 100.0
    );
    println!(
        "Drop due to lowercasing alone in raw Avro: -{:.2}% points!",
        (raw_avro_strict_hits - raw_avro_strict_lower_hits) as f64 / n_targets * 100.0
    );

    // Per-row evaluation across all 9,279 rows
    let mut row_strict_hits = 0;
    let mut row_lower_hits = 0;
    for (bn, _) in &dev_pairs {
        let avro_str = transform_avro_strict(bn);
        if &canonicalize(&parser.convert(&avro_str)) == bn {
            row_strict_hits += 1;
        }
        if &canonicalize(&parser.convert(&avro_str.to_lowercase())) == bn {
            row_lower_hits += 1;
        }
    }
    let n_rows = dev_pairs.len() as f64;
    println!("\nPer-Row Evaluation (N = 9,279):");
    println!(
        "Raw Avro parser on exact Avro-strict:      {}/{} ({:.2}%)",
        row_strict_hits,
        dev_pairs.len(),
        row_strict_hits as f64 / n_rows * 100.0
    );
    println!(
        "Raw Avro parser on LOWERCASED Avro-strict:  {}/{} ({:.2}%)",
        row_lower_hits,
        dev_pairs.len(),
        row_lower_hits as f64 / n_rows * 100.0
    );
    println!(
        "Drop due to lowercasing alone per-row:     -{:.2}% points!",
        (row_strict_hits - row_lower_hits) as f64 / n_rows * 100.0
    );
}
