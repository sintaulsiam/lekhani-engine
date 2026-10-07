use hashbrown::HashMap;
use lekhani_ai::embeddings::{WordEmbeddings, EMBEDDING_DIM};
use lekhani_ai::trainer::is_junk_token;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const VOCAB_SIZE: usize = 4096;
const WINDOW_SIZE: usize = 4;
const POWER_ITERATIONS: usize = 3;

fn main() -> anyhow::Result<()> {
    let start_time = Instant::now();
    println!("=== Lekhani 32-D Dense Semantic Word Embeddings Trainer ===");

    let corpus_path = PathBuf::from("./data/corpus/bengali_corpus.txt");
    if !corpus_path.exists() {
        anyhow::bail!("Corpus file not found at {:?}", corpus_path);
    }

    println!("[1/5] Pass 1: Scanning unigram frequencies from {:?}...", corpus_path);
    let file = File::open(&corpus_path)?;
    let reader = BufReader::with_capacity(4 * 1024 * 1024, file);

    let mut unigram_counts: HashMap<String, u32> = HashMap::with_capacity(100_000);
    let mut total_tokens = 0usize;

    for line_result in reader.lines() {
        let line = line_result?;
        for raw_tok in line.split_whitespace() {
            let tok = raw_tok.trim_matches(|c: char| {
                c.is_ascii_punctuation()
                    || c == '।'
                    || c == '—'
                    || c == '‘'
                    || c == '’'
                    || c == '“'
                    || c == '”'
                    || c == '\''
                    || c == '"'
                    || c == ','
            });
            if !tok.is_empty() && !is_junk_token(tok) {
                *unigram_counts.entry(tok.to_string()).or_insert(0) += 1;
                total_tokens += 1;
            }
        }
    }

    println!(
        "Processed {} total tokens. Unique vocabulary: {} words.",
        total_tokens,
        unigram_counts.len()
    );

    println!("[2/5] Selecting top {} vocabulary words...", VOCAB_SIZE);
    let mut sorted_vocab: Vec<(String, u32)> = unigram_counts.into_iter().collect();
    sorted_vocab.sort_by(|a, b| b.1.cmp(&a.1));
    sorted_vocab.truncate(VOCAB_SIZE);

    let mut word_to_id: HashMap<String, usize> = HashMap::with_capacity(VOCAB_SIZE);
    let mut id_to_word: Vec<String> = Vec::with_capacity(VOCAB_SIZE);
    for (i, (word, _)) in sorted_vocab.iter().enumerate() {
        word_to_id.insert(word.clone(), i);
        id_to_word.push(word.clone());
    }

    println!("[3/5] Pass 2: Accumulating symmetric co-occurrence matrix (window={})...", WINDOW_SIZE);
    let file = File::open(&corpus_path)?;
    let reader = BufReader::with_capacity(4 * 1024 * 1024, file);

    let mut cooccur = vec![0u32; VOCAB_SIZE * VOCAB_SIZE];
    let mut sentence_ids = Vec::with_capacity(64);

    for line_result in reader.lines() {
        let line = line_result?;
        sentence_ids.clear();

        for raw_tok in line.split_whitespace() {
            let tok = raw_tok.trim_matches(|c: char| {
                c.is_ascii_punctuation()
                    || c == '।'
                    || c == '—'
                    || c == '‘'
                    || c == '’'
                    || c == '“'
                    || c == '”'
                    || c == '\''
                    || c == '"'
                    || c == ','
            });
            if let Some(&id) = word_to_id.get(tok) {
                sentence_ids.push(id);
            }
        }

        let len = sentence_ids.len();
        for i in 0..len {
            let id_i = sentence_ids[i];
            let start = i.saturating_sub(WINDOW_SIZE);
            let end = (i + WINDOW_SIZE + 1).min(len);
            for j in start..end {
                if i != j {
                    let id_j = sentence_ids[j];
                    cooccur[id_i * VOCAB_SIZE + id_j] += 1;
                }
            }
        }
    }

    println!("[4/5] Computing Positive Pointwise Mutual Information (PPMI)...");
    let mut row_sums = vec![0.0f64; VOCAB_SIZE];
    let mut total_cooccur = 0.0f64;

    for i in 0..VOCAB_SIZE {
        let mut r = 0.0f64;
        let row_start = i * VOCAB_SIZE;
        for j in 0..VOCAB_SIZE {
            r += cooccur[row_start + j] as f64;
        }
        row_sums[i] = r;
        total_cooccur += r;
    }

    let mut ppmi = vec![0.0f32; VOCAB_SIZE * VOCAB_SIZE];
    let log_total = total_cooccur.ln();

    for i in 0..VOCAB_SIZE {
        let row_i_sum = row_sums[i];
        if row_i_sum <= 0.0 {
            continue;
        }
        let log_row_i = row_i_sum.ln();
        let row_start = i * VOCAB_SIZE;

        for j in 0..VOCAB_SIZE {
            let count = cooccur[row_start + j];
            if count > 0 {
                let row_j_sum = row_sums[j];
                if row_j_sum > 0.0 {
                    let log_pmi = (count as f64).ln() + log_total - log_row_i - row_j_sum.ln();
                    if log_pmi > 0.0 {
                        ppmi[row_start + j] = log_pmi as f32;
                    }
                }
            }
        }
    }

    // Free cooccur to save RAM
    drop(cooccur);

    println!("[5/5] Spherical Subspace Matrix Factorization to {} dimensions...", EMBEDDING_DIM);
    // Initialize random projection matrix Omega (VOCAB_SIZE x EMBEDDING_DIM)
    // Deterministic pseudo-random number generator (xorshift64)
    let mut seed = 0x8543290123891047u64;
    let mut rand_f32 = || -> f32 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed as f32) / (u64::MAX as f32)) * 2.0 - 1.0
    };

    let mut y = vec![0.0f32; VOCAB_SIZE * EMBEDDING_DIM];
    for val in &mut y {
        *val = rand_f32();
    }

    // Power iterations: Y = PPMI * (PPMI * Y)
    let mut temp = vec![0.0f32; VOCAB_SIZE * EMBEDDING_DIM];
    for iter in 0..POWER_ITERATIONS {
        println!("  Subspace Power Iteration {}/{}...", iter + 1, POWER_ITERATIONS);
        // temp = PPMI * Y
        temp.fill(0.0);
        for i in 0..VOCAB_SIZE {
            let row_start = i * VOCAB_SIZE;
            for j in 0..VOCAB_SIZE {
                let p = ppmi[row_start + j];
                if p > 0.0 {
                    let y_offset = j * EMBEDDING_DIM;
                    let temp_offset = i * EMBEDDING_DIM;
                    for k in 0..EMBEDDING_DIM {
                        temp[temp_offset + k] += p * y[y_offset + k];
                    }
                }
            }
        }

        // y = PPMI * temp
        y.fill(0.0);
        for i in 0..VOCAB_SIZE {
            let row_start = i * VOCAB_SIZE;
            for j in 0..VOCAB_SIZE {
                let p = ppmi[row_start + j];
                if p > 0.0 {
                    let temp_offset = j * EMBEDDING_DIM;
                    let y_offset = i * EMBEDDING_DIM;
                    for k in 0..EMBEDDING_DIM {
                        y[y_offset + k] += p * temp[temp_offset + k];
                    }
                }
            }
        }

        // Orthonormalize columns of Y via modified Gram-Schmidt
        for col in 0..EMBEDDING_DIM {
            // Subtract projections onto prior columns
            for prev_col in 0..col {
                let mut dot = 0.0f32;
                for row in 0..VOCAB_SIZE {
                    dot += y[row * EMBEDDING_DIM + col] * y[row * EMBEDDING_DIM + prev_col];
                }
                for row in 0..VOCAB_SIZE {
                    y[row * EMBEDDING_DIM + col] -= dot * y[row * EMBEDDING_DIM + prev_col];
                }
            }
            // Normalize column
            let mut norm_sq = 0.0f32;
            for row in 0..VOCAB_SIZE {
                let v = y[row * EMBEDDING_DIM + col];
                norm_sq += v * v;
            }
            let norm = norm_sq.sqrt().max(1e-10);
            for row in 0..VOCAB_SIZE {
                y[row * EMBEDDING_DIM + col] /= norm;
            }
        }
    }

    // Build WordEmbeddings container
    let mut embeddings = WordEmbeddings::new();
    for i in 0..VOCAB_SIZE {
        let word = &id_to_word[i];
        let mut vec = [0.0f32; EMBEDDING_DIM];
        for k in 0..EMBEDDING_DIM {
            vec[k] = y[i * EMBEDDING_DIM + k];
        }
        embeddings.insert(word, vec);
    }

    // Verification of semantic proximity
    println!("\n=== Semantic Proximity Verification ===");
    let test_pairs = &[
        ("আজ", "কাল"),
        ("ভালো", "সুন্দর"),
        ("ভাত", "খাবার"),
        ("বই", "পড়া"),
        ("বাংলাদেশ", "ঢাকা"),
        ("কেমন", "আছেন"),
        ("খারাপ", "ভালো"),
        ("ভাত", "কম্পিউটার"),
    ];

    for &(w1, w2) in test_pairs {
        let sim = embeddings.cosine_similarity(w1, w2);
        println!("  sim({:8}, {:8}) = {:+.4}", w1, w2, sim);
    }

    let bin_bytes = embeddings.save_binary();
    println!("\nSerialized embeddings binary size: {} KB", bin_bytes.len() / 1024);

    let target_paths = &[
        Path::new("./data/dictionaries/bengali_embeddings.bin"),
        Path::new("/mnt/data/lekhani/data/dictionaries/bengali_embeddings.bin"),
        Path::new("/mnt/data/lekhani-android/data/dictionaries/bengali_embeddings.bin"),
        Path::new("/mnt/data/lekhani-android/android/app/src/main/assets/dictionaries/bengali_embeddings.bin"),
    ];

    for path in target_paths {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut out = File::create(path)?;
        out.write_all(&bin_bytes)?;
        println!("Successfully written to {:?}", path);
    }

    println!("\nEmbeddings training completed in {:.2}s!", start_time.elapsed().as_secs_f32());
    Ok(())
}
