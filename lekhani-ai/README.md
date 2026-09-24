# Lekhani AI (`lekhani-ai`)

High-performance, pure Rust on-device statistical language modeling, contextual candidate scoring, homophone disambiguation, and next-word predictive typing for the Bengali language.

Designed for sub-microsecond inference latency (< 300 ns per evaluation), zero runtime heap allocations, and zero telemetry.

## Features

- **Blazing Fast**: Sub-microsecond N-gram candidate evaluation (~280 ns per candidate).
- **Statistical N-Gram Language Model**: Unigram, Bigram, and Trigram probabilities smoothed with Stupid Backoff.
- **Contextual Homophone Disambiguation**: Intelligently scores confusable Bengali homophones (e.g. `পড়া` vs `পরা`, `খাব` vs `যাব`) based on preceding words.
- **Next-Word Prediction**: Instantaneous continuation suggestions given preceding context tokens.
- **Beam Search Global Decoder**: Global sequence lattice decoding across candidate word alternatives.
- **Zero-Copy Memory-Mapped Architecture**: Loads precompiled binary models (`*.bin`) via `mmap` with zero copy and zero heap expansion.
- **Streaming Model Trainer**: Built-in multi-threaded trainer capable of compiling raw corpus text files into compact binary language models.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
lekhani-ai = "1.0"
```

### Quick Start Example

```rust
use lekhani_ai::{LanguageModel, ContextScorer, NextWordPredictor};

fn main() {
    let lm = LanguageModel::new();
    let scorer = ContextScorer::new();
    let predictor = NextWordPredictor::new();

    // 1. Trigram Scoring
    let score = lm.score_candidate(Some("আমি"), Some("ভাত"), "খাচ্ছি");
    println!("Log probability: {:.3}", score);

    // 2. Homophone Disambiguation
    let homophones = vec!["পড়া".to_string(), "পরা".to_string()];
    let ranked = scorer.rank_candidates(&["বই"], &homophones);
    assert_eq!(ranked[0], "পড়া"); // Correctly selects পড়া after বই

    // 3. Next-Word Prediction
    let next_words = predictor.predict_next(&["বাংলাদেশ", "একটি"], 3);
    println!("Predictions: {:?}", next_words);
}
```

## Running the Example

```bash
cargo run -p lekhani-ai --example score_candidates
```

## License

Licensed under GPL-3.0-or-later.
