# lekhani-engine

Pure Rust engine crates powering the **Lekhani** Bengali input method across all platforms.

## Crates

| Crate | Description |
|---|---|
| [`lekhani-parser`](lekhani-parser/) | Ultra-fast, zero-allocation Avro Phonetic grammar parser and transliterator (11 ns/char) |
| [`lekhani-core`](lekhani-core/) | Headless IME state machine — fixed layouts, Trie dictionary, Bijoy converter, morphemes, autocorrect |
| [`lekhani-ai`](lekhani-ai/) | On-device quantized LLM3 N-gram LM, beam search arena, contextual candidate ranking |
| [`lekhani-neural`](lekhani-neural/) | Micro-neural INT8 GRU recurrent language model and bilingual BPE subword predictor |

## Usage in your project

### From crates.io

```toml
lekhani-parser = "1.2.0"
lekhani-core   = "1.2.0"
lekhani-ai     = "1.2.0"
lekhani-neural = "1.2.0"
```

### From git (latest)

```toml
lekhani-parser = { git = "https://github.com/sintaulsiam/lekhani-engine", tag = "v1.2.0" }
lekhani-core   = { git = "https://github.com/sintaulsiam/lekhani-engine", tag = "v1.2.0" }
lekhani-ai     = { git = "https://github.com/sintaulsiam/lekhani-engine", tag = "v1.2.0" }
lekhani-neural = { git = "https://github.com/sintaulsiam/lekhani-engine", tag = "v1.2.0" }
```

## Platform consumers

- **Android IME**: [sintaulsiam/lekhani-android](https://github.com/sintaulsiam/lekhani-android)
- **Linux / Windows keyboard**: [sintaulsiam/lekhani](https://github.com/sintaulsiam/lekhani)

## Core Innovations: Project বোধ (Bodh)

1. **Project "অন্বয়" (Anvaya)**: Pragmatic & Honorific Agreement Finite State Lattice (`lekhani-ai::anvaya`). Evaluates sociolinguistic honorific tiers (Formal `আপনি`, Familiar `তুমি`, Intimate `তুই`) and discourse styles (Cholit vs Sadhu) with zero heap allocations in < 50 ns, proactively generating and scoring concordant verbs (e.g. `[আপনি, কেমন] + aso` ➔ `আছেন`).
2. **Project "রূপমূল" (Rupmul)**: Multi-Tier Agglutinative Morphological Synthesizer & DAG Verification (`lekhani-core::morpheme`). Verifies strict Bengali morpheme transition DAGs (Root ➔ Classifier ➔ Plural ➔ Case ➔ Particle) and synthesizes valid Sandhi compound inflections (e.g. `মানুষগুলোরও`) without manual dictionary explosion.
3. **Project "ধ্বনি-সেতু" (Dhvani-Setu)**: Phonological Acoustic-Spatial Latent Distance Engine (`lekhani-core::phonetic::fuzzy`). Clusters phonetic consonant families (sibilants, palatals, aspirates) and syncope vowel reductions, unifying Romanized phonetic variations (`shundor`, `sundor`, `shndr`) in $O(1)$ space.
4. **Project "পশ্চাৎ-শোধন" (Retro-Correction)**: Clause-level contextual retro-correction engine (`lekhani-core::session::RetroCorrection`, `AndroidLekhaniSession`). Surfaces non-intrusive 1-tap quick fix chips for honorific mismatches and homophone errors (`পরা বই` ➔ `পড়া`, `আপনি আছো` ➔ `আছেন`).

## License

GPL-3.0-or-later © Sintaul Mahdi Siam

