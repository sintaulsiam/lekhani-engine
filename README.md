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
lekhani-parser = "1.1.0"
lekhani-core   = "1.1.0"
lekhani-ai     = "1.1.0"
lekhani-neural = "1.1.0"
```

### From git (latest)

```toml
lekhani-parser = { git = "https://github.com/sintaulsiam/lekhani-engine", tag = "v1.1.0" }
lekhani-core   = { git = "https://github.com/sintaulsiam/lekhani-engine", tag = "v1.1.0" }
lekhani-ai     = { git = "https://github.com/sintaulsiam/lekhani-engine", tag = "v1.1.0" }
lekhani-neural = { git = "https://github.com/sintaulsiam/lekhani-engine", tag = "v1.1.0" }
```

## Platform consumers

- **Android IME**: [sintaulsiam/lekhani-android](https://github.com/sintaulsiam/lekhani-android)
- **Linux / Windows keyboard**: [sintaulsiam/lekhani](https://github.com/sintaulsiam/lekhani)

## License

GPL-3.0-or-later © Sintaul Mahdi Siam
