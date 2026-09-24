# Lekhani Parser (`lekhani-parser`)

Ultra-fast, zero-allocation pure Rust Bengali phonetic grammar parser and transliterator.

`lekhani-parser` is an independent, standalone Rust library providing 100% Avro Phonetic muscle memory compatibility, compiled Longest-Prefix L1 Trie matching, branchless byte-level condition checks, and canonical diacritic sequencing. It is designed as a modern, high-performance drop-in replacement for `rupantor`.

## Features

- **Blazing Fast**: Sub-microsecond hot path (< 100 ns per typical word transliteration).
- **Zero Allocations on Keystroke**: `convert_into(&str, &mut String)` reuses caller-provided scratch buffers with zero heap allocations during transliteration.
- **Embedded Default Layout**: Ships with the standard Avro Phonetic grammar embedded into `.rodata` via `default_avro_parser()` — 0 ms boot time, zero runtime disk I/O.
- **Custom Layout Support**: Can parse and compile arbitrary Avro-compatible phonetic JSON schema files using `CompiledLayout::from_json`.
- **Fidelity**: 100% faithful to the official Avro Phonetic specification, accurately handling complex conjuncts (যুক্তবর্ণ), reph (র্), ja-phala (্য), chandra-bindu (ঁ), and case-sensitive phonetic nuances.
- **100% Safe Pure Rust**: No unsafe blocks, no C/C++ FFI dependencies, zero regex engines.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
lekhani-parser = "1.0"
```

### Basic Example

```rust
use lekhani_parser::default_avro_parser;

fn main() {
    let parser = default_avro_parser();

    // Standard string conversion
    let output = parser.convert("ami banglay gan gai");
    assert_eq!(output, "আমি বাংলায় গান গাই");

    // Zero-allocation buffer reuse
    let mut buffer = String::with_capacity(64);
    parser.convert_into("bangladesh", &mut buffer);
    assert_eq!(buffer, "বাংলাদেশ");
}
```

### Loading a Custom Layout

```rust
use std::sync::Arc;
use lekhani_parser::{CompiledLayout, LekhaniParser};

let json_str = r#"{"patterns": [...]}"#;
let value: serde_json::Value = serde_json::from_str(json_str).unwrap();
let layout = CompiledLayout::from_json(&value).unwrap();
let parser = LekhaniParser::new(Arc::new(layout));
```

## Architecture

1. **256-Byte Bitmask Character Classification Table**: Branchless $O(1)$ lookups for vowels, consonants, punctuation, digits, and case-sensitivity.
2. **Byte-Level Trie Index**: Pattern rules are indexed in an L1 prefix trie. Key matching is $O(K)$ where $K$ is the pattern length (typically 1 to 4 bytes).
3. **Canonical Diacritic Sequencing**: Automatically reorders modifier glyphs according to Bengali Unicode canonical rules (e.g. vowel sign before chandra-bindu, halant conjunct sequences).

## License

Licensed under GPL-3.0-or-later.
