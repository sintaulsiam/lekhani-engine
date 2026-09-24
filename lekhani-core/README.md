# Lekhani Core (`lekhani-core`)

Core Bengali Input Method Engine (IME) state machine, candidate ranking, dictionary Trie, autocorrect, and session management framework.

Powering Lekhani's desktop input methods across Fcitx5, IBus, and Wayland.

## Features

- **Multi-Layout Support**: Avro Phonetic (via `lekhani-parser`), Probhat, National, Munir, Borno, and Bijoy.
- **Trie-Based Dictionary**: Fast sub-microsecond prefix lookups and candidate generation.
- **Statistical AI Ranking**: Integrated with `lekhani-ai` for contextual homophone ranking and next-word suggestions.
- **User Learning & Personalization**: Dynamic frequency adaptation and committed word observation.
- **Bijoy to Unicode Converter**: Fast bi-directional transliteration.

## Usage

```rust
use lekhani_core::InputSession;

let mut session = InputSession::new();
// Handle keystrokes, candidate suggestions, and commitments
```

## License

Licensed under GPL-3.0-or-later.
