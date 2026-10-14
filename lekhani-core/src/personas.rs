//! Typing Personas Generator for Lekhani Benchmark Evaluation
//!
//! Generates reproducible, seeded Latin inputs corresponding to different
//! typist behaviors:
//! 1. Natural: Raw human-annotated romanizations (Dakshina / Aksharantar).
//! 2. Lazy: No-shift, lowercase, chat digraph folds (aa->a, ee->i, oo->u, w->o).
//! 3. VowelDropped(p): Non-initial vowels dropped with probability p (0.3, 0.6, 0.9).
//! 4. AvroStrict: Canonical Avro phonetic keystrokes forward-mapped from Bengali.

use crate::chars::canonicalize;

/// Ultra-fast, zero-dependency, deterministic 64-bit pseudo-random number generator (PCG/SplitMix)
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut x = self.state;
        x ^= x >> 27;
        x = x.wrapping_mul(0x2545F4914F6CDD1D);
        x ^= x >> 32;
        x
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Natural Persona: Raw human romanization as-is
pub fn transform_natural(input: &str) -> String {
    input.trim().to_string()
}

/// Lazy Persona: Lowercase, collapse Shift distinctions, and apply common chat folds
pub fn transform_lazy(input: &str) -> String {
    let lower = input.trim().to_ascii_lowercase();
    let mut out = lower
        .replace("aa", "a")
        .replace("ee", "i")
        .replace("oo", "u")
        .replace("w", "o");
    if out.is_empty() {
        out = input.to_ascii_lowercase();
    }
    out
}

/// Vowel-Dropped Persona: Drops non-initial vowel characters ('a', 'e', 'i', 'o', 'u') with probability p
pub fn transform_vowel_drop(input: &str, p: f64, rng: &mut SimpleRng) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut chars = trimmed.chars();
    let mut out = String::with_capacity(trimmed.len());

    // Always preserve the very first character
    if let Some(first) = chars.next() {
        out.push(first);
    }

    for c in chars {
        let is_vowel = matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U');
        if is_vowel && rng.next_f64() < p {
            // Drop vowel
            continue;
        }
        out.push(c);
    }

    out
}

/// Maps a single Bengali consonant to its basic Avro representation
fn cons_to_avro(c: char) -> Option<&'static str> {
    match c {
        'ক' => Some("k"),
        'খ' => Some("kh"),
        'গ' => Some("g"),
        'ঘ' => Some("gh"),
        'ঙ' => Some("Ng"),
        'চ' => Some("c"),
        'ছ' => Some("ch"),
        'জ' => Some("j"),
        'ঝ' => Some("jh"),
        'ঞ' => Some("NG"),
        'ট' => Some("T"),
        'ঠ' => Some("Th"),
        'ড' => Some("D"),
        'ঢ' => Some("Dh"),
        'ণ' => Some("N"),
        'ত' => Some("t"),
        'থ' => Some("th"),
        'দ' => Some("d"),
        'ধ' => Some("dh"),
        'ন' => Some("n"),
        'প' => Some("p"),
        'ফ' => Some("f"),
        'ব' => Some("b"),
        'ভ' => Some("bh"),
        'ম' => Some("m"),
        'য' => Some("z"),
        'র' => Some("r"),
        'ল' => Some("l"),
        'শ' => Some("sh"),
        'ষ' => Some("Sh"),
        'স' => Some("s"),
        'হ' => Some("h"),
        _ => None,
    }
}

/// Avro-Strict Persona: Maps a canonical Bengali word to its deterministic canonical Avro keystrokes
pub fn transform_avro_strict(bn_word: &str) -> String {
    let canonical = canonicalize(bn_word);
    let chars: Vec<char> = canonical.chars().collect();
    let mut out = String::with_capacity(chars.len() * 2);

    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];

        // 1. Nukta compounds (U+09BC)
        if i + 1 < chars.len() && chars[i + 1] == '\u{09BC}' {
            match c {
                'ড' => {
                    out.push('R');
                    i += 2;
                    continue;
                }
                'ঢ' => {
                    out.push_str("Rh");
                    i += 2;
                    continue;
                }
                'য' => {
                    out.push('Y');
                    i += 2;
                    continue;
                }
                _ => {}
            }
        }

        // 2. Modifiers (Terminal diacritics - never take inherent vowels)
        match c {
            'ং' => { out.push_str("ng"); i += 1; continue; }
            'ঃ' => { out.push(':'); i += 1; continue; }
            'ঁ' => { out.push('^'); i += 1; continue; }
            'ৎ' => { out.push_str("t``"); i += 1; continue; }
            _ => {}
        }

        // 3. Independent Vowels
        match c {
            'অ' => { out.push('o'); i += 1; continue; }
            'আ' => { out.push('a'); i += 1; continue; }
            'ই' => { out.push('i'); i += 1; continue; }
            'ঈ' => { out.push('I'); i += 1; continue; }
            'উ' => { out.push('u'); i += 1; continue; }
            'ঊ' => { out.push('U'); i += 1; continue; }
            'ঋ' => { out.push_str("rri"); i += 1; continue; }
            'এ' => { out.push('e'); i += 1; continue; }
            'ঐ' => { out.push_str("oi"); i += 1; continue; }
            'ও' => { out.push('O'); i += 1; continue; }
            'ঔ' => { out.push_str("ou"); i += 1; continue; }
            _ => {}
        }

        // 4. Dependent Vowel Signs (Kars)
        match c {
            'া' => { out.push('a'); i += 1; continue; }
            'ি' => { out.push('i'); i += 1; continue; }
            'ী' => { out.push('I'); i += 1; continue; }
            'ু' => { out.push('u'); i += 1; continue; }
            'ূ' => { out.push('U'); i += 1; continue; }
            'ৃ' => { out.push_str("rri"); i += 1; continue; }
            'ে' => { out.push('e'); i += 1; continue; }
            'ৈ' => { out.push_str("oi"); i += 1; continue; }
            'ো' => { out.push('O'); i += 1; continue; }
            'ৌ' => { out.push_str("ou"); i += 1; continue; }
            _ => {}
        }

        // 5. Special Conjuncts, Pholas, & Reph
        // 5a. Reph: র + ্ + Consonant -> "rr" + consonant
        if c == 'র' && i + 2 < chars.len() && chars[i + 1] == '্' {
            let next_c = chars[i + 2];
            if next_c != '্' && next_c != 'র' && next_c != 'য' {
                out.push_str("rr");
                i += 2;
                continue;
            }
        }

        if i + 2 < chars.len() && chars[i + 1] == '্' {
            let next_c = chars[i + 2];
            // ক + ্ + ষ -> kkh
            if c == 'ক' && next_c == 'ষ' {
                out.push_str("kkh");
                i += 3;
                if i < chars.len()
                    && !matches!(chars[i], 'া'..='ৌ' | '্' | '\u{09BC}' | 'ং' | 'ঃ' | 'ঁ')
                {
                    out.push('o');
                }
                continue;
            }
            // জ + ্ + ঞ -> gg
            if c == 'জ' && next_c == 'ঞ' {
                out.push_str("gg");
                i += 3;
                if i < chars.len()
                    && !matches!(chars[i], 'া'..='ৌ' | '্' | '\u{09BC}' | 'ং' | 'ঃ' | 'ঁ')
                {
                    out.push('o');
                }
                continue;
            }
            // ব-ফলা: Consonant + ্ + ব -> Consonant + w
            if next_c == 'ব' && c != 'ব' {
                if let Some(cs) = cons_to_avro(c) {
                    out.push_str(cs);
                    out.push('w');
                    i += 3;
                    if i < chars.len()
                        && !matches!(chars[i], 'া'..='ৌ' | '্' | '\u{09BC}' | 'ং' | 'ঃ' | 'ঁ')
                    {
                        out.push('o');
                    }
                    continue;
                }
            }
            // য-ফলা: Consonant + ্ + য -> Consonant + y
            if next_c == 'য' {
                if let Some(cs) = cons_to_avro(c) {
                    out.push_str(cs);
                    out.push('y');
                    i += 3;
                    if i < chars.len()
                        && !matches!(chars[i], 'া'..='ৌ' | '্' | '\u{09BC}' | 'ং' | 'ঃ' | 'ঁ')
                    {
                        out.push('o');
                    }
                    continue;
                }
            }
        }

        // 6. Consonants
        if let Some(cs) = cons_to_avro(c) {
            out.push_str(cs);
            // Check if next char needs inherent vowel 'o'
            if i + 1 < chars.len() {
                let next_c = chars[i + 1];
                let is_kar = matches!(next_c, 'া'..='ৌ');
                let is_hasanta = next_c == '্';
                let is_nukta = next_c == '\u{09BC}';
                let is_terminal = matches!(next_c, 'ং' | 'ঃ' | 'ঁ');
                if !is_kar && !is_hasanta && !is_nukta && !is_terminal {
                    out.push('o');
                }
            }
            i += 1;
            continue;
        }

        // 7. Virama (Hasanta)
        if c == '্' {
            i += 1;
            continue;
        }

        out.push(c);
        i += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_persona() {
        assert_eq!(transform_natural(" anri "), "anri");
        assert_eq!(transform_natural("akkharguli"), "akkharguli");
    }

    #[test]
    fn test_lazy_persona() {
        assert_eq!(transform_lazy("Taka"), "taka");
        assert_eq!(transform_lazy("Daktar"), "daktar");
        assert_eq!(transform_lazy("bhaai"), "bhai");
        assert_eq!(transform_lazy("deepto"), "dipto");
    }

    #[test]
    fn test_vowel_drop_persona() {
        let mut rng = SimpleRng::new(42);
        let res_p0 = transform_vowel_drop("bondhu", 0.0, &mut rng);
        assert_eq!(res_p0, "bondhu");

        let res_p100 = transform_vowel_drop("bondhu", 1.0, &mut rng);
        assert_eq!(res_p100, "bndh");

        let res_kmn = transform_vowel_drop("kemon", 1.0, &mut rng);
        assert_eq!(res_kmn, "kmn");
    }

    #[test]
    fn test_avro_strict_persona() {
        let parser = lekhani_parser::default_avro_parser();
        let test_words = ["ভারত", "স্বাধীনতা", "অক্ষর", "বন্ধু", "দেশ"];
        for w in &test_words {
            let avro_str = transform_avro_strict(w);
            let converted = parser.convert(&avro_str);
            assert_eq!(
                canonicalize(&converted),
                canonicalize(w),
                "Avro-strict forward mapping mismatch for '{}' (keystrokes: '{}', converted: '{}')",
                w,
                avro_str,
                converted
            );
        }
    }
}
