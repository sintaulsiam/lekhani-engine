//! Bijoy (ANSI) ⇄ Unicode Bengali Conversion Engine

use hashbrown::HashMap;
use std::sync::LazyLock;

static BIJOY_TO_UNICODE: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::with_capacity(100);
    // Vowels
    m.insert("Av", "আ");
    m.insert("A", "অ");
    m.insert("B", "ই");
    m.insert("C", "ঈ");
    m.insert("D", "উ");
    m.insert("E", "ঊ");
    m.insert("F", "ঋ");
    m.insert("G", "এ");
    m.insert("H", "ঐ");
    m.insert("I", "ও");
    m.insert("J", "ঔ");

    // Consonants
    m.insert("K", "ক");
    m.insert("L", "খ");
    m.insert("M", "গ");
    m.insert("N", "ঘ");
    m.insert("O", "ঙ");
    m.insert("P", "চ");
    m.insert("Q", "ছ");
    m.insert("R", "জ");
    m.insert("S", "ঝ");
    m.insert("T", "ঞ");
    m.insert("U", "ট");
    m.insert("V", "ঠ");
    m.insert("W", "ড");
    m.insert("X", "ঢ");
    m.insert("Y", "ণ");
    m.insert("Z", "ত");
    m.insert("_", "থ");
    m.insert("`", "দ");
    m.insert("a", "ধ");
    m.insert("b", "ন");
    m.insert("c", "প");
    m.insert("d", "ফ");
    m.insert("e", "ব");
    m.insert("f", "ভ");
    m.insert("g", "ম");
    m.insert("h", "য");
    m.insert("i", "র");
    m.insert("j", "ল");
    m.insert("k", "শ");
    m.insert("l", "ষ");
    m.insert("m", "স");
    m.insert("n", "হ");
    m.insert("o", "ড়");
    m.insert("p", "ঢ়");
    m.insert("q", "য়");
    m.insert("r", "ৎ");
    m.insert("s", "ং");
    m.insert("t", "ঃ");
    m.insert("u", "ঁ");

    // Kars
    m.insert("v", "া");
    m.insert("w", "ি");
    m.insert("x", "ী");
    m.insert("y", "ু");
    m.insert("~", "ূ");
    m.insert("„", "ৃ");
    m.insert("†", "ে");
    m.insert("‰", "ৈ");
    m.insert("Š", "ৌ");

    // Digits
    m.insert("0", "০");
    m.insert("1", "১");
    m.insert("2", "২");
    m.insert("3", "৩");
    m.insert("4", "৪");
    m.insert("5", "৫");
    m.insert("6", "৬");
    m.insert("7", "৭");
    m.insert("8", "৮");
    m.insert("9", "৯");

    // Punctuation
    m.insert("|", "।");

    m
});

static UNICODE_TO_BIJOY: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut u_to_b = HashMap::with_capacity(100);
    for (&k, &v) in BIJOY_TO_UNICODE.iter() {
        u_to_b.insert(v, k);
    }
    u_to_b
});

/// Convert Bijoy (ANSI / SutonnyMJ) encoded text to Unicode Bengali
pub fn bijoy_to_unicode(input: &str) -> String {
    let mut result = String::with_capacity(input.len() * 2);
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Check for 3-char clusters
        if i + 2 < len {
            let sub: String = chars[i..i + 3].iter().collect();
            if let Some(unicode) = BIJOY_TO_UNICODE.get(sub.as_str()) {
                result.push_str(unicode);
                i += 3;
                continue;
            }
        }

        // Check for 2-char clusters
        if i + 1 < len {
            let sub: String = chars[i..i + 2].iter().collect();
            if let Some(unicode) = BIJOY_TO_UNICODE.get(sub.as_str()) {
                result.push_str(unicode);
                i += 2;
                continue;
            }
        }

        // Single char
        let single: String = chars[i..i + 1].iter().collect();
        if let Some(unicode) = BIJOY_TO_UNICODE.get(single.as_str()) {
            result.push_str(unicode);
        } else {
            result.push(chars[i]);
        }
        i += 1;
    }

    post_process_bijoy_to_unicode(&result)
}

/// Convert Unicode Bengali text to Bijoy (ANSI / SutonnyMJ)
pub fn unicode_to_bijoy(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Multi-char clusters
        if i + 2 < len {
            let sub: String = chars[i..i + 3].iter().collect();
            if let Some(bijoy) = UNICODE_TO_BIJOY.get(sub.as_str()) {
                result.push_str(bijoy);
                i += 3;
                continue;
            }
        }

        if i + 1 < len {
            let sub: String = chars[i..i + 2].iter().collect();
            if let Some(bijoy) = UNICODE_TO_BIJOY.get(sub.as_str()) {
                result.push_str(bijoy);
                i += 2;
                continue;
            }
        }

        let single: String = chars[i..i + 1].iter().collect();
        if let Some(bijoy) = UNICODE_TO_BIJOY.get(single.as_str()) {
            result.push_str(bijoy);
        } else {
            result.push(chars[i]);
        }
        i += 1;
    }

    result
}

fn post_process_bijoy_to_unicode(input: &str) -> String {
    let mut chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i + 1 < chars.len() {
        // Fix misplaced E-Kar / I-Kar (if kar is before consonant, swap them)
        if (chars[i] == 'ি' || chars[i] == 'ে' || chars[i] == 'ৈ') && is_consonant(chars[i + 1]) {
            chars.swap(i, i + 1);
        }
        // Fix Ref (্ + র at end of consonant)
        if chars[i] == '©' { // Bijoy Reph
            chars[i] = 'র';
            chars.insert(i + 1, '্');
        }
        i += 1;
    }
    chars.into_iter().collect()
}

fn is_consonant(c: char) -> bool {
    matches!(c, '\u{0995}'..='\u{09B9}' | '\u{09DC}'..='\u{09DF}' | '\u{09CE}')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bijoy_conversion() {
        assert_eq!(bijoy_to_unicode("evsjv"), "বাংলা");
        assert_eq!(bijoy_to_unicode("Avwg"), "আমি");
    }
}
