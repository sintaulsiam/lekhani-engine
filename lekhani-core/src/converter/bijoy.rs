//! Bijoy (ANSI) ⇄ Unicode Bengali Conversion Engine

use hashbrown::HashMap;

/// Convert Bijoy (ANSI / SutonnyMJ) encoded text to Unicode Bengali
pub fn bijoy_to_unicode(input: &str) -> String {
    let text = input.to_string();

    // Reorder pre-kar symbols in Bijoy:
    // In Bijoy: 'w' (i-kar), 'e' (e-kar), '†' (e-kar), '‰' (oi-kar) appear BEFORE consonant
    // We adjust their positions so standard Unicode mappings succeed.

    let mut result = String::with_capacity(text.len() * 2);
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    let map = get_bijoy_to_unicode_map();

    while i < len {
        // Check for 3-char clusters
        if i + 2 < len {
            let sub: String = chars[i..i + 3].iter().collect();
            if let Some(unicode) = map.get(&sub) {
                result.push_str(unicode);
                i += 3;
                continue;
            }
        }

        // Check for 2-char clusters
        if i + 1 < len {
            let sub: String = chars[i..i + 2].iter().collect();
            if let Some(unicode) = map.get(&sub) {
                result.push_str(unicode);
                i += 2;
                continue;
            }
        }

        // Single char
        let single: String = chars[i..i + 1].iter().collect();
        if let Some(unicode) = map.get(&single) {
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
    let map = get_unicode_to_bijoy_map();
    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Multi-char clusters
        if i + 2 < len {
            let sub: String = chars[i..i + 3].iter().collect();
            if let Some(bijoy) = map.get(&sub) {
                result.push_str(bijoy);
                i += 3;
                continue;
            }
        }

        if i + 1 < len {
            let sub: String = chars[i..i + 2].iter().collect();
            if let Some(bijoy) = map.get(&sub) {
                result.push_str(bijoy);
                i += 2;
                continue;
            }
        }

        let single: String = chars[i..i + 1].iter().collect();
        if let Some(bijoy) = map.get(&single) {
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

fn get_bijoy_to_unicode_map() -> HashMap<String, String> {
    let mut m = HashMap::new();
    // Vowels
    m.insert("Av".to_string(), "আ".to_string());
    m.insert("A".to_string(), "অ".to_string());
    m.insert("B".to_string(), "ই".to_string());
    m.insert("C".to_string(), "ঈ".to_string());
    m.insert("D".to_string(), "উ".to_string());
    m.insert("E".to_string(), "ঊ".to_string());
    m.insert("F".to_string(), "ঋ".to_string());
    m.insert("G".to_string(), "এ".to_string());
    m.insert("H".to_string(), "ঐ".to_string());
    m.insert("I".to_string(), "ও".to_string());
    m.insert("J".to_string(), "ঔ".to_string());

    // Consonants
    m.insert("K".to_string(), "ক".to_string());
    m.insert("L".to_string(), "খ".to_string());
    m.insert("M".to_string(), "গ".to_string());
    m.insert("N".to_string(), "ঘ".to_string());
    m.insert("O".to_string(), "ঙ".to_string());
    m.insert("P".to_string(), "চ".to_string());
    m.insert("Q".to_string(), "ছ".to_string());
    m.insert("R".to_string(), "জ".to_string());
    m.insert("S".to_string(), "ঝ".to_string());
    m.insert("T".to_string(), "ঞ".to_string());
    m.insert("U".to_string(), "ট".to_string());
    m.insert("V".to_string(), "ঠ".to_string());
    m.insert("W".to_string(), "ড".to_string());
    m.insert("X".to_string(), "ঢ".to_string());
    m.insert("Y".to_string(), "ণ".to_string());
    m.insert("Z".to_string(), "ত".to_string());
    m.insert("_".to_string(), "থ".to_string());
    m.insert("`".to_string(), "দ".to_string());
    m.insert("a".to_string(), "ধ".to_string());
    m.insert("b".to_string(), "ন".to_string());
    m.insert("c".to_string(), "প".to_string());
    m.insert("d".to_string(), "ফ".to_string());
    m.insert("e".to_string(), "ব".to_string());
    m.insert("f".to_string(), "ভ".to_string());
    m.insert("g".to_string(), "ম".to_string());
    m.insert("h".to_string(), "য".to_string());
    m.insert("i".to_string(), "র".to_string());
    m.insert("j".to_string(), "ল".to_string());
    m.insert("k".to_string(), "শ".to_string());
    m.insert("l".to_string(), "ষ".to_string());
    m.insert("m".to_string(), "স".to_string());
    m.insert("n".to_string(), "হ".to_string());
    m.insert("o".to_string(), "ড়".to_string());
    m.insert("p".to_string(), "ঢ়".to_string());
    m.insert("q".to_string(), "য়".to_string());
    m.insert("r".to_string(), "ৎ".to_string());
    m.insert("s".to_string(), "ং".to_string());
    m.insert("t".to_string(), "ঃ".to_string());
    m.insert("u".to_string(), "ঁ".to_string());

    // Kars
    m.insert("v".to_string(), "া".to_string());
    m.insert("w".to_string(), "ি".to_string());
    m.insert("x".to_string(), "ী".to_string());
    m.insert("y".to_string(), "ু".to_string());
    m.insert("~".to_string(), "ূ".to_string());
    m.insert("„".to_string(), "ৃ".to_string());
    m.insert("†".to_string(), "ে".to_string());
    m.insert("‰".to_string(), "ৈ".to_string());
    m.insert("Š".to_string(), "ৌ".to_string());

    // Digits
    m.insert("0".to_string(), "০".to_string());
    m.insert("1".to_string(), "১".to_string());
    m.insert("2".to_string(), "২".to_string());
    m.insert("3".to_string(), "৩".to_string());
    m.insert("4".to_string(), "৪".to_string());
    m.insert("5".to_string(), "৫".to_string());
    m.insert("6".to_string(), "৬".to_string());
    m.insert("7".to_string(), "৭".to_string());
    m.insert("8".to_string(), "৮".to_string());
    m.insert("9".to_string(), "৯".to_string());

    // Punctuation
    m.insert("|".to_string(), "।".to_string());

    m
}

fn get_unicode_to_bijoy_map() -> HashMap<String, String> {
    let b_to_u = get_bijoy_to_unicode_map();
    let mut u_to_b = HashMap::new();
    for (k, v) in b_to_u {
        u_to_b.insert(v, k);
    }
    u_to_b
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
