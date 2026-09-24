//! Bijoy (ANSI / SutonnyMJ) ⇄ Unicode Bengali Conversion Engine

use hashbrown::HashMap;
use std::sync::LazyLock;

static BIJOY_TO_UNICODE: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::with_capacity(200);

    // Compound Conjuncts & Special Ligatures
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

    // Standard Consonants
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

    // Kars & Diacritics
    m.insert("v", "া");
    m.insert("w", "ি");
    m.insert("x", "ী");
    m.insert("y", "ু");
    m.insert("~", "ূ");
    m.insert("„", "ৃ");
    m.insert("†", "ে");
    m.insert("‡", "ে");
    m.insert("ˆ", "ৈ");
    m.insert("‰", "ৈ");
    m.insert("Š", "ৌ");
    m.insert("¨", "্");
    m.insert("&", "্");
    m.insert("«", "্র");
    m.insert("ª", "্র");
    m.insert("º", "্র");

    // Common and Rare Sanskrit Conjuncts
    m.insert("¶¥", "ক্ষ্ম");
    m.insert("¶", "ক্ষ");
    m.insert("³", "ক্ত");
    m.insert("K¬", "ক্ল");
    m.insert("²", "ক্স");
    m.insert("µ", "ক্র");
    m.insert("Á", "জ্ঞ");
    m.insert("Â", "ঞ্চ");
    m.insert("Ã", "ঞ্ছ");
    m.insert("Ä", "ঞ্জ");
    m.insert("Æ", "ট্ট");
    m.insert("Ç", "ট্ঠ");
    m.insert("È", "ড্ড");
    m.insert("É", "ণ্ট");
    m.insert("Ê", "ণ্ঠ");
    m.insert("Ë", "ণ্ড");
    m.insert("Ì", "ত্ব");
    m.insert("Í", "ত্ম");
    m.insert("Î", "ত্র");
    m.insert("Ï", "থ্ব");
    m.insert("Ð", "দ্দ");
    m.insert("Ñ", "দ্ধ");
    m.insert("Ò", "দ্ব");
    m.insert("Ó", "দ্ম");
    m.insert("Ô", "ধ্ন");
    m.insert("Õ", "ধ্ব");
    m.insert("×", "ন্ট");
    m.insert("Ø", "ন্ঠ");
    m.insert("Ù", "ন্ড");
    m.insert("Ú", "ন্ত");
    m.insert("Û", "ন্থ");
    m.insert("Ü", "ন্দ");
    m.insert("Ý", "ন্ধ");
    m.insert("Þ", "ন্ন");
    m.insert("ß", "ন্ম");
    m.insert("à", "ন্ব");
    m.insert("á", "প্ট");
    m.insert("â", "প্ত");
    m.insert("ã", "প্ন");
    m.insert("ä", "প্প");
    m.insert("å", "প্ল");
    m.insert("æ", "প্স");
    m.insert("ç", "ফ্ট");
    m.insert("è", "ফ্ফ");
    m.insert("é", "ফ্ল");
    m.insert("ê", "ব্জ");
    m.insert("ë", "ব্দ");
    m.insert("ì", "ব্ধ");
    m.insert("í", "ব্ব");
    m.insert("î", "ব্ল");
    m.insert("ï", "ভ্ল");
    m.insert("ð", "ম্ন");
    m.insert("ñ", "ম্প");
    m.insert("ò", "ম্ফ");
    m.insert("ó", "ম্ব");
    m.insert("ô", "ম্ভ");
    m.insert("õ", "ম্ম");
    m.insert("ö", "ম্ল");
    m.insert("÷", "স্ট");
    m.insert("ø", "স্ন");
    m.insert("ù", "স্ফ");
    m.insert("ú", "্প");
    m.insert("û", "হু");
    m.insert("ü", "হৃ");
    m.insert("ý", "হ্ন");
    m.insert("þ", "হ্ম");
    m.insert("ÿ", "হ্ল");
    m.insert("nè", "হ্ণ");
    m.insert("j^", "ল্ব");
    m.insert("j§", "ল্ম");
    m.insert("j¬", "ল্ল");
    m.insert("¯‹", "স্ক");
    m.insert("¯Í", "স্ত");
    m.insert("¯’", "স্থ");
    m.insert("¯œ", "স্ন");
    m.insert("¯ú", "স্প");
    m.insert("¯¢", "স্ফ");
    m.insert("¯^", "স্ব");
    m.insert("¯§", "স্ম");
    m.insert("¯ø", "স্ল");

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
    let mut u_to_b = HashMap::with_capacity(200);

    // Primary deterministic mappings
    u_to_b.insert("ে", "†");
    u_to_b.insert("ৈ", "ˆ");
    u_to_b.insert("্", "&");
    u_to_b.insert("্র", "«");

    for (&k, &v) in BIJOY_TO_UNICODE.iter() {
        u_to_b.entry(v).or_insert(k);
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

fn pre_process_unicode_to_bijoy(input: &str) -> String {
    let mut chars: Vec<char> = Vec::with_capacity(input.len());

    // First pass: decompose ো -> ে + া, and ৌ -> ে + ৗ
    for c in input.chars() {
        if c == 'ো' {
            chars.push('ে');
            chars.push('া');
        } else if c == 'ৌ' {
            chars.push('ে');
            chars.push('ৗ');
        } else {
            chars.push(c);
        }
    }

    // Second pass: move pre-kars (ি, ে, ৈ) before their preceding consonant cluster
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == 'ি' || chars[i] == 'ে' || chars[i] == 'ৈ' {
            let kar = chars[i];
            if i > 0 {
                let mut start_cons = i - 1;
                while start_cons >= 2
                    && chars[start_cons - 1] == '্'
                    && is_consonant(chars[start_cons - 2])
                {
                    start_cons -= 2;
                }
                if is_consonant(chars[start_cons]) {
                    chars.remove(i);
                    chars.insert(start_cons, kar);
                    i += 1;
                    continue;
                }
            }
        }
        i += 1;
    }

    // Third pass: convert Reph (র + ্ + Consonant) to Consonant + ©
    let mut i = 0;
    while i + 2 < chars.len() {
        if chars[i] == 'র' && chars[i + 1] == '্' && is_consonant(chars[i + 2]) {
            chars.remove(i);
            chars.remove(i);
            let mut end_cons = i;
            while end_cons + 2 < chars.len()
                && chars[end_cons + 1] == '্'
                && is_consonant(chars[end_cons + 2])
            {
                end_cons += 2;
            }
            chars.insert(end_cons + 1, '©');
            i = end_cons + 2;
            continue;
        }
        i += 1;
    }

    chars.into_iter().collect()
}

/// Convert Unicode Bengali text to Bijoy (ANSI / SutonnyMJ)
pub fn unicode_to_bijoy(input: &str) -> String {
    let preprocessed = pre_process_unicode_to_bijoy(input);
    let mut result = String::with_capacity(preprocessed.len() * 2);
    let chars: Vec<char> = preprocessed.chars().collect();
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

    while i < chars.len() {
        // Fix misplaced pre-kars (ি, ে, ৈ) placed before consonant
        if (chars[i] == 'ি' || chars[i] == 'ে' || chars[i] == 'ৈ') && i + 1 < chars.len() {
            let mut end_cons = i + 1;
            while end_cons + 2 < chars.len()
                && chars[end_cons + 1] == '্'
                && is_consonant(chars[end_cons + 2])
            {
                end_cons += 2;
            }
            if is_consonant(chars[end_cons]) {
                let kar = chars.remove(i);
                chars.insert(end_cons, kar);
                i = end_cons + 1;
                continue;
            }
        }

        // Fix Reph '©' at end of consonant (transforms to র্ before consonant cluster)
        if chars[i] == '©' {
            chars.remove(i);
            let mut start_cons = if i > 0 { i - 1 } else { 0 };
            while start_cons >= 2
                && chars[start_cons - 1] == '্'
                && is_consonant(chars[start_cons - 2])
            {
                start_cons -= 2;
            }
            chars.insert(start_cons, 'র');
            chars.insert(start_cons + 1, '্');
            i += 2;
            continue;
        }

        i += 1;
    }

    // Synthesize O-kar (ে + া -> ো) and OU-kar (ে + ৗ -> ৌ)
    let mut synthesized = String::with_capacity(chars.len());
    let mut iter = chars.into_iter().peekable();

    while let Some(c) = iter.next() {
        if c == 'ে' {
            if let Some(&next) = iter.peek() {
                if next == 'া' {
                    synthesized.push('ো');
                    iter.next();
                    continue;
                } else if next == 'ৗ' || next == 'Š' {
                    synthesized.push('ৌ');
                    iter.next();
                    continue;
                }
            }
        }
        synthesized.push(c);
    }

    synthesized
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
        assert_eq!(bijoy_to_unicode("†mvbvi"), "সোনার");
        assert_eq!(bijoy_to_unicode("¯^vaxb"), "স্বাধীন");
        assert_eq!(bijoy_to_unicode("¶wZ"), "ক্ষতি");
        assert_eq!(bijoy_to_unicode("ü`q"), "হৃদয়");
        assert_eq!(bijoy_to_unicode("wPý"), "চিহ্ন");
        assert_eq!(bijoy_to_unicode("eªvþY"), "ব্রাহ্মণ");
        assert_eq!(bijoy_to_unicode("÷vd"), "স্টাফ");
        assert_eq!(bijoy_to_unicode("wK‡kvi"), "কিশোর");
        assert_eq!(unicode_to_bijoy("হৃদয়"), "ü`q");
        assert_eq!(unicode_to_bijoy("সোনার"), "†mvbvi");
        assert_eq!(unicode_to_bijoy("কিশোর"), "wK†kvi");
        assert_eq!(unicode_to_bijoy("বাংলা"), "evsjv");
        assert_eq!(unicode_to_bijoy("আমি"), "Avwg");
        assert_eq!(bijoy_to_unicode(&unicode_to_bijoy("সোনার")), "সোনার");
        assert_eq!(bijoy_to_unicode(&unicode_to_bijoy("কিশোর")), "কিশোর");
    }
}
