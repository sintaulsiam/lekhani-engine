//! Phonetic text normalization, Chandra Bindu ordering, and punctuation splitting utilities.

use crate::chars::BengaliCharExt;

/// Normalize caret position in Latin phonetic input (e.g. "ch^ad" -> "cha^d", "k^ada" -> "ka^da")
pub fn normalize_chandra_latin(input: &str) -> String {
    if !input.contains('^') {
        return input.to_string();
    }
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '^' && i + 1 < chars.len() {
            let next = chars[i + 1];
            if is_latin_vowel(next) {
                let mut v_end = i + 1;
                while v_end < chars.len() && is_latin_vowel(chars[v_end]) {
                    v_end += 1;
                }
                for v in &chars[i + 1..v_end] {
                    out.push(*v);
                }
                out.push('^');
                i = v_end;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

#[inline]
fn is_latin_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U' | 'y' | 'Y' | '`'
    )
}

#[inline(always)]
pub fn fix_chandra_bindu_position(text: &str) -> String {
    let mut res = text.to_string();
    if res.contains('\u{09BC}') {
        res = res.replace("\u{09AF}\u{09BC}", "য়");
        res = res.replace("\u{09A1}\u{09BC}", "ড়");
        res = res.replace("\u{09A2}\u{09BC}", "ঢ়");
    }
    if !res.contains('ঁ') {
        return res;
    }
    res = res.replace("ঁআ", "াঁ");
    res = res.replace("ঁা", "াঁ");
    res = res.replace("ঁএ", "েঁ");
    res = res.replace("ঁে", "েঁ");
    res = res.replace("ঁই", "িঁ");
    res = res.replace("ঁি", "িঁ");
    res = res.replace("ঁউ", "ুঁ");
    res = res.replace("ঁু", "ুঁ");
    res = res.replace("ঁও", "োঁ");
    res = res.replace("ঁো", "োঁ");
    res = res.replace("ঁঈ", "ীঁ");
    res = res.replace("ঁী", "ীঁ");
    res = res.replace("ঁঊ", "ূঁ");
    res = res.replace("ঁূ", "ূঁ");
    res = res.replace("ঁঋ", "ৃঁ");
    res = res.replace("ঁৃ", "ৃঁ");
    res = res.replace("ঁঐ", "ৈঁ");
    res = res.replace("ঁৈ", "ৈঁ");
    res = res.replace("ঁঔ", "ৌঁ");
    res = res.replace("ঁৌ", "ৌঁ");
    res
}

/// Returns true if `s` is a single Bengali vowel, consonant, or consonant+kar CV syllable.
pub fn is_atomic_cv_syllable(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return false;
    }
    if chars.len() == 1 {
        return chars[0].is_vowel() || chars[0].is_consonant();
    }
    if chars.len() == 2 {
        return chars[0].is_consonant() && chars[1].is_kar();
    }
    false
}

#[inline(always)]
pub fn split_word_punct(input: &str) -> (&str, &str, &str) {
    if input.is_empty() {
        return ("", "", "");
    }

    // Preserve Avro escape sequences like ".`", ":`", ",`", etc.
    if input.ends_with('`') && input.len() == 2 && is_punct_byte(input.as_bytes()[0]) {
        return (input, "", "");
    }

    let bytes = input.as_bytes();
    let mut start = 0;
    let mut end = input.len();

    while start < bytes.len() && is_punct_byte(bytes[start]) {
        start += 1;
    }
    while end > start && is_punct_byte(bytes[end - 1]) {
        end -= 1;
    }

    (&input[..start], &input[start..end], &input[end..])
}

#[inline]
fn is_punct_byte(b: u8) -> bool {
    b.is_ascii_punctuation() && b != b'`'
}
