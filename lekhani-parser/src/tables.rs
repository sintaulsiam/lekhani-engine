//! ASCII Character Classification and Normalization Tables
//!
//! Provides branchless 1-cycle byte classification for vowel, consonant,
//! punctuation, number, and Avro-specific case sensitivity flags.

pub const FLAG_VOWEL: u8 = 1 << 0;
pub const FLAG_CONSONANT: u8 = 1 << 1;
pub const FLAG_PUNCTUATION: u8 = 1 << 2;
pub const FLAG_NUMBER: u8 = 1 << 3;
pub const FLAG_CASE_SENSITIVE: u8 = 1 << 4;

/// Static 256-byte classification lookup table for ASCII bytes.
/// Precomputed at compile time.
pub static ASCII_CLASS_TABLE: [u8; 256] = {
    let mut table = [FLAG_PUNCTUATION; 256];

    // Numbers: 0-9
    let mut b = b'0';
    while b <= b'9' {
        table[b as usize] = FLAG_NUMBER | FLAG_PUNCTUATION;
        b += 1;
    }

    // Lowercase consonants
    let consonants = b"bcdfghjklmnpqrstvwxyz";
    let mut i = 0;
    while i < consonants.len() {
        table[consonants[i] as usize] = FLAG_CONSONANT;
        table[consonants[i].to_ascii_uppercase() as usize] = FLAG_CONSONANT;
        i += 1;
    }

    // Lowercase vowels
    let vowels = b"aeiou";
    let mut i = 0;
    while i < vowels.len() {
        table[vowels[i] as usize] = FLAG_VOWEL;
        table[vowels[i].to_ascii_uppercase() as usize] = FLAG_VOWEL;
        i += 1;
    }

    // Avro case-sensitive characters: "oiudgjnrstyz"
    let case_sens = b"oiudgjnrstyz";
    let mut i = 0;
    while i < case_sens.len() {
        table[case_sens[i] as usize] |= FLAG_CASE_SENSITIVE;
        table[case_sens[i].to_ascii_uppercase() as usize] |= FLAG_CASE_SENSITIVE;
        i += 1;
    }

    table
};

/// 256-byte lookup table mapping ASCII bytes to their Avro-normalized case.
/// Characters in `casesensitive` ("oiudgjnrstyz") keep their case.
/// All other characters are mapped to ASCII lowercase.
pub static AVRO_CASE_NORM_TABLE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut b = 0u16;
    while b < 256 {
        let byte = b as u8;
        if (ASCII_CLASS_TABLE[byte as usize] & FLAG_CASE_SENSITIVE) != 0 {
            table[byte as usize] = byte;
        } else {
            table[byte as usize] = byte.to_ascii_lowercase();
        }
        b += 1;
    }
    table
};

#[inline(always)]
pub fn classify_byte(b: u8) -> u8 {
    ASCII_CLASS_TABLE[b as usize]
}

#[inline(always)]
pub fn normalize_avro_byte(b: u8) -> u8 {
    AVRO_CASE_NORM_TABLE[b as usize]
}

#[inline(always)]
pub fn needs_avro_norm(b: u8) -> bool {
    b.is_ascii_uppercase() && (ASCII_CLASS_TABLE[b as usize] & FLAG_CASE_SENSITIVE) == 0
}

#[inline(always)]
pub fn is_vowel_byte(b: u8) -> bool {
    (ASCII_CLASS_TABLE[b as usize] & FLAG_VOWEL) != 0
}

#[inline(always)]
pub fn is_consonant_byte(b: u8) -> bool {
    (ASCII_CLASS_TABLE[b as usize] & FLAG_CONSONANT) != 0
}

#[inline(always)]
pub fn is_number_byte(b: u8) -> bool {
    (ASCII_CLASS_TABLE[b as usize] & FLAG_NUMBER) != 0
}

#[inline(always)]
pub fn is_punctuation_byte(b: u8) -> bool {
    // In Avro semantics: punctuation is anything that is neither vowel nor consonant
    (ASCII_CLASS_TABLE[b as usize] & (FLAG_VOWEL | FLAG_CONSONANT)) == 0
}
