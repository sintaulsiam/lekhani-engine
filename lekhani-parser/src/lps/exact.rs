//! LPS v1.0 Section 12: Exactness Guarantee and Canonical Exact-Form Generator
//!
//! Generates the canonical exact Latin form for any arbitrary Bengali text such that
//! `exact(text)` is deterministic, unique, and round-trips without dictionaries or scoring.

/// Generates the canonical exact LPS Latin form for the given Bengali text.
pub fn exact(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    let mut i = 0;
    while i < n {
        let ch = chars[i];
        let next_ch = if i + 1 < n { Some(chars[i + 1]) } else { None };

        // 1. ZWJ / ZWNJ
        if ch == '\u{200D}' {
            out.push_str(r"\w");
            i += 1;
            continue;
        }
        if ch == '\u{200C}' {
            out.push_str(r"\z");
            i += 1;
            continue;
        }

        // 2. Marks and Modifiers
        if ch == 'ঁ' {
            out.push('^');
            i += 1;
            continue;
        }
        if ch == 'ং' {
            out.push_str("ng''");
            i += 1;
            continue;
        }
        if ch == 'ঃ' {
            out.push_str("h'");
            i += 1;
            continue;
        }
        if ch == 'ৎ' {
            out.push_str("t''");
            i += 1;
            continue;
        }

        // 3. Vowel Signs (Kars)
        let kar_latin = match ch {
            'া' => Some("a"),
            'ি' => Some("i"),
            'ী' => Some("ii"),
            'ু' => Some("u"),
            'ূ' => Some("uu"),
            'ৃ' => Some("ri'"),
            'ে' => Some("e"),
            'ৈ' => Some("oi'"),
            'ো' => Some("oo"),
            'ৌ' => Some("ou'"),
            _ => None,
        };
        if let Some(kl) = kar_latin {
            out.push_str(kl);
            i += 1;
            continue;
        }

        // 4. Independent Vowels
        let ind_vowel_latin = match ch {
            'অ' => Some("o'"),
            'আ' => Some("a"),
            'ই' => Some("i"),
            'ঈ' => Some("ii"),
            'উ' => Some("u"),
            'ঊ' => Some("uu"),
            'ঋ' => Some("ri'"),
            'এ' => Some("e"),
            'ঐ' => Some("oi'"),
            'ও' => Some("oo"),
            'ঔ' => Some("ou'"),
            _ => None,
        };
        if let Some(iv) = ind_vowel_latin {
            out.push_str(iv);
            i += 1;
            continue;
        }

        // 5. Consonants
        let consonant_latin = match ch {
            'ক' => Some("k"),
            'খ' => Some("kh"),
            'গ' => Some("g"),
            'ঘ' => Some("gh"),
            'ঙ' => Some("ng'"),
            'চ' => Some("ch"),
            'ছ' => Some("chh"),
            'জ' => Some("j"),
            'ঝ' => Some("jh"),
            'ঞ' => Some("ny"),
            'ট' => Some("t'"),
            'ঠ' => Some("th'"),
            'ড' => Some("d'"),
            'ঢ' => Some("dh'"),
            'ণ' => Some("n'"),
            'ত' => Some("t"),
            'থ' => Some("th"),
            'দ' => Some("d"),
            'ধ' => Some("dh"),
            'ন' => Some("n"),
            'প' => Some("p"),
            'ফ' => Some("ph"),
            'ব' => Some("b"),
            'ভ' => Some("bh"),
            'ম' => Some("m"),
            'য' => Some("j'"),
            'র' => Some("r"),
            'ল' => Some("l"),
            'শ' => Some("sh"),
            'ষ' => Some("s'"),
            'স' => Some("s"),
            'হ' => Some("h"),
            'ড়' => Some("r'"),
            'ঢ়' => Some("rh"),
            'য়' => Some("y'"),
            _ => None,
        };

        if let Some(cl) = consonant_latin {
            out.push_str(cl);

            // Lookahead after consonant:
            // - Hasanta: '+'
            // - Followed directly by another consonant or independent vowel without kar: '-'
            if let Some(next) = next_ch {
                if next == '্' {
                    out.push('+');
                    i += 2; // consume consonant and hasanta
                    continue;
                } else if is_consonant(next) || is_independent_vowel(next) {
                    out.push('-');
                }
            } else {
                // Word-final bare consonant without sign: needs '-' to prevent inherent 'o' or cluster
                // but by spec section 12.1: "A consonant with no sign that is followed by another consonant
                // or an independent vowel is always written with - or +."
            }
            i += 1;
            continue;
        }

        // 6. Hasanta standalone at word end
        if ch == '্' {
            out.push('+');
            i += 1;
            continue;
        }

        // 7. ASCII / Latin / Punctuation
        if ch.is_ascii() {
            out.push(ch);
            i += 1;
            continue;
        }

        // 8. Anything else: \u{hex}
        out.push_str(&format!(r"\u{{{:04X}}}", ch as u32));
        i += 1;
    }

    out
}

fn is_consonant(ch: char) -> bool {
    matches!(
        ch,
        'ক'..='হ' | 'ড়' | 'ঢ়' | 'য়' | 'ৎ'
    )
}

fn is_independent_vowel(ch: char) -> bool {
    matches!(
        ch,
        'অ'..='ঔ'
    )
}
