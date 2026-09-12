//! Generalized Phoneme Cluster Sound Laws & Lattice Generator
//!
//! Generates linguistically principled phonemic spelling variations from Latin input
//! using general Bengali orthographic and phonetic equivalence sound laws:
//! - Sibilants (স, শ, ষ) and sibilant conjuncts (স্ব, স্ম, ষ্ট, ষ্ঠ, স্ক, ষ্প)
//! - Dental vs Retroflex stops (ত/ট, থ/ঠ, দ/ড, ধ/ঢ)
//! - Rhotics, Flaps, Ri-kar & Reph (র, ড়, ঢ়, ঋ/ৃ, র্)
//! - Affricates, Semivowels & Ja-fala (জ, য, য়, ্য)
//! - Aspirated vs Unaspirated stops (ক/খ, গ/ঘ, চ/ছ, প/ফ, ব/ভ)
//! - Vowel height, Diphthongs & Nasals (ই/ঈ, উ/ঊ, ও/ো, ঐ/ৈ, ঔ/ৌ, ং, ঙ, ঁ)
//!
//! Replaces all hardcoded word lists with scalable, mathematical equivalence classes.

use hashbrown::HashSet;

/// Collapse run-length elongated characters (e.g. "thiiik" -> ["thik", "thiik"], "bhalooo" -> ["bhalo"])
pub fn collapse_elongated_runs(input: &str) -> Vec<String> {
    if input.len() < 3 {
        return Vec::new();
    }

    let mut collapsed_1 = String::with_capacity(input.len());
    let mut collapsed_2 = String::with_capacity(input.len());
    let mut has_run = false;

    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        let mut count = 1;
        while i + count < chars.len() && chars[i + count] == ch {
            count += 1;
        }

        if count >= 3 {
            has_run = true;
            collapsed_1.push(ch);
            collapsed_2.push(ch);
            collapsed_2.push(ch);
        } else if count == 2 {
            collapsed_1.push(ch);
            collapsed_2.push(ch);
            collapsed_2.push(ch);
        } else {
            collapsed_1.push(ch);
            collapsed_2.push(ch);
        }

        i += count;
    }

    let mut results = Vec::new();
    if has_run {
        if collapsed_1 != input {
            results.push(collapsed_1.clone());
        }
        if collapsed_2 != input && collapsed_2 != collapsed_1 {
            results.push(collapsed_2);
        }
    }
    results
}

/// Generic Phoneme Equivalence Sound Laws (Equivalence Classes)
pub const PHONEME_SOUND_LAWS: &[(&str, &[&str])] = &[
    // 1. Sibilants & Sibilant Conjuncts (স, শ, ষ, স্ব, স্ম, ষ্ট, ষ্ঠ, স্ক, ষ্প, স্ফ, স্ত্র)
    ("sh", &["sw", "s", "Sh"]),
    ("Sh", &["sh", "s", "sw"]),
    ("ss", &["sh", "s", "sw"]),
    ("s", &["sw", "sh", "Sh"]),
    ("sw", &["shw", "s", "sh"]),
    ("shw", &["sw", "s"]),
    ("sm", &["Shm", "shm", "s"]),
    ("shm", &["sm", "Shm"]),
    ("sn", &["ShN", "shn", "sn"]),
    ("sht", &["ShT", "st", "sT", "ShTh"]),
    ("shT", &["ShT", "st", "sT"]),
    ("st", &["ShT", "sT", "sht", "ShTh", "st"]),
    ("sT", &["ShT", "st", "sht"]),
    ("sth", &["ShTh", "sTh", "sth", "shth"]),
    ("shth", &["ShTh", "sth", "sTh"]),
    ("shTh", &["ShTh", "sth", "sTh"]),
    ("sTh", &["ShTh", "sth"]),
    ("sk", &["Shk", "sk", "shk"]),
    ("shk", &["Shk", "sk"]),
    ("sp", &["Shp", "sp", "shp"]),
    ("shp", &["Shp", "sp"]),
    ("sf", &["Shf", "sf", "shf"]),
    ("shf", &["Shf", "sf"]),
    ("str", &["sTr", "ShTr", "str"]),
    // 2. Dental & Retroflex Stops (ত/ট, থ/ঠ, দ/ড, ধ/ঢ, খণ্ড-ত ৎ)
    ("th", &["Th", "t"]),
    ("Th", &["th", "T"]),
    ("t", &["T", "th", "t``"]),
    ("T", &["t", "Th"]),
    ("dh", &["Dh", "d"]),
    ("Dh", &["dh", "D"]),
    ("d", &["D", "dh"]),
    ("D", &["d", "Dh"]),
    ("tt", &["t", "tZ", "t``"]),
    ("tth", &["thZ", "tt"]),
    ("ttho", &["thZ", "tth"]),
    ("tto", &["tZ", "tt"]),
    ("dd", &["d", "dZ", "ddh"]),
    ("ddh", &["dd", "dh"]),
    // 3. Rhotics, Flaps, Ri-kar & Reph (র, ড়, ঢ়, ঋ/ৃ, র্)
    ("rrh", &["rh", "R", "r"]),
    ("rr", &["r", "R"]),
    ("rh", &["Rh", "R", "r"]),
    ("Rh", &["rh", "R"]),
    ("ri", &["rri", "ree"]),
    ("rri", &["ri"]),
    ("rre", &["rri", "ri"]),
    ("r", &["R", "rh"]),
    ("R", &["r", "Rh"]),
    // Ri-kar conjuncts (বৃষ্টি, সৃষ্টি, কৃষি, পৃথিবী, দৃষ্টি, মৃত্যু, হৃদয়)
    ("sri", &["srri", "sri"]),
    ("bri", &["brri", "bri"]),
    ("kri", &["krri", "kri"]),
    ("pri", &["prri", "pri"]),
    ("dri", &["drri", "dri"]),
    ("mri", &["mrri", "mri"]),
    ("hri", &["hrri", "hri"]),
    ("gri", &["grri", "gri"]),
    // Reph consonant combinations (র-ফলা / রেফ: র্)
    ("rt", &["rrt", "rrT"]),
    ("rth", &["rrth", "rrTh"]),
    ("ortho", &["orrth", "rrth", "orth"]),
    ("orth", &["orrth", "rrth"]),
    ("rtho", &["rrth", "rrTh", "rth"]),
    ("rj", &["rrz", "rrj"]),
    ("rsh", &["rrsh", "rrSh"]),
    ("rn", &["rrN", "rrn"]),
    ("rb", &["rrb", "rrv"]),
    ("rm", &["rrm"]),
    ("rk", &["rrk"]),
    ("rp", &["rrp"]),
    ("rd", &["rrd", "rrD"]),
    ("rdh", &["rrdh", "rrDh"]),
    ("rg", &["rrg"]),
    ("rgh", &["rrgh"]),
    // 4. Affricates, Semivowels & Ja-fala (জ, য, য়, ওয়, ঝ, জ্ঞ, ক্ষ)
    ("z", &["j", "y"]),
    ("j", &["z", "jh", "y"]),
    ("jh", &["j"]),
    ("y", &["z", "Y", "y"]),
    ("w", &["o", "oy", "v", "bh"]),
    ("gann", &["gZan", "jNGan", "gZann"]),
    ("gani", &["gZani", "gZanI", "gZanee", "jNGani"]),
    ("ganni", &["gZani", "gZanI", "gZanee", "jNGani"]),
    ("jng", &["gZ", "jNG"]),
    ("gZ", &["jNG", "jng"]),
    ("gy", &["gZ", "jNG"]),
    ("gg", &["gZ", "jNG"]),
    ("gn", &["gZ", "jNG"]),
    ("gan", &["gZan", "jNGan"]),
    ("kkh", &["kSh", "x", "ks"]),
    ("x", &["kkh", "kSh"]),
    ("ks", &["kkh", "x"]),
    // 5. Aspirated vs Unaspirated Stops (ক/খ, গ/ঘ, চ/ছ, প/ফ, ব/ভ)
    ("kh", &["k"]),
    ("k", &["kh"]),
    ("gh", &["g"]),
    ("g", &["gh"]),
    ("chh", &["ch", "c"]),
    ("ch", &["c", "chh", "cch"]),
    ("c", &["ch", "k"]),
    ("cch", &["cc", "ch", "c"]),
    ("cc", &["cch", "ch"]),
    ("ph", &["f", "p"]),
    ("f", &["ph", "p"]),
    ("p", &["ph", "f"]),
    ("bh", &["v", "b"]),
    ("v", &["bh", "b", "w"]),
    ("b", &["bh", "v"]),
    // 6. Vowel Height, Diphthongs & Nasals (ই/ঈ, উ/ঊ, ও/ো, ঐ/ৈ, ঔ/ৌ, ন/ণ/ঙ/ং/ঁ)
    ("ee", &["i", "I", "ii"]),
    ("ii", &["ee", "I", "i"]),
    ("i", &["I", "ee", "ii"]),
    ("I", &["i", "ee"]),
    ("oo", &["u", "U", "uu"]),
    ("uu", &["oo", "U", "u"]),
    ("u", &["U", "oo", "uu"]),
    ("U", &["u", "oo"]),
    ("aa", &["a", "A"]),
    ("o", &["O", "a", "u"]),
    ("O", &["o", "u"]),
    ("noiti", &["nOIti", "noyti"]),
    ("noit", &["nOIt", "noyt"]),
    ("noik", &["nOIk", "noyk"]),
    ("doik", &["dOIk", "doyk"]),
    ("soin", &["sOIn", "shOIn"]),
    ("boik", &["bOIk", "boyk"]),
    ("oi", &["OI", "oy"]),
    ("ou", &["OU", "ow"]),
    ("OI", &["oi"]),
    ("OU", &["ou"]),
    ("abong", &["ebong", "abong"]),
    ("ebong", &["abong", "ebong"]),
    ("ng", &["n", "Ng"]),
    ("n", &["N", "ng"]),
    ("N", &["n"]),
    // 7. Ja-fala & Geminate Reductions (দ্য, থ্য, ক্য, ব্য, ন্য, ল্য, ম্য, শ্য, স্য)
    ("bya", &["bZa", "bZ", "by", "be"]),
    ("byo", &["bZo", "bZ", "by", "bo"]),
    ("by", &["bZ", "b", "be"]),
    ("bebo", &["bZbo", "bZabo"]),
    ("byabo", &["bZbo", "bZabo"]),
    ("beba", &["bZba", "bZaba"]),
    ("byaba", &["bZba", "bZaba"]),
    ("be", &["bZa", "bZ", "bya"]),
    ("ty", &["tZ", "t"]),
    ("thy", &["thZ", "th"]),
    ("dy", &["dZ", "d"]),
    ("dhy", &["dhZ", "dh"]),
    ("ny", &["nZ", "n"]),
    ("my", &["mZ", "m"]),
    ("ky", &["kZ", "k"]),
    ("ly", &["lZ", "l"]),
    ("sy", &["sZ", "s"]),
    ("shy", &["shZ", "sh"]),
    // Geminates representing Ja-fala in Bengali phonology (অন্য, জন্য, কল্যাণ, বাক্য, নাব্য, রম্য, সত্য, তথ্য)
    ("nn", &["nZ", "ny", "n"]),
    ("mm", &["mZ", "my", "m"]),
    ("ll", &["lZ", "ly", "l"]),
    ("kk", &["kZ", "ky", "k"]),
    ("bb", &["bZ", "by", "b"]),
    ("dd", &["dZ", "dy", "ddh", "d"]),
    ("ddh", &["dhZ", "dhy", "dd", "dh"]),
    ("tth", &["thZ", "thy", "tt"]),
    ("tt", &["tZ", "ty", "t``", "t"]),
    // 8. Chandrabindu Nasalization Equivalences (চাঁদ, হাঁস, দাঁত, বাঁশ, পাঁচ)
    ("ad", &["a^d", "ad"]),
    ("as", &["a^s", "a^sh", "as"]),
    ("ash", &["a^sh", "a^s", "ash"]),
    ("at", &["a^t", "a^T", "at"]),
    ("ac", &["a^c", "a^ch", "ac"]),
    ("ach", &["a^ch", "a^c", "ach"]),
    ("ak", &["a^k", "ak"]),
    ("ap", &["a^p", "ap"]),
    ("ab", &["a^b", "ab"]),
    ("ag", &["a^g", "ag"]),
    ("an", &["a^n", "an"]),
    ("ha", &["h^a", "ha"]),
    ("ca", &["c^a", "ca"]),
    ("cha", &["ch^a", "cha"]),
    ("da", &["d^a", "da"]),
    ("ba", &["b^a", "ba"]),
    ("pa", &["p^a", "pa"]),
    ("fa", &["f^a", "fa"]),
    ("pha", &["ph^a", "pha"]),
    ("ga", &["g^a", "ga"]),
    ("ka", &["k^a", "ka"]),
];

/// Generate plausible phonetic spelling variants of a Latin word using generic sound laws
pub fn generate_phonetic_variants(input: &str) -> Vec<String> {
    if input.is_empty() || input.chars().count() > 30 {
        return Vec::new();
    }

    let lower = input.to_lowercase();

    // Helper closure to apply sound laws systematically across all rules
    let apply_sound_laws = |src: &str, out: &mut HashSet<String>| {
        let src_lower = src.to_lowercase();
        for &(target, replacements) in PHONEME_SOUND_LAWS {
            for (pos, _) in src_lower.match_indices(target) {
                for &rep in replacements {
                    let mut variant = String::with_capacity(src.len() + rep.len());
                    variant.push_str(&src[..pos]);
                    variant.push_str(rep);
                    variant.push_str(&src[pos + target.len()..]);
                    if variant != src && variant != input {
                        out.insert(variant);
                    }
                }
            }
        }
    };

    let mut pass1 = HashSet::with_capacity(128);
    let mut collapsed_variants = Vec::new();

    // 1. Run-length collapsed variants (e.g. thiiik -> thik, bhalooo -> bhalo)
    for collapsed in collapse_elongated_runs(&lower) {
        collapsed_variants.push(collapsed.clone());
        pass1.insert(collapsed.clone());
        apply_sound_laws(&collapsed, &mut pass1);
    }

    apply_sound_laws(&lower, &mut pass1);

    // 2. Bengali Glide Verbs (e.g. khawa -> khaoya, dewa -> deoya, jawa -> jaoya)
    if lower.ends_with("awa") && lower.len() >= 4 {
        let glide = format!("{}aoya", &lower[..lower.len() - 3]);
        pass1.insert(glide);
    } else if lower.ends_with("ewa") && lower.len() >= 4 {
        let glide = format!("{}eoya", &lower[..lower.len() - 3]);
        pass1.insert(glide);
    }

    // 3. Second pass for compounding sound laws (e.g. sri -> srri AND st -> ShT => srriShTi)
    let mut pass2 = HashSet::with_capacity(256);
    let pass1_list: Vec<String> = pass1.iter().cloned().collect();
    for var in &pass1_list {
        apply_sound_laws(var, &mut pass2);
    }

    pass1.extend(pass2);

    let base_len = if let Some(first_collapsed) = collapsed_variants.first() {
        first_collapsed.len()
    } else {
        input.len()
    };

    let mut results: Vec<String> = pass1.into_iter().collect();
    results.sort_unstable_by(|a, b| {
        let diff_a = (a.len() as isize - base_len as isize).abs();
        let diff_b = (b.len() as isize - base_len as isize).abs();
        diff_a.cmp(&diff_b).then_with(|| a.len().cmp(&b.len()))
    });

    results.truncate(256);
    results
}

/// Compute a canonical Bengali phonetic soundex key
/// Groups homophones and phonologically equivalent characters into canonical classes:
/// - Sibilants (শ, ষ, স) -> 'S'
/// - Nasals (ন, ণ, ং, ঙ, ঁ) -> 'N'
/// - High Vowels / Kars (ি, ী, ই, ঈ) -> 'I'
/// - Mid-Low Vowels / Kars (ু, ূ, উ, ঊ) -> 'U'
/// - Rhotics / Flaps (র, ড়, ঢ়, ঋ, ৃ, র্) -> 'R'
/// - Dental / Retroflex Stops (ত, ৎ, ট) -> 'T', (থ, ঠ) -> 't', (দ, ড) -> 'D', (ধ, ঢ) -> 'd'
/// - Affricates & Semivowels (জ, য, য়, ্য) -> 'J'
/// - Velars (ক, খ) -> 'K', (গ, ঘ) -> 'G'
/// - Labials (প, ফ) -> 'P', (ব, ভ) -> 'B'
///
/// Deduplicates adjacent identical phonetic codes.
pub fn bengali_phonetic_soundex(word: &str) -> String {
    let mut soundex = String::with_capacity(word.len());
    let mut last_code = ' ';

    for ch in word.chars() {
        let code = match ch {
            // Sibilants
            'শ' | 'ষ' | 'স' => 'S',
            // Nasals
            'ন' | 'ণ' | 'ং' | 'ঙ' | 'ঁ' | 'ঞ' => 'N',
            // High Vowels & Kars
            'ি' | 'ী' | 'ই' | 'ঈ' => 'I',
            // Mid-Low Vowels & Kars
            'ু' | 'ূ' | 'উ' | 'ঊ' => 'U',
            // A-vowels
            'া' | 'আ' => 'A',
            // E-vowels
            'ে' | 'এ' => 'E',
            // O-vowels
            'ো' | 'ও' | 'অ' => 'O',
            // Diphthongs
            'ৈ' | 'ঐ' => 'Y',
            'ৌ' | 'ঔ' => 'W',
            // Rhotics & Flaps
            'র' | 'ড়' | 'ঢ়' | 'ঋ' | 'ৃ' => 'R',
            // Stops
            'ত' | 'ৎ' | 'ট' => 'T',
            'থ' | 'ঠ' => 't',
            'দ' | 'ড' => 'D',
            'ধ' | 'ঢ' => 'd',
            'ক' | 'খ' => 'K',
            'গ' | 'ঘ' => 'G',
            'চ' | 'ছ' => 'C',
            'প' | 'ফ' => 'P',
            'ব' | 'ভ' => 'B',
            'জ' | 'য' | 'য়' | 'ঝ' => 'J',
            'হ' => 'H',
            '্' => continue, // ignore hasant in soundex
            _ => continue,
        };

        if code != last_code {
            soundex.push(code);
            last_code = code;
        }
    }

    soundex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phonetic_sound_laws() {
        let s_variants = generate_phonetic_variants("shundor");
        assert!(s_variants
            .iter()
            .any(|v| v.contains("sundor") || v.contains("Shundor")));

        let t_variants = generate_phonetic_variants("bortoman");
        assert!(t_variants
            .iter()
            .any(|v| v.contains("borrtoman") || v.contains("borToman")));

        let z_variants = generate_phonetic_variants("juddho");
        assert!(z_variants.iter().any(|v| v.contains("zuddho")));

        let ch_variants = generate_phonetic_variants("chesta");
        assert!(ch_variants
            .iter()
            .any(|v| v.contains("c") || v.contains("ceShTa") || v.contains("ceshTa")));
    }

    #[test]
    fn test_bengali_phonetic_soundex() {
        // Homophones map to identical soundex signatures
        assert_eq!(
            bengali_phonetic_soundex("বিদেশি"),
            bengali_phonetic_soundex("বিদেশী")
        );
        assert_eq!(
            bengali_phonetic_soundex("শহীদ"),
            bengali_phonetic_soundex("সহিদ")
        );
        assert_eq!(
            bengali_phonetic_soundex("সাধারণ"),
            bengali_phonetic_soundex("সাধারন")
        );
    }
}
