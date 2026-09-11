//! Fuzzy Phonetic Variation Generator
//!
//! Generates common phoneme spelling variations for Latin input to handle
//! ambiguous Bengali phonemes (e.g. s/sh/ss, z/j, i/ee/ii, u/oo/uu, r/rr/rrh, n/N).

use hashbrown::HashSet;

/// Generate plausible phonetic spelling variants of a Latin word
pub fn generate_phonetic_variants(input: &str) -> Vec<String> {
    if input.is_empty() || input.len() > 20 {
        return Vec::new();
    }

    let mut variants = HashSet::new();
    let lower = input.to_lowercase();

    // 1. Single and multi-character replacements
    let rules: &[(&str, &[&str])] = &[
        // Sibilants: s <-> sh <-> ss
        ("sh", &["s", "ss"]),
        ("ss", &["sh", "s"]),
        ("s", &["sh", "ss"]),
        // Affricates: z <-> j <-> jh
        ("z", &["j", "jh"]),
        ("j", &["z", "jh"]),
        ("jh", &["j", "z"]),
        // Labials: v <-> bh <-> b <-> w
        ("v", &["bh", "b", "w"]),
        ("bh", &["v", "b"]),
        ("b", &["bh", "v"]),
        ("w", &["o", "v", "b"]),
        // Aspirated stops: f <-> ph <-> p, kh <-> k, gh <-> g, ch <-> chh <-> c
        ("ph", &["f", "p"]),
        ("f", &["ph", "p"]),
        ("kh", &["k"]),
        ("k", &["kh"]),
        ("gh", &["g"]),
        ("g", &["gh"]),
        ("chh", &["ch", "c"]),
        ("ch", &["chh", "c"]),
        ("c", &["ch", "k"]),
        // High Front Vowels: ee <-> i <-> ii <-> y
        ("ee", &["i", "ii"]),
        ("ii", &["ee", "i"]),
        ("i", &["ee", "ii", "y"]),
        ("y", &["e", "i"]),
        // High Back Vowels: oo <-> u <-> uu <-> o
        ("oo", &["u", "uu"]),
        ("uu", &["oo", "u"]),
        ("u", &["oo", "uu", "o"]),
        // Low/Mid Vowels: o <-> a <-> aa
        ("aa", &["a", "o"]),
        ("a", &["aa", "o"]),
        ("o", &["a", "u", "oo"]),
        // Flaps / Rhotics: r <-> R <-> rr <-> rrh <-> rh
        ("rrh", &["rr", "r", "R", "rh"]),
        ("rr", &["rrh", "r", "R"]),
        ("rh", &["R", "r", "rrh"]),
        ("r", &["R", "rr", "rrh"]),
        ("R", &["r", "rh", "rr"]),
        // Nasals: n <-> N <-> ng
        ("ng", &["n", "N"]),
        ("n", &["N", "ng"]),
        ("N", &["n"]),
        // Retroflex / Dentals: t <-> T, d <-> D, th <-> Th, dh <-> Dh
        ("th", &["Th", "t", "T"]),
        ("Th", &["th", "T", "t"]),
        ("t", &["T", "th"]),
        ("T", &["t", "Th"]),
        ("dh", &["Dh", "d", "D"]),
        ("Dh", &["dh", "D", "d"]),
        ("d", &["D", "dh"]),
        ("D", &["d", "Dh"]),
        // Clusters: x <-> kkh, gy <-> jnh
        ("x", &["kkh", "ks"]),
        ("kkh", &["x"]),
        ("gy", &["jnh", "gg"]),
    ];

    for &(target, replacements) in rules {
        for (pos, _) in lower.match_indices(target) {
            for &rep in replacements {
                let mut variant = String::with_capacity(input.len() + rep.len());
                variant.push_str(&input[..pos]);
                variant.push_str(rep);
                variant.push_str(&input[pos + target.len()..]);
                if variant != input {
                    variants.insert(variant);
                }
            }
        }
    }

    // Sort by length similarity and keep top 8 most probable variants
    let mut list: Vec<String> = variants.into_iter().collect();
    list.sort_unstable_by_key(|v| (v.len() as isize - input.len() as isize).abs());
    list.truncate(8);
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phonetic_variants() {
        let variants_s = generate_phonetic_variants("shob");
        assert!(variants_s.contains(&"sob".to_string()) || variants_s.contains(&"ssob".to_string()));

        let variants_z = generate_phonetic_variants("jontu");
        assert!(variants_z.contains(&"zontu".to_string()));

        let variants_vowel = generate_phonetic_variants("tumi");
        assert!(variants_vowel.contains(&"toomi".to_string()) || variants_vowel.contains(&"tuumi".to_string()));
    }
}
