//! Fuzzy Phonetic Variation Generator
//!
//! Generates linguistically accurate phoneme spelling variations for Latin input to handle
//! ambiguous Bengali phonemes (e.g. s/sh/Sh, z/j, i/ee/ii, u/oo/uu, r/R/rh, n/N, t/T/th/Th).

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

/// Generate plausible phonetic spelling variants of a Latin word
pub fn generate_phonetic_variants(input: &str) -> Vec<String> {
    if input.is_empty() || input.len() > 25 {
        return Vec::new();
    }

    let lower = input.to_lowercase();

    // 1. Comprehensive Phoneme, Sibilant, Retroflex, Conjunct & Cluster Rules
    let rules: &[(&str, &[&str])] = &[
        // Sibilants (স, শ, ষ)
        ("sh", &["s", "Sh"]),
        ("Sh", &["sh", "s"]),
        ("ss", &["sh", "s"]),
        ("s", &["sh", "Sh"]),

        // Sibilant Stop / Fricative Conjuncts (ষ্ট, ষ্ঠ, স্ক, ষ্প, স্ফ)
        ("st", &["ShT", "sT", "ShTh"]),
        ("sth", &["ShTh", "sTh", "sth"]),
        ("sT", &["ShT", "st"]),
        ("shT", &["ShT", "st"]),
        ("shth", &["ShTh", "sth"]),
        ("shTh", &["ShTh", "sth"]),
        ("sk", &["Shk", "sk"]),
        ("shk", &["Shk", "sk"]),
        ("sp", &["Shp", "sp"]),
        ("shp", &["Shp", "sp"]),
        ("sf", &["Shf", "sf"]),
        ("shf", &["Shf", "sf"]),

        // Retroflex / Dental Stops (ত/ট, থ/ঠ, দ/ড, ধ/ঢ)
        ("th", &["Th", "t"]),
        ("Th", &["th"]),
        ("t", &["T", "th"]),
        ("T", &["t"]),
        ("dh", &["Dh", "d"]),
        ("Dh", &["dh"]),
        ("d", &["D", "dh"]),
        ("D", &["d"]),

        // Flaps / Rhotics / Ri-kar (র, ড়, ঢ়, ঋ/ৃ)
        ("rrh", &["rh", "R", "r"]),
        ("rr", &["r", "R"]),
        ("rh", &["Rh", "R", "r"]),
        ("Rh", &["rh", "R"]),
        ("ri", &["rri", "ree"]),
        ("rri", &["ri"]),
        ("r", &["R", "rh"]),
        ("R", &["r", "Rh"]),

        // Nasals & Anusvara (ন, ণ, ঙ, ং)
        ("ng", &["n", "Ng"]),
        ("n", &["N", "ng"]),
        ("N", &["n"]),

        // Labials & Affricates (ভ, ব, য, জ, ঝ)
        ("v", &["bh", "b", "w"]),
        ("bh", &["v", "b"]),
        ("b", &["bh", "v"]),
        ("z", &["j"]),
        ("j", &["z", "jh"]),
        ("jh", &["j"]),
        ("w", &["o", "oy", "v"]),

        // Aspirated / Unaspirated Consonants (ক/খ, গ/ঘ, চ/ছ, প/ফ)
        ("kh", &["k"]),
        ("k", &["kh"]),
        ("gh", &["g"]),
        ("g", &["gh"]),
        ("chh", &["ch", "c"]),
        ("ch", &["chh", "c"]),
        ("c", &["ch", "k"]),
        ("ph", &["f", "p"]),
        ("f", &["ph", "p"]),
        ("p", &["ph", "f"]),

        // Homorganic Vowels & Diphthongs (ই/ঈ, উ/ঊ, ও/ো, ঐ/ৈ, ঔ/ৌ)
        ("ee", &["i", "I", "ii"]),
        ("ii", &["ee", "I", "i"]),
        ("i", &["I", "ee", "ii"]),
        ("I", &["i", "ee"]),
        ("oo", &["u", "U", "uu"]),
        ("uu", &["oo", "U", "u"]),
        ("u", &["U", "oo", "uu"]),
        ("U", &["u", "oo"]),
        ("aa", &["a"]),
        ("o", &["O"]),
        ("O", &["o"]),
        ("oi", &["OI", "oy"]),
        ("ou", &["OU", "ow"]),
        ("OI", &["oi"]),
        ("OU", &["ou"]),

        // Reph Consonants (র-ফলা / রেফ: র্)
        ("rt", &["rrt"]),
        ("rth", &["rrth"]),
        ("rj", &["rrz", "rrj"]),
        ("rsh", &["rrsh", "rrSh"]),
        ("rn", &["rrN", "rrn"]),
        ("rb", &["rrb"]),
        ("rm", &["rrm"]),
        ("rk", &["rrk"]),
        ("rp", &["rrp"]),
        ("rd", &["rrd"]),
        ("rdh", &["rrdh"]),
        ("rg", &["rrg"]),
        ("rgh", &["rrgh"]),

        // Verb Inflections & Past/Present Continuous (চ্ছেন, ছেন, ইত্যাদি)
        ("chhen", &["chen", "cchen", "shen"]),
        ("chho", &["cho", "ccho", "sho"]),
        ("chhe", &["che", "cche", "she"]),
        ("chhi", &["chi", "cchi", "shi"]),
        ("chhilo", &["chilo", "cchilo"]),
        ("chhilam", &["chilam", "cchilam"]),
        ("chhile", &["chile", "cchile"]),
        ("chhilenn", &["chilen", "cchilen"]),

        // Motion Verbs & Initial/Medial Antastha-Ja (য)
        ("jawa", &["zawa", "jaoya", "zaoya"]),
        ("jabo", &["zabo", "zao"]),
        ("jacchi", &["zacchi"]),
        ("jaccho", &["zaccho"]),
        ("jacchen", &["zacchen"]),
        ("jacche", &["zacche"]),
        ("jete", &["zete"]),
        ("juddho", &["zuddho"]),
        ("jontro", &["zontro"]),
        ("jogajog", &["zogazog"]),
        ("projukti", &["prozukti"]),
        ("onujayi", &["onuzayi"]),
        ("joggo", &["zogg"]),

        // Geminates & Ja-fala Conjuncts (দ্ব, দ্য, থ্য, ন্য)
        ("biddaloy", &["bidZaloy", "bidyaloy"]),
        ("biddut", &["bidZut``", "bidyut``"]),
        ("biddan", &["bidZan", "bidwan", "bidyan"]),
        ("tottho", &["tothZ", "tothyo", "tothya"]),
        ("totto", &["tottwo", "totwo"]),
        ("jonne", &["jonyo", "jonZ", "janya"]),
        ("khettro", &["kheZtro", "khetro"]),

        // Sibilant Harmonization (স্ব, স্ম, প্রমিত স)
        ("shadhinota", &["swadhinota", "shwadhinota"]),
        ("shadhin", &["swadhin", "shwadhin"]),
        ("shagotom", &["swagotom", "shwagotom"]),
        ("shastho", &["swastho", "swasthyo", "shwastho"]),
        ("sriti", &["smrriti", "shmrriti"]),
        ("smarok", &["shmarok", "smarok"]),
        ("shorkar", &["sorkar"]),
        ("shorkari", &["sorkari", "sorkaree"]),
        ("shobai", &["sobai"]),
        ("shongbidhan", &["songbidhan"]),
        ("shongbidhane", &["songbidhane"]),
        ("shongbidhaner", &["songbidhaner"]),
        ("shonkha", &["songkhya", "shongkhya"]),
        ("sonkha", &["songkhya"]),
        ("shonkhagorishtho", &["songkhyagoriShTho", "shongkhyagoriShTho"]),
        ("shonkhagoristho", &["songkhyagoriShTho", "shongkhyagoriShTho"]),
        ("sonkhagorishtho", &["songkhyagoriShTho"]),
        ("shodoshyo", &["sodoshyo", "sodosZ"]),
        ("shodoshyoder", &["sodoshyoder", "sodosZder"]),
        ("sodoshyoder", &["sodosZder"]),
        ("shiddhanto", &["siddhanto"]),
        ("proshashon", &["proshason"]),
        ("proshashonik", &["proshasonik"]),
        ("shochetonota", &["sochetonota"]),
        ("shocheton", &["socheton"]),
        ("shasroy", &["sashroy"]),
        ("shasroyi", &["sashroyi"]),
        ("shomporko", &["somporko", "somporrko"]),
        ("shangbadik", &["sangbadik"]),
        ("shadharon", &["sadharon", "sadharoN"]),
        ("sadharon", &["sadharoN"]),
        ("shondha", &["shondhya", "sondha", "sondhya"]),

        // Demonstratives & Emphatics (এটা, এটাই, সেটা, সেটাই, ওটা, ওটাই)
        ("etai", &["eTai"]),
        ("eta", &["eTa"]),
        ("setai", &["seTai"]),
        ("seta", &["seTa"]),
        ("otai", &["OTai", "oTai"]),
        ("ota", &["OTa", "oTa"]),
        ("kotai", &["koNTai", "konTai"]),
        ("jotai", &["zoTai", "zotai"]),
        ("shanto", &["shanto", "santo"]),
        ("shanti", &["shanti", "santi"]),
        ("shompurno", &["sompoorrNo", "sompoorno", "somporrNo"]),
        ("sompurno", &["sompoorrNo", "sompoorno", "somporrNo"]),
        ("shongkhipto", &["songkhipto", "shongkhipTo", "songkhipTo"]),
        ("songkhipto", &["songkhipTo"]),
        ("ottotag", &["attotZag", "attotyag"]),
        ("attotag", &["attotZag", "attotyag"]),
        ("shartho", &["swartho", "shwartho", "sartho"]),
        ("sharthokota", &["sarthokota"]),
        ("shakkhi", &["sakkhi", "shakshi"]),
        ("shakhor", &["swakkhor", "shwakkhor", "sakkhor"]),
        ("shundor", &["sundor"]),
        ("shobuj", &["sobuj"]),
        ("shukh", &["sukh"]),
        ("shukhi", &["sukhi"]),
        ("shustho", &["sustho"]),
        ("oshustho", &["osustho"]),
        ("shombhob", &["sombhob"]),
        ("oshombhob", &["osombhob"]),
        ("shombhabona", &["sombhabona"]),
        ("drishti", &["drriShTi", "dristi"]),
        ("drishtikon", &["drriShTikoN", "dristikon"]),
        ("onnanno", &["onZanZ", "onnano"]),
        ("onnorokom", &["onZrokom", "onnorokom"]),
        ("poriborton", &["porriborton", "porriborrtan"]),
        ("shathe", &["sathe"]),
        ("songskriti", &["soNskrriti", "shongskriti"]),
        ("songskar", &["soNskar", "shongskar"]),
        ("songstha", &["soNstha", "shongstha"]),
        ("utshob", &["ut``shob", "utsab"]),
        ("utshaho", &["ut``shaho", "utsaho"]),
        ("utpadon", &["ut``padon", "utpadoN"]),
        ("utkrishto", &["ut``krriShTo", "utkrishto"]),
        ("utshorgo", &["ut``shorrgo", "utsorgo"]),
        ("utsho", &["ut``sho", "utso"]),
        ("chitkar", &["cit``kar", "chit``kar"]),
        ("boshonto", &["bosonto", "bOshonto"]),
        ("cokh", &["chokh", "cokhe"]),
        ("chokh", &["cokhe", "cokh"]),
        ("jorano", &["juRanO", "juranO"]),
        ("fute", &["fuTe"]),
        ("shiter", &["sheeter"]),
        ("hat-te", &["ha^Tte", "hatte"]),
        ("hatte", &["ha^Tte"]),
        ("cheShTa", &["ceShTa"]),
        ("chesta", &["ceShTa", "ceshTa"]),
        ("chao", &["cao"]),
        ("chomotkar", &["comotkar", "comot``kar"]),
        ("abong", &["ebong"]),
        ("madhyom", &["madhZom", "madhyom"]),
        ("maDhyom", &["madhZom", "madhyom"]),
        ("juktakkhor", &["zuktakkhor", "juktakshor"]),
        ("kono", &["kOnO", "kOno"]),
        ("shudhu", &["shUdhu"]),
        ("borno", &["borrno", "borrNo"]),
        ("gulo", &["gulO"]),
        ("tipe", &["Tipe"]),
        ("holo", &["holO"]),
        ("koro", &["korO"]),
        ("kosto", &["koShTo", "koshTo"]),
        ("phonetic", &["fOneTik", "fonetik", "fOnetik"]),
        ("typin-g", &["Taiping", "typing", "TaipiN"]),
        ("typing", &["Taiping", "TaipiN"]),
        ("sahajjo", &["sahazZo", "sahajZ"]),
        ("eti", &["eTi"]),
        ("eTi", &["eTi"]),
        ("shohoj", &["sohoj"]),
        ("shokale", &["sokale"]),

        // Ri-kar, Long Vowels & Retroflex Stops
        ("kritrim", &["krritrim", "krriTrim"]),
        ("matribhumi", &["matrribhUmi", "matrribhumi", "matrribhoomi"]),
        ("prakritik", &["prakrritik"]),
        ("briddhi", &["brriddhi", "brriddhee"]),
        ("durniti", &["durrneeti", "durrniti"]),
        ("buddhijibi", &["buddhijebee", "buddhijibi"]),
        ("shahid", &["shOheed", "shaheed"]),
        ("dushon", &["dooshon", "dooshoN", "dUShoN"]),
        ("bayumondol", &["bayumonDol", "bayumonDoli"]),
        ("bayumondoliyo", &["bayumonDoleeyo", "bayumonDoli", "bayumonDolee", "bayumonDoliyo"]),
        ("ghotche", &["ghoTche"]),
        ("ghotona", &["ghoTona"]),
        ("ekta", &["ekTa"]),
        ("ekti", &["ekTi"]),
        ("uthe", &["uThe"]),
        ("uthbo", &["uThbo"]),
        ("kothin", &["koThin"]),
        ("chhot", &["chhoT"]),
        ("chhotota", &["chhoTota"]),
        ("bhorta", &["bhorTa", "bhorrTa"]),
        ("ulto", &["ulTo"]),
        ("ghonta", &["ghonTa", "ghonTa"]),

        // Juktoborno Clusters (জ্ঞ, ক্ষ, স্ম, ষ্ণ, স্ব, ইত্যাদি)
        ("sristi", &["srriShTi"]),
        ("bristi", &["brriShTi"]),
        ("kristi", &["krriShTi"]),
        ("krishi", &["krriShi"]),
        ("sristy", &["srriShTi"]),
        ("bristy", &["brriShTi"]),
        ("rist", &["rriShT", "rrisT"]),
        ("kkh", &["x", "ks", "kh"]),
        ("x", &["kkh", "ks"]),
        ("ks", &["x", "kkh"]),
        ("ggan", &["jNGan", "GGan"]),
        ("ggani", &["jNGani", "jNGanee", "jNGanI", "GGani", "GGanee", "GGanI"]),
        ("ganni", &["jNGani", "jNGanee", "jNGanI", "GGani", "GGanee", "GGanI"]),
        ("ggyan", &["jNGan", "GGan"]),
        ("gyan", &["jNGan", "GGan"]),
        ("gyani", &["jNGani", "jNGanee", "jNGanI", "GGani", "GGanee", "GGanI"]),
        ("gg", &["jNG", "GG"]),
        ("gy", &["jNG", "GG"]),
        ("gn", &["jNG", "GG"]),
        ("sm", &["Shm", "shm"]),
        ("sn", &["ShN", "shn"]),
        ("sw", &["shw", "s"]),
        ("cch", &["cc", "ch"]),
        ("cc", &["cch", "ch"]),
        ("ddh", &["dd", "dh"]),
        ("dd", &["ddh", "d"]),
        ("tt", &["t"]),
        ("jj", &["j"]),
        ("orthonoi", &["orrthonOI"]),
        ("ortho", &["orrtho"]),
        ("noitik", &["nOItik"]),
        ("boiggan", &["bOIjNGan"]),
        ("oitihas", &["OItihas"]),
        ("bebostha", &["byobostha", "byabostha", "bZabostha"]),
        ("bebo", &["byobo", "byabo"]),
        ("beba", &["byoba", "byaba"]),
        ("beb", &["byob", "byab"]),
        ("bya", &["by", "be"]),
        ("by", &["b", "be"]),
    ];

    // Helper closure to apply one pass of substitutions
    let apply_rules = |src: &str, out: &mut HashSet<String>| {
        let src_lower = src.to_lowercase();
        for &(target, replacements) in rules {
            for (pos, _) in src_lower.match_indices(target) {
                for &rep in replacements {
                    let mut variant = String::with_capacity(src_lower.len() + rep.len());
                    variant.push_str(&src_lower[..pos]);
                    variant.push_str(rep);
                    variant.push_str(&src_lower[pos + target.len()..]);
                    if variant != src && variant != input {
                        out.insert(variant);
                    }
                }
            }
        }
    };

    let mut pass1 = HashSet::new();

    // 2. Include run-length collapsed variants if any (e.g. thiiik -> thik)
    for collapsed in collapse_elongated_runs(&lower) {
        pass1.insert(collapsed.clone());
        apply_rules(&collapsed, &mut pass1);
    }

    apply_rules(&lower, &mut pass1);

    // 3. Bengali Glide Verbs (e.g. khawa -> khaoya, dewa -> deoya, jawa -> jaoya)
    if lower.ends_with("awa") && lower.len() >= 4 {
        let glide = format!("{}aoya", &lower[..lower.len() - 3]);
        pass1.insert(glide);
    } else if lower.ends_with("ewa") && lower.len() >= 4 {
        let glide = format!("{}eoya", &lower[..lower.len() - 3]);
        pass1.insert(glide);
    } else if lower.ends_with("owa") && lower.len() >= 4 {
        let glide = format!("{}ooya", &lower[..lower.len() - 3]);
        pass1.insert(glide);
    }

    // 4. Chandrabindu (ঁ) variants (e.g. chad -> ca^d, dat -> da^t, has -> ha^s, bas -> ba^sh, pach -> pa^c)
    let chandrabindu_words: &[(&str, &str)] = &[
        ("chad", "ca^d"),
        ("cad", "ca^d"),
        ("chaad", "ca^d"),
        ("chand", "ca^d"),
        ("dat", "da^t"),
        ("daat", "da^t"),
        ("faka", "pha^ka"),
        ("badh", "ba^dh"),
        ("has", "ha^s"),
        ("bas", "ba^sh"),
        ("pach", "pa^c"),
        ("panc", "pa^c"),
        ("thot", "Tho^T"),
        ("khuji", "khu^ji"),
        ("kach", "ka^c"),
    ];
    for &(exact, rep) in chandrabindu_words {
        if lower == exact {
            pass1.insert(rep.to_string());
        }
    }

    // 5. Khanda-Ta (ৎ) & Hasanta variants (e.g. hothat -> hoThat``, utsob -> ut``sob, utsaho -> ut``saho)
    let khanda_ta_words: &[(&str, &str)] = &[
        ("hothat", "hoThat``"),
        ("hotat", "hoThat``"),
        ("utsob", "ut``sob"),
        ("utshob", "ut``sob"),
        ("utsaho", "ut``saho"),
        ("utshaho", "ut``saho"),
        ("utponno", "ut``ponno"),
        ("utpadon", "ut``padon"),
        ("utkrishto", "ut``krriShTo"),
        ("utkristo", "ut``krriShTo"),
        ("bikkhat", "bikkhat``"),
        ("bikhyat", "bikkhat``"),
        ("totpor", "tot``por"),
        ("biddut", "bidZut``"),
        ("attotag", "attotag``"),
        ("ottotag", "attotag``"),
        ("ashirbad", "ashirrbaad"),
        ("ashirbade", "ashirrbaade"),
        ("shurjo", "shoorrzo"),
    ];
    for &(exact, rep) in khanda_ta_words {
        if lower == exact {
            pass1.insert(rep.to_string());
        }
    }

    // 6. Pass 2: Combined 2nd-level variations for multi-phoneme words (e.g. sristi -> sriShTi, onusthan -> onuShThan)
    let mut list: Vec<String> = pass1.into_iter().collect();
    list.sort_unstable_by_key(|v| (v.len() as isize - input.len() as isize).abs());

    let mut pass2 = HashSet::new();
    for p1 in list.iter().take(48) {
        apply_rules(p1, &mut pass2);
    }

    let mut pass2_list: Vec<String> = pass2.into_iter().filter(|v| !list.contains(v) && v != input).collect();
    pass2_list.sort_unstable_by_key(|v| (v.len() as isize - input.len() as isize).abs());

    list.extend(pass2_list);
    list.truncate(96);
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

        // Elongation collapse
        let collapsed_thik = collapse_elongated_runs("thiiik");
        assert!(collapsed_thik.contains(&"thik".to_string()));

        let collapsed_bhalo = collapse_elongated_runs("bhalooo");
        assert!(collapsed_bhalo.contains(&"bhalo".to_string()));

        // Chandrabindu & Khandata variants
        let variants_chad = generate_phonetic_variants("chad");
        assert!(variants_chad.contains(&"ca^d".to_string()));

        let variants_hothat = generate_phonetic_variants("hothat");
        assert!(variants_hothat.contains(&"hoThat``".to_string()));

        // Glide verbs
        let variants_khawa = generate_phonetic_variants("khawa");
        assert!(variants_khawa.contains(&"khaoya".to_string()));

        // Word safety check: basha should NOT become b^asha
        let variants_basha = generate_phonetic_variants("basha");
        assert!(!variants_basha.contains(&"b^asha".to_string()));
    }
}
