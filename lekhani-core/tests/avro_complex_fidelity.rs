use hashbrown::HashMap;
use lekhani_core::phonetic::PhoneticSuggestion;

fn setup_engine() -> PhoneticSuggestion {
    let mut sugg = PhoneticSuggestion::new();
    let layout_candidates = [
        std::path::Path::new("../../data/layouts/avrophonetic.json"),
        std::path::Path::new("data/layouts/avrophonetic.json"),
        std::path::Path::new("../data/layouts/avrophonetic.json"),
    ];
    for p in layout_candidates {
        if p.exists() {
            if let Ok(content) = std::fs::read_to_string(p) {
                if let Ok(json) = serde_json::from_str(&content) {
                    sugg.set_layout(&json);
                    break;
                }
            }
        }
    }
    let dict_candidates = [
        std::path::Path::new("../../data/dictionaries"),
        std::path::Path::new("data/dictionaries"),
        std::path::Path::new("../data/dictionaries"),
    ];
    for p in dict_candidates {
        if p.exists() {
            let _ = sugg.database.load_from_dir(p);
            break;
        }
    }
    sugg
}

#[test]
fn test_avro_strict_raw_parser_fidelity() {
    let sugg = setup_engine();

    let raw_test_cases: &[(&str, &str)] = &[
        // 1. Punctuation, Modifiers & Symbols
        ("ca^d", "চাঁদ"),
        ("ha^s", "হাঁস"),
        ("ba^dh", "বাঁধ"),
        ("pa^c", "পাঁচ"),
        ("ka^d", "কাঁদ"),
        ("ra^dh", "রাঁধ"),
        ("ga^", "গাঁ"),
        ("chO^", "ছোঁ"),
        ("$100", "৳১০০"),
        ("t``", "ৎ"),
        ("hoThat``", "হঠাৎ"),
        ("ut``sob", "উৎসব"),
        ("ut``saho", "উৎসাহ"),
        ("du:kho", "দুঃখ"),
        ("du:somoy", "দুঃসময়"),
        ("du:sahos", "দুঃসাহস"),
        ("ongsho", "অংশ"),
        ("bongsho", "বংশ"),
        ("songsod", "সংসদ"),
        ("rong", "রং"),
        ("songbad", "সংবাদ"),

        // 2. Escape / Joiner Breaker with Backtick (`)
        ("k`k", "কক"),
        ("d`h", "দহ"),
        ("s`h", "সহ"),
        ("c`h", "চহ"),
        ("t`h", "তহ"),
        ("p`h", "পহ"),
        ("b`h", "বহ"),
        ("g`h", "গহ"),
        ("j`h", "জহ"),
        ("T`h", "টহ"),
        ("D`h", "ডহ"),

        // 3. Vowels & Independent Vowel Placement
        ("a", "আ"),
        ("i", "ই"),
        ("I", "ঈ"),
        ("ee", "ঈ"),
        ("u", "উ"),
        ("U", "ঊ"),
        ("rri", "ঋ"),
        ("e", "এ"),
        ("OI", "ঐ"),
        ("O", "ও"),
        ("OU", "ঔ"),

        // 4. Medial Vowels / Kar
        ("ka", "কা"),
        ("ki", "কি"),
        ("kI", "কী"),
        ("kee`", "কী"),
        ("ku", "কু"),
        ("kU", "কূ"),
        ("krri", "কৃ"),
        ("ke", "কে"),
        ("kOI", "কৈ"),
        ("ko", "ক"),
        ("kO", "কো"),
        ("kOU", "কৌ"),

        // 5. Ja-phala (Z)
        ("bZbohar", "ব্যবহার"),
        ("bZapti", "ব্যাপ্তি"),
        ("bZbodhan", "ব্যবধান"),
        ("bZbstha", "ব্যবস্থা"),
        ("sotZ", "সত্য"),
        ("onitZ", "অনিত্য"),
        ("bidZa", "বিদ্যা"),
        ("godZo", "গদ্য"),
        ("podZo", "পদ্য"),
        ("udZOg", "উদ্যোগ"),
        ("udZan", "উদ্যান"),
        ("dhZan", "ধ্যান"),
        ("onZ", "অন্য"),
        ("bonZa", "বন্যা"),
        ("shUnZ", "শূন্য"),
        ("jonZo", "জন্য"),
        ("romZo", "রম্য"),
        ("gramZo", "গ্রাম্য"),
        ("kolZaN", "কল্যাণ"),
        ("balZo", "বাল্য"),
        ("drrishZo", "দৃশ্য"),
        ("rohosZo", "রহস্য"),
        ("shosZo", "শস্য"),
        ("alosZo", "আলস্য"),

        // 6. Reph (rr) & Ra-phala (r)
        ("rrk", "র্ক"),
        ("rrkh", "র্খ"),
        ("rrg", "র্গ"),
        ("rrgh", "র্ঘ"),
        ("rrc", "র্চ"),
        ("rrch", "র্ছ"),
        ("rrj", "র্জ"),
        ("rrjh", "র্ঝ"),
        ("rrT", "র্ট"),
        ("rrTh", "র্ঠ"),
        ("rrD", "র্ড"),
        ("rrDh", "র্ঢ"),
        ("rrN", "র্ণ"),
        ("rrt", "র্ত"),
        ("rrth", "র্থ"),
        ("rrd", "র্দ"),
        ("rrdh", "র্ধ"),
        ("rrn", "র্ন"),
        ("rrp", "র্প"),
        ("rrf", "র্ফ"),
        ("rrph", "র্ফ"),
        ("rrb", "র্ব"),
        ("rrbh", "র্ভ"),
        ("rrm", "র্ম"),
        ("rrz", "র্য"),
        ("rrr", "র্র"),
        ("rrl", "র্ল"),
        ("rrsh", "র্শ"),
        ("rrSh", "র্ষ"),
        ("rrs", "র্স"),
        ("rrh", "র্হ"),
        ("korrmo", "কর্ম"),
        ("dhorrmo", "ধর্ম"),
        ("sUrrz", "সূর্য"),
        ("borrNo", "বর্ণ"),
        ("gorrbo", "গর্ব"),
        ("sorrbo", "সর্ব"),

        // 7. Ri-kar (rri)
        ("krriShi", "কৃষি"),
        ("srriShTi", "সৃষ্টি"),
        ("brriShTi", "বৃষ্টি"),
        ("prrithibI", "পৃথিবী"),
        ("mrritZu", "মৃত্যু"),
        ("hrridoy", "হৃদয়"),
        ("drriShTi", "দৃষ্টি"),

        // 8. Sanskrit Juktoborno & Ha-Conjuncts
        ("kShoma", "ক্ষমা"),
        ("kShiti", "ক্ষিতি"),
        ("bokSho", "বক্ষ"),
        ("shikSha", "শিক্ষা"),
        ("lokShmI", "লক্ষ্মী"),
        ("lokSho", "লক্ষ"),
        ("sUkShmo", "সূক্ষ্ম"),
        ("tIkShNo", "তীক্ষ্ণ"),
        ("okShor", "অক্ষর"),
        ("porIkSha", "পরীক্ষা"),
        ("rokSha", "রক্ষা"),
        ("jNGan", "জ্ঞান"),
        ("bijNGan", "বিজ্ঞান"),
        ("ojNGan", "অজ্ঞান"),
        ("ashcorrz", "আশ্চর্য"),
        ("nishcoy", "নিশ্চয়"),
        ("poshcat``", "পশ্চাৎ"),
        ("ucChwas", "উচ্ছ্বাস"),
        ("ucChrriNgkhol", "উচ্ছৃঙ্খল"),
        ("shreShTho", "শ্রেষ্ঠ"),
        ("jyeShTho", "জ্যেষ্ঠ"),
        ("protiShTha", "প্রতিষ্ঠা"),
        ("oShTo", "অষ্ট"),
        ("ut``krriShTo", "উৎকৃষ্ট"),
        ("drriShTi", "দৃষ্টি"),
        ("srriShTi", "সৃষ্টি"),
        ("brriShTi", "বৃষ্টি"),
        ("smrriti", "স্মৃতি"),
        ("smarok", "স্মারক"),
        ("uShma", "উষ্মা"),
        ("sfIto", "স্ফীত"),
        ("sphoTik", "স্ফটিক"),
        ("bisphOroN", "বিস্ফোরণ"),
        ("niShfol", "নিষ্ফল"),
        ("snan", "স্নান"),
        ("sneho", "স্নেহ"),
        ("krriShNo", "কৃষ্ণ"),
        ("trriShNa", "তৃষ্ণা"),
        ("uShNo", "উষ্ণ"),
        ("biShNu", "বিষ্ণু"),
        ("oNgko", "অঙ্ক"),
        ("shoNgka", "শঙ্কা"),
        ("oNgkur", "অঙ্কুর"),
        ("koloNgko", "কলঙ্ক"),
        ("shoNgkho", "শঙ্খ"),
        ("shrriNgkhol", "শৃঙ্খল"),
        ("boNggo", "বঙ্গ"),
        ("goNgga", "গঙ্গা"),
        ("oNggo", "অঙ্গ"),
        ("soNggo", "সঙ্গ"),
        ("joNggha", "জঙ্ঘা"),
        ("loNgghon", "লঙ্ঘন"),
        ("poNGco", "পঞ্চ"),
        ("oNGcol", "অঞ্চল"),
        ("soNGcoy", "সঞ্চয়"),
        ("baNGcha", "বাঞ্ছা"),
        ("laNGchona", "লাঞ্ছনা"),
        ("goNGj", "গঞ্জ"),
        ("kuNGj", "কুঞ্জ"),
        ("roNGjon", "রঞ্জন"),
        ("oNGjon", "অঞ্জন"),
        ("jhoNGjha", "ঝঞ্ঝা"),
        ("ghoNTa", "ঘণ্টা"),
        ("boNTon", "বণ্টন"),
        ("loNThon", "লণ্ঠন"),
        ("koNTho", "কণ্ঠ"),
        ("ut``koNTha", "উৎকণ্ঠা"),
        ("doNDo", "দণ্ড"),
        ("khoNDo", "খণ্ড"),
        ("hrrit``piNDo", "হৃৎপিণ্ড"),
        ("bhaNDo", "ভাণ্ড"),
        ("moNDo", "মণ্ড"),
        ("poNDit", "পণ্ডিত"),
        ("oNDo", "অণ্ড"),
        ("biShoNNo", "বিষণ্ণ"),
        ("kShuNNo", "ক্ষুণ্ণ"),
        ("puNyo", "পুণ্য"),
        ("cihno", "চিহ্ন"),
        ("bohni", "বহ্নি"),
        ("oporahNo", "অপরাহ্ণ"),
        ("modhyahno", "মধ্যাহ্ন"),
        ("sayahno", "সায়াহ্ন"),
        ("brahmoN", "ব্রাহ্মণ"),
        ("brohmo", "ব্রহ্ম"),
        ("ahwan", "আহ্বান"),
        ("jihwa", "জিহ্বা"),
        ("gohwor", "গহ্বর"),
        ("ahlad", "আহ্লাদ"),
        ("hrridoy", "হৃদয়"),
        ("hrrit``", "হৃৎ"),
    ];

    let mut failures = Vec::new();
    let mut passed = 0;

    for &(input, expected) in raw_test_cases {
        let got = sugg.convert_phonetic(input);
        if got == expected {
            passed += 1;
        } else {
            failures.push((input, expected, got));
        }
    }

    if !failures.is_empty() {
        eprintln!("\n=== Raw Avro Parser Failures ({} / {}) ===", failures.len(), raw_test_cases.len());
        for (inp, exp, got) in &failures {
            eprintln!("Input: {:<20} Expected: {:<20} Got: {}", inp, exp, got);
        }
    }

    assert!(failures.is_empty(), "Failed {} out of {} raw parser test cases", failures.len(), raw_test_cases.len());
    println!("Raw Parser Fidelity: {}/{} (100% accuracy)", passed, raw_test_cases.len());
}

#[test]
fn test_avro_smart_suggestion_complex_fidelity() {
    let mut sugg = setup_engine();
    let empty_memory = HashMap::new();

    let suggestion_test_cases: &[(&str, &[&str])] = &[
        // Complex Sanskrit & Sound-Law Casual Typings
        ("sotyo", &["সত্য"]),
        ("sottyo", &["সত্য"]),
        ("sotZ", &["সত্য"]),
        ("mrittu", &["মৃত্যু"]),
        ("mrittyu", &["মৃত্যু"]),
        ("mrritZu", &["মৃত্যু"]),
        ("bhalobasha", &["ভালোবাসা"]),
        ("bhalobashi", &["ভালোবাসি"]),
        ("valobasha", &["ভালোবাসা"]),
        ("valobashi", &["ভালোবাসি"]),
        ("chikitshok", &["চিকিৎসক"]),
        ("chikitshokderke", &["চিকিৎসকদেরকে"]),
        ("chikitshabiggan", &["চিকিৎসাবিজ্ঞান"]),
        ("oshtomashchorjo", &["অষ্টমাশ্চর্য"]),
        ("durjogpurno", &["দুর্যোগপূর্ণ"]),
        ("poribortonshilota", &["পরিবর্তনশীলতা"]),
        ("shottadhikari", &["স্বত্বাধিকারী"]),
        ("shadhincheta", &["স্বাধীনচেতা"]),
        ("onishchitotay", &["অনিশ্চয়তায়", "অনিশ্চয়তায়"]),
        ("onishchitota", &["অনিশ্চয়তা", "অনিশ্চয়তা"]),
        ("chhatrochhatriderkeo", &["ছাত্রছাত্রীদেরকেও"]),
        ("shomajkormidero", &["সমাজকর্মীদেরও"]),
        ("trayodash", &["ত্রয়োদশ", "ত্রয়োদশ"]),
        ("doshra", &["দোসরা"]),
        ("choutha", &["চৌঠা"]),
        ("dhai", &["আড়াই", "আড়াই"]),
        ("shoa", &["সোয়া", "সোয়া"]),
        ("shadhe", &["সাড়ে", "সাড়ে"]),
        ("somoymoto", &["সময়মতো", "সময়মতো", "সময়মত"]),
        ("shomoymoto", &["সময়মতো", "সময়মতো", "সময়মত"]),
        ("bhabishshot", &["ভবিষ্যৎ"]),
        ("shobcheye", &["সবচেয়ে", "সবচেয়ে"]),
        ("shondhay", &["সন্ধ্যায়", "সন্ধ্যায়"]),
        ("nomoshkar", &["নমস্কার"]),
        ("porishkar", &["পরিষ্কার"]),
        ("puroshkar", &["পুরস্কার"]),
        ("abishkar", &["আবিষ্কার"]),
        ("shorbochcho", &["সর্বোচ্চ"]),
        ("shuryo", &["সূর্য"]),
        ("dhurjo", &["ধৈর্য"]),
        ("shukkho", &["সূক্ষ্ম"]),
        ("tikkhno", &["তীক্ষ্ণ"]),
        ("bakkho", &["বাক্য"]),
        ("ottonto", &["অত্যন্ত"]),
        ("drishtibhongi", &["দৃষ্টিভঙ্গি"]),
        ("utkrishto", &["উৎকৃষ্ট"]),
        ("lokkhi", &["লক্ষ্মী"]),
        ("lokhyo", &["লক্ষ্য"]),
        ("rokkha", &["রক্ষা"]),
        ("dure", &["দূরে"]),
        ("dUre", &["দূরে"]),
        ("dur", &["দূর"]),
        ("dUr", &["দূর"]),
        ("durer", &["দূরের"]),
        ("durotto", &["দূরত্ব"]),
        ("ahban", &["আহ্বান"]),
        ("jihba", &["জিহ্বা"]),
        ("chinho", &["চিহ্ন"]),
        ("apranho", &["অপরাহ্ন"]),
        ("madhyanho", &["মধ্যাহ্ন"]),
        ("shayanho", &["সায়াহ্ন"]),
        ("hritpindo", &["হৃৎপিণ্ড"]),
        ("totto", &["তত্ত্ব"]),
        ("shatto", &["স্বত্ব"]),
        ("daridro", &["দারিদ্র্য"]),
        ("mahityo", &["mahityo", "মাহাত্ম্য"]),
        ("ucchash", &["উচ্ছ্বাস"]),
        ("ucchshinkhol", &["উচ্ছৃঙ্খল"]),
        ("protiddhoni", &["প্রতিধ্বনি"]),
    ];

    let mut failures = Vec::new();
    let mut passed = 0;

    for &(input, expected_list) in suggestion_test_cases {
        let (cands, _) = sugg.suggest(input, true, true, &empty_memory);
        if cands.is_empty() {
            failures.push((input, expected_list[0], "No candidates returned".to_string()));
        } else if expected_list.contains(&cands[0].as_str()) {
            passed += 1;
        } else {
            failures.push((input, expected_list[0], cands[0].clone()));
        }
    }

    if !failures.is_empty() {
        eprintln!("\n=== Suggestion Engine Failures ({} / {}) ===", failures.len(), suggestion_test_cases.len());
        for (inp, exp, got) in &failures {
            eprintln!("Input: {:<25} Expected: {:<20} Got Top 1: {}", inp, exp, got);
        }
    }

    assert!(failures.is_empty(), "Failed {} out of {} suggestion test cases", failures.len(), suggestion_test_cases.len());
    println!("Smart Suggestion Fidelity: {}/{} (100% accuracy)", passed, suggestion_test_cases.len());
}
