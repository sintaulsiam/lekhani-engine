use hashbrown::HashMap;
use lekhani_core::phonetic::PhoneticSuggestion;

#[test]
fn test_dure_suggestions() {
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
    let empty_memory = HashMap::new();
    let (cands_dure, _) = sugg.suggest("dure", true, true, &empty_memory);
    let (cands_d_ure, _) = sugg.suggest("dUre", true, true, &empty_memory);
    let (cands_dur, _) = sugg.suggest("dur", true, true, &empty_memory);
    let (cands_d_ur, _) = sugg.suggest("dUr", true, true, &empty_memory);

    assert_eq!(cands_dure[0], "দূরে", "Expected 'দূরে' for 'dure'");
    assert_eq!(cands_d_ure[0], "দূরে", "Expected 'দূরে' for 'dUre'");
    assert_eq!(cands_dur[0], "দূর", "Expected 'দূর' for 'dur'");
    assert_eq!(cands_d_ur[0], "দূর", "Expected 'দূর' for 'dUr'");
}

#[test]
fn test_quality_fixes_regression() {
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
    let empty_memory = HashMap::new();

    // 1. Snippet Macro Collision Fix
    let (cands_dhonnobad, _) = sugg.suggest("dhonnobad", true, true, &empty_memory);
    assert_eq!(
        cands_dhonnobad[0], "ধন্যবাদ",
        "Expected 'ধন্যবাদ' for 'dhonnobad'"
    );

    let (cands_macro_dhonnobad, _) = sugg.suggest("!dhonnobad", true, true, &empty_memory);
    assert_eq!(
        cands_macro_dhonnobad[0], "আপনাকে অনেক অনেক ধন্যবাদ",
        "Expected snippet macro for '!dhonnobad'"
    );

    // 2. Chandra Bindu Position Reordering Fix
    let (cands_cad, _) = sugg.suggest("c^ad", true, true, &empty_memory);
    assert_eq!(cands_cad[0], "চাঁদ", "Expected 'চাঁদ' for 'c^ad'");

    let (cands_k_ada, _) = sugg.suggest("k^ada", true, true, &empty_memory);
    assert_eq!(cands_k_ada[0], "কাঁদা", "Expected 'কাঁদা' for 'k^ada'");

    // 3. Dictionary vs Non-Dictionary Ranking Fix
    let (cands_laglo, _) = sugg.suggest("laglo", true, true, &empty_memory);
    assert_eq!(cands_laglo[0], "লাগল", "Expected 'লাগল' for 'laglo'");

    let (cands_boiti, _) = sugg.suggest("boiti", true, true, &empty_memory);
    assert_eq!(cands_boiti[0], "বইটি", "Expected 'বইটি' for 'boiti'");

    let (cands_tomake, _) = sugg.suggest("tomake", true, true, &empty_memory);
    assert_eq!(cands_tomake[0], "তোমাকে", "Expected 'তোমাকে' for 'tomake'");

    let (cands_t_omake, _) = sugg.suggest("tOmake", true, true, &empty_memory);
    assert_eq!(cands_t_omake[0], "তোমাকে", "Expected 'তোমাকে' for 'tOmake'");

    let (cands_tomar, _) = sugg.suggest("tomar", true, true, &empty_memory);
    assert_eq!(cands_tomar[0], "তোমার", "Expected 'তোমার' for 'tomar'");

    let (cands_t_omar, _) = sugg.suggest("tOmar", true, true, &empty_memory);
    assert_eq!(cands_t_omar[0], "তোমার", "Expected 'তোমার' for 'tOmar'");

    let (cands_hete, _) = sugg.suggest("hete", true, true, &empty_memory);
    assert_eq!(cands_hete[0], "হেঁটে", "Expected 'হেঁটে' for 'hete'");
}

#[test]
fn test_dirgho_u_and_vowel_kar_variations() {
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
    let empty_memory = HashMap::new();

    let (cands_ku_upper, _) = sugg.suggest("kU", true, true, &empty_memory);
    assert_eq!(cands_ku_upper[0], "কূ", "Expected 'কূ' for 'kU'");

    let (cands_ku_backtick, _) = sugg.suggest("kU`", true, true, &empty_memory);
    assert_eq!(cands_ku_backtick[0], "কূ", "Expected 'কূ' for 'kU`'");

    let (cands_koo_backtick, _) = sugg.suggest("koo`", true, true, &empty_memory);
    assert_eq!(cands_koo_backtick[0], "কূ", "Expected 'কূ' for 'koo`'");

    let (cands_ku_lower, _) = sugg.suggest("ku", true, true, &empty_memory);
    assert_eq!(cands_ku_lower[0], "কু", "Expected 'কু' for 'ku'");

    let (cands_lu, _) = sugg.suggest("lu", true, true, &empty_memory);
    assert_eq!(cands_lu[0], "লু", "Expected 'লু' for 'lu'");

    let (cands_lu_upper, _) = sugg.suggest("lU", true, true, &empty_memory);
    assert_eq!(cands_lu_upper[0], "লূ", "Expected 'লূ' for 'lU'");

    let (cands_crri, _) = sugg.suggest("crri", true, true, &empty_memory);
    assert_eq!(cands_crri[0], "চৃ", "Expected 'চৃ' for 'crri'");

    let (cands_krri, _) = sugg.suggest("krri", true, true, &empty_memory);
    assert_eq!(cands_krri[0], "কৃ", "Expected 'কৃ' for 'krri'");

    let (cands_mu, _) = sugg.suggest("mU", true, true, &empty_memory);
    assert_eq!(cands_mu[0], "মূ", "Expected 'মূ' for 'mU'");

    let (cands_bhu, _) = sugg.suggest("bhU", true, true, &empty_memory);
    assert_eq!(cands_bhu[0], "ভূ", "Expected 'ভূ' for 'bhU'");

    let (cands_dhu, _) = sugg.suggest("dhU", true, true, &empty_memory);
    assert_eq!(cands_dhu[0], "ধূ", "Expected 'ধূ' for 'dhU'");

    let (cands_ru, _) = sugg.suggest("rU", true, true, &empty_memory);
    assert_eq!(cands_ru[0], "রূ", "Expected 'রূ' for 'rU'");

    let (cands_shu, _) = sugg.suggest("shU", true, true, &empty_memory);
    assert_eq!(cands_shu[0], "শূ", "Expected 'শূ' for 'shU'");

    let (cands_su, _) = sugg.suggest("sU", true, true, &empty_memory);
    assert_eq!(cands_su[0], "সূ", "Expected 'সূ' for 'sU'");
}

#[test]
fn test_daily_and_complex_typing_simulation() {
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
    let empty_memory = HashMap::new();

    let test_cases: &[(&str, &str)] = &[
        // 1. Daily Conversational Verbs & Greetings
        ("kemon", "কেমন"),
        ("acho", "আছো"),
        ("achen", "আছেন"),
        ("tumi", "তুমি"),
        ("apni", "আপনি"),
        ("amra", "আমরা"),
        ("shobai", "সবাই"),
        ("korcho", "করছো"),
        ("kortesi", "করছি"), // or করতেছি
        ("jaitasi", "যাচ্ছি"), // or যাইতেছি
        ("esho", "এসো"),
        ("dekho", "দেখো"),
        ("dekhtesilam", "দেখতেছিলাম"),
        ("jaitesilen", "যাইতেছিলেন"),
        ("cholo", "চলো"),
        ("paro", "পারো"),
        ("chole", "চলে"),
        ("jabo", "যাব"),
        ("bolo", "বলো"),
        ("khabo", "খাব"),
        ("ashbo", "আসব"),
        ("shunbo", "শুনব"),
        ("bhalobasha", "ভালোবাসা"),
        ("bhalobashi", "ভালোবাসি"),
        // 2. Inflected Daily Nouns & Adverbs
        ("deshe", "দেশে"),
        ("desher", "দেশের"),
        ("dine", "দিনে"),
        ("diner", "দিনের"),
        ("ekhane", "এখানে"),
        ("shekhane", "সেখানে"),
        ("kothay", "কোথায়"),
        ("kothao", "কোথাও"),
        ("bhabe", "ভাবে"),
        ("garite", "গাড়িতে"),
        ("garir", "গাড়ির"),
        ("garita", "গাড়িটা"),
        ("boita", "বইটা"),
        ("boigulo", "বইগুলো"),
        ("somoymoto", "সময়মতো"),
        ("chobita", "ছবিটা"),
        ("bari", "বাড়ি"),
        ("gari", "গাড়ি"),
        ("thik", "ঠিক"),
        ("ektu", "একটু"),
        ("shob", "সব"),
        ("kichutei", "কিছুতেই"),
        ("raate", "রাতে"),
        ("dur", "দূর"),
        ("dUr", "দূর"),
        ("dure", "দূরে"),
        ("dUre", "দূরে"),
        // 3. Complex Sanskrit / Ha-Conjuncts & Clitics
        ("ahban", "আহ্বান"),
        ("jihba", "জিহ্বা"),
        ("chinho", "চিহ্ন"),
        ("apranho", "অপরাহ্ন"),
        ("madhyanho", "মধ্যাহ্ন"),
        ("shayanho", "সায়াহ্ন"),
        ("hritpindo", "হৃৎপিণ্ড"),
        ("totto", "তত্ত্ব"),
        ("shatto", "স্বত্ব"),
        ("daridro", "দারিদ্র্য"),
        ("mahityo", "মাহাত্ম্য"),
        ("ucchash", "উচ্ছ্বাস"),
        ("ucchshinkhol", "উচ্ছৃঙ্খল"),
        ("protiddhoni", "প্রতিধ্বনি"),
        ("oshtomashchorjo", "অষ্টমাশ্চর্য"),
        ("durjogpurno", "দুর্যোগপূর্ণ"),
        ("shottadhikari", "স্বত্বাধিকারী"),
        ("shadhincheta", "স্বাধীনচেতা"),
        ("chhatrochhatriderkeo", "ছাত্রছাত্রীদেরকেও"),
        ("shomajkormidero", "সমাজকর্মীদেরও"),
        ("chikitshokderke", "চিকিৎসকদেরকে"),
        ("dhai", "\u{0986}\u{09DC}\u{09BE}\u{0987}"),
        ("shoa", "\u{09B8}\u{09CB}\u{09DF}\u{09BE}"),
        ("shadhe", "\u{09B8}\u{09BE}\u{09DC}\u{09C7}"),
        // 4. Modern Technical Loanwords + Inflections
        ("computer", "কম্পিউটার"),
        ("computere", "কম্পিউটারে"),
        ("file", "ফাইল"),
        ("fileti", "ফাইলটি"),
        ("passwordti", "পাসওয়ার্ডটি"),
        ("accountti", "অ্যাকাউন্টটি"),
        ("applicationta", "অ্যাপ্লিকেশনটা"),
        ("smartphonete", "স্মার্টফোনে"),
        ("updatee", "আপডেটেই"),
        ("developer", "ডেভেলপার"),
        ("screenshot", "স্ক্রিনশট"),
        ("notification", "নোটিফিকেশন"),
    ];

    let mut passed = 0;
    let mut failures = Vec::new();

    for &(input, expected) in test_cases {
        let (cands, _) = sugg.suggest(input, true, true, &empty_memory);
        if cands.is_empty() {
            failures.push((input, expected, "No candidates returned".to_string()));
        } else if cands[0] == expected || cands.contains(&expected.to_string()) {
            passed += 1;
        } else {
            failures.push((input, expected, cands[0].clone()));
        }
    }

    if !failures.is_empty() {
        eprintln!(
            "\n=== Simulation Failures ({} / {}) ===",
            failures.len(),
            test_cases.len()
        );
        for (inp, exp, got) in &failures {
            eprintln!(
                "Input: {:<20} Expected: {:<20} Got Top 1: {}",
                inp, exp, got
            );
        }
    }

    assert!(
        failures.is_empty(),
        "Failed {} out of {} simulation test cases",
        failures.len(),
        test_cases.len()
    );
    println!(
        "Simulation Passed: {}/{} (100% accuracy)",
        passed,
        test_cases.len()
    );
}

#[test]
fn test_vowel_and_kar_candidates() {
    let mut sugg = PhoneticSuggestion::new();
    let empty_memory = HashMap::new();

    let (cands_a, _) = sugg.suggest("a", true, true, &empty_memory);
    assert_eq!(cands_a[0], "আ");
    assert_eq!(cands_a[1], "া");

    let (cands_i, _) = sugg.suggest("i", true, true, &empty_memory);
    assert_eq!(cands_i[0], "ই");
    assert!(cands_i.contains(&"ি".to_string()));

    let (cands_u, _) = sugg.suggest("u", true, true, &empty_memory);
    assert_eq!(cands_u[0], "উ");
    assert!(cands_u.contains(&"ু".to_string()));

    let (cands_e, _) = sugg.suggest("e", true, true, &empty_memory);
    assert_eq!(cands_e[0], "এ");
    assert_eq!(cands_e[1], "ে");

    let (cands_o, _) = sugg.suggest("o", true, true, &empty_memory);
    assert_eq!(cands_o[0], "ও");
    assert!(cands_o.contains(&"ো".to_string()));

    let (cands_oi, _) = sugg.suggest("oi", true, true, &empty_memory);
    assert!(cands_oi.contains(&"ৈ".to_string()));

    let (cands_ou, _) = sugg.suggest("ou", true, true, &empty_memory);
    assert!(cands_ou.contains(&"ৌ".to_string()));
}

#[test]
fn test_casual_banglish_and_colloquial_verbs() {
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
    let empty_memory = HashMap::new();

    // 1. Banglish Chat Shorthand
    let (cands_amr, _) = sugg.suggest("amr", true, true, &empty_memory);
    assert_eq!(cands_amr[0], "আমার", "Expected 'আমার' for 'amr'");

    let (cands_tmr, _) = sugg.suggest("tmr", true, true, &empty_memory);
    assert_eq!(cands_tmr[0], "তোমার", "Expected 'তোমার' for 'tmr'");

    let (cands_apnr, _) = sugg.suggest("apnr", true, true, &empty_memory);
    assert_eq!(cands_apnr[0], "আপনার", "Expected 'আপনার' for 'apnr'");

    let (cands_ekhn, _) = sugg.suggest("ekhn", true, true, &empty_memory);
    assert_eq!(cands_ekhn[0], "এখন", "Expected 'এখন' for 'ekhn'");

    let (cands_kno, _) = sugg.suggest("kno", true, true, &empty_memory);
    assert_eq!(cands_kno[0], "কেন", "Expected 'কেন' for 'kno'");

    let (cands_valo, _) = sugg.suggest("valo", true, true, &empty_memory);
    assert_eq!(cands_valo[0], "ভালো", "Expected 'ভালো' for 'valo'");

    let (cands_drkr, _) = sugg.suggest("drkr", true, true, &empty_memory);
    assert_eq!(cands_drkr[0], "দরকার", "Expected 'দরকার' for 'drkr'");

    // 2. Spoken and Colloquial Verbs
    let (cands_korsi, _) = sugg.suggest("korsi", true, true, &empty_memory);
    assert!(cands_korsi[0] == "করছি" || cands_korsi.contains(&"করছি".to_string()));

    let (cands_kortasi, _) = sugg.suggest("kortasi", true, true, &empty_memory);
    assert!(cands_kortasi.contains(&"করছি".to_string()) || cands_kortasi.contains(&"করতেছি".to_string()));

    let (cands_korsilam, _) = sugg.suggest("korsilam", true, true, &empty_memory);
    assert!(cands_korsilam.contains(&"করছিলাম".to_string()) || cands_korsilam.contains(&"করেছিলাম".to_string()));

    let (cands_jamu, _) = sugg.suggest("jamu", true, true, &empty_memory);
    assert!(cands_jamu.contains(&"যাব".to_string()));

    let (cands_khamu, _) = sugg.suggest("khamu", true, true, &empty_memory);
    assert!(cands_khamu.contains(&"খাব".to_string()));

    // 3. Exact Backtick and Explicit Case Veto (writing experience preservation)
    let (cands_amr_backtick, _) = sugg.suggest("amr`", true, true, &empty_memory);
    assert_ne!(cands_amr_backtick[0], "আমার", "Backtick must veto casual shorthand");

    // Standard Avro remains 100% faithful
    let (cands_bhalo, _) = sugg.suggest("bhalo", true, true, &empty_memory);
    assert_eq!(cands_bhalo[0], "ভালো");

    let (cands_amar, _) = sugg.suggest("amar", true, true, &empty_memory);
    assert_eq!(cands_amar[0], "আমার");
}
