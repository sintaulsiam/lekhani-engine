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
        ("dhai", "আড়াই"),
        ("shoa", "সোয়া"),
        ("shadhe", "সাড়ে"),

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
        eprintln!("\n=== Simulation Failures ({} / {}) ===", failures.len(), test_cases.len());
        for (inp, exp, got) in &failures {
            eprintln!("Input: {:<20} Expected: {:<20} Got Top 1: {}", inp, exp, got);
        }
    }

    assert!(failures.is_empty(), "Failed {} out of {} simulation test cases", failures.len(), test_cases.len());
    println!("Simulation Passed: {}/{} (100% accuracy)", passed, test_cases.len());
}
