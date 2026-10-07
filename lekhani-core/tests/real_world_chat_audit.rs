//! Real-World Chat & Conversational Experience Integration Test

use hashbrown::HashMap;
use lekhani_core::phonetic::PhoneticSuggestion;

#[test]
fn test_real_world_chat_diagnostic_scenarios() {
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

    // 1. [আমি] + "korbo" -> 1st person "করব" / "করবো", NOT 2nd/3rd person "করবে"
    let (cands1, _) = sugg.suggest_with_context("korbo", Some("আমি"), true, true, &empty_memory);
    assert!(
        cands1.first() == Some(&"করব".to_string()) || cands1.first() == Some(&"করবো".to_string()),
        "Expected 'করব' or 'করবো' as Rank 1 after 'আমি', got: {:?}",
        cands1
    );

    // 2. [নাম] + "ki" -> Wh-question "কী"
    let (cands2, _) = sugg.suggest_with_context("ki", Some("নাম"), true, true, &empty_memory);
    assert_eq!(
        cands2.first(),
        Some(&"কী".to_string()),
        "Expected 'কী' after 'নাম', got: {:?}",
        cands2
    );

    // 3. [তোমার] + "moto" -> Postposition "মতো"
    let (cands3, _) = sugg.suggest_with_context("moto", Some("তোমার"), true, true, &empty_memory);
    assert_eq!(
        cands3.first(),
        Some(&"মতো".to_string()),
        "Expected 'মতো' after genitive 'তোমার', got: {:?}",
        cands3
    );

    // 4. [কী] + "hol" -> Top 3 includes "হলো"
    let (cands4, _) = sugg.suggest_with_context("hol", Some("কী"), true, true, &empty_memory);
    assert!(
        cands4.iter().take(3).any(|c| c == "হলো"),
        "Expected Top 3 to contain 'হলো' after 'কী', got: {:?}",
        cands4
    );

    // 5. [কোনো] + "shomossha" -> "সমস্যা"
    let (cands5, _) = sugg.suggest_with_context("shomossha", Some("কোনো"), true, true, &empty_memory);
    assert_eq!(
        cands5.first(),
        Some(&"সমস্যা".to_string()),
        "Expected 'সমস্যা' for 'shomossha', got: {:?}",
        cands5
    );

    // 6. [বাসায়] + "pouchhe" -> "পৌঁছে"
    let (cands6, _) = sugg.suggest_with_context("pouchhe", Some("বাসায়"), true, true, &empty_memory);
    assert_eq!(
        cands6.first(),
        Some(&"পৌঁছে".to_string()),
        "Expected 'পৌঁছে' for 'pouchhe', got: {:?}",
        cands6
    );

    // 7. [আমি] + "r" -> Single-char shorthand "আর"
    let (cands7, _) = sugg.suggest_with_context("r", Some("আমি"), true, true, &empty_memory);
    assert_eq!(
        cands7.first(),
        Some(&"আর".to_string()),
        "Expected 'আর' for shorthand 'r', got: {:?}",
        cands7
    );

    // 8. [ভালো] + "asi" -> "আছি" (S-for-CH resolution)
    let (cands8, _) = sugg.suggest_with_context("asi", Some("ভালো"), true, true, &empty_memory);
    assert_eq!(
        cands8.first(),
        Some(&"আছি".to_string()),
        "Expected 'আছি' after 'ভালো', got: {:?}",
        cands8
    );

    // 9. [কোথায়] + "jaba" -> Colloquial "যাবা"
    let (cands9, _) = sugg.suggest_with_context("jaba", Some("কোথায়"), true, true, &empty_memory);
    assert_eq!(
        cands9.first(),
        Some(&"যাবা".to_string()),
        "Expected 'যাবা' for 'jaba', got: {:?}",
        cands9
    );

    // 10. [সমস্যা] + "nai" -> "নাই"
    let (cands10, _) = sugg.suggest_with_context("nai", Some("সমস্যা"), true, true, &empty_memory);
    assert_eq!(
        cands10.first(),
        Some(&"নাই".to_string()),
        "Expected 'নাই' after 'সমস্যা', got: {:?}",
        cands10
    );

    println!("🎉 All 10/10 Real-World Conversational Chat Scenarios Passed Perfectly!");
}
