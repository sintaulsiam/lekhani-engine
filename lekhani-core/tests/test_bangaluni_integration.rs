use hashbrown::HashMap;
use lekhani_core::PhoneticSuggestion;

#[test]
fn test_all_fixed_words_upstream() {
    std::env::set_var("LEKHANI_MODELS_DIR", "/mnt/data/lekhani-engine/data/dictionaries");

    let mut suggestion = PhoneticSuggestion::new();

    let dict_dir = std::path::PathBuf::from("/mnt/data/lekhani-engine/data/dictionaries");
    if dict_dir.exists() {
        let dict_bin = dict_dir.join("dictionary.bin");
        if dict_bin.exists() {
            let _ = suggestion.load_from_dir_with_dict(&dict_dir, "dictionary.bin");
        } else {
            let _ = suggestion.load_from_dir_with_dict(&dict_dir, "dictionary.json");
        }
    }

    if suggestion.pragyan_engine.is_none() {
        if let Some(engine) = lekhani_core::phonetic::PragyanEngine::probe() {
            suggestion.set_pragyan_engine(engine);
        }
    }

    // Set layout
    let layout_path = std::path::PathBuf::from("/mnt/data/lekhani-engine/data/layouts/avrophonetic.json");
    if layout_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&layout_path) {
            if let Ok(json) = serde_json::from_str(&content) {
                suggestion.set_layout(&json);
            }
        }
    }

    let test_cases = [
        ("okritokarjo", "অকৃতকার্য"),
        ("akritakarjo", "অকৃতকার্য"),
        ("akhandata", "অখণ্ডতা"),
        ("auxfordey", "অক্সফোর্ডে"),
        ("ajuhat", "অজুহাত"),
        ("ateetkale", "অতীতকালে"),
        ("akkharke", "অক্ষরকে"),
        ("rajniti", "রাজনীতি"),
        ("somajseba", "সমাজসেবা"),
        ("deshprem", "দেশপ্রেম"),
        ("anchalsamuha", "অঞ্চলসমূহ"),
        ("antorvuktimulok", "অন্তর্ভুক্তিমূলক"),
        ("oppohoron", "অপহরণ"),
        ("dhonnobad", "ধন্যবাদ"),
        ("bangladesh", "বাংলাদেশ"),
        ("shokti", "শক্তি"),
        ("ottacharer", "অত্যাচারের"),
        ("avatar", "অবতার"),
        ("atyacharita", "অত্যাচারিত"),
        ("advut", "অদ্ভুত"),
        ("amio", "আমিও"),
        ("antarvukta", "অন্তর্ভুক্ত"),
        ("otiiteo", "অতীতেও"),
        ("abatar", "অবতার"),
        ("abadamaner", "অবদমনের"),
        ("agradut", "অগ্রদূত"),
        ("bishoy", "বিষয়"),
        ("dayitto", "দায়িত্ব"),
        ("bisheshbhabe", "বিশেষভাবে"),
    ];

    let empty_mem = HashMap::new();
    for (roman, expected) in test_cases {
        let (candidates, _) = suggestion.suggest(roman, false, true, &empty_mem);
        let top = candidates.first().map(|s| s.as_str()).unwrap_or("<none>");
        assert_eq!(
            top, expected,
            "Failed for '{}': expected '{}', got candidates {:?}",
            roman, expected, candidates
        );
    }
}
