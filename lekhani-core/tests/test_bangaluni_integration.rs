use hashbrown::HashMap;
use lekhani_core::PhoneticSuggestion;

#[test]
fn test_all_fixed_words_upstream() {
    let mut suggestion = PhoneticSuggestion::new();

    let layout_candidates = [
        std::path::Path::new("../../data/layouts/avrophonetic.json"),
        std::path::Path::new("data/layouts/avrophonetic.json"),
        std::path::Path::new("../data/layouts/avrophonetic.json"),
    ];
    for p in layout_candidates {
        if p.exists() {
            if let Ok(content) = std::fs::read_to_string(p) {
                if let Ok(json) = serde_json::from_str(&content) {
                    suggestion.set_layout(&json);
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
            let _ = suggestion.load_from_dir(p);
            break;
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
