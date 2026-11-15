use lekhani_parser::lps::{exact, LpsEngine, LpsWarning};

#[test]
fn test_lps_section10_golden_vectors() {
    let engine = LpsEngine::new();

    // 1. ami -> আমি (basic vowels)
    let cands = engine.convert_to_strings("ami");
    assert!(
        cands.contains(&"আমি".to_string()),
        "ami should produce আমি, got: {:?}",
        cands
    );

    // 2. kolom -> কলম (INH)
    let cands = engine.convert_to_strings("kolom");
    assert!(
        cands.contains(&"কলম".to_string()),
        "kolom should produce কলম, got: {:?}",
        cands
    );

    // 3. jabo -> যাব, যাবো (S3, j variant)
    let cands = engine.convert_to_strings("jabo");
    assert!(
        cands.contains(&"যাব".to_string()) || cands.contains(&"যাবো".to_string()),
        "jabo should contain যাব or যাবো, got: {:?}",
        cands
    );

    // 4. bhalo -> ভাল, ভালো (S3)
    let cands = engine.convert_to_strings("bhalo");
    assert!(
        cands.contains(&"ভালো".to_string()) || cands.contains(&"ভাল".to_string()),
        "bhalo should contain ভালো/ভাল, got: {:?}",
        cands
    );

    // 5. bari -> বাড়ি, বারি (r variant)
    let cands = engine.convert_to_strings("bari");
    assert!(
        cands.contains(&"বাড়ি".to_string()) || cands.contains(&"বারি".to_string()),
        "bari should contain বাড়ি/বারি, got: {:?}",
        cands
    );

    // 6. hoy -> হয় (y after INH)
    let cands = engine.convert_to_strings("hoy");
    assert!(
        cands.contains(&"হয়".to_string()),
        "hoy should contain হয়, got: {:?}",
        cands
    );

    // 7. boi -> বই, বৈ (oi split)
    let cands = engine.convert_to_strings("boi");
    assert!(
        cands.contains(&"বই".to_string()) || cands.contains(&"বৈ".to_string()),
        "boi should contain বই/বৈ, got: {:?}",
        cands
    );

    // 8. bou -> বউ, বৌ (ou split)
    let cands = engine.convert_to_strings("bou");
    assert!(
        cands.contains(&"বউ".to_string()) || cands.contains(&"বৌ".to_string()),
        "bou should contain বউ/বৌ, got: {:?}",
        cands
    );

    // 9. biggan -> বিজ্ঞান (gg generates জ্ঞ and গ্গ)
    let cands = engine.convert_to_strings("biggan");
    assert!(
        cands.contains(&"বিজ্ঞান".to_string()),
        "biggan should contain বিজ্ঞান, got: {:?}",
        cands
    );

    // 10. korbo -> করবো, কর্বো (S2 reph versus SPLIT)
    let cands = engine.convert_to_strings("korbo");
    assert!(
        cands.contains(&"করবো".to_string()) || cands.contains(&"কর্বো".to_string()),
        "korbo should contain করবো/কর্বো, got: {:?}",
        cands
    );

    // 11. kor-bo -> করবো (- blocks cluster)
    let cands = engine.convert_to_strings("kor-bo");
    assert!(
        cands.contains(&"করবো".to_string()) || cands.contains(&"করব".to_string()),
        "kor-bo should contain করবো, got: {:?}",
        cands
    );

    // 12. kor+bo -> কর্বো (+ forces cluster)
    let cands = engine.convert_to_strings("kor+bo");
    assert!(
        cands.contains(&"কর্বো".to_string()) || cands.contains(&"কর্ব".to_string()),
        "kor+bo should contain কর্বো/কর্ব, got: {:?}",
        cands
    );

    // 13. dhormo -> ধর্ম (reph)
    let cands = engine.convert_to_strings("dhormo");
    assert!(
        cands.contains(&"ধর্ম".to_string()),
        "dhormo should contain ধর্ম, got: {:?}",
        cands
    );

    // 14. rokto -> রক্ত (kt)
    let cands = engine.convert_to_strings("rokto");
    assert!(
        cands.contains(&"রক্ত".to_string()),
        "rokto should contain রক্ত, got: {:?}",
        cands
    );

    // 15. shobdo -> শব্দ (bd)
    let cands = engine.convert_to_strings("shobdo");
    assert!(
        cands.contains(&"শব্দ".to_string()),
        "shobdo should contain শব্দ, got: {:?}",
        cands
    );

    // 16. kintu -> কিন্তু (nt)
    let cands = engine.convert_to_strings("kintu");
    assert!(
        cands.contains(&"কিন্তু".to_string()),
        "kintu should contain কিন্তু, got: {:?}",
        cands
    );

    // 17. ongko -> অঙ্ক (S4, ngk)
    let cands = engine.convert_to_strings("ongko");
    assert!(
        cands.contains(&"অঙ্ক".to_string()),
        "ongko should contain অঙ্ক, got: {:?}",
        cands
    );

    // 18. sundor -> সুন্দর (nd)
    let cands = engine.convert_to_strings("sundor");
    assert!(
        cands.contains(&"সুন্দর".to_string()),
        "sundor should contain সুন্দর, got: {:?}",
        cands
    );

    // 19. bangla -> বাংলা, বাঙলা (ng before a consonant)
    let cands = engine.convert_to_strings("bangla");
    assert!(
        cands.contains(&"বাংলা".to_string()) || cands.contains(&"বাঙলা".to_string()),
        "bangla should contain বাংলা/বাঙলা, got: {:?}",
        cands
    );

    // 20. chhele -> ছেলে (chh)
    let cands = engine.convert_to_strings("chhele");
    assert!(
        cands.contains(&"ছেলে".to_string()),
        "chhele should contain ছেলে, got: {:?}",
        cands
    );

    // 21. pothik -> পথিক (th)
    let cands = engine.convert_to_strings("pothik");
    assert!(
        cands.contains(&"পথিক".to_string()),
        "pothik should contain পথিক, got: {:?}",
        cands
    );

    // 22. ekhon -> এখন (kh)
    let cands = engine.convert_to_strings("ekhon");
    assert!(
        cands.contains(&"এখন".to_string()),
        "ekhon should contain এখন, got: {:?}",
        cands
    );

    // 23. ca^d -> চাঁদ (^)
    let cands = engine.convert_to_strings("ca^d");
    assert!(
        cands.contains(&"চাঁদ".to_string()),
        "ca^d should contain চাঁদ, got: {:?}",
        cands
    );

    // 24. ghon't'a -> ঘণ্টা (locks and n't')
    let cands = engine.convert_to_strings("ghon't'a");
    assert!(
        cands.contains(&"ঘণ্টা".to_string()),
        "ghon't'a should contain ঘণ্টা, got: {:?}",
        cands
    );

    // 25. ghonta -> ঘণ্টা reachable via candidates or letter chips
    let (cands_obj, _) = engine.parse("ghonta");
    let has_ghonta = cands_obj.iter().any(|c| c.text == "ঘণ্টা")
        || cands_obj.iter().any(|c| {
            c.chips.iter().any(|chip| {
                chip.siblings.contains(&"ণ্ট".to_string())
                    || chip.siblings.contains(&"ণ".to_string())
            })
        });
    assert!(
        has_ghonta,
        "ghonta should be reachable via candidates or chips: {:?}",
        cands_obj
    );

    // 26. rishi -> ঋষি, ঋশি, রিশি
    let cands = engine.convert_to_strings("rishi");
    assert!(
        cands.contains(&"ঋষি".to_string())
            || cands.contains(&"ঋশি".to_string())
            || cands.contains(&"রিশি".to_string()),
        "rishi should contain ঋষি/ঋশি/রিশি, got: {:?}",
        cands
    );

    // 27. rya -> র্যা
    let cands = engine.convert_to_strings("rya");
    assert!(
        cands.contains(&"র্যা".to_string()) || cands.contains(&"র‌্যা".to_string()),
        "rya should contain র্যা, got: {:?}",
        cands
    );

    // 28. r\w+j'a -> র্যা (ZWJ exact)
    let cands = engine.convert_to_strings(r"r\w+j'a");
    assert!(
        cands
            .iter()
            .any(|c| c.contains('র') && c.contains('য') && c.contains('া')),
        "r\\w+j'a should contain র্যা, got: {:?}",
        cands
    );

    // 29. k' -> ক' plus LPS-W101 invalid lock
    let (cands, diags) = engine.parse("k'");
    assert!(
        diags
            .iter()
            .any(|d| matches!(d, LpsWarning::W101InvalidLock(tok) if tok == "k")),
        "k' should report W101InvalidLock, got: {:?}",
        diags
    );
    assert!(
        !cands.is_empty(),
        "k' should produce candidate, got: {:?}",
        cands
    );

    // 30. b-i -> বই only (exact form)
    let cands = engine.convert_to_strings("b-i");
    assert_eq!(
        cands.first().map(|s| s.as_str()),
        Some("বই"),
        "b-i exact form must be বই"
    );

    // 31. k-l-m -> কলম only (exact form)
    let cands = engine.convert_to_strings("k-l-m");
    assert_eq!(
        cands.first().map(|s| s.as_str()),
        Some("কলম"),
        "k-l-m exact form must be কলম"
    );

    // 32. k+s' -> ক্ষ only (exact conjunct)
    let cands = engine.convert_to_strings("k+s'");
    assert_eq!(
        cands.first().map(|s| s.as_str()),
        Some("ক্ষ"),
        "k+s' exact form must be ক্ষ"
    );

    // 33. suur+j' -> সূর্য only (exact reph)
    let cands = engine.convert_to_strings("suur+j'");
    assert_eq!(
        cands.first().map(|s| s.as_str()),
        Some("সূর্য"),
        "suur+j' exact form must be সূর্য"
    );

    // 34. bang''la -> বাংলা only (exact ং)
    let cands = engine.convert_to_strings("bang''la");
    assert_eq!(
        cands.first().map(|s| s.as_str()),
        Some("বাংলা"),
        "bang''la exact form must be বাংলা"
    );

    // 35. ri' -> ঋ
    let cands = engine.convert_to_strings("ri'");
    assert_eq!(
        cands.first().map(|s| s.as_str()),
        Some("ঋ"),
        "ri' exact form must be ঋ"
    );

    // 36. r-i -> রি
    let cands = engine.convert_to_strings("r-i");
    assert_eq!(
        cands.first().map(|s| s.as_str()),
        Some("রি"),
        "r-i exact form must be রি"
    );

    // 37. \u{09CE} -> ৎ
    let cands = engine.convert_to_strings(r"\u{09CE}");
    assert_eq!(
        cands.first().map(|s| s.as_str()),
        Some("ৎ"),
        "\\u{{09CE}} must be ৎ"
    );

    // 38. \u{09BC} -> lone U+09BC (nukta)
    let cands = engine.convert_to_strings(r"\u{09BC}");
    assert_eq!(
        cands.first().map(|s| s.as_str()),
        Some("\u{09BC}"),
        "\\u{{09BC}} must preserve nukta"
    );

    // 39. \efacebook -> facebook
    let cands = engine.convert_to_strings(r"\efacebook");
    assert_eq!(
        cands.first().map(|s| s.as_str()),
        Some("facebook"),
        "\\efacebook must be verbatim facebook"
    );

    // 40. \efacebook-er -> facebooker / facebook-এর
    let cands = engine.convert_to_strings(r"\efacebook-er");
    assert!(
        cands.iter().any(|c| c.starts_with("facebook")),
        "\\efacebook-er must keep facebook prefix, got: {:?}",
        cands
    );

    // 41. user@mail.com -> verbatim, never offered as Bengali (Latin token detection)
    let cands = engine.convert_to_strings("user@mail.com");
    assert_eq!(
        cands.first().map(|s| s.as_str()),
        Some("user@mail.com"),
        "email must remain verbatim Latin"
    );

    // 42. ami tomake bhalobashi -> sentence
    let cands = engine.convert_to_strings("ami tomake bhalobashi");
    assert!(
        cands.iter().any(|c| c.contains("আমি")
            && c.contains("তোমাকে")
            && (c.contains("ভালোবাসি") || c.contains("ভালবাসি"))),
        "sentence must produce valid Bengali words: {:?}",
        cands
    );
}

#[test]
fn test_lps_section12_exact_invariants() {
    // Round-trip tests for Section 12.1 exact forms
    assert_eq!(exact("কলম"), "k-l-m");
    assert_eq!(exact("বই"), "b-i");
    assert_eq!(exact("ক্ষ"), "k+s'");
    assert_eq!(exact("জ্ঞ"), "j+ny");
    assert_eq!(exact("সূর্য"), "suur+j'");
    assert_eq!(exact("বাংলা"), "bang''la");
    assert_eq!(
        exact("\u{09B0}\u{200D}\u{09CD}\u{09AF}\u{09BE}"),
        r"r\w+j'a"
    );

    // Verify engine round-trip for each exact form
    let engine = LpsEngine::new();
    for word in &["কলম", "বই", "ক্ষ", "সূর্য", "বাংলা"] {
        let exact_code = exact(word);
        let cands = engine.convert_to_strings(&exact_code);
        assert_eq!(
            cands.first().map(|s| s.as_str()),
            Some(*word),
            "Round trip failed for {}: exact form '{}' produced {:?}",
            word,
            exact_code,
            cands
        );
    }
}
