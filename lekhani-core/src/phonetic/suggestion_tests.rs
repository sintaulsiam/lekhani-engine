use super::*;

#[test]
fn test_split_word_punct() {
    assert_eq!(split_word_punct("(kotha)"), ("(", "kotha", ")"));
    assert_eq!(split_word_punct("!ami,"), ("!", "ami", ","));
    assert_eq!(split_word_punct("bangla"), ("", "bangla", ""));
}

#[test]
fn test_emoji_and_symbol_suggestions() {
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

    // Emoji shortcode :smile:
    let (cands, _) = sugg.suggest(":smile:", true, true, &empty_memory);
    assert!(!cands.is_empty());
    assert_eq!(cands[0], "😊");

    // Bengali emoji :bhalobasha:
    let (cands, _) = sugg.suggest(":bhalobasha:", true, true, &empty_memory);
    assert!(!cands.is_empty());
    assert_eq!(cands[0], "❤️");

    // Taka symbol *taka*
    let (cands, _) = sugg.suggest("*taka*", true, true, &empty_memory);
    assert!(!cands.is_empty());
    assert_eq!(cands[0], "৳");

    // Smiley :)
    let (cands, _) = sugg.suggest(":)", true, true, &empty_memory);
    assert!(!cands.is_empty());
    assert_eq!(cands[0], "😊");

    // Snippet !shubhechha
    let (cands, _) = sugg.suggest("!shubhechha", true, true, &empty_memory);
    assert!(!cands.is_empty());
    assert_eq!(cands[0], "আন্তরিক শুভেচ্ছা ও অভিনন্দন");

    // Prefix emoji search :sm
    let (cands_prefix, _) = sugg.suggest(":sm", true, true, &empty_memory);
    assert!(!cands_prefix.is_empty());
    assert!(
        cands_prefix.contains(&"😊".to_string()) || cands_prefix.contains(&"😃".to_string())
    );

    // Prefix symbol search *t
    let (cands_sym, _) = sugg.suggest("*t", true, true, &empty_memory);
    assert!(!cands_sym.is_empty());
    assert_eq!(cands_sym[0], "৳");

    // Inline Math Formula =125*8
    let (cands_math, _) = sugg.suggest("=125*8", true, true, &empty_memory);
    assert!(!cands_math.is_empty());
    assert_eq!(cands_math[0], "১,০০০");
    assert_eq!(cands_math[1], "1,000");

    // Currency Conversion #usd50
    let (cands_curr, _) = sugg.suggest("#usd50", true, true, &empty_memory);
    assert!(!cands_curr.is_empty());
    assert!(cands_curr[0].contains("৳"));

    // Snippet prefix search !sh
    let (cands_snip_prefix, _) = sugg.suggest("!sh", true, true, &empty_memory);
    assert!(!cands_snip_prefix.is_empty());
    assert!(cands_snip_prefix.contains(&"আন্তরিক শুভেচ্ছা ও অভিনন্দন".to_string()));
    assert!(cands_snip_prefix.contains(&"স্বাগতম".to_string()));

    // Bengali symbol aliases *chandrabindu*, *khandata*, $$
    let (cands_chandra, _) = sugg.suggest("*chandrabindu*", true, true, &empty_memory);
    assert_eq!(cands_chandra[0], "ঁ");

    let (cands_khanda, _) = sugg.suggest("*khandata*", true, true, &empty_memory);
    assert_eq!(cands_khanda[0], "ৎ");

    let (cands_dollar, _) = sugg.suggest("$$", true, true, &empty_memory);
    assert_eq!(cands_dollar[0], "৳");

    // Transliterate phrase or sentence
    let res_sentence = sugg.transliterate_phrase_or_sentence("!shubhechha *taka* =125*8");
    assert!(res_sentence.contains("আন্তরিক শুভেচ্ছা ও অভিনন্দন"));
    assert!(res_sentence.contains('৳'));
    assert!(res_sentence.contains("১,০০০"));

    // Common Bengali words & candidate ranking
    let (cands_gari, _) = sugg.suggest("gari", true, true, &empty_memory);
    assert!(cands_gari.contains(&"গাড়ি".to_string()));
    let (cands_ga_ri, _) = sugg.suggest("gaRi", true, true, &empty_memory);
    assert_eq!(cands_ga_ri[0], "গাড়ি");

    let (cands_valo, _) = sugg.suggest("valo", true, true, &empty_memory);
    assert!(cands_valo.contains(&"ভালো".to_string()));
    let (cands_val_o, _) = sugg.suggest("valO", true, true, &empty_memory);
    assert_eq!(cands_val_o[0], "ভালো");

    let (cands_sundor, _) = sugg.suggest("sundor", true, true, &empty_memory);
    assert_eq!(cands_sundor[0], "সুন্দর");

    let (cands_tomake, _) = sugg.suggest("tomake", true, true, &empty_memory);
    assert!(cands_tomake.contains(&"তোমাকে".to_string()));
    let (cands_t_o_make, _) = sugg.suggest("tOmake", true, true, &empty_memory);
    assert_eq!(cands_t_o_make[0], "তোমাকে");

    let (cands_bhalo, _) = sugg.suggest("bhalobashi", true, true, &empty_memory);
    assert!(cands_bhalo.contains(&"ভালোবাসি".to_string()));
    let (cands_bhal_o, _) = sugg.suggest("bhalObasi", true, true, &empty_memory);
    assert_eq!(cands_bhal_o[0], "ভালোবাসি");

    // Transliterate full natural sentences with emojis
    let res_love = sugg.transliterate_phrase_or_sentence("ami tomake bhalobashi :heart:");
    assert_eq!(res_love, "আমি তোমাকে ভালোবাসি ❤️");

    let res_car = sugg.transliterate_phrase_or_sentence("gaRi diye bari jabo");
    assert!(res_car.contains("গাড়ি"));
}

#[test]
fn test_bilingual_keyword_emoji_suggestions() {
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

    // 1. Bengali word "cha" should suggest "চা" at top, and "☕" in trailing candidates
    let (cands_cha, _) = sugg.suggest("cha", true, true, &empty_memory);
    assert_eq!(cands_cha[0], "চা");
    assert!(cands_cha.contains(&"☕".to_string()));

    // 2. Bengali word "bhalobasha" should suggest "ভালোবাসা" at top, and "❤️" in trailing candidates
    let (cands_love, _) = sugg.suggest("bhalobasha", true, true, &empty_memory);
    assert_eq!(cands_love[0], "ভালোবাসা");
    assert!(cands_love.contains(&"❤️".to_string()));

    // 3. English word "coffee" / "tea" should suggest "☕"
    let (cands_coffee, _) = sugg.suggest("coffee", true, true, &empty_memory);
    assert!(cands_coffee.contains(&"☕".to_string()));

    // 4. English word "dog" / "cat" should suggest emojis
    let (cands_dog, _) = sugg.suggest("dog", true, true, &empty_memory);
    assert!(cands_dog.contains(&"🐶".to_string()));

    let (cands_cat, _) = sugg.suggest("cat", true, true, &empty_memory);
    assert!(cands_cat.contains(&"🐱".to_string()));
}

#[test]
fn test_candidate_relevance() {
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

    // 1. "ami" should produce natural inflections and not dictionary words like "আমিষ" / "আমিন"
    let (cands_ami, _) = sugg.suggest("ami", true, true, &empty_memory);
    assert_eq!(cands_ami[0], "আমি");
    assert!(cands_ami.contains(&"আমাকে".to_string()));
    assert!(cands_ami.contains(&"আমার".to_string()));
    assert!(cands_ami.contains(&"আমাদের".to_string()));
    assert!(!cands_ami.contains(&"আমিষ".to_string()));
    assert!(!cands_ami.contains(&"আমিন".to_string()));

    // 2. "gari" should produce vehicle emoji and inflections
    let (cands_gari, _) = sugg.suggest("gari", true, true, &empty_memory);
    assert!(cands_gari.contains(&"গাড়ি".to_string()));
    assert!(cands_gari.contains(&"🚗".to_string()));
    assert!(
        cands_gari.contains(&"গাড়িতে".to_string()) || cands_gari.contains(&"গাড়ির".to_string())
    );

    // 3. "taka" should suggest Taka symbol and "টাকা"
    let (cands_taka, _) = sugg.suggest("taka", true, true, &empty_memory);
    assert!(cands_taka.contains(&"টাকা".to_string()));
    assert!(cands_taka.contains(&"৳".to_string()));
    let (cands_ta_ka, _) = sugg.suggest("Taka", true, true, &empty_memory);
    assert_eq!(cands_ta_ka[0], "টাকা");

    // 4. "shob" should produce "সব", "শব", "সবাই"
    let (cands_shob, _) = sugg.suggest("shob", true, true, &empty_memory);
    assert!(cands_shob.contains(&"সব".to_string()));
    assert!(cands_shob.contains(&"শব".to_string()));
    assert!(cands_shob.contains(&"সবাই".to_string()));
}

#[test]
fn test_contextual_homophones() {
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

    // NOTE: Strict rank-1 ordering for contextual homophones requires the
    // corpus-trained LM (Fix 4B: `lekhani dev train`). With only the 383-word
    // static table, পরা’s +3500 exact-match bonus (Avro directly maps "pora"→পরা)
    // narrowly outweighs the bigram LM edge for পড়া. We assert the weaker but
    // still linguistically correct property: পড়া must appear in candidates,
    // and the শার্ট context must still produce পরা at rank-1 (that one is robust).
    // When previous word is "বই", "পড়া" must be rank 1
    let (cands_book, _) =
        sugg.suggest_with_context("pora", Some("বই"), true, true, &empty_memory);
    println!("cands_book: {:?}", cands_book);
    assert_eq!(
        cands_book[0], "পড়া",
        "With context বই, পড়া must be ranked #1; got {:?}",
        cands_book
    );

    // When previous word is "শার্ট", "pora" should rank "পরা" first (robust even without trained LM)
    let (cands_shirt, _) =
        sugg.suggest_with_context("pora", Some("শার্ট"), true, true, &empty_memory);
    assert_eq!(cands_shirt[0], "পরা");

    // When previous word is "বাংলা", "bhasha" ranks "ভাষা"
    let (cands_lang, _) =
        sugg.suggest_with_context("bhasha", Some("বাংলা"), true, true, &empty_memory);
    assert_eq!(cands_lang[0], "ভাষা");
}

#[test]
fn test_next_word_predictions() {
    let sugg = PhoneticSuggestion::new();
    let next_ami = sugg.suggest_next_words("আমি");
    assert!(!next_ami.is_empty());
    assert!(
        next_ami.contains(&"মনে".to_string())
            || next_ami.contains(&"জানি".to_string())
            || next_ami.contains(&"আমার".to_string())
            || next_ami.contains(&"ভালো".to_string())
            || next_ami.contains(&"যাচ্ছি".to_string())
            || next_ami.contains(&"তোমাকে".to_string())
    );

    let next_thanks = sugg.suggest_next_words("ধন্যবাদ");
    assert!(!next_thanks.is_empty());
    assert!(
        next_thanks.contains(&"জানান".to_string())
            || next_thanks.contains(&"জানাই".to_string())
            || next_thanks.contains(&"তোমাকে".to_string())
            || next_thanks.contains(&"ভাই".to_string())
            || next_thanks.contains(&"আপনাকে".to_string())
    );
}

#[test]
fn test_bilingual_loanwords() {
    let mut sugg = PhoneticSuggestion::new();
    let empty_memory = HashMap::new();

    let (cands_meet, _) = sugg.suggest("meeting", true, true, &empty_memory);
    assert!(!cands_meet.is_empty());
    assert!(cands_meet.contains(&"মিটিং".to_string()));
    assert!(cands_meet.contains(&"meeting".to_string()));
}

#[test]
fn test_multi_token_context_and_beam_transliteration() {
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

    // 1. Multi-token context suggestion
    let (cands_multi, _) =
        sugg.suggest_with_multi_context("khacchi", &["আমি", "ভাত"], true, true, &empty_memory);
    assert!(!cands_multi.is_empty());
    assert_eq!(cands_multi[0], "খাচ্ছি");

    // 2. Full sentence beam search decoding
    let sentence = sugg.transliterate_phrase_or_sentence("ami banglay gaan gai");
    assert!(sentence.contains("বাংলা") || sentence.contains("বাংলায়"));
    assert!(sentence.contains("গান"));
    assert!(sentence.contains("গাই"));
}

#[test]
fn test_juktoborno_casual_typing() {
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

    // 1. "kosto" accurately suggests "কষ্ট" at Rank #1
    let (cands_kosto, _) = sugg.suggest("kosto", true, true, &empty_memory);
    assert!(!cands_kosto.is_empty());
    assert_eq!(cands_kosto[0], "কষ্ট");

    let (cands_koshto, _) = sugg.suggest("koshto", true, true, &empty_memory);
    assert_eq!(cands_koshto[0], "কষ্ট");
    let (cands_ko_sh_to, _) = sugg.suggest("koShTo", true, true, &empty_memory);
    assert_eq!(cands_ko_sh_to[0], "কষ্ট");

    // 2. "nosto" accurately suggests "নষ্ট" at Rank #1
    let (cands_nosto, _) = sugg.suggest("nosto", true, true, &empty_memory);
    assert!(!cands_nosto.is_empty());
    assert_eq!(cands_nosto[0], "নষ্ট");

    let (cands_noshto, _) = sugg.suggest("noshto", true, true, &empty_memory);
    assert_eq!(cands_noshto[0], "নষ্ট");
    let (cands_no_sh_to, _) = sugg.suggest("noShTo", true, true, &empty_memory);
    assert_eq!(cands_no_sh_to[0], "নষ্ট");

    // 3. "biganni" and "biggani" suggest "বিজ্ঞানী"
    let (cands_biganni, _) = sugg.suggest("biganni", true, true, &empty_memory);
    assert!(cands_biganni.contains(&"বিজ্ঞানী".to_string()));

    let (cands_biggani, _) = sugg.suggest("biggani", true, true, &empty_memory);
    assert!(cands_biggani.contains(&"বিজ্ঞানী".to_string()));

    // 4. "biggan" suggests "বিজ্ঞান"
    let (cands_biggan, _) = sugg.suggest("biggan", true, true, &empty_memory);
    assert!(cands_biggan.contains(&"বিজ্ঞান".to_string()));

    // 5. "sristi" suggests "সৃষ্টি" at Rank #1
    let (cands_sristi, _) = sugg.suggest("sristi", true, true, &empty_memory);
    assert_eq!(cands_sristi[0], "সৃষ্টি");

    let (cands_srish_ti, _) = sugg.suggest("srriShTi", true, true, &empty_memory);
    assert_eq!(cands_srish_ti[0], "সৃষ্টি");

    // 6. "bristi" suggests "বৃষ্টি" at Rank #1
    let (cands_bristi, _) = sugg.suggest("bristi", true, true, &empty_memory);
    assert_eq!(cands_bristi[0], "বৃষ্টি");

    // 7. "bebostha" suggests "ব্যবস্থা" at Rank #1
    let (cands_bebostha, _) = sugg.suggest("bebostha", true, true, &empty_memory);
    assert_eq!(cands_bebostha[0], "ব্যবস্থা");

    // 8. "onusthan" suggests "অনুষ্ঠান" at Rank #1
    let (cands_onusthan, _) = sugg.suggest("onusthan", true, true, &empty_memory);
    assert_eq!(cands_onusthan[0], "অনুষ্ঠান");
}

#[test]
fn test_natural_mistake_tolerance() {
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

    // 1. Elongation collapse suggestions
    let (cands_thik, _) = sugg.suggest("thiiik", true, true, &empty_memory);
    assert!(!cands_thik.is_empty());
    assert!(cands_thik.contains(&"ঠিক".to_string()));

    let (cands_bhalo, _) = sugg.suggest("bhalooo", true, true, &empty_memory);
    assert!(!cands_bhalo.is_empty());
    assert!(cands_bhalo.contains(&"ভালো".to_string()));

    let (cands_na, _) = sugg.suggest("naaa", true, true, &empty_memory);
    assert!(!cands_na.is_empty());
    assert!(cands_na.contains(&"না".to_string()));

    // 2. Vai literal transliteration
    let (cands_vai, _) = sugg.suggest("vai", true, true, &empty_memory);
    assert_eq!(cands_vai[0], "ভাই");

    // 3. Chandrabindu natural spelling suggestions
    let (cands_chad, _) = sugg.suggest("chad", true, true, &empty_memory);
    assert!(cands_chad.contains(&"ছাদ".to_string()) && cands_chad.contains(&"চাঁদ".to_string()));

    let (cands_cad, _) = sugg.suggest("cad", true, true, &empty_memory);
    assert!(cands_cad.contains(&"চাঁদ".to_string()));

    let (cands_c_ad, _) = sugg.suggest("ca^d", true, true, &empty_memory);
    assert_eq!(cands_c_ad[0], "চাঁদ");

    let (cands_dat, _) = sugg.suggest("dat", true, true, &empty_memory);
    assert!(cands_dat.contains(&"দাঁত".to_string()));

    let (cands_has, _) = sugg.suggest("has", true, true, &empty_memory);
    assert!(cands_has.contains(&"হাঁস".to_string()));

    let (cands_bas, _) = sugg.suggest("bas", true, true, &empty_memory);
    assert!(cands_bas.contains(&"বাঁশ".to_string()));

    let (cands_pach, _) = sugg.suggest("pach", true, true, &empty_memory);
    assert!(cands_pach.contains(&"পাঁচ".to_string()));

    // 4. Khanda-Ta suggestions
    let (cands_hothat, _) = sugg.suggest("hothat", true, true, &empty_memory);
    assert!(cands_hothat.contains(&"হঠাৎ".to_string()));

    let (cands_utsob, _) = sugg.suggest("utsob", true, true, &empty_memory);
    assert!(cands_utsob.contains(&"উৎসব".to_string()));

    let (cands_utsaho, _) = sugg.suggest("utsaho", true, true, &empty_memory);
    assert!(cands_utsaho.contains(&"উৎসাহ".to_string()));

    // 5. Bengali Glide Verbs
    let (cands_khawa, _) = sugg.suggest("khawa", true, true, &empty_memory);
    assert!(cands_khawa.contains(&"খাওয়া".to_string()));

    let (cands_dewa, _) = sugg.suggest("dewa", true, true, &empty_memory);
    assert!(cands_dewa.contains(&"দেওয়া".to_string()));

    let (cands_jawa, _) = sugg.suggest("jawa", true, true, &empty_memory);
    assert!(cands_jawa.contains(&"যাওয়া".to_string()));

    let (cands_pawa, _) = sugg.suggest("pawa", true, true, &empty_memory);
    assert!(cands_pawa.contains(&"পাওয়া".to_string()));
}

#[test]
fn test_complex_word_ergonomics_and_standards() {
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

    // 1. Reph Consonants
    let (cands_bortoman, _) = sugg.suggest("bortoman", true, true, &empty_memory);
    assert!(cands_bortoman.contains(&"বর্তমান".to_string()));

    let (cands_orthonoitik, _) = sugg.suggest("orthonoitik", true, true, &empty_memory);
    assert!(cands_orthonoitik.contains(&"অর্থনৈতিক".to_string()));

    let (cands_karjokrom, _) = sugg.suggest("karjokrom", true, true, &empty_memory);
    assert!(cands_karjokrom.contains(&"কার্যক্রম".to_string()));

    let (cands_durniti, _) = sugg.suggest("durniti", true, true, &empty_memory);
    assert!(cands_durniti.contains(&"দুর্নীতি".to_string()));

    // 2. Conjunction 'o'
    let (cands_o, _) = sugg.suggest("o", true, true, &empty_memory);
    assert_eq!(cands_o[0], "ও");

    // 3. Ja-fala & Geminates
    let (cands_tottho, _) = sugg.suggest("tottho", true, true, &empty_memory);
    assert!(cands_tottho.contains(&"তথ্য".to_string()));

    let (cands_biddaloy, _) = sugg.suggest("biddaloy", true, true, &empty_memory);
    assert!(
        cands_biddaloy.contains(&"বিদ্যালয়".to_string())
            || cands_biddaloy.contains(&"বিদ্যালয়".to_string())
    );

    let (cands_jonne, _) = sugg.suggest("jonne", true, true, &empty_memory);
    assert!(
        cands_jonne.contains(&"জন্য".to_string()) || cands_jonne.contains(&"জন্যে".to_string())
    );

    // 4. Motion Verbs & Antastha-Ja
    let (cands_jacchi, _) = sugg.suggest("jacchi", true, true, &empty_memory);
    assert!(cands_jacchi.contains(&"যাচ্ছি".to_string()));

    let (cands_juddho, _) = sugg.suggest("juddho", true, true, &empty_memory);
    assert!(cands_juddho.contains(&"যুদ্ধ".to_string()));

    let (cands_projukti, _) = sugg.suggest("projukti", true, true, &empty_memory);
    assert!(cands_projukti.contains(&"প্রযুক্তি".to_string()));

    // 5. Verb Inflections
    let (cands_korchhen, _) = sugg.suggest("korchhen", true, true, &empty_memory);
    assert!(
        cands_korchhen.contains(&"করছেন".to_string())
            || cands_korchhen.contains(&"করছেন".to_string())
    );

    // 6. Sibilants
    let (cands_shadhinota, _) = sugg.suggest("shadhinota", true, true, &empty_memory);
    assert!(cands_shadhinota.contains(&"স্বাধীনতা".to_string()));

    let (cands_shorkari, _) = sugg.suggest("shorkari", true, true, &empty_memory);
    assert!(cands_shorkari.contains(&"সরকারি".to_string()));

    let (cands_shobai, _) = sugg.suggest("shobai", true, true, &empty_memory);
    assert!(cands_shobai.contains(&"সবাই".to_string()));

    // 7. Ri-kar & Long Vowels
    let (cands_kritrim, _) = sugg.suggest("kritrim", true, true, &empty_memory);
    assert!(cands_kritrim.contains(&"কৃত্রিম".to_string()));

    let (cands_matribhumi, _) = sugg.suggest("matribhumi", true, true, &empty_memory);
    assert!(cands_matribhumi.contains(&"মাতৃভূমি".to_string()));

    // 8. Morphological Suffixes
    let (cands_boita, _) = sugg.suggest("boita", true, true, &empty_memory);
    assert!(cands_boita.contains(&"বইটা".to_string()));

    let (cands_kajta, _) = sugg.suggest("kajta", true, true, &empty_memory);
    assert!(cands_kajta.contains(&"কাজটা".to_string()));
}

#[test]
fn test_unified_rank1_probabilistic_promotion() {
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

    // 1. Invalid naive transliterations are demoted in favor of real dictionary words at Rank #1
    let (cands_kosto, _) = sugg.suggest("kosto", true, true, &empty_memory);
    assert_eq!(cands_kosto[0], "কষ্ট");

    let (cands_nosto, _) = sugg.suggest("nosto", true, true, &empty_memory);
    assert_eq!(cands_nosto[0], "নষ্ট");

    let (cands_bortoman, _) = sugg.suggest("bortoman", true, true, &empty_memory);
    assert_eq!(cands_bortoman[0], "বর্তমান");

    let (cands_chesta, _) = sugg.suggest("chesta", true, true, &empty_memory);
    assert_eq!(cands_chesta[0], "চেষ্টা");

    let (cands_sristi, _) = sugg.suggest("sristi", true, true, &empty_memory);
    assert_eq!(cands_sristi[0], "সৃষ্টি");

    let (cands_shundor, _) = sugg.suggest("shundor", true, true, &empty_memory);
    assert_eq!(cands_shundor[0], "সুন্দর");

    let (cands_shadhinota, _) = sugg.suggest("shadhinota", true, true, &empty_memory);
    assert_eq!(cands_shadhinota[0], "স্বাধীনতা");
}

#[test]
fn test_sound_law_prefix_autocomplete_and_trailing_emojis() {
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

    // 1. Sound-law prefix autocompletion for "shadhin" suggests "স্বাধীনতা"
    let (cands_shadhin, _) = sugg.suggest("shadhin", true, true, &empty_memory);
    assert_eq!(cands_shadhin[0], "স্বাধীন");
    assert!(cands_shadhin.contains(&"স্বাধীনতা".to_string()));

    // 2. Trailing emoji demotion: emojis should not displace top grammatical words
    let (cands_love, _) = sugg.suggest("bhalobasha", true, true, &empty_memory);
    assert_eq!(cands_love[0], "ভালোবাসা");
    if let Some(pos) = cands_love.iter().position(|c| c == "❤️") {
        assert!(pos >= 3, "Emoji should not displace top candidate words");
    }

    let (cands_tea, _) = sugg.suggest("cha", true, true, &empty_memory);
    assert_ne!(cands_tea[0], "☕");
}

#[test]
fn test_colloquial_verb_suggestions() {
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

    let (cands_kor, _) = sugg.suggest("kortesi", true, true, &empty_memory);
    assert!(
        cands_kor.contains(&"করছি".to_string()) || cands_kor.contains(&"করতেছি".to_string())
    );

    let (cands_jai, _) = sugg.suggest("jaitasi", true, true, &empty_memory);
    assert!(
        cands_jai.contains(&"যাচ্ছি".to_string()) || cands_jai.contains(&"যাইতেছি".to_string())
    );
}

#[test]
fn test_inflection_and_ranking_regressions() {
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

    // 1. Inflected Nouns (should rank exact inflections at #1, not longer compound autocompletions)
    let (cands_deshe, _) = sugg.suggest("deshe", true, true, &empty_memory);
    assert_eq!(cands_deshe[0], "দেশে");

    let (cands_desher, _) = sugg.suggest("desher", true, true, &empty_memory);
    assert_eq!(cands_desher[0], "দেশের");

    let (cands_dine, _) = sugg.suggest("dine", true, true, &empty_memory);
    assert_eq!(cands_dine[0], "দিনে");

    let (cands_ekhane, _) = sugg.suggest("ekhane", true, true, &empty_memory);
    assert_eq!(cands_ekhane[0], "এখানে");

    let (cands_bhabe, _) = sugg.suggest("bhabe", true, true, &empty_memory);
    assert_eq!(cands_bhabe[0], "ভাবে");

    // 2. Everyday common words outranking obscure dictionary entries
    let (cands_tomra, _) = sugg.suggest("tomra", true, true, &empty_memory);
    assert_eq!(cands_tomra[0], "তোমরা");

    let (cands_gari, _) = sugg.suggest("gari", true, true, &empty_memory);
    assert_eq!(cands_gari[0], "গাড়ি");

    let (cands_shob, _) = sugg.suggest("shob", true, true, &empty_memory);
    assert_eq!(cands_shob[0], "সব");

    let (cands_jabo, _) = sugg.suggest("jabo", true, true, &empty_memory);
    assert_eq!(cands_jabo[0], "যাব");

    let (cands_kono, _) = sugg.suggest("kono", true, true, &empty_memory);
    assert_eq!(cands_kono[0], "কোনো");

    // 3. Suffixes and Classifiers
    let (cands_boita, _) = sugg.suggest("boita", true, true, &empty_memory);
    assert_eq!(cands_boita[0], "বইটা");

    let (cands_garite, _) = sugg.suggest("garite", true, true, &empty_memory);
    assert_eq!(cands_garite[0], "গাড়িতে");

    // 4. Sound-law Stem Propagation on Inflected Words
    let (cands_manusher, _) = sugg.suggest("manusher", true, true, &empty_memory);
    assert_eq!(cands_manusher[0], "মানুষের");

    let (cands_porikkhay, _) = sugg.suggest("porikkhay", true, true, &empty_memory);
    assert!(cands_porikkhay[0] == "পরীক্ষায়" || cands_porikkhay[0] == "পরীক্ষায়");

    // 5. Casual Phonetic Typing
    let (cands_ektu, _) = sugg.suggest("ektu", true, true, &empty_memory);
    assert_eq!(cands_ektu[0], "একটু");

    let (cands_bisshash, _) = sugg.suggest("bisshash", true, true, &empty_memory);
    assert_eq!(cands_bisshash[0], "বিশ্বাস");

    let (cands_bissho, _) = sugg.suggest("bissho", true, true, &empty_memory);
    assert_eq!(cands_bissho[0], "বিশ্ব");

    let (cands_ditiyo, _) = sugg.suggest("ditiyo", true, true, &empty_memory);
    assert_eq!(cands_ditiyo[0], "দ্বিতীয়");

    let (cands_shikhok, _) = sugg.suggest("shikhok", true, true, &empty_memory);
    assert_eq!(cands_shikhok[0], "শিক্ষক");

    let (cands_shartho, _) = sugg.suggest("shartho", true, true, &empty_memory);
    assert_eq!(cands_shartho[0], "স্বার্থ");

    // 6. Conversational 2nd-person Verbs (acho, korcho, esho, dekho, cholo, paro)
    let (cands_acho, _) = sugg.suggest("acho", true, true, &empty_memory);
    assert_eq!(cands_acho[0], "আছো");

    let (cands_korcho, _) = sugg.suggest("korcho", true, true, &empty_memory);
    assert_eq!(cands_korcho[0], "করছো");

    let (cands_esho, _) = sugg.suggest("esho", true, true, &empty_memory);
    assert_eq!(cands_esho[0], "এসো");

    let (cands_dekho, _) = sugg.suggest("dekho", true, true, &empty_memory);
    assert_eq!(cands_dekho[0], "দেখো");

    let (cands_cholo, _) = sugg.suggest("cholo", true, true, &empty_memory);
    assert_eq!(cands_cholo[0], "চলো");

    let (cands_paro, _) = sugg.suggest("paro", true, true, &empty_memory);
    assert_eq!(cands_paro[0], "পারো");

    // 7. Complex Sanskrit / Ha-Conjuncts & Clitics
    let (cands_ahban, _) = sugg.suggest("ahban", true, true, &empty_memory);
    assert_eq!(cands_ahban[0], "আহ্বান");

    let (cands_chinho, _) = sugg.suggest("chinho", true, true, &empty_memory);
    assert_eq!(cands_chinho[0], "চিহ্ন");

    let (cands_apranho, _) = sugg.suggest("apranho", true, true, &empty_memory);
    assert_eq!(cands_apranho[0], "অপরাহ্ন");

    let (cands_hrit, _) = sugg.suggest("hritpindo", true, true, &empty_memory);
    assert_eq!(cands_hrit[0], "হৃৎপিণ্ড");

    let (cands_chhatro, _) = sugg.suggest("chhatrochhatriderkeo", true, true, &empty_memory);
    assert_eq!(cands_chhatro[0], "ছাত্রছাত্রীদেরকেও");

    let (cands_dhai, _) = sugg.suggest("dhai", true, true, &empty_memory);
    assert_eq!(cands_dhai[0], "\u{0986}\u{09DC}\u{09BE}\u{0987}");

    // 8. Inflected Loanwords (computere -> কম্পিউটারে, fileti -> ফাইলটি)
    let (cands_comp, _) = sugg.suggest("computere", true, true, &empty_memory);
    assert!(cands_comp[0] == "কম্পিউটারে" || cands_comp.contains(&"কম্পিউটারে".to_string()));

    let (cands_file, _) = sugg.suggest("fileti", true, true, &empty_memory);
    assert!(cands_file[0] == "ফাইলটি" || cands_file.contains(&"ফাইলটি".to_string()));

    let (cands_phone, _) = sugg.suggest("smartphonete", true, true, &empty_memory);
    assert!(cands_phone[0] == "স্মার্টফোনে" || cands_phone.contains(&"স্মার্টফোনে".to_string()));

    let (cands_chair, _) = sugg.suggest("chaire", false, true, &empty_memory);
    assert_eq!(cands_chair[0], "চেয়ারে");
}

#[test]
fn test_auto_dari_toggle() {
    let mut sugg = PhoneticSuggestion::new();
    let empty_memory = HashMap::new();

    // Default: auto_dari = true
    let (cands, _) = sugg.suggest("..", true, true, &empty_memory);
    assert_eq!(cands[0], "।");

    // Disabled: auto_dari = false
    sugg.config.auto_dari = false;
    let (cands_disabled, _) = sugg.suggest("..", true, true, &empty_memory);
    assert_ne!(cands_disabled.first().map(|s| s.as_str()), Some("।"));
}

#[test]
fn test_malformed_layout_json_handling() {
    let mut sugg = PhoneticSuggestion::new();

    // 1. Null layout
    assert!(!sugg.set_layout(&serde_json::Value::Null));

    // 2. String layout
    assert!(!sugg.set_layout(&serde_json::json!("invalid layout string")));

    // 3. Object without patterns
    assert!(!sugg.set_layout(&serde_json::json!({"name": "broken"})));

    // 4. Object with non-array patterns
    assert!(!sugg.set_layout(&serde_json::json!({"patterns": 123})));
}

#[test]
fn test_strict_avro_mode_toggle() {
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

    // 1. Default (strict_avro_mode = false): smart adaptive ranking promotes dictionary word 'চেষ্টা' over naive 'ছেস্তা'
    assert!(!sugg.config.strict_avro_mode);
    let (cands_adaptive, _) = sugg.suggest("chesta", true, true, &empty_memory);
    assert_eq!(cands_adaptive[0], "চেষ্টা");

    // 2. Strict Avro Mode enabled: locks direct naive transliteration 'ছেস্তা' at Slot 0
    sugg.config.strict_avro_mode = true;
    let (cands_strict, _) = sugg.suggest("chesta", true, true, &empty_memory);
    assert_eq!(cands_strict[0], "ছেস্তা");
}

#[test]
fn test_inherent_vowel_sound_law() {
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
    for d in dict_candidates {
        if d.join("dictionary.json").exists() {
            let _ = sugg.database.load_from_dir_with_dict(d, "dictionary.json");
            break;
        }
    }
    let empty_memory = HashMap::new();
    // Inherent vowel words that users type with 'a' (e.g. upanibeshbad, chalaman, badan)
    let (cands, _) = sugg.suggest("upanibeshbad", true, true, &empty_memory);
    assert!(
        cands.contains(&"উপনিবেশবাদ".to_string()),
        "Candidates {:?} should contain উপনিবেশবাদ",
        cands
    );

    let (cands2, _) = sugg.suggest("chalaman", true, true, &empty_memory);
    assert!(
        cands2.contains(&"চলমান".to_string()),
        "Candidates {:?} should contain চলমান",
        cands2
    );

    let (cands3, _) = sugg.suggest("anchale", true, true, &empty_memory);
    assert!(
        cands3.contains(&"অঞ্চলে".to_string()),
        "Candidates {:?} should contain অঞ্চলে",
        cands3
    );

    let (cands4, _) = sugg.suggest("shadharon", true, true, &empty_memory);
    assert!(
        cands4.contains(&"সাধারণ".to_string()),
        "Candidates {:?} should contain সাধারণ",
        cands4
    );
}
