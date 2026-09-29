use lekhani_ai::{BeamSearchDecoder, ContextScorer, LanguageModel, NextWordPredictor};

fn main() {
    println!("=== Lekhani AI: Standalone Bengali Language Intelligence ===\n");

    let mut lm = LanguageModel::new();
    let candidate_paths = [
        "/mnt/data/lekhani-android/data/dictionaries/bengali_lm.bin",
        "../lekhani-android/data/dictionaries/bengali_lm.bin",
        "../../lekhani-android/data/dictionaries/bengali_lm.bin",
        "data/dictionaries/bengali_lm.bin",
    ];
    for path in candidate_paths {
        if std::path::Path::new(path).exists() {
            if let Ok(_) = lm.load_binary_file(path) {
                println!("Loaded bengali_lm.bin from {} successfully!", path);
                break;
            }
        }
    }
    let scorer = ContextScorer::with_language_model(lm.clone());
    let predictor = NextWordPredictor::new();
    let decoder = BeamSearchDecoder::new();

    // 1. Language Model Trigram Scoring
    println!("1. N-Gram Scoring:");
    let score1 = lm.score_candidate(Some("আমি"), Some("ভাত"), "খাচ্ছি");
    let score2 = lm.score_candidate(Some("আমি"), Some("ভাত"), "যাচ্ছি");
    println!("  'আমি ভাত [খাচ্ছি]' log-prob: {:.3}", score1);
    println!("  'আমি ভাত [যাচ্ছি]' log-prob: {:.3}", score2);
    println!("  -> Winner: {}\n", if score1 > score2 { "খাচ্ছি" } else { "যাচ্ছি" });

    // 2. Homophone Disambiguation
    println!("2. Homophone Context Reranking:");
    let s_book_poda = lm.score_candidate(None, Some("বই"), "পড়া");
    let s_book_pora = lm.score_candidate(None, Some("বই"), "পরা");
    println!("  'বই [পড়া]' log-prob: {:.3}", s_book_poda);
    println!("  'বই [পরা]' log-prob: {:.3}", s_book_pora);

    let s_shirt_pora = lm.score_candidate(None, Some("শার্ট"), "পরা");
    let s_shirt_poda = lm.score_candidate(None, Some("শার্ট"), "পড়া");
    println!("  'শার্ট [পরা]' log-prob: {:.3}", s_shirt_pora);
    println!("  'শার্ট [পড়া]' log-prob: {:.3}", s_shirt_poda);

    let homophones_rev = vec!["পরা".to_string(), "পড়া".to_string()];
    let book_context = scorer.rank_candidates(&["বই"], &homophones_rev);
    let shirt_context = scorer.rank_candidates(&["শার্ট"], &homophones_rev);
    println!("  Input [\"পরা\", \"পড়া\"] with Context 'বই': {:?}", book_context);
    println!("  Input [\"পরা\", \"পড়া\"] with Context 'শার্ট': {:?}\n", shirt_context);

    // 3. Next-Word Prediction
    println!("3. Next-Word Continuations:");
    let continuations = predictor.predict_next(&["বাংলাদেশ", "একটি"], 3);
    println!("  'বাংলাদেশ একটি ...' -> {:?}\n", continuations);

    // 4. Global Beam Search Sequence Decoding
    println!("4. Beam Search Sentence Decoding:");
    let lattice = vec![
        vec!["আমি".to_string()],
        vec!["বই".to_string()],
        vec!["পরা".to_string(), "পড়া".to_string()],
    ];
    let decoded = decoder.decode(&lattice);
    println!("  Decoded optimum sentence: {}\n", decoded.join(" "));
}
