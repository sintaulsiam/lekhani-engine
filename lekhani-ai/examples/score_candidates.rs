use lekhani_ai::{BeamSearchDecoder, ContextScorer, LanguageModel, NextWordPredictor};

fn main() {
    println!("=== Lekhani AI: Standalone Bengali Language Intelligence ===\n");

    let lm = LanguageModel::new();
    let scorer = ContextScorer::new();
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
    let homophones = vec!["পড়া".to_string(), "পরা".to_string()];
    let book_context = scorer.rank_candidates(&["বই"], &homophones);
    let shirt_context = scorer.rank_candidates(&["শার্ট"], &homophones);
    println!("  Context 'বই': {:?}", book_context);
    println!("  Context 'শার্ট': {:?}\n", shirt_context);

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
