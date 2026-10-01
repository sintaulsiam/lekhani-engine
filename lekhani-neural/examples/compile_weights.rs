use std::path::Path;
use lekhani_neural::{BpeVocabulary, MicroGruModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict_dir = Path::new("/mnt/data/lekhani-android/data/dictionaries");
    let weights_json = dict_dir.join("neural_weights.json");
    let vocab_json = dict_dir.join("neural_vocab.json");

    println!("Loading vocab from {:?}...", vocab_json);
    let vocab = BpeVocabulary::load_json(&vocab_json)?;
    println!("Loaded vocab with {} tokens.", vocab.len());

    let vocab_bin = dict_dir.join("bengali_vocab.bin");
    vocab.save_binary(&vocab_bin)?;
    println!("Saved binary vocab to {:?} ({} bytes).", vocab_bin, std::fs::metadata(&vocab_bin)?.len());

    println!("Loading model weights from {:?}...", weights_json);
    let model = MicroGruModel::load_json(&weights_json)?;
    println!("Loaded model: vocab_size={}, emb_dim={}, hidden_dim={}", model.vocab_size, model.embedding_dim, model.hidden_dim);

    let model_bin = dict_dir.join("bengali_gru.bin");
    model.save_binary(&model_bin)?;
    println!("Saved binary model to {:?} ({} bytes).", model_bin, std::fs::metadata(&model_bin)?.len());

    // Test inference
    let predictor = lekhani_neural::NeuralContextPredictor::new(std::sync::Arc::new(model), std::sync::Arc::new(vocab));
    let preds = predictor.predict_candidates("আমি ভাত", 3);
    println!("Predictions for 'আমি ভাত': {:?}", preds);

    let preds2 = predictor.predict_candidates("কেমন", 3);
    println!("Predictions for 'কেমন': {:?}", preds2);

    Ok(())
}
