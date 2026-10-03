use std::path::Path;
use lekhani_neural::{BpeVocabulary, MicroGruModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict_dirs = [
        Path::new("/mnt/data/lekhani-engine/data/dictionaries"),
        Path::new("/mnt/data/lekhani-android/data/dictionaries"),
    ];

    for dict_dir in &dict_dirs {
        println!("[*] Processing {:?}", dict_dir);

        // 1. Compile v1 model & vocab
        let weights_json = dict_dir.join("neural_weights.json");
        let vocab_json = dict_dir.join("neural_vocab.json");
        if weights_json.exists() && vocab_json.exists() {
            println!("  Loading v1 vocab from {:?}...", vocab_json);
            let vocab = BpeVocabulary::load_json(&vocab_json)?;
            let vocab_bin = dict_dir.join("bengali_vocab.bin");
            vocab.save_binary(&vocab_bin)?;

            println!("  Loading v1 weights from {:?}...", weights_json);
            let model = MicroGruModel::load_json(&weights_json)?;
            let model_bin = dict_dir.join("bengali_gru.bin");
            model.save_binary(&model_bin)?;
            println!("  [✓] Saved v1 binary model to {:?}", model_bin);
        }

        // 2. Compile v2 model & vocab
        let weights_v2_json = dict_dir.join("neural_weights_v2.json");
        let vocab_v2_json = dict_dir.join("bengali_vocab_v2.json");
        if weights_v2_json.exists() && vocab_v2_json.exists() {
            println!("  Loading v2 vocab from {:?}...", vocab_v2_json);
            let vocab_v2 = BpeVocabulary::load_json(&vocab_v2_json)?;
            println!("  Loaded v2 vocab with {} tokens.", vocab_v2.len());

            println!("  Loading v2 weights from {:?}...", weights_v2_json);
            let model_v2 = MicroGruModel::load_json(&weights_v2_json)?;
            println!(
                "  Loaded v2 model: vocab_size={}, emb_dim={}, hidden_dim={}",
                model_v2.vocab_size, model_v2.embedding_dim, model_v2.hidden_dim
            );

            assert_eq!(
                model_v2.vocab_size,
                vocab_v2.len(),
                "v2 model vocab_size must match v2 vocab length"
            );

            let model_v2_bin = dict_dir.join("bengali_gru_v2.bin");
            model_v2.save_binary(&model_v2_bin)?;
            println!("  [✓] Saved v2 binary model to {:?}", model_v2_bin);

            // Test inference on v2
            let predictor = lekhani_neural::NeuralContextPredictor::try_new(
                std::sync::Arc::new(model_v2),
                std::sync::Arc::new(vocab_v2),
            )?;
            let preds1 = predictor.predict_candidates("আমি ভাত", 3);
            println!("  v2 predictions for 'আমি ভাত': {:?}", preds1);

            let preds2 = predictor.predict_candidates("কেমন", 3);
            println!("  v2 predictions for 'কেমন': {:?}", preds2);
        }
    }

    Ok(())
}
