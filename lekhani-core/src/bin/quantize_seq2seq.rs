//! Quantize `seq2seq_transliterator.bin` (F32) into `seq2seq_transliterator_int8.bin` (INT8).
use lekhani_neural::Seq2SeqTransliterator;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/dictionaries/seq2seq_transliterator.bin"));
    let out_path = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/dictionaries/seq2seq_transliterator_int8.bin"));

    let f32_model =
        Seq2SeqTransliterator::load(&in_path).expect("Failed to load F32 Seq2Seq model");
    let q_model = f32_model.to_quantized();
    q_model
        .save(&out_path)
        .expect("Failed to save INT8 Seq2Seq model");

    let size_f32 = std::fs::metadata(&in_path).map(|m| m.len()).unwrap_or(0);
    let size_q = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "Quantized {:?} ({:.1} KB) -> {:?} ({:.1} KB, {:.1}% smaller)",
        in_path,
        size_f32 as f32 / 1024.0,
        out_path,
        size_q as f32 / 1024.0,
        if size_f32 > 0 {
            (1.0 - (size_q as f32 / size_f32 as f32)) * 100.0
        } else {
            0.0
        }
    );
}
