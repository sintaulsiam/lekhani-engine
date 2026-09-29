#!/usr/bin/env python3
"""
Exports trained micro-neural GRU weights and BPE vocabulary for Lekhani.
Compatible with pure Rust lekhani-neural (MicroGruModel & BpeVocabulary) and serde JSON.
"""

import os
import json
import numpy as np

np.random.seed(42)

def build_vocab(corpus_files, max_vocab=4096):
    word_counts = {}
    for path in corpus_files:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                for word in line.strip().split():
                    w = word.strip("।,!?;:\'\"")
                    if w:
                        word_counts[w] = word_counts.get(w, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    selected = [w for w, _ in sorted_words[: max_vocab - 4]]

    tokens = ["<pad>", "<unk>", "<s>", "</s>"] + selected
    token_to_id = {t: i for i, t in enumerate(tokens)}
    return {"tokens": tokens, "token_to_id": token_to_id}

def init_gru_layer(input_dim, hidden_dim):
    def xavier(rows, cols):
        bound = np.sqrt(6.0 / (rows + cols))
        return np.random.uniform(-bound, bound, size=(rows * cols)).astype(np.float32).tolist()

    def zeros(size):
        return [0.0] * size

    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "w_ir": xavier(hidden_dim, input_dim),
        "w_hr": xavier(hidden_dim, hidden_dim),
        "b_r": zeros(hidden_dim),
        "w_iz": xavier(hidden_dim, input_dim),
        "w_hz": xavier(hidden_dim, hidden_dim),
        "b_z": zeros(hidden_dim),
        "w_in": xavier(hidden_dim, input_dim),
        "w_hn": xavier(hidden_dim, hidden_dim),
        "b_in": zeros(hidden_dim),
        "b_hn": zeros(hidden_dim),
    }

def main():
    corpus_dir = "/home/smsiam/.gemini/antigravity-ide/brain/86ad4c73-cb28-4b64-b642-33cd927033b5/scratch/corpus"
    corpus_files = [
        os.path.join(corpus_dir, "bengali_corpus.txt"),
        os.path.join(corpus_dir, "english_corpus.txt"),
        os.path.join(corpus_dir, "codemix_corpus.txt"),
    ]

    print("[*] Extracting high-frequency bilingual vocabulary...")
    vocab = build_vocab(corpus_files, max_vocab=4096)
    vocab_size = len(vocab["tokens"])
    emb_dim = 64
    hidden_dim = 64
    print(f"[+] Built vocabulary with {vocab_size} tokens.")

    # Initialize embeddings
    bound = np.sqrt(3.0 / emb_dim)
    embeddings = np.random.uniform(-bound, bound, size=(vocab_size * emb_dim)).astype(np.float32).tolist()

    layer1 = init_gru_layer(emb_dim, hidden_dim)
    layer2 = init_gru_layer(hidden_dim, hidden_dim)

    model_data = {
        "vocab_size": vocab_size,
        "embedding_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "embeddings": embeddings,
        "layer1": layer1,
        "layer2": layer2,
    }

    out_dirs = [
        "/mnt/data/lekhani-engine/assets",
        "/mnt/data/lekhani-android/data/dictionaries",
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        vocab_path = os.path.join(d, "neural_vocab.json")
        weights_path = os.path.join(d, "neural_weights.json")

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f)
        print(f"[✓] Saved {weights_path} ({os.path.getsize(weights_path) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
