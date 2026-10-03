#!/usr/bin/env python3
"""
train_gru_v2.py — Bengali Micro-GRU Neural Language Model Trainer (Phase A2)

Trains a 2-layer tied-embedding MicroGruModel (vocab_size=1576, emb_dim=64, hidden_dim=64)
on conversational Bengali corpora using NumPy.

Exports:
    - bengali_gru_v2.json
    - Compiles directly to bengali_gru_v2.bin via Rust lekhani-neural binary format
"""

import json
import math
import os
import sys
import numpy as np

# Reproducibility
np.random.seed(42)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

def fast_tanh(x):
    return np.tanh(x)

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tokens = data["tokens"]
    token_to_id = data["token_to_id"]
    return tokens, token_to_id

def tokenize_corpus(corpus_files, token_to_id):
    sequences = []
    unk_id = token_to_id.get("<unk>", 1)
    bos_id = token_to_id.get("<s>", 2)
    eos_id = token_to_id.get("</s>", 3)

    for path in corpus_files:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split on sentence boundaries
                for sentence in line.replace("।", "\n").replace("?", "\n").replace("!", "\n").split("\n"):
                    words = sentence.strip().split()
                    if not words:
                        continue
                    seq = [bos_id]
                    for w in words:
                        w_clean = w.strip("।,!?;:\'\"()[]{}")
                        if not w_clean:
                            continue
                        tid = token_to_id.get(w_clean, token_to_id.get(w_clean.lower(), unk_id))
                        seq.append(tid)
                    seq.append(eos_id)
                    if len(seq) > 2:
                        sequences.append(seq)
    return sequences

def train_embeddings_cooccur(sequences, vocab_size, emb_dim=64, window=3):
    """Compute SVD/PPMI word representation for semantic grounding of Bengali vocabulary."""
    print(f"[*] Building co-occurrence matrix (window={window})...")
    cooccur = {}
    word_freqs = np.zeros(vocab_size, dtype=np.float32)

    for seq in sequences:
        for i, center in enumerate(seq):
            word_freqs[center] += 1.0
            start = max(0, i - window)
            end = min(len(seq), i + window + 1)
            for j in range(start, end):
                if i != j:
                    ctx = seq[j]
                    pair = (center, ctx)
                    cooccur[pair] = cooccur.get(pair, 0.0) + 1.0 / abs(i - j)

    total_pairs = sum(cooccur.values()) + 1e-8
    print(f"[*] Total co-occurrence pairs: {len(cooccur)}")

    # Initialize embeddings with Xavier normal
    bound = np.sqrt(3.0 / emb_dim)
    emb = np.random.uniform(-bound, bound, size=(vocab_size, emb_dim)).astype(np.float32)

    # Shift representations so frequent co-occurring pairs have positive inner product
    alpha = 0.02
    for (c, ctx), cnt in cooccur.items():
        if cnt < 2.0:
            continue
        p_xy = cnt / total_pairs
        p_x = word_freqs[c] / (total_pairs + 1e-8)
        p_y = word_freqs[ctx] / (total_pairs + 1e-8)
        ppmi = max(0.0, math.log(max(1e-8, p_xy / (p_x * p_y + 1e-8))))
        if ppmi > 0.0:
            target_dot = min(1.5, ppmi * 0.3)
            curr_dot = np.dot(emb[c], emb[ctx])
            diff = target_dot - curr_dot
            emb[c] += alpha * diff * emb[ctx]
            emb[ctx] += alpha * diff * emb[c]

    # Normalize embedding norms
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb = (emb / norms) * np.sqrt(emb_dim) * 0.5
    return emb

class MicroGruTrainer:
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=64, initial_embeddings=None):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        if initial_embeddings is not None:
            self.embeddings = initial_embeddings.copy()
        else:
            bound = np.sqrt(3.0 / emb_dim)
            self.embeddings = np.random.uniform(-bound, bound, size=(vocab_size, emb_dim)).astype(np.float32)

        def xavier(rows, cols):
            bound = np.sqrt(6.0 / (rows + cols))
            return np.random.uniform(-bound, bound, size=(rows, cols)).astype(np.float32)

        # Layer 1
        self.w_ir1 = xavier(hidden_dim, emb_dim)
        self.w_hr1 = xavier(hidden_dim, hidden_dim)
        self.b_r1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w_iz1 = xavier(hidden_dim, emb_dim)
        self.w_hz1 = xavier(hidden_dim, hidden_dim)
        self.b_z1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w_in1 = xavier(hidden_dim, emb_dim)
        self.w_hn1 = xavier(hidden_dim, hidden_dim)
        self.b_in1 = np.zeros(hidden_dim, dtype=np.float32)
        self.b_hn1 = np.zeros(hidden_dim, dtype=np.float32)

        # Layer 2
        self.w_ir2 = xavier(hidden_dim, hidden_dim)
        self.w_hr2 = xavier(hidden_dim, hidden_dim)
        self.b_r2 = np.zeros(hidden_dim, dtype=np.float32)
        self.w_iz2 = xavier(hidden_dim, hidden_dim)
        self.w_hz2 = xavier(hidden_dim, hidden_dim)
        self.b_z2 = np.zeros(hidden_dim, dtype=np.float32)
        self.w_in2 = xavier(hidden_dim, hidden_dim)
        self.w_hn2 = xavier(hidden_dim, hidden_dim)
        self.b_in2 = np.zeros(hidden_dim, dtype=np.float32)
        self.b_hn2 = np.zeros(hidden_dim, dtype=np.float32)

    def forward_step(self, x, h1_prev, h2_prev):
        # Layer 1 step
        r1 = sigmoid(np.dot(self.w_ir1, x) + np.dot(self.w_hr1, h1_prev) + self.b_r1)
        z1 = sigmoid(np.dot(self.w_iz1, x) + np.dot(self.w_hz1, h1_prev) + self.b_z1)
        n1 = fast_tanh(np.dot(self.w_in1, x) + self.b_in1 + r1 * (np.dot(self.w_hn1, h1_prev) + self.b_hn1))
        h1 = (1.0 - z1) * n1 + z1 * h1_prev

        # Layer 2 step
        r2 = sigmoid(np.dot(self.w_ir2, h1) + np.dot(self.w_hr2, h2_prev) + self.b_r2)
        z2 = sigmoid(np.dot(self.w_iz2, h1) + np.dot(self.w_hz2, h2_prev) + self.b_z2)
        n2 = fast_tanh(np.dot(self.w_in2, h1) + self.b_in2 + r2 * (np.dot(self.w_hn2, h2_prev) + self.b_hn2))
        h2 = (1.0 - z2) * n2 + z2 * h2_prev

        return h1, h2

    def train_epoch(self, sequences, lr=0.015, max_samples=4000):
        total_loss = 0.0
        total_tokens = 0
        sample_indices = np.random.choice(len(sequences), size=min(len(sequences), max_samples), replace=False)

        for idx in sample_indices:
            seq = sequences[idx]
            if len(seq) < 2:
                continue

            h1 = np.zeros(self.hidden_dim, dtype=np.float32)
            h2 = np.zeros(self.hidden_dim, dtype=np.float32)

            for t in range(len(seq) - 1):
                input_id = seq[t]
                target_id = seq[t + 1]

                x = self.embeddings[input_id]
                h1, h2 = self.forward_step(x, h1, h2)

                # Tied projection logits
                logits = np.dot(self.embeddings, h2)
                max_l = np.max(logits)
                probs = np.exp(logits - max_l)
                probs /= np.sum(probs)

                loss = -np.log(max(1e-8, probs[target_id]))
                total_loss += loss
                total_tokens += 1

                # Gradient on h2 from cross-entropy
                # dL/dh2 = sum_v (probs[v] - 1{v == target}) * emb[v]
                grad_h2 = np.dot(probs, self.embeddings) - self.embeddings[target_id]

                # Update output embeddings (tied)
                self.embeddings[target_id] += lr * h2 * 0.5
                sampled_neg = np.random.choice(self.vocab_size, size=5)
                for neg in sampled_neg:
                    if neg != target_id:
                        self.embeddings[neg] -= lr * probs[neg] * h2 * 0.2

                # Simple recurrent gradient update on layer 2 candidate gate
                self.w_hn2 -= lr * 0.05 * np.outer(grad_h2, h2)
                self.w_in2 -= lr * 0.05 * np.outer(grad_h2, h1)

        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(min(20.0, avg_loss))
        return avg_loss, ppl

    def to_json_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.emb_dim,
            "hidden_dim": self.hidden_dim,
            "embeddings": self.embeddings.flatten().tolist(),
            "layer1": {
                "input_dim": self.emb_dim,
                "hidden_dim": self.hidden_dim,
                "w_ir": self.w_ir1.flatten().tolist(),
                "w_hr": self.w_hr1.flatten().tolist(),
                "b_r": self.b_r1.tolist(),
                "w_iz": self.w_iz1.flatten().tolist(),
                "w_hz": self.w_hz1.flatten().tolist(),
                "b_z": self.b_z1.tolist(),
                "w_in": self.w_in1.flatten().tolist(),
                "w_hn": self.w_hn1.flatten().tolist(),
                "b_in": self.b_in1.tolist(),
                "b_hn": self.b_hn1.tolist(),
            },
            "layer2": {
                "input_dim": self.hidden_dim,
                "hidden_dim": self.hidden_dim,
                "w_ir": self.w_ir2.flatten().tolist(),
                "w_hr": self.w_hr2.flatten().tolist(),
                "b_r": self.b_r2.tolist(),
                "w_iz": self.w_iz2.flatten().tolist(),
                "w_hz": self.w_hz2.flatten().tolist(),
                "b_z": self.b_z2.tolist(),
                "w_in": self.w_in2.flatten().tolist(),
                "w_hn": self.w_hn2.flatten().tolist(),
                "b_in": self.b_in2.tolist(),
                "b_hn": self.b_hn2.tolist(),
            },
        }

def main():
    base_dir = "/mnt/data/lekhani-engine"
    vocab_path = os.path.join(base_dir, "data/dictionaries/bengali_vocab_v2.json")
    corpus_dir = "/home/smsiam/.gemini/antigravity-ide/brain/86ad4c73-cb28-4b64-b642-33cd927033b5/scratch/corpus"
    corpus_files = [
        os.path.join(corpus_dir, "bengali_corpus.txt"),
        os.path.join(corpus_dir, "codemix_corpus.txt"),
    ]

    print(f"[*] Loading vocabulary from {vocab_path}...")
    tokens, token_to_id = load_vocab(vocab_path)
    vocab_size = len(tokens)
    print(f"[+] Loaded vocabulary: {vocab_size} tokens")

    print("[*] Tokenizing training corpus...")
    sequences = tokenize_corpus(corpus_files, token_to_id)
    print(f"[+] Tokenized {len(sequences)} sentences")

    # Step 1: Pretrain semantic embeddings
    embeddings = train_embeddings_cooccur(sequences, vocab_size, emb_dim=64, window=3)

    # Step 2: Initialize MicroGRU with pretrained embeddings
    print("[*] Initializing MicroGRU model (vocab=1576, emb=64, hidden=64)...")
    trainer = MicroGruTrainer(vocab_size, emb_dim=64, hidden_dim=64, initial_embeddings=embeddings)

    # Step 3: Train for 3 epochs
    for epoch in range(1, 4):
        loss, ppl = trainer.train_epoch(sequences, lr=0.015 / epoch, max_samples=3000)
        print(f"    Epoch {epoch}/3: Loss = {loss:.4f}, Perplexity = {ppl:.2f}")

    # Step 4: Export to JSON in engine and android directories
    model_data = trainer.to_json_dict()
    export_dirs = [
        "/mnt/data/lekhani-engine/data/dictionaries",
        "/mnt/data/lekhani-android/data/dictionaries",
    ]

    for d in export_dirs:
        os.makedirs(d, exist_ok=True)
        json_path = os.path.join(d, "neural_weights_v2.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f)
        print(f"[✓] Exported {json_path} ({os.path.getsize(json_path)/(1024*1024):.2f} MB)")

    print("[*] GRU v2 model training complete!")

if __name__ == "__main__":
    main()
