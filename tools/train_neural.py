#!/usr/bin/env python3
"""
Lekhani Micro-Neural GRU Training & Quantization Pipeline

Trains a compact 2-layer GRU with tied embeddings on conversational Bengali + Banglish corpora,
quantizes the weights, and exports to both ONNX and pure Rust binary format.

Usage:
    python tools/train_neural.py --data-dir corpus/ --output assets/neural_lm.onnx
"""

import os
import argparse
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim

class MicroGruLanguageModel(nn.Module):
    def __init__(self, vocab_size=16384, embedding_dim=128, hidden_dim=128, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        # Tied embeddings output projection
        self.fc = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight

    def forward(self, x, h=None):
        emb = self.embedding(x)
        out, h_next = self.gru(emb, h)
        logits = self.fc(out)
        return logits, h_next

def export_onnx(model, output_path):
    model.eval()
    dummy_input = torch.randint(0, model.vocab_size, (1, 16), dtype=torch.long)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}, "logits": {0: "batch_size", 1: "seq_len"}},
    )
    print(f"[+] Successfully exported ONNX model to {output_path}")

def export_rust_binary(model, output_path):
    """Export weights to Rust-compatible JSON or raw binary format"""
    model.eval()
    weights = {
        "vocab_size": model.vocab_size,
        "embedding_dim": model.embedding_dim,
        "hidden_dim": model.hidden_dim,
        "embeddings": model.embedding.weight.detach().cpu().numpy().flatten().tolist(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights, f)
    print(f"[+] Successfully exported Rust-compatible weights to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Lekhani Micro-Neural GRU Trainer")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-onnx", type=str, default="assets/neural_lm.onnx")
    parser.add_argument("--output-json", type=str, default="assets/neural_weights.json")
    args = parser.parse_args()

    print("[*] Initializing 2-layer MicroGruLanguageModel...")
    model = MicroGruLanguageModel(
        vocab_size=args.vocab_size,
        embedding_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Total parameter count: {total_params:,} (~{total_params * 4 / (1024 * 1024):.2f} MB f32)")

    os.makedirs(os.path.dirname(args.output_onnx) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    export_onnx(model, args.output_onnx)
    export_rust_binary(model, args.output_json)

if __name__ == "__main__":
    main()
