#!/usr/bin/env python3
"""
Build a Bengali-first BPE vocabulary for the Lekhani GRU model.

Usage:
    python3 build_vocab.py \
        --corpus-dir /path/to/text/files \
        --overrides /path/to/phonetic_overrides.json \
        --output-vocab /path/to/bengali_vocab_v2.json \
        --vocab-size 8192

Requirements:
    pip install tokenizers
"""
import argparse
import json
import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFC

def load_overrides_as_text(overrides_path: str) -> str:
    """Extract Bengali word forms from phonetic_overrides.json as training text."""
    with open(overrides_path, encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for romanized, candidates in data.items():
        for (word, _score) in candidates:
            lines.append(word)
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=False, help="Directory of .txt corpus files")
    parser.add_argument("--overrides", required=True, help="Path to phonetic_overrides.json")
    parser.add_argument("--output-vocab", required=True, help="Output JSON vocabulary path")
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()

    # Collect training files
    training_files = []
    if args.corpus_dir:
        for p in Path(args.corpus_dir).rglob("*.txt"):
            training_files.append(str(p))

    # Write overrides as a temp file
    overrides_text = load_overrides_as_text(args.overrides)
    tmp_overrides = "/tmp/lekhani_overrides_corpus.txt"
    with open(tmp_overrides, "w", encoding="utf-8") as f:
        f.write(overrides_text)
    training_files.append(tmp_overrides)

    if not training_files:
        raise ValueError("No training files found. Provide --corpus-dir or --overrides.")

    print(f"Training BPE on {len(training_files)} file(s) with target vocab_size={args.vocab_size}")

    # Initialize BPE tokenizer with Bengali-aware settings
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFC()           # Always NFC-normalize Bengali text
    tokenizer.pre_tokenizer = Whitespace() # Word-level split before BPE

    # Special tokens must be first (matching lekhani_neural's vocab.rs constants)
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        # Do not split on Bengali unicode ranges — treat grapheme clusters as atomic units
        initial_alphabet=[],
    )

    tokenizer.train(training_files, trainer)

    # Export as JSON matching BpeVocabulary::from_tokens format
    vocab_json = tokenizer.get_vocab(with_added_tokens=True)
    # Sort by ID to get ordered token list
    id_to_token = {v: k for k, v in vocab_json.items()}
    tokens = [id_to_token[i] for i in range(len(id_to_token))]

    output = {"tokens": tokens, "token_to_id": vocab_json}
    with open(args.output_vocab, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print stats
    bengali_count = sum(1 for t in tokens if any('\u0980' <= c <= '\u09FF' for c in t))
    print(f"Vocabulary built: {len(tokens)} tokens")
    print(f"  Bengali tokens: {bengali_count} ({100*bengali_count/len(tokens):.1f}%)")
    print(f"  Output: {args.output_vocab}")

if __name__ == "__main__":
    main()
