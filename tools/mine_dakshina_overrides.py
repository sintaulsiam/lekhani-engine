#!/usr/bin/env python3
"""
Mine high-confidence phonetic transliteration overrides from Dakshina lexicon datasets.
Preserves 100% of existing curated/manual overrides, merging only high-confidence,
frequently attested Latin-Bengali pairs.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

def is_valid_bengali(word: str) -> bool:
    if not word:
        return False
    return all('\u0980' <= ch <= '\u09FF' for ch in word)

def is_valid_latin(key: str) -> bool:
    if len(key) < 3:
        return False
    return key.isalnum() and key.isascii() and key.islower()

def mine_overrides(tsv_paths, existing_json_path, min_count=2, min_confidence=0.70):
    # 1. Load existing overrides
    existing = {}
    if os.path.exists(existing_json_path):
        with open(existing_json_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    print(f"📖 Loaded {len(existing)} existing baseline overrides from {existing_json_path}")

    # 2. Aggregate counts across TSVs
    latin_to_bn = defaultdict(lambda: defaultdict(int))
    total_tsv_lines = 0
    for p in tsv_paths:
        p = Path(p)
        if not p.exists():
            print(f"⚠️ Warning: {p} does not exist, skipping.")
            continue
        print(f"📥 Reading {p}...")
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                bn, latin, count_str = parts[0], parts[1].strip().lower(), parts[2]
                try:
                    cnt = int(count_str)
                except ValueError:
                    continue

                if not is_valid_latin(latin):
                    continue
                if not is_valid_bengali(bn):
                    continue

                latin_to_bn[latin][bn] += cnt
                total_tsv_lines += 1

    print(f"📊 Processed {total_tsv_lines} entries across {len(latin_to_bn)} unique Latin keys.")

    # 3. Mine high-confidence entries
    merged = dict(existing)
    added_count = 0

    for latin, bn_counts in latin_to_bn.items():
        # Never override existing curated pairs
        if latin in merged:
            continue

        total = sum(bn_counts.values())
        if total < min_count:
            continue

        cands = []
        for bn, cnt in bn_counts.items():
            conf = cnt / total
            if conf >= min_confidence and cnt >= min_count:
                cands.append([bn, round(conf, 4)])

        if cands:
            cands.sort(key=lambda x: -x[1])
            # Cap candidates per Latin key to 2
            merged[latin] = cands[:2]
            added_count += 1

    print(f"✨ Mined {added_count} new high-confidence overrides (min_count={min_count}, min_conf={min_confidence})")
    print(f"📦 Total merged overrides: {len(merged)}")
    return merged

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    scratch_dir = Path("/home/smsiam/.gemini/antigravity-ide/brain/6e8f5951-b148-4374-87c2-2e54e03df641/scratch/dakshina/dakshina_dataset_v1.0/bn/lexicons")
    
    tsvs = [
        scratch_dir / "bn.translit.sampled.train.tsv",
        scratch_dir / "bn.translit.sampled.dev.tsv",
    ]

    target_json = base_dir / "data" / "dictionaries" / "phonetic_overrides.json"
    
    merged = mine_overrides(tsvs, target_json, min_count=2, min_confidence=0.70)
    
    with open(target_json, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"💾 Written to {target_json}")
