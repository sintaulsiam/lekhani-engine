#!/usr/bin/env python3
"""
Pragyan Phase 1 Step 2: Data Accounting Tool (Python verification script)
"""

import sys
from collections import defaultdict
from pathlib import Path

def has_bengali_conjunct(s: str) -> bool:
    chars = list(s)
    for i in range(len(chars) - 1):
        if chars[i] == '\u09cd':
            next_c = chars[i + 1]
            if ('\u0995' <= next_c <= '\u09b9') or next_c in ['\u09ce', '\u09dc', '\u09dd', '\u09df']:
                return True
    return False

def analyze_split(name: str, path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip().split('\t') for l in f if l.strip()]

    total = len(lines)
    targets = set(p[0].strip() for p in lines if len(p) >= 2)
    romans = set(p[1].strip().lower() for p in lines if len(p) >= 2)

    bn_chars = [len(p[0].strip()) for p in lines if len(p) >= 2]
    lat_chars = [len(p[1].strip()) for p in lines if len(p) >= 2]
    bn_bytes = [len(p[0].strip().encode('utf-8')) for p in lines if len(p) >= 2]

    conj_count = sum(1 for p in lines if len(p) >= 2 and has_bengali_conjunct(p[0].strip()))

    kept = [p for p in lines if len(p) >= 2 and len(p[0].strip().encode('utf-8')) <= 24]
    dropped = [p for p in lines if len(p) >= 2 and len(p[0].strip().encode('utf-8')) > 24]

    kept_conj = sum(1 for p in kept if has_bengali_conjunct(p[0].strip()))
    dropped_conj = sum(1 for p in dropped if has_bengali_conjunct(p[0].strip()))

    return {
        'name': name.upper(),
        'total': total,
        'unique_targets': len(targets),
        'unique_romans': len(romans),
        'avg_bn_chars': sum(bn_chars) / len(bn_chars),
        'avg_lat_chars': sum(lat_chars) / len(lat_chars),
        'conj_count': conj_count,
        'conj_rate': (conj_count / total) * 100.0,
        'kept_count': len(kept),
        'kept_conj_rate': (kept_conj / len(kept)) * 100.0 if kept else 0.0,
        'kept_avg_bn': sum(len(p[0].strip()) for p in kept) / len(kept) if kept else 0.0,
        'kept_avg_lat': sum(len(p[1].strip()) for p in kept) / len(kept) if kept else 0.0,
        'dropped_count': len(dropped),
        'dropped_conj_rate': (dropped_conj / len(dropped)) * 100.0 if dropped else 0.0,
        'dropped_avg_bn': sum(len(p[0].strip()) for p in dropped) / len(dropped) if dropped else 0.0,
        'dropped_avg_lat': sum(len(p[1].strip()) for p in dropped) / len(dropped) if dropped else 0.0,
    }

def main():
    splits = [
        ('TRAIN', Path('data/eval/bn.translit.sampled.train.tsv')),
        ('DEV', Path('data/eval/bn.translit.sampled.dev.tsv')),
        ('TEST', Path('data/eval/bn.translit.sampled.test.tsv')),
    ]
    for name, p in splits:
        res = analyze_split(name, p)
        print(f"[{res['name']}] Total: {res['total']}, Targets: {res['unique_targets']}, Romans: {res['unique_romans']}, Conjuncts: {res['conj_count']} ({res['conj_rate']:.2f}%)")
        print(f"  Kept <= 24 bytes: {res['kept_count']} (Conjunct: {res['kept_conj_rate']:.2f}%, Avg Len: {res['kept_avg_bn']:.2f})")
        print(f"  Dropped > 24 bytes: {res['dropped_count']} (Conjunct: {res['dropped_conj_rate']:.2f}%, Avg Len: {res['dropped_avg_bn']:.2f})")

if __name__ == '__main__':
    main()
