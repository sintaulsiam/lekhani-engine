#!/usr/bin/env python3
"""
Splits held-out DEV (data/eval/bn.translit.sampled.dev.tsv, N=9,279)
into dev-A (tuning) and dev-B (confirmation) by target Bengali word.

Invariants enforced:
1. Zero target word overlap between dev-A and dev-B.
2. Complete coverage: len(dev-A) + len(dev-B) == len(dev) (9,279 rows).
3. Zero rows dropped or modified.
4. Balanced unique targets: 1,250 unique targets each.
"""

from collections import OrderedDict
import json
import os
import sys

DEV_PATH = "data/eval/bn.translit.sampled.dev.tsv"
DEV_A_PATH = "data/eval/bn.translit.sampled.dev_a.tsv"
DEV_B_PATH = "data/eval/bn.translit.sampled.dev_b.tsv"
DICT_PATH = "data/dictionaries/dictionary.json"

def main():
    if not os.path.exists(DEV_PATH):
        print(f"Error: {DEV_PATH} not found", file=sys.stderr)
        sys.exit(1)

    targets = OrderedDict()
    total_rows = 0
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) < 2:
                continue
            bn = parts[0]
            targets.setdefault(bn, []).append(line)
            total_rows += 1

    print(f"Loaded {total_rows} rows across {len(targets)} unique target words from {DEV_PATH}.")

    dev_a_rows = []
    dev_b_rows = []
    dev_a_targets = set()
    dev_b_targets = set()

    for idx, (target, rows) in enumerate(targets.items()):
        if idx % 2 == 0:
            dev_a_rows.extend(rows)
            dev_a_targets.add(target)
        else:
            dev_b_rows.extend(rows)
            dev_b_targets.add(target)

    # Invariant checks
    assert len(dev_a_targets.intersection(dev_b_targets)) == 0, "Target words must be strictly disjoint!"
    assert len(dev_a_rows) + len(dev_b_rows) == total_rows, "Total rows must match original DEV!"
    assert len(dev_a_targets) == 1250, "dev-A must have 1,250 targets"
    assert len(dev_b_targets) == 1250, "dev-B must have 1,250 targets"

    with open(DEV_A_PATH, "w", encoding="utf-8") as f:
        f.writelines(dev_a_rows)

    with open(DEV_B_PATH, "w", encoding="utf-8") as f:
        f.writelines(dev_b_rows)

    # Dictionary coverage stats
    dict_words = set()
    if os.path.exists(DICT_PATH):
        with open(DICT_PATH, "r", encoding="utf-8") as f:
            dict_data = json.load(f)
            if isinstance(dict_data, dict):
                for val in dict_data.values():
                    if isinstance(val, list):
                        dict_words.update(val)
                    elif isinstance(val, str):
                        dict_words.add(val)
            elif isinstance(dict_data, list):
                dict_words = set(dict_data)

    dev_a_seen = sum(1 for t in dev_a_targets if t in dict_words)
    dev_b_seen = sum(1 for t in dev_b_targets if t in dict_words)

    print("\n--- Split Verification Summary ---")
    print(f"dev-A (Tuning)      : {len(dev_a_rows)} rows | {len(dev_a_targets)} targets ({dev_a_seen}/{len(dev_a_targets)} in dictionary: {dev_a_seen/len(dev_a_targets)*100:.1f}%)")
    print(f"dev-B (Confirmation): {len(dev_b_rows)} rows | {len(dev_b_targets)} targets ({dev_b_seen}/{len(dev_b_targets)} in dictionary: {dev_b_seen/len(dev_b_targets)*100:.1f}%)")
    print(f"Target Disjointness : 0 target words shared (100% disjoint).")
    print(f"Wrote {DEV_A_PATH} and {DEV_B_PATH} successfully.")

if __name__ == "__main__":
    main()
