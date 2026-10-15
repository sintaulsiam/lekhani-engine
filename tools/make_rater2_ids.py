#!/usr/bin/env python3
"""
Picks 50 ids uniformly at random from the 200 audit rows with a fixed seed (42).
Writes the IDs (one per line) to data/eval/align_audit_rater2_ids.txt.
"""

import random
from pathlib import Path

def main():
    rng = random.Random(42)
    # 200 rows with IDs 1 to 200
    all_ids = list(range(1, 201))
    rater2_ids = sorted(rng.sample(all_ids, 50))
    
    out_path = Path("data/eval/align_audit_rater2_ids.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row_id in rater2_ids:
            f.write(f"{row_id}\n")
            
    print(f"Generated {len(rater2_ids)} IDs in {out_path} (seed=42):")
    print(rater2_ids)

if __name__ == "__main__":
    main()
