#!/usr/bin/env python3
"""
Auditing Analysis Script for Lekhani M2M Alignment Audit.

Reads data/eval/align_audit_rater1.csv (and optionally data/eval/align_audit_rater2.csv).
Outputs:
1. Counts and percentages of Correct / Incorrect / Unsure overall and by has_conjunct.
2. Wilson 95% confidence intervals for the Correct rate:
   - Treating 'Unsure' as Not Correct (conservative)
   - With 'Unsure' excluded
3. Error-type breakdown.
4. If rater 2 CSV is supplied or exists:
   - Agreement % on overlapping rows
   - Cohen's Kappa coefficient
   - List of IDs where raters disagreed

READ-ONLY: This script NEVER modifies files or assigns grades.
"""

import csv
import math
import sys
from pathlib import Path

def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculates Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95% two-sided normal quantile
    p = k / n
    denominator = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denominator
    margin = (z / denominator) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower * 100.0, upper * 100.0

def load_audit_csv(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    results = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_id = int(row.get("id", 0))
            except ValueError:
                continue
            results[row_id] = row
    return results

def get_grade(r: dict) -> str:
    return (r.get("grade") or r.get("alignment_grade") or "").strip().upper()

def compute_stats_for_rater(rows: list[dict], rater_name: str):
    total = len(rows)
    if total == 0:
        print(f"No rows found for {rater_name}.")
        return

    grades = [get_grade(r) for r in rows]
    correct_count = sum(1 for g in grades if g == "CORRECT")
    incorrect_count = sum(1 for g in grades if g == "INCORRECT")
    unsure_count = sum(1 for g in grades if g == "UNSURE")
    ungraded_count = sum(1 for g in grades if g not in {"CORRECT", "INCORRECT", "UNSURE"})

    graded_total = correct_count + incorrect_count + unsure_count

    print(f"\n{'='*75}")
    print(f"   📊 AUDIT STATISTICS: {rater_name.upper()} (N = {total} rows)")
    print(f"{'='*75}")
    print(f"Overall Grades:")
    print(f"  - Correct:   {correct_count:3d} ({correct_count/total*100:6.2f}%)")
    print(f"  - Incorrect: {incorrect_count:3d} ({incorrect_count/total*100:6.2f}%)")
    print(f"  - Unsure:    {unsure_count:3d} ({unsure_count/total*100:6.2f}%)")
    if ungraded_count > 0:
        print(f"  - Ungraded:  {ungraded_count:3d} ({ungraded_count/total*100:6.2f}%)")

    # Wilson 95% Confidence Intervals
    w_low_all, w_high_all = wilson_interval(correct_count, total)
    print(f"\nWilson 95% Confidence Intervals for Correct Rate:")
    print(f"  - Conservative (Unsure as not correct, N={total}): {correct_count/total*100:.2f}% [{w_low_all:.2f}%, {w_high_all:.2f}%]")
    
    decided = correct_count + incorrect_count
    if decided > 0:
        w_low_dec, w_high_dec = wilson_interval(correct_count, decided)
        print(f"  - Conditional  (Unsure excluded, N={decided}):       {correct_count/decided*100:.2f}% [{w_low_dec:.2f}%, {w_high_dec:.2f}%]")

    # Stratified by Conjunct
    conj_rows = [r for r in rows if str(r.get("has_conjunct", "")).lower() == "true"]
    non_conj_rows = [r for r in rows if str(r.get("has_conjunct", "")).lower() == "false"]

    print(f"\nStratified Breakdown by Conjunct:")
    for label, subset in [("With Conjunct (যুক্তবর্ণ)", conj_rows), ("Without Conjunct", non_conj_rows)]:
        n_sub = len(subset)
        if n_sub == 0:
            continue
        c_sub = sum(1 for r in subset if get_grade(r) == "CORRECT")
        i_sub = sum(1 for r in subset if get_grade(r) == "INCORRECT")
        u_sub = sum(1 for r in subset if get_grade(r) == "UNSURE")
        low, high = wilson_interval(c_sub, n_sub)
        print(f"  ├─ {label:<28} (N={n_sub:3d}): Correct: {c_sub:2d} ({c_sub/n_sub*100:5.1f}%) [CI: {low:.1f}-{high:.1f}%] | Incorr: {i_sub:2d} | Unsure: {u_sub:2d}")

    # Error type breakdown
    error_types = {}
    for r in rows:
        err = r.get("error_type", "").strip()
        g = get_grade(r)
        if g in {"INCORRECT", "UNSURE"} and err:
            error_types[err] = error_types.get(err, 0) + 1

    if error_types:
        print(f"\nError Type Breakdown (Incorrect / Unsure items):")
        for err, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  - {err:<22}: {count:2d} ({count/total*100:5.1f}%)")

def compute_inter_rater_stats(r1_map: dict[int, dict], r2_map: dict[int, dict]):
    shared_ids = sorted(set(r1_map.keys()) & set(r2_map.keys()))
    if not shared_ids:
        return

    n_shared = len(shared_ids)
    print(f"\n{'='*75}")
    print(f"   🤝 INTER-RATER AGREEMENT (N = {n_shared} Overlapping Pairs)")
    print(f"{'='*75}")

    categories = ["CORRECT", "INCORRECT", "UNSURE"]
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    matrix = [[0] * len(categories) for _ in range(len(categories))]

    agreements = 0
    disagreements = []

    for row_id in shared_ids:
        g1 = get_grade(r1_map[row_id])
        g2 = get_grade(r2_map[row_id])

        if g1 in cat_to_idx and g2 in cat_to_idx:
            matrix[cat_to_idx[g1]][cat_to_idx[g2]] += 1

        if g1 == g2:
            agreements += 1
        else:
            disagreements.append((
                row_id,
                r1_map[row_id].get("bengali_word", ""),
                r1_map[row_id].get("roman_input", ""),
                g1,
                g2,
                r1_map[row_id].get("error_type", ""),
                r2_map[row_id].get("error_type", ""),
            ))

    po = agreements / n_shared

    # Marginal probabilities for Cohen's Kappa
    r1_marginals = [sum(matrix[i][j] for j in range(3)) / n_shared for i in range(3)]
    r2_marginals = [sum(matrix[i][j] for i in range(3)) / n_shared for j in range(3)]
    pe = sum(r1_marginals[k] * r2_marginals[k] for k in range(3))

    kappa = 1.0 if pe == 1.0 else (po - pe) / (1.0 - pe) if pe < 1.0 else 0.0

    print(f"Overall Agreement Rate : {agreements} / {n_shared} ({po * 100.0:.2f}%)")
    print(f"Expected Agreement (Pe): {pe * 100.0:.2f}%")
    print(f"Cohen's Kappa (κ)      : {kappa:.4f}")

    if disagreements:
        print(f"\nDisagreement Cases ({len(disagreements)} pairs):")
        print(f"  {'ID':<4} {'Bengali':<12} {'Roman':<12} {'Rater 1':<10} {'Rater 2':<10} {'Notes'}")
        print(f"  {'-'*65}")
        for d in disagreements:
            print(f"  {d[0]:<4} {d[1]:<12} {d[2]:<12} {d[3]:<10} {d[4]:<10} R1:{d[5]} vs R2:{d[6]}")
    else:
        print("\n✓ Perfect agreement on all overlapping rows (Zero disagreements).")

def main():
    r1_path = Path("data/eval/align_audit_rater1.csv")
    r2_path = Path("data/eval/align_audit_rater2.csv")

    if len(sys.argv) >= 2:
        r1_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        r2_path = Path(sys.argv[2])

    if not r1_path.exists():
        print(f"Rater 1 audit CSV not found at: {r1_path}")
        print("Usage: python3 tools/audit_stats.py [rater1_csv] [rater2_csv]")
        sys.exit(1)

    r1_data = load_audit_csv(r1_path)
    compute_stats_for_rater(list(r1_data.values()), r1_path.stem)

    if r2_path.exists():
        r2_data = load_audit_csv(r2_path)
        compute_stats_for_rater(list(r2_data.values()), r2_path.stem)
        compute_inter_rater_stats(r1_data, r2_data)

if __name__ == "__main__":
    main()
