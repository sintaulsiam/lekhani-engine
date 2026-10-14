# Pragyan Phase 1 Step 2: Data Accounting Report

## 1. Execution Metadata & Git Provenance
- **Command(s) Run**:
  - `cargo run --release -p lekhani-core --bin data_accounting`
  - `python3 tools/data_accounting.py`
  - `cargo test --release --test phonetic_eval`
- **Parent Commit SHA**: `61b565f`
- **Tree Hash**: `004edc4eab074533bb149251e334b57184578ecb`
- **Dictionary**: `data/dictionaries/dictionary.json` (SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`)

---

## 2. Split Dimensions & Conjunct Distribution (Filters OFF)

| Split | Raw Rows | Unique Targets | Unique Romanizations | Avg Target Chars | Avg Roman Chars | Rows with Conjuncts | Conjunct Rate (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TRAIN** | 94,546 | 25,000 | 89,381 | 7.24 | 8.31 | 48,854 | 51.67% |
| **DEV** | 9,279 | 2,500 | 9,200 | 6.88 | 7.96 | 4,298 | 46.32% |
| **TEST** | 9,228 | 2,500 | 9,146 | 6.88 | 8.00 | 4,276 | 46.34% |

- **Target Disjointness**: Train, dev, and test sets are 100% disjoint at the target word level (0 target words shared between train and dev/test).
- **Conjunct Count (Juktoborno)**:
  - Train: **48,854** rows (51.67%)
  - Dev: **4,298** rows (46.32%)
  - Test: **4,276** rows (46.34%)

---

## 3. Filter Accounting: Exact Rows Flagged per Filter Rule

| Filter Rule | TRAIN Flagged | DEV Flagged | TEST Flagged | Notes |
|:---|:---:|:---:|:---:|:---|
| Empty field / malformed columns | 0 | 0 | 0 | Zero corrupted lines across all files |
| Non-ASCII Roman characters | 0 | 0 | 0 | All Roman strings are strictly `[a-zA-Z]` |
| Non-Bengali script characters | 0 | 0 | 0 | All target characters within `U+0980..U+09FF` or ZWJ |
| Canonicalization discrepancy | 0 | 0 | 0 | All targets match standard canonical Unicode |
| Roman character length > 24 | 0 | 0 | 0 | Max Roman length is 22 chars (train), 20 (dev/test) |
| Bengali character length > 24 | 0 | 0 | 0 | Max Bengali length is 22 chars (train), 17 (dev/test) |
| **UTF-8 Byte Length > 24 (`bn.len() > 24`)** | **28,178** | **2,129** | **2,125** | **Byte length vs. character length truncation** |

---

## 4. Root Cause Analysis: The 66,368 of 94,546 Train Rows Drop

In earlier exploratory alignment code (`align_m2m.rs`), the filter `bn.len() <= 24` was evaluated on Rust `String`. In Rust, `.len()` returns length in **bytes**, not Unicode characters. 

Because Bengali characters in UTF-8 are 3 bytes each (e.g. `ক` = 3 bytes), `bn.len() <= 24` truncated all words with more than 8 characters ($24 / 3 = 8$).

### Systemic Hardness of Byte-Truncated Rows

| Split | Partition | Rows (%) | Avg BN Chars | Avg Roman Chars | Conjunct Rate (%) | Relative Hardness |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **TRAIN** | Kept (`<= 24` bytes) | 66,368 (70.2%) | 5.85 | 7.03 | **36.75%** | Shorter, simpler syllables |
| | **Dropped (`> 24` bytes)** | **28,178 (29.8%)** | **10.53** | **11.32** | **86.83%** | **Systematically harder (86.83% conjuncts)** |
| | Full Split (Filters OFF) | 94,546 (100.0%) | 7.24 | 8.31 | 51.67% | Authentic full training set |
| **DEV** | Kept (`<= 24` bytes) | 7,150 (77.1%) | 5.89 | 7.03 | **35.90%** | Shorter, simpler syllables |
| | **Dropped (`> 24` bytes)** | **2,129 (22.9%)** | **10.23** | **11.08** | **81.31%** | **Systematically harder (81.31% conjuncts)** |
| | Full Split (Filters OFF) | 9,279 (100.0%) | 6.88 | 7.96 | 46.32% | Authentic full dev set |
| **TEST** | Kept (`<= 24` bytes) | 7,103 (77.0%) | 5.88 | 7.11 | **35.49%** | Shorter, simpler syllables |
| | **Dropped (`> 24` bytes)** | **2,125 (23.0%)** | **10.24** | **11.01** | **82.59%** | **Systematically harder (82.59% conjuncts)** |
| | Full Split (Filters OFF) | 9,228 (100.0%) | 6.88 | 8.00 | 46.34% | Authentic full test set |

**Conclusion**: The dropped rows were systematically harder (average character length: 10.53 vs. 5.85; conjunct rate: 86.83% vs. 36.75%). Truncating them would deprive the Step 3 tokenization and Step 4 alignment models of complex conjunct patterns.

---

## 5. Filtering Decision for Phase 1

- **Rule Applied**: **Filters OFF (100% Inclusion)**.
- Dev and test are filtered by the **exact same rule as train: no rows removed**. All 94,546 train pairs, 9,279 dev pairs, and 9,228 test pairs are clean, canonical, and fully retained.

---

## 6. What Is Not Verified
- Unaligned full sentence text (e.g. BanglaTLit / BanTH): requires sentence-to-word token alignment before word-level metrics apply.
- Performance of Step 3 tokenization on words with $> 2$ consecutive conjunct clusters (e.g. `সংকীর্ণতা`, `মনস্তাত্ত্বিক`), which will be tested in Step 3.

---

## 7. Gate Status
- **Regression Gates (CI)**: **PASS** (4/4 tests passed in 6.17s).
- **Dictionary SHA-256 Gate**: **PASS** (verified against `baseline.json`, tamper test confirmed).
