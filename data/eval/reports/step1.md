# Step 1 Report: Re-baseline on the Real Engine Configuration

## 1. Commands Run and Commit SHA

- **Commit SHA**: `7c025c45`
- **Commands Run**:
  ```bash
  # Step 1 evaluation on Dev and Test splits with explicit dictionary.json
  cargo run --release -p lekhani-core --bin eval_phonetic

  # Verification of CI regression gate
  cargo test --release --test phonetic_eval
  ```
- **Configuration & Provenance**:
  - Engine Dictionary: `data/dictionaries/dictionary.json`
  - Dictionary SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`
  - Canonicalization: NFC + decomposed nuktas (`U+09BC`) + atomic Khanda Ta (`U+09CE`)
  - Target Clusters: 2,500 words (1,000 cluster bootstrap iterations)
  - Random Seed: 42

---

## 2. Raw Result Tables

### 2.1 DEV Split Baseline ($N = 9,279$, Unique Targets = 2,500)
Raw Avro Parser Top-1: **12.37%**

| Persona | N | Top-1 (Exact) | Top-1 95% Cluster CI | Any Attested | Top-3 | Top-5 | MRR | Not In Top-5 | Unreachable | KSPC (Eff / Hit) | Latency p95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `natural` | 9,279 | 24.9% | [23.7, 26.1] | 25.2% | 32.5% | 33.6% | 0.2869 | 66.4% | 65.9% | 1.99 / 1.30 | 1,560 µs |
| `lazy` | 9,279 | 26.1% | [24.8, 27.4] | 26.0% | 34.2% | 35.4% | 0.3014 | 64.6% | 64.0% | 1.94 / 1.27 | 1,454 µs |
| `vowel_drop_p30` | 9,279 | 13.5% | [12.7, 14.4] | 13.5% | 17.9% | 18.7% | 0.1572 | 81.3% | 81.0% | 2.00 / 1.27 | 1,290 µs |
| `vowel_drop_p60` | 9,279 | 6.2% | [5.6, 6.8] | 6.2% | 8.5% | 8.9% | 0.0730 | 91.1% | 90.9% | 1.96 / 1.24 | 1,045 µs |
| `vowel_drop_p90` | 9,279 | 2.1% | [1.7, 2.5] | 2.1% | 2.9% | 3.1% | 0.0249 | 96.9% | 96.9% | 1.87 / 1.17 | 715 µs |
| `avro_strict` | 9,279 | 90.3% | [88.8, 91.8] | 90.3% | 94.8% | 95.7% | 0.9257 | 4.3% | 3.6% | 1.32 / 1.27 | 786 µs |

### 2.2 TEST Split Baseline ($N = 9,228$, Unique Targets = 2,500)
Raw Avro Parser Top-1: **12.72%**

| Persona | N | Top-1 (Exact) | Top-1 95% Cluster CI | Any Attested | Top-3 | Top-5 | MRR | Not In Top-5 | Unreachable | KSPC (Eff / Hit) | Latency p95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `natural` | 9,228 | 25.3% | [24.2, 26.5] | 25.7% | 32.8% | 33.8% | 0.2901 | 66.2% | 65.6% | 1.99 / 1.30 | 1,561 µs |
| `lazy` | 9,228 | 27.2% | [25.8, 28.5] | 27.4% | 34.9% | 36.1% | 0.3101 | 63.9% | 63.1% | 1.94 / 1.27 | 1,520 µs |
| `vowel_drop_p30` | 9,228 | 13.2% | [12.4, 14.0] | 13.2% | 17.6% | 18.3% | 0.1542 | 81.7% | 81.1% | 2.01 / 1.28 | 1,291 µs |
| `vowel_drop_p60` | 9,228 | 6.4% | [5.8, 7.0] | 6.4% | 8.8% | 9.4% | 0.0764 | 90.6% | 90.3% | 1.96 / 1.23 | 1,051 µs |
| `vowel_drop_p90` | 9,228 | 2.0% | [1.7, 2.5] | 2.0% | 2.8% | 3.1% | 0.0244 | 96.9% | 96.7% | 1.87 / 1.16 | 734 µs |
| `avro_strict` | 9,228 | 90.9% | [89.6, 92.3] | 90.9% | 94.3% | 95.3% | 0.9265 | 4.7% | 4.2% | 1.33 / 1.28 | 787 µs |

---

## 3. What Changed from the Previous Number

| Metric (Test Split) | Previous (`dictionary.bin` with Wikipedia) | Step 1 (`dictionary.json` root lexicon only) | Difference |
| :--- | :---: | :---: | :---: |
| `natural` Top-1 | 30.83% [29.57, 32.17] | **25.28% [24.16, 26.51]** | -5.55% |
| `lazy` Top-1 | 31.81% [30.41, 33.15] | **27.18% [25.82, 28.53]** | -4.63% |
| `vowel_drop_p30` Top-1 | 14.77% [13.91, 15.68] | **13.20% [12.40, 14.03]** | -1.57% |
| `vowel_drop_p60` Top-1 | 6.74% [6.12, 7.37] | **6.40% [5.82, 7.02]** | -0.34% |
| `vowel_drop_p90` Top-1 | 2.06% [1.72, 2.47] | **2.05% [1.70, 2.46]** | -0.01% |
| `avro_strict` Top-1 | 94.08% [93.08, 95.13] | **90.89% [89.58, 92.29]** | -3.19% |
| Unseen Share | 12.7% (target in test) | **54.0% (target in test)** | +41.3% |

**Explanation**: 
The 5.55% difference confirms that `dictionary.bin` contained 66,000 Wikipedia-derived unigrams that artificially overlapped with the Dakshina test set vocabulary (raising dictionary target coverage from 46.0% to 87.3%). By enforcing `dictionary.json` (159,381 OpenBangla roots) across both the suggestion engine and evaluation, test set contamination is eliminated. The true baseline Top-1 on natural human typing is **25.28%** (cluster CI $[24.16\%, 26.51\%]$).

---

## 4. What is Not Verified

- **Downstream Learned Grapheme Performance**: The statistical language model (Steps 3–8) has not yet been integrated into this baseline.
- **External G2P Baseline**: Phonetisaurus/Sequitur external baseline on these splits is scheduled for Step 6.

---

## 5. Gate Status

- **CI Regression Gate**: **PASS**
  - Verification: `cargo test --release --test phonetic_eval`
  - Output: `Dakshina Baseline Gate Verification (N=9228): Top-1 = 25.28% (threshold >= 24.16%), p95 = 1561.0µs (threshold <= 1873.2µs)`
  - Status: 3 passed, 0 failed.
