# Pragyan Phase 1 Step 7: Joint Translation Unit N-Gram Language Model Report

## 1. Execution Metadata & Git Provenance
- **Parent Commit SHA**: `74c9571`
- **Command Executed**: `cargo run --release -p lekhani-core --bin train_unit_lm`
- **Lexicon**: `data/dictionaries/dictionary.json` (Hermetic SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`)
- **Dataset Splits (Filters OFF Policy, Zero Drops)**:
  - **Train**: 94,546 pairs (0 dropped, 100.0% kept)
  - **Dev**: 9,279 pairs (0 dropped, 100.0% kept)
  - **Test**: 9,228 pairs (Strictly gated; zero evaluation on test during development)
- **Viterbi Aligned Training Sequences**: 93,756 valid sequences (1,434 vocabulary tokens including boundary markers `<s>` and `</s>`).
- **Smoothing**: Witten-Bell backoff smoothing.
- **Decoding**: Standalone Joint N-Gram Beam Search ($K = 16$). **Pure sequence model alone** (Zero dictionary bias, zero Avro heuristics, zero external unigrams).

---

## 2. Weighting Scheme Comparison on Held-Out DEV ($N = 9,279$)

Evaluated at Order = 3, Pruning Level = 0 (Count $\ge 1$):

| Weighting Scheme | DEV Top-1 (%) | DEV Top-3 (%) | Reachability (%) | File Size (KB) |
|:---|:---:|:---:|:---:|:---:|
| **Row-Level (Dakshina counts)** | 36.31% | 56.83% | **70.05%** | 1,497.9 KB |
| **Type-Level (Uniform 1.0) [Selected]** | **36.39%** | **57.08%** | 69.67% | 1,497.9 KB |

- **Decision on DEV**: Selected **Type-Level weighting**. Weighting each unique phonetic pair equally prevents overfitting to high-frequency head words and generalizes better across phonetically diverse Bengali words.

---

## 3. Work Order Step 7 Grid: Order $\times$ Pruning Level on Held-Out DEV ($N = 9,279$)

| Order | Prune Level | Description | Weighting | File Size (KB) | DEV Top-1 (%) | DEV Top-3 (%) | Reachability (%) | p95 Latency (µs) |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **3** | Level 0 | Count $\ge 1$ (Unpruned) | Type-Level | **1,497.9** | **36.39%** | 57.08% | 69.67% | 20,022.0 µs |
| **3** | Level 1 | Count $\ge 2$ (Pruned singletons) | Type-Level | **1,013.6** | **35.60%** | 55.89% | 69.12% | 17,576.0 µs |
| **3** | Level 2 | Count $\ge 5$ (Aggressive prune) | Type-Level | **566.8** | **31.77%** | 51.10% | 63.90% | 25,343.0 µs |
| **4** | Level 0 | Count $\ge 1$ (Unpruned) | Type-Level | **5,176.7** | **41.52%** | **60.61%** | **70.10%** | 18,544.0 µs |
| **4** | Level 1 | Count $\ge 2$ (Pruned singletons) [Optimal] | Type-Level | **3,067.5** | **41.21%** | **59.97%** | **69.60%** | 19,196.0 µs |
| **4** | Level 2 | Count $\ge 5$ (Aggressive prune) | Type-Level | **1,318.5** | **33.51%** | 50.57% | 59.98% | 20,395.0 µs |
| **5** | Level 0 | Count $\ge 1$ (Unpruned) | Type-Level | **12,110.9** | **41.34%** | 60.47% | 69.07% | 22,480.0 µs |
| **5** | Level 1 | Count $\ge 2$ (Pruned singletons) | Type-Level | **6,231.8** | **40.49%** | 58.70% | 67.40% | 21,114.0 µs |
| **5** | Level 2 | Count $\ge 5$ (Aggressive prune) | Type-Level | **2,109.4** | **31.91%** | 48.14% | 57.01% | 21,294.0 µs |

---

## 4. Key Findings & Architectural Insights

1. **Resolution of the Step 4 Model Error**:
   - In Step 4, unigram decoding achieved only **7.73%** Top-1 because individual units lacked sequence memory, resulting in 97.8% Model Error.
   - Moving from unigram to **4-gram Joint Unit LM** increases Top-1 from **7.73% to 41.52%** (+33.79% absolute gain) and Reachability from **19.00% to 70.10%** (+51.10% absolute gain).
   - Sequence context successfully arbitrates ambiguous vowels (`a` $\rightarrow$ `া` vs `অ` vs $\varepsilon$) and consonants (`s` $\rightarrow$ `স` vs `শ`).
2. **Order 4 vs Order 5**:
   - Order 4 achieves **41.52%** Top-1.
   - Order 5 achieves **41.34%** Top-1 while more than doubling the model size (12.1 MB vs 5.1 MB). Order 4 is the clear sweet spot.
3. **Pruning & Gate G6 Budget Compliance**:
   - Pruning Level 1 (Count $\ge 2$ on higher orders) preserves **41.21%** Top-1 (within 0.31% of unpruned) while shrinking serialized binary footprint to **3,067.5 KB (3.06 MB)**.
   - This strictly complies with the Gate G6 budget constraint ($\le 3.5$ MB).

---

## 5. Gate G1 Formal Verification

| Requirement | Threshold | Measured Result | Status |
|:---|:---:|:---:|:---:|
| **Gate G1: N-Gram Alone > Raw Avro on DEV** | $> 12.37\%$ | **41.52%** (Order 4 Level 0)<br>**41.21%** (Order 4 Level 1) | **PASSED (+29.15% points over baseline)** |

---

## 6. Next Step (Step 8: Log-Linear Decoder)
Integrate the 4-gram Joint Translation Unit LM into the multi-feature log-linear decoder:
$$\text{Score}(W) = \ln P_{\text{JSC}}(W) + \lambda_1 \ln P_{\text{lex}}(W) + \lambda_2 \cdot \mathbf{1}_{W \in \text{dict}} + \lambda_3 \cdot \mathbf{1}_{W == \text{Avro}(R)}$$
Tuning $\lambda_1, \lambda_2, \lambda_3$ on held-out DEV to lift Top-1 accuracy toward the 70–80%+ target.
