# Pragyan Phase 1 Step 7b: Rigorous Verification of Joint Translation Unit LM

## 1. Execution Metadata & Provenance
- **Command Executed**: `cargo run --release -p lekhani-core --bin verify_unit_lm_7b`
- **Lexicon**: `data/dictionaries/dictionary.json` (Hermetic SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`)
- **Dataset Splits (Filters OFF Policy, Zero Drops on Load)**:
  - **Train**: 94,546 pairs (0 dropped on load, 100.0% kept)
  - **Dev**: 9,279 pairs (0 dropped on load, 100.0% kept)
  - **Test**: 9,228 pairs (Strictly gated; zero evaluation on test during development)
- **Viterbi Aligned Training Sequences**: 93,756 valid sequences (1,434 vocabulary tokens including boundary markers `<s>` and `</s>`).
- **Accounting of Unaligned Train Rows (790 rows, 0.84%)**:
  - *Zero-probability co-occurrence lattice under 1:1 constraints*: 789 rows (99.87%) — primarily acronyms/initialisms (e.g. `আইআইজি` $\leftrightarrow$ `iig`, `আইআইটি` $\leftrightarrow$ `iit`, `আরডিআরএস` $\leftrightarrow$ `rdrs`), extreme double vowels/glides (`ইউনিভার্সিটি` $\leftrightarrow$ `iiuniibharsiitii`), and non-standard loan transliterations where no valid 1:1 monotone path has non-zero probability under the trained unit lexicon.
  - *Extreme token length disparity (Bengali >> Roman)*: 1 row (0.13%).
- **Smoothing**: Witten-Bell backoff smoothing.
- **Evaluation Status**: Gate G1 passed on dev, unverified pending Step 8 combined decoder.

---

## 2. 7b Item 1: Exact Scoring Comparability on Held-Out DEV ($N = 9,279$)

Evaluated on exact held-out DEV pairs using the identical scoring harness and canonicalization as `eval_phonetic`:

| Metric | Hits / Total | Accuracy (%) | Old Baseline Engine (%) | Absolute Gain |
|:---|:---:|:---:|:---:|:---:|
| **Exact-Row Top-1** | 3,857 / 9,279 | **41.57%** | 24.86% (2,307/9,279) | **+16.71%** |
| **Most-Frequent Attested Top-1** | 3,861 / 9,279 | **41.61%** | — | — |
| **Any-Attested Top-1** | 3,908 / 9,279 | **42.12%** | 25.10% | **+17.02%** |
| **Exact-Row Top-3** | 5,609 / 9,279 | **60.45%** | 39.42% | **+21.03%** |
| **Beam Reachability (Gold in Top-16)** | 6,501 / 9,279 | **70.06%** | — | — |

---

## 3. Gate G6 Latency Audit: Severe Failure Against Budget

A single-threaded latency benchmark was conducted on 1,000 sequential DEV words ($K = 16$):

| Latency Metric | Measured Value (µs) | Gate G6 Budget | Status | Comparison vs Baseline Engine |
|:---|:---:|:---:|:---:|:---:|
| **Mean** | 7,873.2 µs | $\le 3,000$ µs | — | — |
| **p50** | 7,362 µs | $\le 3,000$ µs | — | $\approx 4.9\times$ slower |
| **p90** | 13,342 µs | $\le 3,000$ µs | — | — |
| **p95** | **14,902 µs** | $\le 3,000$ µs | **FAILED (5.0× over budget)** | Baseline engine is $\approx 1,500$ µs |
| **p99** | 19,413 µs | — | — | — |
| **Max** | 26,415 µs | — | — | — |

- **Profile of 7+ Character Conjunct Word**: `"অংশগ্রহন"` (`"angshogragon"`, 8 chars) averaged **9,441.1 µs** per decode (100 iterations).
- **Root Cause Diagnosis**: Dynamic heap allocations on the hot decoding path (cloning `Vec<UnitId>` histories and `String` candidates, plus `HashMap` rehashing during candidate deduplication at every beam step).
- **Audit Verdict**: While serialized model size ($\le 3.5\text{ MB}$) passes G6, **latency decisively fails Gate G6**. Latency optimization is deferred to Step 8 after accuracy calibration, where a flat pre-allocated beam buffer and zero-allocation score tracing must be implemented.

---

## 4. 7b Item 2: Cluster-Bootstrap 95% Confidence Intervals (1,000x Resampling by Target)

To test whether differences between Order 4 vs 5, Type-Level vs Row-Level, and Pruned vs Unpruned are statistically significant or inside the noise, 1,000 cluster-bootstrap resamples by target Bengali word cluster were computed across all 18 grid configurations:

| Weighting | Order | Prune Level | Description | Model Size (KB) | DEV Top-1 (%) | Cluster-Bootstrap 95% CI | DEV Top-3 (%) | Reachability (%) |
|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| Type-Level | 3 | Level 0 | Count $\ge 1$ (Unpruned) | 1,497.9 | 36.39% | [34.75%, 38.17%] | 57.08% | 69.67% |
| Type-Level | 3 | Level 1 | Count $\ge 2$ (Pruned singletons) | 1,013.6 | 35.60% | [34.01%, 37.28%] | 55.89% | 69.12% |
| Type-Level | 3 | Level 2 | Count $\ge 5$ (Aggressive prune) | 566.8 | 31.77% | [30.29%, 33.33%] | 51.10% | 63.90% |
| Type-Level | 4 | Level 0 | Count $\ge 1$ (Unpruned) | 5,176.7 | 41.52% | [39.83%, 43.28%] | 60.61% | 70.10% |
| **Type-Level** | **4** | **Level 1** | **Count $\ge 2$ (Pruned singletons)** | **3,067.5** | **41.21%** | **[39.54%, 42.87%]** | **59.97%** | **69.60%** |
| Type-Level | 4 | Level 2 | Count $\ge 5$ (Aggressive prune) | 1,318.5 | 33.51% | [32.08%, 34.98%] | 50.57% | 59.98% |
| Type-Level | 5 | Level 0 | Count $\ge 1$ (Unpruned) | 12,110.9 | 41.34% | [39.70%, 43.02%] | 60.47% | 69.07% |
| Type-Level | 5 | Level 1 | Count $\ge 2$ (Pruned singletons) | 6,231.8 | 40.49% | [38.78%, 42.20%] | 58.70% | 67.40% |
| Type-Level | 5 | Level 2 | Count $\ge 5$ (Aggressive prune) | 2,109.4 | 31.91% | [30.46%, 33.43%] | 48.14% | 57.01% |
| Row-Level | 3 | Level 0 | Count $\ge 1$ (Unpruned) | 1,497.9 | 36.31% | [34.65%, 38.08%] | 56.83% | 70.05% |
| Row-Level | 3 | Level 1 | Count $\ge 2$ (Pruned singletons) | 1,076.3 | 35.91% | [34.30%, 37.57%] | 55.99% | 69.74% |
| Row-Level | 3 | Level 2 | Count $\ge 5$ (Aggressive prune) | 636.8 | 32.35% | [30.87%, 33.94%] | 51.40% | 65.22% |
| Row-Level | 4 | Level 0 | Count $\ge 1$ (Unpruned) | 5,176.7 | 41.39% | [39.74%, 43.10%] | 60.84% | 70.23% |
| **Row-Level** | **4** | **Level 1** | **Count $\ge 2$ (Pruned singletons) [Best]** | **3,427.1** | **41.57%** | **[39.92%, 43.31%]** | **60.45%** | **70.06%** |
| Row-Level | 4 | Level 2 | Count $\ge 5$ (Aggressive prune) | 1,630.1 | 34.67% | [33.17%, 36.21%] | 51.99% | 62.29% |
| Row-Level | 5 | Level 0 | Count $\ge 1$ (Unpruned) | 12,110.9 | 40.96% | [39.29%, 42.63%] | 60.11% | 68.75% |
| Row-Level | 5 | Level 1 | Count $\ge 2$ (Pruned singletons) | 7,421.6 | 41.01% | [39.27%, 42.69%] | 59.82% | 68.40% |
| Row-Level | 5 | Level 2 | Count $\ge 5$ (Aggressive prune) | 2,850.8 | 33.01% | [31.52%, 34.44%] | 49.59% | 58.75% |

### Statistical Decisions from CIs:
1. **Order 3 vs Order 4**: Order 3 (36.31% [34.65%, 38.08%]) does NOT overlap Order 4 (41.57% [39.92%, 43.31%]). Order 4 provides a statistically genuine +5.2% gain.
2. **Order 4 vs Order 5**: Order 5 (41.01% [39.27%, 42.69%]) is completely nested inside the Order 4 CI while doubling memory footprint. Order 5 is eliminated.
3. **Type-Level vs Row-Level**: The difference between Type-Level Level 1 (41.21% [39.54%, 42.87%]) and Row-Level Level 1 (41.57% [39.92%, 43.31%]) is 0.36 points and well inside mutual 95% CIs. Both models overlap the best model.
4. **Model Selection**: **Row-Level Order 4 Level 1** achieves highest Top-1 (41.57%) and fits within G6 size budget at **3.43 MB** ($\le 3.5$ MB). The smallest overlapping model within budget is **Type-Level Order 4 Level 1** at **3.07 MB**.

---

## 5. 7b Item 3: Failure Classification across Beam Widths ($K = 16, 64, 256$)

To diagnose why Top-1 was frozen across beam widths and explain the search vs model dynamics:

| Beam Width ($K$) | DEV Top-1 (%) | DEV Top-3 (%) | Reachability (%) | Total Errors | Coverage Error | Search Error | Model Error |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K = 16$** | 41.57% | 60.45% | 70.06% | 5,422 | 126 (2.3%) | 372 (6.9%) | **4,924 (90.8%)** |
| **$K = 64$** | 42.55% | 63.13% | 81.27% | 5,331 | 126 (2.4%) | 111 (2.1%) | **5,094 (95.6%)** |
| **$K = 256$** | 42.87% | 63.78% | 87.52% | 5,301 | 126 (2.4%) | 39 (0.7%) | **5,136 (96.9%)** |

### Why Top-1 Barely Moves (+1.30%) While Reachability Surges (+17.46%):
When $K$ is expanded from 16 to 256, beam reachability climbs from **70.06% to 87.52%**, and Search Error drops from 6.9% to 0.7%. However, Top-1 accuracy only increases from 41.57% to 42.87%.
The printed scores for sample failures show why:
- In $>90\%$ of errors, $S_{\text{forced}} \le S_{\text{decoded}}$.
- The model generates the gold candidate inside the beam, but the pure character-unit sequence model assigns a higher probability to an alternative candidate that is either a non-word homophone or a variant with different vowel length.
- Because the N-gram operates strictly at the translation-unit level without a word-level dictionary prior, search expansion simply uncovers more competing unit combinations that out-score the gold word.
- **Direct Conclusion for Step 8**: Search error is negligible ($< 7\%$). The bottleneck is entirely **Model Error (90.8%)**. A word-level lexicon prior ($\lambda_1 \ln P_{\text{lex}}(W) + \lambda_2 \mathbf{1}_{W \in \text{dict}}$) is required to re-rank the candidates that are already present in the reachability set ($70.06\% \rightarrow 87.52\%$).

### Sample 20 DEV Failures (Forced vs Best Decoded Score under Selected Model):
| ID | Target | Roman | Top Decoded | Class | $S_{\text{forced}}$ | $S_{\text{decoded}}$ | Status |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|
| 1 | অংকুশ | angkush | আংকুশ | Model | -26.59 | -25.19 | FAIL |
| 2 | অংকুশ | ankush | অঙ্কুষ | Model | -22.19 | -19.87 | FAIL |
| 5 | অংশগ্রহন | angshogragon | অংশগ্রগণ | Model | -49.44 | -20.30 | FAIL |
| 6 | অংশগ্রহন | angshograhon | অংশগ্রহণ | Model | -17.44 | -16.46 | FAIL |
| 7 | অংশগ্রহন | angshogrohon | অংশগ্রহণ | Model | -15.58 | -15.11 | FAIL |
| 8 | অংশগ্রহন | angsogrohon | অংশগ্রহণ | Model | -16.46 | -15.98 | FAIL |
| 9 | অংশগ্রহন | ongshograhon | অংশগ্রহণ | Model | -17.39 | -16.41 | FAIL |
| 10 | অংশগ্রহন | ongshogrohon | অংশগ্রহণ | Model | -15.53 | -15.06 | FAIL |
| 11 | অংশগ্রহনের | angsogrohone | অংশগ্রহণে | Model | -28.67 | -17.49 | FAIL |
| 12 | অংশগ্রহনের | anshograhone | অংশগ্রহণে | Model | -32.43 | -20.74 | FAIL |
| 13 | অংশগ্রহনের | ongshogrohoner | অংশগ্রহণের | Model | -16.53 | -15.66 | FAIL |
| 14 | অংশগ্রহনের | onshogrohoner | অংশগ্রহণের | Model | -19.36 | -18.49 | FAIL |
| 15 | অকটেন | akten | আঁকতেন | Model | -26.29 | -12.63 | FAIL |
| 16 | অকটেন | octane | অকট্যানে | Model | -31.46 | -19.29 | FAIL |
| 17 | অকৃতকার্য | akreetokarjo | আকৃতকর্য | Model | -36.61 | -34.88 | FAIL |
| 18 | অকৃতকার্য | akritakarja | আকৃতাকার্য | Model | -38.42 | -26.05 | FAIL |
| 19 | অকৃতকার্য | akritakarjo | আকৃতাকার্য | Model | -37.66 | -25.30 | FAIL |
| 20 | অকৃতকার্য | okritakarja | অকৃতাকার্য | Model | -38.25 | -33.03 | FAIL |
| 21 | অকৃতকার্য | okritakarjo | অকৃতাকার্য | Model | -37.49 | -32.28 | FAIL |
| 25 | অক্ষরকে | akkharke | অক্ষারকে | Model | -17.07 | -16.95 | FAIL |

---

## 6. 7b Item 4: Stratified Results vs Baseline Engine

| Stratum | N | N-Gram Top-1 (%) | Old Baseline Top-1 (%) | Absolute Diff |
|:---|:---:|:---:|:---:|:---:|
| **With Conjunct (যুক্তবর্ণ)** | 4,298 | **43.39%** | 22.8% | **+20.59%** |
| **Without Conjunct** | 4,981 | **39.99%** | 26.6% | **+13.39%** |
| **Length 1–3 chars** | 433 | **38.57%** | 35.8% | **+2.77%** |
| **Length 4–6 chars** | 3,990 | **39.80%** | 26.1% | **+13.70%** |
| **Length 7+ chars** | 4,856 | **43.29%** | 22.4% | **+20.89%** |
| **Seen in dictionary.json** | 4,211 | **43.55%** | 29.5% | **+14.05%** |
| **Unseen in dictionary.json** | 5,068 | **39.92%** | 20.8% | **+19.12%** |

- The N-gram engine dramatically outperforms the baseline on long words (+20.89%) and conjuncts (+20.59%), proving that multi-token history resolves the severe fragmentation issues seen in context-free parsing.
- Crucially, on words **unseen in dictionary.json** (5,068 dev pairs), the pure N-gram achieves **39.92%** vs the engine's 20.8% (+19.12%), demonstrating generalization beyond lexicon coverage.

---

## 7. 7b Item 5: Multi-Persona Evaluation on Held-Out DEV

| Persona | N-Gram Top-1 (%) | Baseline Top-1 (%) | N-Gram Top-3 (%) | N-Gram Reach (%) |
|:---|:---:|:---:|:---:|:---:|
| **natural** | **41.57%** | 24.86% | 60.45% | 70.06% |
| **lazy** | **38.73%** | 26.05% | 57.28% | 66.80% |
| **vowel_drop_p30** | **18.72%** | 13.51% | 27.75% | 33.07% |
| **vowel_drop_p60** | **6.51%** | 6.19% | 10.25% | 12.63% |
| **vowel_drop_p90** | **1.43%** | 2.12% | 2.40% | 2.97% |
| **avro_strict** | **39.98%** | **90.32%** | 59.53% | 67.33% |

### Persona Findings:
- `avro_strict` regressed from 90.32% to 39.98%. This occurs because the unit tokenizer lowercases Roman inputs, discarding Avro case distinctions (e.g. `T` vs `t`, `S` vs `s`). Step 8 must preserve case distinction or inject deterministic Avro hypotheses directly into the beam.
- Severe vowel dropping (`vowel_drop_p60/p90`) drops reachability to 12.6% and 3.0%, confirming the necessity of Step 7c (synthetic persona data augmentation on TRAIN).

---

## 8. 7b Item 6: Sample of 50 Unreachable Cases (Gold Missing from Top-16)

| ID | Target | Roman | Top-3 Decoded Candidates | Failure Diagnosis |
|:---:|:---|:---|:---|:---|
| 5 | অংশগ্রহন | angshogragon | অংশগ্রগণ, অংশগ্রগন, অংশগ্রাগণ | Homophone / character ambiguity pruned out of beam |
| 11 | অংশগ্রহনের | angsogrohone | অংশগ্রহণে, অংশগ্রহনে, অংশগ্রহণ | Vowel syncope / insertion mismatch |
| 12 | অংশগ্রহনের | anshograhone | অংশগ্রহণে, অংশগ্রহনে, অংশগ্রাহনে | Vowel syncope / insertion mismatch |
| 15 | অকটেন | akten | আঁকতেন, একতেন, অকতেন | Vowel syncope / insertion mismatch |
| 16 | অকটেন | octane | অকট্যানে, অক্টানে, অকট্যাণে | Vowel syncope / insertion mismatch |
| 17 | অকৃতকার্য | akreetokarjo | আকৃতকর্য, আক্রিতকর্য, আকৃতকর্য্য | Vowel syncope / insertion mismatch |
| 18 | অকৃতকার্য | akritakarja | আকৃতাকার্য, আকৃতাকার্য্য, আকৃতাকার্যা | Vowel syncope / insertion mismatch |
| 19 | অকৃতকার্য | akritakarjo | আকৃতাকার্য, আকৃতাকার্য্য, আকৃতাঁকার্য | Vowel syncope / insertion mismatch |
| 20 | অকৃতকার্য | okritakarja | অকৃতাকার্য, ওকৃতাকার্য, অকৃতাকার্য্য | Vowel syncope / insertion mismatch |
| 21 | অকৃতকার্য | okritakarjo | অকৃতাকার্য, ওকৃতাকার্য, অকৃতাকার্য্য | Vowel syncope / insertion mismatch |
| 29 | অক্ষুণ্ণ | akhhunno | অক্ষুণ্ন, অক্ষুন্ন, অক্ষুণ্য | Homophone / character ambiguity pruned out of beam |
| 30 | অক্ষুণ্ণ | akkhunno | অক্ষুণ্ন, অক্ষুন্ন, আক্ষুণ্ন | Homophone / character ambiguity pruned out of beam |
| 31 | অক্ষুণ্ণ | okhhunno | অক্ষুণ্ন, অক্ষুন্ন, অক্ষুণ্য | Homophone / character ambiguity pruned out of beam |
| 32 | অক্সফোর্ডে | auxfordey | আওক্সফোর্ডে, অএক্সফোর্ডে, আওএক্সফোর্ডে | Vowel syncope / insertion mismatch |
| 36 | অখণ্ডতা | akhandota | আখন্ডতা, অখন্ডতা, আঁখন্ডতা | Homophone / character ambiguity pruned out of beam |
| 40 | অগভীর | agahir | আগহীর, আগাহীর, আগাহির | Homophone / character ambiguity pruned out of beam |
| 53 | অঙ্কিতা | angkita | আংকিত, আংকিতা, অংকিত | Vowel syncope / insertion mismatch |
| 54 | অঙ্কিতা | ankeeta | আঁকেট, অংকেট, আঁকেইট | Vowel syncope / insertion mismatch |
| 57 | অঙ্কিতা | onkeeta | অংকেট, ওঁকেট, অনকেট | Vowel syncope / insertion mismatch |
| 59 | অঙ্কিতা | onnkita | অনকিত, অনকিতা, অন্যকিত | Vowel syncope / insertion mismatch |
| 61 | অজুহাত | ajuhath | অযুহঠ, আযুহঠ, আজুহঠ | Vowel syncope / insertion mismatch |
| 63 | অঞ্চলও | anchalo | অঞ্চাল, আঞ্চাল, অঞ্চলো | Homophone / character ambiguity pruned out of beam |
| 122 | অতর্কিতে | atarkitey | আটার্কিটে, অ্যাটার্কিটে, আটার্কিতে | Vowel syncope / insertion mismatch |
| 125 | অতর্কিতে | attorkite | অ্যাটর্কিটে, অ্যাটর্কিতে, এটর্কিটে | Vowel syncope / insertion mismatch |
| 126 | অতর্কিতে | otorkeitey | অতরকেইতে, অতর্কেইতে, অতরকেইটে | Conjunct missed (decoded as separate letters) |
| 135 | অতীতেও | otiiteo | লীতেও, অততেও, ওততেও | Vowel syncope / insertion mismatch |
| 136 | অতীতেও | otiteo | অতিতেও, ওটিতেও, অতিতে | Homophone / character ambiguity pruned out of beam |
| 137 | অত্যাচারিত | attacharito | অট্টাচারিত, অট্টাচারিতো, অট্টাচরিত | Homophone / character ambiguity pruned out of beam |
| 138 | অত্যাচারিত | atyacarito | অত্যাকারিত, অত্যাঁচারিত, অত্যাকারিতো | Homophone / character ambiguity pruned out of beam |
| 141 | অত্যাচারিত | ottacharito | অট্টাচারিত, অট্টাচারিতো, অট্টাচরিত | Homophone / character ambiguity pruned out of beam |
| 148 | অত্যাচারের | ottacharer | অট্টাচারের, অট্টাচরের, অট্টাচাের | Homophone / character ambiguity pruned out of beam |
| 150 | অদুরে | oddure | লীর্থড্ডুরে, লীর্থদ্যুরে, লীর্থদ্দুরে | Vowel syncope / insertion mismatch |
| 154 | অদুরে | oodoore | ওডোর, অডোর, উডোর | Vowel syncope / insertion mismatch |
| 155 | অদ্ভুত | adbhut | অ্যাডভূত, অ্যাডভুট, অ্যাডভুত | Vowel syncope / insertion mismatch |
| 156 | অদ্ভুত | advoot | অ্যাডভূত, এডভূত, অ্যাডভোট | Vowel syncope / insertion mismatch |
| 157 | অদ্ভুত | advut | অ্যাডভূত, এডভূত, অ্যাডভুট | Vowel syncope / insertion mismatch |
| 164 | অনধিকার | anadhikar | অনাধীকার, আনাধীকার, এনাধীকার | Vowel syncope / insertion mismatch |
| 170 | অনুচ্ছেদের | anucheder | অনুচেঁদের, অনুচেদের, আনুচেঁদের | Vowel syncope / insertion mismatch |
| 171 | অনুচ্ছেদের | onucheeder | অনুচিদের, অনুচীদের, অনুচিডের | Vowel syncope / insertion mismatch |
| 176 | অনুর্বর | awnurbor | অনূর্বর, এভনূর্বর, অনূরবর | Homophone / character ambiguity pruned out of beam |
| 179 | অনেকটাই | anektai | অনেকটয়, অনেকটে, অনেকটা | Homophone / character ambiguity pruned out of beam |
| 181 | অনেকটাই | onektai | অনেকটয়, অনেকটে, অনেকটা | Homophone / character ambiguity pruned out of beam |
| 186 | অন্তর্ভুক্ত | antarbhukta | অন্তর্ভূক্ত, আন্তর্ভূক্ত, অন্তভুক্ত | Homophone / character ambiguity pruned out of beam |
| 187 | অন্তর্ভুক্ত | antarbhutto | অন্তভুট্টো, আন্তভুট্টো, অন্তরভুট্টো | Vowel syncope / insertion mismatch |
| 190 | অন্তর্ভুক্ত | antorbhukta | অর্ন্তভুক্ত, অন্তভুক্ত, আন্তভুক্ত | Homophone / character ambiguity pruned out of beam |
| 191 | অন্তর্ভুক্ত | antorvukta | অর্ন্তভুক্ত, অন্তর্ভূক্ত, আন্তর্ভূক্ত | Homophone / character ambiguity pruned out of beam |
| 192 | অন্তর্ভুক্ত | antorvukto | অর্ন্তভুক্ত, অন্তর্ভূক্ত, আন্তর্ভূক্ত | Homophone / character ambiguity pruned out of beam |
| 193 | অন্তর্ভুক্তিমূলক | antarbhuktimulak | অন্তর্ভূক্তিমূলক, অন্তভুক্তিমূলক, আন্তর্ভূক্তিমূলক | Homophone / character ambiguity pruned out of beam |
| 194 | অন্তর্ভুক্তিমূলক | antarbhuttirmulok | অন্তভুট্টির্মূলক, আন্তভুট্টির্মূলক, অন্তর্ভূরির্মূলক | Homophone / character ambiguity pruned out of beam |
| 195 | অন্তর্ভুক্তিমূলক | antarvuktimulak | অন্তর্ভূক্তিমূলক, আন্তর্ভূক্তিমূলক, অর্ন্তর্ভূক্তিমূলক | Homophone / character ambiguity pruned out of beam |

---

## 9. Realistic Target Expectations for Step 8 (Removing Unsubstantiated Forecasts)

- **Previous Erroneous Forecast**: "Push Top-1 from 41% into 70–80%+".
- **Correction**: Reachability is bounded at **70.06%** at $K = 16$ (and 81.27% at $K=64$). Furthermore, `dictionary.json` covers **45.38%** (4,211 / 9,279) of DEV targets.
- A word-level lexicon unigram prior and dictionary presence feature can only reorder candidates that are already within reachability.
- **Calibrated Realistic Target for Step 8 Combined Decoder**: Mid-40s to low-50s Top-1 accuracy on DEV.
