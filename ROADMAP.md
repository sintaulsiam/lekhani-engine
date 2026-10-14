# Lekhani Engine (প্রজ্ঞান / Pragyan) — Master Roadmap & Progress Archive

This document is the authoritative, persistent source of truth for the development, architecture, empirical findings, and remaining milestones of the **Lekhani প্রজ্ঞান (Pragyan) Dynamic Phonetic Engine** (`lekhani-engine`).

---

## 1. 🎯 Architectural Core: The 2-Mode IME Design

There is a fundamental misconception to avoid: **we are NOT creating a 3rd layout or 3rd mode.**
The Lekhani phonetic engine has exactly **two operating modes**, controlled by a single boolean switch (`strict_avro_mode`):

```
                       ┌───────────────────────────────┐
                       │      AVRO PHONETIC INPUT      │
                       └───────────────┬───────────────┘
                                       │
                      Is "Strict Avro Mode" enabled?
                                      / \
                                    /     \
                             YES  /         \  NO (Default)
                                /             \
                    ┌──────────▼─────┐   ┌─────▼──────────┐
                    │    MODE 1:     │   │    MODE 2:     │
                    │  STRICT AVRO   │   │ SMART ADAPTIVE │
                    │ (Classic 2003) │   │   (PRAGYAN)    │
                    └────────────────┘   └────────────────┘
```

1. **Mode 1: Strict Avro Mode (`strict_avro_mode = true`)**:
   - The deterministic 2003 desktop rules compiled into pure Rust (`lekhani-parser`).
   - Runs in **11 nanoseconds per character**, zero heap allocations.
   - Retained 100% intact for purists, power users, and exact Shift-key typists.
2. **Mode 2: Smart Adaptive Mode (`strict_avro_mode = false`)**:
   - **Before Phase 1**: Was a crude, half-baked prototype with ~20 hardcoded overrides. Achieved only **24.8% Top-1** accuracy and failed on 74% of long conjunct words.
   - **Pragyan Phase 1 Objective**: Completely replace this broken prototype with a mathematically sound statistical transliteration engine (M2M alignment + N-gram LM + lexicon-constrained beam search).
   - **Result**: Mode 2 becomes an intelligent mobile engine (targeting 60–70%+ accuracy) while Mode 1 remains classic manual control.

---

## 2. 🔍 The Forensic Journey: How We Got Here

### Phase 0: Baseline Diagnostics & The 24.8% Reality
- Built the automated evaluation harness ([`eval_phonetic.rs`](lekhani-core/src/bin/eval_phonetic.rs)) and simulated human typing personas on Google's Dakshina benchmark:
  - **Avro-Strict Persona** (ideal desktop typist with Shift): **88.52% Top-1**.
  - **Natural Persona** (real-world mobile human typing): **24.80% Top-1** (DEV), **25.28% Top-1** (TEST).
- **Finding**: Classical Avro fails 3 out of 4 times on real mobile text because users type in lowercase, drop inherent vowels, and use chat shortcuts.

### Root-Cause Discovery: The Byte-vs-Character Truncation Bug
- Early Phase 1 alignment attempts produced bizarre results: the decoder achieved only 3.7% Top-1 and learned 16,612 bloated, noisy units (`cer → সার`, `ohi → অহি`, `kon → কণ্ঠ`).
- **Forensic Discovery (Step 2)**: In Rust, `String::len()` measures UTF-8 **bytes**, not Unicode characters. Because Bengali characters are 3 bytes each, `bn.len() <= 24` was silently cutting off every word with more than 8 characters ($24 / 3 = 8$).
- **Impact**: **28,178 training rows (29.8% of the data)** and **half (50.1%) of all conjunct rows (24,464 rows)** were silently discarded. The model was trained on an amputated, artificially simple dataset, making it impossible to learn complex conjuncts.
- **Decision**: Implemented an unconditional **Filters OFF (100% inclusion)** policy across train, dev, and test.

### Hermetic Lexicon & Anti-Tampering Gates (Step 1)
- Discovered that previous 30.8% accuracy figures were contaminated with Wikipedia-derived unigram frequency lists.
- Stripped all unverified unigrams and locked the lexicon strictly to `data/dictionaries/dictionary.json` (SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`).
- Added automated CI regression gates that panic if the dictionary hash or accuracy thresholds are tampered with.

### Step 3: Conjunct Failure Localization & Tokenization Inventory
- Evaluated baseline Avro stratified by conjunct density and character length:
  - Non-conjunct words: **28.63% Top-1** (60.17% unreachable).
  - **Long words with conjuncts ($7+$ chars)**: **19.91% Top-1** (**74.32% unreachable**).
  - Proved that the 74% unreachable conjunct-heavy words are the primary bottleneck of Bengali typing.
- Built fine-grained linguistic tokenization ([`tokenization.rs`](lekhani-core/src/tokenization.rs)):
  - Grouped atomic conjuncts ($C + \text{্} + C$) and separated vowel kars.
  - Collapsed combinatorial explosion down to a clean, bounded inventory: **437 Bengali tokens** and **66 Roman tokens**.
  - Verified **100.0000% lossless round-trip concatenation** across all 113,053 Dakshina pairs and all 159,423 `dictionary.json` words.

---

## 3. 📊 Master Milestone Status (Pragyan Phase 1)

| Step | Milestone | Status | Key Deliverable / Finding | Git Commit |
|:---:|:---|:---:|:---|:---:|
| **1** | **Hermetic Re-Baseline** | **DONE** | Established true baseline: 24.80% DEV, 25.28% TEST. Locked SHA-256 dictionary gate. | [`61b565f`](file:///mnt/data/lekhani-engine) |
| **2** | **Data Accounting & Audit** | **DONE** | Uncovered byte length bug (50.1% conjunct rows dropped). Established Filters-OFF policy. | [`3129ff1`](file:///mnt/data/lekhani-engine), [`4261f73`](file:///mnt/data/lekhani-engine) |
| **3** | **Linguistic Tokenization** | **DONE** | 437 BN / 66 Roman units. 100% round-trip verified. Measured 74.3% conjunct failure. | [`f6e7654`](file:///mnt/data/lekhani-engine) |
| **4** | **M2M Alignment on Full Data** | **NEXT** | EM alignment on full 94,546 pairs with Avro soft prior. Export 200-sample audit CSV. | *In Progress* |
| **5** | **Translation Unit N-Gram LM** | Pending | Joint $(R, B)$ unit language model. Full DEV reachability gap analysis. | — |
| **6** | **Trie Beam Search Decoder** | Pending | Lexicon-constrained beam search integrating PrefixTrie. Sub-100 µs latency. | — |
| **7** | **Perceptron Feature Ranker** | Pending | Structured perceptron with 6 feature templates + Dakshina row weighting. | — |
| **8** | **Hyperparameter DEV Tuning** | Pending | Tune beam width, LM weights ($\lambda$), and pruning thresholds exclusively on DEV. | — |
| **9** | **Final TEST Set Gate & Release** | Pending | Evaluate gated `test.tsv`, log ISO-8601 release in `test_runs.log`. | — |
| *2b* | *BanglaTLit External Benchmark* | Queued | External real-world Banglish evaluation (EMNLP 2024 dataset). | — |

---

## 4. 🚀 The Remaining Road Ahead: Detailed Implementation Plan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STEP 4: M2M ALIGNMENT ENGINE                           │
│  - Run Expectation-Maximization (EM) on full 94,546 pairs (Filters OFF)     │
│  - Seed EM with Avro soft prior (frown upon non-Avro splits)                │
│  - Enforce hard constraints (atomic conjuncts, KAR vowel dependencies)      │
│  - Export 200 random training pairs to data/eval/align_audit.csv            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                  STEP 5: TRANSLATION UNIT LANGUAGE MODEL                    │
│  - Estimate joint transition probabilities P(unit_i | unit_{i-1}, ...)      │
│  - Compute reachability across the full DEV set (all 9,279 rows)            │
│  - Verify whether the previous 99% vs 38% gap holds on uncorrupted data     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│              STEP 6: LEXICON-CONSTRAINED BEAM SEARCH DECODER                │
│  - Integrate PrefixTrie with M2M alignments                                 │
│  - Prune illegal non-dictionary paths dynamically during search             │
│  - Enforce sub-microsecond latency budget (< 100 µs per keystroke)          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│             STEP 7: PERCEPTRON RANKER & DAKSHINA ROW WEIGHTING              │
│  - Implement 6 linguistic feature templates (homophone, context, shape)     │
│  - Weight training updates by Dakshina row frequency (penalize rare typos)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                 STEP 8: DEV HYPERPARAMETER TUNING & ABLATION                │
│  - Optimize λ blending weights, beam width, and pruning thresholds on DEV   │
│  - Zero test set leakage                                                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                    STEP 9: FINAL TEST GATE & PRODUCTION LOG                 │
│  - Run official release gate against bn.translit.sampled.test.tsv           │
│  - Verify Top-1 accuracy exceeds baseline (> 50-60%)                        │
│  - Append SHA and configuration to data/eval/test_runs.log                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 🛡️ Inviolable Engineering Rules & Evaluation Discipline

As codified in [`AGENTS.md`](AGENTS.md):
1. **Measured Numbers Only**: Never write subjective adjectives ("100% sound", "clean"). Report only code-measured metrics, exact commands, and commit SHAs.
2. **Strict Split Purity**:
   - `train.tsv`: For fitting and training only.
   - `dev.tsv`: For all tuning, feature selection, and error diagnoses.
   - `test.tsv`: Strictly gated until Step 9.
3. **Identical Filter & Drop Transparency**: Any filter must be applied identically to train, dev, and test. Its drop count must be printed on every single run.
4. **Hermetic Lexicon**: Use `dictionary.json` only (SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`).
5. **Single Canonical Tools**: Maintain exactly one canonical implementation of any tool in `lekhani-core/src/bin/`. Never duplicate tools across directories.
6. **Sub-Microsecond Latency**: Fast-path token scoring must execute in $< 100\,\mu\text{s}$. Never allocate on the heap in touch loops.
