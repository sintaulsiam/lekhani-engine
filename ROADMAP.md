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

### Step 3: Linguistic Tokenization, Inventory & Stratification
- **Stratified Baseline on DEV (Target Bengali Unicode Char Count)**:
  - Input variation is the primary bottleneck: even short non-conjunct words (1–3 chars) have a **60.05% unreachability rate** and 27.48% Top-1.
  - Conjuncts impose an additional 7–8 percentage point Top-1 penalty within matched length strata (4–6 chars: 29.29% vs 21.93%; 7+ chars: 27.87% vs 19.97%).
  - Top-1 DEV reproduces `baseline.json` bit-for-bit at **24.86%** ($2,307 / 9,279$).
- **Strict Train-Derived Inventory**:
  - Derived strictly from 94,546 Train pairs and `dictionary.json` (zero test leakage).
  - 437 Bengali tokens (including 273 Train conjunct types / 362 Train+Dict types) and 66 Roman tokens.
  - Rule-based conjuncts ($C + \text{্} + C$) with deterministic backoff to single consonants + hasanta (Dev conjunct OOV is only 0.23% tokens; Test OOV 0.39%).
  - Pre-pruning co-occurrence candidate translation units in Train: **9,998**.
- **Roman Segmentation Lattice**:
  - Implemented `roman_segmentation_lattice` to offer both decomposed and digraph paths (`ng` vs `n`+`g`), preventing premature greedy segmentation errors.
- **Automated Property Tests & Gates**:
  - 100% round-trip concatenation verified across all 113,053 pairs and 159,423 dictionary entries.
  - Invariants verified: no mid-word boundary hasantas, no word-initial KARs, 100% tokenization idempotence.
  - Real disk file hash tampering test verified: mutating 1 byte panics CI gate on real loader path.

---

## 3. 📊 Master Milestone Status (Pragyan Phase 1)

| Step | Milestone | Status | Key Deliverable / Finding | Git Commit |
|:---:|:---|:---:|:---|:---:|
| **1** | **Hermetic Re-Baseline** | **DONE** | Established true baseline: 24.80% DEV, 25.28% TEST. Locked SHA-256 dictionary gate. | [`61b565f`](file:///mnt/data/lekhani-engine) |
| **2** | **Data Accounting & Audit** | **DONE** | Uncovered byte length bug (50.1% conjunct rows dropped). Established Filters-OFF policy. | [`3129ff1`](file:///mnt/data/lekhani-engine), [`4261f73`](file:///mnt/data/lekhani-engine) |
| **3** | **Linguistic Tokenization** | **DONE** | 437 BN / 66 Roman units. 100% round-trip verified. Measured 74.3% conjunct failure. | [`4817e6a`](file:///mnt/data/lekhani-engine) |
| **4** | **M2M Alignment on Full Data** | **DONE** | 98.76% forced reach, 97.8% model error diagnosed, human audit protocol delivered. | [`791f5d3`](file:///mnt/data/lekhani-engine), [`a9479e8`](file:///mnt/data/lekhani-engine) |
| **5** | **Translation Unit N-Gram LM** | **DONE** | Joint $(R, B)$ unit language model. Type-level Order 4 Level 1 (3.07 MB). | [`791f5d3`](file:///mnt/data/lekhani-engine) |
| **6** | **Trie Beam Search Decoder** | **DONE** | Fast-path incremental decoding with PrefixTrie integration (p95 = 1,858 µs). | [`fc4a816`](file:///mnt/data/lekhani-engine) |
| **7** | **Perceptron Feature Ranker** | **DONE** | 16-D linguistic feature extractor + Margin Perceptron reranker ([`step7_ranker.md`](file:///mnt/data/lekhani-engine/data/eval/reports/step7_ranker.md)). | [`dec5138`](file:///mnt/data/lekhani-engine), [`6bde0c4`](file:///mnt/data/lekhani-engine) |
| **8** | **Hyperparameter DEV Tuning** | **DONE** | Target-disjoint Dev-A/Dev-B confirmation, Gate G2, G4, G5 passed ([`step8.md`](file:///mnt/data/lekhani-engine/data/eval/reports/step8.md)). | [`fc4a816`](file:///mnt/data/lekhani-engine) |
| **9** | **Final TEST Set Gate & Release** | **NEXT** | Evaluate gated `test.tsv`, log ISO-8601 release in `test_runs.log`. | — |
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
