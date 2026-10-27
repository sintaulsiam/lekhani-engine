# AGENTS.md — Engineering Rules for Lekhani Engine

Welcome to **Lekhani Engine** (`lekhani-engine`). This document defines the engineering standards, architecture rules, performance budgets, empirical ceilings, and anti-circular development workflows that all human contributors and AI agents **must strictly follow**.

---

## 1. 🏗️ Architecture & Crate Map

```
┌────────────────────────────────────────────────────────┐
│     lekhani-core (Headless IME State Machine)          │
│  - suggestion.rs: Candidate generation & multi-token   │
│  - ranking.rs: 20-D Perceptron & Feature Ranker        │
│  - database.rs: Fast-path Trie & Morphological Engine  │
│  - chars.rs: Canonical NFC normalization               │
└──────────────────────────▲─────────────────────────────┘
                           │
┌──────────────────────────┴─────────────────────────────┐
│  lekhani-parser          │  lekhani-ai / neural        │
│  - Zero-alloc Avro Trie  │  - Order-4 n-gram LM        │
│  - LPS Grammar Engine    │  - MicroGRU & Seq2Seq       │
│  - Byte-scanner hot-path │  - Bidirectional rescorer   │
└────────────────────────────────────────────────────────┘
```

- **`lekhani-parser`**: 11 ns/char pure Avro Trie grammar engine. Must remain strictly zero runtime heap allocation on byte slices.
- **`lekhani-ai`**: Local N-gram scorer (`bengali_lm.bin`), MicroGRU neural predictor, and bidirectional context rescorer.
- **`lekhani-neural`**: Pretrained Seq2Seq transliterator weights (`seq2seq_transliterator.bin`).
- **`lekhani-core`**: The primary coordinator. Contains candidate generation (`suggestion.rs`), feature extraction and scoring (`ranking.rs`), morphological Sandhi compounding, and database lookup.

---

## 2. 🛡️ Inviolable Core Principles & Performance Budgets

1. **Sub-Microsecond Hot Path & Latency Budget**:
   - Hot-path keystroke processing: **< 100 µs** per character.
   - Integration Release Latency Budget: **p95 $\le 3,617.5$ µs** under `cargo test --release -p lekhani-core --test phonetic_eval test_dakshina_baseline_gate_against_json`.
2. **Strict Memory & Binary Size Discipline**:
   - Zero runtime heap allocations inside fast-path token scanning (`onDraw`/keystroke loops).
   - Candidate pipelines use bounded, pre-allocated vectors (`Vec::with_capacity(8)`).
   - Binary model budgets:
     - `bengali_lm.bin` $\le 5.5$ MB.
     - `seq2seq_transliterator.bin` $\le 1.5$ MB.
3. **Hermetic Lexicon**:
   - `dictionary.json` (SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`) must remain completely untouched. Never mutate it or inject unverified word lists into eval.
4. **Script Integrity & Privacy**:
   - 100% on-device. Zero network telemetry.
   - All text committed must be canonical NFC Unicode. Never corrupt conjuncts (juktoborno) or diacritics.

---

## 3. 🔬 Scientific Evaluation Discipline (Pragyan Rules)

1. **Measured Numbers Only**:
   Report only measured numbers from code and benchmarks, each with the command that produced it and the commit SHA. Never write subjective assertions ("generalizes better", "100% sound") without attached data tables.
2. **Strict Split Purity**:
   - `train.tsv`: For model training and alignment fitting only.
   - `dev.tsv`: For all tuning, feature decisions (unit size, order, pruning thresholds, λ weights), and error diagnoses.
   - `test.tsv`: Strictly gated. Every test-set evaluation must be logged with ISO-8601 timestamp, commit SHA, and exact configuration in `data/eval/test_runs.log` (append-only).
3. **Commit-First Discipline**:
   Commit the code first, run the test, and record that commit SHA with the results.
4. **Canonical Baselines**:
   Quote baselines only from the committed `baseline.json`.
5. **Tool & Code Hygiene**:
   Maintain exactly one canonical implementation of any tool or binary in `lekhani-core/src/bin/`. Never duplicate tools across directories. Explicitly state every file deletion.
6. **Atomic Commits**:
   One commit per step formatted as `<type>(<scope>): <description>`.

---

## 4. 🚫 Anti-Circular Engineering Directives (STRICTLY ENFORCED)

### A. Absolute Ban on Micro-Tuning `fuzzy.rs`
- Candidate generation in `fuzzy.rs` is saturated (producing candidates for 94.8% of in-dictionary targets).
- Adding more hand-crafted regex rules, digraph mappings, or character tweaks in `fuzzy.rs` or `sound_laws.json` is **strictly forbidden**. It is a local hill-climbing trap that produces trivial +2 gains while distracting from real architecture.

### B. Empirical Census of the 5,193 Misses (N=9,279 Dev)
- **Target is ALREADY generated in candidates (Rank 2–7)**: **2,531 words (48.7% of all misses / 27.3% of Dev)**. This is where all gains live.
- **Target is NOT in dictionary.json (OOV names/loanwords)**: **2,390 words (46.0% of misses / 25.8% of Dev)**. Pure dictionary-bound ceiling is 74.2%.
- **Target was not generated at all**: **272 words (5.2% of misses / 2.9% of Dev)**.

### C. Mandatory Focus: Candidate Strip Ranker & Feature Scorer (`ranking.rs` & `suggestion.rs`)
1. **Direct Avro Monopoly Prohibition**: Non-words (`freq: 0`, `!is_in_dict`, e.g. `আঞ্ছালসাহা`) must never receive an automatic +10,000 pt advantage over valid dictionary or compound candidates (`অঞ্চলসহ`). Strip unearned `is_exact_phonetic` bonuses when direct Avro parse is not a dictionary word.
2. **Preserve Typed Endings (Anti-Root Truncation)**: High unigram frequency of bare roots must not strip user-typed terminal vowels or suffixes (e.g. typing `ankita` or `oncholo` must not truncate the ending to pick base `অঙ্কিত` or `অঞ্চল`).
3. **Compound Splitting via Lattice**: Agglutinative Sanskrit compounds (`akritakarja` -> `অকৃতকার্য`) must be resolved by 2-stem lattice segmentation (`অকৃত` + `কার্য`), not monolithic single-pass Avro parsing.

---

## 5. ⚠️ Graveyard of Negative Results (DO NOT REPEAT)

1. **Bengali Inherent Schwa (`[ɔ]`) vs Clitic-O (`-ও`)**:
   - In Latin Banglish, terminal `o` is heavily used for the unwritten Bengali inherent consonant schwa (e.g. `akhhunno` for `অক্ষুণ্ণ`, `somuho` for `সমূহ`, `karjo` for `কার্য`, `shobdo` for `শব্দ`).
   - Penalizing bare consonant stems whenever `middle.ends_with('o')` or promoting clitic forms `stem + ও` caused a severe regression (-322 hits: 4072 down to 3750) by mutating `অক্ষুণ্ণ` into `অক্ষুণ্ণও`, `অকৃতকার্য` into `অকৃতকার্যও`, and `অঞ্চলসমূহ` into `অঞ্চলসমূহও`.
   - Never penalize a candidate for "dropping" `o` into a bare consonant without checking explicit clitic targets or conjugation registers.
2. **English Loanwords & Silent Terminal `e`**:
   - In Dakshina transliterations of Latin/English loanwords (e.g. `augustine`), Latin terminal `e` is silent and maps to a bare consonant (`অগাস্টিন`).
   - A generic `dropped_e_suffix` penalty erroneously forces the locative form `অগাস্টিনে` over the base unigram `অগাস্টিন`.
3. **Sound Law Symmetries (`th` -> `t`)**:
   - Adding `th` $\to$ `t` globally in `fuzzy.rs:96` caused a collision on authentic aspirated `থ` words (-1 net hit: 4072 down to 4071).

---

## 6. ✅ Mandatory Pre-Flight Verification Checklist

Before concluding any task or reporting progress, all 5 checks must pass:
1. `cargo clippy -p lekhani-core --lib -- -D warnings` $\to$ 0 warnings.
2. `cargo test --release -p lekhani-core --test phonetic_eval test_dakshina_baseline_gate_against_json` $\to$ Passes within latency budget (p95 $\le 3,617.5$ µs).
3. `cargo test --release -p lekhani-core --test phonetic_eval test_bin_vs_json_dev_natural_hit_parity` $\to$ Must be 100% identical between JSON and BIN (diff count = 0).
4. `cargo test -p lekhani-core --test simulation_suite` $\to$ 13/13 passed.
5. `cargo test -p lekhani-core --test real_world_chat_audit` $\to$ 10/10 passed.

To run diagnostic evaluations on Dev misses:
`cargo run --release -p lekhani-core --bin diagnose_dev`
