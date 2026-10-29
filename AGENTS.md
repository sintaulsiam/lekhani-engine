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
7. **Living AGENTS.md Invariant (Continuous Documentation Update)**:
   Every agent and contributor MUST automatically update `AGENTS.md` whenever:
   - A new empirical baseline or gate is reached (update baseline figures and commit SHA).
   - A structural limit or ceiling is discovered (update the census data).
   - An experiment fails or regresses (immediately document the mechanism in the Graveyard of Negative Results so it is never repeated).
   - An architectural invariant is added or refined.
   Never leave findings buried in conversation history or ephemeral logs.

---

## 4. 🚫 Anti-Circular Engineering Directives (STRICTLY ENFORCED)

### A. Absolute Ban on Micro-Tuning `fuzzy.rs`
- Candidate generation in `fuzzy.rs` is saturated (producing candidates for 94.8% of in-dictionary targets).
- Adding more hand-crafted regex rules, digraph mappings, or character tweaks in `fuzzy.rs` or `sound_laws.json` is **strictly forbidden**. It is a local hill-climbing trap that produces trivial +2 gains while distracting from real architecture.

### B. Empirical Census of the 5,128 Misses (N=9,279 Dev, Commit `6398c79`, 4,151 Hits / 44.74%)
- **Target is ALREADY generated in candidates (Rank 2–7)**: **2,488 words (48.5% of all misses / 26.8% of Dev)**. This is where all future gains live.
- **Target is NOT in dictionary.json (OOV names/loanwords)**: **2,389 words (46.6% of misses / 25.7% of Dev)**. Pure dictionary-bound ceiling is 74.25%.
- **Target was not generated at all**: **251 words (4.9% of misses / 2.7% of Dev)**.

### C. Morphological & Phonotactic Rules Established
1. **Consonant Stem Protection against Genitive -র / Glide -য়**: In Bengali grammar, genitive `-র` and glide `-য়` strictly attach to vowel/kar-ending stems (`বাবা` $\to$ `বাবার`, `বাড়ি` $\to$ `বাড়ির`). Consonant stems take `-এর`/`-ে` (`অত্যাচার` $\to$ `অত্যাচারের`). Peeling or attaching `-র` to consonant stems generates illegitimate doubled consonant words (`অত্যাচারর` $\to$ `অত্যাচাররের`). Restricting `-র` and `-য়` to vowel stems advanced Dev natural hits from 4,086 to **4,114** (+28 net gain, Commit `8f5bdd2`).
2. **Elimination of Artificial `trie_contains` Boost on Bare Roots**: Awarding `is_high_freq += 0.5` (+1,000 pts) solely to words in `trie.contains_exact()` gave base roots an unearned advantage over their own legitimate morphological inflections. Removing this artificial boost advanced Dev natural hits from 4,114 to **4,151** (+37 net gain, Commit `6398c79`), resolved 108 frequency-trap errors, and cut dropped terminal 'e' root truncations in half (from 67 to 34).

### D. Mandatory Focus: Candidate Strip Ranker & Feature Scorer (`ranking.rs` & `suggestion.rs`)
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
4. **Blanket Suppression of Direct Avro via `has_dominant_homophone` (Commit `9f7407a`)**:
   - Stripping `is_exact_phonetic` from DirectTransliteration whenever `has_dominant_homophone` is true caused a -192 regression (4086 down to 3894).
   - Reason: `has_dominant_homophone` is triggered by ANY candidate with unigram frequency $\ge 30$. Stripping the Avro exact bonus causes valid transliterations of OOV words, proper nouns, and inflections to be overtaken by unrelated dictionary words (e.g. typing `ankita` produced `নাকিতা` instead of `অঙ্কিতা`).
5. **Grapheme Edit Distance (`dist <= 1`) Guard on FuzzySoundLaw (Commit `9f7407a`)**:
   - Restricting `is_exact_phonetic = 0.85` in `FuzzySoundLaw` to `dist <= 1` starves legitimate Dakshina transliterations (e.g. where vowel and consonant variations yield grapheme distance $\ge 2$ from raw Avro) of their ranking score, leaving phonetic noise at Rank 1.
6. **Unsegmented Single-Pass Compound Dumping into Perceptron (Commit `9f7407a`)**:
   - Bypassing the 2,500-pt cap on `SegmentedLattice` compound words without compound-specific feature extraction damages the single-word ranking balance. Compounds require dedicated segmentation features before ranker scoring.
7. **Scaling `dist == 0` on `MorphologicalInflection` without clitic-O protection**:
   - Boosting `dist == 0` over `dist == 1` causes candidates with clitic `-ও` (`অক্ষুণ্ণও`) that have `dist == 0` from raw Avro transliteration (`akhhunno`) to defeat true bare consonant schwa targets (`অক্ষুণ্ণ`) (-91 hits: 4151 down to 4060).
8. **Naive `dropped_conjunct` Guard on `NeuralTransducer`/`PragyanBeam`**:
   - Penalizing candidates lacking `্` whenever raw Avro contains `্` heavily regresses (-91 hits) because Avro creates spurious conjuncts for common Latin digraphs (`sh`, `th`, `ch`) and loanwords where Bengali maps to a single consonant.
9. **Increasing `phonetic_similarity` weight in linear ranker**:
   - Increasing `phonetic_similarity` weight from 200.0 to 400.0 degrades accuracy (-13 hits) due to asymmetric negative penalties on edit distances.

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
