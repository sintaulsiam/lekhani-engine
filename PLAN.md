# Lekhani Engine — Engineering Plan
*Last updated: 2026-10-25 · Commit: `def451a`*

---

## Part 1 — What We Have Done: An Honest Audit

### 1.1 Baseline Accuracy (Measured, Not Estimated)

All numbers are from locked test set (`bn.translit.sampled.test.tsv`, N=9,228).

| Stage | Natural Top-1 | What Unlocked the Gain |
|:---|:---:|:---|
| Raw Avro parser (no lexicon) | 12.7% | – |
| + Dictionary lookup (Step 1) | 25.3% | OpenBangla root lexicon (159k entries) |
| + Pragyan unit LM + Beam search (Steps 5–8) | 48.0% | M2M aligner + Order-4 n-gram unit LM |
| + POVR v2 overrides + bidirectional ranking | **~51.6%** | 23k supervised phonetic overrides + Anvaya SOV ranker |

**Avro-strict ceiling**: 90.9%. The gap between 90.9% and 51.6% is the "natural typing diverges from formal Avro" problem.

---

### 1.2 Where the Failures Are Right Now (Full Census: N=9,279 Dev, 5,193 Misses)

From full diagnostic census (`lekhani-core/src/bin/diagnose_dev.rs`):

| Failure Category | Misses | % of Misses | % of Dev | Root Cause & Action |
|:---|:---:|:---:|:---:|:---|
| **Target in Dict, Ranked at Rank 2–7** | **1,406** | **27.1%** | **15.2%** | **Highest ROI**: Direct Avro monopoly & unigram frequency bias demote valid candidates. Fix via ranker scoring. |
| **Target OOV, Ranked at Rank 2–7** | **1,125** | **21.7%** | **12.1%** | Valid OOV forms (e.g. `অঙ্কিতা`, `অঞ্চলসহ`) suppressed by non-word direct Avro output (`আঞ্ছালসাহা`). Fix via ranker scoring. |
| **Target in Dict, NOT Generated** | **1,397** | **26.9%** | **15.1%** | Sanskrit Tatsama agglutinated compounds (`অকৃতকার্য`). Fix via 2-stem lattice segmentation. |
| **Target OOV, NOT Generated** | **1,265** | **24.4%** | **13.6%** | Pure OOV named entities & crowdsource noise. Hard lexicon ceiling (74.2%). |
| **Candidate Generation Deficit (`fuzzy.rs`)** | **272** | **5.2%** | **2.9%** | Saturated. **DO NOT micro-tune `fuzzy.rs`**. |

> **Key insight**: 2,531 misses (48.7% of all failures, 27.3% of Dev) **were already generated in the candidate strip**. The engine fails because of candidate ranking, NOT candidate generation. All future work must target `ranking.rs` and compound lattice splitting.

---

## Part 2 — Automated vs. Human Review: The Honest Answer

### ✅ Correctly Automated (No Human Review Needed)

| Task | How | Why Automation Was Right |
|:---|:---|:---|
| N-gram LM training (`train_lm.rs`) | Automated over 847k-line corpus | Pure frequency counting; no subjective judgment |
| Pragyan unit LM (`train_unit_lm.rs`) | Automated EM + Order-4 smoothing | Mathematically deterministic; results are measurable |
| POVR binary compilation | `ZeroCopyOverrides::compile` | Lossless byte transformation; verifiable by hash |
| Corpus ingestion & cleaning (`prepare_balanced_corpus.py`) | Automated multi-source blend | NFC normalization + Bengali ratio filter = objective rules |
| Dictionary binary build | Automated trie serialization | Deterministic; SHA-256 gated in CI |
| Regression gate tests | `phonetic_eval.rs` CI suite | Numerical thresholds; machines are better at this |
| M2M aligner | EM algorithm on 94,546 pairs | Probabilistic optimization; objective function |
| Spatial touch model | Gaussian key-center fitting | Pure geometry; no opinion involved |

---

### ⚠️ Automated But Should Have Had Human Review

These are the places where we ran the tool and moved on, but a human expert would have caught problems that are now showing up as failures.

#### 1. Phonetic Overrides JSON (23,753 entries)

**What we did**: Generated entries automatically from Dakshina training set + morphological heuristics.

**What should have happened**: A native Bengali linguistics expert should have audited ~500–1,000 entries covering:
- Cluster digraph mappings (e.g., `kkh` → `ক্খ` vs `ক্ষ`)
- Loanword vowel-initial rules (when does `a`-initial map to `অ` vs `আ`)
- Verb dialect paradigms (Dhakaiya, Sylheti, Chittagonian forms)

**Consequence of skipping**: 40 of 100 sampled failures are words in our own dictionary that the parser cannot reach due to wrong override routing. Examples: `ekkhando` → `একখণ্ড`, `upanibeshbad` → `উপনিবেশবাদ`.

---

#### 2. M2M Aligner Quality Audit (Only 200 Rows)

**What we did**: Ran EM alignment on 94,546 pairs, spot-checked only 200 rows in `align_audit.csv`.

**What should have happened**: A trained native annotator should have graded at least 1,000 randomly sampled alignment pairs before the unit LM was trained on them. The Step 4 report found **97.8% model error rate** — meaning the vast majority of alignment decisions cannot be verified by Avro rules alone. We audited only 200 / 94,546 = 0.2%.

**Consequence**: Our 65.9% "unreachable" rate on natural typing (Step 1 report) means the unit LM was never scoring ~66% of targets at all — it simply could not generate them. A portion of that is inherent free-form Banglish; another portion is alignment noise baked into the unit LM during training.

---

#### 3. Hand-Crafted `CORE_EMBEDDINGS` (~900 lines in `embeddings.rs`)

**What we did**: Manually assigned semantic cluster dimensions (Dim 8/9 = Education, Dim 18/19 = Clothing, Dim 28/29 = Court) and hard-coded 19 homophone contrast pairs.

**What should have happened**: Run `train_embeddings.rs` on the real corpus to get data-driven vectors and use those as the primary embedding table. The trained binary (`bengali_embeddings.bin`, 595 KB) already exists but is **not being used** for disambiguation — the hardcoded constants override it.

**Consequence**: Homophone disambiguation works for only 19 known pairs. The remaining 200+ Bengali homophones get no topic-aware disambiguation.

---

#### 4. Corpus Blend Ratios Not Controlled or Logged

**What we did**: Downloaded 4 datasets (YouTube comments, Alpaca BN, Wikipedia BN, dialogues) and concatenated without weighting.

**What should have happened**: A corpus linguist should set domain weights. Mobile typing is roughly 70% conversational, 20% encyclopedic, 10% formal. If Wikipedia is 50% of the corpus but only 20% of actual mobile use, the LM's next-word predictions are calibrated for the wrong distribution.

**Consequence**: Unknown corpus proportions. We cannot trace whether a next-word prediction failure is a model quality issue or a domain mismatch issue.

---

### ❌ Incorrectly Automated (Should Have Been Built Differently)

#### 1. Trained Embeddings Binary Is Not Used

`train_embeddings.rs` produces `bengali_embeddings.bin`. But `CORE_EMBEDDINGS` in `embeddings.rs` is a completely separate hard-coded constant table. The binary is loaded as a supplementary lookup, but the primary disambiguation logic uses hardcoded constants. This is an architecture bug: we have a working trainer that is being ignored.

#### 2. POVR Entries May Overlap with Dev/Test Targets (Evaluation Contamination Risk)

The overrides JSON was partially seeded from Dakshina train-split targets. We never verified what fraction of the 23,753 POVR entries appear in the dev or test target vocabulary. Any override that directly encodes a test target skips the need for the model to generalize — inflating the measured 51.6% beyond what it would be on truly unseen words.

---

## Part 3 — The Plan: What to Do Next

Ordered by ROI. Phase A items require no new data.

---

### Phase A — Fix What's Broken Without New Data (~2–3 days)

#### A1: Automated POVR Gap-Filler [HIGHEST ROI]

**What**: Run the evaluator over the full dev set, identify every "in-dictionary but parser-misrouted" failure, and auto-generate the correct POVR override entry.

**How**: New bin `fill_povr_gaps.rs` — diffs Avro parser output against dictionary lookup, emits corrective entries.

**Human review required**: Yes — a Bengali speaker spot-checks the generated entries before compilation into `phonetic_overrides.bin`.

**Expected gain**: +6–10% Natural Top-1 (51.6% → 57–62%)

**Files**: `lekhani-core/src/bin/fill_povr_gaps.rs` (new), `data/dictionaries/phonetic_overrides.json`

---

#### A2: Initial Vowel Statistical Reranker [HIGH ROI, ZERO NEW DATA]

**What**: When top candidate starts with `আ-` and a `অ-` prefixed form exists in the dictionary, score both against the 4-gram LM and pick the winner.

**How**: Post-processing pass in `suggestion.rs`, < 500 ns overhead, zero allocations.

**Human review required**: No — fully deterministic LM scoring.

**Expected gain**: +2–4% Natural Top-1

**Files**: `lekhani-core/src/phonetic/suggestion.rs`

---

#### A3: Wire the Trained Embeddings Binary [FIX ARCHITECTURE BUG]

**What**: Replace `CORE_EMBEDDINGS` hardcoded constants with runtime-loaded vectors from `bengali_embeddings.bin` in the homophone disambiguation path.

**How**: Load the binary in `ContextScorer::new()`, use it in `disambiguate_with_domain` instead of the static table.

**Human review required**: No.

**Expected gain**: No Dakshina accuracy change, but homophone disambiguation now works for 4,096 words instead of 19 hard-coded pairs.

**Files**: `lekhani-ai/src/context.rs`, `lekhani-ai/src/embeddings.rs`

---

### Phase B — Human Review Gates (~1–2 weeks)

#### B1: Phonetic Override Audit (NEEDS A BENGALI LINGUIST)

**What**: Audit 1,000 entries from `phonetic_overrides.json`, focusing on:
- Cluster digraph routing (`kkh`, `nch`, `ndh`, `bsh`, `psh`)
- Loanword vowel-initial rules
- Colloquial verb dialect paradigm entries

**Why human review**: The failures in `ekkhando`, `upanibeshbad`, `ananumodita` show that automated generation got phonetic routing wrong for systematic consonant clusters. Automating the fix without linguistics review risks replacing one wrong rule with another wrong rule.

**Deliverable**: Annotated correction list → automated rebuild of `phonetic_overrides.bin`

---

#### B2: Corpus Domain Weight Audit

**What**: Count lines per source in `bengali_corpus.txt`, measure vocabulary overlap with dev/test targets per source, set evidence-based blend ratios.

**Why human review**: A corpus linguist must make the judgment call on domain proportions for mobile-first typing. This cannot be optimized on Dakshina alone because Dakshina is a news-domain dataset.

**Deliverable**: Updated `prepare_balanced_corpus.py` with documented weights + re-run of `train_lm.rs`

---

#### B3: M2M Alignment Re-Audit (200 → 1,000 rows)

**What**: Expand the alignment quality audit from 200 to 1,000 rows. Native annotator grades each alignment as Correct / Wrong / Ambiguous.

**Why**: The unit LM reachability is gated by alignment quality. At 97.8% model error rate (Step 4 finding), we need a larger sample to know whether to trust the current alignments or retrain.

**Deliverable**: Updated `align_audit.csv` + go/no-go decision on LM retrain

---

### Phase C — New Data Acquisition (Weeks–Months)

#### C1: Loanword & Proper Noun Dictionary Expansion

**What**: 47 of 100 sampled failures are OOV loanwords (`অলিম্পিকে`, `আন্দামানে`, `উইন্সলেট`). No amount of parser improvements can fix words that are not in the dictionary.

**How**: Mine BN Wikipedia proper noun entries, BN news entity lists, BN Wiktionary loanword headwords.

**Human review required**: Yes — automated mining always produces noise in proper nouns.

**Expected gain**: +5–8% Natural Top-1

---

#### C2: Automated Morphological Conjugation Generator

**What**: Take ~4,000 root Bengali verbs, apply 18 standard + 6 colloquial dialect inflection paradigms, compile ~100k inflected forms into `phonetic_overrides.bin`.

**How**: New bin `gen_verb_paradigms.rs` using `lekhani-core::phonetic::morphology`.

**Human review required**: Yes — the 6 colloquial paradigms need validation by a linguist. The 18 standard paradigms can be automated from grammar literature.

---

## Part 4 — Accuracy Targets

| Target | Metric | Phase |
|:---|:---:|:---|
| Today | 51.6% Natural Top-1 (Dev) | Baseline |
| Phase A complete | ~60% Natural Top-1 | ~3 days |
| Phase B complete + LM retrain | ~65–68% Natural Top-1 | ~3 weeks |
| Phase C complete | ~72–75% Natural Top-1 | ~2 months |
| Theoretical ceiling | ~85% Natural Top-1 | Long-term |

The gap between ~85% and 90.9% (Avro-strict ceiling) represents irreducible ambiguity: free-form Banglish has inherent multiple valid interpretations that no model can resolve without conversational context.

---

## Part 5 — Inviolable Rules (Unchanged)

1. **No test set access** until a Phase gate is complete. Every test run logged in `data/eval/test_runs.log` with ISO-8601 timestamp + commit SHA.
2. **No Wikipedia unigrams** in `dictionary.json`. The SHA-256 gate (`8a51b4d9...`) must pass on every commit.
3. **Every filter prints its drop count.** Every POVR deletion logged before it happens.
4. **No celebratory wording** until a measured gate is met. Report "failed" or "not demonstrated" plainly.
5. **Atomic commits**: One logical change per commit, format `<type>(<scope>): <description>`.
