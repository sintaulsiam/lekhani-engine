# Pragyan Phase 1 Step 7: Structured Feature Ranker on DEV

## 1. Execution Metadata & Provenance
- **Command Executed**: `cargo run --release -p lekhani-core --bin step7_feature_ranker`
- **Commit SHA**: `6bde0c4` (and `dec5138` initial, `cfb713d` weight-decay, `8c370b9` decoupled intent)
- **Lexicon**: `data/dictionaries/dictionary.json` (Hermetic SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`)
- **Dataset Partition (Target-Disjoint Protocol)**:
  - **Train**: 94,546 pairs (0 dropped on load, 100.0% kept)
  - **Dev-A (Tuning)**: 4,572 rows, 1,250 unique targets, 0 dropped.
  - **Dev-B (Confirmation)**: 4,707 rows, 1,250 unique targets, 0 dropped.
  - **Target Disjointness**: Exactly 0 shared Bengali targets between Dev-A and Dev-B (100% disjoint).
  - **Full DEV**: 9,279 rows, 2,500 unique targets.
  - **Test**: 9,228 rows (STRICTLY GATED; zero evaluation on test during development).
- **Core Model**: 2:1 Constrained M2M Aligner + Type-Level Order 4 Level 1 Joint Unit LM (Count $\ge 2$).
- **Ranker Architecture**: 16-Dimensional Linguistic Feature Extractor + Margin Perceptron ($E = 8$ epochs, $\eta = 0.05, \gamma = 1.0, \lambda_{\text{decay}} = 0.9995$).

---

## 2. Candidate Pool Oracle Ceiling ($K = 16$)

Candidate pools were pre-extracted by combining:
1. $K=16$ N-gram beam search hypotheses from the Joint Unit LM.
2. Injected Avro hypotheses (exact Avro transliteration and uncapitalized initial Avro transliteration).
3. Injected Dakshina supervised engine suggestions.

| Dataset Split | Rows ($N$) | Pool Reachability (Oracle Ceiling) |
|:---|:---:|:---:|
| **Dev-A (Tuning)** | 4,572 | **77.06%** |
| **Dev-B (Confirmation)** | 4,707 | **75.61%** |
| **Full DEV** | 9,279 | **76.32%** |

*Finding*: At beam width $K=16$, **76.32%** of all true targets in Full DEV are present within the candidate pool (at $K=256$, the reachability is 84.11%).

---

## 3. 16-Dimensional Linguistic Feature Weights

Learned averaged perceptron weights across the 16 feature templates:

| Feature Index | Feature Name | Learned Weight | Linguistic Role & Interpretation |
|:---|:---|:---:|:---|
| **$f_0$** | **Unit LM Score** | **38.37** | Strong base phonotactic sequence score from Order 4 Unit LM |
| **$f_1$** | **Dict Word Presence** | **26.62** | Major positive lexical prior for valid Bengali dictionary words |
| **$f_2$** | **Dict Key Equivalence** | **17.76** | Soft homophone equivalence key match (`শ/ষ/স`, `ই/ঈ`, `উ/ঊ`) |
| **$f_3$** | **Core Closed-Class Prior** | **0.05** | Auxiliary boost for closed-class pronouns and particles |
| **$f_4$** | **Avro Exact Match (in dict)** | **30.75** | Positive transliteration anchor when forming valid words |
| **$f_5$** | **Avro Uncap Match (in dict)** | **24.05** | Mobile auto-capitalization invariant anchor |
| **$f_6$** | **Avro Explicit Intent** | **58.78** | Dominant protection weight when user explicitly types Shift/casing |
| **$f_7$** | **Engine Suggestion** | **15.67** | Dakshina supervised corpus match |
| **$f_8$** | **Engine #1 Top Rank** | **21.96** | High confidence supervised engine winner |
| **$f_9$** | **Length Ratio Fidelity** | **0.67** | Penalizes candidates that deviate in character length |
| **$f_{10}$** | **Initial Char Harmony** | **1.34** | Matches vowel vs consonant category at word onset |
| **$f_{11}$** | **Final Char Harmony** | **-1.19** | Prevents spurious word-final Kar / hasanta attachments |
| **$f_{12}$** | **Cluster/Conjunct Harmony** | **-0.53** | Discriminates simple consonants from complex conjuncts |
| **$f_{13}$** | **Vowel Count Harmony** | **-1.19** | Regulates excessive or missing vowel diacritics |
| **$f_{14}$** | **Beam Reciprocal Rank** | **37.98** | Strong preference for Top-1/Top-2 hypotheses from the beam |
| **$f_{15}$** | **Bengali Suffix Match** | **2.07** | Morphological case marker / clitic match (`-er`, `-ke`, `-te`) |

---

## 4. Empirical Evaluation on Held-Out Dev Splits

### Natural Typing Persona:
| Dataset Partition | Step 8 Baseline Top-1 | Structured Ranker Top-1 | Absolute Δ | Net Hits Recovered |
|:---|:---:|:---:|:---:|:---:|
| **Dev-A (Tuning)** | 47.77% (2,184 / 4,572) | **48.43%** (2,214 / 4,572) | **+0.66%** | **+30 words** |
| **Dev-B (Confirmation)** | 47.95% (2,257 / 4,707) | **48.35%** (2,276 / 4,707) | **+0.40%** | **+19 words** |
| **Full DEV ($N=9,279$)** | 47.86% (4,441 / 9,279) | **48.39%** (4,490 / 9,279) | **+0.53%** | **+49 words** |

*(In unconstrained tuning where Avro intent was not forced, Natural Top-1 reached **49.02%** on Dev-A and **48.25%** on Dev-B with **+71 words** recovered on Full DEV).*

### Gate G4 & Gate G5 Evaluation:
- **Gate G4 (Avro-Strict Preservation)**:
  - Scored **85.49%** (4,024 / 4,707) on held-out Dev-B (unconstrained scored 68.79%, while Step 8 baseline reached 89.52%).
- **Gate G5 (Vowel-Drop Robustness)**:
  - Scored **23.31%** (1,097 / 4,707) on Dev-B `vowel_drop_p30` (Gate G5 threshold $> 13.51\%$: **PASSED**).

---

## 5. Architectural Findings & The Natural vs Avro Frontier
The empirical data demonstrates a fundamental mathematical insight:
1. **The Avro Trade-off is Real**: An unconstrained machine learning ranker trained on natural phonetic human typing naturally learns a negative weight for pure 2003 Avro mappings (because real humans drop vowels and casing that 2003 Avro expects).
2. **Explicit Intent Decoupling**: To achieve high natural accuracy while retaining muscle memory for Avro power users, explicit intent features ($f_6$) must decouple casing signals from unigram priors.
3. **Candidate Pool Headroom**: The $K=16$ candidate pool ceiling is **76.32%**. Structured ranking advances Top-1 toward this ceiling, but further gains require multi-word sentence context LM to resolve remaining homophone ambiguities.
