# Pragyan Phase 1 Step 3: Linguistic Tokenization and Unit Inventory Report

## 1. Execution Metadata & Git Provenance
- **Command(s) Run**:
  - `cargo run --release -p lekhani-core --bin diagnose_conjunct`
  - `cargo run --release -p lekhani-core --bin token_inventory`
  - `cargo test -p lekhani-core --test tokenization_properties`
- **Parent Commit SHA**: `3129ff1`
- **Dictionary**: `data/dictionaries/dictionary.json` (SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`)
- **Rule Invariant Added**: Any filter must be applied identically to train, dev, and test, and its drop count must be printed every run.

---

## 2. Baseline Stratification on DEV: Conjuncts vs. Non-Conjuncts and Length

As established in Step 2, conjuncts represent **51.67% of train rows** and **46.32% of dev rows**. The baseline suggestion engine (dictionary.json only, no Wikipedia contamination) was evaluated across linguistic strata on the full DEV split ($N = 9,279$).

### Persona: Natural Typing (Human Romanization)
| Stratum | DEV Rows | Top-1 Accuracy (%) | Top-3 Accuracy (%) | Unreachable (%) |
|:---|:---:|:---:|:---:|:---:|
| **Overall DEV** | 9,279 | **24.80%** | 32.48% | 65.97% |
| ├─ **With Conjunct (যুক্তবর্ণ)** | 4,298 | **20.36%** | 26.41% | **72.68%** |
| └─ **Without Conjunct** | 4,981 | **28.63%** | 37.72% | 60.17% |
| Length: 1–3 chars | 433 | 27.48% | 36.95% | 60.05% |
| Length: 4–6 chars | 3,990 | 27.37% | 37.57% | 59.97% |
| Length: 7+ chars | 4,856 | 22.45% | 27.90% | 71.42% |
| **Conjunct $\times$ Length: 4–6** | 1,003 | 21.83% | 30.71% | 67.30% |
| **Conjunct $\times$ Length: 7+** | 3,295 | **19.91%** | 25.10% | **74.32%** |
| Non-Conjunct $\times$ Length: 1–3 | 433 | 27.48% | 36.95% | 60.05% |
| Non-Conjunct $\times$ Length: 4–6 | 2,987 | 29.23% | 39.87% | 57.52% |
| Non-Conjunct $\times$ Length: 7+ | 1,561 | 27.80% | 33.82% | 65.28% |

### Key Diagnostic Finding:
Classical Avro suggestions fail disproportionately on long words containing conjuncts. While non-conjunct words achieve **28.63% Top-1** (60.17% unreachable), words with conjuncts achieve only **20.36% Top-1** with **72.68% unreachable**. For long conjunct-bearing words ($7+$ chars), reachability drops to a low **25.68%** (**74.32% unreachable**). Treating conjuncts as single atomic linguistic tokens in alignment is the primary architectural requirement.

---

## 3. Linguistic Token Inventory (Pre-Pruning)

Tokenization (`lekhani_core::tokenization`) extracts fine-grained atomic units across all 94,546 train pairs, 9,279 dev pairs, 9,228 test pairs, and all 159,423 entries in `dictionary.json`.

### Bengali Token Inventory
| Category | Unique Tokens | Examples | Notes |
|:---|:---:|:---|:---|
| **Consonants (`CONS`)** | 45 | `ক`, `খ`, `গ`, `ঘ`, `ঙ`, `ৎ`, `ড়`, `ঢ়`, `য়` | Base consonants + atomic nukta + khanda ta |
| **Conjuncts (`CONJ`)** | 365 | `ক্ক`, `ক্ট`, `ক্ট্র`, `ক্ত`, `ক্ত্য`, `ক্ষ`, `জ্ঞ` | Complete atomic conjunct clusters ($C + \text{্} + C$). Never split. |
| **Vowel Kars (`KAR`)** | 10 | `া`, `ি`, `ী`, `ু`, `ূ`, `ৃ`, `ে`, `ৈ`, `ো`, `ৌ` | Dependent vowel diacritics |
| **Independent Vowels (`IVOW`)** | 11 | `অ`, `আ`, `ই`, `ঈ`, `উ`, `ঊ`, `ঋ`, `এ`, `ঐ`, `ও`, `ঔ` | Syllable-initial or standalone vowels |
| **Modifiers (`MOD`)** | 3 | `ঁ` (chandrabindu), `ং` (anusvara), `ঃ` (visarga) | Nasal and aspirate diacritics |
| **Other (Symbols)** | 3 | `্` (isolated hasanta), `২`, `‌` (ZWNJ) | Retained for exact lossless reconstruction |
| **Total Bengali Tokens** | **437** | — | **Zero vocabulary loss; bounded inventory** |

### Roman Token Inventory
- **Total Unique Roman Tokens**: **66**
- Inventory components:
  - 11 Consonant digraphs: `chh`, `kh`, `gh`, `ch`, `th`, `dh`, `ph`, `bh`, `sh`, `ng`, `nj`
  - 7 Vowel digraphs: `aa`, `ee`, `oo`, `oi`, `ou`, `ai`, `au`
  - 13 Doubled consonants: `tt`, `ss`, `kk`, `dd`, `bb`, `mm`, `ll`, `nn`, `pp`, `cc`, `rr`, `gg`, `jj`
  - 26 Single ASCII letters: `a`..`z`
  - 9 Infrequent single characters / symbols (e.g. punctuation or loan markers)

---

## 4. Property Test Verification & Hard Constraints

Automated property tests in `lekhani-core/tests/tokenization_properties.rs`:

| Test Name | Dataset / Scope | Measured Result | Status |
|:---|:---|:---:|:---:|
| `test_bengali_roundtrip_concatenation_dev_split` | All 9,279 DEV pairs | 9,279 / 9,279 (100.0%) | **PASS** |
| `test_roman_roundtrip_concatenation_dev_split` | All 9,279 DEV pairs | 9,279 / 9,279 (100.0%) | **PASS** |
| Full Dataset Round-Trip Concatenation | 113,053 pairs (Train+Dev+Test) | 113,053 / 113,053 (**100.0000%**) | **PASS** |
| Dictionary Round-Trip Concatenation | 159,423 `dictionary.json` entries | 159,423 / 159,423 (**100.0000%**) | **PASS** |
| Hard Constraint Verification | Translation unit whitelist & rules | 0 Violations | **PASS** |
| Silent Filter Guard | Unlogged drop detection | Kept: 9,279 / Dropped: 0 | **PASS** |

### Translation Unit Rules Enforced:
1. **Ratio Budget**: Default ratio is 1:1, 1:0 (deletion), or 0:1 (insertion).
2. **1:2 Whitelist Only**: Only explicit phonetic compounds authorized (`x → ক্স`, `a → অ্যা`, `ri → ঋ`/`ৃ`).
3. **2:1 Doubled Consonant Rule**: Authorized exclusively for geminates (`kk → ক`, `tt → ট`).
4. **Vowel Dependency Invariant**: No unit begins with a `VowelKar` (`KAR`) on the Bengali side unless the Roman side contains a vowel token (`a`, `e`, `i`, `o`, `u`, `aa`, etc.).
5. **Conjunct Atomicity**: Every conjunct is represented as a single atomic token; hasantas cannot be fragmented into isolated pieces.
