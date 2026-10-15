# Pragyan Phase 1 Step 3 (Revised): Linguistic Tokenization and Unit Inventory Report

## 1. Execution Metadata & Git Provenance
- **Parent Commit SHA**: `5110f5a`
- **Commands Executed**:
  - `cargo run --release -p lekhani-core --bin diagnose_conjunct`
  - `cargo run --release -p lekhani-core --bin token_inventory`
  - `cargo test --release -p lekhani-core --test tokenization_properties`
  - `cargo test --release -p lekhani-core --test phonetic_eval`
- **Dictionary Tested**: `data/dictionaries/dictionary.json` (Hermetic SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`)
- **Filter Invariant**: Zero silent drops. All filters apply identically across splits with logged drop counts.

---

## 2. Baseline Stratification on DEV: Input Variation vs. Conjunct Degradation

Length is explicitly measured as **Target Bengali Unicode Character Count** (scalar values).

### Persona: Natural Typing (Human Romanization, N = 9,279 DEV pairs)
| Stratum | DEV Rows | Top-1 Accuracy (%) | Top-3 Accuracy (%) | Unreachable (%) |
|:---|:---:|:---:|:---:|:---:|
| **Overall DEV** | 9,279 | **24.86%** (2,307 / 9,279) | 32.55% | 65.90% |
| ├─ **With Conjunct (যুক্তবর্ণ)** | 4,298 | **20.40%** (877 / 4,298) | 26.43% | **72.64%** |
| └─ **Without Conjunct** | 4,981 | **28.71%** (1,430 / 4,981) | 37.82% | 60.09% |
| Length: 1–3 chars | 433 | 27.48% (119 / 433) | 36.95% | 60.05% |
| Length: 4–6 chars | 3,990 | 27.42% (1,094 / 3,990) | 37.62% | 59.90% |
| Length: 7+ chars | 4,856 | 22.53% (1,094 / 4,856) | 27.99% | 71.36% |
| **Conjunct $\times$ Length: 4–6** | 1,003 | 21.93% (220 / 1,003) | 30.81% | 67.20% |
| **Conjunct $\times$ Length: 7+** | 3,295 | **19.97%** (658 / 3,295) | 25.16% | **74.26%** |
| Non-Conjunct $\times$ Length: 1–3 | 433 | 27.48% (119 / 433) | 36.95% | 60.05% |
| Non-Conjunct $\times$ Length: 4–6 | 2,987 | 29.29% (875 / 2,987) | 39.94% | 57.48% |
| Non-Conjunct $\times$ Length: 7+ | 1,561 | 27.87% (435 / 1,561) | 33.95% | 65.21% |

### Corrected Linguistic Finding:
- **Base Failure Rate**: Even short non-conjunct words (1–3 chars) exhibit a **60.05% unreachability rate** and only 27.48% Top-1.
- **The Dominant Bottleneck**: The primary challenge of natural mobile romanization is pervasive vowel and consonant phonetic variation (`a` $\leftrightarrow$ অ/আ/dropped, `t` $\leftrightarrow$ ত/ট, `s` $\leftrightarrow$ স/শ), which creates a baseline unreachability of ~60% across all length buckets.
- **The Secondary Conjunct Penalty**: Within identical length strata, conjuncts impose a measurable 7–8 percentage point Top-1 penalty:
  - Length 4–6: 29.29% (non-conjunct) vs. 21.93% (conjunct) $\rightarrow$ **$-7.36\%$ penalty**
  - Length 7+: 27.87% (non-conjunct) vs. 19.97% (conjunct) $\rightarrow$ **$-7.90\%$ penalty**
- **Conclusion**: Conjunct-aware atomic tokenization is necessary for phonotactic fidelity, but the primary gains in Pragyan will come from learning translation distributions for ambiguous consonants and vowels.

---

## 3. Train-Derived Linguistic Token Inventory & OOV Analysis

To prevent data leakage, the model's token inventory is derived **strictly from the 94,546 Train pairs and `dictionary.json`**, without referencing Dev or Test splits.

### Model Token Inventory (Train + Hermetic Dictionary Only)
| Category | Train Types | Train+Dict Types | Rule Definition / Sample Items |
|:---|:---:|:---:|:---|
| **Consonants (`CONS`)** | 42 | 45 | Base consonants + atomic nukta + khanda ta (`ক`, `খ`, `ৎ`, `ড়`, etc.) |
| **Conjuncts (`CONJ`)** | 273 | 362 | Deterministic rule-based cluster ($C + \text{্} + C$). Not statistical. |
| **Vowel Kars (`KAR`)** | 10 | 10 | Dependent vowel diacritics (`া`, `ি`, `ী`, `ু`, `ূ`, `ৃ`, `ে`, `ৈ`, `ো`, `ৌ`) |
| **Independent Vowels (`IVOW`)** | 11 | 14 | Standalone vowels + composite vowels (`অ`, `আ`, `অ্যা`, `এ্যা`, etc.) |
| **Modifiers (`MOD`)** | 3 | 3 | Diacritics (`ঁ`, `ং`, `ঃ`) |
| **Other / Symbols** | 2 | 3 | Punctuation / Numerals (`্`, `২`, `\u{200c}`) |
| **Total Model Tokens** | — | **437** | **Zero Dev/Test Data Leakage** |

### Conjunct Out-of-Vocabulary (OOV) Analysis Against Train
- **Dev Conjunct OOV**: 3 / 151 types (**1.99%**), 12 / 5,179 tokens (**0.23%**)
  - Unseen types in Dev: `স্ত্ত` (`s-t-t`), `ত্খ` (`t-kh`), `র্ন্দ` (`r-n-d`)
- **Test Conjunct OOV**: 3 / 154 types (**1.95%**), 20 / 5,064 tokens (**0.39%**)
  - Unseen types in Test: `দ্দ্য` (`d-d-y`), `গ্ম` (`g-m`), `স্ক্ল` (`s-k-l`)
- **OOV Backoff Mechanism**: `CONJ` is a rule-based grapheme cluster ($C + \text{্} + C$), not an open-ended vocabulary. Any unseen conjunct at runtime deterministically backs off to component single consonants and hasanta ($C_1 + \text{্} + C_2$), guaranteeing 100% reachability.

---

## 4. Roman Segmentation Lattice

Greedy longest-match on the Roman side commits prematurely to single tokenizations (e.g. `ng` $\rightarrow$ `ঙ` vs `n`+`g` $\rightarrow$ `ন`+`গ` in `bhangon` vs `bangla`).
We implemented `roman_segmentation_lattice(text: &str)` in `lekhani-core/src/tokenization.rs`.
- Constructs a directed acyclic graph (DAG) offering both decomposed single-letter paths and composite digraph/geminate paths.
- Verified in `test_roman_segmentation_lattice_offers_both_decomposed_and_digraph_paths`.
- The M2M aligner will search the product lattice between Roman alternatives and Bengali tokens, and log the disagreement frequency against greedy segmentation.

---

## 5. Strengthened Property Tests & Invariant Verification

Automated in `lekhani-core/tests/tokenization_properties.rs`:

| Test Name | Dataset Scope | Verified Invariant | Status |
|:---|:---|:---|:---:|
| `test_no_hasanta_at_token_boundary_except_word_end` | All 9,279 DEV pairs | No mid-word tokens end with hasanta; atomic grouping preserves clusters. | **PASS** |
| `test_no_kar_at_word_start` | All 113,053 pairs (Train+Dev+Test) | No tokenized word starts with a dependent `KAR` diacritic. | **PASS** |
| `test_tokenization_idempotence_on_dev` | All 9,279 DEV pairs | `tokenize(join(tokenize(w))) == tokenize(w)`. 0 deviations. | **PASS** |
| `test_bengali_roundtrip_concatenation_dev_split` | All 9,279 DEV pairs | Lossless exact round-trip string reconstruction. | **PASS** |
| `test_roman_roundtrip_concatenation_dev_split` | All 9,279 DEV pairs | Lossless exact Roman input reconstruction. | **PASS** |
| `test_special_character_accounting_and_decisions` | DEV split | Explicit audit: 0 `২`, 9 trailing hasanta (`আল্লাহ্`, `টাইমস্`), 0 ZWNJ. | **PASS** |
| `test_hard_constraints_on_translation_units` | Whitelist rules | Structural unit validity: max 2:2, valid ratio, no naked diacritics. | **PASS** |
| `test_real_tampered_dictionary_file_fails_gate` | Disk file mutation | Real loader panics with hash mismatch when 1 byte is corrupted. | **PASS** |

### Special Character Accounting & Policy:
1. `২` (Numeral 2): 0 rows in Dev, 16 in Train. Kept as valid conversational tokens (`২জি` $\rightarrow$ `duiji`, `২টি` $\rightarrow$ `duiti`).
2. **Trailing Hasanta**: 9 rows in Dev (`আল্লাহ্`, `শাহ্`, `টাইমস্`, `পিপলস্`). Kept intact: phonologically required for Arabic loan honorifics and English plural loanwords.
3. **ZWNJ (`\u{200c}`)**: 0 rows in Dev, 2 in Train. Handled safely.

### Hard Constraints Body & Whitelist:
`test_hard_constraints_on_translation_units` enforces:
1. Max Roman chunk size: 2.
2. Max Bengali chunk size: 2.
3. Roman tokens $\le 2$ chars must be drawn from digraph/geminate inventory or ASCII letters.
4. Bengali tokens cannot be an unattached `KAR` unless mapped from a roman vowel.
5. Deletion (1:0) allowed only for silent letters (`h`, `e`, `a`); insertion (0:1) allowed only for inherent vowel (`অ`).

---

## 6. Pre-Pruning Unit Inventory Size

- Unique Roman Tokens (Train): **66**
- Unique Bengali Tokens (Train+Dict): **437**
- Theoretical unconstrained cross-product: $66 \times 437 = 28,842$ candidate units.
- Pre-pruning co-occurrence filtering (pairs co-occurring within the same word in Train + hard phonetic constraints): **9,998 candidate translation units**.
- Step 4 EM will further prune this inventory down to significant high-probability units.
