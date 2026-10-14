# AGENTS.md — Engineering Rules for Lekhani Engine

Instructions and architectural constraints for AI agents developing `lekhani-engine`.

## 1. Ground Rules & Pragyan Evaluation Discipline
1. **Measured Numbers Only**: Report only measured numbers from code and benchmarks, each with the command that produced it and the commit SHA. Do not write "100% sound," "generalizes better," or any subjective conclusion unsupported by attached data tables.
2. **Strict Split Purity**:
   - `train.tsv`: For model training and alignment fitting only.
   - `dev.tsv`: For all tuning, feature decisions (unit size, order, pruning thresholds, λ weights), and error diagnoses.
   - `test.tsv`: Gated strictly. Every test-set evaluation must be logged with ISO-8601 timestamp, commit SHA, and exact configuration in `data/eval/test_runs.log` (append-only).
3. **Identical Filter & Drop Transparency**: Any filter or truncation must be applied identically to train, dev, and test. Its drop count must be printed on every single run. Silent filtering is strictly forbidden.
4. **Hermetic Lexicon**: Use `dictionary.json` only (SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`). Never introduce unverified Wikipedia-derived unigram lists or unversioned binary dictionaries into the evaluation pipeline.
5. **Tool & Code Hygiene**: Maintain exactly one canonical implementation of any tool or binary in `lekhani-core/src/bin/`. Never duplicate tools across directories. Explicitly state every file deletion.
6. **Atomic Commits**: One commit per step formatted as `<type>(<scope>): <description>`.

## 2. Inviolable Architectural Principles
- **Sub-Microsecond Hot Path**: Keystroke parsing and scoring must execute in < 100 µs. Never block the UI thread or input event loop.
- **Strict Memory Discipline**: Zero runtime heap allocations inside the fast-path token scanning hot path. Bounded, pre-sized allocations for candidate ranking.
- **Privacy**: 100% on-device. Zero network telemetry.
- **Script Integrity**: All text committed to the input connection must be canonical NFC Unicode. Never corrupt conjuncts or diacritics.
