# AGENTS.md — Engineering Rules for Lekhani Engine

Instructions and architectural constraints for AI agents developing `lekhani-engine`.

## 1. Standing Rules
1. **No Synthetic / Scripted Judgments**: No grade, judgment or agreement figure may come from a script.
2. **Filter & Drop Transparency**: Every filter prints its drop count. Every deletion is reported before it happens.
3. **Commit-First Discipline**: Commit the code first, then run it, and record that commit SHA with the results.
4. **Canonical Baselines**: Quote baselines only from the committed `baseline.json`.
5. **Strict Test Split Gating**: No tuning on test. Every test access is appended to `test_runs.log`.
6. **Plain Objective Reporting**: Report "failed" or "not demonstrated" plainly. No celebratory wording until a gate is met.

## 2. Pragyan Evaluation Discipline
1. **Measured Numbers Only**: Report only measured numbers from code and benchmarks, each with the command that produced it and the commit SHA. Do not write "100% sound," "generalizes better," or any subjective conclusion unsupported by attached data tables.
2. **Strict Split Purity**:
   - `train.tsv`: For model training and alignment fitting only.
   - `dev.tsv`: For all tuning, feature decisions (unit size, order, pruning thresholds, λ weights), and error diagnoses.
   - `test.tsv`: Gated strictly. Every test-set evaluation must be logged with ISO-8601 timestamp, commit SHA, and exact configuration in `data/eval/test_runs.log` (append-only).
3. **Hermetic Lexicon**: Use `dictionary.json` only (SHA-256: `8a51b4d98bcbe147771751c60fa1d2919a84b0b910232bdcf863617ecc88fbdb`). Never introduce unverified Wikipedia-derived unigram lists or unversioned binary dictionaries into the evaluation pipeline.
4. **Tool & Code Hygiene**: Maintain exactly one canonical implementation of any tool or binary in `lekhani-core/src/bin/`. Never duplicate tools across directories. Explicitly state every file deletion.
5. **Atomic Commits**: One commit per step formatted as `<type>(<scope>): <description>`.

## 3. Inviolable Architectural Principles
- **Sub-Microsecond Hot Path**: Keystroke parsing and scoring must execute in < 100 µs. Never block the UI thread or input event loop.
- **Strict Memory Discipline**: Zero runtime heap allocations inside the fast-path token scanning hot path. Bounded, pre-sized allocations for candidate ranking.
- **Privacy**: 100% on-device. Zero network telemetry.
- **Script Integrity**: All text committed to the input connection must be canonical NFC Unicode. Never corrupt conjuncts or diacritics.
