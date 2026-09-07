# Tx1 GMM-ridge Implementation Plan

**Goal:** Implement the approved basal-only distribution baseline and saved-model evaluation.

**Architecture:** `src/baselines/tx1_gmm.py` owns train-only feature fitting and gene
readouts. `src/experiments/tx1_gmm_ridge.py` composes prepared inputs, persistence and
the existing evaluator; the joint trainer remains unchanged.

**Tech Stack:** Existing NumPy, pandas, scikit-learn and its joblib dependency.

**Spec:** [Approved design](2026-09-07-tx1-gmm-ridge-design.md).

## Execution

- [x] Add `Tx1GMMRidge.fit(inputs)` and `context_features`/`predict` with equal-line
  cell sampling, both train-only scalers, GMM64 and per-gene alpha=1 Ridge. Tests use
  small synthetic dimensions and mixture sizes; the command fixes production K=64.
- [x] Add `fit --config --out-dir` (train and val exports) and
  `evaluate --model --split` (saved preprocessing, no fitting). Save model before
  export and report fitting/evaluation status separately. Reject reuse of a fit dir.
- [x] Test held-out mutation invariance, unequal bag contribution, missing labels,
  opposing gene slopes, invalid inputs, occupancy math, saved-model prediction
  equality and real prepared-cache command/export behavior.
- [x] Document commands and diagnostic boundaries in the runbook, preserving current
  user edits. Run the new tests plus existing baseline/evaluation/CLI tests, Ruff,
  CLI help and final diff inspection. Keep changes on the existing P1 branch.

No new experiment or performance claim is part of implementation verification.

Verification: 57 targeted tests passed across `test_tx1_gmm_ridge.py`,
`test_joint_cli.py`, `test_run_r1_residual_ladder.py` and the serial cases in
`test_joint_evaluation.py`; three unchanged distributed evaluator cases were excluded.
Ruff passed for the new modules/tests and the affected CLI test. The prepared-cache
integration uses synthetic data, including the production default K64 path, and
does not establish performance on the 226-line benchmark.
