# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Context

- **Project**: Cancer dependency prediction from perturbation-induced
  transcriptomes.
- **Goal**: Connect AIVC / virtual-cell perturbation modeling to cancer
  dependency and synthetic-lethality target prioritization.
- **Core task**: Predict DepMap-style CRISPR gene-effect / dependency scores from
  observed or predicted post-perturbation transcriptomic responses.
- **Current branch state**: `main` contains the experiment pipeline.
- **Role**: Act as a careful junior engineer. Follow **Plan -> Confirm -> Code**
  for non-trivial research or implementation changes.

## Commands

```bash
# Environment
uv sync                              # Install/sync all dependencies
uv run python -c "import anndata, scanpy, torch; print('ok')"

# Lint and format
uv run ruff check .                  # Lint
uv run ruff format .                 # Format
uv run ruff check --fix .            # Auto-fix lint issues

# Tests
uv run python -m pytest              # Full test suite
uv run python -m pytest tests/test_dependency_baseline.py  # Pseudobulk tests
uv run python -m pytest tests/test_single_cell_baseline.py # Deep Sets tests
uv run python -m pytest -k "test_build_features"           # Single test by name

# Pipeline CLI (requires data files not in git)
uv run vcc-dep-baseline build-features --config configs/experiments/01_replogle_k562_pseudobulk_b_to_c_and_adamson_transfer/full_model_ladder.yaml
uv run vcc-dep-baseline run-cv --config configs/experiments/01_replogle_k562_pseudobulk_b_to_c_and_adamson_transfer/full_model_ladder.yaml
uv run vcc-dep-baseline build-cell-bags --config configs/experiments/03_replogle_k562_single_cell_deepsets_adamson/deepsets_cv_and_adamson.yaml
uv run vcc-dep-baseline run-single-cell-cv --config configs/experiments/03_replogle_k562_single_cell_deepsets_adamson/deepsets_cv_and_adamson.yaml
uv run vcc-dep-baseline summarize --results-dir <run_dir>
```

## Pipeline Architecture

The `src/dependency_baseline/` package implements a two-track prediction pipeline:

### Track 1: Pseudobulk Delta Baseline

```
h5ad + overlap_csv → build-features → features.npz
                                           ↓
                              run-cv (RepeatedStratifiedKFold)
                                           ↓
                              fit-final → checkpoints + rankings
```

**Feature construction** (`features.py`): Reads backed h5ad in chunks
(`_chunked_group_sums`), computes per-perturbation pseudobulk mean, subtracts
control mean to get delta expression. Derives response burden scalars, curated
biological program scores, and optional NAR viability-axis scores.

**Feature sets** (`datasets.py:feature_sets()`): Builds named feature matrices
from the NPZ — `delta_all`, `delta_mask_target`, `response_burden`,
`program_scores`, `nar_viability_scores`, `nuisance_resid_delta_all`, etc. The
`compatible_model_feature` function in `models.py` gates which models can
consume which feature sets.

**Model ladder** (`models.py:build_model_specs()`): Constructs sklearn pipelines
from config. Each `ModelSpec` has a name, estimator, and `supports_weight` flag.
Models include DummyRegressor, Ridge, ElasticNet, PCA+Ridge, PCA+RandomForest,
XGBoost, TorchMLPRegressor, and signal-decomposition variants with
`NuisanceResidualizer`.

**Evaluation loop** (`evaluation.py:execute_cv()`): Iterates over scopes ×
folds × features × models × weightings. Each fit is identified by a `job_key`
string for resume support. Results accumulate in `ArtifactStore` which writes
fold metrics, predictions, model manifests, and top-k candidates.

### Track 2: Single-Cell Deep Sets

```
h5ad → build-cell-bags → bags.npz (PCA-projected per-cell arrays)
                              ↓
              run-single-cell-cv (DeepSetsRegressor)
                              ↓
              evaluate-single-cell-external (Adamson transfer)
```

**Cell bags** (`cell_bags.py`): Computes HVG → PCA on control cells, projects
all cells into PC space, subtracts control centroid, groups by perturbation gene
into ragged bags stored as offsets into a flat array.

**Predicted-B runner** (`predicted_b.py`): Runs fold-local linear A->B baselines
that generate predicted post-perturbation bags, then train scVI/GMM Ridge B_hat
-> C predictors inside each CV fold.

**Deep Sets** (`single_cell.py:DeepSetsRegressor`): phi network encodes each
cell, mean-pooling aggregates the bag, rho network predicts GeneEffect. Handles
ragged bags via padding + mask. Inner early-stopping validation split.

### Shared Infrastructure

- **Config** (`config.py`): Frozen dataclasses loaded from YAML. `BaselineConfig`
  is the root containing `DataConfig`, `FeatureConfig`, `CvConfig`,
  `SingleCellConfig`, `ViabilityAxisConfig`, `ExperimentConfig`, `SelectionConfig`.
- **Artifacts** (`artifacts.py`): `ArtifactStore` manages incremental parquet
  writes, checkpoint paths, run manifests, and resume state.
- **Metrics** (`metrics.py`): Spearman, Pearson, RMSE, MAE, R2 for regression;
  AUROC, AUPRC, top-5% enrichment at configurable thresholds for ranking.
- **SelectionConfig**: Runtime filter that narrows scopes, features, models,
  folds, and weightings without changing model definitions. Applied via CLI
  `--scope`, `--feature-set`, `--model`, `--fold`, `--weighting` flags.

### Config Files

Configs in `configs/experiments/` are grouped to match `docs/experiment/`.
Adamson held-out test settings are embedded in the experiment configs that use
Adamson instead of being maintained as a standalone Adamson config. Key patterns:
- `models:` block defines the model ladder with `enabled` flags and hyperparams.
- `selection:` block filters what actually runs (useful for partial reruns).
- `predicted_b:` configures fold-local A->B->C linear predicted-B baselines.
- `viability_axis:` enables NAR death-signature residualization.
- `models.signal_decomposition:` enables program-score and nuisance-residualized
  model families.
- Model `variants:` lists create grid searches (e.g., multiple alpha values).

## Research Framing

```text
cell line + perturbation gene
    → observed or predicted post-perturbation transcriptome
    → dependency / essentiality score
    → context-specific target ranking
```

The matched training key is **(cell line, perturbation gene)**. Preserve it in
all intermediate tables, filenames, and model inputs.

### Stage 1: Observed Transcriptome to Dependency

Build the supervised downstream task with real post-perturbation transcriptomes.
Labels default to continuous DepMap/Achilles CRISPR gene-effect scores.

### Stage 2: Virtual-Cell Extension

After Stage 1, first test observed single-cell MIL / Set-learning (Deep Sets)
before connecting a forward perturbation model. If Set-MIL does not improve over
pseudobulk baselines, predicted-transcriptome experiments are high risk.

### Stage 3: SL Candidate Prioritization

Synthetic lethality requires context specificity. A DepMap essential gene is not
automatically an SL target. Require mutation, copy-number, lineage, or pathway
context evidence before using SL language.

## Data Rules

- Prioritize CRISPRi or knockout Perturb-seq data for DepMap alignment.
- K562 is the proof-of-concept cell line. HCT116/A549 are extensions.
- Norman is CRISPRa — treat as auxiliary, not primary label alignment.
- DepMap labels are population-level fitness readouts, not single-cell death.
- Raw `*.h5ad`, `*.csv`, checkpoints, and large artifacts are gitignored.

## Code Style

- Python 3.11+, strict type hints, absolute imports, Google-style docstrings.
- Prefer composition and small functions. Target <50 lines per function, <600
  lines per file.
- No `print` in library code; use `logging`.
- No hardcoded paths or thresholds; use config.
- Handle specific exceptions. No bare `except`.

## Environment

- Uses `uv` with project-local `.venv`.
- Prefix all Python/pytest/ruff invocations with `uv run`.
- Run `uv sync` if dependencies are missing or lockfile changed.
- The only valid CLI entry point is `uv run vcc-dep-baseline`.

## Commit Guidelines

- Conventional Commits: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`,
  `chore`, `ci`.
- PRs should summarize the change, list touched data assumptions, and include
  verification results.

## Terminology Guardrails

- Say **dependency prediction** or **essentiality ranking** for the DepMap task.
- Say **SL candidate prioritization** only with context-specificity evidence.
- Do not call DepMap scores single-cell death labels.
- Do not equate gene essentiality with synthetic lethality.
- Do not align CRISPRa perturbations with knockout dependency labels without a
  modality caveat.
