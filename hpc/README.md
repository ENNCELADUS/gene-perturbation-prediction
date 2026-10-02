# Joint GeneEffect execution

Run from the repository root on H20. The launcher uses `.venv-tx1/bin/python`;
set `PYTHON_BIN` to select another installed environment. Training uses Torch's visible
GPU count and respects `CUDA_VISIBLE_DEVICES`. Synchronize code with Git before using the
remote checkout.

```bash
hpc/run.sh all configs/geneeffect_joint.yaml [--run-id <id>]
hpc/run.sh test outputs/geneeffect_joint/<id>/train/best.pt
uv run python -m src.evaluate --checkpoint outputs/geneeffect_joint/<id>/train/best.pt --split val
```

`all` runs the whole pipeline in one command and prints its run id first (default
`all_<UTC timestamp>`). Steps, in order, each skipped when its output already exists:

1. **Preparation** into `prepared_root` (`data/geneeffect_joint/v2`): reuse the Tx1 cache
   (only missing lines are encoded), one pass over the basal sources computing library
   sizes, the target total `T`, log-space HVG bags and `q_sc`, the log-space response
   cache, and the manifest with `expression_space`. Skipped when the manifest exists.
2. **STATE sanity line**: the released checkpoint scored on each anchor against no-change.
   Printed and written; not a gate.
3. **Response-model comparison**: six arms, leave-one-anchor-out, folds spread over the
   visible GPUs ([protocol §9](../docs/03-geneeffect-protocol.md#9-response-model-comparison-and-the-all-run)).
4. **Joint training** on all visible GPUs. Rerunning continues from `train/last.pt`.
5. **Validation evaluation** of `train/best.pt`, the baseline ladder (gene mean, K562 copy
   prior, nearest line, context-PCA ridge on Tx1 and on log HVG) and the readout head with
   the explicit gene-specific context slope on the new backbone's cached features.
6. **`summary.md`**: target total `T`, the sanity line, the comparison table and verdicts,
   and the validation table for the joint model, readout head and every baseline.

The run directory is `outputs/geneeffect_joint/<run_id>/{comparison/, train/, evaluation/val/,
baselines/val/, readout/, summary.md}`. An interrupted run is resumed by rerunning the same
command with the same `--run-id`; a fresh run needs a new run id. Resume uses the
checkpoint's configuration and rejects any conflicting configuration, so a batch-size change
needs a new run id.

`all` never evaluates the test split. `hpc/run.sh test CHECKPOINT` is the only route to it
and restores the checkpoint's fitted preprocessing, weights and ESM2 vectors without
optimizer steps. Standalone evaluation exports
`evaluation/<checkpoint-name>/<split>/predictions.parquet`, `metrics.json`, `per_line.csv`
and `per_gene.csv`; per-gene details carry residual target and prediction SD, SD ratio, RMSE
and MAE on the same finite rows and train-derived variable genes, and undefined quantities
keep explicit counts and null scalar values. An export failure is retried by rerunning the
same evaluation command. Nothing here is SL interaction evidence; held-out lines retain the
documented Tx1 pretraining exposure boundary.

## Configuration and inputs

All configuration fields are explicit in `configs/geneeffect_joint.yaml`; unknown or
missing keys are errors. Input paths are relative to the repository root. Preparation
requires the raw source registry, GeneEffect CSV, supplied ESM2 table, STATE checkpoint and
gene order, and the response and basal sources. Missing Tx1 caches additionally need the
configured local Tx1 model and a GPU; newly encoded cells use collation seed 0, and the Tx1
cache's existing on-disk format is read, never rewritten. Tx1 reads raw UMI counts; every
other expression quantity is log-normalised by whole-library size to `T`, the median
library size of the non-targeting cells in the Nadig Jurkat and HepG2 sources. Response
sampling seed 42 is a preparation setting distinct from the runtime seeds 0/0/0. Training
only opens prepared caches and never rebuilds raw inputs; a prepared root or checkpoint
from before the expression-space change is refused for lacking `expression_space`.

## Tx1 GMM-ridge baseline

This CPU/scikit-learn baseline uses the existing Tx1 basal bags directly, without
PCA or ST responses. It fits a shared diagonal GMM64 on equal numbers of cells from
each labeled training line (seed 0), then a standardized 68-feature context vector
and independent Ridge(alpha=1) per gene. Both standardizers and the GMM are train-only.
The frozen settings and comparison boundaries are in the
[design](../docs/specs/2026-09-07-tx1-gmm-ridge-design.md).

```bash
uv run python -m src.experiments.tx1_gmm_ridge fit --config configs/geneeffect_joint.yaml --out-dir outputs/baselines/tx1_gmm_seed0
uv run python -m src.experiments.tx1_gmm_ridge evaluate --model outputs/baselines/tx1_gmm_seed0/model.joblib --split val
```

Fit requires a new output directory and automatically exports train and val, never
test. `model.joblib` stores the fitted model, preprocessing and split; `run.json`
records source/configuration and separate fitting/evaluation status. `diagnostics.json`
reports convergence, component weights, training occupancies and per-line fit-cell
positions/counts. Check convergence before interpreting scores; the code does not
silently lower K or change the embedding when fitting is difficult.

`evaluation/<split>/` contains joint-training-evaluator predictions/metrics, per-line/per-gene tables and
`context_features.csv` (64 occupancies plus entropy, effective component count,
assignment confidence and negative log likelihood). Train scalars use `train_eval_`;
the response table is empty. A failed export can be retried with `evaluate`, restoring
the saved transforms and readout without refitting. Use only trusted local joblib
artifacts with their recorded sklearn version. This is a candidate baseline; no
performance improvement is established by the implementation tests.
