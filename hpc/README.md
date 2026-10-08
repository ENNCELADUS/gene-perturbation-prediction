# Joint GeneEffect execution

Run from the repository root on H20. The launcher uses `.venv-tx1/bin/python`;
set `PYTHON_BIN` to select another installed environment. Every GPU step of `all` uses
every visible GPU (Torch's count, respecting `CUDA_VISIBLE_DEVICES`), or only the ids
given by `--gpus`. Synchronize code with Git before using the remote checkout.

```bash
hpc/run.sh all configs/geneeffect_joint.yaml [--run-id <id>] [--gpus 0,1,2,3]
hpc/run.sh revision configs/revision/frozen_huber.yaml [--run-id <id>] [--gpus 0,1,2,3]
hpc/run.sh test outputs/geneeffect_joint/<id>/train/best.pt
hpc/run.sh prior configs/context_prior/bridge_remedies.yaml [--run-id <id>] [--experiments affine,contrastive]
hpc/run.sh prior-export configs/context_prior/default_prior.yaml --run-id default_prior_export
hpc/run.sh revision configs/correction/huber_no_state.yaml --run-id correction_huber_no_state_seed0
uv run python -m src.evaluate --checkpoint outputs/geneeffect_joint/<id>/train/best.pt --split val
```

`all` runs the whole pipeline in one command and prints its run id first (default
`all_<UTC timestamp>`). Steps, in order, each skipped when its output already exists:

1. **Preparation** into `prepared_root` (`data/geneeffect_joint/v2`): reuse the Tx1 cache
   (only missing lines are encoded), one pass over the basal sources computing library
   sizes, the target total `T`, log-space HVG bags and `q_sc`, the log-space response
   cache, and the manifest with `expression_space`. Skipped when the manifest exists.
2. **Untrained response-model comparison arms** (no change, global mean effect, released
   STATE checkpoint) on the first chosen GPU, which also give the **STATE sanity line**: the
   released checkpoint scored on each anchor against no-change. Printed and written; not a
   gate.
3. **Joint training** through `accelerate launch` with one process per chosen GPU.
   Rerunning continues from `train/last.pt`. `best.pt` and early stopping follow the
   validation selective-gene Spearman (higher wins), not the GeneEffect loss.
4. **Trained response-model comparison arms** (STATE as in the joint model, MLP on HVG, MLP
   on Tx1; one job per arm and held-out anchor, twelve in all), one job per chosen GPU at a
   time, the next starting as soon as a GPU frees, then the comparison table and verdicts
   ([protocol §9](../docs/03-geneeffect-protocol.md#9-response-model-comparison-and-the-all-run)).
5. **Validation evaluation** of `train/best.pt`, the baseline ladder (gene mean, K562 copy
   prior, nearest line, context-PCA ridge on Tx1 and on log HVG) and the readout head with
   the explicit gene-specific context slope on the new backbone's cached features.
6. **`summary.md`**: target total `T`, the sanity line, the comparison table and verdicts,
   and the validation table (selective-gene Spearman and AUPR lift, Huber, correlations) for
   the joint model, readout head and every baseline.

The run directory is `outputs/geneeffect_joint/<run_id>/{comparison/, train/, evaluation/val/,
baselines/val/, readout/, logs/, summary.md}`; `summary.md` is rewritten on every rerun.
An interrupted run is resumed by rerunning the same command with the same `--run-id`; a
fresh run needs a new run id. A run directory is bound to the config it started with
(`run_config.json`), and training resumes from `train/last.pt` only when the config saved
there equals the current one, so a batch-size or any other config change needs a new run id.

Each GPU step uses the whole set of chosen GPUs, so the GeneEffect result arrives before
the comparison tail. `--gpus` takes ids as `CUDA_VISIBLE_DEVICES` lists them (or 0..N−1
without it) and is not bound to the run: a resumed run may use other GPUs for the
comparison jobs, which are independent and seeded per job, but unfinished training resumes
only on as many GPUs as it started on (`train/run.json`); another count is refused before
launch. Without CUDA every step runs in-process on the CPU, in the same order (the test
path). Subprocesses log to `logs/comparison_untrained.log`, `logs/train.log` and
`logs/comparison_<arm>__<anchor>.log` in the run directory. A failed subprocess stops the
run with its step, exit code and log path; comparison jobs already running finish first and
no new one starts. SIGINT or SIGTERM to `all` terminates its subprocess groups before it
exits, so no GPU worker is orphaned.

## The `revision` command

`hpc/run.sh revision CONFIG [--run-id <id>] [--gpus 0,1,2,3]` runs one variant of the
[GeneEffect revision](../docs/specs/2026-10-03-geneeffect-revision-design.md) (objective and
STATE setting chosen by the config, one per variant in `configs/revision/`) and prints its run
id first (default `revision_<UTC timestamp>`). Steps, in order, each skipped when its output
already exists:

1. **Preparation check**: `prepare_inputs` returns at once on an existing `prepared_root`; the
   prepared root of `all` is reused and nothing is re-prepared.
2. **Joint training** through `accelerate launch` with one process per chosen GPU, as in
   `all`; rerunning continues from `train/last.pt`. Selection and early stopping follow the
   validation selective-gene Spearman.
3. **Validation, then test evaluation** of `train/best.pt` and the **baselines** on the same
   split (gene mean, K562 copy prior, nearest line, context-PCA ridge on Tx1 and on log HVG),
   on the first chosen GPU. `best.pt` is chosen on validation alone; test scores it once.
4. **`revision.json` and `summary.md`**: per split, one table (selective Spearman, selective
   AUPR lift, residual Pearson over variable genes, Huber, SD ratio) for the joint model and
   every baseline and the paired line bootstrap of selective Spearman (joint model minus the Tx1
   context-PCA ridge; 1,000 resamples, seed 0); then the best epoch with its training-diagnostic
   and validation selective Spearman. Both files are rewritten on every rerun.

With `paths.prior` set (the single-cell correction, `configs/correction/`,
[protocol §12](../docs/03-geneeffect-protocol.md#12-single-cell-correction)) the run is a
correction: the model predicts the prior export's offset plus the head, validation also runs
before the first update (epoch −1, the prior alone, eligible for `best.pt`), the baselines add
the prior alone and the prior plus the Tx1 context-PCA ridge, `summary.md` bootstraps the stack
against each of them as well, and every summary carries a per-lineage table read from the split
table next to the split file. `uv run python -m src.experiments.compare_runs RUN_A RUN_B --split val`
prints the paired line bootstrap of selective Spearman between two finished runs (A minus B).

The run directory is `<output_root>/<run_id>/{train/, evaluation/{val,test}/,
baselines/{val,test}/, logs/, revision.json, summary.md}`; `output_root` is `outputs/geneeffect_revision` in
`configs/revision/*.yaml` and `outputs/geneeffect_joint` in the base config. There is no
response-model comparison and no readout head. Training uses every chosen GPU (every visible
GPU unless `--gpus` names some); unfinished training resumes only on as many GPUs as it
started on, and `--gpus` is not bound to the run. Resume, config binding
(`run_config.json`; a changed config needs a new run id), the training subprocess log
(`logs/train.log`) and SIGINT or SIGTERM handling are those of `all`. One config is one experiment at one seed:
training with validation selection, then test.

`all` never evaluates the test split; `revision` scores its own `best.pt` on test.
`hpc/run.sh test CHECKPOINT` scores any other checkpoint on test and restores the checkpoint's fitted preprocessing, weights and ESM2 vectors without
optimizer steps. Standalone evaluation (`test`, or `src.evaluate --split val|train`)
writes `evaluation/<checkpoint-name>/<split>/` beside the checkpoint, holding
`predictions.parquet`, `metrics.json`, `per_line.csv` and `per_gene.csv`; per-gene details cover
the union of the train-derived variable and selective genes with boolean `variable` and
`selective` columns, carry residual target and prediction SD, SD ratio, RMSE and MAE on the
same finite rows, and add the dependent-line `aupr_lift`; the `residual_*` metrics use the
variable rows only, `selective_spearman` and `selective_aupr_lift` the selective rows, and
undefined quantities keep explicit counts and null scalar values. An export failure is retried by rerunning the
same evaluation command. Nothing here is SL interaction evidence; held-out lines retain the
documented Tx1 pretraining exposure boundary.

## The `prior` command

`hpc/run.sh prior CONFIG [--run-id <id>] [--experiments A,B]` runs the experiment runner of the linear context prior
([protocol §10](../docs/03-geneeffect-protocol.md#10-linear-context-prior); `configs/context_prior/bridge_remedies.yaml`).
The run id defaults to a timestamp, and the route needs no GPU. The config lists experiments, each a bridge remedy
with its settings and block sets; the runner first prepares pseudo-bulk into `prepared_root` (summing the raw UMI of
each line's basal cells; the one step that reads raw data; the prepared root of `all` is reused), then for every
setting builds the bridged inputs, fits the prior for every block set and penalty, and scores validation and test
(and the bulk-input oracle on validation). It decides nothing: gains are measured against the reference row the config
pins, and the table is read by a person. `configs/context_prior/default_prior.yaml` holds the default prior alone.

`hpc/run.sh prior-selected CONFIG --run-id <id>` fits the config's reference row (its last stage must be the
data-selected genes) and writes `data_selected/{features.csv, targets.csv, ablation.json, summary.md}` into the same run
directory: feature use, per-target gains of the stage, and scores with the stage masked to or without its top features.

`hpc/run.sh prior-export CONFIG --run-id <id>` writes the config's reference row for the single-cell correction into
`outputs/context_prior/<run_id>/export/{prior.npz, prior.json}`: validation and test from the reference fit, and every
labelled single-cell training line out of fold (the bridge and every stage refitted without the line's patient-grouped
fold). An existing export is never replaced; a correction checkpoint records the export's run id, so a new export
needs a new run id. It runs on CPU (set `OMP_NUM_THREADS` and `MKL_NUM_THREADS`); `prior.json` holds its own
validation and test scores, which reproduce the reference row of the `prior` run.

The run directory is `outputs/context_prior/<run_id>/{run_config.json, rows/<experiment>__<n>.json, results.md}`.
Each setting writes its row file when finished and is skipped when the file exists, so an interrupted run resumes by
rerunning the same command with the same run id; `results.md` is rewritten from whatever rows exist on every run.
`--experiments` restricts a process to the named experiments (all of them by default), so several processes can
share one run directory: rows are per setting and never shared. A run directory is bound to the config it started
with; another config needs a new run id. The reference tables it reads are committed under
`configs/context_prior/reference/` and the extra-line membership at `configs/benchmarks/extra_bulk_lines_26Q1.json`;
the H20 host has no internet and downloads nothing.

## Configuration and inputs

All configuration fields are explicit in `configs/geneeffect_joint.yaml`; unknown or
missing keys are errors; a checkpoint from before the selective-gene revision lacks its
fitted selective genes and residual scale and is refused, and one from before the single-cell
correction lacks its recorded prior export and is refused too. Input paths are relative to the repository root. Preparation
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
design (`docs/specs/2026-09-07-tx1-gmm-ridge-design.md`, removed; `git show 1694f5c:<path>`).

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
assignment confidence and negative log likelihood). Train scalars use `train_eval_`.
A failed export can be retried with `evaluate`, restoring
the saved transforms and readout without refitting. Use only trusted local joblib
artifacts with their recorded sklearn version. This is a candidate baseline; no
performance improvement is established by the implementation tests.
