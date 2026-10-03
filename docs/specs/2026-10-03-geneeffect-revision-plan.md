# GeneEffect Revision — Implementation Plan

> **For agentic workers:** executed with superpowers:dispatching-parallel-agents. Each task is owned by
> one agent; an agent edits **only** the files its task lists. Hard tasks run on Opus 5.5, easy ones on
> Sonnet 5.5. The orchestrator owns integration, the full suite, docs and the H20 launch.

**Goal:** the joint GeneEffect model selects on selective-gene Spearman, has a factorised gene × context
head, trains under one of three objectives and one of three STATE settings, and one command runs a
variant through training, validation and baselines.

**Spec:** `docs/specs/2026-10-03-geneeffect-revision-design.md` — read it first.

## Global constraints

- Python 3.11, `uv run python -m …` from the repo root; imports are `src.*`. Run only your own test
  files: `uv run python -m pytest tests/<file> -q > <scratch>/out.txt 2>&1` (redirect: a shell hook
  corrupts foreground pytest output). Lint: `.venv/bin/ruff check <files>`; format only files you
  touched with `.venv/bin/ruff format <files>`.
- Other agents edit other files at the same time. A failure in a file you do not own is not yours to
  fix; report it. Do not commit; the orchestrator commits.
- Config is strict (`src/experiments/config.py`, already updated): never `.get(key, default)` on config.
  New keys: `train.objective` (`huber` | `standardized_mse` | `pearson_blocks`), `train.state_mode`
  (`frozen` | `trainable`), `train.warmup_epochs`, `train.genes_per_block`,
  `features.selective_min_lines`, `features.selective_max_fraction`,
  `features.residual_sd_floor_percentile`, `model.factor_rank`. `configs/geneeffect_joint.yaml` and
  `configs/revision/*.yaml` already carry them.
- "No STATE" is not a `state_mode`: it is `head_blocks.use_delta_proj: false` and `use_s: false`. With
  both off, the model never calls STATE and the optimiser holds neither STATE nor the ESM2 adapter.
- Dependent means `gene_effect < -0.5`; one constant `DEPENDENCY_THRESHOLD = -0.5` in
  `src/data/geneeffect.py`.
- No compatibility layers: checkpoints from before this change need not load. Checkpoint loads stay
  strict.

## Shared interfaces

**Preprocessing (Task 1).** `PreparedInputs` gains `selective_genes: frozenset[str]` and
`residual_scale: pd.Series` (index `inputs.genes`, every value finite and > 0). Fitted in `load_inputs`
on labelled training lines only, or restored from `preprocessing["selective_genes"]` (list in gene
order) and `preprocessing["residual_scale"]` (`{"symbols": [...], "values": [...]}`); both written by
`preprocessing_state()`. Missing keys in a restored state raise `ValueError` naming them.

**Batches and dataset (Task 3).** `OnlineConditionBatch.gene_index: LongTensor[batch]` (position in
`inputs.genes`). `DependencyBatch.residual_scale: FloatTensor[batch]` (σ_g per row) and
`DependencyBatch.selective: BoolTensor[batch]`. `DependencyDataset.rows_by_gene() -> list[np.ndarray]`:
row positions per gene in `inputs.genes` order (empty arrays for genes without rows).

**Model (Task 3).** `GeneEffectE2EModel.forward(batch, response=None)` keeps its signature;
`delta_hat` is in residual units: `residual_scale[gene_index] * head(...)`, the scale held as a model
buffer filled from `inputs.residual_scale`. `GeneEffectE2EModel.uses_state -> bool`. The model's
`backbone.state` and `backbone.perturbations` attributes stay as they are.

**Metrics (Task 2).** `aggregate_geneeffect(..., selective_genes=...)` (required keyword) adds
`selective_spearman`, `selective_spearman_scored`, `selective_spearman_undefined`,
`selective_aupr_lift`, `selective_aupr_lift_scored`, `selective_aupr_lift_undefined`; with the split
prefix the selector is `val_selective_spearman`. `src/eval/metrics.py:paired_line_bootstrap(left,
right, selective_genes, *, repeats, seed) -> {"difference": float, "interval": [lo, hi]}`.

**Training (Task 4).** Selector `val_selective_spearman`, higher is better; `TrainState.best_score`.

## Task 1 — preprocessing (Sonnet)

Files: `src/data/geneeffect.py`, `src/data/prepared.py`, `tests/test_geneeffect_data.py`.

- `fit_selective_genes(labels, train_lines, genes, *, min_lines, max_fraction) -> frozenset[str]`:
  per gene over labelled training rows, `count = #(gene_effect < -0.5)`, `fraction = count / rows`;
  selective iff `count >= min_lines and fraction < max_fraction`.
- `fit_residual_scale(labels, train_lines, genes, *, floor_percentile) -> pd.Series`: population SD
  (ddof 0) of `residual` over training rows per gene; floor = that percentile over finite SDs; genes
  with < 2 rows or SD below the floor get the floor.
- Wire both into `PreparedInputs`/`load_inputs`/`preprocessing_state` as in Shared interfaces.
- Tests: hand-computed toy labels for both functions; fit → `preprocessing_state()` → restore round trip.

## Task 2 — metrics and every caller (Sonnet)

Files: `src/eval/geneeffect.py`, `src/eval/metrics.py`, `src/eval/readout.py`,
`src/data/readout_cache.py`, `src/experiments/baselines.py`, `src/experiments/readout.py`,
`src/experiments/tx1_gmm_ridge.py`, `tests/test_residual_metrics.py`, `tests/test_readout_entry.py`,
`tests/test_tx1_gmm_ridge.py`.

- Selective Spearman: per selective gene, Spearman of `residual_prediction` vs `residual` across lines
  (existing `_unit_spearman`, undefined stays NaN and is counted); macro mean.
- AUPR lift: per selective gene with ≥ 1 dependent and ≥ 1 non-dependent scored row,
  `sklearn.metrics.average_precision_score(gene_effect < -0.5, -geneeffect_prediction) - prevalence`.
  A constant prediction scores 0. Macro mean; count scored and undefined.
- `per_gene` table: rows for the union of variable and selective genes in `genes` order, boolean columns
  `variable` and `selective`, plus `aupr_lift`; every existing `residual_*` metric is computed over the
  variable rows only, so its values do not change.
- `paired_line_bootstrap`: both frames hold `model_id, gene_symbol, residual, residual_prediction` on the
  same keys; resample lines with replacement, recompute macro selective Spearman of each, return the
  observed difference (left − right) and the 2.5/97.5% interval. Vectorise over genes (pivot to
  gene × line arrays), 1,000 repeats over 3,000 genes × 27 lines must take seconds.
- Pass `selective_genes` from every caller; the readout cache metadata stores `selective_genes`.
- `evaluate_model` passes `inputs.selective_genes`.

## Task 3 — factorised head, batches, model assembly (Opus)

Files: `src/model/head.py`, `src/model/geneeffect.py`, `src/model/initialization.py`,
`src/data/batches.py`, `src/data/datasets.py`, `tests/test_geneeffect_head.py`,
`tests/test_joint_data.py`.

- Head: `GeneEffectResidualHead(dims, blocks, hidden, n_hidden_layers, n_genes, factor_rank)`;
  `forward(..., gene_index)` returns `MLP(F) + <G(g), C(g, c)> / sqrt(rank)`. `G(g)` = free
  `nn.Embedding(n_genes, rank)` (normal, std 0.02) + `Linear(e_g -> rank)`; `C` = an MLP
  (`hidden`, LayerNorm, GELU) over the enabled blocks among `z_c`, `q_sc`+mask, `delta_proj`,
  `s`+masks → `rank`. Masks and zeroing exactly as the existing head does them. `MLP(F)` is the existing
  trunk unchanged.
- Model: skip `predict_bags` when `delta_proj` and `s` are both disabled; `uses_state` property;
  `residual_scale` buffer indexed by `gene_index`; `delta_hat` in residual units.
- Assembly: `architecture` records `genes` (list), `head.factor_rank`, `head.n_genes`; restore asserts
  the gene order equals `inputs.genes`. The standardiser fit (`fit_startup_standardizer`) must still
  see every enabled block.
- Dataset/batches: the new fields of Shared interfaces, on the dataset device.
- Tests: head output shape, gene-index dependence, a block-disabled head never receiving STATE tensors,
  the no-STATE model never calling STATE (mock), scale applied, build → save state → restore equality.

## Task 4 — objectives, sampler, optimiser, selection (Opus)

Files: `src/model/losses.py`, `src/training/trainer.py`, `src/training/sampling.py`,
`src/training/checkpoint.py`, `tests/test_joint.py`, `tests/test_joint_checkpoint.py`,
`tests/test_objectives.py` (new).

- `geneeffect_loss(prediction, target, scale, *, objective, gene_index, selective)`: `huber` = Huber
  (delta 1) on residual units; `standardized_mse` = mean of `((prediction - target) / scale) ** 2`;
  `pearson_blocks` = standardized_mse + mean over selective genes in the batch with ≥ 3 rows of
  `1 - pearson(prediction_g, target_g)`. FP32.
- Gene-blocked sampler for `pearson_blocks`: per epoch a seeded permutation (seed from epoch) of genes
  with training rows, blocks of `genes_per_block`, dealt to ranks round-robin with the tail dropped so
  every rank takes the same number of updates; a batch is `dataset.collate` of the block's rows. Other
  objectives keep the current loader.
- Optimiser groups: head always (`head_learning_rate`); adapter (`adapter_learning_rate`) when
  `model.uses_state`; STATE (`state_learning_rate`) only when `uses_state` and `state_mode ==
  "trainable"`. Frozen STATE: `requires_grad_(False)` and kept in eval mode during training.
- Schedule: linear warmup over `warmup_epochs` epochs of updates, then cosine to 0 at `max_epochs`;
  stepped per update; its state saved in and restored from `last.pt`.
- Response replay only when `response_weight > 0`.
- Selection: `record_validation` on `val_selective_spearman` (higher wins, must be finite);
  `TrainState.best_score` replaces `best_loss`.
- Tests: each loss on hand-made tensors (Pearson term ignores non-selective genes and genes with < 3
  rows); sampler covers each gene at most once per epoch with equal per-rank counts; frozen STATE
  parameters unchanged after an update while the adapter changes; scheduler values at warmup end and
  final step; selector picks the higher Spearman. Update existing `test_joint.py` expectations.

## Task 5 — the revision route (Sonnet)

Files: `src/experiments/revision.py` (new), `hpc/run.sh`, `tests/test_revision.py` (new).

- `python -m src.experiments.revision CONFIG [--run-id ID] [--gpus 0,1,2,3]`, run dir
  `<output_root>/<run id>`, the same run-config binding, GPU choice, process pool, SIGTERM handling and
  resume rule as `src/experiments/all.py` (import its helpers; do not edit `all.py`). Steps, each
  skipped when its output exists: prepare check (`prepare_inputs`, returns at once on an existing
  root), training on every chosen GPU, `evaluation/val`, `baselines/val`, `summary.md`.
- `summary.md`: one table, rows the joint model and every baseline method, columns selective Spearman,
  selective AUPR lift, residual Pearson (variable genes), Huber, SD ratio; the paired line bootstrap of
  selective Spearman, joint minus Tx1 context-PCA ridge; best epoch and the training-diagnostic
  selective Spearman at that epoch from `train/metrics.jsonl`. Also `revision.json` with the same
  numbers for tabulating runs.
- `hpc/run.sh revision CONFIG [--run-id ID] [--gpus ...]`.
- Tests: summary and `revision.json` from synthetic metrics files; argument handling; step skipping.
