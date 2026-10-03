# GeneEffect implementation

Import reusable code through `src.<area>` and run Python commands as modules from
the repository root. Hatch packages the complete `src` package.

| Package | Responsibility |
| --- | --- |
| `data` | Split and batch records, GeneEffect targets with the fitted selective-gene set and per-gene residual SD (`data.geneeffect`, restored from checkpoints by `data.prepared`), STATE's log expression transform, basal/response assembly, the Tx1 embedding cache, ESM2 tables, prepared inputs, the readout-head feature cache and gene order |
| `data.prepare` | One-off raw-input builders (split, source registry, atlas and Kinker raw UMI, ESM2 universe, copy prior, PC9/HeLa basal), plus pure gene-universe helpers |
| `model` | STATE on its own HVG basal path with ESM2 perturbation tokens, the STATE-free response MLP, initialization, response computation, features, normalization, the factorised residual head (trunk MLP plus a rank-r gene × context product, predicting in units of the per-gene residual SD; STATE is skipped when its blocks are off), the training objectives (`model.losses`: Huber, standardised MSE, gene-blocked Pearson) and the readout head |
| `eval` | GeneEffect evaluation of one split, metrics (including selective-gene Spearman, selective AUPR lift and the paired line bootstrap), readout-head evaluation and comparison |
| `baselines` | Residual baseline ladder and the Tx1 GMM-ridge baseline |
| `training` | Joint optimizer loop with per-module learning rates (STATE frozen or trainable) and a warmup-cosine schedule, random or gene-blocked batches (`training.sampling`), optional anchor-balanced response replay, checkpoint selection on selective-gene Spearman and resumable state; readout-head training |
| `experiments` | Command wiring: preparation, the response-model comparison, the readout head, baselines, the Tx1 GMM-ridge baseline, the `all` run and the `revision` run |
| `experiments.historical` | Completed Tx1 probes and the separate SL context-screen builder |

Data and model modules do not import training, evaluation or experiment modules.
`data.batches` and `data.prepared` own the shared records. Pure response functions
live in `model.response`, so feature construction does not depend on a trainer. Model
construction uses `model.initialization` for the released STATE weights or a saved
checkpoint's architecture; Tx1 loader and encoder construction live in `model.tx1`.

The standard route is one command, `hpc/run.sh all CONFIG`, which runs
`src.experiments.all`: preparation, the untrained response-model comparison arms,
joint training on every chosen GPU, the trained comparison arms one job per GPU, then
validation evaluation, baselines and the readout head, then `summary.md`. [The HPC guide](../hpc/README.md) lists launch commands. One variant of the
GeneEffect revision runs through `hpc/run.sh revision CONFIG`, which runs
`src.experiments.revision`: training, validation evaluation and baselines for one config, then
`summary.md` and `revision.json`, without the response comparison or the readout head. The steps are
also modules:

```bash
uv run python -m src.experiments.all configs/geneeffect_joint.yaml --run-id <id> [--gpus 0,1]
uv run python -m src.experiments.revision configs/revision/frozen_huber.yaml --run-id <id> [--gpus 0,1]
uv run python -m src.experiments.prepare configs/geneeffect_joint.yaml
uv run python -m src.experiments.response_comparison --config configs/geneeffect_joint.yaml --out-dir <dir>
uv run python -m src.train --config configs/geneeffect_joint.yaml --run-dir <dir>
uv run python -m src.evaluate --checkpoint <best.pt> --split val
uv run python -m src.experiments.baselines --config configs/geneeffect_joint.yaml --split val --out-dir <dir>
uv run python -m src.experiments.readout --help
```

Preparation writes fixed inputs once, in STATE's log expression space (Tx1 alone reads
raw UMI). Training reads those caches and validates once per epoch on the 27 validation lines;
`train.objective` picks the loss, and the four response anchors are revisited every fourth
update only when `response_weight` is positive. The highest `val_selective_spearman` (macro
per-gene Spearman over the selective genes, fitted on training lines) selects `best.pt` and
controls early stopping. Training, collation and projection use seed 0.

The atlas preparation configuration is `configs/data/cell_line_atlas_raw_umi_27.json`;
`data.split_build` owns shared split construction helpers. The design is
[`docs/specs/2026-10-02-expression-space-and-all-pipeline-design.md`](../docs/specs/2026-10-02-expression-space-and-all-pipeline-design.md).
