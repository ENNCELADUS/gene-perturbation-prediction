# GeneEffect implementation

Import reusable code through `src.<area>` and run Python commands as modules from
the repository root. Hatch packages the complete `src` package; there is no
`aivc_model` compatibility package.

| Package | Responsibility |
| --- | --- |
| `data` | Split and batch records, GeneEffect targets, basal/response assembly, ESM2 tables, fixed-input caches and gene order |
| `data.prepare` | Retained dataset and cache preparation commands, plus pure gene-universe helpers |
| `model` | STATE and perturbation adapters, initialization, response computation, features, normalization, pooling and the residual head |
| `eval` | Epoch and held-out evaluation: loss terms, correlations, aligned predictions and coverage |
| `baselines` | Residual baseline fitting and prediction |
| `training` | Joint optimizer loop, balanced response replay, checkpoint selection, resumable state and distributed setup |
| `experiments` | Concrete command wiring |
| `experiments.historical` | Retained Tx1 probes and the separate context-screen builder |

Data and model modules do not import training, evaluation or experiment modules.
`data.splits.FixedSplit` is independent of baseline evaluation. Model construction
uses `model.initialization` for fresh upstream weights or saved checkpoint architecture.
`data.batches` and `data.gene_bags` own the shared records. Pure response functions
live in `model.response`, so feature construction does not depend on a trainer.

Retained preparation commands keep their basenames:

```bash
uv run python -m src.data.prepare.build_exp13_tx1_cache --help
uv run python -m src.data.prepare.build_tx1_basal_embeddings --help
uv run python -m src.data.prepare.precompute_esm2_embeddings --help
uv run python -m src.experiments.baselines --help
```

The standard route is `src.experiments.prepare`, then `src.train`, then
`src.evaluate`; [the HPC guide](../hpc/README.md) lists launch commands.
Preparation writes fixed inputs once. Training reads those caches, revisits the
four response anchors every fourth update, and validates once per epoch. The
lowest `val_geneeffect_loss` selects `best.pt` and controls early stopping.

The atlas preparation configuration is
`configs/data/cell_line_atlas_raw_umi_27.json`. `data.split_build` owns shared split
construction helpers. Tx1 loader and encoder construction helpers live in
`model.tx1`; reusable code does not import a CLI to load a model.

Training, collation and projection use seed 0. The fixed benchmark membership
is unchanged. The approved design is in
[`docs/specs/2026-09-06-modular-joint-training-design.md`](../docs/specs/2026-09-06-modular-joint-training-design.md).
