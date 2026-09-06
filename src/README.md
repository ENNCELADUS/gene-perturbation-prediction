# GeneEffect implementation

Import reusable code through `src.<area>` and run Python commands as modules from
the repository root. Hatch packages the complete `src` package; there is no
`aivc_model` compatibility package.

| Package | Responsibility |
| --- | --- |
| `data` | Split and batch records, GeneEffect targets, basal/response assembly, ESM2 tables, fixed-input caches and gene order |
| `data.prepare` | Retained dataset and cache preparation commands, plus pure gene-universe helpers |
| `model` | STATE and perturbation adapters, initialization, response computation, features, normalization, pooling and the residual head |
| `eval` | Correlation and residual evaluation metrics |
| `baselines` | Residual baseline fitting and prediction |
| `training` | Distributed setup and coordinated error handling |
| `experiments` | Concrete command wiring |
| `experiments.historical` | Retained Tx1 probes and the separate context-screen builder |
| `experiments.exp13_legacy` | Temporary relocated stage-specific trainers, seals, feature stores and historical scoring; removed at joint-trainer cutover |

Data and model modules do not import training, evaluation or experiment modules.
`data.splits.FixedSplit` is independent of baseline evaluation. Model construction
uses `model.initialization`; legacy artifact loading is confined to experiments.
`data.batches` and `data.gene_bags` own the shared records. Pure response functions
live in `model.response`, so feature construction does not depend on a trainer.

Retained preparation commands keep their basenames:

```bash
uv run python -m src.data.prepare.build_exp13_tx1_cache --help
uv run python -m src.data.prepare.build_tx1_basal_embeddings --help
uv run python -m src.data.prepare.precompute_esm2_embeddings --help
uv run python -m src.experiments.baselines --help
```

The atlas preparation configuration is
`configs/data/cell_line_atlas_raw_umi_27.json`. `data.split_build` owns shared split
construction helpers. Tx1 loader and encoder construction helpers live in
`model.tx1`; reusable code does not import a CLI to load a model.

This extraction preserves retained behavior and state-dictionary parameter names.
It does not implement the new joint trainer or change the frozen benchmark split.
The approved design is in
[`docs/specs/2026-09-06-modular-joint-training-design.md`](../docs/specs/2026-09-06-modular-joint-training-design.md).
