# Operational utilities

`download_tahoe_source_shards.sh` downloads requested Tahoe source shards. Model
training and evaluation use [`hpc/run.sh`](../hpc/README.md).

Python preparation commands live in `src/data/prepare/` and run from the
repository root as modules:

```bash
uv run python -m src.experiments.prepare configs/geneeffect_joint.yaml
uv run python -m src.data.prepare.prepare_kinker_umi_h5ad --help
uv run python -m src.data.prepare.build_exp13_tx1_cache --help
uv run python -m src.data.prepare.precompute_esm2_embeddings --help
```

The atlas configuration is `configs/data/cell_line_atlas_raw_umi_27.json`.
Shared split helpers live in `src/data/split_build.py`; Tx1 construction helpers
live in `src/model/tx1.py`. The four retained historical preparation and diagnostic
commands are documented in
[`src/experiments/historical/`](../src/experiments/historical/README.md).

The retired Stage 1/2 commands are available in Git snapshot `e6341d2`.
