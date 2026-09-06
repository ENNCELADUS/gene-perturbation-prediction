# Joint GeneEffect execution

Run from the repository root on H20. The launcher uses `.venv-tx1/bin/python`;
set `PYTHON_BIN` to select another installed environment. Preparation runs once
in one process; training uses Torch's visible GPU count and respects
`CUDA_VISIBLE_DEVICES`. Synchronize code with Git before using the remote checkout.

```bash
hpc/run.sh prepare configs/geneeffect_joint.yaml
hpc/run.sh train configs/geneeffect_joint.yaml --run-id joint_seed0
hpc/run.sh train configs/geneeffect_joint.yaml --resume outputs/geneeffect_joint/joint_seed0/last.pt
hpc/run.sh test outputs/geneeffect_joint/joint_seed0/best.pt
uv run python -m src.evaluate --checkpoint outputs/geneeffect_joint/joint_seed0/best.pt --split val
uv run python -m src.experiments.baselines --config configs/geneeffect_joint.yaml --split test --out-dir outputs/geneeffect_joint/baselines_seed0
```

All configuration fields are explicit in `configs/geneeffect_joint.yaml`. Input
paths are relative to the repository root. Preparation requires the raw source
registry, GeneEffect CSV, supplied ESM2 table, STATE gene order and response
sources. Missing Tx1 caches additionally require the configured local Tx1 model
and a GPU. Existing Tx1 cache seed provenance is preserved. Newly encoded cells
use collation seed 0. q_sc uses raw UMI counts; response sampling seed 42 and
the fixed 10%/seed-13 holdout are preparation settings, distinct from runtime
seeds 0/0/0. The response cache header records gene order established during
raw target alignment; old headers without gene order require preparation.

Training only opens prepared caches and never rebuilds raw inputs. A fresh run
requires a new run ID. Resume uses the checkpoint's configuration and rejects
any conflicting supplied configuration. `last.pt` supports epoch-boundary resume;
`best.pt` strictly minimizes validation GeneEffect Huber loss. `metrics.jsonl`
contains every update and one validation record per completed epoch.

`run.json` records separate training and evaluation states. Testing is explicit
and does not control training completion. Checkpoint evaluation restores fitted
preprocessing, weights and actual ESM2 vectors, then exports
`evaluation/<checkpoint-name>/<split>/predictions.parquet`, `metrics.json`,
`per_line.csv`, `per_gene.csv` and `response.csv`. An export failure can be retried
with the same evaluation command without optimizer steps. Scalar test names have
`test_` prefixes. These commands produce GeneEffect evidence, not SL interaction
evidence; held-out lines retain the documented Tx1 pretraining exposure boundary.
