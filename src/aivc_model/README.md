# `aivc_model` Implementation Notes

This package is the Exp13 GeneEffect residual path plus the Tx1 basal/response
machinery it runs on. Keep it narrow and easy to review. The contract is
[`docs/01-blueprint.md`](../../docs/01-blueprint.md) §3-4; the executable protocol is
[`docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md`](../../docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md).

## File Roles

**Exp13 residual path**

- `benchmark_split.py`: the 226-line membership authority and `assert_fit_eligible`,
  the hard guard every fitting path must call.
- `residual_target.py`: train-only `mu_hat_g` and the residual target. Pure pandas.
- `residual_ladder.py` / `residual_metrics.py`: the R1 baseline ladder and the
  per-gene across-line metric axis it and the head are both scored on.
- `geneeffect_head.py`: the axis-aware loss and the five-block `h_delta` readout.

**Tx1 / STATE substrate**

- `tx1_basal.py`: per-cell-line basal AnnData assembly from Tahoe, X-Atlas-Orion,
  and Perturb-seq h5ad sources; `tx1_response_streaming.py` bounds its memory.
- `tx1_embed_cache.py`: the Tx1-3B basal embedding cache — writer, reader, verifier.
- `tx1_response_data.py` + `tx1_response_gene_bags_cache.py`: observed-response
  `GeneBags` assembly and its fingerprinted cache.
- `tx1_predicted_response.py`: forward-only ST loading and predicted-response
  generation.
- `state_core.py`: the single definition of the STATE/gene-bag primitives
  (`GeneBags`, `StateForwardAdapter`, `Esm2PerturbationAdapter`, …).
- `state_warm_start.py`: shape-filtered warm start for Arc STATE checkpoints.
- `gene_embeddings.py`: precomputed ESM2 per-gene embeddings and the adapter.
- `gene_splits.py`: gene-universe outer-fold manifests.

Do not add a module unless an existing one becomes unreviewable.

## Edit Rules

- Keep changes local to this directory plus the matching script and tests.
- Do not weaken `benchmark_split.assert_fit_eligible`, and never fit anything —
  normalization, projection, hyperparameters, donors — on a `val`, `test`, or
  `unlabeled_train` line.
- Never center a *prediction* on a fold-fit gene mean; see `residual_ladder.py`'s
  module docstring for why that scores per-gene Spearman `+1.0` by construction.
- Validate every checkpoint load (`validate_load_result` pattern). A bare
  `strict=False` load reports success with randomly initialized weights.
- Do not preserve backward compatibility: delete obsolete paths rather than
  adding fallbacks.
- Do not hardcode data paths, checkpoint paths, dimensions, or thresholds; take
  them from CLI arguments or the frozen split JSON.

## Local Assets

Large model files and caches belong under gitignored directories, not in Git.
Download commands are explicit and user-triggered; code must never silently fetch
remote weights. If a checkpoint path is missing or incompatible, fail early
instead of falling back to random initialization.

## Runtime

There is no CLI and no training loop in this package. Entrypoints are scripts,
run through the project `uv` environment:

```bash
uv run python scripts/run_r1_residual_ladder.py --labels <geneeffect_long.csv> \
  --split-json configs/benchmarks/cell_line_geneeffect_226_split.json --out-dir <run_dir>
uv run python scripts/build_cell_line_geneeffect_226_split.py --help
```

GPU work (Tx1-3B encoding, ST forward passes) runs on the HPC under `.venv-tx1`
with `PYTHONPATH=src:.` — see the `hpc-execution` skill and `docs/04` §5.

## Verification

```bash
uv run python -m pytest tests/test_geneeffect_head.py tests/test_benchmark_split.py \
  tests/test_residual_target.py tests/test_residual_metrics.py tests/test_state_core.py
```

Always run pytest from the repository root: `tests/conftest.py` sets torch
environment variables before import and a test file run directly segfaults.
Full data or checkpoint runs are local/remote experiment jobs, not a requirement
for small implementation edits.
