# AGENTS.md

## Research contract and sources

- The current task is context-conditioned SL ranking from basal single-cell transcriptomes in held-out **cell lines**, not held-out genes. No SL graph enters the feature path. The Feng2024 gene-holdout formulation is a separate historical track.
- `docs/01-blueprint.md` defines the research contract and claim boundaries; `docs/02-literature-review.md` defines prior art; `docs/03-experiment-protocol.md` defines the SL-pair protocol. Exp13 uses `docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md`.
- Read the relevant `docs/data/` card before using a dataset. Results and scientific status live in `docs/results/` and the blueprint's Current Scientific State, not in this file. `.superpowers/sdd/` contains local execution notes; `docs/specs/` contains tracked designs.
- Contract documents define admissible experiments and claims; code and artifacts establish what was implemented and executed. Reconcile discrepancies against the designated contract rather than treating implementation drift as a new protocol.

## Environment and implementation

- Sync working code between machines through Git only: commit, push, then pull on the H20 checkout. Do not rsync/scp/tar working trees.
- GPU work uses the H20 container and `hpc/run.sh`, without a scheduler or general qualification ladder. Ordinary jobs auto-size to visible GPUs; sweep-specific masks and runner details are in the runbook. Direct worker invocation and `--max-steps` are debug-only.
- `src/aivc_model/` contains the Exp13 and Tx1 backbone work. The other packages provide features, baselines and benchmark support; package inclusion is explicit in `pyproject.toml` under `[tool.hatch.build.targets.wheel]`.
- Run Python, pytest and Ruff from the repository root through `uv run` (for example, `rtk proxy uv run python -m pytest tests/<file>.py`). Preserve `tests/conftest.py` initialization rather than executing test files as scripts.
- `uv sync` installs the default `dev` group; research extras such as `scib` and `datasets` are optional. `arc-state` is pinned to a Git commit. Missing gitignored assets or optional dependencies can skip tests; an import check does not exercise an asset-dependent pipeline.

## Data and claim boundaries

- Join cell lines by DepMap ModelID through the checked-in map, never informal names. `K-562` and `K562` are not interchangeable join keys.
- Exp13 membership comes from `configs/benchmarks/cell_line_geneeffect_226_split.json` and `benchmark_split.assert_fit_eligible`. It does not substitute for `configs/benchmarks/context_screen_v2_split.json` or the retired Phase-A role column.
- Never center a prediction on a fold-fit gene mean: targets use `mu_hat_g^(-c)`, predictions use fold-independent `mu_bar`. Context claims require residual evaluation against context-blind priors.
- Apply blueprint §§7–8: train-side fitting/calibration only; no per-context prediction z-scoring; one label-independent pair universe; missing scores remain missing; qualify held-out-context results by Tx1 Tahoe-100M pretraining exposure.
- Single-gene essentiality and scope-closed Exp13 are not SL evidence. The current classifier supplies no joint or measured genetic-interaction quantity, so its ranking gain is not an interaction claim. Preserve the blueprint's limits on unseen genes, cross-context significance, label reversal and experimental target validation.
- Dataset meanings are not interchangeable: `jost_replogle_dual_sgrna` measures knockdown efficacy, not epistasis; `essential` and `gwps` h5ads have different gene panels; Horlbeck uses `gi_score` with negative meaning SL; screened non-hits are not universal non-SL labels.
- Preserve each protocol's evaluation unit and evidence requirements; do not impose Feng2024's five-fold reporting on the separate context or Exp13 tracks.
