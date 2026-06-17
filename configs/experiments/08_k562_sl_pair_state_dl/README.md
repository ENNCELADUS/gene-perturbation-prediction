# exp08 — STATE-Adapter DL Model for K562 SL-Pair Ranking

Configs for the 5-phase exp08 implementation (see
`docs/superpowers/specs/2026-06-17-exp08-state-dl-sl-ranking-design.md` and
`docs/superpowers/plans/2026-06-17-exp08-state-dl-sl-ranking.md`).

## Prerequisites

1. Run the ESM2 precompute script once (network node):
   ```bash
   uv run python scripts/precompute_esm2_embeddings.py \
       --benchmark-csv data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv \
       --out data/esm2/k562_sl_universe_esm2_650M.npz \
       --seq-cache data/esm2/symbol_to_sequence.json
   ```

2. Build and cache gwps bags (local or interactive node):
   ```bash
   uv run python -c "
   from pathlib import Path
   from sl_dl_model.bags import build_gwps_bags, save_bags_npz
   from sl_dl_model.config import SLDLConfig
   cfg = SLDLConfig()
   bags = build_gwps_bags(cfg)
   save_bags_npz(bags, Path('data/exp08_cache/k562_gwps_bags.npz'))
   "
   ```

## Phase 0 — Harness parity (exp06 in-harness baseline)

Run: `uv run python -m sl_dl_model run-cv --config phase0_parity.yaml --producer zero`

**Gate:** CV2 NDCG@10 ≈ 0.042, CV3 ≈ 0.002 (match exp06 XGB). If not, harness is wrong.

## Phase 2 — SL classifier (BCE)

Run: `sbatch ../../../scripts/sl_dl_model.sh phase2_bce.yaml state_dl`

**Gate:** CV2 AUROC > 0.704, AUPR > 0.732. If not, debug encoder/pair head before Phase 3.

## Phase 3 — Bag supervision (primary)

Run: `sbatch ../../../scripts/sl_dl_model.sh phase3_bag_supervision.yaml state_dl`

**Gate (primary):** CV2/CV3 NDCG@k and MAP@k beat exp06; lift concentrated on the
covered-pair slice. This is the pass/fail for exp08.

## Phase 4 — Ablations

- Coverage-flag: `ablation_coverage_flag.yaml` (report both; no-flag is the honest one).
- Pooling swap: duplicate `phase3_bag_supervision.yaml`, set `pooling: gmm` (not yet
  implemented; mean_std is the default).
- RankNet: `lambda_rank` is **recorded in the manifest but not yet consumed by the
  training loop** (V1 is BCE-only per the spec). Wiring a pairwise RankNet term into
  `_train` is future work; setting `lambda_rank` in a config has no effect today.

## Output layout

`run_cv` writes, under `output_dir`:
- `CV1/`, `CV2/`, `CV3/` subdirs (per split present), each with `fold_metrics.csv`,
  `summary.csv`, `manifest.json`;
- top-level `fold_metrics.csv` / `summary.csv` / `manifest.json` (all splits combined);
- `official_metrics_summary.csv` (combined official summary across splits).

Artifacts are written only by the main process when launched under Accelerate.

## Interpreting Results

Read `summary.csv` → filter `metric=="ndcg@10"` and `slice=="full_universe"` → compare
mean ± std against exp06 (0.042 ± 0.008 for CV2 XGB). Lift within fold noise is null;
lift concentrated on the covered-pair slice validates the premise but documents the
uncovered-gene dilution.
