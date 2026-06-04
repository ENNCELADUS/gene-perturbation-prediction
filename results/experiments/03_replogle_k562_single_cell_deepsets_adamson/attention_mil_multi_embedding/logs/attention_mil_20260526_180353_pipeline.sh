#!/usr/bin/env bash
set -euo pipefail
cd ~/gene-perturbation-prediction
CONFIG="configs/experiments/03_replogle_k562_single_cell_deepsets_adamson/attention_mil_multi_embedding.yaml"
OUT="results/experiments/03_replogle_k562_single_cell_deepsets_adamson/attention_mil_multi_embedding"
RUN_ID="attention_mil_20260526_180353"
printf "started_at=%s\n" "$(date -Is)"
printf "run_id=%s\n" "$RUN_ID"
uv run vcc-dep-baseline build-cell-bags --config "$CONFIG"
uv run vcc-dep-baseline run-single-cell-cv --config "$CONFIG" --run-id "$RUN_ID"
uv run vcc-dep-baseline build-external-cell-bags --config "$CONFIG" --reference-bags "$OUT/features/single_cell_bags/replogle_k562_single_cell_bags.npz" --external-name adamson_k562 --feature-set single_cell_pc_delta
uv run vcc-dep-baseline evaluate-single-cell-external --config "$CONFIG" --run-dir "$OUT/runs/$RUN_ID" --external-bags "$OUT/features/external/adamson_k562_single_cell_bags/adamson_k562_single_cell_bags.npz" --external-name adamson_k562 --feature-set single_cell_pc_delta
uv run vcc-dep-baseline build-external-cell-bags --config "$CONFIG" --reference-bags "$OUT/features/single_cell_bags/single_cell_scvi_delta/replogle_k562_single_cell_scvi_delta_bags.npz" --external-name adamson_k562 --feature-set single_cell_scvi_delta
uv run vcc-dep-baseline evaluate-single-cell-external --config "$CONFIG" --run-dir "$OUT/runs/$RUN_ID" --external-bags "$OUT/features/external/adamson_k562_single_cell_scvi_delta_bags/adamson_k562_single_cell_scvi_delta_bags.npz" --external-name adamson_k562 --feature-set single_cell_scvi_delta
uv run vcc-dep-baseline build-external-cell-bags --config "$CONFIG" --reference-bags "$OUT/features/single_cell_bags/single_cell_hvg_delta/replogle_k562_single_cell_hvg_delta_bags.npz" --external-name adamson_k562 --feature-set single_cell_hvg_delta
uv run vcc-dep-baseline evaluate-single-cell-external --config "$CONFIG" --run-dir "$OUT/runs/$RUN_ID" --external-bags "$OUT/features/external/adamson_k562_single_cell_hvg_delta_bags/adamson_k562_single_cell_hvg_delta_bags.npz" --external-name adamson_k562 --feature-set single_cell_hvg_delta
printf "finished_at=%s\n" "$(date -Is)"
