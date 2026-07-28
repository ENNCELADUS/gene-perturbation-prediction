#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-tx1/bin/python}"
TX1_GPU="${TX1_GPU:-0}"
HVG_GPU="${HVG_GPU:-1}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results/state}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

test -x "$PYTHON_BIN"
test "$TX1_GPU" != "$HVG_GPU"
if ((BASH_VERSINFO[0] < 5)); then
  echo "run_phase_d_parallel.sh requires Bash 5 or newer" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

COMMON_ARGS=(
  --line-manifest results/phase_a_tx1_20260724/cell_line_manifest.csv
  --phase-a-dir results/phase_a_tx1_20260724
  --tx1-cache-dir data/tx1_basal_embeddings/v1
  --depmap-gene-effect data/sl_dependency_v0/raw/depmap_26q1/CRISPRGeneEffect.csv
)
EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

CUDA_VISIBLE_DEVICES="$TX1_GPU" "$PYTHON_BIN" \
  scripts/train_tx1_geneeffect_head.py \
  --config configs/experiments/12_tx1_st_geneeffect/phase_d/tx1_arm.yaml \
  "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" \
  >"$LOG_DIR/phase_d_tx1_${RUN_STAMP}.log" 2>&1 &
TX1_PID=$!

CUDA_VISIBLE_DEVICES="$HVG_GPU" "$PYTHON_BIN" \
  scripts/train_tx1_geneeffect_head.py \
  --config configs/experiments/12_tx1_st_geneeffect/phase_d/hvg_arm.yaml \
  "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" \
  >"$LOG_DIR/phase_d_hvg_${RUN_STAMP}.log" 2>&1 &
HVG_PID=$!

trap 'kill "$TX1_PID" "$HVG_PID" 2>/dev/null || true' INT TERM
set +e
wait -n "$TX1_PID" "$HVG_PID"
FIRST_STATUS=$?
if ((FIRST_STATUS != 0)); then
  kill "$TX1_PID" "$HVG_PID" 2>/dev/null || true
  wait "$TX1_PID" 2>/dev/null
  wait "$HVG_PID" 2>/dev/null
  echo "Phase D arm failed with status $FIRST_STATUS; peer terminated" >&2
  exit 1
fi
wait
SECOND_STATUS=$?
set -e

if ((SECOND_STATUS != 0)); then
  echo "Phase D arm failed with status $SECOND_STATUS" >&2
  exit 1
fi
