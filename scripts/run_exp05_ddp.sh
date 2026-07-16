#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${1:-configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_fixed.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$REPO_ROOT/.venv/bin/accelerate}"

test -x "$PYTHON_BIN"
test -x "$ACCELERATE_BIN"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$PYTHON_BIN" "$ACCELERATE_BIN" launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m aivc_model.cross_validate \
  --config "$CONFIG_PATH"
