#!/bin/bash
# Launch both Phase C ST response-model arms (fix-round-3, Fix 4).
#
# SEQUENTIAL BY DEFAULT: runs the Tx1 arm to completion, then the HVG arm.
# This is the documented default -- concurrent arms are what killed both
# processes in the 2026-07-26 incident (see
# .superpowers/sdd/phase-c/progress.md): each arm independently rebuilt the
# identical, arm-independent observed-response target data at the same
# time (spec C9 -- only ST's INPUT space differs between arms), and both
# climbed to ~621-625 GB RSS on a shared 2015 GB node before being killed.
# Running sequentially, with a SHARED --response-cache-dir (fix-round-3,
# Fix 2), means the second arm reuses the first's expensive assembly
# instead of repeating it -- eliminating both the memory-doubling problem
# and most of the redundant compute in one change.
#
# Usage:
#   scripts/launch_phase_c_arms.sh
#
# Environment overrides (all optional):
#   CACHE_DIR             Tx1 basal embedding cache
#                         (default: data/tx1_basal_embeddings/v1)
#   LINE_MANIFEST         Frozen cell-line manifest
#                         (default: results/phase_a_tx1_20260724/cell_line_manifest.csv)
#   RESPONSE_CACHE_DIR    Shared response-targets cache (fix-round-3, Fix 2)
#                         (default: data/tx1_response_targets_cache)
#   NUM_PROCESSES         accelerate launch --num_processes (default: 4;
#                         must match required_world_size in both arm configs)
#   PYTHON_BIN            Python interpreter (default: .venv-tx1/bin/python)
#   ACCELERATE_BIN        accelerate binary (default: .venv-tx1/bin/accelerate)
#   TX1_ARM_CONFIG        (default: configs/experiments/12_tx1_st_geneeffect/phase_c/tx1_arm.yaml)
#   HVG_ARM_CONFIG        (default: configs/experiments/12_tx1_st_geneeffect/phase_c/hvg_arm.yaml)
#   ARM_ORDER             "tx1,hvg" (default) or "hvg,tx1" -- whichever arm
#                         runs first pays the one-time cache-build cost;
#                         the second is the one that benefits.
#   PHASE_C_ALLOW_CONCURRENT_ARMS=1
#       Explicit, NON-DEFAULT opt-out of sequential execution. Runs both
#       arms at once in the background. Do not set this on a shared node
#       without first checking Fix 3's peak-RSS log lines
#       ("peak RSS so far (...)") from a prior sequential run -- concurrent
#       arms roughly DOUBLE peak RSS relative to either arm alone, which is
#       exactly what caused the 2026-07-26 incident.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CACHE_DIR="${CACHE_DIR:-data/tx1_basal_embeddings/v1}"
LINE_MANIFEST="${LINE_MANIFEST:-results/phase_a_tx1_20260724/cell_line_manifest.csv}"
RESPONSE_CACHE_DIR="${RESPONSE_CACHE_DIR:-data/tx1_response_targets_cache}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-tx1/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$REPO_ROOT/.venv-tx1/bin/accelerate}"
TX1_ARM_CONFIG="${TX1_ARM_CONFIG:-configs/experiments/12_tx1_st_geneeffect/phase_c/tx1_arm.yaml}"
HVG_ARM_CONFIG="${HVG_ARM_CONFIG:-configs/experiments/12_tx1_st_geneeffect/phase_c/hvg_arm.yaml}"
ARM_ORDER="${ARM_ORDER:-tx1,hvg}"

test -x "$PYTHON_BIN"
test -x "$ACCELERATE_BIN"

_launch_arm() {
  local label="$1"
  local config="$2"
  echo "=== launching Phase C arm: ${label} (${config}) ==="
  "$PYTHON_BIN" "$ACCELERATE_BIN" launch \
    --num_processes "$NUM_PROCESSES" \
    --num_machines 1 \
    --dynamo_backend no \
    scripts/train_tx1_st_response.py \
    --config "$config" \
    --cache-dir "$CACHE_DIR" \
    --line-manifest "$LINE_MANIFEST" \
    --response-cache-dir "$RESPONSE_CACHE_DIR"
}

_config_for() {
  case "$1" in
    tx1) echo "$TX1_ARM_CONFIG" ;;
    hvg) echo "$HVG_ARM_CONFIG" ;;
    *) echo "unknown arm in ARM_ORDER: $1" >&2; exit 1 ;;
  esac
}

IFS=',' read -r first_arm second_arm <<< "$ARM_ORDER"

if [ "${PHASE_C_ALLOW_CONCURRENT_ARMS:-0}" = "1" ]; then
  echo "WARNING: PHASE_C_ALLOW_CONCURRENT_ARMS=1 -- running both arms" \
       "CONCURRENTLY. This roughly doubles peak RSS relative to either arm" \
       "alone; check Fix 3's peak-RSS log lines from a prior sequential run" \
       "before doing this on a shared node (see this script's header)." >&2
  _launch_arm "$first_arm" "$(_config_for "$first_arm")" &
  first_pid=$!
  _launch_arm "$second_arm" "$(_config_for "$second_arm")" &
  second_pid=$!
  wait "$first_pid"
  wait "$second_pid"
else
  _launch_arm "$first_arm" "$(_config_for "$first_arm")"
  _launch_arm "$second_arm" "$(_config_for "$second_arm")"
fi
