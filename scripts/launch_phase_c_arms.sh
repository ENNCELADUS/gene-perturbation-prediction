#!/bin/bash
# Launch both Phase C ST response-model arms (fix-round-3 Fix 4, fix-round-4).
#
# THREE STEPS, IN THIS ORDER, every invocation:
#   1. Warm the shared response-targets cache ONCE, single-process (plain
#      python, no accelerate) -- fix-round-4.
#   2. Launch the first arm under `accelerate launch --num_processes N`.
#   3. Launch the second arm the same way, reusing step 1's cache.
#
# Step 1 exists because of a real, measured failure mode. `train_tx1_st_
# response.py`'s real-training path calls `assemble_and_project` --
# Task 5's expensive, arm-independent raw-source read -- for EVERY DDP
# rank, unconditionally, BEFORE any accelerator/DDP object exists (there is
# no rank gating in that call; the `distributed.require_exact_world_size`
# 4-rank gate is only reached afterward, inside `run_training`). A
# single-process run measured peak RSS reaching 194.6 GB after assembling
# all four `train_response_and_head` lines. Under a COLD
# --response-cache-dir, N concurrent `accelerate launch` ranks would each
# pay that same ~195 GB peak AT ONCE -- N=4 projects to ~780 GB on a shared
# 2015 GB node. `--warm-response-cache` (fix-round-4) runs that assembly
# exactly once, single-process, before any multi-rank launch, and writes
# --response-cache-dir; every subsequent reader (both arms' every rank)
# then hits the cache instead of rebuilding, because the cache's
# fingerprint is a pure function of file contents/config values, never a
# rank id, hostname, or PID (see
# `aivc_model.tx1_response_cache_warmer`/`tx1_response_gene_bags_cache`).
#
# SEQUENTIAL ARM EXECUTION BY DEFAULT (fix-round-3, Fix 4): runs the Tx1 arm
# to completion, then the HVG arm. Concurrent arms are what killed both
# processes in the 2026-07-26 incident (see
# .superpowers/sdd/phase-c/progress.md): each arm independently rebuilt the
# identical, arm-independent observed-response target data at the same
# time (spec C9 -- only ST's INPUT space differs between arms), and both
# climbed to ~621-625 GB RSS on a shared 2015 GB node before being killed.
# The shared warm cache (step 1) plus a SHARED --response-cache-dir passed
# to every arm means every read after the warm step reuses that one
# assembly instead of repeating it -- eliminating both the
# memory-multiplication problem and almost all the redundant compute.
#
# Usage:
#   scripts/launch_phase_c_arms.sh
#
# Environment overrides (all optional):
#   CACHE_DIR             Tx1 basal embedding cache
#                         (default: data/tx1_basal_embeddings/v1)
#   LINE_MANIFEST         Frozen cell-line manifest
#                         (default: results/phase_a_tx1_20260724/cell_line_manifest.csv)
#   RESPONSE_CACHE_DIR    Shared response-targets cache (fix-round-3 Fix 2,
#                         warmed by fix-round-4's step 1 below)
#                         (default: data/tx1_response_targets_cache)
#   NUM_PROCESSES         accelerate launch --num_processes (default: 4;
#                         must match required_world_size in both arm configs)
#   PYTHON_BIN            Python interpreter (default: .venv-tx1/bin/python)
#   ACCELERATE_BIN        accelerate binary (default: .venv-tx1/bin/accelerate)
#   TX1_ARM_CONFIG        (default: configs/experiments/12_tx1_st_geneeffect/phase_c/tx1_arm.yaml)
#   HVG_ARM_CONFIG        (default: configs/experiments/12_tx1_st_geneeffect/phase_c/hvg_arm.yaml)
#   ARM_ORDER             "tx1,hvg" (default) or "hvg,tx1" -- whichever arm
#                         is named FIRST also supplies the config the warm
#                         step (step 1) uses; the response-targets cache
#                         fingerprint has no input_view field (C9), so
#                         either arm's config warms an identical cache.
#   CUDA_VISIBLE_DEVICES  GPU selection (default: 0,1,2,3), mirrors
#                         scripts/run_exp05_ddp.sh's convention.
#   PHASE_C_ALLOW_CONCURRENT_ARMS=1
#       Explicit, NON-DEFAULT opt-out of sequential ARM execution (the warm
#       step always still runs first, single-process). Runs both arms at
#       once in the background afterward. Do not set this on a shared node
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

# Mirrors scripts/run_exp05_ddp.sh's own env exports for the identical
# accelerate-launch house pattern -- this script's own `_launch_arm` was
# missing all four before fix-round-4.
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

_config_for() {
  case "$1" in
    tx1) echo "$TX1_ARM_CONFIG" ;;
    hvg) echo "$HVG_ARM_CONFIG" ;;
    *) echo "unknown arm in ARM_ORDER: $1" >&2; exit 1 ;;
  esac
}

IFS=',' read -r first_arm second_arm <<< "$ARM_ORDER"

_warm_response_cache() {
  local warm_config
  warm_config="$(_config_for "$first_arm")"
  echo "=== warming response-targets cache (single process; config=${warm_config}) ==="
  "$PYTHON_BIN" scripts/train_tx1_st_response.py \
    --config "$warm_config" \
    --cache-dir "$CACHE_DIR" \
    --line-manifest "$LINE_MANIFEST" \
    --response-cache-dir "$RESPONSE_CACHE_DIR" \
    --warm-response-cache
}

_launch_arm() {
  local label="$1"
  local config="$2"
  echo "=== launching Phase C arm: ${label} (${config}) ==="
  "$PYTHON_BIN" "$ACCELERATE_BIN" launch \
    --num_processes "$NUM_PROCESSES" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    scripts/train_tx1_st_response.py \
    --config "$config" \
    --cache-dir "$CACHE_DIR" \
    --line-manifest "$LINE_MANIFEST" \
    --response-cache-dir "$RESPONSE_CACHE_DIR"
}

# Step 1, unconditionally, before any multi-rank launch (both the default
# sequential path and the opt-in concurrent-arms override below): a cold
# --response-cache-dir read by several DDP ranks at once is exactly the
# memory-multiplication failure mode this step exists to prevent.
_warm_response_cache

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
