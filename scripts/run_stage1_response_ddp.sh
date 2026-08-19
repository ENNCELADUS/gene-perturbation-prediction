#!/bin/bash
# Launch the Exp13 Stage 1 response-model trainer
# (scripts/train_geneeffect_response_model.py).
#
# TWO PHASES, IN THIS ORDER, every invocation:
#   1. Assemble/warm the response-gene-bags cache ONCE, single-process
#      (plain python, no accelerate).
#   2. `accelerate launch` the training ranks against the already-populated
#      --response-cache-dir from step 1.
#
# Step 1 exists because of a real, measured failure mode inherited from
# Phase C's identical two-phase design (scripts/launch_phase_c_arms.sh,
# deleted at 873c99c along with the rest of the exp05/Bridge-A/Phase-A-F
# tree, but the lesson is not experiment-specific). `assemble_train_
# response_gene_bags` (src/aivc_model/tx1_response_data.py:203) runs on
# EVERY rank before any accelerator/DDP object exists -- there is no rank
# gating in that call. In Phase C, a single process measured peak RSS
# reaching 194.6 GB after assembling one arm's response bags. On
# 2026-07-26, two concurrent COLD arms independently rebuilt that same
# arm-independent target data at once and both climbed to ~621-625 GB RSS,
# killing a shared node (see the deleted Phase C launcher's header and
# .superpowers/sdd/phase-c/progress.md). Under a cold --response-cache-dir,
# N concurrent `accelerate launch` ranks here would pay the same ~195 GB
# peak AT ONCE -- do not "simplify" this back into a one-phase launch that
# hands a cold cache to accelerate directly.
#
# --assemble-only (phase 1) runs that assembly exactly once, single-process,
# and writes --response-cache-dir; every subsequent reader (every rank of
# phase 2) then hits the warm cache instead of rebuilding, because the
# cache's fingerprint is a pure function of file contents/config values,
# never a rank id, hostname, or PID.
#
# Usage:
#   scripts/run_stage1_response_ddp.sh
#   scripts/run_stage1_response_ddp.sh --help
#
# Required environment:
#   RESPONSE_CACHE_DIR   Shared response-gene-bags cache directory. No
#                         default -- the two-phase launch this script exists
#                         to implement depends on both phases pointing at
#                         the SAME cache dir, so a silently-defaulted path
#                         here is exactly the kind of mistake that causes a
#                         second cold assembly instead of a cache hit.
#
# Optional environment overrides:
#   CONFIG_PATH           Stage 1 training config
#                         (default: configs/experiments/13_geneeffect_226/stage1_response.yaml)
#   SPLIT_JSON             Frozen 226-line split
#                         (default: configs/benchmarks/cell_line_geneeffect_226_split.json)
#   CELL_LINE_MANIFEST     Frozen cell-line manifest
#                         (default: results/phase_a_tx1_20260724/cell_line_manifest.csv)
#   TX1_CACHE_DIR          Tx1 basal embedding cache
#                         (default: data/tx1_basal_embeddings/v1)
#   STATE_MODEL_DIR        Released ST checkpoint dir (var_dims.pkl source)
#                         (default: model/checkpoints/state/ST-HVG-Replogle/fewshot/k562)
#   STATE_CHECKPOINT       Released ST checkpoint file (architecture hparams)
#                         (default: $STATE_MODEL_DIR/checkpoints/final.ckpt)
#   ESM2_EMBEDDINGS        Precomputed ESM2 embeddings .npz
#                         (default: data/esm2/exp13_anchor_union_esm2_650M.npz)
#   PERTURBSEQ_SOURCES     Perturb-seq source manifest
#                         (default: configs/experiments/13_geneeffect_226/perturbseq_sources.json)
#   OUT_DIR                Checkpoint/metrics/manifest output dir
#                         (default: results/experiments/13_geneeffect_226/stage1_response)
#   NUM_PROCESSES          accelerate launch --num_processes (default:
#                         detected GPU count via nvidia-smi, or the
#                         CUDA_VISIBLE_DEVICES device count if that is
#                         already set, or 1 if neither is available --
#                         never hardcoded, the box hardware changes)
#   PYTHON_BIN             Python interpreter (default: .venv-tx1/bin/python
#                         -- .venv has no torch, see the hpc-execution
#                         skill's venv table)
#   ACCELERATE_BIN         accelerate binary (default: .venv-tx1/bin/accelerate)
#   SKIP_ASSEMBLE=1        Skip phase 1 for a re-run against an already-warm
#                         RESPONSE_CACHE_DIR.
#   CUDA_VISIBLE_DEVICES   GPU selection; if already set, NUM_PROCESSES is
#                         derived from its device count instead of nvidia-smi.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,80p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

# RESPONSE_CACHE_DIR has no default -- see the header. Checked first, before
# any binary or file existence check, so a misconfigured launch fails on
# the one thing that would otherwise silently defeat the whole point of
# this script's two-phase design.
: "${RESPONSE_CACHE_DIR:?RESPONSE_CACHE_DIR must be set -- the shared response-gene-bags cache dir both phase 1 (assemble) and phase 2 (accelerate launch) must point at. See the header comment above.}"

CONFIG_PATH="${CONFIG_PATH:-configs/experiments/13_geneeffect_226/stage1_response.yaml}"
SPLIT_JSON="${SPLIT_JSON:-configs/benchmarks/cell_line_geneeffect_226_split.json}"
CELL_LINE_MANIFEST="${CELL_LINE_MANIFEST:-results/phase_a_tx1_20260724/cell_line_manifest.csv}"
TX1_CACHE_DIR="${TX1_CACHE_DIR:-data/tx1_basal_embeddings/v1}"
STATE_MODEL_DIR="${STATE_MODEL_DIR:-model/checkpoints/state/ST-HVG-Replogle/fewshot/k562}"
STATE_CHECKPOINT="${STATE_CHECKPOINT:-$STATE_MODEL_DIR/checkpoints/final.ckpt}"
ESM2_EMBEDDINGS="${ESM2_EMBEDDINGS:-data/esm2/exp13_anchor_union_esm2_650M.npz}"
PERTURBSEQ_SOURCES="${PERTURBSEQ_SOURCES:-configs/experiments/13_geneeffect_226/perturbseq_sources.json}"
OUT_DIR="${OUT_DIR:-results/experiments/13_geneeffect_226/stage1_response}"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-tx1/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$REPO_ROOT/.venv-tx1/bin/accelerate}"

if ! test -x "$PYTHON_BIN"; then
  echo "ERROR: PYTHON_BIN ($PYTHON_BIN) is not an executable. Stage 1 is" \
       "Tx1 work and needs the .venv-tx1 interpreter, not .venv (no torch)" \
       "-- see the hpc-execution skill's venv table." >&2
  exit 1
fi
# The console script is NOT present in .venv-tx1 (verified 2026-08-19: the
# `accelerate` package imports fine at 1.13.0, but no bin/accelerate entry
# point was installed). Prefer the binary when it exists, otherwise drive the
# same launcher through `-m accelerate.commands.launch`, which is the same
# entry point the console script wraps.
if test -x "$ACCELERATE_BIN"; then
  _LAUNCH=("$PYTHON_BIN" "$ACCELERATE_BIN" launch)
elif "$PYTHON_BIN" -c "import accelerate.commands.launch" 2> /dev/null; then
  _LAUNCH=("$PYTHON_BIN" -m accelerate.commands.launch)
else
  echo "ERROR: neither ACCELERATE_BIN ($ACCELERATE_BIN) nor the" \
       "accelerate.commands.launch module is available under $PYTHON_BIN." \
       "Stage 1 is Tx1 work -- see the hpc-execution skill's venv table." >&2
  exit 1
fi

# Plain `PYTHONPATH=src` fails: the scripts do `import scripts....`, which
# needs the repo root on the path too, not just `src`.
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

if [ -z "${NUM_PROCESSES:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_PROCESSES="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
  elif command -v nvidia-smi > /dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    if [ -z "$NUM_PROCESSES" ] || [ "$NUM_PROCESSES" = "0" ]; then
      NUM_PROCESSES=1
    fi
  else
    NUM_PROCESSES=1
  fi
fi

# --config is in the COMMON args, not phase 2 only: assembly reads
# max_cells_per_gene / total_cells_per_line / data_seed from it, and those
# feed the response cache's fingerprint. A phase 1 that omitted --config
# would warm the cache under different values than phase 2 then reads,
# silently defeating the two-phase design instead of failing.
_common_args=(
  --config "$CONFIG_PATH"
  --split-json "$SPLIT_JSON"
  --cell-line-manifest "$CELL_LINE_MANIFEST"
  --tx1-cache-dir "$TX1_CACHE_DIR"
  --state-model-dir "$STATE_MODEL_DIR"
  --state-checkpoint "$STATE_CHECKPOINT"
  --esm2-embeddings "$ESM2_EMBEDDINGS"
  --perturbseq-sources "$PERTURBSEQ_SOURCES"
  --response-cache-dir "$RESPONSE_CACHE_DIR"
  --out-dir "$OUT_DIR"
)

if [ "${SKIP_ASSEMBLE:-0}" = "1" ]; then
  echo "=== SKIP_ASSEMBLE=1: skipping phase 1, reusing warm RESPONSE_CACHE_DIR=${RESPONSE_CACHE_DIR} ==="
else
  echo "=== phase 1: single-process assemble/warm (response-cache-dir=${RESPONSE_CACHE_DIR}) ==="
  "$PYTHON_BIN" scripts/train_geneeffect_response_model.py \
    --assemble-only \
    "${_common_args[@]}"
fi

echo "=== phase 2: accelerate launch, num_processes=${NUM_PROCESSES} ==="
"${_LAUNCH[@]}" \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train_geneeffect_response_model.py \
  "${_common_args[@]}" \
  --device auto
