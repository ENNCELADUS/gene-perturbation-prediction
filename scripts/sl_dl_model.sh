#!/bin/bash
#SBATCH -J SL_DL
#SBATCH -p hexm_l40
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAL40:4
#SBATCH --output=results/experiments/08_k562_sl_pair_state_dl/slurm_%j.out
#SBATCH --error=results/experiments/08_k562_sl_pair_state_dl/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$REPO_ROOT"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

# Args: $1 = config YAML (default phase3), $2 = producer (zero|state_dl).
CONFIG_PATH="${1:-configs/experiments/08_k562_sl_pair_state_dl/phase3_bag_supervision.yaml}"
PRODUCER="${2:-state_dl}"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run 'uv sync' before running exp08 SL DL."
  exit 1
fi

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

uv run python scripts/setup_exp08_assets.py bags

# NOTE: Fold-level task parallelism (no gradient all-reduce). run_cv shards the
# (split_type, fold_id) jobs round-robin across the 4 ranks; each rank trains +
# embeds + scores its own folds on one GPU, then gather_object collects every
# rank's metric rows onto the main process, which writes the cvN/ + combined
# artifacts. Each fold is computed exactly once by exactly one rank, so the
# 4-process output is byte-identical to a 1-process run. If a fold crashes, the
# gather is a collective and the whole run aborts non-zero (fail-fast).
echo "Running exp08 SL DL with config: $CONFIG_PATH (producer: $PRODUCER)"
srun uv run --locked --no-sync --offline accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m sl_dl_model run-cv \
  --config "$CONFIG_PATH" \
  --producer "$PRODUCER"
