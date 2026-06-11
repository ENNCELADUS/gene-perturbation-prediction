#!/bin/bash
#SBATCH -J STATE-ABL
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIATITANRTX:1
#SBATCH --output=results/state/slurm_%j.out
#SBATCH --error=results/state/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$REPO_ROOT"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

CONFIG_PATH="configs/experiments/05_aivc_a_to_b_to_c/state_frozen_feature_ablation.yaml"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run 'uv sync' before running frozen STATE feature ablation."
  exit 1
fi

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Running frozen STATE feature ablation with config: $CONFIG_PATH"
srun uv run --locked --no-sync --offline python \
  src/aivc_model/state_feature_ablation.py \
  --config "$CONFIG_PATH"
