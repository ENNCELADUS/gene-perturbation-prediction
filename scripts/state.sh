#!/bin/bash
#SBATCH -J STATE
#SBATCH -p hexm_l40
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAL40:4
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

CONFIG_PATH="${CONFIG_PATH:-configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml}"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run 'uv sync' before running STATE AIVC."
  exit 1
fi

echo "Running STATE AIVC with config: $CONFIG_PATH"
srun scripts/run_exp05_ddp.sh "$CONFIG_PATH"
