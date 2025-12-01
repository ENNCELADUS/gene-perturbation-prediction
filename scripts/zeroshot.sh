#!/bin/bash
#SBATCH -J scGPT_zeroshot
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIATITANRTX:2
#SBATCH --exclude=ai_gpu28
#SBATCH --output=logs/zeroshot/slurm_%j.out
#SBATCH --error=logs/zeroshot/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

# Set project root
ROOT_DIR="/public/home/wangar2023/VCC_Project"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

# Initialize Conda
if [[ -f "$HOME/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc"
fi
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Activate Environment
echo "Activating conda environment 'vcc'..."
conda activate vcc

# Run Zero-Shot Pipeline
echo "Starting Zero-Shot Perturbation Experiment..."
echo "Date: $(date)"
echo "Config: src/configs/config.yaml"

set -euo pipefail

# Add local hpdex package (numba backend, no C++ build required)
export PYTHONPATH="${ROOT_DIR}/hpdex/src:${PYTHONPATH:-}"

python src/main.py --model_type baseline --threads -1

echo "Experiment complete. Results saved to paths defined in config."
