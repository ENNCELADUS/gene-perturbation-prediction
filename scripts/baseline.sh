#!/bin/bash
#SBATCH -J baseline_model
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIATITANRTX:2
#SBATCH --exclude=ai_gpu28
#SBATCH --output=logs/baseline/slurm_%j.out
#SBATCH --error=logs/baseline/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

# Set project root
ROOT_DIR="/public/home/wangar2023/VCC_Project"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

mkdir -p logs/baseline results/baseline

# Activate Environment
source ~/.bashrc
conda activate vcc

set -euo pipefail

# Add scGPT and local hpdex package to Python path
export PYTHONPATH="${ROOT_DIR}/scGPT:${ROOT_DIR}/hpdex/src:${PYTHONPATH:-}"

python src/main.py --config src/configs/baseline.yaml --model_type baseline
