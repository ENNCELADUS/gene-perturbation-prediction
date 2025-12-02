#!/bin/bash
#SBATCH -J linear_regression
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIATITANRTX:1
#SBATCH --exclude=ai_gpu28
#SBATCH --output=logs/linear/slurm_%j.out
#SBATCH --error=logs/linear/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

# Set project root
ROOT_DIR="/public/home/wangar2023/VCC_Project"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

# Load conda environment
source ~/.bashrc
conda activate vcc

# Add scGPT and local hpdex package to Python path
export PYTHONPATH="${ROOT_DIR}/scGPT:${ROOT_DIR}/hpdex/src:${PYTHONPATH:-}"

set -euo pipefail

# ========== Step 1: Train Linear Regression Model ==========
echo "=========================================="
echo "Step 1: Training Linear Regression Model..."
echo "=========================================="

python src/linear.py \
    --config src/configs/linear.yaml \
    --seed 42

# ========== Step 2: Evaluate Linear Model on Test Set ==========
echo "=========================================="
echo "Step 2: Evaluating linear model on held-out test genes..."
echo "=========================================="

python src/main.py \
    --config src/configs/linear.yaml \
    --model_type linear \
    --threads -1

echo "=========================================="
echo "Pipeline complete. Results saved to results/linear/"
echo "=========================================="
