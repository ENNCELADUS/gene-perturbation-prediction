#!/bin/bash
#SBATCH -J scGPT_finetune
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIATITANRTX:3
#SBATCH --exclude=ai_gpu28
#SBATCH --output=logs/finetune/slurm_%j.out
#SBATCH --error=logs/finetune/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

# Set project root
ROOT_DIR="/public/home/wangar2023/VCC_Project"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

# Load conda environment
source ~/.bashrc
conda activate vcc

# Add local hpdex package (numba backend, no C++ build required)
export PYTHONPATH="${ROOT_DIR}/hpdex/src:${PYTHONPATH:-}"

set -euo pipefail

# ========== Step 0: Split Data (exclude test genes) ==========
echo "=========================================="
echo "Step 0: Splitting data (30 test genes excluded from train)..."
echo "=========================================="

python scripts/data_process/split_data.py

# ========== Step 1: Convert Data to GEARS Format ==========
echo "=========================================="
echo "Step 1: Converting data to GEARS format..."
echo "=========================================="

# Force re-conversion since train.h5ad was regenerated
rm -rf data/processed/gears/vcc
python scripts/convert_to_gears.py \
    --train_path data/processed/train.h5ad \
    --output_dir data/processed/gears \
    --dataset_name vcc

# ========== Step 2: Finetune scGPT (DDP) ==========
echo "=========================================="
echo "Step 2: Finetuning scGPT with DDP..."
echo "=========================================="

# Automatically detect number of GPUs from SLURM allocation
NGPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NGPUS GPUs"

torchrun --nproc_per_node=$NGPUS \
    src/finetune.py \
    --config src/configs/finetune.yaml \
    --seed 42

# ========== Step 3: Evaluate Finetuned Model on Test Set ==========
echo "=========================================="
echo "Step 3: Evaluating finetuned model on held-out test genes..."
echo "=========================================="

# Run evaluation using the finetuned model
python src/main.py \
    --config src/configs/config.yaml \
    --model_type scgpt_finetuned \
    --threads -1

echo "=========================================="
echo "Pipeline complete. Results saved to results/scgpt_finetuned/"
echo "=========================================="
