#!/bin/bash
#SBATCH -J Tahoe-Data-Preprocessing
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/tahoe/slurm_%j.out
#SBATCH --error=logs/tahoe/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

ROOT_DIR="/public/home/wangar2023/VCC_Project"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

source ~/.bashrc
conda activate vcc

set -euo pipefail

mkdir -p logs/scgpt

detect_num_gpus() {
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        echo "$SLURM_GPUS_ON_NODE"
        return
    fi
    if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
        echo "${SLURM_GPUS_PER_NODE%%(*}"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L | wc -l
        return
    fi
    echo 0
}

NUM_GPUS="$(detect_num_gpus)"
echo "Detected GPUs: ${NUM_GPUS}"

run_ddp() {
    if [[ "${NUM_GPUS}" -gt 1 ]]; then
        torchrun --standalone --nproc_per_node="${NUM_GPUS}" "$@"
    else
        python "$@"
    fi
}

echo "=============================================="
echo "STEP1: Data Preparation"
echo "=============================================="

echo "-> Normalizing and log1p preprocessing"
python src/data/tahoe/preprocess_tahoe.py \
    --input "data/tahoe/tahoe.h5ad" \
    --output "data/processed/tahoe/tahoe_log1p.h5ad"

echo "-> Creating drug-based train/val/test split"
python -m src.data.tahoe.tahoe_dataset \
    --input "data/processed/tahoe/tahoe_log1p.h5ad" \
    --output "data/processed/tahoe/splits/tahoe_drug_split_seed42.json" \
    --seed 42 \
    --test-ratio 0.2 \
    --single-val-ratio 0.1 \
    --multi-val-ratio 0.15 \
    --single-target-test-per-gene 0 \
    --multi-test-max-targets 3 \
    --multi-test-fallback-max-targets 4
