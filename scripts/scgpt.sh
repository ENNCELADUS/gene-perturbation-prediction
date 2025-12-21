#!/bin/bash
#SBATCH -J scGPT
#SBATCH -p hexm
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA40:4
#SBATCH --output=logs/scgpt/slurm_%j.out
#SBATCH --error=logs/scgpt/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

ROOT_DIR="/public/home/wangar2023/VCC_Project"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

# Load conda environment
source ~/.bashrc
conda activate vcc

set -euo pipefail

# Create log directory
mkdir -p logs/scgpt

latest_run_dir() {
    local base_dir="$1"
    local latest_dir=""
    latest_dir="$(ls -dt "${base_dir}"/* 2>/dev/null | head -n 1 || true)"
    if [[ -z "${latest_dir}" ]]; then
        return 1
    fi
    echo "${latest_dir}"
}

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

run_finetune() {
    if [[ "${NUM_GPUS}" -gt 1 ]]; then
        torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m src.train.finetune "$@"
    else
        python -m src.train.finetune "$@"
    fi
}

echo "=============================================="
echo "scGPT Ablative Study: Three Modes"
echo "=============================================="
echo "Mode 1: Frozen scGPT encoder (baseline)"
echo "Mode 2: scGPT + trainable retrieval head"
echo "Mode 3: scGPT + LoRA + retrieval head"
echo "=============================================="

# ============================================
# MODE 1: Frozen scGPT Encoder (Baseline)
# ============================================
echo ""
echo "=============================================="
echo "[MODE 1] Frozen scGPT Retrieval (Baseline)"
echo "=============================================="
python -m src.main --config src/configs/scgpt.yaml
SCGPT_BASELINE_DIR="$(latest_run_dir results/scgpt)" || {
    echo "Error: No scGPT baseline results found in results/scgpt" >&2
    exit 1
}
echo "[1/3] Frozen scGPT completed!"

# ============================================
# MODE 2: scGPT + Trainable Retrieval Head
# ============================================
echo ""
echo "=============================================="
echo "[MODE 2] scGPT + Trainable Retrieval Head"
echo "=============================================="
echo "Training retrieval head with InfoNCE loss..."
run_finetune \
    --config src/configs/scgpt_finetune.yaml \
    --mode head_only \
    --loss infonce
echo "Evaluating head-only fine-tuned model..."
python -m src.main \
    --config src/configs/scgpt.yaml \
    --experiment_name scgpt_head_only
SCGPT_HEAD_ONLY_DIR="$(latest_run_dir results/scgpt_head_only)" || {
    echo "Error: No scGPT head-only results found in results/scgpt_head_only" >&2
    exit 1
}
echo "[2/3] Head-only fine-tuning completed!"

# ============================================
# MODE 3: scGPT + LoRA + Retrieval Head
# ============================================
echo ""
echo "=============================================="
echo "[MODE 3] scGPT + LoRA + Retrieval Head"
echo "=============================================="
echo "Training with LoRA adapters + retrieval head..."
run_finetune \
    --config src/configs/scgpt_finetune.yaml \
    --mode lora_head \
    --loss infonce
echo "Evaluating LoRA fine-tuned model..."
python -m src.main \
    --config src/configs/scgpt.yaml \
    --experiment_name scgpt_lora_head
SCGPT_LORA_HEAD_DIR="$(latest_run_dir results/scgpt_lora_head)" || {
    echo "Error: No scGPT LoRA head results found in results/scgpt_lora_head" >&2
    exit 1
}
echo "[3/3] LoRA fine-tuning completed!"

echo ""
echo "=============================================="
echo "All scGPT modes completed!"
echo "=============================================="

# ============================================
# Generate Ablative Comparison Report
# ============================================
echo ""
echo "Generating ablative comparison report..."
python scripts/compare.py \
    --results "${SCGPT_BASELINE_DIR}" "${SCGPT_HEAD_ONLY_DIR}" "${SCGPT_LORA_HEAD_DIR}" \
    --output results/reports \
    --name scgpt_ablative_comparison
echo "Ablative comparison report saved to results/reports/"

# ============================================
# Optional: Classification Loss Variants
# ============================================
# Uncomment to also run with classification loss
# echo ""
# echo "=============================================="
# echo "[BONUS] Training with Classification Loss"
# echo "=============================================="
# 
# # Head-only with classification
# python -m src.train.finetune \
#     --config src/configs/scgpt_finetune.yaml \
#     --mode head_only \
#     --loss classification
# 
# # LoRA with classification
# python -m src.train.finetune \
#     --config src/configs/scgpt_finetune.yaml \
#     --mode lora_head \
#     --loss classification
