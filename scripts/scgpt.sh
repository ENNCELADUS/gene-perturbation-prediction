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

# echo "=============================================="
# echo "Route A + Route B1 Experiments"
# echo "=============================================="

# # ============================================
# # Route A: Forward Model + Retrieval
# # ============================================
# echo ""
# echo "=============================================="
# echo "[Route A] Data Preparation"
# echo "=============================================="
# python -m src.main --config src/configs/scgpt_forward.yaml --mode data

# echo ""
# echo "=============================================="
# echo "[Route A] Train Forward Model (DDP if multi-GPU)"
# echo "=============================================="
# run_ddp -m src.main --config src/configs/scgpt_forward.yaml --mode train

echo ""
echo "=============================================="
echo "[Route A] Build Reference Database"
echo "=============================================="
python -m src.main --config src/configs/scgpt_forward.yaml --mode build_db

echo ""
echo "=============================================="
echo "[Route A] Evaluate Retrieval"
echo "=============================================="
python -m src.main --config src/configs/scgpt_forward.yaml --mode evaluate

# ============================================
# Route B1: Gene-Level Scoring + Composition
# ============================================
echo ""
echo "=============================================="
echo "[Route B1] Train Gene-Score Model (DDP if multi-GPU)"
echo "=============================================="
run_ddp -m src.main --config src/configs/scgpt_discriminative.yaml --mode route_b1_train

echo ""
echo "=============================================="
echo "[Route B1] Evaluate Gene-Score Model"
echo "=============================================="
python -m src.main --config src/configs/scgpt_discriminative.yaml --mode route_b1_eval

echo ""
echo "=============================================="
echo "All Route A and Route B1 runs completed!"
echo "=============================================="
