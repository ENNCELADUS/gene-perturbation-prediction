#!/bin/bash
#SBATCH -J scGPT_Tahoe
#SBATCH -p hexm_l40
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAL40:4
#SBATCH --output=logs/scgpt_tahoe/slurm_%j.out
#SBATCH --error=logs/scgpt_tahoe/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

ROOT_DIR="/public/home/wangar2023/VCC_Project"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

source ~/.bashrc
conda activate vcc

set -euo pipefail

mkdir -p logs/scgpt_tahoe

# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_BLOCKING_WAIT=1
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

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

CONFIG="${CONFIG:-src_tahoe/configs/scgpt_discriminative_tahoe.yaml}"

echo ""
echo "=============================================="
echo "Tahoe Data Preparation"
echo "=============================================="
python -m src_tahoe.main --config "${CONFIG}" --mode data

echo ""
echo "=============================================="
echo "Tahoe [Route B1] Train Gene-Score Model (DDP if multi-GPU)"
echo "=============================================="
run_ddp -m src_tahoe.main --config "${CONFIG}" --mode route_b1_train

echo ""
echo "=============================================="
echo "Tahoe [Route B1] Evaluate Gene-Score Model"
echo "=============================================="
python -m src_tahoe.main --config "${CONFIG}" --mode route_b1_eval

echo ""
echo "=============================================="
echo "Tahoe Route B1 run completed!"
echo "=============================================="
