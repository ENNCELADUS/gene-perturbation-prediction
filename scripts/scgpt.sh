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

# Create logs dir
mkdir -p logs/scgpt

# Run scGPT retrieval pipeline
python -m src.main --config src/configs/scgpt.yaml

echo "scGPT retrieval completed!"
