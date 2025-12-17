#!/bin/bash
#SBATCH -J baseline_model
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/baseline/slurm_%j.out
#SBATCH --error=logs/baseline/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

ROOT_DIR="/public/home/wangar2023/VCC_Project"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

# Load conda environment
source ~/.bashrc
conda activate vcc

# Create logs dir
mkdir -p logs/baseline

# Run PCA baseline pipeline
python -m src.main --config src/configs/pca.yaml

echo "PCA baseline completed!"

