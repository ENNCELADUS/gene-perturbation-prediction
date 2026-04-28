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

set -euo pipefail

ROOT_DIR="/public/home/wangar2023/VCC_Project"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

uv run accelerate launch --num_processes 4 --module src.main \
  --config src/scgpt/configs/norman.yaml
