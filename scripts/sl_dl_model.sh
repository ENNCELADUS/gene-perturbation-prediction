#!/usr/bin/env bash
#SBATCH --job-name=sl_dl_model
#SBATCH --output=logs/sl_dl_model_%j.out
#SBATCH --error=logs/sl_dl_model_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

CONFIG="${1:?usage: $0 <config.yaml> [producer]}"
PRODUCER="${2:-state_dl}"

accelerate launch --multi_gpu --num_processes=4 \
    -m sl_dl_model run-cv --config "$CONFIG" --producer "$PRODUCER"
