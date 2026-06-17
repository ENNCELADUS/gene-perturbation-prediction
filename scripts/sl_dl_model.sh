#!/usr/bin/env bash
#SBATCH --job-name=sl_dl_model
#SBATCH --output=logs/sl_dl_model_%j.out
#SBATCH --error=logs/sl_dl_model_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

CONFIG="${1:?usage: $0 <config.yaml> [producer]}"
PRODUCER="${2:-state_dl}"

# Single-process launch. StateDlProducer trains per fold on the unwrapped model
# (gene-at-a-time forwards through the frozen STATE backbone), so DDP gradient
# sync is not engaged — launching --multi_gpu would fork ranks that each re-run
# the full CV loop and race on artifact writes. run_cv guards writes to the main
# process, but the compute itself is single-process by design. To genuinely use
# multiple GPUs, batch the per-gene forwards and wrap the train step in DDP first.
accelerate launch --num_processes=1 \
    -m sl_dl_model run-cv --config "$CONFIG" --producer "$PRODUCER"
