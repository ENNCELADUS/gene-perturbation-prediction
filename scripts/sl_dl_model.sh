#!/bin/bash
#SBATCH -J SL_DL
#SBATCH -p hexm_l40
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAL40:4
#SBATCH --output=results/experiments/08_k562_sl_pair_state_dl/slurm_%j.out
#SBATCH --error=results/experiments/08_k562_sl_pair_state_dl/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$REPO_ROOT"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

# Args: $1 = config YAML (default phase3), $2 = producer (zero|state_dl).
CONFIG_PATH="${1:-configs/experiments/08_k562_sl_pair_state_dl/phase3_bag_supervision.yaml}"
PRODUCER="${2:-state_dl}"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run 'uv sync' before running exp08 SL DL."
  exit 1
fi

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# NOTE: Fold-level task parallelism via a FILESYSTEM WORK-QUEUE (no collective,
# no gradient all-reduce). run_cv builds the full (split_type, fold_id) job list;
# every rank walks it and atomically claims (os.mkdir, scoped by run token) each
# unfinished job, trains + embeds + scores it on its GPU, and writes
# <split>_fold<k>.result.json under <output_dir>/_fold_results/. A fold that
# raises is quarantined (.failed marker) and the run continues. Rank 0 then polls
# the filesystem until every job is terminal (or assembly_timeout_seconds) and
# writes the cvN/ + combined artifacts; output is byte-identical to a 1-process
# run. There is NO torch.distributed collective, so uneven fold runtimes can no
# longer cause a gather/NCCL timeout.
#
# RESUME: re-submitting skips folds whose .result.json already exists AND matches
# the current run fingerprint (input CSV + result-affecting config + ESM2/bags/
# gwps cache size+mtime). A crashed/incomplete fold (claim but no result) resumes
# automatically. A QUARANTINED fold (.failed marker) will NOT re-run on resubmit
# unless you delete its <split>_fold<k>.failed marker or change the config/input
# (which bumps the fingerprint) — the assembly error names the exact markers.
echo "Running exp08 SL DL with config: $CONFIG_PATH (producer: $PRODUCER)"
srun uv run --locked --no-sync --offline accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m sl_dl_model run-cv \
  --config "$CONFIG_PATH" \
  --producer "$PRODUCER"
