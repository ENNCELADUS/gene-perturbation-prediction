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

CONFIG_DIR="${CONFIG_DIR:-src/scgpt/configs/0429}"
NGPUS="${NGPUS:-4}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

run_config() {
  local config_path="$1"
  echo "Running scGPT pipeline with config: $config_path"
  uv run --locked --no-sync --offline python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NGPUS" \
    --module src.main \
    --config "$config_path"
}

if [[ -n "${CONFIG_PATH:-}" ]]; then
  run_config "$CONFIG_PATH"
else
  mapfile -t CONFIG_PATHS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "*.yaml" | sort)
  if [[ "${#CONFIG_PATHS[@]}" -eq 0 ]]; then
    echo "Error: No YAML configs found in CONFIG_DIR: $CONFIG_DIR" >&2
    exit 1
  fi
  for config_path in "${CONFIG_PATHS[@]}"; do
    run_config "$config_path"
  done
fi
