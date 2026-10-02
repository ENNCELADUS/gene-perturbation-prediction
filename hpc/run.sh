#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: hpc/run.sh prepare CONFIG
       hpc/run.sh train CONFIG (--run-id NAME | --resume CHECKPOINT)
       hpc/run.sh test CHECKPOINT
PYTHON_BIN overrides the H20 .venv-tx1/bin/python environment.
CUDA_VISIBLE_DEVICES is respected when detecting training workers.
EOF
}

if [[ $# == 0 || $1 == --help || $1 == -h ]]; then
  usage
  exit 0
fi
command=$1
shift
case "$command" in prepare|train|test) ;; *) usage >&2; exit 2 ;; esac
if [[ $# == 0 ]]; then usage >&2; exit 2; fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
python_bin=${PYTHON_BIN:-"$repo_root/.venv-tx1/bin/python"}
case "$command" in
  prepare) exec "$python_bin" -m src.experiments.prepare "$@" ;;
  test) checkpoint=$1; shift; exec "$python_bin" -m src.evaluate --checkpoint "$checkpoint" --split test "$@" ;;
  train)
    if [[ $1 == --help || $1 == -h ]]; then exec "$python_bin" -m src.train "$@"; fi
    workers=$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')
    if ! [[ $workers =~ ^[0-9]+$ ]] || (( workers < 1 )); then
      echo 'No visible GPUs in the selected Python environment.' >&2
      exit 1
    fi
    launch=( -m accelerate.commands.launch --num_processes "$workers" --num_machines 1 --mixed_precision bf16 )
    if (( workers > 1 )); then launch+=( --multi_gpu ); fi
    exec "$python_bin" "${launch[@]}" --module src.train "$@"
    ;;
esac
