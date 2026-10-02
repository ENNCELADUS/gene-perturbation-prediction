#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: hpc/run.sh all CONFIG [--run-id ID]
       hpc/run.sh test CHECKPOINT
PYTHON_BIN overrides the H20 .venv-tx1/bin/python environment.
CUDA_VISIBLE_DEVICES is respected when `all` schedules GPUs.
EOF
}

if [[ $# == 0 || $1 == --help || $1 == -h ]]; then
  usage
  exit 0
fi
command=$1
shift
case "$command" in all|test) ;; *) usage >&2; exit 2 ;; esac
if [[ $# == 0 ]]; then usage >&2; exit 2; fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
python_bin=${PYTHON_BIN:-"$repo_root/.venv-tx1/bin/python"}
case "$command" in
  all) exec "$python_bin" -m src.experiments.all "$@" ;;
  test) checkpoint=$1; shift; exec "$python_bin" -m src.evaluate --checkpoint "$checkpoint" --split test "$@" ;;
esac
