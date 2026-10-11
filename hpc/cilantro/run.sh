#!/usr/bin/env bash
# Phase 1 of the Cilantro-SL reproduction: the released code end to end for one run.
#
#   hpc/cilantro/run.sh RUN_ID --dictionary gc95M|gc30M [--gpus 0,1,2,3] [--procs-per-gpu 2]
#   hpc/cilantro/run.sh RUN_ID --dictionary gc95M --gene-subset 200 --epochs 2      # smoke run only
#
# Steps (released notebook in brackets): prepare_inputs, tokenize [1], original_embeddings, perturb [2, one shard
# per process, --procs-per-gpu per GPU], generate [2], sample_negatives [3], setup_input [4], pretrain [5], then
# classify [6]: the five folds of pair-holdout (cv1) and gene-holdout (cv3), ten processes spread over the GPUs, then
# each regime's merge and conformal prediction, then report.py. A step is skipped once its done marker
# exists, so rerunning with the same RUN_ID resumes; the run's settings are fixed in run.json at first launch.
set -euo pipefail
cd "$(dirname "$0")/../.."

run_id=${1:?RUN_ID required}
shift
dictionary="" gpus=0,1,2,3 procs=2 subset=null epochs=100
while [[ $# -gt 0 ]]; do
  case $1 in
    --dictionary) dictionary=$2; shift 2 ;;
    --gpus) gpus=$2; shift 2 ;;
    --procs-per-gpu) procs=$2; shift 2 ;;
    --gene-subset) subset=$2; shift 2 ;;
    --epochs) epochs=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ $dictionary == gc95M || $dictionary == gc30M ]] || { echo "--dictionary gc95M|gc30M is required" >&2; exit 2; }

run_dir=outputs/cilantro_reproduction/$run_id
py=.venv-cilantro/bin/python
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
trap 'kill 0' INT TERM
mkdir -p "$run_dir/logs"

config="{\"dictionary\": \"$dictionary\", \"gene_subset\": $subset, \"epochs\": $epochs}"
if [[ -f $run_dir/run.json ]]; then
  [[ $(cat "$run_dir/run.json") == "$config" ]] || { echo "run.json is $(cat "$run_dir/run.json"), not $config" >&2; exit 1; }
else
  echo "$config" > "$run_dir/run.json"
fi
echo "$(date -u +%FT%TZ) $(git rev-parse HEAD) $*" >> "$run_dir/launches.txt"

IFS=, read -ra gpu_list <<< "$gpus"
first_gpu=${gpu_list[0]}

step() {  # name, done marker relative to the run directory, command...
  local name=$1 marker=$2
  shift 2
  if [[ -e $run_dir/$marker ]]; then echo "skip  $name"; return; fi
  echo "$(date -u +%FT%TZ) start $name"
  "$@" > "$run_dir/logs/$name.log" 2>&1 || { echo "failed $name: $run_dir/logs/$name.log" >&2; exit 1; }
  echo "$(date -u +%FT%TZ) done  $name"
}

step prepare_inputs inputs/provenance.json "$py" hpc/cilantro/prepare_inputs.py --run-dir "$run_dir"
step tokenize tokenized/done.json "$py" hpc/cilantro/tokenize_lines.py --run-dir "$run_dir"
step original_embeddings original_embeddings/done.json \
  env CUDA_VISIBLE_DEVICES="$first_gpu" "$py" hpc/cilantro/original_embeddings.py --run-dir "$run_dir"

if [[ -e $run_dir/perturb/done.json ]]; then
  echo "skip  perturb"
else
  shards=$(( ${#gpu_list[@]} * procs ))
  echo "$(date -u +%FT%TZ) start perturb ($shards shards)"
  pids=()
  for ((s = 0; s < shards; s++)); do
    CUDA_VISIBLE_DEVICES=${gpu_list[$((s % ${#gpu_list[@]}))]} "$py" hpc/cilantro/perturb.py --run-dir "$run_dir" \
      --shard "$s" --num-shards "$shards" > "$run_dir/logs/perturb_$s.log" 2>&1 &
    pids+=($!)
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ $failed == 0 ]] || { echo "a perturb shard failed: $run_dir/logs/perturb_*.log" >&2; exit 1; }
  "$py" -c "import json, sys; from pathlib import Path; d = Path(sys.argv[1]); n = int(sys.argv[2]); \
r = [json.loads((d / 'shards' / f'{s:02d}' / 'done.json').read_text()) for s in range(n)]; \
t = {'shards': n, 'genes': sum(x['genes'] for x in r), 'genes_written': sum(x['genes_written'] for x in r), \
'max_shard_seconds': max(x['seconds'] for x in r)}; (d / 'done.json').write_text(json.dumps(t) + '\n'); print(t)" \
    "$run_dir/perturb" "$shards"
  echo "$(date -u +%FT%TZ) done  perturb"
fi

step generate generated_df/sub_embs/done.json "$py" hpc/cilantro/generate_inputs.py --run-dir "$run_dir"
step sample_negatives pairs/done.json "$py" hpc/cilantro/sample_negatives.py --run-dir "$run_dir"
step setup_input generated_df/gene2vec/done.json "$py" hpc/cilantro/setup_input.py --run-dir "$run_dir"
step pretrain metrics/pretrain/done.json \
  env CUDA_VISIBLE_DEVICES="$first_gpu" "$py" hpc/cilantro/pretrain.py --run-dir "$run_dir" --epochs "$epochs"

wait_all() {  # pids...; fails if any failed
  local failed=0
  for pid in "$@"; do wait "$pid" || failed=1; done
  [[ $failed == 0 ]]
}

# Each fold is its own process; classify.py replays the RNG draws of the folds before it, so the folds are the
# sequential run's.
pids=() job=0
for cv in 1 3; do
  for fold in 0 1 2 3 4; do
    gpu=${gpu_list[$((job % ${#gpu_list[@]}))]}
    step "classify_cv${cv}_fold$fold" "metrics/cv$cv/fold$fold/done.json" env CUDA_VISIBLE_DEVICES="$gpu" "$py" \
      hpc/cilantro/classify.py --run-dir "$run_dir" --cv "$cv" --fold "$fold" --epochs "$epochs" &
    pids+=($!)
    job=$((job + 1))
  done
done
wait_all "${pids[@]}" || exit 1
pids=()
for cv in 1 3; do
  gpu=${gpu_list[$(( (cv / 2) % ${#gpu_list[@]} ))]}
  step "classify_cv$cv" "metrics/cv$cv/done.json" env CUDA_VISIBLE_DEVICES="$gpu" "$py" hpc/cilantro/classify.py \
    --run-dir "$run_dir" --cv "$cv" --merge --epochs "$epochs" &
  pids+=($!)
done
wait_all "${pids[@]}" || exit 1
"$py" hpc/cilantro/report.py --run-dir "$run_dir" > "$run_dir/logs/report.log" 2>&1 \
  || { echo "failed report: $run_dir/logs/report.log" >&2; exit 1; }
echo "$(date -u +%FT%TZ) complete $run_dir"
