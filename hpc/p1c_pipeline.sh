#!/usr/bin/env bash
# P1-C interface isolation: one full round on a two-GPU H20 container.
#
# Waves: fold preparation (CPU) -> Tier 0 + native evaluation + Tier 4 heads ->
# one wave per variant label over the four folds -> comparison -> learning-rate
# arms and (conditionally) V3 -> final comparison. Every GPU job runs through a
# small queue that keeps at most one job per visible GPU. Design:
# docs/specs/2026-09-08-p1c-interface-isolation-design.md
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

DRY_RUN=${PIPELINE_DRY_RUN:-0}
GPUS=${GPUS:-"0 1"}
SKIP_TIER4=${PIPELINE_SKIP_TIER4:-0}
POLL_SECONDS=${PIPELINE_POLL_SECONDS:-30}
python_bin=${PYTHON_BIN:-.venv-tx1/bin/python}

FOLDS="jurkat k562 hepg2 hct116"
WAVE_LABELS="V0 V1 V2-null V2"

: "${RUN:?RUN (the run root directory) is required}"
: "${P0_CHECKPOINT:?P0_CHECKPOINT (the P0 joint checkpoint) is required}"
: "${P1B_PREPARED:?P1B_PREPARED (the P1-B prepared directory) is required}"
: "${P1B_RUNS:?P1B_RUNS (the P1-B runs directory) is required}"
: "${P1A_FEATURES:?P1A_FEATURES (the P1-A feature cache) is required}"
: "${P1A_REFERENCE_P0:?P1A_REFERENCE_P0 (P0 validation predictions) is required}"
: "${P1A_REFERENCE_PCA:?P1A_REFERENCE_PCA (PCA-ridge validation predictions) is required}"

if [[ "$DRY_RUN" != 1 ]]; then
  for required in \
    "$P0_CHECKPOINT" \
    "$P1B_PREPARED" \
    "$P1B_PREPARED/manifest.json" \
    "$P1B_RUNS" \
    "$P1B_RUNS/evaluation" \
    "$P1A_FEATURES" \
    "$P1A_REFERENCE_P0" \
    "$P1A_REFERENCE_PCA"; do
    if [[ ! -e "$required" ]]; then
      printf 'missing required input: %s\n' "$required" >&2
      exit 1
    fi
  done
fi

if [[ -e "$RUN/phase.txt" ]]; then
  printf 'run already started: %s exists\n' "$RUN/phase.txt" >&2
  exit 1
fi
mkdir -p "$RUN"

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$RUN/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# Every background job started by spawn, so a signal can stop the whole round.
# A reaped job's entry becomes 0 so a recycled pid is never signalled.
tracked_pids=()
tracked_names=()
tracked_kids=()
interrupted=0

untrack_pid() {
  local pid=$1 i
  for (( i = 0; i < ${#tracked_pids[@]}; i++ )); do
    if [[ "${tracked_pids[$i]}" == "$pid" ]]; then
      tracked_pids[$i]=0
      tracked_names[$i]=""
      tracked_kids[$i]=""
    fi
  done
}

on_exit() {
  local rc=$?
  if (( interrupted != 0 )); then
    return
  fi
  printf '%s\n' "$rc" > "$RUN/exit_code"
  if [[ "$rc" != 0 ]]; then
    printf 'failed\n' > "$RUN/phase.txt"
  fi
}

# Signal a job shell and its own worker. hpc/run.sh execs Python, so the process
# holding the GPU is the job subshell's direct child; it must be recorded before
# the subshell dies, or it is reparented to PID 1 and pkill -P can no longer
# find it. The shell is signalled first so it cannot race ahead and record a
# normal exit for an interrupted job; the recorded children (plus a best-effort
# pkill for anything started since) follow immediately.
signal_job() {
  local signal=$1 pid=$2 index=$3 kid
  if [[ -z "${tracked_kids[$index]}" ]]; then
    tracked_kids[$index]=$({ pgrep -P "$pid" 2>/dev/null || true; } | tr '\n' ' ')
  fi
  kill "-$signal" "$pid" 2>/dev/null || true
  for kid in ${tracked_kids[$index]}; do
    kill "-$signal" "$kid" 2>/dev/null || true
  done
  pkill "-$signal" -P "$pid" 2>/dev/null || true
}

on_signal() {
  interrupted=1
  trap - INT TERM
  local i
  for (( i = 0; i < ${#tracked_pids[@]}; i++ )); do
    if [[ "${tracked_pids[$i]}" != 0 ]]; then
      signal_job TERM "${tracked_pids[$i]}" "$i"
    fi
  done
  sleep 2
  for (( i = 0; i < ${#tracked_pids[@]}; i++ )); do
    if [[ "${tracked_pids[$i]}" == 0 ]]; then
      continue
    fi
    signal_job KILL "${tracked_pids[$i]}" "$i"
    if [[ ! -s "$RUN/${tracked_names[$i]}.exit" ]]; then
      printf '143\n' > "$RUN/${tracked_names[$i]}.exit"
    fi
  done
  printf 'interrupted\n' > "$RUN/phase.txt"
  printf '143\n' > "$RUN/exit_code"
  exit 143
}

trap on_exit EXIT
trap on_signal INT TERM

phase() {
  printf '%s\n' "$1" > "$RUN/phase.txt"
}

run_sh() {
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "hpc/run.sh $*"
    return 0
  fi
  hpc/run.sh "$@"
}

# One unit of queued work. Every step returns its own failure so the job stops
# at the first failing command regardless of the caller's errexit state.
job_body() {
  local kind=$1
  shift
  case "$kind" in
    tier0)
      run_sh p1c tier0 \
        --p1b-runs "$P1B_RUNS" \
        --p1b-prepared "$P1B_PREPARED" \
        --out-dir "$RUN/tier0" || return $?
      ;;
    native)
      run_sh p1c evaluate-native \
        --prepared "$RUN/prepared/jurkat" \
        --runs "$RUN/runs/N-native/jurkat" \
        --batch-indices 0 1 2 3 4 || return $?
      ;;
    tier4)
      local seed=$1
      run_sh p1a train \
        --cache "$P1A_FEATURES" \
        --out-dir "$RUN/heads/seed$seed" \
        --head-seed "$seed" \
        --arms A0 A1 A2 A3 || return $?
      run_sh p1a compare \
        --cache "$P1A_FEATURES" \
        --runs "$RUN/heads/seed$seed" \
        --out-dir "$RUN/heads/seed$seed/comparison" \
        --reference-val "$P1A_REFERENCE_P0" \
        --reference-val "$P1A_REFERENCE_PCA" || return $?
      ;;
    variant)
      local label=$1 variant=$2 fold=$3 lr=$4
      local prepared="$RUN/prepared/$fold"
      local runs="$RUN/runs/$label/$fold"
      run_sh p1c train \
        --prepared "$prepared" --runs "$runs" \
        --variant "$variant" --lr "$lr" || return $?
      run_sh p1c evaluate --prepared "$prepared" --runs "$runs" || return $?
      run_sh p1c evaluate --prepared "$prepared" --runs "$runs" --external || return $?
      ;;
    *)
      printf 'unknown job kind: %s\n' "$kind" >&2
      return 2
      ;;
  esac
}

# Start one job in the background on "$gpu"; sets spawn_pid.
spawn_pid=0
spawn() {
  local name=$1 gpu=$2
  shift 2
  (
    trap - EXIT INT TERM
    export CUDA_VISIBLE_DEVICES="$gpu"
    rc=0
    job_body "$@" || rc=$?
    # Dry-run jobs stay alive long enough to exercise the signal path, with a
    # child of their own standing in for the Python worker hpc/run.sh execs.
    if [[ "$DRY_RUN" == 1 && -n "${PIPELINE_DRY_RUN_SLEEP:-}" ]]; then
      sleep "$PIPELINE_DRY_RUN_SLEEP" &
      child=$!
      printf '%s\n' "$child" > "$RUN/$name.child.pid"
      wait "$child" || true
    fi
    printf '%s\n' "$rc" > "$RUN/$name.exit"
    exit "$rc"
  ) > "$RUN/$name.log" 2>&1 &
  spawn_pid=$!
  tracked_pids+=("$spawn_pid")
  tracked_names+=("$name")
  tracked_kids+=("")
  printf '%s\n' "$spawn_pid" > "$RUN/$name.pid"
}

# Reap a finished job into job_rc: the code the job recorded, else wait's status.
# A job killed before it could record its own code still gets an exit file, so
# the run root alone always shows every job's outcome.
job_rc=0
reap_job() {
  local name=$1 pid=$2 recorded=""
  job_rc=0
  wait "$pid" 2>/dev/null || job_rc=$?
  if [[ ! "$job_rc" =~ ^[0-9]+$ ]]; then
    job_rc=1
  fi
  if [[ -r "$RUN/$name.exit" ]]; then
    recorded=$(<"$RUN/$name.exit")
  fi
  if [[ "$recorded" =~ ^[0-9]+$ ]]; then
    job_rc=$recorded
  else
    printf '%s\n' "$job_rc" > "$RUN/$name.exit"
  fi
  untrack_pid "$pid"
}

queue_names=()
queue_specs=()

enqueue() {
  local name=$1
  shift
  local IFS='|'
  queue_names+=("$name")
  queue_specs+=("$*")
}

spec_parts=()
split_spec() {
  local IFS='|'
  read -r -a spec_parts <<< "$1"
}

# Run every queued job, at most one per GPU at a time, starting the next queued
# job on a GPU as soon as its previous job exits. Drains the queue even after a
# failure and then returns 1.
run_queue() {
  local -a gpu_list
  local gpu
  gpu_list=()
  for gpu in $GPUS; do
    gpu_list+=("$gpu")
  done
  if [[ ${#gpu_list[@]} -eq 0 ]]; then
    printf 'GPUS must name at least one device\n' >&2
    return 2
  fi
  local total=${#queue_names[@]}
  if (( total == 0 )); then
    return 0
  fi

  local -a slot_pid slot_name
  local slots=${#gpu_list[@]}
  local i next=0 failed=0 running rc
  for (( i = 0; i < slots; i++ )); do
    slot_pid[$i]=0
    slot_name[$i]=""
  done

  while :; do
    for (( i = 0; i < slots; i++ )); do
      if (( slot_pid[i] == 0 )); then
        continue
      fi
      if kill -0 "${slot_pid[$i]}" 2>/dev/null && [[ ! -f "$RUN/${slot_name[$i]}.exit" ]]; then
        continue
      fi
      reap_job "${slot_name[$i]}" "${slot_pid[$i]}"
      rc=$job_rc
      if [[ "$rc" != 0 ]]; then
        printf 'job %s failed with exit code %s\n' "${slot_name[$i]}" "$rc" >&2
        failed=1
      fi
      slot_pid[$i]=0
      slot_name[$i]=""
    done

    for (( i = 0; i < slots; i++ )); do
      if (( next >= total )); then
        break
      fi
      if (( slot_pid[i] == 0 )); then
        split_spec "${queue_specs[$next]}"
        spawn "${queue_names[$next]}" "${gpu_list[$i]}" ${spec_parts[@]+"${spec_parts[@]}"}
        slot_pid[$i]=$spawn_pid
        slot_name[$i]=${queue_names[$next]}
        next=$(( next + 1 ))
      fi
    done

    running=0
    for (( i = 0; i < slots; i++ )); do
      if (( slot_pid[i] != 0 )); then
        running=1
      fi
    done
    if (( next >= total && running == 0 )); then
      break
    fi
    sleep "$POLL_SECONDS"
  done

  queue_names=()
  queue_specs=()
  if (( failed != 0 )); then
    return 1
  fi
}

phase preparing
for fold in $FOLDS; do
  (
    export CUDA_VISIBLE_DEVICES=""
    run_sh p1c prepare \
      --checkpoint "$P0_CHECKPOINT" \
      --out-dir "$RUN/prepared/$fold" \
      --fold "$fold" \
      --reference-manifest "$P1B_PREPARED/manifest.json"
  ) > "$RUN/prepare-$fold.log" 2>&1
done

phase tier0-native-heads
spawn tier0 "" tier0
tier0_pid=$spawn_pid
enqueue N-native-jurkat native
if [[ "$SKIP_TIER4" != 1 ]]; then
  for seed in 1 2; do
    enqueue "tier4-seed$seed" tier4 "$seed"
  done
fi
wave_failed=0
run_queue || wave_failed=1
reap_job tier0 "$tier0_pid"
tier0_rc=$job_rc
if [[ "$tier0_rc" != 0 ]]; then
  printf 'job tier0 failed with exit code %s\n' "$tier0_rc" >&2
  wave_failed=1
fi
if (( wave_failed != 0 )); then
  exit 1
fi

for label in $WAVE_LABELS; do
  phase "wave-$label"
  for fold in $FOLDS; do
    enqueue "$label-$fold" variant "$label" "$label" "$fold" 1e-4
  done
  run_queue
done

phase compare-1
run_sh p1c compare --root "$RUN" --out-dir "$RUN/comparison" > "$RUN/compare-1.log" 2>&1
if [[ "$DRY_RUN" == 1 && ! -f "$RUN/comparison/kept.json" && -n "${PIPELINE_DRY_RUN_KEPT:-}" ]]; then
  mkdir -p "$RUN/comparison"
  cp "$PIPELINE_DRY_RUN_KEPT" "$RUN/comparison/kept.json"
fi

phase wave-lr-and-v3
# Read the predicate before queueing anything: an unreadable kept.json must not
# cost the learning-rate arms, but it still fails the round at the end.
v3_read_failed=0
v3_eligible=0
kept_read=$("$python_bin" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["V3_eligible"]))' \
  "$RUN/comparison/kept.json" 2> "$RUN/v3_read.log") || v3_read_failed=1
if (( v3_read_failed == 0 )) && [[ "$kept_read" == 0 || "$kept_read" == 1 ]]; then
  v3_eligible=$kept_read
else
  v3_read_failed=1
fi

enqueue V0-lr1e-6-jurkat variant V0-lr1e-6 V0 jurkat 1e-6
enqueue V0-lr1e-5-jurkat variant V0-lr1e-5 V0 jurkat 1e-5
if (( v3_read_failed != 0 )); then
  printf 'V3 skipped: could not read V3_eligible from %s (see %s)\n' \
    "$RUN/comparison/kept.json" "$RUN/v3_read.log" > "$RUN/v3_skipped.txt"
elif [[ "$v3_eligible" == 1 ]]; then
  for fold in $FOLDS; do
    enqueue "V3-$fold" variant V3 V3 "$fold" 1e-4
  done
else
  printf 'V3 skipped: V3_eligible is false in %s\n' "$RUN/comparison/kept.json" \
    > "$RUN/v3_skipped.txt"
fi
run_queue

phase compare-final
run_sh p1c compare --root "$RUN" --out-dir "$RUN/comparison" > "$RUN/compare-final.log" 2>&1

if (( v3_read_failed != 0 )); then
  printf 'V3 eligibility could not be read from %s; V3 was not run\n' \
    "$RUN/comparison/kept.json" >&2
  exit 1
fi

phase completed
