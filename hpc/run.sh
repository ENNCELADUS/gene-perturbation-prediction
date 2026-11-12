#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: hpc/run.sh all CONFIG [--run-id ID] [--gpus 0,1,2,3]
       hpc/run.sh revision CONFIG [--run-id ID] [--gpus 0,1,2,3]
       hpc/run.sh test CHECKPOINT
       hpc/run.sh linear-model CONFIG [--run-id ID] [--experiments A,B]   (linear model, CPU)
       hpc/run.sh linear-model-selected CONFIG --run-id ID   (the reference linear model's data-selected genes, CPU)
       hpc/run.sh linear-model-export CONFIG --run-id ID   (the reference linear model's predictions for the correction, CPU)
       hpc/run.sh linear-model-production CONFIG --sl-split SPLIT --run-id ID   (stage one's production fit for stage two, CPU)
       hpc/run.sh linear-model-stack CONFIG --run-id ID   (single-cell statistics stacked on a linear model export, CPU)
       hpc/run.sh linear-model-noise CONFIG --run-id ID [--sections A,B]   (what limits the linear model, CPU)
       hpc/run.sh pie-reproduce --run-id ID [--device cuda] [--steps fetch,download,evaluate,compare]   (PIE's released checkpoints, GPU)
       hpc/run.sh ladder-inputs CONFIG [--steps lines,cells,mean,summaries|tx1] [--shard I/N]   (stage-one inputs; tx1 on a GPU)
       hpc/run.sh pie-precompute CONFIG --steps rescore|pilot|basis|full|assemble [--shard I/N] [--device cuda]   (frozen-PIE deltas, GPU)
       hpc/run.sh gene-sources proteins|afdb|esm|align ARGS   (the gene encoder's sources)
       hpc/run.sh stage-one CONFIG [--run-id ID] [--fold K] [--device cuda]   (one stage-one run, 1 GPU)
       hpc/run.sh stage-one-queue --gpus 0,1,2,3 [--folds main,0,1,2,3,4] CONFIG...   (stage-one runs, one per listed GPU)
       hpc/run.sh stage-one-report CONFIG [--device cuda]   (a stage-one report over runs, 1 GPU)
       hpc/run.sh stage-one-linear-model CONFIG --run-id ID   (the default linear model refitted on the stage-one frame, CPU)
       hpc/run.sh sl-paris CONFIG --run-id ID [--genes N]   (PARIS, ported to Python, CPU, .venv-pie)
       hpc/run.sh sl-slidr CONFIG --run-id ID   (SLIdR, ported to Python, CPU)
       hpc/run.sh sl-stage-two CONFIG --run-id ID   (SL stage two on stage one's GeneEffect, CPU, .venv-pie)
       hpc/run.sh sl-baselines CONFIG --run-id ID   (baselines on the multi-study SL benchmark, CPU, .venv-pie)
       hpc/run.sh sl-conditional CONFIG --run-id ID   (the SL backbone: held-out reader, conditional score, head, CPU)
       hpc/run.sh cell-states CONFIG --steps fit,project,inputs|report   (cell-state telemetry, CPU; fit runs STATES_PYTHON, scVI on a GPU)
       hpc/run.sh pie-precompute CONFIG --steps states|assemble_states --set NAME[,NAME]|ALL [--shard I/N]   (frozen PIE on state inputs, GPU)
PYTHON_BIN overrides the H20 .venv-tx1/bin/python environment; pie-reproduce,
pie-precompute and the stage-one commands except stage-one-linear-model always run in
.venv-pie.
`all` uses every visible GPU (CUDA_VISIBLE_DEVICES is respected) for every
GPU step; --gpus restricts it to the listed ids among the visible ones.
`revision` runs training, validation evaluation and baselines for one config
(no response comparison, no readout) and writes summary.md and revision.json.
`linear-model` runs the named experiments (default: all) of a linear model
config: every bridge setting, block set and penalty, scored on validation and
test, into outputs/linear_model/<id>/ (rows/, results.md).
`linear-model-selected` fits the config's reference row and tabulates its
data-selected stage into outputs/linear_model/<id>/data_selected/.
`linear-model-export` writes the config's reference row, out of fold for the
single-cell training lines, into outputs/linear_model/<id>/export/.
`linear-model-production` fits the config's reference row once on every labelled line
(stage one's production fit) and writes the GeneEffect of every single-cell
line into outputs/linear_model/<id>/production/; stage two reads it.
`linear-model-stack` stacks each arm's single-cell statistics on the export of the
config's linear_model_run, into outputs/linear_model/<id>/single_cell_stack/.
`linear-model-noise` runs the noise diagnosis's sections (default: all) against the
config's linear_model_run, into outputs/linear_model/<id>/noise/.
`pie-reproduce` fetches PIE's released checkpoints, scores them with the
vendored PIE on their test splits and compares the released metrics, into
outputs/pie_reproduction/<id>/ (results.md); PIE's assets go to data/pie/
(HF_ENDPOINT defaults to https://hf-mirror.com).
`ladder-inputs` writes the stage-one inputs of a configs/stage_one config under
its ladder_root; `pie-precompute` the frozen PIE's deltas beside them (one
process per --shard; the pilot report is read before `full`). `gene-sources`
runs one step of the gene sources: `proteins` (.venv-pie), `afdb` and `esm`
(ESM_PYTHON, default the H20's existing esm environment, with ESM_HF_HOME its
shared Hugging Face cache; hpc/esm/setup.sh builds .venv-esm), `align` (this
environment).
`stage-one` trains a configs/stage_one config (kind: train) into
outputs/stage_one/<task>/<run id, default the config stem>[__fold<K>]/ (best.pt,
predictions.parquet, V.npy, row.json); --fold K trains without frame fold K.
`stage-one-queue` runs every training config x fold lacking row.json, one
process per listed GPU, a GPU listed twice running two (other kinds of config are
skipped); `stage-one-report`
scores a report config's rows into outputs/stage_one/<task>/<config stem>/
(results.md); `stage-one-linear-model` writes the linear model reference the report and the
components reader read.
`cell-states` runs the state telemetry of a configs/stage_one/states config
(src.experiments.states.state_telemetry): `fit` writes the fit samples and plan, runs
src.experiments.states.programs in STATES_PYTHON (default .venv-states/bin/python,
built by hpc/states/setup.sh) for the cNMF programs and scVI, then fits the
mixtures; `project`, `inputs` and `report` follow. `pie-precompute` on the same
config with `--steps states --set ALL` runs the frozen PIE on the state inputs.
EOF
}

if [[ $# == 0 || $1 == --help || $1 == -h ]]; then
  usage
  exit 0
fi
command=$1
shift
case "$command" in all|revision|test|linear-model|linear-model-selected|linear-model-export|linear-model-production|linear-model-stack|linear-model-noise|pie-reproduce|ladder-inputs|pie-precompute|gene-sources|stage-one|stage-one-queue|stage-one-report|stage-one-linear-model|sl-paris|sl-slidr|sl-stage-two|sl-baselines|sl-conditional|cell-states) ;; *) usage >&2; exit 2 ;; esac
if [[ $# == 0 ]]; then usage >&2; exit 2; fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
python_bin=${PYTHON_BIN:-"$repo_root/.venv-tx1/bin/python"}
# The ESM steps reuse the H20's existing esm environment and shared Hugging Face cache
# (ESM C 600M and ESM3-open at the pinned revisions); hpc/esm/setup.sh builds .venv-esm instead.
esm_python=${ESM_PYTHON:-/2023533015/ppi-mutation-ddg/.venv/bin/python}
esm_hf_home=${ESM_HF_HOME:-/2023533015/.cache/huggingface}
pie_environment() {
  export PIE_DATA_ROOT="$repo_root/data/pie" PIE_CACHE_DIR="$repo_root/data/pie/cache"
  export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
  # TLS handshakes from the H20 to hf-mirror's Xet redirect can exceed the 10 s default.
  export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60} HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
}
case "$command" in
  all) exec "$python_bin" -m src.experiments.joint.all "$@" ;;
  revision) exec "$python_bin" -m src.experiments.joint.revision "$@" ;;
  linear-model) exec "$python_bin" -m src.experiments.linear_model.linear_model "$@" ;;
  linear-model-selected) exec "$python_bin" -m src.experiments.linear_model.data_selected_analysis "$@" ;;
  linear-model-export) exec "$python_bin" -m src.experiments.linear_model.linear_model_export "$@" ;;
  linear-model-production) exec "$python_bin" -m src.experiments.linear_model.production "$@" ;;
  linear-model-stack) exec "$python_bin" -m src.experiments.linear_model.single_cell_stack "$@" ;;
  linear-model-noise) exec "$python_bin" -m src.experiments.linear_model.noise_diagnosis "$@" ;;
  pie-reproduce)
    pie_environment
    exec "$repo_root/.venv-pie/bin/python" -m src.experiments.stage_one.pie_reproduction "$@" ;;
  ladder-inputs) exec "$python_bin" -m src.experiments.stage_one.ladder_inputs "$@" ;;
  cell-states)
    export STATES_PYTHON=${STATES_PYTHON:-"$repo_root/.venv-states/bin/python"}
    exec "$python_bin" -m src.experiments.states.state_telemetry "$@" ;;
  pie-precompute)
    pie_environment
    exec "$repo_root/.venv-pie/bin/python" -m src.experiments.stage_one.pie_precompute "$@" ;;
  stage-one)
    pie_environment
    exec "$repo_root/.venv-pie/bin/python" -m src.experiments.stage_one run "$@" ;;
  stage-one-queue)
    pie_environment
    exec "$repo_root/.venv-pie/bin/python" -m src.experiments.stage_one queue "$@" ;;
  stage-one-report)
    pie_environment
    exec "$repo_root/.venv-pie/bin/python" -m src.experiments.stage_one report "$@" ;;
  stage-one-linear-model) exec "$python_bin" -m src.experiments.stage_one.linear_model_reference "$@" ;;
  sl-paris) exec "$repo_root/.venv-pie/bin/python" -m src.experiments.sl.paris "$@" ;;
  sl-slidr) exec "$python_bin" -m src.experiments.sl.slidr "$@" ;;
  sl-stage-two) exec "$repo_root/.venv-pie/bin/python" -m src.experiments.sl.stage_two "$@" ;;
  sl-baselines) exec "$repo_root/.venv-pie/bin/python" -m src.experiments.sl.benchmark_baselines "$@" ;;
  sl-conditional) exec "$python_bin" -m src.experiments.sl.conditional "$@" ;;
  gene-sources)
    step=$1; shift
    case "$step" in
      proteins) exec "$repo_root/.venv-pie/bin/python" -m src.data.builders.embed_protein_sources --steps proteins "$@" ;;
      afdb) exec "$esm_python" -m src.data.builders.embed_protein_sources --steps afdb "$@" ;;
      esm)
        export HF_HOME="$esm_hf_home" HF_HUB_OFFLINE=1
        exec "$esm_python" -m src.data.builders.embed_protein_sources --steps esmc,esm3 "$@" ;;
      align) exec "$python_bin" -m src.data.gene_sources "$@" ;;
      *) usage >&2; exit 2 ;;
    esac ;;
  test) checkpoint=$1; shift; exec "$python_bin" -m src.experiments.joint.evaluate --checkpoint "$checkpoint" --split test "$@" ;;
esac
