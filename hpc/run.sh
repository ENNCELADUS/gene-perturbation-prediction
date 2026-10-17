#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: hpc/run.sh all CONFIG [--run-id ID] [--gpus 0,1,2,3]
       hpc/run.sh revision CONFIG [--run-id ID] [--gpus 0,1,2,3]
       hpc/run.sh test CHECKPOINT
       hpc/run.sh prior CONFIG [--run-id ID] [--experiments A,B]   (linear context prior, CPU)
       hpc/run.sh prior-selected CONFIG --run-id ID   (the reference prior's data-selected genes, CPU)
       hpc/run.sh prior-export CONFIG --run-id ID   (the reference prior's predictions for the correction, CPU)
       hpc/run.sh prior-stack CONFIG --run-id ID   (single-cell statistics stacked on a prior export, CPU)
       hpc/run.sh prior-noise CONFIG --run-id ID [--sections A,B]   (what limits the prior, CPU)
       hpc/run.sh pie-reproduce --run-id ID [--device cuda] [--steps fetch,download,evaluate,compare]   (PIE's released checkpoints, GPU)
       hpc/run.sh ladder-inputs CONFIG [--steps lines,cells,mean,summaries|tx1] [--shard I/N]   (stage-one inputs; tx1 on a GPU)
       hpc/run.sh pie-precompute CONFIG --steps rescore|pilot|basis|full|assemble [--shard I/N] [--device cuda]   (frozen-PIE deltas, GPU)
       hpc/run.sh gene-sources proteins|afdb|esm|align ARGS   (the gene encoder's sources)
       hpc/run.sh cell-states CONFIG --steps fit,project,inputs|report   (cell-state telemetry, CPU; fit runs STATES_PYTHON, scVI on a GPU)
       hpc/run.sh pie-precompute CONFIG --steps states|assemble_states --set NAME[,NAME]|ALL [--shard I/N]   (frozen PIE on state inputs, GPU)
PYTHON_BIN overrides the H20 .venv-tx1/bin/python environment; pie-reproduce
and pie-precompute always run in .venv-pie.
`all` uses every visible GPU (CUDA_VISIBLE_DEVICES is respected) for every
GPU step; --gpus restricts it to the listed ids among the visible ones.
`revision` runs training, validation evaluation and baselines for one config
(no response comparison, no readout) and writes summary.md and revision.json.
`prior` runs the named experiments (default: all) of a linear context prior
config: every bridge setting, block set and penalty, scored on validation and
test, into outputs/context_prior/<id>/ (rows/, results.md).
`prior-selected` fits the config's reference row and tabulates its
data-selected stage into outputs/context_prior/<id>/data_selected/.
`prior-export` writes the config's reference row, out of fold for the
single-cell training lines, into outputs/context_prior/<id>/export/.
`prior-stack` stacks each arm's single-cell statistics on the export of the
config's prior_run, into outputs/context_prior/<id>/single_cell_stack/.
`prior-noise` runs the noise diagnosis's sections (default: all) against the
config's prior_run, into outputs/context_prior/<id>/noise/.
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
`cell-states` runs the state telemetry of a configs/stage_one/states config
(src.experiments.state_telemetry): `fit` writes the fit samples and plan, runs
src.data.prepare.state_programs in STATES_PYTHON (default .venv-states/bin/python,
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
case "$command" in all|revision|test|prior|prior-selected|prior-export|prior-stack|prior-noise|pie-reproduce|ladder-inputs|pie-precompute|gene-sources|cell-states) ;; *) usage >&2; exit 2 ;; esac
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
  all) exec "$python_bin" -m src.experiments.all "$@" ;;
  revision) exec "$python_bin" -m src.experiments.revision "$@" ;;
  prior) exec "$python_bin" -m src.experiments.context_prior "$@" ;;
  prior-selected) exec "$python_bin" -m src.experiments.data_selected_analysis "$@" ;;
  prior-export) exec "$python_bin" -m src.experiments.prior_export "$@" ;;
  prior-stack) exec "$python_bin" -m src.experiments.single_cell_stack "$@" ;;
  prior-noise) exec "$python_bin" -m src.experiments.noise_diagnosis "$@" ;;
  pie-reproduce)
    pie_environment
    exec "$repo_root/.venv-pie/bin/python" -m src.experiments.pie_reproduction "$@" ;;
  ladder-inputs) exec "$python_bin" -m src.experiments.ladder_inputs "$@" ;;
  cell-states)
    export STATES_PYTHON=${STATES_PYTHON:-"$repo_root/.venv-states/bin/python"}
    exec "$python_bin" -m src.experiments.state_telemetry "$@" ;;
  pie-precompute)
    pie_environment
    exec "$repo_root/.venv-pie/bin/python" -m src.experiments.pie_precompute "$@" ;;
  gene-sources)
    step=$1; shift
    case "$step" in
      proteins) exec "$repo_root/.venv-pie/bin/python" -m src.data.prepare.embed_protein_sources --steps proteins "$@" ;;
      afdb) exec "$esm_python" -m src.data.prepare.embed_protein_sources --steps afdb "$@" ;;
      esm)
        export HF_HOME="$esm_hf_home" HF_HUB_OFFLINE=1
        exec "$esm_python" -m src.data.prepare.embed_protein_sources --steps esmc,esm3 "$@" ;;
      align) exec "$python_bin" -m src.data.gene_sources "$@" ;;
      *) usage >&2; exit 2 ;;
    esac ;;
  test) checkpoint=$1; shift; exec "$python_bin" -m src.evaluate --checkpoint "$checkpoint" --split test "$@" ;;
esac
