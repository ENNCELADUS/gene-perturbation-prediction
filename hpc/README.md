# Running on the H20

Run from the repository root on an H20 container (the `hpc-execution` skill holds ports, Git-only sync and launch
rules). `hpc/run.sh` runs `.venv-tx1/bin/python` unless `PYTHON_BIN` names another; the PIE and stage-one commands run
`.venv-pie` (`hpc/pie/setup.sh`) and the cell-state fits `.venv-states` (`hpc/states/setup.sh`). `hpc/run.sh --help`
prints every command's arguments.

## Commands

| `hpc/run.sh` | Runs | Where | Section |
| --- | --- | --- | --- |
| `all CONFIG` | the joint pipeline to `summary.md` (`joint.all`) | every chosen GPU | [`all`](#the-all-command) |
| `revision CONFIG` | one joint variant: training, then validation and test (`joint.revision`) | chosen GPUs | [`revision`](#the-revision-command) |
| `test CHECKPOINT` | one checkpoint on test (`joint.evaluate`) | one GPU | [`revision`](#the-revision-command) |
| `linear-model`, `linear-model-selected`, `linear-model-export`, `linear-model-stack`, `linear-model-noise` | the linear model and its analyses (`linear_model.*`) | CPU | [`linear-model`](#the-linear-model-command) |
| `pie-reproduce` | PIE's released checkpoints re-scored (`stage_one.pie_reproduction`) | one GPU, `.venv-pie` | [PIE harness](#the-pie-harness) |
| `ladder-inputs`, `gene-sources`, `pie-precompute` | the stage-one inputs and the frozen PIE's deltas | CPU; Tx1, ESM and PIE on GPUs | [Stage-one inputs](#the-stage-one-inputs) |
| `stage-one`, `stage-one-queue`, `stage-one-report`, `stage-one-linear-model` | stage-one runs, their queue and report, the linear model reference | one GPU per run (`.venv-pie`); the linear model on CPU | [Stage-one model](#the-stage-one-model) |
| `cell-states`, `pie-precompute --steps states` | the cell-state telemetry and the frozen PIE on its states | CPU; scVI and PIE on GPUs | [Cell states](#the-cell-states) |
| `sl-paris`, `sl-stage-two` | PARIS (`sl.paris`) and SL stage two on stage one's GeneEffect (`sl.stage_two`) | CPU, `.venv-pie` | [SL stage two](#sl-stage-two) |
| `sl-baselines` | baselines on the multi-study SL benchmark (`sl.benchmark_baselines`) | CPU, `.venv-pie` | [SL benchmark baselines](#sl-benchmark-baselines) |
| `sl-conditional` | the modular SL backbone: round zero, the buffer-aware reader (`sl.conditional`) | CPU | [SL benchmark baselines](#sl-benchmark-baselines) |
| `sl-slidr` | SLIdR, ported to Python, scores for `sl-baselines` (`sl.slidr`) | CPU | [SL benchmark baselines](#sl-benchmark-baselines) |

GPU use, measured on 30838 in October 2026 (one H20 per job unless noted; SM is `nvidia-smi dmon`'s busy share):

| Command, job | Before | After | GPU while computing |
| --- | --- | --- | --- |
| `stage-one`, `film_linear_model_dE` for 3 epochs | 258 s; a training epoch 14.9 s, its validation ~14 s of pandas | 114 s; a training epoch 2.2 s (the first 22 s, compiling), validation 2.3 s | a step 2.0 ms, 1.6 ms of it on the GPU (was ~12 ms of launches against ~0.4 ms) |
| `stage-one`, a PIE-trunk arm (`direct_readout`) | 109 s an epoch | unchanged, eager (106 s compiled) | GPU-bound |
| `stage-one`, a full-panel arm (`film_dE_full`) | frame-mean pass 1058–1349 s; 134–163 s an epoch | pass 46–53 s; 151 s an epoch, streaming from the page cache | |
| `stage-one-queue`, four 3-epoch runs on one GPU | 417 s, one at a time | 227 s, two at a time | |
| `ladder-inputs --steps tx1`, row T1 | | unchanged | SM 96% mean, ≥85% in 1163 of 1211 s |
| `pie-precompute --steps states`, cycle shard 0/16 (94 contexts) | | unchanged: 1.79 s a context after 136 s of setup a process | SM 97–100% |
| `cell-states --steps fit` | scVI minutes, cNMF hours of CPU holding no GPU | unchanged | |

Validation scoring in one pass and the threaded frame-mean pass leave every number as it was; the CUDA-graph step,
fused AdamW and the masked dependency term move them at rounding level, so the stage-one rows trained before
2026-10-23 were rerun (`results/stage_one_architecture/`, `results/stage_one_linear_model_input/`).

Direct entry points, `python -m src.<module>` from the repository root (`.venv-tx1` unless noted):

| Module | Does | Documented in |
| --- | --- | --- |
| `experiments.prepare` | preparation alone, or growing the prepared root by new lines | [Extra single-cell lines](#extra-single-cell-lines) |
| `experiments.joint.evaluate`, `experiments.joint.compare_runs` | a checkpoint on validation or train; the paired line bootstrap of two runs | [`revision`](#the-revision-command) |
| `experiments.joint.train`, `.response_comparison`, `.baselines`, `.readout` | the steps `all` launches; alone only to debug | [`all`](#the-all-command) |
| `experiments.joint.tx1_gmm_ridge` | the Tx1 GMM-ridge baseline | [Tx1 GMM-ridge baseline](#tx1-gmm-ridge-baseline) |
| `experiments.states.programs` | the cNMF, starCAT and scVI fits (`.venv-states`), launched by `cell-states --steps fit` | [Cell states](#the-cell-states) |
| `data.builders.build_extra_single_cell_sources`, `recount_extra_single_cell_sources`, `build_extra_single_cell_lines`, `plan_second_sources`, `build_cell_line_geneeffect_split` | the single-cell extras and the final split | [Extra single-cell lines](#extra-single-cell-lines) |
| `data.builders.cellosaurus_context`, `embed_protein_sources`; `data.gene_sources` | the stage-one context text and gene sources | [Stage-one inputs](#the-stage-one-inputs) |
| `data.builders.build_extra_bulk_lines`, `build_cell_line_geneeffect_226_split`, `build_sl_screen_benchmark`, `build_sl_study_benchmark`, `build_sl_split`, `build_pc9_hela_basal` | tracked membership files and benchmarks | their cards: [bulk lines](../docs/data/extra-bulk-lines-26q1.md), [226](../docs/data/cell-line-geneeffect-226.md), [SL benchmark](../docs/data/sl-benchmark.html) |
| `data.builders.build_context_reference`, `build_state_programs` | the linear model's reference tables and pinned cell-state programs (on the Mac) | [context-generalization design](../docs/specs/2026-10-04-context-generalization-design.md), [noise-diagnosis design](../docs/specs/2026-10-12-noise-diagnosis-design.md) |
| `data.builders.prepare_kinker_umi_h5ad`, `prepare_cell_line_atlas_raw_umi`, `materialize_exp13_original47_raw_umi`, `bootstrap_exp13_uniprot_cache`, `build_exp13_esm2_universe`, `precompute_esm2_embeddings` | the 226 benchmark's raw-UMI basal h5ad and ESM2 table | [226 card](../docs/data/cell-line-geneeffect-226.md), [Tx1 input record](../results/exp13_stage0/README.md) |
| `experiments.historical.*` | retired probes and builders, kept runnable | [historical README](../src/experiments/historical/README.md) |

## The `all` command

```bash
hpc/run.sh all configs/geneeffect_joint.yaml [--run-id <id>] [--gpus 0,1,2,3]
```

`all` runs the whole pipeline in one command and prints its run id first (default
`all_<UTC timestamp>`). Steps, in order, each skipped when its output already exists:

1. **Preparation** into `prepared_root` (`data/geneeffect_joint/v2`): reuse the Tx1 cache
   (only missing lines are encoded), one pass over the basal sources computing library
   sizes, the target total `T`, log-space HVG bags and `q_sc`, the log-space response
   cache, and the manifest with `expression_space`. Once the manifest exists, only lines
   without `lines/<ModelID>.npz` (new single-cell extras) are prepared, in the recorded space.
2. **Untrained response-model comparison arms** (no change, global mean effect, released
   STATE checkpoint) on the first chosen GPU, which also give the **STATE sanity line**: the
   released checkpoint scored on each anchor against no-change. Printed and written; not a
   gate.
3. **Joint training** through `accelerate launch` with one process per chosen GPU.
   Rerunning continues from `train/last.pt`. `best.pt` and early stopping follow the
   validation selective-gene Spearman (higher wins), not the GeneEffect loss.
4. **Trained response-model comparison arms** (STATE as in the joint model, MLP on HVG, MLP
   on Tx1; one job per arm and held-out anchor, twelve in all), one job per chosen GPU at a
   time, the next starting as soon as a GPU frees, then the comparison table and verdicts
   ([protocol §9](../docs/03-geneeffect-protocol.md#9-response-model-comparison-and-the-all-run)).
5. **Validation evaluation** of `train/best.pt`, the baseline ladder (gene mean, K562 copy
   linear model, nearest line, context-PCA ridge on Tx1 and on log HVG) and the readout head with
   the explicit gene-specific context slope on the new backbone's cached features.
6. **`summary.md`**: target total `T`, the sanity line, the comparison table and verdicts,
   and the validation table (selective-gene Spearman and AUPR lift, Huber, correlations) for
   the joint model, readout head and every baseline.

The run directory is `outputs/geneeffect_joint/<run_id>/{comparison/, train/, evaluation/val/,
baselines/val/, readout/, logs/, summary.md}`; `summary.md` is rewritten on every rerun.
An interrupted run is resumed by rerunning the same command with the same `--run-id`; a
fresh run needs a new run id. A run directory is bound to the config it started with
(`run_config.json`), and training resumes from `train/last.pt` only when the config saved
there equals the current one, so a batch-size or any other config change needs a new run id.

Each GPU step uses the whole set of chosen GPUs, so the GeneEffect result arrives before
the comparison tail. `--gpus` takes ids as `CUDA_VISIBLE_DEVICES` lists them (or 0..N−1
without it) and is not bound to the run: a resumed run may use other GPUs for the
comparison jobs, which are independent and seeded per job, but unfinished training resumes
only on as many GPUs as it started on (`train/run.json`); another count is refused before
launch. Without CUDA every step runs in-process on the CPU, in the same order (the test
path). Subprocesses log to `logs/comparison_untrained.log`, `logs/train.log` and
`logs/comparison_<arm>__<anchor>.log` in the run directory. A failed subprocess stops the
run with its step, exit code and log path; comparison jobs already running finish first and
no new one starts. SIGINT or SIGTERM to `all` terminates its subprocess groups before it
exits, so no GPU worker is orphaned.

## The `revision` command

`hpc/run.sh revision CONFIG [--run-id <id>] [--gpus 0,1,2,3]` runs one variant of the
[GeneEffect revision](../docs/specs/2026-10-03-geneeffect-revision-design.md) (objective and
STATE setting chosen by the config, one per variant in `configs/revision/`) and prints its run
id first (default `revision_<UTC timestamp>`). Steps, in order, each skipped when its output
already exists:

1. **Preparation check**: `prepare_inputs` returns at once on an existing `prepared_root`; the
   prepared root of `all` is reused and nothing is re-prepared.
2. **Joint training** through `accelerate launch` with one process per chosen GPU, as in
   `all`; rerunning continues from `train/last.pt`. Selection and early stopping follow the
   validation selective-gene Spearman.
3. **Validation, then test evaluation** of `train/best.pt` and the **baselines** on the same
   split (gene mean, K562 copy baseline, nearest line, context-PCA ridge on Tx1 and on log HVG),
   on the first chosen GPU. `best.pt` is chosen on validation alone; test scores it once.
4. **`revision.json` and `summary.md`**: per split, one table (selective Spearman, selective
   AUPR lift, residual Pearson over variable genes, Huber, SD ratio) for the joint model and
   every baseline and the paired line bootstrap of selective Spearman (joint model minus the Tx1
   context-PCA ridge; 1,000 resamples, seed 0); then the best epoch with its training-diagnostic
   and validation selective Spearman. Both files are rewritten on every rerun.

With `paths.linear_model` set (the single-cell correction, `configs/correction/`,
[protocol §12](../docs/03-geneeffect-protocol.md#12-single-cell-correction)) the run is a
correction: the model predicts the linear model export's offset plus the head, validation also runs
before the first update (epoch −1, the linear model alone, eligible for `best.pt`), the baselines add
the linear model alone and the linear model plus the Tx1 context-PCA ridge, `summary.md` bootstraps the stack
against each of them as well, and every summary carries a per-lineage table read from the split
table next to the split file. `uv run python -m src.experiments.joint.compare_runs RUN_A RUN_B --split val`
prints the paired line bootstrap of selective Spearman between two finished runs (A minus B).

The run directory is `<output_root>/<run_id>/{train/, evaluation/{val,test}/,
baselines/{val,test}/, logs/, revision.json, summary.md}`; `output_root` is `outputs/geneeffect_revision` in
`configs/revision/*.yaml` and `outputs/geneeffect_joint` in the base config. There is no
response-model comparison and no readout head. Training uses every chosen GPU (every visible
GPU unless `--gpus` names some); unfinished training resumes only on as many GPUs as it
started on, and `--gpus` is not bound to the run. Resume, config binding
(`run_config.json`; a changed config needs a new run id), the training subprocess log
(`logs/train.log`) and SIGINT or SIGTERM handling are those of `all`. One config is one experiment at one seed:
training with validation selection, then test.

`all` never evaluates the test split; `revision` scores its own `best.pt` on test.
`hpc/run.sh test CHECKPOINT` scores any other checkpoint on test and restores the checkpoint's fitted preprocessing, weights and ESM2 vectors without
optimizer steps. Standalone evaluation (`test`, or `src.experiments.joint.evaluate --split val|train`)
writes `evaluation/<checkpoint-name>/<split>/` beside the checkpoint, holding
`predictions.parquet`, `metrics.json`, `per_line.csv` and `per_gene.csv`; per-gene details cover
the union of the train-derived variable and selective genes with boolean `variable` and
`selective` columns, carry residual target and prediction SD, SD ratio, RMSE and MAE on the
same finite rows, and add the dependent-line `aupr_lift`; the `residual_*` metrics use the
variable rows only, `selective_spearman` and `selective_aupr_lift` the selective rows, and
undefined quantities keep explicit counts and null scalar values. An export failure is retried by rerunning the
same evaluation command. Nothing here is SL interaction evidence; held-out lines retain the
documented Tx1 pretraining exposure boundary.

## The Cilantro-SL harness

`hpc/cilantro/` runs the released Cilantro-SL code (phase 1 of
`docs/specs/2026-10-08-cilantro-sl-reproduction-design.md`) in its own environment,
`.venv-cilantro` (Python 3.9.20), not `.venv-tx1`; its scripts are plain files run by that
interpreter, not `src.*` modules. The released code is not tracked here. The H20 cannot reach
GitHub, so from the laptop, with clones at the pinned commits:

```bash
ssh -p 30838 root@10.15.171.204 'cd /2023533015/VCC_Project && git init -q third_party/cilantro_sl && git init -q third_party/geneformer_cilantro'
git -C <clone of kaileyhh/Cilantro-SL> push ssh://root@10.15.171.204:30838/2023533015/VCC_Project/third_party/cilantro_sl c674b55e6fb126b3630502fbe8eb8ab49c5c9bc8:refs/heads/pinned
git -C <clone of kaileyhh/geneformer> push ssh://root@10.15.171.204:30838/2023533015/VCC_Project/third_party/geneformer_cilantro 0e2bf2acbfdbc070a595d4ee66f5cda654dc624a:refs/heads/pinned
```

then on the H20 `git -C third_party/<name> checkout pinned` and `hpc/cilantro/setup.sh`, which checks
the commits, applies `official.patch` and `geneformer.patch`, builds the environment and fetches Geneformer from
ModelScope's copy, checked against the Hugging Face hashes. The four small official data files sit in `data/cilantro/official/`.

`hpc/cilantro/run.sh RUN_ID --dictionary gc95M|gc30M [--gpus 0,1,2,3] [--procs-per-gpu 2]` runs
every released step for one run under `outputs/cilantro_reproduction/RUN_ID/`. The deletion step
runs one shard per process, `--procs-per-gpu` per GPU. The classifier runs each of the five folds of
both split regimes as its own process (`classify.py --fold K`, ten processes spread over the GPUs),
each replaying the earlier folds' RNG draws so it trains exactly the sequential run's fold; then
`classify.py --merge` per regime writes the metrics pickle and runs conformal prediction on fold 0.
A step is skipped once its `done.json`
exists, so rerunning the same RUN_ID resumes; `run.json` fixes the run's settings at first launch.
Logs are in `logs/<step>.log`. `--gene-subset N --epochs E` makes a smoke run (the N genes with
the most SynLethDB pairs, E epochs), only to check the chain.

The deletion, stage one and classifier steps run the harness's fast paths by default
(`hpc/cilantro/fast_perturb.py`, `fast_classifier.py`, `fast_core.py`): the released computation
with its per-gene process pools, repeated model loads, discarded forward passes and per-item data
loaders removed, byte-identical outputs (design §4.4). Each step script takes `--released` to run the
released code instead. `hpc/cilantro/check_equivalence.py --reference <finished run> --out <dir>`
recomputes a few genes and one fold of stage one and the classifier both ways and compares them
(`equivalence.json`); run it after any change to the fast paths (2026-10-09, smoke run: five genes
byte-identical, two classifier folds and stage one equal in weights, losses, predictions,
probabilities, conformal p-values and RNG state; released deletion 464 s for the five genes against
97 s, released classifier epoch 4.3 s against 0.7 s on the smoke fold; 2026-10-10: the fold replay
leaves every RNG where a trained fold does, and the fold-per-process classifier reproduces the smoke
run's sequential weights, test scores, metrics pickle and conformal outputs for all ten folds,
`outputs/cilantro_reproduction/check_fold_parallel_smoke40/`). A run interrupted on the released
path resumes on the fast path unchanged: the gene files are the same, and `perturb.py` first
removes any gene whose files are incomplete.

`hpc/cilantro/run.sh RUN_ID --dictionary gc95M --benchmark` runs Cilantro-SL on the multi-study SL benchmark
(`benchmark_gc95M`, 2026-11-05). The token set is the 500 HVGs and the benchmark's genes (18,729 genes) in place of
the SynLethDB genes, and only the SL split's lines are tokenized (1,144). The released deletion step runs as above
(about 4 h on four H20s). Then `benchmark.py` refits the released stage one, with the released settings, on the
rows of the split's single-cell training and bulk labelled lines that have GeneEffect. It trains the released classifier on
the training lines' genetic pairs, with each line's own label (SL when half or more of its rows are SL). It writes the SL
probability of each held-out (line, pair) to `benchmark/scores.parquet`. A pair whose genes are not both in the line's
top 2,048 tokens gets 0.5, so these pairs tie. `--gene-subset 200 --epochs 2` makes a smoke run (`benchmark_smoke`, 5 min).

The port's coverage report (step 0, CPU only) runs in the project environment:
`.venv-tx1/bin/python -m src.experiments.historical.cilantro_coverage --out outputs/cilantro_coverage/<id>`.

## The PIE harness

PIE (Arc Institute, [ArcInstitute/pie](https://github.com/ArcInstitute/pie)) is vendored and tracked in
`third_party/pie/` at `21a526a` ([VENDORED.md](../third_party/pie/VENDORED.md); CC BY-NC-SA 4.0, which our additions
there share), so it arrives with `git pull`. It runs in its own environment, `.venv-pie` (Python 3.12, `torch==2.10.0`
from the cu128 index, pandas 3), built from PIE's `uv.lock` by `hpc/pie/setup.sh`, which ends with a CUDA check (a
bf16 matmul and scaled-dot-product attention on `cuda:0`; the cu128 wheels run on the H20's driver 550, CUDA 12.4, through
minor-version compatibility). PIE's own tests run from its directory (814 passed on the H20, 2026-10-16):
`(cd third_party/pie && ../../.venv-pie/bin/python -m pytest tests -q -p no:cacheprovider)`.

`hpc/run.sh pie-reproduce --run-id ID [--device cuda] [--steps fetch,download,evaluate,compare]` is Task 1 of the
[stage-one design](../docs/specs/2026-10-16-pie-geneeffect-ladder-design.md): PIE's released checkpoints
(`PIE_replogle_xdataset` on Replogle `test_seen` and `test_unseen`, the four `PIE_replogle_wdataset` folds on
`unseen_ctx`) re-scored by the vendored code and compared with the metric files released with them. It exports
`PIE_DATA_ROOT=data/pie` and `PIE_CACHE_DIR=data/pie/cache` and, unless set, `HF_ENDPOINT=https://hf-mirror.com`;
the runner sets `PIE_RUNS_ROOT` to the run's `runs/`. Steps, each skipped once its outputs exist (rerun the same id to
resume): `fetch` (checkpoints and released tables at pinned revisions into `data/pie/released/`), `download` (every split,
dataset and source the checkpoints pin, through PIE's resolver, each retried up to 100 times with backoff: the H20's
connections to hf-mirror's CDN redirect drop for minutes at a time), `evaluate`
(`pie.evaluate.run_eval` per target, from the target's run directory so the checkpoints' legacy relative paths
`data/splits/…` and `data/sources/aliases.yaml` map to PIE's Hugging Face copies, as PIE intends), `compare`
(`comparison.csv`, `granular_comparison.csv`, `results.md`). Evaluation reads the data config saved in each
checkpoint and rebuilds same-perturbation evidence from its training data, so `download` fetches every dataset and
source the checkpoints pin (all five cross-dataset training sets; `ncbi_text` alone is 43 GB) into
`data/pie/hf/`; PIE checks each array against its `meta.json` SHA-256. Output: `outputs/pie_reproduction/<id>/`
(`run.json` with pins, environment and step times).

## The stage-one inputs

Task 2 of the stage-one design ([protocol §15](../docs/03-geneeffect-protocol.md#15-stage-one-inputs),
[data card](../docs/data/ladder-inputs.md)) writes under `data/geneeffect_joint/v2/ladder/` from
`configs/stage_one/ladder_inputs.yaml`. Every step skips what exists, so a rerun resumes; shards on one host share the
reference pass and `run.json` through a lock file (`pie/.lock`). Launch each with `setsid nohup … > outputs/launches/<name>.log 2>&1 < /dev/null &`.

```bash
CFG=configs/stage_one/ladder_inputs.yaml
OMP_NUM_THREADS=32 hpc/run.sh ladder-inputs $CFG --steps lines,cells,mean,summaries       # CPU, ~20 min
CUDA_VISIBLE_DEVICES=$i hpc/run.sh ladder-inputs $CFG --steps tx1 --shard $i/4            # one per GPU, ~7 cells/s each
SRC=data/pie/hf/datasets/arcinstitute/pie_sources/cb1aaa4e7655605bdc70a9bd77bbd62016b8c7d7
OUT=data/geneeffect_joint/v2/ladder/gene_sources
hpc/run.sh gene-sources proteins --pie-sources $SRC --out $OUT/protein                    # .venv-pie, CPU
hpc/run.sh gene-sources afdb --afdb-dir data/afdb --out $OUT/protein                      # EBI FTP, 5.2 GB
CUDA_VISIBLE_DEVICES=0 hpc/run.sh gene-sources esm --out $OUT/protein                       # ESM C, ESM3 (afdb.json), 1 GPU
hpc/run.sh gene-sources align --genes data/geneeffect_joint/v2/ladder/genes.parquet --pie-sources $SRC \
  --protein $OUT/protein --gene2vec data/cilantro/official/gene2vec_embs.pt \
  --gene-names data/cilantro/geneformer/geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl --out $OUT
hpc/run.sh pie-precompute $CFG --steps rescore                                            # 1 GPU
CUDA_VISIBLE_DEVICES=$i hpc/run.sh pie-precompute $CFG --steps pilot --shard $i/N         # then --steps basis: pilot/pilot.md
CUDA_VISIBLE_DEVICES=$i hpc/run.sh pie-precompute $CFG --steps full --shard $i/N          # after pilot.md is read
hpc/run.sh pie-precompute $CFG --steps assemble
```

The ESM steps run in `ESM_PYTHON` (default the H20's
`/2023533015/ppi-mutation-ddg/.venv`, `esm` 3.2.1) with `ESM_HF_HOME` (default `/2023533015/.cache/huggingface`) offline;
`hpc/esm/setup.sh` builds an own `.venv-esm` should that environment go. The Cellosaurus text is built on the Mac
(`uv run --isolated --no-project --python 3.12 --with-editable 'third_party/pie[sources]' python -m
src.data.builders.cellosaurus_context --out DIR`, with `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` set on that
command line only; a SOCKS `ALL_PROXY` in the shell needs `--with socksio`) and copied to `ladder/context_text/` with a
SHA-256 check.

## The stage-one model

Task 3 of the stage-one design ([protocol §16](../docs/03-geneeffect-protocol.md#16-stage-one-model)) trains the
architecture arms on the ladder inputs, one config per run under `configs/stage_one/architecture/`. Training, the
queue and the report run in `.venv-pie` (re-sync it after a pull that changes `third_party/pie/uv.lock`:
`hpc/pie/setup.sh` syncs the `geneeffect` group, scikit-learn, which the project scorer reads); the linear model reference runs
in the project environment. A run directory is `outputs/stage_one/<task>/<config stem>[__fold<K>]/` (`--run-id`
overrides the stem); a run whose `row.json` exists is skipped, an interrupted one resumes from `last.pt`, and a run
directory refuses another config.

```bash
A=configs/stage_one/architecture
.venv-pie/bin/python -c "import src.eval.accuracy, src.eval.metrics, src.linear_model.targets"   # import check
hpc/run.sh stage-one-linear-model $A/linear_model_reference.yaml --run-id linear_model_reference_seed0        # CPU; components_reader needs it
CUDA_VISIBLE_DEVICES=0 hpc/run.sh stage-one $A/film_dH.yaml [--fold K]                    # one run, 1 GPU
setsid nohup hpc/run.sh stage-one-queue --gpus 0,1,2,3,0,1,2,3 --folds main,0,1,2,3,4 $A/*.yaml \
  > outputs/launches/stage_one_3a.log 2>&1 < /dev/null &   # two runs per GPU; skips linear_model_reference and report_3a
CUDA_VISIBLE_DEVICES=0 hpc/run.sh stage-one-report $A/report_3a.yaml                      # results.md; the dH ridge on 1 GPU
P=configs/stage_one/linear_model_input   # protocol §16's linear model input
setsid nohup hpc/run.sh stage-one-queue --gpus 0,1,2,3,0,1,2,3 --folds main,0,1,2,3,4 $P/*.yaml \
  > outputs/launches/stage_one_linear_model_input.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 hpc/run.sh stage-one-report $P/report_linear_model_input.yaml
G=configs/stage_one/gene_reader   # protocol §18's per-gene reader
OMP_NUM_THREADS=32 hpc/run.sh stage-one-linear-model $G/linear_model_reference.yaml --run-id linear_model_reference_20261113
setsid nohup hpc/run.sh stage-one-queue --gpus 0,1,2,3,0,1,2,3 --folds main,0,1,2,3,4 $G/*gene_reader*.yaml \
  > outputs/launches/stage_one_gene_reader.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 hpc/run.sh stage-one-report $G/report_gene_reader.yaml
```

Each run holds its delta on the GPU in bf16 (ΔH is 28 GB, ΔE 19 GB), copied by gene chunk from the memory-mapped
fp32 file. A run peaks at about 21 GB on ΔE, 30 GB on ΔH and 35 GB on ΔH with the multi-encoder, so two share an
H20 and three ΔH runs do not; listing a GPU twice in `--gpus` runs two jobs on it. A training step is a few ms of
kernel launches, so a lone run leaves its GPU mostly idle through setup, validation and the final writes; a second
run fills them. Without a PIE trunk the step runs as CUDA graphs (`torch.compile`, `reduce-overhead`) on the block
gathered before it, and the first epoch carries the compilation (about 20 s, a few once the inductor cache under
`/tmp` holds it); a trunk arm trains eager. The exceptions are `film_dE_full` and `multi_dE_full`, which stream the
full-panel predicted change (~311 GB under `pie/full/`) from its memory maps every epoch: they want it in the host
page cache (1.5 TB on the H20; the frame-mean pass at their start fills it), and each epoch's `epoch_seconds` in the
log shows whether they read from memory (about 1–1.5 min) or from disk. The queue logs each run to `outputs/stage_one/<task>/logs/<run>.log`, where the
epoch-0 line (before any update) shows validation within-gene R² equal to the base row's in the report, and the
base's per-line residual Spearman, the selector (protocol §6).

## The cell states

Task 8 of the stage-one design ([protocol §17](../docs/03-geneeffect-protocol.md#17-cell-states),
[data card](../docs/data/cell-states.md)) writes `states/` (the frozen PIE's state outputs under `states/pie/`) beside the stage-one inputs from
`configs/stage_one/states/telemetry.yaml`. The cNMF, starCAT and scVI fits run in `.venv-states` (`hpc/states/setup.sh`:
cnmf 1.7.1, starcatpy 1.0.10, scvi-tools 1.4.2, torch 2.10.0 cu128), which the project environment's `fit` step calls
through `STATES_PYTHON` (default `.venv-states/bin/python`); the two exchange files only. `fit` needs one GPU visible
for scVI; the rest of `fit`, `project` and `inputs` is CPU. Every step skips what exists.

```bash
hpc/states/setup.sh                                                                        # once
CFG=configs/stage_one/states/telemetry.yaml
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=32 hpc/run.sh cell-states $CFG --steps fit,project,inputs
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i nohup hpc/run.sh pie-precompute $CFG --steps states --set ALL --shard $i/4 > outputs/launches/states_$i.log 2>&1 & done
wait                                                                                        # every shard has exited
hpc/run.sh pie-precompute $CFG --steps assemble_states --set ALL
hpc/run.sh cell-states $CFG --steps report                                                 # results.md
```

`--set ALL` runs the real sets and permutation seed 0 before seeds 1 and 2; a shard can be stopped there and the others
assembled by name (`--set cycle,programs,…`). A one-set trial before the queue: `--steps states --set cycle --shard 0/16`.

## SL stage two

`sl-stage-two` (`src.experiments.sl.stage_two`, SL protocol §10) reads the SL screen benchmark
(`data/sl_screens/benchmark/pairs.parquet`), the PARIS pairs and each arm's finished GeneEffect (a linear model reference, or a
stage-one run with its five fold runs and `V.npy`). It fits a PARIS head per arm and a viability-embedding head per
stage-one run on the single-cell SL training lines and writes
`scores.parquet`, `metrics.json` and `results.md` to `outputs/sl/stage_two/<run id>/`, in a few minutes on the CPU. The
run directory refuses another config.

```bash
OMP_NUM_THREADS=1 hpc/run.sh sl-paris configs/sl/paris.yaml --run-id paris_seed0   # ~4,250 genes, n_jobs 60
hpc/run.sh sl-stage-two configs/sl/stage_two.yaml --run-id stage_two_seed0
```

## SL benchmark baselines

Stage one and stage two are separate (decision 2026-11-04). The stage-one benchmark keeps the final split. Since
2026-11-12, stage two reads the SL backbone's GeneEffect table, fitted without the SL held-out lines (below).
`linear-model-production` (one fit on every labelled line) is kept for reference only: it reads the GeneEffect of
most SL held-out lines.

`sl-conditional` (`src.experiments.sl.conditional`, SL protocol §11 and §12) is the modular SL backbone. It fits
the reader of its linear model config on the SL split's training side (training lines out of fold) and writes
`predictions.parquet` (GeneEffect of every SL line), `scores/{conditional,conditional_ablated,head,head_ablated}.parquet`
(with the columns `covered` and `partner`), `metrics.json` and `results.md` (stage-one metrics beside the ceiling of
two DepMap screens of one line, the head's penalties) to `outputs/sl/conditional/<run id>/`.
`configs/sl/conditional.yaml` reads the default linear expression model (round zero); `configs/sl/conditional_buffer.yaml`
reads the buffer-aware reader (the partner block added). The terms come from each line's edited rows; a full run
queries each line and each lost gene once.

`sl-baselines` (`src.experiments.sl.benchmark_baselines`) reads the multi-study SL benchmark's labels, the SL split
and the backbone's GeneEffect table ([data card](../docs/data/sl-benchmark.html)). It scores each validation and test
list with pair identity, pan-essentiality, the predicted and the measured GeneEffect (a reference, on lines with
GeneEffect only), and writes `scores.parquet`, `lists.csv`, `lines.csv`, `metrics.json` and `results.md` to
`outputs/sl/benchmark_baselines/<run id>/`. The headline is recall in the top 10% of each list (the known positives of
a line's lists pooled, then the mean over lines), with the per-line difference from pair identity; AUROC and AUPRC over
the positive rate follow. Each row is given for all lines, the lines with and without GeneEffect, and the cancer and
non-cancer lines. An anchor list is ranked by gene B, so drug anchors keep their lists. Table 3 repeats the headline for each stratum of held-out pairs, ranked inside the
stratum of each screen: a seen pair (the same study labelled it in a training line) whose label agrees with the
majority of those labels, a seen pair whose label differs, and a new pair whose genes the training rows have. The
labels as the score (the ceiling) and a seeded random score (about 0.10) check the code. Table 3 also gives two
subsets: the partner pairs (the `partner` column of a score table) and the rows where no gene of the pair is a
dependency alone (measured GeneEffect at or above −0.5). Table 5 gives the per-line difference of each (scorer,
control) pair in the config's `paired_differences`, with the same line bootstrap.

```bash
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 nohup bash -c "hpc/run.sh sl-conditional configs/sl/conditional.yaml \
  --run-id conditional_seed0 \
  && hpc/run.sh sl-baselines configs/sl/benchmark_baselines.yaml --run-id benchmark_backbone_seed0" \
  > outputs/launches/sl_backbone_chain.log 2>&1 &
# the buffer-aware reader (SL protocol §12), then the harness with both backbone runs
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 nohup hpc/run.sh sl-conditional configs/sl/conditional_buffer.yaml \
  --run-id conditional_buffer_seed0 > outputs/launches/conditional_buffer_seed0.log 2>&1 &
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 nohup hpc/run.sh sl-baselines configs/sl/benchmark_baselines.yaml \
  --run-id benchmark_buffer_seed0 > outputs/launches/benchmark_buffer_seed0.log 2>&1 &
```

The config's `extra_scores` adds scores from other harnesses, per (gene A, gene B), with the line and the design
when the table has `model_id` and `design`; the run stops if a held-out row is missing from a table, and a NaN score
leaves the row unranked by that scorer. KG4SL makes one (`hpc/kg4sl/`, its own environment `.venv-kg4sl`, Python 3.11 and
TensorFlow 2.15 on CPU). The released code (commit `be3626a`) needs TensorFlow 1.15 and Python 3.6, which the H20
cannot install, so `official.patch` runs it in TensorFlow 2's `compat.v1` graph mode: the import line and Xavier as
`glorot_uniform` (the same initializer). Send the code from the laptop, then build:

```bash
ssh -p 30838 root@10.15.171.204 'cd /2023533015/VCC_Project && git init -q third_party/kg4sl'
git -C <clone of JieZheng-ShanghaiTech/KG4SL> push ssh://root@10.15.171.204:30838/2023533015/VCC_Project/third_party/kg4sl be3626a6970479affb95fdf6b1d84a6244e52637:refs/heads/pinned
# on the H20
git -C third_party/kg4sl checkout pinned && hpc/kg4sl/setup.sh
PYTHONPATH=third_party/kg4sl/src .venv-kg4sl/bin/python hpc/kg4sl/train.py --out outputs/sl/kg4sl/seed0   # ~1 min
```

`train.py` uses the official SynLethKG without its SL, non-SL and SR edges and without every edge between the two
genes of a benchmark pair. It trains the official model with the official settings on the training lines' genetic
pairs (SL when half or more of the pair's rows are SL), with one tested non-SL pair for each SL pair (the official
ratio; with all tested pairs, which are 1% SL, the score became one constant). It keeps the epoch with the best
validation AUROC. A pair with a gene outside the graph (5% of the held-out rows) gets 0.5, so these pairs tie.

`sl-slidr` (`src.experiments.sl.slidr`, a Python port of the official R code, `third_party/slidr/PROVENANCE.md`) runs
SLIdR's rank test per lineage (R runs per cancer type) on the training-side lines of the SL split in the held-out line's
lineage (at least five). Each gene of a held-out pair with at least two
mutated (damaging or hotspot call, or 0 copies) and two wild-type lines is a driver. A pair scores -log10 of the
smaller mutated-lines p-value of its two orientations, 0 when the wild-type test rejects it (p < 0.1), and -log10 0.5
when no orientation can be tested (`covered` false); the scores are per (line, pair). The hotspot matrix came from the laptop (SHA-256 `a86cd8c9…`).

```bash
hpc/run.sh sl-slidr configs/sl/slidr.yaml --run-id seed0   # outputs/sl/slidr/seed0/scores.parquet
```

PARIS on the benchmark (`configs/sl/paris_benchmark.yaml`) fits each gene of a held-out pair on DepMap 26Q1
GeneEffect against the bulk expression of every protein-coding gene, on the SL split's training-side lines. A pair
scores the larger importance of a selected pair with r > 0, in either orientation, else 0. A gene takes about 290 s
on one thread, so the 18,435 genes run as four shards (two on each container, `n_jobs` 60) and a merge:

```bash
for i in 0 1; do OMP_NUM_THREADS=1 hpc/run.sh sl-paris configs/sl/paris_benchmark.yaml --run-id seed0_shard${i}of4 --shard $i/4; done   # 30757
for i in 2 3; do OMP_NUM_THREADS=1 hpc/run.sh sl-paris configs/sl/paris_benchmark.yaml --run-id seed0_shard${i}of4 --shard $i/4; done   # 30838
hpc/run.sh sl-paris configs/sl/paris_benchmark.yaml --run-id seed0 --merge seed0_shard{0,1,2,3}of4
```

The two paralog classifiers run in `.venv-tx1` (scikit-learn 1.7). `hpc/paralog/dekegel_scores.py` looks up De Kegel
2021's released score of each held-out pair in Table S8 (`data/sl_baselines/dekegel/`, from
cancergd.org/static/paralogs, SHA-256 checked); a pair outside the 36,648 paralog pairs gets the table's median score.
`hpc/paralog/kebabci.py` refits Kebabci 2026's contextualised random forest (the official 16 features and settings,
commit `f966551`) on the training-side lines' paralog rows. Its inputs: DepMap 26Q1 (GeneEffect ranks, expression, copy
number); mutations and common essentials of DepMap 24Q4 (`data/sl_baselines/depmap24q4/`, from figshare through the
laptop: the H20 reaches neither figshare nor DepMap's bucket, and 26Q1's links need a browser); STRING v11.5
(`data/sl_baselines/string/`); GOA human release 218 (`data/sl_baselines/go/`). By default its features read the
held-out line's measured GeneEffect, so that run is a reference row. With `--held-gene-effect`, the held-out lines read
stage one's predicted GeneEffect (the SL backbone's `predictions.parquet`), and the fit keeps the training-side lines'
measured GeneEffect, both on the predicted genes: that run is a baseline row.

```bash
.venv-tx1/bin/python hpc/paralog/dekegel_scores.py --out outputs/sl/dekegel   # seconds
OMP_NUM_THREADS=32 .venv-tx1/bin/python hpc/paralog/kebabci.py --out outputs/sl/kebabci/seed0
OMP_NUM_THREADS=32 .venv-tx1/bin/python hpc/paralog/kebabci.py --out outputs/sl/kebabci/predicted_seed0 \
  --held-gene-effect outputs/sl/conditional/conditional_seed0/predictions.parquet
```

MiT4SL (`hpc/mit4sl/`, `.venv-mit4sl`: torch 2.6 on CUDA 12.4, PyG 2.6.1, torch-sparse built from source; the released
torch 1.12 has no Hopper kernels) needs the released `data.zip` (1,030,474,683 bytes, SHA-256 `d06124f2…`) from the
authors' Google Drive, which the H20 cannot reach: download it on the laptop, check the SHA-256, copy it to
`data/mit4sl/official/` and unzip it (Python `zipfile`; the H20 has no `unzip`). `prepare.py` builds each line's
contextualised PPI graph from DepMap TPM (the tutorial's threshold 400 and 32 neighbours; 106 lines) and the train,
validation and test rows, and removes every PrimeKG and PPI edge inside a benchmark pair (about 5 min). `train.py` runs
the official model and training step (cross-cell-line protocol, 20 epochs of one full-batch step each, as released) for
one run and keeps the epoch with the best validation AUROC (about 35 min on one GPU). Two changes let the released code
run: the official `HGTConv(..., group='sum')` drops the argument (PyG 2.3+ always sums), and each line's 64-layer graph
encoder is checkpointed (the same outputs and gradients; 106 graphs do not fit on one GPU otherwise).

```bash
hpc/mit4sl/setup.sh
PYTHONPATH=third_party/mit4sl/src .venv-mit4sl/bin/python hpc/mit4sl/prepare.py --out data/mit4sl/benchmark
PYTHONPATH=third_party/mit4sl/src .venv-mit4sl/bin/python hpc/mit4sl/train.py --data data/mit4sl/benchmark \
  --out outputs/sl/mit4sl/seed0 --device 0
```

## The `linear-model` command

`hpc/run.sh linear-model CONFIG [--run-id <id>] [--experiments A,B]` runs the experiment runner of the linear model
([protocol §10](../docs/03-geneeffect-protocol.md#10-linear-model); `configs/linear_model/bridge_remedies.yaml`).
The run id defaults to a timestamp, and the route needs no GPU. The config lists experiments, each a bridge remedy
with its settings and block sets; the runner first prepares pseudo-bulk into `prepared_root` (summing the raw UMI of
each line's basal cells; the one step that reads raw data; the prepared root of `all` is reused), then for every
setting builds the bridged inputs, fits the linear model for every block set and penalty, and scores validation and test
(and the bulk-input oracle on validation). It decides nothing: gains are measured against the reference row the config
pins, and the table is read by a person. `configs/linear_model/default_linear_model.yaml` holds the default linear model alone;
it reads the final split through its joint config (single-cell training lines, then the bulk-only lines; lymphoid and
myeloid lines of either kind leave its training side). Run it with `linear-model`, then `linear-model-export`, then the two stacks:

```bash
R=default_linear_model_final_split_20261012
OMP_NUM_THREADS=32 nohup bash -c "hpc/run.sh linear-model configs/linear_model/default_linear_model.yaml --run-id $R \
  && hpc/run.sh linear-model-export configs/linear_model/default_linear_model.yaml --run-id $R" > outputs/launches/$R.log 2>&1 &
OMP_NUM_THREADS=32 nohup hpc/run.sh linear-model-stack configs/linear_model/single_cell_stack.yaml \
  --run-id single_cell_stack_final_split_20261012 > outputs/launches/single_cell_stack_final_split_20261012.log 2>&1 &
```

`hpc/run.sh linear-model-selected CONFIG --run-id <id>` fits the config's reference row (its last stage must be the
data-selected genes) and writes `data_selected/{features.csv, targets.csv, ablation.json, summary.md}` into the same run
directory: feature use, per-target gains of the stage, and scores with the stage masked to or without its top features.

`hpc/run.sh linear-model-export CONFIG --run-id <id>` writes the config's reference row for the single-cell correction into
`outputs/linear_model/<run_id>/export/{linear_model.npz, linear_model.json}`: validation and test from the reference fit, and every
labelled single-cell training line out of fold (the bridge and every stage refitted without the line's patient-grouped
fold), all 377 on the final split; `load_inputs` refuses an export lacking a line the run exposes. An existing export is never replaced; a correction checkpoint records the export's run id, so a new export
needs a new run id. It runs on CPU (set `OMP_NUM_THREADS` and `MKL_NUM_THREADS`); `linear_model.json` holds its own
validation and test scores, which reproduce the reference row of the `linear-model` run.

`hpc/run.sh linear-model-stack CONFIG --run-id <id>` stacks per-gene single-cell statistics (the prepared `q_sc` caches)
on the export of the config's `linear_model_run` (GeneEffect protocol §13) and writes
`outputs/linear_model/<run_id>/{run_config.json, single_cell_stack/{rows/<arm>.json, facts.json, results.md}}`. It
reads the linear model run's bound config, so its genes, residual SD and lines are the export's, and refuses an export that
does not match them. It runs on CPU in one process (set `OMP_NUM_THREADS` and `MKL_NUM_THREADS`); a finished run
directory is refused, so a rerun needs a new run id.

`hpc/run.sh linear-model-noise CONFIG --run-id <id> [--sections A,B]` runs the noise diagnosis (GeneEffect protocol §14;
`configs/linear_model/noise_diagnosis.yaml`) against the config's `linear_model_run` and writes
`outputs/linear_model/<run_id>/{run_config.json, noise/{<section>.json, cell_sampling_genes.csv, results.md}}`; a
finished section is skipped, so a rerun with the same run id resumes. The first section that needs them prepares
`noise/subsamples.parquet` and `noise/programs/` under `prepared_root` from every line's raw cells (CPU, 16 processes);
Sections run in the order `--sections` lists them, so the two that read no raw cells go first. `deposit_swap` first needs
the second-deposit builds (Extra single-cell lines, below). The Cilantro-SL encodings run through `linear-model`:

```bash
OMP_NUM_THREADS=32 nohup hpc/run.sh linear-model configs/linear_model/cilantro_encodings.yaml \
  --run-id cilantro_encodings_20261012 > outputs/launches/cilantro_encodings_20261012.log 2>&1 &
OMP_NUM_THREADS=32 nohup hpc/run.sh linear-model-noise configs/linear_model/noise_diagnosis.yaml --run-id noise_20261012 \
  --sections learning_curve,deposit_confound,cell_sampling,cell_states > outputs/launches/noise_20261012.log 2>&1 &
```

The run directory is `outputs/linear_model/<run_id>/{run_config.json, rows/<experiment>__<n>.json, results.md}`.
Each setting writes its row file when finished and is skipped when the file exists, so an interrupted run resumes by
rerunning the same command with the same run id; `results.md` is rewritten from whatever rows exist on every run.
`--experiments` restricts a process to the named experiments (all of them by default), so several processes can
share one run directory: rows are per setting and never shared. A run directory is bound to the config it started
with; another config needs a new run id. The reference tables it reads are committed under
`configs/linear_model/reference/`; the lines come from the final split (`configs/benchmarks/cell_line_geneeffect_split.json`).
The H20 reaches GitHub, PyPI, EBI/ENA/BioStudies and hf-mirror, but not NCBI GEO, figshare, Zenodo, the 10x CDN or
Cellosaurus: files from those hosts are downloaded on the Mac and copied to the H20 data directory with their SHA-256.

## Extra single-cell lines

Labelled DepMap lines outside the 226 with public basal single cells, and SL benchmark lines without GeneEffect (as
`unlabeled_train`; a line without a ModelID has its ID from `configs/benchmarks/sl_line_identities.csv`), join the
prepared root as raw-UMI h5ad
([card](../docs/data/single-cell-atlas-search.md)), then the final split as training lines
([card](../docs/data/cell-line-geneeffect-split.md)). One wave of the source spec at a time, on CPU; after the
membership file changes, rebuild the final split (on the Mac; it reads tracked files only) and commit it before
`prepare`, which reads the lines from it:

```bash
S=configs/data/extra_single_cell_sources_26Q1.json
.venv-tx1/bin/python -m src.data.builders.build_extra_single_cell_sources --spec $S --wave 1 --print-downloads \
  | while IFS=$'\t' read url path; do mkdir -p "$(dirname "$path")"; wget -c -O "$path" "$url"; done
OMP_NUM_THREADS=16 nohup .venv-tx1/bin/python -m src.data.builders.build_extra_single_cell_sources --spec $S --wave 1 \
  --output-dir data/sl_dependency_v0/processed/basal_raw_umi > outputs/launches/extras_wave1.log 2>&1 &
.venv-tx1/bin/python -m src.data.builders.build_extra_single_cell_lines build \
  --candidates configs/benchmarks/single_cell_atlas_candidates.csv --model data/sl_dependency_v0/raw/depmap/Model.csv \
  --gene-effect data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv \
  --split configs/benchmarks/cell_line_geneeffect_226_split.json --sl-rules configs/benchmarks/sl_label_rules.csv \
  --qc-dir data/sl_dependency_v0/processed/basal_raw_umi/qc --out configs/benchmarks/extra_single_cell_lines_26Q1.json
uv run python -m src.data.builders.build_cell_line_geneeffect_split \
  --split-226 configs/benchmarks/cell_line_geneeffect_226_split.json \
  --single-cell configs/benchmarks/extra_single_cell_lines_26Q1.json \
  --bulk configs/benchmarks/extra_bulk_lines_26Q1.json --out configs/benchmarks/cell_line_geneeffect_split.json
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 nohup .venv-tx1/bin/python -m src.experiments.prepare configs/geneeffect_joint.yaml \
  --pseudobulk-genes data/sl_dependency_v0/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv \
  > outputs/launches/prepare_extras.log 2>&1 &
```

The download step lists the files the wave lacks; files on hosts the H20 cannot reach are fetched on the Mac and
copied to the same paths. The builder skips a source whose `qc/<source>.json` exists and refuses to overwrite an
h5ad; an h5ad without its source's report (an interrupted run) stops it. The membership file is committed after the
build. The last command grows `data/geneeffect_joint/v2` and its pseudo-bulk by the new lines (Tx1 encodes them on
the visible GPU); the 226's files are left as written. To re-prepare one line, delete its h5ad, `lines/<ModelID>.npz`,
Tx1 cache directory and pseudo-bulk row.

Wave 3 (2026-11-04) adds SL benchmark lines (the sources are in the
[atlas card](../docs/data/single-cell-atlas-search.md#wave-3-2026-11-04)). Its files come from the Mac (GEO, figshare+
and 10x), copied to the same paths with their SHA-256; `kinker_scp542` reads the 226's `UMIcount_data.txt` in place, so
`--print-downloads` lists nothing for wave 3 once the copies are there. It also needs `gse311521` and `emtab13204`
built again, for MeWo, MP41 and PL-21: delete their `qc/<source>.json` and the h5ad of each line they wrote before the
build.

Wave 2 adds sources whose libraries are re-counted from SRA runs. Before building, the GENCODE reference is indexed
once and each re-counted library is counted (sratoolkit's `fasterq-dump` and kb-python, installed outside the
checkout under `/2023533015/tools/`; `pysam` from the `data` extra is in `.venv-tx1`):

```bash
K="--kb /2023533015/tools/kb-venv/bin/kb --fasterq-dump /2023533015/tools/sratoolkit/bin/fasterq-dump"
.venv-tx1/bin/python -m src.data.builders.recount_extra_single_cell_sources --spec $S --wave 2 $K index
.venv-tx1/bin/python -m src.data.builders.recount_extra_single_cell_sources --spec $S --wave 2 $K count \
  [--sources gse142750_recount]
```

The `.sra` runs and Cell Ranger BAMs come from SRA's public AWS bucket (`sra-pub-run-odp`, `sra-pub-src-1`), whose
single-stream rate from the H20 is a few MB/s; parallel ranged requests (about eight at once) are several times
faster, while ENA's FASTQ host was slower still and drops the barcode reads of some runs. A line whose chosen source
changes (a wave-2 source with more or better cells than its wave-1 one) is re-prepared: its h5ad, prepared npz, Tx1
cache directory and pseudo-bulk row are moved under `data/superseded/` before the new source is built.

**Second deposits** for the noise diagnosis's `deposit_swap`: `configs/data/second_source_lines_26Q1.json` (written on
the Mac by `plan_second_sources.py`) names, for 99 labelled training lines, a second source of the extras spec. The
builder's `--second-source` mode builds those lines, lines of the 226 included, into a directory of its own, one wave at a
time; the training caches are not touched:

```bash
S=configs/data/extra_single_cell_sources_26Q1.json; P=configs/data/second_source_lines_26Q1.json
O=data/sl_dependency_v0/processed/basal_raw_umi_second_source
for W in 1 2; do OMP_NUM_THREADS=16 .venv-tx1/bin/python -m src.data.builders.build_extra_single_cell_sources --spec $S \
  --wave $W --second-source $P --output-dir $O; done > outputs/launches/second_sources.log 2>&1
```

## Configuration and inputs

All configuration fields are explicit in `configs/geneeffect_joint.yaml`; unknown or
missing keys are errors; a checkpoint from before the selective-gene revision lacks its
fitted selective genes and residual scale and is refused, and one from before the single-cell
correction lacks its recorded linear model export and is refused too. Input paths are relative to the repository root. Preparation
requires one raw-UMI h5ad per line of the split at `paths.basal_raw_umi/<ModelID>.h5ad` (the 226 and the built
single-cell extras), the GeneEffect CSV, supplied ESM2 table, STATE checkpoint and
gene order, and the response sources. Missing Tx1 caches additionally need the
configured local Tx1 model and a GPU; newly encoded cells use collation seed 0, and the Tx1
cache's existing on-disk format is read, never rewritten. Tx1 reads raw UMI counts; every
other expression quantity is log-normalised by whole-library size to `T`, the median
library size of the non-targeting cells in the Nadig Jurkat and HepG2 sources. Response
sampling seed 42 is a preparation setting distinct from the runtime seeds 0/0/0. Training
only opens prepared caches and never rebuilds raw inputs; a prepared root or checkpoint
from before the expression-space change is refused for lacking `expression_space`.

## Tx1 GMM-ridge baseline

This CPU/scikit-learn baseline uses the existing Tx1 basal bags directly, without
PCA or ST responses. It fits a shared diagonal GMM64 on equal numbers of cells from
each labeled training line (seed 0), then a standardized 68-feature context vector
and independent Ridge(alpha=1) per gene. Both standardizers and the GMM are train-only.
The frozen settings and comparison boundaries are in the
design (`docs/specs/2026-09-07-tx1-gmm-ridge-design.md`, removed; `git show 1694f5c:<path>`).

```bash
uv run python -m src.experiments.joint.tx1_gmm_ridge fit --config configs/geneeffect_joint.yaml --out-dir outputs/baselines/tx1_gmm_seed0
uv run python -m src.experiments.joint.tx1_gmm_ridge evaluate --model outputs/baselines/tx1_gmm_seed0/model.joblib --split val
```

Fit requires a new output directory and automatically exports train and val, never
test. `model.joblib` stores the fitted model, preprocessing and split; `run.json`
records source/configuration and separate fitting/evaluation status. `diagnostics.json`
reports convergence, component weights, training occupancies and per-line fit-cell
positions/counts. Check convergence before interpreting scores; the code does not
silently lower K or change the embedding when fitting is difficult.

`evaluation/<split>/` contains joint-training-evaluator predictions/metrics, per-line/per-gene tables and
`context_features.csv` (64 occupancies plus entropy, effective component count,
assignment confidence and negative log likelihood). Train scalars use `train_eval_`.
A failed export can be retried with `evaluate`, restoring
the saved transforms and readout without refitting. Use only trusted local joblib
artifacts with their recorded sklearn version. This is a candidate baseline; no
performance improvement is established by the implementation tests.
