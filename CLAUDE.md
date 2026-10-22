# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research task and document authority

The task is context-specific synthetic-lethality (SL) prediction in held-out **cell lines**, read from their basal
profiles with a single-cell foundation model: stage one predicts DepMap GeneEffect, stage two ranks SL partners against
experimental SL-database labels (`docs/01-blueprint.md` §3). 

`docs/` outranks this file. Start at `docs/01-blueprint.md` (introduction, related work and task formulation, no modelling; it mirrors the
Notion page https://app.notion.com/p/3f4c3591aaf681f68db1ccb9fd595c41, so edit both together);
`docs/02-literature-review.md` is the extended related work (it mirrors the Notion page https://app.notion.com/p/3f6c3591aaf68105996dc71d2543c9d0, so edit both together); `docs/03-geneeffect-protocol.md` is the executable protocol of the implemented
GeneEffect track (its §11 holds the rules) and `docs/04-sl-ranking-protocol.md` the separate SL-pair protocol built on it (its §8 holds the rules; superseded in part by the 2026-10-09 blueprint). The current expression space,
STATE wiring and `all` run follow `docs/specs/2026-10-02-expression-space-and-all-pipeline-design.md`. Read
the `docs/data/` card before using a dataset. Results live in `results/`.

Name every model, head, arm, variant, run and stage by what it is ("shared MLP head", "the seed-0 joint backbone"), never by a
bare internal label (`A0`, `V1`, `Tier 3`). Where code uses a label, give it once in parentheses at first use.

## Current state (2026-10-22)

Replace this section on every update; history lives in `results/` and the protocol.

- **Stage one** (PIE design, [plan](docs/specs/2026-10-17-pie-stage-one-plan.md), protocol §16): no architecture arm
  reaches the default prior refitted on the frame (per-line residual Spearman 0.215 validation / 0.228 test;
  [record](results/stage_one_architecture/README.md)). FiLM reading the prior's prediction recovers most of it
  (0.199–0.203) but not all, and ΔE beside the prior adds nothing that line-permuted ΔE does not
  ([record](results/stage_one_prior_input/README.md)). Cell-state telemetry (Task 8) ran
  ([record](results/state_telemetry_seed0/README.md)).
- **Next**: the rest of Wave 2 and Task 9 (five vocabularies), after the whole-repo trim (`refactor/trim`) and the
  GPU-efficiency pass that follows it.
- The SL pair head is **unimplemented**. One config is one experiment at seed 0: train, select `best.pt` on
  **validation**, then score it once on test; there is no multi-seed stage. Nothing here is SL evidence.

## Commands

Python 3.11–3.12, managed by `uv`. Every Python invocation goes through `uv run`, from the repo root, as a module (`-m`);
imports are `src.*`.

```bash
uv sync                                                   # core deps + dev group (pytest, ruff, xgboost); scib/datasets are optional extras
uv run python -m pytest tests -q > pytest.txt 2>&1       # full suite; redirect — see rtk note
uv run python -m pytest tests/experiments/joint/test_all.py -q  # one file; -k for one test
.venv/bin/ruff check .                                    # lint (E,W,F only; import order not enforced)
.venv/bin/ruff format <files you touched>                 # never `ruff format .` — rewrites unrelated files

hpc/run.sh all configs/geneeffect_joint.yaml [--run-id <id>] [--gpus 0,1,2,3]  # prepare, comparison, train, val eval, baselines, readout, summary.md
hpc/run.sh revision CONFIG [--run-id <id>] [--gpus 0,1,2,3]   # one variant: train, val then test eval with baselines, summary.md + revision.json in outputs/geneeffect_revision/<id>/ (configs/revision/*.yaml; no comparison, no readout)
hpc/run.sh test outputs/geneeffect_joint/<id>/train/best.pt   # test for any checkpoint; `all` never runs test, `revision` scores its own best.pt
hpc/run.sh prior CONFIG [--run-id ID] [--experiments A,B]   # linear context prior, CPU: bridge remedies scored on validation and test; rows/, results.md in outputs/context_prior/<id>/
hpc/run.sh prior-selected CONFIG --run-id ID                # the reference row's data-selected genes, CPU: data_selected/ in the same run directory
hpc/run.sh prior-export CONFIG --run-id ID                  # the reference row for the correction, CPU: export/ (val/test, training lines out of fold); never overwritten
hpc/run.sh prior-stack CONFIG --run-id ID                   # single-cell q_sc statistics stacked on a prior export, CPU: single_cell_stack/ (protocol §13)
hpc/run.sh prior-noise CONFIG --run-id ID [--sections A,B]  # noise diagnosis against a prior run, CPU: noise/ (protocol §14)
hpc/run.sh pie-reproduce --run-id ID                        # PIE's released checkpoints re-scored by the vendored PIE in .venv-pie (hpc/pie/setup.sh), 1 GPU
hpc/run.sh stage-one CONFIG [--fold K]                      # one stage-one run (configs/stage_one/), .venv-pie, 1 GPU; stage-one-queue --gpus ... CONFIG... runs many
hpc/run.sh stage-one-report CONFIG                          # a stage-one report (results.md); stage-one-prior CONFIG --run-id ID: the frame-refitted prior, CPU
uv run python -m src.experiments.joint.compare_runs RUN_A RUN_B --split val   # paired line bootstrap of selective Spearman, A minus B
uv run python -m src.experiments.joint.evaluate --checkpoint <best.pt> --split val   # --split train for checkpoint diagnostics
```

`all` and `revision` skip every step whose output exists, so rerunning with the same run id resumes (training from `train/last.pt`). Every
GPU step uses every visible GPU, or those `--gpus` lists: untrained comparison arms on the first, training on all, then one
trained comparison job per GPU at a time (`revision` has only the training step). Direct worker invocation is debug-only; runner
details are in `hpc/README.md`.

- **Local Mac has no GPU, no model weights, no raw data.** Preparation, training and evaluation run on the H20 container
  (`hpc-execution` skill), without a scheduler. Tests use synthetic fixtures and skip silently when gitignored data or
  `accelerate` is missing — an "it imports" check proves nothing about an asset-dependent path.
- A global `rtk` hook rewrites Bash output: `ruff check .` prints `[]`, foreground pytest shows fake collection errors. Use
  `.venv/bin/ruff` and redirect pytest to a file. `rtk proxy <cmd>` when exact output matters.
- **Baseline the suite before changing code** (full run to a file) so a failure you cause is distinguishable from one you inherit.
- `tests/conftest.py` is load-bearing: it sets `PYTORCH_ENABLE_MPS_FALLBACK` and `OMP_NUM_THREADS` and imports xgboost before
  torch. Running a test file as a script segfaults. `-m` markers do nothing.

## Project rules

- When a change alters behaviour a doc describes, update that doc in the same change.
- Experiment code reports results; decisions are made by reading them, not in code.
- No formalism gates: do not add digest pinning, contract verifiers, eligibility ceremony or any check that blocks a run on
  model quality. Record provenance and proceed; model-quality signals are telemetry. The existing guards against silent wrong
  artifacts (below) fail closed and stay.
- GPU work needs the H20 container: finish everything local first (config, tests, commit, push) and end with the exact
  `hpc/run.sh` launch command.
- Sync code between machines through Git only: commit, push, then pull on the H20 checkout. Never rsync/scp/tar working trees.
- Work done on its own branch or worktree is merged into `main` as soon as the feature is established as working — suite and
  lint pass, any requested review is adjudicated, and an asset-dependent path has run on the H20 host — then `main` is pushed and the branch
  deleted. Do not leave finished work on a side branch.
- Commits use Conventional Commits (`feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`).
- Codex review is optional and runs **only when the user asks**. Then run it from Bash (the slash commands are
  `disable-model-invocation`) in a fresh `CODEX_HOME` holding only `auth.json` and a `config.toml` with `model = "gpt-6.1-sol"`,
  `model_reasoning_effort = "high"`, `approval_policy = "never"` — recreate it per review, or Codex loads MCP servers and hangs.
  Background `codex review --base <sha>` to a file (~200 KB) and adjudicate the findings against the code.

## Architecture

One route, `hpc/run.sh all`: **prepare → untrained comparison → train → trained comparison → evaluate (+ baselines, readout) → summary**,
driven by one strict YAML config. `data` and `model` never import `training`, `eval` or `experiments`.

- **Expression space.** Everything except Tx1's input is log space: whole-library `normalize_total` to a recorded target `T`,
  `log1p`, then STATE's HVG slice. Tx1 reads raw UMI, not CPM.
- **Preparation** (`src/experiments/prepare.py`) is the only place raw data is read. It writes fixed caches and a manifest
  recording `expression_space` under `data/geneeffect_joint/v2`; training opens those caches and never rebuilds them. Its
  lines are the split's (the 226 and the built single-cell extras), each read from `<paths.basal_raw_umi>/<ModelID>.h5ad`;
  an existing root grows by the lines it lacks, in the recorded space (the split path is not a preparation setting). The
  extras' h5ad are built from public deposits by `src/data/builders/build_extra_single_cell_sources.py`, after
  `src/data/builders/recount_extra_single_cell_sources.py` re-counts the SRA reads of deposits without raw counts.
- **Training** (`src/experiments/joint/train.py:run_training` → `src/training/trainer.py`) fits on the split's 377
  labelled training lines, as do the gene means, selective genes, residual SD and context PCA. STATE (`arc-state`, pinned commit)
  reads basal cells through its released encoder, driven by an ESM2 adapter's perturbation token; frozen Tx1 embeddings feed
  the head's context. The head is a rank-`factor_rank` product of a gene factor and a line factor read from the
  128-component context PCA (fitted on training lines), plus a per-(g, c) correction that never sees the context, and predicts in units of the
  per-gene training residual SD. One GeneEffect objective per update, `train.objective` (`huber`, `standardized_mse`,
  `pearson_blocks`); response replay on the four anchor lines every fourth update only when `response_weight > 0` (the base config
  has 0). `train.state_mode` freezes STATE at the released weights or trains it; no STATE is `head_blocks` `use_delta_proj` and
  `use_s` both false, which skips STATE and the adapter. Warmup then cosine learning rate. `best.pt` and early stopping use
  **only** `val_selective_spearman` (higher wins), over the selective genes fitted on training lines.
- **Evaluation** (`src/experiments/joint/train.py:evaluate_checkpoint`) restores fitted preprocessing (gene means, variable and
  selective genes, residual SD, context PCA) from the checkpoint;
  `src/baselines/residual.py` fits the control ladder (gene-mean, copy-prior, nearest-line, context-PCA-ridge) on train only;
  `src/experiments/joint/all.py` chains every step and writes `summary.md`; `src/experiments/joint/revision.py` chains training, validation and test
  evaluation and baselines for one config.
- `configs/benchmarks/cell_line_geneeffect_split.json` (the **final split**, [card](docs/data/cell-line-geneeffect-split.md)) is
  the sole membership authority: 379 single-cell training lines (the 226's 171, PC9 and HeLa unlabelled, then the 208
  single-cell extras), 27 validation, 28 test (A549 since 2026-10-11), 713 labelled and 576 unlabelled bulk-only lines
  that join the prior's training side through bulk RNA, and 12 excluded. `src/data/builders/build_cell_line_geneeffect_split.py`
  builds it from `cell_line_geneeffect_226_split.json`, `extra_single_cell_lines_26Q1.json` and `extra_bulk_lines_26Q1.json`,
  which no fit reads; its table (`.csv`) gives each line's lineage and patient. `src/data/splits.py:assert_fit_eligible`
  guards every fit. Only split, config, provenance files and the context prior's reference tables are tracked data.

## Pitfalls and claim boundaries

Mistakes here produce a complete-looking **wrong artifact**, not an exception.

- `src/experiments/config.py` raises on unknown or missing keys; never add `.get(key, default)` config reads.
- `load_inputs` refuses manifests without `expression_space`, and checkpoint loads raise on zero or mismatched keys
  (`load_released_state`). Never bypass either or add a bare `load_state_dict(..., strict=False)`.
- Resume rejects a conflicting config; a batch-size change needs a new run id. A checkpoint records the split
  membership it was fitted under, and `load_inputs` refuses to restore it under another: every checkpoint from
  before the 2026-10-11 in-place re-split (A549 to test) is refused for resume, evaluation and readout.
- Residuals: targets center on the fold-fit `mu_hat_g^(-c)`, predictions on fold-independent `mu_bar` — centering a prediction
  on `mu_hat` scores Spearman +1.0 by construction. Gene-mean and copy-prior residual correlations are undefined, not zero.
- Join cell lines by DepMap ModelID, never by informal names at run time (`K-562` ≠ `K562`): a source's spelling becomes a
  ModelID only through a reviewed row of `configs/benchmarks/single_cell_atlas_candidates.csv`. Fit means, gene
  membership and normalization on labeled training lines only.
- Context claims need residual evaluation against context-blind priors; the rules are in the GeneEffect protocol (§11) and the
  SL protocol (§8). Single-gene predictions are not SL or genetic-interaction evidence.
