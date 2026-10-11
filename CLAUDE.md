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

## Current state (2026-10-11)

The joint GeneEffect model (`configs/geneeffect_joint.yaml`) trained, tested and baselined once at seed 0, before the
**expression-space change** (STATE was fed raw counts), so those numbers are not like for like with the current pipeline
(`docs/03-geneeffect-protocol.md` §7, `results/joint_geneeffect_seed0/`). The 2026-10-02 `all` run
(`all_20261002T174946Z`) finished: validation residual Pearson 0.068 for the joint model against 0.133 for the Tx1 context ridge,
and the response comparison favours the HVG MLP over fine-tuned STATE and over the Tx1 MLP. The current work is the **revision**
(`docs/specs/2026-10-03-geneeffect-revision-design.md`, then `docs/specs/2026-10-03-geneeffect-head-revision-design.md`): a
selective-gene selector, a nested low-rank head over a training-line context PCA, and objective and STATE screens on the 30734
container. The other current work is the **linear context prior**
(`docs/specs/2026-10-04-context-generalization-design.md`, protocol §10; branch `feat/context-prior`): a closed-form
CPU prior from bulk and pseudo-bulk expression, with DepMap lines outside the 226 as extra training-side lines
(`docs/data/extra-bulk-lines-26q1.md`). Its 2026-10-04 run (`results/context_prior_seed0/`): validation chose the config
without the haematopoietic extras, a ridge on 128 expression components over 953 labelled lines read from bridged
pseudo-bulk; selective Spearman 0.223 validation / 0.223 test against 0.130 / 0.121 for the Tx1 context ridge; gene-level
blocks did not survive the bridge. The bridge-remedy runs (`results/bridge_remedies_seed0/`, a minimal runner that
decides nothing; plan `docs/specs/2026-10-04-bridge-remedies-plan.md`) found no remedy better than the affine bridge,
and, once every components and gene penalty pair is reported (`configs/context_prior/bridge_remedies_penalty_grid.yaml`),
the affine bridge with all gene-level blocks at components penalty 1 and gene penalty 10 scores 0.230 validation /
0.235 test (+0.007 and +0.013 over components alone, intervals above zero). Its follow-ups
(`results/default_prior_followups_seed0/`) found that the data-selected genes carry that gain, so the **default prior**
(`configs/context_prior/default_prior.yaml`) is the affine bridge with components at penalty 1 and data-selected genes at gene
penalty 10: 0.229 validation / 0.234 test. The **single-cell correction** (the nested head stacked on that prior's out-of-fold export,
protocol §12) ran wave one (`docs/specs/2026-10-07-single-cell-correction-plan.md`, `results/correction_wave_one_seed0/`):
`paths.prior` names the export, the stack starts at the prior and validates before its first update, the baselines add
the prior alone and the prior plus the Tx1 ridge, and `configs/correction/` holds the objective screen and the STATE
runs. In all six runs (four objectives with STATE absent, then standardised MSE with frozen and trainable STATE)
`best.pt` is the prior alone (0.229 / 0.234): every trained epoch lowers validation, and the prior plus the Tx1 ridge
scores 0.183 / 0.182. STATE absent won. The **research plan** agreed on 2026-10-11 (blueprint §5) sets the bar at the default prior, not
the gene mean, and asks first whether single-cell resolution adds to pseudo-bulk at all: the **single-cell statistics
stack** (protocol §13, `configs/context_prior/single_cell_stack.yaml`, `hpc/run.sh prior-stack`) stacks per-gene
detected fraction and variance on the prior's export against a mean-only arm. It ran on 2026-10-11
(`results/single_cell_stack_seed0/`): no single-cell arm beats the mean arm on validation (largest +0.0011, interval
spanning zero) and no arm's best row clears the prior (best 0.2340 against 0.2290, interval spanning zero), so the
distributional and AI-virtual-cell layer (the second-wave plan) has no single-cell statistic to start from. A reproduction of Cilantro-SL (Hua et al. 2026), then its
port to the context screen and basal single cells, is designed (`docs/specs/2026-10-08-cilantro-sl-reproduction-design.md`);
phase 1's as-released run is complete (`results/cilantro_reproduction_seed0/`): stage one reproduces but equals a
gene mean from other lines, labels are one per pair in every line, and gene-holdout two-class AP is 0.54 against the
paper's 0.71; the port is not built. The **SL benchmark** (chosen 2026-10-11, blueprint §5, SL protocol) is the
screen set of Kebabci et al. 2026: the Ito, Klingbeil and Harle multi-line screens, labelled by Kebabci's rule (Harle
by its own call), trained pooled and scored within one screen, with a single-cell headline and a bulk-read
complement. It needs one re-version of the 226 split, moving A549 to test, before any further stage-one work; the
default prior and the single-cell stack are then rerun on it. Neither the new split nor a data card or builder exists
yet. The SL pair head is **unimplemented**. One config is one experiment at seed 0: train, select `best.pt` on
**validation**, then score it once on test; there is no multi-seed stage. Nothing here is SL evidence.

## Commands

Python 3.11–3.12, managed by `uv`. Every Python invocation goes through `uv run`, from the repo root, as a module (`-m`);
imports are `src.*`.

```bash
uv sync                                                   # core deps + dev group (pytest, ruff, xgboost); scib/datasets are optional extras
uv run python -m pytest tests -q > pytest.txt 2>&1       # full suite; redirect — see rtk note
uv run python -m pytest tests/test_all.py -q             # one file; -k for one test
.venv/bin/ruff check .                                    # lint (E,W,F only; import order not enforced)
.venv/bin/ruff format <files you touched>                 # never `ruff format .` — rewrites unrelated files

hpc/run.sh all configs/geneeffect_joint.yaml [--run-id <id>] [--gpus 0,1,2,3]  # prepare, comparison, train, val eval, baselines, readout, summary.md
hpc/run.sh revision CONFIG [--run-id <id>] [--gpus 0,1,2,3]   # one variant: train, val then test eval with baselines, summary.md + revision.json in outputs/geneeffect_revision/<id>/ (configs/revision/*.yaml; no comparison, no readout)
hpc/run.sh test outputs/geneeffect_joint/<id>/train/best.pt   # test for any checkpoint; `all` never runs test, `revision` scores its own best.pt
hpc/run.sh prior CONFIG [--run-id ID] [--experiments A,B]   # linear context prior, CPU: bridge remedies scored on validation and test; rows/, results.md in outputs/context_prior/<id>/
hpc/run.sh prior-selected CONFIG --run-id ID                # the reference row's data-selected genes, CPU: data_selected/ in the same run directory
hpc/run.sh prior-export CONFIG --run-id ID                  # the reference row for the correction, CPU: export/ (val/test, training lines out of fold); never overwritten
hpc/run.sh prior-stack CONFIG --run-id ID                   # single-cell q_sc statistics stacked on a prior export, CPU: single_cell_stack/ (protocol §13)
uv run python -m src.experiments.compare_runs RUN_A RUN_B --split val   # paired line bootstrap of selective Spearman, A minus B
uv run python -m src.evaluate --checkpoint <best.pt> --split val   # --split train for checkpoint diagnostics
```

`all` and `revision` skip every step whose output exists, so rerunning with the same run id resumes (training from `train/last.pt`). Every
GPU step uses every visible GPU, or those `--gpus` lists: untrained comparison arms on the first, training on all, then one
trained comparison job per GPU at a time (`revision` has only the training step). Direct worker invocation is debug-only; runner
details are in `hpc/README.md`.

- **Local Mac has no GPU, no Tx1 weights, no raw data.** Preparation, training and evaluation run on the H20 container
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
  recording `expression_space` under `data/geneeffect_joint/v2`; training opens those caches and never rebuilds them.
- **Training** (`src/experiments/geneeffect.py:run_training` → `src/training/trainer.py`): STATE (`arc-state`, pinned commit)
  reads basal cells through its released encoder, driven by an ESM2 adapter's perturbation token; frozen Tx1 embeddings feed
  the head's context. The head is a rank-`factor_rank` product of a gene factor and a line factor read from the
  128-component context PCA (fitted on training lines), plus a per-(g, c) correction that never sees the context, and predicts in units of the
  per-gene training residual SD. One GeneEffect objective per update, `train.objective` (`huber`, `standardized_mse`,
  `pearson_blocks`); response replay on the four anchor lines every fourth update only when `response_weight > 0` (the base config
  has 0). `train.state_mode` freezes STATE at the released weights or trains it; no STATE is `head_blocks` `use_delta_proj` and
  `use_s` both false, which skips STATE and the adapter. Warmup then cosine learning rate. `best.pt` and early stopping use
  **only** `val_selective_spearman` (higher wins), over the selective genes fitted on training lines.
- **Evaluation** (`src/experiments/geneeffect.py:evaluate_checkpoint`) restores fitted preprocessing (gene means, variable and
  selective genes, residual SD, context PCA) from the checkpoint;
  `src/baselines/residual.py` fits the control ladder (gene-mean, copy-prior, nearest-line, context-PCA-ridge) on train only;
  `src/experiments/all.py` chains every step and writes `summary.md`; `src/experiments/revision.py` chains training, validation and test
  evaluation and baselines for one config.
- `configs/benchmarks/cell_line_geneeffect_226_split.json` is the sole membership authority (171 train / 27 val / 28 test; A549 moved to test on 2026-10-11);
  `src/data/splits.py:assert_fit_eligible` guards every fit. Only split, config, provenance files and the context prior's reference tables are tracked data.

## Pitfalls and claim boundaries

Mistakes here produce a complete-looking **wrong artifact**, not an exception.

- `src/experiments/config.py` raises on unknown or missing keys; never add `.get(key, default)` config reads.
- `load_inputs` refuses manifests without `expression_space`, and checkpoint loads raise on zero or mismatched keys
  (`load_released_state`). Never bypass either or add a bare `load_state_dict(..., strict=False)`.
- Resume rejects a conflicting config; a batch-size change needs a new run id.
- Residuals: targets center on the fold-fit `mu_hat_g^(-c)`, predictions on fold-independent `mu_bar` — centering a prediction
  on `mu_hat` scores Spearman +1.0 by construction. Gene-mean and copy-prior residual correlations are undefined, not zero.
- Join cell lines by DepMap ModelID through the checked-in map, never informal names (`K-562` ≠ `K562`). Fit means, gene
  membership and normalization on labeled training lines only.
- Context claims need residual evaluation against context-blind priors; the rules are in the GeneEffect protocol (§11) and the
  SL protocol (§8). Single-gene predictions are not SL or genetic-interaction evidence.
