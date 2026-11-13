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

## Current state (2026-11-13)

Replace this section on every update; history lives in `results/` and the protocol.

- **Stage one** (protocol §16, §18): no arm beats the default linear model refitted on the frame (per-line residual
  Spearman 0.226 validation / 0.227 test on the frame rebuilt 2026-11-13; two screens of one line agree at 0.246). The
  per-gene reader (linear model + `u_g · h(line)`) is null: 7 of 8 runs select epoch 0, the best gains +0.002 / +0.005
  ([record](results/stage_one_gene_reader/README.md)); older arms: [architecture](results/stage_one_architecture/README.md),
  [linear-model input](results/stage_one_linear_model_input/README.md). Outputs before 2026-11-13 are gone from the host.
- **Stage two**: modular SL backbone, buffer-aware head 0.394 / 0.361 < training SL fraction. Three modules of the [plan](docs/specs/2026-11-12-pie-and-single-cell-plan.md) ran, all null: PIE's profile as the editor (SL protocol §14, [record](results/sl_pie_editor_seed0/README.md); below the simple edit, equal to another line's profile; 66 of 86 lines had no context text),
  own-gene space and co-detection (§13, [record](results/sl_single_cell_inputs_seed0/README.md)). Next: test if the measured profile after A's loss holds SL signal under a fitted reader.
- **Multi-study SL benchmark** ([card](docs/data/sl-benchmark.html), [scoreboard](docs/sl-benchmark-results.html)): labels in `data/sl_studies/benchmark_v2` (H20);
  headline = recall in the top 10% of genetic two-class screens, over min(SL pairs, top places) so the ceiling is 1,
  averaged over screens, studies, then lines (77 lines, 44/17/16, placed by hand). 2026-11-13 (run `_v3`), val / test:
  the training SL fraction 0.540 / 0.541 leads; Kebabci on predicted GeneEffect 0.355 / 0.395, KG4SL 0.367 / 0.338,
  De Kegel 0.281 / 0.293, PARIS 0.219 / 0.213, MiT4SL 0.154 / 0.138, Cilantro-SL 0.123 / 0.134, SLIdR 0.095 / 0.115;
  none follows context; labels 1.00 / 1.00, random 0.10 / 0.12.
- One config is one experiment at seed 0: train, select `best.pt` on **validation**, score it once on test.

## Commands

Python 3.11–3.12, managed by `uv`, run from the repo root as modules (`-m`); imports are `src.*`. Every `hpc/run.sh`
subcommand and `python -m` entry point is in `hpc/README.md`; the module map is `src/README.md`.

```bash
uv sync                                                   # core deps + dev group (pytest, ruff)
uv run python -m pytest tests -q > pytest.txt 2>&1       # full suite; redirect — see rtk note
uv run python -m pytest tests/experiments/joint/test_all.py -q  # one file; -k for one test
.venv/bin/ruff check .                                    # lint (E,W,F only; import order not enforced)
.venv/bin/ruff format <files you touched>                 # never `ruff format .` — rewrites unrelated files
```

- **Local Mac has no GPU, no model weights, no raw data.** Preparation, training and evaluation run on the H20 container
  (`hpc-execution` skill), without a scheduler. Tests use synthetic fixtures and skip silently when gitignored data or
  `accelerate` is missing — an "it imports" check proves nothing about an asset-dependent path. The vendored PIE's
  tests skip unless `PYTHONPATH=third_party/pie/src` is set.
- A global `rtk` hook rewrites Bash output: `ruff check .` prints `[]`, foreground pytest shows fake collection errors. Use
  `.venv/bin/ruff` and redirect pytest to a file. `rtk proxy <cmd>` when exact output matters.
- **Baseline the suite before changing code** (full run to a file) so a failure you cause is distinguishable from one you inherit.
- `tests/conftest.py` is load-bearing: it sets `PYTORCH_ENABLE_MPS_FALLBACK` and `OMP_NUM_THREADS` before torch loads.
  Running a test file as a script segfaults. `-m` markers do nothing.

## Project rules

- When a change alters behaviour a doc describes, update that doc in the same change. Update this file and
  `README.md` in every conversation: replace old text, and keep both short (this file under 120 lines).
- Experiment code reports results (format: `.claude/skills/results-report`); decisions are made by reading them, not in code.
- No formalism gates: do not add digest pinning, contract verifiers, eligibility ceremony or any check that blocks a run on
  model quality. Record provenance and proceed; model-quality signals are telemetry. The existing guards against silent wrong
  artifacts (below) fail closed and stay.
- GPU work needs the H20 container: finish everything local first (config, tests, commit, push) and end with the exact
  `hpc/run.sh` launch command.
- Sync code between machines through Git only: commit, push, then pull on the H20 checkout. Never rsync/scp/tar working trees.
- Work done on its own branch or worktree is merged into `main` as soon as the feature is established as working — suite and
  lint pass, any requested review is adjudicated, and an asset-dependent path has run on the H20 host — then `main` is pushed and the branch
  deleted. Do not leave finished work on a side branch. These pushes of `main`, and the push before GPU work, are approved.
- Commits use Conventional Commits (`feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`).
- Codex review is optional and runs **only when the user asks**. Then run it from Bash (the slash commands are
  `disable-model-invocation`) in a fresh `CODEX_HOME` holding only `auth.json` and a `config.toml` with `model = "gpt-6.1-sol"`,
  `model_reasoning_effort = "high"`, `approval_policy = "never"` — recreate it per review, or Codex loads MCP servers and hangs.
  Background `codex review --base <sha>` to a file (~200 KB) and adjudicate the findings against the code.

## Architecture rules

- `data` and `model` never import `training`, `eval` or `experiments`. Each track's commands live in
  `src/experiments/{joint,linear_model,stage_one,states}`, retired ones in `historical/`. Tests mirror `src`; shared synthetic
  worlds live in `tests/fixtures/`.
- **Expression space.** Everything except Tx1's input is log space: whole-library `normalize_total` to a recorded target `T`,
  `log1p`, then STATE's HVG slice. Tx1 reads raw UMI, not CPM.
- **Preparation** (`src/experiments/prepare.py`) is the only place raw data is read. It writes fixed caches and a manifest
  recording `expression_space` under `data/geneeffect_joint/v2`; training opens those caches and never rebuilds them. An
  existing root grows by the lines it lacks, in the recorded space.
- `configs/benchmarks/cell_line_geneeffect_split.json` (the **final split**, [card](docs/data/cell-line-geneeffect-split.md)) is
  the sole membership authority: 400 single-cell training lines (16 unlabelled), 27 validation, 28 test (A549 in train
  since 2026-11-04), 708 labelled and 570 unlabelled bulk-only lines that join the linear model's training side through bulk
  RNA, and 7 excluded (patient-mates of held-out lines without any data). Gene
  means, selective genes, residual SD and context PCA fit on its labelled training lines;
  `src/data/splits.py:assert_fit_eligible` guards every fit.
- Checkpoint selection reads validation only: `val_selective_spearman` for the joint model, the per-line residual Spearman
  for stage one (protocol §6). `all` never runs test; `revision` and `test` score a chosen `best.pt` once.
- Only split, config, provenance files and the linear model's reference tables are tracked data.

## Pitfalls and claim boundaries

Mistakes here produce a complete-looking **wrong artifact**, not an exception.

- `src/experiments/config.py` raises on unknown or missing keys; never add `.get(key, default)` config reads.
- `load_inputs` refuses manifests without `expression_space`, and checkpoint loads raise on zero or mismatched keys
  (`load_released_state`). Never bypass either or add a bare `load_state_dict(..., strict=False)`.
- Resume rejects a conflicting config; a batch-size change needs a new run id. A checkpoint does not record the split
  membership: use a checkpoint only with the split it was trained on.
- Residuals: targets center on the fold-fit `mu_hat_g^(-c)`, predictions on fold-independent `mu_bar` — centering a prediction
  on `mu_hat` scores Spearman +1.0 by construction. Gene-mean and copy-baseline residual correlations are undefined, not zero.
- Join cell lines by DepMap ModelID, never by informal names at run time (`K-562` ≠ `K562`): a source's spelling becomes a
  ModelID only through a reviewed row of `configs/benchmarks/single_cell_atlas_candidates.csv`. A line without a ModelID
  has the ID of its reviewed row in `configs/benchmarks/sl_line_identities.csv` (Cellosaurus accession, else
  `SL-<study>-<line>`); `read_models` adds those rows. Fit means, gene membership and normalization on labeled training
  lines only.
- Context claims need residual evaluation against context-blind baselines; the rules are in the GeneEffect protocol (§11) and the
  SL protocol (§8). Single-gene predictions are not SL or genetic-interaction evidence.
