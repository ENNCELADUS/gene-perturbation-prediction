# CLAUDE.md

## Research task and document authority

The task is context-specific synthetic-lethality (SL) prediction in held-out **cell lines**, read from their basal
profiles with a single-cell foundation model: stage one predicts DepMap GeneEffect, stage two ranks SL partners against
experimental SL-database labels (`docs/01-blueprint.md` §3). **The goal is SL performance in held-out lines.** Stage one is
a means: its accuracy is no goal and no bar for SL work. Report SL first, then the stage-one numbers of the reader behind
it. Spend effort on what separates this work from Cilantro-SL (one score per pair in every line, no line held out): a
score that changes with the cell line and holds in unseen lines, not only a different foundation model.

`docs/` outranks this file. Start at `docs/01-blueprint.md` (introduction, related work, task; no modelling); `docs/02-literature-review.md`
is the extended related work; `docs/03-geneeffect-protocol.md` (rules §11) and `docs/04-sl-ranking-protocol.md` (rules §8; superseded in
part by the 2026-10-09 blueprint) are the executable protocols. Expression space, STATE wiring and `all` follow
`docs/specs/2026-10-02-expression-space-and-all-pipeline-design.md`. Read the `docs/data/` card before using a dataset. Results live in `results/`, local Mac only.

Name every model, head, arm, variant, run and stage by what it is ("shared MLP head", "the seed-0 joint backbone"), never by a
bare internal label (`A0`, `V1`, `Tier 3`). Where code uses a label, give it once in parentheses at first use.

## Current state (2026-11-26)

- **Best SL model**: the gene-knowledge, cell-set model with stage one on GeneEffect itself (SL protocol §19, [record](results/sl_line_term_seed0/summary.html), `hpc/run.sh sl-embedding`). Recall in the top 10%: 0.553 / 0.496 (val / test), above every published baseline (best: Kebabci on predicted GeneEffect, 0.355 / 0.395; [model status](model-status.html)). Its score changes with the line: own input minus another line's input +0.045 [+0.007, +0.083] / +0.012 [−0.007, +0.030]. One seed.
- **Open problem**: no model orders one pair between two unseen lines beyond chance (0.5 = a score without the line). The cross-line ranking loss and the wide-row changes fit the 44 training lines and do not carry to held-out lines (SL protocol §20–§21, [record](results/sl_cross_line_seed0/summary.html), [record](results/sl_context_rows_seed0/summary.html)); only the Kebabci classifier on the line's measured GeneEffect does (0.645 [0.612, 0.673]). Ceiling: a second Harle 2025 replicate, 0.602 / 0.709.
- **Benchmark** ([card](docs/data/sl-benchmark.html)): labels in `data/sl_studies/benchmark_v2` (H20); 77 lines, 44/17/16. Headline = recall in the top 10% of genetic two-class screens (ceiling 1), averaged over screens, studies, then lines. The harness also reports cross-line ordering, recall at 10% false positives, AUROC and the chance-corrected AUPRC.
- **Stage one**: no reader beats the linear model refitted on the frame (per-line residual Spearman 0.226 / 0.227; protocol §16, §18).
- One config is one experiment at seed 0: train, select `best.pt` on **validation**, score it once on test. Since 2026-11-26 the phase of stage two, and a design's version on `model-status.html`, is the most line-specific candidate (validation own minus another line's input) within one standard error of the best validation recall (SL protocol §8); the eleven harness runs were trained again under it (`_seed0_select`, harness `benchmark_context_rows_seed0_select`, `results/sl_selection_seed0/`): every run keeps the same phase; the cross-line version is now the line term trained with the model (test recall 0.524, ordering 0.553 [0.510, 0.596], 0.468 on validation); the ordering of one config moves by up to 0.08 between trainings.

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
  `README.md` in every conversation: replace old text (the current state holds the latest status only; history lives in
  `results/` and the protocols), and keep both short (this file under 100 lines). When a model design finishes its run, update `model-status.html` (its design, figure and SL-benchmark scores; test metrics only, validation stays in the run record).
- Experiment code reports results (format: `.claude/skills/results-report`); decisions are made by reading them, not in code.
- No formalism gates: do not add digest pinning, contract verifiers, eligibility ceremony or any check that blocks a run on
  model quality. Record provenance and proceed; model-quality signals are telemetry. The existing guards against silent wrong
  artifacts (below) fail closed and stay.
- GPU work needs the H20 container: finish everything local first (config, tests, commit, push) and end with the exact
  `hpc/run.sh` launch command.
- Sync code between machines through Git only: commit, push, then run `hpc/pull.sh` in the H20 checkout (HTTPS with retries, no agent forwarding). Never rsync/scp/tar working trees.
- Work done on its own branch or worktree is merged into `main` as soon as the feature is established as working — suite and
  lint pass, any requested review is adjudicated, and an asset-dependent path has run on the H20 host — then `main` is pushed and the branch
  deleted. Do not leave finished work on a side branch. These pushes of `main`, and the push before GPU work, are approved.
- Commits use Conventional Commits (`feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`).
- Codex review runs **only when the user asks**, from Bash (the slash commands are `disable-model-invocation`), in a fresh
  `CODEX_HOME` with only `auth.json` and a `config.toml` (`model = "gpt-6.1-sol"`, `model_reasoning_effort = "high"`,
  `approval_policy = "never"`; recreate it per review, or Codex loads MCP servers and hangs). Background `codex review --base <sha>` to a file and adjudicate the findings against the code.

## Architecture rules

- `data` and `model` never import `training`, `eval` or `experiments`. Each track's commands live in `src/experiments/{joint,linear_model,sl,stage_one,states}`;
  retired code is removed (cite `git show <sha>:<path>`). Tests mirror `src`; shared synthetic worlds live in `tests/fixtures/`.
- **Expression space.** Everything except Tx1's input is log space: whole-library `normalize_total` to a recorded target `T`,
  `log1p`, then STATE's HVG slice. Tx1 reads raw UMI, not CPM.
- **Preparation** (`src/experiments/prepare.py`) is the only place raw data is read. It writes fixed caches and a manifest
  recording `expression_space` under `data/geneeffect_joint/v2`; training opens those caches and never rebuilds them. An
  existing root grows by the lines it lacks, in the recorded space.
- `configs/benchmarks/cell_line_geneeffect_split.json` (the **final split**, [card](docs/data/cell-line-geneeffect-split.md)) is
  the sole membership authority (line counts in its card; bulk-only lines join the linear model's training side through bulk RNA).
  Gene means, selective genes, residual SD and context PCA fit on its labelled training lines; `src/data/splits.py:assert_fit_eligible` guards every fit.
- Checkpoint selection reads validation only: `val_selective_spearman` for the joint model, the per-line residual Spearman
  for stage one (protocol §6). `all` never runs test; `revision` and `test` score a chosen `best.pt` once.
- Only split, config, provenance files and the linear model's reference tables are tracked data.

## Pitfalls and claim boundaries

Mistakes here produce a complete-looking **wrong artifact**, not an exception.

- One null run with new data or a new foundation model does not close that route. Report it, find why it did not help
  (input, reader, training, scale), and test the next change on the same route.
- Never weaken the fail-closed loads: no `.get(key, default)` config reads, no bypass of the `expression_space` manifest
  check, no bare `load_state_dict(..., strict=False)`. Use a checkpoint only with the split it was trained on.
- Residuals: targets center on the fold-fit `mu_hat_g^(-c)`, predictions on `mu_bar`; centering a prediction on `mu_hat`
  scores Spearman +1.0 by construction.
- Join cell lines by DepMap ModelID (or the reviewed ID in `configs/benchmarks/sl_line_identities.csv`), never by names
  (`K-562` ≠ `K562`). Fit means, gene membership and normalization on labelled training lines only.
- Context claims need residual evaluation against context-blind baselines (GeneEffect protocol §11, SL protocol §8).
  Single-gene predictions are not SL evidence.