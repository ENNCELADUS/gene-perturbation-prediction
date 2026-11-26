# CLAUDE.md

## Research task and document authority

The task is context-specific synthetic-lethality (SL) prediction in held-out **cell lines**, read from their basal
profiles with a single-cell foundation model: stage one predicts DepMap GeneEffect, stage two ranks SL partners against
experimental SL-database labels (`docs/01-blueprint.md` §3). **The goal is SL performance in held-out lines.** Stage one is
a means: its accuracy is no goal and no bar for SL work. Report SL first, then the stage-one numbers of the reader behind
it. Spend effort on what separates this work from Cilantro-SL (one score per pair in every line, no line held out): a
score that changes with the cell line and holds in unseen lines, not only a different foundation model.

`docs/` outranks this file. Start at `docs/01-blueprint.md` (introduction, related work and task formulation, no modelling);
`docs/02-literature-review.md` is the extended related work; `docs/03-geneeffect-protocol.md` is the executable protocol of the implemented
GeneEffect track (its §11 holds the rules) and `docs/04-sl-ranking-protocol.md` the separate SL-pair protocol built on it (its §8 holds the rules; superseded in part by the 2026-10-09 blueprint). The current expression space,
STATE wiring and `all` run follow `docs/specs/2026-10-02-expression-space-and-all-pipeline-design.md`. Read
the `docs/data/` card before using a dataset. Results live in `results/`, on the local Mac only (untracked since 2026-11-25).

Name every model, head, arm, variant, run and stage by what it is ("shared MLP head", "the seed-0 joint backbone"), never by a
bare internal label (`A0`, `V1`, `Tier 3`). Where code uses a label, give it once in parentheses at first use.

## Current state (2026-11-26)

Replace this section on every update; history lives in `results/` and the protocol.

- **Stage two**: the gene-knowledge, cell-set model with stage one on GeneEffect itself (SL protocol §19, [record](results/sl_line_term_seed0/summary.html), [plan](docs/specs/2026-11-20-line-term-and-direct-target-plan.md), `hpc/run.sh sl-embedding`, 2026-11-20) scores 0.553 / 0.496 (val / test) with Tx1 cell tokens and 0.545 / 0.505 without, above every published baseline (against Kebabci +0.198 [+0.121, +0.274] / +0.102 [+0.054, +0.149]). Stage one now keeps a trained epoch (loss 1.022 untrained, about 0.20 trained), and the head on the frozen pretrained model scores 0.425 on validation (0.109 before). Against the residual-target runs of 2026-11-19 (§18, [record](results/sl_embedding_seed0/summary.html): 0.500 / 0.485, and 0.508 / 0.501 without cell tokens; their early stop kept the untrained model) the gain holds on validation only: +0.052 [+0.019, +0.088] / +0.011 [−0.027, +0.055]. The score now changes with the line: the model minus the same weights on the mean input +0.088 [+0.049, +0.128] / +0.067 [+0.015, +0.119]; minus another line's input +0.045 [+0.007, +0.083] / +0.012 [−0.007, +0.030] (without cell tokens +0.062 [+0.032, +0.099] / +0.032 [+0.005, +0.067]). Null: Tx1 cell tokens (+0.008 / −0.009, intervals across zero) and the line term (a line head on gene-in-line features, added to the model's score as a deviation from its mean over the training lines): own expression +0.001 / +0.001, PIE change +0.002 / +0.005, Tx1-3B gene embeddings −0.046 / −0.012 (it fits the training lines). By row set: seen pairs of unseen lines 0.560 / 0.520, new pairs 0.548 / 0.518, seen pairs that no training line called SL 0.443 / 0.330. One seed. Caches of gene-in-line features: `hpc/run.sh sl-gene-in-line` ([card](docs/data/gene-in-line-features.md)); the Tx1-3B caches of ACH-000696 (versioned Ensembl IDs, padding tokens only) were encoded again on 2026-11-23; the runs trained again on them (`_seed0_v3`) change no conclusion (model 0.553 / 0.499, each difference within ±0.02; record Table 11). The cross-line loss study ran on 2026-11-24 (SL protocol §20, [record](results/sl_cross_line_seed0/summary.html), [design](docs/specs/2026-11-24-context-specific-sl-architecture-design.md), [plan](docs/specs/2026-11-24-cross-line-loss-plan.md); five runs `configs/sl/cross_line_*.yaml`). Rows: the cross-line ranking loss (two training lines of one study, the same pair) with the line term trained last 0.555 / 0.514; line term trained with the model 0.528 / 0.510; line head on the frozen-source gene vectors 0.533 / 0.512; plus wide expression and buffers 0.525 / 0.495; plus DepMap mutations and copy number 0.540 / 0.524. No gain in recall over the model before the loss (0.549 / 0.505; +0.006 [−0.023, +0.035] / +0.008 [−0.027, +0.045]). Own input minus another line's input (mean over four lines) is positive in every run (+0.027 to +0.052 / +0.026 to +0.062; model before the loss +0.060 / +0.042); the line term adds at most +0.011. The cross-line ordering (0.5 = a score without the line) is 0.74 to 0.99 on the training lines and 0.50 to 0.55 on held-out lines, and no run is above 0.5 on validation and on test (best validation 0.553 [0.519, 0.589] / 0.521 [0.474, 0.568]; DepMap row 0.499 / 0.552; a wrong-input control reaches 0.572 on test): the loss fits (line, pair) on the 44 training lines and does not carry to unseen lines. Three more rows on the wide row, each one change (SL protocol §21, [record](results/sl_context_rows_seed0/summary.html), harness run `benchmark_context_rows_seed0`, 2026-11-26): the wide row on corrected caches (silenced flag on every measured gene) 0.511 / 0.488; plus 37 bulk-RNA training lines 0.525 / 0.498; wide mean token (about 18,300 genes) 0.528 / 0.497; pair-in-line features 0.504 / 0.493. Each change minus its base is within ±0.017, intervals across zero. Cross-line ordering on held-out lines: the corrected wide row 0.554 [0.521, 0.585] / 0.595 [0.551, 0.635] is the first row above 0.5 on both splits, but wrong-input controls reach 0.423 to 0.615 and the cache fix alone moves it by +0.03 / +0.05, so the interval over gene pairs is too narrow and the ordering in unseen lines is not shown. The replicate ceiling inside Harle 2025: a second replicate orders unseen lines at 0.602 / 0.709 (all lines 0.695); the rows reach 0.37 to 0.57 there. Harness metrics since 2026-11-25 (run `benchmark_cross_line_seed0_v2`, report Table 7): the cross-line ordering of every scorer, recall in the top 10%, recall at 10% false positives (random 0.10 in every screen), AUROC and the chance-corrected AUPRC, (AUPRC − positive rate) / (1 − positive rate), in place of AUPRC over the positive rate. On test the cross-line loss model scores 0.521 / 0.514 / 0.540 / 0.781 / 0.304 in that order (Kebabci on predicted GeneEffect 0.491 / 0.395 / 0.383 / 0.664 / 0.240). No model orders a pair between two unseen lines above 0.5 beyond its interval (ours 0.521 to 0.533; a random score 0.517 [0.470, 0.563]); the Kebabci classifier on the line's measured GeneEffect does (0.645 [0.612, 0.673]).
  Earlier: modular SL backbone, buffer-aware head 0.394 / 0.361 < training SL fraction. Three modules of the [plan](docs/specs/2026-11-12-pie-and-single-cell-plan.md) ran, all null: PIE's profile as the editor (SL protocol §14, [record](results/sl_pie_editor_seed0/README.md); below the simple edit, equal to another line's profile; 66 of 86 lines had no context text),
  own-gene space and co-detection (§13, [record](results/sl_single_cell_inputs_seed0/README.md)). The labels hold a small line-specific signal (§15).
- **Multi-study SL benchmark** ([card](docs/data/sl-benchmark.html), [model status](model-status.html)): labels in `data/sl_studies/benchmark_v2` (H20);
  headline = recall in the top 10% of genetic two-class screens, over min(SL pairs, top places) so the ceiling is 1,
  averaged over screens, studies, then lines (77 lines, 44/17/16, placed by hand). 2026-11-13 (run `_v3`), val / test:
  the training SL fraction 0.540 / 0.541 leads; Kebabci on predicted GeneEffect 0.355 / 0.395, KG4SL 0.367 / 0.338,
  De Kegel 0.281 / 0.293, PARIS 0.219 / 0.213, MiT4SL 0.154 / 0.138, Cilantro-SL 0.123 / 0.134, SLIdR 0.095 / 0.115;
  none follows context; labels 1.00 / 1.00, random 0.10 / 0.12.
- **Line-specific signal in the SL labels** (SL protocol §15, §16, no model): between two studies the line-specific part of a pair's score repeats at 0.04–0.12; inside Harle 2025 two replicates repeat at 0.35–0.39. Of four target options (run `target_options_seed0_v2`), the top 10% of the rank as a continuous target repeats best (0.392 against 0.348 for the SL call, 0.178 for the plain rank); the mean of measurements helps but two studies share 0.7% of training cells; error weights and a simple common score definition do not help. The logistic head trained on that target (§17) equals the head trained on the calls: 0.389 / 0.367, difference −0.005 [−0.014, +0.004] / +0.005 [−0.004, +0.018]; neither follows the line, because the head has one weight for each input over all pairs.
- **Stage one**, the reader behind the SL model (protocol §16, §18): no arm beats the default linear model refitted on the frame (per-line residual
  Spearman 0.226 validation / 0.227 test on the frame rebuilt 2026-11-13; two screens of one line agree at 0.246). The
  per-gene reader (linear model + `u_g · h(line)`) is null: 7 of 8 runs select epoch 0, the best gains +0.002 / +0.005
  ([record](results/stage_one_gene_reader/README.md)); older arms: [architecture](results/stage_one_architecture/README.md),
  [linear-model input](results/stage_one_linear_model_input/README.md). Outputs before 2026-11-13 are gone from the host.
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
  `README.md` in every conversation: replace old text, and keep both short (this file under 120 lines). When a model design finishes its run, update `model-status.html` (its design, figure and SL-benchmark scores).
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
- Codex review is optional and runs **only when the user asks**. Then run it from Bash (the slash commands are
  `disable-model-invocation`) in a fresh `CODEX_HOME` holding only `auth.json` and a `config.toml` with `model = "gpt-6.1-sol"`,
  `model_reasoning_effort = "high"`, `approval_policy = "never"` — recreate it per review, or Codex loads MCP servers and hangs.
  Background `codex review --base <sha>` to a file (~200 KB) and adjudicate the findings against the code.

## Architecture rules

- `data` and `model` never import `training`, `eval` or `experiments`. Each track's commands live in
  `src/experiments/{joint,linear_model,sl,stage_one,states}`; retired code is removed (cite `git show <sha>:<path>`). Tests mirror `src`; shared synthetic
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