# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

`AGENTS.md` (imported above) is the shared research contract for Claude and Codex: task framing, document authority, data
and claim boundaries. Edit it there, once, and keep the tracked `.codex/` agents in step with it. This file adds what Claude
Code needs beyond that contract. `docs/` outranks both — start at `docs/01-blueprint.md`; invoke the `research-vault` skill
before editing anything under `docs/`.

## Current state (2026-09-07)

The joint GeneEffect model (`configs/geneeffect_joint.yaml`) has **trained, tested and been baselined** once, seed 0. Joint
Huber beats the context-blind gene-mean by 0.08% and residual correlations trail a Tx1 PCA-ridge baseline — a working path, not a
result (`docs/01-blueprint.md` §7, `docs/results/joint_geneeffect_seed0/`). The SL pair head in `docs/04-sl-ranking-protocol.md`
is **unimplemented**. Further model decisions use **validation**; the test split is spent. Nothing here is SL evidence.

## Commands

Every Python invocation goes through `uv run`, from the repo root, as a module (`-m`); imports are `src.*`.

```bash
uv sync                                                   # core deps + dev group (pytest, ruff, xgboost)
uv run python -m pytest tests -q > pytest.txt 2>&1       # full suite (~50 s); redirect — see rtk note
uv run python -m pytest tests/test_joint_training.py -q  # one file; -k for one test
.venv/bin/ruff check .                                    # lint (E,W,F only; import order not enforced)
.venv/bin/ruff format <files you touched>                 # never `ruff format .` — rewrites unrelated files

hpc/run.sh prepare configs/geneeffect_joint.yaml                       # fixed inputs, once, single process
hpc/run.sh train   configs/geneeffect_joint.yaml --run-id <new_id>     # accelerate launch on visible GPUs
hpc/run.sh train   configs/geneeffect_joint.yaml --resume outputs/geneeffect_joint/<id>/last.pt
hpc/run.sh test    outputs/geneeffect_joint/<id>/best.pt               # explicit; training never runs test
uv run python -m src.evaluate --checkpoint <best.pt> --split val
uv run python -m src.experiments.baselines --config configs/geneeffect_joint.yaml --split val --out-dir <dir>
```

- **Local Mac has no GPU, no Tx1 weights, no raw data.** Training, preparation and evaluation run on the H20 host
  (`hpc-execution` skill, `hpc/README.md`). Sync code via git push/pull only. Tests use synthetic fixtures and skip silently
  when gitignored data or `accelerate` is missing — an "it imports" check proves nothing about an asset-dependent path.
- A global `rtk` hook rewrites Bash output: `ruff check .` prints `[]`, foreground pytest shows fake collection errors. Use
  `.venv/bin/ruff` and redirect pytest to a file. `rtk proxy <cmd>` when exact output matters.
- **Baseline the suite before changing code.** At `f988109`, 568 pass and 2 fail in `tests/test_geneeffect_sampler.py`
  (fake accelerator lacks `.device`, introduced at `bbe4670`) — pre-existing, not yours.
- `tests/conftest.py` is load-bearing: it sets `PYTORCH_ENABLE_MPS_FALLBACK` and `OMP_NUM_THREADS` and imports xgboost before
  torch. Running a test file as a script segfaults. `-m` markers do nothing.
- Commits use Conventional Commits (`feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`).

## Architecture

One route: **prepare → train → evaluate**, all driven by one strict YAML config.

- `src/experiments/prepare.py` builds fixed inputs under `prepared_root` (`data/geneeffect_joint/v1`): Tx1 basal-embedding
  cache, q_sc cache, ESM2 table, response cache with gene order, common gene panel. Preparation is the only place raw data is
  read; **training opens caches and never rebuilds them** — a missing cache is an error, not a trigger.
- `src/train.py` → `src/experiments/geneeffect.py:run_training` → `src/training/trainer.py`. Frozen Tx1 embeddings feed a STATE
  transition model (`arc-state`, pinned to a git commit) plus ESM2 adapter and residual head (`src/model/`). Every update is a
  GeneEffect Huber step; every fourth update replays response reconstruction on four anchor lines (`src/training/sampling.py`).
  One validation per epoch; `best.pt` and early stopping use **only** `val_geneeffect_loss`. Seeds are fixed at 0/0/0.
- `src/evaluate.py` → `evaluate_checkpoint` restores fitted preprocessing from the checkpoint, exports
  `evaluation/<ckpt>/<split>/{predictions.parquet,metrics.json,per_line.csv,per_gene.csv,response.csv}`.
  `src/baselines/residual.py` fits the control ladder (gene-mean, copy-prior, nearest-line, context-PCA-ridge) on train only.
- Import direction: `data` and `model` never import `training`, `eval` or `experiments`. `data.batches` and `data.gene_bags`
  own shared records; `model.response` holds pure response functions so features don't depend on a trainer.
- `src/data/prepare/` = retained preparation CLIs (split builder, Tx1 cache, ESM2, atlas raw-UMI). `src/experiments/historical/`
  = the `context_screen_v2` SL benchmark builder and finished diagnostics. Old staged Exp13 runtime lives at git `e6341d2`;
  retired dependency/Feng2024 code is in gitignored `archive/`.
- `configs/benchmarks/cell_line_geneeffect_226_split.json` is the sole membership authority (172 train / 27 val / 27 test lines;
  170 train lines have labels). `src/data/splits.py:assert_fit_eligible` guards every fit. The split builder SHA-256-pins the
  Phase-A manifest under `configs/benchmarks/provenance/`. Split, config and provenance files are the only tracked "data";
  `data/`, `outputs/`, `*.pt`, `*.csv`, `*.parquet`, `*.h5ad` are gitignored, with explicit `!` exceptions in `.gitignore`.

## Silent failures — the dominant risk

Mistakes here tend to produce a complete-looking **wrong artifact**, not an exception.

- `src/experiments/config.py` **raises** on unknown, missing or out-of-domain keys and hard-codes protocol constants
  (`selection`, seeds, `hvg_dim`, holdout seeds). Keep it that way; do not add `.get(key, default)` config reads.
  (`src/training/sampling.py` still defaults `dependency_batch_size` via `.get` — the config validator is what protects it.)
- Checkpoint loads are `strict=False` by design (`src/model/initialization.py:warm_start_state_dict`,
  `src/model/tx1.py:validate_load_result`) but **must** return a report and raise on zero loaded keys. Never add a bare
  `load_state_dict(..., strict=False)`.
- Resume rejects a config that conflicts with the checkpoint's; a batch-size change needs a **new run id** and a documented derived
  checkpoint (`hpc/README.md`). Seed-0 epochs 1–2 ran at batch 256 and 3–8 at 1024 — a mixed continuation, not an ablation.
- Residual metrics: targets center on the fold-fit `mu_hat_g^(-c)`, predictions on fold-independent `mu_bar`. Centering a
  prediction on `mu_hat` scores Spearman +1.0 by construction. Gene-mean and copy-prior have **undefined** residual
  correlations (constant per gene across contexts) — not zero.
- Tx1 reads raw UMI, not CPM (measured in `docs/results/exp13_stage0/`); response holdout is 10%/seed 13 and sampling seed 42 at
  preparation time, distinct from the runtime 0/0/0 seeds.

## After each implementation wave

**You run the Codex review, not the user.** Both `/codex:review` and `/codex:adversarial-review` are `disable-model-invocation`,
which blocks the slash commands only; call the CLI or the plugin runtime from Bash. Output can reach ~200 KB — background it to a
file and read the findings when it finishes. Findings are input to adjudicate against the code, not patches to apply blindly.

```bash
CH=<scratch>/codex-home; mkdir -p "$CH"; cp ~/.codex/auth.json "$CH/"
printf 'model = "gpt-6-astra"\nmodel_reasoning_effort = "medium"\napproval_policy = "never"\nsandbox_mode = "workspace-write"\n' > "$CH/config.toml"
CODEX_HOME="$CH" codex review --base <WAVE_BASE_SHA> > <wave>-review.txt 2>&1              # built-in reviewer over the wave diff
CODEX_HOME="$CH" node ~/.claude/plugins/cache/openai-codex/codex/<ver>/scripts/codex-companion.mjs \
  adversarial-review --background --base <WAVE_BASE_SHA> "<focus text>"                       # design challenge; shows in /codex:status
```

- The **clean `CODEX_HOME`** is load-bearing: `-c 'mcp_servers={}'` merges rather than replaces, so the `[mcp_servers.*]` tables
  in `~/.codex/config.toml` survive and the review hangs on `gitnexus/detect_changes` with zero output growth. That home also
  carries model and effort, so no `-c` flags. Stall signature: file size stops growing after an `mcp: … started` line with no
  `(completed)` — kill and relaunch.
- Status as of 2026-09-07: codex-cli 0.150.1, plugin 1.0.6, no jobs in `/codex:status`, stop-review gate off. The CLI's
  `codex review` now takes a positional prompt plus `--base/--commit/--uncommitted`; the plugin's `review` subcommand still
  rejects focus text, so a targeted review goes through `adversarial-review`. Focus text may name a file outside the repo.
