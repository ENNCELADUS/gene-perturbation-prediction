# CLAUDE.md — guidance for Claude Code and Codex in this repository

## What this is

Research code for **generalizable synthetic-lethality discovery by virtual-cell composition**: rank SL gene pairs for genes withheld from
SL-pair/graph training (vs. SLMGAE / KR4SL), and separately transfer a context-conditioned score to cell lines excluded from training.
**No SL graph in the feature path.** `docs/` is the authority, not this file — start at `docs/01-blueprint.md`. Role: careful junior engineer;
**Plan → Confirm → Code**.

**State:** Active work is **Exp13**, the 226-line cell-line GeneEffect *residual* benchmark (`docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md`): contract
written, Stage 0 open, **no run started**. The `context_screen_v2` SL split is built and unrun. Feng2024 SOTA reproduction is not started.

## Engineering rules

- Keep documents clean and concise: replacement edits must preserve the line count or reduce it; never increase it.
- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility
  layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirement. Avoid speculative
  abstraction, configuration, and indirection.
- Grow the system in layers: start from the smallest version that works end to end, and add each
  capability on top of something that already works. Never trade a working product for unfinished
  complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce complexity or improve reliability.
  Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding
  packages; check a library's docs and types before assuming it lacks a capability.
- Make architectural decisions for the long term. No stopgap meant to be replaced later.

## Skills — load before touching the area

`research-vault` (editing `docs/`, writing a claim or a result) · `tx1-cache` (`tx1_basal.py`, `tx1_embed_cache.py`,
`build_tx1_basal_embeddings.py`) · `benchmark-harness` (`data/SL_benchmark/` splits, SL metrics, `vcc-dep-baseline`, the
`models:`/`selection:` config pattern) · `hpc-execution` (GPU, GWPS h5ad, ESM2, Tx1-3B, ST checkpoints).

## Environment and testing

- The five `src/` packages look peer-level but are not: **`aivc_model/` is the active backbone — new work goes there**; it holds the Exp13
  head, split guard and residual ladder, the Tx1 basal→embedding→response path, and forward-only STATE loading. The rest are benchmark,
  features, baselines. Most entrypoints need gitignored assets, so "it imports" ≠ "it runs here".
- `.superpowers/sdd/` is gitignored but holds the **live plan and execution ledger** (its `progress.md` lags the code); `docs/specs/` holds
  the tracked designs, both now describing retired T1/T2 work. **`AGENTS.md` is a symlink to `CLAUDE.md`** — edit `CLAUDE.md` only, and keep
  the tracked **`.codex/`** Codex setup (`config.toml`, five agents, four mirrored skills, its own `AGENTS.md`) in sync with it.
- Prefix every Python/pytest/ruff call with `uv run`. A global `rtk hook claude` hook **rewrites Bash output** (`ruff check .` prints `[]`)
  — call `.venv/bin/ruff` for real output. `uv sync` installs the `dev` group only (`scib`/`datasets` absent); a new `src/<pkg>/` is
  invisible until added to `[tool.hatch.build.targets.wheel] packages`; `arc-state` is pinned to a **git commit** (bumps are deliberate).
- **No pytest config**: no testpaths, no addopts, **no markers** (`-m` does nothing). Imports resolve only via the editable `.pth` — always
  `uv run python -m pytest` **from the repo root**. **`tests/conftest.py` is load-bearing**: it sets `PYTORCH_ENABLE_MPS_FALLBACK` and
  `OMP_NUM_THREADS` *before* torch imports, imports xgboost first, and covers the whole suite — **a test file run directly segfaults**.
- **The suite is green at `873c99c`** (497 passed, ~25 s); baseline it anyway, and note tests **skip silently** on missing gitignored data or
  `accelerate`. `ruff format .` rewrites two dozen unrelated files — **format only what you touched**; import order is **not** enforced (`E,W,F`).

## Silent failures — the dominant risk

Warnings and defaults are preferred over exceptions, so mistakes usually produce a complete-looking **wrong artifact**, not an error.

- **`dependency_baseline` config loading does no schema validation** — every field is `.get(key, default)`, so a **misspelled YAML key silently
  takes the default**; `ddgcn/config.py:81`, `sl_profile_baseline/config.py:73` and Exp13's `stage1_config.py` *raise* instead.
- The Tx1 cache zero-fills missing genes, reuses stale cells when the provenance sidecar is absent, and skips completeness checks on a sharded
  verify. The T2 evaluator that hash-pinned the Phase-A contract is gone; the surviving pin is `PINNED_SHA256` in the Exp13 split builder.
- **Checkpoint loads must be validated.** Every surviving loader mirrors `validate_load_result` (`scripts/verify_tx1_obsm_width.py:97`) — see
  `state_warm_start.py`, `tx1_predicted_response.py`. A bare `load_state_dict(..., strict=False)` silently leaves weights randomly initialized.
- **Residuals: never center a *prediction* on a fold-fit gene mean.** `mu_hat_g^(-c)` is an exact affine function of the held-out label, so
  `constant_g - mu_hat_g^(-c)` scores per-gene Spearman `+1.0` by construction. Targets use `mu_hat`, predictions the fold-independent `mu_bar`.

## Data

**Every dataset has a card in `docs/data/` — read it before using the file.** Several are not what their names suggest:
`jost_replogle_dual_sgrna` is knockdown efficacy, **not epistasis**; the two Replogle h5ads differ by one word (`essential`/`gwps`) but have
different gene panels; Horlbeck's field is `gi_score`, **not** `gamma`, and **negative = SL**; `sl_context_screen_v1` is **unsplit** with
screened-non-hit negatives — not universal non-SL, and no CV folds. **Never join cell lines on name** — DepMap rows are ACH ModelIDs and its
`CellLineName` is `K-562` while code says `K562`. `configs/benchmarks/cell_line_geneeffect_226_split.json` is the sole membership authority for
Exp13, guarded by `benchmark_split.assert_fit_eligible`. Large artifacts are gitignored except the tracked `configs/benchmarks/` splits and the four
`results/phase_a_tx1_20260724/` files — of which only `cell_line_manifest.csv` is still SHA-256-pinned (by the Exp13 split builder, which **raises**).

## Claim discipline

Full gates in `docs/01-blueprint.md` §7-8. The four broken most easily —

- **Never claim SL from single-gene essentiality** — an explicit interaction null is required: `interaction = joint - psi(singles)`; declare `psi`.
  DepMap GeneEffect is single-gene, never a double-knockout quantity. **Exp13 is scope-closed** (`01` §8): never SL evidence, never `context_screen_v2`.
- **Name the generalization axis.** CV2/CV3 test unseen *genes*, not unseen cell lines; CV1 tests seen genes. Cross-cell-line claims need
  context splits. **A single-fold or test-fold-selected result is not a result** — 5-fold mean ± spread.
- **A pan-essentiality lift is not an SL result**; **a benchmark rank is not a mechanism**; K562 mechanism ≠ multi-cell-line mechanism;
  Norman CRISPRa is auxiliary only.
- **Raw GeneEffect scores are inflated by `mu_g`** (`Var(mu_g) >> Var(delta)`): a context-blind per-gene mean already wins most of it, so
  context claims are residual-only.