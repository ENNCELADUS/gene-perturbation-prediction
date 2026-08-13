# CLAUDE.md — guidance for Claude Code and Codex in this repository

## What this is

Research code for **generalizable synthetic-lethality discovery by virtual-cell composition**: rank SL gene pairs for genes withheld from
SL-pair/graph training (vs. SLMGAE / KR4SL), and separately transfer a context-conditioned score to cell lines excluded from training.
**No SL graph in the feature path.** `docs/` is the authority, not this file — start at `docs/README.md`. Role: careful junior engineer;
**Plan → Confirm → Code**.

**State:** T1 (K562 Bridge-A vs Horlbeck mechanism) and the registered T2 gate (few-shot cross-cell-line GeneEffect) both **completed
negative** and are paused for redesign — their nine test lines are opened and binding. Active work is **R1**, the DepMap GeneEffect
*residual* ladder (`residual_target.py`, `residual_ladder.py`, `scripts/run_r1_residual_ladder.py`), plus the unsplit
`sl_context_screen_v1` pair × cell-line table. Feng2024 SOTA reproduction is still not started.

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
`tx1_geneeffect_eval.py`, `build_tx1_basal_embeddings.py`) · `benchmark-harness` (`data/SL_benchmark/` splits, SL metrics,
`vcc-dep-baseline`, `models:`/`selection:` configs) · `hpc-execution` (GPU, GWPS h5ad, ESM2, Tx1-3B, frozen exp05 checkpoint).

## After each implementation wave

**You run the review, not the user.** Output runs to ~200 KB — background it to a file and read the findings when it finishes.

```bash
CH=<scratch>/codex-home; mkdir -p "$CH"; cp ~/.codex/auth.json "$CH/"
printf 'model = "gpt-5.6-sol"\nmodel_reasoning_effort = "high"\napproval_policy = "never"\nsandbox_mode = "workspace-write"\n' > "$CH/config.toml"
CODEX_HOME="$CH" codex review --base <WAVE_BASE_SHA> > <wave>-review.txt 2>&1
```

The **clean `CODEX_HOME`** is load-bearing: `-c 'mcp_servers={}'` merges instead of replacing, so `~/.codex/config.toml`'s `[mcp_servers.*]`
survive it and the review hangs on `gitnexus/detect_changes` with zero output growth (verified 2026-07-25; it killed three reviews). That
home also carries model and effort, so no `-c` flags. `review` accepts only `--base/--scope/--model/--cwd`; `--effort` is not a flag and a
stray value is parsed as focus text and rejected. `/codex:review` is `disable-model-invocation` — that blocks the slash command, not the CLI.

## Environment and testing

- The five `src/` packages look peer-level but are not: **`aivc_model/` (exp05 → exp12 Tx1 → R1) is the active backbone — new work goes
  there**; the rest are benchmark, features, baselines. Most entrypoints need gitignored assets, so "it imports" ≠ "it runs here".
- `.superpowers/sdd/` is gitignored but holds the **live plan and execution ledger** (its `progress.md` lags the code); `docs/specs/` holds
  the tracked designs. **`AGENTS.md` is a symlink to `CLAUDE.md`** — edit `CLAUDE.md` only. **`.codex/` is a current, tracked Codex setup**
  (`config.toml`, five agents, four mirrored skills, its own supplementary `AGENTS.md`) — keep both trees in sync.
- Prefix every Python/pytest/ruff call with `uv run`. A global `rtk hook claude` hook **rewrites Bash output** (`ruff check .` prints `[]`)
  — call `.venv/bin/ruff` for real output. `uv sync` installs the `dev` group only (`scib`/`datasets` absent); a new `src/<pkg>/` is
  invisible until added to `[tool.hatch.build.targets.wheel] packages`; `arc-state` is pinned to a **git commit** (bumps are deliberate).
- **No pytest config**: no testpaths, no addopts, **no markers** (`-m` does nothing). Imports resolve only via the editable `.pth` — always
  `uv run python -m pytest` **from the repo root**. **`tests/conftest.py` is load-bearing**: it sets `PYTORCH_ENABLE_MPS_FALLBACK` and
  `OMP_NUM_THREADS` *before* torch imports, imports xgboost first, and covers the whole suite — **a test file run directly segfaults**.
- **The suite is not green** — baseline it first; a failure may not be yours. Tests **skip silently** on missing gitignored data or
  `accelerate`. `ruff format .` rewrites two dozen unrelated files — **format only what you touched**; with `E,W,F` only, **import order is
  not enforced**, so don't "fix" imports.

## Silent failures — the dominant risk

Warnings and defaults are preferred over exceptions, so mistakes usually produce a complete-looking **wrong artifact**, not an error.

- **Config loading does no schema validation** — every field is `.get(key, default)`, so a **misspelled YAML key silently takes the
  default**. All exp05 config dataclasses live in the `@dataclass(frozen=True)` block atop `prepare.py`; add fields there, not locally.
  `_path_or_none` is truthiness-based, so `checkpoint_path: ""` silently **disables** the feature; unset `data.state_embed_key` falls back
  to `adata.X`, not `obsm`. **STATE loads with `strict=False`** (`model.py`) — copy the Tx1 `validate_load_result` pattern instead.
- `--skip-hash-check` / `--allow-partial` / any off-contract threshold downgrade an evaluation to `formal:false` **and still exit 0** —
  never report such a run as formal. The Tx1 cache separately zero-fills missing genes, reuses stale cells, and skips completeness checks
  on a sharded verify.
- **R1 residuals: never center a *prediction* on a fold-fit gene mean.** `mu_hat_g^(-c)` is an exact affine function of the held-out label, so
  `constant_g - mu_hat_g^(-c)` scores per-gene Spearman `+1.0` by construction. Targets use `mu_hat`, predictions the fold-independent `mu_bar`.

## Data

**Every dataset has a card in `docs/data/` — read it before using the file.** Several are not what their names suggest:
`jost_replogle_dual_sgrna` is knockdown efficacy, **not epistasis**; the two Replogle h5ads differ by one word (`essential`/`gwps`) but have
different gene panels; Horlbeck's field is `gi_score`, **not** `gamma`, and **negative = SL**; X-Atlas/Orion HCT116 is **no longer**
untouched; `sl_context_screen_v1` is **unsplit** with screened-non-hit negatives — not universal non-SL, and no CV folds. **Never join cell
lines on name** — DepMap rows are ACH ModelIDs and its `CellLineName` is `K-562` while code says `K562`. Large artifacts are gitignored
except the hash-pinned `results/phase_a_tx1_20260724/` contract.

## Claim discipline

Full gates in `docs/02-acceptance-criteria.md` §7-8. The four broken most easily —

- **Never claim SL from single-gene essentiality** — an explicit interaction null is required: `interaction = joint - psi(singles)`; declare
  `psi`. DepMap GeneEffect is a relative growth-rate effect: single-gene, never a double-knockout quantity.
- **Name the generalization axis.** CV2/CV3 test unseen *genes*, not unseen cell lines; CV1 is diagnostic. Cross-cell-line claims need
  untouched line splits. **A single-fold or test-fold-selected result is not a result** — 5-fold mean ± spread.
- **A pan-essentiality lift is not an SL result**; **a benchmark rank is not a mechanism**; K562 mechanism ≠ multi-cell-line mechanism;
  Norman CRISPRa is auxiliary only.
- **Raw GeneEffect scores are inflated by `mu_g`** (`Var(mu_g) >> Var(delta)`): a context-blind per-gene mean already wins most of it, so
  context claims are residual-only.