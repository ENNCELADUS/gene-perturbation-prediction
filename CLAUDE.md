# CLAUDE.md

## What this is

Research code for **generalizable synthetic-lethality discovery by virtual-cell
composition**: rank SL gene pairs for genes withheld from SL-pair/graph training (vs.
SLMGAE / KR4SL on the Feng2024 benchmark), and separately transfer a context-conditioned
score to cell lines excluded from training. **No SL graph in the feature path.** `docs/`
is the authority, not this file — start at `docs/README.md`. Role: careful junior
engineer; **Plan → Confirm → Code**.

## After each implementation wave

Ask the user to run **`/codex:review --wait`** — Claude cannot invoke it
(`disable-model-invocation`), and no `--model`/`--effort` flags. See `codex-review`.

## Skills — load before touching the area

| Skill | Load before |
|---|---|
| `research-vault` | editing `docs/`, writing a claim, recording a result |
| `tx1-cache` | `tx1_basal.py`, `tx1_embed_cache.py`, `tx1_geneeffect_eval.py`, `build_tx1_basal_embeddings.py` |
| `benchmark-harness` | `data/SL_benchmark/` splits, SL metrics, `vcc-dep-baseline`, `models:`/`selection:` configs |
| `hpc-execution` | anything needing GPU, GWPS h5ad, ESM2, Tx1-3B, or the frozen exp05 checkpoint |
| `codex-review` | finishing a wave |

## Environment

- The six `src/` packages look peer-level but are not: **`aivc_model/` (exp05 → exp12 Tx1)
  is the active backbone — new work goes there**; the rest are benchmark, features, and
  baselines. Most entrypoints need gitignored assets, so "it imports" ≠ "it runs here".
- `.superpowers/sdd/` is gitignored but holds the **live plan and execution ledger**;
  `docs/specs/` holds the tracked designs.
- **`AGENTS.md` is a symlink to `CLAUDE.md`** — edit `CLAUDE.md` only. **`.codex/AGENTS.md`
  is stale and describes a different project**; ignore it.
- Prefix every Python/pytest/ruff call with `uv run`. A global `rtk hook claude` hook
  **rewrites Bash output** (`ruff check .` prints `[]`) — call `.venv/bin/ruff` for real output.
- `uv sync` installs the `dev` group only (`scib`/`datasets` absent). A new `src/<pkg>/` is
  invisible until added to `[tool.hatch.build.targets.wheel] packages`. `arc-state` is
  pinned to a **git commit** — STATE API changes are a deliberate bump.

## Testing

- **No pytest config**: no testpaths, no addopts, **no markers** (`-m` does nothing).
  Imports resolve only via the editable `.pth` — always `uv run python -m pytest` **from
  the repo root**.
- **`tests/conftest.py` is load-bearing**: it sets `PYTORCH_ENABLE_MPS_FALLBACK` and
  `OMP_NUM_THREADS` *before* torch imports, imports xgboost first, and covers
  `tests/sl_dl_model/` too. **Running a test file directly segfaults.** Reuse its fixtures.
- **The suite is not green** — baseline it first; don't assume a failure is yours. Tests
  also **skip silently** on missing gitignored data or `accelerate`, so a green run may
  have covered far less than it looks.
- `ruff format .` rewrites two dozen unrelated files — **format only what you touched**;
  and with `E,W,F` only, **import order is not enforced**, so don't "fix" imports.

## Silent failures — the dominant risk

Warnings and defaults are preferred over exceptions here, so mistakes usually produce a
complete-looking **wrong artifact** rather than an error.

- **Config loading does no schema validation** — every field is `.get(key, default)`, so a
  **misspelled YAML key silently takes the default**. All exp05 config dataclasses live in
  the `@dataclass(frozen=True)` block atop `prepare.py`; add fields there, not locally.
  `_path_or_none` is truthiness-based, so `checkpoint_path: ""` silently **disables** the
  feature; unset `data.state_embed_key` falls back to `adata.X`, not `obsm`.
- **STATE loads with `strict=False`** (`model.py`) — missing weights load silently. Copy
  the Tx1 `validate_load_result` pattern instead.
- `--skip-hash-check` / `--allow-partial` / any off-contract threshold downgrade an
  evaluation to `formal:false` **and still exit 0** — never report such a run as formal.
  The Tx1 cache separately zero-fills missing genes, reuses stale cells, and skips
  completeness checks on a sharded verify. Load `tx1-cache`.

## Data

**Every dataset has a card in `docs/data/` — read it before using the file.** Several are
not what their names suggest: `jost_replogle_dual_sgrna` is knockdown efficacy, **not
epistasis**; the two Replogle h5ads differ by one word (`essential`/`gwps`) but have
different gene panels; Horlbeck's field is `gi_score`, **not** `gamma`, and **negative = SL**;
X-Atlas/Orion HCT116 is **no longer** untouched. **Never join cell lines on name** — DepMap
rows are ACH ModelIDs and its `CellLineName` is `K-562` while code says `K562`. Large
artifacts are gitignored except the hash-pinned `results/phase_a_tx1_20260724/` contract.

## Claim discipline

Full gates in `docs/02-acceptance-criteria.md` §7-8; **load `research-vault` before writing
any result.** The four broken most easily —

- **Never claim SL from single-gene essentiality** — an explicit interaction null is
  required: `interaction = joint - psi(singles)`; declare `psi`. DepMap GeneEffect is a
  relative growth-rate effect: single-gene, never a double-knockout quantity.
- **Name the generalization axis.** CV2/CV3 test unseen *genes*, not unseen cell lines;
  CV1 is diagnostic. Cross-cell-line claims need untouched line splits.
- **A pan-essentiality lift is not an SL result**; **a benchmark rank is not a mechanism**;
  K562 mechanism ≠ multi-cell-line mechanism; Norman CRISPRa is auxiliary only.
- **A single-fold or test-fold-selected result is not a result.** 5-fold mean ± spread.

## Code style

Python 3.11+, strict type hints, absolute imports, Google-style docstrings, `logging` not
`print`, no hardcoded paths/thresholds, no bare `except`, Conventional Commits. Files
should stay <600 lines — several hot ones already exceed that 5×; the rule binds new code.