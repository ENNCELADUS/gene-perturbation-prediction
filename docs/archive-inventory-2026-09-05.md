# Local archive inventory — 2026-09-05

Archive: `archive/2026-09-05-inactive-routes/` (gitignored; local only).
40 original paths, 496 files, 12,501,355,692 bytes were moved, with no original-path symlinks. `manifest.json` inside the archive records every file's original path, size and SHA-256 (or symlink target). Same-filesystem rename preserved inodes; destination sizes and inodes were checked.

| Archived group | Reason |
| --- | --- |
| `src/dependency_baseline`, `src/sl_benchmark_baseline`, `src/sl_profile_baseline` | Retired dependency and Feng2024 program, outside current held-out-cell-line protocol; no imports from retained aivc_model/scripts |
| 26 matching tests, legacy K562/Horlbeck builders and DDGCN verifier/config | Paired with retired implementations; DDGCN source was already absent before this task |
| `results/experiments` (03, 05–09, 11, 12) | Historical experiments, including paused T1/T2; no local Exp13 output directory was present |
| `logs/downloads` | Old GWPS download log/PID; no matching local running download process |
| `data/SL_benchmark` | Historical Feng2024 checkout, including its nested Git history |
| `data/SL_Benchmark_Formal/derived/context_screen_v1` | Superseded pre-provenance snapshot |
| `data/sl_dependency_v0/{interim,splits,processed/horlbeck_2018}` | Retired overlap tables, gene-split artifacts and T1 GI outputs |

Retained: `src/aivc_model` and its tests/scripts, current context_screen_v2 data and authorities, shared raw Perturb-seq/DepMap/ESM2 assets, STATE weights, `results/stage0`, and `results/phase_a_tx1_20260724`. The last directory still supplies registration and cell-line manifests to Exp13 configuration, builders and tests; its age does not make it removable. Stage0 records the input-representation decision still used by the current substrate. Curated `docs/results` reports remain scientific provenance, including negative results.

Exp13 is scope-closed GeneEffect evidence, not SL evidence. Its reusable substrate and registered R1 control are retained; completion alone is not a reason to remove current dependencies.

Restore selected paths from the archive to their original locations after checking for conflicts. Restore package/CLI declarations from `reference/pyproject.toml` when restoring a historical implementation. `reference/README.md` preserves pre-archive navigation. This archive is not included in Git or backed up remotely by this operation.

Local process inspection found no running research training/evaluation/download process using these routes. No remote checkout or job was inspected or changed. Pre-existing CLAUDE.md edits and scGPT/DDGCN deletions were preserved.

## Verification

- Retained suite: 804 passed, 1 skipped, 2 warnings (32.86 s), using `uv run --offline --no-sync python -m pytest tests -q --disable-warnings --maxfail=3`.
- `uv run --offline --no-sync ruff check .`: passed.
- Diff whitespace check for edited README, package configuration and data card: passed. Full-tree check still reports pre-existing trailing whitespace in CLAUDE.md, left untouched.
- Wheel build was attempted but not executed because the local environment lacks `hatchling`; packaging configuration is updated, but wheel generation is unverified.
- All 40 archived original paths are absent; the archive retains all 496 manifest-listed files.

## Script organization follow-up

Three additional diagnostic utilities were moved unchanged into tracked `scripts/historical_data_preparation/`: `audit_tx1_basal_batch.py`, `prepare_kinker_cpm_h5ad.py`, and `stage0_tx1_input_probe.py`. Unlike the local archive above, this folder remains in Git. Their tests import the new locations; the Stage0 reproduction command uses module execution from the repository root. Shared raw-UMI/Tx1/ESM2 and Exp13/R1 processing remains at the top level. See `scripts/historical_data_preparation/README.md` for the per-script rationale.

Follow-up validation: 20 focused tests passed; Ruff and changed-file whitespace checks passed; all three new module entrypoints returned success for `--help`.

The script folder was subsequently renamed to `scripts/historical_data_preparation/`, and the separate SL-pair benchmark builder was moved there. Top-level processing is limited to Exp13 plus its required shared inputs, listed in `scripts/README.md`.

Exp13-only organization validation: 36 focused tests passed; all four historical module help commands, Ruff and scoped whitespace checks passed. No old script-directory references remain in scripts, tests, README or docs.
