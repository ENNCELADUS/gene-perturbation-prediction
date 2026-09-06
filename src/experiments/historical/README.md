# Historical data preparation and diagnostics

Data processing for the separate SL benchmark and completed exploratory diagnostics
lives here. These commands remain tracked for reproducibility.

| Script | Scope |
| --- | --- |
| `build_sl_context_benchmark.py` | context_screen_v2 SL pair-label benchmark; separate from the Exp13 GeneEffect benchmark |
| `audit_tx1_basal_batch.py` | Historical Tahoe-DMSO versus CCLE bulk confounding audit |
| `prepare_kinker_cpm_h5ad.py` | Historical non-raw CPM sensitivity artifacts, superseded by raw-UMI preparation |
| `stage0_tx1_input_probe.py` | Completed raw-versus-CPM and collator-seeding diagnostic |

Run from the repository root with module execution, for example:

```bash
uv run python -m src.experiments.historical.build_sl_context_benchmark --help
uv run python -m src.experiments.historical.audit_tx1_basal_batch --help
uv run python -m src.experiments.historical.prepare_kinker_cpm_h5ad --help
uv run python -m src.experiments.historical.stage0_tx1_input_probe --help
```

The older K562/Horlbeck builders remain alongside their retired datasets in
`archive/2026-09-05-inactive-routes/scripts/`. See the
[implementation guide](../../README.md) for the current module organization.
