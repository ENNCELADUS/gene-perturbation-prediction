# Historical data preparation and diagnostics

Data processing for non-Exp13 experiments and completed exploratory diagnostics lives here. These files remain tracked for reproducibility; former top-level paths have no wrappers or symlinks.

| Script | Scope |
| --- | --- |
| `build_sl_context_benchmark.py` | context_screen_v2 SL pair-label benchmark; separate from the Exp13 GeneEffect benchmark |
| `audit_tx1_basal_batch.py` | Historical Tahoe-DMSO versus CCLE bulk confounding audit |
| `prepare_kinker_cpm_h5ad.py` | Historical non-raw CPM sensitivity artifacts, superseded by raw-UMI preparation |
| `stage0_tx1_input_probe.py` | Completed raw-versus-CPM and collator-seeding diagnostic |

Run from the repository root with module execution, for example:

```bash
uv run python -m scripts.historical_data_preparation.build_sl_context_benchmark --help
uv run python -m scripts.historical_data_preparation.stage0_tx1_input_probe --help
```

The older K562/Horlbeck builders are already sealed alongside their retired datasets in `archive/2026-09-05-inactive-routes/scripts/`; they are not active top-level scripts. See `scripts/README.md` for the Exp13 dependency inventory.
