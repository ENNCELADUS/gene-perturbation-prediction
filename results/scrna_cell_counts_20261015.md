# Single-cell inventory, 2026-10-15

Read directly from the primary per-line H5AD files on H20 host `dg3ptii1ego9q-0`, checkout `/2023533015/VCC_Project`, HEAD `7c69fcf`. Membership: `configs/benchmarks/cell_line_geneeffect_split.csv`.

Available cells are the length of each H5AD observation index under `data/sl_dependency_v0/processed/basal_raw_umi/<ModelID>.h5ad`. These counts describe the assembled primary basal input, not all cells in the original public deposit. Second-source deposits and bulk-only lines are excluded. All 434 files and prepared line caches were present.

| Split | Lines | Total cells | Minimum | Q1 | Median | Q3 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 379 | 551464 | 11 | 128 | 250 | 1210 | 58324 |
| val | 27 | 10564 | 78 | 215 | 246 | 434.5 | 1623 |
| test | 28 | 11504 | 128 | 226.75 | 307 | 430 | 1573 |

Train includes 379 single-cell lines, of which the split documentation identifies 377 as GeneEffect-labelled.

| Minimum available cells | Train / 379 | Val / 27 | Test / 28 |
| --- | ---: | ---: | ---: |
| 32 | 370 | 27 | 28 |
| 64 | 353 | 27 | 28 |
| 128 | 309 | 23 | 28 |
| 256 | 188 | 13 | 19 |
| 512 | 141 | 7 | 5 |
| 1024 | 99 | 2 | 3 |

The prepared `basal_hvg` arrays all have 128 rows. This is a cache shape, not an available-cell count. `select_context_cells` repeats cell positions when fewer cells are available. There are 70 training and 4 validation lines with fewer than 128 available cells, and none in test.

Per-line counts, source and cache size: [CSV](scrna_cell_counts_20261015.csv). A cell-line name falls back to ModelID when absent in H5AD metadata.

Only metadata was inspected; expression matrices were not loaded and no experiment was run.
