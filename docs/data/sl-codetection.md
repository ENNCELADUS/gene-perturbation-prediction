# Co-detection of SL pairs in the screened lines' cells

**Status:** built on the H20 (container 30838) on 2026-11-13 under `data/geneeffect_joint/v2/codetection/` by
`hpc/run.sh sl-codetection` (`src/experiments/prepare.py`, `prepare_codetection`), for
[SL protocol §13](../04-sl-ranking-protocol.md#13-two-single-cell-inputs-2026-11-13).

| Path | Contents | Counts |
| --- | --- | --- |
| `<ModelID>.parquet` | per (gene A, gene B) of the line's labelled rows in `data/sl_studies/benchmark_v2/labels.parquet`: the line's cell count; the share of cells that detect both genes (count > 0); the share expected under independence at each cell's depth, with the mean-profile rate and with each gene's rate matched to its detection share; the two single-gene detection shares; the same five values after `shuffle_counts` (seed 0) permuted each gene's counts across the cells (`shuffled_*`) | 86 lines of the SL split, 2,015,789 pairs; 1,528,342 (75.8%) with a value (NaN: a gene, or a drug anchor, that the line's source does not list); 18 to 58,324 cells per line (median 292) |
| `manifest.json` | the labels, the SL split, the seed and the lines; a root written from other labels or another split is refused | |

Source: the raw-UMI h5ad of each line (`paths.basal_raw_umi` of the joint config). The statistics read every cell of
the line. A gene listed twice in a source is one column (the counts are summed), so its shuffled detection share can
differ from the real one (largest difference 0.008).
