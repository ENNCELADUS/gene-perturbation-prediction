# Linear context prior experiments, run default_prior_final_split_20261012

Shared expression space: 8737 genes (201 training lines took the training mean for some). The prior fits 947 labelled lines of 1399 training-side lines with bulk RNA; 343 of the 377 labelled single-cell training lines have bulk RNA and fit the bridge. 4323 selective genes.

Selective Spearman: macro mean over the selective genes of the Spearman across lines of the residual. Gain: selective Spearman minus that of the reference row (experiment `affine`, first setting, block set `selected`, components penalty 1, gene penalty 10), with its 95% paired line-bootstrap interval. Oracle: validation lines' bulk RNA as input (off-contract). A gene-level block set has a row per components and gene penalty pair. Nothing here is synthetic-lethality evidence.

## affine

Remedy `affine`.

| Setting | Block set | Components penalty | Gene penalty | Val selective Spearman | Test selective Spearman | Val gain | Test gain | Val AUPR lift | Test AUPR lift | Val residual Pearson | Test residual Pearson | Oracle val selective Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | selected | 1 | 10 | 0.2045 | 0.2025 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.1963 | 0.1827 | 0.2220 | 0.2221 | 0.2216 |

Bridge diagnostics: per-gene Pearson across the paired lines between out-of-fold bridged pseudo-bulk and bulk; counts at or above 0.3 / 0.5 / 0.7.

| Setting | Median | Q25 | Q75 | All genes | Selective genes | Selective genes' paralogs | Gene space |
| --- | --- | --- | --- | --- | --- | --- | --- |
| - | 0.453 | 0.319 | 0.593 | 6831 / 3538 / 1062 of 8737 | 2097 / 897 / 194 of 2928 | 1974 / 1137 / 404 of 2394 | all |
