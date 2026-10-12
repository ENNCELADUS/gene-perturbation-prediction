# Linear context prior experiments, run default_prior_resplit_20261012

Shared expression space: 8737 genes (59 training lines took the training mean for some). The prior fits 951 labelled lines of 1403 training-side lines with bulk RNA; 166 of the 169 labelled single-cell training lines have bulk RNA and fit the bridge. 3071 selective genes.

Selective Spearman: macro mean over the selective genes of the Spearman across lines of the residual. Gain: selective Spearman minus that of the reference row (experiment `affine`, first setting, block set `selected`, components penalty 1, gene penalty 10), with its 95% paired line-bootstrap interval. Oracle: validation lines' bulk RNA as input (off-contract). A gene-level block set has a row per components and gene penalty pair. Nothing here is synthetic-lethality evidence.

## affine

Remedy `affine`.

| Setting | Block set | Components penalty | Gene penalty | Val selective Spearman | Test selective Spearman | Val gain | Test gain | Val AUPR lift | Test AUPR lift | Val residual Pearson | Test residual Pearson | Oracle val selective Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | selected | 1 | 10 | 0.2302 | 0.2318 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.1987 | 0.1858 | 0.2242 | 0.2266 | 0.2481 |

Bridge diagnostics: per-gene Pearson across the paired lines between out-of-fold bridged pseudo-bulk and bulk; counts at or above 0.3 / 0.5 / 0.7.

| Setting | Median | Q25 | Q75 | All genes | Selective genes | Selective genes' paralogs | Gene space |
| --- | --- | --- | --- | --- | --- | --- | --- |
| - | 0.424 | 0.266 | 0.570 | 6141 / 3165 / 850 of 8737 | 1351 / 567 / 85 of 2257 | 1373 / 778 / 236 of 1821 | all |
