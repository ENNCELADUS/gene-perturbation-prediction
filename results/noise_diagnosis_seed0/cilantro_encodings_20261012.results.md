# Linear context prior experiments, run cilantro_encodings_20261012

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

## rank_value

Remedy `rank_value`.

| Setting | Block set | Components penalty | Gene penalty | Val selective Spearman | Test selective Spearman | Val gain | Test gain | Val AUPR lift | Test AUPR lift | Val residual Pearson | Test residual Pearson | Oracle val selective Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| encoding rank_value, top_k None | selected | 1 | 10 | 0.2023 | 0.2037 | -0.0022 [-0.0076, 0.0033] | 0.0013 [-0.0036, 0.0067] | 0.1901 | 0.1839 | 0.2193 | 0.2261 | 0.2190 |
| encoding rank_value, top_k 2048 | selected | 1 | 10 | 0.1853 | 0.1948 | -0.0192 [-0.0300, -0.0092] | -0.0077 [-0.0212, 0.0063] | 0.1841 | 0.1841 | 0.2004 | 0.2059 | 0.2111 |
| encoding expression, top_k 2048 | selected | 1 | 10 | 0.1904 | 0.1996 | -0.0142 [-0.0201, -0.0067] | -0.0029 [-0.0141, 0.0080] | 0.1885 | 0.1862 | 0.2090 | 0.2210 | 0.2149 |

Bridge diagnostics: per-gene Pearson across the paired lines between out-of-fold bridged pseudo-bulk and bulk; counts at or above 0.3 / 0.5 / 0.7.

| Setting | Median | Q25 | Q75 | All genes | Selective genes | Selective genes' paralogs | Gene space |
| --- | --- | --- | --- | --- | --- | --- | --- |
| encoding rank_value, top_k None | 0.466 | 0.356 | 0.575 | 7313 / 3657 / 654 of 8737 | 2315 / 933 / 117 of 2928 | 2075 / 1166 / 236 of 2394 | all |
| encoding rank_value, top_k 2048 | 0.302 | 0.142 | 0.446 | 4391 / 1572 / 222 of 8737 | 1318 / 384 / 39 of 2928 | 1279 / 499 / 75 of 2394 | all |
| encoding expression, top_k 2048 | 0.252 | 0.122 | 0.384 | 3460 / 1001 / 215 of 8737 | 923 / 187 / 30 of 2928 | 1049 / 341 / 83 of 2394 | all |
