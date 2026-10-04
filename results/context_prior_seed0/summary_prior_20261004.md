# Linear context prior, run prior_20261004

Selective Spearman is the macro mean over the selective genes of the Spearman across lines of the residual; intervals are 95% paired line bootstraps (1,000 resamples, seed 0) of the gain. Validation chooses; the chosen prior is scored once on test. Nothing here is synthetic-lethality evidence.

## Learning curve

Validation selective Spearman. `oracle` scores validation lines' bulk RNA (off-contract upper bound), `val` bridged pseudo-bulk; ridge penalties are chosen on validation per input.

| Point | Lines | Config | Input | Penalties | Score |
| --- | --- | --- | --- | --- | --- |
| single_cell_train | 167 | components | oracle | 100.0 | 0.1819 |
| single_cell_train | 167 | components | val | 100.0 | 0.1658 |
| single_cell_train | 167 | components_and_selected | oracle | 100.0, 100.0 | 0.1895 |
| single_cell_train | 167 | components_and_selected | val | 100.0, 100.0 | 0.1741 |
| random_100_0 | 100 | components | oracle | 10.0 | 0.1032 |
| random_100_0 | 100 | components | val | 10.0 | 0.0970 |
| random_100_0 | 100 | components_and_selected | oracle | 10.0, 10.0 | 0.1331 |
| random_100_0 | 100 | components_and_selected | val | 10.0, 10.0 | 0.1232 |
| random_100_1 | 100 | components | oracle | 10.0 | 0.1077 |
| random_100_1 | 100 | components | val | 10.0 | 0.1047 |
| random_100_1 | 100 | components_and_selected | oracle | 10.0, 10.0 | 0.1382 |
| random_100_1 | 100 | components_and_selected | val | 10.0, 10.0 | 0.1258 |
| random_100_2 | 100 | components | oracle | 10.0 | 0.1218 |
| random_100_2 | 100 | components | val | 100.0 | 0.1345 |
| random_100_2 | 100 | components_and_selected | oracle | 10.0, 10.0 | 0.1527 |
| random_100_2 | 100 | components_and_selected | val | 100.0, 100.0 | 0.1426 |
| random_400_0 | 400 | components | oracle | 10.0 | 0.1792 |
| random_400_0 | 400 | components | val | 10.0 | 0.1653 |
| random_400_0 | 400 | components_and_selected | oracle | 10.0, 100.0 | 0.1967 |
| random_400_0 | 400 | components_and_selected | val | 10.0, 100.0 | 0.1832 |
| random_400_1 | 400 | components | oracle | 10.0 | 0.1735 |
| random_400_1 | 400 | components | val | 10.0 | 0.1507 |
| random_400_1 | 400 | components_and_selected | oracle | 10.0, 100.0 | 0.1929 |
| random_400_1 | 400 | components_and_selected | val | 10.0, 100.0 | 0.1696 |
| random_400_2 | 400 | components | oracle | 100.0 | 0.2089 |
| random_400_2 | 400 | components | val | 100.0 | 0.1881 |
| random_400_2 | 400 | components_and_selected | oracle | 100.0, 100.0 | 0.2041 |
| random_400_2 | 400 | components_and_selected | val | 100.0, 100.0 | 0.1809 |
| random_700_0 | 700 | components | oracle | 100.0 | 0.2144 |
| random_700_0 | 700 | components | val | 100.0 | 0.1937 |
| random_700_0 | 700 | components_and_selected | oracle | 100.0, 1.0 | 0.2024 |
| random_700_0 | 700 | components_and_selected | val | 100.0, 100.0 | 0.1804 |
| random_700_1 | 700 | components | oracle | 10.0 | 0.2144 |
| random_700_1 | 700 | components | val | 100.0 | 0.1941 |
| random_700_1 | 700 | components_and_selected | oracle | 10.0, 100.0 | 0.2236 |
| random_700_1 | 700 | components_and_selected | val | 100.0, 100.0 | 0.1805 |
| random_700_2 | 700 | components | oracle | 100.0 | 0.2237 |
| random_700_2 | 700 | components | val | 100.0 | 0.2036 |
| random_700_2 | 700 | components_and_selected | oracle | 100.0, 1.0 | 0.2159 |
| random_700_2 | 700 | components_and_selected | val | 100.0, 100.0 | 0.1886 |
| all | 1086 | components | oracle | 0.01 | 0.2345 |
| all | 1086 | components | val | 100.0 | 0.2119 |
| all | 1086 | components_and_selected | oracle | 0.01, 1.0 | 0.2377 |
| all | 1086 | components_and_selected | val | 100.0, 1.0 | 0.1911 |

## Extra-lines decision

All 1086 labelled training-side lines minus the 167 single-cell training lines, expression components plus data-selected genes, val input: 0.0171 [-0.0064, 0.0389]. The extra lines fail (binding).

Bulk input (off-contract): 0.0482 [0.0206, 0.0707]. The gain appears with bulk input but not with bridged input: the bridge is failing, and contrastive-PCA alignment is the next step.

The prior trains on the single-cell training lines alone.

## Block selection

| Block | Setting | Score | Gain interval | Kept |
| --- | --- | --- | --- | --- |
| expression_components | penalty 100.0 | 0.1658 | base | yes |
| pathway_scores | penalty 100.0 | 0.1608 | [-0.0281, 0.0186] | no |
| predicted_genotype | penalty 100.0 | 0.1695 | [-0.0053, 0.0133] | no |
| own_expression | shrinkage 1.0 | 0.1233 | [-0.0549, -0.0265] | no |
| partners | shrinkage 1.0 | 0.1079 | [-0.0724, -0.0377] | no |
| data_selected | penalty 100.0, N 200 | 0.1750 | [-0.0031, 0.0198] | no |
| reduced_rank | rank 64 | 0.1553 | [-0.0169, -0.0037] | no |

Chosen prior: `{"stages": [{"block": "expression_components", "penalty": 100.0, "selected": 0}], "rank": null}`

Shared expression space: 9711 genes, the bulk genes every validation and test line measures; 60 training lines lacked some of them in their source and took the training mean for those.

## Cross-fitting

Bridge quality, the per-gene Pearson across out-of-fold training lines between bridged pseudo-bulk and bulk: median 0.4289 over 9711 genes (0 undefined).

View weights: not tried (fewer than two context blocks, or disabled).

## Validation

| Model | Selective Spearman | Selective AUPR lift | Residual Pearson (per variable gene) | Huber | SD ratio (per gene) |
| --- | --- | --- | --- | --- | --- |
| Linear context prior | 0.1658 | 0.1548 | 0.1562 | 0.0163 | 0.0043 |
| Gene mean | undefined | 0.0000 | undefined | 0.0163 | 0.0000 |
| K562 copy prior | undefined | 0.0000 | undefined | 0.0337 | 0.0000 |
| Nearest line (Tx1) | 0.0810 | 0.1120 | 0.0851 | 0.0286 | 1.0225 |
| Nearest line (HVG) | 0.0806 | 0.1134 | 0.0787 | 0.0282 | 0.9888 |
| Context-PCA ridge (Tx1) | 0.1296 | 0.1335 | 0.1330 | 0.0163 | 0.2727 |
| Context-PCA ridge (HVG) | 0.1174 | 0.1265 | 0.1145 | 0.0162 | 0.2373 |

Selective Spearman, linear context prior minus context-PCA ridge (Tx1): 0.0361 [0.0119, 0.0623] over 479084 common (line, gene) pairs.

## Test

| Model | Selective Spearman | Selective AUPR lift | Residual Pearson (per variable gene) | Huber | SD ratio (per gene) |
| --- | --- | --- | --- | --- | --- |
| Linear context prior | 0.1721 | 0.1561 | 0.1634 | 0.0161 | 0.0040 |
| Gene mean | undefined | 0.0000 | undefined | 0.0161 | 0.0000 |
| K562 copy prior | undefined | 0.0000 | undefined | 0.0339 | 0.0000 |
| Nearest line (Tx1) | 0.0599 | 0.1052 | 0.0608 | 0.0287 | 1.0254 |
| Nearest line (HVG) | 0.0709 | 0.1019 | 0.0715 | 0.0283 | 1.0052 |
| Context-PCA ridge (Tx1) | 0.1206 | 0.1263 | 0.1216 | 0.0163 | 0.2687 |
| Context-PCA ridge (HVG) | 0.1185 | 0.1211 | 0.1105 | 0.0163 | 0.2504 |

Selective Spearman, linear context prior minus context-PCA ridge (Tx1): 0.0515 [0.0134, 0.0871] over 478501 common (line, gene) pairs.

## Validation, bulk input (off-contract upper bound; never a model or a comparison row)

| Model | Selective Spearman | Selective AUPR lift | Residual Pearson (per variable gene) | Huber | SD ratio (per gene) |
| --- | --- | --- | --- | --- | --- |
| Linear context prior (bulk input, off-contract) | 0.1819 | 0.1662 | 0.1700 | 0.0163 | 0.0118 |

## Per lineage

Mean per-line Pearson over the selective genes, in residual-SD units; descriptive only.

| Split | Lineage | Lines | Mean |
| --- | --- | --- | --- |
| val | Biliary Tract | 1 | -0.021 |
| val | Bladder/Urinary Tract | 1 | 0.015 |
| val | Bowel | 1 | 0.188 |
| val | Breast | 4 | 0.061 |
| val | CNS/Brain | 3 | 0.009 |
| val | Esophagus/Stomach | 2 | 0.053 |
| val | Head and Neck | 3 | -0.014 |
| val | Kidney | 1 | 0.115 |
| val | Liver | 1 | 0.041 |
| val | Lung | 4 | 0.084 |
| val | Ovary/Fallopian Tube | 2 | 0.107 |
| val | Pancreas | 1 | -0.061 |
| val | Pleura | 1 | 0.049 |
| val | Skin | 1 | -0.009 |
| val | Uterus | 1 | 0.123 |
| test | Biliary Tract | 1 | 0.050 |
| test | Bladder/Urinary Tract | 1 | 0.043 |
| test | Bowel | 2 | -0.002 |
| test | Breast | 7 | 0.055 |
| test | CNS/Brain | 2 | 0.160 |
| test | Esophagus/Stomach | 1 | 0.103 |
| test | Head and Neck | 3 | -0.019 |
| test | Kidney | 1 | 0.180 |
| test | Liver | 1 | 0.004 |
| test | Lung | 3 | 0.126 |
| test | Ovary/Fallopian Tube | 1 | 0.076 |
| test | Pancreas | 1 | 0.112 |
| test | Pleura | 1 | 0.172 |
| test | Skin | 1 | 0.026 |
| test | Uterus | 1 | 0.035 |
