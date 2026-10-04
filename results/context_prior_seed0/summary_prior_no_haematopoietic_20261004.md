# Linear context prior, run prior_no_haematopoietic_20261004

Selective Spearman is the macro mean over the selective genes of the Spearman across lines of the residual; intervals are 95% paired line bootstraps (1,000 resamples, seed 0) of the gain. Validation chooses; the chosen prior is scored once on test. Nothing here is synthetic-lethality evidence.

## Learning curve

Validation selective Spearman. `oracle` scores validation lines' bulk RNA (off-contract upper bound), `val` bridged pseudo-bulk; ridge penalties are chosen on validation per input.

| Point | Lines | Config | Input | Penalties | Score |
| --- | --- | --- | --- | --- | --- |
| single_cell_train | 167 | components | oracle | 100.0 | 0.1750 |
| single_cell_train | 167 | components | val | 100.0 | 0.1666 |
| single_cell_train | 167 | components_and_selected | oracle | 100.0, 100.0 | 0.1888 |
| single_cell_train | 167 | components_and_selected | val | 100.0, 100.0 | 0.1739 |
| random_100_0 | 100 | components | oracle | 10.0 | 0.0910 |
| random_100_0 | 100 | components | val | 100.0 | 0.0939 |
| random_100_0 | 100 | components_and_selected | oracle | 10.0, 10.0 | 0.1362 |
| random_100_0 | 100 | components_and_selected | val | 100.0, 100.0 | 0.1265 |
| random_100_1 | 100 | components | oracle | 10.0 | 0.1030 |
| random_100_1 | 100 | components | val | 10.0 | 0.0970 |
| random_100_1 | 100 | components_and_selected | oracle | 10.0, 10.0 | 0.1446 |
| random_100_1 | 100 | components_and_selected | val | 10.0, 10.0 | 0.1206 |
| random_100_2 | 100 | components | oracle | 1.0 | 0.1142 |
| random_100_2 | 100 | components | val | 1.0 | 0.0961 |
| random_100_2 | 100 | components_and_selected | oracle | 1.0, 10.0 | 0.1297 |
| random_100_2 | 100 | components_and_selected | val | 1.0, 10.0 | 0.1130 |
| random_400_0 | 400 | components | oracle | 10.0 | 0.1911 |
| random_400_0 | 400 | components | val | 100.0 | 0.1855 |
| random_400_0 | 400 | components_and_selected | oracle | 10.0, 100.0 | 0.2122 |
| random_400_0 | 400 | components_and_selected | val | 100.0, 100.0 | 0.1834 |
| random_400_1 | 400 | components | oracle | 100.0 | 0.1854 |
| random_400_1 | 400 | components | val | 10.0 | 0.1779 |
| random_400_1 | 400 | components_and_selected | oracle | 100.0, 100.0 | 0.1911 |
| random_400_1 | 400 | components_and_selected | val | 10.0, 100.0 | 0.1912 |
| random_400_2 | 400 | components | oracle | 0.1 | 0.1804 |
| random_400_2 | 400 | components | val | 1.0 | 0.1644 |
| random_400_2 | 400 | components_and_selected | oracle | 0.1, 10.0 | 0.1839 |
| random_400_2 | 400 | components_and_selected | val | 1.0, 10.0 | 0.1798 |
| random_700_0 | 700 | components | oracle | 10.0 | 0.2191 |
| random_700_0 | 700 | components | val | 10.0 | 0.2077 |
| random_700_0 | 700 | components_and_selected | oracle | 10.0, 100.0 | 0.2313 |
| random_700_0 | 700 | components_and_selected | val | 10.0, 100.0 | 0.2122 |
| random_700_1 | 700 | components | oracle | 10.0 | 0.2126 |
| random_700_1 | 700 | components | val | 100.0 | 0.2060 |
| random_700_1 | 700 | components_and_selected | oracle | 10.0, 100.0 | 0.2254 |
| random_700_1 | 700 | components_and_selected | val | 100.0, 100.0 | 0.1931 |
| random_700_2 | 700 | components | oracle | 10.0 | 0.2192 |
| random_700_2 | 700 | components | val | 100.0 | 0.2025 |
| random_700_2 | 700 | components_and_selected | oracle | 10.0, 100.0 | 0.2317 |
| random_700_2 | 700 | components_and_selected | val | 100.0, 100.0 | 0.1932 |
| all | 953 | components | oracle | 10.0 | 0.2358 |
| all | 953 | components | val | 10.0 | 0.2232 |
| all | 953 | components_and_selected | oracle | 10.0, 100.0 | 0.2440 |
| all | 953 | components_and_selected | val | 10.0, 100.0 | 0.2248 |

## Extra-lines decision

All 953 labelled training-side lines minus the 167 single-cell training lines, expression components plus data-selected genes, val input: 0.0510 [0.0274, 0.0692]. The extra lines pass (binding).

Bulk input (off-contract): 0.0552 [0.0307, 0.0754].

## Block selection

| Block | Setting | Score | Gain interval | Kept |
| --- | --- | --- | --- | --- |
| expression_components | penalty 10.0 | 0.2232 | base | yes |
| pathway_scores | penalty 100.0 | 0.2236 | [-0.0119, 0.0121] | no |
| predicted_genotype | penalty 100.0 | 0.2242 | [-0.0016, 0.0035] | no |
| own_expression | shrinkage inf | 0.2144 | [-0.0124, -0.0040] | no |
| partners | shrinkage inf | 0.2214 | [-0.0038, 0.0005] | no |
| data_selected | penalty 100.0, N 10 | 0.2276 | [-0.0005, 0.0092] | no |
| reduced_rank | rank 64 | 0.2049 | [-0.0287, -0.0068] | no |

Chosen prior: `{"stages": [{"block": "expression_components", "penalty": 10.0, "selected": 0}], "rank": null}`

Shared expression space: 9711 genes, the bulk genes every validation and test line measures; 60 training lines lacked some of them in their source and took the training mean for those.

## Cross-fitting

Bridge quality, the per-gene Pearson across out-of-fold training lines between bridged pseudo-bulk and bulk: median 0.4298 over 9711 genes (0 undefined).

View weights: not tried (fewer than two context blocks, or disabled).

## Validation

| Model | Selective Spearman | Selective AUPR lift | Residual Pearson (per variable gene) | Huber | SD ratio (per gene) |
| --- | --- | --- | --- | --- | --- |
| Linear context prior | 0.2232 | 0.1959 | 0.2141 | 0.0162 | 0.0161 |
| Gene mean | undefined | 0.0000 | undefined | 0.0163 | 0.0000 |
| K562 copy prior | undefined | 0.0000 | undefined | 0.0337 | 0.0000 |
| Nearest line (Tx1) | 0.0810 | 0.1120 | 0.0851 | 0.0286 | 1.0225 |
| Nearest line (HVG) | 0.0806 | 0.1134 | 0.0787 | 0.0282 | 0.9888 |
| Context-PCA ridge (Tx1) | 0.1296 | 0.1335 | 0.1330 | 0.0163 | 0.2727 |
| Context-PCA ridge (HVG) | 0.1174 | 0.1265 | 0.1145 | 0.0162 | 0.2373 |

Selective Spearman, linear context prior minus context-PCA ridge (Tx1): 0.0935 [0.0578, 0.1241] over 479084 common (line, gene) pairs.

## Test

| Model | Selective Spearman | Selective AUPR lift | Residual Pearson (per variable gene) | Huber | SD ratio (per gene) |
| --- | --- | --- | --- | --- | --- |
| Linear context prior | 0.2227 | 0.1862 | 0.2177 | 0.0160 | 0.0159 |
| Gene mean | undefined | 0.0000 | undefined | 0.0161 | 0.0000 |
| K562 copy prior | undefined | 0.0000 | undefined | 0.0339 | 0.0000 |
| Nearest line (Tx1) | 0.0599 | 0.1052 | 0.0608 | 0.0287 | 1.0254 |
| Nearest line (HVG) | 0.0709 | 0.1019 | 0.0715 | 0.0283 | 1.0052 |
| Context-PCA ridge (Tx1) | 0.1206 | 0.1263 | 0.1216 | 0.0163 | 0.2687 |
| Context-PCA ridge (HVG) | 0.1185 | 0.1211 | 0.1105 | 0.0163 | 0.2504 |

Selective Spearman, linear context prior minus context-PCA ridge (Tx1): 0.1021 [0.0662, 0.1314] over 478501 common (line, gene) pairs.

## Validation, bulk input (off-contract upper bound; never a model or a comparison row)

| Model | Selective Spearman | Selective AUPR lift | Residual Pearson (per variable gene) | Huber | SD ratio (per gene) |
| --- | --- | --- | --- | --- | --- |
| Linear context prior (bulk input, off-contract) | 0.2358 | 0.2046 | 0.2276 | 0.0160 | 0.0442 |

## Per lineage

Mean per-line Pearson over the selective genes, in residual-SD units; descriptive only.

| Split | Lineage | Lines | Mean |
| --- | --- | --- | --- |
| val | Biliary Tract | 1 | 0.113 |
| val | Bladder/Urinary Tract | 1 | 0.061 |
| val | Bowel | 1 | -0.120 |
| val | Breast | 4 | 0.039 |
| val | CNS/Brain | 3 | 0.173 |
| val | Esophagus/Stomach | 2 | 0.039 |
| val | Head and Neck | 3 | 0.139 |
| val | Kidney | 1 | 0.076 |
| val | Liver | 1 | 0.168 |
| val | Lung | 4 | 0.078 |
| val | Ovary/Fallopian Tube | 2 | 0.134 |
| val | Pancreas | 1 | 0.110 |
| val | Pleura | 1 | 0.093 |
| val | Skin | 1 | 0.214 |
| val | Uterus | 1 | 0.165 |
| test | Biliary Tract | 1 | 0.076 |
| test | Bladder/Urinary Tract | 1 | 0.147 |
| test | Bowel | 2 | 0.054 |
| test | Breast | 7 | 0.056 |
| test | CNS/Brain | 2 | 0.149 |
| test | Esophagus/Stomach | 1 | 0.137 |
| test | Head and Neck | 3 | 0.115 |
| test | Kidney | 1 | 0.138 |
| test | Liver | 1 | 0.129 |
| test | Lung | 3 | 0.127 |
| test | Ovary/Fallopian Tube | 1 | 0.155 |
| test | Pancreas | 1 | 0.078 |
| test | Pleura | 1 | 0.111 |
| test | Skin | 1 | 0.118 |
| test | Uterus | 1 | 0.040 |
