# Revision run correction_dependency_classification_no_state_seed0

Config `configs/correction/dependency_classification_no_state.yaml`, git revision `75b825a4b12bfdb21668e114f85e214d7692320f`. `train/best.pt` is chosen on validation alone, then scored once on test. Nothing here is synthetic-lethality evidence.

Selective Spearman and AUPR lift are macro means over the selective genes; residual Pearson and the SD ratio are macro means over the variable genes. Undefined correlations come from predictors that are constant per gene across lines (gene mean, copy prior); they are not zero.

## Validation lines

| Model | Selective Spearman | Selective AUPR lift | Residual Pearson (per variable gene) | Huber | SD ratio (per gene) |
|---|---|---|---|---|---|
| Joint model | 0.2290 | 0.1974 | 0.2229 | 0.0156 | 0.1179 |
| Gene mean | undefined | 0.0000 | undefined | 0.0163 | 0.0000 |
| K562 copy prior | undefined | 0.0000 | undefined | 0.0337 | 0.0000 |
| Nearest line (Tx1) | 0.0810 | 0.1120 | 0.0851 | 0.0286 | 1.0225 |
| Nearest line (HVG) | 0.0806 | 0.1134 | 0.0787 | 0.0282 | 0.9888 |
| Context-PCA ridge (Tx1) | 0.1296 | 0.1335 | 0.1330 | 0.0163 | 0.2727 |
| Context-PCA ridge (HVG) | 0.1174 | 0.1265 | 0.1145 | 0.0162 | 0.2373 |
| Linear context prior | 0.2290 | 0.1974 | 0.2229 | 0.0156 | 0.1179 |
| Linear context prior + Tx1 context-PCA ridge | 0.1826 | 0.1683 | 0.1836 | 0.0160 | 0.2936 |

Selective Spearman, Joint model minus Context-PCA ridge (Tx1): 0.0993 [0.0663, 0.1266], paired bootstrap over validation lines (1000 resamples, seed 0).
Selective Spearman, Joint model minus Linear context prior: 0.0000 [0.0000, 0.0000], paired bootstrap over validation lines (1000 resamples, seed 0).
Selective Spearman, Joint model minus Linear context prior + Tx1 context-PCA ridge: 0.0464 [0.0220, 0.0684], paired bootstrap over validation lines (1000 resamples, seed 0).

Per lineage, descriptive: mean over the lineage's validation lines of the per-line residual Spearman across the selective genes.

| Lineage | Lines | Joint model | Context-PCA ridge (Tx1) | Linear context prior | Linear context prior + Tx1 context-PCA ridge |
|---|---|---|---|---|---|
| Biliary Tract | 1 | 0.2950 | 0.0899 | 0.2950 | 0.1499 |
| Bladder/Urinary Tract | 1 | 0.1187 | 0.0490 | 0.1187 | 0.0876 |
| Bowel | 1 | 0.0689 | 0.0873 | 0.0689 | 0.1138 |
| Breast | 4 | 0.3170 | 0.2389 | 0.3170 | 0.2808 |
| CNS/Brain | 3 | 0.2858 | 0.1238 | 0.2858 | 0.1714 |
| Esophagus/Stomach | 2 | 0.1811 | 0.1336 | 0.1811 | 0.1634 |
| Head and Neck | 3 | 0.2455 | 0.1907 | 0.2455 | 0.2180 |
| Kidney | 1 | 0.2871 | 0.2967 | 0.2871 | 0.3554 |
| Liver | 1 | 0.3793 | 0.1686 | 0.3793 | 0.2929 |
| Lung | 4 | 0.2679 | 0.0957 | 0.2679 | 0.1888 |
| Ovary/Fallopian Tube | 2 | 0.3111 | 0.2455 | 0.3111 | 0.2747 |
| Pancreas | 1 | 0.2384 | 0.0906 | 0.2384 | 0.1175 |
| Pleura | 1 | 0.1902 | -0.0003 | 0.1902 | 0.0741 |
| Skin | 1 | 0.3956 | 0.2238 | 0.3956 | 0.3136 |
| Uterus | 1 | 0.2944 | 0.0767 | 0.2944 | 0.1924 |

## Test lines

| Model | Selective Spearman | Selective AUPR lift | Residual Pearson (per variable gene) | Huber | SD ratio (per gene) |
|---|---|---|---|---|---|
| Joint model | 0.2338 | 0.1910 | 0.2286 | 0.0155 | 0.1139 |
| Gene mean | undefined | 0.0000 | undefined | 0.0161 | 0.0000 |
| K562 copy prior | undefined | 0.0000 | undefined | 0.0339 | 0.0000 |
| Nearest line (Tx1) | 0.0599 | 0.1052 | 0.0608 | 0.0287 | 1.0254 |
| Nearest line (HVG) | 0.0709 | 0.1019 | 0.0715 | 0.0283 | 1.0052 |
| Context-PCA ridge (Tx1) | 0.1206 | 0.1263 | 0.1216 | 0.0163 | 0.2687 |
| Context-PCA ridge (HVG) | 0.1185 | 0.1211 | 0.1105 | 0.0163 | 0.2504 |
| Linear context prior | 0.2338 | 0.1910 | 0.2286 | 0.0155 | 0.1139 |
| Linear context prior + Tx1 context-PCA ridge | 0.1823 | 0.1596 | 0.1824 | 0.0159 | 0.2879 |

Selective Spearman, Joint model minus Context-PCA ridge (Tx1): 0.1131 [0.0784, 0.1411], paired bootstrap over test lines (1000 resamples, seed 0).
Selective Spearman, Joint model minus Linear context prior: 0.0000 [0.0000, 0.0000], paired bootstrap over test lines (1000 resamples, seed 0).
Selective Spearman, Joint model minus Linear context prior + Tx1 context-PCA ridge: 0.0514 [0.0264, 0.0730], paired bootstrap over test lines (1000 resamples, seed 0).

Per lineage, descriptive: mean over the lineage's test lines of the per-line residual Spearman across the selective genes.

| Lineage | Lines | Joint model | Context-PCA ridge (Tx1) | Linear context prior | Linear context prior + Tx1 context-PCA ridge |
|---|---|---|---|---|---|
| Biliary Tract | 1 | 0.2753 | 0.2248 | 0.2753 | 0.3056 |
| Bladder/Urinary Tract | 1 | 0.2795 | 0.1195 | 0.2795 | 0.1782 |
| Bowel | 2 | 0.3516 | 0.1491 | 0.3516 | 0.2547 |
| Breast | 7 | 0.2304 | 0.1231 | 0.2304 | 0.1720 |
| CNS/Brain | 2 | 0.2904 | 0.1908 | 0.2904 | 0.2347 |
| Esophagus/Stomach | 1 | 0.2271 | 0.1179 | 0.2271 | 0.1690 |
| Head and Neck | 3 | 0.2312 | 0.1744 | 0.2312 | 0.2038 |
| Kidney | 1 | 0.4296 | 0.2201 | 0.4296 | 0.3854 |
| Liver | 1 | 0.3593 | 0.0445 | 0.3593 | 0.1637 |
| Lung | 3 | 0.3052 | 0.1696 | 0.3052 | 0.2622 |
| Ovary/Fallopian Tube | 1 | 0.4105 | 0.3165 | 0.4105 | 0.3678 |
| Pancreas | 1 | 0.2582 | 0.2031 | 0.2582 | 0.3492 |
| Pleura | 1 | 0.1804 | 0.1802 | 0.1804 | 0.2145 |
| Skin | 1 | 0.1663 | 0.1197 | 0.1663 | 0.1767 |
| Uterus | 1 | -0.0064 | -0.1712 | -0.0064 | -0.1062 |

## Training

`train/best.pt` is the model before its first update (the prior alone); 5 epochs trained. At that point the selective Spearman is 0.1967 on the training diagnostic lines and 0.2290 on the validation lines.
