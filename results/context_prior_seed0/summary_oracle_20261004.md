# Linear context prior, run oracle_20261004

Selective Spearman is the macro mean over the selective genes of the Spearman across lines of the residual; intervals are 95% paired line bootstraps (1,000 resamples, seed 0) of the gain. Validation chooses; the chosen prior is scored once on test. Nothing here is synthetic-lethality evidence.

## Learning curve

Validation selective Spearman. `oracle` scores validation lines' bulk RNA (off-contract upper bound), `val` bridged pseudo-bulk; ridge penalties are chosen on validation per input.

| Point | Lines | Config | Input | Penalties | Score |
| --- | --- | --- | --- | --- | --- |
| single_cell_train | 167 | components | oracle | 100.0 | 0.1738 |
| single_cell_train | 167 | components_and_selected | oracle | 100.0, 100.0 | 0.1843 |
| random_100_0 | 100 | components | oracle | 10.0 | 0.1015 |
| random_100_0 | 100 | components_and_selected | oracle | 10.0, 10.0 | 0.1248 |
| random_100_1 | 100 | components | oracle | 10.0 | 0.1074 |
| random_100_1 | 100 | components_and_selected | oracle | 10.0, 10.0 | 0.1276 |
| random_100_2 | 100 | components | oracle | 10.0 | 0.1276 |
| random_100_2 | 100 | components_and_selected | oracle | 10.0, 10.0 | 0.1461 |
| random_400_0 | 400 | components | oracle | 1.0 | 0.1824 |
| random_400_0 | 400 | components_and_selected | oracle | 1.0, 10.0 | 0.1914 |
| random_400_1 | 400 | components | oracle | 10.0 | 0.1728 |
| random_400_1 | 400 | components_and_selected | oracle | 10.0, 100.0 | 0.1893 |
| random_400_2 | 400 | components | oracle | 10.0 | 0.2076 |
| random_400_2 | 400 | components_and_selected | oracle | 10.0, 100.0 | 0.2222 |
| random_700_0 | 700 | components | oracle | 1.0 | 0.2143 |
| random_700_0 | 700 | components_and_selected | oracle | 1.0, 10.0 | 0.2240 |
| random_700_1 | 700 | components | oracle | 1.0 | 0.2116 |
| random_700_1 | 700 | components_and_selected | oracle | 1.0, 10.0 | 0.2239 |
| random_700_2 | 700 | components | oracle | 1.0 | 0.2246 |
| random_700_2 | 700 | components_and_selected | oracle | 1.0, 10.0 | 0.2371 |
| all | 1086 | components | oracle | 0.01 | 0.2364 |
| all | 1086 | components_and_selected | oracle | 0.01, 10.0 | 0.2374 |

## Extra-lines decision

All 1086 labelled training-side lines minus the 167 single-cell training lines, expression components plus data-selected genes, oracle input: 0.0531 [0.0220, 0.0777]. The extra lines pass (not binding (bulk input)).
