# P1-C fold comparison

| label | folds | kept_folds | pooled_external_delta | pooled_external_ci_low | pooled_external_ci_high | pooled_ratio | pooled_ratio_ci_low | pooled_ratio_ci_high | kept_all |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 4 | 0 | 320.34 | 318.477 | 322.235 | 17.3279 | 16.8108 | 17.8792 | False |
| V1 | 4 | 0 | 0.519102 | 0.449451 | 0.579687 | 1.03063 | 1.02797 | 1.03313 | False |
| V2 | 4 | 0 | 200.37 | 199.239 | 201.54 | 11.9982 | 11.6264 | 12.392 | False |
| V2-null | 4 | 0 | 326.191 | 324.616 | 327.751 | 18.192 | 17.7183 | 18.7208 | False |

The verdict is `kept_all`: legs (a) and (c) on every fold plus a pooled held-out ratio whose interval lies below 1. The per-fold `kept` and `kept_b` columns in `summary.csv` are reported diagnostics.

Single training seed 0. Jurkat was observed before this design was registered, so its fold is a diagnostic re-evaluation, not a held-out test. The four leave-one-anchor-out folds are related diagnostics on the same four lines, not independent contexts. ST/Tx1 pretraining exposure remains unresolved. Response results are not GeneEffect evidence.
