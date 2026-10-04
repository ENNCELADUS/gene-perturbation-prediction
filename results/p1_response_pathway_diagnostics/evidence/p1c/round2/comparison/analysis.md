# P1-C fold comparison

| label | folds | kept_folds | pooled_external_delta | pooled_external_ci_low | pooled_external_ci_high | pooled_ratio | pooled_ratio_ci_low | pooled_ratio_ci_high | kept_all |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 4 | 0 | 20.5812 | 20.5624 | 20.5998 | 25.7299 | 25.3263 | 26.1651 | False |
| V1 | 4 | 1 | 0.0459936 | 0.0439815 | 0.0480177 | 1.08838 | 1.08551 | 1.09129 | False |
| V2 | 4 | 0 | 8.34884 | 8.33621 | 8.3623 | 10.4892 | 10.3218 | 10.6615 | False |
| V2-null | 4 | 0 | 1.71186 | 1.70515 | 1.71896 | 3.87797 | 3.80541 | 3.95124 | False |

The verdict is `kept_all`: legs (a) and (c) on every fold plus a pooled held-out ratio whose interval lies below 1. The per-fold `kept` and `kept_b` columns in `summary.csv` are reported diagnostics.

Single training seed 0. Jurkat was observed before this design was registered, so its fold is a diagnostic re-evaluation, not a held-out test. The four leave-one-anchor-out folds are related diagnostics on the same four lines, not independent contexts. ST/Tx1 pretraining exposure remains unresolved. Response results are not GeneEffect evidence.
