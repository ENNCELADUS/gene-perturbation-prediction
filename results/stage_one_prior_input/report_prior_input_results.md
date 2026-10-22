# Stage-one report: prior_input (6 rows)

Scoring units: every prediction is GeneEffect re-centred on the protocol's 377-line mean (`(m_g - mu_g) / sigma_g + z`); metrics as protocol §6 and §14 (within-gene R² pooled over the selective genes; calibration slope: measured residual on predicted, each gene's offset removed). Validation 27 lines, test 28, out of fold 377 single-cell training lines (each scored by the run without its fold; each fold's base differs, so the rank metrics are scored within each fold and averaged, and within-gene R², RMSE and per-line Pearson are pooled). The per-line residual Spearman (the selector, protocol §6) correlates each line's predicted and measured residuals across the 4,323 selective genes and averages over lines; it is pooled out of fold, as each line is compared with its own truth. Selective Spearman covers the 4,323 protocol genes (genes scored: those defined), the panel column the selective genes in the decode panel; a contrast's Spearman difference covers the genes defined on both sides. `undefined`: a prediction constant within every gene, or a run not finished. Nothing is chosen here; this is not SL evidence.

## Validation

| Row | Per-line residual Spearman | Lines scored | Selective Spearman | Genes scored | Within-gene R² | Selective Spearman, panel | AUPR lift | Residual Pearson | Per-line Pearson | RMSE | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.0098 | 27.0000 | undefined | 0.0000 | -0.0024 | undefined | 0.0000 | undefined | 0.9095 | 0.1815 | undefined |
| prior | 0.2146 | 27.0000 | 0.1938 | 4323.0000 | 0.0462 | 0.1937 | 0.1809 | 0.2120 | 0.9140 | 0.1773 | 1.4896 |
| film dE | 0.1457 | 27.0000 | 0.1289 | 4269.0000 | -0.0092 | 0.1289 | 0.1489 | 0.1541 | 0.9098 | 0.1813 | 0.5398 |
| film prior | 0.1988 | 27.0000 | 0.1937 | 4269.0000 | 0.0373 | 0.1937 | 0.1784 | 0.2121 | 0.9133 | 0.1779 | 1.7645 |
| film prior dE | 0.2033 | 27.0000 | 0.1862 | 4269.0000 | 0.0452 | 0.1862 | 0.1740 | 0.2090 | 0.9140 | 0.1773 | 0.9455 |
| permuted film prior dE | 0.1972 | 27.0000 | 0.1944 | 4269.0000 | 0.0368 | 0.1944 | 0.1775 | 0.2115 | 0.9134 | 0.1779 | 1.7237 |

## Test

| Row | Per-line residual Spearman | Lines scored | Selective Spearman | Genes scored | Within-gene R² | Selective Spearman, panel | AUPR lift | Residual Pearson | Per-line Pearson | RMSE | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.0475 | 28.0000 | undefined | 0.0000 | 0.0017 | undefined | 0.0000 | undefined | 0.9112 | 0.1795 | undefined |
| prior | 0.2283 | 28.0000 | 0.2001 | 4323.0000 | 0.0474 | 0.2004 | 0.1840 | 0.2227 | 0.9154 | 0.1755 | 1.5227 |
| film dE | 0.1531 | 28.0000 | 0.1476 | 4269.0000 | -0.0165 | 0.1476 | 0.1469 | 0.1643 | 0.9114 | 0.1794 | 0.5294 |
| film prior | 0.2144 | 28.0000 | 0.2005 | 4269.0000 | 0.0375 | 0.2005 | 0.1808 | 0.2234 | 0.9149 | 0.1760 | 1.8242 |
| film prior dE | 0.1954 | 28.0000 | 0.1808 | 4269.0000 | 0.0335 | 0.1808 | 0.1635 | 0.2018 | 0.9147 | 0.1763 | 0.8825 |
| permuted film prior dE | 0.2122 | 28.0000 | 0.1996 | 4269.0000 | 0.0366 | 0.1996 | 0.1808 | 0.2226 | 0.9149 | 0.1760 | 1.7880 |

## Out of fold (telemetry)

| Row | Per-line residual Spearman | Lines scored | Selective Spearman | Genes scored | Within-gene R² | Selective Spearman, panel | AUPR lift | Residual Pearson | Per-line Pearson | RMSE | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | -0.0111 | 377.0000 | undefined | 0.0000 | -0.0063 | undefined | 0.0000 | undefined | 0.9128 | 0.1775 | undefined |
| prior | 0.1920 | 377.0000 | 0.1928 | 4323.0000 | 0.0417 | 0.1931 | 0.1350 | 0.2152 | 0.9170 | 0.1733 | 1.2998 |
| film dE | 0.1290 | 377.0000 | 0.1281 | 4269.0000 | -0.0422 | 0.1281 | 0.1044 | 0.1437 | 0.9111 | 0.1794 | 0.4448 |
| film prior | 0.1783 | 377.0000 | 0.1931 | 4269.0000 | 0.0313 | 0.1931 | 0.1331 | 0.2158 | 0.9165 | 0.1740 | 1.5993 |
| film prior dE | 0.1816 | 377.0000 | 0.1871 | 4269.0000 | 0.0249 | 0.1871 | 0.1314 | 0.2085 | 0.9162 | 0.1744 | 0.7971 |
| permuted film prior dE | 0.1789 | 377.0000 | 0.1917 | 4269.0000 | 0.0319 | 0.1917 | 0.1329 | 0.2148 | 0.9167 | 0.1739 | 1.4882 |

## Contrasts (left minus right; 95% paired line-bootstrap intervals)

| Left | Right | Val per-line Spearman | Test per-line Spearman | Val selective Spearman | Test selective Spearman | Val within-gene R² | Test within-gene R² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| film prior dE | film prior | 0.0046 [-0.0116, 0.0202] | -0.0189 [-0.0386, -0.0016] | -0.0074 [-0.0199, 0.0052] (4269 genes) | -0.0197 [-0.0329, -0.0054] (4269 genes) | 0.0080 [-0.0026, 0.0195] | -0.0040 [-0.0210, 0.0105] |
| film prior dE | permuted film prior dE | 0.0062 [-0.0100, 0.0219] | -0.0167 [-0.0365, 0.0002] | -0.0082 [-0.0204, 0.0045] (4269 genes) | -0.0188 [-0.0317, -0.0045] (4269 genes) | 0.0085 [-0.0022, 0.0202] | -0.0031 [-0.0201, 0.0117] |
| film prior dE | prior | -0.0113 [-0.0275, 0.0051] | -0.0329 [-0.0537, -0.0143] | -0.0075 [-0.0200, 0.0051] (4269 genes) | -0.0197 [-0.0328, -0.0053] (4269 genes) | -0.0010 [-0.0110, 0.0094] | -0.0139 [-0.0298, -0.0002] |
| film prior | prior | -0.0159 [-0.0219, -0.0098] | -0.0140 [-0.0184, -0.0095] | -0.0000 [-0.0001, 0.0001] (4269 genes) | 0.0000 [-0.0001, 0.0002] (4269 genes) | -0.0089 [-0.0121, -0.0058] | -0.0099 [-0.0125, -0.0075] |
| film prior dE | film dE | 0.0577 [0.0353, 0.0825] | 0.0423 [0.0100, 0.0745] | 0.0573 [0.0370, 0.0739] (4269 genes) | 0.0332 [0.0044, 0.0599] (4269 genes) | 0.0544 [0.0342, 0.0803] | 0.0500 [0.0318, 0.0706] |

## Training (selected epoch, and the gap from training to validation)

| Row | Best epoch | Epochs | Updates | Lines trained | Train per-line Spearman | Val per-line Spearman | Train selective Spearman | Val selective Spearman | Train within-gene R² | Val within-gene R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| film dE | 16 | 21 | 23184 | 981 | not recorded | 0.1457 | 0.3525 | 0.1289 | 0.1448 | -0.0092 |
| film prior | 1 | 6 | 6624 | 981 | 0.1822 | 0.1988 | 0.1806 | 0.1937 | 0.0343 | 0.0373 |
| film prior dE | 1 | 6 | 6624 | 981 | 0.2310 | 0.2033 | 0.2226 | 0.1862 | 0.0715 | 0.0452 |
| permuted film prior dE | 1 | 6 | 6624 | 981 | 0.1810 | 0.1972 | 0.1804 | 0.1944 | 0.0335 | 0.0368 |
