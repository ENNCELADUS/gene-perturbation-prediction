# Stage-one report: gene_reader (10 rows)

Scoring units: every prediction is GeneEffect re-centred on the protocol's 377-line mean (`(m_g - mu_g) / sigma_g + z`); metrics as protocol §6 and §14 (within-gene R² pooled over the selective genes; calibration slope: measured residual on predicted, each gene's offset removed). Validation 27 lines, test 28, out of fold 384 single-cell training lines (each scored by the run without its fold; each fold's base differs, so the rank metrics are scored within each fold and averaged, and within-gene R², RMSE and per-line Pearson are pooled). The per-line residual Spearman (the selector, protocol §6) correlates each line's predicted and measured residuals across the 4,323 selective genes and averages over lines; it is pooled out of fold, as each line is compared with its own truth. Selective Spearman covers the 4,323 protocol genes (genes scored: those defined), the panel column the selective genes in the decode panel; a contrast's Spearman difference covers the genes defined on both sides. `undefined`: a prediction constant within every gene, or a run not finished. Nothing is chosen here; this is not SL evidence.

## Validation

| Row | Per-line residual Spearman | Lines scored | Selective Spearman | Genes scored | Within-gene R² | Selective Spearman, panel | AUPR lift | Residual Pearson | Per-line Pearson | RMSE | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.0177 | 27.0000 | undefined | 0.0000 | -0.0015 | undefined | 0.0000 | undefined | 0.9089 | 0.1822 | undefined |
| linear model | 0.2261 | 27.0000 | 0.2013 | 4344.0000 | 0.0495 | 0.2013 | 0.1828 | 0.2191 | 0.9136 | 0.1777 | 1.5108 |
| reader on components | 0.2261 | 27.0000 | 0.2013 | 4344.0000 | 0.0495 | 0.2013 | 0.1828 | 0.2392 | 0.9118 | 0.1795 | 1.5108 |
| permuted reader on components | 0.2261 | 27.0000 | 0.2013 | 4344.0000 | 0.0495 | 0.2013 | 0.1828 | 0.2392 | 0.9118 | 0.1795 | 1.5108 |
| reader on expression | 0.2261 | 27.0000 | 0.2013 | 4344.0000 | 0.0495 | 0.2013 | 0.1828 | 0.2392 | 0.9118 | 0.1795 | 1.5108 |
| permuted reader on expression | 0.2261 | 27.0000 | 0.2013 | 4344.0000 | 0.0495 | 0.2013 | 0.1828 | 0.2392 | 0.9118 | 0.1795 | 1.5108 |
| reader on components with strong decay | 0.2278 | 27.0000 | 0.1969 | 4344.0000 | 0.0548 | 0.1971 | 0.1788 | 0.2411 | 0.9122 | 0.1791 | 1.1896 |
| permuted reader on components with strong decay | 0.2261 | 27.0000 | 0.2013 | 4344.0000 | 0.0495 | 0.2013 | 0.1828 | 0.2392 | 0.9118 | 0.1795 | 1.5108 |
| reader on expression with strong decay | 0.2261 | 27.0000 | 0.2013 | 4344.0000 | 0.0495 | 0.2013 | 0.1828 | 0.2392 | 0.9118 | 0.1795 | 1.5108 |
| permuted reader on expression with strong decay | 0.2261 | 27.0000 | 0.2013 | 4344.0000 | 0.0495 | 0.2013 | 0.1828 | 0.2392 | 0.9118 | 0.1795 | 1.5108 |

## Test

| Row | Per-line residual Spearman | Lines scored | Selective Spearman | Genes scored | Within-gene R² | Selective Spearman, panel | AUPR lift | Residual Pearson | Per-line Pearson | RMSE | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.0544 | 28.0000 | undefined | 0.0000 | 0.0025 | undefined | 0.0000 | undefined | 0.9107 | 0.1799 | undefined |
| linear model | 0.2266 | 28.0000 | 0.1964 | 4344.0000 | 0.0482 | 0.1966 | 0.1821 | 0.2192 | 0.9150 | 0.1758 | 1.5187 |
| reader on components | 0.2266 | 28.0000 | 0.1964 | 4344.0000 | 0.0482 | 0.1966 | 0.1821 | 0.2367 | 0.9135 | 0.1773 | 1.5187 |
| permuted reader on components | 0.2266 | 28.0000 | 0.1964 | 4344.0000 | 0.0482 | 0.1966 | 0.1821 | 0.2367 | 0.9135 | 0.1773 | 1.5187 |
| reader on expression | 0.2266 | 28.0000 | 0.1964 | 4344.0000 | 0.0482 | 0.1966 | 0.1821 | 0.2367 | 0.9135 | 0.1773 | 1.5187 |
| permuted reader on expression | 0.2266 | 28.0000 | 0.1964 | 4344.0000 | 0.0482 | 0.1966 | 0.1821 | 0.2367 | 0.9135 | 0.1773 | 1.5187 |
| reader on components with strong decay | 0.2311 | 28.0000 | 0.1996 | 4344.0000 | 0.0535 | 0.1998 | 0.1780 | 0.2372 | 0.9138 | 0.1770 | 1.2137 |
| permuted reader on components with strong decay | 0.2266 | 28.0000 | 0.1964 | 4344.0000 | 0.0482 | 0.1966 | 0.1821 | 0.2367 | 0.9135 | 0.1773 | 1.5187 |
| reader on expression with strong decay | 0.2266 | 28.0000 | 0.1964 | 4344.0000 | 0.0482 | 0.1966 | 0.1821 | 0.2367 | 0.9135 | 0.1773 | 1.5187 |
| permuted reader on expression with strong decay | 0.2266 | 28.0000 | 0.1964 | 4344.0000 | 0.0482 | 0.1966 | 0.1821 | 0.2367 | 0.9135 | 0.1773 | 1.5187 |

## Out of fold (telemetry)

| Row | Per-line residual Spearman | Lines scored | Selective Spearman | Genes scored | Within-gene R² | Selective Spearman, panel | AUPR lift | Residual Pearson | Per-line Pearson | RMSE | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | -0.0112 | 384.0000 | undefined | 0.0000 | -0.0063 | undefined | 0.0000 | undefined | 0.9128 | 0.1775 | undefined |
| linear model | 0.1911 | 384.0000 | 0.1907 | 4344.0000 | 0.0418 | 0.1910 | 0.1302 | 0.2141 | 0.9171 | 0.1733 | 1.2621 |
| reader on components | 0.1896 | 384.0000 | 0.1875 | 4344.0000 | 0.0407 | 0.1878 | 0.1290 | 0.2306 | 0.9157 | 0.1747 | 1.1653 |
| permuted reader on components | 0.1911 | 384.0000 | 0.1907 | 4344.0000 | 0.0418 | 0.1910 | 0.1302 | 0.2347 | 0.9158 | 0.1746 | 1.2621 |
| reader on expression | 0.1890 | 384.0000 | 0.1883 | 4344.0000 | 0.0390 | 0.1887 | 0.1277 | 0.2312 | 0.9156 | 0.1747 | 1.1273 |
| permuted reader on expression | 0.1911 | 384.0000 | 0.1907 | 4344.0000 | 0.0418 | 0.1910 | 0.1302 | 0.2347 | 0.9158 | 0.1746 | 1.2621 |
| reader on components with strong decay | 0.1896 | 384.0000 | 0.1871 | 4344.0000 | 0.0408 | 0.1874 | 0.1290 | 0.2304 | 0.9157 | 0.1747 | 1.1574 |
| permuted reader on components with strong decay | 0.1911 | 384.0000 | 0.1907 | 4344.0000 | 0.0418 | 0.1910 | 0.1302 | 0.2347 | 0.9158 | 0.1746 | 1.2621 |
| reader on expression with strong decay | 0.1890 | 384.0000 | 0.1883 | 4344.0000 | 0.0393 | 0.1887 | 0.1285 | 0.2315 | 0.9157 | 0.1747 | 1.1324 |
| permuted reader on expression with strong decay | 0.1911 | 384.0000 | 0.1907 | 4344.0000 | 0.0418 | 0.1910 | 0.1302 | 0.2347 | 0.9158 | 0.1746 | 1.2621 |

## Contrasts (left minus right; 95% paired line-bootstrap intervals)

| Left | Right | Val per-line Spearman | Test per-line Spearman | Val selective Spearman | Test selective Spearman | Val within-gene R² | Test within-gene R² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reader on components | linear model | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] |
| reader on components | permuted reader on components | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| reader on expression | linear model | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] |
| reader on expression | permuted reader on expression | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| reader on components with strong decay | linear model | 0.0018 [-0.0157, 0.0221] | 0.0045 [-0.0106, 0.0222] | -0.0044 [-0.0199, 0.0135] (4344 genes) | 0.0032 [-0.0113, 0.0189] (4344 genes) | 0.0052 [-0.0038, 0.0156] | 0.0053 [-0.0036, 0.0151] |
| reader on components with strong decay | permuted reader on components with strong decay | 0.0018 [-0.0157, 0.0221] | 0.0045 [-0.0106, 0.0222] | -0.0044 [-0.0199, 0.0135] (4344 genes) | 0.0032 [-0.0113, 0.0189] (4344 genes) | 0.0052 [-0.0038, 0.0156] | 0.0053 [-0.0036, 0.0151] |
| reader on expression with strong decay | linear model | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] |
| reader on expression with strong decay | permuted reader on expression with strong decay | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [0.0000, 0.0000] (4344 genes) | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

## Training (selected epoch, and the gap from training to validation)

| Row | Best epoch | Epochs | Updates | Lines trained | Train per-line Spearman | Val per-line Spearman | Train selective Spearman | Val selective Spearman | Train within-gene R² | Val within-gene R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reader on components | 0 | 5 | 1360 | 987 | 0.1911 | 0.2261 | 0.1785 | 0.2013 | 0.0418 | 0.0495 |
| permuted reader on components | 0 | 5 | 1360 | 987 | 0.1911 | 0.2261 | 0.1785 | 0.2013 | 0.0418 | 0.0495 |
| reader on expression | 0 | 5 | 1360 | 987 | 0.1911 | 0.2261 | 0.1785 | 0.2013 | 0.0418 | 0.0495 |
| permuted reader on expression | 0 | 5 | 1360 | 987 | 0.1911 | 0.2261 | 0.1785 | 0.2013 | 0.0418 | 0.0495 |
| reader on components with strong decay | 2 | 7 | 1904 | 987 | 0.2446 | 0.2278 | 0.2317 | 0.1969 | 0.0678 | 0.0548 |
| permuted reader on components with strong decay | 0 | 5 | 1360 | 987 | 0.1911 | 0.2261 | 0.1785 | 0.2013 | 0.0418 | 0.0495 |
| reader on expression with strong decay | 0 | 5 | 1360 | 987 | 0.1911 | 0.2261 | 0.1785 | 0.2013 | 0.0418 | 0.0495 |
| permuted reader on expression with strong decay | 0 | 5 | 1360 | 987 | 0.1911 | 0.2261 | 0.1785 | 0.2013 | 0.0418 | 0.0495 |
