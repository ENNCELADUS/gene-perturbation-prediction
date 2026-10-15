# Noise diagnosis `noise_lines_20261014`

Prior run `outputs/context_prior/default_prior_final_split_20261012`. Selective Spearman is the macro mean over the selective genes; gains are 95% paired line bootstraps. Nothing is chosen here; nothing here is synthetic-lethality evidence.

## Learning curve

The reference row refitted on nested patient-grouped fractions of the training side (lines: training side / labelled / paired), each beside its gene mean (the prior's intercept on the fraction's labelled lines). Within-gene R² is pooled over the selective genes against the training gene mean (which scores 0); gains are 95% paired line bootstraps.

Line-specific part:

| Fraction | Lines | Split | Selective Spearman | Gain over full | Within-gene R² | Gene mean R² | R² over gene mean | R² gain over full | AUPR lift | Residual Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.125 | 173 / 126 / 41 | val | 0.1132 | -0.0913 [-0.1151, -0.0586] | -0.0000 | -0.0108 | 0.0108 [0.0027, 0.0184] | -0.0429 [-0.0585, -0.0283] | 0.1437 | 0.1171 |
| 0.125 | 173 / 126 / 41 | test | 0.1331 | -0.0694 [-0.0857, -0.0476] | 0.0130 | -0.0037 | 0.0167 [0.0077, 0.0265] | -0.0293 [-0.0399, -0.0193] | 0.1514 | 0.1413 |
| 0.25 | 354 / 251 / 86 | val | 0.1617 | -0.0428 [-0.0625, -0.0196] | 0.0256 | -0.0062 | 0.0317 [0.0258, 0.0376] | -0.0173 [-0.0250, -0.0101] | 0.1627 | 0.1719 |
| 0.25 | 354 / 251 / 86 | test | 0.1412 | -0.0613 [-0.0760, -0.0420] | 0.0220 | -0.0019 | 0.0239 [0.0162, 0.0335] | -0.0203 [-0.0261, -0.0152] | 0.1494 | 0.1547 |
| 0.5 | 708 / 486 / 165 | val | 0.1778 | -0.0267 [-0.0353, -0.0157] | 0.0342 | -0.0051 | 0.0394 [0.0306, 0.0492] | -0.0086 [-0.0131, -0.0041] | 0.1754 | 0.1906 |
| 0.5 | 708 / 486 / 165 | test | 0.1753 | -0.0272 [-0.0374, -0.0149] | 0.0351 | 0.0005 | 0.0346 [0.0255, 0.0456] | -0.0072 [-0.0109, -0.0037] | 0.1685 | 0.1938 |
| 0.75 | 1,060 / 721 / 246 | val | 0.1930 | -0.0115 [-0.0198, -0.0033] | 0.0385 | -0.0034 | 0.0420 [0.0344, 0.0504] | -0.0043 [-0.0066, -0.0019] | 0.1847 | 0.2094 |
| 0.75 | 1,060 / 721 / 246 | test | 0.1903 | -0.0122 [-0.0200, -0.0036] | 0.0394 | 0.0007 | 0.0386 [0.0296, 0.0492] | -0.0029 [-0.0054, -0.0004] | 0.1760 | 0.2132 |
| 0.875 | 1,232 / 839 / 297 | val | 0.2051 | 0.0006 [-0.0050, 0.0062] | 0.0427 | -0.0026 | 0.0453 [0.0366, 0.0553] | -0.0001 [-0.0019, 0.0019] | 0.1950 | 0.2230 |
| 0.875 | 1,232 / 839 / 297 | test | 0.2007 | -0.0018 [-0.0067, 0.0034] | 0.0414 | 0.0015 | 0.0399 [0.0310, 0.0501] | -0.0009 [-0.0022, 0.0006] | 0.1814 | 0.2205 |
| 1 | 1,399 / 947 / 343 | val | 0.2045 | 0.0000 [0.0000, 0.0000] | 0.0428 | -0.0026 | 0.0454 [0.0366, 0.0555] | 0.0000 [0.0000, 0.0000] | 0.1963 | 0.2220 |
| 1 | 1,399 / 947 / 343 | test | 0.2025 | 0.0000 [0.0000, 0.0000] | 0.0423 | 0.0022 | 0.0401 [0.0314, 0.0499] | 0.0000 [0.0000, 0.0000] | 0.1827 | 0.2221 |

Whole prediction (per-line Pearson across genes; RMSE over every pair):

| Fraction | Split | Absolute Pearson | Gene mean | RMSE | Gene mean RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.125 | val | 0.9096 | 0.9088 | 0.1814 | 0.1822 |
| 0.125 | test | 0.9119 | 0.9106 | 0.1788 | 0.1801 |
| 0.25 | val | 0.9119 | 0.9091 | 0.1792 | 0.1819 |
| 0.25 | test | 0.9131 | 0.9108 | 0.1778 | 0.1799 |
| 0.5 | val | 0.9127 | 0.9092 | 0.1785 | 0.1818 |
| 0.5 | test | 0.9142 | 0.9110 | 0.1767 | 0.1797 |
| 0.75 | val | 0.9132 | 0.9094 | 0.1780 | 0.1816 |
| 0.75 | test | 0.9147 | 0.9111 | 0.1762 | 0.1797 |
| 0.875 | val | 0.9135 | 0.9094 | 0.1777 | 0.1816 |
| 0.875 | test | 0.9148 | 0.9111 | 0.1761 | 0.1796 |
| 1 | val | 0.9136 | 0.9095 | 0.1776 | 0.1816 |
| 1 | test | 0.9149 | 0.9112 | 0.1760 | 0.1795 |

Cilantro-SL's stage one (as released, trained on every labelled line, so in-sample on these lines) against the full prior and its gene mean, all on the validation and test pairs stage one covers:

| Predictor | Split | Pairs | Selective Spearman | Within-gene R² | Absolute Pearson | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cilantro | val | 49,509 (20,503 selective) | 0.1692 (1252 genes) | 0.0391 | 0.9326 | 0.2298 |
| prior | val | 49,509 (20,503 selective) | 0.2153 (1262 genes) | 0.0518 | 0.9277 | 0.2387 |
| gene_mean | val | 49,509 (20,503 selective) | undefined (0 genes) | -0.0031 | 0.9234 | 0.2453 |
| cilantro | test | 53,272 (21,911 selective) | 0.1963 (1270 genes) | 0.0491 | 0.9316 | 0.2296 |
| prior | test | 53,272 (21,911 selective) | 0.2210 (1285 genes) | 0.0508 | 0.9260 | 0.2393 |
| gene_mean | test | 53,272 (21,911 selective) | undefined (0 genes) | 0.0032 | 0.9222 | 0.2451 |

| Split | Cilantro R² over gene mean | Prior R² over Cilantro |
| --- | ---: | ---: |
| val | 0.0423 [0.0150, 0.0659] | 0.0127 [-0.0095, 0.0375] |
| test | 0.0458 [0.0275, 0.0622] | 0.0018 [-0.0154, 0.0215] |
