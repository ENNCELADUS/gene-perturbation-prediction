# Learning curve over DepMap lines, seed 0

Run `noise_lines_20261014` of 2026-10-14 on the H20 container on port 30838, CPU only, branch `feat/line-curve` at
`8de5f1f`: `hpc/run.sh prior-noise configs/context_prior/noise_diagnosis.yaml --run-id noise_lines_20261014 --sections
learning_curve`, against the final split's default prior (`default_prior_final_split_20261012`: 0.2045 validation /
0.2025 test, [record](../final_split_seed0/README.md)). Design: [noise diagnosis §3.2](../../docs/specs/2026-10-12-noise-diagnosis-design.md);
protocol §14. Tables: [results](noise_lines_20261014.results.md), [JSON](noise_lines_20261014.learning_curve.json).
Cilantro-SL's predictions: `hpc/cilantro/stage_one_predictions.py` on the released `as_released_gc95M` stage one
(`outputs/cilantro_analysis/stage_one_split_lines.csv` on the H20; 102,972 pairs, 4,681 genes, 54 of the 55 validation
and test lines). Nothing here is synthetic-lethality evidence.

**Question.** Does using up DepMap's labelled lines make the prior predict GeneEffect more accurately, above all in the
part it adjusts per line? One patient order (seed 0); the variance from which lines are dropped is not measured.

## Line-specific part

Gene mean is the prior's intercept refitted on the fraction's labelled lines. Within-gene R² is pooled over the
selective genes in residual-SD units against the 377-line training gene mean (which scores 0). Gains are 95% paired line
bootstraps; "over full" is the row minus the full prior.

| Fraction | Lines (side / labelled) | Split | Selective Spearman | Over full | Within-gene R² | Gene mean R² | R² over gene mean | R² over full | AUPR lift | Residual Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/8 | 173 / 126 | val | 0.1132 | −0.0913 [−0.1151, −0.0586] | −0.0000 | −0.0108 | +0.0108 [+0.0027, +0.0184] | −0.0429 [−0.0585, −0.0283] | 0.1437 | 0.1171 |
| 1/8 | | test | 0.1331 | −0.0694 [−0.0857, −0.0476] | 0.0130 | −0.0037 | +0.0167 [+0.0077, +0.0265] | −0.0293 [−0.0399, −0.0193] | 0.1514 | 0.1413 |
| 1/4 | 354 / 251 | val | 0.1617 | −0.0428 [−0.0625, −0.0196] | 0.0256 | −0.0062 | +0.0317 [+0.0258, +0.0376] | −0.0173 [−0.0250, −0.0101] | 0.1627 | 0.1719 |
| 1/4 | | test | 0.1412 | −0.0613 [−0.0760, −0.0420] | 0.0220 | −0.0019 | +0.0239 [+0.0162, +0.0335] | −0.0203 [−0.0261, −0.0152] | 0.1494 | 0.1547 |
| 1/2 | 708 / 486 | val | 0.1778 | −0.0267 [−0.0353, −0.0157] | 0.0342 | −0.0051 | +0.0394 [+0.0306, +0.0492] | −0.0086 [−0.0131, −0.0041] | 0.1754 | 0.1906 |
| 1/2 | | test | 0.1753 | −0.0272 [−0.0374, −0.0149] | 0.0351 | 0.0005 | +0.0346 [+0.0255, +0.0456] | −0.0072 [−0.0109, −0.0037] | 0.1685 | 0.1938 |
| 3/4 | 1,060 / 721 | val | 0.1930 | −0.0115 [−0.0198, −0.0033] | 0.0385 | −0.0034 | +0.0420 [+0.0344, +0.0504] | −0.0043 [−0.0066, −0.0019] | 0.1847 | 0.2094 |
| 3/4 | | test | 0.1903 | −0.0122 [−0.0200, −0.0036] | 0.0394 | 0.0007 | +0.0386 [+0.0296, +0.0492] | −0.0029 [−0.0054, −0.0004] | 0.1760 | 0.2132 |
| 7/8 | 1,232 / 839 | val | 0.2051 | +0.0006 [−0.0050, +0.0062] | 0.0427 | −0.0026 | +0.0453 [+0.0366, +0.0553] | −0.0001 [−0.0019, +0.0019] | 0.1950 | 0.2230 |
| 7/8 | | test | 0.2007 | −0.0018 [−0.0067, +0.0034] | 0.0414 | 0.0015 | +0.0399 [+0.0310, +0.0501] | −0.0009 [−0.0022, +0.0006] | 0.1814 | 0.2205 |
| all | 1,399 / 947 | val | 0.2045 | | 0.0428 | −0.0026 | +0.0454 [+0.0366, +0.0555] | | 0.1963 | 0.2220 |
| all | | test | 0.2025 | | 0.0423 | 0.0022 | +0.0401 [+0.0314, +0.0499] | | 0.1827 | 0.2221 |

## Whole prediction

Per-line Pearson across genes and RMSE over every pair, prior against its gene mean: 0.9096 / 0.9088 at 1/8 to 0.9136 /
0.9095 at all lines on validation (test 0.9119 / 0.9106 to 0.9149 / 0.9112); RMSE 0.1814 / 0.1822 to 0.1776 / 0.1816
(test 0.1788 / 0.1801 to 0.1760 / 0.1795).

## Cilantro-SL's stage one on its covered pairs

In-sample (it trained on every labelled line, these included), so an upper bound for it; scored with the full prior and
its gene mean on the pairs it covers (a gene among the line's top 2,048 expressed).

| Predictor | Split | Pairs (selective) | Selective Spearman (genes) | Within-gene R² | Absolute Pearson | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cilantro | val | 49,509 (20,503) | 0.1692 (1,252) | 0.0391 | 0.9326 | 0.2298 |
| prior | val | | 0.2153 (1,262) | 0.0518 | 0.9277 | 0.2387 |
| gene mean | val | | undefined | −0.0031 | 0.9234 | 0.2453 |
| Cilantro | test | 53,272 (21,911) | 0.1963 (1,270) | 0.0491 | 0.9316 | 0.2296 |
| prior | test | | 0.2210 (1,285) | 0.0508 | 0.9260 | 0.2393 |
| gene mean | test | | undefined | 0.0032 | 0.9222 | 0.2451 |

Cilantro's R² over the gene mean +0.0423 [+0.0150, +0.0659] / +0.0458 [+0.0275, +0.0622]; the prior's over Cilantro
+0.0127 [−0.0095, +0.0375] / +0.0018 [−0.0154, +0.0215].

## Reading

- **More DepMap lines no longer help.** The curve rises steeply to 3/4 and then flattens: the last eighth (167
  training-side lines, 108 labelled) changes selective Spearman by −0.0006 [−0.0062, +0.0050] validation / +0.0018
  [−0.0034, +0.0067] test and within-gene R² by +0.0001 / +0.0009, every interval spanning zero. The earlier record's
  "still rises" came from the 3/4 → all step, which is all in 3/4 → 7/8.
- **The per-line adjustment is right in size, but small.** Within-gene R² is above the row's gene mean at every
  fraction (intervals above zero), and the full prior explains 4.3% / 4.2% of the selective genes' line-specific
  variance.
- **Whole-prediction metrics hide it.** Absolute Pearson and RMSE move in the third decimal; they measure the gene mean.
- **Cilantro does not beat the prior even in-sample.** On its covered pairs it carries line-specific signal (R² over the
  gene mean above zero) but scores lower selective Spearman and R², with the R² difference's interval spanning zero. Its
  higher absolute Pearson and lower RMSE come with a gene mean fitted on these lines.
