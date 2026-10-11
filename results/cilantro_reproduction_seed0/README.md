# Cilantro-SL reproduction (phase 1, as released), seed 0

Run `as_released_gc95M`, finished 2026-10-10 13:20 UTC on the H20 container on port 30838, `hpc/cilantro/run.sh
as_released_gc95M --dictionary gc95M` (five launches resumed by run id, the last at `f4bc6d3`), released Cilantro-SL
code at `c674b55` through the harness of the [reproduction design](../../docs/specs/2026-10-08-cilantro-sl-reproduction-design.md)
(phase 1, §4), 100 epochs, the fast deletion and classifier paths (design §4.4), DepMap 26Q1 in place of the paper's
23Q4. Tables: [report.json](report.json) (the run's `report.md` in JSON); the two analyses below,
[stage_one_vs_gene_mean.json](stage_one_vs_gene_mean.json) and [default_prior_whole_matrix.json](default_prior_whole_matrix.json),
were one-off scripts on the host (`outputs/cilantro_analysis/`). Nothing here is synthetic-lethality evidence for this
project's task.

## What ran

Geneformer `gf-12L-30M-i2048` read the bulk profile of 1,719 tokenised lines with the gc95M dictionary the fork loads
by default (design §3: very likely mismatched with the 30M model; the matched gc30M run has not been launched). A
gene is deleted only in lines where it is among the line's top 2,048 expressed genes: 7,321 of 9,987 genes got a
deletion embedding somewhere, and stage one has 2,169,637 (line, gene) rows over 1,140 lines and 6,670 genes (median
124 lines per gene; 11 genes are in every line's top 2,048). The pair corpus is 22,119 SynLethDB SL pairs (of 35,943),
1,810 experimental non-SL pairs (of 2,899) and 108,785 expression-correlation negatives; each pair is expanded over the lines where
both genes have a row, with its one global label: 12,808,732 (line, pair) rows.

## Results against the paper

| | This run | Paper |
| --- | ---: | ---: |
| Stage-one fit on its own training rows: RMSE / Spearman / R² | 0.229 / 0.851 / 0.868 | 0.230 / 0.864 / 0.85 |
| Gene-holdout two-class AP (5 folds) | 0.542 ± 0.015 | 0.7148 |
| Pair-holdout two-class AP | 0.806 ± 0.007 | — |
| Gene-holdout AUROC; positive-class AUPR minus prior | 0.556; 0.045 | — |
| Pair-holdout AUROC; positive-class AUPR minus prior | 0.823; 0.436 | — |
| Conformal coverage at every α (both regimes) | on the target | on the target |
| Singleton prediction sets at 98% target coverage | 6% gene-holdout, 21% pair-holdout | most |

Stage one and conformal coverage reproduce; the headline gene-holdout result does not (two-class AP 0.54 against
0.71; AUROC 0.556 on unseen genes). The dictionary mismatch and the newer DepMap release are untested candidate
causes.

## Stage one is the gene mean

Over the same 2,169,637 rows, 85% of GeneEffect variance lies between genes:

| Predictor | RMSE | Spearman | R² |
| --- | ---: | ---: | ---: |
| Stage one, on its own training rows | 0.229 | 0.851 | 0.868 |
| Gene mean over all lines | 0.243 | 0.856 | 0.851 |
| Gene mean from other lines (five folds over lines, held out) | 0.244 | 0.854 | 0.850 |

On the line-specific part (each gene's deviation from its mean over lines), stage one explains 16% of the
within-gene variance on its own training rows (per-gene Spearman mean 0.139, median 0.098, over 4,756 genes with
at least 20 lines). The paper scores stage one only on the rows it trained on and never on held-out lines.

For comparison, our default linear context prior on all 17,787 panel genes and every labelled entry: whole-matrix
Spearman 0.732 / 0.715 / 0.733 (training out of fold / validation / test) against 0.730 / 0.708 / 0.727 for the
training-line gene mean ([default_prior_whole_matrix.json](default_prior_whole_matrix.json)). Whole-matrix
agreement is the gene mean in both projects; context accuracy is not.

## How it treats cell lines

- **Labels are global.** The released code reads only the pair names from SynLethDB; the `r.cell_line` field
  (present on 69% of SL rows and every non-SL row) is never read. A pair's label is copied to every line where both
  genes have a stage-one row, so which lines a pair appears in depends on expression alone.
- **No line is held out.** The splits hold out pairs (pair-holdout) or genes (gene-holdout); every line is on both
  sides, and stage one trained on every line and gene, including the gene-holdout test genes (design §3).
- **Pairs without a shared line are dropped.** 5,242 of 22,119 SL pairs (23.7%) and 35,544 of 110,595 non-SL pairs
  (32.1%) have no line where both genes are in the top 2,048 and never reach the classifier. Kept SL pairs average
  about 207 rows, non-SL about 124 (the anti-correlated sampled negatives are rarely co-expressed), so the number of
  lines a pair appears in tracks its label (inference, untested).
