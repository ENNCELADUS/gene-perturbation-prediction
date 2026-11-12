# Vendored SLIdR (reference only)

Source: https://github.com/cbg-ethz/slidr, commit `01faebeb6904850e53182cd26e4dc8ee599b4b1d`. Licence: GPL-3
(`LICENSE`). Copied: `R/sl_test_funcs.R`, `R/data_process_funcs.R`, `DESCRIPTION`, `LICENSE`, `README.md`. Not copied:
`data/LiverData.rda`, `Scripts/`, `man/`, plotting code. The R code is not run; it is the specification.

The Python port is `src/experiments/sl/slidr.py` (`configs/sl/slidr.yaml`). It reproduces `identifySLHits` (per driver:
each line's viabilities ranked over all genes, the partner's ranks summed over the mutated lines and divided by the gene
count, `IH_CDF` of that sum; the same test on the wild-type lines, two-sided) and `prepareDataObjects` with `CN_Thr = 1`
(a driver is altered by a mutation or by 0 copies; lines with no alteration drop out; drivers need `min_Nmut = 2`
mutated and wild-type lines; missing viabilities take the gene's mean over the lines). The tests are deterministic, so
the port has no RNG deviation.

Deviations from R:

- `fp_thresh = Inf`: R stops testing a driver's partners when the Bonferroni-scaled p-value exceeds the allowed false
  positives; the port tests every benchmark pair, so each keeps its p-values.
- One data object per lineage (DepMap's OncotreeLineage; the SL split table's lineage for its lines), not per primary
  site with subtype filters: the training-side lines of the SL split (train and bulk_labelled) in the held-out line's
  lineage, at least `minNrcelllines = 5`. A pan-cancer run (2026-11-12) lost 97% of its tests to the two-sided wild-type
  filter on ~1,100 lines. Drivers are the genes of the held-out pairs, not MutSig drivers; both genes of a pair are
  tested as the driver.
- No essential-gene filter (`essential_genes = NULL`, the default of `prepareDataObjects`).
- Inputs are DepMap 26Q1: viabilities `CRISPRGeneEffect.csv`; mutations `OmicsSomaticMutationsMatrixDamaging.csv`
  and `OmicsSomaticMutationsMatrixHotspot.csv` (count > 0); copy number `PortalOmicsCNGeneLog2.csv`, read as
  log2(relative CN + 1), so a line has 0 copies when round(2 (2^x - 1)) = 0 (R: round(2 · 2^x) on a log2 ratio).
- The benchmark score is -log10 of the mutated-lines p-value (it keeps the order of p-values below 1e-16), 0 when the
  wild-type p-value is below 0.1, and -log10 0.5 for a pair that no orientation tests.
