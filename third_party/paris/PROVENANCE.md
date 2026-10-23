# Vendored PARIS (reference only)

Source: https://github.com/sbenfatto/PARIS, commit `179b67a58b2855b411d06877a43624595b71babb`. Licence: MIT
(`LICENSE.md`). Copied: `R/*.R` (without the empty `R/.Rhistory`), `DESCRIPTION`, `LICENSE.md`, `README.md`. Not copied:
`data/*.rda` (140 MB of 2020 DepMap tables), `man/`, `NAMESPACE`. The R code is not run; it is the specification.

The Python port is `src/experiments/sl/paris.py` (`configs/sl/paris.yaml`). It reproduces `RF_exp` for
`imp.score = "raw"` (`Boruta(maxRuns = 500, getImp = getImpRfRaw)`, confirmed attributes only, importance =
`attStats` `medianImp`) and `"Ginicorrected"` (`ranger(importance = "impurity_corrected")`, then
`importance_pvalues` with the default Janitza method, keeping p-value 0; a gene with no negative importance is skipped, as
R's `try` / `next` does). Per gene: the response's NaN lines dropped, `set.seed(43)`, then `cor.test` Pearson r and p-value of
each selected feature. Boruta and ranger semantics were read from Boruta 10.0.0 and ranger 0.18.0 (CRAN sources).

Unavoidable deviations (R/ranger/Boruta vs the port; sklearn trees, numpy RNG):

- RNG: `np.random.default_rng(43)` per gene instead of R's `set.seed(43)`; the bootstrap, shadows and tree seeds differ.
- Trees: sklearn `DecisionTreeRegressor` (best split by variance reduction, `max_features = floor(sqrt(p))`) replaces
  ranger's. Ranger's `min.node.size = 5` (a node of at most 5 rows is a leaf) is `min_samples_split = 6`; sklearn counts
  distinct bootstrap rows (zero-weight rows are dropped), ranger counts draws.
- Permutation importance: ranger's out-of-bag estimator (MSE rise when a feature's OOB values are permuted, summed over
  trees, divided by 500) is reimplemented exactly for each tree, but the permutations are drawn with numpy and each tree's
  predictions come from a vectorised walk over sklearn's tree arrays. Only features a tree splits on can change it.
- Corrected impurity: the permuted copies follow ranger (one row permutation per tree, `mtry` drawn from the 2p columns,
  SSE decrease of the real feature minus its copy's, divided by the trees); the decrease comes from sklearn's node
  impurities.
- Boruta's `getImpRfRaw` also uses 500 trees per run; shadows (at least five, duplicated as in R) are drawn with
  `Generator.permuted`; the loop stops after `maxRuns - 1 = 499` runs as R's does; the Bonferroni factor is the number of
  attributes, as `p.adjust` over the full vector.
- Inputs are not DepMap's 2020 tables: dependency is stage one's measured GeneEffect, expression the stage-one mean channel,
  on the reference lines only (no benchmark line); a gene's own expression column is dropped when it is fitted.
  The forests run one thread each and genes run on `n_jobs` joblib processes.
