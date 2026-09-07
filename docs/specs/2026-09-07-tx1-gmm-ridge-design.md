# Tx1 GMM-ridge baseline

Approved in chat on 2026-09-07. P1-A uses the same prepared basal Tx1 bags,
GeneEffect split and common gene panel as Tx1 PCA8-ridge. It predicts no perturbation
response and makes no unseen-gene or SL claim.

Fit a global cell-dimension StandardScaler and diagonal GaussianMixture with K=64
on equal numbers of cells from each labeled training line. Sample without replacement
from the existing bags, seed 0, using the minimum available bag length; do not create
new cells or deduplicate cached bags. Record selected positions and bag lengths so
repeated cache positions are not described as independent biological cells. No PCA
or per-line centering is applied. GMM uses reg_covar=1e-4, max_iter=200, n_init=1,
tol=1e-3 and random_state=0.

For each full prepared bag, compute 64 mean posterior responsibilities and four
statistics: occupancy entropy, exp(entropy), mean maximum responsibility and mean
negative log likelihood. Fit a context-feature StandardScaler on labeled training
lines, then one Ridge(alpha=1, fit_intercept=True) per gene on its finite labels.
Targets subtract the saved training gene mean; absolute predictions add that same
mean. Missing labels are never imputed. Require at least three train contexts per
gene, matching the existing context baseline's eligibility floor.

Fit produces a saved model and train/val predictions using the P0 evaluator:
per-line absolute and per-gene residual correlations, amplitude/error metrics and
coverage. Train scalars use train_eval_; validation uses val_. Export context features
and GMM convergence, component weights, iteration count, lower bound and training
fit counts. Non-convergence is prominently reported, not silently repaired by changing
K or fitting on held-out cells. No hyperparameter selection is performed.

The artifact stores both scalers, GMM, gene order, coefficients/intercepts, fitted
gene means/variable genes, input configuration, fixed split and preprocessing needed
by the current prepared-input reader. Separate evaluation reloads it without fitting.
Fit never runs test; test evaluation requires an explicit command. Model artifacts
use joblib and must be loaded only from trusted local runs with the recorded sklearn
version. Output directories for fitting must be new; failed exports can be retried
from the saved model. This implementation request does not launch an H20 experiment.
