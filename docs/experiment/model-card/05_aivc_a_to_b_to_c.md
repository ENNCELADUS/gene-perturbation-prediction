# Model Card: AIVC A->B->C STATE Pipeline for K562 Dependency Ranking

Source experiment:
[`05_aivc_a_to_b_to_c.md`](../05_aivc_a_to_b_to_c.md)

## Purpose and Claim Boundary

This model predicts the K562 DepMap/Achilles GeneEffect associated with a
perturbation gene from non-targeting control cells and perturbation identity.
GeneEffect is a population-level relative growth-rate/dependency label under a
population-dynamics model. It is not a cell-death label, a per-cell fate label,
a mechanism label, or a synthetic-lethality label.

For each inner-train perturbation gene, an ESM-2 adapter produces the STATE
perturbation token. The trainable STATE checkpoint predicts a 2,000-feature
post-perturbation response bag from non-targeting control cells. Predicted and
observed response cells pass through the same Linear(2000, 128) response encoder,
trainable diagonal-GMM pooler, and GeneEffect head. Observed B supplies auxiliary
response and GeneEffect supervision for inner-train genes only. Validation and
primary outer-test GeneEffect predictions use control cells plus perturbation
identity only.

## Implemented Model Graph

1. A fixed ESM-2 vector for the perturbation gene enters a trainable adapter
   whose output is the STATE perturbation token.
2. The unfrozen ST-HVG-Replogle STATE checkpoint combines that token with a
   non-targeting control-cell bag and predicts a 2,000-feature response bag.
3. One shared `Linear(2000, 128)` plus `LayerNorm(128)` encoder maps predicted,
   observed, and control cells into the same 128-dimensional response space.
4. One trainable 64-component diagonal GMM pools response bags into occupancy,
   control-relative occupancy, mean, variance, and entropy features.
5. A `[64, 32]` MLP maps the pooled features to one GeneEffect prediction.

The ESM-2 adapter, STATE checkpoint, response encoder, GMM parameters, and C
head are trainable. Predicted and observed branches share the response encoder,
GMM, and C head; the observed branch does not pass through the ESM-2 adapter or
STATE.

The authoritative model is fully differentiable and has no precursor fitting
stage. scVI artifacts, ridge projector artifacts, and fixed-GMM fit caches are
absent from the authoritative run. Legacy scVI, ridge, and fixed-GMM helpers
retained for other experiments are not called by audited exp05 training.

## Prediction and Supervision Units

The unit is one K562 perturbation gene from the frozen 9,338-gene GWPS-DepMap
overlap. The five outer folds are fixed before any fit. Each outer-train partition
is split again into inner-train and inner-validation genes.

For an inner-train gene, observed GWPS response cells provide auxiliary
response alignment and observed-B GeneEffect supervision. Inner-validation and
primary outer-test prediction do not consume the evaluated gene's observed
response cells, target cell count, or target batch labels. Their input is
non-targeting controls plus perturbation identity.

After checkpoint selection is frozen, outer-test observed responses may be
opened only for generation-quality metrics and the shared observed-B oracle.
These diagnostic routes cannot update parameters, statistics, thresholds,
epochs, checkpoints, or representations. Adamson is a secondary assay-transfer
evaluation and is excluded from fitting and selection.

## Loss Graph

The inner-train objective contains seven configured terms:

| Term | Weight | Gradient role |
| --- | ---: | --- |
| HVG mean-delta MSE | 0.01 | STATE and ESM-2 adapter response supervision |
| Shared-latent mean MSE | 0.10 | Align predicted cells to detached observed encodings |
| GMM occupancy MSE | 0.10 | Align predicted occupancy to detached observed occupancy |
| Observed-latent GMM NLL | 0.01 | Fit the trainable diagonal GMM on observed inner-train cells |
| Predicted-B GeneEffect MSE | 2.00 | Train the full predicted path through the C head |
| Observed-B GeneEffect MSE | 0.25 | Train the shared encoder, GMM, and C head |
| Predicted GeneEffect RankNet | 5.00 | Rank predictions across the four-rank global gene batch |

The four response-alignment/GMM terms anneal during the first five epochs to
10% of their initial weights. Both energy-distance weights are zero, so the
authoritative run skips those computations. The observed branch remains
differentiable for its GeneEffect loss but is detached when it serves as an
alignment target.

RankNet differentiably all-gathers the prediction from each rank and ordinarily
all-gathers detached labels and validity masks. The loss therefore uses the
four-gene global batch on every optimizer step, with padded entries removed by
the gathered mask.

## Mandatory DDP Runtime

Each outer fold is one four-GPU DDP training job. Rank 0 through rank 3 process
disjoint gene batches from the same fold and synchronize gradients every optimizer
step. The five outer folds run sequentially; GPUs are not assigned independent
fold-local models. Per-device gene batch size is one, so the global gene batch size
is four.

Training requires exactly four Accelerate processes. The model, optimizer, and
both fold loaders are prepared together; padded distributed loaders keep step
counts symmetric. Every epoch verifies that all four ranks performed optimizer
steps. Rank-zero I/O failures are broadcast and raised on every rank.

## Selection and Readouts

The best checkpoint is selected by inner-validation Spearman from the
prediction-only path. The primary readout is the mean and standard deviation of
the five `internal_outer_test` GeneEffect prediction folds.

The aggregate run emits `summary.csv`, `run_manifest.json`, and:

- `artifacts/fold_metrics.csv`
- `artifacts/predictions.csv`
- `artifacts/gene_splits.csv`
- `artifacts/fold_roles.csv`
- `artifacts/fit_access_audit.csv`
- `artifacts/external_alignment_qa.csv`

Each fold also emits its train log, best and final checkpoints, fitted-component
hash summaries, and runtime evidence. There are no scVI teacher, ridge
projector, or fixed-GMM cache directories in this contract.

## Limitations

- Adamson evaluates assay transfer and does not guarantee held-out-gene
  generalization relative to Replogle.
- RankNet depends on four-rank collectives and is not an independent per-rank
  objective.
- Generation-quality and observed-B-oracle metrics use outer-test response only
  after freeze and are diagnostics, not primary prediction paths.
- GeneEffect supports population-level relative growth-rate/dependency claims
  only. The model does not establish death, fate, mechanism, causation, or
  synthetic lethality.
