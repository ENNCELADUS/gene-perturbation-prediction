# P1-A: fixed-backbone head diagnostics

The owner approved the four-arm experimental design on 2026-09-07 and subsequently
authorized its implementation with TDD. This scope does not include a training
launch or P1-B.
The owner subsequently confirmed the execution settings below, replacing the
proposed fixed 50-epoch budget and three seeds with early stopping and seed 0 only.

## Question and scope

Determine whether an explicit gene-specific context slope improves readout from
the selected P0 backbone, and whether its fixed response features add predictive
value. This is a held-out-cell-line GeneEffect diagnostic on training-covered
genes, not an SL interaction experiment or an unseen-gene claim.

Use the current GeneEffect protocol in [03-geneeffect-protocol.md](../03-geneeffect-protocol.md).
P0 train/validation values and the [validation PCA/GMM reference](../results/tx1_gmm_ridge_seed0/README.md)
motivate the experiment. Reuse the PCA-ridge results. Do not use the already observed
test set to choose arms, hyperparameters or checkpoints.

## Shared representation and training boundary

Restore the checkpoint that produced P0, including STATE, the ESM2 adapter,
projection, gene order and original normalization. Verify checkpoint identity
against the P0 exports before extracting features.

- D contains the existing z_c, e_g, q_sc and applicable masks.
- R contains the 256-dimensional delta projection, six response summaries and
  the two masks associated with the response-summary block.
- Extract raw features once with fixed basal bags, inference mode and precision.
  Store context features by ModelID, gene features by gene, and pair features by
  (ModelID, gene); preserve label and coverage masks.
- All arms consume identical cached values, labels, split and gene membership.
  Fit a shared new block standardizer using training features only, then retain
  each arm's enabled blocks. Preserve current coverage-mask semantics.
- Train fresh heads only. Backbone parameters and behavior are fixed; no response
  replay or backbone optimization occurs in this diagnostic.

## Four main arms

| Arm | Head | Inputs | Primary comparison |
| --- | --- | --- | --- |
| A0 | Current shared MLP | D | Direct-feature reference |
| A1 | Current shared MLP | D + R | A1 minus A0: response-block effect |
| A2 | Current MLP plus gene-specific context slope | D | A2 minus A0: explicit-readout effect |
| A3 | Current MLP plus gene-specific context slope | D + R | A3 minus A1: explicit-readout effect; A3 minus A2: response-block effect |

The MLP retains two Linear -> LayerNorm -> GELU hidden layers of width 256 and
the scalar output. The explicit arms predict

```text
residual_hat(g,c) = shared_mlp(F(g,c)) + w[g]^T u[c]
u[c] = train-fitted PCA8 scores, scaled to unit training population SD
```

Full z_c remains an MLP input. On the 170 unique training-context rows, remove
constant z_c dimensions, standardize each retained dimension using training
statistics, and fit PCA8 with the full SVD solver. Divide each PC score by its
training population SD (ddof=0); validation applies the saved transforms only.
This additional PC scaling differs from the existing unwhitened PCA-ridge reference.
Each training-covered gene has an independent eight-dimensional w initialized to
zero with explicit L2 regularization. Do not
insert fitted ridge predictions or introduce a cell-line parameter lookup.

Copy identical MLP initial weights within A0/A2 and within A1/A3. Initialize one
canonical full-input MLP with seed 0, then copy its direct-feature columns and all
remaining layers to the no-response arms, retaining the canonical fan-in scaling.
Equal seeds alone do not ensure equal weights when input dimensions differ.
Removing R also removes its
associated masks, so the contrast measures the response block as a whole.

## Auxiliary checks

Reuse the existing P0 predictions and metrics; do not rerun the original head with
the original scaler. Check cached feature order, pair keys and labels. Align
validation exports with
the reference's 479,084 observed keys and 4,447 train-defined variable genes;
verify targets as well as membership.

Optionally train A1 with the old scaler if feature-scale diagnostics motivate it.
Use the same initialization and training as A1, changing only the scaler. Do not
apply a new scaler to the old head as a substitute. The old scaler used at most
32 sampled conditions per training line: measure representation drift on matched
conditions to distinguish it from changing the statistics-fitting sample.

## Training, reporting and interpretation

Use Huber delta 1 with the current label weighting. Run A0-A3 once each with head
seed 0: four main training runs, with no multi-seed repeats in this round.

| Parameter | Setting |
| --- | --- |
| Maximum epochs | 50; a cap, not a required duration |
| Early stopping and checkpoint selection | Minimize validation GeneEffect Huber; patience 5 |
| Global batch size | 1024 pairs, independent of GPU count; retain the final partial batch |
| Optimizer | AdamW, betas (0.9, 0.999), epsilon 1e-8 |
| Learning rate | Constant 1e-4 for both MLP and explicit branch; no scheduler or warmup |
| MLP weight decay | 0.01 |
| Explicit-branch weight decay | 0; use the separate L2 term below |
| Gradient clipping | Global norm 1.0 over all trainable head parameters |
| Head initialization seed | 0 |
| Data-order seed | 0, with deterministic epoch-specific permutations shared across arms |
| Head training precision | FP32 |

For A2/A3, the training objective is

```text
L = mean_batch Huber_delta1(prediction, residual_target)
    + (0.01 / number_of_training_covered_genes) * sum_g sum_k w[g,k]^2
```

The regularizer covers all training-covered genes at every update, not only genes
present in the batch. Lambda is 0.01, with the eight coefficients summed within
each gene and averaged across genes. This is a fixed diagnostic setting, not a
validated optimal lambda or a translation of ridge alpha=1. A0/A1 have no branch
regularizer. Validation Huber excludes regularization.

Evaluate complete train and validation sets once at each epoch end. A strict
decrease in validation Huber saves best.pt and resets patience; a tie or increase
retains the earlier best and increments patience. Stop each arm after five
consecutive non-improving validations or at epoch 50. Do not continue stopped arms
to equalize budgets or require a minimum warmup duration.

Report each arm's minimum-Huber checkpoint as the primary result, including its
selected epoch/update and actual stopping epoch/update. Show full learning curves
and pairwise comparisons at shared completed epoch/update points before the
earlier stop. Epochs 10, 25 and 50 can be displayed only where actually reached;
none is mandatory. With the recorded 3,015,332 finite training pairs, one epoch
has 2,945 updates and the cap is 147,250 updates; verify counts against the aligned
cache. Actual budgets may differ because of early stopping, so selected-checkpoint
contrasts are comparisons under the same stopping policy, not equal-update claims.

Report train/validation residual Pearson, Spearman, Huber and per-gene SD ratios,
with per-gene differences and scored/undefined counts. Correlations and SD ratios
use train-defined variable genes; Huber covers all finite labeled pairs. Lower
Huber and higher correlations are favorable; SD ratio is descriptive, not a target
to force toward one. Preserve undefined metrics rather than replacing them by zero.

For paired uncertainty, resample contexts together across arms, grouping by
PatientID where applicable and available. Recompute per-gene correlations and
macro summaries within each replicate, reporting comparable defined-gene support.
Pairs within one context are not independent generalization samples. This is a
single-seed diagnostic; context bootstrap does not measure initialization or
backbone-training variability and does not repeat checkpoint selection.

- A2/A0 and A3/A1 improvements support the explicit readout under the tested setup.
- A1/A0 and A3/A2 improvements support incremental value of the fixed response block.
- Different response effects across heads indicate dependence on the readout.
- Train-only improvements indicate increased fitting without demonstrated transfer.
- Negative response contrasts establish lack of benefit in these settings, not
  absence of information in every possible response representation or decoder.
- Improvement over historical joint inference shows additional readable signal;
  it does not establish replay interference or gradient conflict as the cause.
- Correlation gains with worse Huber must be reported as a tradeoff, not overall
  prediction improvement or an automatic decision to adopt the branch.

This validation diagnostic is conditional on the selected P0 checkpoint. Features
from that supervised backbone do not support independent pipeline cross-validation
within its training contexts; such validation must also exclude upstream supervision
by fold. P1-B separately addresses perturbation prediction and backbone adaptation.

## Execution handoff

Before extraction, record the exact P0 checkpoint/cache identity and inference
precision. The numerical head-training settings above are fixed before comparing
validation outcomes; do not treat the
production seed validator or early-stopping configuration as an existing head-only
runner. The approved design uses the existing feature/head/evaluation interfaces
with a dedicated diagnostic entry point rather than changing production joint
training behavior.

Implementation checks should cover cached feature order and identity,
zero-branch paired initialization, optimizer scope, train-only preprocessing,
pair/target alignment and the existing metric semantics. The diagnostic entry point
is `hpc/run.sh p1a`, with separate extract, train, evaluate and compare commands;
see the [runbook](../../hpc/README.md#p1-a-fixed-backbone-head-diagnostics).
Head runs use one process/device per arm, avoiding world-size-dependent batch or
gradient scaling. No production experiment has been run during implementation.
