# Joint GeneEffect residual failure diagnosis

2026-09-07. Scope: diagnosis and design recommendations, not implementation or a
new training run. Autoresearch orchestrator → decide-design, bounded evidence-led
reasoning with an independent adversarial review. No blind-panel convergence claim.

## Verified evidence

Read existing predictions on H20 `apg3op6hp3v99-0`, checkout
`/2023533015/VCC_Project`, evaluation revision
`a0af4999b73ba6539af58b415291fbac9ea6d90e`. Run
`joint_seed0_20260906T174818Z_b1024`, selected epoch 3. Curated evidence records
checkpoint SHA256 `37405454717cb12164497542623722cd31e9dc793ebbfef3d79a3ef868e63c59`;
the hash was not recomputed in this analysis. Local HEAD was `45f4ead`; model,
training, baseline, evaluation and inspected dataset source has no diff from the
evaluation revision. Existing `docs/03-experiment-protocol.md` edits were preserved.

The CPU diagnostic independently rejoined all methods on 478501 observed
`(ModelID, gene_symbol)` keys and checked target agreement. It recomputed reported
residual Pearson exactly. Missing observations remain missing. No training,
checkpoint inference, calibration, test-driven hyperparameter search, or remote
file mutation was performed. Script and aggregate JSON accompany this report.

Metrics below use the same 4447 train-defined variable genes and 27 test contexts.
Per-gene standard deviations are population SD across observed contexts; the ratio
is computed within each gene and then summarized, not a ratio of medians.

| Diagnostic | Joint | Tx1 PCA-ridge |
|---|---:|---:|
| Residual Pearson macro per gene | 0.05414 | 0.12161 |
| Median prediction SD | 0.01292 | 0.05683 |
| Median target SD | 0.22832 | 0.22832 |
| Median prediction SD / target SD | 0.05752 | 0.25307 |
| 10th–90th percentile SD ratio | 0.02758–0.15703 | 0.14466–0.40892 |
| Genes with positive Pearson | 61.48% | 72.90% |
| Huber on variable-gene subset | 0.0403226 | 0.0404712 |
| Gene-mean Huber on that subset | 0.0404210 | 0.0404210 |

Ridge wins Pearson on 2683/4447 genes (60.33%); joint wins on 1764. Median joint
minus ridge difference is -0.06350. These are descriptive comparisons, not a
significance result; genes are dependent and only one seed was run.

On the 4315 variable genes with all 27 labels observed, subtract each gene's mean
across contexts solely to analyze the matrix spectrum. Joint prediction variation
has 58.90% energy in its first singular direction, versus ridge 42.92%. Participation
rank, `(sum s²)²/sum s⁴`, is 2.58 versus 3.66; target rank is 20.26, including noise.
This is evidence of concentrated predicted variation, not a requirement to match
the target's rank. Identical shared cell-line offsets explain only 6.05% of joint
centered prediction energy, so the model is not merely a single global line bias.

## What failed, and what did not

The 0.91 absolute Pearson largely reflects the fixed gene mean: gene mean alone
achieves 0.91047. Overall joint Huber improves only 0.0773% over that prior. The
specific failure is weak generalization of gene-dependent changes across contexts.
There is measurable signal, but it is strongly shrunk and less correctly ordered
than ridge. Positive scaling cannot improve Pearson or Spearman. Increasing output
variance indiscriminately can worsen Huber, as the nearest-line controls illustrate.

The model already learns `y - mu_train[g]`, adds the same fixed mean back, and has a
direct basal-context input. The comparison uses the same prepared Tx1 mean/variance
representation and label universe. There is no evidence here that Tx1 lacks useful
signal or that adding context/residualization alone would repair the model.

No detached response graph was found in the inspected forward path. Global errors
and finite-gradient checks are present. A decreasing train Huber does not establish
strong train residual correlation: the current logs do not report that quantity.

## Ranked design explanations

### 1. Indirect gene-dependent context mapping is the leading structural hypothesis

Ridge uses train-only standardization, eight PCs and an independent regularized
regression per gene (`src/baselines/residual.py:323-350`, defaults at 669-675).
It directly estimates `r_hat[g,c] = b[g] + w[g]^T PCA(z[c])`.

The joint head concatenates 5120 basal-context coordinates, 1280 ESM2 coordinates,
256 projected response coordinates, six response summaries, three expression
features and three masks. A 6668→256→256→1 MLP mixes all blocks. Its roughly 1.77M
head parameters must learn gene-conditioned context slopes indirectly through
nonlinear activation patterns (`src/model/head.py:162-174,309-310`). A nonlinear MLP
can represent those interactions; the issue is inductive bias and optimization,
not mathematical impossibility or a complete absence of interactions.

Gene-only corrections or common context effects are easier to share than thousands
of different context slopes. Since residuals are already gene-centered, a weak
context map can converge near zero with good aggregate Huber. The millions of
gene/context rows supply shared supervision, but do not create millions of distinct
contexts: contextual coverage remains 170 labeled training lines. Ridge makes much
stronger use of this small-context structure. Measured weak, concentrated variation
is consistent with this hypothesis, but a matched direct-head comparison is needed
to isolate it from optimizer and response-feature effects.

### 2. Reconstruction improvement does not establish GeneEffect-relevant learning

The shared STATE/ESM adapter also optimizes response MSE + energy distance on only
four training anchors. Those losses can improve expression reconstruction without
improving GeneEffect differences on unseen contexts. Held-out response conditions
are still from those four anchors, not the 27 held-out GeneEffect lines.

Moreover, subtracting the same control mean from predicted and observed expression
cancels in mean-delta MSE (`src/model/response.py`). That formulation is not itself
an error, but the subtraction does not prevent a predictor from improving mainly
through context-average reconstruction. Establish perturbation specificity using
a no-perturbation/context-only control and a gene-shuffle diagnostic.

On replay steps, `L = L_GE + L_response`, followed by one global gradient clip
(`src/training/trainer.py:69-85`). Response loss is hundreds while GE is ~0.015;
this is a scale warning, not a measured gradient ratio. A response-dominated norm
could attenuate GE-only head gradients through global clipping, and shared
gradients may conflict. Adam's history and the three non-replay steps also matter.
Record task-specific norms and cosines on STATE/adapter, head norm, clipping factor,
and update norms; compare matched runs with replay weight zero and one before
declaring negative transfer. Do not choose a loss coefficient by dividing scalar
loss magnitudes alone.

### 3. A moving response representation is mixed with fixed initialization scales

STATE outputs pass through predicted-minus-basal moments, a fixed 4000→256 sparse
projection, and summary statistics. Initialization-fitted standardization remains
fixed while STATE changes (`src/model/normalization.py:219-278`). Basal and ESM
features are stable, but the response blocks can drift. The head must track a
changing upstream predictor from the first update, while basal/perturbation
interfaces can include newly initialized weights (`src/model/initialization.py`).

This is a conditioning risk, not proof of a normalization bug. Measure each block's
mean/SD relative to saved scales, near-zero fitted scales and first-layer weighted
contribution at startup/best/last, plus train/eval differences. The 4000→256
projection restricts what response information the head sees but remains
differentiable. One thresholded summary (fraction beyond basal p95) is
nondifferentiable; other response channels retain gradients. Neither observation
alone explains the whole failure because direct context bypasses STATE.

### 4. The objective and selection do not specifically reward contextual ordering

Training averages Huber over all observed genes, while residual correlation
macro-averages only 4447 variable genes. Huber penalizes erroneous amplitude; Pearson
measures direction irrespective of amplitude. Consequently joint can win Huber
while ridge wins correlation. Residualizing the target does not change this:
`Huber((mu+r_hat)-(mu+r)) = Huber(r_hat-r)`.

Epoch 3 was correctly selected by the declared minimum-Huber rule. Its validation
residual Pearson is 0.04043, whereas the best observed validation residual Pearson
is only 0.04946 at epoch 6. Thus selection mismatch exists, but available curves do
not demonstrate a strong contextual model hidden at another epoch. Do not compare
validation and test coefficients as matched performance or retrospectively choose
a checkpoint on the observed test set.

Epochs 1–2 used batch 256/rank and 5889 updates/~1472 replays per epoch. Epochs 3–8
used 1024/rank and 1472 updates/368 replays. The batch transition changes exposure
and optimizer steps; it confounds claims that an epoch-3 change proves overfitting
or response interference. Future comparisons must share the full schedule.

## Recommended modification and discriminating sequence

1. **Establish a stable direct contextual branch first.** Retain frozen Tx1,
   train-only eight-PC context preprocessing and independent ridge heads as the
   reference model. Eight is the actual current baseline setting, not a proposed
   value selected from test. If a neural implementation is needed, first reproduce
   this map using explicit per-gene coefficients with the same regularization.
   Gene-specific coefficients fit the held-out-cell-line contract; this does not
   support unseen-gene claims. ESM2 need not bear the entire burden of producing
   independent gene-specific coefficients. Partial sharing/low-rank coefficient
   models are later controlled alternatives, not mandatory complexity.
2. **Separate contextual direction from calibration.** Estimate ridge shrinkage
   using training-context out-of-fold predictions, with PCA, scaling and gene means
   fitted within each fold. Preserve the fold-independent prediction-centering
   contract for cross-fold diagnostics. Keep magnitude-calibrated GE errors and
   residual ordering as separate reports. Never use the descriptive test SD ratios
   as calibration factors and never z-score predictions per target context.
3. **Ask whether response features add anything beyond that branch.** Freeze the
   response extractor initially and add a small, regularized gene-conditioned
   correction to the direct context predictor; a single gene-agnostic scorer may
   repeat the present limitation. Train it against honest out-of-fold
   baseline errors; in-sample ridge errors would understate the task. Validate the
   full fitted pipeline because fold-to-full-fit shifts remain possible. Preserve
   the useful direct branch while testing response value. If frozen response adds
   no information on validation, a larger joint system is not yet justified.
4. **Only then revisit end-to-end coupling.** On fixed train/validation probes,
   measure per-task gradients and live-feature drift. If replay conflicts with GE,
   compare controlled weighting or separate optimization/clipping; if coupling has
   no measurable benefit, keep response training separate. A frozen-feature failure
   alone cannot rule out every possible jointly learned response representation.
5. **Make any objective change explicit.** The current protocol minimizes Huber.
   Retain that rule for matched diagnostics. If the scientific primary aim changes
   to residual ranking, prospectively revise the objective/selector and consider a
   within-gene, across-context loss using enough contexts per gene. Avoid arbitrary
   inverse-variance amplification of near-constant genes and tiny-batch correlation
   losses. The observed test set is not a fresh model-selection resource.

Required evidence before a new long run: train-versus-validation per-gene residual
correlation/variance; direct gene-specific branch against ridge; frozen response
increment; matched replay on/off comparison with gradient measurements. These
distinguish inadequate fitting from generalization failure, unhelpful response
features, gradient interference and selection effects.

One concrete implementation limitation affects this plan: feature block ablations
are not wired end-to-end. The head rejects tensors for disabled blocks, while
`GeneEffectE2EModel.forward_features` always supplies every block and initialization
does not expose the block config. Repair that wiring before claiming a valid
no-response-feature ablation. It does not explain the completed all-enabled run.

These results concern single-gene GeneEffect, not demonstrated SL interactions.
Tx1 pretraining-exposure and single-seed limits remain. Feature drift, task gradient
conflict and train residual fit were not measured in this read-only analysis.
