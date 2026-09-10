# Future work: readout objective, selection and context representation

**Status:** proposed plan, 2026-09-10. Not implemented, no launch. Follows the
[GeneEffect protocol](../03-geneeffect-protocol.md) §7–§8 and the
[response-pathway diagnostics](../results/p1_response_pathway_diagnostics/README.md);
bound by blueprint §§7–8.

## What the learning curves say

The explicit context-slope readout on the frozen backbone (protocol §7.1) shows two
curves moving apart. Validation GeneEffect Huber reaches its minimum at epoch 3 and
rises afterwards, so selection stops there. Validation residual Pearson keeps rising to
0.136 at epoch 6, and training residual Pearson climbs to 0.34. Three properties of the
objective explain this.

1. **The Huber loss is a mean-squared error with a large irreducible floor.** Residual
   targets are bounded well inside the Huber transition ($|r|<1$ for essentially every
   pair), so the loss is $\tfrac12$ MSE over all 479,084 observed pairs. Most of that
   mass is measurement noise and non-variable genes; the context signal moves it by
   $10^{-4}$. A criterion whose signal is $10^{-4}$ of its value cannot select for it.
2. **The loss weights genes by residual variance, the metric weights them equally.** The
   reported residual Pearson is a macro mean over 4,447 genes; the loss is dominated by
   the genes with the largest residual spread. The readout is optimised for one and
   scored on the other.
3. **Selection on the loss discards the epochs that carry the correlation gain.** With
   170 training lines, the gene-specific slopes overfit the squared error within three
   epochs while the direction of the per-gene context effect keeps improving; the
   selection rule keeps the earlier, more shrunk model (SD ratio 0.19).

Redesigning the objective and the selection rule is therefore the first lever, and it
is cheap: every arm below trains a fresh head on the cached direct features in minutes.

## Tier A: objective and selection on the frozen backbone

Incumbent: explicit context-slope readout, head seed 0, validation residual Pearson
0.1315 (0.133 and 0.134 at seeds 1 and 2). Every arm keeps the backbone, the features,
the split and the variable-gene set fixed.

| Arm | Change | Rationale |
| --- | --- | --- |
| A-sel | Select the checkpoint and stop early on validation residual Pearson; report Huber | Aligns selection with the metric; expected from the curve alone |
| A-std | Train on the per-gene standardised residual $r_{cg}/s_g$, $s_g$ the training residual SD per gene, stored in the checkpoint; de-standardise at prediction | Equal gene weighting in the loss, matching the macro metric; train-side only (blueprint §7) |
| A-corr | Gene-major batches (all training lines of a gene set) with loss $1-\bar\rho_{\text{batch}}+\lambda\,\mathrm{MSE}_{\text{std}}$ | Optimises the reported quantity directly; MSE term keeps scale anchored |
| A-cal | Post-hoc per-gene linear calibration $a_g\hat r+b_g$ fitted on training lines | Addresses shrinkage (SD ratio 0.19) without touching correlation; never per-context |
| A-reg | Slope L2 $\times\{0.1,1,10\}$ and context components $\{4,8,16,32\}$ | Bounds how much of the ceiling is regularisation |
| A-ens | Mean of three head seeds | Cheapest variance reduction; also the seed-stability record |

Combinations follow a keep: A-std then A-sel is the expected incumbent after the first
pass; A-corr is tried on top of whichever is kept.

**Keep rule.** An arm is kept when, against the incumbent, the paired 27-line bootstrap
of validation residual Pearson has an interval excluding 0 in the arm's favour, the
sign of the difference agrees at two further head seeds, and validation Huber is no
worse than the gene mean by more than 0.1%. Otherwise discarded. Two consecutive
discards after a keep end the tier.

**Selection change and the blueprint.** Blueprint §4 states that only minimum
validation GeneEffect loss selects the checkpoint. A-sel changes that rule for the
readout. It requires a dated in-place amendment of blueprint §4, decided by the owner
before A-sel results are registered; until then A-sel is reported as a diagnostic.

## Tier B: context representation

The readout ties an eight-component linear projection of the pooled Tx1 embedding, so
the next ceiling is the representation. Each arm replaces or extends $u_c$ and is
scored under the Tier A incumbent objective.

| Arm | Context input | Rationale |
| --- | --- | --- |
| B-hvg | Principal components of pooled basal HVG expression (train-fitted) | The direct expression signal, no foundation model |
| B-both | Tx1 components and HVG components side by side | Tests complementarity |
| B-cell | Per-cell Tx1 embeddings with a learned attention pooling in the head | Recovers what mean/variance pooling discards |
| B-gene | Gene-specific basal covariates extended to the expression of the gene's pathway neighbours in $c$ | Puts gene-by-context structure in the input rather than the slope |

Same keep rule as Tier A. B-hvg is the control for any Tx1 claim: a Tx1 arm that does not
beat B-hvg licenses no foundation-model claim.

## Tier C: a response model that transfers

Only after Tiers A and B, and only for a response model that meets the standard in the
[interface-isolation design](2026-09-08-p1c-interface-isolation-design.md) §8: a
leave-one-anchor-out held-out loss ratio below no-change with an interval excluding 1
and a positive identity advantage on every held-out anchor. Candidates, in order:

1. The expression-residual interface, the only interface that did not fail on unseen
   lines, with more adaptation anchors where a compatible platform exists.
2. A response model pretrained across many contexts, evaluated untrained under the
   corrected preparation before any adaptation.
3. Readout on response features from a kept response model, against the Tier A/B
   incumbent, to close the response-block question with a working pathway.

## Tier D: seeds and the test split

Three training seeds before any further test use. The test split stays closed until a
validation-kept model beats the eight-component context ridge by at least +0.02 residual
Pearson with an interval excluding 0 at all three seeds. One test evaluation, with the
baseline ladder, closes the track.

## Metrics, predicates and budget (autoresearch configuration)

- **Scope:** `src/model/readout.py` (loss and selection options, standardiser,
  calibration), `src/training/readout.py` (gene-major sampler, selection metric),
  `src/experiments/p1a.py` (`--select`, `--loss`, `--components`, `--slope-l2`,
  `--context` flags), tests in `tests/test_p1a.py`. Backbone, cached features, split and
  variable-gene set unchanged. Blueprint §4 amended only for A-sel, by the owner.
- **Primary metric:** validation residual Pearson, macro mean over the 4,447
  train-defined variable genes, higher is better. **Secondary:** residual Spearman,
  SD ratio, validation Huber. **Guard:** validation Huber within 0.1% of the gene mean.
- **Verify:** `uv run python -m src.experiments.p1a compare --cache <features> --runs
  <runs> --out-dir <dir> --reference-val <incumbent predictions>` emits
  `comparison/selected.csv` and `comparison/paired.json`; an arm is kept when the paired
  interval of residual Pearson against the incumbent excludes 0 in its favour at head
  seed 0 and the sign agrees at seeds 1 and 2.
- **Iterations:** Tier A six arms plus up to three combinations, three head seeds each;
  Tier B four arms; hard ceiling 40 head trainings. Plateau: two consecutive discards
  after a keep, per tier.
- **Compute:** one head training on cached features runs in minutes on one GPU; Tier A
  under two GPU-hours in total; Tier B adds one feature-extraction pass per context
  input (under one hour each on the H20 host). No backbone training.
- **Gates before GPU:** unit tests for the standardiser round-trip (de-standardised
  predictions reproduce the raw-residual Huber), gene-major sampler coverage (every
  training line of every batch gene present), calibration fitted on training rows only,
  and the selection rule recorded in `run.json`; blueprint §4 amendment text agreed
  before A-sel is registered.

## Leakage and claim boundaries

Every fit, standardiser, calibration and component projection uses training lines
only and is stored with the checkpoint. Validation is the decision surface; the test
split is not opened in Tiers A–C. No per-context prediction z-scoring. A readout gain is
a held-out-line dependency result and no SL evidence; a Tx1 gain is claimed only against
the HVG control; Tx1 Tahoe-100M pretraining exposure qualifies every held-out result.
