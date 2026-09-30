# Interface-isolation diagnostic (P1-C): interface isolation and basal reconstruction

**Status:** round 1 (count space) executed, seed 0, closed with nothing kept ([result](../results/p1_response_pathway_diagnostics/README.md)); round 2 (log space, below) launched 2026-09-09. Follows the
[fixed-backbone head diagnostic (P1-A)](2026-09-07-p1a-fixed-backbone-head-diagnostics-design.md) and the
[response-adaptation diagnostic (P1-B)](2026-09-07-p1b-response-adaptation-design.md); uses the current
[GeneEffect protocol](../03-geneeffect-protocol.md) and blueprint §§7–8.

## What the fixed-backbone head and response-adaptation diagnostics established, in the wording the evidence supports

- The response pathway of the seed-0 joint backbone (P0) gives no detectable perturbation-identity use and
  is 21× worse than predicting no change on held-out conditions of its own anchors.
- Interface-only adaptation at 1e-4 recovers reconstruction on the source anchors
  (900 → 33) and produces a small positive identity advantage, but never beats the
  no-change reference on held-out conditions, and its Jurkat error grows as source
  training continues (136 → 174 → 180). This supports over-adaptation to the source
  anchors; the mechanism is unresolved.
- Explicit gene-specific PCA8 context slopes raise validation residual Pearson from
  about 0.05 to 0.13, matching Tx1 PCA8-ridge. The current shared MLP head (A0) and training
  procedure do not learn transferable gene × context relationships; the explicit
  context-slope head (A2) does. No correlation advantage over PCA8-ridge is detected. The
  response block (R) of the joint backbone as trained (B-joint) gives no detectable gain under the two tested heads.
- Two mechanistic hypotheses have priority but are **not isolated**: (H1) insufficient
  interface adaptation, consistent with the new basal encoder receiving the STATE rate
  1e-6 in the seed-0 joint backbone; (H2) representation mismatch at the basal skip, because the composite
  feeds Tx1 embeddings to a new basal encoder while STATE adds the basal *encoding* to
  the transformer output in hidden space before a frozen decoder
  (`project_out(res_pred + control_cells)`), so nothing guarantees that known basal
  expression reaches the output. Native STATE also uses a learned basal encoding in that
  skip, so H2 is a mismatch hypothesis, not a claim that the architecture cannot work.
- The success of interface-only adaptation (B-interface) cannot be credited to learning rate alone: relative to the seed-0 joint backbone it also
  changed the objective, frozen set, supervised contexts and schedule.

## Question and scope

1. How much of each state's error is a context-level shared bias versus
   condition-specific deviation, on the source anchors and on Jurkat?
2. Does the released STATE checkpoint, driven through its own basal encoder with the
   cached HVG expression, behave as a functional response predictor on this data?
3. Holding the response-adaptation diagnostic's adaptation recipe fixed, what does interface learning rate alone
   contribute?
4. Does an interface that lets known basal expression enter the output directly produce
   perturbation effects that transfer to a context held out from adaptation, and does
   Tx1 context conditioning add anything to that?
5. Are the fixed-backbone head diagnostic's explicit context-slope head − shared MLP head and explicit context-slope head − ridge results stable to head initialisation?

No GeneEffect head training on adapted representations, no SL claim, no test split, no
new pretrained checkpoint. ST-Tahoe is out of scope; §8 states the standard it must meet.

## Fixed inputs and boundaries

- The response-adaptation diagnostic's prepared bundle (`180b95d3…`): 1,957 common coordinates, condition splits,
  derangements, train-only baselines, coordinate provenance. Preparation is reused; a
  bundle rebuilt for leave-one-anchor-out folds must reproduce the same coordinates and
  fail on any count disagreement.
- **Jurkat has been observed and used to guide this design.** Every Jurkat number in
  the interface-isolation diagnostic is a *diagnostic re-evaluation*, labelled as such. Independent transfer evidence
  comes from leave-one-anchor-out (LOAO) folds over the four anchors: adapt on three,
  hold the fourth out from adaptation, baselines, selection and stage decisions. Four
  folds are four related diagnostics on the same lines, not four independent contexts;
  ST/Tx1 pretraining exposure of all four remains unresolved.
- No-change is a **reference**, not an irreducible floor. Every loss is also reported as
  a ratio to the no-change reference on the identical condition set.
- Seed 0 throughout unless a tier states otherwise. Intervals are 1,000 paired
  gene-bootstrap replicates conditional on selected checkpoints.

## Existing-export analysis (Tier 0): no GPU, no inference

Inputs: each state's `evaluation/<state>/{internal,external}/effects.npz` and
`conditions.parquet` from run `p1b_seed0_20260907T164813Z`.

- **0a Bias decomposition.** For each state × context × role, with
  e_{g,c} = Δ̂_{g,c} − Δ_{g,c} on the complete 1,957 panel,
  mean_g‖e_{g,c}‖² = ‖ē_c‖² + mean_g‖e_{g,c} − ē_c‖². Report both terms, the shared-bias
  fraction, and the same decomposition for the three references. A large shared-bias
  fraction on Jurkat for interface-only adaptation, extended interface-only adaptation (B-continue) and
  interface adaptation with STATE unfrozen (B-unfreeze) supports H2-type context
  shift; a small one points to condition-level error. Diagnostic only: a bias estimated
  from Jurkat truth is never used to calibrate predictions.
- **0b Matched-coverage references.** Recompute no-change and global-mean on exactly the
  perturbation-mean covered conditions per anchor (val: 874/983, 234/238, 864/1,826;
  Jurkat: 2,373) with paired intervals, and restate the negative-transfer magnitude on
  that set. Retire the "≈40% attainable headroom" reading: the train-side
  perturbation-mean value includes each condition's own effect and is not an independent
  ceiling estimate.
- **0c Anchor scale audit.** Per anchor: cells per condition and per control bag, source
  file and normalisation, control matching, zero-filled coordinates (HCT116's 43), and
  the finite-sample component of the energy term. State whether HepG2's five-fold
  no-change reference (53.7 vs 9.9 for K562) is explained by these before any further
  per-anchor comparison is made.
- **0d Cross-context reference definitions.** For a reference whose predicted effect is
  identical in both anchors (global-mean), the predicted effect difference is exactly
  zero and Pearson must be undefined. The response-adaptation diagnostic's summary reports non-zero Pearson for
  global-mean and no-change; verify whether `np.std(p) > 0` passed on float32 noise.
  If so, this is an implementation artifact: guard with a tolerance, mark those
  correlations undefined, and re-issue the cross-context table. Report which references
  have a defined Pearson and why.

Existing-export analysis outputs replace the corresponding statements in the response-adaptation diagnostic's analysis before any
GPU work starts.

## Native functional check (Tier 1): evaluation only

State **released STATE checkpoint, untrained** (N-native): released ST-HVG-Replogle checkpoint with its own 2000 → 328 basal
encoder (currently shape-skipped at load; must be loaded and verified), its native
one-hot perturbation vocabulary, `ctrl_cell_emb` = cached control HVG bags,
`pert_emb` = one-hot, output as STATE emits it. No Tx1 input, no ESM2 adapter.

- Preprocessing candidates, each an explicitly labelled variant, never a claim of
  recovered native preprocessing: (i) cached HVG values in checkpoint gene order — these
  are raw UMI counts from the raw source h5ads, not normalised expression; (ii) only if
  (i) fails the gate below, documented alternatives (for example log1p or total-count
  normalisation of the same values). Measured: (i) fails on every anchor (K562 18.9 ×
  no-change, identity advantage 0); (ii) as `log1p_norm` over HVG-panel row sums has its
  optimum at target 3,500 (K562 1.28, HepG2 0.84, HCT116 8.9, Jurkat 0.95; identity
  advantage 22–66% of loss except HCT116 3%). The K562 gate is missed at 27.5%. Batch index: 0 as in the composite, plus a sensitivity sweep over a handful
  of indices reported as spread, since the gem_group mapping is unknown.
- Panels: K562 internal holdout native-common (193), the other two source anchors'
  native-common panels, and Jurkat native-common (2,006, diagnostic re-evaluation).
- Gate: under the mean over native identities, or a non-targeting token if present, the
  prediction's response loss is within 10% of the no-change reference on K562. Passing
  establishes a working numeric path; failing on every candidate keeps the released STATE checkpoint on native inputs (B-native)
  "comparable evaluation cannot be established" and records what was tried.
- Report: loss ratio to no-change, identity advantage with the ten fixed derangements,
  existing-export bias decomposition, and cross-context metrics. This bounds what any interface
  built on this checkpoint can achieve and is a prerequisite for the native-basal-path variants below.

## Learning-rate isolation (Tier 2)

Two runs of the exact interface-only adaptation recipe (interface-only parameters, same sampler,
budget cap 50 epochs / 12,850 updates, patience 5, seed 0), changing only the interface
learning rate: **1e-6** and **1e-5**; 1e-4 exists. Compare at best checkpoints and at
matched completed epochs; LOAO is not required here. This quantifies H1's independent
contribution under fixed objective, frozen set, contexts and schedule. Exposure counts
differ from the seed-0 joint backbone (which replayed 64 conditions/rank every fourth update), so the contrast is
"LR under the response-adaptation recipe", not a reconstruction of the seed-0 joint backbone.

## Interface-variant comparison (Tier 3): variants with a held-out context

All variants train with the interface-only adaptation recipe (interface parameters only, 1e-4, same
sampler, cap and selection) and are scored with the full response-adaptation diagnostic battery: references,
identity derangements, own-coordinate removal, cross-context effect differences, existing-export bias
decomposition. Each variant runs the four LOAO folds; the Jurkat fold is reported as
a diagnostic re-evaluation and the LOAO mean is the transfer metric.

| Variant | Basal path | Perturbation path | Output | Untrained behaviour |
|---|---|---|---|---|
| adapted Tx1 basal encoder (V0) = interface-only adaptation | Tx1 → new encoder (328×2560) | ESM2 adapter | STATE hidden skip, decoder | reference (900 on val) |
| expression-residual interface (V1): null-subtracted residual | as adapted Tx1 basal encoder | ESM2 adapter; a fixed null token p₀ | X̂ = X⁰_c + (Ŷ(p_g) − Ŷ(p₀)) | exactly no-change |
| native basal path + trainable Tx1 context term (V2) | native 2000→328 encoder (loaded) + A_tx1(E_tx1), A_tx1 zero-init | ESM2 adapter | STATE hidden skip, decoder | same as the untrained released STATE checkpoint |
| native basal path (ESM-2 perturbation tokens) (V2-null), control | native encoder only, no Tx1 | ESM2 adapter | as the native basal path + trainable Tx1 context term | same as the untrained released STATE checkpoint |
| native basal path + Tx1 term + expression residual (V3) | as native basal path + trainable Tx1 context term | as expression-residual interface | as expression-residual interface | exactly no-change |

- The expression-residual interface uses the decoder in the space it was trained for (absolute expression) and takes the
  effect as the difference between the perturbed and null-token outputs; Ŷ(p₀) is
  computed once per context bag per update and reused. p₀ is the ESM2 adapter's output
  for a fixed null embedding, frozen at zero, so identity swaps still change only the
  perturbed branch.
- The native basal path + trainable Tx1 context term requires the native functional check to pass; otherwise it runs with the (i) values and is labelled
  accordingly. The native basal path (ESM-2 perturbation tokens) isolates whether Tx1 context conditioning adds anything to
  response prediction beyond the native basal path.
- **Initialisation check, per variant:** before training, the untrained expression-residual interface and native basal path + Tx1 term +
  expression residual must return exactly the control bag on every anchor; the native basal path + trainable Tx1 context term and
  the native basal path (ESM-2 perturbation tokens) must have a zeroed adapter final
  layer and batch index 0, and must equal the native forward under a zero perturbation
  vector (unit-tested). Failure of any of these is an implementation bug and blocks the
  arm. The untrained loss of the native basal path + trainable Tx1 context term and of the native basal path (ESM-2 perturbation
  tokens) is *also* compared against the non-targeting identity of the released STATE checkpoint, untrained, on the native validation panel and recorded, but not gated: a one-hot identity
  cannot equal a zero adapter output by construction, so the two are different inputs to
  the same weights, not the same model. This verifies the intended zero-effect behaviour
  of these designs; it is not a requirement placed on pretrained models in general.
- Order and stop rule (bounded iteration): expression-residual interface, native basal path (ESM-2 perturbation tokens),
  native basal path + trainable Tx1 context term, then native basal path + Tx1 term + expression residual only if the
  expression-residual interface or the native basal path + trainable Tx1 context term is
  kept. A variant is **kept** when, at its selected checkpoint, (a) internal-val loss
  ratio to no-change is below 1 with a paired interval excluding 0 on at least two of
  three source anchors in every fold, (b) the equal-fold mean of the per-fold held-out
  ratios (model over no-change on the held-out anchor) is below 1 with its synchronous
  gene-bootstrap interval excluding 1, and (c) identity advantage is positive with
  interval excluding 0 on every held-out fold. Per-fold held-out deltas and intervals
  are reported as diagnostics and are not individually required. Otherwise
  **discarded**. Two consecutive discards after a keep end the tier. Copying basal expression is not evidence of a learned response; (b) and (c) are
  required, and the cross-context effect-difference MSE must not exceed no-change on the
  held-out fold for the variant to be described as transferring effects.

## Head-seed stability (Tier 4): explicit context-slope head, GeneEffect side, independent of the native functional check, learning-rate isolation and interface-variant comparison

On the existing fixed-backbone head diagnostic feature cache, train the shared MLP head and the explicit context-slope head with head seeds 1 and 2 (same
recipe, same data-order seed). Reproduce explicit context-slope head − shared MLP head and explicit context-slope head − PCA8-ridge with the 27-context
paired bootstrap, the per-gene improvement distribution, and the sign of the Huber
difference. The explicit context-slope head remains the head control for every later response representation; a
future adapted response block is compared pairwise against the explicit context-slope head, and response quality is a
prioritisation signal, not a logical prerequisite for downstream value.

## Metrics, predicates and budget (autoresearch configuration)

- **Scope:** `src/model/p1b.py` (parameter provenance and variant construction),
  `src/data/p1b.py` (LOAO anchor roles), `src/experiments/p1b*.py` (native loader,
  `--holdout-anchor`, `--variant`), `src/eval/p1b.py` (bias decomposition, Pearson
  guard), new `hpc/run.sh p1c`; tests in `tests/test_p1c.py`. Production joint training
  unchanged.
- **Primary metric:** held-out-context response loss ratio to no-change (LOAO mean),
  lower is better. **Secondary:** internal-val ratio, identity-advantage share,
  cross-context effect-difference MSE ratio, shared-bias fraction.
- **Verify:** `hpc/run.sh p1c compare` emits `p1c/comparison/summary.csv` with one row
  per (variant, fold) and a `kept` boolean computed from the predicate above; the
  local analysis reads that file only.
- **Iterations:** 3 planned variants plus 1 conditional (native basal path + Tx1 term + expression residual); 2 LR arms; 1 native
  evaluation; 4 head runs. Hard ceiling: 7 training units. Plateau: two consecutive
  discards after a keep.
- **Compute:** one response-adaptation diagnostic arm took under 3.5 h on one GPU at the 50-epoch cap. Native functional check under
  1 h; learning-rate isolation two GPUs in parallel, ≤ 3.5 h; interface-variant comparison four folds per variant on four GPUs,
  ≤ 3.5 h per variant; head-seed stability minutes per head. Whole round ≈ 20 GPU-hours if every
  variant runs.
- **Gates before GPU:** existing-export analysis delivered and reviewed; unit tests for initialisation
  checks, zero-init gates, parameter ownership per variant, LOAO membership (held-out
  anchor absent from baselines, sampler, selection), null-token caching equivalence,
  native-weight load report, and the Pearson guard.

## Leakage and claim boundaries

Jurkat labels were observed before this design; no Jurkat number selects a checkpoint,
a variant, or a preprocessing candidate. LOAO held-out anchors enter nothing on their
fold. GeneEffect labels and the test split are untouched. Response improvements are
response findings; GeneEffect value is established only by a later pairwise comparison
of an adapted response block against the explicit context-slope head under the fixed-backbone head diagnostic protocol. Stronger cross-context claims
require contexts outside the four anchors.

## Standard for any future pretrained response model (ST-Tahoe or other)

Pretrained weights with compatible tensor shapes do not guarantee that the composed
model retains a functional response predictor. A candidate checkpoint must first pass
the native functional check on its native pathway, then show how much of that functionality survives the
Tx1 and genetic-perturbation interfaces under the interface-variant comparison's predicate, before it replaces
ST-HVG-Replogle.

## Round 2: log space

Round 1 ran every arm in count space and established that the composite fed raw UMI
counts to a decoder trained on `normalize_total` + `log1p` expression; nothing was kept
and the native-basal-path family was uninformative as configured
([result](../results/p1_response_pathway_diagnostics/README.md)). Round 2 repeats the interface-variant comparison
with every prepared fold built under `--transform log1p_norm --target-sum 3500`: control
bags and targets are transformed at preparation, and the loss, the references, the
identity derangements and the cross-context metrics are all computed in that space.
Arms: adapted Tx1 basal encoder, expression-residual interface, native basal path (ESM-2 perturbation tokens),
native basal path + trainable Tx1 context term, then native basal path + Tx1 term + expression residual under the same
keep rule; the released STATE checkpoint, untrained, at the same transform is the reference arm on every fold (batch index 0); the two
learning-rate isolation arms rerun on the Jurkat fold. The existing-export analysis and head-seed stability are not repeated. Recipe, folds, budget,
selection and predicate are unchanged. The K562 gate miss (27.5%) is carried as a caveat:
the HVG-panel row sum is a proxy for the library size, so "native space" is approximate.
Round-1 results are retained as the record of the numeric-interface defect and are never
pooled with round 2. If no log-space variant is kept and the K562 reference arm stays
above the gate, the next step is to rebuild preparation with whole-library normalisation
from the raw h5ads and repeat the reference arm before any further variant.

Implementation: `train_variant` accepts any registered transform and records it in
`training.json`; `hpc/p1c_pipeline.sh` takes `PIPELINE_TRANSFORM`, `PIPELINE_TARGET_SUM`,
`PIPELINE_SKIP_TIER0` and `PIPELINE_NATIVE_BATCH_INDICES`.

