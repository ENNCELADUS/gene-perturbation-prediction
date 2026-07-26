# Experiment Roadmap: Generalizable SL Discovery

**Status:** active plan. The first HCT116 frozen-K562-backbone single-gene audit
completed with a negative transport result. **T1, the K562 Bridge-A-vs-Horlbeck
mechanism kill-test, has completed and is negative**: the composed frozen backbone
does not recover measured K562 epistasis (|Spearman| < 0.01; AUROC(s_A -> strong-SL)
approximately 0.52), so the composition mechanism is paused for redesign and not
extended across contexts ([result](results/exp05-bridge-a-horlbeck-kill-test.md)).
With composition paused, the active near-term development focus is **T2, sharpened:
a few-shot cross-cell-line GeneEffect backbone** — an accurate, context-conditioned
single-gene predictor that both bridges depend on, pursued as the Phase 3
backbone-transfer exit and reported as backbone transfer, not as an SL claim
(§1.1, §5). T2's own Phase A (data-audit/manifest freeze) and Phase B (Tx1-3B
basal embeddings) are **complete**; Phase C (ST response model) is code-complete
with a training run in progress and no checkpoint yet; Phase D (rebuilt hybrid
head) is partially built and its training runner is under construction — no
Phase 3 result exists. Formal SOTA reproduction, Bridge B, and any pairwise
cross-cell-line result remain not started.
**Contract:** [`01-blueprint.md`](01-blueprint.md) · **Acceptance criteria:**
[`02-acceptance-criteria.md`](02-acceptance-criteria.md) · **Related work:**
[`03-literature-review.md`](03-literature-review.md)

## 1. Objective and order of operations

Build and evaluate a general synthetic-lethality discovery model in two stages:

1. compare a context-agnostic pair score `q(a,b)` with the Feng2024 SOTA on the
   official benchmark; then
2. test a context-conditioned score `s(a,b | c)` on untouched held-out cancer
   cell lines.

K562 is the current implementation context for the virtual-cell backbone and one
mechanistic validation context. It is not the final target population. Work does
not advance to a stronger claim merely because an earlier dataset is unavailable;
the corresponding claim remains not evaluable.

### 1.1 Immediate execution order (updated 2026-07-23)

T1 (below) has completed and is negative, pausing the composition mechanism; the
near-term work is now a single track — **T2, the few-shot cross-cell-line
GeneEffect backbone**. It runs independently of the Feng-axis reproduction
(Phases 1–2), opens no untouched held-out cell-line labels, and yields no formal
`02` verdict before its Phase 0 registrations are frozen. Concrete design:
[`specs/2026-07-23-tx1-st-geneeffect-backbone-design.md`](specs/2026-07-23-tx1-st-geneeffect-backbone-design.md).

- **T1 — K562 mechanism kill-test (completed 2026-07-22; negative).** Composed the
  frozen exp05 backbone with **Bridge A** (observed `a`-perturbed Replogle state as
  basal, query `b`) over the 83,028 exp05-ready covered pairs (1,281 strong-SL at
  `gi_score < -3.0`), with independent-N panel matching and both pooler reference
  conventions. The symmetrized co-dependency spike does **not** correlate with the
  frozen Horlbeck `gi_score` (|Spearman| < 0.01; AUROC(s_A -> strong-SL)
  approximately 0.52, below the single-gene GeneEffect floor). Bridge B was not run
  on this checkpoint: its ESM2 per-gene perturbation embeddings admit no trained
  two-gene operator, so a combined-embedding input is an untrained out-of-distribution
  query rather than a validated joint-fitness prediction. Per the kill-test rule the
  composition mechanism is **paused for redesign and not extended across contexts**.
  Development diagnostic, not the MECHANISTIC verdict (§9 and `02` §6). Result:
  [`results/exp05-bridge-a-horlbeck-kill-test.md`](results/exp05-bridge-a-horlbeck-kill-test.md).
- **T2 — few-shot cross-cell-line GeneEffect backbone (active near-term focus;
  execution under way).** Build `F(X_c, c, g) -> GeneEffect`: a context-conditioned
  single-gene predictor that predicts DepMap GeneEffect for cancer lines seen only
  through their basal single-cell state — not Feng2024, which has no cell-type axis
  — and (b) adapts to a held-out line from a few of its own labels (k-shot line
  adaptation). Data uses a **fixed 28 train / 5 validation / 9 test split (no
  cross-validation)**: the four CRISPRi Perturb-seq lines (K562, HCT116, Jurkat,
  HepG2) supply observed-response ST supervision, head anchors, and remain in
  training (their supervision is not spent on model selection); 24 of the 38
  Tahoe DMSO-basal∩DepMap lines train the head on predicted response; 5 more,
  lineage-stratified with a fixed seed and drawn only from the 29 non-anchor
  training-pool lines, are held out for validation/model-selection only; the
  remaining 9, lineage-stratified and frozen at Phase A, are the untouched unseen
  test. Scored on the differentially-essential slice (587 genes; slice membership
  fixed on training lines only) against copy-K562 transfer, cross-line mean,
  nearest-line transfer, lineage-only, a CCLE-bulk regression, and a
  pseudobulk-basal regression, with a few-shot curve (accuracy vs. k held-out-line
  labels). This is the Phase 3 single-gene backbone-transfer exit, reported as
  task-data-held-out transfer (test lines are inside the Tx1 encoder's Tahoe
  pretraining), not cross-cell-line SL. Phase A (data audit/manifest freeze) and
  Phase B (Tx1-3B basal embeddings, verified for all 42 manifest lines) are
  **complete**; Phase C (ST response model) is code-complete and dry-run-validated
  on real data for both arms, but **no ST checkpoint has been produced** and a
  training run is in progress; Phase D (rebuilt hybrid head) is **partially
  built** — the head, moment pooling, losses, few-shot calibrator, and evaluator
  exist, but the runner that trains the head is under construction. No result
  exists yet. Concrete architecture:
  [`specs/2026-07-23-tx1-st-geneeffect-backbone-design.md`](specs/2026-07-23-tx1-st-geneeffect-backbone-design.md)
  (Tx1-3B-conditioned ST + rebuilt hybrid head, HVG-ST encoder-unseen control).

## 2. Phase 0 — freeze data and evaluation contracts

### 2.1 Official Feng2024 contract

- Verify the official 9,845-gene entity table, five-fold CV1/CV2/CV3 caches,
  `Rand` 1:1 labels, candidate construction, and `cal_metrics` behavior.
- Record exact local checkout commit, archive checksums, seeds, and metric code.
- Keep the official benchmark separate from the derived K562-DepMap-filtered
  subset. The latter is an ablation/coverage surface only.
- Materialize one immutable manifest per split and fold before model runs.

**Exit:** a small metric-parity test reproduces a known official row or explains
any mismatch with a checked artifact and no hidden protocol change.

### 2.2 Cell-line data audit

Inventory candidate datasets containing pairwise SL/GI labels with explicit cell
line, intervention modality, continuous/binary outcome, negative definition, and
gene/pair coverage. Separately inventory perturbation-response and DepMap inputs
needed to construct `F(X_c,g)`.

For every candidate cell line, record:

- whether it can be train, validation, or untouched test;
- whether labels are measured, curated, or sampled;
- intervention modality and time scale;
- overlap with model inputs and Feng2024 genes;
- known exposure in foundation-model/checkpoint pretraining;
- source-level leakage/circularity with SynLethDB, Feng2024, and calibration
  labels; and
- whether per-anchor ranking and line-level inference are statistically evaluable.

For every proposed label source, freeze the relevance/sign rule, candidate
universe, handling of unmeasured pairs, minimum anchors/positives/coverage,
assay/time-scale compatibility, and study/batch-confounding analysis. A
prospective simulation fixes the minimum practically meaningful effects and the
paired hierarchical estimators before any formal model result is opened. If an
held-out line/assay contributed to a calibration source, purge every label derived
from that line/assay from `q` and contextual calibration, then use pair-disjoint
records from the assay for evaluation. If source lineage or the purge cannot be
verified, declare the line ineligible.

The current K562 resources are included in the audit. The HCT116 single-gene
transport audit is complete and its audited GeneEffect labels are open. HCT116
may be assigned to a declared GeneEffect component training/development role,
but it is no longer eligible as an untouched test line and does not count as
pairwise SL evaluation.

**Exit:** at least two training cell lines and two untouched test cell lines with
eligible pairwise labels, or an explicit `CELL-LINE-GENERALIZATION: not evaluable`
decision. Exact dataset roles, pretraining-exposure status, hashes, effect-size
thresholds, and estimators are frozen before formal modeling. Two held-out lines
support claims about those named contexts; a broader claim additionally requires
a powered design with cell lines as inferential units.

## 3. Phase 1 — reproduce the Feng2024 SOTA bar

Run the official model ladder under one isolated, verified benchmark environment.
At minimum reproduce:

- SLMGAE and KR4SL as the strong cold-start targets;
- KG4SL as a named reference;
- the best remaining official method on CV2/CV3;
- the dependency-only floor; and
- a degree probe as the CV1 gameability diagnostic.

Use the official five folds and report NDCG@10, MAP@10, NDCG@{20,50}, AUROC,
AUPR, and F1. Do not use the K562-filtered reproduction as the formal SOTA bar.

**Exit:** the best eligible reproduced comparator is known on CV2 and CV3 and all
input/metric differences are controlled.

## 4. Phase 2 — establish the context-free model `q(a,b)`

Build the smallest ladder that isolates where signal comes from:

1. gene marginals and dependency-only floor;
2. ESM2/GenePT identity embeddings with a linear swap-invariant head;
3. zero-shot Bridge A and Bridge B scores aggregated by a prespecified
   context-free rule; and
4. optional SL-label-calibrated heads, reported separately.

No SL graph topology enters feature construction. Each more complex row must beat
the simpler row before its added machinery is credited.

**Exit:** five-fold Feng2024 CV1/CV2/CV3 predictions and metrics for the full
ladder, with train-only model selection and coverage accounting. The selected
`q(a,b)` artifact is produced by a prespecified retrain on all admissible
calibration data or a fixed ensemble of all folds, using train-only-selected
hyperparameters. It is never the best test-fold checkpoint. The artifact and its
auxiliary-data exposure manifest are frozen before Phase 3; contextual fitting
cannot update it.

## 5. Phase 3 — generalize the virtual-cell backbone across contexts

Extend the forward model from its current K562 implementation to a declared
multi-cell-line interface:

```text
F(control state X_c, cell-line context c, perturbation gene g)
    -> predicted response
    -> declared single-gene fitness readout
```

Implementation proceeds only after Phase 0 fixes train/validation/test cell-line
roles and records known checkpoint-pretraining exposure. Unknown exposure narrows
the eventual claim to task-data-held-out transfer. Required controls are:

- cell-line identity/lineage only;
- gene identity only;
- direct K562 transfer without contextual adaptation (copy-K562);
- cross-line mean and nearest-line transfer;
- observed-response upper bound where permitted; and
- frozen-backbone versus trained-context-adapter versus k-shot-line-adapted
  variants, reported as an accuracy-vs-k curve.

The existing K562 exp05 run is prior evidence. The one-shot HCT116
frozen-backbone GeneEffect transport audit is complete and negative: direct K562
GeneEffect transfer retained Spearman 0.554, while the frozen response head had
Spearman -0.001 and collapsed output variance. This rejects direct transport of
the frozen K562 response-to-fitness head, not the conserved gene-dependency
prior. HCT116 labels are now unsealed. HCT116 may enter later GeneEffect component
training or development if that role is declared in the data manifest, but it
cannot then be presented as an untouched held-out line. This does not change the
completed audit's negative verdict. See the
[`closeout`](results/exp05-hct116-frozen-backbone-transport.md). The audit does
not count as pairwise or cross-cell-line SL generalization.

**Exit:** on the differentially-essential slice (slice membership fixed on
training lines only), over the fixed few held-out lines, the context-conditioned
predictor beats copy-K562, cross-line mean, nearest-line transfer, lineage-only,
a CCLE-bulk regression, and a pseudobulk-basal regression on the paired
rank-correlation difference. The T2 gate is at a registered k (k=10): the
macro-averaged paired difference has a **line-level bootstrap 95% CI (cell lines as
the inferential units) excluding zero**, with per-line CIs and fraction-positive
reported as descriptive support and k=0 reported as a stress point — all measured
without opening pairwise labels from untouched test lines. A predictor that only
matches copy-K562 on this slice is reported as conserved-prior transfer, not
context-specific GeneEffect.

## 6. Phase 4 — implement contextual Bridge A and Bridge B

### Bridge A

- sequentially simulate `a` then `b`, and `b` then `a`, in the same cell-line
  context;
- symmetrize the dependency spike; and
- compare with observed co-dependency and context-free dependency controls.

### Bridge B

- simulate the joint perturbation in the same context;
- compute both additive and min/HSA interaction residuals; and
- compare with a GenePert-style linear/additive baseline and any eligible GEARS
  interaction residual.

Both bridges emit zero-shot scores before any SL-label calibration. Failures of
the underlying response model, fitness head, composition, and label calibration
are reported separately.

**Exit:** deterministic per-pair, per-cell-line scores with swap-invariance tests,
declared nulls, and no access to held-out cell-line pair labels.

## 7. Phase 5 — formal Feng2024 evaluation

- Evaluate `q(a,b)` on all five official folds.
- Compare head-to-head with the reproduced SOTA and simple model ladder.
- Bind the headline to CV2/CV3 NDCG@10; report CV1 as diagnostic.
- Repeat the comparison on the non-pan-essential slice.
- Report zero-shot composition separately from calibrated heads.

**Exit:** one of the BEAT-SOTA verdicts in `02`, including a negative verdict if
the bar is not cleared.

## 8. Phase 6 — formal held-out-cell-line evaluation

After every model and threshold is frozen:

- open each held-out cell line once;
- score the fixed pair universe for that line;
- compare `s(a,b | c)` with `q(a,b)`, gene marginals, context-only controls, and
  the strongest eligible contextual baseline;
- report per-line metrics, coverage, and uncertainty before the macro-average;
- require every prespecified eligible held-out line to pass the named-context
  criterion; no subset may be selected after opening labels;
- repeat on the non-pan-essential slice; and
- test whether the contextual increment remains after lineage/context covariates.

A cell line assigned to the formal Phase 6 held-out role may not be recycled into
development after its labels are opened. A failed or underpowered line remains
in the report. The earlier HCT116 single-gene component audit is not a Phase 6
held-out SL assignment; under the current protocol HCT116 may be used for
GeneEffect development but cannot later be selected as an untouched Phase 6 line.

**Exit:** a named-context or population-level CELL-LINE-GENERALIZATION verdict and
the SPECIFICITY verdict in `02`, or “not evaluable” if Phase 0 could not establish
an eligible contract.

## 9. Phase 7 — mechanistic validation

### K562 anchor

- use the acquired, provenance-checked Horlbeck 2018 continuous fitness GI and
  its frozen [exp05 coverage audit](results/horlbeck-k562-exp05-coverage.md);
- use evaluation pairs/genes disjoint from SL-label calibration;
- use Adamson UPR only as a qualitative transcriptomic check; and
- exclude Jost/Replogle dual-sgRNA from GI claims.

### Non-K562 anchor

Use at least one eligible non-K562 combinatorial perturbation context with a
continuous or prespecified binary GI definition. Audit the Horlbeck Jurkat arm as
the first candidate; do not assume eligibility until provenance, coverage, and
independence are verified. Apply the same frozen interaction null and prevent
overlap with any contextual calibration labels.

**Exit:** K562-only or multi-cell-line MECHANISTIC verdict, stated at exactly the
contexts supported by the data.

## 10. Required artifacts

Every formal stage writes:

- immutable data-role and split manifests with hashes;
- configuration and code/checkpoint identity;
- fit-access and held-out-access audits;
- per-pair predictions with fold and cell-line provenance;
- per-anchor and aggregate metrics with confidence intervals;
- gene/pair/anchor/cell-line coverage tables; and
- a `docs/results/<slug>.md` note only after the analysis completes.

Status is then synchronized across this file, `docs/README.md`, and root
`README.md`. Planned numbers never enter result documents.

## 11. Stop and downgrade rules

- If official metric parity cannot be established, do not claim a SOTA
  comparison.
- If fewer than two eligible training and two eligible held-out cell lines exist,
  do not claim general cross-cell-line SL discovery.
- If the contextual model only transfers single-gene GeneEffect, report backbone
  transfer, not SL transfer.
- If gains vanish off pan-essential pairs, downgrade to essentiality signal.
- If `s(a,b | c)` does not beat `q(a,b)`, report general pair-prior transfer, not
  context-specific SL.
- If Bridge B does not beat the additive/linear ablation, do not credit the
  virtual-cell machinery.
- If measured GI is unavailable, MECHANISTIC is not evaluable.
