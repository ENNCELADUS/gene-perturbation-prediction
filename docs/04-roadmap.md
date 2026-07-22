# Experiment Roadmap: Generalizable SL Discovery

**Status:** active plan. The first HCT116 frozen-K562-backbone single-gene audit
completed with a negative transport result; no Bridge A/Bridge B, formal SOTA,
or pairwise cross-cell-line result has completed.
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
transport audit is complete and consumed; it does not count as pairwise SL
evaluation.

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
3. prior exp08 pooled-response pair head as a negative/internal control;
4. zero-shot Bridge A and Bridge B scores aggregated by a prespecified
   context-free rule; and
5. optional SL-label-calibrated heads, reported separately.

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
- direct K562 transfer without contextual adaptation;
- observed-response upper bound where permitted; and
- frozen-backbone versus trained-context-adapter variants.

The existing K562 exp05 run is prior evidence. The one-shot HCT116
frozen-backbone GeneEffect transport audit is complete and negative: direct K562
GeneEffect transfer retained Spearman 0.554, while the frozen response head had
Spearman -0.001 and collapsed output variance. This rejects direct transport of
the frozen K562 response-to-fitness head, not the conserved gene-dependency
prior. HCT116 labels are now unsealed; this line is consumed and cannot be
recycled into development or reused as a formal held-out line. See the
[`closeout`](results/exp05-hct116-frozen-backbone-transport.md). The audit does
not count as pairwise or cross-cell-line SL generalization.

**Exit:** single-gene response/fitness transfer is measured on development cell
lines without opening pairwise labels from untouched test lines.

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

No held-out line may be recycled into development after its labels are opened. A
failed or underpowered line remains in the report.

**Exit:** a named-context or population-level CELL-LINE-GENERALIZATION verdict and
the SPECIFICITY verdict in `02`, or “not evaluable” if Phase 0 could not establish
an eligible contract.

## 9. Phase 7 — mechanistic validation

### K562 anchor

- acquire and provenance-check Horlbeck 2018 continuous fitness GI;
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
