# Plan: K562 Mechanism Kill-Test and GeneEffect Cross-Cell-Type Generalization

**Status:** T1 completed 2026-07-22 and is **negative** (composition mechanism
paused; [result](../results/exp05-bridge-a-horlbeck-kill-test.md)). T2 is the
active near-term track, **sharpened 2026-07-23 to a few-shot cross-cell-line
GeneEffect backbone**; its concrete architecture, data split, and current
per-phase execution status are tracked in
[design](2026-07-23-tx1-st-geneeffect-backbone-design.md) §3/§6, which
supersedes the abstract sketch in §3 below.
Authority: [`../01-blueprint.md`](../01-blueprint.md) (frozen contract) and
[`../02-acceptance-criteria.md`](../02-acceptance-criteria.md) (frozen bar) govern;
[`../04-roadmap.md`](../04-roadmap.md) is the tracked phase plan this refines (see
its §1.1). A local model-boundary design note is kept at
`docs/superpowers/specs/2026-07-17-virtual-cell-sl-composition-design.md`.

## 1. Objective and first principles

Two independent development tracks were set, each retiring one load-bearing risk
before any cross-cell-line SL claim is attempted. **T1 has completed and is
negative** (composition paused); **T2 is the active track**:

- **T1 — does the interaction mechanism work at all, in the home context?** A
  single-gene backbone composed into a pair score is only worth extending across
  contexts if it recovers *measured* K562 epistasis. Test it against Horlbeck 2018
  before building any multi-cell-line machinery.
- **T2 — can single-gene GeneEffect become cell-line-specific at all?** The
  context-specific part of any SL score, `s(a,b|c) - q(a,b)`, can only exist where
  dependencies vary across lines. If the backbone cannot produce line-specific
  GeneEffect on that slice, no context-specific SL is possible. Test it on DepMap.

Neither track opens pairwise labels from an untouched held-out cell line; neither
is itself the CELL-LINE-GENERALIZATION result.

## 2. Track 1 — K562 Bridge-A-vs-Horlbeck mechanism kill-test

### 2.1 The bridge choice is forced by the frozen checkpoint

exp05 encodes each perturbation as a per-gene ESM2 embedding and predicts
single-gene GeneEffect from `basal_state + pert_emb`; combination conditions were
excluded from training. Therefore:

- **Bridge B is not run on this checkpoint.** It has no trained two-gene
  perturbation operator: perturbations enter as per-gene embeddings, so a
  combined-embedding input is an untrained out-of-distribution query, not a
  validated joint-fitness prediction. (The additive null itself needs no joint
  input — it is `ψ(ĉ_a, ĉ_b)` computed from two single-gene passes; what the
  checkpoint cannot supply is a trustworthy joint `ĉ_ab` to test against that
  null.) A native double-input model needs combinatorial training data (§4) and is
  out of scope here.
- **Bridge A is natural.** `basal_state` is just a cell state, so feed the
  *observed* `a`-perturbed K562 cells as basal and query `b`. It reuses only frozen
  components; note that feeding an `a`-perturbed state as basal is itself a
  basal-distribution shift (STATE was trained on unperturbed control basal), which
  the state-shift control (§2.3) is designed to bound.

### 2.2 Score and the real-intermediate argument

For an ordered step, define the co-dependency spike

```text
Δ(a→b) = ĉ[b | control_state] − ĉ[b | a-perturbed_state]
```

(GeneEffect is more negative when more essential, so `Δ > 0` means `b` becomes
more essential once `a` is lost). Symmetrize:

```text
s_A(a,b) = ½ · [ Δ(a→b) + Δ(b→a) ]
```

When `a` is an observed Replogle perturbation, the `a`-perturbed state fed as
basal is **real measured data**, not a model-simulated state — so Bridge A here
avoids compounding a simulated-state error onto the read-out. Both GeneEffect
terms in `Δ` are still model predictions, as any GeneEffect read-out is. Whether
the resulting `s_A` corresponds to Horlbeck's fitness GI — an epistasis deviation
of the double-knockdown growth phenotype from the single-gene expectation — is
precisely the hypothesis this kill-test evaluates, not an assumed identity.

### 2.3 Data, coverage, controls

- Target: the frozen `gi_score` from
  [`../data/horlbeck-2018-k562-gi.md`](../data/horlbeck-2018-k562-gi.md); `s_A`
  should rank pairs concordantly with `−gi_score` (more negative `gi_score` =
  synergistic/synthetic-sick-lethal ↔ larger `s_A`). Strong-SL slice
  `gi_score < -3.0` (1,523 pairs) reported separately.
- Coverage decision (require both genes observed so either direction uses a real
  intermediate): exp05-ready bound 408 genes / 83,028 pairs; broader
  observed-response bound 436 genes / 94,830 pairs
  ([coverage](../results/horlbeck-k562-exp05-coverage.md)). The exp05-ready bound
  (both genes in the trained pool) is the in-distribution default for a
  frozen-checkpoint test; the broader bound adds pairs whose queried gene is
  outside the exp05 pool — out-of-distribution for the ESM2 perturbation head.
  Register which bound is the candidate universe before the formal run.
- Controls: (i) `Δ` against a generic non-partner-perturbed state (state-shift
  control — the `a`-specific signal must exceed a generic basal shift); (ii)
  observed co-dependency where available; (iii) a GenePert-style linear-on-ESM2
  ablation. `s_A` is symmetric by construction, so a swap-invariance check
  verifies only the implementation, not the science.

### 2.4 Kill-test versus formal verdict

The immediate run is a **development feasibility diagnostic**: is there rank
correlation with `gi_score` on covered pairs, in the SL direction, beyond the
state-shift control? A null here means the interaction mechanism fails in its
best-supported context — do not extend across contexts before rethinking
composition. It is **not** a MECHANISTIC verdict: the formal `02` §6
EXISTS/MEANINGFUL result additionally requires a Phase-0-registered `rho_min`, a
frozen candidate universe, and disjointness from any SL-label calibration
(trivially satisfiable now, since no calibration exists, but materialized before
the formal run).

This instantiates roadmap Phases 3 (frozen backbone) + 4 (Bridge A) + 7 (K562
anchor), pulled forward. It is independent of the Feng `q(a,b)` track
(Phases 1–2, 5).

## 3. Track 2 — GeneEffect cross-cell-line generalization and few-shot adaptation (DepMap, fixed split)

**Concrete architecture (2026-07-23):** the model, data roles, and execution plan
are specified in
[`2026-07-23-tx1-st-geneeffect-backbone-design.md`](2026-07-23-tx1-st-geneeffect-backbone-design.md)
(Tx1-3B-conditioned ST + rebuilt hybrid head; HVG-ST encoder-unseen control). The
data are re-scoped there to a **fixed most-train / few-test split (no
cross-validation)**: the four CRISPRi Perturb-seq lines (K562, HCT116, Jurkat,
HepG2) are training anchors, and a lineage-stratified few of the Tahoe
DMSO-basal∩DepMap lines are the fixed unseen test. The T2 contract below (testbed,
differentially-essential slice, baselines) stands and governs; the design doc
specifies how it is met.

**Objective (active round).** Build `F(X_c, c, g) -> GeneEffect`, a
context-conditioned single-gene predictor that (a) generalizes across held-out
cancer lines seen only through their basal single-cell state
and (b) adapts to a held-out line from a few of its own labels (k-shot line
adaptation), and is accurate against real DepMap GeneEffect on the
differentially-essential slice. This is the Phase 3 backbone-transfer exit,
reported as backbone transfer, not SL. The model-adaptation method is pending
selection and does not change this contract.

### 3.1 Testbed: DepMap, not Feng

The Feng2024 benchmark has **no cell-type axis**: entities are genes
(`fin_entities.csv`), splits are gene-based CV1/CV2/CV3, and the only
cell-line-resolved artifact is a locally derived K562-only subset (`ACH-000551`).
GeneEffect generalization is therefore evaluated on **DepMap GeneEffect × CCLE**
baseline expression across many lines. DepMap plays three roles kept strictly
separate: test labels, multi-line supervision, and — as a plain
omics→GeneEffect regression — the baseline to beat. The virtual cell is the
method, not the regression.

### 3.2 Mechanism and modality bridge

The backbone's only cell-line input is the basal state, so give the C-head a
cell-line context and train it across many lines. Line-scarcity note (distinct from
k-shot line adaptation below): the four Perturb-seq lines cannot alone define a
context→GeneEffect function; the design supplies head breadth from the ~28–30 Tahoe
DMSO **basal single-cell** lines (their DepMap GeneEffect via predicted response),
with the Perturb-seq lines as the observed-response subset that additionally
exercises the perturbation pathway. CCLE bulk enters as a baseline (and a possible
later genomic-context augmentation), not as the method's context channel — see the
design doc.

### 3.3 Metric and controls

Over the fixed few held-out Tahoe basal lines, in two regimes: **zero-shot** (no
held-out-line labels) and **k-shot line adaptation** (the predictor may see k
GeneEffect labels from the held-out line for a prespecified schedule of k; the
adapted predictor is scored on the remaining, disjoint genes of that line). Primary
score = rank correlation on the **differentially-essential slice** — genes with
high cross-line GeneEffect variance and non-common-essential, both computed on
**training lines only** so the held-out line never informs slice membership. Beat
each baseline on the **paired difference** of rank correlation (method − baseline);
the gate is at a registered k (k=10): the macro-averaged paired difference has a
**line-level bootstrap 95% CI (cell lines as inferential units) excluding zero**,
with per-line CIs and fraction-positive as descriptive support and k=0 as a stress
point (estimator and `rho_min` registered in Phase 0). Baselines: (i) single
source-line transfer (copy-K562) and the training-line mean / nearest line; (ii)
lineage-identity-only; (iii) the CCLE-bulk→GeneEffect regression and a
pseudobulk-basal→GeneEffect regression. In the k-shot regime each baseline gets its
natural k-shot correction (e.g. copy-K562 plus a k-label offset/calibration), so
added machinery must beat "copy-K562 + k labels", not just copy-K562. Report the
**few-shot curve**: slice rank correlation and paired-difference-vs-baseline as a
function of k. Aggregate correlation over the full evaluated gene set is reported
but is inflated by pan-essential genes and is not the gate (cf. the K562→HCT116
transfer Spearman 0.554 on shared genes).

### 3.4 Scope

Single-gene **backbone/component** evidence (roadmap Phase 3 exit), not
CELL-LINE-GENERALIZATION SL (`02` §4), which still needs pairwise labels in
untouched lines. On DepMap-only lines only single-gene GeneEffect is testable —
no perturbation or GI data there. HCT116 is consumed as a *test* line but is
eligible as a *development/training* line here.

## 4. Norman and native multi-gene input (auxiliary, deferred)

Norman 2019 is CRISPR**a** (activation) and transcriptomic, not knockdown fitness
GI; per the contract it is auxiliary with a modality caveat and cannot validate
KO-SL. Its only role is a separate, CRISPRa-quarantined capability testbed for
whether a composition operator beats an additive null on combinatorial
*transcriptional* response (GEARS-style). A native multi-gene-input model that
learns non-additivity requires combinatorial training data (Norman is the only
large perturb-seq source) and is deferred; it is not needed for T1, which
validates the frozen-checkpoint Bridge A directly against Horlbeck.

## 5. Execution order

T1 has completed and is negative, pausing the composition mechanism; T2 is now the
active track, run independently of the Feng-axis reproduction (Phases 1–2). T2
gates context-specific GeneEffect — a precondition for any future context-specific
SL, and the substrate a redesigned composition would compose. It touches no
untouched held-out cell-line SL labels and produces no formal `02` verdict until
its Phase-0 registrations are frozen. Results enter `../results/<slug>.md` only
after each analysis runs.

## 6. Claim boundaries

- A Bridge-A/Horlbeck correlation is K562 mechanism only, never multi-cell-line
  mechanism.
- Single-gene GeneEffect cross-cell-line transfer is backbone transfer, never SL
  transfer or cross-cell-line SL.
- No SL from single-gene essentiality; the interaction null is explicit.
- Norman CRISPRa is never aligned to knockout labels without the modality caveat.
- No planned number is reported as a result.
