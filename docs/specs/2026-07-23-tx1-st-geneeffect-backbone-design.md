# Design: Tx1-conditioned ST GeneEffect backbone (few-shot, cross-cell-line)

**Status:** design approved 2026-07-23; test-set re-scoped 2026-07-23; not yet
built. Concrete architecture for roadmap **T2 / Phase 3** (the few-shot
cross-cell-line GeneEffect backbone). No listed step has run; no number here is a
result. Line counts (49 Tahoe DMSO lines, 38 with DepMap) are audited-from-disk
facts, not model results; Phase A freezes the final manifest.
**Authority:** [`../01-blueprint.md`](../01-blueprint.md) (frozen contract) and
[`../02-acceptance-criteria.md`](../02-acceptance-criteria.md) (frozen bar) govern;
[`../04-roadmap.md`](../04-roadmap.md) §1.1/§5 is the phase plan this refines; it
supersedes the abstract T2 sketch in
[`2026-07-22-k562-mechanism-and-geneeffect-generalization-plan.md`](2026-07-22-k562-mechanism-and-geneeffect-generalization-plan.md)
§3 with a concrete model.

## 1. Objective and scope

Build `F(X_c, c, g) -> GeneEffect`: a context-conditioned single-gene predictor
that predicts real DepMap GeneEffect on cancer cell lines whose **only** input is
basal single-cell state (no genetic perturbation data of their own), trained on
the few lines that carry genetic Perturb-seq. It adapts to a held-out line from a
few of its own labels (k-shot), and is accurate on the differentially-essential
slice. This is the **Phase 3 single-gene backbone-transfer exit**, reported as
**task-data-held-out cross-line GeneEffect transfer** (the lead encoder is
pretrained on the target lines; see §7) — **not** an SL result, **not** a
cross-cell-line SL claim (roadmap §5; `02` §4), and **not** an unseen-context
claim. It opens no pairwise labels from any untouched cell line.

Motivation: the composition mechanism (Bridge A) is paused after a negative
kill-test, and the frozen K562 response-to-GeneEffect head did not transfer to
HCT116 — the measured failure (collapsed output variance, Spearman approximately
0) lives **downstream of the encoder**, in the K562-fit GMM pooler and MSE head,
which regress to the K562 mean off-distribution. This design fixes that failure
and tests whether a stronger cell-state encoder plus multi-line supervision yields
a cross-line-generalizable single-gene backbone that predicts fitness for lines
seen only through their basal transcriptome.

## 2. Architecture

### 2.1 Inference path (test line `c`, gene `g`)

```text
control / DMSO single cells of c
   -> frozen Tx1-3B encoder            (offline; .obsm cell embeddings)
   -> basal cell embeddings
   -> ST (cell-set transformer):        (basal set + one-hot gene g)
        predicted post-perturbation response          [output_space = gene]
   -> moment pool(predicted response) ++ basal-embedding context summary
   -> context-conditioned head (rank-trained)
   -> GeneEffect scalar Ĉ(g | c)
   -> [if k labels for c available] light per-line calibration
```

### 2.2 Component decisions

| Component | Decision | First-principles reason |
| --- | --- | --- |
| Prediction path | ST perturbation-forward → scalar GeneEffect | Keeps the virtual-cell forward model (the composition backbone the broader program needs); the round reads a single-gene fitness scalar off it. |
| Encoder | **Tx1-3B**, frozen/offline (`.obsm`); lead. **HVG-ST** attribution control; SE deferred | The encoder is not in ST's graph — "using Tx1" replaces the per-cell input vector. Tx1's a-priori edge for held-out **cancer** lines is its Tahoe-100M pretraining on the ~50 cancer lines (SE saw only general atlases). Only 3B is the full model (70M/1B saw less data, no drug token), so 3B is the fair test; HVG-ST isolates whether a win is the encoder vs. the rebuilt head + multi-line training. HVG-ST also carries the **encoder-unseen** reference, since the test lines are inside Tx1 pretraining but not inside any HVG pretraining (§7). |
| ST | Arc `StateTransitionPerturbationModel`, cell-set transformer, `output_space=gene`; warm-start where the input space matches (`ST-HVG-Replogle` for the HVG arm), retrain the input projection for Tx1 | Input space (`embed_key`) and output space are decoupled: Tx1 embeddings in, gene expression out. Encoder is frozen (no backprop into Tx1). No released ST takes Tx1 input, so the Tx1 arm cannot hot-start from an existing checkpoint's input layer. |
| Readout | **Rebuild**: drop K562-fit GMM; permutation-invariant moment pool; head conditioned on the **basal embedding** (inductive, **no one-hot line ID**); **rank/correlation loss + variance-matching** | A one-hot line covariate has no slot for an unseen line — structurally cannot generalize; context must be inductive (the basal embedding the model can compute for any line). Pure MSE is gamed by predicting the line mean (exactly the observed collapse), so the objective must be rank/correlation with variance-matching. |
| Perturbation | One-hot gene identity (ST-native `pert_rep=onehot`); no drug tokens | Genetic-only task; the Tx1 drug token is inert for CRISPR cells. |
| Composition / MoE | **None** | A drug-vs-gene router is deterministic (not real MoE), drug→genetic transfer is unproven (demonstrated transfer runs genetic→chemical), and a modality split starves the sparse genetic pathway. Excluded. |

### 2.3 How the model knows cell-line context

There is **no cell-line label or one-hot line ID** (a one-hot has no slot for an
unseen line and cannot generalize). The line is identified by its **basal
(unperturbed / DMSO) cells' Tx1 embeddings** — the baseline transcriptome is the
line's signature, and Tx1 baseline embeddings are already shown to carry
cross-line dependency signal (DepMap Fig 2C). Context enters through two channels:

1. **Into ST** — the basal input to the cell-set transformer is the target line's
   control cells, so the predicted perturbation response is line-specific.
2. **Into the head** — a pooled summary (mean/moments) of the same basal
   embeddings is concatenated into the head input. This is the more robust channel:
   ST is trained only on the four Perturb-seq lines, so the head's direct basal
   conditioning does not depend on ST generalizing its *response* to unseen lines.

What forces the model to use context rather than predict a line-agnostic value:
the differentially-essential slice is, by construction, genes whose GeneEffect
differs across lines, and the rank/correlation objective cannot be won by
predicting the line mean. Whether transcriptome-only context suffices is exactly
what the kill test judges.

**Scope:** context is transcriptome-only this round. Augmenting the head with a
per-line **genomic-context vector** (DepMap hotspot/damaging mutations,
copy-number, bulk expression) is the **first registered improvement** if the
transcriptome-only differentially-essential result is weak; DepMap OmicsExpression
is acquired for the baselines (§5) and would also feed this improvement.

## 3. Data plan

### 3.1 Cell-line roles (fixed split, no cross-validation)

The two input pools with DepMap GeneEffect labels are (a) lines carrying genetic
Perturb-seq (rich: observed response available) and (b) Tahoe DMSO basal-only
lines (basal single-cell + DepMap only). The test set is drawn from pool (b): the
lines that have **no** genetic perturbation data of their own. The split is
**fixed** — most of pool (b) trains the head, a lineage-stratified few are held
out — with **no rotation/cross-validation**.

| Line(s) | Source (local) | DepMap | Role |
| --- | --- | --- | --- |
| **K562** | Replogle GWPS CRISPRi (66 GB) | yes (`ACH-000551`) | **train** — observed-response ST + head anchor |
| **HCT116** | X-Atlas/Orion CRISPRi (44 GB) | yes (`ACH-000971`) | **train** — observed-response ST + head anchor |
| **Jurkat** | Nadig 2025 CRISPRi (262,956 cells, 2,394 targets) | yes (`ACH-000995`) | **train** — observed-response ST + head anchor |
| **HepG2** | Nadig 2025 CRISPRi | yes (`ACH-000739`) | **train** — observed-response ST + head anchor |
| ~28–30 Tahoe DMSO lines | Tahoe-100M DMSO single-cell (basal only) | yes | **train (head)** — GeneEffect via *predicted* response (ST forward from basal) |
| **~8–10 Tahoe DMSO lines** | Tahoe-100M DMSO single-cell (basal only) | yes | **test** — held out, never in training; scored k=0 + k-shot |

All four Perturb-seq training lines are **CRISPRi** (verified: same
non-targeting/dual-guide design across Replogle, Nadig, X-Atlas), so the response
model carries no hidden CRISPRi/CRISPRko mix. The knockdown→KO-GeneEffect modality
gap is the accepted program premise (`01` §2), unchanged.

**Audited eligible pool (to freeze in Phase A):** the Tahoe DMSO subset holds
**49** single-cell lines; **38** carry DepMap GeneEffect; none of the four
Perturb-seq training lines fall in that 38, so the split is clean. Lineage spread
of the 38: Lung 12, Bowel 7, Pancreas 6, CNS/Brain 3, Skin 3, then smaller
(Esophagus/Stomach, Bladder, Breast, Cervix, Liver, PNS). The ~8–10 test lines are
selected by a **frozen rule** — stratified across lineages for diversity, fixed
seed — recorded in the manifest before any run. Contract floor satisfied: ≥2
labeled training lines and ≥2 held-out test lines.

### 3.2 Tahoe as basal context, not drug transfer

Tahoe-100M enters **only** as single-cell DMSO control cells supplying basal states
for the DepMap-labeled cancer lines. No drug perturbation, no compound token, no
MoE. This is the one use the evidence supports (cross-line context breadth),
distinct from the drug-perturbation data deliberately excluded in §2.2.

### 3.3 Observed vs. predicted response (train/inference consistency)

- **ST** is supervised on *observed* KO responses from the four Perturb-seq lines
  (the response-prediction objective). ST never trains on Tahoe basal; it forwards
  on it.
- **The head** trains on *predicted* response uniformly — ST forward from basal for
  every line, including the four anchors — so head training and test both use ST
  out-of-distribution predictions on basal-only lines, keeping the
  observed-before-predicted discipline: forward-model error is isolated to ST.

### 3.4 Basal-source batch confound (top data risk)

Training-line basal comes from Perturb-seq non-targeting controls; test-line and
Tahoe-training-line basal comes from Tahoe DMSO. If those sources carry a
systematic assay/batch signature, the head can separate lines by **batch** rather
than biology and still appear line-specific. Phase A audits this: cluster basal
embeddings and quantify line-vs-batch separation; the gate is reported with and
without a batch/lineage covariate.

## 4. Few-shot adaptation

Context is inductive, so k=0 (zero-shot) is well-defined: a held-out line is just
its basal Tx1 embeddings. The k labels adapt via a **light per-line calibration** —
an affine/low-rank correction on the readout, or a few gradient steps on the head
only — fit on k labeled genes from the held-out line and scored on the remaining,
disjoint genes. No full ST fine-tune (overfits from few labels; would make the
curve reflect fine-tuning dynamics, not representation quality). Every baseline
gets its natural k-shot correction, so machinery must beat "copy-K562 + k labels,"
not just copy-K562. Registered k-schedule: **{0, 5, 10, 25, 50}**.

## 5. Evaluation and kill test

- **Testbed:** DepMap GeneEffect on the ~8–10 held-out Tahoe basal lines. Not
  Feng2024 (no cell-type axis).
- **Primary metric:** paired rank-correlation on the **differentially-essential
  slice** (slice membership — high cross-line GeneEffect variance,
  non-common-essential — fixed on **training lines only**), per held-out line, over
  **copy-K562 + k labels**.
- **Single gate (kill test):** at the registered **k = 10**, the macro-averaged
  paired difference `Tx1-3B-ST − (copy-K562 + 10 labels)` on the
  differentially-essential slice has a **line-level bootstrap 95% CI (cell lines as
  the inferential units) excluding zero** in the SL direction, clearing the
  registered `rho_min`. Failing that pauses the direction. `rho_min`, the
  k-schedule, the slice, and the estimator are registered before the formal run and
  are not revisable to fit a result.
- **Baselines:** copy-K562(+k), cross-line mean, nearest-line transfer,
  lineage-only, **CCLE-bulk→GeneEffect regression** (DepMap OmicsExpression), and
  **pseudobulk-basal→GeneEffect regression** (mean of the line's DMSO/control cells
  → GeneEffect, same basal input modality as the method under the same fixed split
  — the decisive "is the virtual cell just a regression?" control).
- **Reported alongside (not the gate):** per-line CIs and the fraction of held-out
  lines with a positive point estimate (descriptive support, no single-line veto);
  the **k=0 zero-shot** result as a stress point; the full few-shot curve (accuracy
  vs. k); the HVG-ST vs. Tx1-3B-ST attribution; a variance-preservation check
  (predicted-output std vs. target std) confirming the rebuild de-collapsed the
  head; aggregate correlation over the full gene set (inflated by pan-essential
  genes; not the gate).

## 6. Execution plan

**Phase A — data audit, manifest freeze, and acquisitions (Phase 0).** Acquire the
two infrastructure items (both complete the 26q1 release; no biological data
invented): **TahoeX1-3B encoder weights** and **DepMap OmicsExpression** (CCLE
bulk). Confirm DepMap GeneEffect coverage for K562, HCT116, Jurkat, HepG2 and the
38-line Tahoe DMSO∩DepMap pool against `CRISPRGeneEffect.csv`; apply the
lineage-stratified ~8–10 hold-out with a fixed seed and **exclude the held-out
lines from all training**. Run the basal-source batch-confound audit (§3.4).
Materialize an immutable train/test manifest with hashes and per-line Tx1
pretraining-exposure status (all test lines = **known present**). Define the
differentially-essential slice on training lines only; register `rho_min`, the
k-schedule, and the estimator. **Exit:** frozen manifest and registrations.

**Phase B — Tx1-3B embeddings (offline).** Build AnnData (`.X` raw counts,
`var["ensembl_id"]`, `obs["cell_type"]`) for all basal cells; run Tx1-3B inference
into `.obsm`. **Verify `.obsm[...].shape[1]` against the known 3B width** to resolve
the `tx-70m-merged`/`tahoe_x1_3b` labeling bug before trusting any run; do not
assume L2-normalization (add it if ST expects unit-norm inputs). Cache HVG matrices
for the control arm. **Exit:** verified embedding caches.

**Phase C — ST response model.** Reactivate the currently-dead `.obsm`
embedding-input path in `prepare.py::_state_input_view` (it is bypassed for the
`state_checkpoint` backend), and update the response-encoder input dim off the
hardcoded 2000; the `_effective_state_pert_dim` assertion (a safety net) stays. The
HVG arm warm-starts from released `ST-HVG-Replogle` (matching input space). The Tx1
arm retrains ST for the 2560-dim Tx1 input space — a fresh input projection,
optionally warm-starting the transformer body — since no released Tx1-input ST
checkpoint exists. Train on K562+HCT116+Jurkat+HepG2 observed KO responses,
`output_space=gene`. Note the HVG-space delta/energy losses become embedding-space
quantities for the Tx1 arm — keep, rename, or drop deliberately. **Exit:** ST
checkpoints (Tx1 and HVG arms).

**Phase D — rebuilt hybrid head.** Implement moment pooling + inductive
basal-conditioned head with the rank/correlation + variance-matching objective;
train on GeneEffect across the four Perturb-seq lines and the ~28–30 Tahoe-training
lines using predicted response uniformly. Run the HVG-ST control with the identical
head and data. **Exit:** trained heads for both encoder arms.

**Phase E — few-shot calibration.** Implement the per-line affine/low-rank (or
head-only) calibration and the k-schedule. **Exit:** k-shot adapter.

**Phase F — evaluation.** Score Tx1-3B-ST and HVG-ST on the ~8–10 held-out lines on
the differentially-essential slice against all baselines; the k=10 population gate,
per-line CIs, k=0 stress point, few-shot curve, and variance-preservation check.
Write `../results/<slug>.md` only after it runs. **Exit:** a Phase-3
backbone-transfer verdict (positive or negative), status synced across the vault.

## 7. Claim boundaries and risks

- **Backbone transfer, not SL.** A single-gene GeneEffect result — however strong —
  is not synthetic lethality and not cross-cell-line SL (`02` §4; roadmap §5).
- **Task-data-held-out, encoder pretrained on the target lines.** Every held-out
  test line is a Tahoe-100M line, so the lead Tx1-3B encoder saw its basal cells in
  pretraining (**known present** exposure, `02` §2.4/§8). The claim is qualified
  accordingly and stated per line. The **HVG-ST arm carries the encoder-unseen
  comparison** (no pretraining touched any line), so a matched HVG result is the
  cleaner "line unseen to the encoder" reference. Neither line's DepMap labels enter
  training (only k disjoint labels at inference under k-shot).
- **Basal-source batch confound (§3.4) is the top data risk** — audited in Phase A,
  reported with and without a batch/lineage covariate.
- **The transcriptome→viability bridge is the accepted program premise** (`01` §2);
  generalizing it across lines is what the rebuilt head bets on and what the kill
  test judges.
- **Held-out lines need single-cell basal** to compute a Tx1 basal embedding — the
  ST path can only reach lines with baseline single-cell data; this bounds the
  reachable generalization set (the 38-line Tahoe DMSO∩DepMap pool this round).
- **Four labeled observed-response lines** anchor ST; the head's cross-line variance
  comes from the ~28–30 Tahoe training lines. If they contribute too little real
  signal, the single-gate kill test will show it.
- **No planned number is a result;** results enter `docs/results/` only after the
  analysis runs.
