# Design: Tx1-conditioned ST GeneEffect backbone (few-shot, cross-cell-line)

**Status:** design approved 2026-07-23; not yet built. Concrete architecture for
roadmap **T2 / Phase 3** (the few-shot cross-cell-line GeneEffect backbone). No
listed step has run; no number here is a result.
**Authority:** [`../01-blueprint.md`](../01-blueprint.md) (frozen contract) and
[`../02-acceptance-criteria.md`](../02-acceptance-criteria.md) (frozen bar) govern;
[`../04-roadmap.md`](../04-roadmap.md) §1.1/§5 is the phase plan this refines; it
supersedes the abstract T2 sketch in
[`2026-07-22-k562-mechanism-and-geneeffect-generalization-plan.md`](2026-07-22-k562-mechanism-and-geneeffect-generalization-plan.md)
§3 with a concrete model.

## 1. Objective and scope

Build `F(X_c, c, g) -> GeneEffect`: a context-conditioned single-gene predictor
that generalizes leave-one-cell-line-out and adapts to a held-out line from a few
of its own labels (k-shot), accurate against real DepMap GeneEffect on the
differentially-essential slice. This is the **Phase 3 single-gene
backbone-transfer exit**, reported as backbone transfer, **not** an SL result and
**not** a cross-cell-line SL claim (roadmap §5; `02` §4). It opens no pairwise
labels from any untouched cell line.

Motivation: the composition mechanism (Bridge A) is paused after a negative
kill-test, and the frozen K562 response-to-GeneEffect head did not transfer to
HCT116 — the measured failure (collapsed output variance, Spearman approximately
0) lives **downstream of the encoder**, in the K562-fit GMM pooler and MSE head,
which regress to the K562 mean off-distribution. This design fixes that failure
and tests whether a stronger cell-state encoder plus multi-line supervision yields
a cross-line-generalizable single-gene backbone.

## 2. Architecture

### 2.1 Inference path (test line `c`, gene `g`)

```text
control single cells of c
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
| Encoder | **Tx1-3B**, frozen/offline (`.obsm`); lead. **HVG-ST** attribution control; SE deferred | The encoder is not in ST's graph — "using Tx1" replaces the per-cell input vector. Tx1's a-priori edge for held-out **cancer** lines is its Tahoe-100M pretraining on ~50 cancer lines (SE saw only general atlases). Only 3B is the full model (70M/1B saw less data, no drug token), so 3B is the fair test; HVG-ST isolates whether a win is the encoder vs. the rebuilt head + multi-line training. |
| ST | Arc `StateTransitionPerturbationModel`, cell-set transformer, `output_space=gene`; warm-start where the input space matches (`ST-HVG-Replogle` for the HVG arm), retrain the input projection for Tx1 | Input space (`embed_key`) and output space are decoupled: Tx1 embeddings in, gene expression out. Encoder is frozen (no backprop into Tx1). No released ST takes Tx1 input, so the Tx1 arm cannot hot-start from an existing checkpoint's input layer. |
| Readout | **Rebuild**: drop K562-fit GMM; permutation-invariant moment pool; head conditioned on the **basal embedding** (inductive, **no one-hot line ID**); **rank/correlation loss + variance-matching** | A one-hot line covariate has no slot for an unseen line — structurally cannot generalize; context must be inductive (the basal embedding the model can compute for any line). Pure MSE is gamed by predicting the line mean (exactly the observed collapse), so the objective must be rank/correlation with variance-matching. |
| Perturbation | One-hot gene identity (ST-native `pert_rep=onehot`); no drug tokens | Genetic-only task; the Tx1 drug token is inert for CRISPR cells. |
| Composition / MoE | **None** | A drug-vs-gene router is deterministic (not real MoE), drug→genetic transfer is unproven (demonstrated transfer runs genetic→chemical), and a modality split starves the sparse genetic pathway. Excluded. |

### 2.3 How the model knows cell-line context

There is **no cell-line label or one-hot line ID** (a one-hot has no slot for an
unseen line and cannot generalize). The line is identified by its **control
(unperturbed) cells' Tx1 embeddings** — the baseline transcriptome is the line's
signature, and Tx1 baseline embeddings are already shown to carry cross-line
dependency signal (DepMap Fig 2C). Context enters through two channels:

1. **Into ST** — the basal input to the cell-set transformer is the target line's
   control cells, so the predicted perturbation response is line-specific.
2. **Into the head** — a pooled summary (mean/moments) of the same basal
   embeddings is concatenated into the head input. This is the more robust channel:
   ST is trained only on K562+HCT116, so the head's direct basal conditioning does
   not depend on ST generalizing its *response* to unseen lines.

What forces the model to use context rather than predict a line-agnostic value:
the differentially-essential slice is, by construction, genes whose GeneEffect
differs across lines, and the rank/correlation objective cannot be won by
predicting the line mean. Whether transcriptome-only context suffices is exactly
what the kill test judges.

**Scope:** context is transcriptome-only this round. Augmenting the head with a
per-line **genomic-context vector** (DepMap hotspot/damaging mutations,
copy-number, bulk expression — all locally available) is the **first registered
improvement** if the transcriptome-only differentially-essential result is weak; it
must beat the transcriptome-only model, and a held-out line would then also need
its DepMap genomics, not only control cells.

## 3. Data plan

### 3.1 Cell-line roles

Single-cell genetic Perturb-seq universe and roles:

| Line | Perturb-seq source | DepMap GeneEffect | Role |
| --- | --- | --- | --- |
| **K562** | Replogle GWPS + essential (local, large) | yes (`ACH-000551`) | **train** — observed-response ST supervision + head |
| **HCT116** | X-Atlas/Orion genome-wide CRISPRi (local, large) | yes | **train only** — observed-response ST + head (unsealed; ineligible as test, `02` §4) |
| ~50 Tahoe lines | Tahoe-100M **DMSO single-cell controls** (basal only) | yes (cancer lines) | **train (head)** — basal context breadth; no drug perturbations enter |
| RPE1 | Replogle GWPS (large) | no (non-cancer) | optional ST response-only pretraining; no head label |
| **Jurkat** | Replogle-Nadig / X-Atlas | yes | **test** — held out |
| **HepG2** | Replogle-Nadig / X-Atlas | yes | **test** — held out |

Contract floor satisfied: ≥2 labeled training lines (K562, HCT116, + ~50 Tahoe
basal-context lines for the head) and ≥2 untouched test lines (Jurkat, HepG2).

### 3.2 Tahoe as basal context, not drug transfer

Tahoe-100M enters **only** as single-cell DMSO control cells supplying basal
states for ~50 DepMap-labeled cancer lines. No drug perturbation, no compound
token, no MoE. This is the one use the evidence supports (cross-line context
breadth), distinct from the drug-perturbation data deliberately excluded in §2.2.

### 3.3 Observed vs. predicted response (train/inference consistency)

- **ST** is supervised on *observed* KO responses from K562 and HCT116 (the
  response-prediction objective). Observed data trains the response model.
- **The head** trains on *predicted* response uniformly — ST forward from basal
  for every line — so training matches inference on test lines where only basal
  exists. The head never sees observed response, keeping the
  observed-before-predicted discipline: forward-model error is isolated to ST.

## 4. Few-shot adaptation

Context is inductive, so k=0 (zero-shot) is well-defined: a held-out line is just
its basal Tx1 embeddings. The k labels adapt via a **light per-line calibration**
— an affine/low-rank correction on the readout, or a few gradient steps on the
head only — fit on k labeled genes from the held-out line and scored on the
remaining, disjoint genes. No full ST fine-tune (overfits from few labels; would
make the curve reflect fine-tuning dynamics, not representation quality).

## 5. Evaluation and kill test

- **Testbed:** DepMap GeneEffect on held-out lines Jurkat and HepG2; leave-one-line
  -out over labeled training lines for development readouts. Not Feng2024 (no
  cell-type axis).
- **Primary metric:** paired rank-correlation on the **differentially-essential
  slice** (slice membership — high cross-line variance, non-common-essential —
  fixed on **training lines only**), per held-out line, over **copy-K562 + k
  labels**; 95% CI via within-line gene bootstrap; per-line reported, then
  macro-averaged.
- **Baselines:** copy-K562 (+k), cross-line mean, nearest-line transfer,
  lineage-only, plain CCLE-bulk→GeneEffect regression; each extended with its
  natural k-shot correction so machinery must beat "copy-K562 + k labels."
- **Single gate (kill test):** Tx1-3B-ST must beat copy-K562 + k on the
  differentially-essential slice with a per-line 95% CI excluding zero at a
  registered k. Failing that pauses the direction. `rho_min`, the k-schedule, and
  the estimator are registered before the formal run and are not revisable to fit
  a result.
- **Reported alongside (not gates):** the full few-shot curve (accuracy vs. k) —
  the safety net against a degenerate "wins-only-at-large-k" result; the HVG-ST vs.
  Tx1-3B-ST attribution; a variance-preservation check (predicted-output std vs.
  target std) to confirm the rebuild de-collapsed the head; aggregate correlation
  over the full gene set (inflated by pan-essential genes; not the gate).

## 6. Execution plan

**Phase A — data audit and manifest freeze (Phase 0).** Confirm DepMap GeneEffect
coverage for K562, HCT116, Jurkat, HepG2 and the RPE1 gap against
`CRISPRGeneEffect.csv`; enumerate which of the ~50 Tahoe lines carry DepMap
GeneEffect and **exclude Jurkat/HepG2 from that set**. Acquire Jurkat/HepG2
Perturb-seq (Replogle-Nadig / X-Atlas) and the Tahoe-100M DMSO-control subset;
confirm local K562, HCT116. Materialize an immutable train/test manifest with
hashes and per-line Tx1 pretraining-exposure status (task-data-held-out). Define
the differentially-essential slice on training lines only; register `rho_min`, the
k-schedule, and the estimator. **Exit:** frozen manifest and registrations.

**Phase B — Tx1-3B embeddings (offline).** Build AnnData (`.X` raw counts,
`var["ensembl_id"]`, `obs["cell_type"]`) for all basal cells; run Tx1-3B inference
into `.obsm`. **Verify `.obsm[...].shape[1]` against the known 3B width** to
resolve the `tx-70m-merged`/`tahoe_x1_3b` labeling bug before trusting any run; do
not assume L2-normalization (add it if ST expects unit-norm inputs). Cache HVG
matrices for the control arm. **Exit:** verified embedding caches.

**Phase C — ST response model.** Reactivate the currently-dead `.obsm`
embedding-input path in `prepare.py::_state_input_view` (it is bypassed for the
`state_checkpoint` backend), and update the response-encoder input dim off the
hardcoded 2000; the `_effective_state_pert_dim` assertion (a safety net) stays.
The HVG arm warm-starts from released `ST-HVG-Replogle` (matching input space). The
Tx1 arm retrains ST for the 2560-dim Tx1 input space — a fresh input projection,
optionally warm-starting the transformer body — since no released Tx1-input ST
checkpoint exists. Train on K562+HCT116 observed KO responses, `output_space=gene`.
Note the HVG-space delta/energy losses become embedding-space quantities for the
Tx1 arm — keep, rename, or drop deliberately. **Exit:** ST checkpoints (Tx1 and HVG arms).

**Phase D — rebuilt hybrid head.** Implement moment pooling + inductive
basal-conditioned head with the rank/correlation + variance-matching objective;
train on GeneEffect across K562, HCT116, and the ~50 Tahoe-control lines using
predicted response uniformly. Run the HVG-ST control with the identical head and
data. **Exit:** trained heads for both encoder arms.

**Phase E — few-shot calibration.** Implement the per-line affine/low-rank (or
head-only) calibration and the k-schedule. **Exit:** k-shot adapter.

**Phase F — evaluation.** Score Tx1-3B-ST and HVG-ST on Jurkat and HepG2 (and
LOCO) on the differentially-essential slice against all baselines; per-line CIs,
macro-average, single-gate verdict, few-shot curve, variance-preservation check.
Write `../results/<slug>.md` only after it runs. **Exit:** a Phase-3
backbone-transfer verdict (positive or negative), status synced across the vault.

## 7. Claim boundaries and risks

- **Backbone transfer, not SL.** A single-gene GeneEffect result — however strong —
  is not synthetic lethality and not cross-cell-line SL (`02` §4; roadmap §5).
- **Task-data-held-out, not unseen-context.** Tx1 pretrained on the Tahoe cancer
  lines; the transfer claim is qualified accordingly and stated per line.
- **The transcriptome→viability bridge is the accepted program premise**
  (`01` §2); generalizing it across lines is what the rebuilt head bets on and what
  the kill test judges.
- **Held-out lines need single-cell basal** to compute a Tx1 basal embedding — the
  ST path can only reach "more cancer lines" that have baseline single-cell data;
  this bounds the reachable generalization set.
- **Two labeled observed-response lines (K562, HCT116)** anchor ST; the head's
  cross-line variance comes from the ~50 Tahoe basal-context lines. If the ~50
  contribute too little real signal, the single-gate kill test will show it.
- **No planned number is a result;** results enter `docs/results/` only after the
  analysis runs.
