# exp08b — Two-Step State-Adapter Response Generator for SL-Pair Ranking

**Status:** Design approved 2026-06-29. Not yet implemented.
**Provenance:** exp06 (dependency-only floor) → exp07 (real mean-pooled gwps
embedding, CV2 win) → exp08 (frozen-STATE + ESM2 adapter, end-to-end, **failed**)
→ **exp08b** (this design: decouple the generator from the SL head).
**Predecessor failure analysis:** `docs/experiment/08_k562_sl_pair_perturbseq_state_dl.md`.
**Numbering note:** `exp09` (cross-cell-line selectivity) and `exp10` (DDGCN
reproduction) are already taken; this experiment is `exp08b` because it is a
direct re-architecture of exp08, not a new task.

## 1. Motivation and the claim under test

### 1.1 What exp08 established (the failure to fix)

exp08 wired a single end-to-end model: `ESM2(gene) → PertAdapter → frozen STATE
→ predicted response bag → MeanStdPool → e_g → SymmetricPairHead → SL logit`,
trained with a 3-part loss (SL BCE + adapter token-distill + real-bag
supervision) in one shared backward pass. Two structural failures, both visible
in the artifacts under `results/experiments/08_k562_sl_pair_state_dl/`:

1. **Loss-dominance collapse (Phase 3).** With `lambda_bag=1.0` and
   `lambda_sl=1.0`, the bag-reconstruction loss (~11.7) is ~10–20× the SL BCE
   (~0.5–1.0). Sharing one backward pass, the optimizer spent the small
   trainable capacity matching bags and let the SL head decay to a constant —
   `val_pair_auroc` collapsed to exactly 0.5 within 4–8 epochs across nearly
   every CV2/CV3 fold.
2. **The measured transcriptome was never read at inference.** In
   `src/sl_dl_model/train.py:produce`, *every* gene — covered or held-out — is
   embedded as `pool(STATE(adapter(ESM2(gene)), control))`. The real gwps bag
   appears only as a training *target* inside `bag_loss`; it never enters
   `embed_gene`. Combined with (1), the generator never learned to reconstruct
   bags, so the transcriptome channel was dead two ways: never learned, never
   read.

Net result: exp08 best-fold CV2 NDCG@10 = 0.0050, an order of magnitude below
exp07's 0.094 and barely above exp06's 0.042; CV2 AUROC 0.667 < exp06 0.704. The
sophisticated DL path underperformed XGBoost on a real mean-pooled bag by ~19× on
CV2 ranking (0.094 / 0.0050).

### 1.2 The exp07 ceiling

exp07 fed the **real** mean-pooled gwps embedding `e_g^pca` (128-d
`pca_delta_meanpool`) directly to XGBoost and beat the dependency-only floor on
CV2: NDCG@10 0.042 → **0.094**, AUROC 0.704 → 0.751. This proves the *target*
embedding carries SL signal and that **mean pooling is sufficient** to preserve
it. The bottleneck in exp08 is therefore not the pooling layer; it is the
quality of the *predicted* embedding for genes the model never saw perturbed.

**Notation (two distinct objects — do not conflate):**
- `e_g^pca` — exp07's 128-d `pca_delta_meanpool` of the real bag. Used **only**
  as the metric-level *ceiling* (its NDCG numbers); it is never a step-1 target.
- `e_g^real` — the **STATE-output-space** real gwps bag
  (`data/exp08_cache/k562_gwps_bags.npz`), and its `MeanStdPool` (2·D) where a
  pooled vector is needed. This is the step-1 `bag_loss` target and the object
  all step-1 monitor metrics (§4.3) compare against.
- `ê_g = MeanStdPool(pred_bag_g)` — the generator's predicted pooled embedding
  (2·D, same space as `MeanStdPool(e_g^real)`), cached for the step-2 SL head.

### 1.3 Claim under test

> A frozen-STATE backbone fed by an ESM2→adapter can predict a **useful**
> perturbation-response embedding for genes it never saw perturbed, **if** the
> adapter is trained to land its tokens inside STATE's true token distribution
> (the out-of-distribution-token fix), and **if** the generator objective is
> decoupled from the SL objective so neither steamrolls the other.

**Falsifiable prediction.** For held-out *covered* genes (genes with a real
gwps bag that are withheld from the fold's generator supervision),
`cosine(ê_g, MeanStdPool(e_g^real))` rises meaningfully above an
ESM2-nearest-neighbor real-bag-copy baseline; and the step-2 SL head on those
predicted embeddings
beats the exp06 floor on CV2 (NDCG@10 > 0.042). exp07's 0.094 is the aspirational
ceiling; exp06's 0.042 is the floor.

**Why this cannot silently die like exp08.** A fold-local held-out-cosine gate
(§4) exposes a dead generator *before* any step-2 SL head is trained, and a
direct-ESM2→MLP control (§5) isolates whether the frozen STATE forward
contributes signal or merely launders ESM2.

## 2. Architecture: two decoupled steps

```
STEP 1 — Response generator   (trainable: PertAdapter only; STATE frozen)
  ESM2(gene) → PertAdapter → raw pert token
                  │
                  ├─ distill loss:  encode(token) ≈ encode(onehot_g)
                  │                 [genes in STATE pert-vocab ∩ fold-train, ≤1,542]
                  │
                  └─ STATE(token, control) → predicted cell bag  (STATE output space, n_cells × D)
                                  │
                                  ├─ scale-normalized bag loss: pred_bag ≈ real_bag
                                  │   [fold-train-covered genes; target = real gwps
                                  │    cell bag in STATE output space, NOT exp07 PCA]
                                  │
                                  └─ MeanStdPool(pred_bag) → ê_g  (2·D, the cached embedding)
  → freeze generator; cache ê_g = MeanStdPool(pred_bag) for all 9,471 universe genes

STEP 2 — SL head              (trainable: SymmetricPairHead only; generator frozen)
  (ê_a, ê_b, GeneEffect 5-block) → SymmetricPairHead → SL logit → BCE
  → official per-anchor ranking metric (sl_benchmark_baseline, verbatim)
```

The decisive change from exp08: **the bag objective and the SL objective never
share a backward pass.** Step 1 optimizes only the generator (adapter); step 2
optimizes only the pair head against a frozen, cached embedding table. The
Phase-3 loss-dominance collapse is structurally impossible.

This also cleanly separates the two scientific questions:

- **Step 1 answers:** can the generator produce an embedding close to the real
  one for held-out covered genes?
- **Step 2 answers:** is a predicted embedding good enough to support SL ranking?

A null result is attributable to one step rather than confounded across both.

## 3. Step-1 objective (the crux)

```
L_step1 = lambda_distill · MSE( encode(adapter(esm_g)), encode(onehot_g) )
        + lambda_bag      · BagLoss_normalized( pred_bag_g, real_bag_g )
```

where `BagLoss` reuses exp08's `sl_dl_model.losses.bag_loss` (mean-delta MSE +
NaN-safe energy distance) on **cell bags** `(n_cells, D)` in STATE's output
space. The real target is the gwps cell bag from
`data/exp08_cache/k562_gwps_bags.npz`. The cached step-2 embedding is
`ê_g = MeanStdPool(pred_bag_g)` (2·D); exp07's 128-d `pca_delta_meanpool`
`e_g^pca` lives in a different (PCA) space and is **not** a step-1 target — it
is used only as a metric-level ceiling (§5.2). Pooling the predicted bag to a
PCA-space target would require a trainable projection head, which would confound
the "can the adapter coerce STATE tokens?" question and is therefore excluded.

### 3.1 Scale normalization (default: fold-local fixed warmup scale)

The exp08 collision (~11.7 vs ~0.5) is fixed by normalizing the bag term to
O(1) so it cannot steamroll the distill anchor:

- During a **warmup window** (first `warmup_epochs`, default 1), accumulate the
  **detached** per-step bag-loss values on the fold's train-covered genes.
- Set `scale = median(detached_bag_losses)` over that window (median chosen over
  mean for robustness to early outliers), `clamp(scale, min=1e-3)`.
- For all subsequent epochs, use `BagLoss_normalized = bag_loss / scale`.
- Write `scale` per fold to the run manifest for auditability.

**Rejected alternatives:** EMA-normalized bag term (target drifts across
training, weaker interpretability — demoted to an ablation, §6); z-score
normalization (can produce negative loss and adds instability — dropped).

### 3.2 Distill anchor weight

`lambda_distill` stays at **full weight throughout** training (not decayed),
since the distill anchor is the only mechanism tying held-out-gene tokens onto
STATE's trained token manifold — the OOD-token fix. Its sufficiency is tested
directly by the §6 ablation grid, not assumed.

## 4. Fold-local training and the held-out-cosine gate

### 4.1 Fold-local step-1 (mandatory leakage fix)

Each CV fold trains its **own** step-1 generator on **only that fold's
train-covered genes' real bags.** A single global generator is forbidden:
because CV2 holds out one gene and CV3 holds out both, a global generator would
let a CV2/CV3 test gene see its true bag during step-1 bag supervision, silently
breaking the cold-start claim.

- Step-1 bag supervision set = `fold_train_genes ∩ gwps_covered`.
- The CV2/CV3 **test** genes are reached purely by the frozen fold-local
  generator at step-2 scoring time. No test-gene bag ever touches step 1.
- Cost: one generator per fold (CV1/CV2/CV3 × 5 = 15 generators). Acceptable;
  reuses exp08's fold-queue orchestration under the contract in §4.4.

### 4.2 Generator-validation split (held-out covered genes)

The cosine gate's held-out set is carved from **train-covered** genes, never
from CV test genes:

- Hold out 20% of `fold_train_genes ∩ gwps_covered` as a **generator-validation**
  set, withheld from step-1 **bag** supervision. (The distill set is independent
  and unaffected — distill uses STATE pert-vocab ∩ fold-train.)
- These genes have a real `e_g^real` (they are covered) and are not bag-supervised,
  so they are a clean in-fold proxy for held-out-gene generalization.

### 4.3 Step-1 monitor metrics (per epoch, on generator-validation set)

Logged as numbers alongside the loss curves — a **monitor, not a hard stop**
(the run proceeds to step 2 regardless; the numbers tell us how to read step 2).
All comparisons use the **STATE-space** real bag of each held-out covered gene
(the same `data/exp08_cache/k562_gwps_bags.npz` object the loss targets), pooled
the same way — **not** exp07's PCA `e_g^pca`. Two metric families at two stages:

1. **Pooled-vector metrics** between `ê_g = MeanStdPool(pred_bag)` and
   `MeanStdPool(real_bag)` (both 2·D, same space): `cosine` (direction) and
   `MSE` / `L2` (magnitude / dispersion that cosine discards — the SL head may
   depend on magnitude). *Energy distance is NOT used here — it is a
   bag/distribution metric, undefined on a single pooled vector.*
2. **Bag-level metric** between `pred_bag` and `real_bag` *before* pooling: the
   `bag_loss` energy-distance term (`sl_dl_model.losses` `_safe_energy_distance`),
   the distribution-level analogue of (1).
3. **ESM2-nearest-neighbor copy baseline** for both families: for a held-out
   covered gene, take the *train-covered* gene nearest in ESM2 space and use its
   real bag (and its `MeanStdPool`) as the prediction, then report the same
   pooled cosine/MSE/L2 and bag energy. If the trained generator cannot beat
   nearest-neighbor copying, the STATE machinery adds nothing — learned in step
   1, cheaply.

### 4.4 Orchestration contract (named constraint, not just "reuse exp08")

exp08's first multi-rank run died with an NCCL `gather_object` timeout; the fix
was to drop all collectives in favor of a filesystem work-queue. exp08b inherits
that contract **as a hard design constraint**, both steps:

- `accelerate launch` starts the ranks; code uses `PartialState` only for
  device/rank/`is_main_process` — never `Accelerator()`.
- Models are **not** wrapped with `Accelerator.prepare` or `DDP`. There is no
  gradient all-reduce; each rank owns whole folds and trains them locally.
- Each rank claims `(step, split_type, fold_id)` jobs from the filesystem queue
  (atomic `os.mkdir` claim under a per-run token; `.result.json` is the sole
  cross-run resume state), exactly as in `src/sl_dl_model/fold_queue.py`.
- **No collective calls anywhere** in the package — enforced by reusing exp08's
  `tests/test_no_collectives.py` AST guard. Rank-0 assembles results by polling
  the queue, not by `gather`.
- Step 1 and step 2 are **separate queue passes**: step 2 jobs do not start until
  the step-1 generator + cached `ê_g` table for that fold exist on disk.

## 5. Evaluation, baselines, and success criteria

### 5.1 Step-2 metric (primary)

Reuses `sl_benchmark_baseline.metrics.official_*` **verbatim**: same CV1/CV2/CV3
Rand 1:1 splits, same seen-pair masking, same diagonal zeroing, same per-anchor
candidate-partner ranking over the 9,471-gene universe, same covered-pair
diagnostic slice.

- **Primary success:** CV2/CV3 per-anchor NDCG@{10,20,50} and MAP@{10,20,50}
  beat the in-harness exp06 floor. CV1 is ignored (degree-gameable; exp06's
  degree probe wins it).

### 5.2 Baseline ladder (all reported side-by-side)

| Rung | Predictor for `e_g` | Role |
| --- | --- | --- |
| exp06 | none (GeneEffect 5-block only) | floor |
| NN-copy | `e_g^real` of ESM2-nearest train-covered gene | trivial-generalization control |
| direct-ESM2-MLP | `MLP(ESM2(gene))` → `ê_g`, **STATE bypassed entirely** | "does STATE add anything?" control |
| **exp08b** | frozen STATE + ESM2 adapter, two-step (this design) | experiment |
| exp07 | `e_g^pca` (real mean-pooled gwps bag) | ceiling |

The **direct-ESM2-MLP** rung is the critical control: if it matches exp08b, the
frozen STATE forward is laundering ESM2 rather than contributing perturbation
structure, and the STATE machinery is unjustified. To make that comparison fair,
it is specified to match exp08b on everything **except** the STATE forward:

- **Trained fold-locally** on the same step-1 split as exp08b — supervised on
  `fold_train_genes ∩ gwps_covered`, with the same 20% generator-validation
  held-out set (§4.2), so neither baseline sees a CV test gene's bag.
- **Same target space and loss — MLP emits a cell bag, not a pooled vector.**
  To keep `bag_loss` (and the §4.3 bag-level energy metric) well-defined and
  directly comparable, the MLP produces a `(T, D)` STATE-output-space bag over
  the **same control template** exp08b feeds STATE (`T = control_template_size`,
  default 256): `pred_bag = control_template + MLP(ESM2(gene))`, where
  `MLP(ESM2(gene))` is a single `D`-dim delta broadcast-added to all `T` control
  rows. Trained against the same `data/exp08_cache/k562_gwps_bags.npz` real bag
  under the same `bag_loss` + scale normalization; `ê_g = MeanStdPool(pred_bag)`
  is cached identically. The only architectural difference from exp08b is
  `control_template + MLP(ESM2)` replacing `adapter → frozen STATE forward` —
  so the control measures precisely whether the STATE forward adds response
  structure beyond a learned per-gene mean shift on the same control cells.
  *(A pooled-vector MLP head is explicitly rejected: it cannot use `bag_loss`
  or the bag-level energy metric, breaking target-space parity with exp08b.)*
- **No `sl_label`** enters this baseline's training — it is a step-1 generator
  exactly like exp08b, evaluated by the §4.3 monitor and then by a step-2 SL head
  trained on its cached `ê_g`. This prevents it from becoming accidentally
  SL-supervised (too strong) or under-trained (too weak).

### 5.3 Scientific readout (publishable in all four outcomes)

The step-1 cosine gate × step-2 SL lift cross-tabulate to a clear diagnosis:

- **cosine high + SL lift** → the claim holds: predicted transcriptome generalizes.
- **cosine high + no SL lift** → generator works, but the (predicted or even real)
  embedding is insufficient for SL ranking — points back at the task, not the model.
- **cosine low** → the OOD-token fix failed; STATE cannot be coerced onto its
  manifold via an ESM2 adapter. Report and stop the line.
- **exp08b vs exp07 gap** → quantifies the predicted-vs-real penalty regardless
  of the above.

## 6. Ablations (mandatory for attribution)

Run the step-1 objective grid so a null result is attributable rather than
confounded between distill-anchor / bag-objective / frozen-STATE-path:

1. `bag only` (`lambda_distill = 0`)
2. `distill only` (`lambda_bag = 0`)
3. `distill + bag` (default, `lambda_distill` full + normalized bag)
4. **Normalization ablation:** default fixed-warmup-scale vs EMA-normalized bag.

Each ablation is judged first by the §4.3 step-1 monitor, then (for promising
ones) by the §5.1 step-2 metric.

## 7. Scope, reuse, and non-goals

### 7.1 Reuse (do not rebuild)

- ESM2 embeddings (`data/esm2/k562_sl_universe_esm2_650M.npz`), gwps bags
  (`data/exp08_cache/k562_gwps_bags.npz`) — the latter is the **step-1 bag-loss
  target** (real cell bags in STATE output space), and the STATE checkpoint —
  all reused verbatim.
- exp07's real mean-pooled `e_g^pca`
  (`results/experiments/07_k562_sl_pair_perturbseq_augmented/gwps_pca_mean_bags/bags.npz`,
  128-d `pca_delta_meanpool`) is **not** a step-1 target — it lives in a
  different (PCA) space. It is reused only as the **metric-level ceiling** (its
  CV2/CV3 NDCG/MAP numbers) in the §5.2 baseline ladder.
- `sl_dl_model` components: `PertAdapter`, `StateEncoder`, `MeanStdPool`,
  `SymmetricPairHead`, `bag_loss`, fold-queue orchestration, official metric.
- exp08b forks `src/sl_dl_model/` with **two new entrypoints** whose code
  ownership is fixed here to prevent the end-to-end re-coupling that sank exp08
  Phase 3 (the package split vs new sibling package is settled in the plan, but
  the boundary below is binding):

  **Step 1 — `train-generator`** (owns the generator + its artifacts; knows
  nothing about SL labels or the pair head):
  - Trains the fold-local generator (`PertAdapter` + frozen `StateEncoder` +
    `MeanStdPool`) under the §3 step-1 objective only (distill + normalized bag).
  - Reads: fold-train-covered genes' real bags, ESM2 table, STATE checkpoint.
  - Writes per fold: frozen generator weights, the cached `ê_g` table for **all
    9,471** universe genes, the chosen bag-loss `scale` (in the manifest), and
    the §4.3 step-1 monitor CSV. **No `sl_label` is read in this entrypoint.**

  **Step 2 — `train-sl-head`** (owns the pair head; treats `ê_g` as a frozen,
  opaque input — never instantiates or calls the generator/STATE):
  - Loads the step-1 cached `ê_g` table for the fold and trains only
    `SymmetricPairHead` on `(ê_a, ê_b, GeneEffect-5)` under SL BCE.
  - Reads: the step-1 `ê_g` table + the SL benchmark pairs. **It must not import
    `StateEncoder`/`PertAdapter` or hold a STATE checkpoint** — this import-level
    separation is the structural guarantee that the bag and SL objectives can
    never again share a backward pass.

  The two steps communicate **only** through the on-disk `ê_g` table + manifest,
  never through a shared in-memory model object.

### 7.2 Non-goals

- **No GMM / learned pooling.** exp07 proved mean pooling preserves SL signal;
  pooling is not the bottleneck. Explicitly out of scope.
- **No unfreezing STATE.** The experiment is specifically "can the adapter coerce
  OOD tokens onto STATE's frozen manifold," which is only meaningful with a
  frozen backbone. Fine-tuning the 8-layer Llama on ~6,070 covered genes would
  be a near-from-scratch generator and a different experiment.
- **No new SL benchmark / no biological SL claims.** `Rand` negatives remain
  unconfirmed non-SL; this is a benchmark-adapter extension of exp06, same
  terminology guardrails.

## 8. Terminology guardrail

exp08b is a benchmark-adapter extension of exp06/07/08. No "SL target" or
biological SL claims. Predicted embeddings are *predicted perturbation-response
embeddings*, not measured single-cell death labels. CRISPRi/knockout modality
caveats from the parent experiments carry over unchanged.
