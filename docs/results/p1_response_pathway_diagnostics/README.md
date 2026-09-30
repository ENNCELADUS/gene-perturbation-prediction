# Response-pathway diagnostics (P1): fixed-backbone heads, response adaptation, interface isolation

**Status:** closed in count space and in expression space (on the first comparison pass; the expression-space learning-rate arms did not complete), diagnostic, seed 0. No context claim, no SL claim, no test-split use.
The count-space interface of the seed-0 joint backbone is identified as defective;
no adapted variant transfers a perturbation response to a held-out cell line.

Designs: [fixed-backbone head diagnostic](../../specs/2026-09-07-p1a-fixed-backbone-head-diagnostics-design.md),
[response-adaptation diagnostic](../../specs/2026-09-07-p1b-response-adaptation-design.md),
[interface-isolation diagnostic](../../specs/2026-09-08-p1c-interface-isolation-design.md).
Protocol: [GeneEffect protocol §8](../../03-geneeffect-protocol.md#8-where-the-model-stalls-response-pathway-diagnostics).
Backbone under test: the selected seed-0 joint checkpoint
(`joint_seed0_20260906T174818Z_b1024/best.pt`, SHA-256 `37405454…63c59`,
[result](../joint_geneeffect_seed0/README.md)), called the seed-0 joint backbone (P0) below.

## What was tested

1. **Fixed-backbone head diagnostic (P1-A).** With the seed-0 joint backbone frozen, does an explicit gene-specific context slope
   (PCA8 of the Tx1 context features) improve validation residual readout over the
   shared MLP head, and does the backbone's response block (R) add anything? Four heads: shared MLP head (A0), an MLP on
   direct features (D); shared MLP head + response block (A1), an MLP on the direct features plus the response block;
   explicit context-slope head (A2), an MLP plus gene-specific slope on the direct features; explicit context-slope
   head + response block (A3), the explicit context-slope head plus the response block.
2. **Response-adaptation diagnostic (P1-B).** Does the seed-0 joint backbone's response pathway use perturbation identity, and does adapting
   only the new interface (basal encoder 328 × 2560, ESM2 adapter, residual head) at 1e-4
   recover response prediction on the three source anchors and transfer to Jurkat, which
   was excluded from every fit and selection step of this stage?
3. **Interface-isolation diagnostic (P1-C).** Existing-export analysis (Tier 0): how much of the adapted states' Jurkat error is a shared per-context
   bias. Native functional check (Tier 1): does the released STATE ST-HVG-Replogle checkpoint, driven through its own
   basal encoder and one-hot vocabulary with the cached HVG bags, predict responses at all,
   and in which numeric space. Learning-rate isolation (Tier 2): interface learning rate alone (1e-6, 1e-5 against
   1e-4) under the response-adaptation recipe. Interface-variant comparison (Tier 3): four interface variants under leave-one-anchor-out
   (LOAO) folds with a pre-registered keep predicate: adapted Tx1 basal encoder (V0; = the response-adaptation interface),
   expression-residual interface (V1; null-subtracted expression residual, exactly no-change at initialisation),
   native basal path (ESM-2 perturbation tokens; V2-null), the native basal path only, native basal path + trainable
   Tx1 context term (V2), with the Tx1 context zero-initialised, and native basal path + Tx1 term + expression residual
   (V3), conditional on a keep. Head-seed stability (Tier 4): the shared MLP head and the explicit context-slope head of
   the fixed-backbone head diagnostic at head seeds 1 and 2.

## Method and provenance

| Stage | Run | Code | Host |
| --- | --- | --- | --- |
| Fixed-backbone head diagnostic | `outputs/p1a/p1a_seed0_20260907T144045Z` | `afdd7fc` | 30838, 4 GPUs |
| Response-adaptation diagnostic | `outputs/p1b/p1b_seed0_20260907T164813Z` | `0723462` | 30838 |
| Interface-isolation diagnostic | `outputs/p1c/p1c_seed0_20260908T112354Z` | `4b48381` (count-space round), `ca6a71e` (native functional check in log-normalised space, Tier 1b) | 30030, 2 × H20 |

- Response supervision, references and metrics: mean-shift MSE plus energy distance on
  the 1,957 HVG coordinates observed on all four anchors; references are no-change (the
  control bag), the global train mean effect and the train-side perturbation-mean effect;
  identity advantage is wrong-identity minus correct-identity loss over ten fixed
  derangements; intervals are 1,000 paired gene resamples, seed 0, conditional on the
  selected checkpoints.
- Anchors: K562 `ACH-000551`, HepG2 `ACH-000739`, HCT116 `ACH-000971`, Jurkat
  `ACH-000995`. The response-adaptation diagnostic trains on the three source anchors' non-holdout conditions
  (27,361), validates on their condition holdouts (3,047) and scores all 2,377 Jurkat
  conditions externally. The interface-isolation diagnostic's leave-one-anchor-out folds hold out each anchor in turn from every fit and
  from the reference means; the Jurkat fold is a diagnostic re-evaluation because Jurkat
  had been observed in the response-adaptation diagnostic before the interface-isolation design was registered.
- Recipe for every trained arm: interface parameters only (2,533,864; 1,694,184 for
  the native basal path (ESM-2 perturbation tokens)), AdamW at the stated rate, balanced anchor sampling, cap 50 epochs / 12,850
  updates, patience 5 on validation response loss, seed 0.
- Keep predicate (interface-variant comparison): (a) internal-validation loss ratio to no-change below 1
  with a paired interval excluding 0 on at least two of three source anchors in every
  fold; (b) equal-fold mean of per-fold held-out ratios below 1 with a synchronous
  gene-bootstrap interval excluding 1; (c) identity advantage positive with interval
  excluding 0 on every held-out fold.
- The interface-isolation diagnostic ran as one pipeline (`hpc/p1c_pipeline.sh`), exit 0 on 2026-09-09; every arm
  exit 0; init checks passed on every variant fold (expression-residual interface max deviation from the control
  bag 0.0; both native-basal-path variants reproduce the native null forward to within 3 × 10⁻⁴ relative).
- Curated evidence under [`evidence/`](evidence/): fixed-backbone head diagnostic `comparison/` and
  `analysis/audit.json`; response-adaptation diagnostic `comparison/` tables; interface-isolation diagnostic `comparison/` (`variants.csv`,
  `summary.csv`, `kept.json`, `learning_curves.png`), `tier0/`, `heads/seed{1,2}/comparison/`,
  native summaries under `runs/`. Full exports remain on the host.

## Result

### Fixed-backbone head diagnostic and head-seed stability: explicit context slope

Validation residual Pearson (macro over 4,447 train-defined variable genes, 27
validation lines), and paired 27-line bootstrap deltas.

| Head seed | Shared MLP | Shared MLP + response | Context slope | Context slope + response | Context slope − shared MLP Pearson [95%] | (Shared MLP + response) − shared MLP | (Context slope + response) − context slope |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | 0.0495 | 0.0496 | 0.1315 | 0.1280 | +0.0820 [0.0577, 0.1047] | +0.0001 [−0.020, 0.016] | −0.0035 [−0.010, 0.003] |
| 1 | 0.0617 | 0.0539 | 0.1328 | 0.1294 | +0.0711 [0.0487, 0.0947] | −0.0079 [−0.018, 0.003] | −0.0035 [−0.008, 0.002] |
| 2 | 0.0542 | 0.0500 | 0.1344 | 0.1316 | +0.0802 [0.0551, 0.1038] | −0.0042 [−0.018, 0.008] | −0.0028 [−0.010, 0.003] |

The explicit context-slope head against the Tx1 PCA8-ridge reference (seed 0): Pearson −0.0015 [−0.0153, 0.0159];
against the seed-0 joint backbone itself +0.0911 [0.0661, 0.1154]. Huber differences are at the 10⁻⁴ level.

### Response-adaptation diagnostic: response functionality and interface adaptation

Response loss on the 1,957 common coordinates. "Val" is the source-anchor condition
holdout; "Jurkat" is the external anchor, no-change reference 34.31.

| State | Trainable | Val | Jurkat | Jurkat / no-change | Identity advantage, Jurkat [95%] |
| --- | --- | ---: | ---: | ---: | --- |
| Untrained composite (B-init; seed-0 joint backbone construction, untrained interface) | none | 900.6 | 479.8 | 14.0 | −0.003 [−0.057, 0.057] |
| Joint backbone as trained (B-joint) | none | 540.0 | 200.6 | 5.85 | not detected |
| Interface-only adaptation (B-interface; 1e-4, 50 epochs) | interface | 33.06 | 135.96 | 3.96 | 0.192 [0.165, 0.221] |
| Extended interface-only adaptation (B-continue; +50 epochs) | interface | 30.73 | 173.55 | 5.06 | 0.317 [0.247, 0.388] |
| Interface adaptation with STATE unfrozen (B-unfreeze; + inherited STATE) | interface + STATE | 26.54 | 179.98 | 5.25 | positive |

The joint backbone as trained is about 21 × the no-change reference on its own anchors' holdout. Adaptation
recovers source-anchor reconstruction (900 → 33) and a small identity advantage but
never beats no-change on any held-out condition set, and the Jurkat error grows as
source training continues.

### Interface-isolation diagnostic, existing-export analysis: bias decomposition and matched coverage

Shared per-context bias fraction of the adapted states' Jurkat error: interface-only adaptation 0.824,
extended interface-only adaptation 0.861, interface adaptation with STATE unfrozen 0.869. The train-side perturbation-mean prior is worse than
no-change on every anchor's covered condition set (Jurkat 39.16 vs 34.33); the adapted
states' ratios on the same covered sets are 1.0–1.7 on source anchors and 4.8–5.1 on
Jurkat.

### Interface-isolation diagnostic, native functional check: the released checkpoint in its own numeric space

The cached HVG bags and response targets are raw UMI counts (integers, HVG-panel row sums
4.0k–6.3k) from the raw source h5ads. arc-state's own preprocessing applies
`normalize_total` then `log1p` before HVG selection. The composite backbone, in the seed-0 joint backbone and in
every response-adaptation state, fed raw counts to a decoder trained in log space and scored its output
against count-space targets. Ratio of the native checkpoint's response loss to the
no-change reference on held-out conditions, batch index 0, panel `native_all`:

| Input space | K562 | HepG2 | HCT116 | Jurkat |
| --- | ---: | ---: | ---: | ---: |
| raw counts (spec candidate i) | 18.9 | 11.1 | 13.9 | 6.3 |
| log1p(norm to 1,500) | 10.6 | 5.9 | 15.6 | 9.8 |
| log1p(norm to 2,500) | 2.51 | 1.52 | 9.37 | 2.21 |
| log1p(norm to 3,000) | 1.55 | 0.99 | 8.86 | 1.22 |
| **log1p(norm to 3,500)** | **1.28** | **0.84** | 8.93 | **0.95** |
| log1p(norm to 4,000) | 1.33 | 0.90 | 9.27 | 0.97 |
| log1p(norm to 4,505, cached-cell median) | 1.55 | 1.06 | 9.71 | 1.11 |
| log1p(norm to 6,000) | 2.61 | 1.75 | 11.0 | 1.74 |
| log1p(norm to 10,000) | 5.28 | 3.50 | 13.9 | 3.34 |

Identity advantage as a fraction of the model's loss at target 3,500: K562 0.22, HepG2
0.66, HCT116 0.03, Jurkat 0.41; in count space 0.00 on every anchor. Batch indices 1–4
move losses by under 0.3%. Normalisation is over the 2,000-gene HVG panel because the
cache carries no per-cell library sizes, so this is a proxy for the checkpoint's own
whole-library normalisation. The pre-registered K562 gate (within 10% of no-change) is
missed at 27.5%.

### Interface-isolation diagnostic, learning-rate isolation: adapted Tx1 basal encoder recipe, Jurkat fold, 50 epochs

| Interface LR | Val at epoch 50 | Jurkat / no-change |
| --- | ---: | ---: |
| 1e-4 (adapted Tx1 basal encoder) | 33.1 | 3.96 |
| 1e-5 | 133.7 | 4.63 |
| 1e-6 | 600.1 | 10.2 |

### Interface-isolation diagnostic, interface-variant comparison under leave-one-anchor-out folds

Held-out ratio (model over no-change on the held-out anchor, panel `all`) per fold, the
equal-fold pooled ratio with its gene-bootstrap interval, and the identity advantage on
the held-out anchor. Best epoch was the cap (50) for every native-basal-path and adapted Tx1 basal encoder fold; the
expression-residual interface early-stopped at epochs 8–31.

| Variant | jurkat | k562 | hepg2 | hct116 | Pooled ratio [95%] | Held-out identity advantage | Kept |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Adapted Tx1 basal encoder (= interface-only adaptation) | 3.96 | 19.9 | 10.2 | 35.3 | 17.33 [16.81, 17.88] | 0.19 / – / – / – | no |
| Expression-residual interface (null-subtracted) | 1.022 | 1.044 | 1.002 | 1.055 | 1.031 [1.028, 1.033] | 0.92 / 0.12 / 0.04 / 0.003 | no |
| Native basal path (ESM-2 perturbation tokens) | 5.15 | 36.5 | 10.0 | 21.1 | 18.19 [17.72, 18.72] | – | no |
| Native basal path + trainable Tx1 context term (zero-init) | 2.86 | 11.85 | 5.62 | 27.7 | 12.00 [11.63, 12.39] | – | no |

Expression-residual interface internal-validation ratios on the source anchors are below 1 in every fold (0.83–0.96).
On the HepG2 fold the train-side perturbation-mean prior (51.4) beats both the expression-residual interface (54.9) and
no-change (54.8). The native basal path (ESM-2 perturbation tokens) starts at the native raw-count loss (336–466) and descends to
288–409 in 50 epochs; the native basal path + trainable Tx1 context term descends to 22–38 because its trainable Tx1 context layer learns
a count-space offset. The native basal path + Tx1 term + expression residual was not eligible and did not run.

### Interface-isolation diagnostic, expression-space round (round 2): the same variants in the decoder's own space

Run `outputs/p1c/p1c_log3500_seed0_20260909T110631Z` (code `3f226e5`, host 30030, 2 × H20):
every fold prepared under log1p over HVG-panel row sums normalised to 3,500; loss,
references, derangements and cross-context metrics computed in that space; existing-export analysis and
head-seed stability not repeated; the released STATE checkpoint, untrained (N-native), evaluated at the same transform on every fold (batch index 0).
Init checks passed on every fold (expression-residual interface deviation 0.0; both native-basal-path variants within 1.8% of the native
null forward). The interface-variant comparison table below is from the first comparison pass (compare-1)
([`evidence/p1c/round2/comparison/`](evidence/p1c/round2/comparison/), copied from the run's
`comparison/`; kept.json all false, the native basal path + Tx1 term + expression residual not eligible). The two adapted Tx1 basal encoder learning-rate arms
(1e-5, 1e-6, Jurkat fold) stopped without an exit record: their logs end on 2026-09-10 at
epochs 22 and 23 of 50, and the final comparison pass (compare-final) never ran.

| Variant | jurkat | k562 | hepg2 | hct116 | Pooled ratio [95%] | Held-out identity share | Kept |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Released STATE checkpoint, untrained (reference) | 0.953 | 1.248 | 0.923 | 7.80 | — | 41% / 22% / 66% / 3% | — |
| Adapted Tx1 basal encoder | 17.6 | 32.9 | 20.5 | 32.0 | 25.7 [25.3, 26.2] | ≤ 2% | no |
| Expression-residual interface | 1.009 | 1.216 | 0.981 | 1.148 | 1.088 [1.086, 1.091] | 3.0 / 3.2 / 2.4 / 1.7% | no (hepg2 fold passes all legs) |
| Native basal path (ESM-2 perturbation tokens) | 1.007 | 1.63 | 1.020 | 11.9 | 3.88 [3.81, 3.95] | 8.8 / 2.4 / 5.1 / 0.2% | no |
| Native basal path + trainable Tx1 context term | 6.04 | 12.7 | 11.5 | 11.7 | 10.5 [10.3, 10.7] | ≤ 1% | no |

Every arm fits its source anchors (internal-validation ratios 0.84–0.98 for the expression-residual interface on every
anchor; the adapted Tx1 basal encoder and the native basal path + trainable Tx1 context term below 1 except HCT116 at 2.7–3.0;
the native basal path (ESM-2 perturbation tokens) cannot fit HCT116, 10.3).
The expression-residual interface passes legs (a) and (c) on all four folds and leg (b) on hepg2 only. The global-mean
effect beats the expression-residual interface on three held-out lines. The native basal path (ESM-2 perturbation tokens)
keeps the largest held-out identity use of any trained arm but never lowers the loss below no-change.

## Interpretation

- **The numeric interface is the defect.** The composite feeds raw UMI counts into a
  STATE decoder trained on log1p-normalised expression and scores against count-space
  targets. Everything measured about the seed-0 joint backbone's response pathway in the response-adaptation
  diagnostic (no identity use, 21 × no-change) and the count-space variants in the interface-isolation diagnostic
  follows from this. The released
  checkpoint carries real response functionality in approximately its own space: it beats
  no-change on HepG2 and Jurkat with identity use of 22–66% of loss, and misses the K562
  gate by a margin (27.5%) that the HVG-panel proxy normalisation may explain. HCT116
  fails at every scale, consistent with an X-Atlas-Orion platform mismatch rather than a
  scale issue.
- **The expression-residual interface isolates the interface without transferring anything.** Starting exactly at
  no-change removes the blow-up on every held-out fold (ratios 1.00–1.06) and beats
  no-change on every source anchor, but the pooled held-out interval excludes 1 from above
  and the held-out identity advantage collapses from 0.92 on Jurkat to 0.003 on HCT116.
  In count space the residual head learns a shared per-context shift, not a
  perturbation-specific effect.
- **In the decoder's own space the failure is the basal skip, then the interface.** The
  held-out error scales with how much of STATE's hidden-space skip is trained on Tx1 input
  (adapted Tx1 basal encoder 18–33 ×, native basal path + trainable Tx1 context term 6–13 ×, native basal path
  (ESM-2 perturbation tokens) ≈ 1 ×); the expression-residual interface, which bypasses the skip, is at parity with
  identity use of 2–3% and is matched by a constant shift. The native basal encoder cannot
  absorb the HCT116 platform shift on either side (8.9 untrained, 10.3 trained).
- **Learning rate is monotone but not isolated.** 1e-6 barely moves the interface within
  the budget, consistent with the insufficient-adaptation hypothesis (H1; the seed-0 joint backbone gave the new basal
  encoder the STATE rate 1e-6), but every arm is fitting a count-space offset, so learning-rate isolation bounds the
  insufficient-adaptation hypothesis rather than isolating it.
- **The head result is stable.** The explicit context-slope head minus the shared MLP head is +0.07 to +0.08 residual
  Pearson with intervals excluding 0 at all three head seeds; the response block adds nothing at any seed; the explicit
  context-slope head does not separate from Tx1 PCA8-ridge. The explicit context-slope head remains the head control for
  any future response representation.
- **Shared bias dominates** the adapted states' Jurkat error (82–87%), and the
  perturbation-mean prior is not a useful reference on covered sets.

## Verdict and scope

Closed-negative for the count-space response interface of the seed-0 joint backbone;
implementation finding, reportable and not claimable. The GeneEffect numbers here are
validation-only and support no context-modelling advantage (the explicit context-slope head ties PCA8-ridge). No SL
evidence. Single training seed and single head-data-order seed throughout; intervals
condition on selected checkpoints and do not estimate initialisation or new-anchor
variability. Jurkat was observed in the response-adaptation diagnostic before the interface-isolation diagnostic
registered its folds, so the Jurkat
fold is a diagnostic re-evaluation; the K562, HepG2 and HCT116 folds supply the leave-one-anchor-out
evidence. The released STATE checkpoint is named for Replogle training data, so K562 is
not held out from its pretraining; exposure of the Nadig 2025 HepG2/Jurkat and
X-Atlas-Orion HCT116 sets is not established here. Tx1 Tahoe-100M exposure applies to
every Tx1-conditioned arm. Any further response work must first move preparation into the
response model's own numeric space; the count-space round is retained as the record of
the defect and is never pooled with a log-space round.

## Reproduction

```bash
# P1-A (afdd7fc): outputs/launches/p1a_seed0_20260907T144045Z/launch.json
# P1-B (0723462): outputs/launches/p1b_seed0_20260907T164813Z/launch.json
# P1-C (4b48381 / ca6a71e): outputs/launches/p1c_seed0_20260908T112354Z/launch.json
RUN=outputs/p1c/<run_id> P0_CHECKPOINT=... P1B_PREPARED=... P1B_RUNS=... P1A_FEATURES=... \
P1A_REFERENCE_P0=... P1A_REFERENCE_PCA=... GPUS="0 1" bash hpc/p1c_pipeline.sh
uv run python -m src.experiments.p1c prepare --checkpoint <best.pt> --out-dir <dir> --fold jurkat \
  --reference-manifest <p1b_prepared>/manifest.json --transform log1p_norm --target-sum 3500   # Tier 1b
uv run python -m src.experiments.p1c evaluate-native --prepared <dir> --runs <runs> --batch-indices 0 1 2 3 4
uv run python -m src.experiments.p1c compare --root $RUN --out-dir $RUN/comparison
```

Operator details: [runbook, interface-isolation section](../../../hpc/README.md#p1-c-interface-isolation).
