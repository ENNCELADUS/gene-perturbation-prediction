# Tx1-3B-ST and HVG-ST cross-line GeneEffect Phase F report

**Status:** completed 2026-07-29; the registered Tx1-3B-ST k=10 primary gate is
**negative**. Tx1-3B-ST did not improve over copy-K562 + 10 labels
(`Delta rho = -0.0048`, line-bootstrap 95% CI `[-0.0941, 0.0769]`, registered
`rho_min = 0.05`). The HVG-ST encoder-unseen attribution control was also
negative (`Delta rho = 0.0326`, 95% CI `[-0.0602, 0.1181]`).
This is a task-data-held-out **single-gene GeneEffect backbone** result, not a
pairwise synthetic-lethality result. The registered primary kill test is closed;
the broader Phase F baseline ladder remains incomplete.
**Superseded (2026-08-17):** Exp13 (`../specs/2026-08-17-exp13-geneeffect-residual-protocol.md`) replaces
this design with the corrected five-block composition (`01-blueprint.md` §3 amendment) on the
226-line benchmark. The numbers below remain registered evidence and are not revised.

## What was tested

The experiment asked whether a frozen perturbation-response model plus a rebuilt
GeneEffect head transfers dependency ranking to nine fixed test cell lines seen
by the fitted task only through their basal state. Two matched arms differed in
the encoder/ST input view:

- **Tx1-3B-ST:** 2,560-dimensional Tx1-3B basal embeddings, with known Tx1
  pretraining exposure to all nine target lines;
- **HVG-ST:** 2,000-gene HVG expression, the encoder-unseen attribution control.

Both arms used the same 28 training lines, 5 validation lines, 9 test lines,
587-gene differentially-essential slice, two-moment response pooling, GeneEffect
head architecture, frozen 20-panel label schedule, and `k = {0, 5, 10, 25, 50}`.
The primary registered comparison was the per-line mean Spearman difference
against **copy-K562 + k labels**, evaluated on genes not used for that panel's
k-shot calibration. The k=10 macro difference used the nine cell lines as the
bootstrap units, with 10,000 resamples and seed 20260725.

## Method and provenance

| Item | Tx1-3B-ST | HVG-ST |
| --- | --- | --- |
| Phase D arm | `tx1_arm` | `hvg_arm` |
| ST checkpoint SHA-256 | `e7d75d369187a5caa82a8a735302c7b752d6309445f227f66459028ae44bdb12` | `045bef12a3e33ee773f3776f1a274ebc33afcffccf2203f35528b03444261e8b` |
| Selected head epoch | 185 / 199 | 61 / 199 |
| Validation rank-variance loss | 0.677380 | 0.691339 |
| ST input/output width | 2560 / 2000 | 2000 / 2000 |
| Response macro-batch | 64 windows | 64 windows |
| Phase F status | primary | attribution control |

Phase F ran at git SHA
`08116dedd4adb364e73f650179d8ecdc39665b31`. The frozen DepMap 26Q1
`CRISPRGeneEffect.csv` SHA-256 was
`e610a4cefb13a82b5b256b47eb08b63ff14843f8dbd0fb164bc0a32688e5b89e`.
The copy-K562 baseline used its registered model ID `ACH-000551` and the same
affine k-shot correction and score-gene exclusions as the candidate arms.

Strict input validation confirmed 9 test lines, 587 slice genes, 20 panels per
line, 50 ordered label genes per panel, the complete k-schedule, no duplicate
prediction keys, and consistent targets across methods. The Tx1 combined-input
SHA-256 was
`25abbc23d4ab68b284068890385d66df5492bc58c588eca881bba50fb5d8169a`;
the HVG combined-input SHA-256 was
`20d0f0129a114168a41f6f71f94a13f1a55599a1c983ad23c232720c82f1fa99`.

### INTEGRITY admissibility

| Gate | Evidence |
| --- | --- |
| Train-only selection | Head selection used the five registered validation lines; all nine test lines were absent from training and checkpoint selection. |
| Fixed manifests | Cell-line roles, 587-gene slice, 20 panels, k-schedule, estimator, seed, and hashes were frozen before Phase F. |
| Pretraining lineage | Tx1 exposure is known present for all target lines and is disclosed; HVG is the encoder-unseen control. |
| Five official folds | Not applicable: this is the separately registered fixed cell-line split, not Feng2024 CV. |
| Identical harness | Both arms used the same split, targets, panels, baseline, and metric code; only the declared encoder/ST arm differed. |
| Zero-shot separated | k=0 is reported separately from every label-adapted row. |
| No cross-axis relabelling | The result is single-gene GeneEffect transfer, not unseen-SL-gene, pairwise SL, or measured-GI evidence. |
| Coverage disclosed | All 9 test lines and all 587 registered genes were scored for every panel/k combination. |
| Estimator registered | Line-level paired differences, panel aggregation, bootstrap unit, repetitions, seed, and threshold were fixed in Phase A. |

## Result

### Full few-shot curves

`Candidate rho` and `copy-K562 rho` are unweighted means of the nine per-line,
20-panel mean Spearman correlations. `Delta rho` is their paired difference;
the CI bootstraps the nine line-level differences.

| Arm | k | Candidate rho | copy-K562 rho | Delta rho | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tx1-3B-ST | 0 | 0.3402 | 0.2130 | +0.1272 | [0.0462, 0.2025] |
| Tx1-3B-ST | 5 | 0.1865 | 0.0999 | +0.0867 | [0.0201, 0.1465] |
| **Tx1-3B-ST** | **10** | **0.1225** | **0.1273** | **-0.0048** | **[-0.0941, 0.0769]** |
| Tx1-3B-ST | 25 | 0.0702 | 0.1341 | -0.0639 | [-0.1498, 0.0160] |
| Tx1-3B-ST | 50 | 0.0626 | 0.1706 | -0.1080 | [-0.2237, -0.0004] |
| HVG-ST | 0 | 0.2695 | 0.2130 | +0.0565 | [-0.0477, 0.1590] |
| HVG-ST | 5 | 0.2010 | 0.0999 | +0.1012 | [0.0369, 0.1600] |
| **HVG-ST** | **10** | **0.1600** | **0.1273** | **+0.0326** | **[-0.0602, 0.1181]** |
| HVG-ST | 25 | 0.1077 | 0.1341 | -0.0264 | [-0.1121, 0.0537] |
| HVG-ST | 50 | 0.0821 | 0.1706 | -0.0885 | [-0.1933, 0.0143] |

Neither k=10 interval excludes zero or clears the registered lower-bound
threshold of 0.05. Tx1's k=0 interval is positive but its lower bound, 0.0462,
also falls just below `rho_min`; k=0 was a registered stress point, not the
primary gate.

### Registered k=10 result by test line

| ModelID | copy-K562 rho | Tx1 rho | Tx1 Delta | HVG rho | HVG Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACH-000178 | 0.0991 | 0.0952 | -0.0039 | 0.1784 | +0.0793 |
| ACH-000348 | 0.0025 | 0.1438 | +0.1413 | 0.2114 | +0.2089 |
| ACH-000389 | -0.0041 | 0.1172 | +0.1213 | 0.1658 | +0.1699 |
| ACH-000496 | 0.0136 | 0.0895 | +0.0759 | 0.0734 | +0.0598 |
| ACH-000552 | 0.1908 | 0.1195 | -0.0714 | 0.1580 | -0.0328 |
| ACH-000790 | 0.3423 | 0.1656 | -0.1766 | 0.2068 | -0.1355 |
| ACH-000793 | 0.3975 | 0.1327 | -0.2648 | 0.1566 | -0.2409 |
| ACH-000890 | -0.0048 | 0.1227 | +0.1275 | 0.1499 | +0.1548 |
| ACH-000950 | 0.1091 | 0.1166 | +0.0075 | 0.1395 | +0.0303 |

Tx1 had a positive point difference on 5/9 lines; HVG did so on 6/9. Large
negative differences on `ACH-000790` and `ACH-000793`, where copy-K562 was
strong, dominated the population estimate. These fractions are descriptive and
do not override the registered line-bootstrap gate.

### Variance preservation

At k=0, the mean across-line prediction standard deviation was 0.363 for Tx1
and 0.485 for HVG, compared with 0.404 for the targets. The mean per-line
prediction/target standard-deviation ratio was 0.91 for Tx1 (range 0.76-1.28)
and 1.22 for HVG (range 0.50-1.88). The rebuilt head therefore did not simply
collapse to a constant output; HVG instead showed greater and less consistent
dispersion across lines.

## Interpretation

1. **Observation:** Tx1 failed the registered k=10 gate and was slightly worse
   than copy-K562 on average. **Interpretation:** the complete Tx1 response/head
   stack did not add reliable task-specific held-out-line ranking information at
   the registered adaptation budget. **Implication:** the Phase-3 backbone-
   transfer kill test is negative; this model is not a supported substrate for a
   stronger context-specific SL claim.
2. **Observation:** both arms deteriorated as k increased, despite k-shot labels
   being intended to adapt the held-out line. **Interpretation:** the current
   few-shot ridge correction is likely unstable or misaligned with the ranking
   objective; this pattern is not explained by simple output collapse.
   **Implication:** any future redesign should compare identity, affine,
   ridge-residual, and rank-aware calibration across the declared splits.
3. **Observation:** HVG's k=10 point estimate exceeded Tx1's, but its CI crossed
   zero and HVG was not the registered primary method. **Interpretation:** there
   is no positive evidence that Tx1's pretrained embedding contributed the
   claimed transfer advantage. **Implication:** this is descriptive attribution
   evidence.
4. **Observation:** performance was heterogeneous by context. **Interpretation:**
   the candidate helps where copy-K562 is weak but loses badly on two lines where
   the conserved K562 dependency prior transfers well. **Implication:** a single
   global adaptation rule is not reliably context-sensitive across this cohort.

## Verdict and scope

The registered Tx1-3B-ST primary gate is **negative**. The allowed statement is:

> Tx1-3B-ST did not establish task-data-held-out cross-line GeneEffect backbone
> transfer beyond copy-K562 + 10 labels under the registered nine-line design;
> the Tx1 encoder was pretrained on the target lines.

HVG-ST is a negative attribution control. No pairwise double-perturbation phenotype,
explicit interaction null, or
SL/GI label was evaluated, so this report provides no synthetic-lethality or
mechanistic verdict.

The broader Phase F exit is not complete because cross-line mean, nearest-line, lineage-only,
CCLE-bulk, and pseudobulk-basal comparators have not yet been materialized and
run through the same frozen harness.

## Suggested next experiments

1. Reproduce the k-shot degradation on training-line leave-one-line-out and the
   five validation lines, comparing identity, affine, ridge-residual, and
   rank-aware calibration across the declared splits.
2. Complete the registered baseline ladder, especially CCLE-bulk and
   pseudobulk-basal regression, under the existing frozen split; report it as
   closeout evidence, not as a new primary-gate opportunity.
3. If the backbone is redesigned, evaluate it on the declared cell-line split.

## Reproduction

- Phase D provenance: [Tx1](../../results/experiments/12_tx1_st_geneeffect/phase_d/runs/tx1_st_geneeffect_phase_d_tx1_arm/provenance.json) and [HVG](../../results/experiments/12_tx1_st_geneeffect/phase_d/runs/tx1_st_geneeffect_phase_d_hvg_arm/provenance.json).
- Tx1 Phase F artifacts: [input manifest](../../results/experiments/12_tx1_st_geneeffect/phase_f/runs/tx1_primary_08116de_20260729T0700Z/input_manifest.json), [per-line results](../../results/experiments/12_tx1_st_geneeffect/phase_f/runs/tx1_primary_08116de_20260729T0700Z/per_line.csv), [few-shot curve](../../results/experiments/12_tx1_st_geneeffect/phase_f/runs/tx1_primary_08116de_20260729T0700Z/curve.csv), and [gate verdict](../../results/experiments/12_tx1_st_geneeffect/phase_f/runs/tx1_primary_08116de_20260729T0700Z/verdict.json).
- HVG artifacts: [input manifest](../../results/experiments/12_tx1_st_geneeffect/phase_f/runs/hvg_control_08116de_20260729T0705Z/input_manifest.json), [per-line results](../../results/experiments/12_tx1_st_geneeffect/phase_f/runs/hvg_control_08116de_20260729T0705Z/per_line.csv), and [few-shot curve](../../results/experiments/12_tx1_st_geneeffect/phase_f/runs/hvg_control_08116de_20260729T0705Z/curve.csv).
- Evaluator invocation: `PYTHONPATH=src:. .venv-tx1/bin/python scripts/evaluate_tx1_backbone.py --predictions <combined_predictions.csv> --phase-a-dir results/phase_a_tx1_20260724 --out-dir <run_dir>`. The evaluator and the whole Phase A-F pipeline were deleted at `873c99c`: check out `a7e2c91` to re-run.
