# Prior Internal Evidence

**Status:** carried forward from the retired dependency / synthetic-lethality program. These are the only internal results the current research direction depends on.
**Why this file exists:** the full write-ups live under `docs/archive/`, which is untracked and gitignored. This note is the versioned record of the numbers the contract cites.

---

## exp02 — Replogle K562 viability-axis audit

Config: [`configs/experiments/02_replogle_k562_viability_axis_audit/`](../../configs/experiments/02_replogle_k562_viability_axis_audit/)
Archived write-up (untracked, still on disk): `docs/archive/experiment/02_replogle_k562_viability_axis_audit.md`
Documented remote artifact path (not present in this local checkout as of this writing; see the archived doc's own note that the per-config result roots were empty placeholders on the 2026-05-26 inspection): `results/experiments/02_replogle_k562_viability_axis_audit/runs/viability_axis_5x1_main_20260515`

**Task.** Audit whether the Replogle K562 pseudobulk B→C signal (perturbation delta expression → DepMap K562 CRISPR GeneEffect) mostly learns a generic cell-death / proliferation ("response burden") axis, using 2019 NAR Achilles/CTRP cell-death-signature coefficients as the generic-viability anchor. 5-fold CV × 1 repeat, seed 42, unweighted, `internal_cv_all` scope.

**Model-comparison table** (5-fold CV Spearman, reproduced from the archived doc — verified against source, all values match what was supplied):

| Model | Feature set | Spearman |
|---|---|---:|
| NAR viability score only | `nar_viability_scores` | 0.244 |
| NAR score + response burden | `nar_viability_scores_plus_burden` | 0.443 |
| Best full-feature pseudobulk baseline | `delta_all` | 0.494 |
| NAR-residualized transcriptome | `nar_resid_delta_all` | 0.503 |
| NAR + burden-residualized transcriptome | `nuisance_resid_delta_all` | 0.469 |

Verification note: all five values above were checked against `docs/archive/experiment/02_replogle_k562_viability_axis_audit.md` §"5-Fold Validation Results" and match exactly. No mismatch found.

**Reading — and this matters.** The archived doc's own "Main Readout" section characterizes this as "generic response burden plus residual transcriptomic structure," which reads as a headwind. That characterization is **superseded**. `SIGNIFICANCE_CRITERIA_AMENDMENT.md` (amendment A6) and `DECISION_MEMO.md` §2 explicitly **retract** the earlier "thin residual headwind" interpretation as **a misreading of our own data**.

The corrected reading: **exp02 is evidence FOR the specificity hypothesis** — a generic response-burden scalar does *not* capture what the full transcriptome captures, on a DepMap-anchored target. Residualizing the generic viability axis out entirely *improves* performance (0.503 vs 0.494 baseline) — generic viability is not what carries the signal. Residualizing out viability **and** burden still leaves the transcriptome at 0.469, only 0.025 below the unresidualized baseline. Burden alone (0.443) and the burden-free residual (0.469) overlap without either being reducible to the other; treating one predictor's success as implying the other's redundancy was the error in the original reading. exp02 is **not** a headwind against the program.

Note that the raw `SIGNIFICANCE_CRITERIA.md` (in `ideaspark_run/cell-fate-outcome-dynamics/`) was never edited and still contains the retracted "thin residual" passage — treat that file's exp02 discussion as superseded by the amendment, not as current.

---

## exp09 — K562 SL pair cross-cell-line selectivity

Archived write-up (untracked, still on disk): `docs/archive/experiment/09_k562_sl_pair_cross_cell_line_selectivity.md`
Artifact directory (gitignored, present locally): [`results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run/`](../../results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run/)

**Task.** Test whether a cross-cell-line DepMap "Selectivity" contrast feature lifts gene-pair synthetic-lethality link prediction over the exp06 dependency-only floor, on the CV1/CV2/CV3 `Rand` 1:1 K562 benchmark splits.

**The non-pan-essential slice collapse** (confound diagnostic, `B_xcl` model, reproduced from the archived doc — verified against source, matches exactly):

| Split | AUROC full | AUROC non-pan-essential | AUPR full | AUPR non-pan-essential |
|---|---:|---:|---:|---:|
| CV3 | 0.645 | 0.583 | 0.651 | 0.490 |

Verification note: both values (AUROC 0.645 → 0.583, AUPR 0.651 → 0.490) were checked against `docs/archive/experiment/09_k562_sl_pair_cross_cell_line_selectivity.md` §"Confound diagnostic" and match exactly. No mismatch found.

**Reading.** When pairs where either gene is broadly pan-essential are removed, the CV3 classification lift shrinks substantially — AUROC drops to near the dependency-only floor and AUPR falls further. The archived doc's own verdict: on CV3, "most of the cold-start classification lift is attributable to essentiality structure, not pair-specific co-dependency." CV1/CV2 retain a genuine pair-specific component after the same slice; CV3 does not. This is what gates any context-specific-residual claim built on the cross-cell-line Selectivity feature — see [`01-research-direction.md`](../01-research-direction.md) §5.4.

---

## A note on sourcing

`docs/archive/` is **untracked and gitignored** in this repository (see `.gitignore`). The two archived documents cited above exist on disk in this checkout but are not part of git history and will not travel with a fresh clone. Every number reproduced in this file has been checked directly against the archived source doc at the time of writing; if either archived file is later edited or removed, this file — not the archive — is the versioned record the rest of the vault should cite.
