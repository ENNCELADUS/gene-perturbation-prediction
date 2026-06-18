# Experiment 09 — Cross-Cell-Line Selectivity SL Model (Design)

**Status:** Design approved 2026-06-18 (brainstorming). Awaiting spec review
before plan.
**Parent:** Experiment 06 (dependency-only SL-pair MVP). Reuses exp06's
benchmark, CV1/CV2/CV3 splits, official metric protocol, and the
`src/sl_benchmark_baseline/` harness.
**Lineage note:** exp06 is GeneEffect-only. exp09 adds three new DepMap omics
modalities (mutation, copy-number, expression) used only to define a
cross-cell-line stratification; the dependency signal compared is still
GeneEffect. Because it crosses into multi-omics and multi-cell-line territory,
it is a **new experiment**, not an in-place exp06 edit. exp06's locked floor
(F1 CV1=0.730, etc.) stays untouched.

## 1. Motivation

DepMap `CRISPRGeneEffect` is a single-gene knockout screen keyed by
`cell_line x single_gene_KO`, not a double-KO screen. Classic cancer synthetic
lethality (SL) evidence therefore comes from **cross-cell-line comparison**:

```text
gene_a-defective cell lines:  KO(gene_b) GeneEffect is strongly negative
gene_a-intact   cell lines:  KO(gene_b) GeneEffect is near 0
  => gene_a defect makes cells dependent on gene_b
  => (gene_a, gene_b) is SL-like
```

exp06 only used the single K562 GeneEffect scalar per gene. exp07 added gwps
Perturb-seq transcript embeddings. Neither used the cross-cell-line dependency
axis. This experiment tests whether the directional **Selectivity** contrast
from `docs/idea/sl_benchmark_formulation_gap.md` lifts SL-pair ranking over the
exp06 dependency-only floor.

This is a **benchmark-adapter feature**, not a validated K562 SL assay. No
context-specific SL biological claim is made (see Terminology Guardrails).

## 2. Task (unchanged from exp06)

- **Target:** binary `D = sl_label(gene_a, gene_b)` from the SynLethDB-derived
  Feng et al. 2024 benchmark, `Rand` 1:1 negatives only.
- **Splits:** CV1 (pair-level), CV2 (one gene held out), CV3 (both genes held
  out, cold-start). CV3 is the make-or-break surface — exp06-B collapses there
  (NDCG@10 0.002).
- **Universe:** 9,471 K562-mappable genes.
- **Official metric:** exp06's `official_classification_metrics` +
  `official_ranking_metrics` (AUROC, AUPR, F1, NDCG@k, Recall@k, Precision@k,
  MAP@k for k in {10,20,50}), per-anchor candidate-partner ranking, seed 17.

## 3. Data Inputs

All under `data/sl_dependency_v0/raw/depmap/` (gitignored). Cell-line axis is
the GeneEffect set of 1,208 lines (K562 = `ACH-000551`, present in all files).
Genes joined by Entrez ID parsed from `SYMBOL (ENTREZ)` column headers.

| File | Lines (∩ GeneEffect) | Genes | Encoding | Role |
|---|---|---|---|---|
| `CRISPRGeneEffect.csv` | 1,208 | 18,531 | float (Chronos) | dependency `d_{c,b}` |
| `OmicsSomaticMutationsMatrixDamaging.csv` | all 1,208 | 19,578 | binary 0/1 | damaging mutation |
| `OmicsSomaticMutationsMatrixHotspot.csv` | all 1,208 | **554 only** | binary 0/1 | hotspot mutation |
| `PortalOmicsCNGeneLog2.csv` | 858 | 19,144 | log2 (~1.0 neutral) | copy-number loss |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | 1,140 | 19,215 | log2(TPM+1) | low expression |

Keying caveats verified during design:
- CN matrix is indexed directly by `ACH-` model ID.
- Mutation and expression matrices carry `ModelID` as a **column** (integer row
  index); must join on `ModelID`, not the index, and drop metadata columns
  (`SequencingID`, `ModelConditionID`, `IsDefaultEntry*`).
- Hotspot covers only 554 genes — it is a sparse minority channel in the OR.

## 4. Defective Call (Composite OR)

For cell line `c` and anchor gene `a`:

```text
defective(c, a) =
      damaging_mut[c, a] == 1
   OR hotspot_mut[c, a]  == 1
   OR CN_log2[c, a] < CN_LOSS_THR          # CN_LOSS_THR = 0.8 (config)
   OR expr[c, a] <= expr_p10(a)            # per-gene bottom decile (config)
```

Missing channel for a gene contributes nothing to the OR (not treated as
defective). `expr_p10(a)` is computed over available expression lines for gene
`a`. All thresholds live in config; none hardcoded in library code.

Feasibility verified: under composite OR, **9,459 / 9,471** benchmark genes
clear the `n >= 20` bar in both groups (low-expression decile alone yields
~120 defective lines/gene). Only ~12 genes hit fallback. Damaging-mutation
alone would have given only 8% — the OR is necessary.

## 5. Selectivity Feature

Directional contrast per ordered pair, restricted to the GeneEffect cell-line
axis. Defined so that **larger = more SL-like** (no hidden negation):

```text
C+_a = { c : defective(c, a) }              (intersected with GeneEffect lines)
C-_a = { c : not defective(c, a) }
dep_def(a, b)    = mean_{c in C+_a}[ d_{c,b} ]   # b's dependency, a-defective lines
dep_intact(a, b) = mean_{c in C-_a}[ d_{c,b} ]   # b's dependency, a-intact lines

sel(a -> b) = dep_intact(a, b) - dep_def(a, b)
```

`d_{c,b}` is GeneEffect (negative = more essential). When `b` becomes more
essential given an `a` defect, `dep_def < dep_intact`, so `sel(a->b) > 0`.
Larger `sel` => stronger SL-like signal.

**Pan-essentiality handling.** A broadly essential `b` is very negative in both
groups, so the *difference* `sel` largely cancels it — but high-variance weakly
essential genes can still inflate `sel`. We therefore (a) surface
`pan_essential(b) = mean_c d_{c,b}` as a separate swap-invariant model feature
(§6), and (b) keep an optional soft penalty
`sel_pen(a->b) = sel(a->b) - LAMBDA * max(0, -pan_essential(b))` with
`LAMBDA` in config, **default 0** (penalty off; the model learns it from the
feature instead). The unpenalized `sel` is the primary feature.

**Min-n fallback (decided):** if `min(|C+_a|, |C-_a|) < N_MIN` (default 20),
set `sel(a->b) = 0` and raise a per-gene `coverage_flag`. Metrics reported on
both the full universe AND a covered-pair slice (mirrors exp07's coverage-flag
pattern).

**Swap-invariant features (decided):** SL labels are symmetric; Selectivity is
directional. Emit two swap-invariant scalars per pair, mirroring exp06's
`sum`/`|diff|` convention:

```text
sel_mean(a,b) = ( sel(a->b) + sel(b->a) ) / 2
sel_absdiff(a,b) = | sel(a->b) - sel(b->a) |
```

## 6. Model Columns

Mirror exp07's additive-column pattern. exp06's `A` (LogReg), `B` (XGB), `C`
(degree probe) re-run unchanged. Add two new columns:

```text
A_xcl (LogReg) = exp06's 5 GeneEffect scalars (min,max,sum,product,|diff|)
                 + sel_mean + sel_absdiff + pan_essential_pair_feature
B_xcl (XGB)    = same feature block
```

- `pan_essential_pair_feature`: a swap-invariant pan-essentiality summary
  (e.g. `min(pan_essential(a), pan_essential(b))`) so the model can separate
  co-dependency from "one gene is just broadly essential."
- Standardization fit on train-fold statistics only, exactly as exp06's
  `Standardizer` (`features.py`). The Selectivity feature itself is computed
  from the **external DepMap matrices and does not depend on the SL train/test
  split** — identical behavior to exp06's K562 scalar, so it introduces no new
  label leakage across CV folds.

**Primary comparison:** A_xcl vs A, B_xcl vs B on CV1/CV2/CV3, with CV3 as the
decisive surface.

## 7. Diagnostics (report-only; decided: raw feature + slices, no in-model controls)

1. **Non-pan-essential slice:** metrics restricted to pairs where neither gene
   is broadly essential (GeneEffect < -0.5 in > 50% of lines). If lift vanishes
   here, Selectivity is re-encoding essentiality, not SL.
2. **Degree-matched permutation null:** shuffle gene_b within anchor preserving
   train-positive degree. Selectivity must beat this on CV2/CV3 to count as
   real (exp06 showed pair-splits are degree-gameable; CV1 especially).
3. **Coverage slice:** full-universe vs covered-pair (non-fallback) metrics.

If the raw feature shows CV2/CV3 lift, per-omic decomposition (separate
Selectivity per defect channel) and an explicit partial-correlation control are
deferred follow-ups, not part of this experiment.

## 8. Hard Guardrails (from methodology review)

- **`Rand` negatives ONLY.** Never `Dep`/`Exp` negatives — they were
  constructed from DepMap dependency/expression covariation and would directly
  leak into a cross-cell-line feature.
- No SL biological claim. "SL candidate prioritization" language requires
  context-specificity evidence not established here. This is a pair-level
  benchmark-adapter feature.
- exp06 floor stays reproducible: exp09 lives in its own results dir; a parity
  gate (see §10) asserts A/B/C still reproduce the locked exp06 numbers when
  re-run through the shared harness.
- All thresholds (`CN_LOSS_THR`, expr decile, `N_MIN`, `LAMBDA`) in config; no
  hardcoded constants in library code.

## 9. Code Touch Points

All within `src/sl_benchmark_baseline/` (already serves exp06 and exp07):

- **`selectivity.py` (new):** load the 4 omics matrices + GeneEffect, build the
  composite defective mask, compute `sel(a->b)` for the full 9,471-gene
  universe (cache the per-pair/universe Selectivity matrix; ~359 MB float32 for
  the dense universe is feasible, build once). Expose coverage flags and
  pan-essentiality vector.
- **`features.py`:** `build_selectivity_pair_features` producing
  `[sel_mean, sel_absdiff, pan_essential_pair_feature]`; append to the exp06
  scalar block for the `_xcl` models.
- **`models.py`:** `LogRegSelectivityModel` (`name="A_xcl"`),
  `XGBSelectivityModel` (`name="B_xcl"`), `build_selectivity_models` factory.
- **`evaluate.py`:** Selectivity score-matrix path for per-anchor ranking; the
  three diagnostic slices (§7); covered-pair reporting.
- **`config.py` + `__main__.py`:** `--selectivity` toggle and omics paths /
  thresholds (`--cn-loss-thr`, `--expr-decile`, `--sel-n-min`, `--sel-lambda`).
  Off by default => exp06/exp07 behavior unchanged when absent.

## 10. Tests & Verification

- **Parity gate:** re-running A/B/C through the harness reproduces the locked
  exp06 official-metric floor (deterministic, seed 17). Asserted in a test.
- **Selectivity unit test:** synthetic GeneEffect + omics with a known contrast
  => `sel(a->b)` matches hand-computed value; symmetrization and min-n fallback
  exercised.
- **Leakage guard test:** assert config rejects non-`Rand` negative sampling
  when `--selectivity` is on.
- **CLI smoke test:** small-universe end-to-end run produces A_xcl/B_xcl rows.
- Standard `uv run ruff check .`, `uv run ruff format .`,
  `uv run python -m pytest`.

## 11. Results Location

```text
docs/experiment/09_k562_sl_pair_cross_cell_line_selectivity.md   (write-up)
results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/
    official_metrics_cv1/  official_metrics_cv2/  official_metrics_cv3/
    official_metrics_summary.csv
    diagnostics/ (non_pan_essential_slice, degree_perm_null, coverage_slice)
```

## 12. Success Criteria

- **Positive result:** A_xcl/B_xcl beat exp06 A/B on CV2 and especially CV3
  official ranking metrics (NDCG@10 / AUPR), AND the lift survives the
  non-pan-essential slice and degree-permutation null.
- **Honest null:** no lift over exp06 on CV2/CV3, or lift that disappears under
  the diagnostics. Reported as a clean negative ("cross-cell-line Selectivity
  does not add over single-cell-line GeneEffect for cold-start SL ranking").
- Either outcome is a publishable, decisive addition. The experiment is
  designed to fail honestly if the signal is not there.


