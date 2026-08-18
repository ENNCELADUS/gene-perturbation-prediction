# Prior Internal Evidence

**Status:** carried forward from the retired dependency / synthetic-lethality program.
These are the only internal results the current research direction depends on.
**Why this file exists:** the full write-ups live under `docs/archive/`, which is untracked
and gitignored. This is the versioned record of the numbers the blueprint cites.

## exp02 — Replogle K562 viability-axis audit

Config: `configs/experiments/02_replogle_k562_viability_axis_audit/`, deleted at `873c99c` (check out `a7e2c91`)

**Task.** Test whether the Replogle K562 pseudobulk B→C signal (perturbation delta
expression → DepMap K562 CRISPR GeneEffect) mostly learns a generic cell-death /
proliferation "response burden" axis. The generic-viability anchor is the 2019 NAR
Achilles/CTRP cell-death-signature coefficients. 5-fold CV × 1 repeat, seed 42, unweighted.

| Model | Feature set | Spearman |
|---|---|---:|
| NAR viability score only | `nar_viability_scores` | 0.244 |
| NAR score + response burden | `nar_viability_scores_plus_burden` | 0.443 |
| Best full-feature pseudobulk baseline | `delta_all` | 0.494 |
| **NAR-residualized transcriptome** | `nar_resid_delta_all` | **0.503** |
| NAR + burden-residualized transcriptome | `nuisance_resid_delta_all` | 0.469 |

**Reading: exp02 supports the specificity hypothesis** ([`01-blueprint.md`](../01-blueprint.md)
§5.2). Residualizing the generic viability axis out of the transcriptome *improves*
prediction of the independent DepMap anchor (0.503 > 0.494) — generic viability is not what
carries the signal. Residualizing out viability **and** burden still leaves 0.469, only
0.025 below the unresidualized baseline: the residual is nearly the entire baseline, not a
thin sliver of it.

Burden alone reaches 0.443 and the burden-free residual reaches 0.469. These are two
**correlated** predictors that overlap without either being reducible to the other —
one's success does not imply the other's redundancy.

## exp09 — K562 SL pair cross-cell-line selectivity

Artifacts: [`results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run/`](../../results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run/) (gitignored)

**Task.** Test whether a cross-cell-line DepMap "Selectivity" contrast feature lifts
gene-pair SL link prediction over the dependency-only floor, on the CV1/CV2/CV3 `Rand` 1:1
K562 benchmark splits.

**The non-pan-essential slice collapse** (`B_xcl` model, CV3):

| Metric | Full | Non-pan-essential slice |
|---|---:|---:|
| AUROC | 0.645 | 0.583 |
| AUPR | 0.651 | 0.490 |

**Reading.** When pairs where either gene is broadly pan-essential are removed, most of the
CV3 lift disappears — it was attributable to essentiality structure, not pair-specific
co-dependency. CV1/CV2 retain a genuine pair-specific component after the same slice; CV3
does not. This is what gates any context-specific-residual claim
([`01-blueprint.md`](../01-blueprint.md) §5.4).

## Sourcing

`docs/archive/` is untracked and gitignored: the archived write-ups exist on disk in this
checkout but are not in git history and will not survive a fresh clone. Every number above
was checked against its archived source. **This file, not the archive, is the record the
rest of the vault cites.**
