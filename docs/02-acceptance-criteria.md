# Acceptance Criteria (Frozen)

**Status:** frozen before evidence. **Not revisable to fit a result.**
**What this is:** the bar a result must clear to count as an answer. [`01-blueprint.md`](01-blueprint.md) says what may be *claimed*; this document says what counts as *established*.
**Governs:** every reported head-to-head, every "beats SOTA" statement, and every mechanistic claim in [`04-roadmap.md`](04-roadmap.md) and the paper.

A threshold set after seeing the data is not a threshold. "More powerful and
accurate" is only meaningful against numbers fixed in advance. This document fixes
them, and it is frozen.

## What "more powerful and accurate" decomposes into

| Phrase | Criterion | Axis |
|---|---|---|
| **more powerful** | **BEAT-SOTA** | beats SLMGAE/KR4SL on cold-start ranking |
| **accurate** | **MECHANISTIC** | the virtual double-KO matches *measured* epistasis |
| (guard) | **PAIR-SPECIFIC** | the win is SL, not pan-essentiality |
| (guard) | **INTEGRITY** | the number is admissible, not an artifact |

All four bind. A result that misses one is reported as missing it — the bar does not
move.

## Operational definitions

### The harness

Every number is produced by the **identical** Feng2024 K562 `cal_metrics` harness:
9,471-gene universe, balanced Rand 1:1 pairs, per-anchor ranking, five folds, fixed
seeds. Method and baselines differ only in the scorer, never in splits, seeds, or
metric code. A cross-run comparison is **inadmissible**; the comparison must be a
true in-harness ablation.

### Primary metric

**Per-anchor NDCG@10** is primary (MAP@10 and NDCG@{20,50} reported alongside).
Classification (AUROC, AUPR, F1) is reported but is **not** the discriminating axis:
the dependency-only floor already reaches AUROC ≈ 0.70, near the label-graph SOTA, so
ranking is where cold-start SL is won or lost.

### The SOTA reference

The bar is the **best reproduced label-graph method under this harness.** Reproduced
K562 reference points (source: retired-program SLBench reproduction):

| Method | CV2 NDCG@10 | CV3 NDCG@10 | cross-CV NDCG@10 |
|---|---:|---:|---:|
| GRSMF | 0.315 | 0.313 | 0.317 |
| DDGCN | 0.241 | 0.243 | 0.257 |
| Dependency-only floor | 0.042 | 0.002 | 0.032 |
| Archive best label-free (exp07) | 0.094 | ~0.001 | 0.090 |

**The strong bar is SLMGAE and KR4SL, not KG4SL.** Per Feng2024 (pan-cancer, Rand
1:1), SLMGAE is the top cold-start model (CV3 AUROC 0.790, NDCG@10 0.039) and KR4SL is
Feng2024's flagged CV3 leader, whereas **KG4SL is weak** (CV3 AUROC 0.562, below the
dependency floor's 0.596). **SLMGAE, KR4SL, and KG4SL must be reproduced under this
identical K562 harness** (a prerequisite deliverable); their reproduced CV2/CV3
NDCG@10 become the formal target. Until then, **GRSMF's ≈ 0.31 is the standing K562
ranking bar** and the method must exceed the best reproduced label-graph method.
(K562-reproduction NDCG differs sharply from Feng2024 pan-cancer — e.g. GRSMF CV3
0.313 vs 0.000 — so only same-harness numbers are comparable.)

### "Pan-essential"

A gene is **pan-essential** iff it is a DepMap common-essential gene (the
DepMap-provided common-essential list for the release in use). A pair is
pan-essential if **either** gene is. The **non-pan-essential slice** is the set of
test pairs with **neither** gene common-essential.

### Measured genetic interaction

The wet-lab interaction score from a **CRISPRi double-perturbation** assay. The
fitness-scale anchor is **Horlbeck 2018** (K562 dual-CRISPRi GI map, ~222k pairs) —
**to be acquired**, evaluated on pairs/genes **disjoint from the benchmark positives**
(Feng2024's K562 positives may be Horlbeck-derived). **Adamson 2016 UPR epistasis**
(3 sensors + combos) is the only *local* combinatorial set — a small **qualitative
transcriptomic** check, not a powered fitness-GI benchmark. The **Jost/Replogle
dual-sgRNA file is NOT a GI dataset** (single-gene knockdown efficacy) and is
excluded. **Norman 2019 is CRISPRa — auxiliary only,** never counted toward the
primary MECHANISTIC verdict, always with the modality caveat.

## BEAT-SOTA — the ranking win (primary)

> On **both CV2 and CV3**, the method's per-anchor NDCG@10 (mean over the five
> folds) must exceed the **best reproduced strong label-graph SOTA (SLMGAE, KR4SL)**
> **and** the dependency-only floor. (KG4SL is a weak reference, not the bar.)

Two claims, never conflated (mirroring "exists vs. large"):

| Claim | Requirement |
|---|---|
| A win **EXISTS** | The paired improvement (method − best SOTA), bootstrapped over anchors and across the five folds, has a **95% CI excluding zero on both CV2 and CV3.** |
| The method is **SOTA-grade (LARGE)** | The **95% CI lower bound** on that improvement **exceeds `δ_win` = +0.02 absolute NDCG@10** over the best SOTA, on both CV2 and CV3. |

A point estimate above SOTA with a CI touching zero is **"above the reproduced SOTA
mean, difference not established"** — not a win. A win on CV2 but not CV3 is reported
as **"CV2 (one-gene-cold) only"** and does not clear the cold-start bar, whose whole
point is CV3.

**CV1 is inadmissible as a win.** It is degree-gameable (a gene-degree probe tops it);
it is reported only as the diagnostic. A method that "beats SOTA" only on CV1 has not
beaten SOTA.

## PAIR-SPECIFIC — the win is SL, not pan-essentiality

> The BEAT-SOTA improvement must **persist on the non-pan-essential slice** of CV3.

| Observed on the non-pan-essential CV3 slice | Verdict |
|---|---|
| Improvement CI still excludes zero | Pair-specific SL signal established |
| Improvement CI includes zero | **"Pan-essentiality signal; pair-specific SL not established."** The full-set win is downgraded, not accepted as SL. |

Grounding: the retired decomposition showed a naive cross-line lift collapses on this
slice (CV3 AUROC 0.645 → 0.583, AUPR 0.651 → 0.490). Pan-essentiality is easy and is
not the biology of interest; the residual is.

## MECHANISTIC — correspondence with measured epistasis

> The virtual double-knockout score `s_B` (or the interaction residual) must correlate
> with the **measured** genetic-interaction score on the pairs covered by a CRISPRi
> dual-perturbation assay.

| Claim | Requirement |
|---|---|
| Correspondence **EXISTS** | Spearman between `s_B` and the **measured fitness GI** (Horlbeck 2018 K562, on pairs disjoint from the benchmark positives) has a **95% CI excluding zero**, in the direction where SL-predicted pairs carry stronger (more negative) measured interaction. |
| **STRONG** correspondence | Spearman **point estimate ≥ 0.30** with CI excluding zero on Horlbeck. |
| **Qualitative local check** | On **Adamson 2016 UPR** (transcriptomic, ~3 sensors), the predicted interaction ordering matches the known UPR-branch epistasis direction. Supportive only — its tiny n can neither certify nor refute EXISTS/STRONG. |

**No leakage:** genes/pairs used for any SL-label head calibration must be disjoint
from the measured-GI evaluation set. A correspondence computed on pairs that also
trained a calibration head is inadmissible.

**Availability caveat:** if Horlbeck 2018 is not acquired, MECHANISTIC is **"not
evaluable"** (only the qualitative Adamson check exists) — it cannot be quietly
downgraded to a pass on the local transcriptomic set alone.

## INTEGRITY — the number is admissible

Every reported result must satisfy all of:

1. **Train-only selection.** Epoch/checkpoint/hyperparameter selection reads **no**
   test-fold metric. A test-fold-selected number (the archive's exp08 flaw) is
   **inadmissible**, not merely caveated.
2. **Five folds.** Report mean ± 95% CI across the benchmark's five folds. A
   single-fold number is a diagnostic, not a result.
3. **Zero-shot reported separately.** The pure composition (no SL labels in features)
   is a standalone row; any label-calibrated head is an additional, clearly-labelled
   row. A win that exists only with a label-calibrated head is reported as such.
4. **Identical harness** (above). 

## Verdict rule

| Condition | Verdict | Meaning |
|---|---|---|
| BEAT-SOTA (≥ EXISTS on CV2 **and** CV3) **and** PAIR-SPECIFIC **and** INTEGRITY | **Beats SOTA on cold-start SL** | The headline success ("more powerful"). |
| …**and** MECHANISTIC STRONG | **Mechanistically-grounded SOTA** | The full claim ("more powerful **and** accurate"). |
| BEAT-SOTA fails **but** MECHANISTIC ≥ EXISTS | **Mechanism validated, ranking not yet SOTA** | Partial. Honest, still a contribution (composition + measured-epistasis correspondence). |
| Neither BEAT-SOTA nor MECHANISTIC | **Negative** | Reported as such. The composition did not beat the floor/SOTA and did not match measured epistasis. |

## What would make these thresholds wrong

Each is deliberately falsifiable, with a named failure mode:

1. **`δ_win` = 0.02 NDCG@10 may be too lax or too strict.** At a SOTA of ≈ 0.31 it is
   a ~6% relative lift — enough to matter, small enough to be reachable. If the
   reproduced SLMGAE/KR4SL bar is much higher or lower than GRSMF's 0.31, the
   *relative* interpretation shifts; the rule (CI-based, on both CV2 and CV3) does
   not.
2. **The non-pan-essential slice may be small**, widening its CI and making
   PAIR-SPECIFIC hard to clear even for a real effect. Report the slice size and its
   power; an underpowered slice routes to "not established," not to a claim either way.
3. **Measured-GI coverage is thin.** Adamson UPR is a 3-sensor qualitative set;
   Horlbeck 2018 (the fitness-GI anchor) must be acquired and de-circularized against
   the benchmark positives. A MECHANISTIC verdict is bounded by that coverage and must
   be stated, not implied; without Horlbeck it is "not evaluable," not a pass.
4. **Ranking primacy may understate a real classification gain.** If the method
   clearly wins AUROC/AUPR on CV2/CV3 but not NDCG@10, that is reported as a
   classification result, explicitly not a ranking-SOTA claim.
