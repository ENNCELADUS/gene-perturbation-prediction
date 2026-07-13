# Amendment 1 to the Frozen Significance Criteria

Date: 2026-07-13. Amends `SIGNIFICANCE_CRITERIA.md` (frozen 2026-07-13 pre-Gate-2).
Source: reviewer pass on Decision Memo v1.

---

## Why this amendment is legitimate, and the test it had to pass

Spec §2.1 exists to stop thresholds from being moved **after** seeing evidence. So an
amendment to frozen criteria is presumptively illegitimate and must clear three bars:

1. **It fixes a specification defect, not a threshold disagreement.** Every change below
   corrects a statistical construction that was *incoherent as written* — it does not adjust
   a number because the number proved inconvenient.
2. **Every change makes the bar STRICTER, never more permissive.** The failure mode §2.1
   guards against is loosening a threshold to admit a result. Every amendment here tightens.
   That is the safe direction.
3. **No evidence has been graded as PASSING against the original thresholds.** Gate 2A and
   Gate 2B both returned **insufficient evidence** on every criterion. Nothing was accepted
   under the old text, so nothing is being retro-fitted. The amendment changes what a
   *future* study must do; it revises no verdict already rendered.

All three hold. If any had failed, the correct action would have been to keep the defective
criteria and record the defect — not to amend.

---

## A1. "3× replicate SD" is not a power criterion — REPLACED

**Original (LARGE, noise bar):** *"that difference must exceed 3× the replicate standard
deviation of the same quantity in the same assay."*

**Defect.** SD is not SE. A ratio of an effect to a *dispersion* is a signal-to-noise
heuristic, not a statistical test, and it carries no defined type-I or type-II error rate.
It cannot license a claim of detection or of absence.

**Replacement.** Two *different* claims must be distinguished, and the original text
conflated them:

| Claim | Requirement |
|---|---|
| **An effect EXISTS** | 95% CI **excludes zero** |
| **The effect is biologically LARGE** | 95% CI **lower bound exceeds the LARGE threshold** (0.20 absolute founder-loss, or 30% relative division-rate) |

**"Point estimate ≥ 0.20 with a CI excluding zero" is NOT sufficient for LARGE.** An estimate
of 0.21 with CI [0.02, 0.40] would pass that rule while remaining fully consistent with a
true effect far below the biological bar. Compatibility with a large effect is not evidence
of one.

> **The LARGE criterion is met only when the 95% CI lower bound exceeds the predefined LARGE
> threshold.** Estimator and sampling frame declared before data collection.

A result whose CI excludes zero but whose lower bound sits below 0.20 is reported as
**"effect established, magnitude not established as LARGE"** — a real and publishable finding,
and not a LARGE verdict.

The 3×-SD heuristic may still be reported as a descriptive signal-to-noise figure. It has
**no inferential standing** and may not appear in a verdict.

---

## A2. Matching is an equivalence claim, not a null-difference claim — REPLACED

**Original (§4.4):** *two perturbations are matched when `|F_net(g1) − F_net(g2)| ≤ tau`,
`tau` = 1× within-screen replicate SD.*

**Defect.** This is "we failed to detect a difference," which is not "they are the same."
Absence of a significant difference is not equivalence — and a *noisier* screen makes `tau`
*larger*, so the original rule perversely made matching **easier** as measurement quality got
**worse**.

**Replacement.** Matching requires a **two-one-sided-tests (TOST) equivalence result against
a predefined equivalence margin `delta`**, declared before data collection. `delta` is a
statement about what difference in net fitness is *biologically negligible*, and it must be
justified on biological grounds — **not** inherited from the noise floor of the assay.

The `tau`, `0.5·tau`, `2·tau` sensitivity reporting requirement is **retained** as a
robustness check, and is no longer the definition of matching.

---

## A3. Anchor transportability — NEW REQUIREMENT

**Defect.** The original criteria implicitly assumed `F_net` from DepMap is the net fitness of
the intervention actually performed. **DepMap Achilles is Cas9 knockout scored by Chronos.**
A CRISPRi study performs a *different* intervention — different penetrance (hypomorph vs
null), strength, kinetics, horizon, guide efficacy, and possibly subclone.

**New requirement.** Any study matching on an external `F_net` must **measure its own achieved
net effect** and report the external-vs-achieved relationship. Until that is done:

- claims are limited to *"incremental information beyond an external DepMap fitness
  reference"*;
- the claim *"matched on the net fitness of the same intervention"* is **not available**;
- **unverified matching is a confound that manufactures false positives** — two genes
  "matched" on Cas9 KO GeneEffect may be badly mismatched under CRISPRi, and the resulting
  trajectory divergence would be transport mismatch wearing the costume of biology.

---

## A4. Prevalence claims require a prevalence power analysis — NEW

**Defect.** The original COMMON criterion (≥20% / 5–20% / <5%-powered-absence) specified
thresholds but **no sample size**, making "adequately powered" undefined in practice.

**New requirement.** Any COMMON verdict states its **sampling frame** (how pairs were drawn —
random, purposive, or exhaustive) and its **prevalence confidence interval**.

Reference points for a **zero-event** result (one-sided 95% upper bound, ideal independent
sampling):

| n matched pairs, 0 divergent | 95% one-sided UCL on prevalence |
|---|---|
| 10 | 25.9% |
| 20 | 13.9% |
| 30 | 9.5% |
| **59** | **5.0%** ← smallest n that can certify "powered absence" |

**Purposive (non-random) pair selection and repeated measurement of the same genes reduce
effective sample size below nominal n.** A study of fewer than ~59 independently sampled
zero-event pairs **cannot** return a powered-absence verdict for COMMON, regardless of how
clean its data is.

---

## A5. R2 is a property of DATASETS, not of papers — CLARIFICATION

The original criteria require **R2** (reproducible across **independent datasets**) as the
minimum bar for a Gate 2 positive. Decision Memo v1 glossed this as *"no single study can
return R2"* — **too absolute**. A single paper can perfectly well contain two independently
designed experimental campaigns.

The correct unit of replication is the **dataset / experimental campaign**, not the
publication:

> **A single dataset or experimental campaign cannot establish R2. R2 requires independent
> replication — whether reported within the same study or across separate studies.**

"Independent" means the second campaign does not share the failure modes of the first: not
the same cell-culture batch, not the same imaging run, not the same guide library
preparation. Two runs of one protocol on one thawed vial are one campaign, not two.

Symmetrically, **powered absence** requires the A4 sample size. Between those two bars sits
the honest space most studies occupy: **effect-size estimation with a stated CI.**

---

## A6. The exp02 calibration anchor was MISREAD — corrected

**Original text:** *"a generic response-magnitude scalar reaches Spearman 0.426 against a
0.494 full-feature baseline. So a naive burden scalar already captures ~86% of the achievable
correlation. Any incremental-information claim is competing for a thin residual, and the
≥0.10 margin is calibrated to be meaningfully larger than that residual."*

**Defect.** The inference is invalid. The actual exp02 table:

| Model | Spearman |
|---|---|
| NAR viability only | 0.244 |
| NAR + burden | 0.443 |
| Full pseudobulk baseline | 0.494 |
| **NAR-residualized transcriptome** | **0.503** |
| **NAR + burden-residualized transcriptome** | **0.469** |

Residualizing the viability axis out **improves** performance (0.503 > 0.494). Residualizing
out viability *and* burden still leaves **0.469**. The residual is **not thin** — it is
nearly the entire baseline.

The original reasoning treated two **correlated** predictors as if one's success implied the
other's redundancy. It does not: burden alone reaches 0.443, the burden-*free* residual
reaches 0.469, and they overlap without either being reducible to the other.

**Corrected reading: exp02 is evidence FOR the specificity hypothesis (§5.2), on a
DepMap-anchored target — not a headwind against it.**

**Effect on the thresholds: none.** The ≥0.10 CONSEQUENTIAL margin **stands**, but its
justification changes. It is no longer "large enough to escape a thin residual" (a claim
built on the misreading). It is now simply *a margin over an `F_net`-only baseline large
enough to be biologically meaningful and not attributable to overfitting*. The number is
unchanged; the reasoning under it is repaired.

**Why this correction does not violate the freeze:** it makes no threshold more permissive,
and it corrects a *factual misreading of internal data*, not a judgment about evidence. Had
the correction loosened a bar, it would have had to be refused.

---

## What this amendment does NOT change

- The **absolute** LARGE bars (0.20 founder-loss / 30% division rate) — unchanged.
- The COMMON thresholds (20% / 5–20% / <5%) — unchanged.
- The CONSEQUENTIAL margins (≥0.10 absolute Spearman, etc.) and the requirement that they be
  tested on endpoints **outside** the trajectory definition — unchanged.
- The **founder-referenced denominator** requirement — unchanged, and reinforced: "fraction of
  observed cells at `T`" remains inadmissible.
