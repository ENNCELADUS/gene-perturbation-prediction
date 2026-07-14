# Significance Criteria (Frozen)

**Status:** frozen 2026-07-13 before any Gate 2 evidence was collected; Amendment 1 (2026-07-13, post-reviewer) folded in. Every amendment made the bar **stricter**; none was applied to evidence already graded.
**Authority:** these thresholds govern the Gate 2 verdicts in [`docs/03-review-findings.md`](03-review-findings.md). They are not revisable to fit a result.
**Sources:** [`SIGNIFICANCE_CRITERIA.md`](../ideaspark_run/cell-fate-outcome-dynamics/SIGNIFICANCE_CRITERIA.md) + [`SIGNIFICANCE_CRITERIA_AMENDMENT.md`](../ideaspark_run/cell-fate-outcome-dynamics/SIGNIFICANCE_CRITERIA_AMENDMENT.md)

## Why this document exists

Governing spec §2.1: the words **common**, **large**, and **consequential** must be
assigned minimum biological-significance thresholds **before evidence is collected**, not
left to post-hoc judgement — otherwise Gate 2's trichotomy (positive / powered-absence /
insufficient) is unfalsifiable, because any effect size can be called "large" once you
have seen it.

**Merge rule.** The amendment is stricter everywhere it differs from the original freeze,
so the amendment's version wins. Two clauses (matching, and the LARGE noise bar) are
**replaced outright**, not supplemented — the amendment authors verified this is
legitimate because (1) it fixes a statistical construction that was incoherent as
written, not a threshold the authors found inconvenient; (2) every change tightens, never
loosens, the bar; (3) no evidence had yet been graded as passing under the original text
— Gate 2A and Gate 2B both independently returned *insufficient evidence* on every
criterion, so nothing is being retro-fitted.

---

## Correction notice — the exp02 calibration anchor was misread in the raw memo

**The raw `SIGNIFICANCE_CRITERIA.md` file was never edited and still contains, at its
lines ~140–145, a RETRACTED passage** describing exp02 as "an honest headwind": it claims
a generic response-magnitude burden scalar (Spearman 0.426) already captures "~86% of the
achievable correlation" against a 0.494 full-feature baseline, so that any
incremental-information claim for the specificity hypothesis is "competing for a thin
residual."

**This passage is retracted.** Amendment A6 and `DECISION_MEMO.md` §2 identify it as **a
misreading of our own data**, not a judgement call that changed: the inference conflated
two *correlated* predictors' overlap with one being redundant given the other. It is not.
Do not treat the "thin residual" framing — in the raw memo, or anywhere else in this
repository — as live.

**Corrected reading: exp02 is evidence FOR the specificity hypothesis (spec §5.2), on a
DepMap-anchored target — not a headwind against it.** Residualizing the viability axis out
of the pseudobulk transcriptome *improves* prediction of the independent DepMap anchor
relative to the full baseline; residualizing out both viability and response burden still
leaves a Spearman correlation that is most of the unresidualized baseline, not a thin
sliver of it. See [`results/prior-internal-evidence.md`](results/prior-internal-evidence.md)
for the exp02 table.

**Effect on the thresholds below: none.** The ≥0.10 absolute Spearman CONSEQUENTIAL margin
(see below) is unchanged in value. Its justification changes: it is no longer "large
enough to escape a thin residual" (a claim built on the misreading) but simply *a margin
over an `F_net`-only baseline large enough to be biologically meaningful and not
attributable to overfitting*. Correcting a factual misreading of internal data, without
loosening any bar, does not violate the freeze.

---

## §4.4 Operational definitions

### "Comparable net fitness" — the matching tolerance

Net fitness `F_net` is a DepMap/Chronos-style GeneEffect — a **relative growth-rate
effect**, never a death label (spec §12).

> **Tolerance definition (retained).** `tau` = 1.0x the empirical within-screen replicate
> standard deviation of GeneEffect, computed from the actual replicate data of the screen
> in use — not asserted from memory, not a round number chosen for convenience.
>
> **Sensitivity requirement (retained).** Any matched-net-fitness result must be reported
> at `tau`, `0.5*tau`, and `2*tau`. A phenomenon that appears only at one tolerance is a
> matching artifact, not a finding.

**Amendment A2 — REPLACES the original matching rule.** The original text defined
"matched" as `|F_net(g1) - F_net(g2)| <= tau` and treated that as sufficient — i.e., a
**null-difference claim**: "we failed to detect a difference." That is not an equivalence
claim, and it created a perverse incentive the amendment names explicitly: **a noisier
screen makes `tau` larger, so matching got *easier* as measurement got *worse*.**

> **Replacement rule.** Matching is an **equivalence** claim. It requires a
> two-one-sided-tests (TOST) equivalence result against a predefined, biologically
> justified equivalence margin `delta`, declared before data collection. `delta` must be
> justified on biological grounds — what difference in net fitness is negligible — **not**
> inherited from the assay's noise floor. The `tau`/`0.5*tau`/`2*tau` sensitivity
> reporting above is retained as a robustness check, no longer as the definition of
> matching itself.

### Amendment A3 — anchor transportability (NEW)

DepMap Achilles is **Cas9 knockout scored by Chronos**. A CRISPRi study performs a
*different* intervention: different penetrance (hypomorph vs. null), strength, kinetics,
horizon, guide efficacy, and possibly subclone. The original criteria implicitly assumed
external `F_net` is the net fitness of the intervention actually performed — it is not.

> **New requirement.** Any study matching on an external `F_net` must **measure its own
> achieved net effect** and report the external-vs-achieved relationship. Until that is
> done:
> - claims are capped at *"incremental information beyond an external DepMap fitness
>   reference"*;
> - the claim *"matched on the net fitness of the same intervention"* is **not
>   available**;
> - **unverified matching is a confound that manufactures false positives** — two genes
>   "matched" on Cas9-KO GeneEffect may be badly mismatched under CRISPRi, and the
>   resulting trajectory divergence would be transport mismatch wearing the costume of
>   biology.

### "Reproducible" — three strengths, three different claims

| Level | Definition | Claim it licenses |
|---|---|---|
| R1 | Holds across technical/biological replicates **within one dataset** | Weakest. Rules out noise, not batch or context. |
| R2 | Holds across **independent datasets** in the same cell line + modality | **Minimum bar for Gate 2 "positive."** |
| R3 | Holds across **cell lines or perturbation modalities** | Licenses a generality claim. Not required for Gate 2. |

> **Gate 2 requires R2. An R1-only result routes to *insufficient evidence* (bounded
> pilot), not to *positive*.**

**Amendment A5 — clarification.** R2 is a property of **datasets / experimental
campaigns**, not of papers. A single dataset or campaign cannot establish R2 by itself,
but a single publication containing two independently designed experimental campaigns
can. "Independent" means the second campaign does not share the first's failure modes —
not the same cell-culture batch, not the same imaging run, not the same guide-library
preparation; two runs of one protocol on one thawed vial are one campaign, not two.
Symmetrically, "powered absence" requires the A4 sample size (below); between the two
bars sits the honest middle ground most studies occupy: effect-size estimation with a
stated CI.

### "Early" — numerically, relative to `T` and to fate commitment

> **Definition.** A state measurement is *early* iff `t < t_commit`, where `t_commit` is
> the fate-commitment time established by an independent assay for that perturbation
> class (L0 is charged with finding whether a citable, measurable `t_commit` exists at
> all).
>
> **Operational fallback when `t_commit` is unknown:** `t <= 0.25 * T`, AND at least one
> full cell cycle before `T`. Anything else is recorded as **`late-state`** and **cannot
> support a prospective (T2) claim** — it can only support T1.

**Standing constraint to test, not assume:** Replogle Perturb-seq is a single, late
timepoint several days post-transduction. Under this definition it is **presumptively
late-state**; Gate 1 was required to confirm or refute this (see
[`docs/03-review-findings.md`](03-review-findings.md#2-gate-1--measurement-and-observation-validity)).

---

## The three Gate 2 words

### 1. COMMON — prevalence threshold

> **`>= 20%` of net-fitness-matched perturbation pairs (matched at `tau`, drawn from
> perturbations with a non-trivial fitness effect) exhibit reproducible (R2) divergence
> in outcome dynamics.**

- `>= 20%` -> satisfies COMMON.
- `5-20%` -> **partial**; routes to bounded pilot, and any resulting program must be
  scoped to the identified sub-population, not to perturbations in general.
- `< 5%` with adequate power (see Amendment A4 below) -> satisfies "powered absence" for
  COMMON.

Rationale for 20%: below roughly this level, net fitness is a *nearly sufficient
statistic* for the outcome trajectory in the population of interest; above it, the
decomposition describes a routine feature of perturbation biology rather than an
exception. The number is a judgment call, stated so it can be argued with.

### 2. LARGE — effect-size threshold

Divergence must clear an absolute bar, with a statistically disciplined test of whether
it does.

> **Absolute bar:** among perturbations matched on net fitness, the divergence in the
> trajectory estimand must reach **>= 0.20 absolute difference in the fraction of
> founders lost (lineage extinction) over `[t0, t0+T]`**, or an equivalent **>= 30%
> relative difference in division rate.**

**Amendment A1 — REPLACES the original noise bar.** The original text required the
absolute difference to "exceed 3x the replicate standard deviation of the same quantity
in the same assay." **Defect:** SD is not SE. A ratio of an effect to a *dispersion* is a
signal-to-noise heuristic, not a statistical test, and carries no defined type-I or
type-II error rate — it cannot license a claim of detection or of absence.

> **Replacement.** Two different claims, previously conflated, must be distinguished:
>
> | Claim | Requirement |
> |---|---|
> | An effect **EXISTS** | 95% CI **excludes zero** |
> | The effect is biologically **LARGE** | 95% CI **lower bound exceeds** the LARGE threshold (0.20 absolute founder-loss, or 30% relative division-rate) |
>
> **"Point estimate >= 0.20 with a CI excluding zero" is NOT sufficient for LARGE.** An
> estimate of 0.21 with CI [0.02, 0.40] would pass that rule while remaining fully
> consistent with a true effect far below the biological bar — compatibility with a large
> effect is not evidence of one. **The LARGE criterion is met only when the 95% CI lower
> bound exceeds the predefined LARGE threshold**, with estimator and sampling frame
> declared before data collection.
>
> A result whose CI excludes zero but whose lower bound sits below 0.20 is reported as
> **"effect established, magnitude not established as LARGE"** — a real, publishable
> finding, and not a LARGE verdict. The 3x-SD heuristic may still be reported as a
> descriptive signal-to-noise figure; it has **no inferential standing** and may not
> appear in a verdict.

**The denominator warning (spec §3.2), reproduced prominently:** "fraction of founders"
is **NOT** "fraction of observed cells at `T`." Observed units at `T` are survivors of a
branching-plus-loss process — differential loss before or during capture, and
extinguished lineages that leave no trace, are both invisible to a raw endpoint count.
**Fraction-of-observed effect sizes are inadmissible** against this threshold until
converted to a founder-referenced quantity, or the conversion is shown to be impossible
(in which case the finding is downgraded, not accepted). This requirement is unchanged by
the amendment and is reinforced by it.

### 3. CONSEQUENTIAL — the value question, tested OUTSIDE the trajectory definition

Recovery, persistence, and extinction are *components* of the trajectory estimand; using
them to prove the estimand is worth having would be circular. The value test must land on
endpoints **not used to construct the trajectory**:

> A fate-resolved trajectory is CONSEQUENTIAL iff it predicts at least one of the
> following better than net fitness alone, by the stated margin:

| Downstream endpoint (outside the trajectory definition) | Minimum margin over an `F_net`-only baseline |
|---|---|
| Long-term regrowth after perturbation withdrawal | >= 0.10 absolute Spearman, or >= 2-fold separation in regrowth rate between trajectory classes |
| Durable clonogenic survival | >= 2-fold difference in colony-forming efficiency between matched-`F_net` perturbations |
| Resistance emergence / persister outgrowth | Detectable difference in resistant-population emergence frequency at R2 reproducibility |
| Change in intervention prioritisation | >= 20% turnover in a top-50 ranked target list, with the turnover shown to be *correct* against an independent endpoint — not merely different |

**The last row's trap, stated explicitly:** a ranking that *changes* is not a ranking that
*improves*. Turnover alone is not value; the changed ranking must be validated against an
endpoint that did not participate in producing it.

---

## Amendment A4 — prevalence power analysis (NEW)

**Defect the amendment fixes:** the original COMMON criterion specified prevalence
thresholds but no sample size, so "adequately powered" was undefined in practice.

**New requirement.** Any COMMON verdict states its **sampling frame** (how pairs were
drawn — random, purposive, or exhaustive) and its **prevalence confidence interval**.
Reference points for a **zero-event** result (one-sided 95% upper bound, ideal
independent sampling):

| n matched pairs, 0 divergent | 95% one-sided UCL on prevalence |
|---:|---:|
| 10 | 25.9% |
| 20 | 13.9% |
| 30 | 9.5% |
| **59** | **5.0% — smallest n that can certify "powered absence"** |

**Purposive (non-random) pair selection and repeated measurement of the same genes reduce
effective sample size below nominal n.** A study of fewer than ~59 independently sampled
zero-event pairs **cannot** return a powered-absence verdict for COMMON, regardless of how
clean its data is.

---

## Gate 2 verdict rule (applied per candidate)

| Condition | Verdict | Routing |
|---|---|---|
| COMMON **and** LARGE **and** CONSEQUENTIAL all met at R2 | **Positive** | Proceed |
| Any one criterion has evidence, adequately powered (per A4 for COMMON; per the A1 CI rule for LARGE), showing it is **not** met | **Powered absence** | Stop, *for that candidate only* |
| Criteria not addressed by any adequately powered study | **Insufficient evidence** | **Bounded validation pilot, NOT a stop** |

**Power requirement for a "powered absence" verdict.** A study counts as adequately
powered only if it could have detected the thresholds above at the stated reproducibility
level. A study that reports "no difference" without the resolution to see a 0.20 absolute
difference in founder loss is *insufficient evidence*, not absence. Spec §4.3: *lack of
data does not falsify either candidate.* **Insufficient evidence routes to a bounded
validation pilot, never to a stop. Absence of evidence is not evidence of absence.**

---

## What would make me wrong

These thresholds are deliberately falsifiable; each has a named failure mode:

1. **20% prevalence may be far too high** if divergence is concentrated in a biologically
   coherent and *targetable* subclass (e.g. only within one pathway). A 5% prevalence
   that is 100% concentrated in one druggable pathway is more valuable than a diffuse
   25%. This is why `5-20%` routes to a scoped pilot rather than a stop.
2. **The 0.20 absolute founder-loss bar assumes founder-referenced quantities are
   recoverable.** If Gate 1 found they are not, this threshold is inoperable as written
   and must be renegotiated *before* Gate 2 reads evidence, not after.
3. **The exp02 calibration anchor is from a pseudobulk dependency-prediction task**, not
   from a trajectory task. It is the best available yardstick in this repository, but it
   is an analogy and may understate the true headroom — independent of the correction
   above, which fixes how the anchor was *read*, not what task it comes from.
