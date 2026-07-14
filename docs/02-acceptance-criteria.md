# Acceptance Criteria (Frozen)

**Status:** frozen before evidence. **Not revisable to fit a result.**
**What this is:** the bar a result must clear to count as an answer. [`01-blueprint.md`](01-blueprint.md) says what may be *claimed*; this document says what counts as *established*.
**Governs:** the completed Gate 2 verdicts in [`03-literature-review.md`](03-literature-review.md), **and** every future result — the three reanalyses, Study 0, and any bounded pilot ([`04-roadmap.md`](04-roadmap.md)).

A threshold set after seeing the data is not a threshold. Without these numbers fixed in
advance, any prevalence can be called *common* and any effect size can be called *large*
once it is on the screen. That is the only reason this document exists, and it is why it
is frozen.

## Where these bind

Every constraint in Study 0's claim ceiling is sourced here. A result that misses one of
these bars is reported as missing it — the bar does not move.

| This criterion | Grades |
|---|---|
| **Equivalence (TOST) matching** | Study 0's "achieved matching" deliverable |
| **Anchor transportability** | Study 0's "anchor transportability" deliverable; any DepMap-matched claim |
| **LARGE — 0.20 founder-loss** | Study 0's "effect-size range" deliverable |
| **Prevalence power — n ≈ 59** | Why Study 0 cannot return powered absence below 5% |
| **R2 reproducibility** | Why Study 0 is R1 by construction and cannot license a Gate 2 `positive` |
| **CONSEQUENTIAL margins** | Any future claim that the trajectory is worth having |

## Operational definitions

### "Comparable net fitness" — the matching tolerance

Net fitness `F_net` is a DepMap/Chronos-style GeneEffect — a **relative growth-rate
effect**, never a death label ([`01-blueprint.md`](01-blueprint.md) §12).

> **Tolerance.** `tau` = 1.0x the empirical within-screen replicate standard deviation of
> GeneEffect, computed from the actual replicate data of the screen in use — not asserted
> from memory, not a round number chosen for convenience.
>
> **Sensitivity.** Any matched-net-fitness result must be reported at `tau`, `0.5*tau`, and
> `2*tau`. A phenomenon that appears only at one tolerance is a matching artifact, not a
> finding.

**Matching is an equivalence claim, not a null-difference claim.** Treating
`|F_net(g1) - F_net(g2)| <= tau` as sufficient only says "we failed to detect a
difference," and it creates a perverse incentive: **a noisier screen makes `tau` larger, so
matching gets *easier* as measurement gets *worse*.**

> **Rule.** Matching requires a two-one-sided-tests (TOST) equivalence result against a
> predefined equivalence margin `delta`, declared before data collection. `delta` must be
> justified on biological grounds — what difference in net fitness is negligible — **not**
> inherited from the assay's noise floor. The `tau` sensitivity reporting above is a
> robustness check, not the definition of matching.

### Anchor transportability

DepMap Achilles is **Cas9 knockout scored by Chronos**. A CRISPRi study performs a
*different* intervention: different penetrance (hypomorph vs. null), strength, kinetics,
horizon, guide efficacy, and possibly subclone. An external `F_net` is therefore **not** the
net fitness of the intervention actually performed.

> **Rule.** Any study matching on an external `F_net` must **measure its own achieved net
> effect** and report the external-vs-achieved relationship. Until it does:
> - claims are capped at *"incremental information beyond an external DepMap fitness
>   reference"*;
> - the claim *"matched on the net fitness of the same intervention"* is **not available**;
> - **unverified matching is a confound that manufactures false positives** — two genes
>   "matched" on Cas9-KO GeneEffect may be badly mismatched under CRISPRi, and the resulting
>   trajectory divergence would be transport mismatch wearing the costume of biology.

### "Reproducible" — three strengths, three different claims

| Level | Definition | Claim it licenses |
|---|---|---|
| R1 | Holds across technical/biological replicates **within one dataset** | Weakest. Rules out noise, not batch or context. |
| R2 | Holds across **independent datasets** in the same cell line + modality | **Minimum bar for a Gate 2 `positive`.** |
| R3 | Holds across **cell lines or perturbation modalities** | Licenses a generality claim. Not required for Gate 2. |

> **An R1-only result routes to *insufficient evidence* (bounded pilot), never to
> *positive*.**

**R2 is a property of datasets and experimental campaigns, not of papers.** One campaign
cannot establish R2 by itself, but a single publication containing two independently
designed campaigns can. "Independent" means the second does not share the first's failure
modes — not the same cell-culture batch, not the same imaging run, not the same guide-library
preparation. Two runs of one protocol on one thawed vial are one campaign, not two.

### "Early" — relative to `T` and to fate commitment

> **Definition.** A state measurement is *early* iff `t < t_commit`, where `t_commit` is the
> fate-commitment time established by an independent assay for that perturbation class.
>
> **The operational fallback governs, because no `t_commit` exists.** L0 established that no
> citable, off-the-shelf commitment cutoff is available for this context: the classical
> restriction point is contradicted by live-cell evidence (a probabilistic *window*, not a
> point), and death's point of no return is called "poorly defined" by the field's own
> nomenclature committee. Therefore: **`t <= 0.25 * T`, AND at least one full cell cycle
> before `T`.** Anything else is **`late-state`** and **cannot support a prospective (T2)
> claim** — only T1.

**Replogle Perturb-seq is late-state** — a single timepoint several days post-transduction.
Nothing in Gate 1 overturned this. It cannot anchor a T2 claim.

## The three thresholds

### COMMON — prevalence

> **`>= 20%` of net-fitness-matched perturbation pairs (matched at `tau`, drawn from
> perturbations with a non-trivial fitness effect) exhibit reproducible (R2) divergence in
> outcome dynamics.**

| Observed prevalence | Verdict |
|---|---|
| `>= 20%` | Satisfies COMMON |
| `5-20%` | **Partial** — routes to a bounded pilot; any resulting program is scoped to the identified sub-population, not to perturbations in general |
| `< 5%` **with adequate power** | Satisfies "powered absence" for COMMON |

Rationale for 20%: below roughly this level, net fitness is a *nearly sufficient statistic*
for the outcome trajectory in the population of interest; above it, the decomposition
describes a routine feature of perturbation biology rather than an exception. The number is
a judgment call, stated so it can be argued with.

### LARGE — effect size

> **Absolute bar:** among perturbations matched on net fitness, divergence in the trajectory
> estimand must reach **>= 0.20 absolute difference in the fraction of founders lost**
> (lineage extinction) over `[t0, t0+T]`, or an equivalent **>= 30% relative difference in
> division rate.**

**The noise bar is a confidence interval, not a multiple of the SD.** A ratio of an effect to
a *dispersion* ("exceeds 3x the replicate SD") is a signal-to-noise heuristic, not a test:
SD is not SE, and such a ratio carries no defined type-I or type-II error rate, so it cannot
license a claim of detection or of absence.

> **Rule.** Two claims, never conflated:
>
> | Claim | Requirement |
> |---|---|
> | An effect **EXISTS** | 95% CI **excludes zero** |
> | The effect is **LARGE** | 95% CI **lower bound exceeds** the LARGE threshold |
>
> **A point estimate `>= 0.20` with a CI excluding zero is NOT sufficient.** An estimate of
> 0.21 with CI [0.02, 0.40] passes that weaker rule while remaining fully consistent with a
> true effect far below the biological bar — compatibility with a large effect is not
> evidence of one.
>
> A result whose CI excludes zero but whose lower bound sits below 0.20 is reported as
> **"effect established, magnitude not established as LARGE"** — a real, publishable finding,
> and not a LARGE verdict.

**The denominator warning** ([`01-blueprint.md`](01-blueprint.md) §3.2): "fraction of
founders" is **NOT** "fraction of observed cells at `T`." Observed units at `T` are survivors
of a branching-plus-loss process — differential loss before or during capture, and
extinguished lineages that leave no trace, are both invisible to a raw endpoint count.
**Fraction-of-observed effect sizes are inadmissible** against this threshold until converted
to a founder-referenced quantity, or the conversion is shown to be impossible (in which case
the finding is downgraded, not accepted).

### CONSEQUENTIAL — tested outside the trajectory definition

Recovery, persistence, and extinction are *components* of the trajectory estimand; using them
to prove the estimand is worth having would be circular. The value test must land on
endpoints **not used to construct the trajectory**.

> A fate-resolved trajectory is CONSEQUENTIAL iff it beats net fitness alone on at least one
> of these, by the stated margin:

| Downstream endpoint | Minimum margin over an `F_net`-only baseline |
|---|---|
| Long-term regrowth after perturbation withdrawal | `>= 0.10` absolute Spearman, or `>= 2-fold` separation in regrowth rate between trajectory classes |
| Durable clonogenic survival | `>= 2-fold` difference in colony-forming efficiency between matched-`F_net` perturbations |
| Resistance emergence / persister outgrowth | Detectable difference in emergence frequency, at R2 |
| Change in intervention prioritisation | `>= 20%` turnover in a top-50 target list, **with the turnover shown to be correct** against an independent endpoint |

**The trap in the last row:** a ranking that *changes* is not a ranking that *improves*.
Turnover alone is not value; the changed ranking must be validated against an endpoint that
did not participate in producing it.

## Prevalence power

A prevalence threshold without a sample size leaves "adequately powered" undefined.

**Rule.** Any COMMON verdict states its **sampling frame** (random, purposive, or exhaustive)
and its **prevalence confidence interval**. Reference points for a **zero-event** result
(one-sided 95% upper bound, ideal independent sampling):

| n matched pairs, 0 divergent | 95% one-sided UCL on prevalence |
|---:|---:|
| 10 | 25.9% |
| 20 | 13.9% |
| 30 | 9.5% |
| **59** | **5.0% — the smallest n that can certify powered absence** |

**Purposive selection and repeated measurement of the same genes reduce effective sample size
below nominal n.** A study of fewer than ~59 independently sampled zero-event pairs **cannot**
return a powered-absence verdict for COMMON, however clean its data.

## Verdict rule (applied per candidate)

| Condition | Verdict | Routing |
|---|---|---|
| COMMON **and** LARGE **and** CONSEQUENTIAL all met at R2 | **Positive** | Proceed |
| Any one criterion has adequately powered evidence showing it is **not** met | **Powered absence** | Stop, *for that candidate only* |
| Criteria not addressed by any adequately powered study | **Insufficient evidence** | **Bounded pilot — NOT a stop** |

**"Adequately powered" is not a courtesy.** A study counts only if it could have detected the
thresholds above at the stated reproducibility level. A study reporting "no difference"
without the resolution to see a 0.20 absolute difference in founder loss is *insufficient
evidence*, not absence.

> **Absence of evidence is not evidence of absence.** Insufficient evidence routes to a
> bounded pilot, never to a stop ([`01-blueprint.md`](01-blueprint.md) §4.3).

## What would make these thresholds wrong

Each is deliberately falsifiable, with a named failure mode:

1. **20% may be far too high** if divergence is concentrated in a biologically coherent and
   *targetable* subclass. A 5% prevalence that is 100% concentrated in one druggable pathway
   is worth more than a diffuse 25%. This is why `5-20%` routes to a scoped pilot rather than
   a stop.
2. **The 0.20 founder-loss bar assumes founder-referenced quantities are recoverable.** Gate 1
   found the design that recovers them (continuous single-cell imaging to a directly observed
   outcome) exists and is validated — so the threshold is operable. But it is **not** operable
   on any pooled, destructively sequenced dataset, and it must not be quietly evaluated on a
   fraction-of-observed denominator instead.
3. **The exp02 calibration anchor comes from a pseudobulk dependency-prediction task**, not a
   trajectory task ([`results/prior-internal-evidence.md`](results/prior-internal-evidence.md)).
   It is the best yardstick available in this repository, but it is an analogy, and it may
   understate the true headroom.
