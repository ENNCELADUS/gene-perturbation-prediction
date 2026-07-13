# Predefined Significance Criteria (spec §2.1, §4.4)

**Status: FROZEN 2026-07-13, before any Gate 2 evidence was collected.**
Authored by the review orchestrator, not derived from the evidence being weighed —
deriving a threshold from the evidence it will judge is the circularity §2.1 exists to
prevent. These are falsifiable commitments, and the user may overturn any of them, but
only *before* Gate 2 evidence is read, not after.

---

## Why this document exists

Spec §2.1: *"The words **common**, **large**, and **consequential** must be assigned
minimum biological significance thresholds in the review memo **before evidence is
collected**. They are not left to post-hoc judgement."*

Without this, Gate 2's trichotomy (positive / powered-absence / insufficient) is
unfalsifiable: any effect size can be called "large" once you have seen it.

---

## §4.4 Operational definitions (required before either candidate runs)

### "Comparable net fitness" — the matching tolerance

Net fitness `F_net` is a DepMap/Chronos-style GeneEffect, a **relative growth-rate
effect** (never a death label — spec §12).

> **Definition.** Two perturbations are *matched on net fitness* when
> `|F_net(g1) - F_net(g2)| <= tau`, where **`tau` = 1.0 x the empirical within-screen
> replicate standard deviation of GeneEffect, computed from the actual replicate data of
> the screen in use** — not asserted from memory, not a round number chosen for
> convenience.

Rationale for tying `tau` to replicate SD rather than fixing a constant: a fixed
constant (say 0.1) is either vacuous or impossibly strict depending on a screen's noise
floor, and its choice would silently determine the result. Matching must be on the
**uncertainty**, not on the point estimate — two perturbations whose GeneEffect
confidence intervals overlap heavily are not distinguishable in net fitness regardless of
how far apart their point estimates sit.

**Sensitivity requirement.** Any matched-net-fitness result must be reported at
`tau`, `0.5*tau`, and `2*tau`. A phenomenon that appears only at one tolerance is a
matching artifact, not a finding.

### "Reproducible" — three strengths, three different claims

| Level | Definition | Claim it licenses |
|---|---|---|
| R1 | Holds across technical/biological replicates **within one dataset** | Weakest. Rules out noise, not batch or context. |
| R2 | Holds across **independent datasets** in the same cell line + modality | The **minimum bar for Gate 2 "positive"**. |
| R3 | Holds across **cell lines or perturbation modalities** | Licenses a generality claim. Not required for Gate 2. |

> **Gate 2 requires R2.** An R1-only result routes to *insufficient evidence* (bounded
> pilot), not to *positive*.

### "Early" — numerically, per dataset, relative to `T` and to fate commitment

The spec defines "early" relative to `T` **and to fate commitment** — not to sample
collection convenience.

> **Definition.** A state measurement is *early* iff it is taken at `t` such that
> `t < t_commit`, where `t_commit` is the fate-commitment time established by an
> independent assay for that perturbation class (L0 is charged with finding whether a
> citable, measurable `t_commit` exists at all).
>
> **Operational fallback when `t_commit` is unknown:** `t <= 0.25 * T`, AND at least one
> full cell cycle before `T`. Any dataset failing this is recorded as **`late-state`**
> and **cannot support a prospective (T2) claim** — it can only support T1.

**Standing constraint to test, not assume:** Replogle Perturb-seq is a single, late
timepoint several days post-transduction. Under this definition it is presumptively
`late-state`. Gate 1 must confirm or refute this.

---

## The three Gate 2 words

### 1. COMMON — prevalence threshold

> **`>= 20%` of net-fitness-matched perturbation pairs (matched at `tau`, drawn from
> perturbations with a non-trivial fitness effect) exhibit reproducible (R2) divergence
> in outcome dynamics.**

Rationale for 20%: below roughly this level, net fitness is a *nearly sufficient
statistic* for the outcome trajectory in the population of interest, and a model that
resolves the residual is optimizing a corner of the distribution. Above it, the
decomposition is describing a routine feature of perturbation biology rather than an
exception. The number is a judgment call — it is stated so it can be argued with, which
is the point.

- `>= 20%` → satisfies COMMON.
- `5-20%` → **partial**; routes to bounded pilot, and any resulting program must be
  scoped to the identified sub-population, not to perturbations in general.
- `< 5%` with adequate power → satisfies "powered absence" for COMMON.

### 2. LARGE — effect-size threshold

Divergence must clear BOTH an absolute bar and a noise bar.

> **Absolute:** among perturbations matched on net fitness, the divergence in the
> trajectory estimand must reach **>= 0.20 absolute difference in the fraction of
> founders lost (lineage extinction) over `[t0, t0+T]`**, or an equivalent **>= 30%
> relative difference in division rate**.
>
> **Noise:** and that difference must exceed **3x the replicate standard deviation** of
> the same quantity in the same assay.

Rationale: 0.20 absolute in fraction-lost is roughly the point at which two perturbations
would be described by a biologist as producing visibly different outcomes rather than the
same outcome with jitter. The 3x-replicate-SD bar exists because the absolute bar alone
is meaningless in a noisy assay.

**Denominator warning (spec §3.2):** "fraction of founders" is NOT "fraction of observed
cells at `T`". Observed cells at `T` are survivors of a branching-plus-loss process. Any
effect size computed on fraction-of-observed is **inadmissible** against this threshold
until it is converted to a founder-referenced quantity, or the conversion is shown to be
impossible (in which case the finding is downgraded, not accepted).

### 3. CONSEQUENTIAL — the value question, tested OUTSIDE the trajectory definition

Spec §2.1 is emphatic: recovery, persistence, and extinction are *components of the
trajectory estimand*. Using them to prove the estimand is worth having is circular. The
value test must land on endpoints **not used to construct the trajectory**:

> **A fate-resolved trajectory is CONSEQUENTIAL iff it predicts at least one of the
> following better than net fitness alone, by the stated margin:**

| Downstream endpoint (outside the trajectory definition) | Minimum margin over an `F_net`-only baseline |
|---|---|
| Long-term regrowth after perturbation withdrawal | >= 0.10 absolute Spearman, or >= 2-fold separation in regrowth rate between trajectory classes |
| Durable clonogenic survival | >= 2-fold difference in colony-forming efficiency between matched-`F_net` perturbations |
| Resistance emergence / persister outgrowth | Detectable difference in resistant-population emergence frequency at R2 reproducibility |
| Change in intervention prioritisation | >= 20% turnover in a top-50 ranked target list, with the turnover shown to be *correct* against an independent endpoint — not merely different |

**The last row's trap, stated explicitly:** a ranking that *changes* is not a ranking that
*improves*. Turnover alone is not value. The changed ranking must be validated against an
endpoint that did not participate in producing it.

**Calibration anchor from this repository (an honest headwind).** Experiment 02 found a
generic **response-magnitude scalar** reaches Spearman **0.426** against a **0.494**
full-feature baseline. So a naive "burden" scalar already captures ~86% of the achievable
correlation. Any incremental-information claim (spec §5.2, specificity hypothesis) is
competing for a **thin residual**, and the >= 0.10 absolute Spearman margin above is
calibrated to be meaningfully larger than that residual rather than lost inside it.

---

## Gate 2 verdict rule (applied per candidate)

| Condition | Verdict |
|---|---|
| COMMON **and** LARGE **and** CONSEQUENTIAL all met at R2 | **positive** → proceed |
| Any one criterion has evidence, adequately powered, showing it is **not** met | **powered absence** → stop, *for that candidate* |
| Criteria not addressed by any adequately powered study | **insufficient evidence** → **bounded validation pilot**, NOT a stop |

**Power requirement for a "powered absence" verdict.** A study counts as adequately
powered only if it could have detected the thresholds above at the stated
reproducibility level. A study that reports "no difference" without the resolution to see
a 0.20 absolute difference in founder loss is **insufficient evidence**, not absence.
Spec §4.3: *lack of data does not falsify either candidate.*

---

## What would make me wrong

These thresholds are deliberately falsifiable, and each has a failure mode worth naming:

- **20% prevalence may be far too high** if the divergence is concentrated in a
  biologically coherent and *targetable* subclass (e.g. only among perturbations of a
  specific pathway). A 5% prevalence that is 100% concentrated in one druggable pathway is
  more valuable than a diffuse 25%. This is why `5-20%` routes to a scoped pilot rather
  than a stop.
- **The 0.20 absolute founder-loss bar assumes founder-referenced quantities are
  recoverable.** If Gate 1 finds they are not, this threshold is inoperable as written and
  must be renegotiated *before* Gate 2 reads evidence, not after.
- **The exp02 calibration anchor is from a pseudobulk dependency-prediction task**, not
  from a trajectory task. It is the best available yardstick in this repository, but it is
  an analogy, and it may understate the headroom.
