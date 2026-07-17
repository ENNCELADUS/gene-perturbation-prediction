# Decision and Roadmap

**Status:** RETIRED-PROGRAM DOCUMENT — the cell-fate outcome-dynamics roadmap, not the active plan. The active direction is the virtual-cell SL composition program ([`01-blueprint.md`](01-blueprint.md)); the new experiment roadmap is **pending** (after the related-work review). Retained as prior evidence; not deleted.
**Graded against:** [`docs/02-acceptance-criteria.md`](02-acceptance-criteria.md) · **Evidence:** [`docs/03-literature-review.md`](03-literature-review.md) · **Contract:** [`docs/01-blueprint.md`](01-blueprint.md)

## 1. The decision

Both candidates: **`narrow-or-pivot`**.

The literature funnel established that the phenomenon is real in drug perturbation, that
no dataset or design currently supports a prospective (Analysis P) claim in genetic
loss-of-function, and that the specificity half of the program is already substantially
answered. What remains open is the trajectory half — and no existing assay captures it.

That does not license a stop (absence of data is not falsification, §8 of the blueprint's
locked decisions), and it does not license production modeling. It licenses **two bounded
steps, in order**.

## 2. The question this program is now answering

> **In genetic loss-of-function, is net fitness close to a sufficient statistic — or does
> it frequently conceal reproducible and consequential division / death / recovery
> dynamics?**

Biological, falsifiable, and the thing the drug literature has answered *yes* to while the
genetic literature has never asked. Everything downstream — transcriptomic utility,
virtual-cell modeling, target prioritization — is contingent on it.

**Attribute-stacking is not a novelty claim.** "Transcriptome + CRISPR + K562 + DepMap +
full trajectory have never been combined" is a description of a gap, not a scientific
question: satisfiable trivially and defensible only by counting attributes. The surviving
novelty statement is in [`03-literature-review.md`](03-literature-review.md) §7.

## 3. Step 1 — three reanalysis extensions

Analyst-days against data already on disk. No wet lab.

| # | Reanalysis | Why this one |
|---|---|---|
| 1 | exp02's residualization audit applied to **Jost 2020's titration series** | A **within-gene** test — cleaner than exp02's cross-gene one |
| 2 | The same audit on **Dixit 2016's 13-gene cell-cycle panel** | Two same-direction-fitness perturbations, materially different transcriptional routes |
| 3 | **Nadal-Ribelles's mean-vs-variance test generalized to Replogle 2022 raw single-cell K562** | The first "distribution beyond mean" test on **mammalian genetic loss-of-function** |

All three reuse the residualization machinery in `src/dependency_baseline/`
(`NuisanceResidualizer`, plus the burden / program-score / NAR feature sets).

**The categorical limit.** This class of work **cannot** address survivorship (the
structurally absent cells), has **no time course**, and has **no trajectory label to
predict**. Every dataset in scope (Jost day 5, Dixit day 7/14, Replogle day 6–8) profiles
cells that already survived to that timepoint, and every available fitness readout is a
single net rate. No residualization, however clever, manufactures a division/death/recovery
label the underlying assay never captured.

> **The reanalysis can address specificity (Gate 3). It categorically cannot address the
> phenomenon question (Gate 2) or the trajectory question.**

**Why reanalysis first.** It costs analyst-days, and its result changes Study 0's scope and
budget ask. A robust "beyond burden, beyond an independent anchor" result across Jost +
Dixit + Replogle raises the expected value of paying for trajectory infrastructure; a null
result narrows the specificity claim before any wet-lab spend. It is a prerequisite, **not
a substitute** — Study 0's claim ceiling is unchanged either way.

## 4. Step 2 — Study 0, a bounded feasibility and calibration study

**Not a phenomenon test. Not a selection study.** Its only purpose is to determine whether
a phenomenon test is justified and, if so, how large it must be.

**Question.** In K562 genetic loss-of-function, does matched-fitness trajectory divergence
appear to exist, and what would it cost to measure it properly?

**What it delivers:**

1. **Anchor transportability** — achieved CRISPRi net effect per gene vs. its DepMap Cas9
   GeneEffect, quantifying the KO→CRISPRi mismatch *before it can masquerade as biology*.
2. **Achieved matching** — via an equivalence test (TOST), not a null-difference test.
3. **Tracking completeness and censoring structure** in suspension K562 — the fraction of
   founders followed to a resolved outcome, and whether loss-to-follow-up is
   state-dependent.
4. **Replicate SD/SE of every trajectory quantity** — the variance inputs any real power
   calculation needs, and that no existing paper reports.
5. **Effect-size range** — is divergence, where seen, anywhere near the 0.20 founder-loss
   bar?

**Claim ceiling — written into the protocol before any data is collected:**

- ❌ Does **not** test transcriptomic utility (there is no molecular linkage arm).
- ❌ Does **not** select between Candidate A and Candidate B.
- ❌ Does **not** support powered absence below 5% (that needs ~59+ zero-event pairs).
- ❌ Does **not** reach R2 (single dataset; R1 by construction).
- ❌ Does **not** license a Gate 2 `positive`.
- ✅ Estimates feasibility, variance, effect-size range, tracking completeness, and anchor
  transportability.

> **Study 0 informs whether Candidate B is affordable and whether Candidate A is
> technically supportable. It does not empirically adjudicate their scientific utility.**

Its phenotype data must never be read as a scientific comparison of the two candidates: it
has no molecular linkage arm and therefore cannot see Candidate A's independent variable at
all. **A candidate cannot lose a contest it was never entered in.**

## 5. Step 3 (unscheduled) — testing Candidate A

Requires a true early molecular-state linkage arm — an A2 scaffold (ReSisTrace, SIS-seq,
CellTag-multi, STRACK), **none of which has ever been run on a CRISPR-knockout library.**
That is the most actionable methodological opportunity the review found, and it is a
different experiment at a different scale, cost, and power analysis. **It cannot be bolted
onto Study 0 and must not be described as if it could.**

## 6. Design constraints carried into any wet-lab work

**DepMap is an external reference, not a matched anchor.** DepMap Achilles is Cas9 knockout
scored by Chronos; a CRISPRi study performs a different intervention (penetrance, strength,
kinetics, horizon, guide efficacy, subclone). It supports *"incremental information beyond
an external DepMap fitness reference"* — never *"matched on the true net fitness of the
same intervention."* Unverified matching is a confound that would manufacture exactly the
positive result this program wants, which makes it the single most dangerous artifact in
the design.

**K562 is a suspension line, so imaging does not come free.** Continuous pedigree imaging
faces out-of-field drift, focus loss, cell overlap, tracking breaks, detachment,
fragmentation, and state-dependent censoring. Confinement (microwells, agarose, hydrogel)
is required and is itself a perturbation to validate. The honest claim: continuous imaging
**preserves a founder denominator and makes attrition auditable** — it does not guarantee
complete tracking.

**FUCCI + caspase/Annexin is not an assumption-free four-state readout.** FUCCI phase ≠
arrest; Annexin positivity can be reversible; caspase-independent death exists; long cycles
and durable arrest need sufficient follow-up; recovery needs predefined criteria. And
**ambiguous disappearance must be logged as its own outcome category, never silently
assigned to death.** It remains the strongest available phenotype route — it is not a
complete identification of the four states.

## 7. The standing risk

> A genuine literature gap has been identified. The temptation is to over-interpret a small
> feasibility pilot as an experiment that simultaneously proves the phenomenon, selects the
> primary unit, validates transcriptomic utility, and demonstrates powered absence.
> **It can do none of those.**
