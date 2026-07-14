# Decision and Roadmap

**Status:** decision recorded 2026-07-13 (Decision Memo revision 2, after a reviewer pass). Active.
**Decision:** both candidates `narrow-or-pivot`. No production modeling. No unit selection yet.
**Graded against:** [`docs/02-significance-criteria.md`](02-significance-criteria.md). Findings: [`docs/03-review-findings.md`](03-review-findings.md). Contract: [`docs/01-research-direction.md`](01-research-direction.md).
**Source:** [`ideaspark_run/cell-fate-outcome-dynamics/DECISION_MEMO.md`](../ideaspark_run/cell-fate-outcome-dynamics/DECISION_MEMO.md)

---

## 1. The decision

Both candidates: **`narrow-or-pivot`**. No production modeling. No unit selection yet. Next step: Study 0 (§5, Step 2 below).

---

## 2. The real question (reframed)

The v1 headline novelty claim was attribute-stacking — "transcriptome + CRISPR + K562 + DepMap + full trajectory have never been combined." Five technical conditions co-occurring is a description of a gap, not a scientific question, and it is a weak claim: satisfiable trivially and defensible only by counting attributes. Attribute-stacking is **rejected**.

The real question:

> **In genetic loss-of-function, is net fitness close to a sufficient statistic — or does it frequently conceal reproducible and consequential division / death / recovery dynamics?**

This is biological, falsifiable, and it is what the drug literature has answered *yes* to and the genetic literature has never asked. Everything downstream — transcriptomic utility, virtual-cell modeling, target prioritization — is contingent on it.

---

## 3. What v1 got wrong

Four evidence readings in the first version of the decision memo were stronger than the sources support, and the errors were introduced **in synthesis**, not by the gate agents — Gate 2B, for instance, correctly flagged the Mitocheck denominator problem, and v1 then committed exactly that error one level up. Reproduced here because it is what makes the memo trustworthy:

| v1 claim | Defect | Corrected reading |
|---|---|---|
| Gross 2023 shows "**opposite** decompositions" | Too dramatic. Gemcitabine extends S–G2 **and** kills; it is not purely cytotoxic. Both drugs suppress division. | Similar final cell number arises from **materially different** division/death mixes. The wedge is real; the two arms are not opposites. |
| Mitocheck's ~4% suggests the wedge may be rare **for genes** | **Denominator trap — the one this review was built to catch.** ~4% is the fraction of ~51,766 **RNAi/HeLa constructs** showing reproducibly-timed phenotypes. It is not the prevalence of divergence among **net-fitness-matched gene pairs**. Different numerator, different denominator, different modality. This error was **committed in synthesis** even though Gate 2B correctly flagged it one level down. | Mitocheck **raises prior concern** that gene perturbation may differ from drug perturbation. It **cannot** enter the stop logic. |
| Mitocheck's r=0.80 arrest–death coupling shows the processes are "coupled, not free" | Computed **within the ~36-siRNA subset already showing both** arrest and death. Conditioning on both being present, then reporting their correlation, cannot establish general coupling. | A within-selected-subset temporal association. Not evidence that net fitness is near-sufficient for genes. |
| Nano 2023 is "**direct counter-evidence**" to the utility hypothesis | Nano measured **caspase-3 activation kinetics** — one low-dimensional, proximate death-effector readout — not a transcriptome. The authors' own reading is that **other unmeasured cell-state factors** determine outcome, which arguably *supports* the idea that a richer state carries information caspase does not. Also, K562 essential-gene CRISPRi is **not** an "unstressed regime." | Downgraded from **direct counter-evidence** to a **boundary condition / plausibility warning**: fate information is context-dependent, and a proximate death-pathway readout alone is insufficient. It says nothing about transcriptome-wide early state in K562 CRISPRi. |
| Chronos has an "**unflagged identifiability hole**"; the paper is "silent" on it | Unfair framing. Separating birth from death is **outside Chronos's estimand**. A tool is not defective for failing to identify a quantity it never claimed. | Chronos is **structurally incapable** of identifying separate division and death rates **because that decomposition lies outside its estimand.** This is a limitation of the *readout*, not a defect of the *tool* — which is exactly why an independent decomposition is needed. |

---

## 4. The frozen novelty statement

### (a) CLOSED — five claims, each of which was protecting a weaker version of the program

1. *"No paper combines graded genetic perturbation strength + Perturb-seq + growth phenotype in K562."* — **Jost 2020 does exactly this; attribute-stacking is dead as a novelty claim.**
2. *"No quantitative burden-vs-fitness correlation exists for K562 genetic perturbation."* — **Replogle 2022 reports Spearman ρ = −0.51**, with **771 of 9,608** perturbations showing a significant transcriptional response but negligible growth effect.
3. *"Distinct transcriptional profiles at matched fitness have never been shown for genetic perturbation."* — **Dixit 2016** (CABP7 vs CIT — both increase fitness via materially different cell-cycle programs).
4. *"Population state distribution (variance, not mean) has never been linked to fitness in a genetic system."* — **Nadal-Ribelles 2025**, yeast, **325 mutants** (note explicitly: *not 3,500; that figure appeared in v1 and is corrected here*).
5. *"Whether the transcriptome beats a burden scalar against an independent fitness anchor is untested."* — **our own exp02 already answers it, affirmatively** (see [`results/prior-internal-evidence.md`](results/prior-internal-evidence.md)).

### (b) WHAT SURVIVES

> Every fitness readout in this entire literature — Jost's `γ`, Dixit's sgRNA fold-change, Replogle's `γ`, Nadal-Ribelles's competition fitness, DepMap's Chronos GeneEffect — **is a single net rate.** Every dataset is one or two snapshots of a **survivor** population.
>
> **No one has asked whether transcriptomic state predicts *how* a given net fitness was reached** — division suppressed, cells lost, or both partially and recovered — **because no existing dataset, public or internal, captures that decomposition at all.**

### (c) The strategic read

The *specificity* half of the program (does state beat burden?) is now **substantially answered — affirmatively, modestly** — by four independent sources (exp02, Jost 2020, Dixit 2016, Replogle 2022), and is no longer the open question. The **trajectory-decomposition** half is untouched by every paper and every reanalysis found. The program's centre of gravity has moved.

---

## 5. Roadmap

### Step 1 — three reanalysis extensions

Analyst-days on already-public data, no wet lab (from `gate3_rerun_expanded.md` §5 and `DECISION_MEMO.md` §3c):

1. Apply exp02's residualization audit to **Jost 2020's titration series** — a within-gene test, cleaner than exp02's cross-gene one.
2. Apply it to **Dixit 2016's 13-gene cell-cycle panel**.
3. Generalize **Nadal-Ribelles's mean-vs-variance test to Replogle 2022's raw single-cell K562 data** — the first "distribution beyond mean" test on **mammalian genetic loss-of-function**.

**The categorical limit, stated explicitly:** this class of work **cannot** address survivorship (the structurally absent cells), has **no time course**, and has **no trajectory label to predict**. Every dataset in scope (Jost day 5, Dixit day 7/14, Replogle day 6–8) profiles cells that already survived to that timepoint, and every available fitness readout is a single net rate — no residualization, however clever, manufactures a division/death/recovery label the underlying assay never captured. Therefore *the reanalysis can address specificity (Gate 3). It categorically cannot address the phenomenon question (Gate 2) or the trajectory question.*

**Sequencing rationale: reanalysis first, Study 0 second.** The reanalysis costs analyst-days against data already on disk, and its result changes Study 0's scope and budget ask. A robust "beyond burden, beyond an independent anchor" result across Jost + Dixit + Replogle raises the expected value of paying for trajectory infrastructure; a null result would justify narrowing the specificity claim before any wet-lab spend. It is a prerequisite, **not a substitute** — Study 0's claim ceiling is unchanged either way.

### Step 2 — Study 0, a bounded feasibility & calibration study

**Not a phenomenon test. Not a selection study.** Its only purpose is to determine whether a phenomenon test is justified and, if so, how large it must be.

**Question.** In K562 genetic LOF, does matched-fitness trajectory divergence appear to exist, and what would it cost to measure properly?

**Five deliverables:**

1. **Anchor transportability** — achieved CRISPRi net effect per gene vs its DepMap Cas9 GeneEffect, quantifying the KO→CRISPRi mismatch *before it can masquerade as biology*.
2. **Achieved matching** via an equivalence test, not a null-difference test.
3. **Tracking completeness and censoring structure** in suspension K562 — fraction of founders followed to a resolved outcome, and whether loss-to-follow-up is state-dependent.
4. **Replicate SD/SE of every trajectory quantity** — the variance inputs any real power calculation needs and that no existing paper reports.
5. **Effect-size range** — is divergence, where seen, anywhere near the 0.20 founder-loss bar?

**Claim ceiling — stated in the protocol, before data:**

- ❌ Does **not** test transcriptomic utility (no molecular linkage arm).
- ❌ Does **not** select between Candidate A and Candidate B.
- ❌ Does **not** support powered absence below 5% (needs ~59+ zero-event pairs).
- ❌ Does **not** reach R2 (single dataset; R1 by construction).
- ❌ Does **not** license a Gate 2 `positive`.
- ✅ Estimates feasibility, variance, effect-size range, tracking completeness, anchor transportability.

> **Study 0 informs whether Candidate B is affordable and whether Candidate A is technically supportable. It does not empirically adjudicate their scientific utility.**

**Study 0's phenotype data must never be read as a scientific comparison of the two candidates**, because it has no molecular linkage arm and therefore cannot see Candidate A's independent variable at all — **a candidate cannot lose a contest it was never entered in.**

### Step 3 (unscheduled) — testing Candidate A

Requires a true early molecular-state linkage arm (the A2 scaffolds: ReSisTrace, SIS-seq, CellTag-multi, STRACK — **none of which has ever used a CRISPR-KO library**). A different experiment at a different scale, cost, and power analysis. **It cannot be bolted onto Study 0 and must not be described as if it could.**

---

## 6. The standing risk, named

> A genuine literature gap has been identified. The temptation is now to over-interpret a small feasibility pilot as an experiment that simultaneously proves the phenomenon, selects the primary unit, validates transcriptomic utility, and demonstrates powered absence. **It can do none of those.** v1 of this memo already made that mistake once.

Two corrections that must not be lost:

(a) **The DepMap anchor claim is downgraded — CRISPRi ≠ Cas9 KO.** DepMap Achilles is Cas9 knockout scored by Chronos; a CRISPRi study performs a different intervention (penetrance, strength, kinetics, horizon, guide efficacy, subclone). DepMap therefore supports "incremental information beyond an external DepMap fitness reference," not "matched on the true net fitness of the same intervention."

(b) **Imaging claims were overstated — "nothing vanishes" is deleted; K562 is a suspension line.** Continuous pedigree imaging faces out-of-field drift, focus loss, cell overlap, tracking breaks, detachment, fragmentation, and state-dependent censoring; confinement (microwells, agarose, hydrogel) is required and is itself a perturbation to validate. The honest claim: continuous imaging **preserves a founder denominator and makes attrition auditable**; it does not guarantee complete tracking. FUCCI + caspase/Annexin is **not** an assumption-free four-state readout — FUCCI phase ≠ arrest, Annexin positivity can be reversible, caspase-independent death exists — and **ambiguous disappearance must be logged as its own outcome category, never silently assigned to death.**
