# Decision Memo v2 — Cell-Fate Outcome-Dynamics Literature Review

Date: 2026-07-13 (revision 2, after reviewer pass).
Governing spec: `docs/superpowers/specs/2026-07-13-cell-fate-prediction-research-direction-design.md`.
Frozen criteria: `SIGNIFICANCE_CRITERIA.md` + `SIGNIFICANCE_CRITERIA_AMENDMENT.md`.

**Decision unchanged. Justification and next step substantially revised.**

Reviewer verdict adopted verbatim:

> The literature decision is acceptable; the novelty matrix requires expansion; the merged
> pilot and its stop/select logic require major revision.

---

## 0. What v1 got wrong

Four evidence readings were stronger than the sources support, and the errors were
introduced **in synthesis**, not by the gate agents — Gate 2B, for instance, correctly
flagged the Mitocheck denominator problem, and v1 then committed exactly that error one
level up. Recording them so they are not silently repaired:

| v1 claim | Defect | Corrected reading |
|---|---|---|
| Gross 2023 shows "**opposite** decompositions" | Too dramatic. Gemcitabine extends S–G2 **and** kills; it is not purely cytotoxic. Both drugs suppress division. | Similar final cell number arises from **materially different** division/death mixes. The wedge is real; the two arms are not opposites. |
| Mitocheck's ~4% suggests the wedge may be rare **for genes** | **Denominator trap — the one this review was built to catch.** ~4% is the fraction of ~51,766 **RNAi/HeLa constructs** showing reproducibly-timed phenotypes. It is not the prevalence of divergence among **net-fitness-matched gene pairs**. Different numerator, different denominator, different modality. | Mitocheck **raises prior concern** that gene perturbation may differ from drug perturbation. It **cannot** enter the stop logic. |
| Mitocheck's r=0.80 arrest–death coupling shows the processes are "coupled, not free" | Computed **within the ~36-siRNA subset already showing both** arrest and death. Conditioning on both being present, then reporting their correlation, cannot establish general coupling. | A within-selected-subset temporal association. Not evidence that net fitness is near-sufficient for genes. |
| Nano 2023 is "**direct counter-evidence**" to the utility hypothesis | Nano measured **caspase-3 activation kinetics** — one low-dimensional, proximate death-effector readout — not a transcriptome. The authors' own reading is that **other unmeasured cell-state factors** determine outcome. That arguably *supports* the idea that a richer state carries information caspase does not. Also, K562 essential-gene CRISPRi is **not** an "unstressed regime." | Downgraded to a **boundary condition / plausibility warning**: fate information is context-dependent, and a proximate death-pathway readout alone is insufficient. It says nothing about transcriptome-wide early state in K562 CRISPRi. |
| Chronos has an "**unflagged identifiability hole**"; the paper is "silent" on it | Unfair framing. Separating birth from death is **outside Chronos's estimand**. A tool is not defective for failing to identify a quantity it never claimed. | Chronos is **structurally incapable** of identifying separate division and death rates **because that decomposition lies outside its estimand.** This is a limitation of the *readout*, not a defect of the *tool* — which is exactly why an independent decomposition is needed. |

---

## 1. The real question (reframed)

v1's headline novelty claim was **attribute-stacking**: "transcriptome + CRISPR + K562 +
DepMap + full trajectory have never been combined." Five technical conditions co-occurring
is a description of a gap, not a scientific question. It is also a weak claim — it can be
satisfied trivially and defended only by counting attributes.

The real question is:

> **In genetic loss-of-function, is net fitness close to a sufficient statistic — or does it
> frequently conceal reproducible and consequential division / death / recovery dynamics?**

This is biological, falsifiable, and it is what the drug literature has answered *yes* to
and the genetic literature has never asked. Everything downstream (transcriptomic utility,
virtual-cell modeling, target prioritization) is contingent on it.

---

## 2. Revised evidence reading

**Established.** The wedge is demonstrated for **selected drug perturbations** (Gross 2023,
PMC10257663): similar final cell numbers via materially different cell-cycle/death rate
mixes, estimated mechanistically from a reporter rather than inferred from net counts, and
an endpoint-only Bliss model mispredicts their combination. Prevalence and importance
remain **unresolved for genetic loss-of-function**.

**Established.** Net-count readouts cannot decompose division from death. Chronos's state
equation carries a single net exponential rate per state (decomposition outside its
estimand); DIP's authors state the limit explicitly in text; GR's ODE contains a kill term
but the GR *statistic* recovers only the net exponent. **An independent decomposition is
required — this is a Gate 1 measurement fact, not a Gate 2 phenomenon fact.**

**Raises prior concern, decides nothing.** Drugs are *selected* to kill; genetic LOF may act
predominantly through division suppression. Mitocheck is consistent with this but, per §0,
cannot quantify it. This remains **the central open risk** and the thing the next study must
be built to interrogate — but it enters as a *hypothesis to test*, not a finding.

**Boundary condition.** Fate information is context-dependent (Nano 2023), and a proximate
death-effector readout alone is insufficient to predict individual fate.

**RETRACTED: the "thin residual" headwind. It was a misreading of our own data.**

v1 and v2 both claimed exp02 shows "a burden scalar reaches ~86% of the full baseline, so
the specificity hypothesis competes for a thin residual." **This is wrong.** The actual
exp02 table (`docs/experiment/02_replogle_k562_viability_axis_audit.md`):

| Model | Spearman |
|---|---|
| NAR viability score only | 0.244 |
| NAR + burden scalar | 0.443 |
| Best pseudobulk baseline (`delta_all`) | 0.494 |
| **NAR-residualized transcriptome** | **0.503** ← *higher than baseline* |
| **NAR+burden-residualized transcriptome** | **0.469** |

Residualize the generic viability axis out entirely and performance **goes up** (0.503 vs
0.494) — generic viability is not what carries the signal. Residualize out viability **and**
burden and the transcriptome still reaches **0.469**, only 0.025 below the unresidualized
baseline.

The error was treating two correlated predictors as if one's success implied the other's
redundancy. Burden alone reaches 0.443; the **burden-free residual** reaches 0.469. They
overlap, but each carries independent signal. **exp02 is evidence FOR the specificity
hypothesis (§5.2), on a DepMap-anchored (Analysis P) target** — not against it.

**Genuine headwinds that survive.** Chang 2023 (ridge ρ=0.88 ≈ DeepDEP ρ=0.87 on
transcriptome→dependency) and Ahlmann-Eltze 2025 (no deep perturbation model beats a linear
baseline at generating `B` at all). These bear on **model complexity** — deep models don't
beat linear ones here — and on §7's generate-`B` hypothesis. **They do not bear on whether
the transcriptome carries information beyond burden.** That question is now answered
affirmatively, and modestly, by exp02, Jost 2020, Dixit 2016, and Replogle 2022.

---

## 3. Novelty matrix — required expansion

Gate 3's "the generic claim is already taken" is **slightly too strong**. The literature has
substantially covered *early/pre-existing state prospectively associated with later fate*.
But most of that work carries **no independent `F_net` anchor and no burden baseline** — so
**"incremental information beyond an independent net-fitness anchor" remains genuinely
under-addressed.** That, not attribute-stacking, is the defensible novelty surface.

**Missing nearest neighbours (verified this pass, must be added to `gate3/`):**

| Paper | Why it is a nearest neighbour |
|---|---|
| **Jost et al. 2020**, *Nat Biotechnol*, `10.1038/s41587-019-0387-5` — "Titrating gene expression using libraries of systematically attenuated CRISPR guide RNAs" | K562 **CRISPRi**, graded **perturbation strength**, Perturb-seq **and** growth phenotypes in one system. This is the closest existing thing to a dose–response axis linking knockdown strength → transcriptome → fitness. Directly relevant to whether transcriptomic change is separable from burden. |
| **Dixit et al. 2016**, *Cell*, `10.1016/j.cell.2016.11.038` — Perturb-Seq | K562 perturbation fitness effects alongside cell-cycle signatures. Prior art for exactly the transcriptome-vs-fitness comparison. |
| **Nadal-Ribelles et al. 2025**, *Nat Commun*, `10.1038/s41467-025-57911-6` — "Transcriptional heterogeneity shapes stress-adaptive responses in yeast" | 3,500 yeast deletions; single-cell **heterogeneity** (not just mean) related to stress fitness. The closest population-level analogue to Candidate B's spirit — and this is the **published** version of the bioRxiv preprint Gate 3 could only reach abstract-only. |

**Also required:** Gate 2's finds were never cross-fed into Gate 3. **Mitocheck**,
**Panagopoulos 2025**, and the **multi-level CRISPR lineage-barcoding** study (BMC Genomics
2019) all belong in the prior-art matrix and are absent from it.

**Norman 2019 / Replogle 2022 — downgraded.** v1 called these "the comparison sitting there
unexploited." They are a **cheap reanalysis opportunity**, and worth taking. But they supply
only **late survivor transcriptomes** and **aggregate growth**. Day 6–8 Perturb-seq profiles
the cells that *survived* — which is the survivorship-selection residual the spec's own §7
warns about. They are **not** prospective fate comparisons, and must not be described as
such.

---

## 3b. FROZEN NOVELTY STATEMENT (post-Gate-3-rerun)

The Gate 3 rerun closed five novelty claims. **Closing them is the point** — each one was
protecting a weaker version of the program.

**CLOSED:**

1. *"No paper combines graded genetic perturbation strength + Perturb-seq + growth phenotype
   in K562."* — **Jost 2020 does exactly this.** Four of the five stacked attributes already
   co-occur in a 2020 paper the first Gate 3 pass missed entirely. Attribute-stacking is
   dead as a novelty claim, confirmed independently a second time.
2. *"No quantitative burden-vs-fitness correlation exists for K562 genetic perturbation."* —
   **Replogle 2022: Spearman ρ = −0.51**, genome-wide, with **771 of 9,608 perturbations
   showing significant transcriptional response but negligible growth effect.** Cite this
   number; stop calling the relationship unquantified. (ρ² ≈ 0.26 — burden leaves ~74% of
   variance unexplained.)
3. *"Distinct transcriptional profiles at matched/same-direction fitness have never been
   shown for genetic perturbation."* — **Dixit 2016**: CABP7 and CIT both *increase* fitness
   via materially different cell-cycle programs. Small (n=13) but a genuine genetic-system
   existence proof of the wedge's qualitative form.
4. *"Population state DISTRIBUTION (variance, not mean) has never been linked to fitness in a
   genetic-perturbation system."* — **Nadal-Ribelles 2025** does it in yeast (**325** YKOC
   mutants — *not* 3,500; that figure was wrong and is corrected here). Candidate B's
   distributional framing is **anticipated in yeast**. The mammalian/CRISPR/DepMap-anchored
   version remains open.
5. *"Whether the transcriptome beats a burden scalar against an independent fitness anchor is
   untested."* — **our own exp02 already answers it, affirmatively** (see §2).

**What survives — the frozen novelty statement:**

> Every fitness readout in this entire literature — Jost's `γ`, Dixit's sgRNA fold-change,
> Replogle's `γ`, Nadal-Ribelles's competition fitness, DepMap's Chronos GeneEffect — **is a
> single net rate.** Every dataset is one or two snapshots of a **survivor** population.
>
> **No one has asked whether transcriptomic state predicts *how* a given net fitness was
> reached** — division suppressed, cells lost, or both partially and recovered — **because no
> existing dataset, public or internal, captures that decomposition at all.**

That is the claim. It is narrower than v1's attribute-stack and narrower than Gate 3's
first-pass "specific conjunction," and unlike both it survives contact with Jost 2020.

**Note the shape of what happened.** The *specificity* half of the program (does state beat
burden?) is now substantially **answered — affirmatively, modestly** — by four independent
sources. It is no longer the open question. The **trajectory-decomposition** half is
untouched by every paper and every reanalysis found. The program's centre of gravity has
moved, and the spec should move with it.

---

## 3c. REVISED SEQUENCING: reanalysis before Study 0

The Gate 3 rerun surfaced a cheaper first step than Study 0, and §9's permitted-work table
already allows it (*"predefined small-scale descriptive reanalysis"*).

**Three reanalysis extensions, analyst-days on already-public data, no wet lab:**

1. Apply exp02's residualization audit to **Jost 2020's titration series** — does the
   burden-residualized transcriptome still predict a gene's growth phenotype *within its own
   dose series*? A cleaner within-gene test than exp02's cross-gene one.
2. Apply it to **Dixit 2016's 13-gene cell-cycle panel** — does a residualized model separate
   CABP7's distinct route from CIT/PTGER2's shared route better than burden alone?
3. Generalize **Nadal-Ribelles's mean-vs-variance test to Replogle 2022's raw single-cell
   K562 data** — the first "distribution beyond mean" test on **mammalian genetic LOF**, on
   public data. This is the cheapest possible probe of Candidate B's core distributional
   claim.

**What the reanalysis class CANNOT do — by construction, not by effort:**

- **Survivorship.** Every dataset profiles cells that *already survived* to day 5–8. A cell
  whose state predicted early death is structurally absent. Not fixable by analysis.
- **No trajectory label exists to predict.** Every available fitness readout is a single net
  rate. No residualization, however clever, manufactures a division/death/recovery label the
  assay never captured.
- Therefore: **the reanalysis can address specificity (Gate 3). It categorically cannot
  address the phenomenon question (Gate 2) or the trajectory question (§1).**

**Sequencing, with the reason stated rather than assumed:** run the reanalysis first because
it costs analyst-days against data already on disk, and its result changes Study 0's scope
and budget ask. A robust "beyond burden, beyond an independent anchor" result across Jost +
Dixit + Replogle **raises** the expected value of paying for trajectory infrastructure. A null
result would justify narrowing the specificity claim **before** any wet-lab spend. It is a
prerequisite, **not a substitute** — Study 0's claim ceiling (§7) is unchanged either way.

---

## 4. The merged pilot was wrong. Three defects.

### 4.1 It does not measure Candidate A's independent variable

Q2 asks whether an **early transcriptomic clone state** predicts a sibling lineage's future.
The v1 pilot contained pedigree imaging, FUCCI, caspase/Annexin, and regrowth — and **no
molecular linkage arm at all.** It therefore cannot answer Q2, cannot test transcriptomic
utility, and **cannot select between A and B**. Claiming "one experiment, both gates" was a
straightforward logical error.

### 4.2 Q3 is not a prerequisite for Candidate A

v1 asserted both candidates depend on Q3. False. Even if different genes show similar
population dynamics at matched fitness, **the same gene** may still show lineage
heterogeneity, predictable survivor/extinction trajectories, and clone-level priming.

- Q3 negative → **stops or narrows Candidate B.** It does **not** falsify Candidate A.
- Q3 positive → does **not** prove early transcriptome carries predictive value.

Correct framing: **Q3 is the cheapest de-risking question for the between-perturbation
population program — not a logical prerequisite for every lineage-level question.**

### 4.3 10–20 pairs cannot support the stop rule (verified)

With **zero divergent pairs observed**, the one-sided 95% upper bound on prevalence is:

| n (matched pairs, 0 events) | 95% one-sided UCL |
|---|---|
| 10 | 25.9% |
| **20** | **13.9%** |
| 30 | 9.5% |
| **59** | **5.0%** — smallest n clearing the bar |

So a 20-pair null cannot exclude a 14% prevalence, let alone certify <5%. Roughly **59
zero-event pairs** are needed under *ideal independent sampling* — and repeated measurement
of the same genes plus purposive (non-random) pair selection further **reduce effective
sample size** below the nominal n.

Worse: **the frozen criteria require R2 (independent datasets) for a Gate 2 positive.** A
single pilot is **R1 by construction.** So a pilot can reach *neither* a Gate 2 positive
*nor* a powered absence, no matter how it is scoped.

**And "effect > 3× replicate SD" is not a power criterion.** SD is not SE. A real design
needs a predefined sampling frame, an equivalence margin, confidence intervals, and a
prevalence power analysis. This is a defect in the frozen criteria, amended separately.

---

## 5. The DepMap anchor claim must be downgraded

v1: *"a new K562 experiment conditioned on the existing DepMap screen is **Analysis P by
construction**."* **Incorrect.**

The pilot proposes **CRISPRi**. DepMap Achilles is **Cas9 knockout**, scored by Chronos.
These differ in perturbation strength, penetrance (hypomorph vs null), kinetics, screening
horizon, guide efficacy, and K562 subclone/culture conditions.

Therefore DepMap supports:

> incremental information **beyond an external DepMap fitness reference**

and **not**:

> matched on the **true net fitness of the same intervention**.

Two consequences:

1. **`|ΔF_net| ≤ 1 replicate SD` is not equivalence.** Failing to detect a difference is not
   demonstrating its absence. Matching requires an **equivalence test against a predefined
   margin** (TOST), not a non-significant difference.
2. **The study must measure its own achieved net effect.** Otherwise apparent trajectory
   divergence may be nothing but **KO→CRISPRi transport mismatch** — two "matched" genes that
   were never actually matched under the intervention actually applied. This confound would
   manufacture exactly the positive result the program wants, which makes it the single most
   dangerous artifact in the design.

---

## 6. Imaging claims were overstated

**Delete "nothing vanishes."** **K562 is a suspension line.** Continuous pedigree imaging
faces out-of-field drift, focus loss, cell overlap, tracking breaks, detachment,
fragmentation, and **state-dependent censoring**. Confinement (microwells, agarose, hydrogel)
is required and is itself a perturbation to be validated.

The honest claim: continuous imaging **preserves a founder denominator and makes attrition
auditable.** It does not guarantee complete tracking.

**FUCCI + caspase/Annexin is not an assumption-free four-state readout.** FUCCI phase ≠
arrest; Annexin positivity can be reversible; caspase-independent death exists; long cycles
and durable arrest need sufficient follow-up; recovery needs predefined criteria; and
**ambiguous disappearance must be logged as its own outcome category**, never silently
assigned to death. It remains the strongest available phenotype route — it is not a complete
identification of the four states.

---

## 7. Revised next step: **Study 0 — a bounded feasibility & calibration study**

Not a phenomenon test. Not a selection study. A study whose **only** purpose is to determine
whether a phenomenon test is justified and, if so, how large it must be.

**Question.** In K562 genetic LOF, does matched-fitness trajectory divergence appear to
exist, and what would it cost to measure it properly?

**What it measures (all of these are the deliverable):**

1. **Anchor transportability** — the achieved CRISPRi net effect per gene vs its DepMap Cas9
   GeneEffect. Quantifies the KO→CRISPRi mismatch *before* it can masquerade as biology.
2. **Achieved matching** — an equivalence test (not a null-difference test) on the observed
   net effects of nominally matched pairs.
3. **Tracking completeness and censoring structure** in suspension K562 — the fraction of
   founders followed to a resolved outcome, and whether loss-to-follow-up is
   state-dependent.
4. **Replicate SD / SE** of every trajectory quantity — the variance inputs that any real
   power calculation needs and that no existing paper reports.
5. **Effect-size range** — is divergence, where seen, anywhere near the 0.20 founder-loss
   bar?

**Claim ceiling — stated up front, in the protocol, before data:**

- ❌ Does **not** test transcriptomic utility (no molecular linkage arm).
- ❌ Does **not** select between Candidate A and Candidate B.
- ❌ Does **not** support powered absence below 5% (needs ~59+ zero-event pairs).
- ❌ Does **not** reach R2 (single dataset; R1 by construction).
- ❌ Does **not** license a Gate 2 `positive`.
- ✅ Estimates feasibility, variance, effect-size range, tracking completeness, anchor
  transportability.

**Exit conditions.** Study 0 does not stop or start the program.

> **Study 0 informs whether Candidate B is affordable and whether Candidate A is technically
> supportable. It does not empirically adjudicate their scientific utility.**

That distinction is load-bearing. Selection between A and B will legitimately turn on
feasibility, cost, and strategic value — and Study 0 speaks to those. But **Study 0's
phenotype data must never be read as a scientific comparison of the two candidates**, because
it contains no molecular linkage arm and therefore cannot see Candidate A's independent
variable at all. A candidate cannot lose a contest it was never entered in.

### What testing Candidate A would actually require

A **true early molecular-state linkage arm** — sibling-split or clone-linked
transcriptomics (the A2 scaffolds: ReSisTrace, SIS-seq, CellTag-multi, STRACK — none of
which has ever used a CRISPR-KO library). That is a **different experiment at a different
scale, cost, and power analysis.** It cannot be bolted onto Study 0 and must not be
described as if it could.

---

## 8. Live-seq — corrected

**An existence proof, not a near-ready screening platform.** Its 85–89% post-biopsy
viability and small DE-gene count establish that **minimally perturbative transcriptomic
measurement is feasible**. But the paper does **not exclude small cell-cycle delays**, and at
4–5 extractions/hour with ~300 high-quality transcriptomes in the entire study, it is a
proof-of-concept under limited conditions — **not a platform for a fate study across tens of
genes.** §4.1's "A2 is the realistic ceiling" should be amended to note A1 exists and is
throughput-bound, without implying it is deployable.

---

## 9. Decision (deliverable 7)

**Both candidates: `narrow-or-pivot`. No production modeling. No unit selection yet.**

**Next step:** Study 0 — a bounded feasibility & calibration study on the Q3 / Candidate B
axis, with the claim ceiling in §7 written into the protocol before any data is collected.

### The standing risk, named

> A genuine literature gap has been identified. The temptation is now to over-interpret a
> small feasibility pilot as an experiment that simultaneously proves the phenomenon,
> selects the primary unit, validates transcriptomic utility, and demonstrates powered
> absence. **It can do none of those.** v1 of this memo already made that mistake once.

### Spec amendments

1. **§1** — "mathematically true and biologically unproven" → *demonstrated for selected drug
   perturbations (Gross 2023); prevalence and importance unresolved for genetic
   loss-of-function.*
2. **§2** — Chronos is **structurally incapable** of separating division from death **because
   that decomposition is outside its estimand** — a readout limitation, not a tool defect.
3. **§4.1** — A1 exists (Live-seq) but is an **existence proof, throughput-bound, not a
   platform**; A2 remains the ceiling for anything pooled. Live-seq's biopsy is a small but
   non-zero perturbation and does not exclude cell-cycle delay.
4. **§4 Candidate B** — add falsifier: *genetic LOF may act predominantly through division
   suppression, making the wedge substantially drug-specific.* Enters as a hypothesis to
   test, **not** as a finding.
5. **§5.1** — DepMap supports *"incremental beyond an external DepMap reference,"* **not**
   *"matched on the true net fitness of the same intervention."* CRISPRi ≠ Cas9 KO.
6. **§5.2** — add boundary condition: fate information is **context-dependent**, and a
   proximate death-effector readout alone is insufficient (Nano 2023).
7. **§7** — generate-`B` is materially weaker: Ahlmann-Eltze 2025 shows no deep model beats a
   linear baseline at generating `B` in the first place.
8. **§12** — add claim boundaries: *do not cite Live-seq as "non-destructive" without the
   85–89% viability caveat; do not present a drug-derived wedge result as evidence for genes;
   do not treat `|ΔF_net| ≤ 1 SD` as fitness equivalence; do not describe Norman/Replogle as
   prospective fate comparisons.*
