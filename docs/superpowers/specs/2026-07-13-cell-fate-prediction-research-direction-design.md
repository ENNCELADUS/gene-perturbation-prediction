# Research Direction: Perturbation-Induced Cell Fate Prediction

Status date: 2026-07-13
Type: research direction + literature review plan (not an implementation spec)
Origin: PI meeting notes on transcriptomics -> cellular phenotype / cell death
prediction, plus a brainstorming session over the current repository state.

## 1. Position Relative to Existing Work

This is a **new research program**, separate from the existing NeurIPS draft in
`docs/report/` (the "transcriptional shockwave carries the partner" SL-benchmark
story). That draft is treated as a finished or parallel artifact. Experiments
01-10 remain valid as evidence and as baselines; they are not re-told under a new
frame.

The new program's center of gravity is **cell fate prediction as the scientific
problem in its own right**. Conditional essentiality and synthetic lethality are
downstream applications of it, not competitors to it.

## 2. The Structural Fact That Motivates the Program

| Object | Coverage | Location |
| --- | --- | --- |
| Label: `GeneEffect(cell line, gene)` | Dense: 1,208 x 18,531 | `data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv` |
| Basal transcriptome per cell line | Dense: all DepMap lines | `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` |
| Perturbation response `B(c, g)` | ~1 cell line (K562), ~6,070 genes | Replogle K562 GWPS (cluster) |

Verified on 2026-07-13: the GeneEffect matrix is 1,208 rows x 18,531 columns.
Perturb-seq `.h5ad` files are not on the local disk; coverage figures are taken
from `docs/data/` and `docs/experiment/07_*.md`, not re-verified here.

The consequence: the dependency/conditional-essentiality label matrix is **not
label-limited**. It is limited on exactly one axis - the perturbation response
transcriptome exists for one cell line out of 1,208. This is the honest reason a
forward / virtual-cell model is needed at all: `B` is unobservable for ~99.9% of
the contexts of interest, so it must be generated. That is a
problem-formulation argument, not an architecture preference.

## 3. Problem Statement

Population fitness screens (DepMap CRISPR GeneEffect, PRISM drug viability)
measure **depletion**. Depletion is the sum of two biologically distinct
processes - cells dying, and cells ceasing to divide. No population screen can
separate them. Single-cell transcriptomes can, in principle, because a dying cell
and an arrested cell are transcriptomically dissimilar.

The task:

```text
f(cell state, perturbation)
    -> resolved fate distribution over the perturbed population
       { P(proliferating), P(arrested), P(dying | mechanism m), timing }
```

subject to a **consistency constraint**:

```text
aggregate(fate distribution) == observed population fitness
```

Depletion is the model's derived quantity, not its target.

The aggregation function itself is an open modeling choice, not a given. A
depletion score is a function of both the dying fraction and the growth-rate
reduction of the surviving fraction, so `aggregate` is at minimum two-term. Its
exact form is deferred until L2 reports how existing fitness-scoring models
(e.g. DepMap's own) treat growth dynamics.

### 3.1 Corollary Structure

Every existing task in the repository becomes a readout of the same model:

| Downstream task | Expression in the fate model |
| --- | --- |
| Dependency prediction (`C`) | Aggregate of the fate composition for `(c, g)` |
| Conditional essentiality | The same aggregate evaluated in an unseen context `c` |
| Synthetic lethality | Interaction residual of the aggregate over two perturbations, relative to an explicit single-perturbation null |

This ordering is the design commitment: one model, several readouts.

### 3.2 Novelty Claim (to be checked, not assumed)

The claim under test is that (a) the arrest-vs-death split has been treated as a
confound rather than a modeling target, and (b) no prior model constrains a
single-cell fate decomposition to reproduce population fitness as its aggregate.
Dependency prediction and cell-death classification plausibly exist as separate
literatures, and the **bridge** is the gap. This is a prior, not a finding.
Literature question L2 (Section 5) exists to falsify it, and it should run first.

## 4. Falsifiability Requirements

Three failure modes must be designed against from the start. Two have already
occurred once, in exp09.

### 4.1 Null Model 1: The Two-Way Additive Baseline

Before any fate model, measure how much of `GeneEffect(c, g)` is explained by:

```text
GeneEffect(c, g) ~ gene_mean(g) + line_mean(c)
```

using **no transcriptome at all**. If this explains most held-out-*line* variance,
then every downstream model is relearning pan-essentiality. This is precisely the
PI's stated fear that "the model may only learn cell line identity," and it is
the same structure that exp09's non-pan-essential diagnostic already exposed:
CV3 AUROC fell from 0.645 to 0.583 and AUPR from 0.651 to 0.490 once
broadly-essential genes were removed.

All fate-model results are reported as **lift over this null**. It is built
first.

### 4.2 Null Model 2: Response Burden

Experiment 02 established that a generic response-magnitude scalar recovers
Spearman 0.426 against a 0.494 full-feature baseline. A fate model that beats
only response burden has discovered that perturbed cells look perturbed.
Response burden, cell count, and cell-cycle composition are **reported
covariates in every fate result**, not hidden shortcuts.

### 4.3 Failure Mode: Survivorship Bias in QC

Standard scRNA-seq QC discards high-mitochondrial-fraction, low-UMI, low-gene
cells. That is the definitional transcriptomic signature of a dying cell, and it
matches the biological priors in the meeting notes (decreased transcript
abundance, RNA degradation, stress response). Every public Perturb-seq dataset in
use here has already been filtered this way.

Consequences:

1. A fate model trained on QC-filtered data estimates **fate among survivors**,
   which is a different quantity and must be named as such in all claims.
2. The QC threshold is an **experimental variable to ablate**, not a fixed
   preprocessing step.

### 4.4 Identifiability Caveat

Per-cell death probability is counterfactual: one observes a cell's state, never
the fate that state would have had. Therefore:

- Per-**bag** death fraction is estimable against a viability anchor.
- Per-**cell** death probability may be unidentifiable without a fate reporter,
  lineage barcode, or live-imaging pairing.

Until literature question L1 settles this, the model's honest output is a
**fraction**, and any per-cell probability is an unvalidated latent. This
constrains the output type of the entire program and is not a presentation
detail.

## 5. Literature Review Plan

Eight questions, ordered by the cost of a bad answer. The survey's purpose is to
resolve program-deciding uncertainties, not to build general background.

| Id | Question | Why it matters |
| --- | --- | --- |
| L1 | Is per-cell fate identifiable? What technologies pair a fate readout (reporter, lineage barcode, live imaging) with a transcriptome? | Decides whether the model's output is a per-cell probability or a per-bag fraction. Changes the output type. |
| L2 | Has anyone separated cytostatic from cytotoxic depletion? Does DepMap's own scoring model already account for growth-rate dynamics? Does prior work decompose a fitness score into arrest vs. death? | **The make-or-break novelty check. Run first.** If this is solved, the headline changes. |
| L3 | Has the QC-driven loss of dying cells been quantified? Do methods exist that *model* the dying population instead of filtering it? | A positive finding supplies both a method and a critique. Gates whether the death signal survives the pipeline. |
| L4 | What signatures/classifiers distinguish apoptosis, ferroptosis, and necroptosis from transcriptome? Which datasets carry known-mechanism perturbagens usable as mechanism labels? | Mechanism is supervisable via known-mechanism agents; this finds the label source. |
| L5 | Which resources pair single-cell transcriptomes with dose, time, viability, and multiple cell lines? | Makes the "acquire dose-response data with real viability" decision concrete. Output: a dataset table with the four axes marked present/absent. |
| L6 | Who already predicts dependency / conditional essentiality from omics, and how well? | Establishes the real external bar, not our internal baselines. |
| L7 | What is the adversarial case against virtual-cell forward models? There is a live critique that deep perturbation models barely beat trivial baselines. | **The entire A->B->C chain rests on `B` being worth generating.** If the critique holds, generating `B` is a liability. Better found in survey than in review. |
| L8 | SL from DepMap: co-essentiality/co-dependency correlation, mutation-stratified differential dependency, published statistical SL-inference pipelines. Which double-perturbation genetic-interaction screens exist and at what scale? | Arms the next PI meeting. Determines whether the multi-gene AIVC idea is testable or aspirational. |

## 6. Staging

Two workstreams can begin immediately because they do not depend on survey
outcomes:

1. **Two-way additive null on DepMap.** All data is on local disk. Calibrates
   every claim the program will make. If the null is strong, the framing must
   change - better known now.
2. **QC ablation on one K562 Perturb-seq dataset.** Re-process with relaxed
   mito/UMI thresholds. Test whether a distinct low-count, high-mito,
   stress-marker-high population appears under strong perturbations and is absent
   under controls. Cheap, and it directly tests whether the death signal survives
   the pipeline.

Everything else - the mechanism head, dose-response data acquisition, timing, and
the drug/genetic modality bridge - waits on L1, L3, and L5.

### 6.1 Deferred Decision: The Modality Bridge

Viability data at scale is drug-based; essentiality and SL are genetic. How the
two relate (shared fate head trained on drug and applied to genetic; drug-centric;
genetic-centric with drug as auxiliary; or multi-task) is **explicitly deferred
until L5 returns the dataset table.** It is not decided in this document.

## 7. Position for the Next PI Meeting (SL from DepMap)

The PI's question - "how can we derive synthetic lethal relationships from DepMap
data?" - has already been partially answered in this repository by experiment 09
(`docs/experiment/09_k562_sl_pair_cross_cell_line_selectivity.md`).

The statistical device used was a cross-cell-line selectivity contrast: DepMap is
a single-gene screen, so pair evidence must come from comparing cell lines in
which the anchor gene is defective (composite OR over damaging mutation, hotspot
mutation, copy-number loss, low expression) against lines in which it is intact.

```text
sel(a -> b) = mean[ d_{c,b} | a-intact ] - mean[ d_{c,b} | a-defective ]
```

Result: a consistent classification lift over the dependency-only floor on all
three splits, largest on CV3 (+0.050 AUROC). But on the non-pan-essential slice
the CV3 lift largely disappears (AUROC 0.583, AUPR 0.490). The recorded verdict
is that most of the cold-start lift is attributable to essentiality structure,
not pair-specific co-dependency.

Therefore the question to bring to the PI is **not** "what statistic should we
use." It is:

> How do we construct a null model that removes pan-essentiality, so that what
> remains is genuinely interaction?

This is the same null-model problem the double-perturbation AIVC idea faces.
Synthetic lethality is *defined* as a deviation from the expected combined
effect. Predicting `P(death | KO a + KO b)` is not sufficient on its own; it
requires an explicit single-perturbation null `psi(f(a), f(b))` to subtract:

```text
interaction(a, b) = fate(a, b) - psi(fate(a), fate(b))
```

The null `psi` (additive, multiplicative, or learned) is a modeling decision that
must be made explicitly, and it is the crux of both the DepMap-statistics route
and the multi-gene AIVC route.

## 8. Open Questions

Flagged as genuinely uncertain rather than assumed:

1. **Is arrest vs. death cleanly separable in practice from CRISPRi Perturb-seq?**
   Cell-cycle scoring is standard, so the arrested fraction is plausibly
   measurable. A clean *death* signature in QC-filtered data may not be. This is
   the central empirical risk of the program.
2. **Does usable double-perturbation data exist?** The local `GSE205310` archive
   (`docs/data/jost-replogle-dual-sgrna-k562-crispri.md`) parses every condition
   to a *single* target gene in the repository's own coverage table, which
   suggests dual-guide-per-gene (efficacy) rather than gene-pair genetic
   interaction. This must be checked, not assumed. Norman 2019 is CRISPRa and
   modality-mismatched to knockout dependency per existing data rules.
3. **Does the perturbation-response transcriptome transfer across cell lines at
   all?** All current evidence is K562-only. The generalization ladder the PI
   proposed (within cancer type -> cross cancer type -> cross tissue) has not been
   tested, and failure at the cross-tissue level may reflect biological
   distribution shift rather than model failure - which must be distinguished, not
   conflated.
4. **The "~0.67 Spearman" figure quoted in the meeting needs a caveat.** In this
   repository 0.664 / 0.668 is the scVI128-GMM+Ridge **Adamson external
   transfer**, computed on 85 UPR-biased gene-level rows. Internal Replogle K562
   5-fold CV is approximately 0.49. Any external presentation should lead with
   0.49 and present 0.67 as a small-n transfer result.

## 9. Claim Boundaries

Extending the existing terminology guardrails in `CLAUDE.md`:

- Do not call a predicted fate fraction a measured death rate.
- Do not call DepMap GeneEffect a cell-death label; it is a depletion score that
  conflates death and arrest.
- Say **fate among survivors** whenever the model is trained on QC-filtered data.
- Say **death fraction** (bag-level) unless a per-cell fate readout has been
  secured; do not say **death probability** (cell-level) before then.
- Do not claim synthetic lethality without an explicit single-perturbation null
  and interaction residual.
