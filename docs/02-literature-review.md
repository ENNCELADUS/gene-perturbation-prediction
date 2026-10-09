# Related Work: Context, Dependency and Synthetic-Lethality Prediction

Updated 2026-10-09. The extended related work of the [research blueprint](01-blueprint.md), written as the
related-work section of a paper. The blueprint states only how this project differs; this document gives the
prior art behind each statement. It is a focused review, not an exhaustive novelty search.

## 1. Synthetic-lethality benchmarks and their holdout settings

**The standard benchmark holds out pairs and genes.** Feng et al. compare SL predictors on pan-cancer
SynLethDB labels under three cross-validation settings, held-out pairs whose genes are seen in training,
pairs with one unseen gene and pairs with both genes unseen (CV1 to CV3 in the paper), together with
negative-sampling choices, classification and ranking, and show that scores fall as the setting tightens
([Feng et al., Nature Communications 2024](https://doi.org/10.1038/s41467-024-52900-7)). These settings test
generalization to new genes, not to new cell lines. A gene-holdout score cannot establish held-out-cell-line
performance, and a cell-line holdout cannot establish unseen-gene generalization.

**Cross-cell-line settings are explicitly defined.** The same benchmark's supplement defines a
"cross-cell-line scenario" (Supplementary Note 2.5): six methods, among them the context-specific MVGCN-iSL,
are trained on one cell line's labels and tested on another's, Jurkat → A549 and A549 → Jurkat, with
positives called at a genetic-interaction score below −3 and no pair shared between the two lines. AUROC is
0.45–0.57 for Jurkat → A549 and 0.48–0.69 for A549 → Jurkat, and the authors conclude that the task "remains
highly challenging"; in Supplementary Note 2.4, models trained on SynLethDB also perform poorly on K562
combinatorial-screen pairs
([Feng et al., Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-52900-7/MediaObjects/41467_2024_52900_MOESM1_ESM.pdf)).
MiT4SL makes the setting a named scenario of its benchmark: one line out of six (A375, A549, Jurkat, MeWo,
22Rv1, PK1) is held out, every labelled pair of that line is the test set and the other five lines are
training. Its released results give AUROC 0.673 with A549 held out and 0.617 with 22Rv1 held out, and the
split does not remove test pairs that also occur, with another line's label, in training
([Tao et al., bioRxiv 2025](https://doi.org/10.1101/2025.04.20.649694);
[code](https://github.com/JieZheng-ShanghaiTech/MiT4SL)). Earlier, SLWise ran a "cell-line transferable
study" among A549, A375 and HT29, with training and test pairs from different lines; transfer is uneven
across source lines, and EXP2SL transfers poorly
([Pu et al., Computational and Structural Biotechnology Journal 2023](https://doi.org/10.1016/j.csbj.2023.10.011)).
ELISL transfers between cancer types and holds out one cancer type at a time, with performance its authors call
"modest for most pairwise cancer combinations"
([Tepeli, Seale and Gonçalves, Bioinformatics 2024](https://doi.org/10.1093/bioinformatics/btad764)). For
paralog pairs, Kebabci et al. evaluate the same pairs in unseen cell lines and new pairs in unseen lines
([Kebabci et al., bioRxiv 2026](https://doi.org/10.64898/2026.01.19.700065)).

**What these settings leave open.** The held-out line enters each of them differently: SLWise reads
line-specific L1000 knockdown signatures and DepMap gene effects, ELISL cancer-type aggregates of cell-line
and tissue profiles, MiT4SL a protein-interaction network filtered by the line's bulk expression together
with knowledge-graph and protein-sequence representations, and the paralog classifier the held-out line's own
DepMap expression, gene-effect, copy-number and mutation profiles. None reads the held-out line through its
basal single-cell transcriptome, and none also holds the line out of a dependency stage; Cilantro-sl, the
foundation-model SL method closest to the two-stage route here, holds out pairs and genes but never a cell
line (§3). Cross-cell-line SL evaluation is therefore not a contribution of this project. The proposed
contribution is the conjunction: SL ranking in lines excluded from both the dependency and the SL stage, with
the basal single-cell transcriptome as the only context input and pair features from a dependency model built
on single-cell foundation-model or perturbation-model representations, scored against matched
context-ablated, pair-identity and pan-essentiality controls. The architecture alone establishes nothing, and
the absence claim is bounded by its search: Europe PMC, Semantic Scholar, GitHub and bioRxiv abstracts up to
2026-10-09, not Google Scholar, IEEE Xplore, RECOMB or ISMB 2026 proceedings or Chinese-language venues.
MiT4SL's held-out A549 and 22Rv1 are this project's validation and test SL contexts, which makes it a
comparable external reference once label sources are matched.

## 2. Synthetic lethality and how it is measured

Two operational definitions identify the same phenomenon. Under the first, a pair $(A,B)$ is a candidate SL
interaction when knocking out $B$ selectively impairs viability in $A$-deficient contexts compared with
$A$-proficient ones, inferred by comparing single-gene dependencies across genetically characterised cancer
cell lines ([Tang et al., Frontiers in Genetics 2022](https://doi.org/10.3389/fgene.2022.961611);
[Haider et al., Nature Genetics 2025](https://doi.org/10.1038/s41588-025-02108-2)). Under the second, a pair is
SL when each single knockout is viable and the double knockout is lethal, an extreme negative genetic
interaction measured against the fitness expected from the two single knockouts
([O'Neil, Bailey and Hieter, Nature Reviews Genetics 2017](https://doi.org/10.1038/nrg.2017.47);
[Shen and Ideker, Journal of Molecular Biology 2018](https://doi.org/10.1016/j.jmb.2018.06.026)).

The quantities differ. GeneEffect measures a single-gene dependency. A screen hit is a pair label under a
particular assay and calling rule. A genetic-interaction estimate additionally needs a joint phenotype and an
explicit expectation under a non-interaction model. Comparing single-gene dependencies between deficient and
proficient lines yields candidates, not measured epistasis, and composing two single-gene predictions does not
estimate a double-knockout phenotype. The data cards keep the distinction for the resources held here:
[Horlbeck](data/horlbeck-2018-k562-gi.md) contains fitness genetic-interaction measurements, whereas the
[Jost/Replogle dual-sgRNA resource](data/jost-replogle-dual-sgrna-k562-crispri.md) measures single-gene
knockdown efficacy.

Experimental SL labels are scarce and concentrated. Curated databases aggregate screen calls, literature
entries and computational predictions; the experimental part covers a few dozen cell lines, dominated by
combinatorial screens in a handful of them. Reliability differs sharply between literature-curated, predicted
and large-screen entries, and a screened non-hit is a negative only for the line and assay that screened it.

## 3. Computational SL prediction

**Cell-line-specific prediction.** EXP2SL predicts SL in a named cell line from L1000 shRNA expression
signatures ([Wan et al., Frontiers in Pharmacology 2020](https://doi.org/10.3389/fphar.2020.00112)). SLWise and
MiT4SL also take the line as an input and evaluate transfer to lines outside training (§1).

**SL from DepMap dependencies.** Recursive Feature Machines predict CRISPR knockout viability from bulk
expression and mutation features of DepMap lines, then derive SL candidates from feature importance and
validate them by recovery of experimentally verified pairs
([Cai, Radhakrishnan and Uhler, bioRxiv 2023](https://doi.org/10.1101/2023.12.03.569803)). It is the closest
published relative of a two-stage, definition-1 route: dependency prediction first, SL candidates second.

**Foundation-model SL.** Cilantro-sl applies Geneformer to bulk expression of cancer cell lines, simulates
single-gene knockouts in silico, supervises a viability embedding on DepMap CRISPR data and trains a pair
classifier with conformal uncertainty. It is precedent for combining foundation-model representations,
dependency supervision and SL classification; its evaluation holds out pairs and genes, within cell lines,
and never holds out a cell line
([Hua, Haber and Ma, bioRxiv 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13160162/)).

## 4. Dependency prediction from molecular context

Predicting dependencies from a cell's molecular profile has prior art. DeepDEP predicts cancer dependencies
from integrative genomic profiles and transfers them to tumours
([Chiu et al., Science Advances 2021](https://doi.org/10.1126/sciadv.abh1275)); Recursive Feature Machines do
the same from expression and mutations (§3). Stage one is therefore not new as a task; what remains open is
whether a basal single-cell profile, read by a foundation model, predicts the context-dependent part of a
dependency in lines held out from training, beyond what bulk expression and a context-blind gene mean
already give.

## 5. Single-cell foundation models and perturbation prediction

| Work | Relevant contribution | Consequence for this project |
| --- | --- | --- |
| [STATE, Arc Institute](https://arcinstitute.org/news/virtual-cell-model-state) | Models changes in cell expression under perturbations across contexts | Response prediction is an intermediate capability whose downstream dependency value must be measured |
| [Tahoe-x1, official repository](https://github.com/tahoebio/tahoe-x1) | Perturbation-trained single-cell foundation model and supplied checkpoints | Frozen embeddings are an input representation; task holdout does not erase pretraining exposure |

Expression-distribution accuracy and dependency accuracy require separate evaluations, and a predicted
single-gene response is not a simulated double knockout.

Ahlmann-Eltze, Huber and Anders found that the evaluated deep perturbation models did not consistently
outperform deliberately simple mean or linear predictors; their combination experiments also show the
importance of an additive baseline. These findings concern the datasets and metrics studied, not every future
model or task ([Nature Methods 2025](https://www.nature.com/articles/s41592-025-02772-6)).

Systema shows how systematic expression variation can dominate evaluation scores and motivates evaluating
perturbation-specific effects; its revised references have limitations of their own, including sensitivity to
weak effects and reference choice
([Viñas Torné et al., online 2025, Nature Biotechnology 2026](https://www.nature.com/articles/s41587-025-02777-8)).

The literature is not unanimous about what a baseline win implies. A subsequent preprint reports stronger
deep-model performance under metrics calibrated against biologically informative controls, challenging
conclusions drawn from conventional MSE and correlations alone. This is a reason to inspect what a metric
measures, not to change a registered selector after observing test outcomes
([Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on Well-Calibrated Metrics, 2025 preprint](https://www.biorxiv.org/content/10.1101/2025.10.20.683304v1.full)).

The same caution applies to dependencies. Correlation across all (line, gene) entries is dominated by which
genes are essential everywhere, which a context-blind gene mean already supplies; the context-dependent part
is measured per gene across held-out lines, against the gene mean and simple context predictors.
