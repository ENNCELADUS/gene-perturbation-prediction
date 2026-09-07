# Related Work: Context, Dependency and Synthetic-Lethality Ranking

Updated 2026-09-07. This review supports the
[research task](01-blueprint.md), the [GeneEffect protocol](03-geneeffect-protocol.md)
and the [SL evaluation proposal](04-sl-ranking-protocol.md).
It separates published evidence from this repository's results and proposed
contribution. It is a focused review, not an exhaustive novelty search.

## 1. The comparison must match the generalization axis

Feng et al. benchmark SL methods under pair and gene holdouts, negative-sampling
choices, classification and ranking, and also include context-specific experiments.
The paper shows why performance depends on the evaluation setting. It does not
supply a directly comparable score for this repository's nine-context screen table.
A gene-holdout score cannot establish held-out-cell-line performance, and a
cell-line holdout cannot establish unseen-gene generalization.
[Feng et al., Nature Communications 2024](https://www.nature.com/articles/s41467-024-52900-7).

The current target is ranking experimentally screened pairs within held-out cell
lines. Comparisons must share context assignments, observed pair keys, class priors
and label provenance. Graph-free inputs are a design choice, not evidence of
novelty or biological generalization. The project claims no external leaderboard
position on its custom table.

## 2. Closest SL prior art

Cilantro-sl combines a pretrained single-cell foundation model applied to bulk
expression, in-silico single-gene perturbation embeddings, CRISPR-viability
supervision and a pair classifier with conformal uncertainty. This is direct
precedent for combining foundation-model representations, dependency supervision
and SL classification without an SL graph. Its pair/gene-holdout evaluation does
not substitute for the cell-line-held-out experiment proposed here.
[Hua, Haber and Ma, 2026 preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC13160162/).

The proposed distinction is using a response-supervised expression predictor to
construct dependency features from basal single-cell bags, then testing whether
context-dependent features add value beyond matched context-ablated and
pan-essentiality controls. That is a **proposed experimental contribution**.
There is no completed SL result and no basis for an absolute “first” or “no prior
work” claim. Composing two single-gene embeddings or dependency predictions does
not by itself estimate a genetic interaction.

## 3. Basal transcriptomes and perturbation prediction

| Work | Relevant contribution | Consequence for this project |
| --- | --- | --- |
| [STATE, Arc Institute](https://arcinstitute.org/news/virtual-cell-model-state) | Models changes in cell expression under perturbations across contexts | Response prediction is an intermediate capability whose downstream dependency value must be measured |
| [Tahoe-x1, official repository](https://github.com/tahoebio/tahoe-x1) | Perturbation-trained single-cell foundation model and supplied checkpoints | Frozen embeddings are an input representation; task holdout does not erase pretraining exposure |
| [DeepDEP, Chiu et al. 2021](https://pubmed.ncbi.nlm.nih.gov/34417181/) | Predicts cancer dependencies from integrative genomic profiles | Dependency prediction from molecular context has prior art; response simulation must justify its added cost |

This repository combines frozen Tx1 basal embeddings, trainable STATE, an ESM2
adapter and a residual GeneEffect head. These components do not make single-gene
fitness a direct readout of a simulated double knockout. Expression-distribution
accuracy and dependency accuracy require separate evaluations.

## 4. Simple baselines and metric choice

Ahlmann-Eltze, Huber and Anders found that the evaluated deep perturbation models
did not consistently outperform deliberately simple mean/linear predictors; their
combination experiments also illustrate the importance of an additive baseline.
These findings concern the datasets and metrics studied, not every future model
or the GeneEffect task in this repository.
[Nature Methods 2025](https://www.nature.com/articles/s41592-025-02772-6).

Systema shows how systematic expression variation can dominate evaluation scores
and motivates evaluating perturbation-specific effects. Its revised references
also have limitations, including sensitivity to weak effects and reference choice.
[Viñas Torné et al., online 2025, Nature Biotechnology 2026](https://www.nature.com/articles/s41587-025-02777-8).

The literature is not unanimous about what a baseline win implies. A subsequent
preprint reports stronger deep-model performance under metrics calibrated against
biologically informative controls, challenging conclusions drawn from conventional
MSE and correlations alone. This is a reason to inspect what a metric measures,
not to change the registered selector after observing test outcomes.
[Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on Well-Calibrated Metrics, 2025 preprint](https://www.biorxiv.org/content/10.1101/2025.10.20.683304v1.full).

For the current GeneEffect task, the relevant controls are gene-mean, K562
copy-prior, nearest-line and context-PCA-ridge, with Tx1 and HVG context features.
Pairwise error measures scale accuracy; per-gene correlation across held-out
lines measures context variation. High correlation across genes within a line
can already be supplied by a context-blind gene mean. These axes are defined
explicitly in the [blueprint](01-blueprint.md#6-what-the-geneeffect-metrics-measure).

## 5. Single-gene dependency, SL labels and genetic interaction

GeneEffect measures a single-gene dependency. A screen hit is a pair label under
a particular assay and calling rule. A genetic-interaction estimate additionally
needs a joint phenotype and an explicit expected phenotype under a non-interaction
model. These quantities are not interchangeable.

The proposed pair head has an essentiality-only minimum control, but no joint
outcome. Improvement over that control is incremental label-ranking value, not
a joint-minus-null interaction. Neither expression reconstruction nor inferred
co-dependency supplies measured epistasis.

Dataset interpretation follows the cards: [Horlbeck](data/horlbeck-2018-k562-gi.md)
contains fitness-GI measurements, whereas the
[Jost/Replogle dual-sgRNA resource](data/jost-replogle-dual-sgrna-k562-crispri.md)
measures single-gene knockdown efficacy. Neither is the current SL pair-label input.
The [context-screen table](data/sl-context-screen.md) retains aggregate-label
inference and unidentified-study limitations; it cannot test same-pair label reversal.

## 6. What the current evidence supports

The completed seed-0 joint experiment achieves test Huber 0.01611905 versus
0.01613152 for gene-mean, a relative reduction of **0.0773%**. Residual Pearson is
0.05414 versus 0.12161 for Tx1 PCA-ridge; residual Spearman is 0.05280 versus
0.11574. The same observed test keys were verified across all methods.
[Full results and provenance](results/joint_geneeffect_seed0/README.md).

This shows a functioning composition but little dependency-error improvement over
a context-blind prior and weaker context correlations than simple predictors.
It does not establish an SL-ranking benefit, superiority to published systems,
statistical significance or multi-seed robustness.

The next discriminating experiments are validation-side checks of residual
prediction scale and a matched-batch response-supervision ablation. A later SL
experiment must retain identity, essentiality, simple-context and matched
context-ablated controls. The scientific contribution depends on those results;
it is not guaranteed by architecture or by the absence of an SL graph.
