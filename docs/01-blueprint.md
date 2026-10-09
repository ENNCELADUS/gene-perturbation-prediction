# Research Blueprint — Context-Specific Synthetic Lethality Prediction Based on AI Virtual Cell Foundation Model

Updated 2026-10-09. This note is the project's introduction, related work and task formulation, written as
for an ML paper. Methods, experiment rules and results are kept in the repository: the GeneEffect protocol
(`docs/03-geneeffect-protocol.md`), the SL protocol (`docs/04-sl-ranking-protocol.md`), the extended related
work with full citations (`docs/02-literature-review.md`) and the result records under `results/`.

## 1. Introduction

Synthetic lethality (SL) is the most direct route from a tumour's genetic defect to a drug target: when a
tumour has lost gene $A$, inhibiting a partner gene $B$ kills the tumour cells while sparing normal cells that
still carry $A$. PARP inhibitors in BRCA1/2-deficient cancers are the clinical proof of the idea. The
clinically motivated question is: *given the molecular state of a tumour and a loss-of-function alteration
it already carries, which additional gene inhibition is selectively lethal?*

Which pairs are lethal depends on the cell. Lineage, expression programmes, and the paralogs and pathway
partners a cell happens to express decide whether losing $B$ is tolerated once $A$ is gone, so a pair that is
lethal in one cell line can be harmless in another. Measured SL is scarce: experimental SL labels exist for
a few dozen cell lines, and a new tumour or cell line arrives with molecular profiles and no screen. DepMap
measures single-gene dependency (GeneEffect) far more broadly, about 1,200 cancer cell lines by 18,500 genes in
release 26Q1, but it too is a fixed panel. A context it never screened can only be served by a model that maps
the context's basal state to its dependencies.

Such a model should read the context at single-cell resolution. The basal single-cell transcriptome is the
profile most likely to be available for a new context, and it keeps the cell-to-cell heterogeneity that a bulk
average discards. Single-cell foundation models, the AI virtual cell models pretrained on large single-cell
atlases and perturbation data, offer a learned representation of cell state to build on.

We therefore pose SL prediction **across cell lines**: train on a set of cell lines, then predict for a cell
line excluded from all supervised training and model selection, seen only through its basal profile. The
task has two stages. Stage one predicts single-gene GeneEffect in the unseen line; stage two predicts which
gene pairs are synthetic lethal there, reasoning from stage one's predicted dependencies.

## 2. Related work and how this work differs

- **Cell-line-specific SL prediction.** EXP2SL, a graph-based cell-specific model and MiT4SL predict SL in a
  named cell line; the latter two address transfer to lines with few or no labels, and the graph-based model's
  authors report uneven transfer, poorest for EXP2SL. They represent a line by
  expression signatures, cell-specific graphs or line-tailored interaction networks. None reads a held-out
  line through its basal single-cell transcriptome with a single-cell foundation model.
- **SL from DepMap dependencies.** Recursive Feature Machines predict knockout viability
  from bulk expression and mutation features, then derive SL candidates from feature importance. That is the
  closest relative of our two-stage, definition-1 logic, without a foundation model or single-cell input, and
  with candidates checked against known pairs rather than per held-out line.
- **Dependency prediction.** DeepDEP predicts dependencies from bulk omics, so stage one on
  its own is not new; what is new is predicting it from basal single cells for lines held out from training.
- **Foundation-model SL.** Cilantro-sl applies Geneformer to bulk expression, simulates
  knockouts in silico, supervises a viability embedding on DepMap and classifies pairs. It is evaluated on
  unseen pairs and unseen genes, never on unseen cell lines.
- **Evaluation.** Benchmarks of SL prediction show that scores depend on the holdout setting; a pair- or
  gene-holdout score says nothing about held-out cell lines. Cross-cell-line settings already exist: Feng et
  al.'s benchmark trains on one line and tests on another, MiT4SL holds out one line of six, and SLWise and
  ELISL transfer between lines or cancer types, all with weak transfer. Held-out-line SL evaluation is
  therefore not new on its own; what this work adds is the basal single-cell context and holding the line out
  of the dependency stage as well.

To our knowledge, no published SL predictor reads a held-out cell line through its basal single-cell
transcriptome with a single-cell foundation model and is evaluated on cell lines excluded from all training.

## 3. Task formulation

### 3.1 Synthetic lethality: two operational definitions

**Definition 1: context-dependent SL, inferred from single-knockout dependency.** A gene pair $(A,B)$ is a
candidate SL interaction when knockout of gene $B$ selectively impairs cell viability in $A$-deficient
contexts compared with $A$-proficient contexts. It is commonly inferred by comparing gene dependencies across
genetically characterised cancer cell lines.

**Definition 2: SL as a genetic interaction, measured by combinatorial double knockout.** A gene pair $(A,B)$
is SL when each single knockout is viable but the double knockout is lethal. Quantitatively it is an extreme
negative genetic interaction: the observed double-knockout fitness is far below that expected from the two
single-knockout effects.

The two are operational approaches to the same biological phenomenon: the first infers candidate SL
relationships from conditional single-gene dependencies, the second measures pairwise interactions directly.

**How this project uses them.** Definition 1 is the working definition the method reasons with, which keeps
its predictions explainable: $B$ scores as an SL partner of $A$ in context $c$ when $B$'s predicted dependency in
$c$ is selectively stronger than where $A$ is intact. Ground truth is a separate matter. An **SL label** is an
experimental entry in an SL database for a named cell line, whichever approach produced it (in practice mostly
combinatorial screens), and the assay behind each label is recorded. A dependency pattern derived from DepMap
is never called an SL label.

### 3.2 The cell line as input

The unit of prediction is a cell line (context) $c$, which the model sees only through its basal, unperturbed
molecular profile $x_c$. The basal single-cell transcriptome is the core of $x_c$; other baseline data of the
line, such as bulk expression, mutation and copy number, may join it where available. No GeneEffect,
perturbation-response or SL measurement of a held-out line is ever an input. When $x_c$ includes data beyond
the single cells, what the single-cell foundation model contributes is established by ablation, not by the
architecture.

### 3.3 Task A — cross-line GeneEffect prediction (stage one)

- **Input:** the basal profile $x_c$ of a cell line and a candidate knockout gene $g$.
- **Output:** predicted GeneEffect $\hat{y}(c,g)$, continuous; more negative means a stronger dependency.
- **Supervision:** observed DepMap (Chronos) GeneEffect $y(c,g)$ of the training lines.

Task A does single-gene dependency prediction only. GeneEffect is a population-level fitness label, not a
single-cell death label, and a single-gene prediction is not SL evidence.

### 3.4 Task B — context-specific SL prediction (stage two)

- **Input:** the basal profile $x_c$, a gene $A$ lost in $c$ (by the line's own defect, or by the knockout of a
  screen), and a candidate gene $B$.
- **Output:** an SL score $s(c,A,B)$ and a ranked list of candidate partners $B$ for each query $(c,A)$.
- **Reasoning:** definition 1, through stage one's predicted dependencies.
- **Ground truth:** experimental SL-database labels in named cell lines. A positive is an experimental SL hit in
  $c$; a negative is a pair screened in $c$ that did not score, which is not evidence of non-SL anywhere else.
  Untested pairs are unlabelled, never negatives. Database pairs are unordered, so each pair serves both query
  directions.

### 3.5 Why two stages

GeneEffect prediction comes first because DepMap provides large-scale, cell-line-specific labels for
single-gene knockouts, whereas labelled SL pairs with defined cellular contexts, suitable for training a model
directly, are scarce. In the SL data assembled for this project, experimental labels cover a few dozen cell
lines, and a pair's label never differs between the lines it was tested in.

GeneEffect supervision lets the virtual-cell backbone learn how cellular context shapes gene dependency, and
that knowledge transfers to SL prediction. GeneEffect alone does not decide whether two genes are synthetic
lethal, so the second stage still needs SL-specific supervision. Definition 1 is the bridge: stage two reads
stage one's predicted dependencies and asks whether $B$'s dependency is selective for contexts that have lost
$A$.

GeneEffect prediction also has value of its own: it predicts which gene knockouts reduce fitness in a given
cancer context, and those predictions help screen context-specific SL candidates.

### 3.6 Splits and generalization settings

Generalization is measured over held-out cell lines. A held-out line is excluded from supervised training
and model selection in both stages; models are selected on validation lines and scored once on test lines.

- **Stage one** uses one fixed split of the cell lines that have basal single-cell profiles and DepMap
  GeneEffect. DepMap lines without single-cell data may join the training side only.
- **Stage two** uses one fixed split of the cell lines that carry SL labels and basal single-cell profiles. The
  splits nest: SL training lines are stage-one training lines, SL validation lines are stage-one validation
  lines, and SL test lines are stage-one test lines. Stage one may hold further lines on each side.

| Test context | Test genes or pairs | Setting | Scope |
| --- | --- | --- | --- |
| Unseen line | Seen genes | Context generalization | **Primary** |
| Unseen line | Unseen genes | Joint context and gene generalization | Reported separately |
| Seen line | Seen genes, new pairs | Pair generalization | Not claimed |
| Seen line | Unseen genes | Gene generalization | Not claimed |

Unseen genes are withheld from every GeneEffect and SL training label. Results on seen and unseen genes are
never pooled. In the primary setting a pair may be labelled in a training line and again in a test line; the
question is whether the model predicts it in the new line, and a pair-identity control (§3.7) measures how much
of a score memorising pairs alone would earn.

### 3.7 Evaluation

**Task A.** Two axes, both headline:

- whole-matrix accuracy: Pearson, Spearman and RMSE over all (line, gene) entries;
- context accuracy: for each gene, the correlation across held-out lines between predicted and measured
  GeneEffect, taken as residuals over the training gene mean, on genes whose dependency varies between lines.

Each axis is read against a context-blind gene-mean baseline, which already reaches whole-matrix Pearson 0.91
on our benchmark, and against simple context predictors (a ridge on context principal components, a linear
expression prior).

**Task B.** For each held-out line: AUPRC against the line's positive rate, precision and recall at top $K$,
ranking within each query $(c,A)$, and calibration. Each score is compared with controls that a real context
effect must beat:

- gene identity and pan-essentiality: rank $B$ by its average dependency;
- $A$-blind: rank $B$ by stage one's predicted dependency in $c$, ignoring $A$;
- pair identity: rank by the pair's labels in training lines, ignoring $c$;
- context-ablated: the same model with the cell line's profile removed.

Pairs whose label already appears in a training line are reported separately from pairs never labelled in
training. Known context-specific pairs (SMARCA4–SMARCA2, ARID1A–ARID1B, MTAP–PRMT5, VPS4A–VPS4B,
ENO1–ENO2, STAG2–STAG1) are reported by name.

## 4. Clinical interpretation and limitations

- A held-out cancer cell line is a practical proxy for an unseen context. A patient's tumour adds further
  shifts: genotype, intra-tumour heterogeneity, microenvironment and treatment history.
- Strong predicted GeneEffect for $B$ is not evidence of SL between $A$ and $B$. Definition-1 reasoning yields
  candidates; an SL claim needs an experimental, context-specific measurement against matching single-knockout
  controls, and attention to selectivity over normal cells.
- SL-database labels are aggregated screen calls. In the assembled data a pair's label never differs between
  lines, and some multi-line screens record identical labels for every line of their panel, so the benchmark
  cannot test recovery of a pair that is SL in one line and not in another. Negatives are screened non-hits in
  a named line only.
- Experimental SL labels exist for few lines with basal single-cell data, so stage two has few validation and
  test lines.

