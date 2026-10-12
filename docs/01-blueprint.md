# Research Blueprint — Context-Specific Synthetic Lethality Prediction Based on AI Virtual Cell Foundation Model

Updated 2026-10-12. This note is the project's introduction, related work, task formulation and research plan, written as
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
lines, and a pair's label never differs between the lines it was tested in, but only by construction: the table's
unanimity filters drop the 8,866 screen pairs whose evidence is positive in some lines and negative in others.
Multi-line combinatorial screens, which test one pair library in every line of a panel, show labels that do differ
(§5).

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

- **Stage one** uses one fixed split of 226 cell lines with basal single-cell profiles: 171 training lines
  (169 with DepMap GeneEffect), 27 validation and 28 test lines. Validation and test lines come only from this
  split. The single-cell training lines are its 171 training lines and 208 further DepMap lines with
  GeneEffect outside it, whose basal single cells come from public deposits: 379 in all, 377 labelled. Other
  DepMap lines join the training side through their bulk RNA-seq only. A line outside the split that shares a
  patient with a validation or test line is not used.
- **Stage two** uses one fixed split of the cell lines that carry SL labels. The splits nest: SL training lines
  are stage-one training lines, SL validation lines are stage-one validation lines, and SL test lines are
  stage-one test lines. Stage one may hold further lines on each side. Validation and test lines carry basal
  single-cell profiles. The headline model trains on the SL lines that have them; a complementary model adds
  SL training lines read through their DepMap bulk RNA, as stage one's bulk-only lines are, and the two are reported
  side by side, never pooled.

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

**Task A.** Context accuracy is the headline: for each gene, the correlation across held-out lines between
predicted and measured GeneEffect, taken as residuals over the training gene mean, on genes whose dependency
varies between lines. Whole-matrix accuracy (Pearson, Spearman and RMSE over all (line, gene) entries) is
reported beside it but decides nothing, because a context-blind gene mean already supplies almost all of it:
whole-matrix Pearson 0.91 and Spearman 0.71 on our validation lines, within 0.01 of our best model, and the
same holds for Cilantro-SL's reported stage-one fit. Context accuracy is read against the gene mean (the
floor), simple context predictors (a ridge on context principal components) and the linear expression prior,
which is the bar a model must clear.

**Task B.** For each held-out line and screen, the headline is AUPRC against the line's positive rate, with
precision and recall at top $K$ and calibration beside it. Ranking within each query $(c,A)$ is secondary and
scored only for genes with at least five screened partners: the screens are paralog libraries, where 43% of genes
have one screened partner. Each score is compared with controls that a real context effect must beat:

- gene identity and pan-essentiality: rank $B$ by its average dependency;
- $A$-blind: rank $B$ by stage one's predicted dependency in $c$, ignoring $A$;
- pair identity: rank by the pair's labels in training lines, ignoring $c$;
- context-ablated: the same model with the cell line's profile removed.

Pairs whose label already appears in a training line are reported separately from pairs never labelled in
training. Known context-specific pairs (SMARCA4–SMARCA2, ARID1A–ARID1B, MTAP–PRMT5, VPS4A–VPS4B,
ENO1–ENO2, STAG2–STAG1) are reported by name where screened; the benchmark of §5 screens only SMARCA4–SMARCA2
and ARID1A–ARID1B.

## 4. Clinical interpretation and limitations

- A held-out cancer cell line is a practical proxy for an unseen context. A patient's tumour adds further
  shifts: genotype, intra-tumour heterogeneity, microenvironment and treatment history.
- Strong predicted GeneEffect for $B$ is not evidence of SL between $A$ and $B$. Definition-1 reasoning yields
  candidates; an SL claim needs an experimental, context-specific measurement against matching single-knockout
  controls, and attention to selectivity over normal cells.
- SL-database labels are aggregated screen calls. In the assembled data a pair's label never differs between
  lines (its filters remove the pairs whose evidence disagrees, and some aggregated rows record one label for every
  line of a panel), so that table cannot test recovery of a pair that is SL in one line and not in another; the
  multi-line combinatorial screens of §5 can. Negatives are screened non-hits in a named line only, and calls agree
  poorly across studies even for the same line, so labels are compared within one screen.
- Experimental SL labels exist for few lines with basal single-cell data, so stage two has few validation and
  test lines.


## 5. Research plan

**Goal.** Predict GeneEffect in unseen cancer cell lines from their basal profile, beyond what a context-blind
gene mean and a linear expression prior already give, and establish whether an AI-virtual-cell (single-cell
foundation-model) representation adds to that. Once a model clears the bar, carry its predicted dependencies
into SL prediction in unseen lines. A clean negative on the AI-virtual-cell question is a result, not a failed
plan.

**Task 1: cross-line GeneEffect prediction and interpretability.**

- *Objective:* predict gene-specific dependency in held-out lines, especially each line's deviation from the
  gene's mean over lines.
- *Evaluation:* context accuracy on held-out lines (§3.7). The bar is the default linear prior of the GeneEffect
  protocol (§10 there, selective Spearman 0.230 validation / 0.232 test on the current split), cleared by a paired
  line-bootstrap interval above zero on validation; the gene mean is the floor.
- *Interpretability:* why the same gene's dependency differs between lines: the features, pathways and cell
  states behind context-specific vulnerabilities, starting from what the winning model exposes (the prior's
  data-selected genes carry its gain). Every proposed explanation is set against lineage, the gene's own
  expression, its copy number and its paralogs' expression on the same residual; one that lineage alone carries
  is not a finding about context.

**Task 2: does single-cell resolution add to pseudo-bulk?**

- *Objective:* establish whether a line's basal single cells carry dependency information that its bulk or
  pseudo-bulk average does not. Held-out lines have basal single cells only: perturbation responses exist for
  four anchor lines, so a held-out line's response is a model's prediction, not a measurement.
- *Approach, in layers:* first, per-gene single-cell statistics (detected fraction and variance across cells)
  stacked linearly on the default prior, against a mean-expression arm built the same way (GeneEffect protocol
  §13); only where that finds signal, distribution-level models of cellular heterogeneity, cell-state
  proportions and perturbation-sensitive subpopulations, with AI-virtual-cell representations, built from the
  statistic that carries it.
- *Evaluation:* single-cell and pseudo-bulk arms under identical conditions: the same lines, genes, split,
  selector and stacking on the same prior; the pseudo-bulk arm is the default prior itself, not a weaker
  pseudo-bulk model.

**Order.** The single-cell statistics stack; then, only on its signal, distributional and AI-virtual-cell
models judged against the default prior; then interpretation of whichever model wins. On 2026-10-11 the 226 split
was rebuilt in place with A549 moved to test (the SL benchmark below needs it). The default prior was rerun on it on
2026-10-12 (0.230 validation / 0.232 test; with the single-cell extras as training lines 0.229 / 0.230, a difference
whose interval spans zero); the single-cell statistics stack is still to be rerun, and the earlier results stay as
records of the 172/27/27 split. Evidence so far: the
joint STATE model trailed the context ridge, and in the single-cell correction every trained epoch lowered
validation and STATE absent beat STATE frozen or trainable (GeneEffect protocol §7, §12).

**Future extension: SL prediction in unseen lines.** Stage one's predicted, line-specific dependencies feed an
SL framework evaluated on held-out lines, on cell-line-specific benchmarks: multi-line combinatorial screens that
test one pair library in every line of a panel and publish a per-line score. The benchmark (chosen 2026-10-11) is
the screen set of Kebabci et al. 2026: Ito et al. 2021 (11 lines), Klingbeil et al. 2024 (22) and Harle et al. 2025
(27). Of the published SL benchmarks with a cross-cell-line setting (literature review §1) it overlaps this
project's DepMap lines most, and it is the only one with lines among the GeneEffect held-out lines; every line of
Feng et al.'s, SLWise's and MiT4SL's settings is a GeneEffect training line. Labels follow Kebabci's rule for Ito
and Klingbeil (GEMINI in the screen's top 10% and LFC below −0.51 and −1.03 respectively) and Harle et al.'s own
call for Harle, whose Kebabci labels were not released; every row keeps its continuous score. The split stays
nested (§3.6); A549, which all three screened, moved from training to test when the 226 split was rebuilt: validation
PA-TU-8988S (Ito) and NCI-H1299 (Klingbeil, Harle), test MDA-MB-231 (Klingbeil), SK-MES-1 (Harle) and A549 (all
three). Stage two trains on the three screens pooled, with the screen as an input: the headline training side is
the 29 screened training lines with basal single cells and GeneEffect, and the complementary model adds 13 more
read through bulk RNA (§3.6). Evaluation stays within one screen: per held-out line and screen, the pairs whose
call differs between the screen's lines are scored separately from the rest, against the pair's hit rate in the
same screen's training lines (the pair-identity control) and pan-essentiality (§3.7); a model that gives a pair one
score in every line, as Cilantro-SL's classifier is trained to, cannot rank them. MiT4SL's own A549 labels are scored
as a secondary external test beside its published A549 result (AUROC 0.673), which reads only the line's bulk RNA;
Kebabci's published scores read the held-out line's measured GeneEffect and are no comparator. The built
benchmark, its counts and its departures from Kebabci's own labels are in its data card
(`docs/data/sl-screen-benchmark.md`). Composed
single-gene predictions yield definition-1 candidates; they do not estimate a double-knockout phenotype.
