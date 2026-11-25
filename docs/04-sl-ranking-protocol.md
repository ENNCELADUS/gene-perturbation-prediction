# Experiment Protocol: Held-Out-Cell-Line SL Ranking

Updated 2026-10-23. This is the **separate SL-pair protocol** under
[the research blueprint](01-blueprint.md); its rules are in §8. It builds on the backbone defined by the
[GeneEffect protocol](03-geneeffect-protocol.md) (§4). The first stage-two run, on the screen benchmark with the
stage-one backbone, is §10; §2–§6 and §9 describe the earlier, unbuilt design.

**Superseded in part (2026-10-09, benchmark 2026-10-11).** The [research blueprint](01-blueprint.md#3-task-formulation) now fixes
the SL task differently, and this protocol is rewritten to match before stage two is implemented:

- Definition 1 (B's dependency selectively stronger where A is lost) is the logic the method reasons with;
  the SL query is a cell line, a gene A lost in it, and candidate partners B.
- SL labels are experimental SL-database entries only; a DepMap-derived dependency pattern is never an SL
  label.
- Stage two uses one fixed SL split nested in the GeneEffect split: SL training, validation and test lines are
  GeneEffect training, validation and test lines respectively.
- **Benchmark (2026-10-11):** the screen set of Kebabci et al. 2026 (bioRxiv 10.64898/2026.01.19.700065; code
  github.com/cancergenetics/context_specific_paralog_SL, MIT): Ito 2021 (11 lines × 4,859 pairs), Klingbeil 2024
  (22 × about 2,700) and Harle 2025 (27 × 472, CC BY 4.0). Of the published benchmarks with a cross-cell-line
  setting it overlaps the 226 most (literature review §1); Feng's, SLWise's and MiT4SL's lines are all GeneEffect
  training lines. Labels compare within one screen. Ito and Klingbeil take Kebabci's rule
  (`notebooks/03_feature_annotation/05_annotate_gemini.ipynb`): GEMINI at or above the screen's 90th percentile and
  LFC below −0.51 (Ito) or −1.03 (Klingbeil). Harle takes the authors' Table S5 call, reproduced exactly; Kebabci's
  Harle labels come from a GEMINI file they did not release. A Harle pair with a gene flagged single-depleted in a
  line is unlabelled there, not a negative, since the authors' call cannot make it a hit. Every row keeps its
  continuous score.
- **Split:** the 226 split was rebuilt in place on 2026-10-11, moving A549 (ACH-000681; all three screens) from
  training to test.
  Validation PA-TU-8988S (ACH-000022; Ito) and NCI-H1299 (ACH-000510; Klingbeil, Harle); test MDA-MB-231
  (ACH-000768; Klingbeil), SK-MES-1 (ACH-000665; Harle) and A549. No screened line shares a PatientID with a
  validation or test line; A549's DepMap patient siblings leave the extra lines.
- **Training side:** the three screens pooled, with the screen as an input. The headline model trains on the 29
  screened training lines with basal single cells and GeneEffect (the 226's training lines and the built single-cell
  extras GI-1, RH-30, T3M-4, COR-L23, LK-2, NCI-H1975, KP-4, BxPC-3, IPC-298, MEL-JUSO, NCI-H209, NCI-H211,
  HPAF-II, MOLM-13, THP-1 and Capan-1, `configs/benchmarks/extra_single_cell_lines_26Q1.json`); a complementary
  model adds the 13 screened lines read through bulk RNA from the labelled extras
  (`configs/benchmarks/extra_bulk_lines_26Q1.json`), reported beside it, never pooled. Flister 2025 and Fong 2024 are not adopted.
- **Built (2026-10-11):** the screen tables fetched on the H20, the line map
  `configs/benchmarks/sl_screen_lines.csv`, the builder `src/data/builders/build_sl_screen_benchmark.py` and its
  output `data/sl_screens/benchmark/` on the H20, described in the [SL benchmark card](data/sl-benchmark.html). The
  labels follow Kebabci's rule but are not Kebabci's labels: on Ito's ten lines the rule gives 1,511 SL calls in
  47,315 points against their 958 in 41,244, because their GEMINI was recomputed from counts and their pairs were
  filtered, neither released. Before a first run: the default linear model and its out-of-fold export rerun on the
  rebuilt split, then the SL head. The context screen's table below stays a built artifact: its unanimity filters drop the 8,866
  screen pairs whose evidence differs between lines, and some of its aggregated rows may not be fitness screens (the
  A549;CAPAN1;MEL202;MEWO;OMM23 group reads, from the Harle and Chan papers' text, as a Cas9-negative
  library-abundance control; unverified).
- Evaluation follows blueprint §3.7: per held-out line and screen, AUPRC against the positive rate is the headline,
  with precision and recall at top K and calibration; ranking within each query (c, A) is secondary and scored only
  for genes with at least five screened partners. Controls are pan-essentiality, A-blind, pair-identity and
  context-ablated. Within a screen every held-out pair is also measured in its training lines, so the pair-identity
  control is the pair's hit rate in the same screen's training lines, and the pairs whose call differs between the
  screen's lines are scored separately from the rest. MiT4SL's A549 labels are a secondary external test beside its
  published A549 AUROC (0.673).

The nine-context split and the separate eligibility rules below (§2, §4, §5) predate those decisions.

## 1. Objective and prediction unit

For each held-out cell line $c$, rank canonical unordered gene pairs $(a,b)$ using
its basal single-cell transcriptome and gene features. Output a positive-is-SL
score $s(a,b\mid c)=s(b,a\mid c)$. No measured GeneEffect or SL labels from $c$
may enter the deployable feature path.

The primary question is whether context-dependent predicted single-gene profiles
add SL-ranking value beyond pair/gene identity and pan-essentiality. Pair labels
score this question; the separate GeneEffect benchmark diagnoses the dependency
predictor. Neither outcome estimates a double-knockout interaction phenotype.

## 2. Benchmark and outstanding prerequisites

The [tracked split](../configs/benchmarks/context_screen_v2_split.json) and
[basal registry](../configs/benchmarks/context_screen_v2_basal_registry.json)
are the authorities. The retired data card (`docs/data/sl-context-screen.md` at 2171e54) documents source
filters, label inference and artifact provenance. The benchmark is this project's own
proposal, built from screen-level pair labels in named cell lines; the Feng 2024 SL
benchmark is not used.

| Split | Context | ModelID | Positive | Negative |
| --- | --- | --- | ---: | ---: |
| train | K562 | ACH-000551 | 1,669 | 10,270 |
| train | Jurkat | ACH-000995 | 95 | 9,124 |
| train | OVCAR8 | ACH-000696 | 89 | 643 |
| train | HAP1 | ACH-002475 | 56,993 | 20 |
| train | HT29 | ACH-000552 | 235 | 7,322 |
| validation | A549 | ACH-000681 | 392 | 1,581 |
| test | 22RV1 | ACH-000956 | 38 | 580 |
| test | PC9 | ACH-000779 | 134 | 2,364 |
| test | HeLa | ACH-001086 | 170 | 2,364 |

The published table contains 94,083 rows across nine contexts. Its split removed
128 source-row groups crossing sides (359 rows); no retained source row or
canonical pair crosses sides. `source_row_id` only identifies one raw aggregate
record, not a study, so this does not prove study independence.

Before a first SL run, resolve and document the existing prerequisites:

- Complete per-context filter attribution for evidence-count mismatch and
  non-atomic context rejection; current failure counts overlap and are not additive.
- Verify model-ready basal inputs for each included context. The card records
  HAP1/22RV1 as source-registered only; registration alone does not prove usable caches.
- Fix the candidate-pair universe and report batch/source confounding and class
  counts. A549 validation is degenerate: all positives contain TRA2A and no
  negative does. HAP1 has only 20 negatives; 22RV1 has only 38 positives.
- Record PC9/HeLa as one aggregate-label cluster, while retaining separate scores.
  Their labels are SL-only: neither ModelID has GeneEffect in the pinned release.
- Freeze backbone/head selection, inner context groups, hyperparameters, random
  seeds and the minimum context-effect margin before accessing SL test labels.

These are unresolved experimental choices and data checks, not completed steps.
Do not silently change membership or remove difficult contexts. Because no model
has run, a registry expansion may replace the nine-context split in place; after
the first model run, any membership change requires a new benchmark version with
its own documented comparison surface. RPE1 stays excluded for exact-model basal
mismatch.

## 3. Inputs and supervision

### 3.1 Basal cells

Use the exact registered model's basal cells: raw UMI or documented count inputs,
not an informal-name match or substituted bulk profile. Tx1 reads the counts; every
other expression quantity uses the log expression space of the
[GeneEffect protocol §3.4](03-geneeffect-protocol.md#34-expression-space). Fit transforms on eligible
training contexts and restore them for evaluation. Frozen Tx1 caches retain their
recorded pretraining and preprocessing provenance.

### 3.2 Response anchors

The four response anchors are those of the
[GeneEffect protocol §3.2](03-geneeffect-protocol.md#32-response-anchors): K562, HCT116,
Jurkat and HepG2, with genetic-perturbation responses and non-targeting controls.
Inner-fold exclusion also removes an excluded context's response supervision.

### 3.3 Dependency labels

Use the pinned DepMap 26Q1 GeneEffect release. Fit gene means, variable-gene
membership and normalization on the eligible training cohort of each fit only (§4).

### 3.4 Pair labels

The sole pair-label input is `data/SL_Benchmark_Formal/sl_integrated_pairs.csv`,
processed into `data/SL_Benchmark_Formal/derived/context_screen_v2/` according to
the data card. Preserve natural class counts, canonical pairs, `source_row_id`,
`screen_cluster`, `split` and `label_confidence=silver_inferred`.
Screened non-hits are not universal non-SL labels. Do not import GeneEffect,
co-dependency, Feng or measured-GI labels as additional SL ground truth.

## 4. Backbone requirements

The backbone is the joint GeneEffect model: architecture, training loop and checkpoint
selection are fixed by the [GeneEffect protocol](03-geneeffect-protocol.md) §3.4–§5 and
its [current design](specs/2026-10-02-expression-space-and-all-pipeline-design.md). This protocol
adds requirements that the GeneEffect run does not meet.

The SL experiment must specify its own eligible dependency cohort and validation
surface before fitting (§8); it must not import the 226-line split by convenience. Record, for every inner model,
response loss and both component terms alongside dependency error and correlation;
improving one task is not evidence of improving the other.

The completed seed-0 `best.pt` is diagnostic evidence only: it beats the context-blind
gene mean by 0.0773% Huber and trails the Tx1 context-PCA ridge on residual
correlation (result: `results/joint_geneeffect_seed0/README.md`, removed with the history of `results/`). It neither supplies
the required SL exclusions nor the inner-fold models of §5, and its 226-line test set was
observed once on 2026-09-07 (§8). Inner fits that
exclude a response anchor need an explicit eligible-anchor configuration; the fixed
four-anchor trainer cannot be reused unchanged for those folds.

## 5. Out-of-fold feature generation

Every backbone-derived block of an **SL training** row — the profile, the
target-context values and $\mu_{\text{train}}$ — must come from one model that excluded
that row's context group from response supervision, dependency fitting and fitted
preprocessing, with the gene mean refit without it. Partition the train-side SL
contexts into inner groups, declare them before any run, and fit one model per group
under the finally selected hyperparameters; models from the hyperparameter search
are not reused. K562 and Jurkat are both SL training contexts and response anchors,
so their inner fits run with a reduced anchor set.

The profile reference cohort $\mathcal{C}_{\text{ref}}$ is the GeneEffect-labelled
cohort with **all nine SL benchmark contexts removed**; it is identical for every arm.
Fit the feature standardiser only on complete out-of-fold vectors. Mixing an
in-sample profile block with an out-of-fold context block inside one vector is
prohibited, and there is no non-out-of-fold fallback for the predicted-profile arm.

The dominant cost is materialising $\hat\delta(g,c')$ for every benchmark gene across
every $\mathcal{C}_{\text{ref}}$ context once per inner model, on the order of
genes × contexts × cells-per-bag STATE forward passes over the once-prepared basal
caches (log-space HVG bags for STATE, Tx1 embeddings for the head's context); predicted responses are reduced online, never cached to disk. Pin the bag size
(128 cells), the cell-sampling seed and the cache layout, and record a measured
GPU-hour estimate for one inner model before launching the full set.

## 6. SL head and controls

Per gene, summarise the residual profile over $\mathcal{C}_{\text{ref}}$ by nine frozen
statistics $\Sigma$: mean, population standard deviation, the 0.10/0.25/0.50/0.75/0.90
quantiles, and the fractions below −0.5 and below −1.0. Target-context features use
absolute GeneEffect $\hat y$; the pan-essentiality block uses the fixed
$\mu_{\text{train}}$ so it can be ablated separately.

```text
prof_g     = Sigma( ( deltahat(g, c') )_{c' in C_ref} )

phi(a,b|c) = [ prof_a + prof_b , |prof_a - prof_b| ,        18  invariant
               rho(deltahat(a,.), deltahat(b,.)) ,           1  invariant
               yhat_ac + yhat_bc , |yhat_ac - yhat_bc| ,     2  context
               psi_min(yhat_ac, yhat_bc) ,                   1  context
               mu_a + mu_b , |mu_a - mu_b| ]                 2  invariant

s(a,b|c)   = sigmoid( f_theta( phi(a,b|c) ) )
```

Twenty-four dimensions, three of which vary with $c$; every block is invariant under
$a\leftrightarrow b$. $\psi_{\min}=\min(\hat y_{a,c},\hat y_{b,c})$ is the declared
non-interaction null (highest single agent; GeneEffect is negative-is-lethal), so
$\psi$ used as a ranking score is negated. An additive null is excluded because it
duplicates the sum feature. Uncovered genes are masked explicitly, never zero-filled;
in the measured-profile arm a pair with a missing profile leaves the common universe.

The head minimises binary cross-entropy over train-side SL contexts,
**context-balanced** so each context contributes equally, with positives and negatives
reweighted inside each context; otherwise HAP1 alone (57,013 of 86,460 train rows,
20 negatives) dominates. Three arms score one identical pair universe: the **predicted-profile arm**
uses out-of-fold predicted profiles, the **measured-profile arm** measured DepMap GeneEffect
over the same $\mathcal{C}_{\text{ref}}$ columns, and the **measured all-lines ceiling**
measured GeneEffect over all DepMap columns.

Controls, all reported as per-context lift on the same universe (§8):

| Control | Definition | Shortcut removed |
| --- | --- | --- |
| Pair-identity baseline | pair identity / train-context label frequency | pair memorisation |
| Anchor-gene baseline | anchor-gene frequency, `max(r_a, r_b)` | gene-level memorisation the pair-identity baseline misses |
| Null alone | $\psi$ alone, predicted and measured | ranking that is only the null |
| Gene-mean block alone | the $\mu_{\text{train}}$ block alone | pan-essentiality |
| Baseline-ladder backbone | the strongest residual control of the GeneEffect baseline ladder replacing the backbone | a backbone beating no simple baseline |
| Context-ablated head | identically trained head with the three context dimensions ablated | a context claim with no context information |
| Measured references | the measured-profile arm and the measured all-lines ceiling | reference only, never a bar |

The pair-identity baseline is a lift, not an absolute bar: on the unsplit first build of the
context-screen table it reached AUPR 1.0 on Jurkat, HeLa and PC9, so the split, not a
threshold, handles those contexts. The context-ablated head also includes
fixed deterministic context derangements and a permutation null. Declare the minimum
per-context incremental AUPR − base rate attributable to the context block before reading
any test label; statistical distinguishability alone does not clear it, and below it no
context claim is licensed at any absolute AUPR.

## 7. Metrics

**GeneEffect.** The [metric definitions](03-geneeffect-protocol.md#6-evaluation)
distinguish cross-gene absolute correlation from per-gene cross-context residual
correlation. The latter is scored on the separate 226-line benchmark; that score is
diagnostic, not an SL test score. Among SL test contexts only 22RV1 is GeneEffect-covered;
its per-context cross-gene Spearman is reportable and cannot support a context claim.

**SL.** Coverage and post-filter class counts precede every performance number. Report
AUPR − base rate per test context and as a macro; AUROC is secondary. Stratify each AUPR
by whether both, one or neither endpoint appeared in SL training. The de-duplicated
diagnostic macro weights observable label clusters: PC9/HeLa share one cluster but keep
separate scores. Uncertainty uses a two-way dyadic bootstrap over both endpoints, 2,000
replicates per context; a one-way anchor-gene bootstrap is invalid because pairs have
two endpoints. Report the predicted-profile arm minus the measured-profile arm both in
full and restricted to the context block; only the restricted form isolates out-of-sample GeneEffect cost, since 21 of 24
dimensions are context-invariant in both arms.

## 8. Rules

The experiments follow the usual train, validation and test practice, with the
[GeneEffect protocol rules](03-geneeffect-protocol.md#11-rules) (fit on training, tune on
validation, report both, experiment code reports and people decide). The SL-specific points:

- Test contexts are absent from response training, dependency training,
  $\mu_{\text{train}}$, SL-head training, hyperparameter and checkpoint selection,
  standardiser fitting, calibration and thresholding.
- Validation contexts select only; they are never promoted into training.
- Join contexts by DepMap ModelID through the checked-in map and fail loudly on an
  unmapped context. Never join on informal name: DepMap's `CellLineName` for K562 is `K-562`.
- (2026-11-04) A line needs basal scRNA and study labels to be in the multi-study SL benchmark; measured GeneEffect
  only supervises stage one. A line without a ModelID has the ID of its reviewed row in
  `configs/benchmarks/sl_line_identities.csv` (Cellosaurus accession, else `SL-<study>-<line>`), with its patient
  group and lineage. Such a line, and a DepMap line without GeneEffect, is an unlabelled training line of the final
  split: it gives stage one no target. Stage one and stage two are separate: the stage-one benchmark keeps the final
  split, and stage two reads stage one's production table (one fit on every labelled line; see the design spec). The
  SL split, placed by hand, holds lines with and without GeneEffect on every side. The stage-two control with measured GeneEffect is scored only on
  lines with GeneEffect. Results give all lines, each GeneEffect group and the non-cancer lines (DepMap
  "Non-Cancerous", or `non_cancer` in the identity table) separately.
- (2026-11-04) The multi-study benchmark's negatives are tested and not called, so its headline is recall in the
  top 10% of each list (positives pooled per line, mean over lines, minus pair identity per line); AUROC and AUPRC
  over the positive rate follow. (2026-11-13) A screen's recall divides by min(SL pairs, top 10% of the screen), so
  a perfect ranking gives 1 in every screen; before, a screen with more than 10% SL pairs capped it below 1.
- Fit the standardiser, any calibrator and any threshold on the training side only.
  **Per-context z-scoring of $\hat y$ is forbidden**: it consumes the test context's own
  distribution and erases the quantity under test.
- One common, label-independent pair universe across arms. Missing scores stay
  missing, never imputed to zero and never counted as negatives.
- Report every test context; none may be dropped after its result is inspected.
- Qualify every result with the Tx1 Tahoe-100M pretraining exposure of the test lines, and response
  results with STATE's pretraining exposure to K562, HepG2 and Jurkat.
- Every backbone-derived SL training vector is out of fold (§5); the profile reference cohort excludes all
  SL benchmark contexts. The standalone 226-line checkpoints are not eligible for held-out-SL claims.
- Pair labels are `silver_inferred` screen hits and screened non-hits in a named context (§3.4), not
  independently reconstructed per-context evidence; the table's known limitations (incomplete filter
  attribution, degenerate A549 validation labels, shared PC9/HeLa aggregate labels) travel with every result.
- A sigmoid score is a ranking score, not a calibrated probability of synthetic lethality. A single-gene
  GeneEffect result is not an SL result, and neither estimates a double-knockout interaction.
- The seed-0 GeneEffect test split has been observed and is not an SL tuning surface.

## 9. Required outputs

```text
context_screen_v2_split.json   copy of the tracked split file
checkpoint_selection.json      per inner model: selection surface, epoch
geneeffect_predictions.csv     out-of-fold yhat, deltahat and mu_train per (gene, context)
geneeffect_metrics.json
sl_features.parquet
sl_scores.csv
sl_metrics.json
run_manifest.json
```

`run_manifest.json` records the git commit, the DepMap
release, exact context lists and inner groups, the gene universe and variable-gene set,
all hyperparameters with the surface each was selected on, and the training, collation
and projection seeds. `sl_scores.csv` records context ModelID, canonical pair, arm,
score, label, endpoint-seen stratum and every exclusion reason. Planned metrics are not
results; a claim enters [`results/`](../results/) only after the frozen run completes and
its integrity checks pass.

## 10. Stage two, first run (2026-10-23)

Superseded by §11: it used the three-screen split, and its heads read the sum of two single-gene
dependencies. No number from it is carried forward.

`src/experiments/sl/stage_two.py`, `configs/sl/stage_two.yaml`, `hpc/run.sh sl-stage-two`. It follows the
2026-10-09 decisions above and blueprint §3.7 on the screen benchmark, and replaces §5–§6's backbone and features
with the stage-one backbone (GeneEffect protocol §16). Two heads, chosen 2026-10-23, share one MLP
(`[inputs] → Linear 64 → ReLU → Linear 1`, standardised on the training rows, binary cross-entropy with each
(screen, line) weighing 1 split equally between its positives and negatives; AdamW, lr 1e-3, weight decay 1e-4,
batches of 1,024, at most 50 epochs, the epoch kept with the highest mean AUPRC lift over the validation
(screen, line) contexts). No pair label is an input.

- **PARIS head.** PARIS (Benfatto et al. 2021), ported to Python from its official R code
  (`third_party/paris/`, `src/experiments/sl/paris.py`), fitted on the reference lines (the stage-one frame lines
  no screen tested): for each screened gene B, a random forest of B's measured GeneEffect on the mean-channel
  expression of every other gene, and the expression genes A it selects, with their importance and Pearson r.
  Input: the pair's PARIS evidence in both directions, order-free (larger and summed importance, summed r and its
  gap, directions selected; an unselected direction reads 0), the arm's GeneEffect of A and B in the line (sum,
  gap, minimum) and the screen. Arms, given as stage-one report rows and read as the report reads them: the
  frame-refitted default linear model, FiLM on the linear model and ΔE, FiLM on ΔE, and the measured GeneEffect as a
  reference, never a bar. A training line reads the arm out of fold (the linear model's fold export, or the stage-one
  run that left its fold out); a held-out line reads the main fit. PARIS runs in its `Ginicorrected` mode:
  a 20-gene pilot of the Boruta mode finished no gene in 32 minutes.
- **Viability-embedding head** (stage-one arms only). Input `[V_A + V_B ; |V_A − V_B| ; screen]` (67), with V from
  the run's main fit for every line. Out-of-fold V cannot be used: each fold run is another network, whose V
  correlates with the main run's at a median of 0.3 per dimension (a linear map explains about 45%), so a head
  fitted on fold V and scored on main-run V reads two coordinate systems. A training line's V is therefore in
  sample for stage one (fitted on its GeneEffect, never on SL labels).
- **Rows.** Training: the labelled rows of the single-cell stratum (the 29 screened training lines) with both
  genes in the decode panel. Held out: the validation and test rows of panel genes whose pair was labelled in a
  training line of the same screen, so every scorer ranks one universe; the rest are counted. A row whose arm
  GeneEffect is missing (measured arm only) scores missing.
- **Scored**, per split, screen, held-out line and arm: both heads; each head's context-ablated control (the
  PARIS head reading each gene's frame mean GeneEffect in every line, one fit shared by the arms; the V head
  with each gene's V averaged over the reference lines);
  PARIS alone (the larger importance, label-free and the same in every line); pan-essentiality (the two genes'
  summed frame mean GeneEffect); A-blind (their summed GeneEffect in the line); pair identity (the pair's hit
  rate in the screen's training lines). AUPRC and its ratio to the positive rate, AUROC, precision and recall at
  the top 20, on all pairs and on the pairs whose call varies between the screen's training lines. Each head's
  AUPRC differences from the controls carry a two-way dyadic bootstrap interval (§7).
- **Result** ([record](../results/sl_stage_two_seed0/README.md)): pair identity wins everywhere. No arm's PARIS
  head beats its gene-mean ablation, measured GeneEffect included. The V head falls below its own ablation.
- **Not yet:** the bulk-stratum complementary model, MiT4SL's A549 test, and per-gene shares of the positives.

## 11. The modular SL backbone, round zero (2026-11-12)

`src/experiments/sl/conditional.py`, `configs/sl/conditional.yaml`, `hpc/run.sh sl-conditional`; the benchmark
harness (`hpc/run.sh sl-baselines`) scores its tables on the latest SL split (`benchmark_v2`, 17 validation and 16
test lines). The backbone has four slots. A later round changes one slot only, so each gain has one cause.

| Slot | Round zero | Later modules |
| --- | --- | --- |
| Line encoder | bridged pseudo-bulk expression | copy number and mutation; single-cell features |
| Dependency reader | the default linear expression model | pair-aware terms; a nonlinear encoder with per-gene weights |
| Context editor | "A lost": A's expression at the training side's 10th percentile, unchanged when lower | PIE's profile after A's loss |
| Pair scorer | the conditional score, then a logistic head | the training SL fraction as an input; a continuous target |

- **Reader.** The default linear expression model, fitted on the training side of the SL split. No SL validation or
  test line's GeneEffect enters the fit, and the run stops if one does. The labelled single-cell training lines are
  read out of fold (bridge and model fitted without their fold); every other line is read by the full fit. This
  fit replaces the production fit for stage two, which read the GeneEffect of most SL held-out lines.
- **Conditional score** of a pair (A, B) in line c, in residual-SD units:
  `s(c, A, B) = −[ŷ(c with A lost, B) − ŷ(c, B)] − [ŷ(c with B lost, A) − ŷ(c, A)]`. An anchor list reads the first
  term only. The round-zero reader is affine in expression, so each term is A's fall in the line times the reader's
  response of B to A. Thus in round zero the line enters only through how far A falls: the ranking of an anchor
  list changes with the line only when A is already low. Since §12 the run reads each term from the reader's
  predictions of the line's edited row, which gives the same terms for an affine reader.
  A gene outside the expression space (or a drug anchor) cannot be edited: its term is 0, a tie. Such rows are
  not covered; the harness's Table 4 ranks the covered rows alone.
- **Context-ablated score.** The same score with the training-side mean profile in place of the line's profile.
  It has one value per pair in every held-out line.
- **Head.** Logistic regression on the training lines' genetic screens with both classes. Inputs: the two terms, the
  predicted GeneEffect and the expression of A and B in the line (0 for an anchor's A), and the design. Weights
  as the headline: each line equal, then its studies, then a study's screens, then the two classes of a screen.
  The penalty is chosen on validation recall in the top 10%; test is scored once. The context-ablated head reads
  the training-side mean profile.
- **Stage-one metrics** of the reader on the SL lines with GeneEffect, beside the same prediction with its lines
  permuted: per-line residual Spearman (the selector), per-gene residual Spearman, the residual skill score
  (within-gene R²) and calibration slope, the top-20 recall of each line's strongest dependencies, the count of
  selective genes with per-gene Spearman of at least 0.3, and the whole-line Pearson as a sanity row.
- **Ceilings.** Stage two: the agreement of two labels of one pair within a line across lists, and within a study
  across lines (the harness's `label_agreement`). Stage one: the agreement of two independent DepMap screens of one
  line, from DepMap's per-screen GeneEffect (on the H20; not measured in round zero, measured since §12).
- **Reading.** If the conditional score is above its context-ablated score on new pairs or on seen pairs whose
  label differs, the SL rule works in linear form. If it is at random, the reader does not see single genes, and
  pair-aware terms come first. If stage one is near its ceiling, the next rounds change the editor and the scorer,
  not the reader.
- **Result** ([record](../results/sl_backbone_round_zero_seed0/README.md)): the conditional score equals its
  context-ablated score (0.295 / 0.288 against 0.296 / 0.295, validation / test). The head (0.374 / 0.322) is the
  best model after the training SL fraction (0.524 / 0.495), but its gain over its ablation is small and
  gene-level. No model ranks the label-differs pairs above random. The equality of the conditional score and its
  ablation is by construction for an affine reader, so it does not put the reader first. Next: the buffer-aware reader
  ([spec](specs/2026-11-12-sl-buffer-aware-reader-design.md)): the reader gets each gene's buffer in the line (its
  paralogs and complex members), so that the response of B to the loss of A differs between lines.

## 12. The buffer-aware reader (2026-11-12)

`configs/sl/conditional_buffer.yaml` (reader: `configs/linear_model/buffer_aware_reader.yaml`), `hpc/run.sh
sl-conditional`; [spec](specs/2026-11-12-sl-buffer-aware-reader-design.md). This module changes one slot of §11, the
dependency reader. The line encoder, the context editor, the pair scorer and the head's inputs do not change.

- **Reader.** The default linear expression model with the partner block between the expression components and the
  data-selected genes (block set `partners_and_selected`, the same penalties). For gene B in a line the block reads
  the lowest expression among B's paralogs, their sum, the closest paralog's expression, the mean expression of B's
  complex members, and the fraction of B's partners that are low. The weights are fitted on GeneEffect of the
  training side. No SL label is read. The reader is not affine in expression.
- **Terms.** Each term is the reader's prediction of B on the line's row with A at `min(x_A, low_A)`, minus its
  prediction on the row as it is. An edit that does not change the row gives exactly 0. Each line and each lost
  gene is queried once. For an affine reader, the terms are equal to those of §11 (a test checks this). The
  context-ablated score reads the training-side mean profile, so it has one value per pair and design.
- **Partner pairs.** A score table marks a row `partner` when A is a paralog or a complex member of B in the
  reference tables of the partner block (for a combinatorial pair, also B of A). The harness reports every row on
  this subset, and on the rows where no gene is a dependency alone (measured GeneEffect of B, and of A for a pair,
  at or above −0.5; lines with GeneEffect only).
- **Paired differences.** The harness config names (scorer, control) pairs. For each pair the harness gives the mean
  per-line difference in recall in the top 10% with the line bootstrap of the difference from the training SL
  fraction, on all rows and on each stratum and subset.
- **Stage-one ceiling.** One DepMap screen's residual as the prediction of another screen's residual of the same line
  (the first two screens in ScreenID order), on the SL lines with two or more screens, with the reader's stage-one
  metrics.
- **Decision rule** (spec §4). If the conditional score is above its context-ablated score on the partner pairs (the
  interval above zero on validation, the same sign on test), the next module is the line encoder in each line's own
  genes. If not, the next module is the context editor (PIE's profile after A's loss).
- **Result** ([record](../results/sl_buffer_aware_reader_seed0/README.md)): a null result. On the partner pairs (15% of
  the held-out rows) the conditional score minus its context-ablated score is −0.004 [−0.018, +0.009] on validation
  and −0.004 on test; on all pairs −0.006 / −0.003. The partner block changes the scores of partner pairs (Spearman
  0.59 against the default reader; 0.9995 elsewhere), but not by line. The head on the new terms is the best model
  (0.383 / 0.328) and is not above its ablation. The stage-one metrics do not change (per-line residual Spearman
  0.183 / 0.194); two DepMap screens of one line agree at
  0.265 / 0.367 (4 and 8 lines; 0.246 on 124 training lines), so a reader of the true dependency can reach about the
  square root of that, about 0.5. On the partner rows that the reader can edit (19.5%), the difference is also null. By the decision rule, the next
  module is the context editor (PIE's profile after A's loss).

## 13. Two single-cell inputs (2026-11-13)

`hpc/run.sh sl-codetection`, `hpc/run.sh sl-conditional` with `configs/sl/conditional_wide.yaml`,
`conditional_codetection.yaml` and `conditional_codetection_shuffled.yaml`, then `hpc/run.sh sl-baselines
configs/sl/benchmark_single_cell_inputs.yaml`; [plan](specs/2026-11-12-pie-and-single-cell-plan.md), round C parts 1
and 3. Each part changes one thing of the §11 backbone and is its own run.

- **Each line read in the genes it measures.** The config key `space_measured_share` (null: the §11 space, the genes
  that every held-out line measures in pseudo-bulk, 6,523) sets the reader's space to the bulk genes that at least
  this share of the labelled single-cell training lines measure. `conditional_wide.yaml` uses 0.5 (18,294 genes): the
  count falls between half and three quarters of the lines (18,302 at half, 17,142 at 75%), and at half or more a
  gene's bridge slope is fitted mostly on measured values. A held-out line's unmeasured value takes the training mean,
  as a training line's does. A row is covered only when its own line measures each gene it loses (the pseudo-bulk's
  mask before the fill). Known side effects: a filled value takes a rank slot in the per-line quantile normalisation,
  a rarely measured gene's bridge slope is fitted partly on filled values, and the expression components are fitted
  on rows with filled values.
- **A and B detected in the same cells.** Preparation (`prepare.py --codetection`) writes, for each SL split line with
  labelled rows and each of its pairs, the share of the line's cells with a count above 0 for both genes, the share
  expected under independence, `mean_i (1 − exp(−r_A L_i / T)) (1 − exp(−r_B L_i / T))` with `L_i` the cell's library
  and `T` = 10^4, both genes' detection shares, and the same after `shuffle_counts` (seed 0) permuted each gene's
  counts across the cells ([card](data/sl-codetection.md)). The cache holds the rate `r` from the line's mean profile
  and the rate matched to each gene's detection share; the head reads the matched version. The config key
  `codetection` (null, `real` or `shuffled`) adds two head inputs: the share and `log((k + 0.5) / (e + 0.5))`, with `k`
  and `e` the observed and expected counts of cells. A value that the source cannot give (a drug anchor, an unlisted
  gene) reads 0. The context-ablated head reads the training rows' mean of both inputs. This part uses the narrow
  space.
- **Rows** in the harness: the backbone in the narrow space (rerun), in each line's own genes, and its head with real
  and with shuffled co-detection, each with its context-ablated version; paired differences of each against the
  narrow backbone, against its ablation, and of the real against the shuffled co-detection.
- **Result** ([record](../results/sl_single_cell_inputs_seed0/README.md)): the narrow rerun reproduces §11. Read in
  its own genes, the reader covers 79.0% of the held-out rows (43.8% before) and its stage-one per-line residual
  Spearman rises to 0.207 / 0.197 (0.183 / 0.193), but the SL ranking does not: head −0.017 on test (+0.007 on
  validation), intervals across zero. Co-detection adds nothing beyond each gene's expression: the head with it minus
  the head with shuffled cells is +0.004 / +0.001, and the real log ratio hardly agrees with the shuffled one (0.047)
  at a median of 292 cells per line. By the plan's decision table, neither part joins the backbone.

## 14. The PIE editor (2026-11-13)

`src/experiments/sl/pie_pass.py` (`hpc/run.sh sl-pie-pass`, `.venv-pie`), `src/experiments/sl/pie_editor.py`
(`hpc/run.sh sl-pie-editor`), `configs/sl/pie_editor.yaml`, `configs/sl/benchmark_pie_editor.yaml`; plan
[round A](specs/2026-11-12-pie-and-single-cell-plan.md). This module changes one slot of §11, the context editor. The
reader (the default linear expression model of the backbone run), the line encoder, the conditional score and the
head's inputs do not change.

- **PIE's input.** The 86 screened SL lines (the lines of the backbone run's rows) in a PIE root of their own
  (`data/sl_pie/v1`). PIE's input is the mean over a line's cells of log1p CP10K, from the ladder's cell files. The
  decoded genes are the reader's 6,523 expression genes. The context text of a line is PIE's released Cellosaurus
  text where PIE has one; a line without text takes PIE's no-source path.
- **PIE's change.** The frozen cross-dataset checkpoint, DepMap source masked, predicts each line's profile change
  after each lost gene (gene A of a row, gene B of a combinatorial row; 13,950 genes with a PIE key) in every
  screened line: the expectation of its delta-p bins, a change of the mean over cells of log1p CP10K.
- **Units.** The reader's raw row is log1p CPM of the summed UMI. Each gene's change is multiplied by the slope of the
  least-squares line from PIE's input to the raw pseudo-bulk over the SL training lines with cells.
- **The row after the loss.** Raw pseudo-bulk plus the scaled change, then the reader's quantile normalisation, the
  line's bridge (its fold's for a labelled single-cell training line) and the reader. A term is B's prediction after
  minus before. With a zero change the row and the terms are the line's own (a test and a check on the run).
- **Rows.** PIE's profile of the line, and PIE's profile of another line (one seeded permutation of the screened
  lines, no line keeps its own), each as the conditional score and through the logistic head (the head fitted on the
  line's own terms, applied to the other line's). Each is reported on the rows the simple edit covers (0 elsewhere,
  as the simple edit, for the paired differences with it) and on every row PIE can edit.
- **Reference rows.** On the test lines Jurkat, HCT116 and HEK293T: the editor with the line's own measured
  Perturb-seq change (PIE's `delta_p`; Replogle for Jurkat, X-Atlas/Orion for HCT116 and HEK293T) against the change
  of another line (K562 for Jurkat; HEK293T and HCT116 for each other), on the rows whose lost genes both lines
  perturbed. These rows are references, not models.
- **PIE's own task.** On PIE's four Replogle folds (one line held out each): PIE's predicted response of the held-out
  line against the mean measured response to the same perturbation in the other lines, as Pearson and Spearman over
  genes of each (line, perturbation), on the measured response and on its deviation from that mean.
- **Decision** (plan, after round A). PIE's profile above the simple edit and above its permuted version: PIE keeps
  the editor job. PIE equal to its permuted version and the measured own-line profile above another line's: a
  line-specific PIE. The measured own-line profile equal to another line's: the profile after A's loss does not
  carry the line effect through this reader; repeat with the per-gene reader.

## 15. Line-specific signal in the labels (2026-11-19)

`src/experiments/sl/line_specificity.py`, `configs/sl/line_specificity.yaml`, `hpc/run.sh sl-line-specificity`. No
model is read. The question comes before any line-specific model: does the difference between lines in a pair's
screen score repeat?

- **Rows.** The genetic screens with both classes and a score, on the 77 lines of the SL split (every side; nothing is
  fitted): 47 studies, 1,737,816 (study, line, pair) values. A score becomes a percentile rank inside its screen,
  turned so that the study's SL calls are on the high side.
- **Inside a study** (14 studies with three or more lines): the share of the variance between pairs, for the rank and
  for the SL call. The remainder is the line-specific part plus noise.
- **Between two studies that share a line** (41 study pairs): the correlation over the common pairs of a line with the
  same line and with another line of the other study; the same-line advantage, that contrast with each screen's own
  level removed; and the repeat of the line-specific part, the correlation between the two studies of a pair's
  deviation from its mean over the shared lines (two shared lines or more, 12 study pairs). Bootstrap over pairs,
  2,000 draws. Each is given for the rank and for the SL call.
- **Result** (run `line_specificity_seed0_v2`, H20 port 30838, code `901bd09`, 9 s on 224 cores; outputs in
  `outputs/sl/line_specificity/line_specificity_seed0_v2/` on the host). Mean over the study pairs:

  | Value | Same line | Another line | Same-line advantage | Repeat of the line-specific part |
  | --- | ---: | ---: | --- | --- |
  | Rank | 0.141 | 0.127 | +0.049 [+0.017, +0.081] | 0.039 [−0.017, +0.093] |
  | SL call | 0.269 | 0.226 | +0.029 [+0.002, +0.065] | 0.060 [+0.006, +0.119] |

  The share of the rank variance between pairs is 0.66 in Harle 2025, 0.48 in Ito 2021, 0.42 in Flister 2025 (hybrid),
  0.28 in Klingbeil 2024 and 0.22 in Ebright 2026.
- **Reading.** A line-specific signal is there, and it is small between studies: the square root of the repeat puts
  the top score of a perfect model near 0.20 (rank) to 0.25 (call), against 0.50 for stage one's GeneEffect. The
  agreement between two studies is low even in the same line, so noise between studies limits this estimate. The
  advantage is clear between the paralog-library studies (Ito × Klingbeil +0.157 [+0.12, +0.20], Ebright × Klingbeil
  +0.364 [+0.20, +0.54], Ebright × Ito +0.302 [+0.16, +0.45]) and absent in most other study pairs.
- **Limit.** The label table holds one score for each pair and screen (constructs are merged), so the repeat inside
  one study, which is what the benchmark's metric needs, is not measured here; it needs the raw replicate files.

## 16. Options for a less noisy training target (2026-11-19)

`src/experiments/sl/target_options.py`, `configs/sl/target_options.yaml`, `hpc/run.sh sl-target-options`. Four ways
to make the training target less noisy are compared by one quantity, §15's repeat of the line-specific part between
two independent measurements. No model is read. The benchmark's labels are not changed; an option changes only what a
model trains on.

- **The continuous score in place of the SL call.** Two forms: the rank inside the screen, and the rank's top 10% as a
  ramp from 0 to 1 (0 below it).
- **The mean of several measurements.** Inside Harle 2025: the mean of two replicates against the third. Between
  studies: the share of the training lines' (line, pair) cells that two studies screened.
- **A weight from the measurement error.** Each pair's weight is the inverse of its mean variance in Harle's Table S4.
- **One score definition.** Ito 2021 and Klingbeil 2024 publish GEMINI; their score becomes the LFC of the double loss
  minus the LFC of each single loss (the other side a cutting control), for 101,447 of their 104,403 labelled rows.
- **Inside one study.** Harle's count matrix (Table S3) gives a score for each replicate, the same difference of LFCs;
  a replicate's SL calls are as many of its lowest scores as the line has SL calls. The mean of a line's replicate
  scores agrees with Harle's published score at Spearman 0.734 (output check).
- **Result** (run `target_options_seed0_v2`, H20 port 30838, code `0667a08`, 63 s; outputs in
  `outputs/sl/target_options/target_options_seed0_v2/` on the host). Repeat of the line-specific part, 95% bootstrap
  over gene pairs:

  | Setting | Rank | Top 10% of the rank, as a ramp | SL call |
  | --- | --- | --- | --- |
  | Two studies (12 study pairs) | 0.039 [−0.017, +0.093] | 0.116 [+0.049, +0.167] | 0.060 [+0.006, +0.119] |
  | Inside Harle, two replicates | 0.178 [0.162, 0.196] | **0.392** [0.353, 0.427] | 0.348 [0.313, 0.381] |
  | Inside Harle, mean of two replicates against the third | 0.225 [0.195, 0.255] | **0.498** [0.439, 0.549] | 0.435 [0.376, 0.490] |

  The weight: 0.130 to 0.134 for Harle against another study (7 study pairs), 0.178 to 0.162 inside Harle. One score
  definition, on the 16 study pairs with Ito or Klingbeil: same-line agreement 0.188 to 0.031, same-line advantage
  +0.059 [+0.028, +0.087] to +0.017 [−0.009, +0.043]. Several studies: 8,297 of 1,258,077 training cells (0.7%).
- **Reading.** The top of the rank as a continuous target repeats best, the plain rank worst: most pairs are not SL,
  and their order is noise. The mean of two measurements repeats better than one, but two studies share too few cells
  to use it, and a published score is already the mean of its replicates. The weight gives no gain. The simple common
  definition is worse than GEMINI. Inside one study the line-specific part repeats at 0.35 to 0.39 for one replicate,
  against 0.04 to 0.12 between two studies.
- **Limits.** One study has replicate data here; its replicates share the library, the guides and the cell population,
  so the value inside the study is an upper bound for the biology. The weight is per pair, not per pair and line. The
  common definition is the simplest one; a model-based score for all studies needs the raw counts. No model was
  trained on these targets.

## 17. The head trained on the top of each screen's rank (2026-11-19)

`head_target` in the `sl_conditional` config (`src/experiments/sl/conditional.py`): `call`, the screen's SL call, or
`ramp`, §16's best target. `configs/sl/conditional_buffer_ramp.yaml` is `conditional_buffer.yaml` (§12) with
`head_target: ramp`; the harness config is `configs/sl/benchmark_ramp_head.yaml`.

- **Target.** The rank of a row's score inside its screen, turned so that the study's SL calls are high, then its top
  10% as a ramp from 0 to 1 (0 below it). A row without a score keeps its call.
- **Fit.** The same logistic head and inputs. A row counts as SL with weight `target` and as negative with weight
  `1 − target`; the benchmark's weighting stays (every line equal, then its studies, its screens and the two classes of
  a screen). The penalty is chosen on validation recall in the top 10% of the calls, as before. Only the training
  target changes: validation and test read the labels as built.
- **Runs** (H20 port 30838, code `9652bb6`, CPU, 100 threads each, 9 min): `conditional_buffer_call_seed0` and
  `conditional_buffer_ramp_seed0`, then `benchmark_ramp_head_seed0`. The rerun of the head on the calls reproduces
  §12's row (0.394 / 0.361).
- **Result.** Recall in the top 10%, AUROC and AUPRC over the positive rate, in that order, mean over lines (17
  validation, 16 test):

  | Head on the buffer-aware reader's terms | Validation | Test |
  | --- | --- | --- |
  | Trained on the SL call | **0.394**, 0.743, 5.07 | 0.361, 0.716, 4.24 |
  | Trained on the SL call, context-ablated | 0.383, 0.741, 4.80 | 0.355, 0.717, 4.52 |
  | Trained on the top 10% of each screen's rank | 0.389, 0.743, 5.06 | **0.367**, 0.716, 4.20 |
  | Trained on the top 10% of each screen's rank, context-ablated | 0.377, 0.740, 4.77 | 0.356, 0.717, 4.45 |

  Paired differences in recall, 95% line bootstrap: ramp − call −0.005 [−0.014, +0.004] validation, +0.005 [−0.004,
  +0.018] test; ramp − its context-ablated version +0.012 [−0.009, +0.030], +0.011 [−0.014, +0.038]; on seen pairs
  whose label differs from the training lines, ramp − call +0.002 [−0.011, +0.018], +0.006 [−0.005, +0.017]. The
  reader's stage-one numbers on the SL lines with GeneEffect: per-line residual Spearman 0.183 validation (12 lines),
  0.194 test (11 lines).
- **Reading.** A null result: the head trained on the better target equals the head trained on the calls, and neither
  is above its context-ablated version. The head has seven inputs with one weight each for all pairs, fitted on 1.4
  million rows, so noise in the target averages out and the target is not what limits it. The line-specific part of
  the target (§16) can be used only by a reader that has capacity for each pair or gene. Validation still selects the
  head trained on the calls.

## 18. The gene-knowledge, cell-set model (2026-11-19)

`src/model/sl_embedding.py`, `src/experiments/sl/embedding.py`, `configs/sl/embedding.yaml`,
`hpc/run.sh sl-embedding`; the [plan](specs/2026-11-19-embedding-transfer-plan.md) gives the reasons. The question:
can a deep model beat the baselines on SL in held-out lines? Stage one is pretraining. Stage two reads the model's
gene-in-line embeddings, not its predicted GeneEffect. A better GeneEffect is not an aim.

- **Model.** Width 32.
  - *Gene vector:* `u_g = Σ_s P_s(k_gs) + r_g`. `k_gs` are frozen gene sources, each centred on its mean over the
    genes it covers: function text (`gene_text`), ESM C 600M protein (`esmc_600m`) and the STRING space
    (`string_space`). `P_s` is the stage-one gene encoder's projection. `r_g` is a free deviation for each gene, zero
    at the start and pulled to zero by weight decay.
  - *Line:* a mean token `t_0` (an MLP over the line's bridged pseudo-bulk, the SL backbone's line input of §11; bulk
    RNA for a bulk-only line) and one token `t_i` for each cell (a linear map of its frozen Tx1-3B embedding, centred
    on the fit lines' mean cell). Any number of cells: at most `cells.per_step` drawn at a training step, none at the
    rate `cells.dropout`, all of them at scoring. A bulk-only line has `t_0` alone.
  - *Reading:* `α_g = softmax_i(q(u_g) · t_i / √32)` over the line's cells, `h_g(c) = t_0 + Σ_i α_gi t_i`,
    `V(g, c) = u_g ⊙ h_g(c)`, `z(g, c) = Σ_k V_k(g, c)`. The weights are normalised over the cells alone, so the
    order and the number of the cells do not change the output. The token maps end in a zero layer: the untrained
    model predicts `z = 0`, the gene mean.
  - *Pair head:* an MLP on `[V_A + V_B ; |V_A − V_B| ; V_A ⊙ V_B ; u_A ⊙ u_B ⊙ Σ_i w_i t_i² ; combinatorial]`, with
    `w_i` the product `α_Ai α_Bi` normalised over the cells. The score is the same for (A, B) and (B, A). An anchor
    that is not a gene of the model has a zero vector, so the row reads gene B alone.
- **Stage one.** Target `(GeneEffect − m_g) / σ_g`, mean squared error. Fit lines: the labelled single-cell training
  lines of the SL split (every lineage, as the stage-one frame) and the labelled bulk-only lines of the linear
  model's training side (without its dropped lineages), without every line that has an SL label and every line of
  such a line's patient. `m_g`, `σ_g`, the expression scaling and the cells' centre
  are fitted on the fit lines. The epoch kept has the lowest loss on the SL training lines with GeneEffect, which are
  out of sample. Thus the model reads every SL line, training or held out, out of sample, in one coordinate system.
  This track does not use the per-line residual Spearman as its selector (GeneEffect protocol §6), by the user's
  decision of 2026-11-19.
  §19 replaces this residual target with GeneEffect itself. With the residual target the early stop of the
  2026-11-19 runs kept the untrained model, so their stage two started from untrained weights.
- **Stage two.** Rows: the SL training lines' genetic screens with both classes. A step draws lines evenly and, in
  each line, rows at the benchmark's weights (§11: each line equal, then its studies, a study's screens, a screen's
  two classes); the loss is binary cross-entropy on the SL call. First the head trains on the frozen stage-one model.
  Then, from that state, the whole model trains with the GeneEffect loss kept at each weight of
  `stage_two.geneeffect_weights`. Validation recall in the top 10% selects the epoch and the phase; test is scored
  once.
- **Outputs.** `scores/embedding.parquet` (one score per line, design and ordered pair) and the two tables of the
  line-input controls (below) for the harness (`configs/sl/benchmark_embedding.yaml`), `predictions.parquet` (the model's GeneEffect of the SL lines with
  GeneEffect), `best.pt`, `metrics.json` and `results.md`: SL on training, validation and test; the stage-one numbers
  on the SL lines, beside those of the fit lines' gene mean alone; each phase's validation recall; stage one's loss
  per epoch.
- **Variant without cell tokens** (`configs/sl/embedding_mean_token.yaml`, `cell_embeddings: null`): the same model and
  training, every line on its mean token alone. The difference from the full model is what the cells add.
- **Variant with CellNet's cell embeddings** (`configs/sl/embedding_cellnet.yaml`): the same model and training with
  another frozen cell encoder. CellNet (arXiv:2610.21943, weights of the repository's release 0.1.5, Apache-2.0 code)
  draws each cell as an image; `hpc/cellnet/embed_lines.py` writes its native 768-wide readout for the same cells as
  the Tx1 cache, from raw counts, with genes matched by Ensembl ID. Nothing is known here about cancer cell lines in
  its pretraining data.
- **Line-input controls.** The finished weights, scored again with each line's input replaced; no model is refitted.
  `mean_input`: the fit lines' mean expression and no cells, so the score reads the two genes alone and a pair has one
  score in every line (the context-ablated row of §8). `other_line`: the expression and the cells of another
  benchmark line, the next one in a seeded order. A run writes `scores/embedding_mean_input.parquet` and
  `scores/embedding_other_line.parquet`; `--line-input-controls` writes them for a finished run from its `best.pt`.
- **Row sets of the harness** (`training_calls` in `src/experiments/sl/benchmark_baselines.py`, 2026-11-20). A pair
  is seen when the same study labelled it in a training line. The seen pairs divide by the calls of those training
  lines: no line called the pair SL, the lines differ, every line called it SL. No label of the held-out line defines
  these sets, and in the first and the third the training SL fraction is constant, so a ranking there comes from the
  genes or from the line. The two older strata (the label agrees with, or differs from, the training lines) read the
  held-out label: a score that follows the training lines is below random in the second one by its construction.
- **Not in these runs.** A row without pretraining and any other ablation of a module; a model refitted without the
  line input; a perturbation model or perturbation data. The full model's run stops if a line with cells has no Tx1
  cache.
- **Runs** (seed 0, one GPU each, about two minutes of training). 2026-11-19: `embedding_seed0` (H20 port 30838, code
  `232b4f0`), `embedding_mean_token_seed0` (port 30757, `ad355ee`), `embedding_cellnet_seed0` (port 30757, `64dbc5c`)
  and `embedding_mean_token_slow_stage_one_seed0` (`configs/sl/embedding_mean_token_slow_stage_one.yaml`: stage one
  at a learning rate of 1e-4; port 30757, `232b4f0`). 2026-11-20: the line-input controls of the full model and of
  the model without cell tokens (port 30838, `5500c8d`), then the harness run `benchmark_embedding_seed0_v3`
  (`2d57c51`). [Record](../results/sl_embedding_seed0/summary.html).
- **Result, all pairs.** Recall in the top 10%, AUROC and AUPRC over the positive rate, in that order, mean over lines
  (17 validation, 16 test):

  | Model | Validation | Test |
  | --- | --- | --- |
  | Gene-knowledge, cell-set model (Tx1-3B cell tokens) | 0.500, 0.787, 20.0 | 0.485, 0.774, 7.9 |
  | The same, context-ablated (the fit lines' mean input, no cells) | 0.509, 0.800, 16.7 | 0.498, 0.781, 8.0 |
  | Gene-knowledge model without cell tokens | **0.508**, 0.789, 17.5 | **0.501**, 0.776, 8.5 |
  | The same, context-ablated (the fit lines' mean input) | 0.460, 0.785, 8.3 | 0.485, 0.776, 7.7 |
  | Gene-knowledge, cell-set model with CellNet's cell embeddings | 0.486, 0.783, 9.8 | 0.473, 0.766, 8.0 |
  | Backbone head on the buffer-aware reader's terms (§12) | 0.394, 0.743, 5.1 | 0.361, 0.716, 4.2 |
  | Kebabci 2026 paralog classifier on stage one's predicted GeneEffect | 0.355, 0.656, 5.7 | 0.395, 0.664, 6.4 |
  | KG4SL | 0.367, 0.689, 3.4 | 0.338, 0.653, 3.0 |
  | Pan-essentiality (gene mean GeneEffect) | 0.380, 0.728, 4.5 | 0.297, 0.679, 3.5 |

  Paired differences in recall of the full model, 95% line bootstrap, validation / test: against Kebabci
  +0.145 [+0.090, +0.205] / +0.090 [+0.038, +0.145]; against KG4SL +0.133 [+0.059, +0.214] / +0.147 [+0.063, +0.228]; against the backbone
  head +0.106 [+0.051, +0.162] / +0.124 [+0.056, +0.195]; against pan-essentiality +0.121 [+0.001, +0.240] / +0.188 [+0.044, +0.329]; against
  the model without cell tokens −0.008 [−0.029, +0.015] / −0.016 [−0.052, +0.018]. CellNet's variant against the model
  without cell tokens: −0.023 [−0.054, +0.005] / −0.028 [−0.076, +0.007].
- **Result, seen and unseen lines and pairs.** Recall in the top 10%, each row set ranked inside it in each screen
  (validation / test; the training lines are in sample):

  | Model | Seen lines (training) | Unseen lines, seen pairs | Unseen lines, new pairs | Seen pairs that no training line called SL | Seen pairs, the training lines differ |
  | --- | --- | --- | --- | --- | --- |
  | Gene-knowledge, cell-set model (Tx1-3B cell tokens) | 0.782 | 0.518 / 0.514 | 0.475 / 0.354 | 0.388 / 0.375 | 0.411 / 0.564 |
  | The same, context-ablated | — | 0.532 / 0.521 | 0.461 / 0.427 | 0.338 / 0.366 | 0.424 / 0.592 |
  | Gene-knowledge model without cell tokens | 0.898 | 0.529 / 0.525 | 0.472 / 0.403 | 0.366 / 0.347 | 0.427 / 0.552 |
  | The same, context-ablated | — | 0.477 / 0.503 | 0.296 / 0.329 | 0.371 / 0.367 | 0.374 / 0.582 |
  | Backbone head on the buffer-aware reader's terms (§12) | 0.407 | 0.364 / 0.359 | 0.419 / 0.534 | 0.216 / 0.232 | 0.329 / 0.438 |
  | Kebabci 2026 paralog classifier on predicted GeneEffect | — | 0.355 / 0.412 | 0.330 / 0.411 | 0.229 / 0.303 | 0.371 / 0.564 |
  | KG4SL | — | 0.374 / 0.368 | 0.310 / 0.323 | 0.290 / 0.226 | 0.287 / 0.461 |
  | Pan-essentiality (gene mean GeneEffect) | 0.318 | 0.353 / 0.292 | 0.329 / 0.382 | 0.409 / 0.261 | 0.263 / 0.301 |
  | Random score | 0.093 | 0.101 / 0.119 | 0.224 / 0.220 | 0.071 / 0.095 | 0.230 / 0.225 |

  Held-out rows: seen pairs 311,834 (57.3%), of which no training line called the pair SL 290,388 (53.4%), the
  training lines differ 17,584 (3.2%), every training line called it SL 3,862 (0.7%; not shown, its random level is
  high); new pairs with both genes seen 231,745. The full model against Kebabci: seen pairs
  +0.163 [+0.093, +0.233] / +0.102 [+0.037, +0.165]; new pairs +0.145 [+0.012, +0.287] / −0.057 [−0.204, +0.077];
  pairs that no training line called SL +0.159 [+0.060, +0.304] / +0.072 [−0.021, +0.163]. The full model
  against pan-essentiality on the pairs that no training line called SL: −0.021 [−0.126, +0.077] / +0.114 [−0.026, +0.269].
- **Result, the line's input.** Paired differences in recall, validation / test. The full model minus its
  context-ablated version: all pairs −0.009 [−0.034, +0.020] / −0.013 [−0.048, +0.018]; seen pairs
  −0.013 [−0.034, +0.009] / −0.007 [−0.034, +0.022]; pairs that no training line called SL
  +0.050 [−0.043, +0.157] / +0.009 [−0.073, +0.104]; pairs on which the training lines differ
  −0.012 [−0.058, +0.033] / −0.028 [−0.116, +0.050]; new pairs +0.014 [−0.063, +0.109] / −0.073 [−0.317, +0.114].
  The full model minus itself on another line's input: all pairs +0.054 [+0.005, +0.120] / −0.011 [−0.042, +0.015]; pairs
  that no training line called SL +0.075 [+0.006, +0.153] / +0.072 [+0.003, +0.154]. The model without cell tokens
  minus its context-ablated version: all pairs +0.049 [+0.014, +0.088] / +0.016 [−0.012, +0.045];
  new pairs +0.177 [−0.024, +0.448] / +0.075 [−0.049, +0.230]; minus itself on another line's
  input: all pairs +0.053 [+0.001, +0.131] / +0.010 [−0.010, +0.031]; new pairs
  +0.005 [−0.037, +0.053] / +0.009 [−0.066, +0.113].
- **Result, training and stage one.** In all four runs the loss on the SL training lines is lowest at epoch 0 (1.0503,
  then 1.0725 for the full model and 1.0521 at the lower learning rate), so the untrained model is kept; the head on
  that frozen model scores 0.109 on validation, and the phase that trains the whole model is chosen. The slower stage
  one thus gives the scores of the model without cell tokens. Stage-one numbers of the full model on the SL lines with
  GeneEffect: per-line residual Spearman 0.033 validation (12 lines), 0.035 test (11 lines); the linear expression
  model gives 0.183, 0.194.
- **Reading.** A deep model is above every published baseline on all pairs, and every interval against a baseline is
  above zero. The gain does not come from the parts that the plan set out to test. GeneEffect pretraining gives
  nothing: the untrained model is kept, its embeddings are zero, and the result comes from the training of the whole
  model on the SL labels. The cells give nothing: the model without cell tokens is the highest row. From the training
  lines to unseen lines the recall falls by a third, and from seen pairs to new pairs it falls again, most on test,
  where the difference from Kebabci is not above zero. On the pairs that no training line called SL the model ranks
  the held-out line's SL pairs above random and above Kebabci on validation, so it does not only copy the training
  lines' calls; but its context-ablated version and pan-essentiality, which read no line, score the same there. The
  full model does not read the line: with its line input taken away it scores the same on all pairs and in every row
  set, also where the training lines differ. A wrong line's input lowers its score on the pairs that no training
  line called SL, so the score depends on the input, but the right line is not above no line. The model without
  cell tokens is above its context-ablated version on validation and not on test, and its gain over the mean input
  on new pairs is the same with another line's input. Thus no row here is context evidence (§8): the score is a
  score of the gene pair. The open parts are a stage one that learns something that transfers to the SL lines, a
  model that must use the line to fit the training lines, and new pairs.

## 19. The direct GeneEffect target and the line term (2026-11-20)

`src/model/sl_embedding.py`, `src/experiments/sl/embedding.py`, `src/experiments/sl/gene_in_line.py`, the four
configs below, `configs/sl/benchmark_line_term.yaml`, `hpc/run.sh sl-embedding`; the
[plan](specs/2026-11-20-line-term-and-direct-target-plan.md) gives the reasons. The question: do a line's single
cells make the SL score change with the line, and does that change hold in unseen lines? This section changes two
parts of §18. All other parts of §18 stay.

- **Stage one.** The target is GeneEffect itself, `(GeneEffect − c) / s`, mean squared error. `c` and `s` are one
  centre and one scale, shared by every gene: the mean and the SD of all the fit lines' values (`target_scale`). The
  model must learn how essential each gene is, and how a gene differs between lines. The bias of the mean-token map
  gives a term that is the same in every line. The fit lines and the early stop do not change: the epoch kept has the
  lowest loss on the SL training lines, which stage one does not fit. The untrained model predicts one value for
  every gene. In the runs of 2026-11-20 its loss was 1.0224 and the kept epoch (1 or 2) had 0.196 to 0.207, so stage
  two started from trained weights. The code has no residual target. Its rows are the finished runs of 2026-11-19 (§18).
- **The line term.** The SL score of a row (line `c`, pair A, B) is

  ```text
  s(c, A, B) = f(c, A, B) + D(c, A, B) − mean over r in R of D(r, A, B)
  ```

  - `f(c, A, B)`: the model's own score, the pair head of the chosen phase read in line `c` (its tokens, all its cells
    at scoring). At a training step, `f` and `D` of the row's line read the same drawn tokens.
  - `D(r, A, B)`: the line head, a second MLP. It reads the pair head's input for the pair in line `r` and the
    gene-in-line features of A and B in line `r`. Its last layer starts at zero, so the untrained line term is 0.
  - `R`: the SL training lines with training rows (genetic screens with both classes). At a training step, `R` is a
    draw of `line_term.reference_lines` of them, with replacement. At scoring, `R` is all of them, with all their cells.
  - The line term trains last, on the chosen phase, with binary cross-entropy on `s` and the rows of stage two. Only
    the line head and its input projections train, so `f` does not change. Validation recall in the top 10% of `s`
    selects the epoch (`line_term.max_epochs`, `patience`, `learning_rate`).
  - The line term is a deviation from the SL training lines' mean. Thus the gene pair alone cannot fill it. The
    line's gain is `s − f`.
  - Changed on 2026-11-20 after the first runs: the base was the pair score on the mean input; that score is weak for
    a model that reads the line (recall in the top 10% of 0.466 / 0.430 against 0.553 / 0.496 for the model with
    Tx1-3B cell tokens), and the line term on it scored 0.288 / 0.279.
- **Gene-in-line features.** Each source gives one vector for each (line, gene). A linear projection takes it to
  width 32, the sources are summed to `e_g`, and the line head reads `[e_A + e_B ; |e_A − e_B| ; e_A ⊙ e_B]`.
  - *Own expression* (`line_term.own_expression`): the line's scaled bridged pseudo-bulk value of the gene and a
    presence flag, built at load time.
  - *Tx1-3B gene level* (`gene_in_line.tx1_gene`): the mean over the line's cells of Tx1-3B's contextual embedding of
    the gene (2,560 wide) and the share of the cells whose input sequence holds the gene.
  - *PIE change* (`gene_in_line.pie_change`): PIE's predicted change of the line's profile after the gene's loss, in
    64 components fitted on the SL training lines.
  - The caches are written by `hpc/run.sh sl-gene-in-line configs/sl/gene_in_line.yaml --steps tx1_gene --shard I/N`
    (one process for each GPU) and `hpc/run.sh sl-gene-in-line configs/sl/gene_in_line.yaml --steps pie_change` under
    `data/sl_gene_in_line/v1/` ([card](data/gene-in-line-features.md)). Each cache holds `genes.json` and one
    `<ModelID>.npy` for each line (genes × width, float16, a NaN row where the line has no value). Each column is
    centred and scaled on every SL training line with benchmark rows. A gene without a value reads zeros and a
    presence flag of 0. The run stops if a line with benchmark rows has no file in a cache, or if a scaled value is
    outside the float16 range.
- **Rows of the study.** One config each, seed 0, run id `<config name>_seed0_v2` (the `_seed0` runs of 2026-11-20
  used the old base, the pair score on the mean input). The configs are
  `configs/sl/embedding.yaml` with a `line_term` section (8 reference lines, at most 30 epochs, patience 5, learning
  rate 1e-3, own expression on):

  | Config (`configs/sl/`) | Cell tokens | Gene-in-line features |
  | --- | --- | --- |
  | `embedding_direct.yaml` | Tx1-3B | own expression |
  | `embedding_direct_mean_token.yaml` | none | own expression |
  | `embedding_direct_pie_change.yaml` | Tx1-3B | own expression, PIE change |
  | `embedding_direct_tx1_gene.yaml` | Tx1-3B | own expression, Tx1-3B gene level |

  The rows with the residual target are the finished runs `embedding_seed0`, `embedding_mean_token_seed0` and
  `embedding_cellnet_seed0` (§18). They are not run again; they are the rows without pretraining.
- **Outputs.** Each run writes five score tables under `outputs/sl/embedding/<run id>/scores/`:
  `embedding.parquet` (the model of the chosen phase), `embedding_mean_input.parquet` and
  `embedding_other_line.parquet` (the line-input controls of §18), `embedding_line_term.parquet` (`s`) and
  `embedding_line_term_other_line.parquet` (`s` with `f` and `D` read in another line, the next one in a seeded order;
  `R` does not change). `metrics.json` has two keys for the line term: `sl_line_term` (SL metrics of `mean_input`,
  `line_term` and `line_term_other_line`) and `line_term` (the validation recall and the history of each epoch).
  Both are null without a line term. `results.md` reports the line term in its own SL table and after the phases of stage two. A `best.pt` written before
  this change has no line head and does not load; the finished runs keep their score tables.
- **Measures** (`configs/sl/benchmark_line_term.yaml`; scorer keys `direct`, `direct_mean_token`,
  `direct_pie_change`, `direct_tx1_gene`, each with `_mean_input`, `_other_line`, `_line_term` and
  `_line_term_other_line`). Recall in the top 10%, AUROC and AUPRC over the positive rate, on all pairs, validation
  and test. Validation selects; test is scored once.
  - *The line's gain:* in each run, the model with the line term minus the context-ablated model, minus itself on
    another line's input, and minus the model without the line term.
  - *What the cells add:* Tx1-3B cell tokens minus no cell tokens (`direct` against `direct_mean_token`, with and
    without the line term); the PIE row and the Tx1-3B gene-level row minus `embedding_direct`, with the line term.
  - *What pretraining on the direct target adds:* `direct` and `direct_mean_token` minus the finished runs of
    2026-11-19 with the same inputs (`embedding`, `embedding_mean_token`).
  - The model with the line term minus Kebabci's paralog classifier on predicted GeneEffect.
  - The same differences on the harness's row sets: pairs on which the training lines differ, pairs that no training
    line called SL, new pairs. The stage-one numbers of each model come after the SL tables.
- **Limits.** The direct target is mostly the gene mean: the line-specific part gets a smaller share of the gradient,
  and the early stop mostly follows the fit of the gene means. The row set "the training lines differ" has 13 / 10
  held-out lines, so a gain below about 0.05 will not show. The line-specific part of the labels repeats at 0.35 to
  0.39 inside Harle 2025 and at 0.04 to 0.12 between two studies (§15, §16). PIE was trained on 42 of the project's
  lines. A549 (ACH-000681) is one of them; it is a training line of the SL split. Which SL validation or test lines
  are among the 42 is not recorded. Tx1-3B's exposure to the project's lines is not recorded.
  In these runs one SL training line, ACH-000696, had no Tx1-3B gene-level value and degenerate Tx1-3B cell
  embeddings: its raw file lists Ensembl IDs with a version suffix, which Tx1-3B's vocabulary does not hold
  ([card](data/gene-in-line-features.md)). It is one of the 53 SL training lines of the feature scaling. Its
  Tx1-3B caches were encoded again on 2026-11-23 (code `63f5010`), and the three runs that read them were trained
  again (`_seed0_v3`, harness `benchmark_cache_repair_seed0`; the run without cell tokens, trained again, gives the
  same scores to the last digit). No conclusion changes: the model scores 0.553 / 0.499, each new − old difference is
  inside ±0.02 (intervals across zero except the mean-input control on validation, −0.009), and the model minus the same weights on the mean input is +0.100 / +0.069
  ([record](../results/sl_line_term_seed0/summary.html), Table 11).
- **Not in this study.** CellNet's gene-level outputs, in-silico deletion with Tx1-3B, adapter training of Tx1-3B,
  and any other foundation model.
- **Runs** (seed 0, 2026-11-20, [record](../results/sl_line_term_seed0/summary.html)). Caches on H20 port 30838:
  `pie_change` (code `d4d1784`; 86 lines, 64 components fitted on 200,000 vectors of the 53 SL training lines) and
  `tx1_gene` (code `f6e242f`; 86 lines, at most 256 cells each, four GPUs, about 30 s for each line). Runs on port
  30757, code `4ad7dd4`, one GPU each: `embedding_direct_seed0_v2`, `embedding_direct_mean_token_seed0_v2`,
  `embedding_direct_pie_change_seed0_v2`, `embedding_direct_tx1_gene_seed0_v2`; then the harness run
  `benchmark_line_term_seed0_v2`. The two `_seed0` runs (code `d4d1784`) used the first base of the line term.
- **Result, all pairs.** Recall in the top 10%, validation / test, mean over lines (17 / 16):

  | Model | Model | On the mean input | On another line's input | With the line term |
  | --- | --- | --- | --- | --- |
  | Direct target, Tx1-3B cell tokens; own expression in the line term | 0.553 / 0.496 | 0.466 / 0.430 | 0.512 / 0.490 | 0.553 / 0.497 |
  | Direct target, no cell tokens; own expression in the line term | 0.545 / **0.505** | 0.398 / 0.399 | 0.485 / 0.471 | 0.549 / **0.505** |
  | Direct target, Tx1-3B cell tokens; PIE change in the line term | 0.553 / 0.496 | 0.466 / 0.430 | 0.512 / 0.490 | **0.555** / 0.501 |
  | Direct target, Tx1-3B cell tokens; Tx1-3B gene embeddings in the line term | 0.553 / 0.496 | 0.466 / 0.430 | 0.512 / 0.490 | 0.507 / 0.484 |
  | Residual target, Tx1-3B cell tokens (§18) | 0.500 / 0.485 | 0.509 / 0.498 | 0.447 / 0.496 | — |
  | Residual target, no cell tokens (§18) | 0.508 / 0.501 | 0.460 / 0.485 | 0.456 / 0.491 | — |

  Paired differences, 95% line bootstrap, validation / test:
  - Direct minus residual target: +0.052 [+0.019, +0.088] / +0.011 [−0.027, +0.055] with cell tokens;
    +0.036 [+0.009, +0.067] / +0.004 [−0.018, +0.025] without. The gain holds on validation only.
  - The model with the line term minus its row on the mean input: +0.088 [+0.049, +0.128] / +0.067 [+0.015, +0.119]
    with cell tokens; +0.150 [+0.090, +0.218] / +0.107 [+0.039, +0.183] without. Minus its row on another line's
    input: +0.045 [+0.007, +0.083] / +0.012 [−0.007, +0.030] with cell tokens;
    +0.062 [+0.032, +0.099] / +0.032 [+0.005, +0.067] without. The score reads the line; against another line's
    input the test interval is above zero only without cell tokens.
  - The line term minus the model: +0.001 [−0.005, +0.006] / +0.001 [−0.004, +0.005] (own expression);
    +0.002 [−0.001, +0.005] / +0.005 [−0.000, +0.010] (PIE change);
    −0.046 [−0.073, −0.018] / −0.012 [−0.035, +0.007] (Tx1-3B gene embeddings). The line term adds nothing, and with
    the Tx1-3B gene embeddings it fits the training lines (recall 0.914) and lowers validation.
  - Cell tokens minus no cell tokens: +0.008 [−0.011, +0.027] / −0.009 [−0.039, +0.020].
  - The model with the line term minus the Kebabci classifier: +0.198 [+0.121, +0.274] / +0.102 [+0.054, +0.149].
- **Result, row sets** (the model with Tx1-3B cell tokens; recall in the top 10%, validation / test). Seen pairs
  0.560 / 0.520; new pairs 0.548 / 0.518 (residual target 0.475 / 0.354; the difference is
  +0.073 [−0.010, +0.159] / +0.164 [−0.048, +0.413]); pairs that no training line called SL 0.443 / 0.330; pairs on
  which the training lines differ 0.438 / 0.627. On the pairs on which the training lines differ, the model with the
  line term minus its row on the mean input is +0.142 [+0.029, +0.255] / +0.211 [+0.121, +0.288], and minus its row on
  another line's input −0.025 [−0.095, +0.036] / +0.029 [−0.002, +0.059].
- **The first base of the line term.** With `p`, the pair score on the mean input, as the base (the `_seed0` runs),
  the model with the line term scored 0.288 / 0.279 with cell tokens and 0.277 / 0.314 without, below `p` itself
  (0.466 / 0.430 and 0.398 / 0.399). `p` is a weak base for a model that reads the line, and a deviation with zero
  mean over the lines cannot repair it.
- **Stage one, for the record.** Per-line residual Spearman of the model's GeneEffect on the SL lines:
  0.033 / 0.033 with cell tokens and 0.070 / 0.070 without. It decides nothing.
- **Reading.** Stage one on GeneEffect itself gives stage two trained weights: the head on the frozen pretrained model
  scores 0.425 (0.464 without cell tokens) on validation, against 0.109 in §18. The score now changes with the line.
  The all-pairs gain over the residual target holds on validation and not on test. A line term on gene-in-line
  features of the cells adds nothing; cell tokens add nothing. One seed.

## 20. The cross-line loss and local gene-in-line data (2026-11-24)

`src/model/sl_embedding.py`, `src/experiments/sl/embedding.py`, `src/experiments/sl/gene_in_line.py`,
`src/experiments/sl/benchmark_baselines.py`, the five configs below, `configs/sl/benchmark_cross_line.yaml`;
the [design](specs/2026-11-24-context-specific-sl-architecture-design.md) (§5, §6) gives the reasons and the
[plan](specs/2026-11-24-cross-line-loss-plan.md) the code changes. The question: does a loss that the gene pair
alone cannot lower make the score follow the line, and does that hold in unseen lines? The model, stage one and the
score `s = f + D − mean over R of D` are those of §19.

- **The cross-line loss** (`line_term.cross_line: {weight, examples_per_step}`, `null` for none). An example is two
  SL training lines of one study with a training row for the same pair and design: one line called the pair SL
  (`c⁺`), the other did not (`c⁻`). The loss is `softplus(−[s(c⁺) − s(c⁻)])`; the mean over `R` is the same in the
  two lines and cancels, so it is computed on `f + D`. A score that is the same in every line gives `log 2` on every
  example. A step draws `examples_per_step` examples, each from a study drawn evenly; the loss is added to the
  binary cross-entropy on `s` at `weight`. The run stops if no study has an example.
- **Two training modes** (`line_term.joint`). `false`: the line term trains last on the frozen model, as in §19.
  `true`: in the phase that trains the whole model, the score is `s` and the model, the pair head and the line term
  train together on the binary cross-entropy, the cross-line loss and the GeneEffect loss; no separate line-term
  phase follows. Validation recall in the top 10% of `s` selects the epoch and the phase.
- **What the line head reads** (`line_term.input`). `model`: the pair head's input (the line's tokens and the free
  gene deviation `r_g`) and the gene-in-line features, as in §19. `genes`: the gene vectors of the frozen sources
  alone (the gene encoder's output without `r_g`) and the gene-in-line features,
  `[u⁰_A + u⁰_B ; |u⁰_A − u⁰_B| ; u⁰_A ⊙ u⁰_B ; combinatorial ; e_A + e_B ; |e_A − e_B| ; e_A ⊙ e_B]`, so the
  deviation must go through features of the two genes in the line. The reason is §19's record: a line term with
  capacity fitted the training lines.
- **New gene-in-line caches** (`hpc/run.sh sl-gene-in-line configs/sl/gene_in_line.yaml --steps <step>`, CPU;
  [card](data/gene-in-line-features.md)). Each is read as §19's caches are: columns scaled on the SL training lines
  with benchmark rows, one presence flag for each (line, gene). `wide_expression`: `log1p(CP10K)` of the line's
  pseudo-bulk over the 19,277 source genes without the bridge, and a flag for a value at or below the gene's 10th
  percentile over the SL training lines. `cell_spread`: the share of the line's cells that detect the gene, and its
  SD minus the depth-expected SD. `paralog_buffer`, `complex_buffer`, `partner_low`: the partner columns of the
  linear model's `knowledge_features` on the wide expression. `depmap_mutation`: DepMap's damaging and hotspot
  mutation values. `depmap_copy_number`: DepMap's `log2(relative copy number + 1)` and a flag for a log2 ratio
  below −0.8. A line without the source has a file of absent rows. The mean token still reads the 6,523 bridged
  genes.
- **Measures added to the harness.**
  - *The other-line control over four lines* (`eval.other_lines: 4`, harness `scorer_means`): the score is read in
    the input of each of four seeded other benchmark lines, and the control's metrics are the mean over the four.
  - *Cross-line ordering* (harness `cross_line_ordering`): over the examples as above with both lines on one side
    (training, validation or test), the share with `s(c⁺) > s(c⁻)` (a tie counts one half), mean over studies, with
    an interval from resampling gene pairs inside each study. A score that is the same in every line gives exactly
    0.5. Both lines are held out because a training line's score is in sample. The table also gives the number of
    examples and studies on each side.
  - *Two metrics with the same random value in every screen* (since 2026-11-25, harness run `_v2`). *Recall at 10%
    false positives*: the share of a screen's SL pairs above the top 10% of its other pairs (a tie block counts in
    part); a random ranking gives 0.10 at every positive rate, where recall in the top 10% gives the positive rate
    when it is above 0.10. *Chance-corrected AUPRC*: (AUPRC − positive rate) / (1 − positive rate), 0 for a random
    ranking and 1 for a perfect one; it replaces AUPRC over the positive rate in the harness, whose top value was
    1 / positive rate. Both are averaged over screens, studies, then lines. The harness report gives every metric on
    the test lines in one table, with the cross-line ordering of every scorer named in the config.
- **Rows of the study.** One config each, seed 0, run id `<config>_seed0`, without cell tokens; each row is the row
  above with one change:

  | Config (`configs/sl/`) | The change |
  | --- | --- |
  | `embedding_direct_mean_token.yaml` (finished, §19) | the reference |
  | `cross_line_frozen.yaml` | the cross-line loss (weight 0.5, 256 examples at a step); the line term trains last |
  | `cross_line_joint.yaml` | `joint: true` |
  | `cross_line_genes_input.yaml` | `input: genes` |
  | `cross_line_wide.yaml` | the wide caches in place of the bridged own expression (the scRNA-only row) |
  | `cross_line_depmap_dna.yaml` | DepMap mutations and copy number (the multi-omic row) |

- **Not in this study.** The 36 bulk-only lines with a two-class genetic screen as training lines (32 labelled, 4
  unlabelled bulk lines of the split; `benchmark_rows` drops them), Tx1-3B cell tokens, the ramp target, and every
  deferred part of the design (§7 there).
- **Limits.** The line-specific part of the labels repeats at 0.35 to 0.39 inside one study (§16), which bounds the
  ordering; its top value is not measured. One seed.
- **Result** (2026-11-24, [record](../results/sl_cross_line_seed0/summary.html)). Recall in the top 10% of the five
  runs is 0.525 to 0.555 validation / 0.495 to 0.524 test, equal to the reference (0.549 / 0.505; the loss with the
  line term trained last minus it +0.006 [−0.023, +0.035] / +0.008 [−0.027, +0.045]). Own input minus another line's
  input (mean over four lines) is positive in every run (+0.027 to +0.052 / +0.026 to +0.062). The cross-line
  ordering is 0.74 to 0.99 on the training lines and 0.50 to 0.55 on held-out lines; no run is above 0.5 on
  validation and on test, and the reference on another line's input reaches 0.572 on test. The loss fits (line,
  pair) on the 44 training lines and does not carry to unseen lines.
- **Runs** (seed 0, 2026-11-24, code `b2de0c9`, H20 port 30838). The seven cache steps ran on CPU (about 6 s for
  each cell-file step); their counts are in the [card](data/gene-in-line-features.md): every one of the 86 labelled
  lines has a cell file, 79 have DepMap mutation calls and 65 have DepMap copy number. The reference run's four
  other-line controls were written from its `best.pt`. The five runs were started with `--device cpu`, because
  every GPU of both containers was in use by another project; the harness run `benchmark_cross_line_seed0` starts
  when the five runs have finished. No result yet.

## 21. More training lines, a wide mean token and pair-in-line features (2026-11-25)

`src/experiments/sl/embedding.py`, `src/model/sl_embedding.py`, `src/experiments/sl/gene_in_line.py`,
`src/experiments/sl/target_options.py`, the three configs below, `configs/sl/benchmark_context_rows.yaml`; the
[design](specs/2026-11-24-context-specific-sl-architecture-design.md) (§5.4, §5.5, §6) gives the reasons. §20 left
one row of the study and two local changes open. Each is one config here, and each is the wide row of §20
(`cross_line_wide.yaml`) with one change. The question of each row is the question of §20: does the score order a
pair between two unseen lines?

- **The bulk-RNA training lines** (`bulk_training_lines: true`, `cross_line_80_lines.yaml`). The split file's
  bulk-only lines (`bulk_labelled`, `bulk_unlabelled`) with a labelled row join stage two's training lines: 37
  lines, of which 36 have a genetic screen with both classes (32 labelled, 4 unlabelled; 80 training lines with
  such a screen in place of 44). They share studies with the single-cell training lines (Flister 2025, Klingbeil
  2024, Ito 2021, Ebright 2026, Najm 2023, Harle 2025) and add Chan 2025, so they add cross-line examples to those
  studies. Such a line has no cells.
  - *The mean token* reads the line's DepMap bulk RNA on the mean token's genes, quantile-normalised to the training
    side's profile, as stage one's bulk-only fit lines are. 5 of the 36 lines are myeloid lines, which the linear
    model's training side leaves out, so the runner reads the bulk matrix for these lines itself.
  - *The wide expression* (`wide_expression`, and the buffer caches on it) is the bulk RNA mapped onto the
    pseudo-bulk gene by gene: a least-squares line for each gene, fitted on the SL split's training lines that have
    a cell file and bulk RNA (410 lines); a mapped value below 0 is 0. `cell_spread` is absent. The 10th percentiles
    stay those of the SL training lines with a cell file, so the caches of the other lines do not change.
  - *Stage one does not change.* Its fit lines hold no line with an SL label (928 lines), and its early-stop lines
    are single-cell lines.
  - *What the row does not show.* The model can tell these lines from the single-cell lines (no spread over cells,
    another kind of expression), so the row reads as "more lines of another kind", not as a clean sample-size
    effect. The harness reads the split's 44 training lines alone: a held-out pair that only the bulk-RNA lines
    hold is a new pair in the harness, and the run's `results.md` gives the count of such rows. Another line's
    input in the other-line control can be a bulk-RNA line's.
- **The wide mean token** (`space_measured_share: 0.5`, `cross_line_wide_token.yaml`). The mean token reads the
  bulk genes that at least half of the labelled single-cell training lines measure in pseudo-bulk (about 18,300),
  not the 6,523 genes that every held-out line measures (`null`). It is the space of §13
  (`conditional_wide.yaml`): a value that a line does not measure takes the training mean, the bridge is fitted on
  the same genes, and stage one's bulk-only fit lines read the same genes of their bulk RNA. Stage one trains again
  in the run, as in every run.
- **The pair-in-line features** (`line_term.pair_in_line: true`, `cross_line_pair_in_line.yaml`;
  `src.model.sl_embedding.pair_in_line`). The gene-in-line features describe each gene alone; these 12 values
  describe the pair in the line, and the line head reads them beside its other inputs:
  - the largest and the mean wide expression of each gene's paralogs other than the pair's other gene (for a
    paralog pair, the buffer that is left when both genes are lost), as the sum and the absolute difference over
    the two genes, and the number of the two genes that have such a paralog;
  - over the partners that A and B share (paralogs and CORUM co-members of both): their mean wide expression, the
    share of them that is silenced, their mean predicted dependency, and whether there is one;
  - silenced(A) × z(B) + silenced(B) × z(A), where z is the predicted dependency;
  - for an anchor row (A is the anchor): silenced(A) × z(B) and z(B).

  The expression and the silenced flag are those of the `wide_expression` cache. The paralog and complex tables are
  the linear model's reference tables (at most 10 paralogs for each gene). The predicted dependency is the model's
  own stage-one output for the gene in the line, read without a gradient: it is out of sample for every SL line.
  A partner counts only where the line measures it. The features are the same for (A, B) and (B, A) on a
  combinatorial row. No STRING partner list is read, and no linear-model GeneEffect (that needs a wide linear model
  refitted without every SL line).
- **The top of the cross-line ordering** (`sl-target-options`, Table 2 of its report). Inside Harle 2025, the SL
  calls of one replicate are the labels and the score of another replicate is the score (the six ordered pairs of
  replicates), and the mean of replicates 1 and 2 is scored against the calls of replicate 3. The value is the
  ordering that a second measurement of the same lines reaches: a model that follows the line cannot be expected
  above it in this study. The benchmark's own calls come from the three replicates together, so their noise is
  below one replicate's, and the true top is at or above the value with the mean of two.
- **Rows.** One config each, seed 0, run id `<config>_seed0`:

  | Config (`configs/sl/`) | The one change against `cross_line_wide.yaml` |
  | --- | --- |
  | `cross_line_80_lines.yaml` | `bulk_training_lines: true` |
  | `cross_line_wide_token.yaml` | `space_measured_share: 0.5` |
  | `cross_line_pair_in_line.yaml` | `line_term.pair_in_line: true` |
