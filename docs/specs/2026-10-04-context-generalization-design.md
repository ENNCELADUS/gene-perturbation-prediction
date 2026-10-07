# Design: a linear context prior and a single-cell correction for held-out-line GeneEffect

**Status:** design, 2026-10-04, agreed section by section in a brainstorming session. Supersedes the
2026-10-03 context-generalization proposal (untracked, removed). Evidence: the literature review
(`docs/notes/2026-10-03-context-generalization-litreview.md`, about 28 papers read at methods level) and the
DepMap survey (`docs/notes/2026-10-03-depmap-context-data-survey.md`, tables in
`docs/notes/2026-10-03-depmap-derived/`), both in the gitignored vault. Follows the [research blueprint](../01-blueprint.md)
and the [GeneEffect protocol rules](../03-geneeffect-protocol.md#11-rules) (a query line supplies basal single cells
only; the rules for extra lines are in §9 below). Builds on
the [head revision](2026-10-03-geneeffect-head-revision-design.md): its nested low-rank head becomes the
single-cell correction.

## 1. Why

The limit is the number of labelled contexts, not the expressiveness of the context encoder.

| Evidence | Source |
| --- | --- |
| 170 labelled training lines give 170 distinct contexts; every learned context map keys line identity: trunk MLP over Tx1 selective Spearman 0.53–0.67 train / 0.15 validation, Tx1 GMM64 ridge 0.66 / 0.078; the eight-component Tx1 context-PCA ridge reaches validation residual Pearson 0.133 | [head revision design](2026-10-03-geneeffect-head-revision-design.md) §1 |
| DepMap's predictability model refit on line subsets: mean per-gene Pearson 0.071 (100 lines), 0.107 (200), 0.125 (400), 0.156 (700), 0.169 (1,021) | survey §5, figshare 26955886; random CV on raw scores, so the effect overstates our protocol |
| Per-gene linear models match or beat shared deep models on held-out lines (LEAP: per-gene LASSO Spearman 0.321 vs pan-perturbation LightGBM 0.309); a shared deep model trained on lines alone is near chance (DeepDEP without pretraining, per-gene Pearson 0.08) | review Q1 |
| Representation barely matters at fixed n: PCA within about 4% of the best deep representation (Gross 2024); the NCI-DREAM winner on 35 lines was a Bayesian multitask multi-view kernel model | review Q1, Q2 |
| In 29–50% of predictable dependencies the key feature is the target's own, a paralog's, a PPI partner's or a complex co-member's expression (Dempster 2020) | review Q1 |
| Expression is the top feature for 93% of genes in the 1,021-line predictability model; copy number 4.9%, damaging mutation 1.5%, lineage 0.2% | survey §5 |

The data has two axes of very different size: about 17.8k genes but 170 single-cell lines. Deep capacity is
cheap on the gene axis and expensive on the context axis. So the design adds information that does not pass
through the 170-line bottleneck: more labelled lines through bulk RNA, context encoders fitted on data that needs
no GeneEffect label, and gene-pair knowledge.

## 2. Decisions

| Topic | Decision |
| --- | --- |
| Extra lines | DepMap lines with bulk RNA and no single cells may enter training; whether their GeneEffect labels train the prior is read from the learning curve (§8.1) |
| Model shape | Two-stage stack: a cross-fitted **linear context prior** (closed form, CPU) predicts the residual; the existing nested head becomes the **single-cell correction**, trained on the 170 single-cell lines against the residual left by the prior |
| Context encoders | One encoder per view, each fitted where that modality's data is plentiful, never on the 170 labelled lines alone |
| Fusion | Linear in the context. In the prior: per-gene ridge with one penalty per view, gene-conditioned view weights as a candidate. Context-dependent cross-attention only as a correction candidate |
| Hyperparameters | Penalties are tuned on validation by point estimate; which blocks and settings to keep is read from the reported rows (§7.1) |
| Fitting scope | What defines the metric (selective genes, `mu_hat`, σ_g) stays on the 170 labelled training lines; model-side fits use the training side |
| Virtual cell | STATE's predicted response enters the correction, projected onto program scores; the counterfactual SL composition is recorded as an interface (§6), not run |
| Large files | Files over 5 GB are downloaded and processed only on the H20 host |

## 3. Data

### 3.1 Lines

| Side | Lines | Use |
| --- | --- | --- |
| Single-cell training lines | 170 labelled (172 members; PC9 and HeLa unlabelled); 167 have 26Q1 bulk RNA (COLO 205, Evsa-T and JHU-011 do not) | prior and correction |
| Extra labelled lines | 919: 26Q1 CRISPRGeneEffect and bulk TPM, outside the 226; 133 haematopoietic, kept | prior, once the curve passes; encoders always |
| Extra unlabelled lines | 576 with bulk RNA and no GeneEffect | label-free encoders (expression components) and, with their mutation and lineage labels, supervised encoders |
| Validation, test | 27 + 27, unchanged; 26 and 27 have bulk RNA | scored only; their bulk RNA is read by the oracle diagnostic alone (§8.3) |

Exclusions apply to every fit, labelled or not: validation and test lines and every line sharing a `PatientID`
with one of them (10 lines on 26Q1, each listed with its reason in the membership file; ACH-000023, for example,
shares a patient with validation line ACH-000022). Extras that share a patient with a training line stay on the
training side, in that line's cross-fitting fold. The training side then holds 1,664 lines with bulk RNA, 1,089 labelled (1,086 with
bulk RNA), 1,907 with mutation calls and 2,077 with a lineage.

The 919 extras become a tracked membership file next to the split,
`configs/benchmarks/extra_bulk_lines_26Q1.json`, written by a builder under `src/data/prepare/` from the pinned
26Q1 files, with each exclusion and its reason. It is a pinned list like the split, not a rule computed from
labels. One run of the prior drops the haematopoietic extras.

### 3.2 Shared expression space and the bridge

- **Genes:** the 26Q1 bulk matrix's protein-coding genes that every validation and test line's single-cell source
  measures, joined by symbol, read from DepMap's `SYMBOL (Entrez)` column headers (not by Entrez ID); 9,711 genes
  on the current sources. The sources differ in gene vocabulary: the 47 original training contexts measure 7,715
  to 18,467 genes, so requiring every line would leave 4,425. A training line whose source lacks a space gene
  takes that gene's mean over the single-cell training lines that measured it, before quantile normalisation;
  the run records how many values each line took (`space.json`).
- **Bulk side:** DepMap `OmicsExpressionTPMLogp1HumanProteinCodingGenes`, default entry per model.
- **Single-cell side:** pseudo-bulk per line: raw UMI summed over all of the line's basal cells, CPM, log1p.
  Computed by preparation (the only reader of raw data) into the prepared root and recorded in its manifest,
  with the expression source of each line.
- **Normalisation:** every line on both sides is quantile-normalised to one reference distribution, the mean
  sorted bulk profile of the training side.
- **Bridge:** one affine map per gene from normalised pseudo-bulk to normalised bulk, fitted by least squares on
  the 167 training lines that have both, and applied to every single-cell line. A per-gene map is required: 3′ UMI
  counts carry no gene-length normalisation and TPM does. A gene is constant when its range across lines is zero
  (not when a floating-point variance is positive); a constant pseudo-bulk gene maps to its bulk mean, and the
  bridge-quality correlation (§8.2) is undefined for a gene constant on either side. Remedies for a lossy bridge (Celligner-style contrastive PCA
  alignment, gating by bridge quality, noise-matched fitting, low-rank denoising) are compared as experiments of the runner (§7.2),
  not as code fallbacks.

### 3.3 Context data

Every view a query line can supply is computed from its own basal cells; for bulk lines the same function is
applied to bulk RNA. Design rule: **the prior reads only what bulk and pseudo-bulk share, derived by the same
function on both sides; the correction reads only what single cells add.**

| View | Content | Encoder fitted on | Enters |
| --- | --- | --- | --- |
| Expression components | 128 principal components of normalised expression, eigen-scaled as the current context PCA | all 1,664 training-side bulk lines, label-free | prior |
| Pathway and program scores | MSigDB hallmark (50; mean within-line rank of the set's genes, centred) and PROGENy (14 pathways; weighted sum over each pathway's top-100 footprint genes) | nothing: fixed published gene sets | prior |
| Predicted genotype | From the 128 expression components: L2 logistic regressions to driver status and lineage, ridge to MSI score. Drivers are the genes of DepMap's hotspot matrix with a hotspot or damaging mutation in at least 5% of training-side lines, a rule that never reads GeneEffect | training-side lines with the label: mutations 1,907, lineage 2,077, MSI from `OmicsSignatures` | prior |
| Inferred arm-level copy number | expression smoothed along chromosome positions against the training-side reference, summarised to 39 arm scores | checked against Cohen-Sharir arm calls and 26Q1 WGS copy number on training lines | prior, second wave |
| Single-cell state | Tx1 context components (existing); fraction of cells scored S or G2/M with the Tirosh cell-cycle sets; for each Kinker and Gavish heterogeneity program, the fraction of cells above the training-side 90th percentile of per-cell scores | Tx1 pretrained on Tahoe-100M; gene sets fixed; percentiles from training lines | correction |

**Out-of-sample rule:** a feature produced by a supervised encoder (predicted genotype, inferred copy number once
calibrated) is always out-of-sample for the line it describes. Training-side lines get inner cross-fitted
predictions; validation and test get predictions from the encoder fitted on the whole training side. Label-free
encoders (expression components, fixed gene sets) are fitted once.

### 3.4 Gene-pair knowledge

Pinned with provenance under `configs/context_prior/reference/`, built on the Mac by
`src/data/prepare/build_context_reference.py` and committed, because the H20 host has no internet: Ensembl human
paralogs with % identity (the lower of the two directions), keeping each gene's 10 closest at 20% identity or
more, and CORUM's current human release of complexes. Data-selected partners (§4) are computed from
training-side lines only, inside each fold.

### 3.5 Fitting scope

The metric's definitions are fitted on the 170 labelled training lines and do not change: the selective genes
(3,111), `mu_hat` behind the residual target and σ_g. Every number stays comparable with earlier runs. The
bridge, encoders, partner lists and ridges are fitted on the training side. Every training-side line has equal
weight.

### 3.6 Not used

Proteomics, RPPA, metabolomics, methylation, miRNA, chromatin, DEMETER2, PRISM and screen confounders: little
shown gain over RNA, partial coverage, no query-time route, and a linear stack has no auxiliary head to consume
them. TCGA tumour expression as unlabelled data for the expression components is deferred (DeepDEP's tumour
pretraining helped, Gross 2024 found pretraining insignificant, and tumours add a purity shift). Lineage, mutation
calls or WGS copy number as query-time inputs would change the contract and are not used.

## 4. Features for each gene in each line

Prior features live in the shared expression space (bulk for bulk lines, bridged pseudo-bulk for single-cell
lines). Each knowledge feature is z-scored per gene with training-side statistics. A feature that does not apply
(no paralog, no complex, gene not measured) is undefined and takes the training mean, so it contributes nothing;
there is no mask bit, because the per-gene intercept makes one redundant. A panel gene with no column in the
shared expression space has undefined own-expression and partner features and is still predicted from context.

| Block | Definition |
| --- | --- |
| Own expression | g's expression in c |
| Paralogs | min and sum of expression over the gene's 10 closest paralogs (at 20% identity or more); expression of the closest by % identity |
| Complex partners | mean expression of CORUM co-members over every complex containing g |
| Low-partner fraction | fraction of g's paralogs and complex partners in the bottom expression decile of the training side (ISLE's cSL score) |
| Data-selected genes | the N genes (N from 10, 50, 200; g excluded) whose expression has the largest absolute Pearson correlation, across training-side lines, with the residual GeneEffect that g has left after the earlier blocks (§5.1, §7.1) |
| Context views | §3.3, prior rows |

**Sharing.** The four knowledge blocks (own expression, paralogs, complex partners, low-partner fraction) have
weights pooled across genes plus a ridge-shrunk per-gene deviation; one shrinkage strength spans "pooled only" to
"per gene only". Data-selected genes and context views have per-gene ridge weights. The proposal's
"co-essential partner expression" is replaced by data-selected genes, which are chosen directly on g's
dependency.

**Inside each fold** every selected or ranked quantity is recomputed: data-selected genes, expression deciles,
ridges and supervised encoders. Otherwise out-of-fold prior predictions are optimistic and the correction learns
the wrong scale.

**Correction features** (single-cell lines only): the existing `q_sc` statistics of g, the fraction of cells
expressing g's closest paralog and its complex partners (gathers from the `q_sc` cache), and the single-cell
state views.

**Coverage.** The prior predicts every gene in the gene order (17,787), so Huber and residual Pearson stay
defined; the selector scores the 3,111 selective genes.

## 5. Model

$$\hat y(g,c)=\mu_g+\hat r_{\text{prior}}(g,c)+\sigma_g\Big[\tfrac{1}{\sqrt r}\big\langle G(g),\,C(v_c)\big\rangle+h(\cdot)\Big]$$

### 5.1 Linear context prior

A stagewise ridge on the residual in units of σ_g: the blocks of §7.1 are fitted one after another, each to
what the earlier blocks leave, and the prior is the sum of the stage predictions. Every stage has a per-gene
intercept and each block its own penalty, multiplied by the number of fitted lines so that one grid serves every
training-set size; a constant feature column gets scale 1 and contributes nothing. The knowledge blocks have
pooled weights by least squares over every (g, c) row, then, per gene, the deviation from them (penalised by the
shrinkage strength); the data-selected and context-view blocks have per-gene ridge weights. A missing training
label (DepMap leaves some line × gene pairs empty) counts as zero residual in fitting and is skipped in scoring.

- **Inputs.** Fitted on bulk RNA for every training-side line that has it. Every single-cell line is predicted
  from bridged pseudo-bulk: out-of-fold training lines, validation and test, so the correction trains on
  residuals under deployment conditions.
- **Cross-fitting.** Five folds over the training side, assigned deterministically from seed 0, each patient in
  one fold. Bridge, supervised encoders, data-selected genes and ridges are refitted inside each fold; training
  single-cell lines get their fold's out-of-fold prediction; validation and test get the prior refitted on the
  whole training side.
- **Extra lines.** The bridge and encoders are always built (the encoders are fitted on bulk). Whether the extra
  lines' GeneEffect labels train the prior is read from the learning curve (§8.1); the alternative trains the prior
  on the 167 single-cell training lines with bulk RNA.
- **Candidates** (read from the reported rows, §7.1): a reduced-rank context term (gene factors fixed to the top k right
  singular vectors of the training-side residual GeneEffect matrix, as in Webster, and a ridge map from the
  context views to the k factor scores), k from {per-gene, 64}; and gene-conditioned view weights, a stacker that
  mixes the per-view out-of-fold predictions with weights softmax over views of a linear map of the gene's
  embedding, trained by gradient, the prior's only learned variant.

### 5.2 Single-cell correction

The nested low-rank head of the head revision, trained on the 170 labelled single-cell lines against
`r − r̂_prior(out of fold)`:

- `C` reads the Tx1 context PCA (128 components, unchanged) and the single-cell state views.
- `h` reads `q_sc`, the partner fraction-expressing features, `e_g`, and STATE's `s` and Δ. Δ is projected onto
  program scores: the change in each hallmark or PROGENy score between predicted and basal expression, for
  programs with at least 10 members among the 2,000 HVGs.
- Output layers start at zero, so before the first update the stack equals the prior; validation also runs then,
  so "prior alone" is always eligible for `best.pt`.
- **Candidates:** linear `C` versus `C` with its SwiGLU branch; cross-attention from `G(g)` over one token per
  single-cell view in place of the inner product (context-conditioned, so this is where overfitting is expected
  and validation decides); STATE frozen, trainable or absent.

Unchanged: Tx1 and ESM2 frozen and cached; `G` a free per-gene embedding plus a linear map of ESM2 (STRING or GO
embeddings deferred, since every gene is seen in training); no per-line lookup anywhere.

## 6. Interface to the virtual cell

**In this design.** STATE's predicted knockout response is a per-(g, c) feature computed from single cells, so it
enters the correction, projected onto the same program space as the context views. The prior measures what
expression alone gives; the correction with and without STATE's features measures what the virtual cell adds.
That is the [blueprint](../01-blueprint.md#1-the-question) research question, asked on GeneEffect.

**Recorded for the SL protocol, not run.** The prior reads a line's expression, so it can read a counterfactual
line: the virtual cell's predicted state of c after knocking out b.

$$\text{SL score}(a,b\mid c)=\hat d\big(a\mid \mathrm{STATE}(c,\ \text{knockout } b)\big)-\hat d(a\mid c)$$

with a control ladder: no counterfactual (pan-essentiality and context); b's own expression set to zero (the
paralog block alone gives "paralog low, dependency high"); STATE's full predicted response. Only the gap between
the last two is the virtual cell's contribution. Constraints: STATE predicts 2,000 HVGs, so this prior must read a
space those genes cover; no tested response interface has yet transferred to a held-out line.

## 7. Training protocol

### 7.1 Selection

One config is one run at seed 0: validation tunes, and test is scored for every row. The prior adds blocks in a
fixed order: expression components; pathway scores; predicted genotype; own expression; the partner group
(paralog, complex-partner and low-partner blocks together); data-selected genes; inferred copy number when
built. Context-view and data-selected blocks take a penalty from a log grid (and N for data-selected genes) with
earlier blocks fixed, by point estimate; the knowledge blocks take a shrinkage strength toward their pooled
weights. Tuning a penalty on validation is the only choice code makes. Whether a block, the reduced-rank k or a
correction candidate earns its place is read from the reported validation and test scores and the 95% paired
bootstrap interval (27 validation lines, 1,000 resamples, seed 0) of its `val_selective_spearman` gain over the
simpler setting; a person or agent reads the table and decides, and the decision is recorded with the results.

### 7.2 The prior as a step

The prior runs on CPU from its own config through `hpc/run.sh prior CONFIG [--run-id <id>] [--experiments A,B]`.
The runner is minimal: the config names experiments, each a bridge remedy (a module under
`src/context_prior/remedies/`) with a list of settings and block sets; every setting is built, fitted for every
block set and penalty, and scored on validation and test (and the bulk-input oracle on validation). It writes
`rows/<experiment>__<n>.json` per setting, `run_config.json` and `results.md` to `outputs/context_prior/<id>/`.
Cross-fitted predictions of the training single-cell lines (`crossfit` in the library) are produced for the
setting a person chooses to build the correction on. The correction's config names the prior run; its
checkpoint records it; evaluation refuses a checkpoint whose recorded prior run differs from the one supplied (a
fail-closed guard against a silently wrong artifact, not a quality gate). Code lives in the package
`src/context_prior/` (bulk and label loading, bridge, views, gene-pair features, block ridge, cross-fitting),
which imports neither `training`, `eval` nor `experiments`; the runner is `src/experiments/context_prior.py`.

### 7.3 Correction training

As in the revision: AdamW, head 1e-3, adapter 1e-4, STATE 1e-5 when trainable, one warmup epoch then cosine,
patience 5, at most 30 epochs, `best.pt` on `val_selective_spearman`, the 27-line training telemetry, the
objective-screen winner. Changes: the target is the residual after the out-of-fold prior, and validation runs
before the first update. The revision's STATE screen on the old target is dropped; it returns as the
correction's STATE comparison against the prior.

### 7.4 Where things run

The bulk-input learning curve uses only files already on the Mac and runs locally. Pseudo-bulk needs raw single
cells, so the bridge, the prior and the correction run on the H20 host (prior on CPU, correction on GPU). Files
over 5 GB (TCGA, any new single-cell atlas) are downloaded and processed only on the H20 host.

## 8. Evaluation

### 8.1 Learning curve and the extra lines

Two fixed prior configs, expression components alone and expression components plus 50 data-selected genes,
trained on 100, 167, 400, 700 and 1,086 training-side lines; below the full set, three random patient-grouped
subsets per size; the 167 point is the actual single-cell training lines. The ridge penalty is chosen on
validation at each size. Scored on validation from bulk RNA (the oracle, 26 lines) and from bridged pseudo-bulk
(27 lines).

The extra lines' GeneEffect labels help when, with bridged pseudo-bulk input, the 1,086-line prior beats the
167-line prior, both in the expression-components-plus-50-data-selected-genes configuration, by a gain whose 95%
paired bootstrap interval excludes zero. This is read from the reported rows, not applied in code. A gain with
bulk input but not with bridged input points at the bridge; the bridge remedies of §3.2 are then compared. The
first runs (2026-10-04) applied this reading as a rule in code; that runner, learning curve included, is replaced by the minimal runner of §7.2, and the run record
([results](../../results/context_prior_seed0/README.md)) holds what it found.

### 8.2 Metrics and controls

Selector and telemetry unchanged (selective Spearman, selective AUPR lift, residual Pearson over variable genes,
Huber, SD ratio). Every summary adds the prior as the strongest control beside gene mean, copy prior, nearest
line and the Tx1 and HVG context ridges; paired bootstraps of stack minus prior and stack minus Tx1 ridge; a
per-lineage table on validation and test, descriptive only (1–5 lines per lineage); and bridge quality, the
per-gene correlation across lines between bridged pseudo-bulk and bulk on out-of-fold training lines.

### 8.3 Oracle diagnostic

The prior applied to validation lines' bulk RNA is an upper bound that splits the gain of more lines from the
loss in the bridge. It is validation-only, marked off-contract in every table, and never a model or a
comparison row.

## 9. Claims and documents

- **Extra lines (protocol rules).** DepMap lines outside the 226 with bulk RNA may join the training side, with the
  exclusions of §3.1. Their bulk RNA and other-omics labels fit context encoders; whether their GeneEffect labels
  fit the prior is read from §8.1.
- **Reporting.** Results that use extra lines are reported as a training-data change, scored on the unchanged
  validation and test lines; validation lines' bulk RNA is read only by the oracle diagnostic and test lines' bulk
  RNA never. These sit in the [GeneEffect protocol rules](../03-geneeffect-protocol.md#11-rules).
- Unchanged: a query line supplies basal single cells only; validation tunes, test reported once per run; Tx1's
  Tahoe-100M exposure qualifies held-out results; nothing here is SL evidence; the counterfactual interface is not
  run.
- `docs/03-geneeffect-protocol.md`, the data cards and `CLAUDE.md` change with the code that changes their
  behaviour; results go under `results/`.

## 10. Order of work

1. **Local groundwork (Mac):** the extra-line membership file and its builder; pinned Ensembl paralogs, CORUM,
   hallmark and PROGENy gene sets, `OmicsSignatures` MSI scores; the bulk-input learning curve; a search for
   public single-cell atlases covering CRISPR-screened lines outside the 226 (they would add single-cell lines
   without a bridge; any download over 5 GB happens on the H20 host).
2. **Bridge and the extra-lines decision (H20):** pseudo-bulk in preparation; bridge and its quality; the curve
   rescored with bridged input; the extra lines read from the rows.
3. **Prior build-up (H20 CPU):** block sets compared on validation and test, predicted-genotype encoders, the prior a person
   chooses from the rows.
4. **Inferred copy number:** build, check against WGS and arm-level calls on training lines, offer as a block.
5. **Correction (H20 GPU):** the nested head on the prior's residual; the STATE comparison (absent, frozen,
   trainable); the cross-attention candidate; linear versus SwiGLU `C`.

The first implementation plan covers local groundwork through the prior build-up. Inferred copy number and the
correction get their own plans once the prior exists, because their inputs depend on its chosen settings and on
the extra-lines decision.

## 11. Open checks

- How far pseudo-bulk from the single-cell atlases departs from DepMap bulk after the bridge (a 2025 bioRxiv
  reports substantial pseudo-bulk/bulk differences; not read).
- Accuracy of expression-inferred arm-level copy number against WGS on these lines (no published benchmark
  found).
- 26Q1 files behind the portal's Cloudflare check (`OmicsGlobalSignatures`, `CRISPRConfounders`, subtype
  tables) are fetched from the H20 host or a browser session if needed; the 24Q4 figshare `OmicsSignatures`
  supplies MSI scores meanwhile.

## 12. Amendments (2026-10-07)

Agreed in the design interview after the default-prior follow-ups
([results](../../results/default_prior_followups_seed0/README.md)); the plans are
[wave one](2026-10-07-single-cell-correction-plan.md) and
[the second wave](2026-10-07-single-cell-correction-second-wave-plan.md).

- **Prior.** The correction stacks on the affine bridge with expression components (penalty 1) and data-selected
  genes (gene penalty 10), 0.2290 validation / 0.2338 test; own expression and partners added +0.0010 on
  validation (interval spanning zero) once data-selected genes were in.
- **Objective (§7.3).** The objective is screened again on the prior's residual with STATE absent: Huber,
  standardised MSE, standardised MSE plus a ListNet cross-entropy across a gene's lines (temperature 1 in σ units),
  and standardised MSE plus a binary cross-entropy of GeneEffect < −0.5 on the stack's output. The winner runs with
  frozen and with trainable STATE. If STATE absent wins, the research plan is discussed before anything else runs.
- **Growth in layers (§5.2).** Wave one is the existing head unchanged. The second wave adds each input on its own
  against the wave-one winner: partner fractions, program-score Δ (with a STATE setting), the state views,
  inferred copy number, then `C` off, linear `C`, `C` over the views and cross-attention; then one combined run.
- **State views (§3.3).** Tirosh cell-cycle sets and Gavish meta-programs only. Kinker's programs were derived from
  the cells of 23 validation and 19 test lines (`kinker_sccle` is their source), so they would expose held-out lines.
- **Inferred copy number (§3.3).** Also computed from single cells for the correction: arm-level scores, their
  spread across cells and an aneuploidy score as part of the state views, and local copy number of g and its
  closest paralog as an `h` block.
- **More labelled contexts (§10).** The search for public single-cell atlases of screened lines outside the 226 runs
  alongside wave one and ends in a pinned membership file; ingesting the lines is a separate decision.
- **Controls (§8.2).** Every correction summary adds the prior alone and the prior plus the Tx1 context-PCA ridge
  fitted on what the prior leaves, each with a paired bootstrap of the stack minus it.
- **Not in these plans.** Contrastive representation learning (a gene-side contrastive term on `G` is a later
  candidate), predicted genotype in the correction, and other single-cell foundation-model embeddings: the
  limit is labelled contexts, and predicted genotype already gave no gain as a prior block.
