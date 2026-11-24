# Cell-Line-Specific SL Score: Cross-Line Loss and Cell-Line Data Upgrade Design

Date: 2026-11-24. Status: proposed design, revised the same day after a review against the records (§9 lists the changes). The design to build is §5 and §6; the context-gated model of the first draft is deferred (§7). Builds on the gene-knowledge, cell-set model of [SL protocol §18–§19](../04-sl-ranking-protocol.md), the [workflow diagram](../Gene-Knowledge%20Cell-Set%20Model%20Workflow.svg), and the [run record of 2026-11-20](../../results/sl_line_term_seed0/summary.html) (with ACH-000696 cache repair of 2026-11-23).

## 1. Intent and Why Now

The goal of this project ([blueprint §3, §5](../01-blueprint.md)) is **synthetic-lethality (SL) candidate ranking in unseen cell lines** excluded from all training and selection across both stages. To separate this work from pair-level predictors such as Cilantro-SL, a pair's score must **change with the cell line** and that change must **hold in unseen test lines**.

Today's gene-knowledge, cell-set model (`src/model/sl_embedding.py`) clears every published baseline on overall top-10% recall (0.553 validation / 0.499 test with Tx1-3B cell tokens on repaired caches; 0.545 / 0.505 without cell tokens, vs 0.355 / 0.395 for the Kebabci paralog classifier). However, strict controls show that **context-specific ability is still mostly absent**:

1. **Against another real cell line's input (`other_line` control)**: With Tx1-3B cell tokens and the line term, own input minus another line's input is +0.044 validation / **+0.003 test [−0.021, +0.026]** (without the line term, test recall is 0.499 on the own line vs 0.507 on a wrong line). Without cell tokens, it is +0.062 / +0.032 [+0.005, +0.067].
2. **On seen pairs where training lines differ**: With cell tokens, own input minus another line's input is **−0.025 validation / +0.029 test [−0.002, +0.059]**.
3. **Single-cell and post-hoc line-term additions are null or negative**: Tx1-3B cell tokens (+0.008 / −0.009), CellNet cell tokens (−0.023 / −0.028), own-expression line term (+0.003 / +0.007), PIE-change line term (+0.001 / +0.004), and Tx1-3B gene-embedding line term (−0.035 / −0.021, overfitting training recall to 0.908).
4. **Probable causes (hypotheses from the code and the records; §6 tests them one change at a time)**:
   - *Global 32-dim bottleneck*: Compressing an entire cell line into one 32-wide token `t_0` (plus a 32-wide cell average) cannot preserve whether specific anchor gene `A`, partner `B`, or their paralog/complex buffers are lost or expressed in line `c`.
   - *64.1% missing-gene blind spot and noisy bridge*: `t_0` and `own_expression` read only the 6,523 genes measured in every held-out pseudo-bulk (35.9% of the 17,787 model genes), passed through a pseudo-bulk-to-bulk affine bridge with median per-line Pearson 0.44.
   - *Pan-essentiality and pair-identity gradient shortcuts*: 85% of the GeneEffect variance lies between genes ([record](../../results/cilantro_reproduction_seed0/README.md)), so stage one's MSE on GeneEffect mostly fits static gene means (0.033–0.070 per-line residual Spearman). Stage-two per-row binary cross-entropy lets static gene vectors (`u_g` plus 570k free `r_g` parameters) drive training recall to 0.85–0.92 without reading the line.
   - *The line term fails in two opposite ways*: Trained last on a frozen, already-saturated `f`, it has almost no signal left and adds nothing (+0.003 / +0.007). On a base that reads no line (the first base of §19), it had signal to fit, fitted the training lines, and scored 0.288 / 0.279, below its base (0.466 / 0.430). With Tx1-3B gene embeddings it reached 0.908 on the training lines and lost validation recall at each epoch. Thus a deviation with capacity fits the 44 training lines: this is the main risk of every design here (§5.3).
   - *Deposit-confounded cell tokens and missing genomic defects*: Raw 2,560-wide Tx1-3B vectors encode lab/chemistry batch effects across 44 lines, while coding loss-of-function mutations, focal copy-number losses, paralog buffers, and pathway states are absent from `sl_embedding`.

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **Primary**: Produce a score that genuinely changes with the cell line in unseen lines — `own_line − other_line` above zero (95% line-bootstrap interval above zero on validation and positive on test) on **all pairs**, **seen pairs where training lines differ**, and **rows where neither gene is a single-gene dependency alone**, while keeping overall test top-10% recall at or above the current 0.505 and new-pair recall ≥ 0.52. The all-pairs difference is already above zero for the model without cell tokens, so the sensitive measure is the cross-line ordering in held-out lines (§5.5).
- **Data attribution**: Measure, one layer for each row, what additional cell-line-specific information lifts held-out SL ranking. Local features of the two genes come first, because a global line state over 44 training lines identifies the line (§5.3):
  1. **Strict scRNA-only query mode**: wide unbridged scRNA expression of the two genes and their paralog and complex buffers;
  2. **Multi-omic query mode** (blueprint §3.2): the same plus masked DepMap damaging and hotspot mutations and copy number of the two genes;
  3. **Expanded Stage-2 training lines** (+36 patient-clean bulk-RNA DepMap lines with genetic SL screens, 44 to 80 lines, reported side-by-side per blueprint §3.6): the line count limits every line × pair effect;
  4. **Deferred to §7**: scRNA-inferred copy number, driver and MSI probes, pathway, meta-program and cell-state activities, Cellosaurus text, and deposit-whitened foundation-model cell tokens.

### Non-Goals
- Turning Stage-1 GeneEffect accuracy into a gate or standalone goal (Stage-1 metrics are diagnostic telemetry reported after SL).
- Unseen-gene generalization (the generalization axis is unseen cell lines).
- End-to-end fine-tuning or LoRA training of 3B single-cell foundation models on 44 SL lines.
- A new model module, a new stage one or a new cache root before the cross-line loss shows a gain in the current model.

### Anti-Goals (Failure Even If Headline Recall Rises)
- **Per-line z-scoring or within-line ranking of predicted GeneEffect** (strictly forbidden by SL protocol §8: consumes the test line's distribution).
- **Dropping held-out lines that lack DepMap multi-omics or GeneEffect**: Every run must score all 17 validation lines (including `CVCL_0058` HeLa S3 and the 5 lines without GeneEffect) and all 16 test lines (including `CVCL_0063` HEK293T and the 5 lines without GeneEffect) using explicit presence flags, never silent zero imputation without a mask.
- **Pair-memorization masquerading as context**: Any gain in overall recall without a positive gain against `other_line` and `context_ablated` is not context evidence.
- **Memorization of (line, pair)**: A gain of the cross-line ordering on the training lines without a gain in held-out lines is not context evidence.

## 3. Project Constraints

- Python 3.11–3.12 managed by `uv`; run from repo root with `-m`; `src/data` and `src/model` never import `training`, `eval`, or `experiments`.
- Fail-closed configuration and checkpoint loading: `require_keys` everywhere, no `.get(key, default)`, no `load_state_dict(..., strict=False)`.
- Preparation/caching writes immutable arrays under `data/`; training opens caches and never reads raw files.
- All means, standard deviations, PCA/whitening bases, `inferCNV` diploid baselines, and linear-model weights fit **strictly on eligible training-side lines** (`assert_fit_eligible`), never on validation, test, or patient-mates.
- One config = one seed-0 experiment: select checkpoint and phase on validation top-10% recall, score test once, and evaluate via `hpc/run.sh sl-baselines`.
- Terminology: say *GeneEffect / dependency prediction* (Stage 1) and *SL candidate ranking* (Stage 2); never write "interaction" for a model output.

## 4. Approaches Considered

Four directions were evaluated against the project's empirical records; weak variants (end-to-end foundation-model fine-tuning on 44 lines; unmasked multi-omic concatenation that drops non-DepMap lines) were eliminated upfront.

### Approach 1: Upgrade the Counterfactual Perturbation Backbone (SL Protocol §11–§14)
- **Sketch**: Keep the Definition-1 counterfactual difference $s(c, A, B) = -[\hat{y}(c \setminus A, B) - \hat{y}(c, B)] - [\hat{y}(c \setminus B, A) - \hat{y}(c, A)]$, replace the linear reader with a nonlinear mutation/buffer-aware dependency model, and fine-tune PIE or STATE with line adapters to predict $x_{c \setminus A}$.
- **Trade-offs**: Directly mirrors single-knockout Definition 1, **but already falsified by §14**: even *measured* Perturb-seq post-knockout profiles on K-562/Jurkat/HCT116/HEK293T ranked SL pairs near random (0.114 against random 0.10, [record](../../results/sl_pie_editor_seed0/README.md)) through the dependency reader, and composing two noisy single-gene predictions discards the +0.15 recall gain that `GeneLineModel` earns from learned gene-knowledge representations. **Dropped.**

### Approach 2: End-to-End Whole-Line Cross-Attention Transformer
- **Sketch**: Represent line `c` as ~18,500 gene tokens (expression + copy number + mutation) plus 256 cell tokens, and let pair queries `(u_A, u_B)` cross-attend over all tokens of line `c` in both stages (similar to `DirectReadout` in `src/model/stage_one.py`).
- **Trade-offs**: Avoids compressing the line into one vector, **but fails sample-complexity reality**: on 981 Stage-1 lines `DirectReadout` trailed the linear model (0.177 vs 0.226) at 50× the compute cost, and on 44 Stage-2 lines dense cross-attention over 18,500 tokens memorizes line/deposit identity (as §19's 82k-parameter `tx1_gene` map already did, dropping validation recall by −0.046). **Dropped.**

### Approach 3 (Recommended Pick): the Current Model with a Cross-Line Loss, then Local Cell-Line Data, One Change for Each Row
- **Sketch**: Keep `GeneLineModel` and its score `s = f + D − mean_R D` (§19). Add a ranking loss between two training lines of one study for the same pair; the gene pair alone cannot lower it. Then give the line term local features of the two genes in the line as gene-in-line caches: wide unbridged scRNA expression, paralog and complex buffers, masked DepMap mutations and copy number. Then add the 36 bulk-RNA training lines. §5 is the design and §6 holds the rows.
- **Trade-offs**: It is the smallest change that tests the central hypothesis, and each row changes one thing, so a null result has one cause. It keeps the 32-wide line summary and the current stage one, so it does not test those parts of Approach 4.
- **Why Recommended**: The loss is the only part of the first draft that static pair knowledge cannot satisfy, and the score it needs is already in the code. The records show that a deviation with capacity fits the 44 training lines (§1), so capacity and data enter in small steps, each with the held-out measure of §5.5 beside it.

### Approach 4 (Deferred; the First Draft's Pick): Factorized Context-Gated Gene-Knowledge Model (`ContextGatedSLModel`)
- **Status**: Not built until a row of §6 is positive. Then its parts enter as single rows (§7).
- **Sketch**: Keep `GeneLineModel`'s fast, high-recall 32-wide gene-knowledge backbone (`u_g` from function text, ESM C 600M, ESM3 structure, and STRING), **and fix all four bottlenecks structurally**:
  1. **Data**: Build a unified cell-line context cache (`data/sl_context/v1/`) supplying **~18,500-gene unbridged scRNA expression**, **local paralog/CORUM/STRING buffer state**, **out-of-sample wide linear-model GeneEffect**, **batch-clean pathway/cell-state proportions**, **scRNA-inferred copy number (`inferCNV`) + imputed driver/MSI scores** (scRNA-only mode), plus **masked DepMap somatic mutations, WGS copy number, and Cellosaurus text** (multi-omic mode) and **+36 bulk-RNA SL training lines** (80-line complementary training mode).
  2. **Stage 1**: Decompose GeneEffect into a static pan-essentiality head on `u_g`, an out-of-sample linear-model anchor, and a zero-centered context-residual embedding $V^{\text{ctx}}(g, c)$ trained with **both** GeneEffect MSE and a **within-gene cross-line cosine residual loss**.
  3. **Stage 2 Architecture**: Score pairs as an **end-to-end decomposed sum** $s(c, A, B) = s_{\text{pair}}(A, B, \text{design}) + [\Delta s_{\text{ctx}}(c, A, B) - \bar{\Delta}_R(A, B)]$, where $s_{\text{pair}}$ reads *no* line inputs (serving directly as the exact in-distribution `context_ablated` score) and $\Delta s_{\text{ctx}}$ uses **frozen-gene FiLM gating** (no free per-gene `r_g` table!) over compact local + global cell-line features.
  4. **Stage 2 Objective**: Train `s_pair` and `Δs_ctx` **jointly** on §16's **top-10% continuous rank ramp target** plus a **within-study cross-line pairwise ranking loss ($\mathcal{L}_{\text{xline}}$)** where $s_{\text{pair}}$ and $\bar{\Delta}_R$ cancel out algebraically, forcing gradients through cell-line differences.
- **Why it was the first draft's pick**: Preserves the 0.55 recall ceiling of gene-knowledge embeddings, injects local biological state (Kebabci scores 0.723 / 0.685 on the 108,052 held-out paralog rows it covers, 19.9% of the rows; a score that reads no line is higher on those rows, so that number does not show the value of local state), eliminates deposit-batch overfitting and post-hoc logit saturation, and stays under ~5 minutes per GPU run.

## 5. Design to Build First: the Cross-Line Loss in the Current Model

No new model module, no new runner and no new cache root. The changes are in `src/experiments/sl/embedding.py`, `src/model/sl_embedding.py`, `src/experiments/sl/gene_in_line.py` and the harness (`src/experiments/sl/benchmark_baselines.py`).

### 5.1 Counts Before Any Model (CPU, No Training)

The counts come from the label table and the SL split. The harness gives the first two: its cross-line ordering table (§5.5) has the examples and the gene pairs of each study on the training, validation and test lines, for the labels as for every score. They decide nothing in code; the numbers go into the report.

- **Cross-line training examples.** For each study: the pairs that at least one training line called SL and at least one training line did not, and the number of line pairs they give. If a few paralog-library studies supply most of them (§15 found the same-line advantage there and in few other study pairs), the loss trains on those studies, and the report must say so.
- **Cross-line held-out examples.** The same count with both lines among the validation lines, and with both among the test lines. These examples carry the measure of §5.5.
- **The 36 bulk-RNA lines.** How many have a two-class genetic screen in the label table, and how many of them are among the 74 lines with an SL label that stage one's fit already leaves out (`stage_one_lines` leaves out every line with an SL label and every line of its patient; the fit has 928 lines). If all 36 are among them, the row with 80 training lines keeps the same stage one. Measured on the H20 on 2026-11-24: 36 bulk-only lines of the split have a two-class genetic screen in the label table (32 `bulk_labelled`, 4 `bulk_unlabelled`). The stage-one fit leaves out every line with an SL label, so it stays at 928 lines. `benchmark_rows` drops the rows of every line outside train, validation and test, so that row needs its own [plan](2026-11-24-cross-line-loss-plan.md).

### 5.2 The Cross-Line Loss

The score is that of §19, `s(c, A, B) = f(c, A, B) + D(c, A, B) − mean_{r ∈ R} D(r, A, B)` (`GeneLineModel.line_term`, `line_term_scores`).

- **Example.** `(study, A, B, c⁺, c⁻)`: one study, one pair and design, two SL training lines; `c⁺` has the SL call, `c⁻` does not.
- **Loss.**
  $$\mathcal{L}_{\text{cross}} = \text{Softplus}\big(-[s(c^+, A, B) - s(c^-, A, B)]\big) = \text{Softplus}\big(-[(f + D)(c^+, A, B) - (f + D)(c^-, A, B)]\big)$$
  The mean over `R` is the same in the two lines and cancels. A score that is the same in every line gives `log 2` on every example, so the gene pair alone cannot lower this loss.
- **A step.** Studies are drawn evenly, then `cross_line.examples_per_step` examples. Each of the two lines reads its drawn tokens, as in `line_term_loss`. The total loss is the binary cross-entropy on `s` (as now) plus `cross_line.weight` times the cross-line loss. The target stays the SL call.
- **Two training modes** (`line_term.joint`):
  - `false`: the line term trains last on the frozen `f`, as in §19. Only `D` gets the gradient of the new loss.
  - `true`: in the phase that trains the whole model, the score is `s` and the loss is the binary cross-entropy, the cross-line loss and the GeneEffect loss; every parameter trains, and no separate line-term phase follows. Validation recall in the top 10% of `s` selects the epoch and the phase, as now.
- **Places in the code.** `line_term_loss`, `fit_line_term`, `fit_pairs` and `train`; `PairRows` gets the index of the examples; the new config keys are read fail closed in `load_config`.

### 5.3 The Risk: the Deviation Fits (Line, Pair)

The records show it twice (§1): the line term on a base that reads no line scored below its base, and the line term with Tx1-3B gene embeddings fitted the training lines. The cross-line loss increases this pressure, because a table of (line, pair) is the cheapest way to lower it on 44 lines.

- **What the line head reads** (`line_term.input`). `model` is the current input: the pair head's input, which holds the line's tokens and the free gene deviation `r_g`, and the gene-in-line features. `genes` is the low-capacity input: the gene vectors from the frozen sources alone (`u⁰_g`, the gene encoder's output without `r_g`), no line token, and the gene-in-line features:
  `[u⁰_A + u⁰_B ; |u⁰_A − u⁰_B| ; u⁰_A ⊙ u⁰_B ; combinatorial ; e_A + e_B ; |e_A − e_B| ; e_A ⊙ e_B]`.
  With `genes`, the deviation must go through features of the two genes in the line.
- **No global line state in the deviation.** Pathway scores, state shares, text and other line-level vectors have about as many values as there are training lines, so they identify the line. They stay in §7.
- **What each run reports.** Recall on the training lines beside validation and test at the kept epoch (the run's `results.md`), and the cross-line ordering (§5.5) on the training lines beside the held-out lines (the harness).

### 5.4 Cell-Line Data: Gene-in-Line Caches

The new data are gene-in-line caches under `data/sl_gene_in_line/v1/<name>/`, written by new CPU steps of `hpc/run.sh sl-gene-in-line` and read by the existing `read_in_line` (each column centred and scaled on the SL training lines with benchmark rows; one presence flag for each row). A cache row has one presence flag, so columns that can be absent separately go to separate caches. A line without a source gets a file of NaN rows, so no line is dropped and the run does not stop.

| Cache | Values for each (line, gene) | Source | Absent when |
| --- | --- | --- | --- |
| `wide_expression` | `log1p(CP10K)` of the line's pseudo-bulk; a silenced flag (at or below the gene's 10th percentile over the SL training lines) | `ladder/cells/<ModelID>.npz`, 19,277 source genes, no bridge | the source does not hold the gene |
| `cell_spread` | share of the line's cells that detect the gene; SD over the depth-expected value | the same file | the line is bulk-only |
| buffer caches | the five partner columns of `knowledge_features` (`src/linear_model/gene_features.py`) on the wide expression: paralog minimum, paralog sum, first paralog, complex mean, share of low partners | the wide expression and the partner index | the gene has no such partner |
| `depmap_mutation` | damaging mutation; hotspot mutation | `OmicsSomaticMutationsMatrixDamaging.csv`, `OmicsSomaticMutationsMatrixHotspot.csv` | the line has no DepMap calls (`CVCL_*`, `SL-*`) |
| `depmap_copy_number` | log2 copy-number ratio; deep loss (< −0.8) | `PortalOmicsCNGeneLog2.csv` | the same |

- With `wide_expression`, `line_term.own_expression` is off: the wide cache holds that value for all genes, not for the 6,523 bridged genes.
- The mean token `t_0` keeps its bridged 6,523-gene input in every row. This design repairs the gene coverage for the two genes of the pair, not for the line summary; a wider `t_0` needs a wide input for stage one's bulk fit lines and is a later change.
- **The 36 bulk-RNA training lines.** Their `wide_expression` is bulk RNA mapped gene by gene with an affine fit on the 343 paired training lines; `cell_spread` is absent. The model can tell these lines from the single-cell lines by that flag, and they come from other studies, so this row reads as "more lines of another kind", not as a clean sample-size effect. Validation and test lines do not change.
- No pair-in-line tensor. The line head already reads `e_A ⊙ e_B` and `|e_A − e_B|`, which can express "A is lost and B is expressed"; shared-partner features of a pair are deferred.

### 5.5 Measures (Read from the Harness Report, Never Gated in Code)

- **Unchanged.** Recall in the top 10%, AUROC and AUPRC on all pairs and on the row sets, on validation and test; the model minus its context-ablated version.
- **The other-line control over several lines.** `eval.other_lines: 4`: the model is scored on the input of four seeded other benchmark lines, and the control's recall is the mean over them. One other line (as now) gives a wider interval for the same training cost.
- **Cross-line ordering in held-out lines.** Examples as in §5.2 with both lines held out on the same side. The value is the share of examples with `s(c⁺) > s(c⁻)` (a tie counts one half), mean over studies, with a bootstrap over gene pairs (as §15). A score that is the same in every line gives exactly 0.5. It is given for the model, for the model on another line's input, and on the training lines.
  - Both lines are held out because a training line's score is in sample: a difference between a held-out line and a training line carries regression to the mean, not the line.
  - The noise limits it: the line-specific part of the labels repeats at 0.35 to 0.39 inside one study (§16). The top value of the ordering is not measured; Harle's replicates in `target_options.py` can give it.
- **Row sets for the claim.** Seen pairs on which the training lines differ, and rows where no gene of the pair is a dependency alone.

## 6. Rows of the Study

One config, one seed-0 run and one change for each row, all without cell tokens (the reference is the best row on test and the only one whose difference from another line's input is above zero on test). Each row starts from the row above. If a row lowers validation, the report says so, and the reader decides from which row the next one starts.

| Config (`configs/sl/`) | The one change against the row above | Question |
| :--- | :--- | :--- |
| `embedding_direct_mean_token.yaml` (finished, `_seed0_v2`) | — | Reference: 0.545 / 0.505; minus another line's input +0.062 [+0.032, +0.099] / +0.032 [+0.005, +0.067] |
| `cross_line_frozen.yaml` | The cross-line loss in the line term; `f` frozen | Does the loss alone make the deviation follow the line? |
| `cross_line_joint.yaml` | `line_term.joint: true` | Does the model follow the line when it trains with the line term (the post-hoc hypothesis of §1)? |
| `cross_line_genes_input.yaml` | `line_term.input: genes` | Does a deviation with less capacity hold better in unseen lines? |
| `cross_line_wide.yaml` (**scRNA-only query, headline**) | `wide_expression`, `cell_spread` and the buffer caches; `own_expression` off | Do the gene coverage and the buffers of the two genes add line signal? |
| `cross_line_depmap_dna.yaml` (**multi-omic query**) | `depmap_mutation` and `depmap_copy_number`, masked | What do measured mutations and copy number add over scRNA? |
| `cross_line_80_lines.yaml` | The wide row with the 36 bulk-RNA training lines | Does the line count limit the line signal? |

### What the Report Must Show

1. The cross-line ordering in held-out lines against 0.5, on validation and test, beside its value on the training lines.
2. The model minus itself on other lines' input, on all pairs and on the two row sets of §5.5. The row set "the training lines differ" has 13 / 10 held-out lines, so a gain below about 0.05 does not show there; the ordering is the sensitive measure.
3. The paired difference of each row minus the row above.
4. Overall test recall against 0.505 and new-pair recall against 0.52.

## 7. Deferred Design: the Context-Gated Model (First Draft)

This section is the first draft's design. It is not built until a row of §6 is positive; then each part enters as one row on the best row of §6. Corrections from the review:

- **Stage-one residual split.** With the linear anchor `γ · r_lin` in the sum, `V_ctx` trains on what the linear model leaves. Every stage-one arm on that residual was null (7 of 8 runs kept epoch 0), so `V_ctx` will probably stay near zero, as the embeddings of §18 did. Use the linear model's prediction as an input feature only, and test the within-gene cosine loss as its own row.
- **Context-blind pair score with a centred deviation.** This is the structure of §19's first base, which scored below its base. It needs the low-capacity input and the report of §5.3 before it is tried again.
- **Ramp target.** §17 found it equal to the calls in a head without capacity for each pair; it is a fair single row in the joint model.
- **Global line state** (pathway and meta-program scores, state shares, MSI and driver probes, Cellosaurus text, lineage). It identifies the line among 44 (§5.3); it enters last.
- **scRNA-inferred copy number.** A 101-gene window finds arm-level changes; a focal loss already shows as a silenced gene in `wide_expression`.
- **Deposit-whitened Tx1-3B tokens.** The design does not say how a held-out line is whitened when its deposit is not among the training deposits or has fewer than three lines; its own statistics are not allowed (SL protocol §8). State that rule first.
- **Wide linear-model GeneEffect as a local feature.** It needs a refit of the wide linear model without every SL line; a later row.


```text
PANEL A — Multi-Scale Cell-Line Context Cache (data/sl_context/v1, train-only fit)
  ┌─ Global line state g(c) [dim ~ 128, masked] ──────────────────────────────────────────┐
  │ • scRNA pathway & meta-programs: PROGENy (14), Hallmark (50), Gavish 3CA (21), Cycle(2)│
  │ • scRNA subpopulation shares: cNMF / scVI state masses & entropy (ladder/states)       │
  │ • Genomic instability: scRNA MSI & inferCNV aneuploidy (+ masked WGS ploidy/MSI)       │
  │ • Identity context: Cellosaurus text PCA (16) + lineage / non-cancer (scRNA or meta)   │
  └────────────────────────────────────────────────────────────────────────────────────────┘
  ┌─ Local gene-in-line state x(c, g) [17,787+ genes x 12, each with presence bit] ───────┐
  │ • Wide unbridged scRNA: log1p(CP10K) train-z, detection share, Poisson overdispersion, │
  │   and <10th-pct silenced flag (~19,277 source genes; no 6,523-gene bridge cutoff)      │
  │ • Out-of-sample wide linear-model GeneEffect residual (in train residual-SD units)     │
  │ • Copy number: scRNA inferCNV chromosomal log-ratio (all lines) + masked WGS CN/loss   │
  │ • Somatic mutations: scRNA driver-activity score + masked DepMap damaging & hotspot    │
  └────────────────────────────────────────────────────────────────────────────────────────┘
  ┌─ Pair-in-line relational state r(c, A, B) [symmetric, ~18 dims] ──────────────────────┐
  │ • Paralog buffer in c: max & sum expression of remaining paralogs C ∉ {A, B}           │
  │ • Shared CORUM complex & STRING partner mean expression & linear-GE in c               │
  │ • Conditional defect terms: defect(A,c)*expr(B,c), min/max expr & CN, co-detection     │
  └────────────────────────────────────────────────────────────────────────────────────────┘

PANEL B — Stage 1: Residual-Preserving Pretraining (928 fit lines, stop on 47 SL train lines)
  Gene sources (text, ESMC-600M, ESM3-struct, STRING) ──► GeneEncoder ──► u_g⁰ (no free r_g)
                                                                      └─► + r_g ──► u_g
  Line global g(c) + whitened cell tokens ──► h_g(c);   local x(c, g) ──► e(c, g)
  V_ctx(g, c) = u_g⁰ ⊙ MLP_ctx([h_g(c) ; e(c, g)])
  z_hat(g, c) = w_base · u_g  +  γ * r_lin(g, c)  +  Σ_k V_ctx,k(g, c)
  Loss_S1 = MSE(z_hat, GeneEffect_scaled) + λ_res * WithinGeneCosineLoss(z_res, GE_residual)

PANEL C — Stage 2: Joint Decomposed Pair + FiLM Context Head (44 scRNA or 80 scRNA+bulk lines)
  • Pair base (context-blind):
      s_pair(A, B, design) = MLP_pair([sym(u_A, u_B) ; static_pair_priors ; combinatorial])
  • Context deviation (zero-init, gated by u_g⁰ WITHOUT free r_g):
      ψ(c, A, B) = [sym(V_ctx(A,c), V_ctx(B,c)) ; sym(e(c,A), e(c,B)) ; r(c,A,B) ; g(c)]
      [γ_AB, β_AB] = FiLM_gate([u_A⁰ + u_B⁰ ; |u_A⁰ − u_B⁰| ; u_A⁰ ⊙ u_B⁰ ; combinatorial])
      Δs_ctx(c, A, B) = MLP_ctx(γ_AB ⊙ LayerNorm(ψ(c, A, B)) + β_AB)
  • Final score (centred over reference training lines R):
      s(c, A, B) = s_pair(A, B, design) + Δs_ctx(c, A, B) − mean_{r ∈ R} Δs_ctx(r, A, B)
  • Joint Stage-2 Loss:
      L_S2 = SoftBCE_screen(s_with_pair_dropout, y_ramp10)
           + λ_xline * PairwiseLogisticLoss_{same study, y(c⁺)>y(c⁻)}(Δs_ctx(c⁺) − Δs_ctx(c⁻))
           + λ_ge * Loss_S1
```

### 7.1 Cell-Line-Specific Data Additions (`src/experiments/sl/context_cache.py`)

Every new data source is materialized once on the H20 into `data/sl_context/v1/` with strict train-only fitting (`fit_lines`: Stage-1 fit lines for Stage-1 features; SL training lines for Stage-2 scaling) and explicit binary presence masks so no line is ever dropped:

1. **Wide Unbridged Basal Expression & Single-Cell Dispersion (`wide_scrna`, 100% of scRNA lines)**:
   - Read directly from `data/geneeffect_joint/v2/ladder/cells/<ModelID>.npz` (~19,277 genes, vs 6,523 previously).
   - Per `(line, gene)`: `log1p(CP10K)` standardized using training-line mean/SD, single-cell detection fraction, excess SD over the depth-matched Poisson expectation (already stored in `ladder/cells/`), binary silenced indicator ($\le$ training 10th percentile), and `is_measured` mask.
   - For bulk-only training lines (Stage-1 fit lines and the +36 bulk SL lines): mapped onto the same gene axis via per-gene rank-matched / affine alignment fitted on the 343 paired training lines, with single-cell dispersion slots masked (`has_sc_dispersion = 0`).

2. **scRNA-Inferred Copy Number (`infer_cnv`, 100% of scRNA lines — Recommended Call for scRNA-Only DNA Proxy)**:
   - Using GENCODE v44 chromosomal coordinates (chromosomes 1–22), compute a 101-gene sliding-window mean of centered `log1p(CP10K)` expression per line, referenced against the non-cancer / diploid-anchored training baseline and calibrated by ridge regression against `PortalOmicsCNGeneLog2` on the labelled training lines that have both.
   - Produces per-gene inferred copy-number log-ratio $\widehat{\text{CN}}(c, g)$ and whole-line chromosomal instability (arm-variance aneuploidy score) for **all 77 SL benchmark lines**, including non-DepMap `CVCL_0058` and `CVCL_0063`.

3. **DepMap Somatic Mutations & WGS Copy Number (`depmap_dna`, Multi-Omic Query Arm)**:
   - From local H20 `OmicsSomaticMutationsMatrixDamaging.csv`, `OmicsSomaticMutationsMatrixHotspot.csv`, and `PortalOmicsCNGeneLog2.csv` (covering 16/17 validation and 15/16 test lines).
   - Per `(line, gene)`: `[damaging_mut, hotspot_mut, has_mut, cn_log2_z, deep_cn_loss (< -0.8), has_cn]`. Non-ACH lines (`CVCL_*`, `SL-*`) set value `0` and mask `0`.
   - For the **strict scRNA-only arm**, `depmap_dna` is switched off at query time; instead, L2-regularized linear probes trained on the 928 fit lines predict pathway-driver alteration scores (`drivers.tsv`) and MSI transcriptional score (`msi.tsv`) from wide expression, plus `depmap_dna` acts as an optional Stage-1 auxiliary multi-task distillation target.

4. **Out-of-Sample Wide Linear-Model Predicted Dependency (`wide_linear_ge`, 100% of lines)**:
   - Fit one unified wide-space linear expression model (`space_measured_share: 0.5`, ~18,294 genes, as in SL protocol §13) strictly on the Stage-1 fit cohort with **all 77 scRNA SL lines, all 36 bulk-RNA SL lines, and all their patient-mates excluded** (the 928 fit lines of §18 already leave out the 74 lines with an SL label and 12 lines of their patients; §5.1 counts whether the 36 are among them).
   - Supplies $\hat{r}_{\text{lin}}(c, g)$ in training residual-SD units (never per-line z-scored!) out-of-sample for every single line in both the 44-line and 80-line Stage-2 arms from one shared cache.

5. **Pair-in-Line Relational Buffer Features (`pair_buffers`, 100% of rows, < 400 MB GPU RAM)**:
   - Stored in factorized form: per-`(line, gene)` buffer summaries (`knowledge_features` from `src/linear_model/gene_features.py` on wide expression: paralog max/min/sum, CORUM complex mean, low-partner fraction) plus a compact `(benchmark_lines × unique_benchmark_pairs × 4)` float16 tensor (~180 MB) holding shared-partner STRING/CORUM weighted mean expression and mean $\hat{r}_{\text{lin}}$, plus pair sequence identity and co-expression prior.
   - Symmetric defect-vulnerability terms: $\mathbb{I}[A \text{ silenced/lost in } c] \cdot \hat{r}_{\text{lin}}(c, B) + \mathbb{I}[B \text{ silenced/lost in } c] \cdot \hat{r}_{\text{lin}}(c, A)$ for combinatorial screens; directional $(1 - \text{combinatorial}) \cdot [x(c, A), \hat{r}_{\text{lin}}(c, B)]$ for anchor screens (`x(c, -1) = 0`), preserving exact $(A, B) \leftrightarrow (B, A)$ symmetry on all combinatorial rows while distinguishing lost anchor `A` from candidate target `B` on anchor screens.

6. **Batch-Cleaned Global Pathway, Cell-State, & Lineage Context (`global_context`, 100% of lines)**:
   - Pathway & program scores on wide expression: PROGENy (14), Hallmark (50), non-CCLE Gavish 3CA meta-programs (21), Tirosh S and G2/M (2) — 87 features standardized on fit lines.
   - Single-cell subpopulation distribution (`ladder/summaries.parquet` & `states/`): K-means / `scVI` / `cNMF` cell-state shares and Shannon entropy across cells (16–24 features; masked for bulk-only lines).
   - Deposit-whitened foundation-model cell tokens (for the foundation-model ablation arm only): fit a 64-component PCA + within-deposit mean removal (fitted on training deposits with $\ge 3$ lines) on cached `Tx1-3B` embeddings so `cell_token` reads 64 whitened dims instead of 2,560 raw deposit-confounded dims.
   - Metadata / text arm (multi-omic / metadata ablation): 16-PC projection of `ladder/context_text` (Cellosaurus embeddings) + lineage one-hot / non-cancer flag from `Model.csv` and `sl_line_identities.csv` (in scRNA-only mode, replaced by scRNA-predicted lineage probabilities + Cellosaurus text off).

7. **Complementary +36 Bulk-RNA Screened SL Training Lines (`bulk_sl_train` option)**:
   - Include the 36 patient-clean `ACH-*` lines in `bulk_labelled` / `bulk_unlabelled` that carry two-class genetic SL screens in `sl_label_rules.csv`, expanding Stage-2 training from **44 → 80 cell lines** (Stage-1 `fit` already excludes every line with an SL label and its patient-mates in every mode, so every SL training line remains out-of-sample for Stage 1 and the fit set does not change with this option). Validation (17) and test (16) stay strictly scRNA-only and identical, reported side-by-side per blueprint §3.6.

### 7.2 Model Equations & Parameter Discipline (`src/model/sl_context_gated.py`)

- **Gene Knowledge Encoder**:
  - Frozen sources: `gene_text`, `esmc_600m`, `esm3_open` (structure-aware protein embedding already built in `ladder/gene_sources/`), and `string_space`.
  - $u_g^0 = \text{LayerNorm}(\sum_s P_s(k_{g,s}) \cdot \text{present}_{g,s}) \in \mathbb{R}^{32}$ (zero free per-gene parameters; generalizes to new pairs and prevents 44-line gene memorization).
  - $u_g = u_g^0 + r_g$ ($r_g \in \mathbb{R}^{32}$ with weight decay `0.05`, used **only** in the context-blind `s_pair` and Stage-1 `w_base · u_g`, **never** in $\Delta s_{\text{ctx}}$!).

- **Stage-1 Forward & Loss**:
  - Local projection $e(c, g) = W_{\text{loc}} x(c, g) \in \mathbb{R}^{32}$; global line state $h_0(c) = \text{MLP}_{\text{glob}}(g(c)) \in \mathbb{R}^{32}$ (plus optional attention over 64-dim whitened cell tokens).
  - Context residual embedding: $V^{\text{ctx}}(g, c) = u_g^0 \odot \text{MLP}_{\text{read}}([h_g(c) ; e(c, g)])$, last layer zero-initialized.
  - Prediction: $\hat{z}(g, c) = w_{\text{base}}^\top u_g + \gamma_{\text{lin}} \hat{r}_{\text{lin}}(g, c) + \sum_{k=1}^{32} V_k^{\text{ctx}}(g, c)$ (where $\hat{r}_{\text{lin}}$ is scaled to the global `target_sd` units).
  - Loss per `(lines_per_step=64, genes_per_step=512)` block:
    $$\mathcal{L}_{\text{S1}} = \text{MSE}(\hat{z}, z_{\text{target}}) + \lambda_{\text{cos}} \frac{1}{|G_{\text{sel}}|} \sum_{g \in G_{\text{sel}}} \Big(1 - \cos\big(\hat{z}_{\text{ctx}}(\cdot, g) - \overline{\hat{z}}_{\text{ctx}}(g),\; r_{\text{true}}(\cdot, g)\big)\Big)$$
    Early stopping selects the epoch minimizing combined validation loss on the out-of-sample SL training lines with GeneEffect.

- **Stage-2 Forward (Exact Additive Context Decomposition)**:
  - **Context-blind pair score** ($s_{\text{pair}}$):
    $$\phi_{\text{pair}}(A, B) = [u_A + u_B ;\; |u_A - u_B| ;\; u_A \odot u_B ;\; p_{\text{static}}(A, B) ;\; \text{combinatorial}]$$
    $$s_{\text{pair}}(A, B, \text{design}) = \text{MLP}_{\text{pair}}(\phi_{\text{pair}}(A, B))$$
    where $p_{\text{static}}(A, B)$ holds context-free paralog identity, CORUM co-membership, and STRING combined score.
  - **Gene-gated context deviation** ($\Delta s_{\text{ctx}}$, zero-initialized output layer, strong dropout `0.25` + weight decay `0.05`, hidden width `64`):
    $$\psi(c, A, B) = [\text{sym}(V^{\text{ctx}}_A, V^{\text{ctx}}_B) ;\; \text{sym}(e_A, e_B) ;\; r(c, A, B) ;\; g(c)]$$
    $$[\gamma_{AB}, \beta_{AB}] = W_{\text{gate}} [u_A^0 + u_B^0 ;\; |u_A^0 - u_B^0| ;\; u_A^0 \odot u_B^0 ;\; \text{combinatorial}]$$
    $$\Delta s_{\text{ctx}}(c, A, B) = \text{MLP}_{\text{ctx}}\big((1 + \tanh(\gamma_{AB})) \odot \text{LayerNorm}(\psi(c, A, B)) + \beta_{AB}\big)$$
  - **Reference-centered score**:
    $$s(c, A, B) = s_{\text{pair}}(A, B, \text{design}) + \Delta s_{\text{ctx}}(c, A, B) - \frac{1}{|R|} \sum_{r \in R} \Delta s_{\text{ctx}}(r, A, B)$$
    At evaluation:
    - `model` score = $s(c, A, B)$;
    - `context_ablated` score = $s_{\text{pair}}(A, B, \text{design})$ (exact zeroing of the centered deviation — always in-distribution!);
    - `other_line` score = $s_{\text{pair}}(A, B, \text{design}) + \Delta s_{\text{ctx}}(c', A, B) - \bar{\Delta}_R(A, B)$ (with seeded permutation $c' \neq c$, $R$ unchanged).

- **Stage-2 Joint Training Objective**:
  1. **Target**: Continuous top-10% screen rank ramp $y_{\text{ramp}}(c, A, B) \in [0, 1]$ from SL protocol §16 (fallback to binary `label` on any screen lacking continuous scores), weighted equally across lines → studies → screens → positive vs zero-ramp mass (`soft_weights` in `conditional.py`).
  2. **Pair-branch dropout in screen loss**: On each training step, with probability $p_{\text{drop}} = 0.30$ per row, detach and scale $s_{\text{pair}}$ by `0` in $\mathcal{L}_{\text{screen}}$ (computing loss on $\Delta s_{\text{ctx}} - \bar{\Delta}_R$ plus screen prior bias alone) so $s_{\text{pair}}$ cannot starve $\Delta s_{\text{ctx}}$ of gradient on agreed pairs.
  3. **Within-study cross-line contrastive loss ($\mathcal{L}_{\text{xline}}$)**: Pre-index all pairs $(A, B)$ within each study that have at least one positive training line ($y_{\text{ramp}}(c^+) \ge 0.5$ or `label == 1`) and one negative training line ($y_{\text{ramp}}(c^-) == 0$ and `label == 0`). Each step samples $B_{\text{xline}} = 256$ such quartets $(\text{study}, A, B, c^+, c^-)$ weighted evenly across studies and minimizes:
     $$\mathcal{L}_{\text{xline}} = \text{Softplus}\big(m - [\Delta s_{\text{ctx}}(c^+, A, B) - \Delta s_{\text{ctx}}(c^-, A, B)]\big), \quad m = 1.0$$
     Because $s_{\text{pair}}$ and $\bar{\Delta}_R$ cancel identically between $c^+$ and $c^-$, static pair identity cannot satisfy $\mathcal{L}_{\text{xline}}$.
  4. **Schedule**: Two phases as before — Phase 1 freezes Stage-1 `GeneEncoder` and `V_ctx` while training `MLP_pair`, `W_gate`, and `MLP_ctx` jointly; Phase 2 unfreezes Stage 1 at `backbone_learning_rate = 5e-5` alongside $\lambda_{\text{ge}} \mathcal{L}_{\text{S1}}$. Validation top-10% recall on binary benchmark calls (unchanged evaluation!) selects best epoch and phase.

### 7.3 The First Draft's Config Matrix (Deferred; Replaced by §6)

Its first row changes the stage-one split, the pair score, the gate, the target, the pair dropout, the cross-line loss and the joint training together, so a null result there has no single cause. Each of these is one row after §6.

To follow the modular attribution discipline of `CLAUDE.md` and SL protocol §11–§19, configs are structured as a clean 6-row experiment matrix (one GPU each, ~3–5 min per run after cache build, all scored in one `sl-baselines` harness run):

| Config (`configs/sl/`) | Query Contract | Added Cell-Line Data | Architectural / Training Change Tested |
| :--- | :--- | :--- | :--- |
| `context_gated_base.yaml` | scRNA-only | Existing narrow pseudo-bulk only (same data as §19 `mean_token`) | **Architecture & Objective alone**: Stage-1 residual split + joint `s_pair + Δs_ctx` FiLM + ramp target + $\mathcal{L}_{\text{xline}}$ vs §19 `embedding_direct_mean_token` |
| `context_gated_scrna_local.yaml` | scRNA-only | + Wide (~19k) unbridged scRNA + `wide_linear_ge` + paralog/CORUM/STRING buffers | **Local scRNA & network buffers** (fixes 64% missing-gene blind spot & adds Kebabci-style relational context to all pairs) |
| `context_gated_scrna_full.yaml` | **scRNA-only (Primary Headline)** | + `inferCNV` copy number & CIN + PROGENy/Hallmark/3CA/cycle + `cNMF`/`scVI` states + scRNA MSI/driver scores | **Full scRNA-derived context** (tests all biological layers legal under strict basal-scRNA-only deployment) |
| `context_gated_scrna_tx1.yaml` | scRNA-only | `scrna_full` + deposit-whitened 64-PC Tx1-3B cell tokens & Stage-1-compressed Tx1 gene residuals | **Foundation-model ablation** (clean test of Blueprint Task 2: what AI-virtual-cell representations add beyond scRNA pseudo-bulk, states, and `inferCNV`) |
| `context_gated_multiomic.yaml` | **Multi-omic (Blueprint §3.2)** | `scrna_full` + masked DepMap somatic mutations (damaging + hotspot) + WGS CN + Cellosaurus text & lineage | **Direct DNA & metadata lift** (measures exact value of coding mutations & focal WGS CN over scRNA-inferred proxies) |
| `context_gated_scrna_80lines.yaml` | scRNA-only at test (+36 bulk train lines) | `scrna_full` trained on 44 scRNA + 36 bulk-RNA SL lines (80 training lines total) | **Context sample-size scaling** (complementary model of Blueprint §3.6; tests whether doubling training lines unlocks sharper generalization) |

#### The First Draft's Success Criteria (Replaced by "What the Report Must Show" in §6)
1. `own_line − other_line` 95% line-bootstrap interval strictly above zero on both validation and test for `context_gated_scrna_full` and `context_gated_multiomic` (currently +0.044 / +0.003 with cell tokens, +0.062 / +0.032 without).
2. Positive `own_line − other_line` lift on **seen pairs where training lines differ** and on **no gene a dependency alone** (proving context lift is not merely single-gene pan-lethality).
3. Positive paired difference `model − context_gated_base` isolating the exact contribution of each added cell-line data layer.

## 8. Explicit Recommended Calls vs Builder Decisions

### Recommended Design Calls for §5 and §6
- **The cross-line loss in the current model before any new module**: the score `f + D − mean_R D` is in the code, and the loss is the only part that static pair knowledge cannot satisfy.
- **One change for each row**: a null result then has one cause (CLAUDE.md: find why a null run did not help).
- **Local features as gene-in-line caches**, not a new `data/sl_context/v1` root: the cache format, the scaling on the SL training lines and the presence flag exist.
- **The SL call stays the target** in §6; the ramp is a later single row (§7).
- **Two query modes side by side** (`cross_line_wide` as the scRNA-only headline, `cross_line_depmap_dna` as the multi-omic row), as in the first draft.
- **The base is the model without cell tokens**; Tx1-3B cell tokens come back as a later row (§7).

### Left to the Builder for §5 and §6
- How a step draws the two lines of an example and their tokens (`cross_line.examples_per_step: 256`, `cross_line.weight: 0.5`, `eval.other_lines: 4`, pinned in the YAML configs).
- The split of the buffer columns into caches by presence.
- The top value of the cross-line ordering from Harle's replicates (optional).

### Recommended Design Calls of the First Draft (They Stand for the Deferred Design of §7)
- **Dual query modes (`scrna_full` headline + `multiomic` side-by-side)** rather than choosing only one: keeps the paper's core basal-scRNA virtual-cell claim intact while quantifying how much true DNA mutation/CN adds when available (Blueprint §3.2).
- **Joint decomposed `s_pair + (Δs_ctx − mean_R Δs_ctx)` with $\mathcal{L}_{\text{xline}}$** instead of post-hoc `line_term`: algebraically eliminates static pair shortcuts in the context loss and makes `s_pair` an exact in-distribution `context_ablated` control.
- **No free `r_g` offset inside $\Delta s_{\text{ctx}}$**: `r_g` stays restricted to `s_pair` and Stage-1 `w_base`, forcing the context modulator to generalize through biological gene sources (`gene_text`, `esmc_600m`, `esm3_open`, `string_space`).
- **Continuous top-10% rank ramp (`ramp`) as Stage-2 supervision**: already proven 2× more repeatable across studies in §16 (`0.116` vs `0.060`), while validation selection stays on the exact benchmark headline (top-10% recall on binary calls).
- **Separate module `src/model/sl_context_gated.py` and runner `src/experiments/sl/context_gated.py`**: leaves finished §18–§19 `sl_embedding.py` checkpoints and tests 100% reproducible.

### Left to the Builder of the Deferred Design (§7)
- Exact chromosomal coordinates source for `inferCNV` windowing (can read Ensembl/GENCODE gene positions from existing UniProt/GTF/BioMart reference tables or track a compact `configs/linear_model/reference/gene_loci_grch38.tsv` built on the Mac).
- Chunk sizes and float16/float32 memory layout of `data/sl_context/v1/` so all 80–1,050 lines fit comfortably inside one H20 GPU (< 12 GB VRAM).
- Exact regularization grid defaults (`λ_cos = 0.2`, `λ_xline = 0.5`, `λ_ge = 1.0`, `p_drop = 0.3`, `head_hidden = 64`), pinned in the YAML configs.

## 9. Changes from the First Draft (Review of 2026-11-24)

- The design to build is the cross-line loss in the current model (§5), not a new model module; the context-gated model is deferred (§7).
- The rows change one thing each (§6). The first draft's first row changed about eight.
- The risk that the deviation fits (line, pair) is named, with its two records, an input of lower capacity and a report for it (§5.3).
- Stage one does not change. The residual split with a linear anchor is deferred, with the reason (§7).
- Data: local features of the two genes first, as gene-in-line caches (§5.4). Inferred copy number, global line state and whitened Tx1-3B tokens are deferred.
- Measures: the cross-line ordering in held-out lines and the other-line control over four lines are new (§5.5). The first draft's first criterion is already met by the reference row.
- Facts corrected against the records: Kebabci's 0.723 / 0.685 is on the 108,052 rows it covers, not on 1.4M pairs; the stage-one fit already leaves out every line with an SL label (928 fit lines, not about 892); "85%" is the share of the GeneEffect variance between genes, not a share of the gradient; the measured post-knockout profile scored 0.114 against random 0.10; the causes in §1 are hypotheses.
- The training SL fraction is not a goal bar and is not named as a control.
