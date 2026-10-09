# Design: reproducing Cilantro-SL, then porting it to the context screen and basal single cells

**Status:** design, 2026-10-08, agreed in conversation (grill session). Nothing is implemented. The
port (phase 2) follows the [SL protocol rules](../04-sl-ranking-protocol.md#8-rules). Paper: Hua,
Haber and Ma, *Uncertainty-aware synthetic lethality prediction with pretrained foundation models*,
bioRxiv 2026 (local PDF under `docs/literature/03_models/synthetic_lethality/`). Phase 1 reproduces
the paper on its own data with its own code; phase 2 carries the method to the
[context screen](../data/sl-context-screen.md) and the basal single cells of the
[226-line membership](../data/cell-line-geneeffect-226.md). Neither phase is SL evidence for the
held-out-line question until phase 2's test contexts are scored.

## 1. The method

Cilantro-SL has two stages.

- **Viability embedding.** Geneformer embeds each line's expression profile. Removing gene g's token
  and re-embedding gives the delta embedding Δ(c, g) = X(c) − X′(c, g), both read from the penultimate
  layer and mean-pooled over tokens. A FiLM layer, conditioned on g's frozen 128-d Gene2vec embedding,
  modulates Δ inside an MLP that regresses DepMap gene effect with MSE; the MLP's 32-d penultimate
  layer is the viability embedding V(c, g). A (c, g) exists only if g is among the line's top 2048
  tokens.
- **SL classifier.** An MLP on [V(c, a); V(c, b)] separates SL from non-SL with class-weighted cross
  entropy. Split conformal prediction uses 1 − softmax probability of the true label as the
  non-conformity score, gives conformal p-values with +1 smoothing, and takes the larger of the two
  labels' p-values as a pair's confidence.

## 2. The official release

| Item | Source |
| --- | --- |
| Code | [kaileyhh/Cilantro-SL](https://github.com/kaileyhh/Cilantro-SL) at `c674b55` (2026-05-24): seven notebooks and `nn_helpers/`; no licence |
| Geneformer | fork [kaileyhh/geneformer](https://github.com/kaileyhh/geneformer) at `0e2bf2a` (a fork of `jkobject/geneformer`, reverted to the 2024 API with gc95M default dictionaries); weights not in the fork |
| Data | Google Drive folder `1HFjP5Lwa1RoGKTpMYSoCxYNNwSFqu5dE` |

The Drive folder holds:

- `Human_SL.csv`: 35,943 SynLethDB 2.0 rows over 9,856 genes. Sources: GenomeRNAi 10,929,
  computational prediction 9,327, CRISPR/CRISPRi 7,296, text mining 5,614, other 2,107.
- `Human_nonSL.csv`: 2,899 rows, all CRISPR/CRISPRi.
- `gene2vec_embs.pt`: 24,447 Ensembl IDs × 128.
- `OmicsExpressionProteinCodingGenesTPMLogp1.csv`: DepMap **23Q4** (md5 `9402aa25a19279bb20a5d6cf8791a88f`
  matches the 23Q4 release index), 1,479 lines × 19,193 genes.
- `metadata.csv`, `Human_SL_pairs.pkl`, `ensembl_mapping_dict_gc95M.pkl`.

Not released: the gene-effect matrix the code reads (`CRISPRGeneEffectProcessed.csv`), the list of
genes to perturb (`Human_SL_genes.pkl`), the original-embedding table, the sampled negatives, fold
assignments, trained weights and every Geneformer output. The paper's gene effect is DepMap 23Q4
`CRISPRGeneEffect.csv` (md5 `8bfb35c18b7469020dbd5ae4b6274492`, 1,100 lines × 18,443 genes): its
lines meet the expression lines in exactly the paper's 1,021, and the paper's 1,078 × 17,453 come
from the unreleased processing, which matters only before that intersection. Phase 1 runs on 26Q1
instead (§4.4). NaN (line, gene) values are dropped, as the official `data_generator` does.

## 3. Where the code and the paper differ

Read from the code; phase 1 measures what each costs where it can.

1. **Model.** The paper names Geneformer-V1-10M (6 layers, 256-d). The code loads `gf-12L-30M-i2048`
   (12 layers, 512-d), and the FiLM net's input is fixed at 512.
2. **Dictionary.** The fork's default dictionaries are gc95M and the tokenizer is called without one.
   30M-cell models need the gc30M dictionary, so token IDs are very likely mismatched, silently.
3. **Stage-one fit is in-sample.** Stage one trains on every row (`train_and_test(0.0)`) and is then
   scored on a random 99.9% of the same rows. The reported RMSE 0.230, R² 0.85 and Spearman 0.864 are
   most likely training fit.
4. **Stage one precedes the split.** In gene-holdout the test genes' gene effects trained the
   viability embedder; only the classifier never saw them.
5. **AUPR is two-class.** `average_precision_score` on one-hot labels averages AP(SL) and AP(non-SL).
   The reported 0.7148 is not a positive-class AUPR. F1 is positive-class at the argmax.
6. **Unit.** Every (line, pair) row is scored; the paper describes averaging over lines to pair level.
7. **Negatives.** Sampling takes the most negatively correlated expression pairs first (ascending
   sort) until negatives reach 5× positives; the paper says low-correlation pairs.
8. **Conformal.** Calibration is a random 10% of the test fold's rows; the Mondrian class map is
   empty, so it is plain split conformal.
9. **Seeds.** Fold assignment and calibration sampling are unseeded (`random.shuffle`).
10. **Environment.** The pinned torch 1.13.1+cu117 has no Hopper kernels and cannot run on the H20.

## 4. Phase 1: reproduction on the paper's data

### 4.1 Runs

Every row uses 5× negatives, pair-holdout (the code's CV1) and gene-holdout (CV3), five folds and
conformal prediction.

| Run | What changes from the as-released run |
| --- | --- |
| As released (the reproduction) | nothing beyond the patch (§4.3): delta + Gene2vec FiLM, `gf-12L-30M-i2048`, gc95M dictionary |
| Matched dictionary | gc30M dictionary and gene medians; a second Geneformer pass |
| Gene-clean stage one | in gene-holdout, stage one fitted without the test fold's genes |
| Held-out stage-one fit | stage one fitted with a fixed, seeded 20% of lines held out and scored on them |
| 222-line stage one | stage one on a seeded random 222 of the 1,021 lines (the port's line count) |
| Paper ablations | perturbed embedding + Gene2vec; delta only; PCA-32 of Δ with no pretraining |
| Gene-only | Gene2vec of both genes through the same classifier, no Geneformer input |
| ESM+MLP | our ESM2 gene embeddings through the paper's ESM+MLP |

The paper's other six baselines (KG4SL, SL2MF, SLMGAE, DDGCN, MVGCN-iSL, ESM4SL) are not run.

### 4.2 Metrics

For every run:

- **Paper-comparable:** two-class AP and positive-class F1 at row level.
- **Standard:** positive-class AUPR, AUPR minus prior and AUROC, at row level and at pair level
  (mean over lines).
- **Stage one:** RMSE, Spearman and R², in-sample and on held-out lines.
- **Conformal:** empirical coverage and mean set size against 1 − α over an α grid, and confidence
  by SynLethDB label source.

The paper's numbers to read against: gene-holdout two-class AP 0.7148 (KG4SL 0.7325); coverage on the
diagonal; singleton sets up to about 98% target coverage; stage-one RMSE 0.230, Spearman 0.864,
R² 0.85; delta only R² −0.255. These are for reading, not a pass/fail check in code.

### 4.3 Where it runs

The harness is `hpc/cilantro/` (`setup.sh`, `run.sh`, one script per released notebook; usage in
`hpc/README.md`). The official repository (`c674b55`) and the Geneformer fork (`0e2bf2a`) live in
`third_party/` on the H20 only, untracked: the code has no licence and this repository is public, and
the H20 cannot reach GitHub, so both arrive by git push from the laptop. They run in their own
environment, `.venv-cilantro`, on Python 3.9.20 with the released pins except torch (2.6.0 on CUDA
12.4, since 1.13.1+cu117 has no Hopper kernels); Python itself comes from the npmmirror copy of
python-build-standalone. Seeds are set at 0 in each step (Python, NumPy, torch, `PYTHONHASHSEED`),
which fixes the released code's unseeded shuffles without editing it. `gf-12L-30M-i2048` and both
dictionary sets come from `ctheodoris/Geneformer` revision `01d3ea8`, through hf-mirror, checked
against the SHA-256 values Hugging Face reports. Run outputs go to `outputs/cilantro_reproduction/`
on the H20; curated results to `results/cilantro_reproduction_seed0/`.

### 4.4 Amendments (2026-10-09)

- **DepMap 26Q1 instead of 23Q4.** The H20 reaches neither Google Drive nor figshare, and the user
  chose the 26Q1 files already on it over moving the 23Q4 files through the laptop. Expression is one
  default profile per model (1,719 lines); 1,140 lines have both expression and gene effect, against
  the paper's 1,021. Phase 1 is therefore the released code on the paper's labels and gene prior
  with a newer DepMap release; its numbers are read against the paper's, not expected to equal them.
- **The four small Drive files went through the laptop.** `Human_SL.csv`, `Human_nonSL.csv`,
  `Human_SL_pairs.pkl` and `gene2vec_embs.pt` (34 MB) were fetched on the laptop, copied to
  `data/cilantro/official/` on the H20, checked by SHA-256 and deleted locally. SynLethDB's own site
  now serves only release 3.0, from Zenodo, which the H20 cannot resolve either.
- **What the released code needed to run.** `official.patch` makes the hard-coded `/work/magroup`
  paths configurable, passes the run's token dictionary to the perturber's extractor and perturber
  (identical to the default in the as-released run), sizes the extractor to the dataset instead of
  the hard-coded 1,479 lines, and fixes `run_cv`, which used the list `probs` as a dictionary key
  and crashed after the last fold. Outside the patch, the step scripts:
  - apply notebook 4's commented-out `set_index`, without which its loop cannot parse the index;
  - rebuild the perturbed gene list (`Human_SL_genes.pkl`, not released) as the tokenized gene
    universe;
  - write the unperturbed embeddings `data_generator` reads, which no notebook writes.
  As in notebook 6, conformal prediction runs on the first fold only.

## 5. Phase 2: the port

### 5.1 Data

- **Pair labels:** the nine-context context screen table and its tracked split (train K562, Jurkat,
  OVCAR8, HAP1, HT29; validation A549; test 22RV1, PC9, HeLa).
- **Cells:** basal single cells of the 226-line membership, which includes all nine SL contexts.
- **Stage-one supervision:** 26Q1 gene effect on the **222 labelled single-cell lines**: the 226
  minus the SL validation and test lines A549 and 22RV1; PC9 and HeLa have no gene-effect row. The
  226 benchmark's own validation and test lines are ordinary training lines here. The MIX-seq
  candidate lines are not used.
- **Gene embeddings:** the official `gene2vec_embs.pt`. A gene without one is unscorable and is
  counted in coverage.

### 5.2 Step 0: coverage report

Before any GPU work, a CPU job on the H20 tokenizes without running the model. For each context it
reports rows, positives and negatives with both genes covered:

- (a) pseudo-bulk input with the paper's vocabulary recipe: all benchmark genes plus 500 HVGs fitted
  on the stage-one lines' pseudo-bulk;
- (b) per-cell bags under §5.3.

The report is read before the deletion pass. Expected outcome: per-cell bags become the main arm, and
pseudo-bulk stays as the paper-faithful arm, scored on its own covered rows.

The reason, from a proxy on the stale four-context table: DepMap bulk TPM was divided by each gene's
across-line nonzero median, as Geneformer divides by its corpus medians, and the top 2048 tokens were
kept. The share of rows with both genes covered, and covered positives, by vocabulary size:

| Vocabulary | A549 | HT29 | Jurkat | K562 |
| ---: | --- | --- | --- | --- |
| 3.9k | 32%, 202/392 | 32%, 101/234 | 43%, 49/95 | 47%, 841/1,668 |
| 10k | 4%, 0/392 | 5%, 0/234 | 0.4%, 5/95 | 2%, 51/1,668 |
| 15.6k | 2%, 0/392 | 2%, 0/234 | 0%, 0/95 | 0.3%, 23/1,668 |

Median normalisation lets line-specific genes take the 2048 slots of a bulk-like profile and pushes
out the ubiquitously expressed genes that dominate SL screens. The paper escaped this by averaging
each pair over about 1,000 lines; each context screen label belongs to one line. A single cell
detects only a few thousand genes, so nearly all of them fit in 2048 tokens.

### 5.3 Geneformer deletion

- **Model:** `gf-12L-30M-i2048` with the matched gc30M dictionary and medians, whatever phase 1's
  dictionary runs show.
- **Per-cell bags (main arm):** the native gc30M vocabulary, unrestricted; raw counts tokenized as
  Geneformer expects. 128 cells per line, sampled with seed 0, or every cell when a line has fewer.
  In each cell where g is present, delete it and take Δ from the penultimate layer, mean-pooled as in
  phase 1. The line's Δ(c, g) is the mean over those cells. (c, g) is covered when g is present in at
  least 10 of the bag's cells.
- **Pseudo-bulk (paper-faithful arm):** summed raw UMI per line, fed in the paper's input form (log1p
  of a per-million scale), with vocabulary (a) of §5.2.
- **Implementation:** a batched deletion in `src/`, checked to give the official perturber's Δ, up to
  float tolerance, on a sample of phase-1 (line, gene) inputs. Per-cell Δ is reduced online; only
  line-level Δ is cached.

### 5.4 Stage one and the classifier

- **Architectures and hyperparameters** are the official code's, reimplemented rather than copied (no
  licence). Stage one: 512 → 256 (ReLU) → FiLM on the 128-d Gene2vec → 64 (ReLU) → 32 (V) → 1;
  Adam at 1e-3, batch 512, 100 epochs, MSE. Classifier: 64 → 32 → 16 → 8 → 2 with ReLU; Adam at
  1e-4, batch 128, 100 epochs. The last epoch is kept; nothing is tuned. A549 is reported only: one
  gene, TRA2A, scores AUPR 1.0 on it.
- **Out of fold** (protocol §5): the inner groups are {K562, Jurkat} (one aggregate screen cluster),
  {OVCAR8}, {HAP1} and {HT29}. Each group's V comes from a stage one fitted without that group's
  lines. Validation and test contexts use the stage one fitted on all 222 lines. The Geneformer Δ is
  label-free and computed once.
- **Symmetry** (protocol §1): the classifier trains on both gene orders and scores the mean of both,
  so s(a, b | c) = s(b, a | c) while the paper's concatenation is kept.
- **Weighting:** classes are balanced within each context and contexts are weighted equally. HAP1 is
  57,013 of 86,460 training rows and has 20 negatives.

### 5.5 Evaluation

- **Main:** the held-out-context split under protocol §7. Coverage and class counts come first, then
  AUPR minus prior per test context and as a macro (PC9 and HeLa count as one cluster in the
  de-duplicated macro), AUROC as secondary, endpoint-seen strata, and a two-way dyadic bootstrap with
  2,000 replicates. The paper's two-class AP and F1 are reported alongside.
- **Secondary:** the paper's pair-holdout and gene-holdout five-fold CV over the five training
  contexts only (gene-holdout with the gene-clean stage one), so test labels stay untouched.

### 5.6 Conformal

Calibration scores are the out-of-fold classifier scores of the four inner-group fits, with contexts
weighted equally. For each test context, empirical coverage and mean set size against 1 − α are
reported, both plain and class-conditional, along with leave-one-group-out coverage on the training
side. Calibration and test contexts are not exchangeable, so coverage on a new line is measured, not
guaranteed. The secondary CV keeps the paper's within-fold 10% calibration.

### 5.7 Controls

All are scored on the same covered rows:

- pair-identity prior and anchor-gene prior;
- null alone, −min(ŷ(a, c), ŷ(b, c)) from Cilantro's stage-one output, plus measured 26Q1 gene
  effect for 22RV1;
- the classifier on the gene-mean block alone;
- context-ablated Cilantro (Δ from fixed deranged lines, plus a permutation null);
- gene-only Gene2vec;
- ESM+MLP.

The protocol's baseline-ladder backbone and measured-profile arms belong to its own 24-d head and are
not built here.

### 5.8 Claim boundaries

- A classifier score is a ranking score. A conformal set on a new line carries empirical coverage
  only.
- Geneformer V1's pretraining corpus excluded malignant cells and immortalized lines, per its paper.
  Gene2vec was trained on GEO co-expression, which can include these lines.
- Phase 1's labels mix CRISPR, RNAi, text mining and computational predictions; phase 1 measures the
  reproduction, not the held-out-line question.

## 6. Order of work

1. Phase 1 environment, released code and patch; a smoke run end to end. Measure Geneformer
   throughput on 50 genes and record the estimate before the full pass.
2. The as-released run, then the fix runs, ablations and controls of §4.1.
3. The `src/` port: batched deletion with its equivalence check, then stage one, classifier and
   conformal, which must reproduce the as-released phase-1 numbers from the official Δ.
4. The coverage report (§5.2), read before GPU work.
5. Measure per-cell throughput on one line and record a GPU-hour estimate. Rough figure: 226 lines ×
   128 cells × about 3k present genes is about 90M forward passes, on the order of 80 GPU-hours.
   Then the full deletion pass.
6. The port: out-of-fold stage one, classifier, conformal, controls and secondary CV; test once.
   Results go to `results/cilantro_port_seed0/` with a run manifest (commits, input hashes, line and
   gene lists, seeds).

## 7. Out of scope

The paper's six other baselines, Geneformer V2, the MIX-seq lines, the SL protocol's 24-d head and
any change to the context screen table or its split.
