# Design: SL-aligned selection, a factorised head and three STATE settings for the joint GeneEffect model

**Status:** design, 2026-10-03, agreed in conversation (grill session). Bound by the
[blueprint](../01-blueprint.md) claim boundaries. Builds on the
[expression-space design](2026-10-02-expression-space-and-all-pipeline-design.md) and the `all` run
`all_20261002T174946Z`. Replaces only the joint model's head, objective, selection and STATE
treatment; preparation, splits, baselines and the response comparison are unchanged.

## 1. Why

1. **Loss and correlation disagree.** Training minimises Huber (delta 1) pooled over every observed
   pair of ~18.5k genes. Residuals have SD ≈ 0.26, so this is MSE dominated by high-variance genes
   and it rewards shrinkage. Residual Pearson is scale-free and macro per gene. In the `all` run
   validation Huber stayed at 0.0163–0.0166 from epoch 0 while residual Pearson rose 0.035 → 0.079,
   prediction SD was 0.10–0.16 of the target SD, and `best.pt` (selected on Huber) was not the
   best-correlated epoch.
2. **The model cannot fit its training set.** Training residual Pearson reached 0.23 after 10,791
   updates at head learning rate 1e-4 and was still rising. The head is one MLP over a flat
   concatenation dominated by 5,120 Tx1 and 1,280 ESM2 channels; a gene × context effect must be
   built from additive pieces. A readout with an explicit rank-8 gene × context slope reached the
   context-ridge level in three epochs on the old backbone; the flat MLP stayed below half of it.
3. **Fine-tuned STATE does not transfer and loses gene identity.** In the response comparison of
   the `all` run, STATE as in the joint model scored 2.16× no-change on held-out anchors without
   HCT116 (4.80× with it), against 1.05× for an MLP on HVG and 1.09× for the untrained released
   checkpoint, and its gene-shuffle identity share fell from 0.20–0.56 (released) to 0.002–0.06.
   Response replay at weight 1 (loss ≈ 0.8 against GeneEffect ≈ 0.015) is what drives STATE there.
4. **The metric should follow the SL computation.** The intended SL computation is statistical and
   cohort-based (DAISY/ISLE/SLIdR style): for a pair (a, b), test whether lines in which b is lost or
   low show stronger dependency on a. That is a rank test on gene a's dependency across lines, so the
   quantity it consumes is the per-gene ranking of lines, concentrated on genes that have a
   dependent tail. A true genetic-interaction score is out of scope: it needs a joint phenotype that
   the single-gene model does not produce.

## 2. Decisions

| Topic | Decision |
| --- | --- |
| Selective genes | Labelled training lines only: GeneEffect < −0.5 (dependent) in ≥ 5 lines and in < 90% of lines. 3,111 genes on the current split (1,004 commonly essential excluded). Fixed preprocessing, stored with the checkpoint |
| Selector | Macro per-gene Spearman of ŷ across the 27 validation lines over selective genes (`val_selective_spearman`). Equal to residual Spearman, since the gene mean is constant per gene. Drives `best.pt` and early stopping |
| Secondary metrics | Per-gene dependent-line AUPR lift (y < −0.5; AUPR − prevalence) over selective genes with ≥ 1 dependent and ≥ 1 non-dependent validation line; all existing metrics kept as telemetry. Same metrics for every baseline |
| Head | Factorised: δ = ⟨G(g), C(g, c)⟩ + MLP(F). G(g) = free per-gene embedding + a linear map of the ESM2 embedding (separate from the adapter that drives STATE), rank 64. C(g, c) from Tx1 z_c, basal statistics q_sc and, when STATE is used, the response blocks Δ_proj and s. MLP(F) is the existing head, kept as an additive path. No per-line lookup anywhere. The head predicts in units of σ_g and the model multiplies by σ_g, for every objective |
| Objectives (variant axis) | (1) Huber on residuals, random batches (current). (2) MSE on residuals standardised by the training-line residual SD σ_g, floored at its 10th percentile over genes, random batches. (3) Gene-blocked batches (every training line of 6 genes per rank) with mean over the block's selective genes of (1 − Pearson across lines) of standardised residuals + 1.0 × objective (2). Every metric scores the σ_g-rescaled prediction |
| STATE settings (variant axis) | (a) frozen at released weights, ESM2 adapter trained by the GeneEffect loss; (b) STATE trainable at 1e-5 with the adapter, GeneEffect loss only; (c) no STATE: the same head without the Δ_proj and s blocks. No response replay in any setting |
| Optimiser | AdamW; head and gene embeddings 1e-3, ESM2 adapter 1e-4, STATE 1e-5 when trainable; linear warmup over the first epoch, cosine decay to 0 at `max_epochs` 30; patience 5 on the selector |
| Fit | Telemetry only: training-line selector, loss and SD ratio every epoch on the fixed 27 training diagnostic lines. No probe and no gate |

## 3. Staging

All runs use the four H20s of the container on port 30734, one run at a time, seed 0. One config is
one experiment: train, select `best.pt` on validation, then score it once on test. There is no
multi-seed stage.

1. **Objective screen.** STATE setting (a) under objectives (1), (2), (3). The winner is the highest
   `val_selective_spearman` at its `best.pt`.
2. **STATE screen.** The winning objective under settings (b) and (c); setting (a) is reused from
   stage 1. The winner is chosen the same way; when two settings differ by less than the 27-line
   paired bootstrap interval, the simpler one (c, then a, then b) is preferred.

Every run reports, on validation and on test, against the `all` run's baselines on the same metrics: the Tx1 and HVG context-PCA
ridges, nearest line, K562 copy prior and gene mean, with a 27-line paired bootstrap of the selector
difference to the Tx1 ridge.

## 4. Execution

- Code on branch `feat/geneeffect-revision`; the 30734 container uses its own git worktree of the
  shared repository (`/2023533015/VCC_Project_revision`) with `data/`, `model/` and `.venv-tx1`
  linked from the main checkout, so the main checkout stays on `main`.
- A route that runs training, then validation and test evaluation with the baseline metrics, for one
  config, without the response comparison: `hpc/run.sh revision CONFIG --run-id ID`. Outputs under
  `outputs/geneeffect_revision/<id>/`.
- One config per variant under `configs/revision/`. The prepared root is reused; nothing is
  re-prepared.

## 5. Documents

`docs/03-geneeffect-protocol.md` §4–§6 (head, objective, selection, metrics) and `CLAUDE.md`
(the `best.pt` rule) change with the code. Results go to `results/geneeffect_revision/` after
the runs.

## 6. Claim boundaries

Variants are chosen on validation; each run's test numbers are reported, never used to choose.
Selective-gene Spearman is a GeneEffect diagnostic
chosen for its alignment with a cohort-based SL test; it is not SL evidence, and no SL computation
runs here. The selective-gene set is fitted on training lines only.
