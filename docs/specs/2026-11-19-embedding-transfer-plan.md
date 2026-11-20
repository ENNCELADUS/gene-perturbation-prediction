# Plan: stage one as pretraining, stage two on the deep model's embeddings

Date: 2026-11-19. Status: run on 2026-11-19 (SL protocol §18,
[record](../../results/sl_embedding_seed0/summary.html)): above the published baselines on all pairs; the pretraining and
the cell tokens add nothing. This plan follows the goal set on 2026-11-19 (blueprint §5):
the result that counts is SL ranking in held-out lines. Stage one is pretraining. Its GeneEffect accuracy is reported
and is no bar. Stage two reads the deep model's embeddings, not its predicted GeneEffect. It is a plan, not a
protocol. The study asks one question of one architecture (§3): can a deep model beat the baselines on SL?

## 1. Why

- **A better GeneEffect value cannot lift SL.** The measured GeneEffect of the two genes as the score gives recall in
  the top 10% of 0.412 validation / 0.269 test; the predicted value gives 0.371 / 0.276
  ([record](../../results/sl_buffer_aware_reader_seed0/README.md)).
- **An embedding gave more than the value, once.** On the superseded three-screen benchmark, a head on the deep
  model's embedding of each gene, averaged over lines, scored a mean test AUPRC of 0.500, against 0.403 for the
  measured GeneEffect and 0.36 for the predicted one ([record](../../results/sl_stage_two_seed0/README.md)). That run
  scored only pairs already labelled in a training line, so the embedding could act as a gene identity. The embedding
  of the gene in the line scored 0.235 to 0.305, below its own average.
- **Two faults of that run are removed here.** The head trained on lines whose GeneEffect the deep model had fitted,
  and was scored on lines it had not seen. And stage one keeps the checkpoint with the best per-line residual
  Spearman, which for 7 of 8 per-gene reader runs was the linear expression model with nothing learned in the deep
  part.
- **The line-specific part of the labels is small** (SL protocol §15): a perfect model reaches a correlation of about
  0.20 to 0.25 with it between studies, and the same-line advantage is clear only between the paralog-library studies
  (Ito 2021, Klingbeil 2024, Ebright 2026). So the plan makes two separate claims (§4).

## 2. Stage-one data and loss

**Stage-one data.** The split is the SL split (`configs/benchmarks/cell_line_sl_split.json`: 44 SL training lines, 17
validation, 16 test). Exact counts are fixed when the inputs are built.

| Data | Lines | Use |
| --- | --- | --- |
| DepMap GeneEffect 26Q1, every GeneEffect gene | about 935: the 979 labelled fit lines of the SL split less the SL training lines | the pretraining target, `(GeneEffect − gene mean) / residual SD` |
| GeneEffect of the SL training lines | up to 44 | not fitted; the early stop of pretraining (out of sample), and the stage-one numbers |
| GeneEffect of the SL validation and test lines | 33 | never read by a model; stage-one numbers only |
| Bulk RNA | the bulk-only labelled lines (about 600) | line input for a line without cells |
| Basal single cells | 455 lines with cells, median about 260 cells | line input: their mean, or the cells themselves (§3) |

No SL line, training or held out, gives a GeneEffect label to the fit. Thus one model reads every SL line out of
sample, in one coordinate system, and no fold models are needed. Gene means, the residual SD and input scaling are
fitted on the labelled fit lines only.

**Stage one.** The model gives a 32-wide gene-in-line embedding `V(g, c)` whose sum is the predicted GeneEffect
residual `z(g, c)`. Pretraining minimises the GeneEffect loss and stops on that loss on the SL training lines. There
is no comparison with the linear expression model.

## 3. The architecture

One model, chosen on 2026-11-19 after the discussion and the survey: the gene-knowledge, cell-set model. It uses the
three kinds of information: what the gene is, what state the line's cells are in, and what the gene does in the line.

```text
gene g ──► gene encoder ──► u_g ───────────────┐
                                               ├─► gene-specific reading ──► V(g, c) ──► sum ──► GeneEffect
cells of line c ──► line encoder ──► tokens t_i ┘

pair (A, B) in line c ──► pair head on V and on the cells ──► SL score
```

| Module | Form | Frozen models and sources | Inductive bias |
| --- | --- | --- | --- |
| Gene encoder | `u_g = P_text(k_text) + P_prot(k_prot) + P_net(k_net) + r_g`: one linear projection for each source, summed, plus a free deviation `r_g` pulled to zero by weight decay; 32 wide | function text (text-embedding-3-large over NCBI Gene descriptions, PIE's source), ESM C 600M protein embedding, STRING v12 network space | genes with related function, protein or network position start close together; the free part holds what GeneEffect adds |
| Line encoder | tokens, 32 wide: the mean token `t_0`, an MLP over the mean expression of the line; a cell token `t_i = W e_i` for each cell `i`. Any number of cells: a random subset at each training step, all at scoring. A bulk-only line has `t_0` only. The number of cells is no input | Tx1-3B embedding `e_i` of each cell (cached) | the line is a set of cells, not one profile; order and count of the cells do not change the output |
| Gene-specific reading | `α_g = softmax_i(q(u_g) · t_i)` over the cells, `h_g(c) = t_0 + Σ_i α_gi t_i`, `V(g, c) = u_g ⊙ h_g(c)`, `z(g, c) = Σ_k V_k(g, c)` | none | each gene weights the line's cells in its own way; the weights are normalised over the cells alone, so the number of cells does not move the mean token's share |
| Pair head | line level: `[V_A + V_B ; \|V_A − V_B\| ; V_A ⊙ V_B]`. Cell level: `Σ_i w_i (u_A ⊙ t_i) ⊙ (u_B ⊙ t_i)`, with `w_i` the normalised product `α_Ai α_Bi` over the cells. Both, with the design of the screen, go into one MLP that gives `s(c, A, B)` | none | the score is the same for (A, B) and (B, A); the cell-level term holds whether the two genes' states are in the same cells, which no line vector can hold |

No perturbation-response data and no perturbation model are used. PIE describes a gene by the same sources as the
gene encoder, plus a DepMap profile that is not allowed here as an input.

## 4. The experiment

One config, seed 0. The split is the SL split; every held-out line enters only through its basal cells.

| Step | Lines | What is fitted or scored |
| --- | --- | --- |
| Stage one, training | about 935 lines with GeneEffect and no SL label (about 340 with cells, about 600 bulk only) | gene encoder, line encoder and reading, on the GeneEffect loss |
| Stage one, early stop | the SL training lines with GeneEffect (up to 44), out of sample | the epoch with the lowest GeneEffect loss |
| Stage two, training | the 44 SL training lines: their genetic screens with both classes | the pair head on the frozen stage one; then the whole model, with the GeneEffect loss kept at weight λ. Loss: binary cross-entropy weighted as the headline (each line equal, then its studies, a study's screens, a screen's two classes) |
| Validation | the 17 SL validation lines | recall in the top 10% selects the epoch, λ, the head's settings, and frozen against trained further |
| Test | the 16 SL test lines, scored once | recall in the top 10%, AUROC, AUPRC over the positive rate; mean over lines, by the harness (`hpc/run.sh sl-baselines`) |

The bar is the best published baseline on the same harness (run of 2026-11-13, validation / test): Kebabci on
predicted GeneEffect 0.355 / 0.395, KG4SL 0.367 / 0.338; the current head scores 0.394 / 0.361. The GeneEffect
numbers of the model on the SL lines are reported after the SL table and decide nothing. Ablations and comparisons
between designs are not part of this plan yet.

## 5. What must be built

| Part | Reuses |
| --- | --- |
| Stage-one inputs on the SL split, with the SL lines as input only; Tx1 embeddings of the SL lines that lack them | the stage-one input steps (`src/experiments/stage_one/ladder_inputs.py`), the Tx1 cache |
| The model: gene encoder with the free deviation, cell tokens, the gene-specific reading | the gene encoder, `GeneReader` and the per-channel attention of `src/model/stage_one.py`; the stage-one trainer |
| A stage-two module on the benchmark rows that trains the pair head, trains the whole model further and writes score tables for the harness | `fit_head` (`src/experiments/sl/stage_two.py`); rows and weights of `src/experiments/sl/conditional.py` |

## 6. Background

Earlier candidates, not in the study now: the same model with a free gene vector only, with the mean token only, or
with PCA cell tokens; an additive form `Γ(u_g + h(c))` (C3TL, arXiv 2603.13051); PIE's predicted response as the
gene-in-line embedding.

Survey of the latest AIVC models (2026-11-19, an agent's reading of abstracts and project pages; two sources
checked directly):

- No model found predicts GeneEffect or SL from basal single cells in held-out lines. The nearest work: Cilantro-SL,
  Mahi (bioRxiv 10.64898/2026.02.17.706433) and SLxGO (bioRxiv 10.64898/2026.07.31.742101, hold-out over genes).
- CellNet (arXiv 2610.21943) is a single-cell foundation model that renders each cell as an image; it predicts no
  perturbation response or dependency.
- "Cross-Cell-Line Perturbation Prediction Needs Controls" (NeurIPS 2026 poster 139089): none of five frozen
  single-cell foundation-model encoders, as the context of each cell, matches a PCA or an empty encoder in a strict
  held-out-line test. This is the main risk of the Tx1 cell tokens.
- Cole et al. (bioRxiv 10.64898/2026.02.18.706454): knowledge embeddings of genes (STRING, text) are above
  foundation-model gene embeddings for perturbation prediction.
- Stack (bioRxiv 10.64898/2026.01.09.698608) and X-Cell (Xaira): a set of cells as the unit, gene tokens that attend
  to knowledge tokens; results reported by their own authors only.

## 7. Decisions

Accepted by the user on 2026-11-19:

1. Stage-one selection for this track is the GeneEffect loss on the SL training lines, not the per-line residual
   Spearman on validation; the whole model is selected on validation SL. GeneEffect protocol §6 and §11 change when
   the first design is built.
2. The SL training lines give no GeneEffect label to the pretraining fit.
3. The aim is whether a deep model beats the baselines on SL; a better GeneEffect is not an aim.
4. One architecture (§3), with no ablations or comparisons between designs for now.
