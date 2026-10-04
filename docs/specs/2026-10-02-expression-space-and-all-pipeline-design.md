# Design: one expression space, STATE on its own basal path, and a single automatic run

**Status:** design, 2026-10-02, agreed in conversation; awaiting review of this written form.
Bound by the [blueprint](../01-blueprint.md) claim boundaries. Supersedes the response wiring of
the joint training design (`2026-09-06-modular-joint-training-design.md`) and the
readout objective plan (`2026-09-10-readout-objective-and-selection-design.md`)'s ordering of
response-model work after readout work. Both were removed from the tree; `git show 1694f5c:docs/specs/<name>`.
Evidence it builds on: [response-pathway diagnostics](../results/p1_response_pathway_diagnostics/README.md).

## 1. Why

1. **STATE was fed the wrong numbers.** The released STATE ST-HVG-Replogle checkpoint was trained
   on arc-state `preprocess_train` output: `sc.pp.normalize_total` over the whole library (target
   = median library size), `log1p`, then the 2,000-gene HVG slice (`X_hvg`). The formal pipeline
   slices raw UMI counts to the HVG panel and never normalises
   (`src/data/response.py:_align_to_checkpoint_order`, `src/data/prepared.py:_open_lines`). The
   decoder's log-space output was scored against count-space targets, and the head's expression
   shift subtracted raw basal counts from log-space predictions. The log-space correction exists
   only inside the diagnostic harnesses, with an HVG-panel proxy for library size.
2. **STATE's basal input was a random layer.** The joint model replaced STATE's 2000→328 basal
   encoder with a new 2560→328 layer on Tx1 embeddings, trained at STATE's 1e-6. In log space this
   interface was the worst on held-out anchors (17.6–32.9× no-change). Published work offers no
   case of Tx1 or SE embeddings beating HVG expression as STATE's basal input on unseen cell lines
   (Tahoe-x1 bioRxiv 2025 Fig. 6B zero-shot: STATE on HVG ≈0.49 Pearson ΔC, on Tx1-3B ≈0.41,
   perturbation mean ≈0.40), and every embedding-input STATE there was trained end to end.
3. **Nobody has measured what STATE adds.** No arm has replaced STATE with a plain model on the
   same inputs.
4. **The code is mostly ceremony.** About 8,400 lines of closed diagnostic harnesses and a formal
   path with roughly 225 validation raises; a complete experiment needs many manual commands.

## 2. Decisions

| Topic | Decision |
| --- | --- |
| Expression spaces | Tx1 reads raw UMI. Every other expression quantity uses one space: `log1p(x · T / L_cell)` then the STATE HVG slice, with `L_cell` the cell's library size over all genes |
| `T` | Median library size of the non-targeting cells in the Nadig 2025 Jurkat and HepG2 sources, the data STATE's Replogle checkpoint was trained on; computed once at preparation and recorded, never configured |
| STATE basal input | STATE's own loaded 2000→328 basal encoder on log-space basal HVG expression |
| Perturbation input | ESM2 adapter → STATE's loaded perturbation encoder (unchanged) |
| Tx1 | Context embedding for the GeneEffect head only, and input to the STATE-free comparison arm |
| Response supervision | Auxiliary task that fine-tunes STATE; all conditions of all four anchors (K562, HepG2, Jurkat, HCT116); **no condition holdout** |
| Validation | One split: the 27 GeneEffect validation lines select `best.pt` and stop training |
| Learning rates | head 1e-4, ESM2 adapter 1e-4, STATE 1e-5 |
| Response-model comparison | Six-arm factorial, leave-one-anchor-out, fixed 50 epochs, reported on the held-out anchor |
| Code | Delete the closed diagnostic harnesses; keep the readout head and all evidence; simplify the whole formal path |
| Run | `hpc/run.sh all CONFIG` runs preparation through validation evaluation in one command; test stays separate |

## 3. Expression space

For each cell, `L_cell` is the sum of its raw UMI over every gene in the AnnData Tx1 reads for that
line (for atlas lines that is the Tx1-vocabulary-mapped matrix). The transform is applied once, in
preparation, before the HVG slice; caches store log-space float32. It covers:

- basal HVG bags of all 226 lines (STATE's basal input, the head's expression shift, the HVG
  context baseline features);
- perturbed-cell response targets and anchor control bags;
- `q_sc` mean and variance (detected fraction is space-invariant).

The prepared manifest records `expression_space = {transform: log1p_normalize_total, target_sum: T,
library_size: all_genes}` and the source of `T`. `load_inputs` refuses a manifest without it.

Known approximation: library sizes are taken over each source's gene universe, which differs
slightly between the Nadig, Replogle, X-Atlas-Orion and atlas matrices.

## 4. Joint GeneEffect model

Unchanged except: STATE's basal input is the log-space basal HVG bag through its own encoder; the
released checkpoint loads with no shape-skipped keys; the three parameter groups use the rates
above; response batches sample all conditions. GeneEffect Huber every update, response loss
(mean-shift MSE + energy distance, weight 1) every fourth update, balanced across anchors, batch
1024 dependency / 64 response conditions per rank, seeds 0. Response validation metrics disappear
from the per-epoch log; GeneEffect validation is unchanged.

## 5. Response-model comparison

Four folds; each trains on all conditions of three anchors and scores every condition of the fourth.

| Arm | Basal input | Response model | Trained |
| --- | --- | --- | --- |
| No-change | — | basal bag copied | no |
| Global mean effect | — | train-anchor mean shift, gene-blind | no |
| Released STATE checkpoint | log HVG | released STATE, one-hot gene vocabulary; scored on genes in that vocabulary | no |
| STATE as in the joint model | log HVG | STATE + ESM2 adapter, joint learning rates | yes |
| MLP on HVG | log HVG cell | `x + f([x ; a(e_g)])`, final layer zero-initialised | yes |
| MLP on Tx1 | Tx1 cell embedding `h` | `x + f([h ; a(e_g)])`, final layer zero-initialised | yes |

Trained arms: AdamW, new layers 1e-4, STATE 1e-5, 50 epochs, no early stopping, balanced anchor
sampling, seed 0. Reported per arm and fold: held-out loss ÷ no-change at the final epoch,
identity share (loss increase under ten fixed gene shuffles, as a fraction of loss), source-anchor
training ratio, and the per-epoch held-out curve for reading only. One interval: 1,000-resample
gene bootstrap of each arm's pooled held-out ratio, pooled over all four folds and over the three
folds without HCT116. Two verdict lines: STATE as in the joint model vs MLP on HVG (what the STATE
transformer adds); MLP on Tx1 vs MLP on HVG (what Tx1 adds as representation).

## 6. The `all` run

`hpc/run.sh all configs/geneeffect_joint.yaml` executes, skipping any step whose output exists, so
a rerun resumes:

1. **Preparation.** Reuse the Tx1 cache (encode only missing lines) and the ESM2 table; one pass over
   the basal sources for library sizes, log HVG bags and `q_sc`; compute `T`; build the log-space
   response cache; write the manifest.
2. **STATE sanity line.** Released checkpoint on each anchor's conditions, ratio to no-change printed
   and written; not a gate.
3. **Response-model comparison** (section 5), folds spread across visible GPUs.
4. **Joint training** with an automatic timestamped run id, all visible GPUs.
5. **Validation evaluation** of `best.pt`, the baseline ladder (gene mean, K562 copy prior, nearest
   line, context-PCA ridge on Tx1 and on log HVG), and the readout head with the explicit
   gene-specific context slope on the new backbone's cached features.
6. **`summary.md`** under the run directory: `T`, the sanity line, the comparison table and
   verdicts, the validation table (Huber, absolute and residual Pearson/Spearman, SD ratio) for the
   joint model, the readout head and every baseline.

`hpc/run.sh test CHECKPOINT` remains the only route to the test split, and `all` never calls it.

## 7. Code boundary

**Delete:** the fixed-backbone head, response-adaptation and interface-isolation harnesses
(`src/{data,model,training,eval,experiments}/p1{a,b,c}*.py`, `src/eval/p1b_comparison.py`,
`src/eval/p1c_tier0.py`, `hpc/p1c_pipeline.sh`, their `hpc/run.sh` entries and tests),
`src/experiments/profile_joint.py`, and the response-condition holdout code.

**Rewrite for simplicity:** `src/experiments/{config,prepare,geneeffect,baselines}.py`,
`src/data/{prepared,response,response_cache,response_streaming,basal,tx1_cache,q_sc,batches,datasets,gene_bags}.py`,
`src/model/{state,initialization,features,geneeffect,response}.py`,
`src/training/{trainer,sampling,checkpoint}.py`, `src/eval/{geneeffect,response}.py`,
`src/baselines/residual.py`, `hpc/run.sh`. The Tx1 cache's on-disk format (`embeddings.npy`,
`hvg.npy`, `obs.parquet` per line) is kept so no line is re-encoded.

**Add:** `src/experiments/all.py` (the run), `src/experiments/response_comparison.py` with
`src/model/response_mlp.py` (the comparison), `src/experiments/readout.py` (readout entry point moved
out of the deleted fixed-backbone head harness).

**Keep untouched:** the readout head (`src/{model,training,eval,data}/readout*.py`,
`src/eval/readout_comparison.py`), `src/eval/metrics.py`, `src/data/{splits,split_build,gene_splits,residual_target,geneeffect,embeddings,esm2_provenance,gene_order}.py`,
the one-off raw-input builders under `src/data/prepare/`, `src/experiments/historical/`, the GMM
ridge baseline, every file under `docs/results/`, and earlier specs.

**Guards kept** (one line each where possible): fitting on training lines only
(`assert_fit_eligible`); target residuals on the fold-fit mean, predictions on the fold-independent
mean; undefined, never zero, correlations for constant predictors; checkpoint loads raise when zero
keys load; config rejects unknown keys; `load_inputs` refuses a manifest without the log expression
space. **Removed:** pinned-value config checks (seeds, selection metric, numeric domains), gene-order
hash sidecars and SHA-256 re-verification on open, the end-of-preparation test-cache reopen,
duplicated reader/writer width and dtype assertions, manifest bookkeeping training does not read.

## 8. Tests

The suite is rewritten to behaviour tests: the expression transform against scanpy on a toy matrix
with library size over all genes; the Tx1 cache reader on the existing format; train-only fitting,
residual centering and undefined correlations; STATE's own basal path loading every released weight;
zero-initialised MLP arms returning exactly no-change; zero-key checkpoint loads raising; resume from
`last.pt` and skip-on-rerun; and one CPU smoke test running `all` end to end on a synthetic fixture to
`summary.md`. Retained-logic tests (splits, metrics, baselines, readout head) stay.

## 9. Documents

- Blueprint: high-level task description only; its model equations, learning rates, metric table
  and result tables move to the protocol.
- Protocol: new §3.4 expression space; §4–§5 for the new wiring, rates and single validation split;
  §9 replaced by the comparison and the `all` run; §8 stays as the diagnostics record.
- `figures/geneeffect_architecture.svg`: STATE's basal input becomes log HVG expression; Tx1 goes to
  the head.
- `AGENTS.md`, `CLAUDE.md`, `hpc/README.md`: new commands, deleted diagnostics, updated
  silent-failure list.
- Tahoe-x1 literature note: lines 37–38 report Fig. 5 separability values as perturbation-prediction
  numbers; replace with Fig. 6 values.

## 10. Execution

Branch `refactor/expression-space-all-pipeline` → behaviour suite and Ruff green locally → Codex
review of the wave, findings adjudicated → merge to `main`, push → pull on H20 port 30838 (4 GPUs) →
`hpc/run.sh all` in the background. First report once preparation has written `T` and the STATE
sanity line; the chain then runs to `summary.md`.

## 11. Claim boundaries

Validation only; the test split stays closed. The seed-0 joint numbers predate the expression-space
change and are not compared as like for like. The comparison has four contexts; a tie at no-change is
an expected, informative outcome. Tx1 Tahoe-100M exposure and STATE's Replogle/Nadig pretraining
exposure (K562, HepG2, Jurkat) qualify every response result. Nothing here is SL evidence.
