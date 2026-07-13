# AIVC A->B->C STATE Pipeline for K562 Dependency Ranking

Run status: the repaired five-fold protocol was locked on 2026-07-13; no repaired
result artifacts are reported here. Older exp05 negative results are invalid as
model evidence because the primary and external expression coordinates were
misaligned.

Model card:
[`docs/experiment/model-card/05_aivc_a_to_b_to_c.md`](model-card/05_aivc_a_to_b_to_c.md)

## Scope

This experiment implements a STATE/ST-style A->B->C training pipeline for K562
dependency ranking. The matched supervised unit is the perturbation gene. The
label is the DepMap K562 CRISPR GeneEffect value for that gene.

```text
Replogle K562 control expression/HVG chunks + perturbation vector
    -> STATE/ST forward module
    -> predicted expression/HVG response
    -> ExpressionToLatentProjector
    -> predicted scVI latent bag B_hat
    -> GMM distribution features + MLP C head
    -> DepMap K562 GeneEffect
```

The implemented 05 surface currently covers the STATE/ST checkpoint path in
`src/aivc_model`. Tahoe-x1 and other AIVC families are future comparators, not
implemented model families in this pipeline.

## Configuration and Entry Points

Authoritative repaired production config:
`configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml`.

The older configs remain historical inputs and must not be used to represent the
repaired experiment.

Frozen STATE feature ablation config:
`configs/experiments/05_aivc_a_to_b_to_c/state_frozen_feature_ablation.yaml`.

Direct command:

```bash
uv run python -m aivc_model.cross_validate \
  --config configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml
```

Slurm entry point:

```bash
sbatch scripts/state.sh
```

Before training, build the strict ESM-2 asset and run the read-only preflight:

```bash
uv run python scripts/precompute_esm2_embeddings.py \
  --benchmark-csv data/sl_dependency_v0/interim/k562_gwps_depmap_overlap.csv \
  --symbol-column perturbation_gene \
  --out data/esm2/k562_gwps_depmap_esm2_650M.npz \
  --seq-cache data/esm2/symbol_to_sequence.json \
  --local-files-only \
  --require-complete-coverage
uv run python -m aivc_model.cross_validate \
  --config configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml \
  --preflight-only
```

## Repaired Five-Fold Claim Contract

The repaired experiment predicts population-level K562 DepMap/Achilles
GeneEffect. GeneEffect is a fitness/proliferation dependency score; it is not a
single-cell death probability and does not establish a death mechanism.

The universe is exactly the pre-frozen 9,338-gene GWPS-DepMap overlap. Its
canonical five-fold manifest and lowercase SHA-256 authority are reported with
every run. GeneEffect labels, GWPS responses, transitions, gene-derived prompts,
and fine-tuning samples inherit the same gene-level outer fold. No failed ESM or
expression match can create a smaller post-filter universe: ESM-2 must resolve
`9338/9338`, and STATE expression remains fixed at `2000/2000` checkpoint genes.

The primary result is the mean and standard deviation across the five
`internal_outer_test` folds. Adamson pilot, UPR epistasis, and UPR Perturb-seq
are secondary assay-transfer evaluations; Adamson never participates in epoch,
checkpoint, or layer selection.

For each fold, outer-test observed response has exactly two permitted routes,
both after the selected checkpoint is frozen: generation-quality evaluation and
the train-fitted observed-B oracle. It is never used for ESM adapter, STATE,
scVI, GMM, normalizer, or projector fitting; fine-tuning; early stopping; or
layer selection. The repaired config fixes the representation to the STATE
output layer rather than searching layers on outer-test data.

## Frozen STATE Feature Ablation

The next 05 ablation is designed as a two-stage validation sweep rather than
another end-to-end AIVC training run. STATE is frozen, predicted/control bags are
exported into several representation spaces, and fold-local K64 diagonal GMM
features plus Ridge C heads are fit with Replogle 5-fold gene CV.

The configured arms are:

| Arm | Role |
| --- | --- |
| Observed scVI128 GMM+Ridge | Diagnostic observed-B anchor only |
| STATE output -> fold-local scVI128 -> GMM+Ridge | Predicted output re-encoded through the experiment 03/04 scVI pattern |
| STATE output/HVG -> GMM+Ridge | Direct STATE output-space C head |
| STATE token hidden -> GMM+Ridge | Transformer hidden-space C head using same-path non-targeting control embeddings |

Primary model selection should read
`external_ensemble_target_heldout:adamson_k562` Spearman as an Adamson-guided
validation sweep. It is not an untouched final external-generalization claim.
The ablation writes fold metrics, per-gene predictions, fold membership, feature
QA, GMM convergence metadata, and `run_manifest.json` under
`results/experiments/05_aivc_a_to_b_to_c`.

## Training and DDP Pipeline

The training loop uses Accelerate DDP over perturbation-gene bags. The pipeline
uses padded per-rank gene loaders, `find_unused_parameters=True` for sparse
per-gene perturbation-vector parameters, tensor-based metric and prediction
gather, rank0-only scVI teacher cache materialization, and rank0-only artifact
and checkpoint writes.

When `projector.teacher: scvi`, rank0 fits a train-only scVI teacher on Replogle
train genes plus controls and writes a validated run-local latent cache. Other
ranks wait for that cache before entering the main AIVC training loop. The
linear projector maps STATE expression/HVG outputs into the scVI latent space
used by the GMM featureizer.

During A->B training, `make_cell_set_chunks` intentionally builds target
response chunks first, records their optional target batch labels, samples
batch-matched control chunks when control batch annotations are available, and
passes the corresponding batch labels into STATE. This is the supervised
training contract for reducing batch confounding in set-level A->B losses; it is
not a validation or final-inference dependency on the target response bag.

Epoch validation is prediction-only and aligned to the final evaluation path:
all available control cells are cached on device, combined with the perturbation
identity in one same-gene STATE call, converted to predicted `B_hat`, and scored
through the C head as `y_pred`. Validation does not build observed target
response chunks, does not use target batch labels or target cell counts, and
selects `models/best/` by `val_spearman` from those prediction-only metrics.

The ranknet/freeze-state config freezes the STATE adapter, trains the projector,
perturbation vectors for missing genes, and C head, adds pairwise RankNet loss on
predicted GeneEffect within each local gene batch, and anneals the A->B losses.

## Current Artifact Contract

Main-rank outputs are:

| Artifact | Path under run directory |
| --- | --- |
| Per-epoch train diagnostics and prediction-only validation metrics | `train_log.csv` |
| Final evaluation metrics | `test_metrics.csv` |
| Final predictions | `artifacts/test_predictions.csv` |
| Gene split record | `artifacts/gene_splits.csv` |
| External-test QA, when configured | `artifacts/external_test_qa.json` |
| Run-local ridge projector cache | `artifacts/ridge_projector_fit/` |
| Run-local fixed GMM cache | `artifacts/fixed_gmm_fit/` |
| Best checkpoint | `models/best/` |
| Final checkpoint | `models/final/` |

For external Adamson runs, `test_metrics.csv` and `test_predictions.csv` use
`evaluation_scope=external:adamson_k562`.

Final internal/external test evaluation uses the same prediction-only contract:
`y_pred` is computed from all available control cells plus perturbation identity,
using one same-gene STATE call per evaluated perturbation gene. The final path
does not use the evaluated gene's observed response bag, target cell count, or
target batch labels. Validation and final artifacts omit observed-B anchor
metrics and A->B reconstruction losses; those remain training diagnostics under
`train_*` columns.

## Known Limitations

- External Adamson evaluation currently evaluates all matched external genes and
  does not filter genes that overlap Replogle train or validation genes. Current
  Adamson metrics should therefore be read as external assay transfer, not as a
  guaranteed held-out perturbation-gene result.
- RankNet pairs are local to the current DDP rank and local gene batch. They are
  not synchronized across ranks.
- Current final metrics include predicted-B->C regression/ranking metrics. They
  do not yet export explicit occupancy RMSE/KL/JS diagnostics, top-k overlap, or
  fold-local summary artifacts.
- STATE checkpoint and mapping assets are loaded from local torch/pickle files.
  Treat those paths as trusted local artifacts only.
