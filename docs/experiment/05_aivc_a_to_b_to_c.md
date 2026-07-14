# AIVC A->B->C STATE Pipeline for K562 Dependency Ranking

Run status: the repaired five-fold protocol was locked on 2026-07-13 and the
end-to-end response-GMM implementation was documented on 2026-07-14. No
repaired result artifacts are reported here. Older exp05 negative results are
invalid as model evidence because the primary and external expression
coordinates were misaligned.

Model card:
[`docs/experiment/model-card/05_aivc_a_to_b_to_c.md`](model-card/05_aivc_a_to_b_to_c.md)

## Scope

This experiment predicts the population-level K562 DepMap/Achilles GeneEffect
for a perturbation gene. GeneEffect is a relative growth-rate/dependency label
under a population-dynamics model. It is not a cell-death label, a per-cell fate
label, or evidence of a death mechanism.

For each inner-train perturbation gene, an ESM-2 adapter produces the STATE
perturbation token. The trainable STATE checkpoint predicts a 2,000-feature
post-perturbation response bag from non-targeting control cells. Predicted and
observed response cells pass through the same Linear(2000, 128) response encoder,
trainable diagonal-GMM pooler, and GeneEffect head. Observed B supplies auxiliary
response and GeneEffect supervision for inner-train genes only. Validation and
primary outer-test GeneEffect predictions use control cells plus perturbation
identity only.

The response encoder applies `LayerNorm(128)` after the shared linear map. The
64-component diagonal GMM returns occupancy, control-relative occupancy, latent
mean, latent variance, and occupancy entropy to the `[64, 32]` MLP GeneEffect
head.

```text
ESM-2(gene) -> trainable adapter -> STATE perturbation token
non-targeting controls + token -> trainable STATE -> predicted response (2000)
predicted response -> shared Linear(2000, 128) + LayerNorm -> trainable GMM -> C head
observed response  -> shared Linear(2000, 128) + LayerNorm -> trainable GMM -> C head
```

The authoritative exp05 path does not fit or cache scVI teacher latents, a
ridge projector, or a fixed sklearn GMM. Consequently, scVI artifacts, ridge
projector artifacts, and fixed-GMM fit caches are absent from the authoritative
run. Retained legacy helpers and the separate frozen-feature ablation are not
part of this run.

## Configuration and Entry Points

The authoritative config is
`configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml`.

Run the read-only asset preflight with:

```bash
uv run python -m aivc_model.cross_validate \
  --config configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml \
  --preflight-only
```

The training launch is:

```bash
sbatch scripts/state.sh
```

`scripts/state.sh` delegates to `scripts/run_exp05_ddp.sh`, which launches four
Accelerate processes with bf16 mixed precision. A single-process invocation is
rejected for authoritative training.

## Five-Fold Data and Access Contract

The experiment uses the pre-frozen 9,338-gene GWPS-DepMap overlap and its
canonical five-fold manifest. GeneEffect labels, GWPS responses, transition
supervision, gene-derived prompts, and fine-tuning samples inherit the same
gene-level outer fold. ESM-2 must resolve all `9338/9338` genes, and the STATE
expression space is fixed to the checkpoint's 2,000 features.

Within each outer fold, only inner-train genes may update the ESM-2 adapter,
STATE, response encoder, GMM, or GeneEffect head. Inner-validation uses the
prediction-only path for checkpoint selection. The primary outer-test path is
also prediction-only and exposes labels for scoring only after the selected
checkpoint is frozen.

Outer-test observed responses have exactly two post-freeze diagnostic uses:
generation-quality evaluation and the shared observed-B oracle. Neither route
may update parameters, normalizers, thresholds, epochs, checkpoints, or
representations. Adamson is a secondary assay-transfer evaluation and does not
participate in fitting or selection.

## Loss Graph

For each inner-train gene, the configured objective is:

```text
0.01 * HVG mean-delta MSE
+ 0.10 * shared-latent mean MSE
+ 0.10 * predicted-vs-observed GMM occupancy MSE
+ 0.01 * observed-latent GMM negative log likelihood
+ 2.00 * predicted-B GeneEffect MSE
+ 0.25 * observed-B GeneEffect MSE
+ 5.00 * predicted GeneEffect RankNet loss
```

The response-alignment and GMM terms anneal over the first five epochs to 10%
of their initial weights. HVG and latent energy-distance weights are zero in the
authoritative config, so those distances are not computed. Observed targets are
detached for alignment losses, while observed-B GeneEffect supervision retains
gradients through the shared response encoder, trainable GMM, and C head.

RankNet uses differentiable all-gather for per-rank predictions and ordinary
all-gather for detached labels and validity masks. It therefore forms pairs over
the four-gene global batch on every optimizer step; padded loader entries are
masked before pair construction.

## Four-Rank DDP Semantics

Each outer fold is one four-GPU DDP training job. Rank 0 through rank 3 process
disjoint gene batches from the same fold and synchronize gradients every optimizer
step. The five outer folds run sequentially; GPUs are not assigned independent
fold-local models. Per-device gene batch size is one, so the global gene batch size
is four.

The model, optimizer, training loader, and validation loader enter one
`accelerator.prepare(...)` call. The loader is padded so every rank performs a
positive number of optimizer steps, and the four step counts are checked at the
end of every epoch. Rank-zero directory creation, logs, checkpoints, fold
outputs, and final aggregation propagate failures symmetrically to all ranks.

## Artifact Contract

Each fold writes `train_log.csv`, `fit_audit_summary.json`,
`runtime_evidence.json`, `models/best/`, `models/final/`, and fold-local copies of
the tabular audit artifacts. The aggregate run writes:

| Artifact | Purpose |
| --- | --- |
| `summary.csv` | Mean and standard deviation across outer folds |
| `artifacts/fold_metrics.csv` | Internal, generation-quality, observed-B-oracle, and external metrics |
| `artifacts/predictions.csv` | Per-gene predictions with evaluation scope |
| `artifacts/gene_splits.csv` | Byte-identical canonical outer-fold manifest |
| `artifacts/fold_roles.csv` | Inner-train, inner-validation, and outer-test roles |
| `artifacts/fit_access_audit.csv` | Recorded gene access by fit and selection stage |
| `artifacts/external_alignment_qa.csv` | External-source alignment checks |
| `run_manifest.json` | Split, source, ESM-2, STATE, fold-seed, and artifact provenance |

## Interpretation Limits

- The primary result is the mean and standard deviation over the five
  `internal_outer_test` folds.
- Adamson is assay transfer, not guaranteed held-out-gene generalization.
- Generation-quality and observed-B-oracle results are post-freeze diagnostics,
  not deployable primary predictions.
- The endpoint remains a population-level relative growth-rate/dependency
  label. No output supports a cell-death, mechanism, or per-cell fate claim.
