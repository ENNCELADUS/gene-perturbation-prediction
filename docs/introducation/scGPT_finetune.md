# Fine-tuning scGPT for Perturbation Response Prediction

## Loaded checkpoint
- Frozen (when `training.freeze_encoder: true`): parameters with prefixes
  `encoder`, `value_encoder`, `transformer_encoder`.
- Trainable (remaining components):
  - `pert_encoder` embedding for perturbation flags.
  - `decoder` (`AffineExprDecoder` -> `ExprDecoder` coeff/bias heads for
    expression prediction; zero-prob heads if `explicit_zero_prob`).
  - `cls_decoder` head (present even if not used in this task).

## Finetune training step
- For each batch, predict perturbation expression and compute composite loss:
  - `sw1`: Sliced Wasserstein-1 (distribution alignment)
  - `proto`: ProtoInfoNCE on pseudobulk deltas
  - `de_rank`: DE rank loss (only when a DE gene map is available)
  - `dir`: DE direction loss (only when a DE gene map is available)
- Total loss (defaults in `config["loss"]`):
  `0.60*sw1 + 0.25*proto + 0.10*de_rank + 0.05*dir`.
- Optimization uses AMP + gradient scaling; early stopping tracks validation
  `overall_score` with patience `config["optimizer"]["early_stop"]`. The best
  checkpoint is saved as `best_model.pt` plus `args.json` and `vocab.json`.

## Evaluation metrics
Computed per perturbation and aggregated via the "score of averages":
- `PDS` (normalized perturbation discrimination score; lower is better)
- `MAE@k` (default `k=2000`, from `config["metrics"]["mae_top_k"]`; lower is better)
- `DES` (differential expression score; higher is better)
- `overall_score` = mean of scaled PDS/MAE/DES * 100 (higher is better); used for
  early stopping.
