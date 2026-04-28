# Directory Structure Design

This repo is organized in a flattened “model-first” manner.

```text
src/
  __init__.py
  main.py

  utils/
    __init__.py
    config.py
    data.py
    metrics.py
    runtime.py

  pca_knn/
    __init__.py
    prepare.py
    train.py
    evaluate.py
    config.yaml

  random_forest/
    __init__.py
    prepare.py
    train.py
    evaluate.py
    config.yaml

  scgpt/
    __init__.py
    configs/
      norman.yaml
    prepare.py
    data.py
    model.py
    train.py
    evaluate.py
```

## Huggingface Accelerate Usage

This repo uses `huggingface/accelerate` for distributed training.

### Accelerate Boundaries

Hugging Face Accelerate is suitable for managing:

- Device placement
- Distributed training
- Gradient accumulation
- Mixed precision
- Checkpoint save/load
- Logging integration

However, it is not recommended for business orchestration. Specifically:

`scgpt/train.py` should:
1. Load config
2. Prepare dataset/model/optimizer
3. `accelerator.prepare(...)`
4. Train loop
5. Save checkpoint

Avoid creating handmade trainer frameworks. Keep it simple, explicit, and readable.

## Configuration File Convention

```yaml
run_config:
  stages: ["prepare", "train", "evaluate"]
  seed: 47
  train_log_path: results/scgpt/train.log
  eval_log_path: results/scgpt/eval_results.json
  save_checkpoint_path: results/scgpt/best_model.pt
  load_checkpoint_path: results/scgpt/best_model.pt
  save_best_only: true

device_config:
  device: "cuda"
  ddp_enabled: true
  use_mixed_precision: true

data_config:
  h5ad_path: data/norman/perturb_processed.h5ad
  condition_key: condition
  control_key: control
  control_n_samples: 16
  num_workers: 0
  condition_split_path: results/scgpt/norman_gene_heldout_split.yaml
  condition_split:
    train: []
    validation: []
    test: []
  split_config:
    strategy: gene_heldout
    train_gene_fraction: 0.7
    validation_gene_fraction: 0.1
    test_gene_fraction: 0.2
    min_cells_per_condition: 1

model_config:
  model: scgpt
  pretrained_dir: model/scGPT
  freeze_encoder: true
  freeze_layers_up_to: 10
  preprocess_binning: 51
  score_mode: dot
  head_hidden_dim: 512
  head_dropout: 0.2

training_config:
  epochs: 50
  batch_size: 32
  learning_rate: 5.0e-5
  weight_decay: 0.01
  max_grad_norm: 1.0

evaluation_config:
  top_k_values: [1, 5, 10, 20, 40]
```
