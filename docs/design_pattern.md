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
  stages: ["evaluate"]
  seed: 47
  train_log_path:
  eval_log_path:
  save_checkpoint_path:
  load_checkpoint_path:
  save_best_only: true

device_config:
  device: "cuda"
  ddp_enabled: true
  use_mixed_precision: true

data_config:
  dataloader:
    train_dataset: "data/PRING/species_processed_data/human/BFS/human_train_ppi.txt"
    valid_dataset: "data/PRING/species_processed_data/human/BFS/human_val_ppi.txt"
    test_dataset: "data/PRING/species_processed_data/human/BFS/human_test_ppi.txt"
    num_workers: 4
    pin_memory: true
    drop_last: true

model_config:
  model: "v3"
  input_dim: 1536
  d_model: 512
  encoder_layers: 3
  cross_attn_layers: 3
  n_heads: 8
  mlp_head:
    hidden_dims: [512, 256, 128]
    dropout: 0.20
    activation: "gelu"
    norm: "layernorm"
  regularization:
    dropout: 0.10
    token_dropout: 0.10
    cross_attention_dropout: 0.10
    stochastic_depth: 0.10

training_config:
  epochs: 50
  batch_size: 32
  early_stopping_patience: 10
  monitor_metric: "auprc"
  logging:
    validation_metrics: ["auprc", "auroc", "f1", "accuracy"]
  optimizer:
    type: "adamw"
    lr: 0.0001
    beta1: 0.9
    beta2: 0.999
    eps: 1.0e-8
    weight_decay: 0.05
  scheduler:
    type: "onecycle"
    max_lr: 0.0001
    pct_start: 0.10
    div_factor: 25
    final_div_factor: 10000
    anneal_strategy: "cos"
  loss:
    type: "bce_with_logits"
    pos_weight: 1.0
    label_smoothing: 0.05
  strategy:
    type: "none"
  domain_adaptation:
    enabled: false
    method: "none"  # none | shot
    target_split: "test"
    epochs: 15
    beta: 0.3
    entropy_weight: 1.0
    diversity_weight: 1.0
    epsilon: 1.0e-5
    freeze_prefixes: ["output_head"]
    optimizer:
      type: "sgd"
      lr: 1.0e-4
      momentum: 0.9
      weight_decay: 1.0e-3
    scheduler:
      type: "shot_poly"  # shot_poly | none
      gamma: 10.0
      power: 0.75

evaluate:
  metrics:
```