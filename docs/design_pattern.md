# Directory Structure Design

This repository uses a flattened, model-first layout. Shared behavior lives in
`src/utils`; each active model owns its own `prepare.py`, `train.py`, and
`evaluate.py` stage files.

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

## CLI Contract

`src.main` is intentionally thin. It accepts only a model config path:

```bash
uv run --module src.main --config src/scgpt/configs/norman.yaml
```

Distributed scGPT runs are launched through Hugging Face Accelerate:

```bash
uv run accelerate launch --module src.main \
  --config src/scgpt/configs/norman.yaml
```

Do not add CLI flags for model name, stage, checkpoint path, output path, or
max-step overrides. The config file is the only source of truth. `src.main`
validates the config, reads `model_config.model`, and executes
`run_config.stages` in order by importing:

```text
src.{model_config.model}.{stage}.run(config)
```

Allowed models are:

- `pca_knn`
- `random_forest`
- `scgpt`

Allowed stages are:

- `prepare`
- `train`
- `evaluate`

Each stage file must expose:

```python
def run(config: dict) -> dict:
    ...
```

## Shared Config Schema

Every model config must contain these top-level sections:

```yaml
run_config:
  stages: ["prepare", "train", "evaluate"]
  seed: 42
  study_name: norman
  train_log_path: logs/pca_knn/train/norman.log
  eval_log_path: logs/pca_knn/evaluate/norman.log
  save_checkpoint_path: model/pca_knn/norman/model.joblib
  load_checkpoint_path:
  save_best_only: true

device_config:
  device: "cpu"          # "cpu" | "cuda"
  ddp_enabled: false     # DDP is activated by accelerate launch
  use_mixed_precision: false

data_config:
  h5ad_path: data/norman/perturb_processed.h5ad
  condition_key: condition
  control_key: control
  condition_split:
    train: []
    validation: []
    test: []

model_config:
  model: pca_knn         # pca_knn | random_forest | scgpt

training_config: {}

evaluation_config:
  top_k_values: [1, 5, 10, 20, 40]
```

`run_config.stages` controls all orchestration. For example, an evaluation-only
run is configured as:

```yaml
run_config:
  stages: ["evaluate"]
  study_name: norman
  load_checkpoint_path: model/scgpt_gene_score/norman/best_model.pt
```

Logs and model artifacts have separate roots:

- study identity: `run_config.study_name`
- runtime logs: `logs/{model_name}/train/{study_name}.log` and
  `logs/{model_name}/evaluate/{study_name}.log`
- trained model artifacts: `model/{model_name}/{study_name}/...`
- pretrained scGPT backbone: `model/scGPT/`

Do not save finetuned scGPT checkpoints under `model/scGPT/`. That directory is
reserved for the upstream backbone files loaded from `pretrained_dir`. Use a
distinct trained-model directory such as
`model/scgpt_gene_score/norman/best_model.pt`; this avoids conflicts with the
pretrained `model/scGPT/best_model.pt`, including on case-insensitive
filesystems.

`flash-attn` is optional and is not part of the locked dependency set. Keep
`use_fast_transformer: false` unless the runtime environment has the required
flash attention stack installed and validated.

## Accelerate Boundaries

The small `src/utils/runtime.py::AccelerateRuntime` wrapper owns device
placement, mixed precision, DDP preparation, gradient clipping, checkpoint save,
and metric gathering.

scGPT training and evaluation should use Accelerate for:

- `accelerator.prepare(...)`
- `accelerator.backward(...)`
- `accelerator.clip_grad_norm_(...)`
- `accelerator.gather_for_metrics(...)`
- main-process checkpoint/log writes
- process synchronization

Accelerate is not the business orchestrator. The stage order still comes only
from `run_config.stages`, and `src.main` remains the only entry point.
