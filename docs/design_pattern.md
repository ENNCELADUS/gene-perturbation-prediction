# Directory Structure Design

This repo is organized in a flattened “model-first” manner.

```
src/
  __init__.py
  cli.py

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

Accelerate 的边界

Hugging Face Accelerate 适合接管：

device placement
distributed training
gradient accumulation
mixed precision
checkpoint save/load
logging integration
但不建议让它承担业务 orchestration。也就是说：

scgpt/train.py
  load config
  prepare dataset/model/optimizer
  accelerator.prepare(...)
  train loop
  save checkpoint
不要再写一套 handmade trainer framework。简单、显式、可读就够。