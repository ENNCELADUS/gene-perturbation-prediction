# VCC - Reverse Perturbation Prediction

Reverse perturbation prediction for CRISPR Perturb-seq data using retrieval-based/classification methods.

## Quick Start

```bash
conda create -n vcc python=3.11 -y
conda activate vcc
pip install -r requirements.txt

python -m src.main --config src/configs/pca.yaml
```

## Repository Layout

```
.
├── src/                # Core pipeline package
│   ├── configs/        # Experiment configuration files
│   ├── data/           # Dataset loading and split logic
│   ├── evaluate/       # Evaluation metrics and helpers
│   ├── model/          # Encoders and retrieval models
│   ├── train/          # Training and fine-tuning flows
│   └── utils/          # Shared utilities
├── scripts/            # Automation helpers and SLURM runners
├── scGPT/              # Vendorized scGPT modules and tests
├── data/
│   ├── raw/            # Raw inputs (gitignored)
│   └── processed/      # Derived features
├── docs/               # Project documentation and references
├── tests/              # Project tests
└── cell-eval/          # Standalone evaluation package
```

## Documentation

- Data splits and AnnData requirements: `docs/data.md`
- Metrics and evaluation: `docs/eval_metrics.md`
- Project overview: `docs/project-intro/introduction.md`
- scGPT reference notes: `docs/references/scGPT.md`
