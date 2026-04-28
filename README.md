# VCC - Reverse Perturbation Prediction

Reverse perturbation prediction for CRISPR Perturb-seq data using retrieval-based/classification methods.

## Quick Start

```bash
# Install Python once if needed
uv python install 3.11

# Create or update the project environment
uv sync

# Run the main pipeline
uv run --module src.main --config src/scgpt/configs/norman.yaml

# Run tests
uv run pytest
```

## Environment Management

This repository now uses `pyproject.toml + uv` as the only supported Python
environment workflow.

- Create or refresh the local virtualenv: `uv sync`
- Run CLI entry points: `uv run --module src.main --config src/scgpt/configs/norman.yaml`
- Run tests: `uv run pytest`
- Run lint/format: `uv run ruff check --fix .` and `uv run ruff format .`

Notes:
- `pyproject.toml` and `uv.lock` are the single sources of truth for Python dependencies.
- Plain `uv sync` installs the default development toolchain, including `ipython`, `pytest`, `ruff`, and `tabulate`.
- Some code paths expect either an installed `scgpt` package or a populated local
  `scGPT/` checkout.
- `flash-attn` is intentionally not locked in the shared environment because it
  requires a CUDA/NVCC build host. Install it manually on GPU machines after
  `uv sync` if your training path still needs it.

## Repository Layout

```
.
├── src/                # Core pipeline package
│   ├── pca_knn/        # PCA+kNN baseline pipeline
│   ├── random_forest/  # Random forest baseline pipeline
│   ├── scgpt/          # scGPT gene-score pipeline
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
- Task definition: `docs/task.md`
- Model overview: `docs/models.md`
- Metrics and evaluation: `docs/metrics.md`
