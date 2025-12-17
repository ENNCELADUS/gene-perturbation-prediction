# VCC - Reverse Perturbation Prediction

Reverse perturbation prediction for CRISPR Perturb-seq data using retrieval-based methods.

## Quick Start

```bash
# Setup environment
conda create -n vcc python=3.11 -y
conda activate vcc
pip install -r requirements.txt

# Run PCA baseline
python -m src.main --config src/configs/pca.yaml
```

## Project Structure

```
VCC/
├── src/                        # Main source code
│   ├── main.py                 # Entry point
│   ├── configs/                # Model configs (pca.yaml, scgpt.yaml)
│   ├── data/                   # Data loading (GEARS wrapper)
│   ├── model/                  # Encoders + retrieval
│   ├── evaluate/               # Metrics (Top-K, MRR, NDCG)
│   ├── train/                  # Training (Stage 2)
│   └── utils/                  # Utilities
│
├── scripts/                    # HPC SLURM scripts
│   ├── baseline.sh             # PCA baseline
│   └── scgpt.sh                # scGPT retrieval
│
├── data/                       # Data directory
│   └── norman/                 # Norman Perturb-seq dataset
│
└── logs/                       # Experiment outputs (gitignored)
```

## Usage

### Local Execution

```bash
# PCA baseline
python -m src.main --config src/configs/pca.yaml

# scGPT retrieval (Stage 2)
python -m src.main --config src/configs/scgpt.yaml

# Evaluate on specific split
python -m src.main --config src/configs/pca.yaml --split test
```

### HPC (SLURM)

```bash
sbatch scripts/baseline.sh   # PCA baseline
sbatch scripts/scgpt.sh      # scGPT with GPU
```

## Data

Uses Norman et al. 2019 CRISPRa Perturb-seq dataset via GEARS:

| Property | Value |
|----------|-------|
| Cells | 91,205 |
| Genes | 5,045 (HVG) |
| Conditions | 284 |
| Single-gene perturbations | 152 |
| Double-gene perturbations | 131 |

See `data/norman/README.md` for details.

## Metrics

- **Top-K Accuracy**: Fraction where true perturbation is in top-K predictions
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

## References

1. Norman et al. (2019). Exploring genetic interaction manifolds. *Science*.
2. Roohani et al. (2023). GEARS: Predicting transcriptional outcomes. *Nature Biotechnology*.
3. Cui et al. (2023). scGPT: Building foundation models for single-cell. *Nature Methods*.
