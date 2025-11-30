# scGPT Finetuning for Perturbation Prediction

## Overview

This document describes the finetuning pipeline for scGPT on the VCC perturbation dataset.

## Prerequisites

- **Dependencies:** `gears`, `torch_geometric`, `scgpt`

Install torch_geometric if missing:
```bash
pip install torch_geometric
```

## Data Pipeline

No new layers or heads are added and no structural layers are removed; the finetune step just re‑initializes certain non‑encoder parameters and reloads encoder‑type weights from the base scGPT checkpoint.

### 1. VCC → GEARS Format Conversion

GEARS requires specific AnnData format:

| VCC Format | GEARS Format |
|------------|--------------|
| `obs['target_gene']` = "GENE" | `obs['condition']` = "GENE+ctrl" |
| `obs['target_gene']` = "non-targeting" | `obs['condition']` = "ctrl" |
| `var.index` = gene names | `var['gene_name']` = gene names |
| N/A | `obs['cell_type']` = "K562" (constant) |

**Script:** `scripts/convert_to_gears.py`

### 2. Processed Output

The conversion script creates GEARS-compatible files in `data/processed/gears_vcc/`:
- `perturb_processed.h5ad`: Processed AnnData with DE genes computed
- `data_pyg/cell_graphs.pkl`: PyTorch Geometric cell graphs
- `splits/`: Train/val/test split pickles

## Training Configuration

### Model Architecture
- **Base model:** `model/scGPT/best_model.pt` (pretrained on 33M human cells)
- **Architecture:** TransformerGenerator (12 layers, 512 dim, 8 heads)
- **Vocab size:** ~60K gene tokens

### Hyperparameters (from tutorial)

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Batch size | 64 |
| Epochs | 15 |
| Early stop patience | 10 (based on training loss) |
| Scheduler | StepLR (γ=0.9) |
| Max sequence length | 1536 |
| AMP | True |

### Training Objectives
- **MLM (Masked Language Modeling):** Primary objective
- MVC, ECS, CLS, CCE: Disabled

### Layers Loaded from Pretrained
- `encoder.*`
- `value_encoder.*`
- `transformer_encoder.*`

## Commands

### Step 1: Convert Data
```bash
python scripts/convert_to_gears.py
```

### Step 2: Train
```bash
# Local
python src/train.py --config src/configs/finetune.yaml

# SLURM
sbatch scripts/finetune.sh
```

### Step 3: Evaluate
```bash
python src/main.py --model_type scgpt_finetuned --threads 8
```

## Output

Finetuned model saved to `model/scGPT_finetuned/`:
- `best_model.pt`: Model weights
- `args.json`: Model configuration
- `vocab.json`: Gene vocabulary
- `run.log`: Training log

## Expected Results

Based on the scGPT tutorial (Adamson dataset):
- Pearson correlation on DE genes: ~0.95+
- Training time: ~5 hours on A40 (15 epochs)

## Notes

1. **Gene matching:** VCC has 18,080 genes; expect ~80-90% match to pretrained vocab
2. **Memory:** A40 handles batch_size=64; TITAN RTX may need batch_size=32
3. **No validation split:** Training uses all 135 perturbation genes; early stopping on training loss

