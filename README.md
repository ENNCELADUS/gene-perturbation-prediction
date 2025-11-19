# VCC - CRISPR Perturbation Gene Expression Dataset

A comprehensive analysis and preprocessing pipeline for the VCC (CRISPR Perturbation) gene expression dataset.

## Project Structure

```
VCC/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── environment.yml                        # Conda environment configuration
│
├── data/                                  # Data directory
│   ├── raw/                               # Raw data files (downloaded)
│   │   ├── adata_Training.h5ad            # Training set (221K cells × 18K genes)
│   │   ├── gene_names.csv                 # Gene name mappings
│   │   └── pert_counts_Validation.csv     # Validation perturbation counts
│   └── processed/                         # Processed data (generated)
│
├── scripts/                               # Executable Python scripts
│   ├── preview_data.py                    # Preview all data files
│   └── visualize_h5ad.py                  # Generate visualizations & analysis
│
├── analysis/                              # Analysis outputs
│   └── visualizations/                    # Generated plots
│       ├── h5ad_visualization.png         # Main analysis plots
│       └── h5ad_expression_details.png    # Expression matrix details
│
├── docs/                                  # Documentation
│   ├── VCC_DATA_DESCRIPTION.md            # Detailed data description
│   └── h5ad_visualize.md                  # H5AD visualization guide
│
└── notebooks/                             # Jupyter notebooks (optional)
    └── (place analysis notebooks here)
```

## Quick Start

### Setup Environment

```bash
# Create conda environment
conda create -n vcc python=3.11 -y
conda activate vcc

# Install dependencies
pip install -r requirements.txt

# Format and lint (PEP 8 via Ruff)
ruff format .
ruff check . --fix
```

## Dataset Overview

### Training Data: `adata_Training.h5ad`

- **Cells (obs):** 221,273
- **Genes (vars):** 18,080
- **Size:** ~7.2 GB (sparse format)
- **Sparsity:** 51.69% zeros
- **Format:** AnnData H5AD

**Observation Metadata:**
- `target_gene` - CRISPR target gene (151 unique)
- `guide_id` - Guide RNA identifier (189 unique)
- `batch` - Experimental batch (48 unique)

**Gene Metadata:**
- `gene_id` - Ensembl gene identifier

**Expression Matrix:**
- Type: Sparse CSR matrix
- Data type: float32
- Non-zero elements: 1.93B

### Gene Names: `gene_names.csv`

- 18,079 gene names in order
- Corresponds to genes in expression matrix

### Validation Perturbations: `pert_counts_Validation.csv`

- 50 target genes for validation
- Cell counts: 161-2,925 cells per gene
- Median UMI per cell: ~54K

## Key Statistics

| Metric | Value |
|--------|-------|
| Total cells | 221,273 |
| Total genes | 18,080 |
| Control cells (non-targeting) | 38,176 |
| Perturbed cells | 183,097 |
| Unique target genes | 151 |
| Unique batches | 48 |
| Sparsity | 51.69% |
| Avg non-zero expression | 6.50 |

## Documentation

- **[h5ad_visualize.md](docs/h5ad_visualize.md)** - Guide to H5AD structure visualization
