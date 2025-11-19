# Data Preprocessing Roadmap

## Overview
This document outlines the data preprocessing steps taken to prepare the Virtual Cell Challenge (VCC) dataset for model training and evaluation.

## 1. Data Splitting Strategy

To rigorously evaluate the model's ability to generalize to **unseen perturbations** (Out-of-Distribution / OOD generalization), we performed a stratified split based on `target_gene` rather than a random cell-wise split.

### 1.1. Methodology
*   **Source File:** `data/raw/adata_Training.h5ad` (approx. 221k cells)
*   **Splitting Criteria:** 
    *   **Control Group:** All cells with `target_gene == 'non-targeting'` are assigned to the **Training Set**. This ensures the model learns the baseline cellular state.
    *   **Perturbation Genes:** The remaining 150 unique perturbation genes were shuffled (random seed: `42`).
        *   **10% (15 genes)** were randomly selected and held out entirely for the **Test Set**.
        *   **90% (135 genes)** were assigned to the **Training Set**.

### 1.2. Processed Artifacts

| Dataset | File Path | Dimensions | Content |
| :--- | :--- | :--- | :--- |
| **Train** | `data/processed/train.h5ad` | 199,022 cells × 18,080 genes | 135 perturbation genes + All 'non-targeting' controls |
| **Test** | `data/processed/test.h5ad` | 22,251 cells × 18,080 genes | 15 held-out perturbation genes (OOD) |

### 1.3. Reproducibility
The splitting logic is implemented in `scripts/split_data.py`. To reproduce the split:

```bash
python scripts/split_data.py
```

## 2. Verification
Post-processing verification confirmed:
*   **No Data Leakage:** The intersection of perturbation genes between Train and Test sets is empty.
*   **Control Representation:** The training set includes the necessary control population for calculating differential expression.
*   **Class Balance:** The test set preserves all cells for the selected 15 genes, maintaining the original experimental distribution.

