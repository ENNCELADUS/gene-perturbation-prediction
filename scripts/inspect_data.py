#!/usr/bin/env python
"""Inspect h5ad data to understand zero expression issue."""

import scanpy as sc
import numpy as np

# Load the data
adata = sc.read_h5ad('data/norman/perturb_processed.h5ad')

print('=== Data Overview ===')
print(f'Shape: {adata.shape}')
print(f'X type: {type(adata.X)}')
print(f'X dtype: {adata.X.dtype}')

# Check for all-zero rows
if hasattr(adata.X, 'toarray'):
    X = adata.X.toarray()
else:
    X = adata.X

row_sums = X.sum(axis=1)
zero_rows = np.sum(row_sums == 0)
print(f'\n=== Zero Expression Analysis ===')
print(f'Cells with all-zero expression: {zero_rows} / {adata.n_obs} ({100*zero_rows/adata.n_obs:.2f}%)')

# Check min/max values
print(f'\n=== Value Range ===')
print(f'Min value: {X.min()}')
print(f'Max value: {X.max()}')
print(f'Mean value: {X.mean():.4f}')

# Check sparsity
nonzero_per_row = np.count_nonzero(X, axis=1)
print(f'\n=== Sparsity ===')
print(f'Min non-zero genes per cell: {nonzero_per_row.min()}')
print(f'Max non-zero genes per cell: {nonzero_per_row.max()}')
print(f'Mean non-zero genes per cell: {nonzero_per_row.mean():.1f}')
print(f'Cells with <10 non-zero genes: {np.sum(nonzero_per_row < 10)}')

# Check the obs columns
print(f'\n=== Obs Columns ===')
print(adata.obs.columns.tolist())

# Check var columns
print(f'\n=== Var Columns ===')
print(adata.var.columns.tolist())

# Now check the REAL issue: binning
print('\n=== Binning Analysis ===')
# Simulate binning as done in collator
n_bins = 51
sample_idx = 0
expr = X[sample_idx]
expr_clip = np.clip(expr, 0, None)
max_val = expr_clip.max() if expr_clip.max() > 0 else 1.0
binned = np.floor(expr_clip / max_val * (n_bins - 1)).astype(int)
binned = np.clip(binned, 0, n_bins - 1)
print(f'Sample cell binned values - unique: {np.unique(binned)}')
print(f'Non-zero after binning: {np.count_nonzero(binned)}')

# Check how many cells become all-zero after binning
all_zero_after_binning = 0
very_sparse_after_binning = 0
for i in range(min(1000, len(X))):  # Check first 1000 cells
    expr = X[i]
    expr_clip = np.clip(expr, 0, None)
    max_val = expr_clip.max() if expr_clip.max() > 0 else 1.0
    binned = np.floor(expr_clip / max_val * (n_bins - 1)).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)
    nz = np.count_nonzero(binned)
    if nz == 0:
        all_zero_after_binning += 1
    if nz < 10:
        very_sparse_after_binning += 1

print(f'\nIn first 1000 cells:')
print(f'  All-zero after binning: {all_zero_after_binning}')
print(f'  <10 non-zero after binning: {very_sparse_after_binning}')

# Check expression distribution
print(f'\n=== Expression Distribution ===')
flat_X = X.flatten()
print(f'Percentiles: 0%={np.percentile(flat_X, 0):.4f}, 25%={np.percentile(flat_X, 25):.4f}, 50%={np.percentile(flat_X, 50):.4f}, 75%={np.percentile(flat_X, 75):.4f}, 99%={np.percentile(flat_X, 99):.4f}, 100%={np.percentile(flat_X, 100):.4f}')
