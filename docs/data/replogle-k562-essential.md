# Replogle K562 Essential CRISPRi Perturb-seq

## Role

Primary observed-transcriptome source for Stage 1 K562 alignment. This is the
v0 CRISPRi-compatible perturbation-response dataset paired with DepMap K562
gene-effect labels.

## Downloaded File

Remote path:
`/home/richard/projects/VCC/data/sl_dependency_v0/raw/replogle/K562_essential_normalized_singlecell_01.h5ad`

- Source: Hugging Face `arcinstitute/Replogle-Nadig-Preprint`
- File: `K562_essential_normalized_singlecell_01.h5ad`
- Size: 10662212844 bytes
- SHA256:
  `dfd2898643aff82f939ae22876e81f25629b9b7240e2f07564046dfa6c0953b2`
- Downloaded on 2026-05-11

Figshare returned HTTP 403 from the remote host, and Zenodo/scPerturb returned
HTTP 504, so the Hugging Face mirror was used.

## AnnData Shape and Fields

- Shape: 310385 cells x 8563 genes
- Perturbation label: `obs["gene"]`
- Cell-line label: `obs["cell_line"]`, value `k562`
- Expression gene identifier: `var.index` Ensembl IDs
- Gene symbol: `var["gene_name"]`
- Control label: `non-targeting`

## Current K562 Alignment

Remote report root:
`/home/richard/projects/VCC/data/sl_dependency_v0/interim/`

- Unique perturbation labels: 2058
- Labels matched to DepMap gene-effect columns: 1967
- Non-control labels without DepMap match: 90
- Control-like cells: 10691
- Per-perturbation cell count: median 121, 10th percentile 47, 90th percentile
  269

Generated alignment table:
`interim/k562_replogle_depmap_overlap.csv`.
