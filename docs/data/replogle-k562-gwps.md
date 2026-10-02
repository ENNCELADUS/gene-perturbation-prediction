# Replogle K562 GWPS CRISPRi Perturb-seq

## Role

Genome-scale K562 CRISPRi Perturb-seq, and the K562 response anchor of the joint
GeneEffect pipeline. The pipeline reads the raw-count release
`data/sl_dependency_v0/raw/replogle/K562_gwps_raw_singlecell_01.h5ad`
(`configs/experiments/13_geneeffect_226/perturbseq_sources.json`); preparation requires
integer UMI counts, so the normalized file described below is not a pipeline input.

## Downloaded File

Local path:
`data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad`

- Source: Replogle/Nadig processed K562 GWPS single-cell h5ad.
- Size: 65830941948 bytes.
- Downloaded on 2026-06-17.

## AnnData Shape and Fields

- Shape: 1989578 cells x 8248 genes.
- Perturbation label: `obs["gene"]`.
- Unique perturbation labels: 9867.
- Control label: `non-targeting`.
- Control cells: 75328.
- Other observed metadata fields include `gene_id`, `transcript`,
  `gene_transcript`, `sgID_AB`, `UMI_count`, `core_scale_factor`, and
  `core_adjusted_UMI_count`.
