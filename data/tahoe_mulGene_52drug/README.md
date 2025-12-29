# Tahoe Multi-Gene 52-Drug Dataset

Single-cell perturbation dataset with multi-gene targets and 52 drugs.

## Overview

| Metric | Value |
|--------|-------|
| Total Samples | 2,345,326 |
| Parquet Files | 24 |
| Unique Target Genes | 100 |
| Unique Drugs | 52 |
| Unique Cell Lines | 49 |
| Unique Plates | 2 |
| Unique Samples | 87 |

## Columns

- `target_gene` – Target gene identifier
- `drug` – Drug compound name
- `cell_line_id` – Cellosaurus cell line ID (CVCL_*)
- `plate` – Plate identifier
- `sample` – Sample identifier
- `label` – Class label (0–277, 100 unique values)
- `overlap_ratio` – Quality metric (min: 0.20, max: 0.70, mean: 0.29 ± 0.08)
- `ctrl_drug` – Control drug (all DMSO_TF)

## Target Gene Distribution

- **Range**: 15,381 – 60,000 samples per target
- **Mean**: 23,453 samples per target

| Top 10 Targets | Samples |
|----------------|---------|
| GNRHR | 60,000 |
| TOP2A | 60,000 |
| ADRA1A | 40,000 |
| JAK3 | 40,000 |
| JAK2 | 40,000 |
| JAK1 | 40,000 |
| ADRA1B | 40,000 |
| ADRA1D | 40,000 |
| EGFR | 40,000 |
| CYP2C9 | 40,000 |

## Drug Distribution

| Top 5 Drugs | Samples |
|-------------|---------|
| Olanzapine | 260,000 |
| Norepinephrine (hydrochloride) | 140,000 |
| Tucidinostat | 120,000 |
| Docetaxel | 120,000 |
| Gemfibrozil | 100,000 |

## Cell Line Distribution

49 cell lines with varying sample counts (4 – 286,630 samples each).

| Top 10 Cell Lines | Samples |
|-------------------|---------|
| CVCL_0480 | 286,630 |
| CVCL_1550 | 199,053 |
| CVCL_0371 | 197,815 |
| CVCL_1547 | 176,344 |
| CVCL_1717 | 166,649 |
| CVCL_1119 | 144,231 |
| CVCL_0293 | 141,137 |
| CVCL_1495 | 138,611 |
| CVCL_1635 | 131,590 |
| CVCL_1055 | 130,063 |

## File Structure

```
data/tahoe_mulGene_52drug/
├── single_target_*.parquet   # 24 parquet files
└── README.md
```
