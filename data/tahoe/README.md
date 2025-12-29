# Tahoe H5AD Dataset

This README documents the merged Tahoe dataset at `data/tahoe/tahoe.h5ad`.

## Overview

| Property | Value |
|----------|-------|
| **Path** | `data/tahoe/tahoe.h5ad` |
| **File size** | ~34 GB |
| **Shape** | 2,345,326 cells × 31,597 genes |
| **X storage** | CSR sparse matrix |
| **Layers** | `ctrl` (control expression) |
| **obsm/varm** | none |
| **uns** | empty |
| **raw** | absent |

> [!NOTE]
> `obs_names` are not unique in this file.

---

## Normalization Status

**Data is RAW COUNTS (not normalized)**

X matrix value statistics (sampled 100k non-zero values):

| Metric | Value |
|--------|-------|
| Min | 1.0 |
| Max | 755.0 |
| Mean | 1.60 |
| Median | 1.0 |
| Std | 5.29 |
| Values are integer | **Yes** |

Value percentiles:
- 25th: 1.0
- 50th: 1.0
- 75th: 1.0
- 95th: 3.0
- 99th: 7.0

> [!IMPORTANT]
> The data is **raw UMI counts** (integer values with max 755). For most analyses, you should apply:
> 1. Library-size normalization (`sc.pp.normalize_total(adata)`)
> 2. Log transformation (`sc.pp.log1p(adata)`)

---

## Matrix Density

| Matrix | Non-zeros | Density | Sparsity |
|--------|-----------|---------|----------|
| X | 3,335,906,034 | ~4.50% | ~95.50% |
| ctrl layer | 980,504,827 | ~1.32% | ~98.68% |

---

## obs Columns

| Column | Description |
|--------|-------------|
| `condition` | Combined perturbation: `target_gene+drug` |
| `condition_name` | Cell line specific: `cell_line_id_condition` |
| `cell_line_id` | Cell line identifier (CVCL format) |
| `target_gene` | Target gene being perturbed |
| `drug` | Drug applied |
| `plate` | Experimental plate |
| `sample` | Sample identifier |
| `label` | Numeric label (0-277) |
| `overlap_ratio` | Overlap quality metric (0.2-0.7) |
| `control` | Control flag (all 0 in perturbation data) |
| `ctrl_drug` | Control drug used (all DMSO_TF) |

---

## Key Cardinalities

| Column | Unique Values |
|--------|---------------|
| **drug** | 52 |
| **target_gene** | 100 |
| **cell_line_id** | 49 |
| **condition** | 118 |
| **condition_name** | 5,115 |
| **label** | 100 |
| **plate** | 2 |

---

## Drug Distribution (for Condition Splitting)

Total: **52 unique drugs** across **2,345,326 cells**

### Top 10 Drugs by Cell Count
| Drug | Cells |
|------|-------|
| Dexmedetomidine (hydrochloride) | 260,000 |
| Isradipine | 240,000 |
| Dobutamine (hydrochloride) | 220,000 |
| Phenylephrine (hydrochloride) | 200,000 |
| Clonidine | 180,000 |
| Fenoldopam (mesylate) | 160,000 |
| Tucidinostat | 120,000 |
| Docetaxel | 120,000 |
| Gemfibrozil | 100,000 |
| Adenosine | 80,000 |

### Drug Cell Count Statistics
| Metric | Value |
|--------|-------|
| Min cells/drug | 19,183 |
| Max cells/drug | 260,000 |
| Mean cells/drug | 45,102 |
| Median cells/drug | 20,000 |
| Std | 42,638 |

---

## Target Gene Distribution

Total: **100 unique target genes**

### Top 10 Target Genes
| Gene | Cells |
|------|-------|
| GNRHR | 60,000 |
| TOP2A | 60,000 |
| ADRA1D | 40,000 |
| EGFR | 40,000 |
| DRD2 | 40,000 |
| MTOR | 40,000 |
| ADRA2C | 40,000 |
| ADRA2A | 40,000 |
| JAK1 | 40,000 |
| ADRA1B | 40,000 |

---

## Cell Line Distribution

Total: **49 unique cell lines**

### Top 10 Cell Lines
| Cell Line | Cells |
|-----------|-------|
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

---

## Plate Distribution

| Plate | Cells |
|-------|-------|
| plate4 | 1,942,983 |
| plate5 | 402,343 |

---

## Overlap Ratio Statistics

| Metric | Value |
|--------|-------|
| Min | 0.2000 |
| Max | 0.7046 |
| Mean | 0.2897 |
| Median | 0.2645 |
| Std | 0.0836 |

---

## Cross-Reference Coverage

### Drug × Target Gene
- **5,200** unique combinations
- Average genes per drug: **2.3**
- Min genes per drug: 1
- Max genes per drug: 13
- No drug targets all 100 genes

### Drug × Cell Line
- **2,548** unique combinations
- Average cell lines per drug: **43.6**
- Drugs present in all 49 cell lines: **1**

---

## Drug-Based Condition Split Strategies

Unlike the Norman dataset (gene knockouts), Tahoe uses:
- **Condition = target_gene + drug**
- Splitting by **drug** tests generalization to unseen drugs

### Strategy 1: Random Drug Holdout
Split drugs randomly into train/val/test:
- Train: 36 drugs (~1.6M cells)
- Val: 5 drugs (~225k cells)
- Test: 11 drugs (~496k cells)

### Strategy 2: Leave-One-Drug-Out
Cross-validation with each drug as test fold:
- 52 test folds
- Each test: ~45k cells (avg)

### Strategy 3: Stratified by Drug Size
Group drugs by cell count quartiles:
- Small (<Q1): 1 drug
- Medium (Q1-Q3): 35 drugs
- Large (≥Q3): 16 drugs

> [!TIP]
> For **unseen drug generalization** evaluation, use Strategy 1 or 2.
> For **balanced evaluation**, use Strategy 3 to ensure similar-sized drugs in each split.

---

## Comparison with Norman Dataset

| Aspect | Norman | Tahoe |
|--------|--------|-------|
| Perturbation type | Gene knockout | Drug + Target gene |
| Condition definition | Gene(s) | target_gene + drug |
| Split strategy | By seen/unseen genes | By seen/unseen drugs |
| Double perturbation | Gene + Gene | N/A (single drug-gene pairs) |
| Cell lines | Single | 49 cell lines |

---

## var Columns

| Column | Description |
|--------|-------------|
| `gene_idx` | Gene index |
