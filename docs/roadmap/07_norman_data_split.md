# Norman (GEARS-processed) Condition-Level Split  
**For Reverse Perturbation Tasks (Final, Implementable Design)**

Below is a **fully implementable condition-level split scheme** for the **Norman dataset (GEARS-processed)**, specifically designed for your **reverse perturbation** task. It is also compatible with your two downstream directions:

- scGPT forward modeling  
- B1 gene scoring as a “multi-class” classification task  

This document **only defines which single-gene and double-gene conditions belong to train/val/test**.  
It **does not** involve reference/query splits or retrieval library construction.

The design follows the GEARS / community-standard principle of **gene-visibility–based stratification**:

- Double perturbations are grouped by how many of their genes have been seen as *single perturbations* during training:  
  **0/2, 1/2, or 2/2 seen**
- This stratification is explicitly used in the GEARS paper to evaluate combinatorial generalization. [PMC][1]
- The GEARS-processed Norman dataset already provides statistics for single- and double-perturbation train/val/test splits  
  (e.g., single: ~70/8/27; double: ~36/16/69; exact numbers vary with filtering). [Virtual Cell Models][2]

---

## 0. Core Principles (Hard Constraints)

**Principle 1 — Split by condition, not by cell**  
The split unit is the **condition**, not individual cells.  
All cells belonging to a condition (e.g., `X` or `X+Y`) must appear in **only one** of train, val, or test.  
No condition-level leakage is allowed.

**Principle 2 — Definition of “unseen”**  
A gene is considered *unseen* if it has **never appeared as a single-gene perturbation in the training set**.  
This is exactly the definition used by GEARS to construct 0/1/2-seen difficulty tiers. [PMC][1]

**Principle 3 — Controls are excluded from condition labels**  
Control cells may remain in the dataset for forward modeling or auxiliary purposes,  
but they are **not part of the perturbation condition label space** you aim to identify.

---

## 1. Data and Label Standardization (Before Splitting)

1. Use the **GEARS-processed Norman dataset** as input.  
   This version is already log-normalized and typically restricted to the top 5,000 HVGs. [Virtual Cell Models][2]

2. Normalize `obs["condition"]` into a canonical format:
   - Single-gene: `X`
   - Double-gene: `X+Y`  
     Enforce **lexicographic ordering**: `min(X,Y)+max(X,Y)`  
     (to avoid treating `A+B` and `B+A` as different conditions).

3. Filter low-quality conditions (strongly recommended):
   - Count cells per condition.
   - Remove any condition with fewer than `min_cells_per_condition`
     (e.g., 30 or 50).  
   - Remove all associated cells.
   
   Without this step, the split becomes unstable and noisy.

---

## 2. Final Split Design (Core Logic)

### 2.1 Select the “Unseen Gene Set”

Let **G** be the full set of single-gene perturbations  
(~105 genes in the GEARS-processed Norman dataset). [Virtual Cell Models][2]

Select a subset **G_unseen ⊂ G** to serve as **test-only unseen genes**.

Recommended settings:
- Size: **20–30%** of all single-gene perturbations
- Fix a random seed for reproducibility
- Stratify approximately by cell count per gene  
  (to avoid removing all high-coverage or all low-coverage genes)

This choice fully determines:
- Which single-gene conditions are test-only
- The difficulty tier of each double-gene condition

---

### 2.2 Single-Gene Condition Assignment

Split single-gene conditions into two categories:

#### A. Unseen single genes → Test only
If `g ∈ G_unseen`:
- The single-gene condition `g` goes **entirely into test**
- Corresponds to GEARS “1/1 unseen” evaluation [PMC][1]

#### B. Seen single genes → Train / Val
If `g ∉ G_unseen`:
- Assign condition `g` to the training side
- Further split **by condition** into train and val

Recommended ratios (by number of conditions):
- 90/10 or 85/15 (train/val)
- Validation should remain small to preserve training diversity

Resulting sets:
- `train_single` (seen)
- `val_single` (seen)
- `test_single` (unseen)

This closely matches common Norman GEARS splits (e.g., ~70/8/27). [Virtual Cell Models][2]

---

### 2.3 Double-Gene Condition Assignment (Stratified)

For each double-gene condition `a+b`, determine how many of its genes are seen  
(i.e., not in `G_unseen`). This yields three GEARS-standard difficulty tiers: [PMC][1]

1. **2/2 seen (combo_seen2)**  
   Both `a` and `b` are seen genes

2. **1/2 seen (combo_seen1)**  
   Exactly one of `a`, `b` is unseen

3. **0/2 seen (combo_seen0)**  
   Both genes are unseen

#### Rule 1 — Strict unseen definition (strongly recommended)
- `combo_seen1` and `combo_seen0` → **test only**

Rationale:  
If a gene never appears as a single perturbation in training,  
it should not appear in training even as part of a combination.  
Otherwise, the notion of “unseen gene” becomes diluted for reverse perturbation.

#### Rule 2 — Only `combo_seen2` may enter training
- `combo_seen2` conditions are split into train / val / test
- Purpose:
  - Allow the model to learn some gene–gene interactions
  - Preserve easier double perturbations as test references

Recommended ratios (by condition count):
- Train: 60–70%
- Val: 10–15%
- Test: 20–30%

As a result, the **double-gene test set naturally contains**:
- `test_double_seen2` (easier)
- `test_double_seen1` (harder)
- `test_double_seen0` (hardest)

This aligns well with reported Norman double-perturbation splits  
(e.g., ~36/16/69). [Virtual Cell Models][2]

---

## 3. Final Split Artifacts to Produce

To support **both scGPT forward training** and **B1 gene-scoring training**  
using the same split, you should output:

1. `train_conditions.txt`  
   - `train_single`  
   - `train_double_seen2`

2. `val_conditions.txt`  
   - `val_single`  
   - `val_double_seen2`

3. `test_conditions.txt`  
   - `test_single_unseen`  
   - `test_double_seen2`  
   - `test_double_seen1`  
   - `test_double_seen0`

Additionally, produce:

4. `test_strata.json` (or equivalent)  
   - Explicitly list test double-gene conditions by tier:
     - `seen2`
     - `seen1`
     - `seen0`

This stratification is essential for:
- Reporting tier-wise top-K hit rates
- Error diagnosis
- Clear experimental narratives in papers

---

**References**  
[1] PMC — GEARS paper  
[2] Virtual Cell Models — GEARS-processed Norman dataset

---

## Final Implementation (Optimized Settings)

**Configuration** (default in `src/configs/*.yaml`):
- `unseen_gene_fraction: 0.15` → 15 unseen genes (15% of singles)
- `min_cells_per_condition: 50` (singles), `min_cells_per_double: 30` (doubles)
- `seen_single_train_ratio: 0.9`, `combo_seen2_train_ratio: 0.7`, `combo_seen2_val_ratio: 0.15`

**Raw Data**:
`data/norman/perturb_processed.h5ad` has shape (91205, 5045) — 91,205 cells by 5,045 genes.

**Results** (Norman dataset, seed=42):
- Total: 236 conditions | 83,803 cells (post-filter)
- **Train**: 147 conditions (56,580 cells)
  - 81 singles, 66 doubles (seen2)
  - 18,262 cells for gene-gene interaction learning
- **Val**: 23 conditions (6,560 cells)
- **Test**: 66 conditions (20,712 cells)
  - 15 single_unseen, 15 combo_seen2, 31 combo_seen1, 5 combo_seen0

**Key metrics**:
- 34% more interaction learning cells vs baseline (18.3k vs 13.6k)
- 27% more double-gene combos (66 vs 52)
- 71% of train combos have ≥200 cells (robust)

**Artifact**: `data/norman/splits/norman_condition_split_seed42.json`
