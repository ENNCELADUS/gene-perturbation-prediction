This roadmap defines two high-level technical routes for **reverse perturbation prediction** using the **GEARS-processed Norman dataset** and the **condition-level split** described at the end of `docs/roadmap/07_norman_data_split.md`. Both routes keep the same train/val/test split unit (condition, not cell), and both operate over the **full processed gene space** (HVGs in the GEARS-processed AnnData), not the original 20-gene subspace from the scGPT paper.

The two routes differ in *how* the model infers perturbation identity from expression: one uses **generative forward modeling + retrieval**, the other uses **direct discriminative inference**.

---

## 1) Route A — Forward Modeling + Retrieval (scGPT-style, adapted to Norman)

### 1.1 Task Objects and Split Units

In this dataset, a “case” corresponds to a **perturbation condition** (e.g., `X`, `Y`, `X+Y`). The basic unit for train/val/test is therefore **condition-level**, not an individual cell. This mirrors the split defined in `docs/roadmap/07_norman_data_split.md`, including the seen/unseen gene tiers for double perturbations.

Unlike the original scGPT paper’s 20-gene subspace, we operate on the **full GEARS-processed Norman set**, which includes all singles and doubles in the processed AnnData (after filtering). The specific condition counts and strata are as follows:

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

### 1.2 Construction of the Reference Database and Query Set

This is the core of the retrieval formulation, adapted to the processed Norman data:

1. **Train a forward perturbation model** to predict perturbed expression from a control cell and a condition label.  
   The model learns a mapping from a control state to a predicted perturbed state, using the condition-level split.

2. **Reference database (search corpus):**  
   For **all conditions in the split universe**, the forward model generates *predicted* perturbed expressions.  
   For each condition, sample a fixed number of control cells (or a balanced subset) and generate predicted profiles.  
   This produces a reference library of predicted profiles tagged by condition.

3. **Query set:**  
   For **test conditions only**, all experimentally measured **ground-truth perturbed cells** are used as queries.

### 1.3 Similarity, Voting, and Output

- **First stage:**  
  Each query cell retrieves its Top-K nearest neighbors from the reference library using a defined similarity measure in expression or embedding space.  
  Each neighbor corresponds to a *(condition, predicted profile)* pair.

- **Second stage:**  
  Retrieval results across all query cells for a test condition are **aggregated by condition via voting or score pooling**.  
  Conditions with stronger aggregated support rank higher.

- **Output:**  
  The final Top-K results are **Top-K conditions**, not individual profiles.

### 1.4 Evaluation Metrics (Top-K Hit)

Both metrics are defined at the **test-condition level**:

- **Correct retrieval (exact match):**  
  For a test condition (e.g., X+Y), if X+Y appears in the Top-K ranked conditions, the case is counted as a hit.

- **Relevant retrieval (one-gene overlap):**  
  If any Top-K condition shares **at least one gene** with the test condition, the case is counted as a hit.

### 1.5 Route A Implementation Details (scGPT Forward Finetune)

This section captures the concrete model components found in the checkpoint and a detailed finetuning plan that freezes most weights.

**Checkpoint inspected:** `model/scGPT/best_model.pt`  
**Observed module families (state_dict prefixes):**
- `encoder.*` → token embedding + encoder normalization (`encoder.embedding`, `encoder.enc_norm`)
- `flag_encoder.*` → perturbation flag encoder (binary/condition flags)
- `value_encoder.*` → expression value projection MLP + norm
- `transformer_encoder.layers.{0..11}.*` → 12-layer Transformer encoder
- `decoder.fc.*` → MLP decoder head for expression reconstruction
- `mvc_decoder.*` → masked value completion head (`W`, `gene2query`)

#### 1.5.1 Finetune Objective (Forward Model)
Given control cells and perturbation condition tokens, predict perturbed expression profiles in the **full GEARS-processed gene space** (5,045 genes).

**Primary loss targets:**
- Reconstruction loss on expressed genes (MSE or Huber, consistent with scGPT training).
- Optional auxiliary loss for MVC head if used in the base checkpoint.

#### 1.5.2 Freeze Strategy (Default)
Freeze **most** weights to preserve pretraining while adapting to Norman:

**Frozen by default:**
- `encoder.*` (token embedding + encoder norm)
- `flag_encoder.*`
- `value_encoder.*`
- `transformer_encoder.layers.0..11.*`

**Trainable by default:**
- `decoder.fc.*` (expression reconstruction head)
- `mvc_decoder.*` (masked value completion head)

**Optional partial unfreeze (if underfitting):**
- Unfreeze only `transformer_encoder.layers.11.*` (last block) and `encoder.enc_norm.*` for light adaptation.
- Keep a lower learning rate for unfrozen encoder blocks (e.g., 5-10x smaller than decoder).

#### 1.5.3 Loading Checkpoint and Freezing (Implementation Skeleton)

1) Build the model with the same architecture as the checkpoint (scGPT base config).  
2) Load `state_dict` from `model/scGPT/best_model.pt`.  
3) Set `requires_grad = False` for frozen modules, keep decoders trainable.  
4) Use an optimizer with parameter groups for different learning rates.

Example (pseudocode, to be adapted in `src/` training entry point):

```python
ckpt = torch.load("model/scGPT/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt, strict=True)

for name, param in model.named_parameters():
    param.requires_grad = False
    if name.startswith("decoder.") or name.startswith("mvc_decoder."):
        param.requires_grad = True

# Optional partial unfreeze
for name, param in model.named_parameters():
    if name.startswith("transformer_encoder.layers.11") or name.startswith("encoder.enc_norm"):
        param.requires_grad = True

optimizer = torch.optim.AdamW(
    [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("decoder.")], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("mvc_decoder.")], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("transformer_encoder.layers.11")], "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.enc_norm")], "lr": 1e-5},
    ],
    weight_decay=0.01,
)
```

#### 1.5.4 Training Loop Details (Forward Modeling)

- **Inputs:** control cell expression, perturbation flags (single or double).  
- **Outputs:** predicted perturbed expression across 5,045 genes.  
- **Batching:** group by condition for stable per-condition gradients; keep condition-level split.  
- **Determinism:** fix RNG seeds; deterministic control-cell sampling per condition for reproducibility.

**Recommended schedule:**
- Warmup 5% steps, cosine decay.
- Early stop on validation reconstruction loss or retrieval Top-K hit proxy.

#### 1.5.5 Retrieval Database Construction (Post-Finetune)

For each condition in the full split universe:
- Sample a fixed number of control cells (e.g., 200 per condition or min available).
- Generate predicted perturbed profiles using the finetuned forward model.
- Store the generated vectors with condition labels to build the retrieval index.

This provides the reference set used in the Route A Top-K retrieval evaluation.

#### 1.5.6 Evaluation Metrics (Implementation Notes)

The retrieval metrics in `src/evaluate/metrics.py` match the Route A definitions:
- `top_k_accuracy` for exact hit@K
- `one_gene_overlap_hit_at_k` for relevant retrieval hit@K
- `mrr` and `ndcg` for ranked-list quality
- Optional macro hit@K across conditions

Use these APIs in `src/main.py` when evaluating the final retrieval rankings on the **test conditions**:
- Build ranked condition lists per query cell.
- Aggregate metrics via `compute_all_metrics(...)`.
- Report both exact and one-gene-overlap hit@K on the test set.

---

## 2) Route B — Direct Discriminative Inference (Compositional)

This route avoids forward generation and instead **infers perturbation identity directly from observed expression**. A closed-set classifier over condition IDs cannot predict unseen combinations, so the discriminative route must be **compositional** to generalize to unseen pairs defined in the condition split.

A clean and fully comparable approach is the following.

### Route B1: Gene-Level Scoring + Condition Composition (Recommended)

Instead of predicting **condition IDs**, the model predicts **perturbed genes as atomic labels**, which naturally supports unseen combinations in the Norman split.

**Definition:**

- **Input:** a single perturbed cell expression (or its embedding).
- **Model:** a discriminative network producing **gene-level scores** over the full processed gene set  
  (scores reflect whether each gene is likely part of the perturbation).

- **Condition scoring from gene scores:**
  - Single-gene condition: score equals the gene’s score.
  - Two-gene condition: score is composed from the two gene scores  
    (sum, log-sum, or another simple composition rule).

- **Ranking:**  
  Enumerate all **conditions defined by the Norman split** and compute a full condition ranking per query cell.

- **Evaluation:**  
  For a test condition, reuse **exact Top-K hit** and **one-gene-overlap Top-K hit**, exactly as in Route A.

**Advantages:**
- The condition split is strictly respected (test conditions are unseen during training).
- Generalization to unseen combinations is handled naturally via label composition.
- The evaluation protocol is directly comparable to Route A.

This is a genuinely different **problem definition**, yet it remains fully aligned with scGPT’s Top-K hit metrics.

---

## 3) Comparing the Two Routes

If the objective is:

> *Given a perturbed expression profile, identify the perturbation condition that generated it, even when that condition was unseen during training*,  

both approaches can be evaluated under the same **condition split** and **Top-K hit metrics**, but they emphasize different capabilities.

### Route A (Forward + Retrieval) Emphasizes
- Learning a **strong forward perturbation generator**, such that generated candidate responses cover the true distribution.
- Inferring causes by **retrieval over generated hypotheses**.
- Supporting conditions without experimental data, which is well aligned with **experimental proposal and extrapolation** scenarios.

### Route B (Discriminative Compositional) Emphasizes
- Directly extracting a **perturbation fingerprint** from expression data without generating candidate profiles.
- Natural generalization to unseen combinations via **gene-level compositionality**.
- Potential limitations in capturing fine-grained expression heterogeneity across different cell states.

---

## 4) A Strictly Comparable Benchmark Setup (Recommended)

Given that you already have a PCA / retrieval-style evaluator framework, the cleanest way to compare problem definitions is:

1. **Fix the condition split** (identical train/val/test conditions).
2. **Fix the query set:** ground-truth perturbed cells from test conditions.
3. **Two scoring routes:**
   - **Route A (forward + retrieval):**  
     Reference = forward-model–generated profiles for all conditions  
     (balanced control-cell sampling per condition) + similarity search + voting.
   - **Route B (gene-level compositional):**  
     Train a gene-level predictor; for each query cell, enumerate and rank all conditions; optionally aggregate scores or votes across cells.
4. **Report the same metrics:**  
   Top-K (e.g., K = 1/5/10/20), with both **exact** and **one-gene-overlap** hit rates.

This comparison isolates **problem definition differences**—generative retrieval versus discriminative compositional inference—rather than confounding factors such as model size or training tricks.
