# Evaluation Metrics for the Project

Notes: All the original data are already be log1p-normalized.

## 1. Perturbation Discrimination Score (PDS)

The Perturbation Discrimination Score evaluates whether the model can correctly identify the true perturbation among all possible perturbations. It measures how well the **predicted perturbation delta vector** matches the **true perturbation delta vector**, using cosine similarity and ranking.

### 1.1 Pseudobulk Construction

For each perturbation class $k \in \{1, \dots, N\}$ and the non-targeting control (NTC):

* True pseudobulk expression:

$$y_k \in \mathbb{R}^G$$

* Predicted pseudobulk expression:

$$\hat{y}_k \in \mathbb{R}^G$$

* True NTC expression:

$$y_{ntc} \in \mathbb{R}^G$$

Values are log1p-normalized averages across cells.

### 1.2 Delta Computation

True perturbation effect:

$$\delta_k = y_k - y_{ntc}$$

Predicted perturbation effect:

$$\hat{\delta}_k = \hat{y}_k - y_{ntc}$$

### 1.3 Cosine Similarity

For each predicted perturbation $k$, compute cosine similarity between its predicted delta and all true deltas:

$$S_{k,j} = \frac{\hat{\delta}_k \cdot \delta_j}{|\hat{\delta}_k|_2 |\delta_j|_2}$$

### 1.4 Ranking and Score

Let $R_k$ be the rank of the *true* perturbation $\delta_k$ among all similarities with $\hat{\delta}_k$.

Mean rank:

$$PDS_{rank} = \frac{1}{N} \sum_{k=1}^N R_k$$

Normalized score:

$$nPDS_{rank} = \frac{1}{N^2} \sum_{k=1}^N R_k$$

Lower = better discrimination.

---

## 2. MAE on Top 2000 High-Variance Genes

**MAE of Top 2000 Genes by Ground Truth Fold Change**

This metric focuses on the biologically most strongly affected genes. For each perturbation, it selects the 2000 genes with the largest ground-truth log2 fold changes and computes the prediction error only on this subset.

### 2.1 Log2 Fold Change Calculation

Using raw counts with a pseudocount:

$$LFC_{k,g} = \left| \log_2(c_{k,g} + 1) - \log_2(c_{ntc,g} + 1) \right|$$

where:

* $c_k$: raw mean counts for perturbation $k$
* $c_{ntc}$: raw mean counts for NTC

### 2.2 Select Top 2000 Genes

For each perturbation:

$$\Omega_k = \text{Top2000}\left( {LFC_{k,g}} \right)$$

### 2.3 Compute MAE Over Selected Genes

$$MAE_{top2k} = \frac{1}{N} \sum_{k=1}^{N} \left( \frac{1}{2000} \sum_{g \in \Omega_k} \left| \hat{y}_{k,g} - y_{k,g} \right| \right)$$

---

## 3. Differential Expression Score (DES)

The Differential Expression Score evaluates how accurately the model predicts differential gene expression—an essential output for functional genomics and biological interpretation.

### 3.1 Differential Expression Testing

For each perturbation $k$:

* Compute differential expression p-values between perturbed and control cells using the **Wilcoxon rank-sum test** with tie correction.
* Apply the **Benjamini–Hochberg (BH) procedure** at **FDR = 0.05** to define significant DE genes.

This yields:

* Predicted DE gene set: $G_{k,\text{pred}}$
* True DE gene set: $G_{k,\text{true}}$

Let:

* $n_{k,\text{pred}} = |G_{k,\text{pred}}|$
* $n_{k,\text{true}} = |G_{k,\text{true}}|$

### 3.2 Case 1: Predicted set is smaller or equal

If

$$n_{k,\text{pred}} \le n_{k,\text{true}}$$

then DES is the fraction of true DE genes correctly predicted:

$$DES_k = \frac{|G_{k,\text{pred}} \cap G_{k,\text{true}}|}{n_{k,\text{true}}}$$

### 3.3 Case 2: Predicted set is larger

If

$$n_{k,\text{pred}} > n_{k,\text{true}}$$

To avoid penalizing predictions that over-call DE genes, construct a truncated predicted set:

* Let $\tilde{G}_{k,\text{pred}}$ contain the **top $n_{k,\text{true}}$** predicted DE genes, ranked by absolute log fold change.

Then compute:

$$DES_k = \frac{|\tilde{G}_{k,\text{pred}} \cap G_{k,\text{true}}|}{n_{k,\text{true}}}$$

### 3.4 Overall DES

The final score is the mean over all perturbations:

$$DES = \frac{1}{N} \sum_{k=1}^N DES_k$$

---

## 4. Final Metrics for Validation and Test

### 4.1 Loss Function

Use PyTorch's built-in **SmoothL1** loss (Huber-style L1/L2 compromise) to directly regress $\hat{y}$ against $y$ in **log1p space**. Either the function name or mathematical definition can be written in code comments.

**PyTorch implementation:**

```python
torch.nn.SmoothL1Loss(beta=1.0, reduction="mean")
```

Note: Older versions may not have the `beta` parameter; use the default SmoothL1 in that case.

**Mathematical definition** (element-wise, let $e = \hat{y} - y$):

When $|e| < \beta$:

$$\ell(e) = \frac{1}{2} \frac{e^2}{\beta}$$

When $|e| \ge \beta$:

$$\ell(e) = |e| - \frac{1}{2}\beta$$

Finally, take the mean over all elements.

### 4.2 Validation Metrics per Epoch

For each epoch, record both the "raw three metrics" and the "normalized three sub-scores" (all in the same direction), then compute a **weighted geometric mean** composite score (default weights all equal to 1). The calculation procedures and meanings for PDS and Top2000-MAE follow the documentation (pseudobulk, delta, cosine similarity, ranking, Top2000 gene selection, and MAE).

#### 4.2.1 Raw Three Metrics (Computed According to Evaluation Definitions)

**PDS (lower is better):**

For each perturbation $k$, first compute pseudobulk to obtain $y_k$, $\hat{y}_k$, and $y_{ntc}$. Construct $\delta_k = y_k - y_{ntc}$ and $\hat{\delta}_k = \hat{y}_k - y_{ntc}$. Compute cosine similarity $S_{k,j}$ with all true deltas $\delta_j$. Obtain the rank $R_k$ of the true perturbation in the sorted similarities with $\hat{\delta}_k$. Then compute the mean rank:

$$\text{MeanRank} = \frac{1}{N} \sum_{k=1}^N R_k$$

If you also want to keep the normalized version from the documentation, you can record it simultaneously:

$$nPDS_{rank} = \frac{1}{N^2} \sum_{k=1}^{N} R_k$$

**Top2000-MAE (lower is better):**

Select the Top2000 gene set $\Omega_k$ for each perturbation based on ground truth LFC, then compute:

$$MAE_{top2k} = \frac{1}{N} \sum_{k=1}^{N} \left( \frac{1}{2000} \sum_{g \in \Omega_k} |\hat{y}_{k,g} - y_{k,g}| \right)$$

**DES (higher is better):**

Compute according to the DES definition:

$$\text{DES} = \frac{1}{N} \sum_{k=1}^N DES_k$$

This uses Wilcoxon rank-sum test + BH (FDR=0.05) to compute set intersection ratios. When the predicted set is larger than the true set, truncate using $|\text{logFC}|$ to the same size before computing the intersection ratio.

#### 4.2.2 Normalized Three Sub-Scores (All "Higher is Better", Range 0~1)

**PDS score:**

Linearly invert MeanRank to 0~1:

$$s_{\text{PDS}} = \frac{N - \text{MeanRank}}{N - 1}$$

When MeanRank = 1, $s_{\text{PDS}} = 1$; when MeanRank = N, $s_{\text{PDS}} = 0$.

**MAE score:**

Transform MAE into a 0~1 "higher is better" score (recommended stable mapping that does not depend on global model min/max):

$$s_{\text{MAE}} = \frac{1}{1 + \frac{MAE_{top2k}}{\tau}}$$

Default value for $\tau$ (choose one for reproducibility):

* $\tau = MAE_{top2k}^{\text{baseline}}$ (baseline, e.g., "always predict NTC pseudobulk" or "always predict training set mean" Top2000-MAE on validation)
* If you don't want to compute a baseline, use $\tau = \text{median}(\{MAE_{top2k}(k)\})$ (median of per-perturbation MAE_top2k for the epoch)

**DES score:**

DES is already 0~1 and higher is better:

$$s_{\text{DES}} = \text{DES}$$

#### 4.2.3 Composite Metric (Weighted Geometric Mean, Default Weights All Equal to 1)

$$S_{\text{geo}} = \left( s_{\text{PDS}}^{w_1} \cdot s_{\text{MAE}}^{w_2} \cdot s_{\text{DES}}^{w_3} \right)^{\frac{1}{w_1 + w_2 + w_3}}$$

With default $w_1 = w_2 = w_3 = 1$:

$$S_{\text{geo}} = (s_{\text{PDS}} \cdot s_{\text{MAE}} \cdot s_{\text{DES}})^{1/3}$$

This is "higher is better". If you prefer a "lower is better" total score, also output:

$$L_{\text{geo}} = 1 - S_{\text{geo}}$$

as the final monitoring/early stopping metric.

### 4.3 Final Test Metrics and `perturbation_metrics.csv`

#### 4.3.1 Final Test Metrics (Global Summary, Written to Log)

Recommended fields (same set as validation for comparison):

* `pds_mean_rank`
* `pds_nrank` (optional)
* `mae_top2000`
* `des`
* `s_pds`
* `s_mae`
* `s_des`
* `s_geo` (and optionally `l_geo = 1 - s_geo`)

The raw calculations for PDS/MAE are consistent with the documentation.

#### 4.3.2 `perturbation_metrics.csv` (Per-Perturbation Records for Diagnosing Underperformers)

One row per perturbation $k$. Recommended minimum columns (all can be obtained from the existing evaluation process):

* `perturbation_id` (or name of $k$)
* `rank_Rk` (the true perturbation rank $R_k$ from PDS)
* `rank_Rk_norm` (recommended to store $R_k/N$ for cross-dataset comparison)
* `cosine_self` ($\cos(\hat{\delta}_k, \delta_k)$, for diagnosing signature alignment)
* `mae_top2000_k` ($\frac{1}{2000}\sum_{g \in \Omega_k} |\hat{y}_{k,g} - y_{k,g}|$)
* `des_k` ($DES_k$ computed according to the DES definition)
* (Optional but useful) `n_true_de` ($n_{k,\text{true}}$), `n_pred_de` ($n_{k,\text{pred}}$), `n_intersect` (intersection size), to help explain why $DES_k$ is high/low

When modifying code according to this checklist, in the validation/test evaluator: first generate pseudobulk and delta (already needed for PDS), simultaneously obtain $\Omega_k$ (already needed for Top2000), plus DES set statistics, and you can output both the global and per-perturbation tables.
