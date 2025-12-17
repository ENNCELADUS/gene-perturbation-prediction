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

### 3.5 Cell-Eval Differential Expression Metrics (STATE §4.7.2)

The STATE paper evaluates DE predictions with several complementary metrics using Wilcoxon rank-sum tests (BH-adjusted, $p_{adj} < 0.05$) on both observed and predicted values.

**Notation**
- Perturbations $t \in [T]$, gene set $G$.
- Significant DE sets: $G^{(DE)}_{t,\text{true}}$, $G^{(DE)}_{t,\text{pred}}$.
- Top-$k$ DE genes (ranked by $|\Delta_{t,g}|$): $G^{(k)}_{t,\text{true}}$, $G^{(k)}_{t,\text{pred}}$ with $k \in \{50,100,200,N\}$ (for $k=N$, use all true DE genes).
- Spearman rank correlation is $\rho_{\text{rank}}$.

**DE Overlap Accuracy**
$$\text{Overlap}_{t,k} = \frac{|G^{(k)}_{t,\text{true}} \cap G^{(k)}_{t,\text{pred}}|}{k}$$

**Top-$k$ Precision**
$$\text{Precision}_{t,k} = \frac{|G^{(k)}_{t,\text{true}} \cap G^{(k)}_{t,\text{pred}}|}{|G^{(k)}_{t,\text{pred}}|}$$

**Directionality Agreement**
Let $G_t^{\cap} = G^{(DE)}_{t,\text{true}} \cap G^{(DE)}_{t,\text{pred}}$ with true log-fold change $\Delta_{t,g}$ and predicted $\hat{\Delta}_{t,g}$.
$$\text{DirAgree}_t = \frac{|\{g \in G_t^{\cap} : \text{sgn}(\hat{\Delta}_{t,g}) = \text{sgn}(\Delta_{t,g})\}|}{|G_t^{\cap}|}$$

**Spearman Correlation on Significant Genes**
Let $G_t^{*} = G^{(DE)}_{t,\text{true}}$.
$$\text{Spearman}_t = \rho_{\text{rank}}(\hat{\Delta}_{t,G_t^{*}}, \Delta_{t,G_t^{*}})$$

**ROC-AUC**
Labels genes as significant (1 if $p < 0.05$, else 0) in observed data; uses predicted $-\log_{10}(p_{adj})$ as scores.
$$\text{ROC-AUC}_t = \int_0^1 \text{TPR}_t(\text{FPR}) \, d\text{FPR}$$

**PR-AUC**
Same labels/scores as ROC-AUC, reporting area under precision–recall.
$$\text{PR-AUC}_t = \int_0^1 \text{Precision}_t(r) \, d\text{Recall}$$

**Effect Size Correlation**
Counts of significant DE genes: $n_t = |G^{(DE)}_{t,\text{true}}|$, $\hat{n}_t = |G^{(DE)}_{t,\text{pred}}|$.
$$\text{SizeCorr} = \rho_{\text{rank}}\big((n_t)_{t=1}^T, (\hat{n}_t)_{t=1}^T\big)$$

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

#### 4.2.2 Normalized Three Sub-Scores

**Baseline** (simply averaging the expression from all perturbations. The baseline scores are pre-calculated on the Training dataset):

* `pds_nrank`: 0.5167
* `mae_top2000`: 0.1258
* `des`: 0.0442

##### **Overall Score**

The overall score on the leaderboard—and ultimately the score used for prize-eligible final entries—is computed by **averaging the improvement of the three metrics** relative to a **baseline cell-mean model**.

The baseline model predicts by averaging expression across all perturbations. Its baseline scores are pre-computed on the Training dataset and appear in the raw score table.

---

### **Scaled Metrics**

#### **1. Differential Expression Score (DES) & Perturbation Discrimination Score (PDS)**

- DES ranges from **0 (worst)** to **1 (best)** (higher is better).
- PDS is a (normalized) mean rank, so **lower is better**. A perfect model has
  $nPDS_{rank} = 1/N$; the baseline is worse (higher).

Their scaled versions measure improvement over the baseline, respecting each
metric's direction:

$$DES_{scaled} = \frac{DES_{prediction} - DES_{baseline}}{1 - DES_{baseline}}$$

$$PDS_{scaled} = \frac{PDS_{baseline} - PDS_{prediction}}{PDS_{baseline}}$$

---

#### **2. Mean Absolute Error (MAE)**

Since lower MAE is better and the ideal value is **0**, the scaled score is:

$$MAE_{scaled} = \frac{MAE_{baseline} - MAE_{prediction}}{MAE_{baseline}}$$

---

### **Score Clipping**

If any scaled score becomes **negative** (i.e., your model performs worse than the baseline), it is **clipped to 0**.

All scaled scores are thus bounded in:

$$0 \le \text{scaled score} \le 1$$

---

### **Final Score**

The final score is the **mean of the three scaled scores**, multiplied by 100:

$$S = \frac{1}{3}(DES_{scaled} + PDS_{scaled} + MAE_{scaled}) \times 100$$
