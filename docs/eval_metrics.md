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

The primary metric is the **Normalized Perturbation Discrimination Score ($nPDS_{rank}$)**:

$$nPDS_{rank} = \frac{1}{N^2} \sum_{k=1}^N R_k$$

* Range: $[1/N, 1]$
* Interpretation: Lower is better (best = $1/N$, worst = $1$).

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

The model minimizes a composite loss function ($L_{\text{total}}$) that balances cell-level reconstruction and perturbation-level discrimination. Default weights are:

$$
L_{\text{total}} = 0.60 \cdot L_{\text{SW1}} + 0.25 \cdot L_{\text{Proto}} + 0.10 \cdot L_{\text{Rank}} + 0.05 \cdot L_{\text{Dir}}
$$

*   **$L_{\text{SW1}}$ (Sliced Wasserstein)**: Aligns distribution shapes between predicted and true cell populations.
*   **$L_{\text{Proto}}$ (ProtoInfoNCE)**: Contrastive loss on pseudobulk deltas to enforce perturbation distinctiveness.
*   **$L_{\text{Rank}}$ (DE Rank)**: Ensures significant DE genes have larger absolute predicted changes than non-DE genes.
*   **$L_{\text{Dir}}$ (DE Direction)**: Penalizes incorrect sign predictions for significant DE genes.

### 4.2 Updated Overall Score (Official-style scaling; adapted to your metric definitions)

Let the validation metrics computed using **your** definitions be:

* $DES_{\text{pred}} \in [0, 1]$ (higher is better)
* $nPDS_{\text{pred}} = \frac{1}{N^2} \sum_{k=1}^N R_k$ (lower is better)
* $MAE_{\text{pred}}$ (your Top2000-MAE; lower is better)

Let the corresponding baseline scores (cell-mean baseline model) under the **same** metric definitions be:
* $DES_{\text{base}} = 0.1075$
* $nPDS_{\text{base}} = 0.5167$
* $MAE_{\text{base}} = 0.1258$

Define a clipping operator:
$$
\text{clip}_{[0,1]}(x) = \min(1, \max(0, x))
$$

### 4.3 Scaled DES (same as official)

Because the best possible $DES$ is $1$:
$$
DES_{\text{scaled}} = \text{clip}_{[0,1]} \left( \frac{DES_{\text{pred}} - DES_{\text{base}}}{1 - DES_{\text{base}}} \right)
$$

### 4.4 Scaled PDS (using $nPDS_{rank}$ definition)

Official scoring logic adapted to your normalized PDS ($nPDS_{\text{pred}}$), where lower is better. The best attainable value is $nPDS_{\min} = \frac{1}{N}$.

$$
nPDS_{\text{scaled}} = \text{clip}_{[0,1]} \left( \frac{nPDS_{\text{base}} - nPDS_{\text{pred}}}{nPDS_{\text{base}}} \right)
$$

This normalization yields zero if performance is no better than baseline, and approaches one as performance improves (implementation simplifies theoretical optimum to 0).

### 4.5 Scaled MAE (official logic, with your Top2000-MAE)

For MAE, the best value is $0$, so the official scaling becomes:
$$
MAE_{\text{scaled}} = \text{clip}_{[0,1]} \left( \frac{MAE_{\text{base}} - MAE_{\text{pred}}}{MAE_{\text{base}}} \right)
$$

### 4.6 Final Overall Score (official aggregation)

Official aggregation is the **unweighted arithmetic mean** of the three scaled scores, multiplied by 100:
$$
S_{\text{overall}} = 100 \cdot \frac{DES_{\text{scaled}} + nPDS_{\text{scaled}} + MAE_{\text{scaled}}}{3}
$$