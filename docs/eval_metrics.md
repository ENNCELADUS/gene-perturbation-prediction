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
