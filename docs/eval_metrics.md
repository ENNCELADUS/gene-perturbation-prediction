# Reverse perturbation metrics

Metrics for evaluating reverse-perturbation retrieval, where each query perturbation is a condition like `t = X + Y` and the model returns a ranked list of candidate conditions `R_K(t)` (top `K`).

1. **Exact-match Hit@K** (scGPT “correct retrieval”)
   - **Definition:** For each test perturbation `t`, retrieve top-`K` candidates `R_K(t)`. Score `1` if the exact condition appears in `R_K(t)`, else `0`. Report the mean over test cases.

     $$
     \text{Hit@K}_{\text{exact}}
     = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[t_i \in R_K(t_i)].
     $$

2. **One-gene-overlap Hit@K** (scGPT “relevant retrieval”)
   - **Definition:** Score `1` if any retrieved condition shares at least one gene with the target (e.g., overlaps `{X, Y}`), else `0`. Report the mean over test cases.

     $$
     \text{Hit@K}_{\ge 1}
     = \frac{1}{N}\sum_{i=1}^{N}
       \mathbf{1}\Big[\exists r \in R_K(t_i):
       |\text{genes}(r)\cap \text{genes}(t_i)| \ge 1\Big].
     $$

3. **MRR** (Mean Reciprocal Rank)
   - **Definition:** For each test case, take the rank `rank(t)` of the exact ground-truth condition in the retrieved list (define as `\infty` if not present).

     $$
     \text{MRR}=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{\mathrm{rank}(t_i)}.
     $$

   - **Bio-meaning:** Measures how quickly the true perturbation is surfaced to a wet-lab shortlist.

4. **mAP** (mean Average Precision) *(optional)*
   - **Definition:** Treat “positives” as a set of relevant conditions (e.g., exact match only; or exact + one-gene overlap). For each query, compute average precision over ranks, then average across queries.
   - **Bio-meaning:** Rewards placing biologically relevant candidates early, not just “somewhere in top-`K`”.

5. **NDCG@K** (Normalized Discounted Cumulative Gain) *(optional)*
   - **Definition:** Assign graded relevance `rel(r)` to each retrieved item (e.g., `2` for exact match, `1` for one-gene overlap, `0` otherwise).

     $$
     \text{DCG@K}=\sum_{j=1}^{K}\frac{2^{\mathrm{rel}_j}-1}{\log_2(j+1)},\quad
     \text{NDCG@K}=\frac{\text{DCG@K}}{\text{IDCG@K}}.
     $$

   - **Bio-meaning:** Distinguishes close-but-not-exact mechanisms from irrelevant ones.

6. **DE gene set overlap** (F1 or Jaccard)
   - **Definition:** Compare predicted vs. observed differentially expressed (DE) gene sets `S_pred` and `S_obs`.

     $$
     \text{Jaccard}=\frac{|S_{\text{pred}}\cap S_{\text{obs}}|}{|S_{\text{pred}}\cup S_{\text{obs}}|},\quad
     \text{F1}=\frac{2PR}{P+R}.
     $$

   - **Bio-meaning:** Tests whether retrieved perturbations reproduce the same transcriptional program, not just nearest-neighbor similarity.

7. **Pathway enrichment agreement** (FDR / NES concordance) *(optional)*
   - **Definition:** Run pathway enrichment on predicted-response DE genes and on observed DE genes; report overlap of significant pathways (e.g., Jaccard on pathway sets at `FDR < 0.05`) and/or correlation of pathway scores (NES).
   - **Bio-meaning:** Evaluates mechanistic consistency at the pathway level, which is often more robust than individual genes.
