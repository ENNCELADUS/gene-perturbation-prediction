# GeneEffect Head Revision — Design

Date 2026-10-03. Follows [the revision design](2026-10-03-geneeffect-revision-design.md); decided with the user
through a design interview after the objective screen.

## 1. Why

The factorised head of the revision overfits: training-diagnostic selective Spearman reached 0.53 (Huber) and 0.67
(standardised MSE) while validation stalled at 0.15. The cause is capacity on the line context, not a bottleneck:

- the Tx1 context embedding $z_c$ (5120 = mean and variance of 2560-d cell embeddings) takes only **170** distinct
  values during training, so its rank is at most 169; z-scored, 90% of its variance lies in 86 components and 95%
  in 114. The trunk's and the context tower's first layers spent 1.7M and 1.4M weights on those 170 points, enough
  to key a lookup table on line identity;
- after per-dimension z-scoring, $z_c$ is 77% of the trunk input width and the basal statistics $q_{g,c}$ 0.06%;
- the trunk sees $e_g$ and $z_c$ together, so it can fit any gene × line function on its own;
- the Tx1 context-PCA ridge, the strongest control, reaches 0.13 with 8 context components.

## 2. Decisions

| Question | Decision |
| --- | --- |
| Head form | Nested low rank: $\hat\delta_{g,c}=\sigma_g\big[\langle G(g),C(\tilde z_c)\rangle/\sqrt r+h(q_{g,c},s,\Delta_{\text{proj}},e_g)\big]$. $C$ reads only the line context; $h$ only the per-(g, c) features. With a linear $C$ and per-gene $G$ the first term is the context-PCA ridge, so the model contains the strongest control |
| Context compression | $\tilde z_c$ = 128 principal components of $z_c$, fitted on the labelled training lines after per-dimension z-scoring (as the Tx1 ridge does); fitted preprocessing, saved in and restored from the checkpoint like the gene means |
| PCA scaling | Eigen-scaled: all scores divided by one constant, $\sqrt{\lambda_1}$, so the first component has unit variance and the tail stays small. Not whitened; $\tilde z_c$ bypasses the per-dimension block standardiser |
| Context tower $C$ | $C(z)=Wz+\operatorname{SwiGLU}(z)$: a linear map 128 → r plus a residual SwiGLU branch 128 → 128 → r whose output layer is zero-initialised, so training starts at a reduced-rank ridge |
| Rank $r$ | 64 (`model.factor_rank`, unchanged) |
| Gene factor $G$ | Free per-gene embedding (17,787 × 64, normal std 0.02) plus Linear(e_g 1280 → 64), unchanged |
| Correction $h$ | Per-block encoders, each Linear → LayerNorm: $q$ (3 + mask) → 16, $s$ (6 + 2 masks) → 16, $\Delta_{\text{proj}}$ 256 → 32, $e_g$ 1280 → 32; concatenated (96, fewer when blocks are off) → SwiGLU hidden 64 → 1, output layer zero-initialised. No $z_c$ |
| Activation | SwiGLU, $(xW_1\odot\operatorname{SiLU}(xW_2))W_3$, in both hidden layers (h and C's residual branch) |
| Regularisation | Dropout 0.1 on h's concatenated encodings and in C's residual hidden layer (`model.dropout`); `train.weight_decay` 0.05 |
| No-STATE | $h$ then sees $q$ and $e_g$ only; unchanged rule (`use_delta_proj` and `use_s` false skip STATE and the adapter) |
| Code | The nested head replaces the trunk-plus-factor head; `GeneEffectMLP` stays for the readout head. Checkpoints of the earlier head load only at its commit |

New config keys: `model.context_components` (128), `model.dropout` (0.1). Removed: `model.head_hidden` and
`model.head_layers` for the joint head (the readout keeps its own widths). Block encoder and hidden widths are fixed in
code, not configured.

## 3. Screen

One run on the four H20s of port 30734, seed 0, train/val + test (one config, one experiment): the new head with STATE
frozen and the objective that wins the objective screen (Huber unless the Pearson-block run beats 0.1515 validation
selective Spearman). It is compared with the frozen-STATE run of the same objective under the earlier head on validation;
test is reported. The STATE screen follows on the head that wins.

## 4. Claim boundaries

As in the revision design: validation chooses, test is reported, nothing here is SL evidence. PCA, gene means,
selective genes and residual SD are fitted on labelled training lines only.
