# Formulation Gap: K562 A->B->C to True SL Benchmark Models

Status date: 2026-06-08

This note formalizes the gap between the current K562 dependency chain and a
true synthetic-lethality benchmark model. It is intentionally about model input
and output shape, not about implementation details.

## Benchmark Anchor

Reference paper:
`docs/literature/literature/02_data/benchmarks/2024_nature_communications_benchmarking_machine_learning_methods_for_synthetic_lethality_prediction_in_cancer.pdf`.

The benchmark paper fixes several requirements that the current K562 model does
not yet satisfy:

- The supervised object is a gene pair, not a single perturbation target.
- Every model must emit a floating-point pair score so the same output can be
  evaluated as both classification and ranking.
- Inputs may include SL labels, GO, PPI, pathways, knowledge graphs, gene
  expression, dependency-score correlations, protein complexes, or sequence.
- Evaluation includes positive-negative ratios such as `1:1`, `1:5`, `1:20`,
  and `1:50`, because true non-SL pairs greatly outnumber known SL pairs.
- Splits include CV1 pair holdout, CV2 one-unseen-gene holdout, and CV3
  two-unseen-gene holdout.
- Context-specific SL remains a harder setting: the same gene pair can be SL in
  one cancer context or cell line and not in another.

## Current Formulation

The current implemented chain is a single-cell-line dependency model:

```text
A: K562 control-cell state + perturbation target
    -> B or B_hat: observed or predicted post-perturbation response
    -> C: DepMap GeneEffect for the same target in K562
```

Let:

$$
c_0 = \mathrm{K562}
$$

$$
g \in \mathcal{G}
$$

where $g$ is the perturbation target gene.

The current A-side input is:

$$
A_{c_0,g} = (X^{\mathrm{ctrl}}_{c_0}, g)
$$

where $X^{\mathrm{ctrl}}_{c_0}$ is the K562 control-cell expression
distribution and $g$ is represented by target masking or another target
encoding.

The B-side object is an observed or predicted post-perturbation response
distribution:

$$
B_{c_0,g} = \{x_{c_0,g,i}\}_{i=1}^{n_{c_0,g}}
$$

or:

$$
\hat{B}_{c_0,g} = F_{A \to B}(A_{c_0,g})
$$

The C-side target is:

$$
y_{c_0,g} = \mathrm{GeneEffect}(c_0, g)
$$

The current downstream predictor is:

$$
\hat{y}_{c_0,g}
= h_C\left(\phi_B(B_{c_0,g})\right)
$$

or in the predicted-B setting:

$$
\hat{y}_{c_0,g}
= h_C\left(\phi_B(\hat{B}_{c_0,g})\right)
$$

Here $\phi_B$ is the response-distribution featureizer, such as scVI latent
encoding plus GMM/prototype occupancy features, and $h_C$ is the dependency
head, such as Ridge.

This is a valid dependency formulation, but it is not a true SL formulation.
It predicts whether target loss is fitness-relevant in one cell context. It
does not model a two-gene interaction, context specificity, or SL benchmark
labels.

## True SL Benchmark Formulation

The SL benchmark task is closer to a gene-pair link-prediction and ranking
problem:

```text
context c + anchor/background gene a + candidate target gene b
    -> SL score for (c, a, b)
    -> ranked candidate partners for a
```

In a context-agnostic benchmark, $c$ may be omitted or treated as a global
context:

$$
s_{a,b} = f_{\mathrm{SL}}(a, b)
$$

In a cancer-relevant context-specific benchmark, the required object is:

$$
s_{c,a,b} = f_{\mathrm{SL}}(c, a, b)
$$

where:

- $c$ is a cancer context, such as cell line, lineage, mutation background, copy
  number state, pathway state, or patient-derived context;
- $a$ is the anchor or background gene/event that defines the vulnerable
  context;
- $b$ is the candidate target gene;
- $s_{c,a,b}$ is a continuous score used for both binary classification and
  ranking.

The benchmark label can be binary:

$$
\ell_{c,a,b} \in \{0, 1\}
$$

or continuous when a combinatorial screen supplies interaction strength:

$$
\gamma_{c,a,b} = \mathrm{GI}(c, a, b)
$$

where lower or more negative interaction scores usually indicate stronger
synthetic interaction, depending on the assay convention.

## Input Gap

| Component | Current K562 A->B->C | True SL benchmark model |
| --- | --- | --- |
| Context | Fixed $c_0=\mathrm{K562}$ | Variable $c$ across cell lines, lineages, mutation states, or patient-like contexts |
| Gene key | Single target gene $g$ | Ordered query pair or unordered pair $(a,b)$ |
| Response input | $B_{c_0,g}$ or $\hat{B}_{c_0,g}$ | Optional $B_{c,b}$, $B_{c,a}$, or double-perturbation $B_{c,a,b}$ if available |
| Label | $\mathrm{GeneEffect}(c_0,g)$ | SL label $\ell_{c,a,b}$, GI score $\gamma_{c,a,b}$, or benchmark pair label $\ell_{a,b}$ |
| Gene prior | Mostly target identity / target coordinate | GO, PPI, pathway, KG, co-expression, co-dependency, protein-complex, sequence, or pair graph features |
| Negative space | Non-essential or held-out genes in K562 dependency regression | Unknown/non-SL gene pairs with explicit negative sampling ratios and false-negative risk |
| Generalization unit | Held-out target gene in K562 / Adamson transfer | CV1 pair split, CV2 one-unseen-gene split, CV3 two-unseen-gene split, plus cell-line/context holdout |

The minimal input extension is:

$$
I_{c,a,b}
= \left[
e_c(c);
e_g(a);
e_g(b);
e_{ab}(a,b);
\phi_B(B_{c,b})
\right]
$$

where:

- $e_c(c)$ encodes cell-line or cancer-context omics;
- $e_g(a)$ and $e_g(b)$ encode individual genes;
- $e_{ab}(a,b)$ encodes pairwise relation evidence;
- $\phi_B(B_{c,b})$ keeps the current perturbation-response signal as a
  target-specific mechanistic feature.

If anchor-response or double-perturbation data exist, the richer input is:

$$
I_{c,a,b}
= \left[
e_c(c);
e_g(a);
e_g(b);
e_{ab}(a,b);
\phi_B(B_{c,a});
\phi_B(B_{c,b});
\phi_B(B_{c,a,b});
\Delta_{ab}
\right]
$$

with an interaction residual such as:

$$
\Delta_{ab}
= \phi_B(B_{c,a,b})
- \psi\left(\phi_B(B_{c,a}), \phi_B(B_{c,b})\right)
$$

where $\psi$ is an additive or independence baseline.

## Output Gap

The current model outputs one dependency score:

$$
\hat{y}_{c_0,g}
$$

A true SL benchmark-compatible model must output at least one pair score:

$$
\hat{s}_{c,a,b}
= h_{\mathrm{SL}}(I_{c,a,b})
$$

For binary SL labels:

$$
\hat{p}_{c,a,b}
= \sigma(\hat{s}_{c,a,b})
\approx P(\ell_{c,a,b}=1)
$$

For ranking:

$$
\mathrm{rank}_c(a)
= \mathrm{sort}_{b \in \mathcal{G}_{\mathrm{candidate}}}
\hat{s}_{c,a,b}
$$

The artifact-level output should therefore be pair-indexed:

```text
context_id
anchor_gene_id
candidate_gene_id
score
probability
label
label_source
split_type
rank_scope
```

This differs from the current dependency artifact, which is effectively:

```text
cell_line_id
perturbation_gene_id
gene_effect_true
gene_effect_pred
```

## Required Intermediate Layer: Context-Selective Dependency

There is a useful bridge between the current dependency model and true SL:

```text
context c + candidate target b
    -> context-selective dependency score
```

This layer is not yet true SL, but it is necessary because SL requires
selective vulnerability rather than broad essentiality.

Let:

$$
d_{c,b} = \mathrm{Dependency}(c,b)
$$

Define a context group:

$$
\mathcal{C}^+_a
= \{c : \mathrm{context}(c,a)=1\}
$$

and matched controls:

$$
\mathcal{C}^-_a
= \{c : \mathrm{context}(c,a)=0\}
$$

A context-selective target score can be:

$$
\mathrm{Selectivity}(a,b)
=
\mathbb{E}_{c \in \mathcal{C}^+_a}[d_{c,b}]
-
\mathbb{E}_{c \in \mathcal{C}^-_a}[d_{c,b}]
-
\lambda \cdot \mathrm{PanEssentialPenalty}(b)
$$

This supports SL-like prioritization:

```text
anchor/background a
    -> targets b whose dependency is stronger in a-positive contexts
```

But it should still be labeled `SL-like`, not true SL, unless the pair is
validated by known SL labels, combinatorial screens, or other interaction
evidence.

## Benchmark-Compatible Objective

A benchmark-fitted SL model should support both classification and ranking.

Classification objective:

$$
\mathcal{L}_{\mathrm{cls}}
=
\sum_{(c,a,b)}
w_{c,a,b}
\cdot
\mathrm{BCEWithLogits}(\hat{s}_{c,a,b}, \ell_{c,a,b})
$$

The weights $w_{c,a,b}$ should be allowed to reflect label source confidence,
positive/negative sample imbalance, and negative sampling strategy.

Ranking objective:

$$
\mathcal{L}_{\mathrm{rank}}
=
\sum_{c,a}
\sum_{b^+ \in \mathcal{P}_{c,a}}
\sum_{b^- \in \mathcal{N}_{c,a}}
\max(0, m - \hat{s}_{c,a,b^+} + \hat{s}_{c,a,b^-})
$$

where $\mathcal{P}_{c,a}$ are known SL partners and $\mathcal{N}_{c,a}$ are
negative or unknown candidate partners.

The benchmark output must be evaluable with:

- AUROC, AUPR, and F1 for pair classification;
- NDCG@K, Recall@K, and Precision@K for partner ranking;
- CV1, CV2, and CV3 gene-pair splits;
- context or cell-line holdout splits for context-specific SL.

## What The Current Model Can Contribute

The current model should be treated as a mechanistic feature generator and
dependency submodel:

$$
\phi_B(B_{c,b}) \to \hat{d}_{c,b}
$$

It can contribute:

- target-response features for candidate target $b$;
- predicted dependency score $\hat{d}_{c,b}$;
- response-burden and residual transcriptomic structure;
- uncertainty or error estimates from observed-B versus predicted-B comparison.

It cannot by itself supply:

- the anchor/background gene $a$;
- a pairwise interaction label;
- negative pair sampling;
- partner ranking for each anchor gene;
- cross-context selectivity;
- CV2/CV3-style cold-start generalization.

## Minimal Next Formulation

The smallest mathematically honest extension is:

$$
\hat{s}_{c,a,b}
=
h_{\mathrm{SL}}
\left(
\left[
e_c(c);
e_g(a);
e_g(b);
e_{ab}(a,b);
\hat{d}_{c,b};
\phi_B(B_{c,b})
\right]
\right)
$$

with:

$$
\hat{d}_{c,b}
= h_C(\phi_B(B_{c,b}))
$$

This keeps the current A->B->C chain as a dependency feature while adding the
missing SL benchmark axes: context, anchor gene, candidate gene, pair priors,
pair label, and ranking output.

The stronger but data-hungry formulation is:

$$
\hat{s}_{c,a,b}
=
h_{\mathrm{SL}}
\left(
\left[
e_c(c);
e_g(a);
e_g(b);
e_{ab}(a,b);
\phi_B(B_{c,a});
\phi_B(B_{c,b});
\phi_B(B_{c,a,b});
\Delta_{ab}
\right]
\right)
$$

This is the first form that directly models interaction response rather than
inferring SL from target dependency plus context.

## Claim Boundary

Use these labels precisely:

| Formulation | Acceptable claim |
| --- | --- |
| $\phi_B(B_{c_0,g}) \to \mathrm{GeneEffect}(c_0,g)$ | K562 dependency prediction |
| $\phi_B(\hat{B}_{c_0,g}) \to \mathrm{GeneEffect}(c_0,g)$ | Predicted-transcriptome dependency ranking |
| $(c,b) \to d_{c,b}$ across many contexts | Multi-context dependency prediction |
| $(a,b)$ or $(c,a,b) \to s_{\mathrm{SL}}$ with SL labels | SL benchmark prediction |
| $(c,a,b)$ with context holdout and validated interaction evidence | Context-specific SL target discovery |

The current repository is strongest at the first two rows. The true benchmark
model starts at the fourth row.
