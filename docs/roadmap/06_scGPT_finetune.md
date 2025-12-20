The Norman dataset can support **task-adaptive fine-tuning of scGPT** for your **reverse perturbation retrieval** benchmark, but the right strategy is almost always **parameter-efficient** (a small head and/or LoRA) rather than full fine-tuning, because Norman is comparatively small and you will otherwise overfit or learn dataset-specific shortcuts.

Below is a top-level, benchmark-oriented answer: what to fine-tune, what objectives make sense for *reverse retrieval*, and a clear **Definition of Done (DoD)** for comparing **frozen encoder** vs **fine-tuned scGPT**.

---

## 1) What “fine-tuning scGPT” should mean for *this* task

Your task is not “reconstruct expression” (ExprDecoder) and not classic cell-type classification; it is:

* Input: a cell (or pseudo-bulk) expression signature (y) (optionally with context (x))
* Output: ranking over perturbation conditions (p) (reverse identification / retrieval)

So the most direct fine-tuning target is the **embedding geometry** used by retrieval, not the expression decoder.

Practically, you want scGPT to produce embeddings where:

* cells/prototypes from the same perturbation condition are close
* different conditions are separated
* masking perturbed genes prevents “cheating” on the target gene identity

That aligns with training either:

* a **retrieval head** (projection + similarity calibration), and/or
* **LoRA adapters** in the transformer blocks to adjust representation minimally.

---

## 2) Where to add trainable modules in your shown checkpoint

Given the architecture you showed, there are three natural adaptation points (ordered from safest to riskiest).

### A) Frozen backbone + train a small retrieval head (recommended baseline fine-tune)

Freeze:

* `encoder`, `value_encoder`, `transformer_encoder`

Train:

* a small projection MLP on top of the pooled cell embedding (e.g., `[512 -> 256/512 -> D]`)
* optional learnable temperature / normalization for cosine similarity
* optional lightweight “prototype aggregator” if you use multiple prototypes per condition

This is usually enough to get a measurable gain over “frozen scGPT encoder + cosine” while staying stable.

### B) Add LoRA to transformer attention (and optionally FFN) + head (recommended “strong” fine-tune)

Freeze all original weights, but insert LoRA into:

* MultiheadAttention projections (conceptually Q/K/V/O; depending on implementation access you may apply LoRA to `out_proj` and/or the internal in-proj weights if exposed)
* optionally FFN `linear1/linear2` in each encoder layer

Train:

* LoRA parameters
* the same retrieval head as in (A)

This gives you more adaptation capacity without the brittleness of full fine-tuning.

### C) Partial/full fine-tuning of last N layers (only if you explicitly need it)

Unfreeze:

* last 1–3 transformer layers (plus head)

This can improve performance but is the easiest way to overfit Norman and weaken generalization claims, especially under hold-out conditions/genes.

---

## 3) Training objectives that match retrieval (and how they map to your metrics)

To compare “inference-only” vs “fine-tuned”, you need a training loss that aligns with ranking.

Two clean options:

### Option 1: Supervised contrastive / InfoNCE (retrieval-native)

Construct positives and negatives:

* positives: two signatures from the same condition (cell-cell, cell-prototype, prototype-prototype)
* negatives: signatures from other conditions in the batch

Loss encourages correct condition clustering; evaluation is still your retrieval pipeline with top-K / MRR / NDCG.

This is the most “retrieval-faithful” approach.

### Option 2: Multiclass condition classification (simple and strong baseline)

Train a classifier head to predict `cond_id` from the embedding.
At inference:

* take top-K classes as retrieved candidates
* compute the same top-K / MRR / NDCG

This is easy to implement and often competitive, but it is a closed-set formulation unless you carefully define candidate space.

In both cases, keep your **mask perturbed genes** rule during training and evaluation to preserve interpretability.

---

## 4) DoD: Fine-tuned scGPT vs Frozen scGPT comparison (benchmark-grade)

This is a concrete “done” checklist that ensures a fair, publishable comparison.

### DoD-A: Experimental design parity

* Same dataset version, same preprocessing, same gene set, same masking policy.
* Same condition-level split files (train/val/test), saved and reused across all runs.
* Same definition of candidate set for retrieval (train-only vs all observed vs predicted; explicitly stated).
* Same reference library strategy (mean vs bootstrap prototypes; same prototype counts).

### DoD-B: Two scGPT baselines are runnable end-to-end

1. **Frozen scGPT encoder** (inference-only)

* Embedding extraction is deterministic under fixed seed.
* Retrieval metrics reported (Top-K exact/relevant, MRR, NDCG), plus masking ablation.

2. **Fine-tuned scGPT** (one of A/B above, plus optionally both)

* A clearly defined trainable component set:

  * “Head-only” fine-tune (must exist)
  * “LoRA + head” fine-tune (recommended second variant)
* Training uses only train split; early stopping and hyperparameter selection use val split only.
* Test split is untouched until final evaluation.

### DoD-C: Training signals are aligned with retrieval and leak-free

* Loss function is explicitly stated (contrastive or classification).
* No use of test conditions (or test cells) in any fit step (including PCA fit, normalization fit, prototype estimation if you treat prototypes as learned parameters).
* Masking is applied consistently (train/val/test) unless you are explicitly running a “no-mask ablation”.

### DoD-D: Reporting artifacts for comparison

For each run (frozen and fine-tuned), produce:

* One summary metrics table (macro over conditions + micro over cells).
* The same plots/tables for both:

  * Top-K curves (K = 1/5/8/10)
  * MRR, NDCG
  * masking ON vs OFF deltas
* At least one stability check:

  * performance across 3 random seeds, or confidence intervals via bootstrap over queries.

### DoD-E: A decision-ready conclusion

* A single comparison table that answers:

  * Does head-only fine-tuning outperform frozen scGPT materially?
  * Does LoRA add additional gain beyond head-only?
  * Are gains preserved under the harder split regime you care about (hold-out conditions, and optionally hold-out genes)?
* A short error analysis showing what changes with fine-tuning (which confusions reduce, which remain).

---

## 5) Practical guidance on what Norman “can support” without overfitting

* Norman is strong enough to support **head-only** and **LoRA+head** fine-tuning credibly, especially if you use pseudo-bulk/prototypes to reduce noise.
* It is generally not strong enough (as a single dataset) to justify full-model fine-tuning unless you are very disciplined with splits and regularization, and even then the generalization story can weaken.

If you want the cleanest benchmark narrative: implement **three scGPT modes** under identical retrieval evaluation:

* scGPT frozen (encoder-only)
* scGPT + trainable retrieval head
* scGPT + LoRA + retrieval head

That trio is usually sufficient to demonstrate whether adaptation helps and how much capacity is needed, without turning the project into architecture exploration.