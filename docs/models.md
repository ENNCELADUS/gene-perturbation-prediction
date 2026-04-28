# Models

The active source tree should be read as model-first: each method produces a
ranked gene list for inverse perturbation retrieval.

## PCA-kNN

Config: `src/pca_knn/config.yaml`

PCA-kNN embeds query cells and training cells into a lower-dimensional
expression space, retrieves nearby training examples, and transfers target-gene
evidence from neighbors to candidate genes.

This is a non-parametric baseline. It is useful for checking whether local
expression similarity already explains the inverse retrieval task.

## Random Forest

Config: `src/random_forest/config.yaml`

The random forest baseline treats the task as multi-label gene prediction from
expression features. It is a compact supervised baseline for testing whether
classical ML can recover target genes without a foundation-model backbone.

## scGPT Gene Score

Config: `src/scgpt/configs/norman.yaml`

The scGPT path uses a pretrained scGPT transformer as a cell encoder and trains
a gene-scoring head. The model compares the query cell embedding, optionally
after subtracting matched control embeddings, against candidate gene embeddings
from the scGPT gene token encoder.

The current scGPT model is direct inverse retrieval, not forward generation.
It does not generate post-perturbation profiles for all candidate conditions.

The shared scGPT loader and inverse gene-score model live in `src/scgpt/model.py`.
Training and evaluation are driven by `run_config.stages` in the model config.

## Architecture Plot

The scGPT architecture visualization is stored under:

```text
docs/plot/scgpt_pipeline.tex
docs/plot/scgpt_pipeline.pdf
```
