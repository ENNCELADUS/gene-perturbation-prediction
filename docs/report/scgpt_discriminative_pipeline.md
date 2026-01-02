# scGPT Discriminative Pipeline: Inputs and Flow

## Short Diagram

```
AnnData X (cell x gene) + obs (condition, control, batch, cell_type)
        |
        | 1) GeneScoreDataset: pick perturbed cell + matched controls
        v
Binned expression vectors (perturbed + control) + gene token IDs
        |
        | 2) tokenize_and_pad_batch
        v
(gene_ids, values, padding_mask) for perturbed + control batches
        |
        | 3) scGPT backbone (Transformer)
        v
cell_emb (perturbed) and control cell_embs
        |
        | 4) delta embedding (perturbed - mean(control))
        v
Gene scoring head (dot or MLP with gene embeddings)
        |
        v
Score vector over all genes (rank targets)
```

## Text Description

The discriminative (Route B1) pipeline operates on **cell expression vectors**, not gene sequences. Each training example starts from AnnData `X` (cell x gene) and uses the scGPT gene vocabulary to build a **gene token ID list** aligned to the dataset's gene order. Expression values are binned to `n_bins` and tokenized by **keeping only non-zero genes**, prepending a `<cls>` token, and padding to a fixed length.

For each perturbed cell, the loader samples matched **control cells** (by batch/cell_type when available) and computes their embeddings in the same way. The model subtracts the mean control embedding from the perturbed embedding to form a delta representation. A lightweight head then scores **every gene** by matching the delta embedding to the gene embeddings from the scGPT encoder. The output is a gene score vector used for ranking and top-K metrics.
