# Exp13 Stage 0 — Tx1 input representation (CPM vs raw counts)

**Status:** completed 2026-08-18. Tx1-3B does **not** read a CPM row like the raw counts
underneath it; the shift is systematic and survives pooling. Branch 1 of
[`../04-exp13-geneeffect-residual-protocol.md`](../04-exp13-geneeffect-residual-protocol.md)
§6 was taken and executed: raw UMI counts now exist for all 152 Kinker lines. Authority:
[`../01-blueprint.md`](../01-blueprint.md) §7-8. This is a substrate measurement, not an SL
or a dependency result.

## What was tested

Whether $x_c = E_{\text{Tx1}}(r_c)$ is trustworthy for the 152 `kinker_sccle` lines, whose
published matrix is processed CPM rather than UMI counts. §6 made this a blocking question
because ST consumes the Tx1 embedding as its Stage-1 *input*, so an untrustworthy embedding
voids the whole forward pass for a line rather than one feature block.

A second question was added after reading the collator: the released checkpoint sets
`max_length: 2048` with `sampling: true`, and `_sample` draws its gene subset with an
**unseeded** `torch.randperm`, so cells above that width are re-sampled on every encode.

## Method and provenance

`scripts/stage0_tx1_input_probe.py` at commit `260b4bf`, run under `.venv-tx1` on one H20-3e.
Four arms over the same cells in the same order: `raw`; `cpm` (`raw * 1e6 / library_size`);
`repeat_seeded`; `repeat_unseeded`. Each seeded arm calls `torch.manual_seed` immediately
before its forward pass — without that pinning the CPM and subsampling effects are
confounded, since both perturb the same embedding. Cells are reported split by whether they
exceed 2048 detected genes, because only wider cells can trigger `_sample` at all.

Two runs: 256 raw-count Adamson cells (`adamson_2016_pilot.h5ad`, integer-audited) as a
proxy, and 256 cells of `ACH-000211` (Daoy) — a Kinker line on the **test** side of
`cell_line_geneeffect_226_split.json`, built by `scripts/prepare_kinker_umi_h5ad.py`.
Artifacts: `results/stage0/tx1_input_probe_adamson_pilot.json`,
`results/stage0/tx1_input_probe_kinker_ACH-000211.json`.

## Result

Cosine to the `raw` arm; `pooled` is the cosine of the per-line mean embedding, which is the
quantity `z_c` and `Delta` actually consume.

| arm | cohort | narrow cells (≤2048 genes) | wide cells | pooled |
| --- | --- | --- | --- | --- |
| CPM vs raw | Adamson | differs, 0.955 | differs, 0.939 | 0.979 / 0.989 |
| CPM vs raw | Kinker (test line) | differs, 0.954 | differs, 0.921 | **0.972 / 0.987** |
| repeat, seeded | both | identical | identical | 1.00000 |
| repeat, unseeded | both | identical | differs, 0.945–0.959 | **0.99970** |

## Interpretation

The CPM arm differs from raw **even on cells below the context window with the RNG pinned**,
so the effect is the input representation itself, not gene subsampling. It is a bias, not
noise: it survives pooling at 0.972–0.987, while the unseeded subsampling difference — of
comparable size per cell — pools away to 0.9997.

A static reading of the collator predicts the opposite and is wrong. `collator_config.yml`
sets `do_binning: true`, and `binning()` bucketizes each cell against quantiles of its own
nonzero values, so `bucketize(k*x, k*q) == bucketize(x, q)` for any `k > 0`. The collator
also clones an unbinned `expr_raw` before binning and the model reads `batch["expr_raw"]`
(`model.py:381`); float32 boundary effects in `torch.quantile` are a second candidate. The
mechanism is not pinned down here, and the measurement does not depend on it.

Seeding makes the encoder bit-identical, so the subsampling finding is a reproducibility
requirement rather than a defect in the data: a collator seed must be pinned wherever the
cell-sampling order is pinned (§5).

## Verdict and scope

§6 is closed by branch 1. `UMIcount_data.txt` (Broad Single Cell Portal SCP542, 3.47 GB,
56,982 pre-QC cells, 207 lines) covers all 40,670 selected cells and all 152 selected lines,
and carries cell-line identity in the file, so no genotype demultiplexing was needed.
`prepare_kinker_umi_h5ad.py` ingested it: 0 non-integer values in 151,986,988 nonzeros,
40,670/40,670 cell-line labels reconciled against the CPM-derived selection, 19,181 genes
mapped to the Tx1 vocabulary, HVG fill rate 0.077 (identical to the CPM build), and all 152
per-line h5ads pass `assert_tx1_input_contract`.

This measurement licenses no claim about GeneEffect, dependency, or synthetic lethality. The
226-line split membership is unchanged — only the numeric source moved — and the frozen line
manifest, whose hash the split builder pins, was not edited. The CPM artifacts remain as the
comparison arm and must not re-enter the standard Tx1 path.

## Reproduction

```bash
PYTHONPATH=src:. .venv-tx1/bin/python scripts/prepare_kinker_umi_h5ad.py \
  --matrix <sources>/kinker_sccle/UMIcount_data.txt \
  --selected-cells <derived>/kinker_sccle/selected_cells.tsv \
  --line-manifest <manifest>/cell_line_atlas_179_manifest.csv \
  --gene-metadata <tahoe>/gene_metadata.parquet --vocab-json <tx1>/vocab.json \
  --hvg-var-dims <state>/var_dims.pkl --output-dir <processed>/kinker_umi_152

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. .venv-tx1/bin/python \
  scripts/stage0_tx1_input_probe.py --adata <processed>/kinker_umi_152/h5ad/ACH-000211.h5ad \
  --model-dir data/models/tahoe_x1_3b/3b-model --n-cells 256 --batch-size 8 \
  --out-json results/stage0/tx1_input_probe_kinker_ACH-000211.json
```

Input hashes are recorded in the ingest's `qc.json: source_hashes` and in each probe JSON's
`source` block. HPC: `data/sl_dependency_v0/processed/cell_line_atlas_179/kinker_umi_152/`.
