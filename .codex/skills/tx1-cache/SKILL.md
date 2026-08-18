---
name: tx1-cache
description: Use before touching src/aivc_model/tx1_basal.py, tx1_embed_cache.py, tx1_predicted_response.py, or scripts/build_tx1_basal_embeddings.py. The Tx1 basal embedding cache has several failure modes that produce a complete-looking, silently wrong cache, and its Phase-A line manifest is SHA-256-pinned by the Exp13 split builder, not by the cache.
---

# Tx1 basal embedding cache

Pipeline: `tx1_basal.py` (assemble AnnData) → `tx1_embed_cache.py` (write/verify)
→ `scripts/build_tx1_basal_embeddings.py` (GPU encode) → `load_line_cache`
(`:416`), consumed by `tx1_predicted_response.py` and `tx1_response_data.py`.

Almost every failure here is **silent**. Errors would be a good day. Work through
the checks below rather than trusting a zero exit code.

## Silent-corruption modes

**Zero-filled HVG matrix.** `_resolve_hvg_matrix`
(`tx1_embed_cache.py:1110-1152`) zero-fills any checkpoint gene missing from the
source. Point it at a *valid but wrong* symbol column and you get a near-all-zero
`hvg.npy` — logged only as a WARNING plus an `hvg_fill_rate` record.
`verify_cache` never checks it. **Read `hvg_fill_rate` after every build.**

**Stale cells after a parameter change.** A *missing* `sample_provenance.json` is
trusted as legacy (`tx1_embed_cache.py:1099`), so changing
`--max-cells-per-line` or `--seed` silently reuses the old sampled cells.
Relatedly, the `hvg_gene_order_sha256` consistency check passes when *all*
entries lack the key (`:687-705`) — unanimous absence reads as agreement.

**Sharded verify ≠ verify.** `verify_cache(only_lines=...)` skips the
untracked-directory check (`:556-561`) and narrows completeness to the requested
shard. A shard exiting 0 tells you nothing about the cache as a whole. Always
finish with one unrestricted pass and require `"status": "verified"`.

**Manifest clobber.** `write_run_manifest` merges `lines` but replaces
`config_snapshot` wholesale, in an unlocked read-then-write
(`tx1_embed_cache.py:323-393`). Concurrent shards lose snapshots — do not
run shards against one manifest in parallel without serializing this.

**Destructive replace.** `write_line_cache` `rmtree`s the final directory before
`os.replace` (`tx1_embed_cache.py:186-190`). An interrupted write leaves no cache
and no backup.

## Gene-ID and column traps

Every source names its genes differently, and one flag applies to a whole run:

| Source | Gene column |
|---|---|
| Tahoe | `gene_symbol` |
| X-Atlas-Orion | `gene_name` |
| Perturb-seq h5ad anchors | `gene_id` in `var.index` |

`--hvg-gene-symbol-col` is **per-run, not per-line** — a mixed-source run must be
sharded by source.

- Tahoe drops token ids `<3` as special (`tx1_basal.py:46,205`). X-Atlas must
  **not** — token ids 0-4 are ordinary genes there (`_row_to_xatlas_cell`, `:800-813`).
- Tokens absent from `gene_metadata.parquet` are dropped with a WARNING only
  (`:268-276`).
- The GPU encoder reads `adata.var["ensembl_id"]`
  (`scripts/build_tx1_basal_embeddings.py:127`); every builder now materializes
  that column explicitly, including the h5ad anchors that carry ids in
  `var.index` (`tx1_basal.py:362-371`). Do not remove that step.
- `_ENSEMBL_ID_PATTERN` (`tx1_basal.py:45`) accepts version suffixes
  (`ENSG…​.12`) that `vocab[gene]` will then reject.
- `GeneBags.for_genes` (`state_core.py:324`) upper-cases requested symbols.

## The Phase-A manifest — one pin survives, three files lost theirs

`verify_cache` resolves its expected line list from
`manifest.json["config_snapshot"]["line_manifest_path"]`, which in practice points
at `results/phase_a_tx1_20260724/cell_line_manifest.csv` (`:491,609-643`). Four
Phase-A files remain tracked there, but the T2 evaluator that pinned
`phase_a_registration.json` by SHA-256, the `freeze_tx1_phase_a.py` builder, and
the byte-identical `configs/experiments/12_tx1_st_geneeffect/phase_a/` copy were
all deleted at `873c99c`.

**`cell_line_manifest.csv` is still pinned** — by the Exp13 split builder, not the
cache: `build_cell_line_geneeffect_226_split.py:24` holds its SHA-256 and `_pin`
(`:36-39`, `pin_inputs=True` by default) **raises** on mismatch, so regenerating it
breaks the split rebuild. The other three files are now unpinned by anything.

Exp13 does not use the Phase-A `train_head`/`test` role column —
`benchmark_split.assert_fit_eligible` over
`configs/benchmarks/cell_line_geneeffect_226_split.json` is the membership
authority, and resurrecting the old role column is an explicit anti-goal
(`benchmark_split.py:1-17`).

## Checkpoint loading

Every surviving loader validates its load result instead of trusting
`strict=False`: `validate_load_result` (`scripts/verify_tx1_obsm_width.py:97-119`),
mirrored by `state_warm_start.py:61-127` and `tx1_predicted_response.py:95,209-223`.
A bare `load_state_dict(..., strict=False)` leaves weights randomly initialized and
reports success. `install_padding_metadata_fallback`
(`verify_tx1_obsm_width.py:45-50`) monkeypatches llmfoundry, so it must run
*before* model construction.

When adding a new backbone loader, copy this pattern.
