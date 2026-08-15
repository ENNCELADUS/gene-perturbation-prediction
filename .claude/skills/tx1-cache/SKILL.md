---
name: tx1-cache
description: Use before touching src/aivc_model/tx1_basal.py, tx1_embed_cache.py, tx1_geneeffect_eval.py, or scripts/build_tx1_basal_embeddings.py. The Tx1 basal embedding cache has several failure modes that produce a complete-looking, silently wrong cache, plus a hash-pinned Phase-A contract that is easy to void.
---

# Tx1 basal embedding cache

Pipeline: `tx1_basal.py` (assemble AnnData) → `tx1_embed_cache.py` (write/verify)
→ `scripts/build_tx1_basal_embeddings.py` (GPU encode) →
`tx1_geneeffect_eval.py` (frozen-contract evaluation).

Almost every failure here is **silent**. Errors would be a good day. Work through
the checks below rather than trusting a zero exit code.

## Silent-corruption modes

**Zero-filled HVG matrix.** `_resolve_hvg_matrix`
(`tx1_embed_cache.py:1136-1152`) zero-fills any checkpoint gene missing from the
source. Point it at a *valid but wrong* symbol column and you get a near-all-zero
`hvg.npy` — logged only as a WARNING plus an `hvg_fill_rate` record.
`verify_cache` never checks it. **Read `hvg_fill_rate` after every build.**

**Stale cells after a parameter change.** A *missing* `sample_provenance.json` is
trusted as legacy (`tx1_embed_cache.py:1100`), so changing
`--max-cells-per-line` or `--seed` silently reuses the old sampled cells.
Relatedly, the `hvg_gene_order_sha256` consistency check passes when *all*
entries lack the key (`:704`) — unanimous absence reads as agreement.

**Sharded verify ≠ verify.** `verify_cache(only_lines=...)` skips the
completeness and untracked-directory checks (`tx1_embed_cache.py:548-561`).
A shard exiting 0 tells you nothing about the cache as a whole. Always finish
with one unrestricted pass and require `"status": "verified"`.

**Manifest clobber.** `write_run_manifest` merges `lines` but replaces
`config_snapshot` wholesale, in an unlocked read-then-write
(`tx1_embed_cache.py:342-352,385-391`). Concurrent shards lose snapshots — do not
run shards against one manifest in parallel without serializing this.

**Destructive replace.** `write_line_cache` `rmtree`s the final directory before
`os.replace` (`tx1_embed_cache.py:189-191`). An interrupted write leaves no cache
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

- Tahoe drops token ids `<3` as special (`tx1_basal.py:44,203`). X-Atlas must
  **not** — token 0 is a real gene there (`:633-646`).
- Tokens absent from `gene_metadata.parquet` are dropped with a WARNING only
  (`:268-276`).
- The GPU encoder reads `adata.var["ensembl_id"]`
  (`scripts/build_tx1_basal_embeddings.py:127`), but `build_perturbseq_basal_adata`
  only re-indexes `var` and keeps the source columns (`tx1_basal.py:360-364`) —
  the three h5ad anchors have no `ensembl_id` column at all.
- `_ENSEMBL_ID_PATTERN` (`tx1_basal.py:43`) accepts version suffixes
  (`ENSG…​.12`) that `vocab[gene]` will then reject.
- `GeneBags.for_genes` (`prepare.py`) upper-cases requested symbols.

## The frozen Phase-A contract

`FROZEN_REGISTRATION_SHA256` (`tx1_geneeffect_eval.py:1116`) pins the exact bytes
of `results/phase_a_tx1_20260724/phase_a_registration.json`. **Regenerating any
Phase-A artifact voids it** — that is the point; it is an out-of-band trust anchor,
not a checksum of convenience. Re-running `scripts/freeze_tx1_phase_a.py`
rewrites the contract files and makes every evaluation raise
`EvaluationContractError`. If a change requires regenerating it, that is a
contract change: stop and raise it with the user.

The four contract files also exist byte-identical under
`configs/experiments/12_tx1_st_geneeffect/phase_a/` (which holds nine files in
total — the extra five are provenance artifacts). The evaluator defaults to the
`results/` copy (`scripts/evaluate_tx1_backbone.py:64`) — edit one and they
desync silently.

The gate evaluator always verifies registered artifact hashes and the full
evaluation contract before writing `verdict.json`.

**Double calibration.** Emitting Phase-E already-adapted predictions *without*
`panel`/`k` columns silently re-applies `affine_kshot_calibrate`
(`tx1_geneeffect_eval.py:256-257,395-405`). Carry the columns through.
`strict=False` (`:1041`) disables all coverage validation.

## STATE vs Tx1 loading

`model.py:922-928` imports STATE by string with `strict=False` — **missing
weights load silently**. Tx1 does the right thing instead:
`validate_load_result` (`scripts/verify_tx1_obsm_width.py:97-119`). And
`install_padding_metadata_fallback` (`:45-50`) monkeypatches llmfoundry, so it
must run *before* model construction.

When adding a new backbone loader, copy the Tx1 pattern, not the STATE one.
