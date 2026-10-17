# Stage-one ladder inputs

**Status:** built on the H20 (container 30838) on 2026-10-16 and 2026-10-17 under
`data/geneeffect_joint/v2/ladder/` (Tx1 cells under `data/tx1_basal_embeddings/v2_all/`), by Task 2 of the
[PIE stage-one design](../specs/2026-10-16-pie-geneeffect-ladder-design.md) (plan
[2026-10-16](../specs/2026-10-16-pie-ladder-inputs-plan.md)); the definitions are in
[protocol §15](../03-geneeffect-protocol.md#15-stage-one-inputs) and the build record in
[`results/ladder_inputs/`](../../results/ladder_inputs/README.md). The stage-one model (protocol §16) reads them
through `src/data/ladder_reader.py`, which refuses a `pie/index.json` whose lines or genes are not `lines.parquet`'s
and the decode panel in order; no model has been trained on them yet.

## Role

Every stage-one arm (Tasks 3–7) reads these fixed inputs and never rebuilds them: the line frame and its folds, the
mean channel and PIE's input, per-line cell channels and summaries, Tx1 embeddings of every cell, gene sources, the
Cellosaurus text, and the frozen PIE's deltas. `manifest.json` records the settings (a root written under other settings
is refused), the expression space of each channel and each step's counts.

## Sources

| Input | Source | Where it was read |
| --- | --- | --- |
| Lines, labels, bulk RNA | the [final split](cell-line-geneeffect-split.md), DepMap 26Q1 (`CRISPRGeneEffect.csv`, `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`, `Model.csv`; [card](depmap-public-26q1.md)) | H20 |
| Basal cells | `data/sl_dependency_v0/processed/basal_raw_umi/<ModelID>.h5ad` (the 226 and the [single-cell extras](single-cell-atlas-search.md)) | H20 |
| `gene_text`, `string_space`, PIE's `context_text` | `arcinstitute/PIE_sources` at `cb1aaa4e` | H20 and Mac |
| Protein sequences | reviewed human UniProtKB, release 2026_03, the accessions behind PIE's `esm2` (19,202 of its 19,203 genes) | H20 |
| ESM C 600M, ESM3-open | `EvolutionaryScale/esmc-600m-2024-12` at `e4d83bc`, `EvolutionaryScale/esm3-sm-open-v1` at `47f0545`, through hf-mirror; run in an existing `esm` 3.2.1 environment on the H20 (`ESM_PYTHON`) | H20 |
| Structures | AlphaFold DB v6 human proteome (`UP000005640_9606_HUMAN_v6.tar`, `ftp.ebi.ac.uk`, SHA-256 `1de6f3d7df03…`) | H20 |
| Gene2vec | Cilantro's `gene2vec_embs.pt`, mapped through Geneformer's `gene_name_id_dict_gc30M.pkl` | H20 |
| Cellosaurus | release 56.0 (PIE's pin) through PIE's client; embedded with `text-embedding-3-large` | Mac (the H20 cannot reach Cellosaurus or OpenAI) |
| Frozen PIE | `PIE_replogle_xdataset` `best_auprc.ckpt` at `f6da9607` | H20 |

## Contents

| Path | Contents | Counts |
| --- | --- | --- |
| `lines.parquet` | every split line: role, frame fold, lineage, patient, paired flag (the RRID is in `context_text_provenance.json`) | frame 981 (377 single-cell training, 28 of them lymphoid or myeloid; 604 bulk-only), 343 paired, 27 validation, 28 test |
| `genes.parquet` | per GeneEffect gene: frame base `m_g`, 377-line mean, residual SD, selective and variable flags, `gene_text` key | 17,787 genes; decode panel 17,607; selective 4,323 (4,269 in the panel) |
| `cells/<ModelID>.npz` | PIE input, pseudo-bulk, spread against depth-expected values, per-cell QC, HVG cells and their shuffle | 434 lines, 573,532 cells, 19,277 source genes |
| `mean/` | mean channel in the default prior's space, PIE input of every line, `pie_listed.parquet` | 1,036 lines, 8,810 space genes |
| `summaries.parquet` | program fractions and means, cluster shares at 8 and 16 | 434 lines × 70 features |
| `gene_sources/` | `gene_text`, `esmc_600m`, `esm3_open`, `string_space`, `gene2vec` on the gene axis; `coverage.json`; `protein/` (sequences, raw ESM tables, structure status) | see below |
| `context_text/` | PIE `context_text` source keyed by ModelID; `context_text_provenance.json` beside it | 1,108 lines with text (42 reused from PIE, 1,066 embedded) |
| `pie/` | frozen-PIE re-score, reference pass, per-gene shards, basis, assembled deltas, `run.json` | see the results record |

Gene-source coverage of the 17,787 genes (selective share in brackets): `gene_text` 17,607 (0.988), ESM C 17,708
(0.993), ESM3-open 17,449 (0.978), `string_space` 17,503 (0.983), Gene2vec 15,922 (0.901). ESM3-open reads a
structure only when its sequence equals the UniProt one: 18,903 proteins matched, 213 AlphaFold models are fragmented,
48 have none and 38 differ in sequence, and those 299 have no ESM3 entry.

## Known limitations

- **Bridge quality.** The out-of-fold affine bridge from pseudo-bulk to bulk has a median per-line Pearson of 0.44
  (minimum −0.24); a line with cells is still read through its cells.
- **PIE's input coverage.** Some sources list few panel genes (minimum listed share 0.43, median 0.996); an unlisted
  gene takes the frame lines' mean.
- **Doublet scores.** 51 lines have under 100 cells and no Scrublet score.
- **Cellosaurus text.** 39 lines have none: 27 have no RRID in `Model.csv`; 11 have an age range PIE's renderer
  refuses (ACH-000206, 000974, 001054, 001301, 001302, 001555, 001556, 001557, 001819, 001977, 002026); EW8
  (ACH-000499, CVCL_V618) is not served by release 56.0. The 1,066 new vectors came through an OpenAI-compatible relay
  (`dapi.aivue.cn`), accepted because it reproduced PIE's released vectors of the 42 reused lines' texts (cosine
  minimum 0.99939, median 0.99999).
- **PIE has seen some of our lines.** PIE was trained on 42 of them, test line A549 among them (Tahoe-100M, chemical
  exposures only); its training evidence is served as released.
- **Protein table.** The UniProt table re-fetched for 2026_03 is not byte-equal to the one behind PIE's `esm2`.
