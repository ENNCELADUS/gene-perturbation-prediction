# Single-Cell Extra Lines, DepMap 26Q1

**Status:** Labelled DepMap lines outside the [226](cell-line-geneeffect-226.md) whose public basal
single cells join the prepared root as training-side lines. The 2026-10-07 search below found the
MIX-seq candidates; the 2026-10-11 survey (local note `docs/notes/Basal single cell DepMap lines.md`)
found 131 more labelled lines, taken in two waves: wave 1 reads deposited count matrices, wave 2
re-counts deposits that hold only processed values. The identity map, the source choice, the
membership file and the builder below are in place; the membership counts are those of the last
`build`. Nothing here is SL evidence.

## Identity map

`configs/benchmarks/single_cell_atlas_candidates.csv` has one row per (source, identifier), where
`identifier` is the source's own spelling of a line: a per-cell `DepMap_ID`, a Cellosaurus RRID, a
GEO sample field, a label-table value, a hashtag-to-line value or a stated pool member. Every label a
source can produce has a row, lines of the 226 included, so the builder can resolve or drop each cell.

| Column | Meaning |
| --- | --- |
| `source`, `accession` | Source key (lower-case accession) and its public accession |
| `model_id`, `identifier_kind` | The ModelID and how it was found: `model_id`, `rrid`, `ccle_name`, `name` (Model.csv `StrippedCellLineName` or `CellLineName`, case and punctuation ignored) or `synonym` (an exact Cellosaurus name or synonym whose accession is a Model.csv RRID) |
| `cells`, `raw_counts`, `condition`, `size_gb` | Expected basal cells, whether the deposit holds integer counts, the basal arm(s), the source's download size |
| `condition_class` | `basal_2d` (untreated or vehicle 2D), `control_guide` (non-targeting or safe-targeting guide), `cutting_control` (Cas9 cut at an unexpressed locus), `other` |
| `review` | `ok`, a flag (`ambiguous`, `no_match`, `problematic`, `derived`) or `dropped: <reason>` |
| `wave` | `1` or `2` when the source is ingested, empty when not |
| `note` | Evidence: Cellosaurus accession, GSM ids, the reviewer's reason |

`build_extra_single_cell_lines propose` fills the rows without a review from the source's own ModelID
or RRID, else a unique Model.csv name, else a Cellosaurus synonym, and flags the rest: two Model.csv
lines share the name (`ambiguous`), nothing matches (`no_match`), Cellosaurus marks the DepMap line
problematic (`problematic`), or the name is a Cellosaurus child of a DepMap line (`derived`). Flagged
rows are settled by hand once (`ok` with a ModelID, or `dropped: <reason>`). Nothing joins lines by
name at run time: the builder resolves labels only through `ok` rows, and a label absent from the map
stops it.

## Source choice and membership

Each line takes its cells from one source: among `ok` rows of an ingested wave, the best
`condition_class`, then integer counts, then most cells, then the source key. The membership file
`configs/benchmarks/extra_single_cell_lines_26Q1.json` (schema `extra-single-cell-lines-26q1-v2`,
`src/data/extra_lines.py:load_extra_single_cell_lines`) sorts the chosen lines outside the 226:
`labelled` (26Q1 GeneEffect row and built cells, with source, identifiers, condition, patient,
lineage, cells and arms), `pending` (labelled, source not built or no cell after QC), `unlabelled`
(no GeneEffect row; never built) and `excluded` (shares a PatientID with a validation or test line).
Preparation and the SL benchmark read only `labelled`. `inputs` records the SHA-256 of every input.

## Builder

`src/data/prepare/build_extra_single_cell_sources.py` reads the tracked source spec
`configs/data/extra_single_cell_sources_26Q1.json` (per source: files and their URLs, archives to
unpack, gene id kind, basal arms, libraries). Per library:

- **Reader:** 10x mtx or h5 (expression and antibody-capture features split; header rows and a
  mixed-species prefix such as `hg19_` handled), a dense count table (Drop-seq DGE, BD Rhapsody
  molecules per cell, with comment lines skipped), an h5ad or a velocyto loom (`X`, `.raw` or a
  layer), or the Tahoe-100M DMSO shards. Archives (tar, zip, gzip) are unpacked first. Counts must
  be integers; a deposit clipped at a small cap passes that check, so it is excluded at review.
- **Cells:** the deposit's called cells, or the inflection of the barcode-rank curve for unfiltered
  matrices (the knee UMI is reported).
- **Line per cell:** the whole library, the authors' per-cell table (with keep filters and an arm
  column), a pattern on the cell id (a line prefix such as `SUM149PT_`, an aggregate's `-1` suffix,
  an arm suffix such as `_C.R1`), hashtags (a singlet's top tag needs at least `min_count` and
  `min_ratio` times the second),
  or, for a pool without labels, k-means clusters assigned to the member whose DepMap bulk RNA deviation
  correlates best, when it beats the second best by `min_margin` (every cluster's correlations and
  margin are reported).
- **Kept:** cells of the lines the choice takes from this source, in a listed basal arm, with at least
  `min_genes` (200) detected genes and at most `max_mito_fraction` (25%) mitochondrial UMI. Cells of a
  226 line are never written; each dropped cell is counted by reason. There is no minimum number of
  cells per line.

A line's libraries are stacked over the genes they all measure, mapped to the Tx1 vocabulary through
Tahoe's gene metadata (unique symbols only). It writes `<basal_raw_umi>/<ModelID>.h5ad` (obs: line,
patient, source, identifier, arm, library) and, last, `qc/<source>.json`; a source with a report is
skipped, an h5ad is never overwritten, and an h5ad without its report stops the run.

## 2026-10-07 search

Only MIX-seq adds a material number of labelled lines. Tahoe-100M and Kinker et al. 2020 are
already almost fully used: their lines outside the 226 have no 26Q1 GeneEffect row, except one
Tahoe line.

| Source | Accession | Raw counts | Condition | Cells per line (median, range) | Identifier | Size | Licence | Resolved | In the 226 | Outside the 226 | Of those labelled |
| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Tahoe-100M | [tahoebio/Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) | yes | `DMSO_TF` vehicle, plate-matched | 43,500 (345–139,224) | RRID | 428.8 GB | CC0-1.0 | 49 of 50 | 38 | 11 | 1 |
| Kinker et al. 2020 | [SCP542](https://singlecell.broadinstitute.org/single_cell/study/SCP542); GEO GSE157220 | yes (SCP542 only) | untreated, pooled | 227 (56–1,990) | CCLE name | 9.3 GB | none stated | 193 of 198 | 162 | 31 | 0 |
| MIX-seq (McFarland et al. 2020) | [figshare 10298696](https://figshare.com/articles/dataset/MIX-seq_data/10298696) | yes | DMSO 3–48 h, untreated 6 h, or control sgRNA | 123 (4–3,453) | ModelID | 2.26 GB | CC BY 4.0 | 205 of 213 | 75 | 130 | 100 |
| sci-Plex 3 (Srivatsan et al. 2020) | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/00109df5-7810-4542-8db5-2288c46e0424); GEO GSE139944 | yes | vehicle, 24 h | 3,359 (3,287–6,358) | RRID | 1.86 GB | CC BY 4.0 | 3 of 3 | 3 | 0 | 0 |
| SARS-CoV-2 cell lines (Wyler et al. 2021) | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/d0e9c47b-4ce7-4f84-b182-eddcfa0b2658); GEO GSE148729 | yes | mock infection, 4–36 h | 16,365 (16,218–16,512) | RRID | 11.4 GB | CC BY 4.0 | 2 of 2 | 1 | 1 | 0 |

"Labelled" means the ModelID has a row in 26Q1 `CRISPRGeneEffect.csv`. Cell counts are exact
counts from each source's metadata (definitions below); sizes are each source's total download.
Across sources, 355 distinct lines resolve, 157 lie outside the 226, and 101 of those are
labelled. Two lines outside the 226 share a `PatientID` with a held-out line: PA-TU-8988T
(`ACH-000023`, MIX-seq, labelled) with PA-TU-8988S (validation), and SW 480 (`ACH-000842`, Tahoe,
unlabelled) with SW 620 (test). The builder excludes both, leaving 100 labelled candidates: 99
from MIX-seq and C3A (`ACH-001021`, Tahoe's HepG2/C3A, which shares a patient with the training
line HepG2). The builder printed 100 labelled, 55 unlabelled and 200 excluded (198 in the 226
and the two patient exclusions); its printed counts are the authority, and this card follows them.

### Per-source notes

- **Tahoe-100M.** Cells carry the Cellosaurus ID (`cell_line` in `metadata/obs_metadata.parquet`).
  Cells per line count `drug == "DMSO_TF"` with `pass_filter == "full"`, which matches the
  `dmso_cells` already recorded for the 38 lines in the 226. Tahoe's own `Cell_ID_DepMap` agrees
  with the RRID resolution for all 49 lines. hTERT-HPNE (`CVCL_C466`) is not in Model.csv. The
  expression shards are 337.6 GB of the 428.8 GB repository.
- **Kinker et al. 2020.** The GEO pool table (`GSE157220_Pool_composition.xlsx`) lists 198
  lines by CCLE name with post-QC cell counts; those are the rows here. GEO holds only CPM; the raw
  UMI matrix (`UMIcount_data.txt`, 3.47 GB, pre-QC, 207 lines) is on SCP542 and needs a sign-in
  to download, so the 9 lines beyond the 198 were not listed. Five CCLE names have no Model.csv
  row (`93VU`, `JHU006`, `SCC47`, `SCC90` upper aerodigestive tract, `NCIH2077_LUNG`). The 31
  resolved lines outside the 226 have no GeneEffect row.
- **MIX-seq.** Each experiment zip holds a 10x Cell Ranger integer matrix (3′ v2 for
  experiments 1–3, v3 for 5 and 10; the README calls the values read counts) and a per-cell
  `classifications.csv` with the SNP-assigned CCLE name (`singlet_ID`) and `DepMap_ID`. Cells per
  line count `cell_quality == "normal"` in the vehicle arms: DMSO 6 h and 24 h plus untreated 6 h
  (experiment 1, 24 lines), DMSO 6 h and 24 h (experiment 3, 99 lines), DMSO 24 h (experiment 10,
  98 lines), and the DMSO-hashed cells of the trametinib time course (experiment 5, 24 lines).
  37 lines appear only in experiment 2, whose controls are Cas9 lines carrying a control sgRNA
  (sgLACZ, sgOR2J2) for 72–96 h; for those rows `cells` counts the control-sgRNA cells and
  `condition` says so. Among the 100 labelled lines outside the 226, 14 have only control-sgRNA
  cells, and the median is 100 cells per line (11–845; 51 lines have at least 100). Experiments 3
  and 10 were superloaded. Of the 213 pooled lines, 8 have no QC-passing cell in any control arm
  and so no row; two of those (`COLO699_LUNG`, `S117_SOFT_TISSUE`) also have no Model.csv row.
- **sci-Plex 3.** CELLxGENE's curation carries the Cellosaurus ID in `tissue_ontology_term_id`
  and integer-valued counts in `X`. Cells per line count `vehicle == True` (product name
  `Vehicle`, dose 0). All three lines (A549, K562, MCF7) are in the 226. The GEO deposit
  (`GSE139944_RAW.tar`, 9.2 GB) was not opened; CELLxGENE is the route that supplies RRIDs.
- **Wyler et al. 2021.** Drop-seq; CELLxGENE keeps the counts in `raw/X`. Cells per line count
  mock-infected cells (no virus strain). NCI-H1299 is in the 226; Calu-3 (`ACH-000392`) is not and
  has no GeneEffect row.

### Pitfalls found

- **MIX-seq Supplementary Table 1.** Its `DepMap ID` column is not aligned with its `CCLE Name`
  column (`22RV1_PROSTATE` is listed beside `ACH-000008`; Model.csv gives `ACH-000956`). None of its
  504 pairs agrees with Model.csv. Pool membership is read from the CCLE names and the
  `experiments` column, which agree with the cells; the identifier is the per-cell `DepMap_ID`,
  which agrees with Model.csv `CCLEName` for every resolved line but one.
- **`ACH-000921`.** MIX-seq assigns cells to `NCIH1339_LUNG` with `DepMap_ID` `ACH-000921`;
  26Q1 names `ACH-000921` NCI-H157-DM and has no `NCIH1339_LUNG`. The row is kept under the ModelID
  the source gives, because the SNP reference that assigned the cells is that model's own
  sequencing.

The 2026-10-07 rows were resolved only from identifiers the source itself gives (a ModelID, an RRID
that one Model.csv row carries, or an exact CCLE name); they carry `review = ok`. Rows added from
informal names go through the identity map's review above.

## Sources found but not usable

| Source | Accession | Lines | Why not in the candidate file |
| --- | --- | ---: | --- |
| Breast cancer cell line atlas (Gambardella et al. 2022) | GEO GSE173634; [figshare 15022698](https://figshare.com/articles/dataset/Single_Cell_Breast_Cancer_cell-line_Atlas/15022698) (CC BY 4.0, 1.93 GB, raw UMI, Drop-seq, untreated) | 32 | Informal names only (`HS578T`, `MDAMB361`; GEO `cell line: AU565`). The 226 already takes 14 lines from it through the earlier atlas manifest. |
| Single-cell atlas of human cell lines (Zhu et al. 2023, Nat Commun 14, 8170) | [OMIX005191](https://ngdc.cncb.ac.cn/omix/release/OMIX005191) (open access, 56 files, about 1.5 GB of wide count tables) | 42 (40 cancer, 2 normal) | Informal names only (`K-562`, `HT29`, `HNSCCUM-02T`); hypoxia and lineage-tracing arms for five lines. The 226 already takes 13 lines from it. |
| Controlled-heterogeneity benchmark (Sci Data 2024) | GEO GSE243665 | several lung lines | Files are named by informal names or ATCC catalog numbers (`A549`, `CCL-185-IG`, `CRL5868`, `DV90`). HCC-78, a labelled line outside the 226, is among them (2026-10-11 survey). |
| Pathway Perturb-seq (Jiang et al. 2025, Nat Cell Biol) | [Zenodo 10520190](https://zenodo.org/records/10520190) (CC BY 4.0, 20.5 GB Seurat objects) | 6 | Released as one object per stimulation (IFN-β, IFN-γ, TNF-α, TGF-β1, insulin), so its controls come from stimulation screens; informal names. Not opened. |
| X-Atlas/Orion and X-Atlas/Pisces (Xaira) | [figshare+ 29190726](https://doi.org/10.25452/figshare.plus.29190726) (about 520 GB); [X-Atlas-Pisces](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Pisces) (CC BY-NC-SA 4.0) | HCT116, HEK293T, HepG2, Jurkat | Informal names; HCT116, HepG2 and Jurkat are already in the 226 and HEK293T is not a cancer line. Pisces files were not yet released when checked. |
| scPerturb | scperturb.org | — | Repackages published perturbation sets, MIX-seq and sci-Plex among them; no further multi-line basal set was identified in it. |
| Arc Virtual Cell Atlas / scBaseCount | arcinstitute.org/tools/virtualcellatlas | — | Reprocessed SRA data with agent-extracted metadata; no ModelID, RRID or CCLE-name field. It does carry a free-text `cell_line` field per experiment, which the 2026-10-11 survey used to find studies (about one match in five was misannotated); each study is ingested from its own deposit. |

CELLxGENE Discover was searched through its curation API: of 2,238 public datasets, five carry a
Cellosaurus cell-line tissue (the three sci-Plex 3 lines and the two SARS-CoV-2 lines above).

## Rebuild

New sources enter as map rows (with an empty `review`) and spec entries; then, on the Mac,
`propose` fills the rows and the flagged ones are reviewed by hand:

```bash
uv run python -m src.data.prepare.build_extra_single_cell_lines propose \
    --candidates configs/benchmarks/single_cell_atlas_candidates.csv \
    --model data/sl_dependency_v0/raw/depmap/Model.csv
```

On the H20, the builder and `build_extra_single_cell_lines build` write the h5ad, the QC reports and
the membership file, and preparation grows the prepared root (commands in `hpc/README.md`, "Extra
single-cell lines"). Re-preparing a line means deleting its h5ad, `lines/<ModelID>.npz`, Tx1 cache
directory and pseudo-bulk row.

## Artifacts

- `configs/benchmarks/single_cell_atlas_candidates.csv`: the identity map, one row per (source,
  identifier); columns above.
- `configs/data/extra_single_cell_sources_26Q1.json`: the source spec, per wave.
- `configs/benchmarks/extra_single_cell_lines_26Q1.json`: the membership file.
- H20: `data/sl_dependency_v0/raw/extra_single_cell/<source>/` (deposits, `SHA256SUMS`) and
  `data/sl_dependency_v0/processed/basal_raw_umi/` (one h5ad per prepared line, the 226 hard-linked
  from their builders' outputs, and `qc/<source>.json`).

## Use

Most extras come from their own study, so a line's lab and chemistry are confounded with the line;
the panels and pools (MIX-seq, the gastric, neuroblastoma, SUM-breast and blood-cancer panels)
give several lines inside one study. The prepared extras are training-side data only: validation and
test lines are unchanged, and whether and how a model uses the extras is decided by its own run.
