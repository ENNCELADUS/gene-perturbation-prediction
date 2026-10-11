# SL Screen Benchmark

**Status:** built on the H20 on 2026-10-11 (`data/sl_screens/benchmark/`). It is the SL
benchmark chosen in [blueprint §5](../01-blueprint.md#5-research-plan): the multi-line
combinatorial screens of Kebabci et al. 2026, namely Ito et al. 2021, Klingbeil et al. 2024 and
Harle et al. 2025, plus MiT4SL's A549 labels as a secondary external test. No SL model has
read it. Nothing here is an SL result.

## Role

Stage two of the [SL protocol](../04-sl-ranking-protocol.md) trains on the screens' training
lines and is scored per held-out line within one screen. One row is one (screen, line,
unordered gene pair) with the published scores, an SL label and the line's side of the
nested split. Labels are compared within one screen only: calls agree poorly across studies
([literature review §1](../02-literature-review.md#1-synthetic-lethality-benchmarks-and-their-holdout-settings)).

## Sources

Fetched to `data/sl_screens/raw/` on the H20 on 2026-10-11. Hashes are recorded by the
build in `manifest.json`.

| Screen | File | Source | SHA-256 (first 12) | Licence |
| --- | --- | --- | --- | --- |
| Ito et al. 2021, Nat Genet | `ito_2021/41588_2021_967_MOESM2_ESM.xlsx` (Supplementary Tables 4, 8, 10) | [Springer](https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-021-00967-z/MediaObjects/41588_2021_967_MOESM2_ESM.xlsx) | `c3dd8427d78b` | not verified |
| Klingbeil et al. 2024, Cancer Discov | `klingbeil_2024/cd-23-1529_supplementary_table_s{2,4,6}_suppst{2,4,6}.xlsx` | PMC open-access bucket, `https://pmc-oa-opendata.s3.amazonaws.com/PMC11609825.1/` | S2 `897016827bef`, S4 `d29cb5951e1a`, S6 `ff5c25c4e457` | not verified |
| Harle et al. 2025, Genome Biol | `harle_2025/13059_2025_3737_MOESM1_ESM.xlsx` (Tables S4, S5) | [Springer](https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-025-03737-w/MediaObjects/13059_2025_3737_MOESM1_ESM.xlsx) | `297ed1ee52a7` | CC BY 4.0 |
| MiT4SL (Tao et al. 2025) | `mit4sl/a549_bench.csv`, `mit4sl/Primenode.csv` | the repository's `data.zip` on [Google Drive](https://drive.google.com/drive/folders/1mVvm6yZTPhbjEXviYi5ZkXhMyqSrsj8C) | `7c5da2c0807d`, `267d39a6d285` | MIT |
| HGNC complete set | `hgnc/hgnc_complete_set.txt` | `https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt` | `64fcdfcdf9bc` | HGNC |

The H20 cannot reach Google Drive, so the two MiT4SL files were extracted on the Mac by HTTP
range requests from `data.zip` and copied with `scp`. They are the only files that did not
come straight from their host.

## Lines

`configs/benchmarks/sl_screen_lines.csv` maps each screen's column label to a DepMap ModelID.
It has 61 rows: Ito 11, Klingbeil 22, Harle 27 and MiT4SL 1. Harle's own `depMapID` column
agrees with it. Harle's C092 has no Model.csv row and is mapped to no ModelID. The build
assigns each line a split side and an input stratum from the
[226 split](cell-line-geneeffect-226.md), the [bulk extra lines](extra-bulk-lines-26q1.md) and
the [single-cell extra lines](single-cell-atlas-search.md):

- **Validation and test lines** are the 226's own lines, read from basal single cells.
- **`single_cell` training lines** are the 226's labelled training lines and the labelled
  single-cell extras whose basal cells are built (`labelled` in the single-cell membership file).
- **`bulk` training lines** are labelled bulk extras without single cells; only the
  complementary model reads them.
- **`unused` lines** carry a reason: no ModelID, no 26Q1 GeneEffect, or no expression source.

| Screen | Validation | Test | Training, single cells | Training, bulk | Unused |
| --- | --- | --- | --- | --- | --- |
| Ito | PA-TU-8988S | A549 | GI-1, IPC-298, MEL-JUSO | 5 | MeWo |
| Klingbeil | NCI-H1299 | MDA-MB-231, A549 | 11 | 7 | NCI-H1436 |
| Harle | NCI-H1299 | SK-MES-1, A549 | 17 | 1 | C092, CHL-1, MeWo, SK-MEL-28, SK-MEL-5, KP-1N |

Across the three screens, 29 distinct training lines have single cells and 13 more are
read through bulk RNA. A549 is a test line because the 226 split was rebuilt in place on
2026-10-11 to put it there. The benchmark was rebuilt the same day after the wave-one
single-cell extras were built: eight lines (nine screen rows) moved from bulk to single cells,
IPC-298 and MEL-JUSO (Ito), NCI-H209, NCI-H211, HPAF-II, MOLM-13 and THP-1 (Klingbeil), Capan-1
and HPAF-II (Harle), taking the single-cell stratum's labelled pairs from 41,926 to 63,864 and the
bulk stratum's from 62,491 to 40,553. Rows, labels and sides are otherwise unchanged; the earlier
build is kept on the H20 as `data/sl_screens/benchmark_2026-10-11_before_extras/`.

## Labels

- **Ito and Klingbeil** take Kebabci et al.'s rule (`05_annotate_gemini.ipynb` in
  [their code](https://github.com/cancergenetics/context_specific_paralog_SL)). A row is SL
  when its GEMINI sensitive-lethality score is at or above the screen's 90th percentile,
  taken over every scored (line, pair) row of the published table, and its LFC is below
  −0.51 (Ito) or −1.03 (Klingbeil). The GEMINI cut is 0.470 for Ito and 0.724 for
  Klingbeil. A row without an LFC stays unlabelled; none lacks one.
- **Klingbeil** screens domain constructs, and some genes under an old and a current symbol
  (KIAA1804 and MAP3K21). Symbols are mapped first. The 65 construct rows that become one
  gene paired with itself are dropped. A gene pair is then SL in a line when any of its
  constructs is, and keeps its strongest GEMINI, LFC and FDR, taking 56,345 construct rows
  to 52,242 pair rows.
- **Harle** takes the authors' Table S5 call, which Table S4's rule reproduces exactly: mean
  normalised GI < −0.5, FDR < 0.01, a Bassik hit, and neither gene single-depleted by both
  MAGeCK and BAGEL2. The call cannot make a pair with a single-depleted gene a hit, so those
  3,282 non-hit rows are unlabelled rather than negative. Kebabci et al.'s own Harle labels
  come from a GEMINI file they did not release.
- **Genes** are named as stage one names them. Each published symbol is first resolved to
  its current HGNC symbol: the approved symbol, else the one approved symbol it was
  previously, else the one it is an alias of. A gene then takes its 26Q1 GeneEffect symbol
  when exactly one GeneEffect symbol resolves to the same HGNC gene (DepMap keeps some
  symbols HGNC has since renamed, such as FAM83A), else its current HGNC symbol. Of 4,535
  published symbols, 89 are renamed (`renamed_symbols.csv`) and 116 are absent from the
  26Q1 GeneEffect columns; `both_in_geneeffect` marks the pairs stage one can serve.

**These labels follow Kebabci's rule but are not Kebabci's labels.** On their ten Ito lines
(MeWo left out), the build calls 1,511 SL in 47,315 points; they report 958 in 41,244. Their
LFC is Ito's Table 4: the mid-point method gives −0.515 on it, matching their −0.51. Their
GEMINI was recomputed from counts rather than taken from Table 8, and their pair set was
filtered; neither is released. On Klingbeil's S2 the mid-point method gives −1.28, not their
−1.03, which is kept as published.

## Counts

| Screen | Side | Lines | Rows | Labelled | SL |
| --- | --- | ---: | ---: | ---: | ---: |
| Ito | validation | 1 | 4,782 | 4,782 | 73 |
| Ito | test | 1 | 4,771 | 4,771 | 168 |
| Ito | training, single cells | 1 | 4,708 | 4,708 | 231 |
| Ito | training, bulk | 7 | 33,054 | 33,054 | 1,039 |
| Klingbeil | validation | 1 | 2,403 | 2,403 | 82 |
| Klingbeil | test | 2 | 4,740 | 4,740 | 274 |
| Klingbeil | training, single cells | 6 | 14,285 | 14,285 | 560 |
| Klingbeil | training, bulk | 12 | 28,431 | 28,431 | 1,542 |
| Harle | validation | 1 | 472 | 328 | 40 |
| Harle | test | 2 | 944 | 700 | 65 |
| Harle | training, single cells | 15 | 7,080 | 5,209 | 514 |
| Harle | training, bulk | 3 | 1,416 | 1,006 | 112 |

Per held-out line, each labelled row is compared with the same screen's single-cell training
lines. "Non-hit here, hit in training" counts pairs labelled negative in this line but SL in
at least one of them. "Hit here, non-hit in training" counts pairs SL in this line but
negative in at least one of them:

| Screen | Line | Side | Labelled | SL | Non-hit here, hit in training | Hit here, non-hit in training |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Ito | PA-TU-8988S | validation | 4,782 | 73 | 203 | 31 |
| Ito | A549 | test | 4,771 | 168 | 183 | 99 |
| Klingbeil | NCI-H1299 | validation | 2,403 | 82 | 248 | 24 |
| Klingbeil | MDA-MB-231 | test | 2,341 | 149 | 171 | 46 |
| Klingbeil | A549 | test | 2,399 | 125 | 209 | 33 |
| Harle | NCI-H1299 | validation | 328 | 40 | 35 | 2 |
| Harle | SK-MES-1 | test | 342 | 42 | 31 | 0 |
| Harle | A549 | test | 358 | 23 | 54 | 1 |

MiT4SL's A549 table has 1,054 pairs, 179 SL and 1,052 with both genes in GeneEffect. Its
labels come from SLKB, which merges several studies, so it is scored apart, beside MiT4SL's
published A549 result.

## Pitfalls

- **Line-varying calls are partly noise.** Of the pairs SL in some line, 700 of 720 (Ito),
  753 of 770 (Klingbeil) and 103 of 117 (Harle) are negative in another line. Cross-line
  agreement is 0.48 (Ito), 0.37 (Klingbeil) and 0.77 (Harle), so each held-out line keeps
  its continuous scores beside the call.
- **Every held-out pair is also measured in training lines.** The pair-identity control,
  the pair's hit rate in the same screen's training lines, has to be beaten.
- **Positives cluster in gene families.** The most frequent genes among positives are
  RPS6KB1 (3.5% of Ito's), RANBP2 (2.4% of Klingbeil's), and SEC23A, CNOT7, GDI1, SAR1A
  and CNOT8 (2.8% each of Harle's). Report per-gene shares before trusting AUPRC.
- **The libraries are paralog-led.** Most genes have one or two screened partners, so
  ranking within a query (line, lost gene) is secondary (blueprint §3.7).

## Outputs

On the H20 under `data/sl_screens/benchmark/`:

| File | Content |
| --- | --- |
| `pairs.parquet` | One row per (screen, line, pair): `screen`, `screen_line`, `model_id`, `gene_a`, `gene_b` (GeneEffect symbol where the gene has one, else current HGNC; `gene_a < gene_b`), `screen_gene_a`, `screen_gene_b` (as published, `\|`-joined when merged), `both_in_geneeffect`, `gemini`, `lfc`, `gi`, `fdr`, `n_constructs`, `label` (1, 0 or missing), `unlabelled_reason`, `split`, `stratum`, `line_reason` |
| `mit4sl_a549.parquet` | MiT4SL's A549 pairs and labels |
| `renamed_symbols.csv` | Published symbols named differently in the benchmark |
| `manifest.json` | Rule constants and GEMINI cuts, the Kebabci reproduction counts, per-screen counts by side, line sides and the SHA-256 of every input |

## Rebuild

```bash
R=data/sl_screens/raw
wget -U Mozilla/5.0 -O $R/ito_2021/41588_2021_967_MOESM2_ESM.xlsx "https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-021-00967-z/MediaObjects/41588_2021_967_MOESM2_ESM.xlsx"
for t in s2_suppst2 s4_suppst4 s6_suppst6; do wget -U Mozilla/5.0 -O $R/klingbeil_2024/cd-23-1529_supplementary_table_$t.xlsx "https://pmc-oa-opendata.s3.amazonaws.com/PMC11609825.1/cd-23-1529_supplementary_table_$t.xlsx"; done
wget -U Mozilla/5.0 -O $R/harle_2025/13059_2025_3737_MOESM1_ESM.xlsx "https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-025-03737-w/MediaObjects/13059_2025_3737_MOESM1_ESM.xlsx"
wget -O $R/hgnc/hgnc_complete_set.txt "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
uv run python -m src.data.prepare.build_sl_screen_benchmark \
    --raw data/sl_screens/raw \
    --lines configs/benchmarks/sl_screen_lines.csv \
    --split configs/benchmarks/cell_line_geneeffect_226_split.json \
    --extra-bulk configs/benchmarks/extra_bulk_lines_26Q1.json \
    --extra-single-cell configs/benchmarks/extra_single_cell_lines_26Q1.json \
    --gene-effect data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv \
    --hgnc data/sl_screens/raw/hgnc/hgnc_complete_set.txt \
    --out data/sl_screens/benchmark
```

The builder refuses an existing output directory. On the H20 it runs with
`.venv-tx1/bin/python`, which needs `openpyxl`, in about 30 seconds on the CPU.
