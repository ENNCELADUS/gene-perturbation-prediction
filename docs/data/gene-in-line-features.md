# Gene-in-line features

**Status:** built on the H20 on 2026-11-20 under `data/sl_gene_in_line/v1/` (`src/experiments/sl/gene_in_line.py`,
`configs/sl/gene_in_line.yaml`). `pie_change`: 86 lines, 64 components fitted on 200,000 vectors of the 53 SL training
lines, 275 MB, about 2 minutes on the CPU (code `d4d1784`). `tx1_gene`: 86 lines, at most 256 cells each, 11 GB, 21 to
41 s for each line on one H20, about 13 minutes on four (code `f6e242f`); a line has a value for about 10,000 to 13,000
of the 25,861 names of the axis. Run record: [`results/sl_line_term_seed0/summary.html`](../../results/sl_line_term_seed0/summary.html).
On 2026-11-23 the `tx1_gene` file of ACH-000696 was encoded again (code `63f5010`, see the known limits).
The seven CPU caches from `wide_expression` to `depmap_copy_number` were added on 2026-11-24 (spec
`docs/specs/2026-11-24-context-specific-sl-architecture-design.md` §5.4). They were built on the H20 on 2026-11-24 (code `b2de0c9`), each for the 86 labelled lines. Lines with the
source / genes with a value in at least one line: `wide_expression` 86 / 18,905; `cell_spread` 86 / 18,905;
`paralog_buffer` 86 / 12,550; `complex_buffer` 86 / 4,848; `partner_low` 86 / 14,604; `depmap_mutation` 79 / 17,851;
`depmap_copy_number` 65 / 18,416. A DepMap column without a gene symbol (the row number, a bare Ensembl ID) is left out.

**Two cache roots since 2026-11-25.** `data/sl_gene_in_line/v2` is the root that the code writes now; `v1` keeps
the caches as the runs up to 2026-11-24 read them. The difference is one rule. Under `v1`, a gene's 10th percentile
was over all SL training lines with a cell file, and a gene that one of those lines does not measure had no
percentile: its silenced flag was 0 in every line and it did not count as a low partner (5,032 of the 18,905 genes
with a value could have the flag). Under `v2` the percentile is over the training lines that measure the gene. The
silenced flag of `wide_expression` and the value of `partner_low` differ between the roots; every other value is
the same. `tx1_gene` and `pie_change` of `v2` are links to `v1`.

Since 2026-11-25 the CPU steps from `wide_expression` on also write the split file's bulk-only lines with a labelled
row (37 lines; SL protocol §21). Such a line has no cell file. Its `wide_expression` is its DepMap bulk RNA
(`bulk_expression` in the config, `log2(TPM + 1)`), mapped onto the pseudo-bulk gene by gene: a least-squares line
for each source gene, fitted on the SL split's training lines that have a cell file and bulk RNA; a mapped value below
0 is 0, and a gene without a fit (not in the bulk matrix, or one bulk value in every fit line) is absent. The buffer
caches read that mapped expression; `cell_spread` is absent for these lines. The 10th percentiles stay those of the
SL training lines with a cell file, so the files of the other lines do not change. `report.json` of
`wide_expression` and of the buffer caches also gives `lines_from_bulk` and `bulk_map_lines` (the lines of the fit).

**Role.** The line term of the gene-knowledge, cell-set SL model reads one vector for each (line, gene) from each
cache (`line_term.gene_in_line` of an SL embedding config; plan `docs/specs/2026-11-20-line-term-and-direct-target-plan.md`
§4). Each vector tells how the gene is in that line.

| Cache | Vector of gene `g` in line `c` | Width | Step |
| --- | --- | --- | --- |
| `tx1_gene/` | the mean over the line's cells of Tx1-3B's contextual embedding of `g` (the transformer's output at `g`'s token), then the share of the cells whose input sequence holds `g` | 2,561 | `tx1_gene`, one GPU for each shard |
| `pie_change/` | PIE's predicted change of the line's profile after the loss of `g` (`<pie.root>/pie/change/<G>.npy`, from `hpc/run.sh sl-pie-pass`), in 64 components | 64 | `pie_change`, CPU |
| `wide_expression/` | the line's pseudo-bulk `log1p(CP10K)` of `g`; a silenced flag (1 at or below `g`'s 10th percentile over the SL training lines, else 0) | 2 | `wide_expression`, CPU |
| `cell_spread/` | the share of the line's cells that detect `g`; the SD of `g` minus its depth-expected SD | 2 | `cell_spread`, CPU |
| `paralog_buffer/` | the minimum and the sum of the wide expression of `g`'s paralogs, and that of its closest paralog | 3 | `paralog_buffer`, CPU |
| `complex_buffer/` | the mean wide expression of `g`'s complex co-members | 1 | `complex_buffer`, CPU |
| `partner_low/` | the share of `g`'s paralogs and co-members at or below their 10th percentile over the SL training lines | 1 | `partner_low`, CPU |
| `depmap_mutation/` | DepMap's damaging call of `g`; its hotspot call (each 0, 1 or 2) | 2 | `depmap_mutation`, CPU |
| `depmap_copy_number/` | DepMap's `log2(relative copy number + 1)` of `g`; a deep-loss flag (1 when the log2 copy-number ratio is below −0.8) | 2 | `depmap_copy_number`, CPU |

**Format.** Each cache holds `genes.json` (a JSON list of gene symbols: the sorted `gene_a` and `gene_b` of
`data/sl_studies/benchmark_v2/labels.parquet`) and `<ModelID>.npy` (genes × width, float16, in the order of
`genes.json`). A row of NaN is a gene without a value in that line. `tx1_gene/report.json` gives, for each line, its
cells and the number of genes with a value; `pie_change/report.json` gives the counts, the sample size and the
labelled lines that PIE does not have. The `report.json` of each CPU cache from `wide_expression` on gives the number of
lines, of lines with the source and of genes with a value in at least one line.

**Sources.**

- `tx1_gene`: the raw-UMI h5ad of each SL split line with a labelled row (`paths.basal_raw_umi` of the joint config),
  at most 256 cells (`tx1_cell_indices`, seed 0), Tx1-3B (`paths.tx1_model_dir`) with a sequence of at most 2,048
  genes. A gene of the axis takes the first `var` row of its symbol and the token of that row's Ensembl ID. A token
  twice in one cell's sequence stops the step, and so does a value that float16 cannot hold.
- `pie_change`: the lines of `<pie.root>/lines.parquet` (the 86 screened lines). A gene's vector in a line is its row
  of PIE's change file (6,523 panel genes); the file name is the upper-case symbol.
- `wide_expression`, `cell_spread`: `<ladder>/cells/<ModelID>.npz` and `<ladder>/cells/genes.json` (19,277 source
  genes, no bridge). A gene of the axis takes the first source gene of its upper-case symbol.
- `paralog_buffer`, `complex_buffer`, `partner_low`: the partner columns of `knowledge_features`
  (`src/linear_model/gene_features.py`) on the pseudo-bulk of the lines with a cell file. The paralog and complex tables
  are the reference tables of `paths.reference` in the linear-model config (`linear_model` key). The paralogs are in
  the table's order; the closest paralog is the first one that the source holds.
- `depmap_mutation`: `OmicsSomaticMutationsMatrixDamaging.csv` and `OmicsSomaticMutationsMatrixHotspot.csv`, the row
  with `IsDefaultEntryForModel == "Yes"` of each line. The gene columns `SYMBOL (entrez)` give the upper-case symbol;
  when two columns give one symbol, the first one is used. A gene of the damaging matrix that the hotspot matrix does
  not list has hotspot 0.
- `depmap_copy_number`: `PortalOmicsCNGeneLog2.csv` (first column the ModelID), with the same gene columns. The deep-loss
  flag is 1 when `2 ** value − 1 < 2 ** −0.8`.

**Fit rule.** The PIE components (centre and basis) are fitted on a seeded sample of at most 200,000 (line, gene)
vectors of the lines with role `train` only. The 10th percentiles of `wide_expression` and `partner_low` are fitted on
the SL training lines (the `train` side of the SL split) that have a cell file and measure the gene. The model centres and scales each column on the SL training lines when
it reads a cache (`read_in_line`).

**Known limits.**

- **One line was encoded again.** ACH-000696, an SL training line, lists its Ensembl IDs with a version suffix
  (`ENSG00000223972.5`), the only one of the 455 raw files to do so. Tx1-3B's vocabulary holds no such ID and gives a
  missing ID the padding token, so the first caches of this line came from padding tokens only. Since code
  `63f5010`, `read_basal_source` removes the suffix. On 2026-11-23 the line was encoded again in `tx1_gene/` (11,863
  of 25,861 genes with a value; 0 before) and in `data/tx1_basal_embeddings/v2_all/{real,shuffled}` (mean spread
  across cells 0.80 and 0.61; 0.24 and 0.05 before). The same code gives the cached `v2_all/real` values of A549
  exactly. The runs of 2026-11-20 read the old values. `data/tx1_basal_embeddings/v1` (the joint GeneEffect track)
  keeps its old values of this line: its other lines come from an older encoder that today's code does not give.
- A gene has a Tx1-3B value only in the cells whose 2,048-gene sequence holds it. A gene that no cell holds, a gene
  whose Ensembl ID is not in Tx1-3B's vocabulary and a gene that the line's source does not list are NaN rows.
- PIE was trained on 42 of the project's lines. A549 is one of them, and it is a training line of the SL split. Which
  SL validation or test lines are among the 42 is not recorded.
- `pie_change` stops when a lost gene of PIE's `genes.parquet` has no change file (an incomplete PIE pass). An axis
  gene that is not a lost gene is a NaN row in every line. A labelled line that PIE does not have gets no file.
- `tx1_gene` skips a line whose file exists. `pie_change` does nothing when every line's file exists; when one is
  missing, it fits the components again and writes every line, so that all lines are in one basis (a new fit can
  change the sign of a component).
- A CPU cache from `wide_expression` on has a file for every SL split line with a labelled row. A line without the
  source (no cell file; for DepMap, no row in the matrix, for example `CVCL_*` and `SL-*` lines) gets a file of NaN
  rows. A row is NaN when one of its values is missing. Each call writes every line again.
- The fit of the PIE components holds about 11 GB in memory (the float32 sample of 200,000 x 6,523 and its centred
  copy).
