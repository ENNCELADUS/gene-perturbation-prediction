# Gene-in-line features

**Status:** code ready on 2026-11-20 (`src/experiments/sl/gene_in_line.py`, `configs/sl/gene_in_line.yaml`); the
caches are not built yet. They go under `data/sl_gene_in_line/v1/` on the H20.

**Role.** The line term of the gene-knowledge, cell-set SL model reads one vector for each (line, gene) from each
cache (`line_term.gene_in_line` of an SL embedding config; plan `docs/specs/2026-11-20-line-term-and-direct-target-plan.md`
§4). Each vector tells how the gene is in that line.

| Cache | Vector of gene `g` in line `c` | Width | Step |
| --- | --- | --- | --- |
| `tx1_gene/` | the mean over the line's cells of Tx1-3B's contextual embedding of `g` (the transformer's output at `g`'s token), then the share of the cells whose input sequence holds `g` | 2,561 | `tx1_gene`, one GPU for each shard |
| `pie_change/` | PIE's predicted change of the line's profile after the loss of `g` (`<pie.root>/pie/change/<G>.npy`, from `hpc/run.sh sl-pie-pass`), in 64 components | 64 | `pie_change`, CPU |

**Format.** Each cache holds `genes.json` (a JSON list of gene symbols: the sorted `gene_a` and `gene_b` of
`data/sl_studies/benchmark_v2/labels.parquet`) and `<ModelID>.npy` (genes × width, float16, in the order of
`genes.json`). A row of NaN is a gene without a value in that line. `tx1_gene/report.json` gives, for each line, its
cells and the number of genes with a value; `pie_change/report.json` gives the counts, the sample size and the
labelled lines that PIE does not have.

**Sources.**

- `tx1_gene`: the raw-UMI h5ad of each SL split line with a labelled row (`paths.basal_raw_umi` of the joint config),
  at most 256 cells (`tx1_cell_indices`, seed 0), Tx1-3B (`paths.tx1_model_dir`) with a sequence of at most 2,048
  genes. A gene of the axis takes the first `var` row of its symbol and the token of that row's Ensembl ID. A token
  twice in one cell's sequence stops the step, and so does a value that float16 cannot hold.
- `pie_change`: the lines of `<pie.root>/lines.parquet` (the 86 screened lines). A gene's vector in a line is its row
  of PIE's change file (6,523 panel genes); the file name is the upper-case symbol.

**Fit rule.** The PIE components (centre and basis) are fitted on a seeded sample of at most 200,000 (line, gene)
vectors of the lines with role `train` only. The model centres and scales each column on the SL training lines when
it reads a cache (`read_in_line`).

**Known limits.**

- A gene has a Tx1-3B value only in the cells whose 2,048-gene sequence holds it. A gene that no cell holds, a gene
  whose Ensembl ID is not in Tx1-3B's vocabulary and a gene that the line's source does not list are NaN rows.
- PIE was trained on 42 of the project's lines. A549 is one of them, and it is a training line of the SL split. Which
  SL validation or test lines are among the 42 is not recorded.
- `pie_change` stops when a lost gene of PIE's `genes.parquet` has no change file (an incomplete PIE pass). An axis
  gene that is not a lost gene is a NaN row in every line. A labelled line that PIE does not have gets no file.
- The fit of the PIE components holds about 11 GB in memory (the float32 sample of 200,000 x 6,523 and its centred
  copy).
