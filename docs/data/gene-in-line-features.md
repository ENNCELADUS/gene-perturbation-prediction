# Gene-in-line features

**Status:** built on the H20 on 2026-11-20 under `data/sl_gene_in_line/v1/` (`src/experiments/sl/gene_in_line.py`,
`configs/sl/gene_in_line.yaml`). `pie_change`: 86 lines, 64 components fitted on 200,000 vectors of the 53 SL training
lines, 275 MB, about 2 minutes on the CPU (code `15a754f`). `tx1_gene`: 86 lines, at most 256 cells each, 11 GB, 21 to
41 s for each line on one H20, about 13 minutes on four (code `d9d2950`); a line has a value for about 10,000 to 13,000
of the 25,861 names of the axis. Run record: [`results/sl_line_term_seed0/summary.html`](../../results/sl_line_term_seed0/summary.html).
On 2026-11-23 the `tx1_gene` file of ACH-000696 was encoded again (code `842c283`, see the known limits).

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

- **One line was encoded again.** ACH-000696, an SL training line, lists its Ensembl IDs with a version suffix
  (`ENSG00000223972.5`), the only one of the 455 raw files to do so. Tx1-3B's vocabulary holds no such ID and gives a
  missing ID the padding token, so the first caches of this line came from padding tokens only. Since code
  `842c283`, `read_basal_source` removes the suffix. On 2026-11-23 the line was encoded again in `tx1_gene/` (11,863
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
- The fit of the PIE components holds about 11 GB in memory (the float32 sample of 200,000 x 6,523 and its centred
  copy).
