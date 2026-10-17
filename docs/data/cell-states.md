# Cell-state telemetry inputs

**Status:** code built 2026-10-17 (Wave 1B of the [stage-one plan](../specs/2026-10-17-pie-stage-one-plan.md), Task 8 of
the [PIE stage-one design](../specs/2026-10-16-pie-geneeffect-ladder-design.md)); not yet run on the H20. Definitions
are in [protocol §17](../03-geneeffect-protocol.md#17-cell-states); the run record will be
`results/state_telemetry_seed0/README.md`.

## Role

Frozen cell-state vocabularies fitted on the single-cell training lines, their memberships of every cell of the
training and validation lines, the state inputs they give the frozen PIE with their controls, and the frozen PIE's
deltas on those inputs. Task 8's report reads them; Task 9 reads the vocabularies people choose from it.

## Sources

| Input | Source |
| --- | --- |
| Lines, cells, PIE's input, Tx1 cells, the frozen PIE's main pass | the [stage-one inputs](ladder-inputs.md) |
| Raw cells | `<basal_raw_umi>/<ModelID>.h5ad` (per-cell PIE input, cNMF, scVI, reference-program scores) |
| Second deposits | the noise diagnosis's `deposit.second_source_root` (99 lines; `configs/data/second_source_lines_26Q1.json`) |
| Chemistry per deposit | `configs/benchmarks/single_cell_source_chemistry.csv` (one reviewed row per deposit; `unknown` where no record names the kit) |
| Kinker's 12 recurrent programs | `configs/context_prior/reference/kinker_programs.tsv` (Kinker et al. 2020, Table S4; provenance in `kinker_programs_provenance.json`) |
| Gavish's 41 meta-programs, hallmarks, Tirosh's cycle | `configs/context_prior/reference/` (`provenance.json`) |

Kinker's programs were derived from CCLE pools that hold some of our lines, validation and test lines among them; they
only name states in the report and never enter a model. Their symbols are of 2020 (IL8, CYR61, KIAA0101, …): a member
whose symbol is not a panel gene is not scored, and the report gives the share of each group's members the sources list.

## Contents

Under `data/geneeffect_joint/v2/ladder/` on the H20:

| Path | Contents |
| --- | --- |
| `states/manifest.json` | settings (a root written under others is refused) and each step's counts |
| `states/fit/sample_seed<s>.parquet` | the cells each training line contributes to a fit (500 per line), with chemistry and S and G2/M scores |
| `states/fit/plan.json` | what `state_programs` reads (lines, genes, settings, h5ad paths) |
| `states/fit/seed<s>/<vocabulary>.npz` | frozen mixtures: weights, means, covariances, centring, scale, principal components, state names |
| `states/programs/seed<s>/` | per-line cNMF results (`cnmf/`), the reference (`reference.npz`, `reference.tsv` in starCAT's format, `reference.json`), usages of every cell (`usages/`, `usages_second/`), `manifest.json` (spectra normalisation, background program, usage-to-weight rule, starCAT's own fit on one line beside ours) |
| `states/scvi/seed<s>/` | the scVI model, latent means of every cell (`latent/`, `latent_second/`), `manifest.json` |
| `states/memberships/<ModelID>.npz`, `states/second/` | memberships of every cell, keyed `<vocabulary>@seed<s>`; second deposits' cells (`second/cells/`) and memberships |
| `states/lines/<ModelID>.npz` | per set: state inputs, masses, effective counts, state names and merges; per vocabulary: state-weighted reference-program scores and QC shifts |
| `states/<set>/inputs.parquet`, `masses.parquet` | the set's contexts `<ModelID>\|<set>\|<state>` (each line's `\|whole` context after its states) × the decode panel, and their masses |
| `pie_states/<set>/` | after `assemble_states`: `dH`, `dE`, `own_h` (float16), `own_dp`, `own_x` (float32), genes × contexts × width, centred on the main pass's frame means; `contexts.parquet`, `index.json`; `pie_states/run.json` |

Sets: every vocabulary at fitting seed 0 (`cycle`, `programs`, `scvi_centred_k{4,6,8}`, `scvi_absolute_k{4,6,8}`,
`hvg_k{4,6,8}`, `tx1_k{4,6,8}`, `qc_k{4,6,8}`), `cycling_weighted`, then `<vocabulary>__perm{0,1,2}` for every vocabulary
but the QC states. `pie-precompute --set ALL` runs them in that order, so the real sets and permutation seed 0 finish
first.

## Known limitations

- **Chemistry is per deposit.** A deposit pooling several kits carries one row, `mixed`, and is one scVI category:
  MIX-seq (3′ v2 and v3 experiments, 83 primary lines), Perturb-seq controls (3′ v3 and HCT116's 5′ v3), GSE147405,
  GSE182694 and GSE331320. Of the 71 rows, none is unknown; two 5′ deposits have no recorded version, and seven rows
  read the version from the deposited files against the record's text (each row's note says which).
- **Programs' K.** cNMF runs at K = 4, 6 and 8 and keeps programs robust across K; no per-line K is chosen.
- **The starCAT projection** fixes each gene's SD from the fit; starCAT's own `fit_transform` rescales by the query's
  SD. The manifest gives the two side by side on one line.
- **Storage.** About 4 KB per (gene, context) in `pie_states/` (some 140 GB for every set at 200 genes); the per-line
  pass files are removed once a set is assembled.
