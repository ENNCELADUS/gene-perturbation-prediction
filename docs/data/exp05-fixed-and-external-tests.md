# Exp05 Fixed Split and Cell-Line Data Roles

**Status:** unified K562 gene-pool fixed split implemented; X-Atlas/Orion HCT116
was downloaded on the experiment HPC and its one-shot frozen-backbone audit is
complete and negative. HCT116 may now be assigned to GeneEffect component
training or development, but it is no longer eligible as an untouched external
test line.

## Role contract

All samples inherit the role of their normalized perturbation gene. A gene may
appear in exactly one role.

| Role | Permitted use |
| --- | --- |
| `train` | Parameter updates only |
| `validation` | Early stopping, checkpoint selection, and hyperparameter selection |
| `internal_test` | Stage-level unbiased checks after checkpoint freeze; never selection |
| `external_test` | Final generalization evaluation; avoid repeated development-time access |

The K562 fixed manifest is built over one 9,341-gene pool: 9,338 Replogle genes
plus three genes observed only in the local non-Replogle sources. All Replogle,
Adamson, and Dixit response cells for the same target gene inherit one shared
role. The deterministic split is 85%/7.5%/7.5%: 7,939 train genes, 701
validation genes, and 701 internal-test genes. There is no K562 `external_test`
role in this configuration.

## Unified local K562 response pool

The pool combines Replogle with the three local Adamson CRISPRi h5ads and exact
single-gene conditions from the local Dixit CRISPR-KO h5ad. Each condition is
mapped to the K-562 DepMap model `ACH-000551`; combination and intergenic
conditions are excluded. Overlapping targets pool their response cells, while
only the three genuinely Replogle-unseen targets expand the gene universe. The
runtime configuration is
[`state_esm2_gwps_fixed.yaml`](../../configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_fixed.yaml).

Build command:

```bash
uv run python scripts/assemble_exp05_fixed_datasets.py \
  --predictions-csv results/experiments/05_aivc_a_to_b_to_c/runs/<audited-run>/artifacts/predictions.csv
```

Generated, gitignored data products:

- `data/sl_dependency_v0/interim/k562_non_replogle_depmap_overlap.csv`
- `data/sl_dependency_v0/interim/k562_non_replogle_depmap_overlap.csv.sha256`
- `data/sl_dependency_v0/splits/k562_pool_depmap_fixed_seed42.csv`
- `data/sl_dependency_v0/splits/k562_pool_depmap_fixed_seed42.csv.sha256`

## External cell-line candidates

| Dataset | Perturbation/readout | Decision | Reason |
| --- | --- | --- | --- |
| X-Atlas/Orion HCT116 | Genome-wide CRISPRi Perturb-seq | Development-eligible after completed audit | The sealed single-gene audit is complete and negative. Its opened GeneEffect labels may now support a declared training/development role, but HCT116 cannot again serve as an untouched test. See the [closeout](../results/exp05-hct116-frozen-backbone-transport.md). |
| X-Atlas/Orion HEK293T | Genome-wide CRISPRi Perturb-seq | Exclude from current target | DepMap Public 26Q1 has no corresponding HEK293T GeneEffect row locally |
| PRISM Repurposing | Small-molecule viability | Exclude | Drug conditions are not perturbation-gene conditions; drug-target assignment would change the supervision contract |
| Tahoe-100M / sci-Plex | Small-molecule transcriptomics | Exclude | Chemical perturbations cannot be directly labeled with target-gene GeneEffect |
| Virtual Cell Challenge H1 | CRISPRi Perturb-seq | Deferred | A DepMap H1 GeneEffect label is unavailable, so it does not meet this exp05 target contract |

Primary sources:

- [Virtual Cell Challenge public datasets](https://virtualcellchallenge.org/datasets)
- [Xaira X-Atlas/Orion dataset card](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion)
- [X-Atlas/Orion processed h5ads and metadata](https://doi.org/10.25452/figshare.plus.29190726)
- [Broad PRISM Repurposing](https://depmap.org/repurposing/)

The completed audit used 109 raw HCT116 parquet files on the experiment HPC,
sample-matched Non-Targeting controls, and an Ensembl/token-aligned 2,000-feature
cache. Its overlap table maps HCT116 `ACH-000971` GeneEffect without merging
HCT116 and K562 rows by gene. The formal result covers 3,982 label-matched genes;
full provenance and metrics are recorded in the
[`closeout`](../results/exp05-hct116-frozen-backbone-transport.md). Its opened
GeneEffect labels may be used in a declared GeneEffect component
training/development role; the completed audit and its negative result remain
unchanged.
