# Exp13 scripts

Top-level data preparation is restricted to Exp13 and its required shared inputs. Non-Exp13 processing and completed exploratory diagnostics are in [`historical_data_preparation/`](historical_data_preparation/README.md).

| Retained scripts | Exp13 dependency |
| --- | --- |
| `build_cell_line_geneeffect_226_split.py`, `cell_line_split_common.py` | Frozen 226-line membership and partition helpers |
| `prepare_kinker_umi_h5ad.py` | Raw-UMI preparation for 152 Kinker lines |
| `prepare_cell_line_atlas_raw_umi.py`, `cell_line_atlas_raw_umi_27_config.json` | Remaining 27 atlas lines |
| `materialize_exp13_original47_raw_umi.py`, `build_exp13_basal_source_registry.py` | Original 47 contexts and complete source registry |
| `build_pc9_hela_basal.py` | PC9/HeLa source h5ads consumed by original47 materialization; both belong to Exp13 train membership without GeneEffect labels |
| `download_tahoe_source_shards.sh`, `download_tahoe_dmso_subset.py` | Tahoe source/control data for the 38 Tahoe original contexts; specify the complete required context set when building Exp13 inputs |
| `register_tx1_source.py`, `build_tx1_basal_embeddings.py`, `verify_tx1_obsm_width.py` | Registered Tx1 source and encoder/loading helpers used by Exp13 cache construction |
| `bootstrap_exp13_uniprot_cache.py`, `build_exp13_esm2_universe.py`, `precompute_esm2_embeddings.py` | Authenticated gene universe and ESM2 embeddings |
| `build_exp13_tx1_cache.py`, `build_exp13_q_sc_cache.py`, `build_exp13_copy_prior.py` | Exp13 caches and copy-prior control |

Training, Stage1 sealing and the registered R1 evaluator also stay at the top level. These files support the retained Exp13 substrate; they do not establish an SL result.
