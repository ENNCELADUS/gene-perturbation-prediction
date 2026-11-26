# Vendored MuSL

Source: https://github.com/JieZheng-ShanghaiTech/MuSL, commit `f8021cfc618fafae8c330b694d0fa7c46db5f1a5`, copied on
2026-11-26 as released: `src/`, `train_model.py`, `prepare_single_cell_data.ipynb`, `LICENSE`. This note is the only
addition.

Licence: MIT (`LICENSE`).

Left out: `processed_data/` (data), the figures and `environment.yml`. `hpc/musl/setup.sh` fetches
`processed_data/merged_ppi_gene_names.csv` from the same commit (SHA-256 checked) into `data/musl/official/`; the
released `protein_embeddings.pt` (Zenodo 10.5281/zenodo.17098066) is copied from the laptop.

Used by `hpc/musl/` (imported under the package name `musl_src`, because the project has its own `src`):
`src/module.py` (`MuSL`, `EdgeDataset`, `train_step`), `src/utils.py` (`calculate_features`, `get_normalize_params`,
`normalize_with_params`, `customAdamW`). `train_model.py` is not run: it reads fixed `./processed_data/` files and
splits by cross-validation; `hpc/musl/train.py` follows its loop. The notebook is the source of the imputation steps.

Local changes: none to the vendored files. `hpc/musl/prepare.py` replaces `src.utils.get_mutual_info` at run time by
the same sklearn call in one process (the official one opens 48 processes for each call), with a fixed noise seed.
