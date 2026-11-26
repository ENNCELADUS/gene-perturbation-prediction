# Vendored PromptCASL

Source: https://github.com/JieZheng-ShanghaiTech/PromptCASL, commit `afcf7e9221f873a6f13e4db8edb570c7810f9bc9`
(`main`, "Update analysis2"), copied on 2026-11-26. Paper: Lu, Feng, Zhang, Rui, Zheng, "Prompt-Driven
Context-Aware Prediction of Cell Line-Specific Synthetic Lethality via Biomedical Large Language Models",
IEEE BIBM 2025.

Licence: MIT (`LICENSE`).

Copied as released: `src/`, `scripts/`, `data/build_splits.py`, `env.yaml`, `LICENSE`.

Left out: `ana/` (interpretability notebooks and helpers), `data/folds_assignment/`, `data/text_template/`,
`data/multi_omics/` (the released labels, prompts and pathway tables of the six paper cell lines; the
harness builds its own for the benchmark lines), `PromptCASL_pipeline.png`, `README.md`.

Why it is vendored: `hpc/promptcasl/` imports the official dataset step (`src/preprocess_datasets`), pair
builder (`src/pair_pathway`), model, training loop (`src/train_model`), argument parser (`src/main_loco`),
the ConvE script and the neighbour functions, and runs them on the benchmark. It is run with
`.venv-promptcasl` (`hpc/promptcasl/setup.sh`).

Local changes:
- `src/__init__.py`, `scripts/__init__.py` (new, empty): the package is a top-level `src` like the
  repository's own; with an `__init__.py` it wins when `third_party/promptcasl` is first on the path, as
  the official namespace package would not.

Everything else is changed in `hpc/promptcasl/` by assigning module attributes, not by editing these files:
`src.pathway_utils` (CCLE and GMT paths, `zscore_ge` on the fit lines), `scripts.build_reactome_pair_evidence`
(line list, output directory), `scripts.train_conve_primekg` (`PrimeKG` returns the MiT4SL copy without the
benchmark pairs' edges), `src.pair_pathway` (artifact directory, lineage of each line, lineage mean on the
fit lines). The official `README.md` command `--omics_mode pair_pathway_evidence` also needs
`--omics_encoder 0`: its default 1 is refused by `main_loco.main`.
