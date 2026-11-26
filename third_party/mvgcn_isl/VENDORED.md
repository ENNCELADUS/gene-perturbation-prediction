# Vendored MVGCN-iSL

Source: https://github.com/kunjiefan/MVGCNiSL, commit `85b7b880be9a86ae0b721bef0b26cc4363026882`, copied on 2026-11-26.
`code/main.py`, `code/model.py` and `code/utils.py` are copied as released, with `LICENSE`.

Licence: MIT (`LICENSE`).

Left out: `data/` and `README.md`. `hpc/mvgcn_isl/setup.sh` fetches the two data files that the harness reads from the
same commit, with a SHA-256 check: `BIOGRID-9606.csv` (45 MB, to `data/sl_baselines/biogrid/`) and
`coexpression_exp_0.5.csv` (7 MB, to `data/mvgcn_isl/official/`). `third_party/mvgcn_isl/data` is a link (ignored by Git)
to `data/mvgcn_isl/official/`, because the official loaders read `../data/...` from `code/`. The released
`coexpression_ess_0.2.csv` is not used: it reads the held-out lines' GeneEffect, so `hpc/mvgcn_isl/prepare.py`
makes it again with the official `calculate_coexpression` on the fit lines.

Local changes: none. The harness imports `main.train_model`, `main.test_model`, `model.GCN_pool`,
`model.EarlyStopping` and `utils.{load_graph_data, calculate_coexpression, generate_torch_edges}` and sets the
module global `main.args` (the official functions read it). A shim in `hpc/mvgcn_isl/prepare.py` restores
`pd.DataFrame.append`, which pandas 2 removed and `load_graph_data` calls.
