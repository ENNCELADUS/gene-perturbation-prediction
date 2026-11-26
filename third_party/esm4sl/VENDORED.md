# Vendored ESM4SL

Source: https://github.com/JieZheng-ShanghaiTech/ESM4SL, commit `348b2bf9a03f89b8a80d9becb329522a5bb13053`,
copied on 2026-11-26 with `git archive`.

Licence: MIT (`LICENSE`); `coach_pl/` is Apache-2.0 (`coach_pl/LICENSE`).

Copied: `esm4sl/`, `coach_pl/`, `data_preprocess/esm2_gen.py`, `train.py`, `test.py`, `LICENSE`.
Left out: `data/` (mapping tables, SLKB scores; the benchmark inputs are built by `hpc/esm4sl/prepare.py`),
`script/`, the notebooks, `environment.sh`, `deepspeed_combine.py`, `README.md`.

It is run through its official command line (`hpc/esm4sl/setup.sh`, `prepare.py`, `collect.py`; commands in
`docs/specs/2026-11-26-published-baselines-plan.md`, Task 8). The config `esm4sl/configuration/attn/new.yaml` has
`DATASET.CELL_LINE: null`, so the code path with the authors' absolute path in `esm4sl/dataset/sl.py` is not used.

Local changes:
- `esm4sl/module/cls_module.py` (2026-11-26): `step_common` no longer appends each batch's attention map
  (`[batch, 2000, 2000]` float, 256 MB at batch 16) to `self.attns`. Nothing reads that list; it kept every map of an
  epoch in memory, which is about 540 GB per training epoch and about 18 TB in the test pass over 1.1 million pairs.
