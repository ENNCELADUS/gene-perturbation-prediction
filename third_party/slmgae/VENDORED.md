# Vendored SLMGAE

Source: https://github.com/DiNg1011/SLMGAE, commit `ba0c7016922fbec569f8adda5153c28a5392ec58` (`master`), copied on
2026-11-26. Copied as released: `code/` and `README.md`.

Licence: the repository has no licence file. The code is vendored for research use only, with the source named here.

Left out: `data/` (the two GO matrices are Git LFS files of 109 MB each; `hpc/slmgae/setup.sh` fetches them, with the
gene list, to `data/slmgae/official/`), `BC_data/`, `.idea/`, `.gitattributes`.

Why it is vendored: `hpc/slmgae/train.py` imports the official `SLMGAE` model, `Optimizer` loss and graph helpers
(`code/models.py`, `code/objective.py`, `code/utils.py`) and runs the official training step. It is not installed as a
package: run with `PYTHONPATH=third_party/slmgae/code`.

Local changes:
- `import tensorflow as tf` became `import tensorflow.compat.v1 as tf` in `code/utils.py`, `code/layers.py`,
  `code/metrics.py` and `code/inits.py` (the released environment, TensorFlow 1.15 on Python 3.7, cannot be built on
  the H20). `code/models.py` and `code/objective.py` take `tf` from `layers.py` by `import *`. `hpc/slmgae/train.py`
  calls `tf.disable_v2_behavior()` and defines the official flags. No other line changed.
