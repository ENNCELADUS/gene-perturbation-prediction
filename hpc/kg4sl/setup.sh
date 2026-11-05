#!/usr/bin/env bash
# Build the KG4SL harness on the H20: the pinned released code with official.patch, and .venv-kg4sl.
#
# The released code arrives by git push from the laptop into third_party/kg4sl (see hpc/README.md); this
# script checks the pinned commit, applies the patch and builds the environment from PyPI.
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=/2023533015/.uv/bin/uv
export UV_PYTHON_INSTALL_DIR=/2023533015/.uv/python
KG4SL_COMMIT=be3626a6970479affb95fdf6b1d84a6244e52637

actual=$(git -C third_party/kg4sl rev-parse HEAD)
[[ $actual == "$KG4SL_COMMIT" ]] || { echo "third_party/kg4sl is at $actual, expected $KG4SL_COMMIT" >&2; exit 1; }
patch=$(pwd)/hpc/kg4sl/official.patch
if git -C third_party/kg4sl apply --reverse --check "$patch" 2>/dev/null; then
  echo "official.patch already applied"
else
  git -C third_party/kg4sl apply "$patch"
fi

[[ -x .venv-kg4sl/bin/python ]] || "$UV" venv --python 3.11 .venv-kg4sl
"$UV" pip install --python .venv-kg4sl/bin/python -r hpc/kg4sl/requirements.txt
PYTHONPATH=third_party/kg4sl/src .venv-kg4sl/bin/python -c "import tensorflow as tf, model, data_loader; print('tensorflow', tf.__version__)"
