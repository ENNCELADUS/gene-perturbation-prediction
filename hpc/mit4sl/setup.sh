#!/usr/bin/env bash
# Build the MiT4SL harness on the H20: the pinned released code (third_party/mit4sl, cloned over HTTPS) and
# .venv-mit4sl. torch-sparse has no wheel for torch 2.6 on PyPI and data.pyg.org is unreachable, so it is built
# from source with torch-scatter, CPU only: HGTLoader samples on the CPU.
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=/2023533015/.uv/bin/uv
export UV_PYTHON_INSTALL_DIR=/2023533015/.uv/python
MIT4SL_COMMIT=68cfb5c0c2e76a98824ae8511799685c78ef4a93

[[ -d third_party/mit4sl ]] || git clone -q https://github.com/JieZheng-ShanghaiTech/MiT4SL.git third_party/mit4sl
actual=$(git -C third_party/mit4sl rev-parse HEAD)
[[ $actual == "$MIT4SL_COMMIT" ]] || { echo "third_party/mit4sl is at $actual, expected $MIT4SL_COMMIT" >&2; exit 1; }

[[ -x .venv-mit4sl/bin/python ]] || "$UV" venv --python 3.10 .venv-mit4sl
"$UV" pip install --python .venv-mit4sl/bin/python -r hpc/mit4sl/requirements.txt
"$UV" pip install --python .venv-mit4sl/bin/python setuptools wheel
# no nvcc on the H20: CPU builds; train.py keeps PyG off torch-scatter on the GPU
FORCE_ONLY_CPU=1 MAX_JOBS=32 "$UV" pip install --python .venv-mit4sl/bin/python --no-build-isolation \
  torch-scatter==2.1.2 torch-sparse==0.6.18
PYTHONPATH=third_party/mit4sl/src .venv-mit4sl/bin/python -c "
import torch, torch_geometric, torch_sparse
from model import MiT4SL
from torch_geometric.loader import HGTLoader
print('torch', torch.__version__, 'pyg', torch_geometric.__version__, 'cuda', torch.cuda.is_available())"
