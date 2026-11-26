#!/usr/bin/env bash
# Build the MVGCN-iSL harness on the H20: the two released data files (pinned commit of kunjiefan/MVGCNiSL, SHA-256
# checked), the link the official loaders read (third_party/mvgcn_isl/data -> ../data), and .venv-mvgcn.
# BIOGRID is shared with other harnesses: it is fetched under a temporary name and moved into place.
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=/2023533015/.uv/bin/uv
export UV_PYTHON_INSTALL_DIR=/2023533015/.uv/python
RAW=https://raw.githubusercontent.com/kunjiefan/MVGCNiSL/85b7b880be9a86ae0b721bef0b26cc4363026882/data
BIOGRID_SHA=6516b66add291999e0fb3705b79f83f5b96a5c4405745b698c2917e3d91c58e7
COEXP_SHA=bd3d86093a27e11669083a29dc8d3d0745e8fdf315ffe6156d38e95b12d23d48

fetch() { # url, target, sha256
  if [[ -f $2 ]] && [[ $(sha256sum "$2" | cut -d' ' -f1) == "$3" ]]; then return; fi
  wget -q -T 30 -t 6 -O "$2.part.$$" "$1"  # the H20's DNS sometimes answers an address that hangs
  [[ $(sha256sum "$2.part.$$" | cut -d' ' -f1) == "$3" ]] || { echo "SHA-256 mismatch for $1" >&2; rm -f "$2.part.$$"; exit 1; }
  mv "$2.part.$$" "$2"
}
mkdir -p data/sl_baselines/biogrid data/mvgcn_isl/official/CCLE
fetch "$RAW/BIOGRID-9606.csv" data/sl_baselines/biogrid/BIOGRID-9606.csv "$BIOGRID_SHA"
fetch "$RAW/coexpression_exp_0.5.csv" data/mvgcn_isl/official/coexpression_exp_0.5.csv "$COEXP_SHA"
# the official loaders read ../data/BIOGRID-9606.csv and ../data/coexpression_exp_0.5.csv from code/
ln -sfn "$PWD/data/sl_baselines/biogrid/BIOGRID-9606.csv" data/mvgcn_isl/official/BIOGRID-9606.csv
ln -sfn "$PWD/data/mvgcn_isl/official" third_party/mvgcn_isl/data

[[ -x .venv-mvgcn/bin/python ]] || "$UV" venv --python 3.10 .venv-mvgcn
"$UV" pip install --python .venv-mvgcn/bin/python -r hpc/mvgcn_isl/requirements.txt
MPLBACKEND=Agg PYTHONPATH=third_party/mvgcn_isl/code .venv-mvgcn/bin/python -c "
import torch
from model import GCN_pool
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
