#!/usr/bin/env bash
# Build the ESM4SL harness on the H20: .venv-esm4sl for the vendored official code
# (third_party/esm4sl, tracked). The ESM-2 650M weights are copied from the laptop to
# $TORCH_HOME/hub/checkpoints (dl.fbaipublicfiles.com answers HTTP 403 on the H20).
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=/2023533015/.uv/bin/uv
export UV_PYTHON_INSTALL_DIR=/2023533015/.uv/python

[[ -x .venv-esm4sl/bin/python ]] || "$UV" venv --python 3.10 .venv-esm4sl
"$UV" pip install --python .venv-esm4sl/bin/python -r hpc/esm4sl/requirements.txt
(cd third_party/esm4sl && ../../.venv-esm4sl/bin/python -c "
import esm, esm4sl, pytorch_lightning, torch
from coach_pl.tool.trainer import build_training_trainer
print('torch', torch.__version__, 'lightning', pytorch_lightning.__version__, 'cuda', torch.cuda.is_available())")
