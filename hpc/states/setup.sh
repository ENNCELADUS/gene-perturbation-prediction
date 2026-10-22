#!/usr/bin/env bash
# Build the cell-state environment on the H20: .venv-states with cNMF, starCAT and scvi-tools, then check that they
# import and that CUDA works through its torch.
#
# src.experiments.states.programs runs here (consensus NMF per training line, the recurrent-program reference and
# its usages, scVI and its latents; PIE stage-one design §3.5, Task 8) and exchanges files only with the project
# environment, which drives it through STATES_PYTHON (hpc/run.sh cell-states). Pinned: cnmf 1.7.1, starcatpy 1.0.10,
# scvi-tools 1.4.2, torch 2.10.0 from the PyTorch cu128 index as in .venv-pie (cu128 wheels run on driver 550 by
# minor-version compatibility). The module imports src.data.basal, src.data.ladder_cells and src.data.tx1_cache, which
# need anndata, pandas, pyarrow and scipy only.
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=/2023533015/.uv/bin/uv
export UV_PYTHON_INSTALL_DIR=/2023533015/.uv/python
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone

"$UV" python install 3.12
"$UV" venv --python 3.12 .venv-states
export VIRTUAL_ENV="$(pwd)/.venv-states"
"$UV" pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.10.0"
"$UV" pip install "scvi-tools==1.4.2" "cnmf==1.7.1" "starcatpy==1.0.10" pyarrow "torch==2.10.0"
"$UV" pip freeze > .venv-states/freeze.txt

echo "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
.venv-states/bin/python - <<'EOF'
import sys

import cnmf
import scvi
import starcat  # noqa: F401
import torch

print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
print("cnmf", cnmf.__version__, "scvi-tools", scvi.__version__)
if not torch.cuda.is_available():
    sys.exit("CUDA is not available through .venv-states's torch")
a = torch.randn(1024, 1024, device="cuda:0")
torch.cuda.synchronize()
assert torch.isfinite(a @ a).all(), "non-finite result on cuda:0"
print("float32 matmul ran on", torch.cuda.get_device_name(0))
EOF
.venv-states/bin/python -c "import src.experiments.states.programs"
echo "STATES_PYTHON=$(pwd)/.venv-states/bin/python"
