#!/usr/bin/env bash
# Build the ESM environment on the H20: .venv-esm with EvolutionaryScale's esm 3.2.3, then fetch the pinned
# ESM C 600M and ESM3-open snapshots through hf-mirror and check CUDA works through it.
#
# esm 3.2.3 is the last release of the line that ships ESM C 600M and ESM3-open as released in 2024; 3.4 requires
# torch 2.11 and CUDA 13 kernels (cuequivariance-ops-torch-cu13), which driver 550 cannot run. Its metadata still lists
# torchvision and torchtext, which no inference module imports and whose wheels pin older torch, so esm goes in with
# --no-deps and its imported dependencies are listed here, with transformers held below 4.48.2 as esm requires. torch is
# 2.10.0 from the PyTorch cu128 index, as in .venv-pie (cu128 wheels run on driver 550 by minor-version compatibility).
# The weights go to data/esm/hf; src.data.prepare.embed_protein_sources refuses any snapshot but its pinned revisions.
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=/2023533015/.uv/bin/uv
export UV_PYTHON_INSTALL_DIR=/2023533015/.uv/python
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
export HF_HOME="$(pwd)/data/esm/hf"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60} HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}

"$UV" python install 3.12
"$UV" venv --python 3.12 .venv-esm
export VIRTUAL_ENV="$(pwd)/.venv-esm"
"$UV" pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.10.0"
"$UV" pip install --no-deps "esm==3.2.3"
"$UV" pip install \
  "transformers==4.48.1" "huggingface_hub<1" "biotite>=1.0.0" "numpy<2.4" pandas attrs einops zstd cloudpathlib \
  scipy brotli msgpack-numpy httpx tenacity biopython
"$UV" pip freeze > .venv-esm/freeze.txt

# esm's loaders download the whole snapshot; the embedding step checks it is the pinned revision.
.venv-esm/bin/python - <<'EOF'
from huggingface_hub import snapshot_download

for repo in ("EvolutionaryScale/esmc-600m-2024-12", "EvolutionaryScale/esm3-sm-open-v1"):
    print(repo, snapshot_download(repo_id=repo))
EOF

echo "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
.venv-esm/bin/python - <<'EOF'
import sys

import esm
import torch
from esm.models.esm3 import ESM3  # noqa: F401
from esm.models.esmc import ESMC  # noqa: F401

print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "esm", esm.__version__)
if not torch.cuda.is_available():
    sys.exit("CUDA is not available through .venv-esm's torch")
device = torch.device("cuda:0")
a = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
c = a @ a
torch.cuda.synchronize()
assert torch.isfinite(c).all(), "non-finite result on cuda:0"
print("bf16 matmul ran on", torch.cuda.get_device_name(device))
EOF
