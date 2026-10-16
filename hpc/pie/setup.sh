#!/usr/bin/env bash
# Build the PIE environment on the H20: .venv-pie from the vendored copy's lock, then check CUDA works through it.
#
# PIE is vendored and tracked in third_party/pie (commit in third_party/pie/VENDORED.md), so it arrives with git pull.
# uv syncs its lock into .venv-pie at the repo root: Python 3.12 (from the npmmirror copy of python-build-standalone),
# PIE installed editable, torch 2.10.0 from the PyTorch cu128 index, and PIE's dev group (pytest).
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=/2023533015/.uv/bin/uv
export UV_PYTHON_INSTALL_DIR=/2023533015/.uv/python
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone

"$UV" python install 3.12
UV_PROJECT_ENVIRONMENT="$(pwd)/.venv-pie" "$UV" sync --frozen --python 3.12 --project third_party/pie

# cu128 wheels nominally need driver 570 or newer; fail here if this driver cannot run them.
echo "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
.venv-pie/bin/python - <<'EOF'
import sys

import torch
import torch.nn.functional as F

print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA is not available through .venv-pie's torch")
device = torch.device("cuda:0")
print("device", torch.cuda.get_device_name(device))
a = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
c = a @ b
q = torch.randn(2, 8, 256, 64, device=device, dtype=torch.bfloat16)
attn = F.scaled_dot_product_attention(q, q, q)
torch.cuda.synchronize()
assert torch.isfinite(c).all() and torch.isfinite(attn).all(), "non-finite result on cuda:0"
print("bf16 matmul and scaled_dot_product_attention ran on cuda:0")
EOF

# PIE's tests resolve paths against the working directory, so they run from third_party/pie.
echo "PIE's tests: (cd third_party/pie && ../../.venv-pie/bin/python -m pytest tests -q -p no:cacheprovider)"
