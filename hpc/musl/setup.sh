#!/usr/bin/env bash
# Build the MuSL harness on the H20: .venv-musl (the vendored code is third_party/musl, tracked) and the
# released PPI graph, fetched from the pinned commit with a SHA-256 check. protein_embeddings.pt (Zenodo
# 10.5281/zenodo.17098066) is not reachable from the H20: copy it from the laptop to data/musl/official/.
# No nvcc on the H20: PyG's SAGEConv runs on torch alone (no torch-scatter / torch-sparse).
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=/2023533015/.uv/bin/uv
export UV_PYTHON_INSTALL_DIR=/2023533015/.uv/python
MUSL_COMMIT=f8021cfc618fafae8c330b694d0fa7c46db5f1a5
PPI_SHA256=10aed182ae9a9bcb114b03194f309c46e536cedaa008364d7d39577de776efb7

mkdir -p data/musl/official
ppi=data/musl/official/merged_ppi_gene_names.csv
if [[ ! -s $ppi ]]; then
  wget -q -O "$ppi" "https://raw.githubusercontent.com/JieZheng-ShanghaiTech/MuSL/$MUSL_COMMIT/processed_data/merged_ppi_gene_names.csv"
fi
echo "$PPI_SHA256  $ppi" | sha256sum -c -

[[ -x .venv-musl/bin/python ]] || "$UV" venv --python 3.10 .venv-musl
"$UV" pip install --python .venv-musl/bin/python -r hpc/musl/requirements.txt
.venv-musl/bin/python -c "
import sys
sys.path.insert(0, 'hpc/musl')
import magic, scprep, torch, torch_geometric, prepare
module = prepare.official('module')
print('torch', torch.__version__, 'pyg', torch_geometric.__version__, 'magic', magic.__version__,
      'scprep', scprep.__version__, 'cuda', torch.cuda.is_available(), module.MuSL.__name__)"
