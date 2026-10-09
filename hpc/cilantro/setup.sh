#!/usr/bin/env bash
# Build the official Cilantro-SL harness on the H20: pinned released code, its Python 3.9 environment, Geneformer files.
#
# The H20 cannot reach GitHub, so the released code arrives by git push from the laptop into third_party/<name>
# (see hpc/README.md); this script checks the pinned commits, applies the two patches, builds .venv-cilantro from
# PyPI (Python itself from the npmmirror copy of python-build-standalone) and fetches Geneformer from
# ModelScope's copy, checked against the Hugging Face hashes.
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=/2023533015/.uv/bin/uv
export UV_PYTHON_INSTALL_DIR=/2023533015/.uv/python
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
CILANTRO_COMMIT=c674b55e6fb126b3630502fbe8eb8ab49c5c9bc8
GENEFORMER_COMMIT=0e2bf2acbfdbc070a595d4ee66f5cda654dc624a

check_commit() {
  local actual
  actual=$(git -C "third_party/$1" rev-parse HEAD)
  [[ $actual == "$2" ]] || { echo "third_party/$1 is at $actual, expected $2" >&2; exit 1; }
}
check_commit cilantro_sl "$CILANTRO_COMMIT"
check_commit geneformer_cilantro "$GENEFORMER_COMMIT"

apply_patch() {  # repository under third_party/, patch file under hpc/cilantro/
  local patch
  patch=$(pwd)/hpc/cilantro/$2
  if git -C "third_party/$1" apply --reverse --check "$patch" 2>/dev/null; then
    echo "$2 already applied"
  else
    git -C "third_party/$1" apply "$patch"
  fi
}
apply_patch cilantro_sl official.patch
apply_patch geneformer_cilantro geneformer.patch

"$UV" python install 3.9.20
[[ -x .venv-cilantro/bin/python ]] || "$UV" venv --python 3.9.20 .venv-cilantro
"$UV" pip install --python .venv-cilantro/bin/python -r hpc/cilantro/requirements.txt \
  --override hpc/cilantro/overrides.txt

.venv-cilantro/bin/python hpc/cilantro/fetch_geneformer.py
# The fork resolves its default (gc95M) dictionaries next to geneformer/__init__.py.
for name in token_dictionary gene_median_dictionary ensembl_mapping_dict; do
  ln -sfn "$(pwd)/data/cilantro/geneformer/geneformer/${name}_gc95M.pkl" \
    "third_party/geneformer_cilantro/geneformer/${name}_gc95M.pkl"
done
# The notebooks import the fork from its source directory (sys.path), as hpc/cilantro/common.py does;
# it is not pip-installed (its setup.py requires Python >= 3.10, the released environment is 3.9).
PYTHONPATH=third_party/geneformer_cilantro .venv-cilantro/bin/python -c "import torch, geneformer, scanpy, pandas; \
print('torch', torch.__version__, torch.cuda.is_available(), 'pandas', pandas.__version__, 'scanpy', scanpy.__version__)"
