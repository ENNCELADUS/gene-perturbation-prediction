#!/usr/bin/env bash
# Build the SLMGAE harness on the H20: .venv-slmgae (TensorFlow CPU) and the data files the harness reads that are
# not tracked: the released gene list and the two released GO matrices (telemetry only; Git LFS files, 109 MB each),
# the GO ontology file of the month of the annotations, and the BioGRID table. The vendored code is third_party/slmgae.
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=/2023533015/.uv/bin/uv
export UV_PYTHON_INSTALL_DIR=/2023533015/.uv/python
COMMIT=ba0c7016922fbec569f8adda5153c28a5392ec58
OFFICIAL=data/slmgae/official
GO=data/sl_baselines/go
BIOGRID=data/sl_baselines/biogrid
mkdir -p "$OFFICIAL" "$GO" "$BIOGRID"

# fetch URL DEST [SHA256]: download to a temporary name, check, then move into place (skip when present)
fetch() {
  local url=$1 dest=$2 sha=${3:-}
  if [[ ! -s $dest ]]; then
    local tmp="$dest.part.$$"
    wget -q --timeout=30 --tries=10 --waitretry=5 -O "$tmp" "$url" || { rm -f "$tmp"; echo "cannot fetch $url" >&2; exit 1; }
    if [[ -n $sha ]]; then
      [[ $(sha256sum "$tmp" | cut -d' ' -f1) == "$sha" ]] || { rm -f "$tmp"; echo "SHA-256 mismatch for $url" >&2; exit 1; }
    fi
    mv "$tmp" "$dest"
  fi
  echo "$dest  $(sha256sum "$dest" | cut -d' ' -f1)"
}

fetch "https://raw.githubusercontent.com/DiNg1011/SLMGAE/$COMMIT/data/List_Proteins_in_SL.txt" \
  "$OFFICIAL/List_Proteins_in_SL.txt" 5c90ee7eb90286c3ed39d635f18448833857c1ae4eb5afb299eeadae87087169
fetch "https://media.githubusercontent.com/media/DiNg1011/SLMGAE/$COMMIT/data/Human_GOsim.txt" \
  "$OFFICIAL/Human_GOsim.txt" dc9b077133c9f468fda94ba0c8cfc1fcc8c05ddd04d7c69de426e4dd6d7b1a1b
fetch "https://media.githubusercontent.com/media/DiNg1011/SLMGAE/$COMMIT/data/Human_GOsim_CC.txt" \
  "$OFFICIAL/Human_GOsim_CC.txt" b227e4a64e2056f355b212aecd569a738278e38c562c0eec544d8c71efe527ad
fetch https://release.geneontology.org/2023-07-27/ontology/go-basic.obo "$GO/go-basic.2023-07-27.obo"
fetch https://raw.githubusercontent.com/kunjiefan/MVGCNiSL/85b7b880be9a86ae0b721bef0b26cc4363026882/data/BIOGRID-9606.csv \
  "$BIOGRID/BIOGRID-9606.csv"

[[ -x .venv-slmgae/bin/python ]] || "$UV" venv --python 3.11 .venv-slmgae
"$UV" pip install --python .venv-slmgae/bin/python -r hpc/slmgae/requirements.txt
PYTHONPATH=third_party/slmgae/code .venv-slmgae/bin/python -c "
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from models import SLMGAE
print('tensorflow', tf.__version__, 'SLMGAE', SLMGAE.__name__)"
