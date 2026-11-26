#!/usr/bin/env bash
# Build the PromptCASL harness on the H20: .venv-promptcasl, the Reactome gene sets, the BiomedBERT encoder
# (hf-mirror.com) and a check of the GenePT embeddings. The official code is vendored in
# third_party/promptcasl (VENDORED.md). The H20 does not reach Zenodo: the GenePT pickle (record 10833191,
# GenePT_emebdding_v2.zip) is copied from the laptop to data/promptcasl/official/ with a SHA-256 check.
# torch 2.6.0 on CUDA 12.4 in place of the released torch 2.1.2 + cu121 (as hpc/mit4sl); the other pins are
# the released env.yaml's. torch-geometric only reads MiT4SL's PrimeKG pickle.
set -euo pipefail
cd "$(dirname "$0")/../.."

UV=/2023533015/.uv/bin/uv
export UV_PYTHON_INSTALL_DIR=/2023533015/.uv/python
OFFICIAL=data/promptcasl/official
GENEPT_SHA=cd64611edab310b517cbb461295605a37821f6cd4f8ddae90ba18a2f861a6618
GMT_SHA=d24ccc3c7ecb9b5de54b0ac29a6db78d0465812dd1778c36fa19e53e9ad48f3e
mkdir -p "$OFFICIAL"

[[ -x .venv-promptcasl/bin/python ]] || "$UV" venv --python 3.10 .venv-promptcasl
"$UV" pip install --python .venv-promptcasl/bin/python -r hpc/promptcasl/requirements.txt

check() { echo "$2  $1" | sha256sum -c - >/dev/null || { echo "SHA-256 of $1 differs from $2" >&2; exit 1; }; }
GMT=$OFFICIAL/c2.cp.v7.5.1.symbols.gmt
[[ -f $GMT ]] || wget -q -O "$GMT" https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.5.1/c2.cp.v7.5.1.symbols.gmt
check "$GMT" "$GMT_SHA"
[[ -f $OFFICIAL/GenePT_gene_protein_embedding_model_3_text.pickle ]] \
  || { echo "copy GenePT_gene_protein_embedding_model_3_text.pickle to $OFFICIAL (see the header)" >&2; exit 1; }
check "$OFFICIAL/GenePT_gene_protein_embedding_model_3_text.pickle" "$GENEPT_SHA"

HF_ENDPOINT=https://hf-mirror.com .venv-promptcasl/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
                  local_dir='$OFFICIAL/biomedbert', allow_patterns=['*.json', '*.txt', '*.bin'])"
(cd third_party/promptcasl && ../../.venv-promptcasl/bin/python -c "
import torch
import src.model, src.train_model, src.main_loco
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
