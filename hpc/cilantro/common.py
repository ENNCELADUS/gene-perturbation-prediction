"""Paths, environment and seeds shared by the official Cilantro-SL harness.

The harness runs the released Cilantro-SL code (third_party/cilantro_sl, patched by
official.patch) in its own Python 3.9 environment (.venv-cilantro). Each step script
transcribes one released notebook; the paths the notebooks hard-code are set here. See
docs/specs/2026-10-08-cilantro-sl-reproduction-design.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CODE = ROOT / "third_party" / "cilantro_sl"
OFFICIAL_DATA = ROOT / "data" / "cilantro" / "official"
GENEFORMER_DATA = ROOT / "data" / "cilantro" / "geneformer"
GENEFORMER_MODEL = GENEFORMER_DATA / "gf-12L-30M-i2048"
DEPMAP_26Q1 = ROOT / "data" / "sl_dependency_v0" / "raw" / "depmap"
RUNS = ROOT / "outputs" / "cilantro_reproduction"

DICTIONARIES = {
    # What the fork loads by default and the notebooks never override (as released).
    "gc95M": {
        "token": GENEFORMER_DATA / "geneformer" / "token_dictionary_gc95M.pkl",
        "median": GENEFORMER_DATA / "geneformer" / "gene_median_dictionary_gc95M.pkl",
        "mapping": GENEFORMER_DATA / "geneformer" / "ensembl_mapping_dict_gc95M.pkl",
    },
    # The dictionaries gf-12L-30M-i2048 was pretrained with (matched dictionary).
    "gc30M": {
        "token": GENEFORMER_DATA
        / "geneformer"
        / "gene_dictionaries_30m"
        / "token_dictionary_gc30M.pkl",
        "median": GENEFORMER_DATA
        / "geneformer"
        / "gene_dictionaries_30m"
        / "gene_median_dictionary_gc30M.pkl",
        "mapping": GENEFORMER_DATA
        / "geneformer"
        / "gene_dictionaries_30m"
        / "ensembl_mapping_dict_gc30M.pkl",
    },
}

# Symbol -> Ensembl map the notebooks use to name genes (the fork's gc95M mapping, also
# in the Drive folder); the same file in both runs, since it only converts identifiers.
SYMBOL_MAP = DICTIONARIES["gc95M"]["mapping"]


def run_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text())


def paths(run_dir: Path) -> dict[str, Path]:
    return {
        "inputs": run_dir / "inputs",
        "expression": run_dir / "inputs" / "expression.csv",
        "metadata": run_dir / "inputs" / "metadata.csv",
        "gene_effect": run_dir / "inputs" / "gene_effect.csv",
        "tokenize_input": run_dir / "tokenize_input",
        "sl_hvg_h5ad": run_dir / "tokenize_input" / "omics_expr_hvg_w_SL.h5ad",
        "tokenized": run_dir / "tokenized" / "hvg_500_w_SL_tokenized_2048.dataset",
        "genes_to_perturb": run_dir / "tokenized" / "genes_to_perturb.pkl",
        "orig_embs": run_dir / "original_embeddings" / "orig_embedding_pretrained.csv",
        "perturb": run_dir / "perturb",
        "generated": run_dir / "generated_df",
        "pairs": run_dir / "pairs" / "all_pairs_NSM_EXP.pkl",
        "models": run_dir / "models",
        "metrics": run_dir / "metrics",
    }


def configure(run_dir: Path, res_dir: Path | None = None) -> None:
    """Set the environment the patched official modules read; put them on sys.path."""
    config = run_config(run_dir)
    p = paths(run_dir)
    dictionary = DICTIONARIES[config["dictionary"]]
    os.environ["CILANTRO_TOKEN_DICTIONARY"] = str(dictionary["token"])
    os.environ["CILANTRO_ENSEMBL_MAPPING"] = str(SYMBOL_MAP)
    os.environ["CILANTRO_EXPRESSION_CSV"] = str(p["expression"])
    os.environ["CILANTRO_GENE_EFFECT_CSV"] = str(p["gene_effect"])
    os.environ["CILANTRO_HVG_H5AD"] = str(p["sl_hvg_h5ad"])
    os.environ["CILANTRO_SL_HVG_H5AD"] = str(p["sl_hvg_h5ad"])
    os.environ["CILANTRO_ORIG_EMBS_CSV"] = str(p["orig_embs"])
    os.environ["CILANTRO_MODEL_DIR"] = str(p["models"])
    os.environ["CILANTRO_RES_DIR"] = str(res_dir or run_dir / "perturb" / "res") + "/"
    for sub in ("", "nn_helpers", "isp", "input_setup"):
        path = str(OFFICIAL_CODE / sub) if sub else str(OFFICIAL_CODE)
        if path not in sys.path:
            sys.path.insert(0, path)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mark_done(directory: Path, **record) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "done.json").write_text(
        json.dumps(record, indent=1, default=str) + "\n"
    )
