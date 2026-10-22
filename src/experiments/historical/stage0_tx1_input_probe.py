"""Historical probe of whether Tx1-3B encodes a CPM-rescaled cell as it encodes the
raw counts underneath it, with the collator's unseeded gene subsampling (``_sample``,
``torch.randperm``) isolated by seeding each arm just before its forward pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy import sparse

_LOGGER = logging.getLogger("stage0_tx1_input_probe")

#: Kinker's published CPM matrix sums to ~1e6 per cell.
CPM_TARGET_SUM: float = 1e6


def _sha256_order(barcodes: np.ndarray) -> np.ndarray:
    """Cells in ascending ``sha256(barcode)``: reproducible without an index file."""
    digests = [
        hashlib.sha256(str(barcode).encode("utf-8")).hexdigest() for barcode in barcodes
    ]
    return np.argsort(np.asarray(digests))


def assert_raw_counts(matrix: sparse.spmatrix | np.ndarray) -> dict[str, float]:
    """Audit numbers of ``matrix``; fails closed unless it is raw counts (finite,
    non-negative, integral), since the probe's claim is vacuous on a reference arm
    that is already normalised."""
    data = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0:
        raise ValueError("matrix has no stored values")
    if not np.all(np.isfinite(data)):
        raise ValueError("matrix contains non-finite values")
    if np.any(data < 0):
        raise ValueError("matrix contains negative values")
    noninteger = float(np.mean(data != np.rint(data)))
    if noninteger > 0.0:
        raise ValueError(
            "matrix is not raw counts: "
            f"{noninteger:.6f} of stored values are non-integer"
        )
    return {
        "stored_values": float(data.size),
        "max_value": float(data.max()),
        "noninteger_fraction": noninteger,
    }


def to_cpm(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Rescale each cell to ``CPM_TARGET_SUM``, preserving the zero pattern."""
    csr = sparse.csr_matrix(matrix, dtype=np.float64)
    sums = np.asarray(csr.sum(axis=1)).ravel()
    if np.any(sums <= 0):
        raise ValueError("every cell must have a positive library size")
    scale = CPM_TARGET_SUM / sums
    scaled = sparse.diags(scale) @ csr
    return sparse.csr_matrix(scaled, dtype=np.float32)


def detected_genes(matrix: sparse.spmatrix) -> np.ndarray:
    """Per-cell count of nonzero genes -- what ``max_length`` is compared to."""
    csr = sparse.csr_matrix(matrix)
    csr.eliminate_zeros()
    return np.diff(csr.indptr)


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = np.maximum(left_norm * right_norm, 1e-12)
    return np.sum(left * right, axis=1) / denominator


def compare(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """Per-cell agreement of ``other`` with ``reference``, and of their pooled
    means, the pooled mean being what the context embedding consumes."""
    cosines = _cosine_rows(reference, other)
    reference_mean = reference.mean(axis=0, keepdims=True)
    other_mean = other.mean(axis=0, keepdims=True)
    return {
        "cells": int(reference.shape[0]),
        "identical": bool(np.array_equal(reference, other)),
        "max_abs_diff": float(np.abs(reference - other).max()),
        "cosine_min": float(cosines.min()),
        "cosine_mean": float(cosines.mean()),
        "pooled_mean_cosine": float(_cosine_rows(reference_mean, other_mean)[0]),
    }


def run_probe(
    adata: ad.AnnData,
    encode: object,
    *,
    seed: int,
    max_length: int,
) -> dict[str, object]:
    """The ``cpm``, ``repeat_seeded`` and ``repeat_unseeded`` arms against the
    ``raw`` encode of the same cells, overall and split by whether a cell exceeds
    ``max_length`` detected genes (only those can trigger ``_sample``)."""
    counts = detected_genes(adata.X)
    wide = counts > max_length

    cpm_adata = adata.copy()
    cpm_adata.X = to_cpm(adata.X)

    torch.manual_seed(seed)
    raw_embeddings = encode(adata)
    torch.manual_seed(seed)
    cpm_embeddings = encode(cpm_adata)
    torch.manual_seed(seed)
    repeat_seeded = encode(adata)
    repeat_unseeded = encode(adata)

    arms = {
        "cpm_vs_raw": cpm_embeddings,
        "repeat_seeded_vs_raw": repeat_seeded,
        "repeat_unseeded_vs_raw": repeat_unseeded,
    }
    report: dict[str, object] = {
        "cells": int(adata.n_obs),
        "embedding_width": int(raw_embeddings.shape[1]),
        "max_length": int(max_length),
        "detected_genes": {
            "min": float(counts.min()),
            "median": float(np.median(counts)),
            "max": float(counts.max()),
            "over_max_length": int(wide.sum()),
        },
        "arms": {},
    }
    for name, embeddings in arms.items():
        entry = {"all_cells": compare(raw_embeddings, embeddings)}
        if wide.any():
            entry["wide_cells"] = compare(raw_embeddings[wide], embeddings[wide])
        if (~wide).any():
            entry["narrow_cells"] = compare(raw_embeddings[~wide], embeddings[~wide])
        report["arms"][name] = entry
    return report


def load_probe_adata(
    path: Path,
    *,
    n_cells: int,
    vocab_genes: set[str] | None,
    var_ensembl_col: str | None,
) -> ad.AnnData:
    """A deterministic cell subset on the genes in ``vocab_genes``: the encoder
    raises on a gene outside the Tx1 vocabulary."""
    adata = ad.read_h5ad(path)
    if var_ensembl_col and var_ensembl_col in adata.var.columns:
        adata.var.index = adata.var[var_ensembl_col].astype(str)
    adata.var["ensembl_id"] = adata.var.index.astype(str)
    if vocab_genes is not None:
        keep = adata.var["ensembl_id"].isin(vocab_genes).to_numpy()
        if not keep.any():
            raise ValueError("no var gene id is present in the Tx1 vocabulary")
        adata = adata[:, keep].copy()
    order = _sha256_order(adata.obs_names.to_numpy())[:n_cells]
    adata = adata[order].copy()
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "probe"
    adata.X = sparse.csr_matrix(adata.X)
    return adata


def build_encoder(model_dir: Path, batch_size: int, max_length: int):
    """The Tx1-3B encode callable, its loader built per call so the global torch
    seed can be set between forward passes (``src.data.tx1_cache`` cannot)."""
    from composer import Trainer
    from tahoe_x1.utils.util import loader_from_adata

    from src.model.tx1 import install_padding_metadata_fallback, load_local_safetensors

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to run the Tx1 input probe")
    install_padding_metadata_fallback()
    model, vocab, collator_config, load_report = load_local_safetensors(model_dir)
    trainer = Trainer(model=model, device="gpu")

    def encode(adata: ad.AnnData) -> np.ndarray:
        genes = adata.var["ensembl_id"].astype(str).tolist()
        gene_ids = np.asarray([vocab[gene] for gene in genes], dtype=int)
        loader = loader_from_adata(
            adata=adata,
            collator_cfg=collator_config,
            vocab=vocab,
            batch_size=batch_size,
            max_length=max_length,
            gene_ids=gene_ids,
            num_workers=0,
            prefetch_factor=None,
        )
        predictions = trainer.predict(loader, return_outputs=True)
        return (
            torch.cat(
                [output["cell_emb"].detach().float().cpu() for output in predictions],
                dim=0,
            )
            .numpy()
            .astype(np.float32)
        )

    return encode, vocab, load_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--n-cells", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--var-ensembl-col", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(message)s")

    encode, vocab, load_report = build_encoder(
        args.model_dir, args.batch_size, args.max_length
    )
    vocab_genes = {str(gene) for gene in vocab.get_stoi()}

    adata = load_probe_adata(
        args.adata,
        n_cells=args.n_cells,
        vocab_genes=vocab_genes,
        var_ensembl_col=args.var_ensembl_col,
    )
    count_audit = assert_raw_counts(adata.X)
    _LOGGER.info("probe cells=%d genes=%d", adata.n_obs, adata.n_vars)

    report = run_probe(adata, encode, seed=args.seed, max_length=args.max_length)
    report["source"] = {
        "adata": str(args.adata),
        "model_dir": str(args.model_dir),
        "seed": args.seed,
        "raw_count_audit": count_audit,
        "checkpoint_load": load_report,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
