#!/usr/bin/env python3
"""Stage 0 probe: does Tx1 read CPM input the same way it reads raw counts?

``docs/04-exp13-geneeffect-residual-protocol.md`` §6 registers one open
question: 152 of the 179 new-atlas lines are Kinker ``processed_cpm``, and
$x_c = E_{Tx1}(r_c)$ is only trustworthy if the encoder treats a
CPM-normalized row the way it treats the raw counts underneath it. §6 rules
out "swap only $z_c$" as a fallback, so an untrustworthy embedding voids the
whole ST pass for those lines.

The static answer is that it does, because the released checkpoint's
``collator_config.yml`` sets ``do_binning: true``: :func:`tahoe_x1.data.
collator.binning` bucketizes each cell against quantiles of *its own*
nonzero values, and ``bucketize(k*x, k*q) == bucketize(x, q)`` for any
``k > 0``. Kinker's matrix is linear CPM (per-cell library sums ~1e6), i.e.
exactly such a ``k`` per cell. This probe measures that claim on real cells
through the real encoder instead of asserting it.

It also isolates a second effect found while reading the same collator:
``max_length`` is 2048 with ``sampling: true``, and ``_sample`` draws its
gene subset with an **unseeded** ``torch.randperm``. Every Kinker cell is
wider than that (detected-gene median 4637), so repeat encodes of one cell
disagree for reasons that have nothing to do with CPM. The two effects are
confounded unless the RNG is pinned, so each arm seeds immediately before
its forward pass; ``repeat_unseeded`` then measures the sampling effect on
its own.

Arms (all on the same cells, same order):

``raw``               reference encode of the raw-count matrix
``cpm``               encode of ``raw * 1e6 / library_size``, same seed
``repeat_seeded``     ``raw`` again, same seed -- isolates seeding
``repeat_unseeded``   ``raw`` again, RNG left running -- gene subsampling

Cells are reported split by whether they exceed ``--max-length`` detected
genes, because only the wide group can trigger ``_sample`` at all.
"""

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

#: Target library size for the CPM arm, matching Kinker's published matrix
#: (``GSE157220_CPM_data.txt.gz``, per-cell sums ~1e6).
CPM_TARGET_SUM: float = 1e6


def _sha256_order(barcodes: np.ndarray) -> np.ndarray:
    """Deterministic cell order: ascending ``sha256(barcode)``.

    Mirrors ``docs/04`` §5's cell-subsample rule so the probe's cells are
    reproducible across runs and machines without carrying an index file.
    """
    digests = [
        hashlib.sha256(str(barcode).encode("utf-8")).hexdigest() for barcode in barcodes
    ]
    return np.argsort(np.asarray(digests))


def assert_raw_counts(matrix: sparse.spmatrix | np.ndarray) -> dict[str, float]:
    """Fail closed unless ``matrix`` really is raw counts.

    The probe's whole claim is "CPM encodes like the counts underneath it",
    which is vacuous if the reference arm is already normalized. Replogle
    ships ``_raw_`` and ``_normalized_`` h5ads whose names differ by one
    word (``docs/data/replogle-k562-gwps.md``), so this is a live confusion.

    Returns:
        Audit numbers worth recording next to the result.

    Raises:
        ValueError: values are negative, non-finite, or not integral.
    """
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
    """Rescale each cell to ``CPM_TARGET_SUM``, preserving the zero pattern.

    This is the transform whose invariance is under test: a strictly
    positive per-cell scalar. It cannot move a zero, reorder a cell's genes,
    or break a tie, which is exactly why per-cell quantile binning should
    not see it.
    """
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
    """Row-wise cosine similarity between two equal-shaped embedding blocks."""
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = np.maximum(left_norm * right_norm, 1e-12)
    return np.sum(left * right, axis=1) / denominator


def compare(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """Summarize how far ``other`` sits from ``reference``.

    Reports the per-cell agreement and the agreement of the pooled mean,
    because the pooled mean is what ``z_c`` and ``Delta`` actually consume
    (``01-blueprint.md`` §3) -- a per-cell wobble that averages out matters
    far less than one that does not.
    """
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
    """Encode the four arms and compare them.

    Args:
        adata: Raw-count AnnData, already restricted to encodable genes.
        encode: Callable taking an AnnData and returning ``[cells, width]``.
        seed: Global torch seed set immediately before the seeded arms.
        max_length: The collator's context window, used only to split the
            report into cells that can and cannot trigger ``_sample``.

    Returns:
        A JSON-ready record: per-arm comparisons, overall and split by
        whether the cell exceeds ``max_length`` detected genes.
    """
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
    """Load a deterministic cell subset restricted to encodable genes.

    Genes are filtered to ``vocab_genes`` because the encoder resolves every
    ``var["ensembl_id"]`` through the Tx1 vocabulary and raises on a miss;
    filtering here keeps the probe's failure modes about the question under
    test rather than about vocabulary coverage.
    """
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
    """Build the real Tx1-3B encode callable.

    Deliberately not reusing ``build_tx1_basal_embeddings._build_tx1_encoder``:
    that one is private and, more to the point, this probe must be able to
    set the global torch seed *between* forward passes, which only works if
    the loader is constructed per call.
    """
    from composer import Trainer
    from tahoe_x1.utils.util import loader_from_adata

    from scripts.verify_tx1_obsm_width import (
        install_padding_metadata_fallback,
        load_local_safetensors,
    )

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
    """Parse command-line arguments."""
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
    """Run the probe against the real Tx1-3B encoder and write its report."""
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
    _LOGGER.info("wrote %s", args.out_json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
