#!/usr/bin/env python3
"""Build Tx1-ready PC9 and HeLa basal-count artifacts on the HPC."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse

from aivc_model.tx1_basal import assert_tx1_input_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_verified(adata: ad.AnnData, output: Path) -> str:
    assert_tx1_input_contract(adata)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.h5ad")
    adata.write_h5ad(temporary, compression="gzip")
    reread = ad.read_h5ad(temporary)
    assert_tx1_input_contract(reread)
    if reread.shape != adata.shape or int(reread.X.sum()) != int(adata.X.sum()):
        raise ValueError("H5AD read-back verification failed")
    temporary.replace(output)
    return _sha256(output)


def _qc(adata: ad.AnnData) -> dict[str, object]:
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    detected = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    symbols = adata.var["gene_symbol"].astype(str)
    mito = symbols.str.upper().str.startswith("MT-").to_numpy()
    mito_counts = np.asarray(adata.X[:, mito].sum(axis=1)).ravel()
    mito_fraction = np.divide(
        mito_counts, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0
    )
    return {
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "raw_integer_counts": bool(
            np.issubdtype(adata.X.dtype, np.integer)
            and (adata.X.data >= 0).all()
        ),
        "total_counts": int(totals.sum()),
        "median_counts_per_cell": float(np.median(totals)),
        "median_detected_genes_per_cell": float(np.median(detected)),
        "mitochondrial_genes_present": int(mito.sum()),
        "max_mito_fraction": float(mito_fraction.max()) if mito.any() else None,
        "zero_count_cells": int((totals == 0).sum()),
    }


def _minimum_qc_mask(adata: ad.AnnData) -> np.ndarray:
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    detected = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    mito = adata.var["gene_symbol"].astype(str).str.upper().str.startswith("MT-")
    mito_counts = np.asarray(adata.X[:, mito.to_numpy()].sum(axis=1)).ravel()
    mito_fraction = np.divide(
        mito_counts, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0
    )
    mito_ok = (mito_fraction <= 0.20) if mito.any() else np.ones(len(adata), dtype=bool)
    return (totals >= 1_000) & (detected >= 500) & mito_ok


def build_pc9(root: Path) -> dict[str, object]:
    source = root / "pc9"
    matrix_path = source / "GSM4932159_sample1_matrix.mtx.gz"
    genes_path = source / "GSM4932159_sample1_genes.tsv.gz"
    barcodes_path = source / "GSM4932159_sample1_barcodes.tsv.gz"
    genes = pd.read_csv(
        genes_path, sep="\t", header=None, names=["ensembl_id", "gene_symbol"]
    )
    barcodes = pd.read_csv(barcodes_path, header=None, names=["barcode"])
    with gzip.open(matrix_path, "rb") as handle:
        matrix = io.mmread(handle).tocsr().transpose().tocsr().astype(np.int64)
    if matrix.shape != (len(barcodes), len(genes)):
        raise ValueError("PC9 matrix dimensions do not match genes and barcodes")
    if genes["ensembl_id"].duplicated().any() or barcodes["barcode"].duplicated().any():
        raise ValueError("PC9 genes or barcodes are not unique")
    obs = pd.DataFrame(index=pd.Index(barcodes["barcode"], name="barcode"))
    obs["cell_type"] = "PC9"
    obs["treatment"] = "untreated_0h"
    var = genes.set_index("ensembl_id", drop=False)
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    source_qc = _qc(adata)
    keep = _minimum_qc_mask(adata)
    adata = adata[keep].copy()
    output = source / "formal/pc9_untreated_tx1_raw_counts.h5ad"
    output_hash = _write_verified(adata, output)
    result = {
        "context": "PC9",
        "model_id": "ACH-000779",
        "cellosaurus_id": "CVCL_B260",
        "source_files": {
            str(path.relative_to(root)): _sha256(path)
            for path in (matrix_path, genes_path, barcodes_path)
        },
        "output_path": str(output.relative_to(root.parent.parent.parent.parent)),
        "output_sha256": output_hash,
        "qc_filter": {
            "minimum_counts": 1000,
            "minimum_detected_genes": 500,
            "maximum_mito_fraction": 0.20,
            "mitochondrial_filter_applied": True,
            "source_qc": source_qc,
            "cells_removed": int((~keep).sum()),
        },
        "qc": _qc(adata),
    }
    (output.parent / "pc9_formal_provenance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _gencode_symbol_map(gtf: Path) -> tuple[dict[str, str], set[str]]:
    ids_by_symbol: dict[str, set[str]] = defaultdict(set)
    pattern = re.compile(r'(gene_id|gene_name) "([^"]+)"')
    with gtf.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            attributes = dict(pattern.findall(line.rsplit("\t", 1)[-1]))
            if "gene_id" in attributes and "gene_name" in attributes:
                ids_by_symbol[attributes["gene_name"]].add(
                    attributes["gene_id"].split(".", 1)[0]
                )
    unique = {
        symbol: next(iter(ids_))
        for symbol, ids_ in ids_by_symbol.items()
        if len(ids_) == 1
    }
    ambiguous = {symbol for symbol, ids_ in ids_by_symbol.items() if len(ids_) > 1}
    return unique, ambiguous


def _repair_hela_symbols(symbols: pd.Index) -> list[str]:
    march_occurrence = {"1-Mar": 0, "2-Mar": 0}
    repaired: list[str] = []
    for raw in symbols.astype(str):
        if raw == "1-Dec":
            repaired.append("DEC1")
        elif re.fullmatch(r"(?:[1-9]|1[0-4])-Sep", raw):
            repaired.append(f"SEPTIN{raw.split('-', 1)[0]}")
        elif re.fullmatch(r"(?:[1-9]|10|11)-Mar", raw):
            number = raw.split("-", 1)[0]
            if raw in march_occurrence:
                march_occurrence[raw] += 1
                prefix = "MARC" if march_occurrence[raw] == 1 else "MARCH"
                repaired.append(f"{prefix}{number}")
            else:
                repaired.append(f"MARCH{number}")
        else:
            repaired.append(raw)
    if march_occurrence != {"1-Mar": 2, "2-Mar": 2}:
        raise ValueError("Unexpected HeLa March-symbol corruption pattern")
    return repaired


def _hela_batch(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    symbol_to_id: dict[str, str],
) -> tuple[ad.AnnData, dict[str, object]]:
    extracted = tar.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(member.name)
    with gzip.open(extracted, "rt") as handle:
        frame = pd.read_csv(handle, sep="\t", index_col=0)
    raw_total = int(frame.to_numpy().sum())
    symbols = (
        _repair_hela_symbols(frame.index)
        if member.name.endswith("_3.txt.gz")
        else list(frame.index.astype(str))
    )
    mapped = pd.Series(symbols, index=frame.index).map(symbol_to_id)
    keep = mapped.notna().to_numpy()
    kept = frame.iloc[keep].copy()
    kept.index = mapped.iloc[np.flatnonzero(keep)].to_numpy()
    kept = kept.groupby(level=0, sort=True).sum()
    matrix = sparse.csr_matrix(kept.to_numpy(dtype=np.int64).T)
    obs = pd.DataFrame(index=pd.Index(kept.columns.astype(str), name="cell_id"))
    obs["cell_type"] = "HELA"
    obs["source_batch"] = member.name
    var = pd.DataFrame(index=pd.Index(kept.index, name="ensembl_id"))
    reverse_symbols: dict[str, str] = {}
    for symbol, ensembl_id in zip(symbols, mapped, strict=True):
        if pd.notna(ensembl_id):
            reverse_symbols.setdefault(str(ensembl_id), symbol)
    var["ensembl_id"] = var.index
    var["gene_symbol"] = [reverse_symbols[value] for value in var.index]
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    stats = {
        "source_gene_rows": int(len(frame)),
        "mapped_gene_rows": int(keep.sum()),
        "output_genes": int(adata.n_vars),
        "source_counts": raw_total,
        "retained_counts": int(matrix.sum()),
    }
    return adata, stats


def build_hela(root: Path, gtf: Path) -> dict[str, object]:
    source = root / "hela"
    tar_path = source / "GSE129447_RAW.tar"
    symbol_to_id, ambiguous = _gencode_symbol_map(gtf)
    batches: list[ad.AnnData] = []
    batch_stats: dict[str, object] = {}
    with tarfile.open(tar_path) as tar:
        members = sorted(tar.getmembers(), key=lambda member: member.name)
        if [member.name for member in members] != [
            "GSM3713084_HeLa_1.txt.gz",
            "GSM3713085_HeLa_2.txt.gz",
            "GSM3713086_HeLa_3.txt.gz",
        ]:
            raise ValueError("Unexpected HeLa archive members")
        for member in members:
            batch, stats = _hela_batch(tar, member, symbol_to_id)
            batches.append(batch)
            batch_stats[member.name] = stats
    adata = ad.concat(batches, axis=0, join="outer", fill_value=0, merge="first")
    adata.X = adata.X.tocsr().astype(np.int64)
    adata.var["ensembl_id"] = adata.var.index.astype(str)
    id_to_symbol: dict[str, str] = {}
    for symbol, ensembl_id in symbol_to_id.items():
        id_to_symbol.setdefault(ensembl_id, symbol)
    adata.var["gene_symbol"] = [id_to_symbol[value] for value in adata.var.index]
    source_qc = _qc(adata)
    keep = _minimum_qc_mask(adata)
    adata = adata[keep].copy()
    output = source / "formal/hela_720_tx1_raw_counts.h5ad"
    output_hash = _write_verified(adata, output)
    result = {
        "context": "HELA",
        "model_id": "ACH-001086",
        "cellosaurus_id": "CVCL_0030",
        "source_files": {str(tar_path.relative_to(root)): _sha256(tar_path)},
        "gencode_gtf": str(gtf),
        "gencode_gtf_sha256": _sha256(gtf),
        "ambiguous_gencode_symbols_excluded": len(ambiguous),
        "excel_symbol_repairs": {
            "1-Dec": "DEC1",
            "N-Sep": "SEPTINN",
            "first 1-Mar/2-Mar": "MARC1/MARC2",
            "remaining N-Mar": "MARCHN",
        },
        "batch_stats": batch_stats,
        "output_path": str(output.relative_to(root.parent.parent.parent.parent)),
        "output_sha256": output_hash,
        "qc_filter": {
            "minimum_counts": 1000,
            "minimum_detected_genes": 500,
            "maximum_mito_fraction": None,
            "mitochondrial_filter_applied": False,
            "source_qc": source_qc,
            "cells_removed": int((~keep).sum()),
        },
        "qc": _qc(adata),
    }
    (output.parent / "hela_formal_provenance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _register_formal_artifacts(root: Path, results: list[dict[str, object]]) -> None:
    manifest_path = root / "SOURCE_MANIFEST.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    manifest.loc[
        (manifest["context"] == "HeLa")
        & (manifest["relative_path"] == "hela/GSE129447_RAW.tar"),
        "admissibility",
    ] = "source archive; formal aligned artifact registered separately"
    rows = {
        "PC9": {
            "context": "PC9",
            "relative_path": "pc9/formal/pc9_untreated_tx1_raw_counts.h5ad",
            "accession": "derived from GSM4932159 / GSE162045",
            "source_record_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4932159",
            "basal_definition": "untreated 0 Hour; minimum-QC raw counts",
            "observed_shape": "2381 cells x 33694 genes",
            "admissibility": "tx1_contract_verified; SL-only test",
        },
        "HELA": {
            "context": "HeLa",
            "relative_path": "hela/formal/hela_720_tx1_raw_counts.h5ad",
            "accession": "derived from GSE129447",
            "source_record_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129447",
            "basal_definition": (
                "HeLa-CCL2 processed raw integer counts; gene-key aligned"
            ),
            "observed_shape": "720 cells x 22120 genes",
            "admissibility": (
                "tx1_contract_verified; SL-only test; mitochondrial QC unavailable"
            ),
        },
    }
    by_context = {str(result["context"]): result for result in results}
    for context, row in rows.items():
        result = by_context[context]
        artifact = root / str(row["relative_path"])
        row["bytes"] = str(artifact.stat().st_size)
        row["sha256"] = str(result["output_sha256"])
        match = (manifest["context"] == row["context"]) & (
            manifest["relative_path"] == row["relative_path"]
        )
        if match.any():
            for column, value in row.items():
                manifest.loc[match, column] = value
        else:
            manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    manifest.to_csv(manifest_path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--gencode-gtf", type=Path, required=True)
    parser.add_argument("--context", choices=["PC9", "HELA", "all"], default="all")
    args = parser.parse_args()
    results = []
    if args.context in {"PC9", "all"}:
        results.append(build_pc9(args.candidate_root))
    if args.context in {"HELA", "all"}:
        results.append(build_hela(args.candidate_root, args.gencode_gtf))
    if args.context == "all":
        _register_formal_artifacts(args.candidate_root, results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
