#!/usr/bin/env python3
"""Build explicitly non-raw Kinker CPM h5ads for sensitivity analysis.

The public Kinker/scCCLE matrix is processed CPM, not UMI counts.  This
script preserves that distinction in every artifact and deliberately marks
the standard Tx1 raw-count contract as blocked.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import re
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.basal import assert_tx1_input_contract

SEMANTICS = "processed_cpm"
STATUS = "verified_processed_cpm_sensitivity_only"
ENSEMBL_PATTERN = re.compile(r"^ENSG\d+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p25", "median", "p75", "max"),
            np.quantile(values, (0.0, 0.25, 0.5, 0.75, 1.0)),
            strict=True,
        )
    }


def _read_header(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return handle.readline().rstrip("\n\r").split("\t")


def _verify_upstream_matrix_checksum(matrix: Path, checksum_path: Path) -> str:
    matches: list[str] = []
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        candidates = (
            checksum_path.parent / relative,
            checksum_path.parent.parent / relative,
        )
        if any(candidate.resolve() == matrix.resolve() for candidate in candidates):
            matches.append(expected)
    if len(matches) != 1:
        raise ValueError("upstream checksum manifest must identify the matrix once")
    actual = _sha256(matrix)
    if actual != matches[0]:
        raise ValueError("Kinker matrix disagrees with the upstream checksum manifest")
    return actual


def _load_metadata(
    selected_path: Path, manifest_path: Path, matrix_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(selected_path, sep="\t", dtype=str)
    required_selected = {
        "cell_id",
        "model_id",
        "source_cell_line_name",
        "source_cluster",
        "matrix_file",
    }
    missing = sorted(required_selected - set(selected.columns))
    if missing:
        raise ValueError(f"selected_cells.tsv is missing columns: {missing}")
    if selected["cell_id"].duplicated().any():
        raise ValueError("selected_cells.tsv contains duplicate cell_id values")
    if set(selected["matrix_file"]) != {matrix_name}:
        raise ValueError(
            "selected_cells.tsv matrix_file must uniquely equal the input matrix name"
        )

    manifest = pd.read_csv(manifest_path, dtype=str)
    required_manifest = {
        "source",
        "model_id",
        "patient_id",
        "cell_line_name",
        "source_cell_line_name",
        "matrix_semantics",
        "published_cells",
    }
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise ValueError(f"line manifest is missing columns: {missing}")
    kinker = manifest.loc[manifest["source"] == "kinker_sccle"].copy()
    if kinker["model_id"].duplicated().any():
        raise ValueError("Kinker line manifest contains duplicate model_id values")
    if set(kinker["matrix_semantics"]) != {SEMANTICS}:
        raise ValueError("Kinker manifest must declare matrix_semantics=processed_cpm")
    if set(selected["model_id"]) != set(kinker["model_id"]):
        raise ValueError("selected cell and Kinker manifest ModelID sets differ")
    joined = selected.merge(
        kinker,
        on=("model_id", "source_cell_line_name"),
        how="left",
        validate="many_to_one",
    )
    if joined["patient_id"].isna().any():
        raise ValueError("selected cells do not map completely to the line manifest")
    for column in ("model_id", "patient_id", "cell_line_name"):
        if joined[column].str.strip().eq("").any():
            raise ValueError(f"selected-cell provenance has empty {column} values")
    observed = joined.groupby("model_id", sort=False).size()
    published = kinker.set_index("model_id")["published_cells"].astype(int)
    if not observed.sort_index().equals(published.sort_index()):
        raise ValueError("per-line selected cell counts disagree with published_cells")
    return joined, kinker


def _map_genes(
    source_symbols: pd.Index, gene_metadata_path: Path, vocab_path: Path
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    metadata = pd.read_parquet(gene_metadata_path)
    required = {"gene_symbol", "ensembl_id", "token_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Tx1 gene metadata is missing columns: {missing}")
    metadata["token_id"] = pd.to_numeric(metadata["token_id"], errors="raise")
    metadata = metadata.loc[metadata["token_id"] >= 3].copy()
    metadata = metadata.astype({"gene_symbol": str, "ensembl_id": str})
    valid_ensembl = metadata["ensembl_id"].map(
        lambda value: bool(ENSEMBL_PATTERN.match(value))
    )
    if not valid_ensembl.all():
        raise ValueError("Tx1 gene metadata contains an invalid Ensembl ID")
    if metadata["ensembl_id"].duplicated().any():
        raise ValueError("Tx1 gene metadata contains duplicate Ensembl IDs")
    if metadata["token_id"].duplicated().any():
        raise ValueError("Tx1 gene metadata contains duplicate token IDs")
    ambiguous_symbols = set(
        metadata.loc[metadata["gene_symbol"].duplicated(False), "gene_symbol"]
    )
    unique = metadata.loc[~metadata["gene_symbol"].isin(ambiguous_symbols)].copy()
    vocab = json.loads(vocab_path.read_text())
    if not isinstance(vocab, dict):
        raise ValueError("Tx1 vocab.json must be a JSON object")
    unique = unique.loc[unique["ensembl_id"].isin(vocab)].copy()
    token_mismatch = unique.apply(
        lambda row: int(row["token_id"]) != int(vocab[row["ensembl_id"]]), axis=1
    )
    if token_mismatch.any():
        raise ValueError("Tx1 gene metadata token_id disagrees with vocab.json")
    by_symbol = unique.set_index("gene_symbol")
    keep = np.asarray([symbol in by_symbol.index for symbol in source_symbols])
    mapped = by_symbol.loc[source_symbols[keep]].copy()
    mapped.insert(0, "gene_symbol", source_symbols[keep].to_numpy())
    mapped = mapped.rename(columns={"ensembl_id": "sensitivity_ensembl_id"})
    mapped.index = mapped["gene_symbol"].astype(str)
    if mapped["sensitivity_ensembl_id"].duplicated().any():
        raise ValueError("gene mapping produced duplicate Ensembl IDs")
    audit = {
        "source_genes": int(len(source_symbols)),
        "ambiguous_metadata_symbols": int(len(ambiguous_symbols)),
        "mapped_tx1_vocab_genes": int(keep.sum()),
        "unmapped_or_out_of_vocab_genes": int((~keep).sum()),
    }
    return keep, mapped, audit


def _line_qc(matrix: sparse.csr_matrix) -> dict[str, object]:
    library_sums = np.asarray(matrix.sum(axis=1)).ravel()
    detected = np.diff(matrix.indptr)
    return {
        "n_cells": int(matrix.shape[0]),
        "n_genes": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "empty_cells": int((library_sums == 0).sum()),
        "empty_genes": int((np.asarray(matrix.getnnz(axis=0)) == 0).sum()),
        "library_sum": _quantiles(library_sums),
        "detected_genes": _quantiles(detected),
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    staging = output_dir.with_name(f".{output_dir.name}.building")
    if output_dir.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir} or {staging}")
    staging.mkdir(parents=True)
    try:
        selected, lines = _load_metadata(
            args.selected_cells, args.line_manifest, args.matrix.name
        )
        upstream_matrix_sha256 = _verify_upstream_matrix_checksum(
            args.matrix, args.upstream_sha256
        )
        if len(lines) != args.expected_lines:
            raise ValueError(
                f"expected {args.expected_lines} Kinker lines, found {len(lines)}"
            )
        if len(selected) != args.expected_selected_cells:
            raise ValueError(
                f"expected {args.expected_selected_cells} selected cells, "
                f"found {len(selected)}"
            )
        header = _read_header(args.matrix)
        if not header or header[0] != "GENE":
            raise ValueError("Kinker matrix first header field must be GENE")
        matrix_cells = header[1:]
        if len(matrix_cells) != args.expected_source_cells:
            raise ValueError(
                f"expected {args.expected_source_cells} source cells, "
                f"found {len(matrix_cells)}"
            )
        if len(matrix_cells) != len(set(matrix_cells)):
            raise ValueError("Kinker matrix header contains duplicate cell IDs")
        selected_ids = selected["cell_id"].tolist()
        missing_cells = sorted(set(selected_ids) - set(matrix_cells))
        if missing_cells:
            raise ValueError(f"selected cells absent from matrix: {missing_cells[:10]}")

        gene_frame = pd.read_csv(
            args.matrix,
            sep="\t",
            usecols=["GENE"],
            dtype={"GENE": str},
        )
        source_symbols = pd.Index(gene_frame["GENE"].astype(str))
        if source_symbols.duplicated().any():
            duplicate = source_symbols[source_symbols.duplicated()].unique().tolist()
            raise ValueError(
                f"Kinker matrix contains duplicate genes: {duplicate[:10]}"
            )
        keep, var, gene_audit = _map_genes(
            source_symbols, args.gene_metadata, args.vocab_json
        )
        mapped_symbols = set(source_symbols[keep])
        matrix_parts: list[sparse.csr_matrix] = []
        full_library_sums = np.zeros(len(selected_ids), dtype=np.float64)
        nonzero = 0
        noninteger = 0
        finite = True
        nonnegative = True
        rows_seen = 0
        chunks = pd.read_csv(
            args.matrix,
            sep="\t",
            usecols=["GENE", *selected_ids],
            dtype={cell_id: np.float32 for cell_id in selected_ids},
            chunksize=256,
        )
        for chunk in chunks:
            symbols = chunk.pop("GENE").astype(str)
            expected_symbols = source_symbols[rows_seen : rows_seen + len(symbols)]
            if symbols.tolist() != expected_symbols.tolist():
                raise ValueError(
                    "Kinker gene order changed between gene-only and numeric passes"
                )
            values = chunk.loc[:, selected_ids].to_numpy(copy=False)
            rows_seen += len(symbols)
            finite = finite and bool(np.isfinite(values).all())
            nonnegative = nonnegative and bool((values >= 0).all())
            full_library_sums += values.sum(axis=0, dtype=np.float64)
            present = values != 0
            nonzero += int(present.sum())
            noninteger += int(
                (present & (np.abs(values - np.rint(values)) > 1e-6)).sum()
            )
            mapped = np.asarray([symbol in mapped_symbols for symbol in symbols])
            if mapped.any():
                matrix_parts.append(sparse.csr_matrix(values[mapped, :].T))
        if rows_seen != len(source_symbols):
            raise ValueError(
                "Kinker gene-only and numeric passes disagree in row count"
            )
        numeric = {
            "finite": finite,
            "nonnegative": nonnegative,
            "nonzero_values": nonzero,
            "noninteger_nonzero_values": noninteger,
            "noninteger_nonzero_fraction": noninteger / nonzero if nonzero else 0.0,
        }
        if not finite or not nonnegative:
            raise ValueError("Kinker matrix contains negative or non-finite values")
        if numeric["noninteger_nonzero_fraction"] < args.min_noninteger_fraction:
            raise ValueError(
                "Kinker matrix does not exhibit the declared CPM semantics"
            )
        library_sum_audit = _quantiles(full_library_sums)
        library_sum_mean = float(full_library_sums.mean())
        library_sum_audit["cv"] = float(full_library_sums.std() / library_sum_mean)
        library_sum_audit["max_relative_deviation_from_median"] = float(
            np.max(
                np.abs(full_library_sums - library_sum_audit["median"])
                / library_sum_audit["median"]
            )
        )
        scale_relative_error = (
            np.abs(full_library_sums - args.expected_cpm_library_sum)
            / args.expected_cpm_library_sum
        )
        scale_pass = scale_relative_error <= args.cpm_library_sum_rtol
        scale_pass_fraction = float(scale_pass.mean())
        library_sum_audit.update(
            {
                "expected_scale": args.expected_cpm_library_sum,
                "relative_tolerance": args.cpm_library_sum_rtol,
                "cells_within_tolerance": int(scale_pass.sum()),
                "cells_outside_tolerance": int((~scale_pass).sum()),
                "fraction_within_tolerance": scale_pass_fraction,
            }
        )
        median_relative_error = (
            abs(library_sum_audit["median"] - args.expected_cpm_library_sum)
            / args.expected_cpm_library_sum
        )
        if (
            median_relative_error > args.cpm_median_rtol
            or library_sum_audit["cv"] > args.max_library_sum_cv
            or scale_pass_fraction < args.min_cpm_scale_pass_fraction
        ):
            raise ValueError(
                "Kinker full-gene cell sums violate the CPM scale gate: "
                f"{json.dumps(library_sum_audit, sort_keys=True)}"
            )
        selected["source_full_gene_library_sum"] = full_library_sums
        selected["cpm_scale_outlier"] = ~scale_pass
        filtered = sparse.hstack(matrix_parts, format="csr", dtype=np.float32)
        hvg_payload = pickle.loads(args.hvg_var_dims.read_bytes())
        hvg_genes = np.asarray(hvg_payload["gene_names"], dtype=object).astype(str)
        if not len(hvg_genes):
            raise ValueError("HVG checkpoint gene order is empty")
        if len(set(hvg_genes)) != len(hvg_genes):
            raise ValueError("HVG checkpoint gene order contains duplicates")
        mapped_gene_symbols = set(var["gene_symbol"].astype(str))
        hvg_missing = [gene for gene in hvg_genes if gene not in mapped_gene_symbols]
        hvg_fill_rate = len(hvg_missing) / len(hvg_genes) if len(hvg_genes) else 0.0
        if hvg_fill_rate > args.max_hvg_fill_rate:
            raise ValueError(
                f"HVG fill rate {hvg_fill_rate:.3f} exceeds "
                f"gate {args.max_hvg_fill_rate:.3f}"
            )

        h5ad_dir = staging / "h5ad"
        h5ad_dir.mkdir()
        var_for_h5ad = var[["gene_symbol"]].copy()
        var_for_h5ad.index = var_for_h5ad["gene_symbol"].astype(str)
        var.to_csv(staging / "gene_mapping.tsv", sep="\t", index=False)
        line_rows: list[dict[str, object]] = []
        line_qc: dict[str, object] = {}
        source_hashes = {
            "matrix_sha256": upstream_matrix_sha256,
            "selected_cells_sha256": _sha256(args.selected_cells),
            "line_manifest_sha256": _sha256(args.line_manifest),
            "gene_metadata_sha256": _sha256(args.gene_metadata),
            "vocab_sha256": _sha256(args.vocab_json),
            "hvg_var_dims_sha256": _sha256(args.hvg_var_dims),
        }
        for _, line in lines.sort_values("model_id").iterrows():
            model_id = str(line["model_id"])
            row_indices = np.flatnonzero(selected["model_id"].to_numpy() == model_id)
            obs = selected.iloc[row_indices].copy().set_index("cell_id")
            obs.index.name = None
            obs["cell_type"] = str(line["cell_line_name"])
            adata = ad.AnnData(
                X=filtered[row_indices], obs=obs, var=var_for_h5ad.copy()
            )
            adata.uns.update(
                {
                    "expression_semantics": SEMANTICS,
                    "source_dataset": "Kinker/scCCLE GSE157220",
                    "tx1_usage": "sensitivity_only",
                    "tx1_standard_raw_count_compatible": False,
                    "tx1_raw_count_block_reason": (
                        "Published processed CPM cannot be reconstructed into "
                        "UMI counts"
                    ),
                    "provenance_sha256": source_hashes,
                }
            )
            try:
                assert_tx1_input_contract(adata)
            except ValueError:
                pass
            else:
                raise ValueError(
                    f"{model_id} unexpectedly satisfies the standard Tx1 "
                    "artifact contract"
                )
            qc = _line_qc(adata.X)
            source_sums = full_library_sums[row_indices]
            qc["source_full_gene_library_sum"] = _quantiles(source_sums)
            qc["cpm_scale_outlier_cells"] = int((~scale_pass[row_indices]).sum())
            qc["cpm_scale_outlier_fraction"] = float((~scale_pass[row_indices]).mean())
            if qc["empty_cells"]:
                raise ValueError(f"{model_id} contains empty cells after vocab mapping")
            if qc["detected_genes"]["min"] < args.min_detected_genes:
                raise ValueError(f"{model_id} fails the detected-gene QC gate")
            path = h5ad_dir / f"{model_id}.h5ad"
            adata.write_h5ad(path, compression="gzip")
            reread = ad.read_h5ad(path, backed="r")
            try:
                if reread.shape != adata.shape:
                    raise ValueError(f"{model_id} h5ad shape changed on disk")
                if list(reread.obs_names) != list(obs.index):
                    raise ValueError(f"{model_id} h5ad cell order changed on disk")
                if not reread.obs_names.is_unique or not reread.var_names.is_unique:
                    raise ValueError(f"{model_id} h5ad has duplicate cells or genes")
                if reread.uns["tx1_standard_raw_count_compatible"] is not False:
                    raise ValueError(f"{model_id} lost its CPM semantic blocker")
            finally:
                reread.file.close()
            line_qc[model_id] = qc
            line_rows.append(
                {
                    "model_id": model_id,
                    "patient_id": line["patient_id"],
                    "cell_line_name": line["cell_line_name"],
                    "source_cell_line_name": line["source_cell_line_name"],
                    "n_cells": len(row_indices),
                    "n_genes": adata.n_vars,
                    "expression_semantics": SEMANTICS,
                    "tx1_usage": "sensitivity_only",
                    "tx1_raw_count_contract": "blocked",
                    "h5ad": str(path.relative_to(staging)),
                    "sha256": _sha256(path),
                }
            )

        manifest_out = staging / "manifest.tsv"
        pd.DataFrame(line_rows).to_csv(manifest_out, sep="\t", index=False)
        processing_config = {
            "matrix": str(args.matrix.resolve()),
            "selected_cells": str(args.selected_cells.resolve()),
            "line_manifest": str(args.line_manifest.resolve()),
            "gene_metadata": str(args.gene_metadata.resolve()),
            "vocab_json": str(args.vocab_json.resolve()),
            "hvg_var_dims": str(args.hvg_var_dims.resolve()),
            "upstream_sha256": str(args.upstream_sha256.resolve()),
            "output_dir": str(output_dir),
            "expression_semantics": SEMANTICS,
            "expected_lines": args.expected_lines,
            "expected_selected_cells": args.expected_selected_cells,
            "expected_source_cells": args.expected_source_cells,
            "expected_cpm_library_sum": args.expected_cpm_library_sum,
            "cpm_library_sum_rtol": args.cpm_library_sum_rtol,
            "cpm_median_rtol": args.cpm_median_rtol,
            "max_library_sum_cv": args.max_library_sum_cv,
            "min_cpm_scale_pass_fraction": args.min_cpm_scale_pass_fraction,
            "min_noninteger_fraction": args.min_noninteger_fraction,
            "max_hvg_fill_rate": args.max_hvg_fill_rate,
            "min_detected_genes": args.min_detected_genes,
        }
        (staging / "processing_config.json").write_text(
            json.dumps(processing_config, indent=2) + "\n"
        )
        qc_payload = {
            "status": STATUS,
            "source_cells_total": len(matrix_cells),
            "selected_cells": len(selected_ids),
            "selected_lines": len(lines),
            "source_genes": len(source_symbols),
            "all_selected_cells_present_in_source": True,
            "source_numeric_audit": numeric,
            "source_full_gene_library_sum": library_sum_audit,
            "gene_mapping": gene_audit,
            "hvg_checkpoint_genes": int(len(hvg_genes)),
            "hvg_missing_genes": int(len(hvg_missing)),
            "hvg_fill_rate": hvg_fill_rate,
            "tx1_gene_mapping_contract": "passed_for_sensitivity_adapter",
            "tx1_raw_count_contract": "blocked",
            "tx1_embedding_executed": False,
            "lines": line_qc,
            "source_hashes": source_hashes,
        }
        (staging / "qc.json").write_text(json.dumps(qc_payload, indent=2) + "\n")
        (staging / "status.json").write_text(
            json.dumps(
                {
                    "status": STATUS,
                    "tx1_raw_count_contract": "blocked",
                    "reason": "Kinker source is processed CPM, not raw UMI counts",
                },
                indent=2,
            )
            + "\n"
        )
        (staging / "README.md").write_text(
            "# Kinker/scCCLE 152-line processed CPM\n\n"
            "Each per-line h5ad preserves the published processed CPM values after "
            "an explicit one-to-one gene-symbol to Tx1-vocabulary Ensembl mapping. "
            "The Ensembl mapping is kept separately in `gene_mapping.tsv`; the h5ad "
            "intentionally exposes only gene symbols so a standard Tx1 loader cannot "
            "mistake it for raw-count input. "
            "These files are **not raw counts** and are restricted to a separately "
            "reported CPM sensitivity analysis. They must not enter the standard "
            "Tx1 raw-count path. `status.json` and every h5ad carry that blocker.\n"
        )
        checksum_lines = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "sha256.txt":
                checksum_lines.append(f"{_sha256(path)}  {path.relative_to(staging)}")
        (staging / "sha256.txt").write_text("\n".join(checksum_lines) + "\n")
        (staging / "CPM_SENSITIVITY_H5AD_VERIFIED").write_text(STATUS + "\n")
        staging.rename(output_dir)
        return qc_payload
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--selected-cells", type=Path, required=True)
    parser.add_argument("--line-manifest", type=Path, required=True)
    parser.add_argument("--gene-metadata", type=Path, required=True)
    parser.add_argument("--vocab-json", type=Path, required=True)
    parser.add_argument("--hvg-var-dims", type=Path, required=True)
    parser.add_argument("--upstream-sha256", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-lines", type=int, default=152)
    parser.add_argument("--expected-selected-cells", type=int, default=40_670)
    parser.add_argument("--expected-source-cells", type=int, default=53_513)
    parser.add_argument("--expected-cpm-library-sum", type=float, default=1_000_000.0)
    parser.add_argument("--cpm-library-sum-rtol", type=float, default=0.05)
    parser.add_argument("--cpm-median-rtol", type=float, default=0.02)
    parser.add_argument("--max-library-sum-cv", type=float, default=0.01)
    parser.add_argument("--min-cpm-scale-pass-fraction", type=float, default=0.999)
    parser.add_argument("--min-noninteger-fraction", type=float, default=0.01)
    parser.add_argument("--max-hvg-fill-rate", type=float, default=0.10)
    parser.add_argument("--min-detected-genes", type=int, default=1_000)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
