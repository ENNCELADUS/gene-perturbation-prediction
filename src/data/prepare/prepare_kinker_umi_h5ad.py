#!/usr/bin/env python3
"""Build raw-UMI Kinker h5ads that satisfy the standard Tx1 input contract.

Sibling of ``src/experiments/historical/prepare_kinker_cpm_h5ad.py``
with the contract inverted. That
script ingests the GEO ``GSE157220_CPM_data.txt.gz`` matrix and deliberately
marks every artifact ``tx1_raw_count_contract: blocked``, because published
CPM cannot be reconstructed into counts. Exp13 Stage 0 measured what feeding
Tx1 that CPM actually costs -- per-cell cosine 0.94 against the raw encode,
and, unlike gene-subsampling noise, a shift that survives pooling to the
per-line mean the model consumes -- so the CPM path is not usable for the
152 Kinker lines and the historical Exp13 protocol §6 branch 1 applies.

The counts exist: Broad Single Cell Portal SCP542 publishes
``UMIcount_data.txt`` (pre-QC UMI counts, 56,982 cells, 207 lines), which
covers all 40,670 selected cells and all 152 selected lines. This script
ingests that matrix instead, and every gate here is the CPM script's gate
turned around:

- values must be **integral** (the CPM script requires a *non*-integer
  fraction, as proof of its declared semantics);
- there is **no library-size gate** -- UMI depth is a free quantity, whereas
  the CPM script requires every cell to sum to ~1e6;
- each h5ad must **pass** :func:`assert_tx1_input_contract`, so ``var`` is
  indexed by Ensembl id and carries ``ensembl_id``; the CPM script keeps
  symbols precisely so a standard Tx1 loader cannot accept its output.

Two structural differences in the source file itself: its header begins with
an empty field rather than ``GENE``, and two metadata rows (``Cell_line``,
``Pool_ID``) precede the gene rows. The ``Cell_line`` row is not skipped
blindly -- it is cross-checked against ``selected_cells.tsv``'s
``source_cell_line_name``, which independently confirms that the cells
selected from the CPM matrix denote the same lines in this one.

The frozen line manifest still declares ``matrix_semantics: processed_cpm``
for these lines and is **not** edited here: its hash is pinned by
``build_cell_line_geneeffect_226_split.py``'s ``PINNED_SHA256``, and the 226
split membership is unchanged by this switch. Only the numeric source moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.basal import assert_tx1_input_contract

SEMANTICS = "raw_umi"
STATUS = "verified_raw_umi"
#: Non-gene rows the SCP matrix carries before its first gene row, in order.
METADATA_ROWS: tuple[str, ...] = ("Cell_line", "Pool_ID")


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
        )
    }


def read_matrix_head(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Read the header and the metadata rows that precede the gene rows.

    Returns:
        ``(cell_ids, metadata)`` where ``metadata`` maps each row label in
        :data:`METADATA_ROWS` to its per-cell values.

    Raises:
        ValueError: the header does not start with an empty field, or the
            metadata rows are absent or out of order -- either means the
            file is not the SCP ``UMIcount_data.txt`` layout and every row
            offset below would be wrong.
    """
    with path.open("rt") as handle:
        header = handle.readline().rstrip("\n\r").split("\t")
        if not header or header[0] != "":
            raise ValueError(
                "UMI matrix header must begin with an empty field; "
                f"found {header[:1]!r}"
            )
        cells = header[1:]
        metadata: dict[str, list[str]] = {}
        for expected in METADATA_ROWS:
            row = handle.readline().rstrip("\n\r").split("\t")
            if not row or row[0] != expected:
                raise ValueError(
                    f"expected metadata row {expected!r}, found {row[:1]!r}"
                )
            if len(row) - 1 != len(cells):
                raise ValueError(f"metadata row {expected!r} has the wrong width")
            metadata[expected] = row[1:]
    if len(cells) != len(set(cells)):
        raise ValueError("UMI matrix header contains duplicate cell IDs")
    return cells, metadata


def assert_line_labels_agree(
    cells: list[str], cell_lines: list[str], selected: pd.DataFrame
) -> int:
    """Cross-check the matrix's own line labels against the selection.

    The selection was derived from the CPM matrix; this confirms the same
    barcodes denote the same cell lines in the UMI matrix, which is the one
    assumption that silently ruins everything downstream if it is false.

    Returns:
        The number of cells checked.

    Raises:
        ValueError: any selected cell's line label disagrees.
    """
    label_by_cell = dict(zip(cells, cell_lines))
    disagreements = [
        (
            str(row.cell_id),
            label_by_cell[str(row.cell_id)],
            str(row.source_cell_line_name),
        )
        for row in selected.itertuples()
        if label_by_cell.get(str(row.cell_id)) != str(row.source_cell_line_name)
    ]
    if disagreements:
        raise ValueError(
            "UMI matrix cell-line labels disagree with the selection for "
            f"{len(disagreements)} cells, e.g. {disagreements[:5]}"
        )
    return len(selected)


def _load_metadata(
    selected_path: Path,
    manifest_path: Path,
    selection_matrix_name: str,
    manifest_semantics: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the selection and the frozen line manifest, and reconcile them.

    ``selection_matrix_name`` is the matrix the *selection* was made from
    (the CPM file), not the matrix read here: ``selected_cells.tsv`` records
    its own provenance and is not rewritten by this script.
    """
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
    if set(selected["matrix_file"]) != {selection_matrix_name}:
        raise ValueError(
            "selected_cells.tsv matrix_file must uniquely equal "
            f"{selection_matrix_name!r}"
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
    if set(kinker["matrix_semantics"]) != {manifest_semantics}:
        raise ValueError(
            f"Kinker manifest must declare matrix_semantics={manifest_semantics}"
        )
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
    observed = joined.groupby("model_id", sort=False).size()
    published = kinker.set_index("model_id")["published_cells"].astype(int)
    if not observed.sort_index().equals(published.sort_index()):
        raise ValueError("per-line selected cell counts disagree with published_cells")
    return joined, kinker


def map_genes_to_ensembl(
    source_symbols: pd.Index, gene_metadata_path: Path, vocab_path: Path
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    """Map gene symbols to Tx1-vocabulary Ensembl ids, one-to-one.

    Same resolution rule as the CPM script -- drop symbols that are
    ambiguous in the Tx1 metadata, keep only ids present in ``vocab.json``,
    and require ``token_id`` to agree with the vocab -- but the surviving
    column is named ``ensembl_id``, because these artifacts are meant to
    satisfy the Tx1 input contract rather than be excluded from it.
    """
    metadata = pd.read_parquet(gene_metadata_path)
    required = {"gene_symbol", "ensembl_id", "token_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Tx1 gene metadata is missing columns: {missing}")
    metadata["token_id"] = pd.to_numeric(metadata["token_id"], errors="raise")
    metadata = metadata.loc[metadata["token_id"] >= 3].copy()
    metadata = metadata.astype({"gene_symbol": str, "ensembl_id": str})
    if metadata["ensembl_id"].duplicated().any():
        raise ValueError("Tx1 gene metadata contains duplicate Ensembl IDs")
    ambiguous = set(
        metadata.loc[metadata["gene_symbol"].duplicated(False), "gene_symbol"]
    )
    unique = metadata.loc[~metadata["gene_symbol"].isin(ambiguous)].copy()
    vocab = json.loads(vocab_path.read_text())
    if not isinstance(vocab, dict):
        raise ValueError("Tx1 vocab.json must be a JSON object")
    unique = unique.loc[unique["ensembl_id"].isin(vocab)].copy()
    mismatch = unique.apply(
        lambda row: int(row["token_id"]) != int(vocab[row["ensembl_id"]]), axis=1
    )
    if len(unique) and mismatch.any():
        raise ValueError("Tx1 gene metadata token_id disagrees with vocab.json")
    by_symbol = unique.set_index("gene_symbol")
    keep = np.asarray([symbol in by_symbol.index for symbol in source_symbols])
    mapped = by_symbol.loc[source_symbols[keep]].copy()
    mapped.insert(0, "gene_symbol", source_symbols[keep].to_numpy())
    mapped.index = mapped["ensembl_id"].astype(str)
    mapped.index.name = None
    if mapped["ensembl_id"].duplicated().any():
        raise ValueError("gene mapping produced duplicate Ensembl IDs")
    audit = {
        "source_genes": int(len(source_symbols)),
        "ambiguous_metadata_symbols": int(len(ambiguous)),
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
    """Ingest the UMI matrix into per-line raw-count h5ads."""
    output_dir = args.output_dir.resolve()
    staging = output_dir.with_name(f".{output_dir.name}.building")
    if output_dir.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir} or {staging}")
    staging.mkdir(parents=True)
    try:
        selected, lines = _load_metadata(
            args.selected_cells,
            args.line_manifest,
            args.selection_matrix_name,
            args.manifest_matrix_semantics,
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
        matrix_sha256 = _sha256(args.matrix)
        if args.expected_matrix_sha256 and matrix_sha256 != args.expected_matrix_sha256:
            raise ValueError("UMI matrix disagrees with --expected-matrix-sha256")

        matrix_cells, row_metadata = read_matrix_head(args.matrix)
        if len(matrix_cells) != args.expected_source_cells:
            raise ValueError(
                f"expected {args.expected_source_cells} source cells, "
                f"found {len(matrix_cells)}"
            )
        selected_ids = selected["cell_id"].tolist()
        missing_cells = sorted(set(selected_ids) - set(matrix_cells))
        if missing_cells:
            raise ValueError(f"selected cells absent from matrix: {missing_cells[:10]}")
        checked = assert_line_labels_agree(
            matrix_cells, row_metadata["Cell_line"], selected
        )

        column_names = ["GENE", *matrix_cells]
        skip = 1 + len(METADATA_ROWS)
        gene_frame = pd.read_csv(
            args.matrix,
            sep="\t",
            names=column_names,
            skiprows=skip,
            usecols=["GENE"],
            dtype={"GENE": str},
        )
        source_symbols = pd.Index(gene_frame["GENE"].astype(str))
        if source_symbols.duplicated().any():
            duplicate = source_symbols[source_symbols.duplicated()].unique().tolist()
            raise ValueError(f"UMI matrix contains duplicate genes: {duplicate[:10]}")
        overlap = sorted(set(source_symbols) & set(METADATA_ROWS))
        if overlap:
            raise ValueError(f"metadata row label reappears as a gene: {overlap}")

        keep, var, gene_audit = map_genes_to_ensembl(
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
            names=column_names,
            skiprows=skip,
            usecols=["GENE", *selected_ids],
            dtype={cell_id: np.float32 for cell_id in selected_ids},
            chunksize=args.chunk_rows,
        )
        for chunk in chunks:
            symbols = chunk.pop("GENE").astype(str)
            expected_symbols = source_symbols[rows_seen : rows_seen + len(symbols)]
            if symbols.tolist() != expected_symbols.tolist():
                raise ValueError(
                    "UMI gene order changed between gene-only and numeric passes"
                )
            values = chunk.loc[:, selected_ids].to_numpy(copy=False)
            rows_seen += len(symbols)
            finite = finite and bool(np.isfinite(values).all())
            nonnegative = nonnegative and bool((values >= 0).all())
            full_library_sums += values.sum(axis=0, dtype=np.float64)
            present = values != 0
            nonzero += int(present.sum())
            noninteger += int((present & (np.abs(values - np.rint(values)) > 0)).sum())
            mapped = np.asarray([symbol in mapped_symbols for symbol in symbols])
            if mapped.any():
                matrix_parts.append(sparse.csr_matrix(values[mapped, :].T))
        if rows_seen != len(source_symbols):
            raise ValueError("UMI gene-only and numeric passes disagree in row count")
        numeric = {
            "finite": finite,
            "nonnegative": nonnegative,
            "nonzero_values": nonzero,
            "noninteger_nonzero_values": noninteger,
            "noninteger_nonzero_fraction": noninteger / nonzero if nonzero else 0.0,
        }
        if not finite or not nonnegative:
            raise ValueError("UMI matrix contains negative or non-finite values")
        if noninteger:
            raise ValueError(
                "UMI matrix is not raw counts: "
                f"{noninteger} of {nonzero} nonzero values are non-integer"
            )

        library_sum_audit = _quantiles(full_library_sums)
        library_sum_audit["cv"] = float(
            full_library_sums.std() / float(full_library_sums.mean())
        )
        selected["source_full_gene_library_sum"] = full_library_sums
        filtered = sparse.hstack(matrix_parts, format="csr", dtype=np.float32)

        hvg_payload = pickle.loads(args.hvg_var_dims.read_bytes())
        hvg_genes = np.asarray(hvg_payload["gene_names"], dtype=object).astype(str)
        if not len(hvg_genes):
            raise ValueError("HVG checkpoint gene order is empty")
        mapped_gene_symbols = set(var["gene_symbol"].astype(str))
        hvg_missing = [gene for gene in hvg_genes if gene not in mapped_gene_symbols]
        hvg_fill_rate = len(hvg_missing) / len(hvg_genes)
        if hvg_fill_rate > args.max_hvg_fill_rate:
            raise ValueError(
                f"HVG fill rate {hvg_fill_rate:.3f} exceeds "
                f"gate {args.max_hvg_fill_rate:.3f}"
            )

        h5ad_dir = staging / "h5ad"
        h5ad_dir.mkdir()
        var_for_h5ad = var[["gene_symbol", "ensembl_id"]].copy()
        var_for_h5ad.index = var["ensembl_id"].astype(str)
        var_for_h5ad.index.name = None
        var.to_csv(staging / "gene_mapping.tsv", sep="\t", index=False)
        source_hashes = {
            "matrix_sha256": matrix_sha256,
            "selected_cells_sha256": _sha256(args.selected_cells),
            "line_manifest_sha256": _sha256(args.line_manifest),
            "gene_metadata_sha256": _sha256(args.gene_metadata),
            "vocab_sha256": _sha256(args.vocab_json),
            "hvg_var_dims_sha256": _sha256(args.hvg_var_dims),
        }
        line_rows: list[dict[str, object]] = []
        line_qc: dict[str, object] = {}
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
                    "source_dataset": "Kinker/scCCLE SCP542 UMIcount_data.txt",
                    "tx1_usage": "standard",
                    "tx1_standard_raw_count_compatible": True,
                    "provenance_sha256": source_hashes,
                }
            )
            assert_tx1_input_contract(adata)
            qc = _line_qc(adata.X)
            qc["source_full_gene_library_sum"] = _quantiles(
                full_library_sums[row_indices]
            )
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
                if reread.uns["tx1_standard_raw_count_compatible"] is not True:
                    raise ValueError(f"{model_id} lost its raw-count semantic")
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
                    "tx1_usage": "standard",
                    "tx1_raw_count_contract": "passed",
                    "h5ad": str(path.relative_to(staging)),
                    "sha256": _sha256(path),
                }
            )

        pd.DataFrame(line_rows).to_csv(staging / "manifest.tsv", sep="\t", index=False)
        processing_config = {
            "matrix": str(args.matrix.resolve()),
            "selected_cells": str(args.selected_cells.resolve()),
            "line_manifest": str(args.line_manifest.resolve()),
            "gene_metadata": str(args.gene_metadata.resolve()),
            "vocab_json": str(args.vocab_json.resolve()),
            "hvg_var_dims": str(args.hvg_var_dims.resolve()),
            "output_dir": str(output_dir),
            "expression_semantics": SEMANTICS,
            "selection_matrix_name": args.selection_matrix_name,
            "manifest_matrix_semantics": args.manifest_matrix_semantics,
            "expected_lines": args.expected_lines,
            "expected_selected_cells": args.expected_selected_cells,
            "expected_source_cells": args.expected_source_cells,
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
            "cell_line_labels_cross_checked": checked,
            "source_numeric_audit": numeric,
            "source_full_gene_library_sum": library_sum_audit,
            "gene_mapping": gene_audit,
            "hvg_checkpoint_genes": int(len(hvg_genes)),
            "hvg_missing_genes": int(len(hvg_missing)),
            "hvg_fill_rate": hvg_fill_rate,
            "tx1_gene_mapping_contract": "passed",
            "tx1_raw_count_contract": "passed",
            "tx1_embedding_executed": False,
            "lines": line_qc,
            "source_hashes": source_hashes,
        }
        (staging / "qc.json").write_text(json.dumps(qc_payload, indent=2) + "\n")
        (staging / "status.json").write_text(
            json.dumps(
                {
                    "status": STATUS,
                    "tx1_raw_count_contract": "passed",
                    "reason": (
                        "SCP542 UMIcount_data.txt supplies raw UMI counts for every "
                        "selected Kinker cell; historical Exp13 protocol §6 branch 1"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        (staging / "README.md").write_text(
            "# Kinker/scCCLE 152-line raw UMI counts\n\n"
            "Per-line h5ads built from Broad Single Cell Portal SCP542 "
            "`UMIcount_data.txt`, replacing the CPM sensitivity artifacts for the "
            "standard Tx1 path. `var` is indexed by Tx1-vocabulary Ensembl ids and "
            "carries `ensembl_id`, so each file satisfies "
            "`assert_tx1_input_contract`. Values are integral UMI counts; the "
            "library size is a free quantity, not normalized.\n"
        )
        checksum_lines = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "sha256.txt":
                checksum_lines.append(f"{_sha256(path)}  {path.relative_to(staging)}")
        (staging / "sha256.txt").write_text("\n".join(checksum_lines) + "\n")
        (staging / "RAW_UMI_H5AD_VERIFIED").write_text(STATUS + "\n")
        staging.rename(output_dir)
        return qc_payload
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--selected-cells", type=Path, required=True)
    parser.add_argument("--line-manifest", type=Path, required=True)
    parser.add_argument("--gene-metadata", type=Path, required=True)
    parser.add_argument("--vocab-json", type=Path, required=True)
    parser.add_argument("--hvg-var-dims", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-matrix-sha256", default=None)
    parser.add_argument("--selection-matrix-name", default="GSE157220_CPM_data.txt.gz")
    parser.add_argument("--manifest-matrix-semantics", default="processed_cpm")
    parser.add_argument("--expected-lines", type=int, default=152)
    parser.add_argument("--expected-selected-cells", type=int, default=40_670)
    parser.add_argument("--expected-source-cells", type=int, default=56_982)
    parser.add_argument("--max-hvg-fill-rate", type=float, default=0.10)
    parser.add_argument("--min-detected-genes", type=int, default=1_000)
    parser.add_argument("--chunk-rows", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
