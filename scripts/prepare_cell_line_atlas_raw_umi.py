#!/usr/bin/env python3
"""Convert the 27 Breast/CCLA raw-UMI lines to audited Tx1-contract h5ad.

The input schema is deliberately supplied by JSON.  In particular, this script
never infers a cell line from a barcode or guesses whether rows are genes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from aivc_model.tx1_basal import assert_tx1_input_contract
from aivc_model.tx1_embed_cache import load_hvg_gene_order


ENSEMBL_RE = re.compile(r"^ENSG\d+$")
MANIFEST_COLUMNS = (
    "source",
    "model_id",
    "patient_id",
    "cell_line_name",
    "source_cell_line_name",
    "source_matrix",
    "matrix_semantics",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_one_column(path: Path, column: int) -> list[str]:
    frame = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if column < 0 or column >= frame.shape[1]:
        raise ValueError(f"{path}: column {column} outside {frame.shape[1]} columns")
    values = frame.iloc[:, column].astype(str).tolist()
    if not values or any(not value for value in values):
        raise ValueError(f"{path}: selected column is empty")
    return values


def _require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")


def _load_gene_mapping(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    path = Path(config["gene_metadata_path"])
    frame = pd.read_parquet(path)
    ensembl_col = str(config["gene_metadata_ensembl_column"])
    symbol_col = str(config["gene_metadata_symbol_column"])
    token_col = str(config["gene_metadata_token_column"])
    _require_columns(frame, [ensembl_col, symbol_col, token_col], path)
    frame = frame.loc[
        pd.to_numeric(frame[token_col], errors="raise")
        >= int(config["minimum_token_id"]),
        [ensembl_col, symbol_col, token_col],
    ].copy()
    frame[ensembl_col] = frame[ensembl_col].astype(str)
    frame[symbol_col] = frame[symbol_col].astype(str)
    bad = sorted(
        value
        for value in frame[ensembl_col].unique()
        if not ENSEMBL_RE.fullmatch(value)
    )
    if bad:
        raise ValueError(f"gene metadata has non-canonical Ensembl ids: {bad[:5]}")
    duplicated_ensembl = frame[frame[ensembl_col].duplicated(False)][
        ensembl_col
    ].unique()
    if len(duplicated_ensembl):
        raise ValueError(
            "gene metadata maps multiple vocabulary rows to Ensembl ids: "
            f"{sorted(duplicated_ensembl)[:5]}"
        )
    vocab_path = Path(config["tx1_vocab_path"])
    vocab = json.loads(vocab_path.read_text())
    if not isinstance(vocab, dict) or not all(
        isinstance(key, str) and isinstance(value, int) for key, value in vocab.items()
    ):
        raise ValueError(f"{vocab_path}: expected a JSON object mapping genes to ints")
    mismatches = frame.loc[
        frame.apply(
            lambda row: vocab.get(str(row[ensembl_col])) != int(row[token_col]),
            axis=1,
        )
    ]
    if not mismatches.empty:
        examples = mismatches[[ensembl_col, token_col]].head().to_dict("records")
        raise ValueError(
            "Tx1 gene_metadata token ids disagree with vocab.json; "
            f"examples: {examples}"
        )
    symbol_counts = frame.groupby(symbol_col)[ensembl_col].nunique()
    unique_symbols = set(symbol_counts[symbol_counts == 1].index.astype(str))
    symbol_to_ensembl = dict(
        frame.loc[
            frame[symbol_col].isin(unique_symbols), [symbol_col, ensembl_col]
        ].itertuples(index=False, name=None)
    )
    indexed = frame.set_index(ensembl_col, verify_integrity=True)
    indexed = indexed.rename(
        columns={symbol_col: "gene_symbol", token_col: "tx1_token_id"}
    )
    return indexed, symbol_to_ensembl


def _map_and_collapse_genes(
    matrix: sparse.spmatrix,
    source_ids: list[str],
    id_kind: str,
    gene_metadata: pd.DataFrame,
    symbol_to_ensembl: dict[str, str],
) -> tuple[sparse.csr_matrix, pd.DataFrame, dict[str, int]]:
    if matrix.shape[1] != len(source_ids):
        raise ValueError(
            f"matrix has {matrix.shape[1]} genes but feature file has {len(source_ids)}"
        )
    if id_kind == "ensembl":
        mapped = [
            value if value in gene_metadata.index else None for value in source_ids
        ]
    elif id_kind == "symbol":
        mapped = [symbol_to_ensembl.get(value) for value in source_ids]
    else:
        raise ValueError(f"unsupported gene_id_kind={id_kind!r}")
    kept = [index for index, value in enumerate(mapped) if value is not None]
    if not kept:
        raise ValueError("no source genes map unambiguously into the Tx1 vocabulary")
    mapped_kept = [str(mapped[index]) for index in kept]
    unique_ids = list(dict.fromkeys(mapped_kept))
    target_position = {value: index for index, value in enumerate(unique_ids)}
    projection = sparse.csr_matrix(
        (
            np.ones(len(kept), dtype=np.int8),
            ([*range(len(kept))], [target_position[value] for value in mapped_kept]),
        ),
        shape=(len(kept), len(unique_ids)),
    )
    collapsed = sparse.csr_matrix(matrix[:, kept], dtype=np.int64) @ projection
    collapsed = sparse.csr_matrix(collapsed, dtype=np.int64)
    source_by_target: dict[str, list[str]] = {value: [] for value in unique_ids}
    for source_index, ensembl_id in zip(kept, mapped_kept, strict=True):
        source_by_target[ensembl_id].append(source_ids[source_index])
    var = gene_metadata.loc[unique_ids, ["gene_symbol", "tx1_token_id"]].copy()
    var["ensembl_id"] = var.index.astype(str)
    var["source_gene_ids"] = ["|".join(source_by_target[value]) for value in unique_ids]
    var.index.name = "ensembl_id_index"
    stats = {
        "source_genes": len(source_ids),
        "mapped_source_genes": len(kept),
        "unmapped_source_genes": len(source_ids) - len(kept),
        "output_genes_before_empty_filter": len(unique_ids),
        "collapsed_duplicate_mappings": len(kept) - len(unique_ids),
        "duplicate_source_gene_ids": len(source_ids) - len(set(source_ids)),
    }
    return collapsed, var, stats


def _assert_raw_integer(matrix: sparse.spmatrix, label: str) -> None:
    data = sparse.csr_matrix(matrix).data
    if data.size and (not np.all(np.isfinite(data)) or np.any(data < 0)):
        raise ValueError(f"{label}: matrix has negative or non-finite values")
    if data.size and not np.all(data == np.floor(data)):
        raise ValueError(f"{label}: matrix is not integer-valued raw UMI")


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _mad_outliers(values: np.ndarray) -> dict[str, Any]:
    logged = np.log1p(values.astype(np.float64))
    median = float(np.median(logged))
    mad = float(np.median(np.abs(logged - median)))
    if mad == 0:
        return {"method": "log1p_median_5mad", "mad": mad, "low": 0, "high": 0}
    return {
        "method": "log1p_median_5mad",
        "mad": mad,
        "low": int(np.count_nonzero(logged < median - 5 * mad)),
        "high": int(np.count_nonzero(logged > median + 5 * mad)),
    }


def _build_adata(
    matrix: sparse.spmatrix,
    source_gene_ids: list[str],
    source_cell_ids: list[str],
    row: pd.Series,
    gene_id_kind: str,
    gene_metadata: pd.DataFrame,
    symbol_to_ensembl: dict[str, str],
    hvg_names: set[str],
) -> tuple[ad.AnnData, dict[str, Any]]:
    model_id = str(row["model_id"])
    if matrix.shape[0] != len(source_cell_ids):
        raise ValueError(
            f"{model_id}: matrix has {matrix.shape[0]} cells but received "
            f"{len(source_cell_ids)} cell ids"
        )
    if len(source_cell_ids) != len(set(source_cell_ids)):
        raise ValueError(f"{model_id}: duplicate source cell ids")
    _assert_raw_integer(matrix, model_id)
    matrix, var, gene_stats = _map_and_collapse_genes(
        matrix, source_gene_ids, gene_id_kind, gene_metadata, symbol_to_ensembl
    )
    empty_genes = np.asarray(matrix.getnnz(axis=0) == 0).ravel()
    n_empty_genes = int(empty_genes.sum())
    matrix = sparse.csr_matrix(matrix[:, ~empty_genes], dtype=np.int64)
    var = var.loc[~empty_genes].copy()
    library_size = np.asarray(matrix.sum(axis=1)).ravel()
    detected_genes = np.asarray(matrix.getnnz(axis=1)).ravel()
    if np.any(library_size == 0):
        raise ValueError(f"{model_id}: output contains empty cells after Tx1 mapping")
    obs_names = [f"{model_id}:{cell_id}" for cell_id in source_cell_ids]
    obs = pd.DataFrame(
        {
            "cell_type": str(row["cell_line_name"]),
            "model_id": model_id,
            "patient_id": str(row["patient_id"]),
            "cell_line_name": str(row["cell_line_name"]),
            "source": str(row["source"]),
            "source_cell_line_name": str(row["source_cell_line_name"]),
            "source_cell_id": source_cell_ids,
        },
        index=obs_names,
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    assert_tx1_input_contract(adata)
    present_symbols = set(adata.var["gene_symbol"].astype(str))
    missing_hvg = sorted(hvg_names - present_symbols)
    qc: dict[str, Any] = {
        "model_id": model_id,
        "source": str(row["source"]),
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "nnz": int(adata.X.nnz),
        "empty_cells": 0,
        "empty_genes_removed": n_empty_genes,
        "library_size": _distribution(library_size),
        "detected_genes": _distribution(detected_genes),
        "library_size_outliers": _mad_outliers(library_size),
        "detected_gene_outliers": _mad_outliers(detected_genes),
        "hvg_expected": len(hvg_names),
        "hvg_present": len(hvg_names) - len(missing_hvg),
        "hvg_fill_rate": len(missing_hvg) / len(hvg_names) if hvg_names else 0.0,
        "tx1_contract": "passed",
        **gene_stats,
    }
    return adata, qc


def _selected_for_line(
    selected: pd.DataFrame,
    source: dict[str, Any],
    model_id: str,
    expected_matrix: str,
    path: Path,
) -> list[str]:
    model_column = str(source["selected_model_id_column"])
    cell_column = str(source["selected_cell_id_column"])
    matrix_column = str(source["selected_matrix_column"])
    _require_columns(selected, [model_column, cell_column, matrix_column], path)
    subset = selected.loc[selected[model_column].astype(str) == model_id].copy()
    matrices = set(subset[matrix_column].astype(str))
    if matrices != {expected_matrix}:
        raise ValueError(
            f"{path}: {model_id} selected-cell matrix values {sorted(matrices)} "
            f"do not match manifest source_matrix={expected_matrix!r}"
        )
    result = subset[cell_column].astype(str).tolist()
    if not result:
        raise ValueError(f"{path}: no selected cells for {model_id}")
    if len(result) != len(set(result)):
        raise ValueError(f"{path}: duplicate selected cells for {model_id}")
    return result


def _process_breast(
    rows: pd.DataFrame,
    source: dict[str, Any],
    gene_metadata: pd.DataFrame,
    symbol_to_ensembl: dict[str, str],
    hvg_names: set[str],
) -> list[tuple[ad.AnnData, dict[str, Any]]]:
    selected_path = Path(source["selected_cells_path"])
    selected = pd.read_csv(selected_path, sep="\t", dtype=str)
    genes = _read_one_column(
        Path(source["features_path"]), int(source["feature_column"])
    )
    cells = _read_one_column(
        Path(source["barcodes_path"]), int(source["barcode_column"])
    )
    if len(cells) != len(set(cells)):
        raise ValueError(f"{source['barcodes_path']}: duplicate cell ids")
    matrix = sparse.csr_matrix(mmread(Path(source["matrix_path"]))).transpose().tocsr()
    if matrix.shape != (len(cells), len(genes)):
        expected_shape = (len(cells), len(genes))
        raise ValueError(
            f"Breast matrix shape {matrix.shape} != cells×genes {expected_shape}"
        )
    positions = {cell: index for index, cell in enumerate(cells)}
    output = []
    for _, row in rows.iterrows():
        expected_matrix = str(row["source_matrix"])
        if Path(source["matrix_path"]).name != expected_matrix:
            raise ValueError(
                f"Breast configured matrix {Path(source['matrix_path']).name!r} "
                f"does not match manifest {expected_matrix!r}"
            )
        chosen = _selected_for_line(
            selected,
            source,
            str(row["model_id"]),
            expected_matrix,
            selected_path,
        )
        missing = sorted(set(chosen) - set(positions))
        if missing:
            raise ValueError(f"Breast barcodes missing selected cells: {missing[:5]}")
        subset = matrix[[positions[cell] for cell in chosen], :]
        output.append(
            _build_adata(
                subset,
                genes,
                chosen,
                row,
                str(source["gene_id_kind"]),
                gene_metadata,
                symbol_to_ensembl,
                hvg_names,
            )
        )
    return output


def _process_ccla(
    rows: pd.DataFrame,
    source: dict[str, Any],
    gene_metadata: pd.DataFrame,
    symbol_to_ensembl: dict[str, str],
    hvg_names: set[str],
) -> list[tuple[ad.AnnData, dict[str, Any]]]:
    selected_path = Path(source["selected_cells_path"])
    selected = pd.read_csv(selected_path, sep="\t", dtype=str)
    source_root = Path(source["source_root"])
    gene_column = str(source["feature_id_column"])
    output = []
    for _, row in rows.iterrows():
        model_id = str(row["model_id"])
        chosen = _selected_for_line(
            selected,
            source,
            model_id,
            str(row["source_matrix"]),
            selected_path,
        )
        path = source_root / str(row["source_matrix"])
        frame = pd.read_csv(path, sep="\t")
        _require_columns(frame, [gene_column, *chosen], path)
        unexpected_duplicates = frame[gene_column].astype(str).duplicated().sum()
        genes = frame[gene_column].astype(str).tolist()
        values = frame.loc[:, chosen].apply(pd.to_numeric, errors="raise").to_numpy().T
        matrix = sparse.csr_matrix(values)
        adata, qc = _build_adata(
            matrix,
            genes,
            chosen,
            row,
            str(source["gene_id_kind"]),
            gene_metadata,
            symbol_to_ensembl,
            hvg_names,
        )
        qc["duplicate_rows_in_source_tsv"] = int(unexpected_duplicates)
        output.append((adata, qc))
    return output


def _load_hvgs(state_model_dir: Path) -> set[str]:
    names = load_hvg_gene_order(state_model_dir).astype(str).tolist()
    if not names or len(names) != len(set(names)):
        raise ValueError("checkpoint HVG gene_names are empty or duplicated")
    return set(names)


def _probe_tx1_loader(adata: ad.AnnData, config: dict[str, Any]) -> None:
    """Materialize one real Tx1 input batch without loading model weights or a GPU."""
    from omegaconf import OmegaConf
    from tahoe_x1.tokenizer import GeneVocab
    from tahoe_x1.utils.util import loader_from_adata

    model_dir = Path(config["tx1_model_dir"])
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    collator_config = OmegaConf.load(model_dir / "collator_config.yml")
    collator_config["use_chem_token"] = False
    if "drug_to_id_path" in collator_config:
        del collator_config["drug_to_id_path"]
    one_cell = adata[:1, :].copy()
    genes = one_cell.var["ensembl_id"].astype(str).tolist()
    gene_ids = np.asarray([vocab[gene] for gene in genes], dtype=int)
    loader = loader_from_adata(
        adata=one_cell,
        collator_cfg=collator_config,
        vocab=vocab,
        batch_size=1,
        max_length=int(config["tx1_loader_probe_max_length"]),
        gene_ids=gene_ids,
        num_workers=0,
        prefetch_factor=None,
    )
    try:
        next(iter(loader))
    except StopIteration as error:
        raise ValueError("Tx1 loader probe returned no batch") from error


def _write_output(
    output_dir: Path,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    built: list[tuple[ad.AnnData, dict[str, Any]]],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temp:
        root = Path(temp)
        (root / "h5ad").mkdir()
        (root / "qc").mkdir()
        (root / "provenance").mkdir()
        qcs = []
        for adata, qc in built:
            model_id = str(qc["model_id"])
            if bool(config["run_tx1_loader_probe"]):
                _probe_tx1_loader(adata, config)
                qc["tx1_loader_probe"] = "passed"
            else:
                qc["tx1_loader_probe"] = "not_run"
            h5ad_path = root / "h5ad" / f"{model_id}.h5ad"
            adata.write_h5ad(h5ad_path, compression="gzip")
            roundtrip = ad.read_h5ad(h5ad_path)
            assert_tx1_input_contract(roundtrip)
            if (
                roundtrip.shape != adata.shape
                or roundtrip.obs_names.tolist() != adata.obs_names.tolist()
            ):
                raise ValueError(
                    f"{model_id}: h5ad round-trip changed shape or cell order"
                )
            (root / "qc" / f"{model_id}.json").write_text(
                json.dumps(qc, indent=2, sort_keys=True) + "\n"
            )
            qcs.append(qc)
        manifest.to_csv(root / "manifest.csv", index=False)
        pd.DataFrame(qcs).to_csv(
            root / "qc" / "cell_line_qc.tsv", sep="\t", index=False
        )
        summary = {
            "status": "source_h5ad_verified_tx1_static_contract_only",
            "n_lines": len(qcs),
            "n_cells": sum(int(qc["n_cells"]) for qc in qcs),
            "sources": manifest["source"].value_counts().sort_index().to_dict(),
            "tx1_loader_probe": sorted(set(qc["tx1_loader_probe"] for qc in qcs)),
            "tx1_minimal_encoding": "not_run_gpu_resource_blocked",
            "input_file_count": len(config["_input_sha256"]),
            "hvg_fill_rate_range": [
                min(float(qc["hvg_fill_rate"]) for qc in qcs),
                max(float(qc["hvg_fill_rate"]) for qc in qcs),
            ],
            "config_sha256": _sha256(Path(config["_config_path"])),
        }
        (root / "provenance" / "config_snapshot.json").write_text(
            json.dumps(config["_config_snapshot"], indent=2, sort_keys=True) + "\n"
        )
        (root / "provenance" / "input_sha256.json").write_text(
            json.dumps(config["_input_sha256"], indent=2, sort_keys=True) + "\n"
        )
        (root / "qc" / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        (root / "README.md").write_text(
            "# Raw-UMI 27 h5ad\n\n"
            "Sparse per-line raw UMI matrices mapped to the Tx1 vocabulary. "
            "Cell membership comes only from the audited selected_cells tables. "
            "`SOURCE_H5AD_VERIFIED` certifies raw-count, mapping, QC, and static "
            "Tx1 input checks; it does not certify GPU encoding or embedding "
            "readiness. `sha256.txt` covers every regular bundle file created "
            "before the success sentinel, including h5ad, QC, manifest, config "
            "snapshot, and input-hash provenance; the sentinel itself is excluded.\n"
        )
        checksum_paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != "sha256.txt"
        )
        (root / "sha256.txt").write_text(
            "".join(
                f"{_sha256(path)}  {path.relative_to(root)}\n"
                for path in checksum_paths
            )
        )
        (root / "SOURCE_H5AD_VERIFIED").write_text(
            json.dumps(summary, sort_keys=True) + "\n"
        )
        Path(temp).replace(output_dir)


def _input_paths(config: dict[str, Any], manifest: pd.DataFrame) -> list[Path]:
    paths = {
        Path(config["manifest_path"]),
        Path(config["gene_metadata_path"]),
        Path(config["tx1_vocab_path"]),
        Path(config["state_model_dir"]) / "var_dims.pkl",
    }
    for source_name, source in config["sources"].items():
        rows = manifest[manifest["source"] == source_name]
        if rows.empty:
            continue
        paths.add(Path(source["selected_cells_path"]))
        if source["kind"] == "matrix_market_genes_by_cells":
            paths.update(
                {
                    Path(source["matrix_path"]),
                    Path(source["features_path"]),
                    Path(source["barcodes_path"]),
                }
            )
        elif source["kind"] == "wide_tsv_genes_by_cells":
            paths.update(
                Path(source["source_root"]) / str(value)
                for value in rows["source_matrix"].astype(str)
            )
    return sorted(paths)


def run(config_path: Path, output_dir: Path, only_model_id: str | None) -> None:
    config_snapshot = json.loads(config_path.read_text())
    config = dict(config_snapshot)
    config["_config_path"] = str(config_path)
    config["_config_snapshot"] = config_snapshot
    manifest_path = Path(config["manifest_path"])
    manifest = pd.read_csv(manifest_path, dtype=str)
    _require_columns(manifest, list(MANIFEST_COLUMNS), manifest_path)
    if manifest["model_id"].duplicated().any():
        raise ValueError("manifest contains duplicate model_id values")
    expected_sources = set(config["sources"])
    manifest = manifest[manifest["source"].isin(expected_sources)].copy()
    if only_model_id is not None:
        if only_model_id not in set(manifest["model_id"]):
            raise ValueError(
                f"--only-model-id {only_model_id!r} is absent from the configured "
                "raw-UMI manifest subset"
            )
        manifest = manifest[manifest["model_id"] == only_model_id].copy()
        expected_lines = 1
    else:
        expected_lines = int(config["expected_lines"])
    if len(manifest) != expected_lines:
        raise ValueError(f"selected {len(manifest)} lines; expected {expected_lines}")
    expected_semantics = str(config["expected_matrix_semantics"])
    if set(manifest["matrix_semantics"]) != {expected_semantics}:
        raise ValueError(
            f"manifest matrix_semantics must be exactly {expected_semantics!r}; got "
            f"{sorted(set(manifest['matrix_semantics']))}"
        )
    input_paths = _input_paths(config, manifest)
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"missing provenance inputs: {missing_inputs}")
    print(f"hashing {len(input_paths)} provenance inputs", flush=True)
    config["_input_sha256"] = {str(path): _sha256(path) for path in input_paths}
    gene_metadata, symbol_to_ensembl = _load_gene_mapping(config)
    hvg_names = _load_hvgs(Path(config["state_model_dir"]))
    built: list[tuple[ad.AnnData, dict[str, Any]]] = []
    for source_name, source in config["sources"].items():
        rows = manifest[manifest["source"] == source_name]
        if rows.empty:
            continue
        kind = str(source["kind"])
        print(f"processing {source_name}: {len(rows)} lines", flush=True)
        if kind == "matrix_market_genes_by_cells":
            built.extend(
                _process_breast(
                    rows, source, gene_metadata, symbol_to_ensembl, hvg_names
                )
            )
        elif kind == "wide_tsv_genes_by_cells":
            built.extend(
                _process_ccla(rows, source, gene_metadata, symbol_to_ensembl, hvg_names)
            )
        else:
            raise ValueError(f"unsupported source kind {kind!r}")
    if len(built) != expected_lines:
        raise ValueError(f"built {len(built)} lines; expected {expected_lines}")
    _write_output(output_dir, manifest, config, built)
    print(f"verified output: {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-model-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.output_dir, args.only_model_id)


if __name__ == "__main__":
    main()
