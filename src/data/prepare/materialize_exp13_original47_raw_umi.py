#!/usr/bin/env python3
"""Materialize the 47 original Exp13 contexts as capped raw-UMI h5ads."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
import uuid

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


from src.data.gene_splits import sha256_file  # noqa: E402
from src.data.basal import (
    _assemble_tahoe_matrix,
    _assemble_token_matrix,
    _filter_xatlas_shard,
    _materialize_rows,
    _row_to_xatlas_cell,
    assert_tx1_input_contract,
)


TAHOE_SOURCE = "Tahoe-100M DMSO"
RAW_UMI_SEMANTICS = "raw_umi_counts"
SPECIAL_MODEL_IDS = {
    "ACH-000696",  # OVCAR8
    "ACH-000779",  # PC9
    "ACH-000956",  # 22Rv1
    "ACH-001086",  # HeLa
    "ACH-002475",  # HAP1
}


def _stable_rank(seed: int, model_id: str, cell_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{model_id}\0{cell_id}".encode()).digest(), "big"
    )


def _stable_indices(
    cell_ids: np.ndarray, *, seed: int, model_id: str, max_cells: int
) -> np.ndarray:
    ranked = sorted(
        range(len(cell_ids)),
        key=lambda index: _stable_rank(seed, model_id, str(cell_ids[index])),
    )[:max_cells]
    return np.asarray(sorted(ranked), dtype=np.int64)


def _update_top_cells(
    heap: list[tuple[int, str, np.ndarray, np.ndarray]],
    *,
    rank: int,
    cell_id: str,
    genes: object,
    expressions: object,
    max_cells: int,
) -> None:
    entry = (
        -rank,
        cell_id,
        np.asarray(genes, dtype=np.int64),
        np.asarray(expressions),
    )
    if len(heap) < max_cells:
        heapq.heappush(heap, entry)
    elif rank < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _atomic_write_h5ad(adata: ad.AnnData, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.h5ad")
    try:
        adata.write_h5ad(temporary, compression="gzip")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_line(adata: ad.AnnData, path: Path, model_id: str) -> None:
    data = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X).ravel()
    if data.size and not np.equal(data, np.floor(data)).all():
        raise ValueError(f"{model_id}: source matrix is not raw integer UMI counts")
    if set(adata.obs["model_id"].astype(str)) != {model_id}:
        raise ValueError(f"{model_id}: materialized obs.model_id is inconsistent")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError(f"{model_id}: materialized axes are not unique")
    assert_tx1_input_contract(adata)
    _atomic_write_h5ad(adata, path)


def _materialize_tahoe(
    manifest: pd.DataFrame,
    shard_dir: Path,
    gene_metadata_path: Path,
    output_dir: Path,
    *,
    max_cells: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    tahoe = manifest.loc[manifest["basal_source"] == TAHOE_SOURCE].copy()
    if len(tahoe) != 38:
        raise ValueError(f"expected 38 Tahoe lines; observed {len(tahoe)}")
    by_cellosaurus = {
        str(row.cellosaurus_id): row for row in tahoe.itertuples(index=False)
    }
    heaps: dict[str, list[tuple[int, str, np.ndarray, np.ndarray]]] = {
        cellosaurus_id: [] for cellosaurus_id in by_cellosaurus
    }
    observed = dict.fromkeys(by_cellosaurus, 0)
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no Tahoe parquet shards under {shard_dir}")
    columns = ["genes", "expressions", "BARCODE_SUB_LIB_ID", "cell_line_id"]
    for shard_number, shard in enumerate(shards, start=1):
        frame = pd.read_parquet(shard, columns=columns)
        frame = frame.loc[frame["cell_line_id"].isin(by_cellosaurus)]
        for row_number, row in frame.iterrows():
            cellosaurus_id = str(row["cell_line_id"])
            info = by_cellosaurus[cellosaurus_id]
            cell_id = f"{shard.name}:{row_number}:{row['BARCODE_SUB_LIB_ID']}"
            observed[cellosaurus_id] += 1
            _update_top_cells(
                heaps[cellosaurus_id],
                rank=_stable_rank(seed, str(info.model_id), cell_id),
                cell_id=cell_id,
                genes=row["genes"],
                expressions=row["expressions"],
                max_cells=max_cells,
            )
        if shard_number % 100 == 0 or shard_number == len(shards):
            print(f"Tahoe shards: {shard_number}/{len(shards)}", flush=True)
    metadata = pd.read_parquet(gene_metadata_path).set_index("token_id")
    records: dict[str, dict[str, object]] = {}
    for cellosaurus_id, info in by_cellosaurus.items():
        model_id = str(info.model_id)
        selected = sorted(heaps[cellosaurus_id], key=lambda item: item[1])
        if len(selected) != max_cells:
            raise ValueError(
                f"{model_id}: expected {max_cells} Tahoe cells; got {len(selected)}"
            )
        matrix, var = _assemble_tahoe_matrix(
            [(item[2], item[3]) for item in selected], metadata
        )
        obs = pd.DataFrame(
            {
                "cell_type": str(info.cell_line_name),
                "cellosaurus_id": cellosaurus_id,
                "model_id": model_id,
                "basal_source": TAHOE_SOURCE,
            },
            index=[item[1] for item in selected],
        )
        adata = ad.AnnData(X=matrix, obs=obs, var=var)
        output = output_dir / f"{model_id}.h5ad"
        _write_line(adata, output, model_id)
        records[model_id] = {
            "source_kind": "tahoe_100m_dmso",
            "source_path": str(shard_dir.resolve()),
            "source_cell_count": int(observed[cellosaurus_id]),
            "selected_cell_count": int(adata.n_obs),
            "output_path": str(output.resolve()),
            "output_sha256": sha256_file(output),
        }
    return records


def _materialize_anchors(
    manifest: pd.DataFrame,
    source_config_path: Path,
    output_dir: Path,
    *,
    max_cells: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    configs = json.loads(source_config_path.read_text())
    records: dict[str, dict[str, object]] = {}
    by_id = manifest.set_index("model_id")
    if set(configs) != {"ACH-000551", "ACH-000739", "ACH-000971", "ACH-000995"}:
        raise ValueError("perturbseq source config does not contain the four anchors")
    for model_id, config in configs.items():
        info = by_id.loc[model_id]
        if config["source_type"] == "h5ad":
            source = Path(config["h5ad_path"])
            adata = _build_perturbseq_hash(
                source,
                config,
                info,
                max_cells=max_cells,
                seed=seed,
            )
        elif config["source_type"] == "xatlas_orion_parquet":
            source = Path(config["shard_dir"])
            adata = _build_xatlas_hash(
                config,
                info,
                model_id=model_id,
                max_cells=max_cells,
                seed=seed,
            )
        else:
            raise ValueError(f"{model_id}: unsupported anchor source type")
        output = output_dir / f"{model_id}.h5ad"
        _write_line(adata, output, model_id)
        records[model_id] = {
            "source_kind": str(config["source_type"]),
            "source_path": str(source.resolve()),
            "selected_cell_count": int(adata.n_obs),
            "output_path": str(output.resolve()),
            "output_sha256": sha256_file(output),
        }
    return records


def _build_perturbseq_hash(
    source: Path,
    config: dict[str, object],
    info: pd.Series,
    *,
    max_cells: int,
    seed: int,
) -> ad.AnnData:
    model_id = str(info.name)
    backed = ad.read_h5ad(source, backed="r")
    try:
        perturbation_col = str(config["perturbation_col"])
        control_label = str(config["control_label"])
        control_indices = np.flatnonzero(
            (backed.obs[perturbation_col].astype(str) == control_label).to_numpy()
        )
        if not control_indices.size:
            raise ValueError(f"{model_id}: no Perturb-seq control cells")
        cell_ids = backed.obs_names.astype(str).to_numpy()
        selected_local = _stable_indices(
            cell_ids[control_indices],
            seed=seed,
            model_id=model_id,
            max_cells=max_cells,
        )
        selected = control_indices[selected_local]
        matrix = sparse.csr_matrix(
            _materialize_rows(backed.X, selected, sparsify_chunks=True)
        )
        var = backed.var.copy()
    finally:
        backed.file.close()
    ensembl_col = str(config["var_ensembl_col"])
    if ensembl_col in var.columns:
        var.index = var[ensembl_col].astype(str)
    elif var.index.name != ensembl_col:
        raise ValueError(f"{model_id}: missing {ensembl_col!r} Ensembl axis")
    var["ensembl_id"] = var.index.astype(str)
    obs = pd.DataFrame(
        {
            "cell_type": str(info["cell_line_name"]),
            "cellosaurus_id": str(info["cellosaurus_id"]),
            "model_id": model_id,
            "basal_source": "Perturb-seq non-targeting control",
        },
        index=cell_ids[selected],
    )
    return ad.AnnData(X=matrix, obs=obs, var=var)


def _build_xatlas_hash(
    config: dict[str, object],
    info: pd.Series,
    *,
    model_id: str,
    max_cells: int,
    seed: int,
) -> ad.AnnData:
    shard_dir = Path(str(config["shard_dir"]))
    paths = sorted(shard_dir.glob(str(config["shard_glob"])))
    if not paths:
        raise FileNotFoundError(f"{model_id}: no X-Atlas shards")
    selected: list[tuple[int, str, object]] = []
    for path in paths:
        frame, _ = _filter_xatlas_shard(
            path,
            str(config["control_label"]),
            int(config["pass_guide_filter_value"]),
        )
        for row in frame.itertuples(index=False):
            cell = _row_to_xatlas_cell(row)
            cell_id = f"{cell.sample}:{cell.cell_barcode}"
            rank = _stable_rank(seed, model_id, cell_id)
            entry = (-rank, cell_id, cell)
            if len(selected) < max_cells:
                heapq.heappush(selected, entry)
            elif rank < -selected[0][0]:
                heapq.heapreplace(selected, entry)
    cells = [item[2] for item in sorted(selected, key=lambda item: item[1])]
    if len(cells) != max_cells:
        raise ValueError(f"{model_id}: insufficient X-Atlas control cells")
    metadata = pd.read_parquet(Path(str(config["gene_metadata_path"]))).set_index(
        "gene_token_id"
    )
    matrix, var = _assemble_token_matrix(
        [(cell.genes, cell.values) for cell in cells],
        metadata,
        metadata_var_columns=("ensembl_id", "gene_name"),
    )
    obs = pd.DataFrame(
        {
            "cell_type": str(info["cell_line_name"]),
            "cellosaurus_id": str(info["cellosaurus_id"]),
            "model_id": model_id,
            "basal_source": "Perturb-seq non-targeting control",
            "sample": [cell.sample for cell in cells],
        },
        index=[f"{cell.sample}:{cell.cell_barcode}" for cell in cells],
    )
    return ad.AnnData(X=matrix, obs=obs, var=var)


def _registered_specials(registry_path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(registry_path.read_text())
    rows = {
        str(row["model_id"]): row
        for row in payload["contexts"]
        if str(row["model_id"]) in SPECIAL_MODEL_IDS
    }
    if set(rows) != SPECIAL_MODEL_IDS:
        raise ValueError("context basal registry does not contain all five specials")
    return rows


def _copy_registered_h5ad(
    source: Path,
    info: dict[str, object],
    *,
    max_cells: int,
    seed: int,
) -> ad.AnnData:
    backed = ad.read_h5ad(source, backed="r")
    try:
        ids = backed.obs_names.astype(str).to_numpy()
        selected = _stable_indices(
            ids, seed=seed, model_id=str(info["model_id"]), max_cells=max_cells
        )
        result = backed[selected, :].to_memory()
    finally:
        backed.file.close()
    result.obs["cell_type"] = str(info["canonical_name"])
    result.obs["cellosaurus_id"] = str(info["cellosaurus_id"])
    result.obs["model_id"] = str(info["model_id"])
    result.obs["basal_source"] = str(info["basal_source"])
    if "ensembl_id" not in result.var.columns:
        result.var["ensembl_id"] = result.var_names.astype(str)
    return result


def _build_hap1(
    source: Path,
    info: dict[str, object],
    *,
    max_cells: int,
    seed: int,
) -> ad.AnnData:
    import scanpy as sc

    full = sc.read_10x_h5(source, gex_only=True)
    ids = full.obs_names.astype(str).to_numpy()
    selected = _stable_indices(
        ids, seed=seed, model_id=str(info["model_id"]), max_cells=max_cells
    )
    result = full[selected, :].copy()
    symbols = result.var_names.astype(str)
    ensembl = result.var["gene_ids"].astype(str).to_numpy()
    result.var.index = ensembl
    result.var["ensembl_id"] = ensembl
    result.var["gene_symbol"] = symbols
    result.obs["cell_type"] = str(info["canonical_name"])
    result.obs["cellosaurus_id"] = str(info["cellosaurus_id"])
    result.obs["model_id"] = str(info["model_id"])
    result.obs["basal_source"] = str(info["basal_source"])
    return result


def _map_symbols_to_ensembl(
    symbols: np.ndarray, metadata: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapping = metadata.loc[:, ["ensembl_id", "gene_symbol"]].dropna().copy()
    mapping["ensembl_id"] = mapping["ensembl_id"].astype(str)
    mapping["gene_symbol"] = mapping["gene_symbol"].astype(str)
    unique_symbols = ~mapping["gene_symbol"].duplicated(keep=False)
    unique_ensembl = ~mapping["ensembl_id"].duplicated(keep=False)
    mapping = mapping.loc[unique_symbols & unique_ensembl]
    lookup = mapping.set_index("gene_symbol")["ensembl_id"]
    source_indices = []
    ensembl_ids = []
    kept_symbols = []
    for index, symbol in enumerate(symbols.astype(str)):
        if symbol in lookup.index:
            source_indices.append(index)
            ensembl_ids.append(str(lookup.loc[symbol]))
            kept_symbols.append(symbol)
    if not source_indices:
        raise ValueError("22Rv1 symbols do not map to any unique Ensembl IDs")
    return (
        np.asarray(source_indices, dtype=np.int64),
        np.asarray(ensembl_ids, dtype=object),
        np.asarray(kept_symbols, dtype=object),
    )


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def _build_22rv1(
    source: Path,
    info: dict[str, object],
    gene_metadata_path: Path,
    *,
    max_cells: int,
    seed: int,
) -> ad.AnnData:
    with h5py.File(source) as handle:
        group = handle["GeneExpressionMatrix"]
        col_data = group["properties/colData"][:]
        mask = (col_data["Cell"] == b"22Rv1") & (col_data["TREATMENT"] == b"22Rv1-DMSO")
        candidates = np.flatnonzero(mask)
        cell_ids = _decode(group["properties/colnames"][:])
        selected_local = _stable_indices(
            cell_ids[candidates],
            seed=seed,
            model_id=str(info["model_id"]),
            max_cells=max_cells,
        )
        selected = candidates[selected_local]
        counts = group["assays/counts"]
        matrix = sparse.csc_matrix(
            (counts["data"][:], counts["indices"][:], counts["indptr"][:]),
            shape=(
                len(group["properties/rownames"]),
                len(group["properties/colnames"]),
            ),
        )[:, selected].T.tocsr()
        symbols = _decode(group["properties/rownames"][:])
    metadata = pd.read_parquet(gene_metadata_path)
    source_indices, ensembl_ids, kept_symbols = _map_symbols_to_ensembl(
        symbols, metadata
    )
    matrix = matrix[:, source_indices]
    obs = pd.DataFrame(
        {
            "cell_type": str(info["canonical_name"]),
            "cellosaurus_id": str(info["cellosaurus_id"]),
            "model_id": str(info["model_id"]),
            "basal_source": str(info["basal_source"]),
        },
        index=cell_ids[selected],
    )
    var = pd.DataFrame(
        {"ensembl_id": ensembl_ids, "gene_symbol": kept_symbols},
        index=ensembl_ids,
    )
    return ad.AnnData(X=matrix, obs=obs, var=var)


def _materialize_specials(
    registry_path: Path,
    gene_metadata_path: Path,
    output_dir: Path,
    *,
    max_cells: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    records = {}
    for model_id, info in _registered_specials(registry_path).items():
        source = Path(str(info["artifact_path"]))
        expected = str(info["artifact_sha256"])
        observed = sha256_file(source)
        if observed != expected:
            raise ValueError(f"{model_id}: registered source SHA-256 mismatch")
        if model_id in {"ACH-000696", "ACH-000779", "ACH-001086"}:
            result = _copy_registered_h5ad(source, info, max_cells=max_cells, seed=seed)
        elif model_id == "ACH-002475":
            result = _build_hap1(source, info, max_cells=max_cells, seed=seed)
        else:
            result = _build_22rv1(
                source,
                info,
                gene_metadata_path,
                max_cells=max_cells,
                seed=seed,
            )
        output = output_dir / f"{model_id}.h5ad"
        _write_line(result, output, model_id)
        records[model_id] = {
            "source_kind": "registered_special",
            "source_path": str(source.resolve()),
            "source_sha256": observed,
            "selected_cell_count": int(result.n_obs),
            "output_path": str(output.resolve()),
            "output_sha256": sha256_file(output),
        }
    return records


def materialize(args: argparse.Namespace) -> dict[str, object]:
    if (
        args.output_dir.exists()
        or args.registry_output.exists()
        or args.manifest_output.exists()
    ):
        raise FileExistsError("refusing to overwrite original-47 outputs")
    manifest = pd.read_csv(args.cell_line_manifest, dtype=str, keep_default_na=False)
    if len(manifest) != 42:
        raise ValueError(f"expected 42 Phase-A manifest rows; observed {len(manifest)}")
    args.output_dir.mkdir(parents=True)
    records = _materialize_tahoe(
        manifest,
        args.tahoe_shard_dir,
        args.tahoe_gene_metadata,
        args.output_dir,
        max_cells=args.max_cells,
        seed=args.seed,
    )
    records.update(
        _materialize_anchors(
            manifest,
            args.perturbseq_sources,
            args.output_dir,
            max_cells=args.max_cells,
            seed=args.seed,
        )
    )
    records.update(
        _materialize_specials(
            args.context_registry,
            args.tahoe_gene_metadata,
            args.output_dir,
            max_cells=args.max_cells,
            seed=args.seed,
        )
    )
    if len(records) != 47:
        raise ValueError(f"expected 47 materialized lines; observed {len(records)}")
    registry = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "source_path": records[model_id]["output_path"],
                "source_kind": "h5ad",
                "matrix_semantics": RAW_UMI_SEMANTICS,
            }
            for model_id in sorted(records)
        ]
    )
    args.registry_output.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.registry_output, index=False)
    payload = {
        "schema_version": "exp13-original47-raw-umi-v1",
        "seed": args.seed,
        "max_cells": args.max_cells,
        "selection_algorithm": "sha256_seed_model_id_cell_id_v1",
        "cell_line_manifest": str(args.cell_line_manifest),
        "cell_line_manifest_sha256": sha256_file(args.cell_line_manifest),
        "records": records,
        "registry_path": str(args.registry_output),
        "registry_sha256": sha256_file(args.registry_output),
    }
    args.manifest_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-line-manifest",
        type=Path,
        default=Path("results/phase_a_tx1_20260724/cell_line_manifest.csv"),
    )
    parser.add_argument(
        "--perturbseq-sources",
        type=Path,
        default=Path("configs/experiments/13_geneeffect_226/perturbseq_sources.json"),
    )
    parser.add_argument(
        "--context-registry",
        type=Path,
        default=Path("configs/benchmarks/context_screen_v2_basal_registry.json"),
    )
    parser.add_argument(
        "--tahoe-shard-dir",
        type=Path,
        default=Path("data/sl_dependency_v0/raw/tahoe_100m_dmso/shards"),
    )
    parser.add_argument(
        "--tahoe-gene-metadata",
        type=Path,
        default=Path(
            "data/sl_dependency_v0/raw/tahoe_100m_metadata/gene_metadata.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sl_dependency_v0/processed/exp13_original47_raw_umi"),
    )
    parser.add_argument(
        "--registry-output",
        type=Path,
        default=Path("data/sl_dependency_v0/processed/exp13_original47_registry.csv"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/sl_dependency_v0/processed/exp13_original47_manifest.json"),
    )
    parser.add_argument("--max-cells", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    payload = materialize(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
