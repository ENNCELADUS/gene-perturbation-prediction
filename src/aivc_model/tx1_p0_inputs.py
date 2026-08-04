"""Materialize authenticated, development-only inputs for Tx1 P0 baselines."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import numpy as np
import pandas as pd

from aivc_model.tx1_embed_cache import verify_cache
from aivc_model.tx1_geneeffect_eval import verify_artifact_hashes

PROTOCOL_ID: Final[str] = "tx1_geneeffect_p0_v1"
TRAIN_ROLE: Final[str] = "train_head"
K562_MODEL_ID: Final[str] = "ACH-000551"
EXPECTED_LINE_COUNT: Final[int] = 29
EXPECTED_GENE_COUNT: Final[int] = 587


@dataclass(frozen=True)
class P0InputsResult:
    """Materialized tables and their provenance payload."""

    gene_effect_long: pd.DataFrame
    copy_k562_prior: pd.DataFrame
    line_context: pd.DataFrame
    provenance: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_authorities(
    phase_a_dir: Path, manifest_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]:
    verify_artifact_hashes(phase_a_dir)
    frozen_manifest = phase_a_dir / "cell_line_manifest.csv"
    if _sha256(manifest_path) != _sha256(frozen_manifest):
        raise ValueError("manifest does not match frozen Phase-A manifest bytes")
    manifest = pd.read_csv(manifest_path)
    required = {"model_id", "role", "basal_source"}
    if not required.issubset(manifest.columns):
        raise ValueError(
            f"manifest is missing columns: {sorted(required - set(manifest))}"
        )
    if manifest["model_id"].isna().any() or manifest["model_id"].duplicated().any():
        raise ValueError("manifest model_id values must be complete and unique")
    train = manifest.loc[manifest["role"] == TRAIN_ROLE].copy()
    if len(train) != EXPECTED_LINE_COUNT:
        raise ValueError("P0 requires exactly 29 train_head lines")
    if set(train["basal_source"].astype(str)) != {"Tahoe-100M DMSO"}:
        raise ValueError("every P0 train_head line must be Tahoe-100M DMSO")
    slice_frame = pd.read_csv(phase_a_dir / "differentially_essential_slice.csv")
    needed_slice = {"depmap_column", "gene_symbol"}
    if not needed_slice.issubset(slice_frame.columns):
        raise ValueError("frozen differential slice is missing gene identifiers")
    if (
        len(slice_frame) != EXPECTED_GENE_COUNT
        or slice_frame["depmap_column"].isna().any()
        or slice_frame["gene_symbol"].isna().any()
        or slice_frame["depmap_column"].duplicated().any()
        or slice_frame["gene_symbol"].duplicated().any()
    ):
        raise ValueError("frozen differential slice must contain 587 unique genes")
    registration = json.loads(
        (phase_a_dir / "phase_a_registration.json").read_text(encoding="utf-8")
    )
    if not isinstance(registration, Mapping):
        raise ValueError("Phase-A registration root must be an object")
    return train.sort_values("model_id"), slice_frame, registration


def _load_gene_effect(
    path: Path,
    registration: Mapping[str, object],
    train_ids: list[str],
    slice_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    depmap_columns = slice_frame["depmap_column"].astype(str).tolist()
    expected_sha = str(registration["sources"]["depmap_gene_effect"]["sha256"])
    if _sha256(path) != expected_sha:
        raise ValueError("GeneEffect CSV SHA256 differs from Phase-A registration")
    header = pd.read_csv(path, nrows=0)
    if header.shape[1] < 2:
        raise ValueError("GeneEffect CSV has no gene columns")
    id_column = str(header.columns[0])
    ids = pd.read_csv(path, usecols=[id_column], dtype=str)[id_column]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("GeneEffect CSV model IDs must be complete and unique")
    wanted_ids = [*train_ids, K562_MODEL_ID]
    positions = {model_id: index for index, model_id in enumerate(ids)}
    missing_ids = sorted(set(wanted_ids) - set(positions))
    if missing_ids:
        raise ValueError(f"GeneEffect CSV is missing rows: {missing_ids}")
    missing_columns = sorted(set(depmap_columns) - set(header.columns.astype(str)))
    if missing_columns:
        raise ValueError(f"GeneEffect CSV is missing columns: {missing_columns}")
    selected_positions = {positions[model_id] for model_id in wanted_ids}
    selected = pd.read_csv(
        path,
        usecols=[id_column, *depmap_columns],
        index_col=id_column,
        skiprows=lambda row_number: (
            row_number != 0 and row_number - 1 not in selected_positions
        ),
    )
    selected.index = selected.index.astype(str)
    selected = selected.loc[wanted_ids, depmap_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    values = selected.to_numpy(dtype=float)
    if values.shape != (EXPECTED_LINE_COUNT + 1, EXPECTED_GENE_COUNT):
        raise ValueError("GeneEffect selection has an unexpected matrix shape")
    if not np.isfinite(values).all():
        raise ValueError("GeneEffect selection contains missing or non-finite values")
    symbol_by_column = dict(
        zip(
            slice_frame["depmap_column"].astype(str),
            slice_frame["gene_symbol"].astype(str),
            strict=True,
        )
    )
    long = (
        selected.loc[train_ids]
        .rename_axis(index="model_id", columns="depmap_column")
        .stack()
        .rename("gene_effect")
        .reset_index()
    )
    long["gene_symbol"] = long["depmap_column"].map(symbol_by_column)
    long = long[["model_id", "gene_symbol", "gene_effect"]].sort_values(
        ["model_id", "gene_symbol"], kind="stable"
    )
    prior = pd.DataFrame(
        {
            "gene_symbol": slice_frame["gene_symbol"].astype(str),
            "gene_effect": selected.loc[K562_MODEL_ID].to_numpy(dtype=float),
        }
    ).sort_values("gene_symbol", kind="stable")
    if long.duplicated(["model_id", "gene_symbol"]).any():
        raise ValueError("materialized GeneEffect table has duplicate keys")
    return long.reset_index(drop=True), prior.reset_index(drop=True)


def _pool_cache(
    cache_root: Path, manifest_path: Path, train_ids: list[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    report = verify_cache(cache_root, frozen_manifest_path=manifest_path)
    if report.get("status") != "verified":
        raise ValueError(
            f"Tx1/HVG cache verification failed: {report.get('discrepancies')}"
        )
    rows: list[dict[str, object]] = []
    hashes: dict[str, dict[str, str]] = {}
    expected_widths: tuple[int, int] | None = None
    for model_id in train_ids:
        line_dir = cache_root / model_id
        embedding_path = line_dir / "embeddings.npy"
        hvg_path = line_dir / "hvg.npy"
        embeddings = np.load(embedding_path, mmap_mode="r")
        hvg = np.load(hvg_path, mmap_mode="r")
        if embeddings.ndim != 2 or hvg.ndim != 2 or embeddings.shape[0] < 1:
            raise ValueError(f"line {model_id}: cache arrays must be non-empty 2D")
        if embeddings.shape[0] != hvg.shape[0]:
            raise ValueError(f"line {model_id}: Tx1/HVG cache row counts differ")
        widths = (int(embeddings.shape[1]), int(hvg.shape[1]))
        if expected_widths is None:
            expected_widths = widths
        elif widths != expected_widths:
            raise ValueError(f"line {model_id}: cache widths differ across lines")
        embedding_mean = np.asarray(embeddings, dtype=np.float64).mean(axis=0)
        hvg_mean = np.asarray(hvg, dtype=np.float64).mean(axis=0)
        if not np.isfinite(embedding_mean).all() or not np.isfinite(hvg_mean).all():
            raise ValueError(f"line {model_id}: pooled cache context is non-finite")
        row: dict[str, object] = {"model_id": model_id}
        row.update(
            {
                f"tx1_mean_{i:04d}": float(value)
                for i, value in enumerate(embedding_mean)
            }
        )
        row.update(
            {f"hvg_mean_{i:04d}": float(value) for i, value in enumerate(hvg_mean)}
        )
        rows.append(row)
        hashes[model_id] = {
            "embeddings.npy": _sha256(embedding_path),
            "hvg.npy": _sha256(hvg_path),
        }
    return pd.DataFrame(rows), hashes


def _add_expression(
    context: pd.DataFrame,
    path: Path,
    registration: Mapping[str, object],
    train_ids: list[str],
) -> pd.DataFrame:
    source = registration["sources"]["depmap_omics_expression"]
    expected = str(source["sha256"])
    if _sha256(path) != expected:
        raise ValueError("expression CSV SHA256 differs from Phase-A registration")
    expression = pd.read_csv(path, index_col=0)
    expression.index = expression.index.astype(str)
    if expression.index.duplicated().any() or not set(train_ids).issubset(
        expression.index
    ):
        raise ValueError("expression CSV must contain every train_head model_id once")
    selected = expression.loc[train_ids].apply(pd.to_numeric, errors="coerce")
    if selected.shape[1] < 1 or not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError(
            "selected expression context must be numeric, finite, and non-empty"
        )
    if selected.columns.astype(str).duplicated().any():
        raise ValueError("expression CSV has duplicate feature columns")
    selected.columns = [
        f"expression__{value}" for value in selected.columns.astype(str)
    ]
    return context.merge(selected.rename_axis("model_id").reset_index(), on="model_id")


def build_p0_inputs(
    *,
    phase_a_dir: Path,
    manifest_path: Path,
    gene_effect_path: Path,
    cache_root: Path,
    expression_path: Path | None = None,
) -> P0InputsResult:
    """Build authenticated P0 inputs without selecting any test-line label."""
    train, slice_frame, registration = _load_authorities(phase_a_dir, manifest_path)
    train_ids = train["model_id"].astype(str).tolist()
    gene_effect, prior = _load_gene_effect(
        gene_effect_path, registration, train_ids, slice_frame
    )
    context, cache_hashes = _pool_cache(cache_root, manifest_path, train_ids)
    if expression_path is not None:
        context = _add_expression(context, expression_path, registration, train_ids)
    if context["model_id"].tolist() != train_ids or context.isna().any().any():
        raise ValueError("line context lost train_head coverage")
    inputs = {
        "cell_line_manifest": _sha256(manifest_path),
        "phase_a_registration": _sha256(phase_a_dir / "phase_a_registration.json"),
        "differentially_essential_slice": _sha256(
            phase_a_dir / "differentially_essential_slice.csv"
        ),
        "depmap_gene_effect": _sha256(gene_effect_path),
    }
    if expression_path is not None:
        inputs["depmap_omics_expression"] = _sha256(expression_path)
    provenance: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "formal": False,
        "development_only": True,
        "test_lines_excluded": True,
        "target_role": TRAIN_ROLE,
        "n_lines": EXPECTED_LINE_COUNT,
        "n_genes": EXPECTED_GENE_COUNT,
        "k562_prior_model_id": K562_MODEL_ID,
        "input_sha256": dict(sorted(inputs.items())),
        "cache_array_sha256": cache_hashes,
        "context": {
            "pooling": "per-line arithmetic mean over cells",
            "label_free": True,
            "tx1_width": sum(column.startswith("tx1_mean_") for column in context),
            "hvg_width": sum(column.startswith("hvg_mean_") for column in context),
            "expression_width": sum(
                column.startswith("expression__") for column in context
            ),
        },
    }
    return P0InputsResult(gene_effect, prior, context, provenance)


def write_p0_inputs(result: P0InputsResult, output_dir: Path) -> None:
    """Atomically write materialized CSVs and stable provenance JSON."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite P0 inputs: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        tx1_columns = [
            column for column in result.line_context if column.startswith("tx1_mean_")
        ]
        hvg_columns = [
            column for column in result.line_context if column.startswith("hvg_mean_")
        ]
        outputs = {
            "gene_effect_long.csv": result.gene_effect_long,
            "copy_k562_prior.csv": result.copy_k562_prior,
            "line_context.csv": result.line_context,
            "tx1_context.csv": result.line_context[["model_id", *tx1_columns]],
            "hvg_context.csv": result.line_context[["model_id", *hvg_columns]],
        }
        output_hashes: dict[str, str] = {}
        for filename, frame in outputs.items():
            path = temporary / filename
            frame.to_csv(path, index=False)
            output_hashes[filename] = _sha256(path)
        provenance = {**result.provenance, "output_sha256": output_hashes}
        (temporary / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
