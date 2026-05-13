"""Feature construction for Replogle K562 dependency baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from dependency_baseline.config import BaselineConfig


@dataclass(frozen=True)
class FeaturePaths:
    features_npz: Path
    metadata_csv: Path
    qa_report_md: Path
    summary_json: Path


def build_features(config: BaselineConfig) -> FeaturePaths:
    """Build perturbation-level delta features from an AnnData file."""
    output_dir = config.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    features_npz = output_dir / "replogle_k562_delta_features.npz"
    metadata_csv = output_dir / "replogle_k562_feature_metadata.csv"
    qa_report_md = output_dir / "replogle_k562_feature_qa.md"
    summary_json = output_dir / "replogle_k562_feature_summary.json"

    overlap = pd.read_csv(config.data.overlap_csv)
    numeric_overlap = _numeric_training_rows(overlap, config)
    qa_nan = _matched_nan_rows(overlap, config)
    external_sanity = _external_overlap_sanity(config)
    labels = numeric_overlap["perturbation_gene"].astype(str).tolist()

    adata = ad.read_h5ad(config.data.h5ad_path, backed="r")
    try:
        obs_labels = adata.obs[config.data.obs_perturbation_col].astype(str).to_numpy()
        group_labels = [config.data.control_label, *labels]
        sums, counts = _chunked_group_sums(
            matrix=adata.X,
            obs_labels=obs_labels,
            group_labels=group_labels,
            chunk_size=config.features.chunk_size,
        )
        control_count = int(counts[0])
        if control_count == 0:
            msg = f"Control label {config.data.control_label!r} has no cells"
            raise ValueError(msg)

        control_mean = sums[0] / control_count
        perturb_counts = counts[1:].astype(np.int64)
        missing_cells = [
            label
            for label, count in zip(labels, perturb_counts, strict=True)
            if count == 0
        ]
        if missing_cells:
            msg = f"{len(missing_cells)} numeric labels have no cells in h5ad"
            raise ValueError(msg)

        perturb_means = sums[1:] / perturb_counts[:, None]
        delta = (perturb_means - control_mean[None, :]).astype(np.float32)
        burden = response_burden(delta, config.features.top_abs_delta_sizes)
        var_symbols = _var_symbols(adata, config.data.var_gene_symbol_col)
        target_indices = _target_indices(labels, var_symbols)
    finally:
        adata.file.close()

    metadata = numeric_overlap.copy()
    metadata["feature_row"] = np.arange(len(metadata))
    metadata["observed_n_cells"] = perturb_counts
    metadata["target_gene_index"] = target_indices
    metadata.to_csv(metadata_csv, index=False)

    np.savez_compressed(
        features_npz,
        delta=delta,
        response_burden=burden.to_numpy(dtype=np.float32),
        y=metadata[config.data.depmap_label_col].to_numpy(dtype=np.float32),
        n_cells=metadata["observed_n_cells"].to_numpy(dtype=np.float32),
        target_gene_index=np.asarray(target_indices, dtype=np.int32),
        perturbation_gene=np.asarray(labels, dtype=object),
        expression_gene_symbol=np.asarray(var_symbols, dtype=object),
        response_burden_columns=np.asarray(burden.columns.tolist(), dtype=object),
    )

    summary = {
        "n_overlap_rows": int(len(overlap)),
        "n_matched_rows": int(
            overlap[config.data.matched_label_col].fillna(False).sum()
        ),
        "n_numeric_training_rows": int(len(metadata)),
        "n_matched_nan_label_rows": int(len(qa_nan)),
        "n_expression_genes": int(delta.shape[1]),
        "control_label": config.data.control_label,
        "control_cells": control_count,
        "min_cells_per_perturbation": int(perturb_counts.min()),
        "median_cells_per_perturbation": float(np.median(perturb_counts)),
        "max_cells_per_perturbation": int(perturb_counts.max()),
        "n_missing_target_gene_indices": int(np.sum(np.asarray(target_indices) < 0)),
        "external_overlap_tables": external_sanity,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    qa_report_md.write_text(
        _qa_report(summary, qa_nan, external_sanity, config),
        encoding="utf-8",
    )

    return FeaturePaths(features_npz, metadata_csv, qa_report_md, summary_json)


def response_burden(delta: np.ndarray, top_abs_sizes: tuple[int, ...]) -> pd.DataFrame:
    """Compute TRADE-inspired scalar summaries from delta expression."""
    abs_delta = np.abs(delta)
    positive = np.clip(delta, 0, None)
    negative = np.clip(delta, None, 0)
    data: dict[str, np.ndarray] = {
        "delta_l1_mean": abs_delta.mean(axis=1),
        "delta_l2": np.linalg.norm(delta, axis=1),
        "delta_variance": delta.var(axis=1),
        "delta_positive_mean": positive.mean(axis=1),
        "delta_negative_abs_mean": np.abs(negative).mean(axis=1),
        "delta_max_abs": abs_delta.max(axis=1),
    }
    n_genes = delta.shape[1]
    for size in top_abs_sizes:
        k = min(size, n_genes)
        top_values = np.partition(abs_delta, n_genes - k, axis=1)[:, n_genes - k :]
        data[f"delta_top{k}_abs_mean"] = top_values.mean(axis=1)
    return pd.DataFrame(data)


def _numeric_training_rows(
    overlap: pd.DataFrame, config: BaselineConfig
) -> pd.DataFrame:
    matched = overlap[config.data.matched_label_col].fillna(False).astype(bool)
    numeric = overlap[config.data.depmap_label_col].notna()
    rows = overlap.loc[matched & numeric].copy()
    rows = rows.sort_values("perturbation_gene").reset_index(drop=True)
    return rows


def _matched_nan_rows(overlap: pd.DataFrame, config: BaselineConfig) -> pd.DataFrame:
    matched = overlap[config.data.matched_label_col].fillna(False).astype(bool)
    missing = overlap[config.data.depmap_label_col].isna()
    return overlap.loc[matched & missing].copy()


def _chunked_group_sums(
    matrix: object,
    obs_labels: np.ndarray,
    group_labels: list[str],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    group_to_index = {label: index for index, label in enumerate(group_labels)}
    row_group = np.full(obs_labels.shape[0], -1, dtype=np.int32)
    for label, index in group_to_index.items():
        row_group[obs_labels == label] = index

    n_groups = len(group_labels)
    n_genes = int(matrix.shape[1])
    sums = np.zeros((n_groups, n_genes), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)

    for start in tqdm(range(0, matrix.shape[0], chunk_size), desc="aggregate h5ad"):
        stop = min(start + chunk_size, matrix.shape[0])
        groups = row_group[start:stop]
        keep = groups >= 0
        if not np.any(keep):
            continue
        block = matrix[start:stop]
        if sparse.issparse(block):
            block_array = block.toarray()
        else:
            block_array = np.asarray(block)
        block_array = block_array[keep]
        groups = groups[keep]
        for group_index in np.unique(groups):
            mask = groups == group_index
            sums[group_index] += block_array[mask].sum(axis=0)
            counts[group_index] += int(mask.sum())
    return sums, counts


def _var_symbols(adata: ad.AnnData, column: str) -> list[str]:
    if column in adata.var.columns:
        return adata.var[column].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _target_indices(perturbation_genes: list[str], var_symbols: list[str]) -> list[int]:
    symbol_to_index = {symbol: index for index, symbol in enumerate(var_symbols)}
    return [symbol_to_index.get(gene, -1) for gene in perturbation_genes]


def _qa_report(
    summary: dict[str, object],
    qa_nan: pd.DataFrame,
    external_sanity: list[dict[str, object]],
    config: BaselineConfig,
) -> str:
    lines = [
        "# Replogle K562 Feature QA",
        "",
        "This report is generated by `vcc-dep-baseline build-features`.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        f"- `{key}`: {value}"
        for key, value in summary.items()
        if key != "external_overlap_tables"
    )
    lines.extend(
        [
            "",
            "## Matched Rows Excluded Because GeneEffect Is NaN",
            "",
            f"Rows with `{config.data.matched_label_col}=True` but missing "
            f"`{config.data.depmap_label_col}` are excluded from regression.",
            "",
        ]
    )
    if qa_nan.empty:
        lines.append("None.")
    else:
        cols = [
            col
            for col in (
                "perturbation_gene",
                "depmap_gene_column",
                config.data.n_cells_col,
            )
            if col in qa_nan.columns
        ]
        lines.append(qa_nan[cols].to_markdown(index=False))
    lines.extend(["", "## External Overlap Sanity Tables", ""])
    if external_sanity:
        lines.append(pd.DataFrame(external_sanity).to_markdown(index=False))
    else:
        lines.append("No external overlap tables configured.")
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "DepMap GeneEffect is a population-level long-term fitness label. "
            "The Replogle features are observed surviving/captured post-CRISPRi "
            "transcriptomes, so survivor bias and CRISPRi-to-CRISPR-KO mismatch "
            "must remain explicit in interpretation.",
            "",
        ]
    )
    return "\n".join(lines)


def _external_overlap_sanity(config: BaselineConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in config.data.external_overlap_csvs:
        if not path.exists():
            rows.append(
                {
                    "table": path.name,
                    "status": "missing",
                    "rows": np.nan,
                    "numeric_labels": np.nan,
                    "unique_numeric_genes": np.nan,
                }
            )
            continue
        overlap = pd.read_csv(path)
        numeric = overlap[config.data.depmap_label_col].notna()
        rows.append(
            {
                "table": path.name,
                "status": "present",
                "rows": int(len(overlap)),
                "numeric_labels": int(numeric.sum()),
                "unique_numeric_genes": int(
                    overlap.loc[numeric, "perturbation_gene"].astype(str).nunique()
                ),
            }
        )
    return rows
