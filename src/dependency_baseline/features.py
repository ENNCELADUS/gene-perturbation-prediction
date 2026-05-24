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

from dependency_baseline.artifacts import (
    feature_metadata_path,
    feature_npz_path,
    feature_qa_path,
    feature_summary_path,
)
from dependency_baseline.config import BaselineConfig, ExternalFeatureSourceConfig
from dependency_baseline.program_scores import build_program_scores
from dependency_baseline.viability_axis import build_viability_axis_scores


@dataclass(frozen=True)
class FeaturePaths:
    features_npz: Path
    metadata_path: Path
    qa_report_md: Path
    summary_json: Path

    @property
    def metadata_csv(self) -> Path:
        return self.metadata_path


def build_features(config: BaselineConfig) -> FeaturePaths:
    """Build perturbation-level delta features from an AnnData file."""
    output_dir = config.data.output_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    features_npz = feature_npz_path(config.data.output_dir)
    metadata_path = feature_metadata_path(config.data.output_dir)
    qa_report_md = feature_qa_path(config.data.output_dir)
    summary_json = feature_summary_path(config.data.output_dir)

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
        program_scores = build_program_scores(
            delta=delta,
            gene_symbols=var_symbols,
            program_sets=config.features.program_score_sets,
        )
        viability_axis = build_viability_axis_scores(
            delta=delta,
            gene_symbols=var_symbols,
            config=config.viability_axis,
            default_cache_dir=config.data.output_dir
            / "external"
            / "nar_viability_axis",
        )
    finally:
        adata.file.close()

    metadata = numeric_overlap.copy()
    metadata["feature_row"] = np.arange(len(metadata))
    metadata["observed_n_cells"] = perturb_counts
    metadata["target_gene_index"] = target_indices
    metadata.to_parquet(metadata_path, index=False)

    feature_payload: dict[str, np.ndarray] = {
        "delta": delta,
        "response_burden": burden.to_numpy(dtype=np.float32),
        "program_scores": program_scores.scores.to_numpy(dtype=np.float32),
        "y": metadata[config.data.depmap_label_col].to_numpy(dtype=np.float32),
        "n_cells": metadata["observed_n_cells"].to_numpy(dtype=np.float32),
        "target_gene_index": np.asarray(target_indices, dtype=np.int32),
        "perturbation_gene": np.asarray(labels, dtype=object),
        "expression_gene_symbol": np.asarray(var_symbols, dtype=object),
        "response_burden_columns": np.asarray(burden.columns.tolist(), dtype=object),
        "program_score_columns": np.asarray(
            program_scores.score_columns,
            dtype=object,
        ),
    }
    viability_axis_qa: list[dict[str, object]] = []
    if viability_axis is not None:
        feature_payload["nar_viability_scores"] = viability_axis.scores.to_numpy(
            dtype=np.float32,
        )
        feature_payload["nar_viability_score_columns"] = np.asarray(
            viability_axis.score_columns,
            dtype=object,
        )
        viability_axis_qa = viability_axis.qa_rows

    np.savez_compressed(features_npz, **feature_payload)

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
        "viability_axis_enabled": bool(config.viability_axis.enabled),
        "viability_axis_score_columns": list(
            viability_axis.score_columns if viability_axis is not None else ()
        ),
        "viability_axis_models": viability_axis_qa,
        "program_score_columns": list(program_scores.score_columns),
        "program_score_sets": program_scores.qa_rows,
        "external_overlap_tables": external_sanity,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    qa_report_md.write_text(
        _qa_report(summary, qa_nan, external_sanity, config),
        encoding="utf-8",
    )

    return FeaturePaths(features_npz, metadata_path, qa_report_md, summary_json)


def build_external_features(
    config: BaselineConfig,
    reference_features_npz: Path,
    external_name: str = "adamson_k562",
) -> FeaturePaths:
    """Build an external feature pack aligned to a reference feature space."""
    if not config.data.external_feature_sources:
        msg = "No data.external_feature_sources configured"
        raise ValueError(msg)
    output_dir = config.data.output_dir / "features" / "external" / external_name
    output_dir.mkdir(parents=True, exist_ok=True)
    features_npz = output_dir / f"{external_name}_features.npz"
    metadata_path = output_dir / "feature_metadata.parquet"
    qa_report_md = output_dir / "feature_qa.md"
    summary_json = output_dir / "feature_summary.json"

    reference = np.load(reference_features_npz, allow_pickle=True)
    reference_genes = reference["expression_gene_symbol"].astype(str).tolist()
    overlap = pd.read_csv(config.data.overlap_csv)
    numeric_overlap = _numeric_training_rows(overlap, config)
    source_rows: list[pd.DataFrame] = []
    source_delta: list[np.ndarray] = []
    source_qa: list[dict[str, object]] = []

    for source in config.data.external_feature_sources:
        source_overlap = _external_source_overlap(numeric_overlap, source.name)
        rows, delta, qa = _build_external_source_rows(
            source=source,
            overlap=source_overlap,
            depmap_label_col=config.data.depmap_label_col,
            reference_genes=reference_genes,
            chunk_size=config.features.chunk_size,
        )
        source_qa.append(qa)
        if rows.empty:
            continue
        source_rows.append(rows)
        source_delta.append(delta)

    if not source_rows:
        msg = "No configured external source produced numeric perturbation rows"
        raise ValueError(msg)

    row_metadata = pd.concat(source_rows, ignore_index=True)
    row_delta = np.vstack(source_delta).astype(np.float32)
    metadata, delta = _aggregate_external_gene_rows(row_metadata, row_delta)
    burden = response_burden(delta, config.features.top_abs_delta_sizes)
    target_indices = _target_indices(
        metadata["perturbation_gene"].astype(str).tolist(),
        reference_genes,
    )
    program_scores = build_program_scores(
        delta=delta,
        gene_symbols=reference_genes,
        program_sets=config.features.program_score_sets,
    )
    metadata["feature_row"] = np.arange(len(metadata))
    metadata["target_gene_index"] = target_indices
    metadata.to_parquet(metadata_path, index=False)

    np.savez_compressed(
        features_npz,
        delta=delta.astype(np.float32),
        response_burden=burden.to_numpy(dtype=np.float32),
        program_scores=program_scores.scores.to_numpy(dtype=np.float32),
        y=metadata[config.data.depmap_label_col].to_numpy(dtype=np.float32),
        n_cells=metadata["observed_n_cells"].to_numpy(dtype=np.float32),
        target_gene_index=np.asarray(target_indices, dtype=np.int32),
        perturbation_gene=metadata["perturbation_gene"]
        .astype(str)
        .to_numpy(dtype=object),
        expression_gene_symbol=np.asarray(reference_genes, dtype=object),
        response_burden_columns=np.asarray(burden.columns.tolist(), dtype=object),
        program_score_columns=np.asarray(program_scores.score_columns, dtype=object),
        source_dataset=metadata["source_dataset"].astype(str).to_numpy(dtype=object),
        external_row_count=metadata["external_row_count"].to_numpy(dtype=np.int32),
    )

    observed_mask = ~np.isnan(delta).all(axis=0)
    summary = {
        "external_name": external_name,
        "n_source_rows": int(len(row_metadata)),
        "n_gene_rows": int(len(metadata)),
        "n_reference_genes": int(len(reference_genes)),
        "n_observed_reference_genes": int(observed_mask.sum()),
        "n_missing_reference_genes": int((~observed_mask).sum()),
        "min_cells_per_gene": int(metadata["observed_n_cells"].min()),
        "median_cells_per_gene": float(np.median(metadata["observed_n_cells"])),
        "max_cells_per_gene": int(metadata["observed_n_cells"].max()),
        "sources": source_qa,
        "program_score_columns": list(program_scores.score_columns),
        "program_score_sets": program_scores.qa_rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    qa_report_md.write_text(
        _external_qa_report(summary, metadata, config),
        encoding="utf-8",
    )

    return FeaturePaths(features_npz, metadata_path, qa_report_md, summary_json)


def response_burden(delta: np.ndarray, top_abs_sizes: tuple[int, ...]) -> pd.DataFrame:
    """Compute TRADE-inspired scalar summaries from delta expression."""
    abs_delta = np.abs(delta)
    positive = np.clip(delta, 0, None)
    negative = np.clip(delta, None, 0)
    observed = ~np.isnan(delta)
    observed_counts = np.maximum(observed.sum(axis=1), 1)
    data: dict[str, np.ndarray] = {
        "delta_l1_mean": np.nanmean(abs_delta, axis=1),
        "delta_l2": np.sqrt(np.nansum(delta * delta, axis=1)),
        "delta_variance": np.nanvar(delta, axis=1),
        "delta_positive_mean": np.nansum(positive, axis=1) / observed_counts,
        "delta_negative_abs_mean": np.nansum(np.abs(negative), axis=1)
        / observed_counts,
        "delta_max_abs": np.nanmax(abs_delta, axis=1),
    }
    n_genes = delta.shape[1]
    for size in top_abs_sizes:
        k = min(size, n_genes)
        partition_input = np.nan_to_num(abs_delta, nan=-np.inf)
        top_values = np.partition(partition_input, n_genes - k, axis=1)[
            :,
            n_genes - k :,
        ]
        top_values[top_values == -np.inf] = np.nan
        data[f"delta_top{k}_abs_mean"] = np.nanmean(top_values, axis=1)
    return pd.DataFrame(data).fillna(0.0)


def _build_external_source_rows(
    *,
    source: ExternalFeatureSourceConfig,
    overlap: pd.DataFrame,
    depmap_label_col: str,
    reference_genes: list[str],
    chunk_size: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    adata = ad.read_h5ad(source.h5ad_path, backed="r")
    try:
        label_col = _source_perturbation_label_col(overlap)
        source_overlap = (
            overlap.assign(source_perturbation_label=overlap[label_col].astype(str))
            .drop_duplicates(["source_perturbation_label", "perturbation_gene"])
            .sort_values("source_perturbation_label")
            .reset_index(drop=True)
        )
        labels = source_overlap["source_perturbation_label"].astype(str).tolist()
        label_metadata = source_overlap.set_index("source_perturbation_label")
        obs_labels = adata.obs[source.obs_perturbation_col].astype(str).to_numpy()
        control_label = _detect_control_label(obs_labels, source.control_label)
        group_labels = [control_label, *labels]
        sums, counts = _chunked_group_sums(
            matrix=adata.X,
            obs_labels=obs_labels,
            group_labels=group_labels,
            chunk_size=chunk_size,
        )
        if counts[0] == 0:
            msg = f"Control label {control_label!r} has no cells in {source.name}"
            raise ValueError(msg)
        control_mean = sums[0] / counts[0]
        keep = counts[1:] > 0
        kept_labels = np.asarray(labels, dtype=object)[keep].astype(str)
        if kept_labels.size == 0:
            qa = {
                "source_dataset": source.name,
                "h5ad_path": str(source.h5ad_path),
                "control_label": control_label,
                "control_cells": int(counts[0]),
                "numeric_labels_with_cells": 0,
            }
            return pd.DataFrame(), np.empty((0, len(reference_genes))), qa
        perturb_means = sums[1:][keep] / counts[1:][keep, None]
        source_delta = (perturb_means - control_mean[None, :]).astype(np.float32)
        source_genes = _var_symbols(adata, source.var_gene_symbol_col)
        aligned_delta = _align_delta_to_reference(
            source_delta,
            source_genes,
            reference_genes,
        )
        rows = pd.DataFrame(
            {
                "source_dataset": source.name,
                "source_perturbation_label": kept_labels,
                "perturbation_gene": label_metadata.loc[
                    kept_labels, "perturbation_gene"
                ]
                .astype(str)
                .to_numpy(),
                "observed_n_cells": counts[1:][keep].astype(int),
            }
        )
        rows["depmap_gene_effect"] = label_metadata.loc[
            kept_labels, depmap_label_col
        ].to_numpy(dtype=float)
        qa = {
            "source_dataset": source.name,
            "h5ad_path": str(source.h5ad_path),
            "control_label": control_label,
            "control_cells": int(counts[0]),
            "numeric_labels_with_cells": int(len(rows)),
            "expression_genes": int(len(source_genes)),
            "matched_reference_genes": int(
                len(set(source_genes).intersection(reference_genes))
            ),
        }
        return rows, aligned_delta, qa
    finally:
        adata.file.close()


def _external_source_overlap(overlap: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "source_dataset" not in overlap.columns:
        return overlap.copy()
    source_rows = overlap.loc[overlap["source_dataset"].astype(str) == source_name]
    if source_rows.empty:
        return overlap.copy()
    return source_rows.copy()


def _source_perturbation_label_col(overlap: pd.DataFrame) -> str:
    if "source_perturbation_label" in overlap.columns:
        return "source_perturbation_label"
    return "perturbation_gene"


def _detect_control_label(labels: np.ndarray, configured: str | None) -> str:
    unique = pd.Index(labels.astype(str)).unique().tolist()
    if configured:
        if configured not in set(unique):
            msg = f"Configured control label {configured!r} not found"
            raise ValueError(msg)
        return configured
    lower_to_label = {label.lower(): label for label in unique}
    for candidate in (
        "control",
        "non-targeting",
        "non-targeting control",
        "unperturbed",
        "mock",
    ):
        if candidate in lower_to_label:
            return lower_to_label[candidate]
    control_like = [
        label
        for label in unique
        if "control" in label.lower() or "non-target" in label.lower()
    ]
    if len(control_like) == 1:
        return str(control_like[0])
    msg = "Could not auto-detect a unique control label"
    raise ValueError(msg)


def _align_delta_to_reference(
    delta: np.ndarray,
    source_genes: list[str],
    reference_genes: list[str],
) -> np.ndarray:
    source_index = {}
    for index, gene in enumerate(source_genes):
        source_index.setdefault(gene, index)
    aligned = np.full((delta.shape[0], len(reference_genes)), np.nan, dtype=np.float32)
    for output_index, gene in enumerate(reference_genes):
        input_index = source_index.get(gene)
        if input_index is not None:
            aligned[:, output_index] = delta[:, input_index]
    return aligned


def _aggregate_external_gene_rows(
    metadata: pd.DataFrame,
    delta: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    deltas = []
    for gene, group in metadata.groupby("perturbation_gene", sort=True):
        indices = group.index.to_numpy()
        weights = group["observed_n_cells"].to_numpy(dtype=np.float64)
        weighted = delta[indices] * weights[:, None]
        denom = np.where(~np.isnan(delta[indices]), weights[:, None], 0.0).sum(axis=0)
        numerator = np.nansum(weighted, axis=0)
        gene_delta = np.divide(
            numerator,
            denom,
            out=np.full(delta.shape[1], np.nan, dtype=np.float64),
            where=denom > 0,
        )
        rows.append(
            {
                "perturbation_gene": gene,
                "depmap_gene_effect": float(group["depmap_gene_effect"].mean()),
                "observed_n_cells": int(group["observed_n_cells"].sum()),
                "source_dataset": ";".join(sorted(group["source_dataset"].unique())),
                "external_row_count": int(len(group)),
            }
        )
        deltas.append(gene_delta.astype(np.float32))
    return pd.DataFrame(rows), np.vstack(deltas).astype(np.float32)


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
        if key
        not in {
            "external_overlap_tables",
            "viability_axis_models",
            "program_score_sets",
        }
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
    lines.extend(["", "## NAR Viability-Axis Models", ""])
    viability_axis_rows = summary.get("viability_axis_models", [])
    if viability_axis_rows:
        lines.append(pd.DataFrame(viability_axis_rows).to_markdown(index=False))
    else:
        lines.append("Not enabled.")
    lines.extend(["", "## Program Score Sets", ""])
    program_score_rows = summary.get("program_score_sets", [])
    if program_score_rows:
        lines.append(pd.DataFrame(program_score_rows).to_markdown(index=False))
    else:
        lines.append("None configured.")
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


def _external_qa_report(
    summary: dict[str, object],
    metadata: pd.DataFrame,
    config: BaselineConfig,
) -> str:
    lines = [
        "# External Feature QA",
        "",
        "This report is generated by `vcc-dep-baseline build-external-features`.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        f"- `{key}`: {value}"
        for key, value in summary.items()
        if key not in {"sources", "program_score_sets"}
    )
    lines.extend(["", "## Sources", ""])
    sources = summary.get("sources", [])
    if sources:
        lines.append(pd.DataFrame(sources).to_markdown(index=False))
    else:
        lines.append("None.")
    lines.extend(["", "## Source Dataset Breakdown", ""])
    breakdown = (
        metadata.assign(source_dataset=metadata["source_dataset"].str.split(";"))
        .explode("source_dataset")
        .groupby("source_dataset", as_index=False)
        .agg(
            genes=("perturbation_gene", "nunique"),
            total_cells=("observed_n_cells", "sum"),
        )
    )
    lines.append(breakdown.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "External features are aligned to the Replogle reference gene order. "
            "Reference genes not observed in the external source are encoded as "
            "`NaN` and should be handled by trained imputers inside model pipelines.",
            "",
            f"Label column: `{config.data.depmap_label_col}`.",
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
