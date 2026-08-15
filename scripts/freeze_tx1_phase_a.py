#!/usr/bin/env python3
"""Freeze the Tx1/ST cross-line GeneEffect Phase-A manifest and registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ANCHORS = {
    "ACH-000551": ("K562", "CVCL_0004", "Myeloid"),
    "ACH-000971": ("HCT116", "CVCL_0291", "Bowel"),
    "ACH-000995": ("Jurkat", "CVCL_0065", "Lymphoid"),
    "ACH-000739": ("HepG2", "CVCL_0027", "Liver"),
}
K562_ID = "ACH-000551"
K_SCHEDULE = [0, 5, 10, 25, 50]
GENE_RE = re.compile(r"^(.*?) \((\d+)\)$")


@dataclass(frozen=True)
class PhaseAConfig:
    seed: int = 20260723
    test_size: int = 9
    slice_variance_quantile: float = 0.75
    dependency_cutoff: float = -0.5
    max_dependency_prevalence: float = 0.80
    min_dependent_lines: int = 2
    min_non_dependent_lines: int = 2
    power_repetitions: int = 1000
    bootstrap_repetitions: int = 1000
    utility_anchor_rho: float = 0.554
    utility_fraction: float = 0.10
    k_label_panels: int = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _allocate_test_counts(counts: pd.Series, test_size: int) -> dict[str, int]:
    eligible = counts[counts >= 2].sort_index()
    raw = eligible * test_size / int(counts.sum())
    allocation = np.floor(raw).astype(int).clip(upper=eligible - 1)
    while int(allocation.sum()) < test_size:
        candidates = [
            lineage
            for lineage in eligible.index
            if allocation[lineage] < eligible[lineage] - 1
        ]
        if not candidates:
            raise ValueError("Cannot allocate the requested hold-out size")
        lineage = max(
            candidates,
            key=lambda value: (raw[value] - allocation[value], counts[value], value),
        )
        allocation[lineage] += 1
    return {str(key): int(value) for key, value in allocation.items() if value}


def stratified_holdout(
    pool: pd.DataFrame, config: PhaseAConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = pool["lineage"].value_counts()
    allocation = _allocate_test_counts(counts, config.test_size)
    rng = np.random.default_rng(config.seed)
    held_out: list[str] = []
    for lineage in sorted(allocation):
        ids = sorted(pool.loc[pool["lineage"] == lineage, "model_id"])
        selected = rng.choice(ids, size=allocation[lineage], replace=False)
        held_out.extend(str(value) for value in selected)
    result = pool.copy()
    result["role"] = np.where(result["model_id"].isin(held_out), "test", "train_head")
    return result.sort_values("model_id").reset_index(drop=True), allocation


def _load_tahoe_pool(
    summary_path: Path, cell_metadata_path: Path, model_path: Path, gene_effect: Path
) -> pd.DataFrame:
    summary = json.loads(summary_path.read_text())
    tahoe_ids = set(summary["cell_line_counts"])
    metadata = pd.read_parquet(cell_metadata_path)
    depmap = pd.read_csv(model_path)
    gene_effect_ids = set(pd.read_csv(gene_effect, usecols=[0]).iloc[:, 0].astype(str))
    mapping = metadata[
        ["Cell_ID_Cellosaur", "Cell_ID_DepMap", "cell_name", "Organ"]
    ].drop_duplicates()
    mapping = mapping[mapping["Cell_ID_Cellosaur"].isin(tahoe_ids)]
    columns = ["ModelID", "RRID", "CellLineName", "OncotreeLineage"]
    depmap = depmap[columns].drop_duplicates()
    merged = mapping.merge(
        depmap,
        left_on=["Cell_ID_DepMap", "Cell_ID_Cellosaur"],
        right_on=["ModelID", "RRID"],
        validate="one_to_one",
    )
    merged = merged[merged["ModelID"].isin(gene_effect_ids)].copy()
    result = merged.rename(
        columns={
            "ModelID": "model_id",
            "RRID": "cellosaurus_id",
            "CellLineName": "cell_line_name",
            "OncotreeLineage": "lineage",
        }
    )
    result["dmso_cells"] = result["cellosaurus_id"].map(summary["cell_line_counts"])
    result["tx1_pretraining_exposure"] = "known_present"
    result["basal_source"] = "Tahoe-100M DMSO"
    return result[
        [
            "model_id",
            "cellosaurus_id",
            "cell_line_name",
            "lineage",
            "dmso_cells",
            "basal_source",
            "tx1_pretraining_exposure",
        ]
    ]


def _anchor_rows() -> pd.DataFrame:
    rows = []
    for model_id, (name, cellosaurus_id, lineage) in ANCHORS.items():
        rows.append(
            {
                "model_id": model_id,
                "cellosaurus_id": cellosaurus_id,
                "cell_line_name": name,
                "lineage": lineage,
                "dmso_cells": None,
                "basal_source": "Perturb-seq non-targeting control",
                "tx1_pretraining_exposure": "declared_separately",
                "role": "train_response_and_head",
            }
        )
    return pd.DataFrame(rows).sort_values("model_id")


def _read_gene_effect(path: Path, model_ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    missing = sorted(set(model_ids) - set(frame.index.astype(str)))
    if missing:
        raise ValueError(f"Missing GeneEffect rows: {missing}")
    frame.index = frame.index.astype(str)
    return frame.loc[model_ids].apply(pd.to_numeric, errors="coerce")


def define_differential_slice(
    gene_effect: pd.DataFrame, config: PhaseAConfig
) -> pd.DataFrame:
    n_lines = len(gene_effect)
    coverage = gene_effect.notna().sum()
    dependent = (gene_effect < config.dependency_cutoff).sum()
    non_dependent = (gene_effect >= config.dependency_cutoff).sum()
    prevalence = dependent / coverage.replace(0, np.nan)
    std = gene_effect.std(axis=0, ddof=1)
    eligible = (
        (coverage == n_lines)
        & (dependent >= config.min_dependent_lines)
        & (non_dependent >= config.min_non_dependent_lines)
        & (prevalence <= config.max_dependency_prevalence)
    )
    variance_cutoff = float(std[eligible].quantile(config.slice_variance_quantile))
    selected = eligible & (std >= variance_cutoff)
    records = []
    for column in gene_effect.columns[selected]:
        match = GENE_RE.match(str(column))
        records.append(
            {
                "depmap_column": str(column),
                "gene_symbol": match.group(1) if match else str(column),
                "entrez_id": int(match.group(2)) if match else None,
                "training_std": float(std[column]),
                "training_dependency_prevalence": float(prevalence[column]),
                "training_line_count": int(coverage[column]),
            }
        )
    result = pd.DataFrame(records).sort_values(["gene_symbol", "entrez_id"])
    result.attrs["variance_cutoff"] = variance_cutoff
    result.attrs["eligible_gene_count"] = int(eligible.sum())
    return result.reset_index(drop=True)


def _expression_rows(path: Path, model_ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "IsDefaultEntryForModel" in frame:
        frame = frame[frame["IsDefaultEntryForModel"].astype(bool)]
    frame = frame.drop_duplicates("ModelID", keep="first").set_index("ModelID")
    available = [model_id for model_id in model_ids if model_id in frame.index]
    if K562_ID not in available:
        raise ValueError("K562 is missing from OmicsExpression")
    metadata = {
        "Unnamed: 0",
        "SequencingID",
        "ModelConditionID",
        "IsDefaultEntryForMC",
        "IsDefaultEntryForModel",
    }
    return frame.loc[available].drop(columns=list(metadata & set(frame.columns)))


def _proxy_line_deltas(
    expression: pd.DataFrame,
    gene_effect: pd.DataFrame,
    slice_columns: list[str],
    evaluate_ids: list[str],
) -> np.ndarray:
    variable = expression.var(axis=0).nlargest(min(2000, expression.shape[1])).index
    x = StandardScaler().fit_transform(expression[variable])
    components = min(10, len(expression) - 2, x.shape[1])
    z = PCA(n_components=components, random_state=0).fit_transform(x)
    positions = {value: index for index, value in enumerate(expression.index)}
    k562 = gene_effect.loc[K562_ID, slice_columns]
    deltas = []
    for model_id in evaluate_ids:
        index = positions[model_id]
        candidates = [i for i in range(len(z)) if i != index]
        nearest = min(
            candidates,
            key=lambda other: float(np.linalg.norm(z[index] - z[other])),
        )
        target = gene_effect.loc[model_id, slice_columns]
        proxy = gene_effect.iloc[nearest][slice_columns]
        proxy_rho = float(spearmanr(proxy, target, nan_policy="omit").statistic)
        k562_rho = float(spearmanr(k562, target, nan_policy="omit").statistic)
        deltas.append(proxy_rho - k562_rho)
    return np.asarray(deltas, dtype=np.float64)


def _bootstrap_lower(values: np.ndarray, rng: np.random.Generator, reps: int) -> float:
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    return float(np.quantile(values[indices].mean(axis=1), 0.025))


def power_table(
    residuals: np.ndarray, config: PhaseAConfig
) -> tuple[pd.DataFrame, float, float, float]:
    centered = residuals - residuals.mean()
    rho_min = float(
        np.floor(config.utility_anchor_rho * config.utility_fraction * 100) / 100
    )
    alternative = float(
        max(rho_min, np.round(residuals.mean() - 0.5 * residuals.std(ddof=1), 2))
    )
    thresholds = np.round(np.arange(0.0, 0.201, 0.01), 2)
    true_deltas = np.round(np.arange(0.0, 0.301, 0.01), 2)
    rng = np.random.default_rng(config.seed + 1)
    rows = []
    for true_delta in true_deltas:
        lowers = []
        for _ in range(config.power_repetitions):
            sample = true_delta + rng.choice(
                centered, size=config.test_size, replace=True
            )
            lowers.append(_bootstrap_lower(sample, rng, config.bootstrap_repetitions))
        lowers_array = np.asarray(lowers)
        for threshold in thresholds:
            rows.append(
                {
                    "true_delta": float(true_delta),
                    "rho_min": float(threshold),
                    "power": float(np.mean(lowers_array > threshold)),
                }
            )
    table = pd.DataFrame(rows)
    at_alternative = table[np.isclose(table["true_delta"], alternative)]
    power_at_rho_min = float(
        at_alternative.loc[
            np.isclose(at_alternative["rho_min"], rho_min), "power"
        ].iloc[0]
    )
    if power_at_rho_min < 0.80:
        raise ValueError(
            f"Phase A is underpowered for rho_min={rho_min}: {power_at_rho_min}"
        )
    return table, rho_min, alternative, power_at_rho_min


def materialize_k_label_panels(
    test_ids: list[str], slice_columns: list[str], config: PhaseAConfig
) -> pd.DataFrame:
    """Freeze nested, result-independent k-label panels for every test line."""
    rows: list[dict[str, object]] = []
    max_k = max(K_SCHEDULE)
    for model_id in sorted(test_ids):
        for panel in range(config.k_label_panels):
            seed_material = f"{config.seed}:{model_id}:{panel}".encode()
            panel_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8], "big"
            )
            rng = np.random.default_rng(panel_seed)
            selected = rng.permutation(slice_columns)[:max_k]
            rows.extend(
                {
                    "model_id": model_id,
                    "panel": panel,
                    "panel_seed": panel_seed,
                    "label_order": order,
                    "depmap_column": gene,
                }
                for order, gene in enumerate(selected, start=1)
            )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def freeze(args: argparse.Namespace) -> dict[str, object]:
    config = PhaseAConfig(seed=args.seed, test_size=args.test_size)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pool = _load_tahoe_pool(
        args.tahoe_summary, args.cell_metadata, args.model, args.gene_effect
    )
    if len(pool) != 38:
        raise ValueError(f"Expected 38 eligible Tahoe lines, found {len(pool)}")
    split, allocation = stratified_holdout(pool, config)
    if set(split["model_id"]) & set(ANCHORS):
        raise ValueError("Tahoe pool overlaps a Perturb-seq anchor")
    manifest = pd.concat([split, _anchor_rows()], ignore_index=True)
    manifest = manifest.sort_values(["role", "model_id"]).reset_index(drop=True)
    training_ids = sorted(manifest.loc[manifest["role"] != "test", "model_id"])
    gene_effect = _read_gene_effect(args.gene_effect, training_ids)
    slice_frame = define_differential_slice(gene_effect, config)
    slice_columns = slice_frame["depmap_column"].tolist()
    all_manifest_ids = sorted(manifest["model_id"].astype(str))
    expression = _expression_rows(args.expression, all_manifest_ids)
    expression_ids = set(expression.index.astype(str))
    manifest["omics_expression_available"] = manifest["model_id"].isin(expression_ids)
    proxy_ids = sorted(
        model_id
        for model_id in split.loc[split["role"] == "train_head", "model_id"]
        if model_id in expression_ids
    )
    proxy_training_ids = [
        model_id for model_id in training_ids if model_id in expression_ids
    ]
    proxy_expression = expression.loc[proxy_training_ids]
    proxy_gene_effect = gene_effect.loc[proxy_training_ids]
    deltas = _proxy_line_deltas(
        proxy_expression, proxy_gene_effect, slice_columns, proxy_ids
    )
    power, rho_min, alternative, power_at_rho_min = power_table(deltas, config)
    test_ids = sorted(split.loc[split["role"] == "test", "model_id"])
    k_panels = materialize_k_label_panels(test_ids, slice_columns, config)

    manifest_path = output / "cell_line_manifest.csv"
    slice_path = output / "differentially_essential_slice.csv"
    proxy_path = output / "training_proxy_line_deltas.csv"
    power_path = output / "rho_min_power_simulation.csv"
    k_panels_path = output / "k_label_panels.csv"
    manifest.to_csv(manifest_path, index=False)
    slice_frame.to_csv(slice_path, index=False)
    pd.DataFrame({"model_id": proxy_ids, "proxy_delta_rho": deltas}).to_csv(
        proxy_path, index=False
    )
    power.to_csv(power_path, index=False)
    k_panels.to_csv(k_panels_path, index=False)

    source_paths = {
        "tahoe_summary": args.tahoe_summary,
        "tahoe_cell_metadata": args.cell_metadata,
        "depmap_model": args.model,
        "depmap_gene_effect": args.gene_effect,
        "depmap_omics_expression": args.expression,
    }
    gate_paths = {
        "batch_audit": args.batch_audit,
        "tx1_source_manifest": args.tx1_source_manifest,
        "tx1_width_verification": args.tx1_width_verification,
    }
    open_gates = [name for name, path in gate_paths.items() if path is None]
    gate_payloads = {
        name: json.loads(path.read_text())
        for name, path in gate_paths.items()
        if path is not None
    }
    if (
        "batch_audit" in gate_payloads
        and gate_payloads["batch_audit"].get("status") != "complete"
    ):
        raise ValueError("Batch audit is not complete")
    if (
        "tx1_source_manifest" in gate_payloads
        and gate_payloads["tx1_source_manifest"].get("status") != "verified"
    ):
        raise ValueError("Tx1 source manifest is not verified")
    if "tx1_width_verification" in gate_payloads:
        width = gate_payloads["tx1_width_verification"]
        if width.get("status") != "verified" or width.get("actual_obsm_width") != 2560:
            raise ValueError("Tx1 .obsm width is not verified at 2560")
        load = width.get("checkpoint_load", {})
        if not load.get("complete_backbone_load") or load.get("missing_keys"):
            raise ValueError("Tx1 checkpoint did not completely load the backbone")
        source = gate_payloads.get("tx1_source_manifest")
        if source and width.get("source_sha256", {}).get(
            "model_safetensors"
        ) != source.get("files", {}).get("model.safetensors", {}).get("sha256"):
            raise ValueError("Width probe and source manifest use different weights")
    registration = {
        "schema_version": 1,
        "status": "frozen" if not open_gates else "provisional_pending_phase_a_gates",
        "open_gates": open_gates,
        "design": "Tx1-conditioned ST GeneEffect backbone Phase A",
        "seed": config.seed,
        "eligible_tahoe_lines": 38,
        "tahoe_train_lines": int((split["role"] == "train_head").sum()),
        "tahoe_test_lines": int((split["role"] == "test").sum()),
        "perturbseq_anchor_lines": 4,
        "omics_expression_missing_ids": sorted(
            set(manifest["model_id"]) - expression_ids
        ),
        "omics_expression_missing_policy": (
            "ACH-001039 remains a Tahoe training line for Tx1/HVG/pseudobulk heads "
            "but is excluded from the CCLE-bulk baseline; no expression imputation"
        ),
        "lineage_test_allocation": allocation,
        "k_schedule": K_SCHEDULE,
        "gate_k": 10,
        "slice": {
            "defined_on": "training lines only",
            "complete_case_required": True,
            "dependency_cutoff": config.dependency_cutoff,
            "min_dependent_lines": config.min_dependent_lines,
            "min_non_dependent_lines": config.min_non_dependent_lines,
            "max_dependency_prevalence": config.max_dependency_prevalence,
            "variance_quantile": config.slice_variance_quantile,
            "variance_cutoff": slice_frame.attrs["variance_cutoff"],
            "eligible_before_variance_filter": slice_frame.attrs["eligible_gene_count"],
            "selected_gene_count": len(slice_frame),
        },
        "estimator": {
            "per_line_metric": "Spearman rank correlation",
            "paired_difference": "Tx1-3B-ST minus copy-K562 + k labels",
            "population_estimate": "unweighted macro mean over held-out lines",
            "inferential_unit": "cell line",
            "confidence_interval": "two-sided 95% percentile bootstrap over lines",
            "bootstrap_repetitions": 10000,
            "bootstrap_seed": config.seed + 2,
            "gate": "at k=10, bootstrap lower bound must exceed rho_min",
            "k_label_sampling": (
                "20 fixed nested panels per held-out line; panel seeds and ordered "
                "genes are materialized in k_label_panels.csv; k uses the first k"
            ),
            "scored_genes": (
                "differential slice minus that panel's k labels; target/prediction "
                "pairwise complete cases, minimum 100 genes"
            ),
            "ties": "average ranks as in scipy.stats.spearmanr",
            "panel_aggregation": (
                "mean Spearman within line across 20 panels, then paired method "
                "difference; bootstrap only the nine line-level means"
            ),
            "adapter_freeze_boundary": (
                "adapter architecture and hyperparameters are selected using "
                "training lines only in Phase E and hashed; all methods use "
                "the same panels"
            ),
        },
        "rho_min": rho_min,
        "rho_min_justification": {
            "utility_anchor_rho": config.utility_anchor_rho,
            "utility_fraction": config.utility_fraction,
            "utility_derivation": (
                "floor(0.10 * 0.554 * 100) / 100 = 0.05; 0.554 is the "
                "previously observed HCT116 retained GeneEffect transfer rho, and "
                "one tenth is the prespecified smallest worthwhile rank lift"
            ),
            "noise_source": (
                "training-only nearest-expression-line vs copy-K562 paired rho deltas"
            ),
            "proxy_delta_mean": float(deltas.mean()),
            "proxy_delta_std": float(deltas.std(ddof=1)),
            "decision_alternative": alternative,
            "decision_alternative_rule": (
                "training proxy mean minus one half between-line SD, rounded to "
                "0.01; no clipping or post-power tuning"
            ),
            "power_at_rho_min": power_at_rho_min,
            "selection_rule": (
                "rho_min is fixed by utility; simulation must show >=80% power at "
                "the training-derived conservative decision alternative or Phase A "
                "fails"
            ),
            "simulation_repetitions": config.power_repetitions,
            "inner_bootstrap_repetitions": config.bootstrap_repetitions,
        },
        "exposure": {
            "held_out_task_labels": "excluded from training",
            "held_out_tx1_pretraining": "known present for every Tahoe test line",
            "claim": "task-data-held-out cross-line GeneEffect transfer",
        },
        "batch_confound_sensitivity": {
            "primary": "unadjusted registered estimator",
            "batch_matched_head": (
                "retrain the GeneEffect head without the four Perturb-seq anchor "
                "lines, using Tahoe-source head-training lines only; ST response "
                "training on the four anchors is unchanged"
            ),
            "lineage_balanced_estimate": (
                "unweighted mean of the six held-out-lineage mean paired effects"
            ),
            "gate_policy": (
                "primary unadjusted estimator remains the sole gate; both fixed "
                "sensitivities are reported and cannot replace it"
            ),
            "not_used": (
                "no line-level source covariate adjustment because source is "
                "constant across all held-out Tahoe lines"
            ),
        },
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "artifacts": {
            "cell_line_manifest_sha256": sha256_file(manifest_path),
            "differential_slice_sha256": sha256_file(slice_path),
            "training_proxy_deltas_sha256": sha256_file(proxy_path),
            "power_simulation_sha256": sha256_file(power_path),
            "k_label_panels_sha256": sha256_file(k_panels_path),
            **{
                f"{name}_sha256": sha256_file(path)
                for name, path in gate_paths.items()
                if path is not None
            },
        },
        "phase_a_gate_evidence": gate_payloads,
    }
    registration["registration_payload_sha256"] = canonical_json_sha256(registration)
    _write_json(output / "phase_a_registration.json", registration)
    return registration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tahoe-summary", type=Path, required=True)
    parser.add_argument("--cell-metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-audit", type=Path)
    parser.add_argument("--tx1-source-manifest", type=Path)
    parser.add_argument("--tx1-width-verification", type=Path)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--test-size", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    registration = freeze(parse_args())
    print(json.dumps(registration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
