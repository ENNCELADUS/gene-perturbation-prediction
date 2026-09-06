"""Shared validation and MILP partition helpers for the cell-line benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


REQUIRED_COLUMNS = (
    "source",
    "model_id",
    "patient_id",
    "cell_line_name",
    "lineage",
    "matrix_semantics",
    "same_patient_as_excluded",
)
SPLITS = ("train", "val", "test")
EXPECTED_SOURCE_SEMANTICS = {
    "breast_cell_line_atlas": "raw_umi_counts",
    "ccla_omix005191": "raw_umi_counts",
    "kinker_sccle": "processed_cpm",
}
EXPECTED_179_MANIFEST_SHA256 = (
    "12ac575e1deea1e8343fa05c04b4bd590c39534d004a682a1d0ba297055b8a37"
)
EXPECTED_179_SELECTION_AUDIT_SHA256 = (
    "7b1ebf963f4897972439e2c337a0bac7d25f40d9f27adf64a9de1b04496a719b"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_manifest(frame: pd.DataFrame, expected_lines: int) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    frame = frame.copy()
    if len(frame) != expected_lines:
        raise ValueError(f"expected {expected_lines} manifest rows, found {len(frame)}")
    for column in ("model_id", "patient_id", "source", "lineage"):
        if (
            frame[column].isna().any()
            or frame[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"manifest has missing/blank {column}")
        frame[column] = frame[column].astype(str)
    if frame["model_id"].duplicated().any():
        raise ValueError("manifest has duplicate model_id values")
    normalized = frame["same_patient_as_excluded"].astype(str).str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("same_patient_as_excluded must contain only true/false")
    frame["same_patient_as_excluded"] = normalized == "true"
    if expected_lines == 179:
        observed = frame.groupby("source")["matrix_semantics"].unique().to_dict()
        expected = {key: [value] for key, value in EXPECTED_SOURCE_SEMANTICS.items()}
        if observed != expected:
            raise ValueError(
                f"source/matrix semantics mismatch: {observed} != {expected}"
            )
    return frame


def _largest_remainder(values: pd.Series, target: int) -> dict[str, int]:
    exact = values.astype(float) * target / float(values.sum())
    result = np.floor(exact).astype(int)
    remaining = target - int(result.sum())
    order = sorted(values.index, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return {str(key): int(result[key]) for key in values.index}


def _source_targets(
    totals: pd.Series, counts: dict[str, int], n_lines: int
) -> dict[str, dict[str, int]]:
    expected_sources = {
        "breast_cell_line_atlas",
        "ccla_omix005191",
        "kinker_sccle",
    }
    if (
        n_lines == 179
        and counts == {"train": 125, "val": 27, "test": 27}
        and set(totals.index) == expected_sources
        and totals.to_dict()
        == {
            "breast_cell_line_atlas": 14,
            "ccla_omix005191": 13,
            "kinker_sccle": 152,
        }
    ):
        return {
            "train": {
                "breast_cell_line_atlas": 8,
                "ccla_omix005191": 7,
                "kinker_sccle": 110,
            },
            "val": {
                "breast_cell_line_atlas": 2,
                "ccla_omix005191": 2,
                "kinker_sccle": 23,
            },
            "test": {
                "breast_cell_line_atlas": 4,
                "ccla_omix005191": 4,
                "kinker_sccle": 19,
            },
        }
    val = _largest_remainder(totals, counts["val"])
    test = _largest_remainder(totals, counts["test"])
    train = {
        str(name): int(totals[name]) - val[str(name)] - test[str(name)]
        for name in totals.index
    }
    return {"train": train, "val": val, "test": test}


def _group_arrays(
    manifest: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    sources = sorted(manifest["source"].unique())
    lineages = sorted(manifest["lineage"].unique())
    source_pos = {value: index for index, value in enumerate(sources)}
    lineage_pos = {value: index for index, value in enumerate(lineages)}
    patient_ids: list[str] = []
    sizes: list[int] = []
    source_counts: list[np.ndarray] = []
    lineage_counts: list[np.ndarray] = []
    for patient_id, rows in manifest.groupby("patient_id", sort=True):
        patient_ids.append(str(patient_id))
        sizes.append(len(rows))
        source = np.zeros(len(sources), dtype=np.int16)
        lineage = np.zeros(len(lineages), dtype=np.int16)
        for value, count in rows["source"].value_counts().items():
            source[source_pos[str(value)]] = int(count)
        for value, count in rows["lineage"].value_counts().items():
            lineage[lineage_pos[str(value)]] = int(count)
        source_counts.append(source)
        lineage_counts.append(lineage)
    return (
        patient_ids,
        np.asarray(sizes, dtype=np.int16),
        np.stack(source_counts),
        np.stack(lineage_counts),
    )


def assign_splits(
    manifest: pd.DataFrame,
    *,
    counts: dict[str, int],
    seed: int,
) -> tuple[pd.Series, dict[str, dict[str, int]], dict[str, object]]:
    """Globally optimize a deterministic patient-grouped assignment."""
    if sum(counts.values()) != len(manifest) or any(counts[key] <= 0 for key in SPLITS):
        raise ValueError("split counts must be positive and sum to manifest rows")
    patients, sizes, source_counts, lineage_counts = _group_arrays(manifest)
    patient_pos = {value: index for index, value in enumerate(patients)}
    patient_sizes = manifest.groupby("patient_id").size()
    forced_patients = sorted(
        set(manifest.loc[manifest["same_patient_as_excluded"], "patient_id"])
        | set(patient_sizes[patient_sizes > 1].index)
    )
    forced_indices = {patient_pos[value] for value in forced_patients}
    candidates = np.asarray(
        [index for index in range(len(patients)) if index not in forced_indices],
        dtype=int,
    )
    source_totals = manifest["source"].value_counts().sort_index()
    source_targets = _source_targets(source_totals, counts, len(manifest))
    source_names = sorted(source_totals.index)
    target_source_vector = {
        split: np.asarray([source_targets[split][name] for name in source_names])
        for split in ("val", "test")
    }
    # Balance lineage within each evidence source. Absolute deviations are
    # linearized, so scipy's MILP solver proves the primary optimum instead of
    # depending on a finite random search.
    strata = sorted(
        set(zip(manifest["source"].astype(str), manifest["lineage"].astype(str)))
    )
    stratum_pos = {value: index for index, value in enumerate(strata)}
    group_strata = np.zeros((len(patients), len(strata)), dtype=float)
    for patient_index, patient in enumerate(patients):
        rows = manifest.loc[manifest["patient_id"] == patient]
        for key, value in rows.groupby(["source", "lineage"]).size().items():
            group_strata[patient_index, stratum_pos[(str(key[0]), str(key[1]))]] = value
    n = len(candidates)
    n_deviation = 2 * len(strata)
    n_variables = 2 * n + n_deviation
    objective = np.zeros(n_variables)
    objective[2 * n :] = 1.0
    lower = np.zeros(n_variables)
    upper = np.concatenate([np.ones(2 * n), np.full(n_deviation, np.inf)])
    integrality = np.concatenate([np.ones(2 * n), np.zeros(n_deviation)])
    rows: list[np.ndarray] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add(coefficients: np.ndarray, lo: float, hi: float) -> None:
        rows.append(coefficients)
        row_lower.append(lo)
        row_upper.append(hi)

    for local_index in range(n):
        row = np.zeros(n_variables)
        row[local_index] = 1.0
        row[n + local_index] = 1.0
        add(row, -np.inf, 1.0)
    for split_index, split in enumerate(("val", "test")):
        offset = split_index * n
        row = np.zeros(n_variables)
        row[offset : offset + n] = sizes[candidates]
        add(row, counts[split], counts[split])
        for source_index, source in enumerate(source_names):
            row = np.zeros(n_variables)
            row[offset : offset + n] = source_counts[candidates, source_index]
            target = target_source_vector[split][source_index]
            add(row, target, target)
        for stratum_index, (source, _lineage) in enumerate(strata):
            source_total = float(source_totals[source])
            stratum_total = float(group_strata[:, stratum_index].sum())
            target = stratum_total * source_targets[split][source] / source_total
            deviation_index = 2 * n + split_index * len(strata) + stratum_index
            observed = group_strata[candidates, stratum_index]
            row = np.zeros(n_variables)
            row[offset : offset + n] = observed
            row[deviation_index] = -1.0
            add(row, -np.inf, target)
            row = np.zeros(n_variables)
            row[offset : offset + n] = -observed
            row[deviation_index] = -1.0
            add(row, -np.inf, -target)
    constraints = LinearConstraint(np.stack(rows), row_lower, row_upper)
    primary = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": 300.0},
    )
    if not primary.success or primary.x is None:
        raise ValueError(f"MILP split optimization failed: {primary.message}")
    # Hold the proven balance optimum and use a stable hash-derived objective to
    # choose one reproducible member when multiple optima exist.
    optimum_row = objective.copy()
    stable = np.zeros(n_variables)
    for split_index, split in enumerate(("val", "test")):
        for local_index, patient_index in enumerate(candidates):
            token = f"{seed}:{split}:{patients[int(patient_index)]}".encode()
            stable[split_index * n + local_index] = int.from_bytes(
                hashlib.sha256(token).digest()[:8], "big"
            ) / float(2**64)
    secondary_constraints = LinearConstraint(
        np.vstack([np.stack(rows), optimum_row]),
        [*row_lower, -np.inf],
        [*row_upper, float(primary.fun) + 1e-7],
    )
    secondary = milp(
        stable,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=secondary_constraints,
        options={"time_limit": 300.0},
    )
    if not secondary.success or secondary.x is None:
        raise ValueError(f"MILP deterministic tie-break failed: {secondary.message}")
    val_indices = candidates[np.flatnonzero(secondary.x[:n] > 0.5)]
    test_indices = candidates[np.flatnonzero(secondary.x[n : 2 * n] > 0.5)]
    assignment_by_patient = {patient: "train" for patient in patients}
    for index in val_indices:
        assignment_by_patient[patients[int(index)]] = "val"
    for index in test_indices:
        assignment_by_patient[patients[int(index)]] = "test"
    assignment = manifest["patient_id"].map(assignment_by_patient)
    observed_counts = assignment.value_counts().to_dict()
    if observed_counts != counts:
        raise AssertionError(f"split count mismatch: {observed_counts} != {counts}")
    for split in ("val", "test"):
        counts_by_source = manifest.loc[assignment == split, "source"].value_counts()
        observed = {name: int(counts_by_source.get(name, 0)) for name in source_names}
        if observed != source_targets[split]:
            raise ValueError(
                f"best split misses exact {split} source quotas: {observed} != "
                f"{source_targets[split]}"
            )
    return (
        assignment,
        source_targets,
        {
            "method": "scipy.optimize.milp",
            "objective": "global minimum source_by_lineage_L1_deviation",
            "primary_objective": float(primary.fun),
            "primary_optimal": True,
            "deterministic_tie_break_seed": seed,
        },
    )


def _require_selection_audit(path: Path, manifest: pd.DataFrame) -> dict[str, object]:
    audit = json.loads(path.read_text())
    if audit.get("status") != "verified_against_local_26q1_and_exclusion_manifests":
        raise ValueError("selection audit is not verified")
    if audit.get("selected_model_ids") != len(manifest):
        raise ValueError("selection audit selected_model_ids mismatch")
    if audit.get("selected_excluded_overlap") != []:
        raise ValueError("selection audit reports excluded ModelID overlap")
    observed_sources = manifest["source"].value_counts().sort_index().to_dict()
    if audit.get("source_counts") != observed_sources:
        raise ValueError("selection audit source_counts mismatch")
    return audit


def _geneeffect_audit(path: Path, split_manifest: pd.DataFrame) -> dict[str, object]:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    missing = sorted(set(split_manifest["model_id"]) - set(frame.index))
    if missing:
        raise ValueError(f"GeneEffect is missing benchmark ModelIDs: {missing}")
    frame = frame.loc[split_manifest["model_id"]]
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    result: dict[str, object] = {
        "n_gene_columns": int(numeric.shape[1]),
        "all_model_ids_covered": True,
        "assignment_used_geneeffect_values": False,
        "slices": {},
    }
    finite_by_split: dict[str, np.ndarray] = {}
    for split in SPLITS:
        ids = split_manifest.loc[split_manifest["split"] == split, "model_id"]
        subset = numeric.loc[ids]
        finite = np.isfinite(subset.to_numpy())
        finite_by_split[split] = finite.sum(axis=0)
        result["slices"][split] = {
            "n_lines": int(len(ids)),
            "finite_values": int(finite.sum()),
            "genes_with_any_finite": int((finite.sum(axis=0) > 0).sum()),
        }
    train_finite = finite_by_split["train"]
    result["train_genes_with_at_least_3_finite"] = int((train_finite >= 3).sum())
    result["train_genes_with_at_least_5_finite"] = int((train_finite >= 5).sum())
    result["common_genes_train_ge5_val_ge3_test_ge3"] = int(
        (
            (train_finite >= 5)
            & (finite_by_split["val"] >= 3)
            & (finite_by_split["test"] >= 3)
        ).sum()
    )
    result["evidence_train_coverage"] = {}
    for semantics, evidence in (
        ("raw_umi_counts", "raw_umi_27"),
        ("processed_cpm", "kinker_processed_cpm_152"),
    ):
        ids = split_manifest.loc[
            (split_manifest["split"] == "train")
            & (split_manifest["matrix_semantics"] == semantics),
            "model_id",
        ]
        finite = np.isfinite(numeric.loc[ids].to_numpy()).sum(axis=0)
        result["evidence_train_coverage"][evidence] = {
            "n_train_lines": int(len(ids)),
            "genes_with_at_least_3_finite": int((finite >= 3).sum()),
            "genes_with_at_least_5_finite": int((finite >= 5).sum()),
        }
    return result


def build(
    manifest_path: Path,
    geneeffect_path: Path,
    selection_audit_path: Path,
    output_dir: Path,
    *,
    expected_lines: int,
    counts: dict[str, int],
    seed: int,
) -> None:
    raise RuntimeError(
        "the standalone 179 split was removed; use "
        "build_cell_line_geneeffect_226_split.py"
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if expected_lines == 179:
        if _sha256(manifest_path) != EXPECTED_179_MANIFEST_SHA256:
            raise ValueError("179-line manifest does not match the pinned SHA-256")
        if _sha256(selection_audit_path) != EXPECTED_179_SELECTION_AUDIT_SHA256:
            raise ValueError(
                "179-line selection audit does not match the pinned SHA-256"
            )
    manifest = _require_manifest(pd.read_csv(manifest_path), expected_lines)
    selection_audit = _require_selection_audit(selection_audit_path, manifest)
    assignment, source_targets, optimizer = assign_splits(
        manifest, counts=counts, seed=seed
    )
    split_manifest = manifest.copy()
    split_manifest.insert(0, "split", assignment.to_numpy())
    split_manifest = split_manifest.sort_values(["split", "model_id"])
    if split_manifest.groupby("patient_id")["split"].nunique().max() != 1:
        raise AssertionError("PatientID crosses split sides")
    forced = split_manifest.loc[
        split_manifest["same_patient_as_excluded"]
        | split_manifest["patient_id"].duplicated(keep=False),
        ["model_id", "split"],
    ]
    if not forced.empty and set(forced["split"]) != {"train"}:
        raise AssertionError("forced patient group escaped train")
    label_audit = _geneeffect_audit(geneeffect_path, split_manifest)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.building")
    if staging.exists():
        raise FileExistsError(f"refusing to replace incomplete staging {staging}")
    staging.mkdir()
    csv_path = staging / "cell_line_atlas_179_split.csv"
    json_path = staging / "cell_line_atlas_179_split.json"
    audit_path = staging / "cell_line_atlas_179_split_audit.json"
    try:
        split_manifest.to_csv(csv_path, index=False)
        membership = {
            split: sorted(
                split_manifest.loc[split_manifest["split"] == split, "model_id"]
            )
            for split in SPLITS
        }
        evidence_slices = {
            "raw_umi_27": {
                split: sorted(
                    split_manifest.loc[
                        (split_manifest["split"] == split)
                        & (split_manifest["matrix_semantics"] == "raw_umi_counts"),
                        "model_id",
                    ]
                )
                for split in SPLITS
            },
            "kinker_processed_cpm_152": {
                split: sorted(
                    split_manifest.loc[
                        (split_manifest["split"] == split)
                        & (split_manifest["matrix_semantics"] == "processed_cpm"),
                        "model_id",
                    ]
                )
                for split in SPLITS
            },
        }
        split_payload: dict[str, object] = {
            "membership": membership,
            "schema_version": "cell-line-atlas-179-split-v1",
            "seed": seed,
            "assignment_inputs": ["source", "lineage", "patient_id"],
            "policy": (
                "Fixed patient-grouped 125/27/27 split; exact source quotas; "
                "globally optimal source-by-lineage MILP balance; GeneEffect values "
                "excluded from assignment. Lines sharing a patient with the prior "
                "excluded context set, or sharing PatientID with another atlas line, "
                "are forced to train. The raw-UMI test slice holds eight contexts; "
                "Kinker remains a separate processed-CPM sensitivity slice."
            ),
            "primary_evidence": "raw_umi_27",
            "sensitivity_evidence": "kinker_processed_cpm_152",
            "evidence_slices": evidence_slices,
        }
        json_path.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n")
        selection_paths: dict[str, Path] = {}
        for evidence, slices in evidence_slices.items():
            path = staging / f"cell_line_atlas_179_{evidence}_selection.json"
            payload = {
                "train": slices["train"],
                "val": slices["val"],
                "test": [],
                "schema_version": "cell-line-atlas-179-selection-authority-v1",
                "evidence": evidence,
                "purpose": "hyperparameter_selection_without_test_evaluation",
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            selection_paths[evidence] = path
        audit = {
            "status": "verified_fixed_split",
            "schema_version": "cell-line-atlas-179-split-audit-v1",
            "seed": seed,
            "optimizer": optimizer,
            "counts": {
                key: int((split_manifest["split"] == key).sum()) for key in SPLITS
            },
            "patient_counts": {
                key: int(
                    split_manifest.loc[
                        split_manifest["split"] == key, "patient_id"
                    ].nunique()
                )
                for key in SPLITS
            },
            "source_counts": {
                key: split_manifest.loc[split_manifest["split"] == key, "source"]
                .value_counts()
                .sort_index()
                .to_dict()
                for key in SPLITS
            },
            "evidence_counts": {
                "raw_umi_27": {
                    key: int(
                        (
                            (split_manifest["split"] == key)
                            & (split_manifest["matrix_semantics"] == "raw_umi_counts")
                        ).sum()
                    )
                    for key in SPLITS
                },
                "kinker_processed_cpm_152": {
                    key: int(
                        (
                            (split_manifest["split"] == key)
                            & (split_manifest["matrix_semantics"] == "processed_cpm")
                        ).sum()
                    )
                    for key in SPLITS
                },
            },
            "lineage_counts": {
                key: split_manifest.loc[split_manifest["split"] == key, "lineage"]
                .value_counts()
                .sort_index()
                .to_dict()
                for key in SPLITS
            },
            "source_targets": source_targets,
            "patient_group_cross_split": 0,
            "model_id_overlap": 0,
            "selection_provenance": selection_audit,
            "forced_train_model_ids": sorted(forced["model_id"].astype(str)),
            "geneeffect": label_audit,
            "input_sha256": {
                "manifest": _sha256(manifest_path),
                "geneeffect": _sha256(geneeffect_path),
                "selection_audit": _sha256(selection_audit_path),
                "builder_script": _sha256(Path(__file__)),
            },
            "output_sha256": {
                "csv": _sha256(csv_path),
                "json": _sha256(json_path),
                **{
                    f"{evidence}_selection": _sha256(path)
                    for evidence, path in selection_paths.items()
                },
            },
        }
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geneeffect", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-lines", type=int, default=179)
    parser.add_argument("--train-count", type=int, default=125)
    parser.add_argument("--val-count", type=int, default=27)
    parser.add_argument("--test-count", type=int, default=27)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


if __name__ == "__main__":
    raise RuntimeError(
        "the standalone 179 split was removed; use "
        "build_cell_line_geneeffect_226_split.py"
    )
