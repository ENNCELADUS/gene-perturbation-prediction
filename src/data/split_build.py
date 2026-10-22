"""Shared validation and MILP partition helpers for the cell-line benchmark."""

from __future__ import annotations

import hashlib
import json
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
