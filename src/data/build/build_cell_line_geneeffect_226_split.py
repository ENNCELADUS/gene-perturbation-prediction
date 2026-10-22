"""Build the 226-context GeneEffect split (the final split's single-cell core): the
42 original Phase-A contexts and the basal registry's nine, then the atlas lines
partitioned by a MILP that balances source and lineage across train, val and test."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


SPLITS = ("train", "val", "test")

REQUIRED_COLUMNS = (
    "source",
    "model_id",
    "patient_id",
    "cell_line_name",
    "lineage",
    "matrix_semantics",
    "same_patient_as_excluded",
)
EXPECTED_SOURCE_SEMANTICS = {
    "breast_cell_line_atlas": "raw_umi_counts",
    "ccla_omix005191": "raw_umi_counts",
    "kinker_sccle": "processed_cpm",
}


def require_manifest(frame: pd.DataFrame, expected_lines: int) -> pd.DataFrame:
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


def require_selection_audit(path: Path, manifest: pd.DataFrame) -> dict[str, object]:
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


#: Original contexts moved out of train. A549 is screened by all three multi-line SL
#: screens of the SL benchmark (Ito, Klingbeil, Harle), which needs it held out of
#: stage one (blueprint §5, 2026-10-11).
ORIGINAL_TEST = frozenset({"ACH-000681"})
EXPECTED_COUNTS = {"train": 171, "val": 27, "test": 28}


def _original_rows(
    phase_path: Path,
    registry_path: Path,
    model_path: Path,
    geneeffect_ids: set[str],
) -> pd.DataFrame:
    phase = pd.read_csv(phase_path, dtype=str).fillna("")
    registry = json.loads(registry_path.read_text())["contexts"]
    registry_by_id = {str(row["model_id"]): row for row in registry}
    phase_ids = set(phase["model_id"].astype(str))
    registry_ids = set(registry_by_id)
    if len(phase_ids) != 42 or len(registry_ids) != 9:
        raise ValueError("expected 42 Phase-A and 9 basal-registry ModelIDs")
    original_ids = phase_ids | registry_ids
    if len(original_ids) != 47:
        raise ValueError("expected a 47-ModelID original-context union")
    if not ORIGINAL_TEST <= original_ids:
        raise ValueError(f"ORIGINAL_TEST is not original: {sorted(ORIGINAL_TEST)}")
    model = pd.read_csv(model_path, dtype=str).set_index("ModelID")
    missing_model = sorted(original_ids - set(model.index))
    if missing_model:
        raise ValueError(
            f"DepMap Model.csv is missing original contexts: {missing_model}"
        )
    phase_by_id = phase.set_index("model_id")
    rows: list[dict[str, object]] = []
    for model_id in sorted(original_ids):
        metadata = model.loc[model_id]
        phase_row = phase_by_id.loc[model_id] if model_id in phase_ids else None
        registry_row = registry_by_id.get(model_id)
        rows.append(
            {
                "split": "test" if model_id in ORIGINAL_TEST else "train",
                "cohort": "original_context_47",
                "model_id": model_id,
                "patient_id": str(metadata["PatientID"]),
                "cell_line_name": str(metadata["CellLineName"]),
                "lineage": str(metadata["OncotreeLineage"]),
                "expression_source": (
                    str(phase_row["basal_source"])
                    if phase_row is not None
                    else str(registry_row["basal_source"])
                ),
                "matrix_semantics": "registered_basal",
                "geneeffect_available": model_id in geneeffect_ids,
                "original_phase_a": model_id in phase_ids,
                "original_basal_registry": model_id in registry_ids,
            }
        )
    return pd.DataFrame(rows)


def _label_audit(geneeffect_path: Path, manifest: pd.DataFrame) -> dict[str, object]:
    labels = pd.read_csv(geneeffect_path, index_col=0)
    labels.index = labels.index.astype(str)
    missing = sorted(set(manifest["model_id"]) - set(labels.index))
    expected_missing = ["ACH-000779", "ACH-001086"]
    if missing != expected_missing:
        raise ValueError(f"unexpected GeneEffect-missing ModelIDs: {missing}")
    available = manifest.loc[manifest["model_id"].isin(labels.index)].copy()
    numeric = labels.loc[available["model_id"]].apply(pd.to_numeric, errors="coerce")
    slices: dict[str, object] = {}
    finite_by_split = {}
    for split in SPLITS:
        ids = available.loc[available["split"] == split, "model_id"]
        finite = numeric.loc[ids].notna().sum(axis=0)
        finite_by_split[split] = finite
        slices[split] = {
            "membership_lines": int((manifest["split"] == split).sum()),
            "geneeffect_lines": int(len(ids)),
            "genes_with_any_finite": int((finite > 0).sum()),
        }
    return {
        "missing_model_ids": missing,
        "available_model_ids": int(len(available)),
        "slices": slices,
        "train_genes_with_at_least_5_finite": int(
            (finite_by_split["train"] >= 5).sum()
        ),
        "common_genes_train_ge5_val_ge3_test_ge3": int(
            (
                (finite_by_split["train"] >= 5)
                & (finite_by_split["val"] >= 3)
                & (finite_by_split["test"] >= 3)
            ).sum()
        ),
    }


def build(
    atlas_manifest_path: Path,
    selection_audit_path: Path,
    phase_path: Path,
    registry_path: Path,
    model_path: Path,
    geneeffect_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    atlas = require_manifest(pd.read_csv(atlas_manifest_path), 179)
    selection_audit = require_selection_audit(selection_audit_path, atlas)
    assignment, source_targets, optimizer = assign_splits(
        atlas,
        counts={"train": 125, "val": 27, "test": 27},
        seed=20260816,
    )
    atlas_rows = pd.DataFrame(
        {
            "split": assignment,
            "cohort": "new_cell_line_atlas_179",
            "model_id": atlas["model_id"],
            "patient_id": atlas["patient_id"],
            "cell_line_name": atlas["cell_line_name"],
            "lineage": atlas["lineage"],
            "expression_source": atlas["source"],
            "matrix_semantics": atlas["matrix_semantics"],
            "geneeffect_available": True,
            "original_phase_a": False,
            "original_basal_registry": False,
        }
    )
    geneeffect_ids = set(
        pd.read_csv(geneeffect_path, usecols=[0]).iloc[:, 0].astype(str)
    )
    original = _original_rows(phase_path, registry_path, model_path, geneeffect_ids)
    if set(atlas_rows["model_id"]) & set(original["model_id"]):
        raise ValueError("179-line atlas overlaps the original 47 contexts")
    manifest = pd.concat([atlas_rows, original], ignore_index=True)
    manifest = manifest.sort_values(["split", "model_id"]).reset_index(drop=True)
    observed = manifest["split"].value_counts().to_dict()
    expected = EXPECTED_COUNTS
    if observed != expected:
        raise AssertionError(f"unified split count mismatch: {observed} != {expected}")
    if manifest.groupby("patient_id")["split"].nunique().max() != 1:
        raise AssertionError("PatientID crosses unified split sides")
    moved = set(original.loc[original["split"] != "train", "model_id"])
    if moved != ORIGINAL_TEST:
        raise AssertionError(f"original contexts outside train {sorted(moved)}")
    label_audit = _label_audit(geneeffect_path, manifest)

    staging = output_dir.with_name(f".{output_dir.name}.building")
    if staging.exists():
        raise FileExistsError(f"refusing to replace incomplete staging {staging}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    csv_path = staging / "cell_line_geneeffect_226_split.csv"
    json_path = staging / "cell_line_geneeffect_226_split.json"
    audit_path = staging / "cell_line_geneeffect_226_split_audit.json"
    try:
        manifest.to_csv(csv_path, index=False)
        payload = {
            split: sorted(manifest.loc[manifest["split"] == split, "model_id"])
            for split in SPLITS
        }
        payload.update(
            {
                "unlabeled_train": ["ACH-000779", "ACH-001086"],
                "schema_version": "cell-line-geneeffect-226-split-v2",
                "policy": (
                    "One 171/27/28 split. The 47 original contexts are fixed in train "
                    "except A549, fixed in test for the SL benchmark; the 179 new "
                    "contexts retain their patient-grouped 125/27/27 MILP partition. "
                    "PC9 and HELA are train members without 26Q1 GeneEffect."
                ),
            }
        )
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        audit = {
            "status": "verified_unified_split",
            "schema_version": "cell-line-geneeffect-226-split-audit-v2",
            "counts": expected,
            "cohort_counts": manifest.groupby(["cohort", "split"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index"),
            "original_contexts_test": sorted(ORIGINAL_TEST),
            "patient_group_cross_split": 0,
            "atlas_original_model_id_overlap": 0,
            "optimizer_for_179_partition": optimizer,
            "source_targets_for_179_partition": source_targets,
            "geneeffect": label_audit,
            "selection_provenance": {
                key: value
                for key, value in selection_audit.items()
                if key != "evidence_sha256"
            },
        }
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-manifest", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--basal-registry", type=Path, required=True)
    parser.add_argument("--depmap-model", type=Path, required=True)
    parser.add_argument("--geneeffect", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        args.atlas_manifest,
        args.selection_audit,
        args.phase_a_manifest,
        args.basal_registry,
        args.depmap_model,
        args.geneeffect,
        args.output_dir,
    )
