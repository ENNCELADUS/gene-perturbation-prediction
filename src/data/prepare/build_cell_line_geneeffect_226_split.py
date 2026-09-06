#!/usr/bin/env python3
"""Build the single 226-context GeneEffect benchmark split."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.data.split_build import (
    EXPECTED_179_MANIFEST_SHA256,
    EXPECTED_179_SELECTION_AUDIT_SHA256,
    _require_manifest,
    _require_selection_audit,
    _sha256,
    assign_splits,
)


PINNED_SHA256 = {
    "phase_a_manifest": (
        "8656a91e5588e66b33e66774aa6c17f7238e94fc1125649e5ac7eed540d4e297"
    ),
    "basal_registry": (
        "19354c54faf15f325de1199ed668117d43e7531b6553c73ae3a224124e5c4f47"
    ),
    "depmap_model": "ea4e0b2a3bc806f81df62689a5ae75f1a100135727a3d7b8a4c7ccc8815183f8",
    "geneeffect": "e610a4cefb13a82b5b256b47eb08b63ff14843f8dbd0fb164bc0a32688e5b89e",
}
SPLITS = ("train", "val", "test")


def _pin(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")


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
                "split": "train",
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
    *,
    pin_inputs: bool = True,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if pin_inputs:
        _pin(atlas_manifest_path, EXPECTED_179_MANIFEST_SHA256, "179 manifest")
        _pin(
            selection_audit_path,
            EXPECTED_179_SELECTION_AUDIT_SHA256,
            "179 selection audit",
        )
        for label, path in (
            ("phase_a_manifest", phase_path),
            ("basal_registry", registry_path),
            ("depmap_model", model_path),
            ("geneeffect", geneeffect_path),
        ):
            _pin(path, PINNED_SHA256[label], label)

    atlas = _require_manifest(pd.read_csv(atlas_manifest_path), 179)
    selection_audit = _require_selection_audit(selection_audit_path, atlas)
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
    expected = {"train": 172, "val": 27, "test": 27}
    if observed != expected:
        raise AssertionError(f"unified split count mismatch: {observed} != {expected}")
    if manifest.groupby("patient_id")["split"].nunique().max() != 1:
        raise AssertionError("PatientID crosses unified split sides")
    if set(original["split"]) != {"train"}:
        raise AssertionError("an original context escaped train")
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
                "schema_version": "cell-line-geneeffect-226-split-v1",
                "policy": (
                    "One 172/27/27 split. All 47 original contexts are fixed in train; "
                    "the 179 new contexts retain their patient-grouped 125/27/27 MILP "
                    "partition. PC9 and HELA are train members without 26Q1 GeneEffect."
                ),
            }
        )
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        audit = {
            "status": "verified_unified_split",
            "schema_version": "cell-line-geneeffect-226-split-audit-v1",
            "counts": expected,
            "cohort_counts": manifest.groupby(["cohort", "split"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index"),
            "original_contexts_all_train": True,
            "patient_group_cross_split": 0,
            "atlas_original_model_id_overlap": 0,
            "optimizer_for_179_partition": optimizer,
            "source_targets_for_179_partition": source_targets,
            "geneeffect": label_audit,
            "selection_provenance": selection_audit,
            "input_sha256": {
                "atlas_manifest": _sha256(atlas_manifest_path),
                "selection_audit": _sha256(selection_audit_path),
                "phase_a_manifest": _sha256(phase_path),
                "basal_registry": _sha256(registry_path),
                "depmap_model": _sha256(model_path),
                "geneeffect": _sha256(geneeffect_path),
                "builder_script": _sha256(Path(__file__)),
            },
            "output_sha256": {
                "csv": _sha256(csv_path),
                "json": _sha256(json_path),
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
