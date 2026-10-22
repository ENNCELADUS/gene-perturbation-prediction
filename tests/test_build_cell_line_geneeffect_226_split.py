from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.baselines.residual import run_r1_ladder
from src.data.splits import load_geneeffect_split


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "configs/benchmarks/cell_line_geneeffect_226_split.csv"
SPLIT = ROOT / "configs/benchmarks/cell_line_geneeffect_226_split.json"
AUDIT = ROOT / "configs/benchmarks/cell_line_geneeffect_226_split_audit.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_single_split_keeps_original_contexts_in_train_except_a549() -> None:
    frame = pd.read_csv(CSV)
    assert frame["split"].value_counts().to_dict() == {
        "train": 171,
        "val": 27,
        "test": 28,
    }
    original = frame.loc[frame["cohort"] == "original_context_47"]
    assert len(original) == 47
    assert set(original.loc[original["split"] != "train", "model_id"]) == {"ACH-000681"}
    assert original.set_index("model_id").loc["ACH-000681", "split"] == "test"
    assert set(original.loc[~original["geneeffect_available"], "model_id"]) == {
        "ACH-000779",
        "ACH-001086",
    }
    assert frame.groupby("patient_id")["split"].nunique().max() == 1


def test_json_is_the_only_three_way_membership_authority() -> None:
    split = load_geneeffect_split(SPLIT)
    assert (len(split.train), len(split.val), len(split.test)) == (171, 27, 28)
    assert not (set(split.train) & (set(split.val) | set(split.test)))
    assert not (set(split.val) & set(split.test))


def test_audit_hashes_and_label_boundary() -> None:
    audit = json.loads(AUDIT.read_text())
    assert audit["status"] == "verified_unified_split"
    assert audit["original_contexts_test"] == ["ACH-000681"]
    assert audit["patient_group_cross_split"] == 0
    assert audit["geneeffect"]["missing_model_ids"] == [
        "ACH-000779",
        "ACH-001086",
    ]
    assert audit["output_sha256"] == {
        "csv": _sha256(CSV),
        "json": _sha256(SPLIT),
    }


def test_exact_authority_runs_with_two_unlabeled_train_members() -> None:
    split = load_geneeffect_split(SPLIT)
    unlabeled = {"ACH-000779", "ACH-001086"}
    labeled_ids = sorted(
        (set(split.train) | set(split.val) | set(split.test)) - unlabeled
    )
    labels = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "gene_symbol": f"G{gene}",
                "gene_effect": -1.0 - 0.01 * gene - 0.0001 * index,
            }
            for index, model_id in enumerate(labeled_ids)
            for gene in range(6)
        ]
    )
    predictions = run_r1_ladder(labels, {}, None, split=split)
    assert set(predictions["slice"]) == {"val", "test"}
    assert set(predictions["model_id"]) == set(split.val) | set(split.test)
