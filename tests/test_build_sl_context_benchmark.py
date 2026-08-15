from __future__ import annotations

import hashlib

import pandas as pd
import pytest

import scripts.build_sl_context_benchmark as benchmark_builder
from scripts.build_sl_context_benchmark import (
    apply_context_split,
    audit_positive_losses,
    context_statistics,
    finalise_benchmark,
    select_atomic_rows,
    validate_split_evidence,
)


def test_generated_manifest_does_not_encode_claim_policy(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(benchmark_builder, "_sha256", lambda _path: "0" * 64)
    manifest = benchmark_builder._manifest(
        tmp_path,
        min_class_count=10,
        selection_stats={},
        pre_split_stats={},
        split_stats={
            "retained_rows": 0,
            "retained_unique_pairs": 0,
            "retained_unique_genes": 0,
            "retained_by_context": {},
        },
    )

    assert "formal" not in manifest["split"]
    assert "claim_status" not in manifest["split"]


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "conflict": 0,
        "cell_lines": "RPE1",
        "en": False,
        "ep": True,
        "evidence_types": "experimental_screen",
        "gene_a_status": "approved",
        "gene_b_status": "updated",
        "has_human_evidence": True,
        "is_sl": True,
        "label": "positive",
        "label_tier": "experimental",
        "min_fdr": 0.01,
        "n_cell_lines": 1,
        "n_evidence": 1,
        "n_neg": 0,
        "n_pos": 1,
        "n_unl": 0,
        "organisms": "human",
        "pair_human_ortholog": "TP53|BRCA1",
        "qc_flag": None,
        "sources": "screen",
    }
    row.update(updates)
    return row


def _negative(**updates: object) -> dict[str, object]:
    row = _row(
        en=True,
        ep=False,
        is_sl=False,
        label="negative",
        label_tier="experimental_negative",
        n_neg=1,
        n_pos=0,
    )
    row.update(updates)
    return row


def _atomic_row(
    pair_id: str,
    context: str,
    label: int,
    min_fdr: float,
    source_row_id: int = 0,
) -> dict[str, object]:
    gene_a, gene_b = pair_id.split("|")
    return {
        "pair_id": pair_id,
        "gene_a": gene_a,
        "gene_b": gene_b,
        "context": context,
        "sl_label": label,
        "label_name": "positive" if label else "negative",
        "label_semantics": (
            "experimental_screen_hit" if label else "experimental_screen_non_hit"
        ),
        "pair_is_ordered": False,
        "label_confidence": "silver_inferred",
        "context_assignment": "unanimous_row_evidence_count_match",
        "source_row_id": source_row_id,
        "source_n_evidence": 1,
        "source_row_min_fdr": min_fdr,
    }


def test_select_atomic_rows_explodes_only_identifiable_contexts() -> None:
    frame = pd.DataFrame(
        [
            _row(
                cell_lines="rpe1;K562",
                n_cell_lines=2,
                n_evidence=2,
                n_pos=2,
            ),
            _row(cell_lines="CTX:literature"),
            _row(cell_lines="RPE1;K562", n_cell_lines=2),
            _row(pair_human_ortholog="TP53|TP53"),
            _row(label_tier="predicted_or_db"),
        ]
    )

    selected, stats = select_atomic_rows(frame)

    assert selected[["context", "pair_id", "sl_label"]].to_dict("records") == [
        {"context": "RPE1", "pair_id": "BRCA1|TP53", "sl_label": 1},
        {"context": "K562", "pair_id": "BRCA1|TP53", "sl_label": 1},
    ]
    assert stats == {
        "raw_rows": 5,
        "quality_label_rows": 4,
        "count_mismatch_rows": 1,
        "invalid_pair_rows": 1,
        "invalid_context_tokens": 1,
        "atomic_rows": 2,
    }


@pytest.mark.parametrize(
    "row",
    [
        _negative(n_evidence=2, n_neg=1, n_pos=1),
        _row(organisms="mouse"),
        _row(conflict=1),
        _row(sources="screen;literature"),
        _row(cell_lines="RPE1;K562", n_cell_lines=2),
        _row(cell_lines="A549+K562"),
    ],
)
def test_select_atomic_rows_fails_closed(row: dict[str, object]) -> None:
    selected, _ = select_atomic_rows(pd.DataFrame([row]))

    assert selected.empty


def test_finalise_excludes_pair_context_label_conflicts_and_weak_contexts() -> None:
    rows = pd.DataFrame(
        [
            _atomic_row("A|B", "RPE1", 1, 0.01),
            _atomic_row("C|D", "RPE1", 0, 0.02),
            _atomic_row("X|Y", "RPE1", 0, 0.03),
            _atomic_row("X|Y", "RPE1", 1, 0.03),
            _atomic_row("E|F", "K562", 1, 0.04),
        ]
    )

    benchmark, inventory, stats = finalise_benchmark(rows, min_class_count=1)

    assert set(benchmark["pair_id"]) == {"A|B", "C|D"}
    assert set(benchmark["context"]) == {"RPE1"}
    assert not inventory.set_index("context").loc[
        "K562", "included_in_pair_classification_table"
    ]
    assert stats["pair_context_label_conflicts_excluded"] == 1


def test_finalise_rejects_nonpositive_minimum() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        finalise_benchmark(pd.DataFrame({"pair_id": ["A|B"]}), 0)


def test_source_row_id_is_the_global_raw_row_and_survives_explosion() -> None:
    frame = pd.DataFrame(
        [_row(cell_lines="RPE1;K562", n_cell_lines=2, n_evidence=2, n_pos=2)],
        index=[41_237],
    )

    selected, _ = select_atomic_rows(frame)

    # Both contexts came from one aggregate row, so they share its id. If their
    # contexts land on different split sides, the complete row is removed.
    assert selected["source_row_id"].tolist() == [41_237, 41_237]


def _split_contract() -> dict[str, object]:
    return {
        "schema_version": "sl-context-screen-v2-split-v1",
        "eligibility": {"minimum_positive_rows": 1, "minimum_negative_rows": 1},
        "contexts": {
            "TRAIN": {"model_id": "ACH-000003", "response_anchor": True},
            "VALID": {"model_id": "ACH-000002", "response_anchor": False},
            "TEST": {"model_id": "ACH-000001", "response_anchor": False},
        },
        "pinned_train_contexts": ["TRAIN"],
        "allocation_counts": {"test": 1, "validation": 1},
        "assignments": {"TRAIN": "train", "TEST": "test", "VALID": "validation"},
    }


def test_context_split_drops_source_rows_crossing_assignment_sides() -> None:
    rows = pd.DataFrame(
        [
            _atomic_row("A|B", "TRAIN", 1, 0.01, source_row_id=1),
            _atomic_row("C|D", "VALID", 1, 0.01, source_row_id=1),
            _atomic_row("E|F", "TRAIN", 1, 0.01, source_row_id=2),
            _atomic_row("G|H", "TRAIN", 0, 0.01, source_row_id=3),
            _atomic_row("I|J", "VALID", 1, 0.01, source_row_id=4),
            _atomic_row("K|L", "VALID", 0, 0.01, source_row_id=5),
            _atomic_row("M|N", "TEST", 1, 0.01, source_row_id=6),
            _atomic_row("O|P", "TEST", 0, 0.01, source_row_id=7),
        ]
    )

    split, stats = apply_context_split(rows, _split_contract())

    assert set(split["source_row_id"]) == {2, 3, 4, 5, 6, 7}
    assert split.groupby("source_row_id")["split"].nunique().max() == 1
    assert stats["cross_split_source_rows_dropped"] == 1
    assert stats["rows_dropped"] == 2
    assert stats["pairs_crossing_splits_after_filter"] == 0


def test_context_split_rejects_assignment_not_implied_by_model_id_order() -> None:
    contract = _split_contract()
    contract["assignments"] = {
        "TRAIN": "train",
        "TEST": "validation",
        "VALID": "test",
    }
    rows = pd.DataFrame(
        [
            _atomic_row("A|B", context, 1, 0.01, source_row_id=offset)
            for offset, context in enumerate(["TRAIN", "VALID", "TEST"], start=1)
        ]
        + [
            _atomic_row("C|D", context, 0, 0.01, source_row_id=offset)
            for offset, context in enumerate(["TRAIN", "VALID", "TEST"], start=4)
        ]
    )

    with pytest.raises(ValueError, match="deterministic rule"):
        apply_context_split(rows, contract)

    anchor_mismatch = _split_contract()
    anchor_mismatch["contexts"]["TRAIN"]["response_anchor"] = False
    with pytest.raises(ValueError, match="response anchors"):
        apply_context_split(rows, anchor_mismatch)


def test_context_split_rejects_a_context_removed_in_full() -> None:
    rows = pd.DataFrame(
        [
            _atomic_row("A|B", "TRAIN", 1, 0.01, source_row_id=1),
            _atomic_row("C|D", "TEST", 1, 0.01, source_row_id=1),
            _atomic_row("E|F", "TRAIN", 0, 0.01, source_row_id=2),
            _atomic_row("G|H", "TEST", 0, 0.01, source_row_id=2),
            _atomic_row("I|J", "VALID", 1, 0.01, source_row_id=3),
            _atomic_row("K|L", "VALID", 0, 0.01, source_row_id=4),
        ]
    )

    with pytest.raises(ValueError, match="made contexts ineligible"):
        apply_context_split(rows, _split_contract())


def test_context_split_rejects_pairs_crossing_on_distinct_source_rows() -> None:
    rows = pd.DataFrame(
        [
            _atomic_row("A|B", "TRAIN", 1, 0.01, source_row_id=1),
            _atomic_row("C|D", "TRAIN", 0, 0.01, source_row_id=2),
            _atomic_row("E|F", "VALID", 1, 0.01, source_row_id=3),
            _atomic_row("G|H", "VALID", 0, 0.01, source_row_id=4),
            _atomic_row("A|B", "TEST", 1, 0.01, source_row_id=5),
            _atomic_row("I|J", "TEST", 0, 0.01, source_row_id=6),
        ]
    )

    with pytest.raises(ValueError, match="Canonical pairs cross"):
        apply_context_split(rows, _split_contract())


def test_split_evidence_hashes_and_model_ids_are_checked(tmp_path) -> None:
    basal_path = tmp_path / "basal.csv"
    gene_effect_path = tmp_path / "gene_effect.csv"
    model_path = tmp_path / "model.csv"
    pd.DataFrame(
        {
            "model_id": ["ACH-1"],
            "basal_source": ["controls"],
            "cellosaurus_id": ["CVCL_0001"],
        }
    ).to_csv(basal_path, index=False)
    pd.DataFrame({"ModelID": ["ACH-1"], "GENE": [-0.5]}).to_csv(
        gene_effect_path, index=False
    )
    pd.DataFrame({"ModelID": ["ACH-1"], "StrippedCellLineName": ["CELL"]}).to_csv(
        model_path, index=False
    )

    def digest(path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    contract = {
        "contexts": {
            "CELL": {
                "model_id": "ACH-1",
                "basal_source": "controls",
                "cellosaurus_id": "CVCL_0001",
                "canonical_name": "CELL",
            }
        },
        "executable_evidence": {
            "basal_manifest": {"path": str(basal_path), "sha256": digest(basal_path)},
            "gene_effect": {
                "path": str(gene_effect_path),
                "sha256": digest(gene_effect_path),
            },
            "model_metadata": {
                "path": str(model_path),
                "sha256": digest(model_path),
            },
        },
    }

    validate_split_evidence(contract)
    contract["contexts"] = {"OTHER": contract["contexts"].pop("CELL")}
    with pytest.raises(ValueError, match="Label context identity mismatch"):
        validate_split_evidence(contract)
    contract["contexts"] = {"CELL": contract["contexts"].pop("OTHER")}
    pd.DataFrame({"ModelID": ["ACH-1"], "StrippedCellLineName": ["WRONG"]}).to_csv(
        model_path, index=False
    )
    contract["executable_evidence"]["model_metadata"]["sha256"] = digest(model_path)
    with pytest.raises(ValueError, match="DepMap identity mismatch"):
        validate_split_evidence(contract)
    pd.DataFrame({"ModelID": ["ACH-1"], "StrippedCellLineName": ["CELL"]}).to_csv(
        model_path, index=False
    )
    contract["executable_evidence"]["model_metadata"]["sha256"] = digest(model_path)
    contract["executable_evidence"]["gene_effect"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_split_evidence(contract)


def test_audit_attributes_dropped_positives_to_the_responsible_condition() -> None:
    frame = pd.DataFrame(
        [
            _row(cell_lines="MCF7", sources="screen;literature"),
            _row(cell_lines="MCF7", sources="screen;literature"),
            _row(cell_lines="GI1", conflict=1),
            _row(cell_lines="RPE1"),
        ]
    )

    audit = audit_positive_losses(frame).set_index(["context", "condition"])

    assert audit.loc[("MCF7", "sources_screen_only"), "positives_dropped"] == 2
    assert audit.loc[("GI1", "conflict_zero"), "positives_dropped"] == 1
    assert ("RPE1", "conflict_zero") not in audit.index


def test_missing_cell_lines_do_not_become_a_nan_context() -> None:
    # str(NaN).upper() == "NAN", which matches the atomic-token pattern.
    frame = pd.DataFrame([_row(cell_lines=float("nan"), conflict=1)])

    audit = audit_positive_losses(frame)
    selected, _ = select_atomic_rows(pd.DataFrame([_row(cell_lines=float("nan"))]))

    assert "NAN" not in set(audit["context"])
    assert selected.empty


def test_context_statistics_expose_a_single_gene_label_function() -> None:
    benchmark = pd.DataFrame(
        [
            _atomic_row("BAIT|X", "A549", 1, 0.01, source_row_id=1),
            _atomic_row("BAIT|Y", "A549", 1, 0.01, source_row_id=2),
            _atomic_row("P|Q", "A549", 0, 0.01, source_row_id=3),
            _atomic_row("M|N", "K562", 1, 0.01, source_row_id=4),
            _atomic_row("R|S", "K562", 0, 0.01, source_row_id=4),
        ]
    )

    stats = context_statistics(benchmark).set_index("context")

    # Every A549 positive contains BAIT, so its label function is an indicator.
    assert stats.loc["A549", "top_positive_gene"] == "BAIT"
    assert stats.loc["A549", "top_positive_gene_share"] == 1.0
    assert stats.loc["K562", "top_positive_gene_share"] == 1.0
    assert stats.loc["A549", "n_distinct_positive_genes"] == 3
    assert stats.loc["A549", "positive_prior"] == pytest.approx(2 / 3)
    assert stats.loc["A549", "n_rows_sharing_source_row_with_other_context"] == 0
