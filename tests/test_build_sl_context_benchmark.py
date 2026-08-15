from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_sl_context_benchmark import (
    audit_positive_losses,
    context_statistics,
    finalise_benchmark,
    select_atomic_rows,
)


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

    # Both contexts came from one aggregate row, so they share its id — this is
    # the coupling that must keep them on the same side of a split.
    assert selected["source_row_id"].tolist() == [41_237, 41_237]


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
