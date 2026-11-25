"""The targets of the SL target options, on small hand-made tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.sl.target_options import (
    harle_replicates,
    interaction,
    ordering_ceiling,
    replicate_rows,
    with_score,
)


def labels() -> pd.DataFrame:
    rows = [("A", "B", 1), ("A", "C", 0), ("B", "C", 0)]
    return pd.DataFrame(
        {"study_id": "Harle2025", "list_id": "Harle2025|L", "model_id": "M",
         "design": "combinatorial", "condition": "genetic", "line": "L",
         "gene_a": [a for a, _, _ in rows], "gene_b": [b for _, b, _ in rows],
         "screen_gene_a": [a for a, _, _ in rows],
         "screen_gene_b": [b for _, b, _ in rows],
         "score": [0.1, 0.2, 0.3], "label": [label for _, _, label in rows]}
    )  # fmt: skip


def test_interaction_is_the_double_loss_minus_both_single_losses():
    lfc = pd.DataFrame(
        {"gene_a": ["A", "A", "AAVS1", "B", "A"],
         "gene_b": ["B", "AAVS1", "B", "A", "C"],
         "L": [-3.0, -0.5, -1.0, -2.0, -0.4]}
    )  # fmt: skip
    lfc = lfc.assign(unit_a=lfc["gene_a"], unit_b=lfc["gene_b"])
    out = interaction(lfc, ["L"]).set_index("pair")["gi"]
    # A|B: the mean of its two constructs; A|C has no single loss of C
    assert out.to_dict() == {"A|B": pytest.approx(-2.5 + 0.5 + 1.0)}


def test_a_harle_replicate_scores_a_pair_from_its_counts():
    counts = pd.DataFrame(
        {"sorted_gene_pair": ["B|A", "B|A", "B|A", "B|A"],
         "guide_type": ["gene|gene", "gene|gene", "gene|safe_targeting",
                        "safe_targeting|gene"],
         "singles_target_gene": [None, None, "A", "B"],
         "L R1": [0.0, 3.0, 1.0, 3.0], "L R2": [7.0, 7.0, 7.0, 7.0],
         "control_mean": [7.0, 7.0, 7.0, 7.0]}
    )  # fmt: skip
    out = harle_replicates(counts).set_index("replicate")
    assert out.loc["1", ["line", "pair"]].tolist() == ["L", "A|B"]
    # mean of log2(1/8), log2(4/8) minus log2(2/8) minus log2(4/8)
    assert out.loc["1", "gi"] == pytest.approx(-2.0 + 2.0 + 1.0)
    assert out.loc["2", "gi"] == pytest.approx(0.0)


def test_replicates_become_studies_with_as_many_calls_as_the_line_has():
    replicates = pd.DataFrame(
        {"line": "L", "replicate": ["1", "1", "1", "2", "2", "2", "3", "3"],
         "pair": ["A|B", "A|C", "B|C", "A|B", "A|C", "B|C", "A|B", "A|C"],
         "gi": [-2.0, 0.5, 0.1, 0.4, -1.0, 0.2, -1.5, 0.3]}
    )  # fmt: skip
    rows = replicate_rows(labels(), replicates)
    assert set(rows["study_id"]) == {"Harle2025|R1", "Harle2025|R2", "Harle2025|R3",
                                     "Harle2025|R1+R2"}  # fmt: skip
    called = rows[rows["label"] == 1].set_index("study_id")
    assert (called["gene_a"] + called["gene_b"]).to_dict() == {
        "Harle2025|R1": "AB", "Harle2025|R2": "AC", "Harle2025|R3": "AB",
        "Harle2025|R1+R2": "AB",
    }  # fmt: skip
    mean = rows[rows["study_id"] == "Harle2025|R1+R2"].set_index("gene_b")["score"]
    assert mean.loc["B"] == pytest.approx(-0.8)


def test_with_score_replaces_one_study_and_leaves_gaps_empty():
    gi = pd.DataFrame({"line": "L", "pair": ["A|B", "A|C"], "gi": [-1.0, 2.0]})
    out = with_score(labels(), "Harle2025", gi)["score"].to_numpy()
    assert out[:2].tolist() == [-1.0, 2.0] and np.isnan(out[2])
    assert with_score(labels(), "Other", gi)["score"].tolist() == [0.1, 0.2, 0.3]


def test_the_ordering_ceiling_scores_one_replicate_against_anothers_calls():
    """Two lines, one pair: replicate 1 calls it SL in M1 only; replicate 2's score
    is lower (more SL) in M1, replicate 3's in M2."""
    rows = []
    for study, m1, m2, call in (
        ("Harle2025|R1", -3.0, 0.0, 1.0), ("Harle2025|R2", -2.0, 0.5, 1.0),
        ("Harle2025|R3", 0.5, -1.0, 0.0), ("Harle2025|R1+R2", -2.5, 0.25, 1.0),
    ):  # fmt: skip
        for model_id, score, label in (("M1", m1, call), ("M2", m2, 1.0 - call)):
            rows.append(
                {"study_id": study, "list_id": f"{study}|{model_id}",
                 "model_id": model_id, "design": "combinatorial",
                 "condition": "genetic", "gene_a": "A", "gene_b": "B",
                 "score": score, "label": label}
            )  # fmt: skip
    records = ordering_ceiling(
        pd.DataFrame(rows), {"M1": "train", "M2": "train"}, reps=10, seed=0
    )
    value = {(r["scored"][-2:], r["called"][-2:]): r["all"]["value"] for r in records}
    assert len(records) == 7 and records[-1]["scored"] == "Harle2025|R1+R2"
    assert value[("R2", "R1")] == 1.0 and value[("R1", "R2")] == 1.0
    # replicate 3 calls M2; replicate 1's score is lower in M1: the wrong order
    assert value[("R1", "R3")] == 0.0 and value[("R2", "R3")] == 0.0
    assert records[0]["train"] == records[0]["all"] and "val" not in records[0]
    assert records[0]["all"]["examples"] == 1 and "by_study" not in records[0]["all"]
