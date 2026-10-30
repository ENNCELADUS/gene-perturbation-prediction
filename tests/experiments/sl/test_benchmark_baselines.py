"""SL benchmark baselines on a synthetic benchmark: scorer values, and the headline
averaged within each line before averaging over lines."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.splits import FixedSplit
from src.experiments.sl.benchmark_baselines import (
    benchmark_rows,
    line_values,
    load_config,
    pair_identity,
    run,
    summary,
)

GENES = ["G0", "G1", "G2", "G3"]


def labels() -> pd.DataFrame:
    rows = []
    for line, side_label in (("T0", (1, 0, 0)), ("T1", (1, 1, 0)), ("V0", (1, 0, 0)),
                             ("E0", (0, 1, 0))):  # fmt: skip
        for (a, b), label in zip(
            [("G0", "G1"), ("G0", "G2"), ("G1", "G3")], side_label
        ):
            rows.append(
                {"study_id": "S", "design": "combinatorial", "list_id": f"S|{line}",
                 "model_id": line, "gene_a": a, "gene_b": b, "label": label}
            )  # fmt: skip
    rows.append(
        {"study_id": "S", "design": "combinatorial", "list_id": "S|E0",
         "model_id": "E0", "gene_a": "G2", "gene_b": "G3", "label": 0}
    )  # fmt: skip
    frame = pd.DataFrame(rows)
    return frame.assign(label=frame["label"].astype("Int8"))


def test_pair_identity_falls_back_to_the_studys_rate():
    frame = labels().assign(label=lambda f: f["label"].astype(int))
    training = frame[frame["model_id"].isin(["T0", "T1"])]
    held = frame[frame["model_id"] == "E0"].reset_index(drop=True)
    rate = pair_identity(training, held)
    assert rate[:3] == pytest.approx([1.0, 0.5, 0.0])
    assert rate[3] == pytest.approx(3 / 6)  # (G2, G3) never seen in training


def test_a_drug_anchor_list_is_ranked_by_gene_b():
    rows = pd.DataFrame(
        {"study_id": "D", "design": "anchor", "list_id": "D|E0", "model_id": "E0",
         "gene_a": "drug:olaparib", "gene_b": ["G0", "G1"], "label": [1, 0]}
    ).astype({"label": "Int8"})  # fmt: skip
    split = FixedSplit(train=("T0",), val=(), test=("E0",), unlabeled_train=())
    kept, dropped = benchmark_rows(rows, split, set(GENES))
    assert len(kept) == 2 and dropped["rows_gene_without_score"] == 0
    effects = pd.DataFrame({"G0": [-1.0], "G1": [0.2]}, index=["E0"])
    assert line_values(effects, kept).tolist() == pytest.approx([-1.0, 0.2])


def test_summary_averages_lists_within_a_line_first():
    lists = pd.DataFrame(
        {"split": "test", "scorer": "x", "model_id": ["L0", "L0", "L1"],
         "auprc_lift": [1.0, 3.0, 4.0], "auroc": [0.5, 0.5, 0.5]}
    )  # fmt: skip
    (row,) = summary(lists, reps=20, seed=0)
    assert row["auprc_lift"] == pytest.approx(3.0) and row["lines"] == 2


def test_run_scores_every_list(tmp_path):
    split = {"schema_version": "cell-line-geneeffect-split-v1", "train": ["T0", "T1"],
             "val": ["V0"], "test": ["E0"], "unlabeled_train": []}  # fmt: skip
    (tmp_path / "split.json").write_text(json.dumps(split))
    effect = pd.DataFrame(
        np.array([[-1.0, -0.2, 0.0, 0.1], [-0.8, -0.4, 0.0, 0.1],
                  [-0.9, -0.1, 0.0, 0.0], [-0.1, -1.5, -2.0, 0.0]]),
        index=pd.Index(["T0", "T1", "V0", "E0"], name="ModelID"),
        columns=[f"{g} ({i})" for i, g in enumerate(GENES)],
    )  # fmt: skip
    effect.to_csv(tmp_path / "ge.csv")
    labels().to_parquet(tmp_path / "labels.parquet")
    (tmp_path / "config.yaml").write_text(
        f"kind: sl_benchmark_baselines\nseed: 0\noutput_root: {tmp_path}\ntask: base\n"
        f"labels: {tmp_path}/labels.parquet\nsplit: {tmp_path}/split.json\n"
        f"gene_effect: {tmp_path}/ge.csv\nprior: null\ntop_k: 2\nbootstrap: 10\n"
    )
    out = run(load_config(tmp_path / "config.yaml"), "r0")
    scores = pd.read_parquet(out / "scores.parquet")
    pan = scores[
        (scores["scorer"] == "pan_essentiality") & (scores["model_id"] == "E0")
    ]
    # mean over T0, T1: G0 -0.9, G1 -0.3, G2 0.0, G3 0.1
    assert pan["score"].tolist() == pytest.approx([1.2, 0.9, 0.2, -0.1])
    measured = scores[
        (scores["scorer"] == "measured_geneeffect") & (scores["model_id"] == "E0")
    ]
    assert measured["score"].tolist() == pytest.approx([1.6, 2.1, 1.5, 2.0])
    metrics = json.loads((out / "metrics.json").read_text())
    by = {(r["split"], r["scorer"]): r for r in metrics["summary"]}
    scorers = ("pair_identity", "pan_essentiality", "measured_geneeffect")
    assert set(by) == {(s, c) for s in ("val", "test") for c in scorers}
    # On E0 the measured GeneEffect ranks the one SL pair (G0, G2) first.
    assert by[("test", "measured_geneeffect")]["auprc_lift"] == pytest.approx(4.0)
