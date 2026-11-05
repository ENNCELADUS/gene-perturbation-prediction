"""SL benchmark baselines on a synthetic benchmark: scorer values, and the headline
averaged over a study's genetic screens, a line's studies, then the lines."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.splits import FixedSplit
from src.experiments.sl.benchmark_baselines import (
    benchmark_rows,
    line_records,
    line_values,
    list_records,
    load_config,
    pair_identity,
    positive_ranks,
    run,
    strata,
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
                {"study_id": "S", "design": "combinatorial", "condition": "genetic",
                 "list_id": f"S|{line}",
                 "model_id": line, "gene_a": a, "gene_b": b, "label": label}
            )  # fmt: skip
    rows.append(
        {"study_id": "S", "design": "combinatorial", "condition": "genetic",
         "list_id": "S|E0",
         "model_id": "E0", "gene_a": "G2", "gene_b": "G3", "label": 0}
    )  # fmt: skip
    frame = pd.DataFrame(rows)
    return frame.assign(label=frame["label"].astype("Int8"))


def test_strata_split_seen_pairs_by_agreement_with_the_training_lines():
    frame = labels().assign(label=lambda x: x["label"].astype(int))
    training = frame[frame["model_id"].isin(["T0", "T1"])]
    held = frame[frame["model_id"] == "E0"]
    # training rates: G0-G1 1.0, G0-G2 0.5, G1-G3 0.0; G2-G3 is never labelled there
    assert strata(training, held).tolist() == [
        "seen_flip",
        "seen_agree",
        "seen_agree",
        "new_pair",
    ]
    unseen = held.iloc[[3]].assign(gene_b="G9")
    assert strata(training, unseen).tolist() == ["unseen_gene"]


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


def test_recall_in_the_top_tenth_and_percentiles_of_a_known_list():
    # 20 pairs, 2 known positives ranked 1st and 5th: the top 10% holds 2 pairs.
    score = -np.arange(20, dtype=float)
    label = np.zeros(20, dtype=int)
    label[[0, 4]] = 1
    credit, percentile = positive_ranks(label, score)
    assert credit.tolist() == [1.0, 0.0]
    assert percentile.tolist() == pytest.approx([18 / 18, 15 / 18])
    # A constant score is a random ranking: each positive's credit is 0.10.
    credit, percentile = positive_ranks(label, np.zeros(20))
    assert credit == pytest.approx([0.1, 0.1]) and percentile.tolist() == [0.5, 0.5]


def test_a_line_averages_its_studies_of_screens_and_the_gap_is_per_line():
    def screen(study, name, condition, label):
        return pd.DataFrame(
            {"split": "test", "design": "anchor", "condition": condition,
             "study_id": study, "list_id": name, "model_id": "L0", "label": label,
             "score": range(10, 0, -1)}
        )  # fmt: skip

    first, last = [1] + [0] * 9, [0] * 9 + [1]
    scores = pd.concat(
        [screen("S", "a", "genetic", first), screen("S", "b", "genetic", last),
         screen("T", "c", "genetic", first),
         screen("T", "d", "treatment", last),  # not in the headline
         screen("T", "e", "genetic", [0] * 10)]  # one class: not in the headline
    )  # fmt: skip
    model = scores.assign(scorer="model")
    control = scores.assign(scorer="pair_identity", score=0.0)
    lists = list_records(pd.concat([model, control]), k=2)
    lines = line_records(lists).set_index("scorer")
    # study S: screens credit 1 and 0 -> 0.5; study T: 1 -> line (0.5 + 1) / 2
    # (pooling the positives would give 2/3)
    assert lines.loc["model", "recall_top10"] == pytest.approx(0.75)
    assert lines.loc["model", ["positives", "studies", "screens"]].tolist() == [3, 2, 3]
    assert lines.loc["pair_identity", "recall_top10"] == pytest.approx(0.1)
    rows = summary(lines.reset_index(), {"all": {"L0"}}, reps=10, seed=0)
    model_row = next(r for r in rows if r["scorer"] == "model")
    assert model_row["minus_pair_identity"] == pytest.approx(0.65)
    assert (model_row["lines"], model_row["positives"]) == (1, 3)


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
    pd.DataFrame(
        {"split": ["train", "train", "val", "test"],
         "model_id": ["T0", "T1", "V0", "E0"], "group": "with_geneeffect"}
    ).to_csv(tmp_path / "split.csv", index=False)  # fmt: skip
    pd.DataFrame(
        {"ModelID": ["T0", "T1", "V0", "E0"],
         "OncotreePrimaryDisease": ["Lung", "Lung", "Lung", "Non-Cancerous"]}
    ).to_csv(tmp_path / "Model.csv", index=False)  # fmt: skip
    labels().to_parquet(tmp_path / "labels.parquet")
    # a line-blind score per pair from another harness: G0-G2 first
    pd.DataFrame(
        {"gene_a": ["G0", "G0", "G1", "G2"], "gene_b": ["G1", "G2", "G3", "G3"],
         "score": [0.2, 0.9, 0.1, 0.3]}
    ).to_parquet(tmp_path / "kg4sl.parquet")  # fmt: skip
    (tmp_path / "config.yaml").write_text(
        f"kind: sl_benchmark_baselines\nseed: 0\noutput_root: {tmp_path}\ntask: base\n"
        f"labels: {tmp_path}/labels.parquet\nsplit: {tmp_path}/split.json\n"
        f"gene_effect: {tmp_path}/ge.csv\nmodel: {tmp_path}/Model.csv\n"
        f"stage_one: null\nextra_scores:\n  kg4sl: {tmp_path}/kg4sl.parquet\n"
        "top_k: 2\nbootstrap: 10\n"
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
    assert {"split": "test", "condition": "genetic", "screens": 1, "two_class": 1,
            "positives": 1} in metrics["screens"]  # fmt: skip
    report = (out / "results.md").read_text()
    assert "Table 1" in report and "Measured GeneEffect" in report
    by = {
        (r["split"], r["scorer"], r["group"]): r
        for r in metrics["summary"]
        if r["stratum"] == "all"
    }
    scorers = (
        "pair_identity",
        "pan_essentiality",
        "measured_geneeffect",
        "labels",
        "random",
        "kg4sl",
    )
    assert {(s, c) for s, c, g in by if g == "all"} == {
        (s, c) for s in ("val", "test") for c in scorers
    }
    assert ("test", "pair_identity", "non_cancer") in by  # E0 is non-cancer
    by = {(s, c): row for (s, c, g), row in by.items() if g == "all"}
    # On E0 the measured GeneEffect ranks the one SL pair (G0, G2) first.
    assert by[("test", "measured_geneeffect")]["auprc_lift"] == pytest.approx(4.0)
    # A list of 4 pairs: its top 10% is 0.4 of a pair, all of it the first pair.
    assert by[("test", "measured_geneeffect")]["recall_top10"] == pytest.approx(0.4)
    # The labels reach the ceiling min(1, 0.1 * 4 / 1); the random score stays below it.
    assert by[("test", "labels")]["recall_top10"] == pytest.approx(0.4)
    assert 0.0 <= by[("test", "random")]["recall_top10"] <= 0.4
    assert by[("test", "kg4sl")]["recall_top10"] == pytest.approx(0.4)
    # flips: G0-G1 in E0 (SL in both training lines) and G0-G2 in V0 (a 0.5 rate
    # counts as an SL majority)
    assert metrics["strata"]["seen_flip"] == 2
    assert "Table 3" in report
