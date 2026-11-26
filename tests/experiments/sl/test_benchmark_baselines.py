"""SL benchmark baselines on a synthetic benchmark: scorer values, and the headline
averaged over a study's genetic screens, a line's studies, then the lines."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.splits import FixedSplit
from src.experiments.sl import benchmark_baselines
from src.experiments.sl.benchmark_baselines import (
    SIDES,
    benchmark_rows,
    cross_line_ordering,
    line_records,
    line_values,
    list_records,
    load_config,
    no_dependency_alone,
    pair_identity,
    positive_ranks,
    run,
    strata,
    summary,
    training_calls,
    with_scorer_means,
)

GENES = ["G0", "G1", "G2", "G3"]


def labels(extra=()) -> pd.DataFrame:
    rows = []
    for line, side_label in (("T0", (1, 0, 0)), ("T1", (1, 1, 0)), ("V0", (1, 0, 0)),
                             ("E0", (0, 1, 0)), *extra):  # fmt: skip
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
    # the same pairs by the training lines' calls alone; E0's label is not read
    pairs = (held["gene_a"] + "-" + held["gene_b"]).to_numpy()
    calls = training_calls(training, held)
    assert {k: pairs[m].tolist() for k, m in calls.items()} == {
        "seen": ["G0-G1", "G0-G2", "G1-G3"],
        "seen_never": ["G1-G3"],
        "seen_mixed": ["G0-G2"],
        "seen_always": ["G0-G1"],
    }
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
    # only gene B can be a dependency alone in an anchor list
    assert no_dependency_alone(effects, kept).tolist() == [False, True]
    pair = kept.assign(design="combinatorial", gene_a=["G1", "G0"], gene_b="G1")
    no_line = effects.rename(index={"E0": "X"})  # a line without GeneEffect
    assert no_dependency_alone(no_line, pair).tolist() == [False] * 2
    assert no_dependency_alone(effects, pair).tolist() == [True, False]


def test_recall_in_the_top_tenth_and_percentiles_of_a_known_list():
    # 20 pairs, 2 known positives ranked 1st and 5th: the top 10% holds 2 pairs.
    score = -np.arange(20, dtype=float)
    label = np.zeros(20, dtype=int)
    label[[0, 4]] = 1
    credit, percentile, at_fpr = positive_ranks(label, score)
    assert credit.tolist() == [1.0, 0.0]
    assert percentile.tolist() == pytest.approx([18 / 18, 15 / 18])
    # 18 negatives: 1.8 false-positive places; 0 and 3 negatives rank above the two.
    assert at_fpr.tolist() == [1.0, 0.0]
    # A constant score is a random ranking: each positive's credit is 0.10.
    credit, percentile, at_fpr = positive_ranks(label, np.zeros(20))
    assert credit == pytest.approx([0.1, 0.1]) and percentile.tolist() == [0.5, 0.5]
    assert at_fpr == pytest.approx([0.1, 0.1])
    # No negative: the percentile and the credit at 10% false positives are undefined.
    assert all(np.isnan(v).all() for v in positive_ranks(np.ones(3), np.zeros(3))[1:])


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


#: The two config keys of the cross-line measures, both empty.
NO_CROSS_LINE = "scorer_means: {}\ncross_line_ordering: []\n"
#: Lines without GeneEffect, one more on each held-out side; their calls differ from
#: V0's and E0's on G0-G1 and G0-G2.
TWINS = (("V1", (0, 1, 0)), ("E1", (1, 0, 0)))


def write_harness(
    tmp_path,
    tail="paired_differences: [[cilantro_sl, kg4sl]]\n" + NO_CROSS_LINE,
    twins=(),
):
    """The synthetic harness inputs and config (``tail`` ends the config); returns the
    config path. ``twins`` adds lines without GeneEffect to the held-out sides."""
    extra = [line for line, _ in twins]
    split = {"schema_version": "cell-line-geneeffect-split-v1", "train": ["T0", "T1"],
             "val": ["V0", *extra[:1]], "test": ["E0", *extra[1:]],
             "unlabeled_train": []}  # fmt: skip
    (tmp_path / "split.json").write_text(json.dumps(split))
    effect = pd.DataFrame(
        np.array([[-1.0, -0.2, 0.0, 0.1], [-0.8, -0.4, 0.0, 0.1],
                  [-0.9, -0.1, 0.0, 0.0], [-0.1, -1.5, -2.0, 0.0]]),
        index=pd.Index(["T0", "T1", "V0", "E0"], name="ModelID"),
        columns=[f"{g} ({i})" for i, g in enumerate(GENES)],
    )  # fmt: skip
    effect.to_csv(tmp_path / "ge.csv")
    pd.DataFrame(
        {"split": ["train", "train", "val", "test", "val", "test"][: 4 + len(extra)],
         "model_id": ["T0", "T1", "V0", "E0", *extra],
         "group": ["with_geneeffect"] * 4 + ["without_geneeffect"] * len(extra)}
    ).to_csv(tmp_path / "split.csv", index=False)  # fmt: skip
    pd.DataFrame(
        {"ModelID": ["T0", "T1", "V0", "E0", *extra],
         "OncotreePrimaryDisease": ["Lung", "Lung", "Lung", "Non-Cancerous",
                                    *["Lung"] * len(extra)]}
    ).to_csv(tmp_path / "Model.csv", index=False)  # fmt: skip
    labels(twins).to_parquet(tmp_path / "labels.parquet")
    # a line-blind score per pair from another harness: G0-G2 first; it does not
    # cover G2-G3 (E0's fourth row)
    pd.DataFrame(
        {"gene_a": ["G0", "G0", "G1", "G2"], "gene_b": ["G1", "G2", "G3", "G3"],
         "score": [0.2, 0.9, 0.1, 0.3], "covered": [True, True, True, False]}
    ).to_parquet(tmp_path / "kg4sl.parquet")  # fmt: skip
    # a per-line score: E0's SL pair (G0, G2) last; G0-G2 is a partner pair
    pd.DataFrame(
        {"model_id": "E0", "gene_a": ["G0", "G0", "G1", "G2"],
         "gene_b": ["G1", "G2", "G3", "G3"], "score": [0.9, 0.1, 0.8, 0.7],
         "partner": [False, True, False, False]}
    ).pipe(lambda f: pd.concat([f.assign(model_id=m) for m in ["E0", "V0", *extra]]))\
        .to_parquet(tmp_path / "cilantro.parquet")  # fmt: skip
    (tmp_path / "config.yaml").write_text(
        f"kind: sl_benchmark_baselines\nseed: 0\noutput_root: {tmp_path}\ntask: base\n"
        f"labels: {tmp_path}/labels.parquet\nsplit: {tmp_path}/split.json\n"
        f"gene_effect: {tmp_path}/ge.csv\nmodel: {tmp_path}/Model.csv\n"
        f"stage_one: null\nextra_scores:\n  kg4sl: {tmp_path}/kg4sl.parquet\n"
        f"  cilantro_sl: {tmp_path}/cilantro.parquet\n"
        "top_k: 2\nbootstrap: 10\n" + tail
    )
    return tmp_path / "config.yaml"


def test_run_scores_every_list(tmp_path):
    out = run(load_config(write_harness(tmp_path)), "r0")
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
        "cilantro_sl",
    )
    # the training lines: every scorer but Cilantro-SL, whose table has no T0, T1 row
    assert {(s, c) for s, c, g in by if g == "all"} == {
        (s, c) for s in ("val", "test") for c in scorers
    } | {("train", c) for c in scorers if c != "cilantro_sl"}
    assert ("test", "pair_identity", "non_cancer") in by  # E0 is non-cancer
    by = {(s, c): row for (s, c, g), row in by.items() if g == "all"}
    # On E0 the measured GeneEffect ranks the one SL pair (G0, G2) first.
    # AUPRC 1 at a positive rate of 0.25: (1 - 0.25) / (1 - 0.25)
    assert by[("test", "measured_geneeffect")]["auprc_corrected"] == pytest.approx(1.0)
    assert by[("test", "measured_geneeffect")]["recall_fpr10"] == pytest.approx(1.0)
    assert "Table 7" in report and "Chance-corrected AUPRC" in report
    # A list of 4 pairs: its top 10% is 0.4 of a place, all of it the first pair, and
    # one SL pair is more than 0.4 places, so the recall is 0.4 / 0.4.
    assert by[("test", "measured_geneeffect")]["recall_top10"] == pytest.approx(1.0)
    # The labels reach the ceiling 1; the random score stays at or below it.
    assert by[("test", "labels")]["recall_top10"] == pytest.approx(1.0)
    assert 0.0 <= by[("test", "random")]["recall_top10"] <= 1.0
    # in sample, T1's rates are its own labels: G0-G1 1.0, G0-G2 0.5, G1-G3 0.0, so
    # its SL pairs G0-G1 and G0-G2 rank first and fill its 0.3 top places
    t1 = pd.read_csv(out / "lines.csv").query(
        "split == 'train' and scorer == 'pair_identity' and model_id == 'T1'"
    )
    assert t1["recall_top10"].item() == pytest.approx(1.0)
    assert "| Model | Training (2 lines)" in report
    table = pd.read_csv(out / "summary.csv")
    assert {"stratum", "scorer", "lines", "recall_top10", "auroc"} <= set(table)
    assert by[("test", "kg4sl")]["recall_top10"] == pytest.approx(1.0)
    assert by[("test", "cilantro_sl")]["recall_top10"] == pytest.approx(0.0)
    # flips: G0-G1 in E0 (SL in both training lines) and G0-G2 in V0 (a 0.5 rate
    # counts as an SL majority)
    assert metrics["strata"]["seen_flip"] == 2
    assert "Table 3" in report
    # Table 4: KG4SL beside the training SL fraction on its 6 of 7 held rows
    entry = metrics["covered"]["kg4sl"]
    assert (entry["rows"], entry["share"]) == (6, pytest.approx(6 / 7))
    assert {(r["split"], r["scorer"]) for r in entry["summary"]} == {
        (s, c) for s in ("val", "test") for c in ("kg4sl", "pair_identity", "labels",
                                                  "random")
    }  # fmt: skip
    # on E0's three covered rows, KG4SL ranks the SL pair (G0, G2) first
    test_kg4sl = next(
        r for r in entry["summary"] if (r["split"], r["scorer"]) == ("test", "kg4sl")
    )
    assert test_kg4sl["auroc"] == pytest.approx(1.0)
    assert "Table 4" in report and "cilantro_sl" not in metrics["covered"]
    assert "row" not in scores.columns
    # the subsets: G0-G2 in V0 and E0; only V0's G1-G3 has no gene below -0.5
    # and the seen pairs by the training lines' calls: one pair of each kind per line
    assert metrics["subsets"] == {
        "seen": {"rows": 6, "share": pytest.approx(6 / 7)},
        "seen_never": {"rows": 2, "share": pytest.approx(2 / 7)},
        "seen_mixed": {"rows": 2, "share": pytest.approx(2 / 7)},
        "seen_always": {"rows": 2, "share": pytest.approx(2 / 7)},
        "partner": {"rows": 2, "share": pytest.approx(2 / 7)},
        "no_dependency_alone": {"rows": 1, "share": pytest.approx(1 / 7)},
    }
    # the paired difference on E0: Cilantro-SL 0.0 minus KG4SL 1.0
    gap = next(
        r for r in metrics["paired_differences"]
        if (r["split"], r["group"], r["stratum"]) == ("test", "all", "all")
    )  # fmt: skip
    assert (gap["scorer"], gap["control"], gap["lines"]) == ("cilantro_sl", "kg4sl", 1)
    assert gap["difference"] == pytest.approx(-1.0)
    assert "Table 5" in report and "partner pairs 2 (28.6%)" in report


def ordering_rows(*rows) -> pd.DataFrame:
    """Rows (split, study, gene A, line, label, score) of combinatorial genetic
    screens of (gene A, B), one screen per (study, line) unless the list is given."""
    frame = pd.DataFrame(
        [r[:6] + (r[6] if len(r) > 6 else "genetic",) for r in rows],
        columns=["split", "study_id", "gene_a", "model_id", "label", "score",
                 "condition"],
    )  # fmt: skip
    return frame.assign(
        design="combinatorial",
        gene_b="B",
        list_id=frame["study_id"] + "|" + frame["model_id"],
    )


def test_a_score_that_is_the_same_in_every_line_orders_at_one_half():
    rng = np.random.default_rng(0)
    rows = [
        (split, study, f"A{pair}", f"{split}{line}", int(rng.integers(2)))
        for split in SIDES
        for study in ("S1", "S2")
        for pair in range(6)
        for line in range(5)
    ]
    frame = ordering_rows(*(r + (0.0,) for r in rows))
    # line 0 of each side has three screens of every study: a mean of three equal
    # scores may differ from the score in its last bit
    first = frame[frame["model_id"].str.endswith("0")]
    frame = pd.concat([frame, first, first], ignore_index=True)
    pair_alone = rng.random(6)[frame["gene_a"].str[1:].astype(int)]
    label = frame["label"].to_numpy(dtype=float)
    for score, value in ((pair_alone, 0.5), (label, 1.0), (-label, 0.0)):
        result = cross_line_ordering(frame, score, reps=20, seed=0)
        assert set(result) == set(SIDES)
        assert {split: r["value"] for split, r in result.items()} == {
            split: value for split in SIDES
        }


def test_cross_line_ordering_pairs_lines_of_one_side_study_and_pair():
    frame = ordering_rows(
        # study S, validation, gene pair (P, B): V0 above V1, tied with V2
        ("val", "S", "P", "V0", 1, 2.0), ("val", "S", "P", "V1", 0, 1.0),
        ("val", "S", "P", "V2", 0, 2.0),
        # (Q, B): V1 above V2, V0 below V2
        ("val", "S", "Q", "V0", 1, 1.0), ("val", "S", "Q", "V1", 1, 3.0),
        ("val", "S", "Q", "V2", 0, 2.0),
        # V3 has both calls for (P, B) in two screens of S: its cell is dropped
        ("val", "S", "P", "V3", 1, 9.0), ("val", "S", "P", "V3", 0, 9.0),
        # a treatment screen and a row without a score are not read
        ("val", "S", "P", "V4", 1, -9.0, "treatment"),
        ("val", "S", "P", "V5", 0, np.nan),
        # lines of test, training and another study: one line per (side, study, pair)
        ("test", "S", "P", "E0", 0, 9.0), ("train", "S", "P", "T0", 0, 9.0),
        ("val", "S2", "P", "V6", 0, 9.0),
    )  # fmt: skip
    frame.loc[7, "list_id"] = "S|V3|second"
    result = cross_line_ordering(frame, frame["score"].to_numpy(), reps=50, seed=0)
    assert set(result) == {"val"}
    val = result["val"]
    # (P, B): V0 > V1 1, V0 = V2 0.5; (Q, B): V0 < V2 0, V1 > V2 1
    assert val["by_study"] == {"S": {"value": 2.5 / 4, "examples": 4, "pairs": 2}}
    counts = (val["examples"], val["pairs"], val["studies"])
    assert val["value"] == 2.5 / 4 and counts == (4, 2, 1)
    low, high = val["interval"]
    assert 0.0 <= low <= high <= 1.0


def test_the_study_mean_weighs_each_study_once():
    frame = ordering_rows(
        ("test", "S1", "P", "E0", 1, 1.0), ("test", "S1", "P", "E1", 0, 0.0),
        *(("test", "S2", "P", f"E{i}", 1, 0.0) for i in range(10)),
        *(("test", "S2", "P", f"N{i}", 0, 1.0) for i in range(10)),
    )  # fmt: skip
    result = cross_line_ordering(frame, frame["score"].to_numpy(), reps=20, seed=0)
    test = result["test"]
    assert {s: (r["value"], r["examples"]) for s, r in test["by_study"].items()} == {
        "S1": (1.0, 1), "S2": (0.0, 100)
    }  # fmt: skip
    assert (test["value"], test["examples"], test["studies"]) == (0.5, 101, 2)
    assert test["interval"] == [0.5, 0.5]


def test_the_harness_reports_the_ordering_for_the_named_scorers(tmp_path):
    tail = "paired_differences: []\nscorer_means: {}\ncross_line_ordering: [labels]\n"
    out = run(load_config(write_harness(tmp_path, tail, twins=TWINS)), "r0")
    metrics = json.loads((out / "metrics.json").read_text())
    ordering = metrics["cross_line_ordering"]["labels"]
    # each side: G0-G1 and G0-G2 differ between its two lines; the labels order them
    assert {s: (r["value"], r["examples"]) for s, r in ordering.items()} == {
        "val": (1.0, 2), "test": (1.0, 2), "train": (1.0, 1)
    }  # fmt: skip
    report = (out / "results.md").read_text()
    assert "Cross-line ordering" in report
    assert (
        "| Labels as the score (check: the ceiling) | 1.000 [1.000, 1.000] (1, 1)"
        in (report)
    )


def test_a_scorer_mean_is_the_mean_of_its_scorers_per_line():
    lines = pd.DataFrame(
        {"split": "val", "scorer": ["a", "a", "b"], "model_id": ["L0", "L1", "L0"],
         "positives": [3, 4, 5], "studies": [2, 1, 1], "screens": [4, 1, 2],
         "recall_top10": [0.2, 0.5, 0.4], "recall_fpr10": [0.1, 0.5, 0.3],
         "auroc": [0.6, 0.5, 0.8], "auprc_corrected": [0.0, 0.0, 0.5]}
    )  # fmt: skip
    out = with_scorer_means(lines, {"m": ["a", "b"]})
    assert list(out.columns) == list(lines.columns)
    assert out.iloc[:3].equals(lines)
    derived = out[out["scorer"] == "m"]
    # L1 has no score of b: no derived row
    assert derived["model_id"].tolist() == ["L0"]
    row = derived.iloc[0]
    metrics = ["recall_top10", "recall_fpr10", "auroc", "auprc_corrected"]
    assert row[metrics].tolist() == pytest.approx([0.3, 0.2, 0.7, 0.25])
    assert row[["positives", "studies", "screens"]].tolist() == [3, 2, 4]


def test_a_paired_difference_can_name_a_scorer_mean(tmp_path, monkeypatch):
    monkeypatch.setitem(benchmark_baselines.NAMES, "two_scores", "Mean of two scores")
    tail = (
        "paired_differences: [[kg4sl, two_scores]]\n"
        "scorer_means: {two_scores: [kg4sl, cilantro_sl]}\ncross_line_ordering: []\n"
    )
    out = run(load_config(write_harness(tmp_path, tail)), "r0")
    metrics = json.loads((out / "metrics.json").read_text())
    # on E0: KG4SL 1.0, Cilantro-SL 0.0, so their mean is 0.5
    gap = next(
        r for r in metrics["paired_differences"]
        if (r["split"], r["group"], r["stratum"]) == ("test", "all", "all")
    )  # fmt: skip
    assert (gap["scorer"], gap["control"], gap["lines"]) == ("kg4sl", "two_scores", 1)
    assert gap["difference"] == pytest.approx(0.5)
    assert "| Mean of two scores | — | " in (out / "results.md").read_text()


@pytest.mark.parametrize(
    "tail",
    [
        "scorer_means: {not_a_scorer: [kg4sl]}\ncross_line_ordering: []\n",
        "scorer_means: {embedding: [kg4sl, pair_identity]}\ncross_line_ordering: []\n",
        "scorer_means: {}\ncross_line_ordering: [not_a_scorer]\n",
        "scorer_means: {}\n",
    ],
)
def test_a_scorer_mean_of_an_unknown_scorer_is_refused(tmp_path, tail):
    path = write_harness(tmp_path, "paired_differences: []\n" + tail)
    with pytest.raises(ValueError):
        load_config(path)


def test_an_extra_training_line_gives_training_rows():
    frame = labels()
    split = FixedSplit(train=("T0",), val=(), test=("E0",), unlabeled_train=())
    kept, dropped = benchmark_rows(frame, split, set(GENES))
    assert "T1" not in set(kept["model_id"]) and dropped["rows_line_not_in_split"] > 0
    more, dropped = benchmark_rows(frame, split, set(GENES), ["T1"])
    assert set(more.loc[more["model_id"] == "T1", "split"]) == {"train"}
    assert dropped["rows_line_not_in_split"] == int(
        (~frame["model_id"].isin(["T0", "T1", "E0"]) & frame["label"].notna()).sum()
    )


def test_published_baseline_config_names_known_scorers():
    from src.experiments.run_dir import REPO_ROOT

    config = benchmark_baselines.load_config(
        REPO_ROOT / "configs" / "sl" / "benchmark_published.yaml"
    )
    scorers = set(config["extra_scores"])
    assert scorers <= set(benchmark_baselines.NAMES)
    assert {"slmgae", "esm4sl", "mvgcn_isl", "musl", "promptcasl"} <= scorers
    assert not {"paris", "slidr", "dekegel_paralog"} & scorers
    assert ["mvgcn_isl", "mvgcn_isl_mean_input"] in config["paired_differences"]
