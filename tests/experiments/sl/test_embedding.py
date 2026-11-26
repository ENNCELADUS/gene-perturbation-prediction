"""The gene-knowledge, cell-set model's experiment on a synthetic world: GeneEffect is
gene vector times line state, a pair is SL where both genes are dependencies in the
line; stage one lowers the loss on the SL training lines, stage two ranks held-out
lines' pairs above random, and the inputs' readers fail closed."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments.sl import benchmark_baselines, embedding
from src.experiments.sl.benchmark_baselines import HEADLINE

CONFIG = {
    "seed": 0,
    "top_k": 20,
    "model": {"hidden": 16, "head_hidden": 16},
    "cells": {"per_step": 6, "dropout": 0.2},
    "stage_one": {"max_epochs": 12, "patience": 3, "learning_rate": 1.0e-2,
                  "weight_decay": 0.0, "lines_per_step": 16, "genes_per_step": 30},
    "stage_two": {"max_epochs": 6, "patience": 3, "steps_per_epoch": 30,
                  "lines_per_step": 4, "rows_per_line": 64, "learning_rate": 1.0e-2,
                  "backbone_learning_rate": 1.0e-3, "weight_decay": 0.0,
                  "geneeffect_weights": [1.0]},
    "eval": {"rows_per_chunk": 64, "genes_per_chunk": 16, "other_lines": 1},
    "line_term": None,
}  # fmt: skip
SIDES = {"train": 8, "val": 4, "test": 4}
LINE_CONFIG = CONFIG | {
    "line_term": {"reference_lines": 3, "max_epochs": 4, "patience": 2,
                  "learning_rate": 1.0e-2, "own_expression": True, "gene_in_line": {},
                  "joint": False, "input": "model", "cross_line": None,
                  "pair_in_line": False}
}  # fmt: skip

CROSS_CONFIG = LINE_CONFIG | {
    "line_term": LINE_CONFIG["line_term"]
    | {"cross_line": {"weight": 0.5, "examples_per_step": 32}}
}

JOINT_CONFIG = CROSS_CONFIG | {"line_term": CROSS_CONFIG["line_term"] | {"joint": True}}


def world(seed: int = 0) -> tuple[embedding.Data, np.ndarray, np.ndarray]:
    """40 fit lines with cells, 40 bulk-only fit lines and 16 SL lines; 30 genes.
    GeneEffect is ``z`` plus a gene offset that is the same in every line; the SL
    labels read ``z``. Returns the data, the true GeneEffect and the offset."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(30)]
    sl = [f"{side}{i}" for side, n in SIDES.items() for i in range(n)]
    with_cells = [f"C{i}" for i in range(40)] + sl
    lines = with_cells + [f"B{i}" for i in range(40)]
    state = rng.normal(size=(len(lines), 2))
    gene = rng.normal(size=(len(genes), 2))
    z = state @ gene.T
    expression = state @ rng.normal(size=(2, 6)) + 0.1 * rng.normal(size=(96, 6))
    to_cell = rng.normal(size=(2, 8))
    counts = np.array([rng.integers(5, 13) for _ in with_cells] + [0] * 40)
    cells = np.concatenate(
        [state[i] @ to_cell + 0.3 * rng.normal(size=(n, 8))
         for i, n in enumerate(counts) if n]
    )  # fmt: skip
    measured = np.full(z.shape, np.nan)
    labelled = [i for i, m in enumerate(lines) if not m.startswith(("val", "test"))]
    measured[labelled] = z[labelled] + 0.1 * rng.normal(size=(len(labelled), 30))
    records = []
    for name in sl:
        line = lines.index(name)
        a, b = rng.integers(0, 30, size=(2, 150))
        keep = a != b
        total = z[line, a[keep]] + z[line, b[keep]]
        records.append(
            pd.DataFrame(
                {"split": name.rstrip("0123456789"), "design": "combinatorial",
                 "condition": HEADLINE, "study_id": "S", "list_id": f"S:{name}",
                 "model_id": name, "gene_a": [genes[i] for i in a[keep]],
                 "gene_b": [genes[i] for i in b[keep]],
                 "label": (total < np.quantile(total, 0.2)).astype(int),
                 "line": line, "a": a[keep], "b": b[keep]}
            )
        )  # fmt: skip
    present = np.ones(30, dtype=bool)
    present[7] = False
    knowledge = gene @ rng.normal(size=(2, 5)) * present[:, None]
    offset = 3 * rng.normal(size=30)  # drawn last: the earlier draws do not change
    measured += offset
    fit = [i for i, m in enumerate(lines) if m[0] in "CB"]
    # the SL lines' one gene-in-line source: z itself and a presence flag
    sl_at = [lines.index(m) for m in sl]
    own = np.concatenate([z[sl_at][:, :, None], np.ones((16, 30, 1))], axis=2)
    own = np.concatenate([own, np.zeros((16, 1, 2))], axis=1)
    in_line_at = torch.full((len(lines),), -1)
    in_line_at[sl_at] = torch.arange(16)
    centre, scale = embedding.target_scale(pd.DataFrame(measured[fit]))
    data = embedding.Data(
        genes=genes,
        lines=lines,
        expression=torch.tensor(expression, dtype=torch.float32),
        cells=torch.tensor(cells, dtype=torch.float16),
        offsets=torch.tensor(np.concatenate([[0], np.cumsum(counts)])),
        targets=torch.tensor((measured - centre) / scale, dtype=torch.float32),
        fit=torch.tensor(fit),
        stop=torch.tensor([i for i, m in enumerate(lines) if m.startswith("train")]),
        rows=pd.concat(records, ignore_index=True),
        sources={"knowledge": (torch.tensor(knowledge, dtype=torch.float32),
                               torch.tensor(present))},
        m_g=measured[fit].mean(axis=0),
        centre=centre,
        scale=scale,
        in_line={"own": torch.tensor(own, dtype=torch.float16)},
        in_line_at=in_line_at,
    )  # fmt: skip
    return data, z + offset, offset


def test_the_target_is_geneeffect_itself_on_one_scale():
    frame = pd.DataFrame(
        {"A": [0.0, -2.0], "B": [1.0, 1.0], "C": [np.nan, 3.0]}, index=["L1", "L2"]
    )
    centre, scale = embedding.target_scale(frame)
    assert np.isclose(centre, 0.6) and np.isclose(scale, 2.64**0.5)


def test_the_model_learns_geneeffect_and_ranks_held_out_pairs():
    data, effect, offset = world()
    torch.manual_seed(0)
    model = embedding.build_model(CONFIG, data)
    result = embedding.train(model, data, CONFIG)
    history = result["stage_one"]
    assert history[0]["epoch"] == 0 and history[0]["train_loss"] is None
    assert min(r["stop_loss"] for r in history) < 0.5 * history[0]["stop_loss"]
    # the kept checkpoint is a trained one, so the frozen head reads embeddings (on an
    # untrained model it reads a constant, and its recall is about 0.1); the gene
    # offset makes the untrained model's loss high
    assert min(history, key=lambda r: r["stop_loss"])["epoch"] >= 1
    assert result["phases"][0]["val_recall_top10"] > 0.2
    phases = result["phases"]
    assert [p["phase"] for p in phases] == ["frozen", "trained further"]
    assert result["chosen"] == embedding.choose_phase(phases)
    assert all(p["val_recall_se"] >= 0 and np.isfinite(p["val_own_minus_other_line"])
               for p in phases)  # fmt: skip
    assert np.isfinite(result["scores"]).all()
    assert result["line_term"] is None
    # the controls: with the fit lines' mean input a pair has one score in every line
    controls = result["controls"]
    assert set(controls) == {"mean_input", "other_line"}
    assert all(np.isfinite(c).all() for c in controls.values())
    pair = ["design", "gene_a", "gene_b"]
    spread = {
        name: data.rows.assign(score=score).groupby(pair)["score"].agg(np.ptp).max()
        for name, score in {**controls, "own": result["scores"]}.items()
    }
    assert spread["mean_input"] < 1e-4 < spread["own"]
    assert not np.allclose(controls["other_line"], result["scores"])
    metrics = embedding.sl_metrics(data.rows, result["scores"], 20)
    assert {s: metrics[s]["lines"] for s in SIDES} == SIDES
    # a random score has a recall in the top 10% near 0.1 and an AUROC near 0.5
    assert metrics["val"]["recall_top10"] > 0.3
    assert metrics["test"]["recall_top10"] > 0.3 and metrics["test"]["auroc"] > 0.7
    # the held-out lines' GeneEffect, never a target, is predicted from their inputs
    held = [m for m in data.lines if m.startswith(("val", "test"))]
    table = embedding.geneeffect_table(model, data, held, CONFIG)
    truth = effect[[data.lines.index(m) for m in held]]
    assert np.corrcoef(table.to_numpy().ravel(), truth.ravel())[0, 1] > 0.9
    # the model holds the gene mean: the offset that is the same in every line
    assert np.corrcoef(table.to_numpy().mean(axis=0), offset)[0, 1] > 0.9
    # and the line-specific part: each gene centred over the held-out lines
    centred = table.to_numpy() - table.to_numpy().mean(axis=0)
    assert (
        np.corrcoef(centred.ravel(), (truth - truth.mean(axis=0)).ravel())[0, 1] > 0.5
    )
    report_metrics = {
        "run_id": "synthetic", "sl": metrics, "phases": phases,
        "chosen_phase": result["chosen"], "stage_one_history": history,
        "stage_one": {s: {"lines": n, "line_spearman": 0.1} for s, n in SIDES.items()},
        "stage_one_base": {s: {"lines": n} for s, n in SIDES.items()},
        "counts": {"fit_lines": 80, "fit_lines_with_cells": 40, "stop_lines": 8,
                   "sl_lines_left_out": 8, "patient_mates_left_out": 0, "cells": 1,
                   "genes": 30, "gene_source_coverage": {"knowledge": 29},
                   "training_rows": 1, "bulk_training_lines": 0,
                   "held_rows_seen_only_in_bulk_training_lines": {"val": 0, "test": 0}},
        "sl_line_term": None, "line_term": None,
    }  # fmt: skip
    report = embedding.results_markdown(report_metrics)
    assert "**Table 1.**" in report and "trained further" in report
    assert "No line term" in report
    with_line_term = embedding.results_markdown(
        report_metrics
        | {"sl_line_term": {n: metrics for n in ("mean_input", "line_term",
                                                 "line_term_other_line")},
           "line_term": {"val_recall_top10": 0.3, "history": phases[0]["history"]}}
    )  # fmt: skip
    assert "The model with the line term" in with_line_term
    assert "| line term |" in with_line_term


def test_line_term_is_a_deviation_from_the_training_lines():
    data, *_ = world()
    torch.manual_seed(0)
    model = embedding.build_model(LINE_CONFIG, data)
    pairs = embedding.pair_rows(data, LINE_CONFIG)
    every = np.arange(len(data.rows))
    torch.nn.init.normal_(model.mean_token[-1].weight, std=0.5)
    model_scores = embedding.pair_scores(model, data, pairs, every, LINE_CONFIG)
    controls = embedding.line_input_controls(model, data, pairs, LINE_CONFIG)
    assert not np.allclose(
        model_scores, controls["mean_input"]
    )  # the model reads the line
    # an untrained line head gives 0: the score is the model's own score in the line
    start = embedding.line_term_scores(model, data, pairs, every, LINE_CONFIG)
    assert np.allclose(start, model_scores, atol=1e-5)
    # read in another line, the model's score is the other-line control
    other = embedding.other_lines(data, LINE_CONFIG)[0]
    moved = embedding.line_term_scores(
        model, data, pairs, every, LINE_CONFIG, other.__getitem__
    )
    assert np.allclose(moved, controls["other_line"], atol=1e-5)
    torch.nn.init.normal_(model.line_head[-1].weight, std=0.5)
    own = embedding.line_term_scores(model, data, pairs, every, LINE_CONFIG)
    assert np.isfinite(own).all() and not np.allclose(own, model_scores)
    # a line head whose input is the same in every line gives no deviation
    data.expression[:] = data.expression[0]
    data.offsets[:] = 0  # no line has a cell
    data.in_line["own"][:] = data.in_line["own"][0]
    same = embedding.line_term_scores(model, data, pairs, every, LINE_CONFIG)
    base = embedding.pair_scores(model, data, pairs, every, LINE_CONFIG)
    assert np.allclose(same, base, atol=1e-4)


def test_line_term_scores_follow_the_definition_row_by_row():
    data, *_ = world()
    data.rows.loc[[0, 1], "a"] = -1  # anchors without a gene vector
    torch.manual_seed(0)
    model = embedding.build_model(LINE_CONFIG, data)
    torch.nn.init.normal_(model.mean_token[-1].weight, std=0.5)
    torch.nn.init.normal_(model.line_head[-1].weight, std=0.5)
    pairs = embedding.pair_rows(data, LINE_CONFIG)
    rows = np.array([0, 1, 5, 400, len(data.rows) - 1])
    got = embedding.line_term_scores(model, data, pairs, rows, LINE_CONFIG)
    other = embedding.other_lines(data, LINE_CONFIG)[0]
    got_other = embedding.line_term_scores(
        model, data, pairs, rows, LINE_CONFIG, other.__getitem__
    )
    assert np.isfinite(got).all() and np.isfinite(got_other).all()
    with torch.no_grad():
        u = model.gene_vectors()

        def features(line: int, gene: int) -> dict[str, torch.Tensor]:
            at = int(data.in_line_at[line])
            return {  # gene -1 reads zeros
                n: (torch.zeros(t.shape[2]) if gene < 0 else t[at, gene].float())[None]
                for n, t in data.in_line.items()
            }

        def deviation(line: int, a: int, b: int, c: torch.Tensor) -> torch.Tensor:
            tokens = embedding.one_line(model, data, line, None, None)
            return model.line_term(
                u[[a]], u[[b]], c, features(line, a), features(line, b),
                torch.zeros(1, 0), *tokens,
            )[0]  # fmt: skip

        def score(line: int, a: int, b: int, c: torch.Tensor) -> torch.Tensor:
            tokens = embedding.one_line(model, data, line, None, None)
            return model.pair(u[[a]], u[[b]], c, *tokens)[0]

        for k, i in enumerate(rows):
            a, b, c = int(pairs.a[i]), int(pairs.b[i]), pairs.combinatorial[[i]]
            reference = torch.stack(
                [deviation(r, a, b, c) for r in sorted(pairs.draws)]
            ).mean()
            for line, value in ((int(pairs.line[i]), got[k]),
                                (other[int(pairs.line[i])], got_other[k])):  # fmt: skip
                want = score(line, a, b, c) + deviation(line, a, b, c) - reference
                assert abs(value - float(want)) < 1e-5


def test_the_line_term_trains_alone_and_train_reports_it():
    data, *_ = world()
    torch.manual_seed(0)
    model = embedding.build_model(LINE_CONFIG, data)
    result = embedding.train(model, data, LINE_CONFIG)
    line = result["line_term"]
    assert set(line) == {"val_recall_top10", "history", "scores", "other_line"}
    assert np.isfinite(line["scores"]).all()
    assert not np.allclose(line["scores"], line["other_line"])
    assert not np.allclose(line["scores"], result["scores"])
    assert not np.allclose(line["scores"], result["controls"]["mean_input"])
    before = copy.deepcopy(model.state_dict())
    state, _, history = embedding.fit_line_term(
        model, data, embedding.pair_rows(data, LINE_CONFIG), LINE_CONFIG,
        torch.Generator().manual_seed(1),
    )  # fmt: skip
    moved = {k for k in before if not torch.equal(before[k], state[k])}
    assert moved and all(k.startswith(("line_head.", "in_line.")) for k in moved)
    assert all(p.requires_grad for p in model.parameters())
    assert history and all("val_recall_top10" in r for r in history)


def test_the_phase_is_the_most_line_specific_one_within_one_standard_error():
    def phase(recall, se, specific):
        return {"val_recall_top10": recall, "val_recall_se": se,
                "val_own_minus_other_line": specific}  # fmt: skip

    # the second phase is within one standard error of the best and more line-specific
    assert (
        embedding.choose_phase([phase(0.50, 0.02, 0.01), phase(0.49, 0.03, 0.05)]) == 1
    )
    # a phase below the floor is not a candidate, however line-specific
    assert (
        embedding.choose_phase([phase(0.50, 0.02, 0.01), phase(0.45, 0.03, 0.09)]) == 0
    )
    # a tie goes to the earlier phase
    assert (
        embedding.choose_phase([phase(0.50, 0.02, 0.03), phase(0.50, 0.02, 0.03)]) == 0
    )


def test_training_reads_only_training_rows_of_two_class_genetic_screens():
    data, *_ = world()
    rows = data.rows.copy()
    one_class = rows["model_id"] == "train0"
    rows.loc[one_class, "label"] = 0
    treated = rows["model_id"] == "train1"
    rows.loc[treated, "condition"] = "drug"
    keep = embedding.training_rows(rows)
    assert set(rows.loc[keep, "split"]) == {"train"}
    assert not keep[one_class.to_numpy()].any() and not keep[treated.to_numpy()].any()
    data.rows = rows
    draws = embedding.pair_rows(data, CONFIG).draws
    assert sorted(draws) == sorted(data.lines.index(f"train{i}") for i in range(2, 8))
    totals = [float(rates.sum()) for _, rates in draws.values()]
    assert np.allclose(totals, totals[0])  # every line weighs the same


def test_cross_line_examples_pair_two_lines_of_one_study_for_one_pair():
    def screen(study, line, rows, split="train"):
        return [
            {"split": split, "design": "combinatorial", "condition": HEADLINE,
             "study_id": study, "list_id": f"{study}:{line}", "model_id": line[:2],
             "gene_a": a, "gene_b": b, "label": label}
            for a, b, label in rows
        ]  # fmt: skip

    records = [
        *screen("S1", "L1", [("A", "B", 1), ("X1", "Y", 0)]),
        *screen("S1", "L2", [("A", "B", 0), ("X2", "Y", 1)]),
        *screen("S1", "L3", [("A", "B", 0), ("X3", "Y", 1)]),
        # one line, both calls in two screens: never an example with itself
        *screen("S1", "L4a", [("A", "B", 1), ("X4", "Y", 0)]),
        *screen("S1", "L4b", [("A", "B", 0), ("X5", "Y", 1)]),
        # not training rows: a validation line and a screen with one class
        *screen("S1", "L5", [("A", "B", 0), ("X6", "Y", 1)], split="val"),
        *screen("S1", "L6", [("A", "B", 0), ("X7", "Y", 0)]),
        # a study whose lines never differ on a pair
        *screen("S2", "L1", [("A", "B", 0), ("X8", "Y", 1)]),
        *screen("S2", "L2", [("A", "B", 0), ("X9", "Y", 1)]),
    ]
    rows = pd.DataFrame(records)
    examples = embedding.cross_line_examples(rows)
    assert len(examples) == 1  # S2 has no example
    positive, negative = examples[0]
    got = {
        (rows.at[p, "list_id"], rows.at[n, "list_id"])
        for p, n in zip(positive.tolist(), negative.tolist())
    }
    assert got == {
        ("S1:L1", "S1:L2"), ("S1:L1", "S1:L3"), ("S1:L1", "S1:L4b"),
        ("S1:L4a", "S1:L2"), ("S1:L4a", "S1:L3"),
    }  # fmt: skip
    assert len(positive) == len(got)
    assert (rows.loc[positive, "gene_a"] == "A").all()


def test_a_score_that_ignores_the_line_gives_log_two():
    data, *_ = world()
    torch.manual_seed(0)
    model = embedding.build_model(CROSS_CONFIG, data)  # untrained
    pairs = embedding.pair_rows(data, CROSS_CONFIG)
    assert pairs.cross and all(p.dtype == torch.long for p, _ in pairs.cross)
    loss = embedding.cross_line_loss(
        model, data, pairs, CROSS_CONFIG, torch.Generator().manual_seed(0), False
    )
    assert abs(float(loss.detach()) - math.log(2)) < 1e-6


def test_the_cross_line_loss_trains_only_the_line_term_when_the_model_is_frozen():
    data, *_ = world()
    torch.manual_seed(0)
    model = embedding.build_model(CROSS_CONFIG, data)
    torch.nn.init.normal_(model.mean_token[-1].weight, std=0.5)
    before = copy.deepcopy(model.state_dict())
    state, _, history = embedding.fit_line_term(
        model, data, embedding.pair_rows(data, CROSS_CONFIG), CROSS_CONFIG,
        torch.Generator().manual_seed(1),
    )  # fmt: skip
    moved = {k for k in before if not torch.equal(before[k], state[k])}
    assert any(k.startswith("line_head.") for k in moved)
    assert all(k.startswith(("line_head.", "in_line.")) for k in moved)
    assert history


def test_cross_line_without_an_example_is_refused():
    data, *_ = world()
    train = data.rows["split"] == "train"
    # the call depends on the pair alone, so two training lines never differ on it
    data.rows.loc[train, "label"] = (
        (data.rows.loc[train, "a"] + data.rows.loc[train, "b"]) % 5 == 0
    ).astype(int)
    assert embedding.training_rows(data.rows).any()
    embedding.pair_rows(data, LINE_CONFIG)  # no cross_line: no example needed
    with pytest.raises(ValueError, match="cross_line"):
        embedding.pair_rows(data, CROSS_CONFIG)


def test_joint_training_moves_the_model_and_the_line_term_together():
    data, *_ = world()
    torch.manual_seed(0)
    model = embedding.build_model(JOINT_CONFIG, data)
    before = copy.deepcopy(model.state_dict())
    state, _, history = embedding.fit_pairs(
        model, data, embedding.pair_rows(data, JOINT_CONFIG), JOINT_CONFIG,
        torch.Generator().manual_seed(1), 1.0, with_line_term=True,
    )  # fmt: skip
    moved = {k for k in before if not torch.equal(before[k], state[k])}
    for part in ("line_head.", "head.", "mean_token.", "gene_encoder.", "deviation"):
        assert any(k.startswith(part) for k in moved), part
    assert history and all(p.requires_grad for p in model.parameters())


def test_joint_training_has_no_separate_line_term_phase(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("fit_line_term runs in joint training")

    monkeypatch.setattr(embedding, "fit_line_term", refuse)
    data, *_ = world()
    torch.manual_seed(0)
    model = embedding.build_model(JOINT_CONFIG, data)
    result = embedding.train(model, data, JOINT_CONFIG)
    line = result["line_term"]
    chosen = result["phases"][result["chosen"]]
    assert line["history"] == chosen["history"]
    assert line["val_recall_top10"] == chosen["val_recall_top10"]
    every = np.arange(len(data.rows))
    pairs = embedding.pair_rows(data, JOINT_CONFIG)
    assert np.allclose(
        line["scores"],
        embedding.line_term_scores(model, data, pairs, every, JOINT_CONFIG),
    )


def with_other_lines(config: dict, count: int) -> dict:
    return config | {"eval": config["eval"] | {"other_lines": count}}


def test_each_other_line_is_another_line_and_the_first_is_the_old_control():
    data, *_ = world()
    maps = embedding.other_lines(data, with_other_lines(CONFIG, 3))
    assert len(maps) == 3
    order = np.random.default_rng(0).permutation(np.unique(data.rows["line"]))
    assert maps[0] == dict(zip(order.tolist(), np.roll(order, 1).tolist()))
    for line in order.tolist():
        others = [m[line] for m in maps]
        assert line not in others and len(set(others)) == 3


def test_more_other_lines_than_benchmark_lines_is_refused():
    data, *_ = world()
    lines = data.rows["line"].nunique()
    embedding.other_lines(data, with_other_lines(CONFIG, lines - 1))
    with pytest.raises(ValueError, match="other_lines"):
        embedding.other_lines(data, with_other_lines(CONFIG, lines))


def test_a_run_writes_one_table_for_each_other_line():
    data, *_ = world()
    config = with_other_lines(LINE_CONFIG, 2)
    torch.manual_seed(0)
    model = embedding.build_model(config, data)
    torch.nn.init.normal_(model.mean_token[-1].weight, std=0.5)
    pairs = embedding.pair_rows(data, config)
    controls = embedding.line_input_controls(model, data, pairs, config)
    assert set(controls) == {"mean_input", "other_line", "other_line_2"}
    assert not np.allclose(controls["other_line"], controls["other_line_2"])
    result = embedding.train(model, data, config)
    assert set(result["controls"]) == set(controls)
    line = result["line_term"]
    assert set(line) == {
        "val_recall_top10", "history", "scores", "other_line", "other_line_2"
    }  # fmt: skip
    assert not np.allclose(line["other_line"], line["other_line_2"])


def test_cell_batch_caps_drops_and_pads():
    data, *_ = world()
    generator = torch.Generator().manual_seed(0)
    lines = torch.tensor([0, 1, len(data.lines) - 1])  # two with cells, one bulk-only
    counts = (data.offsets[lines + 1] - data.offsets[lines]).tolist()
    cells, mask = embedding.cell_batch(data, lines, None, None)
    assert mask.sum(dim=1).tolist() == counts and counts[2] == 0
    assert torch.equal(cells[0, : counts[0]], data.cells[: counts[0]])
    capped, mask = embedding.cell_batch(data, lines, 3, generator)
    assert capped.shape[1] == 3 and mask.sum(dim=1).tolist() == [3, 3, 0]
    assert embedding.cell_batch(data, lines, 3, generator, dropout=1.0) == (None, None)
    assert embedding.cell_batch(data, lines[2:], None, None) == (None, None)


def test_gene_sources_align_to_the_genes_and_cells_need_their_cache(tmp_path: Path):
    path = tmp_path / "source.npz"
    np.savez(
        path,
        axis=np.array(["B", "A", "X"]),
        present=np.array([True, True, False]),
        vectors=np.array([[1.0, 3.0], [3.0, 5.0], [9.0, 9.0]]),
    )
    sources = embedding.read_sources({"s": str(path)}, ["A", "B", "X", "Y"])
    vectors, present = sources["s"]
    assert present.tolist() == [True, True, False, False]
    assert np.allclose(vectors, [[1, 1], [-1, -1], [0, 0], [0, 0]])
    with pytest.raises(ValueError, match="covers no gene"):
        embedding.read_sources({"s": str(path)}, ["Y"])
    for name, values in (("L1", [[1.0, 1.0], [3.0, 3.0]]), ("L2", [[10.0, 10.0]])):
        (tmp_path / name).mkdir()
        np.save(tmp_path / name / "embeddings.npy", np.array(values, np.float32))
    cells, counts = embedding.read_cells(tmp_path, ["L1", "L2"], {"L1"})
    assert counts.tolist() == [2, 1] and cells.dtype == np.float16
    assert np.allclose(cells, [[-1, -1], [1, 1], [8, 8]])  # centred on L1's mean cell
    with pytest.raises(FileNotFoundError, match="L3"):
        embedding.read_cells(tmp_path, ["L1", "L3"], {"L1"})
    cells, counts = embedding.read_cells(None, ["L1", "L3"], {"L1"})  # no cell tokens
    assert cells.shape == (0, 1) and counts.tolist() == [0, 0]


def test_gene_in_line_features_align_scale_on_the_fit_lines_and_fail_closed(
    tmp_path: Path,
):
    (tmp_path / "genes.json").write_text(json.dumps(["B", "A", "X"]))
    nan = [np.nan, np.nan]
    for name, values in (
        ("L1", [[1.0, 10.0], [3.0, 30.0], nan]),
        ("L2", [[3.0, 30.0], [5.0, 50.0], [9.0, 90.0]]),
        ("L3", [[5.0, 50.0], nan, [0.0, 0.0]]),
    ):
        np.save(tmp_path / f"{name}.npy", np.array(values, np.float16))
    out = embedding.read_in_line(
        tmp_path, ["L1", "L2", "L3"], ["A", "B", "Y"], fit=["L1", "L2"]
    )
    assert out.shape == (3, 4, 3) and out.dtype == np.float16
    root = 2**0.5  # the fit lines' values of the model's genes: 3, 1, 5, 3
    assert np.allclose(out[:, 0, 0], [0, root, 0], atol=1e-3)  # A; L3 has no value
    assert np.allclose(out[:, 1, 0], [-root, 0, root], atol=1e-3)  # B
    assert out[:, :, 2].tolist() == [[1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]]
    assert not out[:, 2:].any()  # a gene off the cache's axis, and the zero row
    with pytest.raises(FileNotFoundError, match="L4"):
        embedding.read_in_line(tmp_path, ["L1", "L4"], ["A"], fit=["L1"])
    huge = tmp_path / "huge"
    huge.mkdir()
    (huge / "genes.json").write_text(json.dumps(["A", "B"]))
    for name, values in (("L1", [[0.0], [np.nan]]), ("L2", [[1.0], [np.nan]])):
        np.save(huge / f"{name}.npy", np.array(values, np.float16))
    np.save(huge / "L3.npy", np.array([[60000.0], [np.nan]], np.float16))
    with pytest.raises(ValueError, match="huge"):  # (60000 - 0.5) / 0.5 > float16
        embedding.read_in_line(huge, ["L1", "L3"], ["A", "B"], fit=["L1", "L2"])


def test_own_expression_is_the_lines_value_of_the_gene():
    frame = pd.DataFrame({"A": [0.5, -1.0], "C": [2.0, 3.0]}, index=["L1", "L2"])
    out = embedding.own_expression(frame, ["L2", "L1"], ["A", "B"])
    assert out.shape == (2, 3, 2)
    assert out[:, 0].tolist() == [[-1.0, 1.0], [0.5, 1.0]]
    assert not out[:, 1:].any()  # B is not an expression gene; the zero row
    frame.loc["L1", "A"] = 1.0e6
    with pytest.raises(ValueError, match="the expression"):
        embedding.own_expression(frame, ["L2", "L1"], ["A", "B"])


def test_the_config_loads_and_a_missing_key_is_refused(tmp_path: Path):
    config = embedding.load_config(Path("configs/sl/embedding.yaml"))
    assert config["kind"] == "sl_embedding" and len(config["gene_sources"]) == 3
    for name, cache in (
        ("embedding_mean_token", None),
        ("embedding_cellnet", "data/cellnet_basal_embeddings/v1"),
    ):
        variant = embedding.load_config(Path(f"configs/sl/{name}.yaml"))
        assert variant["cell_embeddings"] == cache
        assert {k: v for k, v in variant.items() if k != "cell_embeddings"} == {
            k: v for k, v in config.items() if k != "cell_embeddings"
        }
    slow = embedding.load_config(
        Path("configs/sl/embedding_mean_token_slow_stage_one.yaml")
    )
    assert slow["stage_one"]["learning_rate"] == 1.0e-4
    assert slow["cell_embeddings"] is None
    text = Path("configs/sl/embedding.yaml").read_text().replace("top_k: 20\n", "")
    (tmp_path / "bad.yaml").write_text(text)
    with pytest.raises(ValueError, match="top_k"):
        embedding.load_config(tmp_path / "bad.yaml")


def test_a_line_term_without_the_new_keys_is_refused(tmp_path: Path):
    text = Path("configs/sl/embedding_direct.yaml").read_text()
    assert (
        embedding.load_config(Path("configs/sl/embedding_direct.yaml"))["eval"][
            "other_lines"
        ]
        == 1
    )
    for line, key in (("  joint: false\n", "joint"), ("  input: model\n", "input"),
                      ("  cross_line: null\n", "cross_line"),
                      ("  pair_in_line: false\n", "pair_in_line"),
                      ("space_measured_share: null\n", "space_measured_share"),
                      ("bulk_training_lines: false\n", "bulk_training_lines"),
                      ):  # fmt: skip
        assert line in text
        (tmp_path / "bad.yaml").write_text(text.replace(line, ""))
        with pytest.raises(ValueError, match=key):
            embedding.load_config(tmp_path / "bad.yaml")
    bad = {
        "input": text.replace("  input: model\n", "  input: cells\n"),
        "cross_line": text.replace(
            "  cross_line: null\n", "  cross_line: {weight: 0.5}\n"
        ),
        "other_lines": text.replace("other_lines: 1", "other_lines: 0"),
        # the pair-in-line features need the wide expression cache
        "pair_in_line": text.replace(
            "  pair_in_line: false\n", "  pair_in_line: true\n"
        ),
    }
    for key, wrong in bad.items():
        (tmp_path / "bad.yaml").write_text(wrong)
        with pytest.raises(ValueError, match=key):
            embedding.load_config(tmp_path / "bad.yaml")


def test_the_study_configs_differ_only_where_the_plan_says():
    base = embedding.load_config(Path("configs/sl/embedding_direct.yaml"))
    finished = embedding.load_config(Path("configs/sl/embedding.yaml"))
    assert {k: v for k, v in base.items() if k != "line_term"} == {
        k: v for k, v in finished.items() if k != "line_term"
    }
    assert finished["line_term"] is None
    assert base["line_term"] == {
        "reference_lines": 8, "max_epochs": 30, "patience": 5, "learning_rate": 1.0e-3,
        "own_expression": True, "gene_in_line": {}, "joint": False, "input": "model",
        "cross_line": None, "pair_in_line": False,
    }  # fmt: skip
    changes = {
        # four other-line controls: the reference of the cross-line study (§20)
        "embedding_direct_mean_token": {("cell_embeddings",): None,
                                        ("eval", "other_lines"): 4},
        "embedding_direct_pie_change": {("line_term", "gene_in_line"): {
            "pie_change": "data/sl_gene_in_line/v1/pie_change"}},
        "embedding_direct_tx1_gene": {("line_term", "gene_in_line"): {
            "tx1_gene": "data/sl_gene_in_line/v1/tx1_gene"}},
    }  # fmt: skip
    for name, change in changes.items():
        expected = copy.deepcopy(base)
        for keys, value in change.items():
            target = expected
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
        assert embedding.load_config(Path(f"configs/sl/{name}.yaml")) == expected


def test_the_harness_names_every_table_of_the_study():
    config = benchmark_baselines.load_config(
        Path("configs/sl/benchmark_line_term.yaml")
    )
    assert set(config["extra_scores"]) <= set(benchmark_baselines.NAMES)
    assert len([k for k in config["extra_scores"] if k.startswith("direct")]) == 20


def test_the_cross_line_configs_change_one_thing_for_each_row():
    """SL protocol §20: each config is the row above with one change."""
    cache = "data/sl_gene_in_line/v1/"
    wide = {n: cache + n for n in ("wide_expression", "cell_spread", "paralog_buffer",
                                   "complex_buffer", "partner_low")}  # fmt: skip
    dna = {n: cache + n for n in ("depmap_mutation", "depmap_copy_number")}
    rows = [
        ("embedding_direct_mean_token", {}),
        ("cross_line_frozen", {("line_term", "cross_line"): {
            "weight": 0.5, "examples_per_step": 256}}),
        ("cross_line_joint", {("line_term", "joint"): True}),
        ("cross_line_genes_input", {("line_term", "input"): "genes"}),
        ("cross_line_wide", {("line_term", "own_expression"): False,
                             ("line_term", "gene_in_line"): wide}),
        ("cross_line_depmap_dna", {("line_term", "gene_in_line"): wide | dna}),
    ]  # fmt: skip
    expected = None
    for name, change in rows:
        config = embedding.load_config(Path(f"configs/sl/{name}.yaml"))
        if expected is not None:
            for (section, key), value in change.items():
                expected[section][key] = value
            assert config == expected, name
        expected = copy.deepcopy(config)


def test_the_harness_names_every_table_of_the_cross_line_study():
    config = benchmark_baselines.load_config(
        Path("configs/sl/benchmark_cross_line.yaml")
    )
    named = set(config["extra_scores"]) | set(config["scorer_means"])
    assert named <= set(benchmark_baselines.NAMES)
    # the harness's own scorers need no table
    own = {"labels", "random", "stage_one", "pan_essentiality", "measured_geneeffect"}
    assert set(config["cross_line_ordering"]) <= set(config["extra_scores"]) | own
    assert {n for pair in config["paired_differences"] for n in pair} <= named


def test_stage_one_fits_no_sl_line_and_no_line_of_an_sl_lines_patient():
    from src.data.splits import FixedSplit

    split = FixedSplit(
        train=("T1", "T2", "T3", "S1", "U1"), val=("V1",), test=("X1",),
        unlabeled_train=("U1",),
    )  # fmt: skip
    patient = pd.Series(
        {"T1": "p1", "T2": "p2", "T3": "p3", "S1": "p4", "U1": "p5", "V1": "p6",
         "X1": "p7", "B1": "p8", "B2": "p4", "B3": "p9", "B4": "p6", "S9": "p2"}
    )  # fmt: skip
    labelled = {"T1", "T2", "T3", "S1", "V1", "X1", "B1", "B2", "B4"}
    fit, stop, left_out = embedding.stage_one_lines(
        split,
        ["B1", "B2", "B3", "B4"],
        patient,
        sl_lines={"S1", "V1", "X1", "S9"},  # S9: an SL line outside the split
        sl_train={"S1"},
        with_geneeffect=labelled,
    )
    # T2 shares a patient with S9, B2 with S1, B4 with V1; B3 has no GeneEffect
    assert fit == ["T1", "T3", "B1"] and stop == ["S1"]
    assert left_out == {"sl_lines_left_out": 1, "patient_mates_left_out": 3}
    with pytest.raises(ValueError, match="nothing to stop on"):
        embedding.stage_one_lines(
            split, [], patient, {"S1"}, sl_train=set(), with_geneeffect=labelled
        )


def test_the_context_rows_change_one_thing_from_the_wide_row():
    """SL protocol §21: the wide row on the v2 caches, then each config is that row
    with one change."""
    old = embedding.load_config(Path("configs/sl/cross_line_wide.yaml"))
    wide = embedding.load_config(Path("configs/sl/cross_line_wide_v2.yaml"))
    caches = old["line_term"]["gene_in_line"]
    assert wide["line_term"]["gene_in_line"] == {
        n: p.replace("/v1/", "/v2/") for n, p in caches.items()
    }
    old["line_term"]["gene_in_line"] = wide["line_term"]["gene_in_line"]
    assert old == wide
    assert wide["space_measured_share"] is None and not wide["bulk_training_lines"]
    assert wide["line_term"]["pair_in_line"] is False
    for name, keys, value in (
        ("cross_line_80_lines", ("bulk_training_lines",), True),
        ("cross_line_wide_token", ("space_measured_share",), 0.5),
        ("cross_line_pair_in_line", ("line_term", "pair_in_line"), True),
    ):
        expected = copy.deepcopy(wide)
        target = expected
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        assert embedding.load_config(Path(f"configs/sl/{name}.yaml")) == expected, name


def test_the_harness_names_every_table_of_the_context_rows():
    config = benchmark_baselines.load_config(
        Path("configs/sl/benchmark_context_rows.yaml")
    )
    named = set(config["extra_scores"]) | set(config["scorer_means"])
    assert named <= set(benchmark_baselines.NAMES)
    own = {"labels", "random", "stage_one", "pan_essentiality", "measured_geneeffect"}
    assert set(config["cross_line_ordering"]) <= set(config["extra_scores"]) | own
    assert {n for pair in config["paired_differences"] for n in pair} <= named
    assert ["cross_wide_v2", "cross_wide"] in config["paired_differences"]
    for run in ("cross_80_lines", "cross_wide_token", "cross_pair_in_line"):
        assert [run, "cross_wide_v2"] in config["paired_differences"]
        assert [run, f"{run}_other_lines"] in config["paired_differences"]


def test_the_pair_tables_follow_the_reference():
    from types import SimpleNamespace

    reference = SimpleNamespace(
        paralogs=pd.DataFrame({"gene": ["A", "A"], "paralog": ["C", "B"],
                               "identity": [60.0, 40.0]}),
        complexes=pd.DataFrame({"complex_id": [1, 1, 2, 2],
                                "gene": ["A", "D", "B", "D"]}),
    )  # fmt: skip
    index = embedding.pair_index(["A", "B", "C", "D"], reference)
    assert index["paralogs"][0].tolist() == [2, 1]  # closest first
    assert (index["paralogs"][1:] == 4).all()  # padded with the row of "no gene"
    member = index["member"]
    assert member.shape == (5, 5) and (member == member.T).all()
    assert not member.diagonal().any() and not member[4].any()
    # B is a paralog of A in the table, so A is a partner of B; D joins A and B
    assert np.flatnonzero(member[1]).tolist() == [0, 3]
    assert index["partners"][1].tolist() == [0, 3, 4]
    assert index["partners"][0].tolist() == [1, 2, 3]


def test_the_pair_in_line_features_reach_the_line_term():
    """With the pair tables, the line head has the wider input and the line term of
    a pair changes when the line's expression of a paralog of one gene changes."""
    data, *_ = world()
    rng = np.random.default_rng(0)
    wide = np.concatenate([rng.normal(size=(16, 30, 2)), np.ones((16, 30, 1))], axis=2)
    wide = np.concatenate([wide, np.zeros((16, 1, 3))], axis=1)
    data.in_line["wide_expression"] = torch.tensor(wide, dtype=torch.float16)
    from types import SimpleNamespace

    reference = SimpleNamespace(
        paralogs=pd.DataFrame({"gene": ["G0"], "paralog": ["G2"], "identity": [50.0]}),
        complexes=pd.DataFrame({"complex_id": [1, 1], "gene": ["G0", "G1"]}),
    )
    data.pair_index = {
        k: torch.from_numpy(v)
        for k, v in embedding.pair_index(data.genes, reference).items()
    }
    config = LINE_CONFIG | {
        "line_term": LINE_CONFIG["line_term"] | {"pair_in_line": True}
    }
    torch.manual_seed(0)
    model = embedding.build_model(config, data)
    plain = embedding.build_model(
        LINE_CONFIG, dataclasses.replace(data, pair_index=None)
    )
    assert (
        model.line_head[0].in_features
        == plain.line_head[0].in_features + embedding.PAIR_WIDTH
    )
    torch.nn.init.normal_(model.mean_token[-1].weight, std=0.5)
    torch.nn.init.normal_(model.line_head[-1].weight, std=0.5)
    line = data.lines.index("train0")
    a, b, flag = torch.tensor([0, 3]), torch.tensor([1, 4]), torch.tensor([True, True])

    def term() -> torch.Tensor:
        with torch.no_grad():
            u, tokens = (
                model.gene_vectors(),
                embedding.one_line(model, data, line, None, None),
            )
            return embedding.line_term_of(
                model, data, u, model.line_vectors(), a, b, flag, line, tokens
            )

    before = term()
    at = int(data.in_line_at[line])
    data.in_line["wide_expression"][at, 2, 0] += 3.0  # G2, the other paralog of G0
    after = term()
    assert before[0] != after[0] and before[1] == after[1]
    pairs = embedding.pair_rows(data, config)
    scores = embedding.line_term_scores(model, data, pairs, np.arange(40), config)
    assert np.isfinite(scores).all()


def test_rows_seen_only_in_the_bulk_training_lines_are_counted():
    rows = pd.DataFrame(
        {"split": ["train", "train", "train", "val", "val", "test"],
         "model_id": ["T", "B", "B", "V", "V", "X"], "study_id": "S",
         "gene_a": ["A", "A", "C", "A", "C", "C"],
         "gene_b": ["B", "B", "D", "B", "D", "D"]}
    )  # fmt: skip
    # (A, B) is also in the training line T; (C, D) is in the bulk line B alone
    assert embedding.seen_only_in(rows, ["B"]) == {"val": 1, "test": 1}
    assert embedding.seen_only_in(rows, []) == {"val": 0, "test": 0}
