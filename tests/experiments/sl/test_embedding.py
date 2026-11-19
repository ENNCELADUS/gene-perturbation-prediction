"""The gene-knowledge, cell-set model's experiment on a synthetic world: GeneEffect is
gene vector times line state, a pair is SL where both genes are dependencies in the
line; stage one lowers the loss on the SL training lines, stage two ranks held-out
lines' pairs above random, and the inputs' readers fail closed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments.sl import embedding
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
    "eval": {"rows_per_chunk": 64, "genes_per_chunk": 16},
}  # fmt: skip
SIDES = {"train": 8, "val": 4, "test": 4}


def world(seed: int = 0) -> tuple[embedding.Data, np.ndarray]:
    """40 fit lines with cells, 40 bulk-only fit lines and 16 SL lines; 30 genes."""
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
    targets = np.full(z.shape, np.nan)
    labelled = [i for i, m in enumerate(lines) if not m.startswith(("val", "test"))]
    targets[labelled] = z[labelled] + 0.1 * rng.normal(size=(len(labelled), 30))
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
    data = embedding.Data(
        genes=genes,
        lines=lines,
        expression=torch.tensor(expression, dtype=torch.float32),
        cells=torch.tensor(cells, dtype=torch.float16),
        offsets=torch.tensor(np.concatenate([[0], np.cumsum(counts)])),
        targets=torch.tensor(targets, dtype=torch.float32),
        fit=torch.tensor([i for i, m in enumerate(lines) if m[0] in "CB"]),
        stop=torch.tensor([i for i, m in enumerate(lines) if m.startswith("train")]),
        rows=pd.concat(records, ignore_index=True),
        sources={"knowledge": (torch.tensor(knowledge, dtype=torch.float32),
                               torch.tensor(present))},
        m_g=np.zeros(30),
        sigma_g=np.ones(30),
    )  # fmt: skip
    return data, z


def test_the_model_learns_geneeffect_and_ranks_held_out_pairs():
    data, z = world()
    torch.manual_seed(0)
    model = embedding.build_model(CONFIG, data)
    result = embedding.train(model, data, CONFIG)
    history = result["stage_one"]
    assert history[0]["epoch"] == 0 and history[0]["train_loss"] is None
    assert min(r["stop_loss"] for r in history) < 0.5 * history[0]["stop_loss"]
    phases = result["phases"]
    assert [p["phase"] for p in phases] == ["frozen", "trained further"]
    assert result["chosen"] == int(np.argmax([p["val_recall_top10"] for p in phases]))
    assert np.isfinite(result["scores"]).all()
    metrics = embedding.sl_metrics(data.rows, result["scores"], 20)
    assert {s: metrics[s]["lines"] for s in SIDES} == SIDES
    # a random score has a recall in the top 10% near 0.1 and an AUROC near 0.5
    assert metrics["val"]["recall_top10"] > 0.3
    assert metrics["test"]["recall_top10"] > 0.3 and metrics["test"]["auroc"] > 0.7
    # the held-out lines' GeneEffect, never a target, is predicted from their inputs
    held = [m for m in data.lines if m.startswith(("val", "test"))]
    table = embedding.geneeffect_table(model, data, held, CONFIG)
    truth = z[[data.lines.index(m) for m in held]]
    assert np.corrcoef(table.to_numpy().ravel(), truth.ravel())[0, 1] > 0.5
    report = embedding.results_markdown(
        {"run_id": "synthetic", "sl": metrics, "phases": phases,
         "chosen_phase": result["chosen"], "stage_one_history": history,
         "stage_one": {s: {"lines": n, "line_spearman": 0.1} for s, n in SIDES.items()},
         "stage_one_base": {s: {"lines": n} for s, n in SIDES.items()},
         "counts": {"fit_lines": 80, "fit_lines_with_cells": 40, "stop_lines": 8,
                    "sl_lines_left_out": 8, "patient_mates_left_out": 0, "cells": 1,
                    "genes": 30, "gene_source_coverage": {"knowledge": 29},
                    "training_rows": 1}}
    )  # fmt: skip
    assert "**Table 1.**" in report and "trained further" in report


def test_training_reads_only_training_rows_of_two_class_genetic_screens():
    data, _ = world()
    rows = data.rows.copy()
    one_class = rows["model_id"] == "train0"
    rows.loc[one_class, "label"] = 0
    treated = rows["model_id"] == "train1"
    rows.loc[treated, "condition"] = "drug"
    keep = embedding.training_rows(rows)
    assert set(rows.loc[keep, "split"]) == {"train"}
    assert not keep[one_class.to_numpy()].any() and not keep[treated.to_numpy()].any()
    data.rows = rows
    draws = embedding.pair_rows(data).draws
    assert sorted(draws) == sorted(data.lines.index(f"train{i}") for i in range(2, 8))
    totals = [float(rates.sum()) for _, rates in draws.values()]
    assert np.allclose(totals, totals[0])  # every line weighs the same


def test_cell_batch_caps_drops_and_pads():
    data, _ = world()
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
    text = Path("configs/sl/embedding.yaml").read_text().replace("top_k: 20\n", "")
    (tmp_path / "bad.yaml").write_text(text)
    with pytest.raises(ValueError, match="top_k"):
        embedding.load_config(tmp_path / "bad.yaml")


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
