"""Opening a prepared root: train-only fitting, restore, and test exposure.

``make_prepared_fixture`` writes a small prepared root in the current layout
directly (no raw sources); other tests reuse it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.prepared import PreparedLine, load_inputs, write_prepared_line
from src.data.q_sc import QScFeatures
from src.data.response_cache import write_response_targets

ANCHORS = [f"ACH-A{i}" for i in range(4)]
TARGET_SUM = 1000.0


def make_prepared_fixture(root: Path, *, hvg_width: int = 2) -> dict:
    """Prepared root with four unequal anchor pools; returns a config."""
    if hvg_width < 2:
        raise ValueError("fixture hvg_width must be at least 2")
    root.mkdir(parents=True, exist_ok=True)
    prepared = root / "prepared"
    split = {
        "train": ANCHORS + ["ACH-TRAIN", "ACH-UNLABELED"],
        "val": ["ACH-VAL"],
        "test": ["ACH-TEST"],
        "unlabeled_train": ["ACH-UNLABELED"],
    }
    panel = ["G1", "G0", "G2"]
    hvg = ["G0", "G1", *[f"HVG{i}" for i in range(2, hvg_width)]]
    esm_order = ["G2", "G0", "G1", "R0", "R1", "R2", "R3"]
    (root / "split.json").write_text(json.dumps(split))
    ids = ANCHORS + ["ACH-TRAIN", "ACH-VAL", "ACH-TEST"]
    pd.DataFrame(
        {
            "G0 (1)": np.arange(7),
            "G1 (2)": np.arange(7) ** 2,
            "G2 (3)": [1, 0, 1, 0, 1, 9, 8],
        },
        index=ids,
    ).to_csv(root / "labels.csv")
    np.savez(
        root / "esm2.npz",
        symbols=np.array(esm_order),
        vectors=np.arange(len(esm_order) * 3, dtype=np.float32).reshape(-1, 3),
        resolved=np.ones(len(esm_order), dtype=bool),
    )
    for line, model_id in enumerate(ids):
        rows = np.arange(5, dtype=np.float32) + line * 10
        write_prepared_line(
            prepared / "lines" / f"{model_id}.npz",
            PreparedLine(
                controls_tx1=np.repeat(rows[:, None], 2560, axis=1),
                basal_hvg=np.log1p(np.repeat(rows[:, None], hvg_width, axis=1)),
                q_sc=QScFeatures(
                    symbols=tuple(panel),
                    values=np.array(
                        [[np.nan] * 3, [line + 1, 0.5, 1], [2, 0.2, 3]],
                        dtype=np.float32,
                    ),
                    available=np.array([False, True, True]),
                ),
            ),
        )
    keys = [
        (anchor, gene)
        for i, anchor in enumerate(ANCHORS)
        for gene in ["G0", "G1", *[f"R{j}" for j in range(i + 1)]]
    ]
    cells = np.log1p(np.arange(len(keys) * 2 * hvg_width, dtype=np.float32))
    cells = cells.reshape(len(keys), 2, hvg_width)
    write_response_targets(
        prepared / "response", keys, [2] * len(keys), hvg_width, list(cells)
    )
    (prepared / "prepared_inputs.json").write_text(
        json.dumps(
            {
                "expression_space": {
                    "transform": "log1p_normalize_total",
                    "target_sum": TARGET_SUM,
                    "library_size": "all_genes",
                    "target_sum_sources": ["ACH-000995", "ACH-000739"],
                },
                "common_gene_panel": panel,
                "hvg_order": hvg,
                "response_anchors": ANCHORS,
            }
        )
    )
    return {
        "prepared_root": str(prepared),
        "paths": {
            "split": str(root / "split.json"),
            "gene_effect": str(root / "labels.csv"),
            "esm2_embeddings": str(root / "esm2.npz"),
        },
        "features": {
            "hvg_dim": hvg_width,
            "esm2_dim": 3,
            "cells_per_context": 5,
            "variable_gene_min_observations": 5,
            "variable_gene_percentile": 75,
            "selective_min_lines": 5,
            "selective_max_fraction": 0.9,
            "residual_sd_floor_percentile": 10,
        },
        "train": {"dependency_batch_size": 2, "response_batch_size": 8},
        "seeds": {"train": 0, "collator": 0, "projection": 0},
    }


def test_fixture_opens_in_log_space(tmp_path):
    inputs = load_inputs(make_prepared_fixture(tmp_path, hvg_width=3))
    assert inputs.target_sum == TARGET_SUM
    assert inputs.genes == ("G1", "G0", "G2")
    assert inputs.hvg_order == ("G0", "G1", "HVG2")
    assert inputs.response_anchors == tuple(ANCHORS)
    assert len(inputs.response_targets.keys) == 2 * 4 + 10
    assert inputs.response_targets.target_bag(0).shape == (2, 3)
    line = inputs.lines["ACH-A0"]
    assert line.controls_tx1.shape == (5, 2560)
    assert line.basal_hvg.shape == (5, 3)
    assert np.isnan(line.q_sc.values[0]).all() and not line.q_sc.available[0]
    assert inputs.esm2_symbols == ("G2", "G0", "G1", "R0", "R1", "R2", "R3")


def test_train_only_fit_ignores_val_and_test_labels(tmp_path):
    config = make_prepared_fixture(tmp_path)
    before = load_inputs(config)
    assert before.train_gene_means["G0"] == 2
    assert set(before.labels.model_id) == {*before.split.supervised_train, "ACH-VAL"}
    wide = pd.read_csv(config["paths"]["gene_effect"], index_col=0)
    wide.loc[["ACH-VAL", "ACH-TEST"]] = 9999
    wide.to_csv(config["paths"]["gene_effect"])
    after = load_inputs(config)
    pd.testing.assert_series_equal(before.train_gene_means, after.train_gene_means)
    assert before.variable_genes == after.variable_genes


def test_test_lines_and_labels_need_include_test(tmp_path):
    config = make_prepared_fixture(tmp_path)
    assert "ACH-TEST" not in load_inputs(config).lines
    opened = load_inputs(config, include_test=True)
    assert "ACH-TEST" in opened.lines
    assert "ACH-TEST" in set(opened.labels.model_id)
    assert "ACH-UNLABELED" not in opened.lines


def test_restore_skips_fitting_and_external_esm2(tmp_path, monkeypatch):
    import src.data.prepared as prepared

    config = make_prepared_fixture(tmp_path)
    fitted = load_inputs(config)
    state = fitted.preprocessing_state()
    Path(config["paths"]["esm2_embeddings"]).unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("restore must not refit")

    monkeypatch.setattr(prepared, "fit_gene_means", forbidden)
    monkeypatch.setattr(prepared, "fit_variable_gene_membership", forbidden)
    restored = load_inputs(config, preprocessing=state)
    pd.testing.assert_series_equal(restored.train_gene_means, fitted.train_gene_means)
    np.testing.assert_array_equal(restored.esm2_vectors, fitted.esm2_vectors)
    state["target_sum"] = TARGET_SUM * 2
    with pytest.raises(ValueError, match="target_sum"):
        load_inputs(config, preprocessing=state)


def test_dependency_batch_carries_gene_index_scale_and_selectivity(tmp_path):
    import dataclasses

    import torch

    from src.data.datasets import DependencyDataset

    opened = load_inputs(make_prepared_fixture(tmp_path))
    scale = pd.Series([0.5, 2.0, 4.0], index=list(opened.genes))
    inputs = dataclasses.replace(
        opened, residual_scale=scale, selective_genes=frozenset({"G0"})
    )
    dataset = DependencyDataset(inputs, "train")
    batch = dataset.collate(range(len(dataset)))
    genes = batch.conditions.genes
    gene_index = batch.conditions.gene_index
    assert gene_index.dtype == torch.long
    assert [inputs.genes[i] for i in gene_index.tolist()] == list(genes)
    assert batch.residual_scale.dtype == torch.float32
    assert batch.residual_scale.tolist() == [scale[gene] for gene in genes]
    assert batch.selective.dtype == torch.bool
    assert batch.selective.tolist() == [gene == "G0" for gene in genes]
    moved = batch.to("meta")
    for value in (
        moved.conditions.gene_index,
        moved.residual_scale,
        moved.selective,
        moved.residual,
    ):
        assert value.device.type == "meta"


def test_rows_by_gene_partitions_rows_in_gene_order(tmp_path):
    import dataclasses

    from src.data.datasets import DependencyDataset

    opened = load_inputs(make_prepared_fixture(tmp_path))
    assert opened.genes == ("G1", "G0", "G2")
    inputs = dataclasses.replace(
        opened, labels=opened.labels.loc[opened.labels.gene_symbol != "G0"]
    )
    dataset = DependencyDataset(inputs, "train")
    groups = dataset.rows_by_gene()
    assert len(groups) == len(inputs.genes)
    assert len(groups[1]) == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate(groups)), np.arange(len(dataset))
    )
    for gene, rows in zip(inputs.genes, groups, strict=True):
        assert {dataset.genes[i] for i in rows} <= {gene}
        assert len(rows) == sum(g == gene for g in dataset.genes)
