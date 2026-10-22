"""A prepared root on disk (``prepared_inputs.json`` and the per-line caches) for the
joint world's lines, as preparation writes it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.prepared import PreparedLine, write_prepared_line
from src.data.q_sc import QScFeatures
from src.data.response import write_response_targets

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
            "prior": None,
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
        # Every Tx1 dimension of a fixture cell is equal: the contexts have rank one.
        "model": {"context_components": 1},
    }
