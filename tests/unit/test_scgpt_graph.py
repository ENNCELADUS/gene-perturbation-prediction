from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from src.scgpt.graph import (
    GeneGraphConfig,
    build_coexpression_gene_graph,
    load_gene_graph,
    save_gene_graph_edges,
)


def test_load_gene_graph_maps_edge_list_gene_names(tmp_path: Path) -> None:
    edge_path = tmp_path / "edges.csv"
    edge_path.write_text("source,target,weight\nA,B,0.7\nB,C,0.2\nC,missing,0.9\n")

    graph = load_gene_graph(
        GeneGraphConfig(
            enabled=True,
            source="edge_list",
            path=edge_path,
            directed=False,
        ),
        gene_names=["A", "B", "C"],
    )

    assert graph.edge_index.tolist() == [
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ]
    assert torch.allclose(
        graph.edge_weight,
        torch.tensor([0.7, 0.7, 0.2, 0.2], dtype=torch.float32),
    )
    assert graph.diagnostics["matched_edges"] == 2
    assert graph.diagnostics["dropped_edges"] == 1


def test_coexpression_graph_uses_only_train_conditions_and_controls() -> None:
    adata = ad.AnnData(
        X=np.asarray(
            [
                [1.0, 0.0, 1.0],
                [2.0, 0.1, 1.2],
                [10.0, 9.0, 0.0],
                [11.0, 9.1, 0.2],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(
            {
                "condition": ["ctrl", "A", "B", "B"],
                "control": [1, 0, 0, 0],
            }
        ),
        var=pd.DataFrame({"gene_name": ["A", "B", "C"]}),
    )

    graph = build_coexpression_gene_graph(
        adata=adata,
        train_conditions=["A"],
        gene_names=["A", "B", "C"],
        condition_key="condition",
        control_key="control",
        top_k=1,
        directed=False,
    )

    assert graph.diagnostics["source"] == "coexpression"
    assert graph.diagnostics["n_cells"] == 2
    assert graph.diagnostics["top_k"] == 1
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] > 0


def test_save_and_load_coexpression_graph_round_trips(tmp_path: Path) -> None:
    graph = build_coexpression_gene_graph(
        adata=ad.AnnData(
            X=np.asarray(
                [
                    [1.0, 2.0, 3.0],
                    [1.2, 2.2, 2.8],
                    [3.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            obs=pd.DataFrame(
                {
                    "condition": ["ctrl", "A", "A"],
                    "control": [1, 0, 0],
                }
            ),
            var=pd.DataFrame({"gene_name": ["A", "B", "C"]}),
        ),
        train_conditions=["A"],
        gene_names=["A", "B", "C"],
        condition_key="condition",
        control_key="control",
        top_k=1,
        directed=False,
    )
    graph_path = tmp_path / "coexpression.csv"

    save_gene_graph_edges(graph, graph_path, gene_names=["A", "B", "C"])
    loaded = load_gene_graph(
        GeneGraphConfig(enabled=True, source="coexpression", path=graph_path),
        gene_names=["A", "B", "C"],
    )

    assert loaded.edge_index.shape == graph.edge_index.shape
    assert torch.allclose(loaded.edge_weight, graph.edge_weight)
