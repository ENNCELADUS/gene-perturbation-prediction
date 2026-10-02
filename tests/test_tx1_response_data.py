"""Response sources, log-space response targets and the response cache."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.data.expression import log_normalize
from src.data.response import (
    PerturbseqSource,
    XatlasOrionSource,
    control_library_sizes,
    load_response_sources,
    read_response_part,
    response_bags,
    response_cells,
    response_part,
    write_response_part,
)
from src.data.response_cache import open_response_targets, write_response_targets

LABELS = ["non-targeting"] * 3 + ["kif11", "KIF11", "TP53", "TP53"]
SYMBOLS = ["H1", "H2", "OTHER"]


def _h5ad(path: Path, *, scale: float = 1.0) -> PerturbseqSource:
    counts = np.arange(1, len(LABELS) * 3 + 1, dtype=np.float32).reshape(-1, 3) * scale
    obs = pd.DataFrame({"gene": LABELS}, index=[f"c{i}" for i in range(len(LABELS))])
    var = pd.DataFrame(
        {"gene_id": [f"ENSG{i:011d}" for i in range(3)], "gene_name": SYMBOLS},
        index=["a", "b", "c"],
    )
    ad.AnnData(X=csr_matrix(counts), obs=obs, var=var).write_h5ad(path)
    return PerturbseqSource(path, "non-targeting", "gene", "gene_id")


def test_sources_parse_both_types(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "ACH-1": {
                    "h5ad_path": "a.h5ad",
                    "perturbation_col": "gene",
                    "control_label": "non-targeting",
                    "var_ensembl_col": "gene_id",
                },
                "ACH-2": {
                    "source_type": "xatlas_orion_parquet",
                    "shard_dir": "shards",
                    "gene_metadata_path": "meta.parquet",
                    "control_label": "Non-Targeting",
                    "shard_glob": "HCT*.parquet",
                },
            }
        )
    )
    sources = load_response_sources(path)
    assert sources["ACH-1"] == PerturbseqSource(
        Path("a.h5ad"), "non-targeting", "gene", "gene_id", "gene_name"
    )
    assert sources["ACH-2"] == XatlasOrionSource(
        Path("shards"), Path("meta.parquet"), "Non-Targeting", "HCT*.parquet", 1
    )


def test_control_library_sizes_cover_every_gene(tmp_path):
    source = _h5ad(tmp_path / "a.h5ad")
    np.testing.assert_array_equal(control_library_sizes(source, "ACH-1"), [6, 15, 24])


def _targets(source, model_id, hvg_order, target_sum):
    """``(keys, bags)`` of one anchor, read in this process."""
    part = response_part(
        response_cells(
            source,
            model_id,
            hvg_order,
            max_cells_per_gene=None,
            total_cells_per_line=None,
            seed=0,
        )
    )
    keys = [(model_id, gene) for gene in part.genes]
    return keys, [part.bag(i, target_sum) for i in range(len(keys))]


def test_targets_are_log_space_over_whole_library(tmp_path):
    counts = np.arange(1, len(LABELS) * 3 + 1, dtype=np.float32).reshape(-1, 3)
    keys, bags = _targets(
        _h5ad(tmp_path / "a.h5ad"), "ACH-1", ("H2", "MISSING", "H1"), 10.0
    )
    assert keys == [("ACH-1", "KIF11"), ("ACH-1", "TP53")]
    for (_, gene), bag in zip(keys, bags, strict=True):
        rows = [i for i, label in enumerate(LABELS) if label.upper() == gene]
        expected = np.zeros((len(rows), 3), dtype=np.float32)
        expected[:, [0, 2]] = log_normalize(
            counts[rows][:, [1, 0]], counts[rows].sum(axis=1), 10.0
        )
        # Labels differing only in case merge; row order within a bag is free.
        np.testing.assert_allclose(
            np.sort(bag, axis=0), np.sort(expected, axis=0), rtol=1e-6
        )


def test_targets_reject_normalised_source(tmp_path):
    with pytest.raises(ValueError, match="integer"):
        _targets(_h5ad(tmp_path / "a.h5ad", scale=0.1), "ACH-1", ("H1",), 10.0)


def test_xatlas_targets(tmp_path):
    pd.DataFrame(
        {
            "ensembl_id": ["ENSG00000000000", "ENSG00000000001"],
            "gene_name": ["H1", "OTHER"],
            "gene_token_id": [0, 1],
        }
    ).to_parquet(tmp_path / "meta.parquet")
    shards = tmp_path / "shards"
    shards.mkdir()
    pd.DataFrame(
        [
            {
                "gene_token_id": np.array([0, 1]),
                "gene_expression": np.array([float(i + 1), 3.0]),
                "cell_barcode": f"b{i}",
                "sample": "s",
                "gene_target": target,
                "pass_guide_filter": 1,
            }
            for i, target in enumerate(["Non-Targeting", "G1", "G1"])
        ]
    ).to_parquet(shards / "HCT116_Batch1.parquet")
    source = XatlasOrionSource(shards, tmp_path / "meta.parquet", "Non-Targeting")
    keys, bags = _targets(source, "ACH-000971", ("H1",), 5.0)
    assert keys == [("ACH-000971", "G1")]
    np.testing.assert_allclose(
        bags[0][:, 0], np.log1p(np.array([2.0, 3.0]) * 5.0 / np.array([5.0, 6.0]))
    )


def test_part_file_round_trip_gives_the_same_bags(tmp_path):
    source = _h5ad(tmp_path / "a.h5ad")
    hvg_order = ("H2", "MISSING", "H1")
    keys, bags = _targets(source, "ACH-1", hvg_order, 10.0)
    path = write_response_part(
        tmp_path / "part.npz",
        source,
        "ACH-1",
        hvg_order,
        max_cells_per_gene=None,
        total_cells_per_line=None,
        seed=0,
    )
    assert read_response_part(path).genes == ("KIF11", "TP53")
    for streamed, bag in zip(
        response_bags({"ACH-1": path}, keys, 10.0), bags, strict=True
    ):
        assert streamed.dtype == np.float32
        np.testing.assert_array_equal(streamed, bag)


def test_response_cache_roundtrip_and_overwrite(tmp_path):
    keys = [("ACH-1", "A"), ("ACH-2", "B")]
    bags = [np.ones((2, 3), dtype=np.float32), np.full((1, 3), 2.0, dtype=np.float32)]
    write_response_targets(tmp_path / "response", keys, [2, 1], 3, iter(bags))
    cache = open_response_targets(tmp_path / "response")
    assert cache.keys == tuple(keys)
    np.testing.assert_array_equal(cache.target_bag(1), bags[1])
    write_response_targets(tmp_path / "response", keys[:1], [2], 3, bags[:1])
    assert open_response_targets(tmp_path / "response").keys == (keys[0],)
    assert [p.name for p in tmp_path.iterdir()] == ["response"]


def test_response_cache_rejects_a_bag_of_the_wrong_length(tmp_path):
    bags = [np.ones((2, 3), dtype=np.float32)]
    with pytest.raises(ValueError, match="bag shape"):
        write_response_targets(tmp_path / "response", [("ACH-1", "A")], [3], 3, bags)
    assert list(tmp_path.iterdir()) == []
