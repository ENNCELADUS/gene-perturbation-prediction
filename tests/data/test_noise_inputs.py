"""The noise diagnosis's per-line inputs: cell draws, program scores, and the
preparation steps that write them from raw cells."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.data.noise_inputs import cell_draws, program_scores
from src.data.pseudobulk import pseudobulk
from src.experiments.prepare import (
    prepare_inputs,
    prepare_noise_inputs,
    prepare_second_source,
)
from tests.fixtures.prepare import CELLS_PER_LINE, SOURCE_GENES, TRAIN, build_world


def test_halves_split_the_cells_and_draws_nest():
    cells = [f"c{i}" for i in range(9)]
    draws = cell_draws("ACH-1", cells, [2, 5, 9, 20])
    assert set(draws) == {"half_a", "half_b", "cells_2", "cells_5"}
    halves = np.concatenate([draws["half_a"], draws["half_b"]])
    assert sorted(halves) == list(range(9))
    assert len(draws["half_a"]) == 5 and len(draws["half_b"]) == 4
    assert list(draws["cells_5"][:2]) == list(draws["cells_2"])
    again = cell_draws("ACH-1", cells, [2, 5])
    assert all(np.array_equal(again[k], draws[k]) for k in again)
    other = cell_draws("ACH-2", cells, [5])
    assert not np.array_equal(other["cells_5"], draws["cells_5"])


def test_program_scores_subtract_the_cells_mean_over_the_scoring_set():
    counts = csr_matrix(np.array([[1.0, 3.0, 0.0, 6.0], [4.0, 0.0, 2.0, 2.0]]))
    symbols = ["a", "B", "C", "OFFSET"]  # symbols are matched upper-cased
    genes = ["A", "B", "C", "D"]  # the scoring set; this source does not list D
    # Q's member "e" lies outside the scoring set and is not part of Q here.
    members = {"P": ["A", "b"], "Q": ["C", "D", "e"]}
    scores, listed = program_scores(counts, symbols, genes, members, 10.0)
    library = np.array([10.0, 8.0])
    logged = np.log1p(counts.toarray()[:, :3] * (10.0 / library)[:, None])
    background = logged.sum(axis=1) / 4  # unlisted D counts as unexpressed
    assert np.allclose(scores[:, 0], logged[:, :2].mean(axis=1) - background)
    assert np.allclose(scores[:, 1], logged[:, 2] / 2 - background)
    assert listed == {"scoring_set": 0.75, "P": 1.0, "Q": 0.5}


def test_dropping_zero_count_genes_does_not_move_a_cells_scores():
    """Some builders drop each line's zero-count genes; identical cells must
    score the same whichever way their source was built."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.5, size=(5, 30)).astype(float)
    counts[:, :10] = 0.0  # never detected in this line
    symbols = [f"G{i}" for i in range(30)]
    members = {"P": symbols[5:15], "Q": symbols[20:25]}
    kept, _ = program_scores(csr_matrix(counts), symbols, symbols, members, 1e4)
    dropped, listed = program_scores(
        csr_matrix(counts[:, 10:]), symbols[10:], symbols, members, 1e4
    )
    assert np.allclose(kept, dropped)
    assert listed["scoring_set"] == pytest.approx(20 / 30)
    assert listed["P"] == 0.5 and listed["Q"] == 1.0


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("noise_world")
    config = build_world(root)
    prepare_inputs(config)
    return config


def test_noise_inputs_are_written_once_from_raw_cells(world):
    genes = ["GA", "GB", "GC", "GD", "H1", "NOT_MEASURED"]
    programs = {"P": ["GA", "GB"], "Q": ["GC", "GD"]}
    path = prepare_noise_inputs(
        world, genes, cells=[3, CELLS_PER_LINE], programs=programs
    )
    root = path.parent
    manifest = json.loads(path.read_text())
    frame = pd.read_parquet(root / "subsamples.parquet")
    line = TRAIN[0]
    source = ad.read_h5ad(Path(world["paths"]["basal_raw_umi"]) / f"{line}.h5ad")
    draws = cell_draws(line, source.obs_names.astype(str), [3, CELLS_PER_LINE])
    assert set(frame.loc[line].index) == {"half_a", "half_b", "cells_3"}
    for name, rows in draws.items():
        expected = pseudobulk(source.X[rows], list(SOURCE_GENES), genes)
        assert np.allclose(frame.loc[(line, name)].to_numpy(), expected, equal_nan=True)
    with np.load(root / "programs" / f"{line}.npz") as payload:
        assert payload["scores"].shape == (CELLS_PER_LINE, 2)
    assert manifest["listed"][line] == {"scoring_set": 5 / 6, "P": 1.0, "Q": 1.0}
    assert manifest["members_in_genes"] == {"P": 2, "Q": 2}
    assert set(manifest["lines"]) == set(frame.index.get_level_values("model_id"))
    # The same settings reuse the root; other settings are refused.
    assert (
        prepare_noise_inputs(world, genes, cells=[CELLS_PER_LINE, 3], programs=programs)
        == path
    )
    with pytest.raises(ValueError, match="other settings"):
        prepare_noise_inputs(world, genes, cells=[4], programs=programs)


def _second(directory: Path, model_id: str, seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, size=(6, len(SOURCE_GENES))).astype(np.float32)
    adata = ad.AnnData(
        X=csr_matrix(counts),
        obs=pd.DataFrame({"model_id": model_id}, index=[f"s{i}" for i in range(6)]),
        var=pd.DataFrame(
            {
                "ensembl_id": [f"ENSG{i:011d}" for i in range(len(SOURCE_GENES))],
                "gene_symbol": list(SOURCE_GENES),
            },
            index=[f"v{i}" for i in range(len(SOURCE_GENES))],
        ),
    )
    directory.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(directory / f"{model_id}.h5ad")
    return adata


def test_second_deposit_pseudobulk_grows_by_new_files(world, tmp_path):
    genes = ["GA", "GB", "H1"]
    second = tmp_path / "second"
    first = _second(second, TRAIN[4], seed=1)
    path = prepare_second_source(world, genes, second)
    frame = pd.read_parquet(path.parent / "pseudobulk.parquet")
    assert list(frame.index) == [TRAIN[4]]
    assert np.allclose(frame.iloc[0], pseudobulk(first.X, list(SOURCE_GENES), genes))
    _second(second, TRAIN[5], seed=2)
    prepare_second_source(world, genes, second)
    grown = pd.read_parquet(path.parent / "pseudobulk.parquet")
    assert list(grown.index) == [TRAIN[4], TRAIN[5]]
    pd.testing.assert_frame_equal(grown.loc[[TRAIN[4]]], frame)
    with pytest.raises(ValueError, match="another second-deposit"):
        prepare_second_source(world, genes[:2], second)
