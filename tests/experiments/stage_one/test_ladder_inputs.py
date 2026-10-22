"""The stage-one inputs on a synthetic world: frame, cells, mean channel,
summaries and Tx1 caches."""

from __future__ import annotations

import copy
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.context_prior.bridge import fit_bridge
from src.experiments.stage_one import ladder_inputs as ladder
from tests.fixtures.ladder import (
    BULK_LABELLED,
    CELLS,
    HVG,
    PROGRAMS,
    TEST,
    TRAIN,
    UNLABELLED,
    VAL,
    build_world,
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("ladder")
    config = build_world(root)
    ladder.run(config, ["lines", "cells", "mean", "summaries"], (0, 1))
    return root, config


def test_frame_lines_genes_and_base(world):
    root, config = world
    lines = ladder.read_lines(root / "ladder")
    frame = [*TRAIN, *BULK_LABELLED[:5]]
    assert list(lines.index) == [*frame, *VAL, *TEST]
    assert lines.loc["ACH-T9", "role"] == "train_cells"  # lymphoid, has cells
    assert lines.loc["ACH-T8", "role"] == "train_cells"  # no bulk RNA
    assert "ACH-B5" not in lines.index and "ACH-U0" not in lines.index
    assert set(lines.loc[frame, "fold"]) == {0, 1}
    assert (lines.loc[[*VAL, *TEST], "fold"] == -1).all()
    assert lines.index[lines["paired"]].tolist() == list(TRAIN[:8])
    genes = ladder.read_genes(root / "ladder")
    assert genes.loc["C1ORF1", "gene_text_key"] == "C1orf1"
    assert not genes.loc["G7", "in_panel"]
    effect = pd.read_csv(root / "gene_effect.csv", index_col=0)
    expected = effect.loc[frame].mean(axis=0).to_numpy()
    assert np.allclose(genes["m_g"].to_numpy(), expected)
    assert (genes["m_g"] < 10).all()  # no validation or test label


def test_cells_read_every_cell_with_case_blind_symbols(world):
    root, _ = world
    record = ladder.read_cells(root / "ladder", "ACH-V0")
    union = json.loads((root / "ladder" / "cells" / "genes.json").read_text())["genes"]
    source = ad.read_h5ad(root / "basal_raw_umi" / "ACH-V0.h5ad")
    counts = source.X.toarray()
    library = counts.sum(axis=1)
    column = list(source.var["gene_symbol"]).index("C1orf1")
    cp10k = counts[:, column] * 1e4 / library
    at = union.index("C1ORF1")
    assert np.isclose(record["pie_input"][at], np.log1p(cp10k).mean(), atol=1e-5)
    assert np.isclose(record["pseudobulk"][at], np.log1p(cp10k.mean()), atol=1e-5)
    assert record["hvg"].shape == (CELLS, len(HVG))
    assert record["qc"].shape == (CELLS, len(ladder.QC_COLUMNS))
    assert np.isnan(record["qc"][:, -1]).all()  # too few cells for Scrublet
    with np.load(root / "prepared" / "noise" / "programs" / "ACH-V0.npz") as payload:
        assert np.allclose(record["qc"][:, 4:6], payload["scores"][:, :2])


def test_mean_channel_reads_cells_out_of_fold_and_bulk_for_bulk_only(world):
    root, _ = world
    out = root / "ladder" / "mean"
    channel = pd.read_parquet(out / "mean_channel.parquet")
    pseudo = pd.read_parquet(out / "pseudobulk.parquet")
    bulk = pd.read_parquet(out / "bulk_paired.parquet")
    lines = ladder.read_lines(root / "ladder")
    assert list(channel.index) == list(lines.index)
    assert float(bulk.to_numpy().max()) < 50  # validation bulk never read
    paired = list(lines.index[lines["paired"]])
    fold = lines.loc["ACH-T3", "fold"]
    outside = [m for m in paired if lines.loc[m, "fold"] != fold]
    manual = fit_bridge(
        pseudo.loc[outside].astype(float), bulk.loc[outside].astype(float)
    ).apply(pseudo.loc[["ACH-T3"]].astype(float))
    assert np.allclose(channel.loc[["ACH-T3"]], manual, atol=1e-5)
    held = fit_bridge(
        pseudo.loc[paired].astype(float), bulk.loc[paired].astype(float)
    ).apply(pseudo.loc[list(VAL)].astype(float))
    assert np.allclose(channel.loc[list(VAL)], held, atol=1e-5)
    assert channel.notna().all().all()


def test_pie_input_rescales_bulk_and_fills_unlisted_genes(world):
    root, _ = world
    out = root / "ladder" / "mean"
    values = pd.read_parquet(out / "pie_input.parquet")
    listed = pd.read_parquet(out / "pie_listed.parquet")
    genes = ladder.read_genes(root / "ladder")
    assert list(values.columns) == list(genes.index[genes["in_panel"]])
    bulk = pd.read_csv(root / "bulk.csv").set_index("ModelID")
    raw = bulk.loc["ACH-B0", "G0 (1)"]
    assert np.isclose(values.loc["ACH-B0", "G0"], np.log1p((2**raw - 1) / 100))
    assert listed.all().all()
    assert values.notna().all().all()


def test_summaries_cover_every_split_line(world):
    root, _ = world
    table = pd.read_parquet(root / "ladder" / "summaries.parquet")
    assert len(table) == len(TRAIN) + len(UNLABELLED) + len(VAL) + len(TEST)
    assert table.shape[1] == 2 * len(PROGRAMS) + 2 + 3
    for k in (2, 3):
        shares = table.loc[:, [c for c in table if c.startswith(f"clusters_{k}|")]]
        assert np.allclose(shares.sum(axis=1), 1.0)


def test_manifest_refuses_other_settings(world):
    root, config = world
    changed = copy.deepcopy(config)
    changed["pie"]["pilot_genes"] = 7
    with pytest.raises(ValueError, match="other settings"):
        ladder.open_manifest(ladder.load_setup(changed))
    manifest = json.loads((root / "ladder" / "manifest.json").read_text())
    assert set(manifest["steps"]) == {"lines", "cells", "mean", "summaries"}
    assert manifest["steps"]["lines"]["frame"] == 15


def test_tx1_embeds_real_and_count_shuffled_cells(world, monkeypatch):
    root, config = world
    import src.model.tx1 as tx1

    seen = []

    def fake_builder(model_dir, batch_size, max_length):
        def encode(adata):
            seen.append(adata.X.sum(axis=0))
            return np.zeros((adata.n_obs, 2560), dtype=np.float32)

        return encode, {}

    monkeypatch.setattr(tx1, "build_tx1_encoder", fake_builder)
    ladder.run(config, ["tx1"], (1, 2))
    setup = ladder.load_setup(config)
    lines = ladder.tx1_lines(ladder.load_frame(setup))[1::2]
    for kind in ("real", "shuffled"):
        assert sorted(p.name for p in (root / "tx1" / kind).iterdir()) == sorted(lines)
    real = np.load(root / "tx1" / "real" / lines[0] / "hvg.npy")
    moved = np.load(root / "tx1" / "shuffled" / lines[0] / "hvg.npy")
    assert not np.array_equal(real, moved)
    for j in range(real.shape[1]):
        assert np.array_equal(np.sort(real[:, j]), np.sort(moved[:, j]))
    ladder.run(config, ["tx1"], (1, 2))  # resumes: nothing left to encode
    assert len(seen) == 2 * len(lines)
