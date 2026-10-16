"""The stage-one inputs on a synthetic world: frame, cells, mean channel,
summaries and Tx1 caches."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.sparse import csr_matrix

from src.context_prior.bridge import fit_bridge
from src.data.splits import SPLIT_SCHEMA_VERSION
from src.experiments import ladder_inputs as ladder
from src.experiments.config import validate_ladder_config

TRAIN = tuple(f"ACH-T{i}" for i in range(10))  # T8 has no bulk; T9 is lymphoid
UNLABELLED = ("ACH-U0",)
VAL = ("ACH-V0", "ACH-V1", "ACH-V2")
TEST = ("ACH-S0", "ACH-S1", "ACH-S2")
BULK_LABELLED = tuple(f"ACH-B{i}" for i in range(6))  # B5 is myeloid
BULK_UNLABELLED = ("ACH-C0", "ACH-C1")
EFFECT_GENES = (*(f"G{i}" for i in range(8)), "C1ORF1")
BULK_GENES = (*EFFECT_GENES, "B1", "B2", "B3")
HVG = ("H1", "H2", "H3", "H4")
SOURCE = (*(f"G{i}" for i in range(8)), "C1orf1", "B1", "B2", "B3", *HVG, "MT-CO1")
PROGRAMS = ("S", "G2M", "P1")
CELLS = 30
TARGET_SUM = 5000.0


def _lineage(model_id: str) -> str:
    return {"ACH-T9": "Lymphoid", "ACH-B5": "Myeloid"}.get(model_id, "Lung")


def build_world(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    split_lines = (*TRAIN, *UNLABELLED, *VAL, *TEST)
    (root / "split.json").write_text(
        json.dumps(
            {
                "schema_version": SPLIT_SCHEMA_VERSION,
                "train": [*TRAIN, *UNLABELLED],
                "val": list(VAL),
                "test": list(TEST),
                "unlabeled_train": list(UNLABELLED),
                "bulk_labelled": list(BULK_LABELLED),
                "bulk_unlabelled": list(BULK_UNLABELLED),
                "excluded": {},
            }
        )
    )
    every = (*split_lines, *BULK_LABELLED, *BULK_UNLABELLED)
    pd.DataFrame(
        {
            "ModelID": every,
            "PatientID": [f"P-{m}" for m in every],
            "OncotreeLineage": [_lineage(m) for m in every],
            "RRID": "",
        }
    ).to_csv(root / "Model.csv", index=False)

    labelled = (*TRAIN, *VAL, *TEST, *BULK_LABELLED)
    effects = rng.normal(-0.4, 0.6, size=(len(labelled), len(EFFECT_GENES)))
    effects[len(TRAIN) : len(TRAIN) + len(VAL) + len(TEST)] += 100.0  # leak marker
    pd.DataFrame(
        effects,
        index=pd.Index(labelled),
        columns=[f"{g} ({i + 1})" for i, g in enumerate(EFFECT_GENES)],
    ).to_csv(root / "gene_effect.csv")

    with_bulk = [m for m in every if m not in ("ACH-T8", *TEST)]
    bulk = rng.normal(4.0, 1.5, size=(len(with_bulk), len(BULK_GENES))).clip(0)
    for row, model_id in enumerate(with_bulk):
        if model_id in VAL:
            bulk[row] += 100.0  # validation bulk RNA must never be read
    frame = pd.DataFrame(
        bulk, columns=[f"{g} ({i + 1})" for i, g in enumerate(BULK_GENES)]
    )
    frame.insert(0, "ModelID", with_bulk)
    frame.insert(1, "IsDefaultEntryForModel", "Yes")
    frame.insert(2, "SequencingID", [f"CDS-{i}" for i in range(len(with_bulk))])
    frame.to_csv(root / "bulk.csv", index=False)

    (root / "basal_raw_umi").mkdir()
    programs = root / "prepared" / "noise" / "programs"
    programs.mkdir(parents=True)
    (root / "prepared" / "lines").mkdir()
    for index, model_id in enumerate(split_lines):
        line_rng = np.random.default_rng(100 + index)
        genes = (
            SOURCE if model_id != "ACH-T1" else tuple(g for g in SOURCE if g != "B3")
        )
        counts = line_rng.poisson(4.0, size=(CELLS, len(genes))).astype(np.float32)
        counts[:, : len(EFFECT_GENES)] *= 1 + index % 3
        ad.AnnData(
            X=csr_matrix(counts),
            obs=pd.DataFrame(
                {"model_id": model_id},
                index=[f"{model_id}-c{i}" for i in range(CELLS)],
            ),
            var=pd.DataFrame(
                {
                    "ensembl_id": [f"ENSG{SOURCE.index(g):011d}" for g in genes],
                    "gene_symbol": list(genes),
                },
                index=[f"v{i}" for i in range(len(genes))],
            ),
        ).write_h5ad(root / "basal_raw_umi" / f"{model_id}.h5ad")
        np.savez(
            programs / f"{model_id}.npz",
            scores=line_rng.normal(size=(CELLS, len(PROGRAMS))).astype(np.float32),
        )
        np.savez(
            root / "prepared" / "lines" / f"{model_id}.npz",
            controls_tx1=np.zeros((4, 8), dtype=np.float32),
            basal_hvg=line_rng.normal(size=(4, len(HVG))).astype(np.float32),
            q_sc_values=np.zeros((len(EFFECT_GENES), 3), dtype=np.float32),
            q_sc_available=np.ones(len(EFFECT_GENES), dtype=bool),
        )
    (root / "prepared" / "prepared_inputs.json").write_text(
        json.dumps(
            {
                "expression_space": {"target_sum": TARGET_SUM},
                "common_gene_panel": list(EFFECT_GENES),
                "hvg_order": list(HVG),
                "lines": list(split_lines),
            }
        )
    )
    gene_text = root / "gene_text"
    gene_text.mkdir()
    keys = [*(f"G{i}" for i in range(7)), "C1orf1", "OTHER"]  # G7 has no text
    (gene_text / "meta.json").write_text(json.dumps({"keys": keys}))

    joint = yaml.safe_load(Path("configs/geneeffect_joint.yaml").read_text())
    joint["prepared_root"] = str(root / "prepared")
    joint["features"]["selective_min_lines"] = 2
    joint["paths"].update(
        split=str(root / "split.json"),
        gene_effect=str(root / "gene_effect.csv"),
        basal_raw_umi=str(root / "basal_raw_umi"),
        tx1_model_dir=str(root / "tx1_model"),
    )
    (root / "joint.yaml").write_text(yaml.safe_dump(joint))
    prior = yaml.safe_load(Path("configs/context_prior/default_prior.yaml").read_text())
    prior["joint_config"] = str(root / "joint.yaml")
    prior["paths"].update(
        model=str(root / "Model.csv"), bulk_expression=str(root / "bulk.csv")
    )
    (root / "prior.yaml").write_text(yaml.safe_dump(prior))
    noise = yaml.safe_load(
        Path("configs/context_prior/noise_diagnosis.yaml").read_text()
    )
    noise["cell_states"].update(
        programs=list(PROGRAMS), cluster_components=2, cluster_counts=[2, 3]
    )
    (root / "noise.yaml").write_text(yaml.safe_dump(noise))
    config = yaml.safe_load(Path("configs/stage_one/ladder_inputs.yaml").read_text())
    config.update(
        prior_config=str(root / "prior.yaml"),
        noise_config=str(root / "noise.yaml"),
        ladder_root=str(root / "ladder"),
        frame={"folds": 2},
    )
    config["paths"].update(
        gene_text=str(gene_text),
        context_text=str(root / "ladder" / "context_text"),
        tx1_cache=str(root / "tx1"),
    )
    return validate_ladder_config(config)


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

    monkeypatch.setattr(tx1, "_build_tx1_encoder", fake_builder)
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
