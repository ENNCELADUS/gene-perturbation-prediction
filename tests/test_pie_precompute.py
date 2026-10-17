"""The frozen-PIE pass: our batches against PIE's own data path, the DepMap mask,
the profile basis and the assembly.

The PIE tests need the vendored PIE (``.venv-pie``, or an isolated env with
``third_party/pie`` installed) and skip without it; the rest run anywhere.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.experiments import pie_precompute as pp

PIE_TESTS = Path(__file__).resolve().parents[1] / "third_party" / "pie" / "tests"
LINES = ("L1", "L2", "L3")  # L3 has no Cellosaurus text
PANEL = ("GA", "GB", "GC", "GD", "GE")


def _tables(n_lines=len(LINES), panel=PANEL, roles=None) -> pp.Tables:
    lines = [f"L{i + 1}" for i in range(n_lines)]
    roles = roles or ["train_cells", "train_bulk", "test"][:n_lines]
    genes = pd.DataFrame(
        {
            "gene_text_key": list(panel),
            "in_panel": True,
            "selective": [i % 2 == 0 for i in range(len(panel))],
        },
        index=pd.Index(list(panel), name="gene"),
    )
    rng = np.random.default_rng(0)
    return pp.Tables(
        lines=pd.DataFrame({"role": roles}, index=pd.Index(lines, name="model_id")),
        genes=genes,
        pie_input=pd.DataFrame(
            rng.uniform(0, 3, size=(n_lines, len(panel))).astype(np.float32),
            index=lines,
            columns=list(panel),
        ),
    )


# --- pure pieces ----------------------------------------------------------------


def test_basis_keeps_low_rank_changes():
    rng = np.random.default_rng(1)
    changes = rng.normal(size=(400, 3)) @ rng.normal(size=(3, 50)) + 2.0
    basis = pp.fit_basis(changes, 3, "cpu")
    assert basis["components"].shape == (3, 50)
    assert np.isclose(basis["explained"].sum(), 1.0, atol=1e-4)
    restored = (changes - basis["mean"]) @ basis["components"].T @ basis["components"]
    assert np.allclose(restored + basis["mean"], changes, atol=1e-3)


def test_variance_split():
    values = np.zeros((2, 3, 1))
    values[0, :, 0] = [1.0, 2.0, 3.0]  # mean 2, variance 2/3
    values[1, :, 0] = [5.0, 5.0, 5.0]  # mean 5
    split = pp.variance_split(values)
    assert np.isclose(split["within_gene"], (2 / 3) / 2)
    assert np.isclose(split["between_genes"], 2.25)


def test_pilot_genes_and_shards():
    tables = _tables(panel=tuple(f"G{i}" for i in range(30)))
    chosen = pp.pilot_genes(tables, 10, 0)
    assert len(set(chosen)) == 10 and chosen == pp.pilot_genes(tables, 10, 0)
    assert chosen == [g for g in tables.panel if g in set(chosen)]
    parts = [pp.shard_genes(chosen, (i, 3)) for i in range(3)]
    assert sorted(sum(parts, [])) == sorted(chosen)


def test_assemble_centres_each_gene_on_the_frame(tmp_path):
    tables = _tables()
    config = {"ladder_root": str(tmp_path), "paths": {}, "pie": {}}
    out = pp.Layout(tmp_path, tmp_path / "ckpt", tmp_path / "text", config)
    rng = np.random.default_rng(2)
    p = len(PANEL)
    basis = {"mean": rng.normal(size=p), "components": rng.normal(size=(2, p))}
    out.pie.mkdir()
    np.savez(out.pie / "basis.npz", **basis)
    shards = {}
    for i, gene in enumerate(PANEL):
        shard = {
            "dH": rng.normal(size=(3, 4)),
            "dL": rng.normal(size=(3, 4)),
            "own_h": rng.normal(size=(3, 4)),
            "own_dp": rng.normal(size=3),
            "own_x": rng.normal(size=3),
        }
        if i == 0:  # a pilot gene: full change kept, projected at assembly
            full = rng.normal(size=(3, p))
            pp._save_npy(out.full_dir / f"{gene}.npy", full)
            expected_dE = (full - basis["mean"]) @ basis["components"].T
        else:
            shard["dE"] = rng.normal(size=(3, 2))
        pp._save_npz(out.genes_dir / f"{gene}.npz", **shard)
        shards[gene] = shard
    pp.assemble(out, tables)
    frame = [0, 1]  # L1 and L2 are frame lines, L3 a test line
    dH = np.load(out.pie / "dH.npy")
    means = np.load(out.pie / "frame_means.npz")
    assert dH.shape == (p, 3, 4)
    assert np.allclose(dH[1], shards["GB"]["dH"] - shards["GB"]["dH"][frame].mean(0))
    assert np.allclose(means["dH"][1], shards["GB"]["dH"][frame].mean(0))
    dE = np.load(out.pie / "dE.npy")
    assert np.allclose(dE[0] + means["dE"][0], expected_dE, atol=1e-5)
    own_dp = np.load(out.pie / "own_dp.npy")
    assert own_dp.shape == (p, 3, 1)
    index = json.loads((out.pie / "index.json").read_text())
    assert index["genes"] == list(PANEL) and index["lines"] == list(LINES)


# --- PIE --------------------------------------------------------------------------


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_pie(tmp_path, monkeypatch):
    """PIE's tiny synthetic world, our three lines as a controls-only dataset with
    their own context text (L3 has none), and a random tiny PIE over them."""
    pytest.importorskip("pie")
    torch = pytest.importorskip("torch")
    from pie.data.datamodule import PieDataModule
    from pie.data.dataset import RowRef
    from pie.data.delta_p import dequantize
    from pie.model import PieModel

    fixtures = _load("pie_test_fixtures", PIE_TESTS / "fixtures.py")
    helpers = _load("pie_test_model_helpers", PIE_TESTS / "model" / "helpers.py")
    tiny = fixtures.build_tiny_data(tmp_path / "tiny")
    monkeypatch.setenv("PIE_CACHE_DIR", str(tiny.cache_dir))
    train = fixtures.tiny_data_config(tiny)
    stats = PieDataModule(train, tmp_path / "run").setup_stats(rank=0)

    tables = _tables()
    out = pp.Layout(tmp_path / "ladder", tmp_path / "ckpt", tmp_path / "text", {})
    pp.write_dataset(out, tables)
    context = fixtures._dense_source(
        tmp_path / "context_text",
        name="context_text",
        index="context",
        keys=["L1", "L2"],
        dim=5,
        dtype="float32",
        seed=11,
    )
    sources = {**train.source_dirs, "context_text": str(context)}
    data = train.model_copy(
        update={
            "preprocessed_dirs": [str(out.dataset)],
            "dataset_weights": None,
            "source_dirs": sources,
        }
    )
    module = PieDataModule(data, None, stats, seed=0, evidence_source=train)
    module.setup_stats(rank=0)
    dataset = module.make_dataset([RowRef(0, -1, m, "GA") for m in LINES], "none")
    torch.manual_seed(0)
    model = PieModel(
        helpers.tiny_config(), stats, dict(module.source_dims), module.gene_query_text()
    ).eval()
    with torch.no_grad():  # zero-initialised adapters would hide a wrong fusion
        for parameter in model.parameters():
            if not parameter.any():
                parameter.normal_(0.0, 0.2)
    pie = pp.Pie(
        model=model,
        module=module,
        dataset=dataset,
        config=SimpleNamespace(trainer=SimpleNamespace(precision="32-true")),
        centres=dequantize(torch.arange(stats.delta_p.n_bins), stats.delta_p),
        temperature=4.5,
        device=torch.device("cpu"),
        pert_sources=("esm2", "ncbi_text", "perturbation_text", "smiles"),
        context_sources=("context_text",),
    )
    return pie, tables


def _theirs(pie, lines, perturbation):
    from pie.data.dataset import RowRef, collate

    dataset = pie.module.make_dataset(
        [RowRef(0, -1, m, perturbation) for m in lines], "none"
    )
    return pp.mask_depmap(
        collate([dataset[i] for i in range(len(lines))], pie.module.source_dims)
    )


@pytest.mark.parametrize("perturbation", ["GA", "GC"])
def test_our_batches_match_pies_data_path(tiny_pie, perturbation):
    import torch

    from pie.predict import _batch_to

    pie, tables = tiny_pie
    theirs = _batch_to(
        _theirs(pie, list(LINES), perturbation), pie.device, torch.float32
    )
    ours = pp.build_batch(
        pie,
        list(LINES),
        perturbation,
        tables.pie_input.to_numpy(dtype=np.float32),
        pp.gene_evidence(pie, perturbation),
    )
    assert set(ours.source_tokens) == set(theirs.source_tokens)
    for key, value in theirs.groups[0].evidence.items():
        assert torch.equal(ours.groups[0].evidence[key], value)
    features = []
    hook = pie.model.decoder[-1].register_forward_hook(
        lambda _m, _i, output: features.append(output)
    )
    with torch.inference_mode():
        reference = pie.model(theirs)[0]
        hook.remove()
        own = tables.panel.index("GE")  # in the second of two gene chunks
        result = pp.forward(pie, ours, own, heads=True, logits=True)
    decoded = torch.cat(features, dim=1)
    assert np.allclose(result.dp_logits, reference.dp_logits.numpy(), atol=1e-5)
    expected = torch.softmax(reference.dp_logits / pie.temperature, -1) @ pie.centres
    assert np.allclose(result.change, expected.numpy(), atol=1e-5)
    assert np.allclose(result.features, decoded.mean(dim=1).numpy(), atol=1e-5)
    assert np.allclose(result.own, decoded[:, own].numpy(), atol=1e-5)


def test_mixed_precision_pass_returns_float32(tiny_pie):
    import dataclasses

    pie, tables = tiny_pie
    mixed = dataclasses.replace(
        pie, config=SimpleNamespace(trainer=SimpleNamespace(precision="bf16-mixed"))
    )
    batch = pp.build_batch(
        mixed,
        list(LINES),
        "GA",
        tables.pie_input.to_numpy(dtype=np.float32),
        pp.gene_evidence(mixed, "GA"),
    )
    result = pp.forward(mixed, batch, 1, heads=True)
    exact = pp.forward(pie, batch, 1, heads=True)
    for name in ("latent", "features", "own", "change"):
        value = getattr(result, name)
        assert value.dtype == np.float32, name
        assert np.allclose(value, getattr(exact, name), atol=0.05), name


def test_cached_decode_equals_a_fresh_one(tiny_pie):
    pie, tables = tiny_pie
    assert pie.model.cfg.inference_chunk_size < len(PANEL)
    ctrl = tables.pie_input.to_numpy(dtype=np.float32)

    def run(gene):
        batch = pp.build_batch(
            pie, list(LINES), gene, ctrl, pp.gene_evidence(pie, gene)
        )
        return pp.forward(pie, batch, 3, heads=True)

    first = run("GA")
    assert any(key[0] == "control" for key in pie.cache)
    other = run("GC")  # another perturbation reads the cached control encoding
    again = run("GA")
    pie.cache.clear()
    fresh = run("GC")
    for name in ("latent", "features", "own", "change"):
        assert np.array_equal(getattr(first, name), getattr(again, name)), name
        assert np.array_equal(getattr(other, name), getattr(fresh, name)), name
    assert not np.array_equal(first.change, other.change)


def test_agreement_reports_float32_and_bf16(tiny_pie):
    import dataclasses

    pie, tables = tiny_pie
    exact = pp.agreement(pie, tables, ["GA", "GC"])
    assert set(exact) == {"float32"} and exact["float32"] < 1e-5
    mixed = dataclasses.replace(
        pie,
        config=SimpleNamespace(trainer=SimpleNamespace(precision="bf16-mixed")),
        cache={},
    )
    both = pp.agreement(mixed, tables, ["GA", "GC"])
    assert both["float32"] < 1e-5
    assert 0 < both["pie_bf16"] and 0 < both["ours_bf16"]
    assert both["ours_bf16"] <= pp.BF16_ERROR_RATIO * both["pie_bf16"] + 1e-3


def test_reference_pass_reads_context_alone(tiny_pie):
    from pie.data.dataset import NO_SOURCE

    pie, tables = tiny_pie
    blank = pp.gene_evidence(pie, pp.NO_PERTURBATION)
    assert all(not value.any() for value in blank.values())
    batch = pp.build_batch(
        pie, list(LINES), None, tables.pie_input.to_numpy(dtype=np.float32), blank
    )
    assert set(batch.source_tokens) == {"context_text", NO_SOURCE}
    assert batch.source_masks[NO_SOURCE][:, 0].tolist() == [False, False, True]
    result = pp.forward(pie, batch, None, heads=False)
    assert result.features.shape == (3, pie.model.cfg.d_model)


def test_mask_removes_depmap_and_carries_emptied_rows():
    pytest.importorskip("pie")
    import torch
    from pie.data.dataset import NO_SOURCE

    helpers = _load("pie_test_model_helpers", PIE_TESTS / "model" / "helpers.py")
    batch = helpers.tiny_batch(sources=("esm2",))
    b = batch.source_masks["esm2"].shape[0]
    batch.source_tokens[pp.DEPMAP_SOURCE] = torch.ones(b, 1, 3)
    batch.source_masks[pp.DEPMAP_SOURCE] = torch.ones(b, 1, dtype=torch.bool)
    batch.source_masks["esm2"][1] = False  # row 1 had DepMap alone
    masked = pp.mask_depmap(batch)
    assert pp.DEPMAP_SOURCE not in masked.source_tokens
    assert masked.source_masks[NO_SOURCE][:, 0].tolist() == [False, True, False]
    batch.source_masks["esm2"][1] = True
    assert NO_SOURCE not in pp.mask_depmap(batch).source_tokens


def test_dataset_opens_as_pie_preprocessed(tmp_path):
    pytest.importorskip("pie")
    from pie.data.preprocessed import PreprocessedDir

    tables = _tables()
    out = pp.Layout(tmp_path, tmp_path / "ckpt", tmp_path / "text", {})
    opened = PreprocessedDir.open(pp.write_dataset(out, tables))
    assert opened.dataset == pp.DATASET and opened.controls_only
    assert opened.contexts == list(LINES) and opened.genes == list(PANEL)
    assert np.allclose(opened.ctrl_means, tables.pie_input.to_numpy())


def test_dataset_refuses_other_lines_or_control_means(tmp_path):
    pytest.importorskip("pie")
    out = pp.Layout(tmp_path, tmp_path / "ckpt", tmp_path / "text", {})
    tables = _tables()
    pp.write_dataset(out, tables)
    assert pp.write_dataset(out, tables) == out.dataset
    with pytest.raises(ValueError, match="other lines or genes"):
        pp.write_dataset(out, _tables(n_lines=2))
    moved = _tables()
    moved.pie_input.iloc[0, 0] += 1.0
    with pytest.raises(ValueError, match="other control means"):
        pp.write_dataset(out, moved)


def test_reference_pass_refuses_other_lines(tmp_path):
    out = pp.Layout(tmp_path, tmp_path / "ckpt", tmp_path / "text", {})
    tables = _tables()
    path = out.pie / "reference.npz"
    latent = np.zeros((3, 4), dtype=np.float32)
    pp._save_npz(
        path, lines=np.asarray(["L1", "L2", "L9"]), latent=latent, features=latent
    )
    with pytest.raises(ValueError, match="other lines"):
        pp.reference_pass(out, None, tables)
    pp._save_npz(path, lines=np.asarray(list(LINES)), latent=latent, features=latent)
    assert set(pp.reference_pass(out, None, tables)) == {"latent", "features"}
    pp._save_npz(
        path, latent=latent, features=latent
    )  # written before lines were recorded
    assert pp.reference_pass(out, None, tables)["latent"].shape == (3, 4)
    pp._save_npz(path, latent=latent[:2], features=latent[:2])
    with pytest.raises(ValueError, match="other lines"):
        pp.reference_pass(out, None, tables)
