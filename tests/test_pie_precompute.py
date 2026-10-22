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

from src.data.io import write_npy, write_npz
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
            write_npy(out.full_dir / f"{gene}.npy", full)
            expected_dE = (full - basis["mean"]) @ basis["components"].T
        else:
            shard["dE"] = rng.normal(size=(3, 2))
        write_npz(out.genes_dir / f"{gene}.npz", **shard)
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
    write_npz(
        path, lines=np.asarray(["L1", "L2", "L9"]), latent=latent, features=latent
    )
    with pytest.raises(ValueError, match="other lines"):
        pp.reference_pass(out, None, tables)
    write_npz(path, lines=np.asarray(list(LINES)), latent=latent, features=latent)
    assert set(pp.reference_pass(out, None, tables)) == {"latent", "features"}
    write_npz(
        path, latent=latent, features=latent
    )  # written before lines were recorded
    assert pp.reference_pass(out, None, tables)["latent"].shape == (3, 4)
    write_npz(path, latent=latent[:2], features=latent[:2])
    with pytest.raises(ValueError, match="other lines"):
        pp.reference_pass(out, None, tables)


# --- state pseudo-contexts (design §3.5, Task 8) --------------------------------------

STATE_GENES = [("GA", 0), ("GE", 4)]  # GE is in the second of two gene chunks


def _evidence(pie, keys=("GA", "GE")):
    return {k: pp.gene_evidence(pie, k) for k in [pp.NO_PERTURBATION, *keys]}


def test_whole_context_equals_the_main_pass(tiny_pie):
    pie, tables = tiny_pie
    ctrl = tables.pie_input.to_numpy(dtype=np.float32)
    evidence = _evidence(pie)
    names = [f"{m}|set|whole" for m in LINES]
    arrays = pp.state_batch(
        pie, names, list(LINES), ctrl, STATE_GENES, evidence, None, 4
    )
    assert "dE" not in arrays and arrays["dH"].shape[:2] == (3, 2)
    reference = pp.forward(
        pie,
        pp.build_batch(pie, list(LINES), None, ctrl, evidence[pp.NO_PERTURBATION]),
        None,
        heads=False,
    )
    for g, (key, own) in enumerate(STATE_GENES):
        main = pp.forward(
            pie,
            pp.build_batch(pie, list(LINES), key, ctrl, evidence[key]),
            own,
            heads=True,
        )
        assert np.allclose(
            arrays["dH"][:, g], main.features - reference.features, atol=1e-5
        )
        assert np.allclose(arrays["own_h"][:, g], main.own, atol=1e-5)
        assert np.allclose(arrays["own_dp"][:, g, 0], main.change[:, own], atol=1e-6)
        assert np.array_equal(arrays["own_x"][:, g, 0], ctrl[:, own])


def test_two_states_of_one_line_differ_and_each_batch_empties_the_cache(tiny_pie):
    pie, tables = tiny_pie
    line = tables.pie_input.to_numpy(dtype=np.float32)[0]
    ctrl = np.vstack([line, 0.5 * line])
    names, owners = ["L1|set|0", "L1|set|1"], ["L1", "L1"]
    evidence = _evidence(pie)
    arrays = pp.state_batch(pie, names, owners, ctrl, STATE_GENES, evidence, None, 2)
    assert not np.allclose(arrays["dH"][0], arrays["dH"][1])
    assert not np.allclose(arrays["own_dp"][0], arrays["own_dp"][1])
    assert not any(key[0] in ("context", "control") for key in pie.cache)
    # A context name read again with another input is decoded afresh.
    again = pp.state_batch(
        pie, names[:1], owners[:1], ctrl[1:], STATE_GENES, evidence, None, 1
    )
    assert np.allclose(again["dH"][0], arrays["dH"][1], atol=1e-5)


def test_padding_changes_nothing(tiny_pie):
    pie, tables = tiny_pie
    ctrl = tables.pie_input.to_numpy(dtype=np.float32)[:2]
    names, owners = ["L1|set|0", "L2|set|0"], ["L1", "L2"]
    evidence = _evidence(pie)
    exact = pp.state_batch(pie, names, owners, ctrl, STATE_GENES, evidence, None, 2)
    padded = pp.state_batch(pie, names, owners, ctrl, STATE_GENES, evidence, None, 5)
    for name, value in exact.items():
        assert padded[name].shape == value.shape
        assert np.allclose(padded[name], value, atol=1e-6), name


def test_hoisted_owner_encoder_equals_the_unhoisted_one(tiny_pie):
    pie, tables = tiny_pie
    full = tables.pie_input.to_numpy(dtype=np.float32)
    ctrl = np.vstack([full[0], 0.3 * full[0], full[2]])
    names, owners = ["L1|set|0", "L1|set|1", "L3|set|0"], ["L1", "L1", "L3"]
    evidence = pp.gene_evidence(pie, "GA")
    batch = pp.build_batch(pie, names, "GA", ctrl, evidence, owners=owners)
    plain = pp.forward_tensors(pie, batch, 1, heads=True)
    hoisted = pp.forward_tensors(
        pie, batch, 1, heads=True, memory=pp.owner_memory(pie, owners, "GA", evidence)
    )
    for name in ("latent", "features", "own", "change"):
        assert np.allclose(hoisted[name].numpy(), plain[name].numpy(), atol=1e-5), name
    # The owner's text, not the context's name, reaches the encoder.
    other = pp.owner_memory(pie, ["L2", "L2", "L3"], "GA", evidence)
    assert not np.allclose(
        other.float().mean(dim=1)[0].numpy(), hoisted["latent"][0].numpy()
    )


def _state_tables_on_disk(root, tables, name):
    line = tables.pie_input.to_numpy(dtype=np.float32)
    contexts = ["L1|set|0", "L1|set|1", "L1|set|whole", "L3|set|0", "L3|set|whole"]
    inputs = np.vstack([line[0], 0.5 * line[0], 0.75 * line[0], line[2], line[2]])
    masses = pd.DataFrame(
        {
            "model_id": ["L1", "L1", "L1", "L3", "L3"],
            "state": ["0", "1", "whole", "0", "whole"],
            "mass": [0.5, 0.5, np.nan, 1.0, np.nan],
            "effective_count": 10.0,
            "merged": "",
        },
        index=pd.Index(contexts, name="context"),
    )
    folder = root / "states" / name
    folder.mkdir(parents=True)
    pd.DataFrame(inputs, index=masses.index, columns=tables.panel).to_parquet(
        folder / "inputs.parquet"
    )
    masses.to_parquet(folder / "masses.parquet")
    return pp.read_state_tables(root / "states", name, tables)


def test_assembly_refuses_a_line_pass_from_other_state_inputs(tmp_path):
    tables = _tables()
    out = pp.Layout(tmp_path, tmp_path / "ckpt", tmp_path / "text", {"seed": 0})
    states = _state_tables_on_disk(tmp_path, tables, "set")
    config = {"pie": {"genes": 2, "contexts_per_batch": 2}}
    genes = pp.state_genes(tables, config, 0)
    layout = pp.state_layout(out)
    for model_id, fingerprint in (("L1", "stale"), ("L3", states.fingerprints["L3"])):
        k = states.owners.count(model_id)
        write_npz(
            layout.line("set", model_id),
            contexts=np.asarray([c for c in states.contexts if c.startswith(model_id)]),
            genes=np.asarray(genes),
            fingerprint=np.asarray(fingerprint),
            own_x=np.zeros((2, k, 1), dtype=np.float32),
        )
    out.pie.mkdir(parents=True)
    np.savez(out.pie / "frame_means.npz", own_x=np.zeros((len(PANEL), 1)))
    with pytest.raises(ValueError, match="L1: the pass was computed from other"):
        pp.assemble_states(out, tables, config, "set")


def test_states_pass_resumes_per_line_and_assembles_centred(tiny_pie, tmp_path):
    import torch

    pie, tables = tiny_pie
    out = pp.Layout(tmp_path, tmp_path / "ckpt", tmp_path / "text", {"seed": 0})
    states = _state_tables_on_disk(tmp_path, tables, "set")
    assert states.lines() == ["L1", "L3"]
    config = {"pie": {"genes": 2, "contexts_per_batch": 2}}
    genes = pp.state_genes(tables, config, 0)
    rng = np.random.default_rng(3)
    basis = {
        "mean": torch.zeros(len(PANEL)),
        "components": torch.from_numpy(
            rng.normal(size=(2, len(PANEL))).astype(np.float32)
        ),
    }
    layout = pp.state_layout(out)
    assert layout.root == tmp_path / "states" / "pie"  # moves with states/
    first = pp.states_pass(pie, tables, states, genes, basis, ["L1"], layout, 2, {})
    assert first["contexts"] == 3
    assert layout.line("set", "L1").is_file()
    assert not layout.line("set", "L3").is_file()
    again = pp.states_pass(
        pie, tables, states, genes, basis, ["L1", "L3"], layout, 2, {}
    )
    assert again["contexts"] == 2
    with np.load(layout.line("set", "L1")) as payload:
        stored = {k: payload[k] for k in payload.files}
    assert stored["contexts"].tolist() == states.contexts[:3]
    assert stored["dH"].dtype == np.float16 and stored["dH"].shape[:2] == (2, 3)
    assert str(stored["fingerprint"]) == states.fingerprints["L1"]
    # The same context names over other state inputs: the stale pass is refused.
    folder = tmp_path / "states" / "set"
    inputs = pd.read_parquet(folder / "inputs.parquet")
    original = inputs.copy()
    inputs.iloc[0, 0] += 0.25
    inputs.to_parquet(folder / "inputs.parquet")
    moved = pp.read_state_tables(tmp_path / "states", "set", tables)
    assert moved.fingerprints["L1"] != states.fingerprints["L1"]
    assert moved.fingerprints["L3"] == states.fingerprints["L3"]
    with pytest.raises(ValueError, match="other state inputs"):
        pp.states_pass(pie, tables, moved, genes, basis, ["L1"], layout, 2, {})
    original.to_parquet(folder / "inputs.parquet")
    d = pie.model.cfg.d_model
    out.pie.mkdir(parents=True, exist_ok=True)
    means = {
        "dH": rng.normal(size=(len(PANEL), d)),
        "dE": rng.normal(size=(len(PANEL), 2)),
        "own_h": rng.normal(size=(len(PANEL), d)),
        "own_dp": rng.normal(size=(len(PANEL), 1)),
        "own_x": rng.normal(size=(len(PANEL), 1)),
    }
    np.savez(out.pie / "frame_means.npz", **means)
    assert pp.assemble_states(out, tables, config, "set") == {
        "contexts": 5,
        "genes": 2,
    }
    dE = np.load(layout.root / "set" / "dE.npy")
    position = [tables.panel.index(g) for g in genes]
    expected = stored["dE"].astype(np.float32) - means["dE"][position][:, None, :]
    assert np.allclose(dE[:, :3].astype(np.float32), expected, atol=1e-2)
    assert not layout.lines("set").exists()
    index = json.loads(layout.index("set").read_text())
    assert index["fingerprints"] == states.fingerprints
    # An assembled set is skipped by the pass and by assembly, not recomputed.
    assert pp.states_pass(pie, tables, states, genes, basis, ["L1"], layout, 2, {}) == {
        "contexts": 0,
        "assembled": True,
    }
    assert pp.assemble_states(out, tables, config, "set")["skipped"]
    # Changed state inputs refuse the assembly at pass and assembly alike.
    inputs.to_parquet(folder / "inputs.parquet")  # the moved inputs
    moved = pp.read_state_tables(tmp_path / "states", "set", tables)
    with pytest.raises(ValueError, match="assembled from other state inputs"):
        pp.states_pass(pie, tables, moved, genes, basis, ["L1"], layout, 2, {})
    with pytest.raises(ValueError, match="assembled from other state inputs"):
        pp.assemble_states(out, tables, config, "set")
