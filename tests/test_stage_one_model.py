"""The stage-one model's pieces without PIE: Cilantro's FiLM layer and tail, the
gene encoder, the FiLM arms, the calibration slope, the config schema and the
training loop."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.splits import FixedSplit
from src.eval.metrics import calibration_slope
from src.experiments.config import validate_stage_one_config
from src.model.stage_one import (
    CilantroTail,
    DeltaFiLM,
    FiLM,
    GeneEncoder,
    InputDims,
    PairBatch,
    build_model,
)
from src.training import stage_one_trainer as trainer

REPO = Path(__file__).resolve().parents[1]
CILANTRO = REPO / "third_party" / "cilantro_sl" / "nn_helpers"
CONFIGS = REPO / "configs" / "stage_one" / "architecture"


def _batch(genes: int, lines: int, *, dim: int = 6, line_width: int = 4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return PairBatch(
        gene_vectors={"gene_text": torch.randn(genes, dim, generator=generator)},
        gene_present={"gene_text": torch.ones(genes, dtype=torch.bool)},
        line=torch.randn(lines, line_width, generator=generator),
    )


def _line_film(line_width: int = 4, dim: int = 6) -> DeltaFiLM:
    return DeltaFiLM(
        input="components",
        input_dim=line_width,
        own_dim=0,
        gene_encoder=GeneEncoder({"gene_text": dim}, 8),
        context=None,
        context_width=0,
        state=None,
        hidden=16,
    )


# --- Cilantro's layers --------------------------------------------------------------


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_film_and_tail_match_cilantros_filmnet128():
    """Same weights, same outputs as Cilantro's FiLMNet128 (the H20's copy of
    ``third_party/cilantro_sl``; skipped where it is absent)."""
    if not (CILANTRO / "net_layers.py").is_file():
        pytest.skip("third_party/cilantro_sl is on the H20 only")
    _load("film", CILANTRO / "film.py")
    net_layers = _load("net_layers", CILANTRO / "net_layers.py")
    torch.manual_seed(0)
    theirs = net_layers.FiLMNet128().eval()
    inflow, film, tail = torch.nn.Linear(512, 256), FiLM(256, 128), CilantroTail(256)
    with torch.no_grad():
        inflow.load_state_dict(theirs.l1.state_dict())
        film.cond_gen.load_state_dict(theirs.film_layer.cond_gen.state_dict())
        for name in ("l2", "l3", "l4"):
            getattr(tail, name).load_state_dict(getattr(theirs, name).state_dict())
        x, protein = torch.randn(7, 512), torch.randn(7, 128)
        expected, expected_v = theirs(x, protein)
        z, v = tail(film(torch.relu(inflow(x)), protein))
    assert torch.allclose(z, expected.squeeze(-1), atol=1e-6)
    assert torch.allclose(v, expected_v, atol=1e-6)


def test_the_untrained_tail_outputs_zero():
    tail = CilantroTail(16)
    z, v = tail(torch.randn(3, 5, 16))
    assert z.shape == (3, 5) and v.shape == (3, 5, 32)
    assert torch.equal(z, torch.zeros_like(z))
    model = _line_film()
    z, _ = model(_batch(4, 3))
    assert torch.equal(z, torch.zeros(4, 3))


def test_a_gene_never_trained_on_decodes_from_its_sources():
    """No per-gene parameter: a new gene's prediction comes from its source vector,
    and two new genes with the same vector predict alike."""
    torch.manual_seed(0)
    model = _line_film()
    with torch.no_grad():
        model.tail.l4.weight.normal_()
    batch = _batch(3, 5)
    vector = torch.randn(1, 6)
    batch.gene_vectors = {
        "gene_text": torch.cat([batch.gene_vectors["gene_text"][:1], vector, vector])
    }
    z, _ = model(batch)
    assert torch.isfinite(z).all()
    assert torch.allclose(z[1], z[2]) and not torch.allclose(z[0], z[1])
    assert not any(
        "gene" in name and p.shape[0] > 100 for name, p in model.named_parameters()
    )


def test_a_masked_source_contributes_nothing():
    encoder = GeneEncoder({"a": 3, "b": 2}, 4)
    vectors = {"a": torch.randn(2, 3), "b": torch.randn(2, 2)}
    present = {"a": torch.tensor([True, True]), "b": torch.tensor([True, False])}
    out = encoder(vectors, present)
    alone = encoder.projections["a"](vectors["a"][1:])
    assert torch.allclose(out[1:], alone)


def test_the_profile_delta_reads_before_and_after_states():
    """ΔE = E(after) - E(before), beside its value at the frame-mean states; a
    linear E makes the centred part the change of the stored coordinates alone."""
    torch.manual_seed(0)
    dims = InputDims(gene_sources={"gene_text": 6}, delta=5, own=0, text=3)
    config = {
        "data": {"input": "dE"},
        "model": {
            "arm": "film",
            "gene_width": 8,
            "hidden": 16,
            "state_width": 5,
            "context": None,
            "trunk": None,
        },
    }
    model = build_model(config, dims)
    linear = torch.nn.Linear(5, 5)
    model.state.net = torch.nn.Sequential(linear)
    batch = _batch(2, 3)
    batch.line = None
    batch.before = torch.randn(3, 5)
    batch.delta = torch.randn(2, 3, 5)
    batch.shift = torch.randn(2, 5)
    with torch.no_grad():
        pair = model.pair_input(batch)
        expected = batch.delta @ linear.weight.T
    assert torch.allclose(pair[..., :5], expected, atol=1e-5)


# --- the calibration slope ---------------------------------------------------------


def test_calibration_slope_is_one_on_scaled_truth():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(30, 6))
    offsets = rng.normal(size=6)
    assert calibration_slope(truth + offsets, truth) == pytest.approx(1.0)
    assert calibration_slope(0.5 * truth, truth) == pytest.approx(2.0)
    truth[3, 2] = np.nan
    assert np.isfinite(calibration_slope(truth, truth))
    constant = np.tile(offsets, (30, 1))
    assert np.isnan(calibration_slope(constant, truth))


# --- configs ------------------------------------------------------------------------


def _config(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_architecture_config_validates(path):
    validate_stage_one_config(yaml.safe_load(path.read_text()))


def test_stage_one_configs_refuse_unknown_and_missing_keys():
    config = _config("film_dH")
    extra = copy.deepcopy(config)
    extra["train"]["momentum"] = 0.9
    with pytest.raises(ValueError, match="unknown"):
        validate_stage_one_config(extra)
    missing = copy.deepcopy(config)
    del missing["data"]["permuted"]
    with pytest.raises(ValueError, match="missing"):
        validate_stage_one_config(missing)
    trunk = _config("direct_readout")
    del trunk["model"]["trunk"]["n_latents"]
    with pytest.raises(ValueError, match="missing"):
        validate_stage_one_config(trunk)
    wrong = copy.deepcopy(config)
    wrong["data"]["own_slots"] = ["own_q"]
    with pytest.raises(ValueError, match="own_slots"):
        validate_stage_one_config(wrong)
    with pytest.raises(ValueError, match="kind"):
        validate_stage_one_config({**config, "kind": "sweep"})


# --- the training loop ---------------------------------------------------------------


SPLIT = FixedSplit(train=("T0", "T1"), val=("V0",), test=("S0",))


def _loop_config(**train) -> dict:
    config = _config("components_reader")
    config["train"].update(
        max_epochs=3, patience=3, warmup_epochs=0, lines_per_step=4, genes_per_step=3
    )
    config["train"].update(train)
    config["eval"].update(lines_per_chunk=3, genes_per_chunk=2)
    return config


def _toy(seed: int = 0):
    """Lines 0-7 train, 8-11 validate; the target is a function of the line input
    and the gene vector."""
    generator = torch.Generator().manual_seed(seed)
    genes, lines = 5, 12
    gene_vectors = torch.randn(genes, 6, generator=generator)
    line_inputs = torch.randn(lines, 4, generator=generator)
    targets = (line_inputs[:, :1] * gene_vectors[:, 0]).float()

    def batch(g, c):
        return PairBatch(
            gene_vectors={"gene_text": gene_vectors[g]},
            gene_present={"gene_text": torch.ones(len(g), dtype=torch.bool)},
            line=line_inputs[c],
        )

    labels = targets.clone()
    labels[8:] = torch.nan
    data = trainer.TrainingData(batch, labels, torch.arange(8), genes)
    return data, targets


def _validator(data, targets, config, *, fail_after=None):
    """Validation on lines 8-11: an undefined score while the prediction is
    constant per gene (as the base is); raises after ``fail_after`` calls."""
    calls = []

    def validate(module):
        calls.append(None)
        if fail_after is not None and len(calls) > fail_after:
            raise RuntimeError("interrupted")
        z = trainer.predict(module, data.batch, torch.arange(8, 12), 5, config)
        error = float(np.mean((z.T - targets[8:].numpy()) ** 2))
        spearman = None if np.allclose(z, z[:, :1]) else -error
        return {"val_selective_spearman": spearman, "val_mse": error}

    return validate


def test_epoch_zero_is_the_base_and_never_a_candidate_when_undefined(tmp_path):
    config = _loop_config()
    data, targets = _toy()
    torch.manual_seed(0)
    result = trainer.fit(
        _line_film(), data, config, tmp_path, SPLIT, _validator(data, targets, config)
    )
    first = result.history[0]
    assert first["epoch"] == 0 and first["candidate"] is False
    assert first["val_mse"] == pytest.approx(float((targets[8:] ** 2).mean()))
    assert result.state.best_epoch >= 1
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "last.pt").is_file()


def test_resume_refuses_another_config_and_checkpoints_another_membership(tmp_path):
    config = _loop_config(max_epochs=1)
    data, targets = _toy()
    validate = _validator(data, targets, config)
    torch.manual_seed(0)
    trainer.fit(_line_film(), data, config, tmp_path, SPLIT, validate)
    changed = _loop_config(max_epochs=1, learning_rate=0.5)
    with pytest.raises(ValueError, match="another config"):
        trainer.fit(_line_film(), data, changed, tmp_path, SPLIT, validate)
    moved = FixedSplit(train=("T0",), val=("V0", "T1"), test=("S0",))
    with pytest.raises(ValueError, match="split membership"):
        trainer.load_strict(tmp_path / "best.pt", _line_film(), config, moved)
    trainer.load_strict(tmp_path / "best.pt", _line_film(), config, SPLIT)
    with pytest.raises(RuntimeError):
        trainer.load_strict(
            tmp_path / "best.pt", _line_film(line_width=3), config, SPLIT
        )


def test_a_resumed_run_ends_where_an_uninterrupted_one_does(tmp_path):
    config = _loop_config(max_epochs=3, patience=5)
    data, targets = _toy()
    torch.manual_seed(0)
    whole = trainer.fit(
        _line_film(),
        data,
        config,
        tmp_path / "whole",
        SPLIT,
        _validator(data, targets, config),
    )
    torch.manual_seed(0)
    with pytest.raises(RuntimeError, match="interrupted"):
        trainer.fit(
            _line_film(),
            data,
            config,
            tmp_path / "parts",
            SPLIT,
            _validator(data, targets, config, fail_after=2),
        )
    resumed = trainer.fit(
        _line_film(),
        data,
        config,
        tmp_path / "parts",
        SPLIT,
        _validator(data, targets, config),
    )
    assert resumed.history == whole.history
