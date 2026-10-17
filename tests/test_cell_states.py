"""Cell-state vocabularies and state inputs (``src.data.cell_states``): the state
inputs reproduce PIE's input, the controls keep what they must, fits refuse
held-out lines, projection is deterministic."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import sparse

from src.data import cell_states as cs
from src.data.ladder_cells import gene_summaries
from src.data.splits import FixedSplit
from src.experiments.config import load_states_config, validate_states_config

PANEL = [f"G{i}" for i in range(12)]
SOURCE = [*(f"g{i}" for i in range(10)), "OTHER", "G0"]  # G0 listed twice, G10-11 not


def _line(seed: int = 0, n: int = 80):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rng.uniform(0.2, 4.0, size=len(SOURCE)), size=(n, len(SOURCE)))
    counts[:5] = 0  # cells without any UMI on the panel stay zero
    counts[:5, 10] = 3
    summaries = gene_summaries(
        sparse.csr_matrix(counts), SOURCE, PANEL, rng=np.random.default_rng(1)
    )
    fill = np.where(summaries.listed, summaries.pie_input, 0.7)  # the frame mean
    cells = cs.per_cell_log_cp10k(counts, SOURCE, PANEL, fill, summaries.listed)
    return cells, fill


def test_per_cell_inputs_average_to_pies_input():
    cells, pie_input = _line()
    assert not cells.listed[10:].any()
    assert np.allclose(cells.mean(), pie_input, atol=1e-6)
    with pytest.raises(ValueError, match="other panel genes"):
        cs.per_cell_log_cp10k(
            np.ones((3, len(SOURCE))), SOURCE, PANEL, pie_input, np.ones(12, bool)
        )


def test_state_masses_times_inputs_reproduce_the_line_input_after_merging():
    cells, pie_input = _line()
    rng = np.random.default_rng(2)
    q = rng.dirichlet([2.0, 2.0, 2.0, 0.05], size=cells.n_cells)
    q[:, 3] = 0.0
    q /= q.sum(axis=1, keepdims=True)
    q[0, 3] = 0.6  # a state of effective count near one
    q[0, :3] = 0.4 / 3
    assert cs.effective_count(q)[3] < 10
    state = cs.state_masses_and_inputs(q, cells, ["a", "b", "c", "rare"], 10.0)
    assert "rare" not in state.labels
    assert any("rare" in merged for merged in state.merged)
    assert np.isclose(state.mass.sum(), 1.0)
    assert (state.effective >= 10).all()
    assert np.allclose(state.mass @ state.inputs, pie_input, atol=1e-6)
    assert np.allclose(state.inputs[:, 10:], 0.7)  # unlisted genes keep the fill
    one = cs.state_masses_and_inputs(q, cells, ["a", "b", "c", "rare"], 1e9)
    assert len(one.labels) == 1 and np.allclose(one.inputs[0], pie_input, atol=1e-6)


def test_permuted_memberships_keep_mass_effective_count_and_line_input():
    cells, pie_input = _line()
    q = np.random.default_rng(3).dirichlet([1.0, 1.0, 1.0], size=cells.n_cells)
    permuted = cs.permute_memberships(q, "ACH-1", "hvg_k4", 0)
    assert not np.array_equal(permuted, q)
    assert np.allclose(permuted.sum(axis=0), q.sum(axis=0))
    assert np.allclose(cs.effective_count(permuted), cs.effective_count(q))
    assert np.allclose(permuted.T @ permuted, q.T @ q)  # overlap kept
    state = cs.state_masses_and_inputs(permuted, cells, ["a", "b", "c"], 1.0)
    assert np.allclose(state.mass @ state.inputs, pie_input, atol=1e-6)
    again = cs.permute_memberships(q, "ACH-1", "hvg_k4", 0)
    assert np.array_equal(again, permuted)
    assert not np.array_equal(cs.permute_memberships(q, "ACH-1", "hvg_k4", 1), permuted)


def test_cycling_weights_and_input():
    cells, _ = _line()
    p = np.linspace(0, 1, cells.n_cells)
    w = cs.cycling_weights(p, 0.05)
    assert np.isclose(w.sum(), 1.0) and (w > 0).all()
    assert np.isclose(w[-1] / w[0], 1.05 / 0.05)
    expected = np.asarray(cells.u.T @ w).ravel()
    assert np.allclose(cells.weighted_inputs(w[:, None])[0, :10], expected[:10])


def _split():
    return FixedSplit(
        train=("ACH-T0", "ACH-T1", "ACH-U0"),
        val=("ACH-V0",),
        test=("ACH-S0",),
        unlabeled_train=("ACH-U0",),
    )


def test_fit_sample_weighs_lines_equally_and_rejects_held_out_lines():
    roles = {"ACH-T0": "train_cells", "ACH-T1": "train_cells", "ACH-V0": "val"}
    sample = cs.fit_sample(
        {"ACH-T0": 40, "ACH-T1": 7}, roles, _split(), cells_per_line=20, seed=0
    )
    assert sample.groupby("model_id").size().to_dict() == {"ACH-T0": 20, "ACH-T1": 20}
    small = sample.loc[sample["model_id"] == "ACH-T1", "row"]
    assert set(small) == set(range(7))  # every cell, then draws with replacement
    big = sample.loc[sample["model_id"] == "ACH-T0", "row"]
    assert big.is_unique
    again = cs.fit_sample(
        {"ACH-T0": 40, "ACH-T1": 7}, roles, _split(), cells_per_line=20, seed=0
    )
    assert again.equals(sample)
    with pytest.raises(ValueError, match="may not enter a fit"):
        cs.fit_sample({"ACH-V0": 40}, roles, _split(), cells_per_line=20, seed=0)
    with pytest.raises(ValueError, match="not a labeled train member"):
        cs.fit_sample(
            {"ACH-V0": 40},
            {"ACH-V0": "train_cells"},
            _split(),
            cells_per_line=5,
            seed=0,
        )


def test_projection_is_deterministic_and_row_independent():
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(4)
    x = np.vstack([rng.normal(m, 1.0, size=(200, 3)) for m in (-3.0, 0.0, 3.0)])
    params = cs.fit_mixture(x, 3, seed=0, inits=2)
    params |= {"centred": np.asarray(True)}
    centre = x.mean(axis=0)
    q = cs.project(params, x, centre)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert np.array_equal(q, cs.project(params, x, centre))
    assert np.allclose(cs.project(params, x[:7], centre), q[:7], atol=1e-12)
    model = GaussianMixture(3, covariance_type="full", n_init=2, random_state=0).fit(x)
    assert np.allclose(
        cs.mixture_responsibilities(params, x - centre),
        model.predict_proba(x - centre),
        atol=1e-8,
    )


def test_pca_and_scale_features():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 6))
    params = {
        "centred": np.asarray(False),
        "scale": np.full(6, 2.0),
        **cs.fit_pca(x / 2.0, 3, 0),
    }
    features = cs.mixture_features(params, x, None)
    assert features.shape == (100, 3)
    with pytest.raises(ValueError, match="centre"):
        cs.mixture_features({**params, "centred": np.asarray(True)}, x, None)


def test_cycle_labels_and_probability():
    means = np.asarray([[0.4, -0.2], [-0.3, -0.3], [-0.1, 0.5]])
    labels = cs.cycle_labels(means)
    assert labels == ["s", "noncycling", "g2m"]
    q = np.asarray([[0.2, 0.7, 0.1]])
    assert np.isclose(cs.cycling_probability(q, labels)[0], 0.3)


def test_program_memberships_drop_cycle_and_fall_back_to_background():
    usages = np.asarray([[1.0, 3.0, 5.0, 0.0], [0.0, 0.0, 2.0, 0.0]])
    kinds = ["recurrent", "recurrent", "cycle", "background"]
    q = cs.program_memberships(usages, kinds)
    assert q.shape == (2, 3)
    assert np.allclose(q[0], [0.25, 0.75, 0.0])
    assert np.allclose(q[1], [0.0, 0.0, 1.0])


def test_names_and_sets():
    config = load_states_config(Path("configs/stage_one/states/telemetry.yaml"))
    name = cs.context_name("ACH-000001", "hvg_k4", "2")
    assert name == "ACH-000001|hvg_k4|2" and cs.context_owner(name) == "ACH-000001"
    with pytest.raises(ValueError):
        cs.context_name("ACH-1", "a|b", "0")
    vocabularies = cs.vocabularies(config)
    assert vocabularies[:2] == ["cycle", "programs"] and "qc_k8" in vocabularies
    sets = cs.state_sets(config)
    assert sets[len(vocabularies)] == cs.CYCLING_SET
    permuted = sets[len(vocabularies) + 1 :]
    assert "qc_k4__perm0" not in permuted and "tx1_k6__perm2" in permuted
    assert permuted[0] == "cycle__perm0"
    assert cs.base_vocabulary("scvi_centred_k8__perm1") == "scvi_centred_k8"
    assert cs.family("scvi_absolute_k4") == "scvi_absolute"


def test_states_config_rejects_unknown_and_missing_keys():
    raw = yaml.safe_load(Path("configs/stage_one/states/telemetry.yaml").read_text())
    validate_states_config(raw)
    extra = copy.deepcopy(raw)
    extra["fit"]["surprise"] = 1
    with pytest.raises(ValueError, match="unknown"):
        validate_states_config(extra)
    missing = copy.deepcopy(raw)
    del missing["inputs"]["cycling_floor"]
    with pytest.raises(ValueError, match="missing"):
        validate_states_config(missing)
    roles = copy.deepcopy(raw)
    roles["project"]["roles"] = ["val"]
    with pytest.raises(ValueError, match="train_cells first"):
        validate_states_config(roles)
