"""Cell-state vocabularies and state inputs (``src.data.cell_states``)."""

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
from src.experiments.states.config import load_states_config, validate_states_config

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


def _states(q, cells, labels, minimum):
    """A line's states after merging the small ones, as the telemetry builds them."""
    q, labels, merged = cs.merge_small(q, cells, labels, minimum)
    return cs.state_inputs(q, cells, labels, merged)


def test_state_masses_times_inputs_reproduce_the_line_input_after_merging():
    cells, pie_input = _line()
    rng = np.random.default_rng(2)
    q = rng.dirichlet([2.0, 2.0, 2.0, 0.05], size=cells.n_cells)
    q[:, 3] = 0.0
    q /= q.sum(axis=1, keepdims=True)
    q[0, 3] = 0.6  # a state of effective count near one
    q[0, :3] = 0.4 / 3
    assert cs.effective_count(q)[3] < 10
    state = _states(q, cells, ["a", "b", "c", "rare"], 10.0)
    assert "rare" not in state.labels
    assert any("rare" in merged for merged in state.merged)
    assert np.isclose(state.mass.sum(), 1.0)
    assert (state.effective >= 10).all()
    assert np.allclose(state.mass @ state.inputs, pie_input, atol=1e-6)
    assert np.allclose(state.inputs[:, 10:], 0.7)  # unlisted genes keep the fill
    one = _states(q, cells, ["a", "b", "c", "rare"], 1e9)
    assert len(one.labels) == 1 and np.allclose(one.inputs[0], pie_input, atol=1e-6)


def test_permuted_memberships_keep_mass_effective_count_and_line_input():
    cells, pie_input = _line()
    q = np.random.default_rng(3).dirichlet([1.0, 1.0, 1.0], size=cells.n_cells)
    permuted = cs.permute_memberships(q, "ACH-1", "hvg_k4", 0)
    assert not np.array_equal(permuted, q)
    assert np.allclose(permuted.sum(axis=0), q.sum(axis=0))
    assert np.allclose(cs.effective_count(permuted), cs.effective_count(q))
    assert np.allclose(permuted.T @ permuted, q.T @ q)  # overlap kept
    state = _states(permuted, cells, ["a", "b", "c"], 1.0)
    assert np.allclose(state.mass @ state.inputs, pie_input, atol=1e-6)
    again = cs.permute_memberships(q, "ACH-1", "hvg_k4", 0)
    assert np.array_equal(again, permuted)
    assert not np.array_equal(cs.permute_memberships(q, "ACH-1", "hvg_k4", 1), permuted)


def test_permuted_sets_permute_the_merged_states():
    cells, pie_input = _line()
    rng = np.random.default_rng(8)
    q = rng.dirichlet([2.0, 2.0, 2.0], size=cells.n_cells)
    q = np.column_stack([q, np.zeros(cells.n_cells)])
    q[:3, :3] *= 0.2
    q[:3, 3] = 0.8  # a small state the merge absorbs by its input
    labels = ["a", "b", "c", "small"]
    real, permuted = cs.permuted_state_inputs(
        q, cells, labels, 10.0, model_id="ACH-1", vocabulary="hvg_k4", seeds=[0, 1]
    )
    assert "small" not in real.labels  # merging happened
    for state in permuted.values():
        assert state.labels == real.labels and state.merged == real.merged
        assert np.allclose(state.mass, real.mass)
        assert np.allclose(state.effective, real.effective)
        assert np.allclose(state.mass @ state.inputs, pie_input, atol=1e-6)
        assert not np.allclose(state.inputs, real.inputs)


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
    assert sample.groupby("model_id").size().to_dict() == {"ACH-T0": 20, "ACH-T1": 7}
    small = sample.loc[sample["model_id"] == "ACH-T1", "row"]
    assert list(small) == list(range(7))  # every cell once, never duplicated
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


def _null_component_shares(padded: bool) -> np.ndarray:
    """Per component of a 50-d mixture fitted on line-centred null cells (no
    states), the largest share of its responsibility held by one line."""
    rng = np.random.default_rng(6)
    counts = {f"ACH-T{i}": (12 if i < 5 else 300) for i in range(30)}
    cells = {m: rng.normal(size=(n, 50)) for m, n in counts.items()}
    split = FixedSplit(train=tuple(counts), val=(), test=())
    roles = dict.fromkeys(counts, "train_cells")
    sample = cs.fit_sample(counts, roles, split, cells_per_line=200, seed=0)
    parts, owners = [], []
    for model_id, group in sample.groupby("model_id", sort=False):
        rows = group["row"].to_numpy()
        if padded and len(rows) < 200:  # the replaced rule: copies up to 200
            extra = np.random.default_rng(1).choice(rows, size=200 - len(rows))
            rows = np.concatenate([rows, extra])
        x = cells[model_id]
        parts.append(x[rows] - x.mean(axis=0))
        owners += [model_id] * len(rows)
    features = np.vstack(parts)
    params = cs.fit_mixture(features, 6, seed=0, inits=1)
    q = cs.mixture_responsibilities(params, features)
    owners = np.asarray(owners)
    shares = np.stack([q[owners == m].sum(axis=0) for m in counts]) / q.sum(axis=0)
    return shares.max(axis=0)


def test_null_mixture_components_are_not_one_lines_duplicated_cells():
    # 30 lines: no line holds more than a tenth of any component.
    assert (_null_component_shares(padded=False) < 0.1).all()
    # Sensitivity: with the 5 small lines padded by copies, components lock on them.
    assert (_null_component_shares(padded=True) > 0.25).sum() >= 3


def test_chemistry_resolves_per_cell_and_fails_closed():
    import pandas as pd

    table = pd.DataFrame(
        [
            ("mixseq", "", "", "mixed"),
            ("mixseq", "", "DMSO_6hr_expt3", "10x 3' v2"),
            ("mixseq", "", "DMSO_24hr_expt10", "10x 3' v3"),
            ("mixseq", "ACH-2", "DMSO_24hr_expt10", "10x 3' v3.1"),
            ("perturb", "", "", "mixed"),
            ("perturb", "ACH-9", "", "10x 5' v3"),
            ("kinker", "", "", "10x 3' v2"),
        ],
        columns=["source", "model_id", "arm", "chemistry"],
    )
    arms = ["DMSO_24hr_expt10", "DMSO_6hr_expt3", "DMSO_24hr_expt10"]
    assert list(cs.resolve_chemistry(table, "mixseq", "ACH-1", arms)) == [
        "10x 3' v3",
        "10x 3' v2",
        "10x 3' v3",
    ]
    # A ModelID and arm row is more specific than the arm alone.
    assert cs.resolve_chemistry(table, "mixseq", "ACH-2", arms[:1])[0] == "10x 3' v3.1"
    assert cs.resolve_chemistry(table, "perturb", "ACH-9", [""])[0] == "10x 5' v3"
    assert (
        cs.resolve_chemistry(table, "kinker", "ACH-5", ["", "x"]).tolist()
        == ["10x 3' v2"] * 2
    )
    with pytest.raises(ValueError, match="covers arm 'DMSO_3hr'"):
        cs.resolve_chemistry(table, "mixseq", "ACH-1", ["DMSO_3hr"])
    with pytest.raises(ValueError, match="no chemistry row"):
        cs.resolve_chemistry(table, "perturb", "ACH-8", [""])  # mixed, no line row


def test_chemistry_table_rows_are_unique_and_mixed_sources_resolve():
    import pandas as pd

    table = pd.read_csv(
        "configs/benchmarks/single_cell_source_chemistry.csv",
        keep_default_na=False,
        dtype=str,
    )
    assert not table.duplicated(["source", "model_id", "arm"]).any()
    assert set(table.loc[table["chemistry"] == "mixed", "source"]) == {
        "mixseq",
        "Perturb-seq non-targeting control",
        "gse147405",
        "gse182694",
        "gse331320",
    }
    expt3, expt10 = "DMSO_6hr_expt3", "DMSO_24hr_expt10"
    spanning = cs.resolve_chemistry(table, "mixseq", "ACH-000035", [expt10, expt3])
    assert spanning.tolist() == ["10x 3' v3", "10x 3' v2"]
    hct116 = cs.resolve_chemistry(
        table, "Perturb-seq non-targeting control", "ACH-000971", [""]
    )
    assert hct116.tolist() == ["10x 5' v3"]


def test_state_fingerprints_change_with_inputs_masses_and_names():
    import pandas as pd

    names = ["L1|s|0", "L1|s|whole", "L2|s|0", "L2|s|whole"]
    inputs = pd.DataFrame(np.arange(8, dtype=np.float32).reshape(4, 2), index=names)
    masses = pd.DataFrame(
        {"model_id": ["L1", "L1", "L2", "L2"], "mass": [1.0, np.nan, 1.0, np.nan]},
        index=names,
    )
    base = cs.set_fingerprints(inputs, masses)
    assert set(base) == {"L1", "L2"} and base == cs.set_fingerprints(inputs, masses)
    moved = inputs.copy()
    moved.iloc[0, 0] += 1e-3
    assert cs.set_fingerprints(moved, masses)["L1"] != base["L1"]
    assert cs.set_fingerprints(moved, masses)["L2"] == base["L2"]
    weighed = masses.copy()
    weighed.iloc[2, 1] = 0.5
    assert cs.set_fingerprints(inputs, weighed)["L2"] != base["L2"]


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
