"""Six-arm response-model comparison on a synthetic four-anchor world.

The world has 4 anchors (one of them HCT116), 12 response genes, 8 control cells
per anchor, 6 expression genes, 5-wide Tx1 cells and 4-wide ESM2 vectors. STATE
is the small STATE-shaped ``LinearMockStateModel``; the released one-hot
vocabulary is a tiny ``pert_onehot_map.pt`` covering 9 of the 12 genes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments import response_comparison as rc
from src.model.state import LinearMockStateModel

ANCHORS = ("ACH-000971", "ACH-A1", "ACH-A2", "ACH-A3")
GENES = tuple(f"G{i}" for i in range(12))
COVERED = GENES[:9]
WIDTH, TX1, ESM2, CELLS, SENTENCE = 6, 5, 4, 8, 4
VOCABULARY = len(COVERED) + 1  # plus non-targeting


class Targets:
    """The prepared response cache's interface: ordered keys and one bag per key."""

    def __init__(self, bags: dict[tuple[str, str], np.ndarray]) -> None:
        self.keys = tuple(bags)
        self._bags = list(bags.values())

    def target_bag(self, index: int) -> np.ndarray:
        return self._bags[index]


def make_inputs(seed: int = 0, noisy_anchor: str | None = None) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    effects = {gene: rng.normal(0.0, 0.5, size=WIDTH) for gene in GENES}
    lines = {
        a: SimpleNamespace(
            basal_hvg=np.log1p(rng.poisson(2.0, size=(CELLS, WIDTH))).astype(
                np.float32
            ),
            controls_tx1=rng.normal(size=(CELLS, TX1)).astype(np.float32),
        )
        for a in ANCHORS
    }
    bags = {}
    for a in ANCHORS:
        for gene in GENES:
            base = np.log1p(rng.poisson(2.0, size=(5, WIDTH)))
            bags[(a, gene)] = (base + effects[gene]).astype(np.float32)
    if noisy_anchor is not None:
        noise = np.random.default_rng(99)
        for key in bags:
            if key[0] == noisy_anchor:
                bags[key] = noise.normal(3.0, 2.0, size=(5, WIDTH)).astype(np.float32)
    return SimpleNamespace(
        response_targets=Targets(bags),
        response_anchors=ANCHORS,
        lines=lines,
        esm2_symbols=GENES,
        esm2_vectors=rng.normal(size=(len(GENES), ESM2)).astype(np.float32),
    )


def make_ragged_inputs(seed: int = 1) -> SimpleNamespace:
    """Anchors of unequal size: basal bags of 6-9 cells (not all whole STATE
    sentences), target bags of 2-7 cells, and a different perturbed-gene set
    per anchor."""
    rng = np.random.default_rng(seed)
    effects = {gene: rng.normal(0.0, 0.5, size=WIDTH) for gene in GENES}
    sizes = dict(zip(ANCHORS, (6, 9, 7, 8), strict=True))
    perturbed = dict(zip(ANCHORS, (GENES[2:], GENES[:7], GENES, GENES[::2])))
    lines = {
        a: SimpleNamespace(
            basal_hvg=np.log1p(rng.poisson(2.0, size=(sizes[a], WIDTH))).astype(
                np.float32
            ),
            controls_tx1=rng.normal(size=(sizes[a], TX1)).astype(np.float32),
        )
        for a in ANCHORS
    }
    bags = {}
    for a in ANCHORS:
        for gene in perturbed[a]:
            cells = int(rng.integers(2, 8))
            base = np.log1p(rng.poisson(2.0, size=(cells, WIDTH)))
            bags[(a, gene)] = (base + effects[gene]).astype(np.float32)
    return SimpleNamespace(
        response_targets=Targets(bags),
        response_anchors=ANCHORS,
        lines=lines,
        esm2_symbols=GENES,
        esm2_vectors=rng.normal(size=(len(GENES), ESM2)).astype(np.float32),
    )


def fold_digest(fold: dict) -> list[float]:
    """Missing count, sum and position-weighted sum of each loss list, then the curve.

    The weighted sum moves when a loss lands on the wrong condition.
    """
    digest = []
    for key in ("loss", "shuffled_loss", "source_loss"):
        values = np.array([np.nan if v is None else v for v in fold[key]], dtype=float)
        weights = 1.0 + np.arange(len(values)) / len(values)
        digest += [
            float(np.isnan(values).sum()),
            float(np.nansum(values)),
            float(np.nansum(weights * values)),
        ]
    return digest + list(fold["curve"] or [])


def make_config(root: Path, *, epochs: int = 2) -> dict:
    vocabulary = {
        np.str_(gene): torch.nn.functional.one_hot(torch.tensor(i), VOCABULARY).float()
        for i, gene in enumerate((*COVERED, "non-targeting"))
    }
    torch.save(vocabulary, root / "pert_onehot_map.pt")
    return {
        "comparison": {
            "epochs": epochs,
            "hidden": 8,
            "learning_rate": 0.01,
            "state_learning_rate": 0.001,
            "batch_size": 6,
            "shuffles": 3,
            "bootstrap": 200,
        },
        "model": {"cell_sentence_len": SENTENCE, "esm2_adapter_hidden": 8},
        "paths": {"state_checkpoint": "mock-released", "state_model_dir": str(root)},
    }


def state_factory():
    torch.manual_seed(7)
    weights = LinearMockStateModel(WIDTH, WIDTH, VOCABULARY, SENTENCE).state_dict()

    def build():
        model = LinearMockStateModel(WIDTH, WIDTH, VOCABULARY, SENTENCE)
        model.load_state_dict(weights, strict=True)
        return model

    return build


def refuse_state():
    raise AssertionError("STATE must not be rebuilt for a finished fold")


def run(tmp_path: Path, *, epochs: int = 2, inputs=None, out: str = "out"):
    torch.set_num_threads(1)
    config = make_config(tmp_path, epochs=epochs)
    out_dir = tmp_path / out
    table = rc.run_comparison(
        config,
        out_dir,
        device="cpu",
        inputs=make_inputs() if inputs is None else inputs,
        state_factory=state_factory(),
    )
    return table.set_index(["arm", "fold"]), out_dir


def test_no_change_ratio_is_one(tmp_path):
    table, _ = run(tmp_path, epochs=1)
    rows = table.loc["no_change"]
    assert (rows.held_out_ratio == 1.0).all()
    assert (rows.source_ratio == 1.0).all()
    assert (rows.n_conditions == len(GENES)).all()


def test_zero_init_arms_start_at_no_change(tmp_path):
    table, out_dir = run(tmp_path, epochs=0)
    for arm in ("mlp_hvg", "mlp_tx1"):
        rows = table.loc[arm]
        assert (rows.held_out_ratio == 1.0).all()
        assert (rows.source_ratio == 1.0).all()
        assert (rows.identity_share == 0.0).all()
    curves = pd.read_csv(out_dir / "curves.csv")
    mlp = curves[curves.arm.str.startswith("mlp")]
    assert set(mlp.epoch) == {0} and (mlp.held_out_ratio == 1.0).all()


def test_folds_never_train_on_held_out_anchor(tmp_path):
    held = "ACH-A2"
    _, clean = run(tmp_path, out="clean")
    _, noisy = run(tmp_path, inputs=make_inputs(noisy_anchor=held), out="noisy")
    for arm in ("global_mean_effect", *rc.TRAINED_ARMS):
        a = json.loads((clean / "folds" / f"{arm}__{held}.json").read_text())
        b = json.loads((noisy / "folds" / f"{arm}__{held}.json").read_text())
        # Changing only the held-out anchor's targets leaves the fitted model, and
        # so its source-condition losses, bit-for-bit unchanged.
        assert a["source_loss"] == b["source_loss"]
        assert a["loss"] != b["loss"]


def test_released_arm_skips_uncovered_genes(tmp_path):
    table, out_dir = run(tmp_path, epochs=1)
    sanity = json.loads((out_dir / "sanity.json").read_text())["anchors"]
    for anchor in ANCHORS:
        assert sanity[anchor]["covered_genes"] == len(COVERED)
        assert sanity[anchor]["total_genes"] == len(GENES)
        assert np.isfinite(sanity[anchor]["ratio_to_no_change"])
        fold = json.loads(
            (out_dir / "folds" / f"released_state__{anchor}.json").read_text()
        )
        scored = {g for g, loss in zip(fold["genes"], fold["loss"]) if loss is not None}
        assert scored == set(COVERED)
    rows = table.loc["released_state"]
    assert (rows.n_covered == len(COVERED)).all()
    assert (rows.n_conditions == len(GENES)).all()
    assert np.isfinite(rows.held_out_ratio).all()
    assert np.isfinite(rows.identity_share).all()


def test_identity_share_zero_for_gene_blind_arm(tmp_path):
    table, _ = run(tmp_path, epochs=1)
    assert (table.loc["global_mean_effect"].identity_share == 0.0).all()
    assert (table.loc["no_change"].identity_share == 0.0).all()


def test_bootstrap_interval_contains_point(tmp_path):
    _, out_dir = run(tmp_path, epochs=1)
    verdicts = json.loads((out_dir / "verdicts.json").read_text())
    for variant in ("all_folds", "without_hct116"):
        for arm in rc.ARMS:
            pooled = verdicts["pooled"][variant][arm]
            low, high = pooled["interval"]
            assert low <= pooled["ratio"] <= high
        for verdict in verdicts["verdicts"][variant].values():
            low, high = verdict["interval"]
            assert low <= verdict["difference"] <= high
            separated = low > 0 or high < 0
            assert verdict["label"] == ("separated" if separated else "overlap")
    no_change = verdicts["pooled"]["all_folds"]["no_change"]
    assert no_change["ratio"] == 1.0 and no_change["interval"] == [1.0, 1.0]


def test_bootstrap_draws_each_gene_once_across_folds():
    folds = {
        arm: {
            a: {
                "genes": ["G1", "G2"],
                "loss": [1.0, 3.0] if arm != "no_change" else [1.0, 1.0],
            }
            for a in ("A", "B")
        }
        for arm in rc.ARMS
    }
    samples = rc._bootstrap(folds, ["A", "B"], {"all": ["A", "B"]}, 200)
    # Both folds share genes, so every resample weights them identically and
    # the two fold ratios agree: no resample can mix G1 in one fold with G2 in
    # the other.
    pooled = samples["all"]["mlp_hvg"]
    assert set(np.round(pooled, 6)) <= {1.0, 2.0, 3.0}


def test_pooled_with_and_without_hct116(tmp_path):
    table, out_dir = run(tmp_path, epochs=1)
    verdicts = json.loads((out_dir / "verdicts.json").read_text())
    assert verdicts["folds"]["without_hct116"] == list(ANCHORS[1:])
    for arm in rc.ARMS:
        ratios = table.loc[arm].held_out_ratio
        pooled = verdicts["pooled"]
        assert pooled["all_folds"][arm]["ratio"] == pytest.approx(ratios.mean())
        assert pooled["without_hct116"][arm]["ratio"] == pytest.approx(
            ratios.drop("ACH-000971").mean()
        )
    state = verdicts["pooled"]["all_folds"]["state_joint"]["ratio"]
    assert state != verdicts["pooled"]["without_hct116"]["state_joint"]["ratio"]


def test_skips_when_verdicts_exist_and_resumes_finished_folds(tmp_path):
    config = make_config(tmp_path, epochs=1)
    inputs = make_inputs()
    out_dir = tmp_path / "out"
    first = rc.run_comparison(
        config, out_dir, device="cpu", inputs=inputs, state_factory=state_factory()
    )
    again = rc.run_comparison(
        config, out_dir, device="cpu", inputs=inputs, state_factory=refuse_state
    )
    pd.testing.assert_frame_equal(first, again)

    removed = out_dir / "folds" / "mlp_hvg__ACH-A1.json"
    original = removed.read_text()
    removed.unlink()
    (out_dir / "verdicts.json").unlink()
    kept = {p: p.stat().st_mtime_ns for p in (out_dir / "folds").iterdir()}
    resumed = rc.run_comparison(
        config, out_dir, device="cpu", inputs=inputs, state_factory=refuse_state
    )
    assert removed.read_text() == original
    assert {p: p.stat().st_mtime_ns for p in kept} == kept
    pd.testing.assert_frame_equal(first, resumed)


def test_outputs_written(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    train = rc._train

    def train_after_sanity(*args, **kwargs):
        assert (out_dir / "sanity.json").is_file()
        assert not (out_dir / "comparison.csv").exists()
        return train(*args, **kwargs)

    monkeypatch.setattr(rc, "_train", train_after_sanity)
    table, _ = run(tmp_path, epochs=1)
    assert len(table) == len(rc.ARMS) * len(ANCHORS)
    assert list(table.reset_index().columns) == [
        "arm",
        "fold",
        "held_out_ratio",
        "identity_share",
        "source_ratio",
        "n_conditions",
        "n_covered",
    ]
    times = [
        (out_dir / name).stat().st_mtime_ns
        for name in ("sanity.json", "comparison.csv", "curves.csv", "verdicts.json")
    ]
    assert times == sorted(times)
    curves = pd.read_csv(out_dir / "curves.csv")
    assert set(curves.arm) == set(rc.TRAINED_ARMS)
    assert set(curves.epoch) == {0, 1}
    for arm in rc.ARMS:
        for anchor in ANCHORS:
            assert (out_dir / "folds" / f"{arm}__{anchor}.json").is_file()


WORLDS = {"default": (make_inputs, 2), "ragged": (make_ragged_inputs, 3)}


@pytest.mark.parametrize("world", sorted(WORLDS))
def test_matches_reference_fold_files(tmp_path, world):
    """Fold files agree with those of the one-condition-at-a-time implementation
    (commit f3ece57), recorded as ``REFERENCE`` digests."""
    make, epochs = WORLDS[world]
    inputs = make()
    _, out_dir = run(tmp_path, epochs=epochs, inputs=inputs)
    paths = sorted((out_dir / "folds").glob("*.json"))
    assert {f"{world}/{p.stem}" for p in paths} == {
        key for key in REFERENCE if key.startswith(f"{world}/")
    }
    keys = inputs.response_targets.keys
    for path in paths:
        fold = json.loads(path.read_text())
        assert fold["genes"] == [g for m, g in keys if m == fold["anchor"]]
        np.testing.assert_allclose(
            fold_digest(fold), REFERENCE[f"{world}/{path.stem}"], rtol=1e-5, atol=1e-6
        )


def test_jobs_in_any_order_give_identical_fold_files(tmp_path):
    config = make_config(tmp_path, epochs=1)
    inputs = make_inputs()
    _, together = run(tmp_path, epochs=1, inputs=inputs, out="together")

    split = tmp_path / "split"
    rc.run_untrained(
        config, split, device="cpu", inputs=inputs, state_factory=state_factory()
    )
    jobs = rc.pending_trained_jobs(config, split, inputs=inputs)
    assert len(jobs) == len(rc.TRAINED_ARMS) * len(ANCHORS)
    for arm, anchor in reversed(jobs):
        # Disturb the global generator: a job must not depend on what ran before.
        torch.manual_seed(1234)
        torch.rand(10)
        rc.run_trained_job(
            config,
            split,
            arm,
            anchor,
            device="cpu",
            inputs=inputs,
            state_factory=state_factory(),
        )
    rc.summarise(config, split, inputs=inputs)

    names = sorted(p.name for p in (together / "folds").iterdir())
    assert names == sorted(p.name for p in (split / "folds").iterdir())
    for name in names:
        assert json.loads((together / "folds" / name).read_text()) == json.loads(
            (split / "folds" / name).read_text()
        )
    for name in ("sanity.json", "comparison.csv", "curves.csv", "verdicts.json"):
        assert (together / name).read_text() == (split / name).read_text()


def test_pending_trained_jobs_skips_finished(tmp_path):
    torch.set_num_threads(1)
    config = make_config(tmp_path, epochs=1)
    inputs = make_inputs()
    out_dir = tmp_path / "out"
    every = [(arm, a) for a in ANCHORS for arm in rc.TRAINED_ARMS]
    assert rc.pending_trained_jobs(config, out_dir, inputs=inputs) == every

    def job(arm, anchor, factory=refuse_state):
        rc.run_trained_job(
            config,
            out_dir,
            arm,
            anchor,
            device="cpu",
            inputs=inputs,
            state_factory=factory,
        )

    with pytest.raises(FileNotFoundError, match="no_change__ACH-A2.json"):
        job("mlp_tx1", "ACH-A2")
    with pytest.raises(ValueError, match="not a trained arm"):
        job("no_change", "ACH-A2")
    with pytest.raises(ValueError, match="not a response anchor"):
        job("mlp_tx1", "ACH-UNKNOWN")

    rc.run_untrained(
        config, out_dir, device="cpu", inputs=inputs, state_factory=state_factory()
    )
    assert rc.pending_trained_jobs(config, out_dir, inputs=inputs) == every
    job("mlp_tx1", "ACH-A2")
    job("state_joint", ANCHORS[0], state_factory())
    finished = [("mlp_tx1", "ACH-A2"), ("state_joint", ANCHORS[0])]
    assert rc.pending_trained_jobs(config, out_dir, inputs=inputs) == [
        j for j in every if j not in finished
    ]
    written = out_dir / "folds" / "state_joint__ACH-000971.json"
    stamp = written.stat().st_mtime_ns
    job("state_joint", ANCHORS[0])  # finished: STATE is not rebuilt
    assert written.stat().st_mtime_ns == stamp


def test_summarise_raises_on_missing_fold(tmp_path):
    _, out_dir = run(tmp_path, epochs=0)
    (out_dir / "verdicts.json").unlink()
    (out_dir / "folds" / "state_joint__ACH-A3.json").unlink()
    (out_dir / "folds" / "no_change__ACH-A1.json").unlink()
    config = make_config(tmp_path, epochs=0)
    with pytest.raises(FileNotFoundError) as error:
        rc.summarise(config, out_dir, inputs=make_inputs())
    assert "state_joint__ACH-A3.json" in str(error.value)
    assert "no_change__ACH-A1.json" in str(error.value)
    assert not (out_dir / "verdicts.json").exists()


def test_cli_modes(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = {**make_config(tmp_path, epochs=1), "prepared_root": str(tmp_path)}
    inputs = make_inputs()
    build = state_factory()
    monkeypatch.setattr(rc, "load_config", lambda path: config)
    monkeypatch.setattr(rc, "load_inputs", lambda config: inputs)
    monkeypatch.setattr(
        rc, "read_manifest", lambda root: {"response_anchors": list(ANCHORS)}
    )
    monkeypatch.setattr(rc, "load_released_state", lambda path, cell_set_len: build())

    def cli(out: str, *mode: str) -> Path:
        out_dir = tmp_path / out
        rc.main(
            ["--config", "c.yaml", "--out-dir", str(out_dir), "--device", "cpu", *mode]
        )
        return out_dir

    split = cli("split", "--untrained")
    assert (split / "sanity.json").is_file()
    assert sorted(p.stem.split("__")[0] for p in (split / "folds").iterdir()) == sorted(
        rc.UNTRAINED_ARMS * len(ANCHORS)
    )
    jobs = rc.pending_trained_jobs(config, split)
    assert len(jobs) == len(rc.TRAINED_ARMS) * len(ANCHORS)
    for arm, anchor in jobs:
        cli("split", "--job", arm, anchor)
    assert rc.pending_trained_jobs(config, split) == []
    assert not (split / "verdicts.json").exists()
    cli("split", "--summarise")

    whole = cli("whole")
    for name in ("sanity.json", "comparison.csv", "curves.csv", "verdicts.json"):
        assert (whole / name).read_text() == (split / name).read_text()
    for path in (whole / "folds").iterdir():
        assert path.read_text() == (split / "folds" / path.name).read_text()
    with pytest.raises(SystemExit):
        cli("split", "--untrained", "--summarise")


# Digests (``fold_digest``) of the fold files written by the one-condition-at-a-time
# implementation at commit f3ece57 on the worlds of ``WORLDS``, 9 significant digits.
REFERENCE = json.loads(
    """{
"default/global_mean_effect__ACH-000971":
    [0.0, 20.9777808, 31.3576577, 0.0, 20.9777808, 31.3576577, 0.0, 50.2492577,
    74.2207836],
"default/global_mean_effect__ACH-A1":
    [0.0, 20.9962887, 31.2064916, 0.0, 20.9962887, 31.2064916, 0.0, 50.8671352,
    74.8699276],
"default/global_mean_effect__ACH-A2":
    [0.0, 17.4079912, 25.9868977, 0.0, 17.4079912, 25.9868977, 0.0, 53.3362665,
    78.5028601],
"default/global_mean_effect__ACH-A3":
    [0.0, 16.0695979, 23.9726845, 0.0, 16.0695979, 23.9726845, 0.0, 54.9579782,
    81.6092784],
"default/mlp_hvg__ACH-000971":
    [0.0, 22.0045122, 32.8078221, 0.0, 22.2218477, 32.9824343, 0.0, 49.6023794,
    72.885216, 1.0, 1.0273172, 1.16698288],
"default/mlp_hvg__ACH-A1":
    [0.0, 21.3482231, 31.7700504, 0.0, 21.5089431, 31.9620462, 0.0, 50.3359023,
    73.9217494, 1.0, 0.974607827, 0.954880304],
"default/mlp_hvg__ACH-A2":
    [0.0, 17.2820044, 25.7655779, 0.0, 17.398118, 25.8925737, 0.0, 53.1499992,
    77.9045975, 1.0, 0.949590189, 0.946528438],
"default/mlp_hvg__ACH-A3":
    [0.0, 15.7069554, 23.4983193, 0.0, 15.9249759, 23.7948528, 0.0, 54.3707065,
    80.8952796, 1.0, 0.947625204, 0.916662697],
"default/mlp_tx1__ACH-000971":
    [0.0, 19.0185436, 28.7614928, 0.0, 19.2386683, 29.0266784, 0.0, 50.7866321,
    74.7098112, 1.0, 0.996979746, 1.00862562],
"default/mlp_tx1__ACH-A1":
    [0.0, 21.1828593, 31.791423, 0.0, 21.4044014, 31.9725119, 0.0, 49.5183839,
    73.2164057, 1.0, 0.972676382, 0.947483781],
"default/mlp_tx1__ACH-A2":
    [0.0, 17.276065, 25.9979042, 0.0, 17.558208, 26.3482994, 0.0, 52.2307096,
    77.2989779, 1.0, 0.968837159, 0.946203141],
"default/mlp_tx1__ACH-A3":
    [0.0, 16.0949801, 24.2658103, 0.0, 16.4234381, 24.6337202, 0.0, 54.6273004,
    81.8524599, 1.0, 0.970052272, 0.939307936],
"default/no_change__ACH-000971":
    [0.0, 18.8558998, 28.7375109, 0.0, 18.8558998, 28.7375109, 0.0, 57.7502015,
    85.0470798],
"default/no_change__ACH-A1":
    [0.0, 22.3569624, 33.7492016, 0.0, 22.3569624, 33.7492016, 0.0, 54.2491388,
    81.0424745],
"default/no_change__ACH-A2":
    [0.0, 18.2583044, 27.4847646, 0.0, 18.2583044, 27.4847646, 0.0, 58.3477969,
    87.2292782],
"default/no_change__ACH-A3":
    [0.0, 17.1349346, 25.8786967, 0.0, 17.1349346, 25.8786967, 0.0, 59.4711667,
    89.2624606],
"default/released_state__ACH-000971":
    [3.0, 51.5516787, 70.0175884, 3.0, 51.549358, 70.1342147, 9.0, 149.512694,
    218.353119],
"default/released_state__ACH-A1":
    [3.0, 47.8230646, 65.8340574, 3.0, 47.9075305, 65.9683988, 9.0, 153.241308,
    222.233372],
"default/released_state__ACH-A2":
    [3.0, 52.1176615, 71.1228384, 3.0, 52.0760313, 71.1157114, 9.0, 148.946712,
    216.175848],
"default/released_state__ACH-A3":
    [3.0, 49.5719683, 67.8154751, 3.0, 49.5839709, 67.8806206, 9.0, 151.492405,
    220.67256],
"default/state_joint__ACH-000971":
    [0.0, 59.4493514, 88.4079657, 0.0, 59.2125541, 87.9411988, 0.0, 172.180892,
    258.258409, 3.67986281, 3.41545394, 3.15282495],
"default/state_joint__ACH-A1":
    [0.0, 56.4391067, 85.3243047, 0.0, 56.1596971, 84.6852286, 0.0, 175.552389,
    261.771112, 2.92409094, 2.72658437, 2.52445326],
"default/state_joint__ACH-A2":
    [0.0, 59.7189109, 88.8956311, 0.0, 59.4034507, 88.3212989, 0.0, 172.036031,
    256.931327, 3.79092173, 3.53319211, 3.27078077],
"default/state_joint__ACH-A3":
    [0.0, 56.5054606, 84.3980777, 0.0, 56.2999481, 83.8445992, 0.0, 175.21161,
    262.6412, 3.84616859, 3.57268296, 3.29767588],
"ragged/global_mean_effect__ACH-000971":
    [0.0, 14.91465, 21.6268595, 0.0, 14.91465, 21.6268595, 0.0, 37.8046992,
    54.31035],
"ragged/global_mean_effect__ACH-A1":
    [0.0, 15.8813672, 22.4059851, 0.0, 15.8813672, 22.4059851, 0.0, 37.3288199,
    54.8414045],
"ragged/global_mean_effect__ACH-A2":
    [0.0, 18.6970359, 27.6793832, 0.0, 18.6970359, 27.6793832, 0.0, 34.7256646,
    51.211498],
"ragged/global_mean_effect__ACH-A3":
    [0.0, 7.58359867, 10.520041, 0.0, 7.58359867, 10.520041, 0.0, 44.8735127,
    66.4809575],
"ragged/mlp_hvg__ACH-000971":
    [0.0, 14.4877514, 21.028158, 0.0, 14.9633202, 21.6552827, 0.0, 36.1446093,
    51.8854664, 1.0, 1.00765822, 1.01052463, 0.982297903],
"ragged/mlp_hvg__ACH-A1":
    [0.0, 15.4886736, 21.8650278, 0.0, 15.6477455, 22.1264045, 0.0, 35.7219473,
    52.2243047, 1.0, 1.07103151, 1.11520202, 1.09859855],
"ragged/mlp_hvg__ACH-A2":
    [0.0, 18.5701067, 27.3804481, 0.0, 19.6950571, 29.2143181, 0.0, 32.9537846,
    48.3898862, 1.0, 1.04164921, 1.10820025, 1.09754891],
"ragged/mlp_hvg__ACH-A3":
    [0.0, 7.06982273, 9.84406048, 0.0, 7.51949237, 10.5079226, 0.0, 41.8647664,
    62.1848655, 1.0, 1.00270048, 0.960684861, 0.913299478],
"ragged/mlp_tx1__ACH-000971":
    [0.0, 14.6562595, 21.2302069, 0.0, 15.1826415, 21.9422536, 0.0, 34.9364284,
    50.1579147, 1.0, 0.989383984, 0.992576349, 0.993723083],
"ragged/mlp_tx1__ACH-A1":
    [0.0, 15.9273318, 22.5307494, 0.0, 16.4920706, 23.3552305, 0.0, 32.9430295,
    48.0348455, 1.0, 1.04909886, 1.10052789, 1.12971221],
"ragged/mlp_tx1__ACH-A2":
    [0.0, 17.8047231, 26.3174215, 0.0, 18.6430374, 27.7015919, 0.0, 31.8723429,
    46.7304768, 1.0, 1.01362601, 1.0366418, 1.05231246],
"ragged/mlp_tx1__ACH-A3":
    [0.0, 7.32592744, 10.1672454, 0.0, 7.70948875, 10.715279, 0.0, 40.8037915,
    60.4074308, 1.0, 1.00198312, 0.973607987, 0.9463838],
"ragged/no_change__ACH-000971":
    [0.0, 14.7488367, 21.2666473, 0.0, 14.7488367, 21.2666473, 0.0, 38.7591597,
    55.6518007],
"ragged/no_change__ACH-A1":
    [0.0, 14.0985745, 20.3628257, 0.0, 14.0985745, 20.3628257, 0.0, 39.4094219,
    57.8960693],
"ragged/no_change__ACH-A2":
    [0.0, 16.9196165, 24.8652072, 0.0, 16.9196165, 24.8652072, 0.0, 36.58838,
    53.9454901],
"ragged/no_change__ACH-A3":
    [0.0, 7.74096876, 10.674878, 0.0, 7.74096876, 10.674878, 0.0, 45.7670277,
    67.5944131],
"ragged/released_state__ACH-000971":
    [3.0, 36.5342145, 47.5907163, 3.0, 36.3577143, 47.4413231, 4.0, 111.620922,
    157.077461],
"ragged/released_state__ACH-A1":
    [0.0, 40.4321346, 56.7925675, 0.0, 40.4712434, 56.9459841, 7.0, 107.723002,
    155.79521],
"ragged/released_state__ACH-A2":
    [3.0, 47.0083323, 62.4172591, 3.0, 47.0483663, 62.582996, 4.0, 101.146804,
    148.493477],
"ragged/released_state__ACH-A3":
    [1.0, 24.180455, 32.2631799, 1.0, 23.9696145, 32.0610484, 6.0, 123.974681,
    179.61117],
"ragged/state_joint__ACH-000971":
    [0.0, 39.7428579, 57.3391196, 0.0, 39.8460597, 57.4741468, 0.0, 102.321739,
    148.034316, 3.45347699, 3.21023486, 2.96954711, 2.69464357],
"ragged/state_joint__ACH-A1":
    [0.0, 33.6248803, 47.3434322, 0.0, 33.5857878, 47.1662871, 0.0, 107.374644,
    157.939176, 2.88297949, 2.72290561, 2.56609515, 2.3849844],
"ragged/state_joint__ACH-A2":
    [0.0, 49.8304625, 71.714371, 0.0, 49.9328369, 71.74008, 0.0, 99.4077692,
    146.373994, 3.58636306, 3.36883039, 3.16386391, 2.94512955],
"ragged/state_joint__ACH-A3":
    [0.0, 21.9765067, 31.0589528, 0.0, 22.1528451, 31.2225151, 0.0, 119.282179,
    176.019817, 3.69288511, 3.41332598, 3.14157874, 2.83898662]
}"""
)
