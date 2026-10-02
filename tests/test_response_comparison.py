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
