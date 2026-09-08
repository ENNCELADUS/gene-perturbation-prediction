import json
import numpy as np
import pandas as pd
import pytest
import torch


def test_bias_decomposition_splits_shared_from_condition_specific():
    from src.eval.p1c_tier0 import bias_decomposition

    observed = np.zeros((4, 3))
    predicted = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 2.0, 0], [1.0, -2.0, 0]])
    result = bias_decomposition(predicted, observed)
    # mean_g ||e||^2 = (1 + 1 + 5 + 5)/4 = 3 ; ||ē||^2 = 1 ; residual = 2
    assert result["total_mse"] == pytest.approx(3.0 / 3)  # per-coordinate mean
    assert result["shared_bias"] == pytest.approx(1.0 / 3)
    assert result["condition_specific"] == pytest.approx(2.0 / 3)
    assert result["shared_bias_fraction"] == pytest.approx(1 / 3)
    assert result["n"] == 4


def test_decompose_effects_reads_npz_by_method_and_anchor(tmp_path):
    from src.eval.p1c_tier0 import decompose_effects

    keys = np.array([("model", "A", "G"), ("model", "A", "H"), ("no_change", "A", "G")])
    predicted = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    observed = np.zeros((3, 2))
    np.savez(
        tmp_path / "effects.npz", keys=keys, predicted=predicted, observed=observed
    )
    frame = decompose_effects(tmp_path / "effects.npz", method="model")
    assert frame.model_id.tolist() == ["A"] and frame.n.tolist() == [2]
    assert frame.shared_bias_fraction.iloc[0] == pytest.approx(1.0)


def test_matched_coverage_restricts_references_to_perturbation_mean_support():
    from src.eval.p1c_tier0 import matched_coverage

    rows = []
    for gene, covered in (("G", True), ("H", False)):
        rows.append(
            dict(
                role="val",
                model_id="A",
                panel="all",
                gene=gene,
                method="no_change",
                response_loss=1.0,
            )
        )
        rows.append(
            dict(
                role="val",
                model_id="A",
                panel="all",
                gene=gene,
                method="global_mean",
                response_loss=2.0,
            )
        )
        if covered:
            rows.append(
                dict(
                    role="val",
                    model_id="A",
                    panel="all",
                    gene=gene,
                    method="perturbation_mean",
                    response_loss=3.0,
                )
            )
    frame = matched_coverage(pd.DataFrame(rows))
    row = frame[(frame.model_id == "A") & (frame.reference == "no_change")].iloc[0]
    assert row.covered == 1 and row.total == 2
    assert row.perturbation_mean_loss == 3.0 and row.reference_loss_on_covered == 1.0


def test_cross_context_pearson_is_undefined_for_constant_effect_references():
    from src.eval.p1b import cross_context

    bundle = {
        "keys": [("a", "G"), ("b", "G"), ("j", "G")],
        "splits": {"train": [], "val": [0, 1], "external": [2]},
        "anchors": ["a", "b"],
        "external": "j",
    }
    # Identical predicted effects in both anchors -> zero difference up to float noise.
    noise = np.array([1e-7, -1e-7, 0.0])
    effects = {
        ("global_mean", "a", "G"): (
            np.array([1.0, 2.0, 3.0]) + noise,
            np.array([1.0, 0.0, 0.0]),
        ),
        ("global_mean", "b", "G"): (
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
    }
    frame = pd.DataFrame({"method": ["global_mean"]})
    result = cross_context(
        frame,
        effects,
        {**bundle, "splits": {"train": [], "val": [0, 1], "external": [2]}},
    )
    assert len(result) == 1 and np.isnan(result.effect_difference_pearson.iloc[0])


def _build_tier0_fixture(tmp_path):
    from src.data.gene_splits import sha256_file

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    response_cache_dir = tmp_path / "cache"
    (response_cache_dir / "response_targets").mkdir(parents=True)
    pd.DataFrame({"model_id": ["a", "a", "b"], "n_cells": [100, 120, 90]}).to_parquet(
        response_cache_dir / "response_targets" / "metadata.parquet"
    )

    bundle = {
        "keys": [("a", "G"), ("b", "G"), ("j", "G")],
        "splits": {"train": [], "val": [0, 1], "external": [2]},
        "anchors": ["a", "b"],
        "external": "j",
        "controls": {
            "a": {"hvg": np.zeros((5, 3))},
            "b": {"hvg": np.zeros((5, 3))},
        },
        "response_cache": str(response_cache_dir),
    }
    torch.save(bundle, prepared / "bundle.pt")
    manifest = {
        "bundle_sha256": sha256_file(prepared / "bundle.pt"),
        "source_missing_genes": {"a": [], "b": []},
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest))

    runs = tmp_path / "runs"
    export_dir = runs / "evaluation" / "B-init" / "internal"
    export_dir.mkdir(parents=True)
    methods = ("model", "no_change", "global_mean", "perturbation_mean")
    conditions = pd.DataFrame(
        [
            dict(
                role="val",
                model_id=anchor,
                panel="all",
                gene="G",
                method=method,
                response_loss=1.0,
            )
            for anchor in ("a", "b")
            for method in methods
        ]
    )
    conditions.to_parquet(export_dir / "conditions.parquet", index=False)
    keys = np.array(
        [(method, anchor, "G") for anchor in ("a", "b") for method in methods]
    )
    predicted = np.tile(np.array([1.0, 0.0]), (len(keys), 1))
    observed = np.zeros((len(keys), 2))
    np.savez(
        export_dir / "effects.npz", keys=keys, predicted=predicted, observed=observed
    )
    (export_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "state": "B-init",
                "bundle": manifest["bundle_sha256"],
            }
        )
    )
    return prepared, runs, export_dir, manifest


def test_run_tier0_verifies_bundle_and_evaluation_identity_and_writes_outputs(tmp_path):
    from src.eval.p1c_tier0 import run_tier0

    prepared, runs, export_dir, manifest = _build_tier0_fixture(tmp_path)
    original_bundle_bytes = (prepared / "bundle.pt").read_bytes()

    out_dir = tmp_path / "out"
    result = run_tier0(runs, prepared, out_dir)
    assert result == out_dir
    for name in (
        "bias_decomposition.csv",
        "matched_coverage.csv",
        "anchor_audit.csv",
        "cross_context_guarded.csv",
        "tier0.md",
    ):
        assert (out_dir / name).exists()

    # Tampering with the prepared bundle after preparation must be caught.
    (prepared / "bundle.pt").write_bytes(original_bundle_bytes + b"\x00")
    with pytest.raises(ValueError, match="identity changed"):
        run_tier0(runs, prepared, out_dir)
    (prepared / "bundle.pt").write_bytes(original_bundle_bytes)

    # An export recorded against a different bundle must also be rejected.
    (export_dir / "evaluation.json").write_text(
        json.dumps({"status": "completed", "state": "B-init", "bundle": "stale-hash"})
    )
    with pytest.raises(ValueError):
        run_tier0(runs, prepared, out_dir)


def test_fold_membership_rotates_one_anchor_out():
    from src.data.p1c import FOLDS, fold_membership, ALL_ANCHORS

    for fold, held in FOLDS.items():
        sources, external = fold_membership(fold)
        assert external == held and held not in sources
        assert tuple(a for a in ALL_ANCHORS if a != held) == sources
    with pytest.raises(KeyError):
        fold_membership("mcf7")


def test_response_view_input_layouts_concatenate_hvg_then_tx1():
    from types import SimpleNamespace
    from src.data.p1b import build_snapshot, ResponseView
    from src.data.response_cache import ResponseTargetsCache

    keys = [(a, g) for a in ("a", "b", "c", "j") for g in ("G", "H")]
    cache = ResponseTargetsCache(
        tuple(a for a, g in keys),
        tuple(g for a, g in keys),
        np.ones((16, 3), dtype=np.float32),
        np.arange(0, 17, 2),
        pd.DataFrame(),
    )
    inputs = SimpleNamespace(
        response_targets=cache,
        response_holdout=frozenset((a, "H") for a in ("a", "b", "c")),
        hvg_order=("G", "H", "I"),
        lines={
            a: SimpleNamespace(
                controls_tx1=np.full((2, 5), 2.0, dtype=np.float32),
                basal_hvg=np.full((2, 3), 1.0, dtype=np.float32),
            )
            for a in ("a", "b", "c", "j")
        },
    )
    bundle = build_snapshot(
        inputs, [0, 1], {"G"}, anchors=("a", "b", "c"), external="j"
    )
    for layout, width, first in (("tx1", 5, 2.0), ("hvg", 3, 1.0), ("hvg_tx1", 8, 1.0)):
        batch = ResponseView(bundle, cache, input_layout=layout).batch([0])
        assert batch.controls_tx1[0].shape == (2, width)
        assert float(batch.controls_tx1[0][0, 0]) == first
        assert batch.control_hvg[0].shape == (2, 3)
    with pytest.raises(ValueError):
        ResponseView(bundle, cache, input_layout="bogus")


def _bundle_matching_p1b_expectations(anchors, external):
    """Synthetic bundle whose check_membership() result equals P1B_EXPECTATIONS."""
    train_counts = (16399, 5481, 5481)  # sums to 27361; largest_pool == 16399
    keys = []
    for anchor, n in zip(anchors, train_counts):
        keys.extend((anchor, f"g{i}") for i in range(n))
    keys.extend((anchors[0], f"v{i}") for i in range(3047))
    keys.extend((external, f"e{i}") for i in range(2377))
    n_train = sum(train_counts)
    splits = {
        "train": list(range(0, n_train)),
        "val": list(range(n_train, n_train + 3047)),
        "external": list(range(n_train + 3047, n_train + 3047 + 2377)),
    }
    panels = {
        f"external/{external}/seen": {"indices": list(range(2373))},
        f"external/{external}/unseen": {"indices": list(range(4))},
        f"external/{external}/native_common": {"indices": list(range(2006))},
        f"external/{external}/native_all": {"indices": list(range(2009))},
    }
    return {"keys": keys, "splits": splits, "panels": panels}


def test_check_membership_none_expectations_checks_total_and_anchor_set():
    from src.experiments.p1b_preparation import check_membership
    from src.data.p1b import SOURCE_ANCHORS, EXTERNAL_ANCHOR

    def bundle_with(total):
        return {
            "keys": [(SOURCE_ANCHORS[0], "g")] * total,
            "splits": {
                "train": list(range(total - 2)),
                "val": [total - 2],
                "external": [total - 1],
            },
            "panels": {},
        }

    good = bundle_with(32785)
    result = check_membership(good, SOURCE_ANCHORS, EXTERNAL_ANCHOR, None)
    assert result["counts"] == {"train": 32783, "val": 1, "external": 1}
    assert result["external_panels"] == {
        "seen": 0,
        "unseen": 0,
        "native_common": 0,
        "native_all": 0,
    }

    with pytest.raises(ValueError, match="unexpected total condition count"):
        check_membership(bundle_with(100), SOURCE_ANCHORS, EXTERNAL_ANCHOR, None)

    with pytest.raises(ValueError, match="approved"):
        check_membership(good, ("x", "y", "z"), "w", None)

    empty_external = bundle_with(32785)
    empty_external["splits"]["train"] = list(range(32785))
    empty_external["splits"]["val"] = []
    empty_external["splits"]["external"] = []
    with pytest.raises(ValueError, match="external"):
        check_membership(empty_external, SOURCE_ANCHORS, EXTERNAL_ANCHOR, None)


def test_check_membership_matches_expectations_exactly_or_raises_on_first_mismatch():
    from src.experiments.p1b_preparation import check_membership, P1B_EXPECTATIONS
    from src.data.p1b import SOURCE_ANCHORS, EXTERNAL_ANCHOR

    tiny = {
        "keys": [(SOURCE_ANCHORS[0], "g")] * 5,
        "splits": {"train": [0, 1, 2], "val": [3], "external": [4]},
        "panels": {},
    }
    with pytest.raises(ValueError, match="counts"):
        check_membership(tiny, SOURCE_ANCHORS, EXTERNAL_ANCHOR, P1B_EXPECTATIONS)

    matching = _bundle_matching_p1b_expectations(SOURCE_ANCHORS, EXTERNAL_ANCHOR)
    result = check_membership(
        matching, SOURCE_ANCHORS, EXTERNAL_ANCHOR, P1B_EXPECTATIONS
    )
    assert result == P1B_EXPECTATIONS


def test_prepare_fold_writes_fold_json_and_checks_reference_coordinates(
    tmp_path, monkeypatch
):
    from pathlib import Path
    import src.experiments.p1c_preparation as p1c_prep
    from src.data.p1c import fold_membership
    from src.experiments.p1b_preparation import P1B_EXPECTATIONS

    calls = []

    def fake_prepare_bundle(checkpoint, directory, *, anchors, external, expectations):
        calls.append((checkpoint, anchors, external, expectations))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps({"coordinates": [0, 1, 2]}))

    monkeypatch.setattr(p1c_prep, "prepare_bundle", fake_prepare_bundle)

    jurkat_dir = tmp_path / "jurkat"
    p1c_prep.prepare_fold("ckpt", "jurkat", jurkat_dir)
    sources, external = fold_membership("jurkat")
    assert json.loads((jurkat_dir / "fold.json").read_text()) == {
        "fold": "jurkat",
        "external": external,
        "sources": list(sources),
    }
    assert calls[0] == ("ckpt", sources, external, P1B_EXPECTATIONS)

    k562_dir = tmp_path / "k562"
    p1c_prep.prepare_fold("ckpt", "k562", k562_dir)
    assert calls[1][3] is None

    # A reference manifest with matching coordinates is accepted.
    p1c_prep.prepare_fold(
        "ckpt",
        "hepg2",
        tmp_path / "hepg2",
        reference_manifest=jurkat_dir / "manifest.json",
    )

    # A reference manifest with different coordinates is rejected.
    def fake_prepare_bundle_mismatch(
        checkpoint, directory, *, anchors, external, expectations
    ):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps({"coordinates": [9, 9, 9]}))

    monkeypatch.setattr(p1c_prep, "prepare_bundle", fake_prepare_bundle_mismatch)
    with pytest.raises(ValueError, match="fold coordinates differ"):
        p1c_prep.prepare_fold(
            "ckpt",
            "hct116",
            tmp_path / "hct116",
            reference_manifest=jurkat_dir / "manifest.json",
        )
