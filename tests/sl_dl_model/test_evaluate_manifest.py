# tests/sl_dl_model/test_evaluate_manifest.py
"""Tests for manifest completeness and empty-run guards in evaluate.py."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sl_dl_model.config import SLDLConfig
from sl_dl_model.evaluate import ZeroEmbeddingProducer, _load_state_dl_caches, run_cv


def _toy_frame() -> pd.DataFrame:
    """Reusable toy frame from test_evaluate_parity.py pattern."""
    rows = []
    genes = [f"G{i}" for i in range(8)]
    rng = np.random.default_rng(0)
    eff = {g: float(rng.normal()) for g in genes}
    pid = 0
    for split in ("CV2",):
        for fold in (0, 1):
            for role in ("train", "test"):
                for i in range(len(genes)):
                    for j in range(i + 1, len(genes)):
                        a, b = genes[i], genes[j]
                        rows.append(
                            {
                                "pair_id": f"p{pid}",
                                "fold_id": fold,
                                "split_type": split,
                                "split_role": role,
                                "sl_label": (i + j) % 2,
                                "gene_a_symbol": a,
                                "gene_b_symbol": b,
                                "gene_a_k562_gene_effect": eff[a],
                                "gene_b_k562_gene_effect": eff[b],
                            }
                        )
                        pid += 1
    return pd.DataFrame(rows)


# FIX 1 RED — manifest missing required fields
def test_manifest_includes_all_required_fields(tmp_path: Path) -> None:
    """Manifest must record candidate_gene_count, pooling, loss_weights.

    Also checks coverage_flag_included and gwps_coverage_gene_count.
    """
    csv = tmp_path / "toy.csv"
    frame = _toy_frame()
    frame.to_csv(csv, index=False)
    cfg = SLDLConfig(
        input_csv=csv,
        output_dir=tmp_path / "run",
        split_types=("CV2",),
        folds=(0, 1),
        include_coverage_flag=True,
    )
    run_cv(cfg, ZeroEmbeddingProducer())

    manifest_path = tmp_path / "run" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())

    # Existing fields (keep these)
    assert "input_csv" in manifest
    assert "split_types" in manifest
    assert "folds" in manifest
    assert "ranking_k" in manifest
    assert "seed" in manifest
    assert "embedding_method" in manifest
    assert "fallback_strategy" in manifest
    assert "include_coverage_flag" in manifest
    assert "esm2_model" in manifest
    assert "state_checkpoint" in manifest

    # NEW required fields (FIX 1)
    assert "candidate_gene_count" in manifest
    unique_genes = set(frame["gene_a_symbol"]) | set(frame["gene_b_symbol"])
    assert manifest["candidate_gene_count"] == len(unique_genes)
    assert manifest["candidate_gene_count"] == 8  # toy frame has G0..G7

    assert "pooling" in manifest
    assert manifest["pooling"] == "mean_std"

    assert "loss_weights" in manifest
    assert isinstance(manifest["loss_weights"], dict)
    assert "lambda_sl" in manifest["loss_weights"]
    assert manifest["loss_weights"]["lambda_sl"] == 1.0

    assert "coverage_flag_included" in manifest
    assert manifest["coverage_flag_included"] is True

    assert "gwps_coverage_gene_count" in manifest
    # ZeroEmbeddingProducer path should set this to None
    assert manifest["gwps_coverage_gene_count"] is None


# FIX 2 RED — empty metric rows produce empty CSV silently
def test_empty_metric_rows_raise_error(tmp_path: Path) -> None:
    """When split_types filter yields no folds, run_cv must raise RuntimeError."""
    csv = tmp_path / "toy.csv"
    frame = _toy_frame()
    # Frame has only CV2 rows
    frame.to_csv(csv, index=False)

    cfg = SLDLConfig(
        input_csv=csv,
        output_dir=tmp_path / "run",
        split_types=("CV1",),  # Request CV1 but frame has only CV2
        folds=(0, 1),
        include_coverage_flag=False,
    )

    with pytest.raises(
        RuntimeError,
        match="no metric rows produced; check split_types and training data",
    ):
        run_cv(cfg, ZeroEmbeddingProducer())


# FIX 3 RED — missing bags_npz warning
def test_missing_bags_npz_warns(tmp_path: Path, caplog, monkeypatch) -> None:
    """When bags_npz is None, _load_state_dl_caches must warn about full h5ad load."""
    import logging

    from sl_dl_model.bags import GwpsBags

    # Stub bags object — match exact GwpsBags signature
    stub_bags = GwpsBags(
        control_template=np.zeros((5, 8), dtype=np.float32),
        bags_by_symbol={
            "G0": np.zeros((10, 8), dtype=np.float32),
            "G1": np.zeros((10, 8), dtype=np.float32),
        },
        input_dim=8,
    )

    # Patch at the source module level since _load_state_dl_caches uses local imports
    import sl_dl_model.bags as bags_mod

    monkeypatch.setattr(bags_mod, "build_gwps_bags", lambda config, rng_seed: stub_bags)

    # Stub ESM2 loader at source module level
    import sl_dl_model.gene_embeddings as ge_mod

    class StubEsm:
        pass

    monkeypatch.setattr(ge_mod, "load_esm2_embeddings", lambda path: StubEsm())

    cfg = SLDLConfig(
        esm2_npz=tmp_path / "esm2.npz",  # Set this to avoid ValueError
        bags_npz=None,  # This is the trigger
        state_backend="linear_mock",
    )

    with caplog.at_level(logging.WARNING):
        _load_state_dl_caches(cfg)

    assert any(
        "bags_npz is not set" in rec.message
        and "full gwps h5ad will be loaded" in rec.message
        for rec in caplog.records
    )


def test_stale_bags_npz_raises_when_checkpoint_dim_differs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A stale cache with the wrong feature width cannot reach STATE."""
    from sl_dl_model.bags import GwpsBags

    stale_bags = GwpsBags(
        control_template=np.zeros((5, 4), dtype=np.float32),
        bags_by_symbol={"G0": np.zeros((10, 4), dtype=np.float32)},
        input_dim=4,
    )

    import sl_dl_model.bags as bags_mod

    monkeypatch.setattr(bags_mod, "load_bags_npz", lambda path: stale_bags)

    import sl_dl_model.gene_embeddings as ge_mod

    class StubEsm:
        pass

    monkeypatch.setattr(ge_mod, "load_esm2_embeddings", lambda path: StubEsm())

    checkpoint = tmp_path / "state" / "checkpoints" / "final.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    with (checkpoint.parent.parent / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"input_dim": 2, "gene_names": ["G0", "G1"]}, handle)
    bags_npz = tmp_path / "stale_bags.npz"
    bags_npz.touch()
    cfg = SLDLConfig(
        esm2_npz=tmp_path / "esm2.npz",
        bags_npz=bags_npz,
        state_checkpoint=checkpoint,
    )

    with pytest.raises(ValueError, match="setup_exp08_assets.py bags"):
        _load_state_dl_caches(cfg)


# FIX 1 RED — multi-process write guard
def test_non_main_process_does_not_write_artifacts(tmp_path: Path, monkeypatch) -> None:
    """Only the main process writes artifacts; non-main ranks must not.

    Simulates a non-main rank by patching PartialState to report
    ``is_main_process == False`` and asserting no output files are written.
    """
    import sl_dl_model.evaluate as ev

    class _NonMainState:
        is_main_process = False
        process_index = 1
        num_processes = 2

    monkeypatch.setattr(ev, "PartialState", lambda: _NonMainState())

    csv = tmp_path / "toy.csv"
    _toy_frame().to_csv(csv, index=False)
    out = tmp_path / "run"
    cfg = SLDLConfig(
        input_csv=csv,
        output_dir=out,
        split_types=("CV2",),
        folds=(0, 1),
        include_coverage_flag=False,
    )
    run_cv(cfg, ZeroEmbeddingProducer())

    # Non-main rank must not have written any artifact files.
    assert not (out / "fold_metrics.csv").exists()
    assert not (out / "manifest.json").exists()


# FIX 7 RED — per-split output layout + combined official summary
def test_per_split_output_layout_and_combined_summary(tmp_path: Path) -> None:
    """run_cv writes per-split subdirs plus a combined official summary CSV.

    Spec §7: ``fold_metrics.csv``/``summary.csv``/``manifest.json`` per split
    under ``<out>/<cvN>/`` plus a combined ``official_metrics_summary.csv``.
    """
    csv = tmp_path / "toy.csv"
    _toy_frame().to_csv(csv, index=False)
    out = tmp_path / "run"
    cfg = SLDLConfig(
        input_csv=csv,
        output_dir=out,
        split_types=("CV2",),
        folds=(0, 1),
        include_coverage_flag=False,
    )
    run_cv(cfg, ZeroEmbeddingProducer())

    # Per-split subdir for CV2.
    assert (out / "CV2" / "fold_metrics.csv").exists()
    assert (out / "CV2" / "summary.csv").exists()
    assert (out / "CV2" / "manifest.json").exists()

    # Combined official summary across splits.
    combined = out / "official_metrics_summary.csv"
    assert combined.exists()
    df = pd.read_csv(combined)
    assert "split_type" in df.columns
    assert set(df["split_type"]) == {"CV2"}


def test_manifest_includes_training_fields() -> None:
    from sl_dl_model.evaluate import _build_manifest

    cfg = SLDLConfig(esm2_model="x", batch_pairs=512, early_stop_patience=4)
    manifest = _build_manifest(
        cfg,
        split_types=("CV2",),
        candidate_gene_count=10,
        gwps_coverage_gene_count=None,
    )
    assert manifest["batch_pairs"] == 512
    assert manifest["early_stop_patience"] == 4
    assert manifest["early_stop_metric"] == "val_pair_auroc"
    assert manifest["val_source"] == "test_fold"
