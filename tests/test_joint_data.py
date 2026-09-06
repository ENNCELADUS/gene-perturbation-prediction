"""Prepared-cache opening, fit isolation, and explicit batch contracts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.batches import DependencyBatch, ResponseBatch
from src.data.datasets import (
    DependencyDataset,
    ResponseDataset,
    make_evaluation_loaders,
)
from src.data.prepared import load_inputs
from src.data.tx1_cache import _hvg_gene_order_signature
from src.training.sampling import make_training_loaders


def make_prepared_fixture(root: Path, *, hvg_width: int = 2) -> dict:
    """Write real small prepared files, including four unequal anchor pools."""
    root.mkdir(parents=True, exist_ok=True)
    anchors = [f"ACH-A{i}" for i in range(4)]
    split = {
        "train": anchors + ["ACH-TRAIN", "ACH-UNLABELED"],
        "val": ["ACH-VAL"],
        "test": ["ACH-TEST"],
        "unlabeled_train": ["ACH-UNLABELED"],
    }
    panel = ["G1", "G0", "G2"]
    if hvg_width < 2:
        raise ValueError("fixture hvg_width must be at least 2")
    hvg = ["G0", "G1", *[f"HVG{i}" for i in range(2, hvg_width)]]
    esm_order = ["G2", "G0", "G1", "R0", "R1", "R2", "R3"]
    (root / "split.json").write_text(json.dumps(split))
    pd.DataFrame({"gene_symbol": panel}).to_csv(root / "panel.csv", index=False)
    ids = anchors + ["ACH-TRAIN", "ACH-VAL", "ACH-TEST"]
    pd.DataFrame(
        {
            "G0 (1)": np.arange(7),
            "G1 (2)": np.arange(7) ** 2,
            "G2 (3)": [1, 0, 1, 0, 1, 9, 8],
        },
        index=ids,
    ).to_csv(root / "labels.csv")
    (root / "state").mkdir()
    (root / "state" / "var_dims.pkl").write_bytes(pickle.dumps({"gene_names": hvg}))
    np.savez(
        root / "esm2.npz",
        symbols=np.array(esm_order),
        vectors=np.arange(len(esm_order) * 3, dtype=np.float32).reshape(-1, 3),
        resolved=np.ones(len(esm_order), dtype=bool),
    )
    (root / "q_sc").mkdir()
    for line, model_id in enumerate(ids):
        directory = root / "tx1" / model_id
        directory.mkdir(parents=True)
        rows = np.arange(5, dtype=np.float32) + line * 10
        np.save(directory / "embeddings.npy", np.repeat(rows[:, None], 2560, axis=1))
        np.save(directory / "hvg.npy", np.repeat(rows[:, None], hvg_width, axis=1))
        pd.DataFrame(index=[f"cell-{i}" for i in range(5)]).to_parquet(
            directory / "obs.parquet"
        )
        (directory / "hvg_gene_order.json").write_text(
            json.dumps(_hvg_gene_order_signature(hvg))
        )
        available = np.array([False, True, True])
        values = np.array(
            [[np.nan, np.nan, np.nan], [line + 1, 0.5, 1], [2, 0.2, 3]],
            dtype=np.float32,
        )
        np.savez(
            root / "q_sc" / f"{model_id}.npz",
            model_id=model_id,
            gene_symbols=np.array(panel),
            values=values,
            available=available,
            source_sha256="historical-only",
        )
    conditions = [
        {"model_id": anchor, "gene": gene}
        for i, anchor in enumerate(anchors)
        for gene in ["G0", "G1", *[f"R{j}" for j in range(i + 1)]]
    ]
    holdout = [{"model_id": anchor, "gene": "G0"} for anchor in anchors]
    response = root / "response" / "response_targets"
    response.mkdir(parents=True)
    n = len(conditions)
    np.save(
        response / "target_cells.npy",
        np.arange(n * 2 * hvg_width, dtype=np.float32).reshape(-1, hvg_width),
    )
    np.save(response / "offsets.npy", np.arange(n + 1, dtype=np.int64) * 2)
    np.save(
        response / "genes.npy",
        np.array([f"{r['gene']}@{r['model_id']}" for r in conditions], dtype=object),
    )
    pd.DataFrame(
        [
            {"model_id": r["model_id"], "perturbation_gene": r["gene"], "n_cells": 2}
            for r in conditions
        ]
    ).to_parquet(response / "metadata.parquet")
    (response / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "n_bags": n,
                "target_dim": hvg_width,
                "hvg_order": hvg,
            }
        )
    )
    metadata = {
        "schema_version": "geneeffect-joint-prepared-v1",
        "split": split,
        "common_gene_panel": panel,
        "hvg_order": hvg,
        "esm2_order": esm_order,
        "response_anchors": anchors,
        "response_conditions": conditions,
        "response_holdout": holdout,
    }
    (root / "prepared_inputs.json").write_text(json.dumps(metadata))
    return {
        "prepared_root": str(root),
        "paths": {
            "split": str(root / "split.json"),
            "common_gene_panel": str(root / "panel.csv"),
            "gene_effect": str(root / "labels.csv"),
            "state_model_dir": str(root / "state"),
            "esm2_embeddings": str(root / "esm2.npz"),
            "tx1_cache": str(root / "tx1"),
            "q_sc_cache": str(root / "q_sc"),
            "response_cache": str(root / "response"),
        },
        "features": {"hvg_dim": hvg_width, "esm2_dim": 3, "cells_per_context": 8},
        "train": {"dependency_batch_size": 2, "response_batch_size": 8},
        "seeds": {"train": 0, "collator": 0, "projection": 0},
    }


def test_load_inputs_preserves_mixed_case_state_hvg_axis(tmp_path: Path) -> None:
    config = make_prepared_fixture(tmp_path)
    hvg_order = ["C1orf109", "C3orf38"]
    state_dir = Path(config["paths"]["state_model_dir"])
    (state_dir / "var_dims.pkl").write_bytes(pickle.dumps({"gene_names": hvg_order}))
    tx1_root = Path(config["paths"]["tx1_cache"])
    for sidecar in tx1_root.glob("*/hvg_gene_order.json"):
        sidecar.write_text(json.dumps(_hvg_gene_order_signature(hvg_order)))
    prepared_path = Path(config["prepared_root"]) / "prepared_inputs.json"
    prepared = json.loads(prepared_path.read_text())
    prepared["hvg_order"] = hvg_order
    prepared_path.write_text(json.dumps(prepared))
    response_manifest_path = (
        Path(config["paths"]["response_cache"]) / "response_targets" / "manifest.json"
    )
    response_manifest = json.loads(response_manifest_path.read_text())
    response_manifest["hvg_order"] = hvg_order
    response_manifest_path.write_text(json.dumps(response_manifest))

    inputs = load_inputs(config)

    assert inputs.hvg_order == tuple(hvg_order)


def test_train_only_fit_and_explicit_test_exposure(tmp_path):
    config = make_prepared_fixture(tmp_path)
    before = load_inputs(config)
    assert set(before.labels.model_id) == set(
        before.split.supervised_train + before.split.val
    )
    assert before.train_gene_means["G0"] == 2
    assert before.variable_genes == frozenset({"G1"})
    batch_before = DependencyDataset(before, "train").collate([0, 1, 2])
    wide = pd.read_csv(config["paths"]["gene_effect"], index_col=0)
    wide.loc[["ACH-VAL", "ACH-TEST"]] = 9999
    wide.to_csv(config["paths"]["gene_effect"])
    after = load_inputs(config)
    pd.testing.assert_series_equal(before.train_gene_means, after.train_gene_means)
    assert before.variable_genes == after.variable_genes
    batch_after = DependencyDataset(after, "train").collate([0, 1, 2])
    from src.model.normalization import BlockStandardizer

    for name in ("q_sc", "e_g", "z_c"):
        torch.testing.assert_close(
            getattr(batch_before.conditions, name),
            getattr(batch_after.conditions, name),
            rtol=0,
            atol=0,
        )
        standardizer = BlockStandardizer().fit(
            {name: getattr(batch_before.conditions, name)}
        )
        torch.testing.assert_close(
            standardizer.transform(name, getattr(batch_before.conditions, name)),
            standardizer.transform(name, getattr(batch_after.conditions, name)),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        batch_before.residual, batch_after.residual, rtol=0, atol=0
    )
    for model_id in before.lines:
        np.testing.assert_array_equal(
            before.lines[model_id].controls_tx1, after.lines[model_id].controls_tx1
        )
        np.testing.assert_array_equal(
            before.lines[model_id].basal_hvg, after.lines[model_id].basal_hvg
        )
    with pytest.raises(ValueError, match="no finite dependency rows"):
        DependencyDataset(after, "test")
    assert set(
        DependencyDataset(load_inputs(config, include_test=True), "test").rows.model_id
    ) == {"ACH-TEST"}


def test_restore_bypasses_fits_raw_builders_and_external_esm2(tmp_path, monkeypatch):
    import src.data.geneeffect as ge
    import src.data.residual_target as residual
    import src.data.response as response
    import src.data.response_cache as response_cache
    import src.data.tx1_cache as tx1_cache

    config = make_prepared_fixture(tmp_path)
    inputs = load_inputs(config)
    state = inputs.preprocessing_state()
    state["esm2_vectors"] = state["esm2_vectors"] + 11
    Path(config["paths"]["esm2_embeddings"]).unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("a raw rebuild, source hash, or fresh fit was invoked")

    for module, names in [
        (ge, ["fit_variable_gene_membership", "load_source_registry"]),
        (residual, ["fit_gene_means", "build_residual_targets"]),
        (response, ["assemble_train_response_gene_bags"]),
        (
            response_cache,
            ["write_response_targets_cache"],
        ),
        (tx1_cache, ["verify_cache", "embed_lines"]),
    ]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    restored = load_inputs(config, preprocessing=state, include_test=True)
    batch = next(iter(make_training_loaders(restored, config, 2, None)[0]))
    index = {gene: i for i, gene in enumerate(restored.esm2_symbols)}
    torch.testing.assert_close(
        batch.conditions.e_g,
        torch.stack([state["esm2_vectors"][index[g]] for g in batch.conditions.genes]),
    )
    assert next(
        iter(make_evaluation_loaders(restored, config, "test", None)[0])
    ).conditions.model_ids == ("ACH-TEST", "ACH-TEST")
    del state["esm2_vectors"]
    with pytest.raises(ValueError, match="requires esm2_symbols and esm2_vectors"):
        load_inputs(config, preprocessing=state)


def test_masks_paired_selection_context_and_device_transfer(tmp_path):
    config = make_prepared_fixture(tmp_path)
    inputs = load_inputs(config)
    data = DependencyDataset(inputs, "train")
    indices = [
        int(
            data.rows.index[
                (data.rows.model_id == "ACH-A0") & (data.rows.gene_symbol == gene)
            ][0]
        )
        for gene in inputs.genes
    ]
    batch = data.collate(indices)
    assert batch.conditions.q_sc_mask.tolist() == [False, True, True]
    assert batch.conditions.gene_in_hvg_panel.tolist() == [True, True, False]
    assert batch.conditions.own_gene_shift_available.tolist() == [False, True, False]
    assert batch.conditions.q_sc[0].tolist() == [0, 0, 0]
    controls, hvg = batch.conditions.controls_tx1[0], batch.conditions.basal_hvg[0]
    torch.testing.assert_close(controls[:, 0], hvg[:, 0])
    assert len(set(controls[:5, 0].tolist())) == 5
    torch.testing.assert_close(controls[5:, 0], controls[:3, 0])
    torch.testing.assert_close(
        batch.conditions.z_c[0],
        torch.cat((controls.mean(0), controls.var(0, unbiased=False))),
    )
    moved = batch.to("meta", non_blocking=True)
    assert moved.conditions.genes == batch.conditions.genes
    assert moved.conditions.model_ids == batch.conditions.model_ids
    assert moved.conditions.controls_tx1[0].device.type == "meta"
    assert moved.valid.device.type == "meta"
    response = ResponseDataset(inputs, holdout=True).collate([0])
    moved_response = response.to("meta", non_blocking=True)
    assert moved_response.model_ids == response.model_ids
    assert moved_response.genes == ("G0",)
    assert moved_response.control_hvg[0].device.type == "meta"


def test_collators_do_not_repeat_full_batch_validation(tmp_path, monkeypatch):
    """Prepared-cache validation must not be repeated in the per-step hot path."""
    config = make_prepared_fixture(tmp_path)
    inputs = load_inputs(config)

    def fail_validation(self):
        raise AssertionError("full batch validation entered the collation hot path")

    monkeypatch.setattr(DependencyBatch, "validate", fail_validation)
    monkeypatch.setattr(ResponseBatch, "validate", fail_validation)

    dependency = DependencyDataset(inputs, "train")
    dependency.collate([0, 1])
    response = ResponseDataset(inputs, holdout=True)
    response.collate([response.indices[0]])


@pytest.mark.parametrize(
    "relative",
    [
        "prepared_inputs.json",
        "panel.csv",
        "tx1/ACH-A0/embeddings.npy",
        "tx1/ACH-A0/hvg.npy",
        "tx1/ACH-A0/obs.parquet",
        "tx1/ACH-A0/hvg_gene_order.json",
        "q_sc/ACH-A0.npz",
        "response/response_targets/target_cells.npy",
        "response/response_targets/offsets.npy",
        "response/response_targets/genes.npy",
        "response/response_targets/metadata.parquet",
        "response/response_targets/manifest.json",
    ],
)
def test_missing_prepared_file_is_read_only_and_actionable(tmp_path, relative):
    config = make_prepared_fixture(tmp_path)
    missing = tmp_path / relative
    missing.unlink()
    before = set(tmp_path.rglob("*"))
    with pytest.raises((ValueError, FileNotFoundError)) as caught:
        load_inputs(config)
    assert str(missing) in str(caught.value)
    assert "hpc/run.sh prepare <config>" in str(caught.value)
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("order", [None, ["G1", "G0"]])
def test_response_cache_requires_actual_matching_order(tmp_path, order):
    config = make_prepared_fixture(tmp_path)
    manifest_path = tmp_path / "response/response_targets/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if order is None:
        del manifest["hvg_order"]
    else:
        manifest["hvg_order"] = order
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hvg_order.*prepare"):
        load_inputs(config)


def test_cache_rejects_misaligned_qsc_and_response_keys(tmp_path):
    config = make_prepared_fixture(tmp_path)
    path = tmp_path / "q_sc/ACH-A0.npz"
    with np.load(path) as shard:
        payload = dict(shard)
    payload["gene_symbols"] = payload["gene_symbols"][::-1]
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="gene order"):
        load_inputs(config)
    payload["gene_symbols"] = payload["gene_symbols"][::-1]
    np.savez(path, **payload)
    path = tmp_path / "response/response_targets/genes.npy"
    keys = np.load(path, allow_pickle=True)
    keys[0] = "G1@ACH-A0"
    np.save(path, keys)
    with pytest.raises(ValueError, match="disagree with metadata"):
        load_inputs(config)


def test_fresh_opening_does_not_build_or_hash_source_files(tmp_path, monkeypatch):
    import src.data.geneeffect as ge
    import src.data.q_sc as q_sc
    import src.data.response as response
    import src.data.response_cache as response_cache
    import src.data.tx1_cache as tx1_cache

    config = make_prepared_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("raw input rebuilding or source-file hashing")

    for module, names in [
        (ge, ["load_source_registry"]),
        (q_sc, ["compute_q_sc"]),
        (response, ["assemble_train_response_gene_bags"]),
        (
            response_cache,
            [
                "write_response_targets_cache",
            ],
        ),
        (tx1_cache, ["verify_cache", "embed_lines", "sha256_file"]),
    ]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    before = set(tmp_path.rglob("*"))
    inputs = load_inputs(config)
    dependency, replay = make_training_loaders(inputs, config, 2, None)
    assert next(iter(dependency)).valid.all()
    assert next(replay).genes
    assert set(tmp_path.rglob("*")) == before


def test_response_gene_requires_esm2_resolution(tmp_path):
    config = make_prepared_fixture(tmp_path)
    state = load_inputs(config).preprocessing_state()
    state["esm2_symbols"] = state["esm2_symbols"][:-1]
    state["esm2_vectors"] = state["esm2_vectors"][:-1]
    metadata_path = tmp_path / "prepared_inputs.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["esm2_order"] = metadata["esm2_order"][:-1]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="response genes missing from ESM2.*R3"):
        load_inputs(config, preprocessing=state)


def test_fixture_supports_full_hvg_width_for_later_model_integration(tmp_path):
    inputs = load_inputs(make_prepared_fixture(tmp_path, hvg_width=2000))
    assert len(inputs.hvg_order) == 2000
    assert inputs.lines["ACH-A0"].basal_hvg.shape == (8, 2000)
    assert inputs.response_targets.target_bag(0).shape == (2, 2000)
