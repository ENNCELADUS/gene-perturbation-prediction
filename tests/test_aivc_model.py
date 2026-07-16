from __future__ import annotations

from dataclasses import replace
import json
import hashlib
import math
from pathlib import Path
import pickle
import sys
import types
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture
import torch
import torch.nn.functional as F

import aivc_model.model as model_module
import aivc_model.prepare as prepare_module
import aivc_model.train as train_module
import aivc_model.gwps_cache as gwps_cache_module
import aivc_model.gene_splits as gene_splits_module
from aivc_model.gwps_cache import (
    source_fingerprint,
)
from aivc_model.model import (
    AivcModel,
    Esm2PerturbationAdapter,
    LossWeights,
    MLPHead,
    PerturbationVectorAdapter,
    StateForwardAdapter,
    fit_fixed_gmm,
    load_state_model,
    _pairwise_ranknet_loss,
)
from aivc_model.response import ResponseEncoder, TrainableDiagonalGMM
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
from aivc_model.prepare import (
    DataConfig,
    ExternalSourceConfig,
    GeneBags,
    GeneSplit,
    ProjectorConfig,
    ResponseEncoderConfig,
    SplitConfig,
    _external_state_input_view,
    _load_metadata,
    _load_scvi_latent_cache,
    _scvi_cache_metadata,
    _merge_external_gene_rows,
    _project_scvi_latent_collections,
    _project_scvi_latent_groups,
    _scvi_datasplitter_kwargs,
    _scvi_latent_cache_dir,
    _suppress_scvi_lightning_warnings,
    _scvi_trainer_kwargs,
    _state_input_view,
    _var_symbols,
    _write_scvi_latent_cache,
    encode_batch_labels,
    load_external_gene_bags,
    load_config,
    load_perturbation_vectors,
    load_gene_bags,
    load_state_batch_lookup,
    make_cell_set_chunks,
    make_gene_split,
    resolve_state_gene_order,
    with_cached_scvi_teacher_latents,
)
from aivc_model.train import _write_csv_if_main, run_training


def _validated_cpu_accelerator(config: object) -> object:
    accelerator = train_module._make_accelerator(config)
    setattr(
        accelerator,
        "_aivc_exp05_cuda_topology",
        (4, tuple(("cuda", index) for index in range(4))),
    )
    return accelerator


def _toy_artifact_authority(
    *,
    source_fingerprint: str = "source",
    canonical_split_sha256: str = "split",
    fit_genes: tuple[str, ...] = ("GENE1",),
) -> object:
    return prepare_module.ArtifactAuthority(
        source_fingerprint=source_fingerprint,
        canonical_split_sha256=canonical_split_sha256,
        outer_fold=0,
        fit_stage="inner_train",
        fit_genes=fit_genes,
        train_genes=fit_genes,
        val_genes=("GENE2",),
        test_genes=("GENE3",),
    )


def _toy_cache_inputs(tmp_path: Path) -> dict[str, object]:
    paths = {
        "h5ad": tmp_path / "source.h5ad",
        "checkpoint": tmp_path / "final.ckpt",
        "var_dims": tmp_path / "var_dims.pkl",
        "pert_onehot_map": tmp_path / "pert_onehot_map.pt",
        "batch_sidecar": tmp_path / "batch_onehot_map.pt",
        "cell_type_sidecar": None,
        "canonical_split": tmp_path / "outer.csv",
        "canonical_split_sha256_file": tmp_path / "outer.csv.sha256",
        "overlap_csv": tmp_path / "overlap.csv",
    }
    for key, path in paths.items():
        if path is not None:
            path.write_bytes(str(key).encode())
    return {
        **paths,
        "feature_names": np.asarray(["A", "B"], dtype=object),
        "canonical_gene_count": 2,
        "cache_seed": 7,
        "cache_cells_per_gene": 1,
        "depmap_label_col": "depmap_gene_effect",
        "matched_label_col": "has_depmap_label",
        "var_gene_symbol_col": "gene_name",
        "obs_perturbation_col": "gene",
        "control_label": "non-targeting",
        "obs_batch_col": "gem_group",
    }


def _toy_gwps_cache_config(
    tmp_path: Path,
    *,
    response_genes: list[str] | None = None,
) -> object:
    tmp_path.mkdir(parents=True, exist_ok=True)
    genes = response_genes or ["G1", "G2"]
    model_dir = tmp_path / "state"
    model_dir.mkdir(exist_ok=True)
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump(
            {"gene_names": ["A", "B"], "input_dim": 2, "output_dim": 2},
            handle,
        )
    (model_dir / "pert_onehot_map.pt").write_bytes(b"perturbations")
    (model_dir / "batch_onehot_map.pt").write_bytes(b"batches")
    checkpoint = model_dir / "checkpoints" / "final.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")

    rows = [
        [float(index + 1), float((index + 1) * 10)] for index in range(len(genes) + 2)
    ]
    adata = ad.AnnData(np.asarray(rows, dtype=np.float32))
    adata.var_names = ["ENSG1", "ENSG2"]
    adata.var["gene_name"] = ["B", "A"]
    adata.var["alternate_gene_name"] = ["B", "A"]
    adata.obs["gene"] = [*genes, "non-targeting", "non-targeting"]
    adata.obs["alternate_gene"] = adata.obs["gene"].astype(str)
    adata.obs["gem_group"] = [
        *("25" if index % 2 == 0 else "31" for index in range(len(genes))),
        "25",
        "31",
    ]
    adata.obs["alternate_batch"] = adata.obs["gem_group"].astype(str)
    h5ad_path = tmp_path / "gwps.h5ad"
    adata.write_h5ad(h5ad_path)

    overlap_csv = tmp_path / "overlap.csv"
    pd.DataFrame(
        {
            "perturbation_gene": ["G1", "G2"],
            "depmap_gene_effect": [-1.0, -0.5],
            "has_depmap_label": [True, True],
        }
    ).to_csv(overlap_csv, index=False)
    outer_manifest = tmp_path / "outer.csv"
    pd.DataFrame({"perturbation_gene": ["G1", "G2"], "outer_fold": [0, 1]}).to_csv(
        outer_manifest, index=False
    )
    outer_sha256 = tmp_path / "outer.csv.sha256"
    outer_sha256.write_text(f"{_sha256(outer_manifest)}\n")
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            h5ad_path=h5ad_path,
            overlap_csv=overlap_csv,
            obs_perturbation_col="gene",
            control_label="non-targeting",
            obs_batch_col="gem_group",
            var_gene_symbol_col="gene_name",
            depmap_label_col="depmap_gene_effect",
            matched_label_col="has_depmap_label",
            cache_seed=42,
            cache_cells_per_gene=1,
        ),
        state=types.SimpleNamespace(
            backend="state_checkpoint",
            model_dir=model_dir,
            checkpoint_path=checkpoint,
        ),
        cv=types.SimpleNamespace(
            outer_split_manifest=outer_manifest,
            outer_split_sha256_file=outer_sha256,
            expected_gene_count=2,
        ),
    )


def test_gwps_cache_manifest_changes_with_state_sidecar(tmp_path: Path) -> None:
    inputs = _toy_cache_inputs(tmp_path)
    first = source_fingerprint(**inputs)
    inputs["var_dims"].write_bytes(b"changed")
    second = source_fingerprint(**inputs)
    assert first != second


def test_gwps_cache_round_trip_preserves_order_and_batches(tmp_path: Path) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    cache_dir = tmp_path / "cache"
    contract = gwps_cache_module._CacheContract(gene_count=2, state_dim=2)
    gwps_cache_module._build_gwps_cache(config, cache_dir, contract)
    bags = gwps_cache_module._load_gwps_cache(config, cache_dir, contract)
    assert bags.feature_names.tolist() == ["A", "B"]
    np.testing.assert_allclose(bags.feature_fill_values, np.asarray([35.0, 3.5]))
    assert bags.genes.tolist() == ["G1", "G2"]
    np.testing.assert_array_equal(bags.gene_outer_folds, np.asarray([0, 1]))
    np.testing.assert_array_equal(bags.control_batch, np.asarray(["25", "31"]))
    np.testing.assert_array_equal(bags.input_bags[0], np.asarray([[10.0, 1.0]]))
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert set(manifest["arrays"]) == set(gwps_cache_module._ARRAY_FILENAMES)
    assert all(
        set(metadata) == {"sha256", "shape", "dtype"}
        for metadata in manifest["arrays"].values()
    )


def test_gwps_cache_rejects_legacy_filename_only_manifest(tmp_path: Path) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    cache_dir = tmp_path / "cache"
    contract = gwps_cache_module._CacheContract(gene_count=2, state_dim=2)
    manifest_path = gwps_cache_module._build_gwps_cache(config, cache_dir, contract)
    manifest = json.loads(manifest_path.read_text())
    manifest["arrays"] = list(gwps_cache_module._ARRAY_FILENAMES)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest array contract"):
        gwps_cache_module._load_gwps_cache(config, cache_dir, contract)
    with pytest.raises(ValueError, match="manifest array contract"):
        gwps_cache_module._build_gwps_cache(config, cache_dir, contract)


@pytest.mark.parametrize("filename", ["cells.npy", "offsets.npy", "batch_labels.npy"])
def test_gwps_cache_rejects_array_mutation(
    tmp_path: Path,
    filename: str,
) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    cache_dir = tmp_path / "cache"
    contract = gwps_cache_module._CacheContract(gene_count=2, state_dim=2)
    gwps_cache_module._build_gwps_cache(config, cache_dir, contract)
    array = np.load(cache_dir / filename, mmap_mode="r+")
    array.flat[0] = "X" if array.dtype.kind in {"U", "S"} else array.flat[0] + 1
    array.flush()

    with pytest.raises(ValueError, match=f"{filename} SHA-256 mismatch"):
        gwps_cache_module._load_gwps_cache(config, cache_dir, contract)
    with pytest.raises(ValueError, match=f"{filename} SHA-256 mismatch"):
        gwps_cache_module._build_gwps_cache(config, cache_dir, contract)


def test_gwps_cache_rejects_bound_structurally_invalid_offsets(tmp_path: Path) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    cache_dir = tmp_path / "cache"
    contract = gwps_cache_module._CacheContract(gene_count=2, state_dim=2)
    manifest_path = gwps_cache_module._build_gwps_cache(config, cache_dir, contract)
    offsets = np.load(cache_dir / "offsets.npy", mmap_mode="r+")
    offsets[1] = offsets[0]
    offsets.flush()
    manifest = json.loads(manifest_path.read_text())
    manifest["arrays"] = gwps_cache_module._array_manifest(cache_dir)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="every gene bag must be non-empty"):
        gwps_cache_module._load_gwps_cache(config, cache_dir, contract)


def test_gwps_cache_rejects_bound_structurally_invalid_batch_length(
    tmp_path: Path,
) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    cache_dir = tmp_path / "cache"
    contract = gwps_cache_module._CacheContract(gene_count=2, state_dim=2)
    manifest_path = gwps_cache_module._build_gwps_cache(config, cache_dir, contract)
    np.save(cache_dir / "batch_labels.npy", np.asarray(["25"]))
    manifest = json.loads(manifest_path.read_text())
    manifest["arrays"] = gwps_cache_module._array_manifest(cache_dir)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="response batches must match response cells"):
        gwps_cache_module._load_gwps_cache(config, cache_dir, contract)


def test_gwps_cache_replaces_nonfinite_from_control_only(tmp_path: Path) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    adata = ad.read_h5ad(config.data.h5ad_path)
    values = np.asarray(adata.X, dtype=np.float32)
    values[0, 1] = np.nan
    values[1, 0] = np.inf
    values[3, 1] = np.nan
    adata.X = values
    adata.write_h5ad(config.data.h5ad_path)

    cache_dir = tmp_path / "cache"
    gwps_cache_module._build_gwps_cache(
        config,
        cache_dir,
        gwps_cache_module._CacheContract(gene_count=2, state_dim=2),
    )

    cells = np.load(cache_dir / "cells.npy")
    controls = np.load(cache_dir / "control_cells.npy")
    fills = np.load(cache_dir / "feature_fill_values.npy")
    assert np.isfinite(cells).all()
    assert np.isfinite(controls).all()
    assert np.isfinite(fills).all()
    np.testing.assert_allclose(fills, np.asarray([30.0, 3.5], dtype=np.float32))
    assert json.loads((cache_dir / "manifest.json").read_text())["schema_version"] == 2


def test_gwps_cache_accepts_extra_source_gene_but_rejects_missing_canonical_gene(
    tmp_path: Path,
) -> None:
    contract = gwps_cache_module._CacheContract(gene_count=2, state_dim=2)
    superset = _toy_gwps_cache_config(
        tmp_path / "superset", response_genes=["G1", "G2", "EXTRA"]
    )
    gwps_cache_module._build_gwps_cache(
        superset, tmp_path / "superset" / "cache", contract
    )
    bags = gwps_cache_module._load_gwps_cache(
        superset, tmp_path / "superset" / "cache", contract
    )
    assert bags.genes.tolist() == ["G1", "G2"]

    missing = _toy_gwps_cache_config(
        tmp_path / "missing", response_genes=["G1", "EXTRA"]
    )
    with pytest.raises(ValueError, match="missing canonical"):
        gwps_cache_module._build_gwps_cache(
            missing, tmp_path / "missing" / "cache", contract
        )


def test_gwps_cache_reads_backed_and_streams_only_selected_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _toy_gwps_cache_config(
        tmp_path, response_genes=["G1", "G1", "G2", "EXTRA"]
    )
    config.data.cache_cells_per_gene = 1
    contract = gwps_cache_module._CacheContract(gene_count=2, state_dim=2)
    real_read_h5ad = ad.read_h5ad

    def guarded_read_h5ad(path: Path, *, backed: str | None = None) -> ad.AnnData:
        assert backed == "r"
        return real_read_h5ad(path, backed=backed)

    monkeypatch.setattr(gwps_cache_module.ad, "read_h5ad", guarded_read_h5ad)
    assert not hasattr(gwps_cache_module, "_dense_slice")

    cache_dir = tmp_path / "cache"
    gwps_cache_module._build_gwps_cache(config, cache_dir, contract)
    bags = gwps_cache_module._load_gwps_cache(config, cache_dir, contract)

    assert [bag.shape for bag in bags.input_bags] == [(1, 2), (1, 2)]
    assert bags.control_input.shape == (2, 2)
    assert bags.feature_names.tolist() == ["A", "B"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_state_checkpoint_input_requires_model_dir(tmp_path: Path) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    config.state.model_dir = None
    adata = ad.read_h5ad(config.data.h5ad_path)
    with pytest.raises(ValueError, match="model_dir"):
        _state_input_view(adata, config)


def test_state_checkpoint_input_requires_checkpoint_path(tmp_path: Path) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    config.state.checkpoint_path = None
    adata = ad.read_h5ad(config.data.h5ad_path)
    with pytest.raises(ValueError, match="checkpoint_path"):
        _state_input_view(adata, config)


@pytest.mark.parametrize(
    ("input_dim", "output_dim", "feature_count", "message"),
    [
        (1999, 2000, 1999, "input_dim=1999"),
        (2000, 1999, 2000, "output_dim=1999"),
        (2000, 2000, 1999, "feature count=1999"),
    ],
)
def test_production_state_contract_rejects_wrong_checkpoint_dimensions(
    tmp_path: Path,
    input_dim: int,
    output_dim: int,
    feature_count: int,
    message: str,
) -> None:
    model_dir = tmp_path / "state"
    model_dir.mkdir()
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump(
            {
                "gene_names": [f"G{i}" for i in range(feature_count)],
                "input_dim": input_dim,
                "output_dim": output_dim,
            },
            handle,
        )
    with pytest.raises(ValueError, match=message):
        gwps_cache_module._validate_state_contract(
            model_dir,
            np.asarray([f"G{i}" for i in range(feature_count)], dtype=object),
            gwps_cache_module._PRODUCTION_CONTRACT,
        )


def test_production_canonical_count_cannot_be_redefined_by_config(
    tmp_path: Path,
) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    with pytest.raises(ValueError, match="9338"):
        gwps_cache_module._load_canonical_manifest(
            config,
            config.cv.outer_split_manifest,
            gwps_cache_module._PRODUCTION_CONTRACT,
        )


def test_gwps_cache_rejects_invalid_canonical_sha256_authority(
    tmp_path: Path,
) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    config.cv.outer_split_sha256_file.write_text(f"{'0' * 64}\n")
    contract = gwps_cache_module._CacheContract(gene_count=2, state_dim=2)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        gwps_cache_module._build_gwps_cache(config, tmp_path / "cache", contract)


def test_gwps_cache_label_change_invalidates_existing_cache(tmp_path: Path) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    cache_dir = tmp_path / "cache"
    contract = gwps_cache_module._CacheContract(gene_count=2, state_dim=2)
    gwps_cache_module._build_gwps_cache(config, cache_dir, contract)
    labels = pd.read_csv(config.data.overlap_csv)
    labels.loc[0, "depmap_gene_effect"] = -9.0
    labels.to_csv(config.data.overlap_csv, index=False)

    with pytest.raises(ValueError, match="GWPS cache fingerprint mismatch"):
        gwps_cache_module._build_gwps_cache(config, cache_dir, contract)


@pytest.mark.parametrize(
    ("setting", "replacement"),
    [
        ("var_gene_symbol_col", "alternate_gene_name"),
        ("obs_perturbation_col", "alternate_gene"),
        ("control_label", "control"),
        ("obs_batch_col", "alternate_batch"),
    ],
)
def test_gwps_cache_setting_change_invalidates_existing_cache(
    tmp_path: Path,
    setting: str,
    replacement: str,
) -> None:
    config = _toy_gwps_cache_config(tmp_path)
    cache_dir = tmp_path / "cache"
    contract = gwps_cache_module._CacheContract(gene_count=2, state_dim=2)
    gwps_cache_module._build_gwps_cache(config, cache_dir, contract)
    setattr(config.data, setting, replacement)

    with pytest.raises(ValueError, match="GWPS cache fingerprint mismatch"):
        gwps_cache_module._build_gwps_cache(config, cache_dir, contract)


def test_state_alignment_uses_gene_name_in_checkpoint_order(tmp_path: Path) -> None:
    adata = ad.AnnData(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    adata.var_names = ["ENSG1", "ENSG2", "ENSG3"]
    adata.var["gene_name"] = ["B", "A", "C"]
    model_dir = tmp_path / "state"
    model_dir.mkdir()
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": ["A", "B"]}, handle)

    indices, names = resolve_state_gene_order(adata, model_dir, "gene_name")

    np.testing.assert_array_equal(indices, np.asarray([1, 0]))
    np.testing.assert_array_equal(names, np.asarray(["A", "B"], dtype=object))


def test_state_alignment_never_falls_back_when_checkpoint_gene_is_missing(
    tmp_path: Path,
) -> None:
    adata = ad.AnnData(np.ones((1, 1), dtype=np.float32))
    adata.var["gene_name"] = ["A"]
    model_dir = tmp_path / "state"
    model_dir.mkdir()
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": ["A", "B"]}, handle)

    with pytest.raises(ValueError, match="1/2"):
        resolve_state_gene_order(adata, model_dir, "gene_name")


def test_make_gene_split_is_disjoint() -> None:
    genes = np.asarray([f"GENE{i}" for i in range(12)], dtype=object)
    y = np.linspace(-2.0, 1.0, len(genes), dtype=np.float32)

    split = make_gene_split(
        genes,
        y,
        SplitConfig(
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            random_state=3,
            stratify_bins=3,
        ),
    )

    assert set(split.train).isdisjoint(set(split.val))
    assert set(split.train).isdisjoint(set(split.test))
    assert set(split.val).isdisjoint(set(split.test))
    assert len(split.train) + len(split.val) + len(split.test) == len(genes)


def test_make_gene_split_supports_zero_internal_test() -> None:
    genes = np.asarray([f"GENE{i}" for i in range(10)], dtype=object)
    y = np.linspace(-2.0, 1.0, len(genes), dtype=np.float32)

    split = make_gene_split(
        genes,
        y,
        SplitConfig(
            train_fraction=0.9,
            val_fraction=0.1,
            test_fraction=0.0,
            random_state=3,
            stratify_bins=3,
        ),
    )

    assert len(split.train) == 9
    assert len(split.val) == 1
    assert len(split.test) == 0
    assert set(split.train).isdisjoint(set(split.val))


def test_missing_perturbation_vector_uses_trainable_mean_initialization() -> None:
    adapter = PerturbationVectorAdapter(
        ["GENE1", "GENE2"],
        {"GENE1": np.asarray([1.0, 3.0], dtype=np.float32)},
        pert_dim=2,
    )

    missing = adapter("GENE2")

    assert missing.requires_grad
    np.testing.assert_allclose(missing.detach().numpy(), np.asarray([1.0, 3.0]))


def test_esm2_perturbation_adapter_maps_all_genes_through_one_network() -> None:
    table = Esm2EmbeddingTable(
        dim=3,
        vectors_by_symbol={
            "KNOWN": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "HELDOUT": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        },
    )
    adapter = Esm2PerturbationAdapter(
        ["KNOWN", "HELDOUT"], table, adapter_hidden=4, pert_dim=2
    )

    assert adapter("KNOWN").shape == (2,)
    assert adapter("HELDOUT").shape == (2,)
    assert adapter("HELDOUT").requires_grad
    assert not hasattr(adapter, "missing_vectors")


def test_esm2_perturbation_adapter_rejects_unresolved_gene() -> None:
    table = Esm2EmbeddingTable(dim=3, vectors_by_symbol={})

    with pytest.raises(ValueError, match="UNRESOLVED"):
        Esm2PerturbationAdapter(["UNRESOLVED"], table, adapter_hidden=4, pert_dim=2)


def test_external_only_perturbation_vector_uses_mean_initialization() -> None:
    adapter = PerturbationVectorAdapter(
        ["TRAIN1", "ADAMSON_ONLY"],
        {"TRAIN1": np.asarray([1.0, 0.0], dtype=np.float32)},
        pert_dim=2,
    )

    missing = adapter("ADAMSON_ONLY")

    assert missing.requires_grad
    assert not adapter.has_known_vector("ADAMSON_ONLY")
    assert adapter.has_known_vector("TRAIN1")
    np.testing.assert_allclose(missing.detach().numpy(), np.asarray([1.0, 0.0]))


def test_one_gene_step_leaves_other_missing_vectors_without_grad() -> None:
    adapter = PerturbationVectorAdapter(["GENE1", "GENE2"], {}, pert_dim=2)

    current = adapter("GENE1")
    other = adapter("GENE2")
    current.square().sum().backward()

    assert current.grad is not None
    assert other.grad is None


def test_state_pt_perturbation_map_loads_vectors(tmp_path: Path) -> None:
    path = tmp_path / "pert_onehot_map.pt"
    torch.save(
        {
            "GENE1": torch.tensor([1.0, 0.0]),
            "GENE2": torch.tensor([0.0, 1.0]),
        },
        path,
    )

    vectors = load_perturbation_vectors(path)

    assert set(vectors) == {"GENE1", "GENE2"}
    np.testing.assert_allclose(vectors["GENE2"], np.asarray([0.0, 1.0]))


def test_state_batch_lookup_encodes_gem_group_labels(tmp_path: Path) -> None:
    model_dir = tmp_path / "state_model"
    model_dir.mkdir()
    torch.save(
        {
            "31": torch.tensor([0.0, 1.0, 0.0]),
            "25": torch.tensor([1.0, 0.0, 0.0]),
        },
        model_dir / "batch_onehot_map.pt",
    )

    lookup = load_state_batch_lookup(model_dir)
    encoded = encode_batch_labels(np.asarray(["31", "missing", "25"]), lookup)

    assert lookup == {"31": 1, "25": 0}
    np.testing.assert_array_equal(encoded, np.asarray([1, 0, 0]))


def test_scvi_teacher_kwargs_reduce_lightning_warning_noise(monkeypatch) -> None:
    config = ProjectorConfig(scvi_num_workers=4)

    datasplitter_kwargs = _scvi_datasplitter_kwargs(config)
    trainer_kwargs = _scvi_trainer_kwargs(config)

    assert datasplitter_kwargs == {"num_workers": 4, "persistent_workers": True}
    assert trainer_kwargs["logger"] is False
    assert trainer_kwargs["enable_progress_bar"] is False
    assert trainer_kwargs["enable_model_summary"] is False

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    auto_kwargs = _scvi_datasplitter_kwargs(ProjectorConfig(scvi_num_workers=None))
    assert auto_kwargs == {"num_workers": 8, "persistent_workers": True}


def test_scvi_teacher_warning_context_filters_known_noise() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _suppress_scvi_lightning_warnings(ProjectorConfig()):
            warnings.warn(
                "The `srun` command is available on your system but is not used.",
                UserWarning,
            )
            warnings.warn(
                "adata.X does not contain unnormalized count data. "
                "Are you sure this is what you want?",
                UserWarning,
            )
            warnings.warn("unrelated warning", UserWarning)

    messages = [str(item.message) for item in caught]
    assert "unrelated warning" in messages
    assert not any("The `srun` command is available" in message for message in messages)
    assert not any(
        "adata.X does not contain unnormalized" in message for message in messages
    )


def test_scvi_projection_loads_teacher_once_for_all_datasets(tmp_path: Path) -> None:
    load_calls = []

    class FakeLoadedModel:
        def __init__(self, query: ad.AnnData) -> None:
            self.query = query

        def get_latent_representation(self, indices: np.ndarray) -> np.ndarray:
            matrix = np.asarray(self.query.X, dtype=np.float32)
            return matrix[np.asarray(indices, dtype=np.int64), :2] + 10.0

    class FakeSCVIModel:
        @staticmethod
        def load(model_dir: str, adata: ad.AnnData) -> FakeLoadedModel:
            load_calls.append((model_dir, int(adata.n_obs)))
            return FakeLoadedModel(adata)

    class FakeModelNamespace:
        SCVI = FakeSCVIModel

    class FakeScvi:
        model = FakeModelNamespace

    control = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    bags = (
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.asarray([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
    )
    external_control = np.asarray([[9.0, 10.0, 11.0]], dtype=np.float32)
    external_bags = (np.asarray([[12.0, 13.0, 14.0]], dtype=np.float32),)
    logs: list[str] = []

    projected = _project_scvi_latent_collections(
        FakeScvi,
        tmp_path / "teacher",
        (
            ("primary", control, bags, np.asarray(["G0", "G1", "G2"], dtype=object)),
            (
                "external:adamson",
                external_control,
                external_bags,
                np.asarray(["G0", "G1", "G2"], dtype=object),
            ),
        ),
        progress_interval=1,
        log_fn=logs.append,
    )
    control_latent, latent_bags = projected[0]
    external_control_latent, external_latent_bags = projected[1]

    assert len(load_calls) == 1
    assert load_calls[0][1] == 7
    np.testing.assert_allclose(control_latent, control[:, :2] + 10.0)
    np.testing.assert_allclose(latent_bags[0], bags[0][:, :2] + 10.0)
    np.testing.assert_allclose(latent_bags[1], bags[1][:, :2] + 10.0)
    np.testing.assert_allclose(
        external_control_latent,
        external_control[:, :2] + 10.0,
    )
    np.testing.assert_allclose(
        external_latent_bags[0],
        external_bags[0][:, :2] + 10.0,
    )
    assert logs[-1] == "Projected external:adamson scVI latents for 1/1 genes"


def test_scvi_projection_group_helper_returns_one_dataset(tmp_path: Path) -> None:
    class FakeLoadedModel:
        def __init__(self, query: ad.AnnData) -> None:
            self.query = query

        def get_latent_representation(self, indices: np.ndarray) -> np.ndarray:
            matrix = np.asarray(self.query.X, dtype=np.float32)
            return matrix[np.asarray(indices, dtype=np.int64), :1]

    class FakeSCVIModel:
        @staticmethod
        def load(_model_dir: str, adata: ad.AnnData) -> FakeLoadedModel:
            return FakeLoadedModel(adata)

    class FakeModelNamespace:
        SCVI = FakeSCVIModel

    class FakeScvi:
        model = FakeModelNamespace

    control_latent, latent_bags = _project_scvi_latent_groups(
        FakeScvi,
        tmp_path / "teacher",
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        (np.asarray([[2.0, 3.0]], dtype=np.float32),),
        None,
        progress_label="single",
    )

    np.testing.assert_allclose(control_latent, np.asarray([[0.0]], dtype=np.float32))
    np.testing.assert_allclose(latent_bags[0], np.asarray([[2.0]], dtype=np.float32))


def test_scvi_latent_cache_round_trips_valid_metadata(tmp_path: Path) -> None:
    config = load_config(_write_scvi_cache_config(tmp_path))
    data = _toy_gene_bags_with_batches()
    split = GeneSplit(
        train=np.asarray([0], dtype=np.int64),
        val=np.asarray([1], dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )
    projected = replace(
        data,
        control_latent=data.control_latent + 1.0,
        latent_bags=tuple(bag + 1.0 for bag in data.latent_bags),
    )
    artifacts_dir = tmp_path / "artifacts"

    _write_scvi_latent_cache(config, projected, split, artifacts_dir, None)
    loaded = _load_scvi_latent_cache(config, data, split, artifacts_dir, None)

    assert (_scvi_latent_cache_dir(artifacts_dir) / "COMPLETE").exists()
    assert loaded is not None
    loaded_data, loaded_external = loaded
    assert loaded_external is None
    np.testing.assert_allclose(loaded_data.control_latent, projected.control_latent)
    np.testing.assert_allclose(loaded_data.latent_bags[0], projected.latent_bags[0])


def test_scvi_latent_cache_rejects_incomplete_or_mismatched_metadata(
    tmp_path: Path,
) -> None:
    config = load_config(_write_scvi_cache_config(tmp_path))
    data = _toy_gene_bags_with_batches()
    split = GeneSplit(
        train=np.asarray([0], dtype=np.int64),
        val=np.asarray([1], dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )
    artifacts_dir = tmp_path / "artifacts"
    _write_scvi_latent_cache(config, data, split, artifacts_dir, None)
    cache_dir = _scvi_latent_cache_dir(artifacts_dir)

    (cache_dir / "COMPLETE").unlink()
    assert _load_scvi_latent_cache(config, data, split, artifacts_dir, None) is None

    (cache_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["latent_dim"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert _load_scvi_latent_cache(config, data, split, artifacts_dir, None) is None


def test_scvi_cache_metadata_binds_split_and_exact_fit_gene_authority(
    tmp_path: Path,
) -> None:
    config = load_config(_write_scvi_cache_config(tmp_path))
    data = _toy_gene_bags_with_batches()
    split = GeneSplit(
        train=np.asarray([0], dtype=np.int64),
        val=np.asarray([1], dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )

    first = _scvi_cache_metadata(
        config,
        data,
        split,
        None,
        authority=_toy_artifact_authority(),
    )
    changed_split = _scvi_cache_metadata(
        config,
        data,
        split,
        None,
        authority=_toy_artifact_authority(canonical_split_sha256="changed"),
    )
    changed_fit = _scvi_cache_metadata(
        config,
        data,
        split,
        None,
        authority=_toy_artifact_authority(fit_genes=("GENE1", "GENE4")),
    )

    assert first != changed_split
    assert first != changed_fit
    assert first["schema_version"] == 2
    assert first["fit_genes_sha256"]


def test_non_rank_scvi_path_requires_valid_latent_cache(tmp_path: Path) -> None:
    config = load_config(_write_scvi_cache_config(tmp_path))
    data = _toy_gene_bags_with_batches()
    split = GeneSplit(
        train=np.asarray([0], dtype=np.int64),
        val=np.asarray([1], dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )
    artifacts_dir = tmp_path / "artifacts"

    try:
        with_cached_scvi_teacher_latents(
            config,
            data,
            split,
            artifacts_dir,
            fit_teacher=False,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing scVI latent cache should fail on non-rank0 path")

    assert not _scvi_latent_cache_dir(artifacts_dir).exists()


def test_scvi_cache_subprocess_env_removes_distributed_state() -> None:
    env = train_module._isolated_scvi_cache_env(
        {
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "4",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "TORCHELASTIC_RUN_ID": "job",
            "ACCELERATE_USE_CPU": "false",
            "SLURM_PROCID": "1",
            "SLURM_NTASKS": "4",
            "SLURM_CPUS_PER_TASK": "32",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "PYTHONPATH": "src",
        }
    )

    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "ACCELERATE_USE_CPU",
        "SLURM_PROCID",
        "SLURM_NTASKS",
    ):
        assert key not in env
    assert env["SLURM_CPUS_PER_TASK"] == "32"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTHONPATH"] == "src"


def test_wait_for_scvi_latent_cache_retries_until_cache_is_valid(
    tmp_path: Path,
) -> None:
    config = load_config(_write_scvi_cache_config(tmp_path))
    data = _toy_gene_bags_with_batches()
    split = GeneSplit(
        train=np.asarray([0], dtype=np.int64),
        val=np.asarray([1], dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )
    artifacts_dir = tmp_path / "artifacts"
    sleeps: list[float] = []

    def write_cache_after_first_miss(seconds: float) -> None:
        sleeps.append(seconds)
        _write_scvi_latent_cache(config, data, split, artifacts_dir, None)

    loaded, external = train_module._wait_for_scvi_latent_cache(
        config,
        data,
        split,
        None,
        artifacts_dir,
        timeout_seconds=1.0,
        poll_seconds=0.01,
        sleep_fn=write_cache_after_first_miss,
    )

    assert sleeps
    assert external is None
    np.testing.assert_allclose(loaded.control_latent, data.control_latent)


def test_wait_for_scvi_latent_cache_timeout_includes_last_reason(
    tmp_path: Path,
) -> None:
    config = load_config(_write_scvi_cache_config(tmp_path))
    data = _toy_gene_bags_with_batches()
    split = GeneSplit(
        train=np.asarray([0], dtype=np.int64),
        val=np.asarray([1], dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )
    artifacts_dir = tmp_path / "artifacts"

    try:
        train_module._wait_for_scvi_latent_cache(
            config,
            data,
            split,
            None,
            artifacts_dir,
            timeout_seconds=0.0,
            poll_seconds=0.0,
        )
    except TimeoutError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing scVI latent cache should time out")

    assert str(artifacts_dir) in message
    assert "COMPLETE" in message


def test_rank0_scvi_orchestration_runs_subprocess_then_reads_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeAccelerator:
        is_main_process = True

        def __init__(self) -> None:
            self.barrier_count = 0

        def wait_for_everyone(self) -> None:
            self.barrier_count += 1

    config = load_config(_write_scvi_cache_config(tmp_path))
    data = _toy_gene_bags_with_batches()
    split = GeneSplit(
        train=np.asarray([0], dtype=np.int64),
        val=np.asarray([1], dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )
    accelerator = FakeAccelerator()
    calls: list[Path] = []

    def fake_subprocess(config_path: Path, artifacts_dir: Path) -> None:
        calls.append(config_path)
        _write_scvi_latent_cache(config, data, split, artifacts_dir, None)

    monkeypatch.setattr(train_module, "_run_scvi_cache_subprocess", fake_subprocess)

    loaded, external = train_module._with_rank_safe_scvi_teacher(
        config,
        data,
        split,
        None,
        tmp_path / "artifacts",
        accelerator,
        config_path=tmp_path / "config.yaml",
    )

    assert calls == [tmp_path / "config.yaml"]
    assert accelerator.barrier_count == 1
    assert external is None
    np.testing.assert_allclose(loaded.control_latent, data.control_latent)


def test_accelerator_uses_static_ddp_graph_without_unused_scan(tmp_path: Path) -> None:
    config = load_config(_write_scvi_cache_config(tmp_path))

    accelerator = train_module._make_accelerator(config)

    assert accelerator.ddp_handler is not None
    assert accelerator.ddp_handler.find_unused_parameters is False


def test_train_config_parses_gene_batch_size_and_defaults(tmp_path: Path) -> None:
    default_config = load_config(_write_scvi_cache_config(tmp_path))
    config_path = tmp_path / "gene_batch.yaml"
    raw = _write_scvi_cache_config(tmp_path).read_text(encoding="utf-8")
    raw = raw.replace(
        "train:\n",
        "loss:\n"
        "  pred_rank_weight: 5.0\n"
        "  pred_rank_tau: 0.25\n"
        "  pred_rank_pair_margin: 0.25\n"
        "  pred_rank_pair_weight_clip: 2.0\n"
        "  gmm_nll_weight: 0.01\n"
        "  b_loss_anneal_epochs: 5\n"
        "  b_loss_anneal_final_fraction: 0.1\n"
        "train:\n",
    )
    raw = raw.replace(
        "  device: cpu\n",
        "  device: cpu\n"
        "  gene_batch_size: 1\n"
        "  learning_rate: 0.000025\n"
        "  state_learning_rate: 0.0000025\n"
        "  max_grad_norm: 0.5\n"
        "  required_world_size: 4\n"
        "  freeze_state: true\n"
        "  input_tensor_cache_max_gib: 12.5\n",
    )
    config_path.write_text(raw, encoding="utf-8")

    parsed = load_config(config_path)

    assert default_config.train.gene_batch_size == 1
    assert default_config.train.learning_rate == 2.5e-5
    assert default_config.train.state_learning_rate == 2.5e-6
    assert default_config.train.max_grad_norm == 1.0
    assert default_config.train.required_world_size == 4
    assert default_config.train.freeze_state is False
    assert default_config.train.input_tensor_cache_max_gib == 24.0
    assert default_config.loss.pred_rank_weight == 0.0
    assert default_config.loss.gmm_nll_weight == 0.0
    assert default_config.loss.b_loss_anneal_epochs == 0
    assert parsed.train.gene_batch_size == 1
    assert parsed.train.learning_rate == 2.5e-5
    assert parsed.train.state_learning_rate == 2.5e-6
    assert parsed.train.max_grad_norm == 0.5
    assert parsed.train.required_world_size == 4
    assert parsed.train.freeze_state is True
    assert parsed.train.input_tensor_cache_max_gib == 12.5
    assert parsed.loss.pred_rank_weight == 5.0
    assert parsed.loss.pred_rank_tau == 0.25
    assert parsed.loss.pred_rank_pair_margin == 0.25
    assert parsed.loss.pred_rank_pair_weight_clip == 2.0
    assert parsed.loss.gmm_nll_weight == 0.01
    assert train_module._loss_weights(parsed).gmm_nll == 0.01
    assert parsed.loss.b_loss_anneal_epochs == 5
    assert parsed.loss.b_loss_anneal_final_fraction == 0.1


def test_train_config_accepts_positive_legacy_world_and_batch_sizes(
    tmp_path: Path,
) -> None:
    config_path = _write_scvi_cache_config(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  device: cpu\n",
            "  device: cpu\n  required_world_size: 1\n  gene_batch_size: 16\n",
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.train.required_world_size == 1
    assert config.train.gene_batch_size == 16


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"state_learning_rate": 0.0}, "state_learning_rate must be positive"),
        (
            {"learning_rate": 2.5e-5, "state_learning_rate": 2.5e-4},
            "state_learning_rate must not exceed learning_rate",
        ),
        ({"max_grad_norm": 0.0}, "max_grad_norm must be positive"),
        ({"required_world_size": 0}, "required_world_size must be positive"),
        ({"gene_batch_size": 0}, "gene_batch_size must be positive"),
    ],
)
def test_train_config_rejects_invalid_e2e_settings(
    tmp_path: Path,
    settings: dict[str, float | int],
    message: str,
) -> None:
    config_path = _write_scvi_cache_config(tmp_path)
    lines = "".join(f"  {key}: {value}\n" for key, value in settings.items())
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  device: cpu\n",
            f"  device: cpu\n{lines}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_response_encoder_config_is_optional_for_legacy_configs(tmp_path: Path) -> None:
    legacy_path = _write_scvi_cache_config(tmp_path)
    legacy = load_config(legacy_path)
    configured_path = tmp_path / "response_encoder.yaml"
    configured_path.write_text(
        legacy_path.read_text(encoding="utf-8").replace(
            "projector:\n",
            "response_encoder:\n"
            "  input_dim: 2000\n"
            "  latent_dim: 128\n"
            "projector:\n",
            1,
        ),
        encoding="utf-8",
    )

    configured = load_config(configured_path)

    assert legacy.response_encoder is None
    assert configured.response_encoder == ResponseEncoderConfig(2000, 128)


@pytest.mark.parametrize(
    "response_encoder",
    [
        None,
        ResponseEncoderConfig(1999, 128),
        ResponseEncoderConfig(2000, 127),
    ],
)
def test_audited_e2e_builder_requires_locked_response_encoder(
    tmp_path: Path,
    response_encoder: ResponseEncoderConfig | None,
) -> None:
    config = replace(
        load_config(_write_scvi_cache_config(tmp_path)),
        response_encoder=response_encoder,
    )

    with pytest.raises(ValueError, match="input_dim=2000.*latent_dim=128"):
        train_module._build_e2e_model(
            config,
            _toy_gene_bags_with_batches(),
            extra_genes=(),
            canonical_gene_order=("GENE1", "GENE2"),
            emit_checkpoint_output=False,
        )


def test_audited_e2e_builder_rejects_nontrainable_gmm(tmp_path: Path) -> None:
    config_path = _write_scvi_cache_config(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "train:\n",
            "gmm:\n  trainable: false\ntrain:\n",
            1,
        ),
        encoding="utf-8",
    )
    config = replace(
        load_config(config_path),
        response_encoder=ResponseEncoderConfig(2000, 128),
    )
    assert config.gmm.trainable is False

    with pytest.raises(ValueError, match="requires trainable GMM"):
        train_module._build_e2e_model(
            config,
            _toy_gene_bags_with_batches(),
            extra_genes=(),
            canonical_gene_order=("GENE1", "GENE2"),
            emit_checkpoint_output=False,
        )


def test_state_config_parses_strict_esm2_fields(tmp_path: Path) -> None:
    default_config = load_config(_write_scvi_cache_config(tmp_path))
    config_path = tmp_path / "esm2.yaml"
    raw = _write_scvi_cache_config(tmp_path).read_text(encoding="utf-8")
    raw = raw.replace(
        "  pert_dim: 2\n",
        "  pert_dim: 2\n"
        "  gene_tokenizer: esm2\n"
        f"  esm2_npz: {tmp_path / 'esm2.npz'}\n"
        "  esm2_adapter_hidden: 16\n"
        "  require_resolved_esm2: true\n",
    )
    config_path.write_text(raw, encoding="utf-8")

    parsed = load_config(config_path)

    assert default_config.state.gene_tokenizer == "state_onehot"
    assert default_config.state.esm2_npz is None
    assert default_config.state.esm2_adapter_hidden == 512
    assert default_config.state.require_resolved_esm2 is False
    assert parsed.state.gene_tokenizer == "esm2"
    assert parsed.state.esm2_npz == tmp_path / "esm2.npz"
    assert parsed.state.esm2_adapter_hidden == 16
    assert parsed.state.require_resolved_esm2 is True


def test_state_config_ignores_legacy_noop_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy_state_fields.yaml"
    config_path.write_text(
        f"""
data:
  h5ad_path: {tmp_path / "unused.h5ad"}
  overlap_csv: {tmp_path / "unused.csv"}
  output_dir: {tmp_path / "outputs"}
  state_embed_key: X_hvg
state:
  backend: state_checkpoint
  model_dir: {tmp_path / "state_model"}
  checkpoint_path: {tmp_path / "state.ckpt"}
  embed_key: ignored
  input_dim: 3
  output_dim: 3
  pert_dim: 2
  hidden_dim: 768
  cell_set_len: 256
  allow_mock: true
train:
  cell_set_len: 32
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.state_embed_key == "X_hvg"
    assert config.train.cell_set_len == 32
    for field in ("embed_key", "hidden_dim", "cell_set_len", "allow_mock"):
        assert not hasattr(config.state, field)


def test_padded_gene_loader_marks_padding_for_even_ddp_steps() -> None:
    padded, is_padding = train_module._pad_gene_indices(
        np.arange(5, dtype=np.int64),
        batch_size=4,
        world_size=2,
    )

    assert padded.tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
    assert is_padding.tolist() == [False, False, False, False, False, True, True, True]

    loader = train_module._gene_loader(
        np.arange(5, dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=4,
        world_size=2,
    )
    batches = list(loader)

    assert [batch["index"].tolist() for batch in batches] == [
        [0, 1, 2, 3],
        [4, 0, 1, 2],
    ]
    assert [batch["is_padding"].tolist() for batch in batches] == [
        [False, False, False, False],
        [False, True, True, True],
    ]


def test_cost_balanced_sampler_groups_similar_work_and_reshuffles_epochs() -> None:
    costs = np.asarray([1, 10, 2, 9, 3, 8, 4, 7], dtype=np.int64)
    sampler = train_module._CostBalancedSampler(costs, world_size=4, seed=17)

    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert sorted(first) == list(range(len(costs)))
    assert sorted(second) == list(range(len(costs)))
    assert first != second
    for start in range(0, len(first), 4):
        assert np.ptp(costs[first[start : start + 4]]) <= 3


def test_gene_chunk_costs_match_condition_chunking() -> None:
    data = _toy_gene_bags_with_batches()
    cell_set_len = 2

    costs = train_module._gene_chunk_costs(data, cell_set_len)
    actual = [
        len(
            make_cell_set_chunks(
                data,
                gene_index,
                cell_set_len=cell_set_len,
                rng=np.random.default_rng(3),
                pad_short=True,
                shuffle=True,
            )
        )
        for gene_index in range(len(data.genes))
    ]

    assert costs.tolist() == actual


def test_model_inputs_tensor_cache_matches_uncached_tensors_and_losses() -> None:
    data = _toy_gene_bags_with_batches()
    batch_lookup = {"batch_a": 3, "batch_b": 4}
    device = torch.device("cpu")
    uncached_inputs, uncached_fallbacks, uncached_chunks = (
        train_module._model_inputs_for_indices(
            data,
            [0, 1],
            np.random.default_rng(17),
            cell_set_len=3,
            device=device,
            batch_lookup=batch_lookup,
            pad_short=True,
        )
    )
    cache = train_module._InputTensorCache.build(
        data,
        batch_lookup,
        device,
    )
    assert cache.control_batch is not None
    torch.testing.assert_close(
        cache.control_batch,
        torch.as_tensor([3, 3, 4], dtype=torch.long),
    )
    cached_inputs, cached_fallbacks, cached_chunks = (
        train_module._model_inputs_for_indices(
            data,
            [0, 1],
            np.random.default_rng(17),
            cell_set_len=3,
            device=device,
            batch_lookup=batch_lookup,
            pad_short=True,
            tensor_cache=cache,
        )
    )

    _assert_nested_tensors_close(cached_inputs, uncached_inputs)
    torch.testing.assert_close(cached_fallbacks, uncached_fallbacks)
    torch.testing.assert_close(cached_chunks, uncached_chunks)

    model, _state_model = _build_two_gene_aivc_model()
    with torch.no_grad():
        uncached_losses = model(weights=_loss_weights(), **uncached_inputs)
        cached_losses = model(weights=_loss_weights(), **cached_inputs)

    for key in uncached_losses:
        torch.testing.assert_close(cached_losses[key], uncached_losses[key])


def test_model_inputs_tensor_cache_respects_estimated_size_cap() -> None:
    data = _toy_gene_bags_with_batches()
    batch_lookup = {"batch_a": 3, "batch_b": 4}
    estimated_bytes = train_module._InputTensorCache.estimate_bytes(
        data,
        batch_lookup,
    )

    assert estimated_bytes > 1
    assert (
        train_module._InputTensorCache.maybe_build(
            data,
            batch_lookup,
            torch.device("cpu"),
            max_bytes=estimated_bytes - 1,
            allow_cpu=True,
        )
        is None
    )


def test_run_epoch_sums_multi_gene_losses_in_one_optimizer_step(monkeypatch) -> None:
    class BatchLossModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(
            self,
            *,
            gene: tuple[str, ...],
            control_chunks: tuple[tuple[torch.Tensor, ...], ...],
            target_expression_chunks: tuple[tuple[torch.Tensor, ...], ...],
            target_latent_chunks: tuple[tuple[torch.Tensor, ...], ...],
            batch_index_chunks: tuple[tuple[torch.Tensor | None, ...], ...],
            y: torch.Tensor,
            weights: LossWeights,
            gene_mask: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            batch_size = len(gene)
            del (
                gene,
                control_chunks,
                target_expression_chunks,
                target_latent_chunks,
                batch_index_chunks,
                y,
                weights,
                gene_mask,
            )
            values = torch.arange(
                1,
                batch_size + 1,
                dtype=torch.float32,
                device=self.param.device,
            )
            total_losses = self.param * values
            return {
                "total": total_losses.sum(),
                "hvg_mean_delta": total_losses.mean(),
                "hvg_energy": total_losses.mean(),
                "latent_mean_delta": total_losses.mean(),
                "latent_energy": total_losses.mean(),
                "pred_c": total_losses.mean(),
                "obs_c": total_losses.mean(),
                "occupancy": total_losses.mean(),
                "pred_rank": total_losses.mean(),
                "pred_y": values,
                "obs_y": values,
                "per_gene_total_loss": total_losses,
                "per_gene_hvg_mean_delta": values,
                "per_gene_hvg_energy": values,
                "per_gene_latent_mean_delta": values,
                "per_gene_latent_energy": values,
                "per_gene_pred_c": values,
                "per_gene_obs_c": values,
                "per_gene_occupancy": values,
                "per_gene_pred_rank": values,
            }

    class CountingOptimizer(torch.optim.SGD):
        def __init__(self, params: list[torch.nn.Parameter]) -> None:
            super().__init__(params, lr=0.1)
            self.step_count = 0
            self.last_grad = math.nan

        def step(self, closure=None):  # type: ignore[override]
            self.step_count += 1
            self.last_grad = float(model.param.grad.detach())
            return None

    del monkeypatch
    model = BatchLossModel()
    accelerator = train_module.Accelerator(cpu=True)
    optimizer = CountingOptimizer([model.param])
    loader = train_module._gene_loader(
        np.arange(2, dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=2,
        world_size=1,
    )

    row = train_module._run_epoch(
        model,
        _toy_gene_bags_with_batches(),
        loader,
        _loss_weights(),
        optimizer,
        np.random.default_rng(1),
        2,
        accelerator,
        {},
        epoch=1,
        max_epochs=1,
        max_grad_norm=10.0,
    )

    assert optimizer.step_count == 1
    assert optimizer.last_grad == 3.0
    assert row["total_loss"] == 1.5


def test_gradient_clipping_is_called_before_optimizer_step(monkeypatch) -> None:
    events: list[str] = []
    clipped_norms: list[float] = []

    class RecordingOptimizer(torch.optim.SGD):
        def step(self, closure=None):  # type: ignore[override]
            events.append("step")
            return super().step(closure)

    model, _state_model = _build_tiny_aivc_model()
    accelerator = train_module.Accelerator(cpu=True)
    original_clip = accelerator.clip_grad_norm_

    def record_clip(
        parameters: object,
        max_norm: float,
    ) -> torch.Tensor:
        events.append("clip")
        clipped_norms.append(max_norm)
        return original_clip(parameters, max_norm)  # type: ignore[arg-type]

    monkeypatch.setattr(accelerator, "clip_grad_norm_", record_clip)
    optimizer = RecordingOptimizer(model.parameters(), lr=0.1)
    loader = train_module._gene_loader(
        np.asarray([0], dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=1,
        world_size=1,
    )

    train_module._run_epoch(
        model,
        _toy_gene_bags_with_batches(),
        loader,
        _loss_weights(),
        optimizer,
        np.random.default_rng(1),
        2,
        accelerator,
        {},
        epoch=1,
        max_epochs=1,
        max_grad_norm=0.25,
    )

    assert events.index("clip") < events.index("step")
    assert clipped_norms == [0.25]


def test_run_epoch_zero_weights_padding_for_loss_metrics_and_count() -> None:
    class MaskedBatchLossModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))
            self.seen_masks: list[list[bool]] = []

        def forward(
            self,
            *,
            gene: tuple[str, ...],
            control_chunks: tuple[tuple[torch.Tensor, ...], ...],
            target_expression_chunks: tuple[tuple[torch.Tensor, ...], ...],
            target_latent_chunks: tuple[tuple[torch.Tensor, ...], ...],
            batch_index_chunks: tuple[tuple[torch.Tensor | None, ...], ...],
            y: torch.Tensor,
            weights: LossWeights,
            gene_mask: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            del (
                gene,
                control_chunks,
                target_expression_chunks,
                target_latent_chunks,
                batch_index_chunks,
                y,
                weights,
            )
            assert gene_mask is not None
            self.seen_masks.append(gene_mask.detach().cpu().tolist())
            values = torch.arange(
                1,
                len(gene_mask) + 1,
                dtype=torch.float32,
                device=self.param.device,
            )
            per_gene_total = self.param * values
            total = per_gene_total[gene_mask].sum()
            return {
                "total": total,
                "hvg_mean_delta": per_gene_total[gene_mask].mean(),
                "hvg_energy": per_gene_total[gene_mask].mean(),
                "latent_mean_delta": per_gene_total[gene_mask].mean(),
                "latent_energy": per_gene_total[gene_mask].mean(),
                "pred_c": per_gene_total[gene_mask].mean(),
                "obs_c": per_gene_total[gene_mask].mean(),
                "occupancy": per_gene_total[gene_mask].mean(),
                "pred_rank": per_gene_total[gene_mask].mean(),
                "pred_y": values,
                "obs_y": values,
                "per_gene_total_loss": per_gene_total,
                "per_gene_hvg_mean_delta": values,
                "per_gene_hvg_energy": values,
                "per_gene_latent_mean_delta": values,
                "per_gene_latent_energy": values,
                "per_gene_pred_c": values,
                "per_gene_obs_c": values,
                "per_gene_occupancy": values,
                "per_gene_pred_rank": values,
            }

    class CountingOptimizer(torch.optim.SGD):
        def __init__(self, params: list[torch.nn.Parameter]) -> None:
            super().__init__(params, lr=0.1)
            self.last_grad = math.nan

        def step(self, closure=None):  # type: ignore[override]
            self.last_grad = float(model.param.grad.detach())
            return None

    model = MaskedBatchLossModel()
    accelerator = train_module.Accelerator(cpu=True)
    optimizer = CountingOptimizer([model.param])
    loader = train_module._gene_loader(
        np.asarray([0], dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=2,
        world_size=1,
    )

    row = train_module._run_epoch(
        model,
        _toy_gene_bags_with_batches(),
        loader,
        _loss_weights(),
        optimizer,
        np.random.default_rng(1),
        2,
        accelerator,
        {},
        epoch=1,
        max_epochs=1,
        max_grad_norm=10.0,
    )

    assert model.seen_masks == [[True, False]]
    assert optimizer.last_grad == 1.0
    assert row["total_loss"] == 1.0
    assert row["hvg_mean_delta"] == 1.0


def test_run_epoch_uses_one_model_forward_for_gene_batch(monkeypatch) -> None:
    class BatchOnlyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))
            self.calls: list[tuple[str, ...]] = []

        def forward(
            self,
            *,
            gene: tuple[str, ...],
            control_chunks: tuple[tuple[torch.Tensor, ...], ...],
            target_expression_chunks: tuple[tuple[torch.Tensor, ...], ...],
            target_latent_chunks: tuple[tuple[torch.Tensor, ...], ...],
            batch_index_chunks: tuple[tuple[torch.Tensor | None, ...], ...],
            y: torch.Tensor,
            weights: LossWeights,
            gene_mask: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            del (
                control_chunks,
                target_expression_chunks,
                target_latent_chunks,
                batch_index_chunks,
                y,
                weights,
                gene_mask,
            )
            if isinstance(gene, str):
                raise AssertionError("expected one batched forward call")
            self.calls.append(gene)
            values = torch.arange(
                1,
                len(gene) + 1,
                dtype=torch.float32,
                device=self.param.device,
            )
            total_losses = self.param * values
            return {
                "total": total_losses.sum(),
                "hvg_mean_delta": total_losses.mean(),
                "hvg_energy": total_losses.mean(),
                "latent_mean_delta": total_losses.mean(),
                "latent_energy": total_losses.mean(),
                "pred_c": total_losses.mean(),
                "obs_c": total_losses.mean(),
                "occupancy": total_losses.mean(),
                "pred_rank": total_losses.mean(),
                "pred_y": values,
                "obs_y": values,
                "per_gene_total_loss": total_losses,
                "per_gene_hvg_mean_delta": values,
                "per_gene_hvg_energy": values,
                "per_gene_latent_mean_delta": values,
                "per_gene_latent_energy": values,
                "per_gene_pred_c": values,
                "per_gene_obs_c": values,
                "per_gene_occupancy": values,
                "per_gene_pred_rank": values,
            }

    class CountingOptimizer(torch.optim.SGD):
        def __init__(self, params: list[torch.nn.Parameter]) -> None:
            super().__init__(params, lr=0.1)
            self.step_count = 0

        def step(self, closure=None):  # type: ignore[override]
            self.step_count += 1
            return None

    def fail_object_gather(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("object gather should not be used for epoch metrics")

    monkeypatch.setattr(
        train_module,
        "gather_object",
        fail_object_gather,
        raising=False,
    )
    model = BatchOnlyModel()
    optimizer = CountingOptimizer([model.param])
    accelerator = train_module.Accelerator(cpu=True)
    loader = train_module._gene_loader(
        np.asarray([0, 1], dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=2,
        world_size=1,
    )

    row = train_module._run_epoch(
        model,
        _toy_gene_bags_with_batches(),
        loader,
        _loss_weights(),
        optimizer,
        np.random.default_rng(1),
        2,
        accelerator,
        {},
        epoch=1,
        max_epochs=1,
    )

    assert model.calls == [("GENE1", "GENE2")]
    assert optimizer.step_count == 1
    assert row["total_loss"] == 1.5


def test_train_and_evaluate_reuse_one_control_batch_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _toy_gene_bags_with_batches()
    lookup = prepare_module._control_indices_by_batch(data)
    build_calls: list[GeneBags] = []
    observed_lookups: list[dict[str, np.ndarray] | None] = []
    original_inputs = train_module._model_inputs_for_indices

    def build_lookup(current: GeneBags) -> dict[str, np.ndarray]:
        build_calls.append(current)
        return lookup

    def capture_lookup(*args: object, **kwargs: object):
        observed_lookups.append(kwargs.get("control_indices_by_batch"))
        return original_inputs(*args, **kwargs)

    monkeypatch.setattr(train_module, "_control_indices_by_batch", build_lookup)
    monkeypatch.setattr(train_module, "_model_inputs_for_indices", capture_lookup)
    accelerator = train_module.Accelerator(cpu=True)
    model, _state_model = _build_tiny_aivc_model()
    loader = train_module._gene_loader(
        np.asarray([0, 0], dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=1,
        world_size=1,
    )
    train_module._run_epoch(
        model,
        data,
        loader,
        _loss_weights(),
        torch.optim.SGD(model.parameters(), lr=0.01),
        np.random.default_rng(1),
        2,
        accelerator,
        {"batch_a": 0, "batch_b": 1, "batch_z": 2},
        epoch=1,
        max_epochs=1,
    )

    assert build_calls == [data]
    assert len(observed_lookups) == 2
    assert all(current is lookup for current in observed_lookups)

    build_calls.clear()
    observed_lookups.clear()
    eval_loader = train_module._gene_loader(
        np.asarray([0, 0], dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=1,
        world_size=1,
    )
    train_module._evaluate(
        model,
        data,
        eval_loader,
        _loss_weights(),
        np.random.default_rng(1),
        2,
        accelerator,
        {"batch_a": 0, "batch_b": 1, "batch_z": 2},
        pad_short=False,
    )

    assert build_calls == [data]
    assert len(observed_lookups) == 2
    assert all(current is lookup for current in observed_lookups)


def test_evaluate_filters_padding_rows_after_forward() -> None:
    model, _state_model = _build_tiny_aivc_model()
    data = _toy_gene_bags_with_batches()
    accelerator = train_module.Accelerator(cpu=True)
    loader = train_module._gene_loader(
        np.asarray([0], dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=2,
        world_size=1,
    )

    summary, predictions = train_module._evaluate(
        model,
        data,
        loader,
        _loss_weights(),
        np.random.default_rng(1),
        2,
        accelerator,
        {},
        pad_short=True,
    )

    assert predictions["perturbation_gene"].tolist() == ["GENE1"]
    assert len(predictions) == 1
    assert math.isfinite(summary["total_loss"])


def test_evaluate_uses_tensor_gather_and_filters_padding(monkeypatch) -> None:
    def fail_object_gather(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("object gather should not be used for eval metrics")

    monkeypatch.setattr(
        train_module,
        "gather_object",
        fail_object_gather,
        raising=False,
    )
    model, _state_model = _build_tiny_aivc_model()
    data = _toy_gene_bags_with_batches()
    accelerator = train_module.Accelerator(cpu=True)
    loader = train_module._gene_loader(
        np.asarray([0], dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=2,
        world_size=1,
    )

    summary, predictions = train_module._evaluate(
        model,
        data,
        loader,
        _loss_weights(),
        np.random.default_rng(1),
        2,
        accelerator,
        {},
        pad_short=True,
    )

    assert predictions["perturbation_gene"].tolist() == ["GENE1"]
    assert len(predictions) == 1
    assert math.isfinite(summary["total_loss"])


def test_final_prediction_only_uses_all_control_chunks_and_control_batches(
    monkeypatch,
) -> None:
    class RecordingState(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))
            self.chunk_sizes: list[int] = []
            self.batch_values: list[list[int]] = []
            self.pert_names: list[list[str]] = []

        def forward(
            self,
            batch: dict[str, torch.Tensor],
            padded: bool = False,
        ) -> torch.Tensor:
            del padded
            control = batch["ctrl_cell_emb"]
            self.chunk_sizes.append(int(control.shape[0]))
            self.batch_values.append(batch["batch"].detach().cpu().tolist())
            self.pert_names.append(list(batch["pert_name"]))
            return control @ self.weight

    def fail_target_chunking(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("final prediction must not inspect target B chunks")

    model, _state_model = _build_two_gene_aivc_model()
    recording_state = RecordingState()
    model.state_adapter = StateForwardAdapter(recording_state)
    data = replace(
        _toy_gene_bags_with_batches(),
        control_input=np.arange(15, dtype=np.float32).reshape(5, 3),
        control_batch=np.asarray(
            ["batch_a", "batch_b", "batch_a", "batch_b", "batch_c"],
            dtype=object,
        ),
    )
    accelerator = train_module.Accelerator(cpu=True)
    loader = train_module._gene_loader(
        np.asarray([0, 1], dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=2,
        world_size=1,
    )
    monkeypatch.setattr(train_module, "make_cell_set_chunks", fail_target_chunking)

    summary, predictions = train_module._evaluate_prediction_only_final(
        model,
        data,
        loader,
        cell_set_len=2,
        accelerator=accelerator,
        batch_lookup={"batch_a": 3, "batch_b": 4, "batch_c": 5},
    )

    assert recording_state.chunk_sizes == [5, 5]
    assert recording_state.batch_values == [[3, 4, 3, 4, 5], [3, 4, 3, 4, 5]]
    assert recording_state.pert_names == [["GENE1"] * 5, ["GENE2"] * 5]
    assert predictions["perturbation_gene"].tolist() == ["GENE1", "GENE2"]
    assert predictions["n_chunks"].tolist() == [1.0, 1.0]
    assert "y_obs_anchor" not in predictions.columns
    assert set(summary) >= {"rmse", "spearman"}


def test_final_prediction_only_matches_chunked_predict_response() -> None:
    model, _state_model = _build_counting_aivc_model()
    data = replace(
        _toy_gene_bags_with_batches(),
        control_input=np.arange(15, dtype=np.float32).reshape(5, 3),
        control_batch=np.asarray(
            ["batch_a", "batch_b", "batch_a", "batch_b", "batch_c"],
            dtype=object,
        ),
    )
    batch_lookup = {"batch_a": 3, "batch_b": 4, "batch_c": 5}
    accelerator = train_module.Accelerator(cpu=True)
    loader = train_module._gene_loader(
        np.asarray([0], dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=1,
        world_size=1,
    )
    predicted_expression_chunks = []
    control_chunks = []
    with torch.no_grad():
        for start in range(0, data.control_input.shape[0], 2):
            end = min(start + 2, data.control_input.shape[0])
            control_chunk = torch.as_tensor(
                data.control_input[start:end],
                dtype=torch.float32,
            )
            batch_indices = train_module._batch_tensor(
                data.control_batch[start:end],
                batch_lookup,
                torch.device("cpu"),
            )
            predicted_expression, _predicted_latent = model.predict_response(
                control_chunk,
                "GENE1",
                batch_indices,
            )
            predicted_expression_chunks.append(predicted_expression)
            control_chunks.append(control_chunk)
        expected = model.predict_c_from_response(
            torch.cat(predicted_expression_chunks, dim=0),
            torch.cat(control_chunks, dim=0),
        )

        _summary, predictions = train_module._evaluate_prediction_only_final(
            model,
            data,
            loader,
            cell_set_len=2,
            accelerator=accelerator,
            batch_lookup=batch_lookup,
        )

    assert predictions["n_chunks"].tolist() == [1.0]
    np.testing.assert_allclose(
        predictions["y_pred"].to_numpy(dtype=np.float32),
        np.asarray([float(expected.detach().cpu().item())], dtype=np.float32),
    )


def test_final_prediction_only_unwraps_prepared_model() -> None:
    class WrappedModel(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, *args: object, **kwargs: object) -> object:
            return self.module(*args, **kwargs)

    class FakeAccelerator:
        is_main_process = True
        device = torch.device("cpu")

        def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
            assert isinstance(model, WrappedModel)
            return model.module

        def gather(self, value: torch.Tensor) -> torch.Tensor:
            return value

    model, _state_model = _build_tiny_aivc_model()
    wrapped = WrappedModel(model)
    data = _toy_gene_bags_with_batches()
    loader = train_module._gene_loader(
        np.asarray([0], dtype=np.int64),
        shuffle=False,
        seed=1,
        gene_batch_size=1,
        world_size=1,
    )

    _summary, predictions = train_module._evaluate_prediction_only_final(
        wrapped,
        data,
        loader,
        cell_set_len=2,
        accelerator=FakeAccelerator(),  # type: ignore[arg-type]
        batch_lookup={},
    )

    assert predictions["perturbation_gene"].tolist() == ["GENE1"]


def test_gpu_peak_memory_uses_tensor_gather(monkeypatch) -> None:
    def fail_object_gather(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("object gather should not be used for gpu memory")

    monkeypatch.setattr(
        train_module,
        "gather_object",
        fail_object_gather,
        raising=False,
    )
    accelerator = train_module.Accelerator(cpu=True)

    value = train_module._global_peak_gpu_memory_allocated_mb(accelerator)

    assert math.isnan(value)


def test_metric_selection_maximizes_spearman() -> None:
    assert train_module._is_better_metric(0.2, 0.1, mode="max")
    assert not train_module._is_better_metric(0.1, 0.2, mode="max")
    assert not train_module._is_better_metric(math.nan, 0.2, mode="max")


def test_train_val_chunks_cover_cells_and_pad_short_chunk() -> None:
    data = _toy_gene_bags_with_batches()
    rng = np.random.default_rng(3)

    chunks = make_cell_set_chunks(
        data,
        0,
        cell_set_len=3,
        rng=rng,
        pad_short=True,
        shuffle=True,
    )

    assert [len(chunk.target_indices) for chunk in chunks] == [3, 3]
    covered = set(np.concatenate([chunk.target_indices for chunk in chunks]).tolist())
    assert covered == {0, 1, 2, 3}


def test_final_test_chunks_are_variable_length_without_padding() -> None:
    data = _toy_gene_bags_with_batches()
    rng = np.random.default_rng(3)

    chunks = make_cell_set_chunks(
        data,
        0,
        cell_set_len=3,
        rng=rng,
        pad_short=False,
        shuffle=True,
    )

    assert sorted(len(chunk.target_indices) for chunk in chunks) == [2, 2]
    covered = set(np.concatenate([chunk.target_indices for chunk in chunks]).tolist())
    assert covered == {0, 1, 2, 3}


def test_batch_matched_control_sampling_and_fallback() -> None:
    data = _toy_gene_bags_with_batches()
    rng = np.random.default_rng(5)

    chunks = make_cell_set_chunks(
        data,
        1,
        cell_set_len=3,
        rng=rng,
        pad_short=True,
        shuffle=False,
    )

    first, second = chunks
    assert first.control_fallback_count == 0
    assert set(data.control_batch[first.control_indices]) == {"batch_a"}
    assert second.control_fallback_count == 3


def test_control_sampling_scans_once_per_unique_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _toy_gene_bags_with_batches()
    original_flatnonzero = np.flatnonzero
    scanned: list[np.ndarray] = []

    def counting_flatnonzero(values: np.ndarray) -> np.ndarray:
        scanned.append(values)
        return original_flatnonzero(values)

    monkeypatch.setattr(prepare_module.np, "flatnonzero", counting_flatnonzero)
    selected, fallback_count = prepare_module._sample_control_indices(
        data,
        n_rows=4,
        rng=np.random.default_rng(5),
        target_batch=np.asarray(
            ["batch_a", "batch_a", "batch_z", "batch_z"],
            dtype=object,
        ),
    )

    assert len(scanned) == 2
    assert fallback_count == 2
    assert set(data.control_batch[selected[:2]]) == {"batch_a"}


def test_fixed_gmm_featureizer_is_differentiable() -> None:
    bag_a = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=np.float32)
    bag_b = np.asarray([[1.0, 1.0], [1.1, 1.0], [1.0, 1.1]], dtype=np.float32)
    featureizer = fit_fixed_gmm(
        (bag_a, bag_b),
        bag_a,
        n_components=2,
        covariance_floor=1e-4,
        random_state=5,
        max_fit_cells=None,
    )
    x = torch.tensor(bag_b, dtype=torch.float32, requires_grad=True)

    feature = featureizer(x)
    feature.sum().backward()

    assert torch.isfinite(feature).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_fixed_gmm_featureizer_matches_sklearn_weighted_responsibilities() -> None:
    control = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [-0.1, 0.05],
            [0.05, -0.05],
            [0.0, 0.1],
            [-0.05, -0.1],
        ],
        dtype=np.float32,
    )
    major_bag = np.asarray(
        [
            [0.2, 0.0],
            [0.15, 0.1],
            [-0.2, 0.0],
            [0.0, -0.2],
            [0.1, -0.15],
            [-0.15, 0.1],
        ],
        dtype=np.float32,
    )
    minor_bag = np.asarray(
        [[4.0, 4.0], [4.2, 3.9], [3.8, 4.1]],
        dtype=np.float32,
    )
    test_bag = np.asarray(
        [[0.0, 0.0], [4.1, 4.0], [2.0, 2.0]],
        dtype=np.float32,
    )
    featureizer = fit_fixed_gmm(
        (major_bag, minor_bag),
        control,
        n_components=2,
        covariance_floor=1e-6,
        random_state=17,
        max_fit_cells=None,
    )
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="diag",
        random_state=17,
        reg_covar=1e-6,
    )
    gmm.fit(np.vstack([control, major_bag, minor_bag]).astype(np.float32))

    actual = featureizer._occupancy(torch.as_tensor(test_bag)).detach().numpy()
    expected = gmm.predict_proba(test_bag).mean(axis=0)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_projector_and_gmm_fit_caches_reuse_and_recompute_on_metadata_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(_write_scvi_cache_config(tmp_path))
    data = _toy_gene_bags_with_batches()
    split = GeneSplit(
        train=np.asarray([0], dtype=np.int64),
        val=np.asarray([1], dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )
    artifacts_dir = tmp_path / "artifacts"
    projector_calls: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []
    gmm_calls: list[int] = []

    def fake_fit_linear_projector(
        train_expr: np.ndarray,
        train_latent: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        projector_calls.append((train_expr.shape, train_latent.shape, float(alpha)))
        value = float(len(projector_calls))
        return (
            np.full(
                (train_expr.shape[1], train_latent.shape[1]),
                value,
                dtype=np.float32,
            ),
            np.full(train_latent.shape[1], value, dtype=np.float32),
        )

    def fake_fit_fixed_gmm(
        bags: tuple[np.ndarray, ...],
        control_bag: np.ndarray,
        *,
        n_components: int,
        covariance_floor: float,
        random_state: int,
        max_fit_cells: int | None,
    ) -> model_module.FixedGMMFeatureizer:
        del bags, n_components, covariance_floor, random_state, max_fit_cells
        gmm_calls.append(1)
        value = float(len(gmm_calls))
        latent_dim = int(control_bag.shape[1])
        return model_module.FixedGMMFeatureizer(
            means=np.full((1, latent_dim), value, dtype=np.float32),
            variances=np.full((1, latent_dim), value + 1.0, dtype=np.float32),
            weights=np.asarray([1.0], dtype=np.float32),
            control_bag=control_bag,
        )

    monkeypatch.setattr(
        train_module,
        "fit_linear_projector",
        fake_fit_linear_projector,
    )
    monkeypatch.setattr(train_module, "fit_fixed_gmm", fake_fit_fixed_gmm)

    first_weight, first_bias = train_module._fit_or_load_projector_cache(
        config,
        data,
        split,
        artifacts_dir,
    )
    first_gmm = train_module._fit_or_load_fixed_gmm_cache(
        config,
        data,
        split,
        artifacts_dir,
    )
    second_weight, second_bias = train_module._fit_or_load_projector_cache(
        config,
        data,
        split,
        artifacts_dir,
    )
    second_gmm = train_module._fit_or_load_fixed_gmm_cache(
        config,
        data,
        split,
        artifacts_dir,
    )

    assert len(projector_calls) == 1
    assert len(gmm_calls) == 1
    np.testing.assert_allclose(second_weight, first_weight)
    np.testing.assert_allclose(second_bias, first_bias)
    torch.testing.assert_close(second_gmm.means, first_gmm.means)
    torch.testing.assert_close(second_gmm.variances, first_gmm.variances)

    changed_data = replace(
        data,
        latent_bags=(data.latent_bags[0] + 0.5, data.latent_bags[1]),
    )
    changed_weight, _changed_bias = train_module._fit_or_load_projector_cache(
        config,
        changed_data,
        split,
        artifacts_dir,
    )
    changed_gmm = train_module._fit_or_load_fixed_gmm_cache(
        config,
        changed_data,
        split,
        artifacts_dir,
    )

    assert len(projector_calls) == 2
    assert len(gmm_calls) == 2
    assert not np.array_equal(changed_weight, first_weight)
    assert not torch.equal(changed_gmm.means, first_gmm.means)
    assert (artifacts_dir / "ridge_projector_fit" / "COMPLETE").exists()
    assert (artifacts_dir / "fixed_gmm_fit" / "COMPLETE").exists()


def test_pred_c_loss_backprops_into_mock_state() -> None:
    model, state_model = _build_tiny_aivc_model()
    losses = model.losses_for_gene(
        gene="GENE1",
        control_chunks=(torch.randn(3, 3), torch.randn(2, 3)),
        target_expression_chunks=(torch.randn(3, 3), torch.randn(2, 3)),
        target_latent_chunks=(torch.randn(3, 2), torch.randn(2, 2)),
        batch_index_chunks=(None, None),
        y=torch.tensor(-1.0),
        weights=_loss_weights(),
    )

    losses["total"].backward()

    grads = [
        parameter.grad
        for parameter in state_model.parameters()
        if parameter.grad is not None
    ]
    assert grads
    assert any(torch.any(grad != 0) for grad in grads)


def test_observed_c_supervision_updates_shared_response_stack_not_state() -> None:
    model, _state_model = _build_shared_response_aivc_model()
    losses = model.losses_for_gene(
        **_shared_response_gene_inputs(),
        weights=LossWeights(
            latent_mean_delta=0.0,
            latent_energy=0.0,
            hvg_mean_delta=0.0,
            hvg_energy=0.0,
            pred_c=0.0,
            obs_c=1.0,
            occupancy=0.0,
            gmm_nll=0.0,
        ),
    )

    losses["total"].backward()

    assert all(parameter.grad is None for parameter in model.state_adapter.parameters())
    assert any(
        parameter.grad is not None for parameter in model.response_encoder.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in model.response_pooler.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.c_head.parameters())


def test_predicted_c_supervision_reaches_unfrozen_state() -> None:
    model, _state_model = _build_shared_response_aivc_model()
    losses = model.losses_for_gene(
        **_shared_response_gene_inputs(),
        weights=LossWeights(
            latent_mean_delta=0.0,
            latent_energy=0.0,
            hvg_mean_delta=0.0,
            hvg_energy=0.0,
            pred_c=1.0,
            obs_c=0.0,
            occupancy=0.0,
            gmm_nll=0.0,
        ),
    )

    losses["total"].backward()

    assert any(
        parameter.grad is not None for parameter in model.state_adapter.parameters()
    )


def test_aivc_forward_matches_loss_helper() -> None:
    model, _state_model = _build_tiny_aivc_model()
    kwargs = {
        "gene": "GENE1",
        "control_chunks": (torch.randn(3, 3), torch.randn(2, 3)),
        "target_expression_chunks": (torch.randn(3, 3), torch.randn(2, 3)),
        "target_latent_chunks": (torch.randn(3, 2), torch.randn(2, 2)),
        "batch_index_chunks": (None, None),
        "y": torch.tensor(-1.0),
        "weights": _loss_weights(),
    }

    forward_losses = model(**kwargs)
    helper_losses = model.losses_for_gene(**kwargs)

    assert set(forward_losses) == set(helper_losses)
    for key in forward_losses:
        assert torch.allclose(forward_losses[key], helper_losses[key])


def test_aivc_forward_processes_chunks_independently_without_changing_losses() -> None:
    model, state_model = _build_counting_aivc_model()
    state_model.contextual = True
    kwargs = {
        "gene": "GENE1",
        "control_chunks": (
            torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
            torch.tensor([[2.0, 3.0, 4.0]]),
        ),
        "target_expression_chunks": (
            torch.tensor([[1.0, 1.5, 2.5], [1.5, 2.5, 3.5]]),
            torch.tensor([[2.5, 3.5, 4.5]]),
        ),
        "target_latent_chunks": (
            torch.tensor([[1.0, 1.5], [1.5, 2.5]]),
            torch.tensor([[2.5, 3.5]]),
        ),
        "batch_index_chunks": (torch.tensor([0, 0]), torch.tensor([1])),
        "y": torch.tensor(-1.0),
        "weights": _loss_weights(),
    }

    expected = _legacy_chunk_loop_losses(model, **kwargs)
    state_model.call_shapes.clear()
    state_model.call_batches.clear()
    actual = model(**kwargs)

    assert state_model.call_shapes == [2, 1]
    assert state_model.call_batches == [[0, 0], [1]]
    assert set(actual) == set(expected)
    for key in actual:
        assert torch.allclose(actual[key], expected[key], atol=1e-6)


def test_aivc_forward_batches_native_state_sentences_without_mixing_chunks() -> None:
    model, state_model = _build_counting_aivc_model()
    state_model.contextual = True
    state_model.cell_sentence_len = 2
    kwargs = {
        "gene": "GENE1",
        "control_chunks": (
            torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
            torch.tensor([[8.0, 9.0, 10.0], [9.0, 10.0, 11.0]]),
        ),
        "target_expression_chunks": (
            torch.tensor([[1.0, 1.5, 2.5], [1.5, 2.5, 3.5]]),
            torch.tensor([[8.5, 9.5, 10.5], [9.5, 10.5, 11.5]]),
        ),
        "target_latent_chunks": (
            torch.tensor([[1.0, 1.5], [1.5, 2.5]]),
            torch.tensor([[8.5, 9.5], [9.5, 10.5]]),
        ),
        "batch_index_chunks": (torch.tensor([0, 0]), torch.tensor([1, 1])),
        "y": torch.tensor(-1.0),
        "weights": _loss_weights(),
    }

    expected = _legacy_chunk_loop_losses(model, **kwargs)
    state_model.call_shapes.clear()
    state_model.call_batches.clear()
    state_model.call_padded.clear()
    actual = model(**kwargs)

    assert state_model.call_shapes == [4]
    assert state_model.call_batches == [[0, 0, 1, 1]]
    assert state_model.call_padded == [True]
    for key in actual:
        assert torch.allclose(actual[key], expected[key], atol=1e-6)


def test_hvg_mean_delta_is_invariant_to_unequal_chunk_partitioning() -> None:
    model, _state_model = _build_counting_aivc_model()
    control = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    target = torch.zeros_like(control)
    weights = LossWeights(
        latent_mean_delta=0.0,
        latent_energy=0.0,
        hvg_mean_delta=1.0,
        hvg_energy=0.0,
        pred_c=0.0,
        obs_c=0.0,
        occupancy=0.0,
        gmm_nll=0.0,
    )
    common = {
        "gene": "GENE1",
        "y": torch.tensor(-1.0),
        "weights": weights,
    }

    complete_bag = model.losses_for_gene(
        **common,
        control_chunks=(control,),
        target_expression_chunks=(target,),
        target_latent_chunks=(torch.zeros(5, 2),),
        batch_index_chunks=(None,),
    )
    unequal_chunks = model.losses_for_gene(
        **common,
        control_chunks=(control[:2], control[2:4], control[4:]),
        target_expression_chunks=(target[:2], target[2:4], target[4:]),
        target_latent_chunks=(
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.zeros(1, 2),
        ),
        batch_index_chunks=(None, None, None),
    )

    assert torch.allclose(
        unequal_chunks["hvg_mean_delta"],
        complete_bag["hvg_mean_delta"],
    )


def test_aivc_batched_forward_backprops_each_gene_vector() -> None:
    model, _state_model = _build_two_gene_aivc_model()

    losses = model(
        gene=("GENE1", "GENE2"),
        control_chunks=(
            (torch.randn(2, 3),),
            (torch.randn(2, 3),),
        ),
        target_expression_chunks=(
            (torch.randn(2, 3),),
            (torch.randn(2, 3),),
        ),
        target_latent_chunks=(
            (torch.randn(2, 2),),
            (torch.randn(2, 2),),
        ),
        batch_index_chunks=((None,), (None,)),
        y=torch.tensor([-1.0, -0.5]),
        weights=_loss_weights(),
    )
    losses["total"].backward()

    assert losses["pred_y"].shape == (2,)
    assert losses["per_gene_total_loss"].shape == (2,)
    assert model.perturbations.missing_vectors["g0"].grad is not None
    assert model.perturbations.missing_vectors["g1"].grad is not None


def test_optimizer_uses_lower_state_learning_rate(tmp_path: Path) -> None:
    model, _state_model = _build_tiny_aivc_model()
    config = load_config(_write_scvi_cache_config(tmp_path))
    config = replace(
        config,
        train=replace(
            config.train,
            state_learning_rate=2.5e-6,
            learning_rate=2.5e-5,
        ),
    )

    groups = train_module._optimizer_parameter_groups(model, config)

    assert [group["lr"] for group in groups] == [2.5e-6, 2.5e-5]
    state_ids = {id(parameter) for parameter in model.state_adapter.parameters()}
    assert {id(parameter) for parameter in groups[0]["params"]} == state_ids
    assert state_ids.isdisjoint(id(parameter) for parameter in groups[1]["params"])


def test_aivc_batched_forward_gene_mask_excludes_padding_from_ranknet_and_loss() -> (
    None
):
    model, _state_model = _build_tiny_aivc_model()
    model.perturbations = PerturbationVectorAdapter(
        ["GENE1", "GENE2", "GENE3"],
        {},
        pert_dim=2,
    )
    common_kwargs = {
        "gene": ("GENE1", "GENE2", "GENE3"),
        "control_chunks": (
            (torch.randn(2, 3),),
            (torch.randn(2, 3),),
            (torch.randn(2, 3),),
        ),
        "target_expression_chunks": (
            (torch.randn(2, 3),),
            (torch.randn(2, 3),),
            (torch.randn(2, 3),),
        ),
        "target_latent_chunks": (
            (torch.randn(2, 2),),
            (torch.randn(2, 2),),
            (torch.randn(2, 2),),
        ),
        "batch_index_chunks": ((None,), (None,), (None,)),
        "y": torch.tensor([-1.0, 10.0, -0.2]),
        "weights": LossWeights(
            latent_mean_delta=1.0,
            latent_energy=1.0,
            hvg_mean_delta=0.1,
            hvg_energy=0.1,
            pred_c=1.0,
            obs_c=0.25,
            occupancy=0.1,
            pred_rank=5.0,
            pred_rank_tau=0.25,
            pred_rank_pair_margin=0.0,
            pred_rank_pair_weight_clip=2.0,
        ),
    }

    masked = model(
        **common_kwargs,
        gene_mask=torch.tensor([True, False, True]),
    )
    unmasked = model(**common_kwargs)
    all_true = model(
        **common_kwargs,
        gene_mask=torch.tensor([True, True, True]),
    )
    expected_rank = _pairwise_ranknet_loss(
        masked["pred_y"][[0, 2]],
        common_kwargs["y"][[0, 2]],
        tau=0.25,
        pair_margin=0.0,
        pair_weight_clip=2.0,
    )
    expected_unmasked_rank = _pairwise_ranknet_loss(
        unmasked["pred_y"],
        common_kwargs["y"],
        tau=0.25,
        pair_margin=0.0,
        pair_weight_clip=2.0,
    )
    expected_total = (
        masked["per_gene_total_loss"][[0, 2]].sum()
        + masked["per_gene_pred_rank"][[0, 2]].sum() * 0.0
    )

    assert torch.allclose(masked["pred_rank"], expected_rank)
    assert masked["per_gene_total_loss"][1].item() == 0.0
    assert torch.allclose(masked["total"], expected_total)
    assert torch.allclose(unmasked["pred_rank"], expected_unmasked_rank)
    assert torch.allclose(all_true["total"], unmasked["total"])
    assert torch.allclose(all_true["pred_rank"], unmasked["pred_rank"])


def test_aivc_batched_forward_all_padding_returns_differentiable_zero() -> None:
    model, _state_model = _build_two_gene_aivc_model()

    losses = model(
        gene=("GENE1", "GENE2"),
        control_chunks=((torch.randn(2, 3),), (torch.randn(2, 3),)),
        target_expression_chunks=((torch.randn(2, 3),), (torch.randn(2, 3),)),
        target_latent_chunks=((torch.randn(2, 2),), (torch.randn(2, 2),)),
        batch_index_chunks=((None,), (None,)),
        y=torch.tensor([-1.0, -0.5]),
        weights=LossWeights(
            latent_mean_delta=1.0,
            latent_energy=1.0,
            hvg_mean_delta=0.1,
            hvg_energy=0.1,
            pred_c=1.0,
            obs_c=0.25,
            occupancy=0.1,
            pred_rank=5.0,
            pred_rank_tau=0.25,
            pred_rank_pair_margin=0.0,
            pred_rank_pair_weight_clip=2.0,
        ),
        gene_mask=torch.tensor([False, False]),
    )

    assert losses["total"].requires_grad
    assert losses["total"].item() == 0.0
    assert losses["pred_rank"].item() == 0.0
    assert losses["per_gene_total_loss"].tolist() == [0.0, 0.0]
    losses["total"].backward()


def test_pairwise_ranknet_filters_small_label_margins() -> None:
    pred_y = torch.tensor([0.0, 0.5, 1.0])
    y = torch.tensor([0.0, 0.1, 1.0])

    filtered = _pairwise_ranknet_loss(
        pred_y,
        y,
        tau=0.25,
        pair_margin=0.25,
        pair_weight_clip=2.0,
    )
    manual_deltas = torch.tensor([0.0 - 1.0, 0.5 - 1.0])
    manual_targets = torch.tensor([0.0 - 1.0, 0.1 - 1.0]).sign()
    manual_weights = torch.tensor([1.0, 0.9])
    expected = (
        F.softplus(-manual_targets * manual_deltas / 0.25) * manual_weights
    ).sum() / manual_weights.sum()

    assert torch.allclose(filtered, expected)
    no_pairs = _pairwise_ranknet_loss(
        pred_y,
        y,
        tau=0.25,
        pair_margin=2.0,
        pair_weight_clip=2.0,
    )
    assert no_pairs.item() == 0.0


def test_global_ranknet_gathers_four_single_gene_ranks_with_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank_predictions = tuple(
        torch.tensor([value], requires_grad=True) for value in (0.2, -0.5, 0.8, 0.1)
    )
    rank_labels = tuple(torch.tensor([value]) for value in (-1.0, 0.5, -0.2, 1.0))
    rank_masks = tuple(torch.tensor([True]) for _ in range(4))
    collective_order: list[str] = []

    monkeypatch.setattr(model_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(model_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_module.dist, "get_world_size", lambda: 4)

    def differentiable_all_gather(
        local_prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        collective_order.append("predictions")
        assert local_prediction.data_ptr() == rank_predictions[0].data_ptr()
        return rank_predictions

    def all_gather(
        outputs: list[torch.Tensor],
        local_value: torch.Tensor,
    ) -> None:
        assert not local_value.requires_grad
        if local_value.dtype == torch.bool:
            collective_order.append("masks")
            gathered = rank_masks
        else:
            collective_order.append("labels")
            gathered = rank_labels
        for output, value in zip(outputs, gathered, strict=True):
            output.copy_(value)

    monkeypatch.setattr(
        model_module,
        "_differentiable_all_gather",
        differentiable_all_gather,
    )
    monkeypatch.setattr(model_module.dist, "all_gather", all_gather)

    loss = model_module._global_pairwise_ranknet_loss(
        rank_predictions[0],
        rank_labels[0],
        rank_masks[0],
        tau=0.25,
        pair_margin=0.0,
        pair_weight_clip=2.0,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert collective_order == ["predictions", "labels", "masks"]
    assert all(
        prediction.grad is not None
        and torch.isfinite(prediction.grad).all()
        and prediction.grad.abs().sum().item() > 0.0
        for prediction in rank_predictions
    )


def test_frozen_state_keeps_adapter_eval_and_trains_downstream_modules() -> None:
    model, state_model = _build_two_gene_aivc_model(freeze_state=True)

    losses = model(
        gene=("GENE1", "GENE2"),
        control_chunks=((torch.randn(2, 3),), (torch.randn(2, 3),)),
        target_expression_chunks=((torch.randn(2, 3),), (torch.randn(2, 3),)),
        target_latent_chunks=((torch.randn(2, 2),), (torch.randn(2, 2),)),
        batch_index_chunks=((None,), (None,)),
        y=torch.tensor([-1.0, -0.5]),
        weights=LossWeights(
            latent_mean_delta=0.0,
            latent_energy=0.0,
            hvg_mean_delta=0.0,
            hvg_energy=0.0,
            pred_c=1.0,
            obs_c=0.25,
            occupancy=0.0,
            pred_rank=5.0,
            pred_rank_tau=0.25,
            pred_rank_pair_margin=0.25,
            pred_rank_pair_weight_clip=2.0,
        ),
    )
    losses["total"].backward()

    assert not model.state_adapter.training
    assert all(parameter.grad is None for parameter in state_model.parameters())
    assert model.response_encoder.linear.weight.grad is not None
    assert model.c_head.net[-1].weight.grad is not None
    assert model.perturbations.missing_vectors["g0"].grad is not None


def test_esm2_state_is_trainable_before_ddp_prepare(
    tmp_path: Path, monkeypatch
) -> None:
    data = _toy_gene_bags_with_batches()
    manifest = tmp_path / "outer.csv"
    pd.DataFrame(
        {"perturbation_gene": ["GENE1", "GENE2"], "outer_fold": [0, 1]}
    ).to_csv(manifest, index=False)
    sha256_file = tmp_path / "outer.csv.sha256"
    sha256_file.write_text(f"{_sha256(manifest)}\n", encoding="utf-8")
    esm2_npz = tmp_path / "esm2.npz"
    np.savez(
        esm2_npz,
        symbols=np.asarray(["GENE1", "GENE2"], dtype=object),
        vectors=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        resolved=np.asarray([True, True]),
    )
    config = load_config(_write_scvi_cache_config(tmp_path))
    config = replace(
        config,
        response_encoder=ResponseEncoderConfig(input_dim=2000, latent_dim=128),
        cv=replace(
            config.cv,
            outer_split_manifest=manifest,
            outer_split_sha256_file=sha256_file,
        ),
        state=replace(
            config.state,
            gene_tokenizer="esm2",
            esm2_npz=esm2_npz,
            esm2_adapter_hidden=4,
            require_resolved_esm2=True,
            pert_dim=5,
        ),
    )
    monkeypatch.setattr(train_module, "CANONICAL_GENE_COUNT", 2)
    monkeypatch.setattr(gene_splits_module, "CANONICAL_GENE_COUNT", 2)
    monkeypatch.setattr(gene_splits_module, "CANONICAL_OUTER_FOLDS", frozenset({0, 1}))
    state_model = model_module.LinearMockStateModel(
        input_dim=3, output_dim=2000, pert_dim=5
    )
    monkeypatch.setattr(
        train_module,
        "load_state_model",
        lambda **_kwargs: state_model,
    )
    model = train_module._build_e2e_model(
        config,
        data,
        extra_genes=(),
        canonical_gene_order=("GENE1", "GENE2"),
        emit_checkpoint_output=False,
    )
    assert model.perturbations("GENE1").shape == (5,)
    _predicted, latent = model.predict_response(
        torch.as_tensor(data.control_input), "GENE1"
    )
    pred_y = model.c_head(model.response_pooler(latent, latent.detach()))
    pred_c = F.mse_loss(pred_y, torch.tensor(-1.0))
    pred_c.backward()

    assert isinstance(model.perturbations, Esm2PerturbationAdapter)
    assert all(
        parameter.requires_grad for parameter in model.state_adapter.parameters()
    )
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.state_adapter.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.perturbations.adapter.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.response_encoder.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.c_head.parameters())


def test_esm2_build_rejects_missing_external_gene_without_filtering(
    tmp_path: Path, monkeypatch
) -> None:
    data = _toy_gene_bags_with_batches()
    manifest = tmp_path / "outer.csv"
    pd.DataFrame(
        {"perturbation_gene": ["GENE1", "GENE2"], "outer_fold": [0, 1]}
    ).to_csv(manifest, index=False)
    sha256_file = tmp_path / "outer.csv.sha256"
    sha256_file.write_text(f"{_sha256(manifest)}\n", encoding="utf-8")
    esm2_npz = tmp_path / "esm2.npz"
    np.savez(
        esm2_npz,
        symbols=np.asarray(["GENE1", "GENE2"], dtype=object),
        vectors=np.ones((2, 3), dtype=np.float32),
        resolved=np.asarray([True, True]),
    )
    config = load_config(_write_scvi_cache_config(tmp_path))
    config = replace(
        config,
        response_encoder=ResponseEncoderConfig(input_dim=2000, latent_dim=128),
        cv=replace(
            config.cv,
            outer_split_manifest=manifest,
            outer_split_sha256_file=sha256_file,
        ),
        state=replace(
            config.state,
            gene_tokenizer="esm2",
            esm2_npz=esm2_npz,
            require_resolved_esm2=True,
        ),
    )
    monkeypatch.setattr(train_module, "CANONICAL_GENE_COUNT", 2)
    monkeypatch.setattr(gene_splits_module, "CANONICAL_GENE_COUNT", 2)
    monkeypatch.setattr(gene_splits_module, "CANONICAL_OUTER_FOLDS", frozenset({0, 1}))
    with pytest.raises(ValueError, match="ADAMSON_ONLY"):
        train_module._build_e2e_model(
            config,
            data,
            extra_genes=("ADAMSON_ONLY",),
            canonical_gene_order=("GENE1", "GENE2"),
            emit_checkpoint_output=False,
        )


def test_esm2_build_accepts_resolved_external_gene(
    tmp_path: Path, monkeypatch
) -> None:
    data = _toy_gene_bags_with_batches()
    manifest = tmp_path / "outer.csv"
    pd.DataFrame(
        {"perturbation_gene": ["GENE1", "GENE2"], "outer_fold": [0, 1]}
    ).to_csv(manifest, index=False)
    sha256_file = tmp_path / "outer.csv.sha256"
    sha256_file.write_text(f"{_sha256(manifest)}\n", encoding="utf-8")
    esm2_npz = tmp_path / "esm2.npz"
    np.savez(
        esm2_npz,
        symbols=np.asarray(["GENE1", "ADAMSON_ONLY", "GENE2"], dtype=object),
        vectors=np.ones((3, 3), dtype=np.float32),
        resolved=np.asarray([True, True, True]),
    )
    config = load_config(_write_scvi_cache_config(tmp_path))
    config = replace(
        config,
        response_encoder=ResponseEncoderConfig(input_dim=2000, latent_dim=128),
        cv=replace(
            config.cv,
            outer_split_manifest=manifest,
            outer_split_sha256_file=sha256_file,
        ),
        state=replace(
            config.state,
            gene_tokenizer="esm2",
            esm2_npz=esm2_npz,
            require_resolved_esm2=True,
        ),
    )
    monkeypatch.setattr(train_module, "CANONICAL_GENE_COUNT", 2)
    monkeypatch.setattr(gene_splits_module, "CANONICAL_GENE_COUNT", 2)
    monkeypatch.setattr(gene_splits_module, "CANONICAL_OUTER_FOLDS", frozenset({0, 1}))

    model = train_module._build_e2e_model(
        config,
        data,
        extra_genes=("ADAMSON_ONLY",),
        canonical_gene_order=("GENE1", "GENE2"),
        emit_checkpoint_output=False,
    )

    assert model.perturbations("ADAMSON_ONLY").shape == (2,)


def test_zero_weight_energy_losses_skip_extra_compute(monkeypatch) -> None:
    model, _state_model = _build_tiny_aivc_model()

    def fail_energy(
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        del predicted, target
        raise AssertionError("zero-weight energy loss should not be computed")

    monkeypatch.setattr(model_module, "_energy_distance", fail_energy)

    losses = model.losses_for_gene(
        gene="GENE1",
        control_chunks=(torch.randn(2, 3),),
        target_expression_chunks=(torch.randn(2, 3),),
        target_latent_chunks=(torch.randn(2, 2),),
        batch_index_chunks=(None,),
        y=torch.tensor(-1.0),
        weights=LossWeights(
            latent_mean_delta=0.1,
            latent_energy=0.0,
            hvg_mean_delta=0.01,
            hvg_energy=0.0,
            pred_c=2.0,
            obs_c=0.25,
            occupancy=0.0,
        ),
    )

    assert losses["hvg_energy"].item() == 0.0
    assert losses["latent_energy"].item() == 0.0
    assert losses["hvg_energy"].requires_grad
    assert losses["latent_energy"].requires_grad
    assert all(torch.isfinite(loss) for loss in losses.values())


def test_nonzero_expensive_losses_match_previous_math() -> None:
    model, _state_model = _build_tiny_aivc_model()
    kwargs = {
        "gene": "GENE1",
        "control_chunks": (torch.randn(3, 3),),
        "target_expression_chunks": (torch.randn(3, 3),),
        "target_latent_chunks": (torch.randn(3, 2),),
        "batch_index_chunks": (None,),
        "y": torch.tensor(-0.5),
        "weights": LossWeights(
            latent_mean_delta=0.1,
            latent_energy=0.2,
            hvg_mean_delta=0.01,
            hvg_energy=0.3,
            pred_c=2.0,
            obs_c=0.25,
            occupancy=0.4,
        ),
    }

    losses = model.losses_for_gene(**kwargs)
    legacy_losses = _legacy_chunk_loop_losses(model, **kwargs)

    for key in (
        "total",
        "hvg_energy",
        "latent_energy",
        "occupancy",
        "pred_c",
        "obs_c",
    ):
        assert torch.allclose(losses[key], legacy_losses[key], atol=1e-6)


def test_a_to_b_set_loss_is_target_order_invariant() -> None:
    model, _state_model = _build_tiny_aivc_model()
    control = torch.zeros(4, 3)
    target_expression = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]]
    )
    target_latent = model.response_encoder(target_expression).detach()
    shuffled = torch.tensor([2, 0, 3, 1])

    first = model.losses_for_gene(
        gene="GENE1",
        control_chunks=(control,),
        target_expression_chunks=(target_expression,),
        target_latent_chunks=(target_latent,),
        batch_index_chunks=(None,),
        y=torch.tensor(-1.0),
        weights=_loss_weights(),
    )
    second = model.losses_for_gene(
        gene="GENE1",
        control_chunks=(control,),
        target_expression_chunks=(target_expression[shuffled],),
        target_latent_chunks=(target_latent[shuffled],),
        batch_index_chunks=(None,),
        y=torch.tensor(-1.0),
        weights=_loss_weights(),
    )

    for key in (
        "hvg_mean_delta",
        "hvg_energy",
        "latent_mean_delta",
        "latent_energy",
        "occupancy",
    ):
        assert torch.allclose(first[key], second[key], atol=1e-6)
    assert "a_to_b_expression" not in first
    assert "a_to_b_latent" not in first


def test_state_forward_adapter_uses_predict_step_batch_schema() -> None:
    class PredictStepState(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.seen: dict[str, object] = {}

        def predict_step(
            self,
            batch: dict[str, object],
            batch_idx: int,
            padded: bool,
        ) -> dict[str, torch.Tensor]:
            self.seen = {"batch": batch, "batch_idx": batch_idx, "padded": padded}
            ctrl = batch["ctrl_cell_emb"]
            assert isinstance(ctrl, torch.Tensor)
            return {"preds": ctrl @ self.weight}

    state = PredictStepState()
    adapter = StateForwardAdapter(state)
    control = torch.eye(2, requires_grad=True)
    batch_indices = torch.tensor([1, 2])

    output = adapter(control, torch.tensor([1.0, 0.0]), "GENE1", batch_indices)
    output.sum().backward()

    seen_batch = state.seen["batch"]
    assert isinstance(seen_batch, dict)
    assert state.seen["padded"] is False
    assert {"ctrl_cell_emb", "pert_emb", "pert_name", "batch"}.issubset(seen_batch)
    assert state.weight.grad is not None
    assert torch.isfinite(output).all()


def test_train_smoke_writes_minimal_csv_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    h5ad_path, overlap_path = _write_toy_inputs(tmp_path)
    config_path = tmp_path / "state_smoke.yaml"
    config_path.write_text(
        f"""
data:
  h5ad_path: {h5ad_path}
  overlap_csv: {overlap_path}
  output_dir: {tmp_path / "outputs"}
  obs_perturbation_col: gene
  control_label: non-targeting
  state_embed_key: null
  scvi_obsm_key: X_scVI
  depmap_label_col: depmap_gene_effect
  matched_label_col: has_depmap_label
  min_cells_per_gene: 2
split:
  train_fraction: 0.5
  val_fraction: 0.25
  test_fraction: 0.25
  random_state: 11
  stratify_bins: 2
state:
  backend: linear_mock
  input_dim: 3
  output_dim: 3
  pert_dim: 2
projector:
  latent_dim: 2
  ridge_alpha: 0.1
  trainable: true
gmm:
  n_components: 2
  covariance_floor: 0.0001
  max_fit_cells: null
model:
  c_hidden_units: [8]
  dropout: 0.0
loss:
  latent_mean_delta_weight: 1.0
  latent_energy_weight: 1.0
  hvg_mean_delta_weight: 0.1
  hvg_energy_weight: 0.1
  pred_c_weight: 1.0
  obs_c_weight: 0.25
  occupancy_weight: 0.1
train:
  run_id: smoke
  seed: 13
  max_epochs: 2
  learning_rate: 0.001
  weight_decay: 0.0
  cell_set_len: 2
  device: cpu
""",
    )
    config = load_config(config_path)
    split = make_gene_split(
        np.asarray(["GENE1", "GENE2", "GENE3", "GENE4"], dtype=object),
        np.asarray([-1.2, -0.7, 0.1, 0.4], dtype=np.float32),
        config.split,
    )
    train_indices = {int(index) for index in split.train}
    chunked_indices: list[int] = []
    original_make_cell_set_chunks = train_module.make_cell_set_chunks

    def fail_non_train_target_chunking(*args: object, **kwargs: object) -> object:
        index = int(args[1])
        chunked_indices.append(index)
        if index not in train_indices:
            raise AssertionError(
                "validation/final prediction must not inspect target chunks"
            )
        return original_make_cell_set_chunks(*args, **kwargs)

    monkeypatch.setattr(
        train_module,
        "make_cell_set_chunks",
        fail_non_train_target_chunking,
    )

    paths = run_training(
        config,
        accelerator=_validated_cpu_accelerator(config),  # type: ignore[arg-type]
    )

    assert paths["train_log"].exists()
    assert paths["test_metrics"].exists()
    assert (paths["run_dir"] / "artifacts" / "test_predictions.csv").exists()
    assert (paths["run_dir"] / "models" / "best" / "pytorch_model.bin").exists()
    assert (paths["run_dir"] / "models" / "best" / "metadata.json").exists()
    assert (paths["run_dir"] / "models" / "final" / "pytorch_model.bin").exists()
    assert (paths["run_dir"] / "models" / "final" / "metadata.json").exists()
    train_log = pd.read_csv(paths["train_log"])
    test_metrics = pd.read_csv(paths["test_metrics"])
    assert len(train_log) == 2
    assert {
        "epoch",
        "gpu_peak_memory_allocated_mb",
        "train_total_loss",
        "val_spearman",
    }.issubset(train_log.columns)
    assert set(chunked_indices).issubset(train_indices)
    expected_loss_cols = {
        "hvg_mean_delta",
        "hvg_energy",
        "latent_mean_delta",
        "latent_energy",
        "occupancy",
        "pred_c",
        "obs_c",
        "pred_rank",
        "total_loss",
    }
    assert {f"val_{column}" for column in expected_loss_cols}.isdisjoint(
        train_log.columns
    )
    best_metadata = json.loads(
        (paths["run_dir"] / "models" / "best" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    selected_epoch = int(best_metadata["epoch"])
    selected_val_spearman = float(
        train_log.loc[train_log["epoch"] == selected_epoch, "val_spearman"].iloc[0]
    )
    assert best_metadata["selection_metric"] == "val_spearman"
    if math.isnan(selected_val_spearman):
        assert math.isnan(float(best_metadata["metric_value"]))
    else:
        assert float(best_metadata["metric_value"]) == selected_val_spearman
    assert expected_loss_cols.isdisjoint(test_metrics.columns)
    assert {"rmse", "spearman"}.issubset(test_metrics.columns)
    assert {"obs_rmse", "obs_spearman"}.isdisjoint(test_metrics.columns)
    predictions = pd.read_csv(paths["run_dir"] / "artifacts" / "test_predictions.csv")
    assert "y_obs_anchor" not in predictions.columns


def test_csv_writer_is_main_process_only(tmp_path: Path) -> None:
    class FakeAccelerator:
        def __init__(self, is_main_process: bool) -> None:
            self.is_main_process = is_main_process

    path = tmp_path / "nested" / "table.csv"
    frame = pd.DataFrame({"value": [1]})

    _write_csv_if_main(frame, path, FakeAccelerator(False))

    assert not path.exists()

    _write_csv_if_main(frame, path, FakeAccelerator(True))

    assert path.exists()
    assert pd.read_csv(path)["value"].tolist() == [1]


def test_external_adamson_sources_merge_and_mean_impute_missing_genes(
    tmp_path: Path,
) -> None:
    h5ad_path, overlap_path = _write_toy_inputs(tmp_path)
    source_a, source_b, external_overlap = _write_toy_external_inputs(tmp_path)
    config_path = _write_external_smoke_config(
        tmp_path,
        h5ad_path,
        overlap_path,
        source_a,
        source_b,
        external_overlap,
        run_id="loader",
        max_epochs=1,
    )
    config = load_config(config_path)
    assert config.external_test is not None
    assert all(
        source.var_gene_symbol_col is None for source in config.external_test.sources
    )
    reference = load_gene_bags(config)

    external = load_external_gene_bags(config, reference, tmp_path / "artifacts")

    assert external is not None
    assert external.data.genes.astype(str).tolist() == ["GENE1", "GENE5"]
    assert external.data.metadata["external_row_count"].tolist() == [2, 1]
    assert external.data.metadata["observed_n_cells"].tolist() == [4, 2]
    assert external.qa["n_gene_rows"] == 2
    assert external.qa["n_control_cells"] == 4
    source_qa = external.qa["sources"]
    assert isinstance(source_qa, list)
    assert source_qa[0]["missing_input_features"] == 1
    reference_fill = reference.control_input.mean(axis=0)
    np.testing.assert_allclose(external.data.input_bags[0][:, 1], reference_fill[1])


def test_fallback_primary_loader_replaces_nonfinite_from_control_only(
    tmp_path: Path,
) -> None:
    h5ad_path, overlap_path = _write_toy_inputs(tmp_path)
    adata = ad.read_h5ad(h5ad_path)
    values = np.asarray(adata.X, dtype=np.float32)
    values[0, 0] = np.nan
    values[4, 1] = np.inf
    values[5, 2] = -7.0
    adata.X = values
    adata.write_h5ad(h5ad_path)
    config = load_config(_write_scvi_cache_config(tmp_path))
    config = replace(
        config,
        data=replace(
            config.data,
            h5ad_path=h5ad_path,
            overlap_csv=overlap_path,
        ),
    )

    bags = load_gene_bags(config)

    assert np.isfinite(bags.feature_fill_values).all()
    assert np.isfinite(bags.control_input).all()
    assert np.isfinite(bags.control_latent).all()
    assert all(np.isfinite(bag).all() for bag in bags.input_bags)
    assert all(np.isfinite(bag).all() for bag in bags.latent_bags)
    np.testing.assert_allclose(
        bags.feature_fill_values,
        np.asarray([0.6, 0.55, 0.65], dtype=np.float32),
    )
    np.testing.assert_allclose(bags.control_latent, bags.control_input)
    for input_bag, latent_bag in zip(bags.input_bags, bags.latent_bags, strict=True):
        np.testing.assert_allclose(latent_bag, input_bag)
    assert bags.input_bags[0][1, 2] == pytest.approx(-7.0)


def test_external_adamson_reuses_reference_fills_for_nonfinite_values(
    tmp_path: Path,
) -> None:
    h5ad_path, overlap_path = _write_toy_inputs(tmp_path)
    source_a, source_b, external_overlap = _write_toy_external_inputs(tmp_path)
    adata = ad.read_h5ad(source_a)
    values = np.asarray(adata.X, dtype=np.float32)
    values[0, 1] = np.inf
    values[2, 0] = np.nan
    adata.X = values
    adata.write_h5ad(source_a)
    config = load_config(
        _write_external_smoke_config(
            tmp_path,
            h5ad_path,
            overlap_path,
            source_a,
            source_b,
            external_overlap,
            run_id="finite_loader",
            max_epochs=1,
        )
    )
    reference = load_gene_bags(config)

    external = load_external_gene_bags(config, reference, tmp_path / "artifacts")

    assert external is not None
    assert all(np.isfinite(bag).all() for bag in external.data.input_bags)
    assert np.isfinite(external.data.control_input).all()
    np.testing.assert_array_equal(
        external.data.feature_fill_values,
        reference.feature_fill_values,
    )


def test_external_var_names_align_to_state_symbols() -> None:
    adata = ad.AnnData(np.asarray([[1.0, 2.0]], dtype=np.float32))
    adata.var_names = ["A", "B"]

    assert _var_symbols(adata, None) == ["A", "B"]


def test_external_alignment_rejects_zero_matches(tmp_path: Path) -> None:
    h5ad_path, overlap_path = _write_toy_inputs(tmp_path)
    config = load_config(_write_scvi_cache_config(tmp_path))
    config = replace(
        config,
        data=replace(
            config.data,
            h5ad_path=h5ad_path,
            overlap_csv=overlap_path,
        ),
    )
    reference = replace(
        load_gene_bags(config),
        feature_names=np.asarray(["A", "B", "C"], dtype=object),
    )
    adata = ad.AnnData(np.ones((2, 2), dtype=np.float32))
    adata.var_names = ["X", "Y"]
    source = ExternalSourceConfig(
        "adamson",
        Path("unused"),
        var_gene_symbol_col=None,
    )

    with pytest.raises(ValueError, match="matched 0"):
        _external_state_input_view(adata, source, config, reference)


def test_primary_metadata_duplicate_labels_aggregate_when_consistent(
    tmp_path: Path,
) -> None:
    overlap_path = tmp_path / "overlap.csv"
    pd.DataFrame(
        {
            "perturbation_gene": ["GENE1", "GENE1", "GENE2"],
            "depmap_gene_effect": [-1.0, -1.0 + 1e-9, -0.5],
            "has_depmap_label": [True, True, True],
        }
    ).to_csv(overlap_path, index=False)

    metadata = _load_metadata(
        DataConfig(
            h5ad_path=tmp_path / "unused.h5ad",
            overlap_csv=overlap_path,
            output_dir=tmp_path / "outputs",
        )
    )

    assert metadata["perturbation_gene"].tolist() == ["GENE1", "GENE2"]
    assert math.isclose(metadata["depmap_gene_effect"].iloc[0], -1.0)


def test_primary_metadata_duplicate_labels_raise_when_conflicting(
    tmp_path: Path,
) -> None:
    overlap_path = tmp_path / "overlap.csv"
    pd.DataFrame(
        {
            "perturbation_gene": ["GENE1", "GENE1"],
            "depmap_gene_effect": [-1.0, -0.9],
            "has_depmap_label": [True, True],
        }
    ).to_csv(overlap_path, index=False)

    try:
        _load_metadata(
            DataConfig(
                h5ad_path=tmp_path / "unused.h5ad",
                overlap_csv=overlap_path,
                output_dir=tmp_path / "outputs",
            )
        )
    except ValueError as exc:
        assert "Conflicting DepMap labels" in str(exc)
    else:
        raise AssertionError("conflicting duplicate labels should fail")


def test_external_merge_duplicate_labels_aggregate_when_consistent() -> None:
    row_metadata = pd.DataFrame(
        {
            "source_dataset": ["source_a", "source_b"],
            "perturbation_gene": ["GENE1", "GENE1"],
            "depmap_gene_effect": [-1.0, -1.0 + 1e-9],
            "observed_n_cells": [2, 3],
        }
    )
    input_bags = [
        np.ones((2, 2), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
    ]
    latent_bags = [bag.copy() for bag in input_bags]

    metadata, merged_input, _merged_latent, _merged_batch = _merge_external_gene_rows(
        row_metadata,
        input_bags,
        latent_bags,
        None,
        "depmap_gene_effect",
    )

    assert metadata["perturbation_gene"].tolist() == ["GENE1"]
    assert metadata["external_row_count"].tolist() == [2]
    assert metadata["observed_n_cells"].tolist() == [5]
    assert merged_input[0].shape == (5, 2)
    assert math.isclose(metadata["depmap_gene_effect"].iloc[0], -1.0)


def test_external_merge_duplicate_labels_raise_when_conflicting() -> None:
    row_metadata = pd.DataFrame(
        {
            "source_dataset": ["source_a", "source_b"],
            "perturbation_gene": ["GENE1", "GENE1"],
            "depmap_gene_effect": [-1.0, -0.9],
            "observed_n_cells": [2, 3],
        }
    )
    input_bags = [
        np.ones((2, 2), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
    ]
    latent_bags = [bag.copy() for bag in input_bags]

    try:
        _merge_external_gene_rows(
            row_metadata,
            input_bags,
            latent_bags,
            None,
            "depmap_gene_effect",
        )
    except ValueError as exc:
        assert "Conflicting DepMap labels" in str(exc)
    else:
        raise AssertionError("conflicting external labels should fail")


def test_train_smoke_writes_external_adamson_outputs(tmp_path: Path) -> None:
    h5ad_path, overlap_path = _write_toy_inputs(tmp_path)
    source_a, source_b, external_overlap = _write_toy_external_inputs(tmp_path)
    config_path = _write_external_smoke_config(
        tmp_path,
        h5ad_path,
        overlap_path,
        source_a,
        source_b,
        external_overlap,
        run_id="external_smoke",
        max_epochs=2,
    )

    config = load_config(config_path)
    paths = run_training(
        config,
        accelerator=_validated_cpu_accelerator(config),  # type: ignore[arg-type]
    )

    run_dir = paths["run_dir"]
    test_metrics = pd.read_csv(paths["test_metrics"])
    predictions = pd.read_csv(run_dir / "artifacts" / "test_predictions.csv")
    assert set(test_metrics["evaluation_scope"]) == {
        "internal_outer_test",
        "external:adamson_k562",
    }
    assert set(predictions["evaluation_scope"]) == {
        "internal_outer_test",
        "external:adamson_k562",
    }
    external_predictions = predictions.loc[
        predictions["evaluation_scope"] == "external:adamson_k562"
    ]
    assert set(external_predictions["perturbation_gene"]) == {"GENE1", "GENE5"}
    assert "source_dataset" in predictions.columns
    assert "y_obs_anchor" not in predictions.columns
    assert not any(column.startswith("obs_") for column in test_metrics.columns)
    assert "hvg_mean_delta" not in test_metrics.columns
    assert "perturbation_has_known_vector" in predictions.columns
    assert not external_predictions["perturbation_has_known_vector"].any()
    assert (run_dir / "artifacts" / "external_test_qa.json").exists()
    assert (run_dir / "models" / "best" / "pytorch_model.bin").exists()
    assert (run_dir / "models" / "final" / "pytorch_model.bin").exists()


def test_external_metadata_does_not_contaminate_shared_internal_gene(
    tmp_path: Path,
) -> None:
    h5ad_path, overlap_path = _write_toy_inputs(tmp_path)
    source_a, source_b, external_overlap = _write_toy_external_inputs(tmp_path)
    config_path = _write_external_smoke_config(
        tmp_path,
        h5ad_path,
        overlap_path,
        source_a,
        source_b,
        external_overlap,
        run_id="external_metadata_scope",
        max_epochs=1,
    )

    config = load_config(config_path)
    paths = run_training(
        config,
        accelerator=_validated_cpu_accelerator(config),  # type: ignore[arg-type]
    )

    predictions = pd.read_csv(paths["run_dir"] / "artifacts" / "test_predictions.csv")
    shared = predictions.loc[predictions["perturbation_gene"] == "GENE1"]
    internal = shared.loc[shared["evaluation_scope"] == "internal_outer_test"].iloc[0]
    adamson = shared.loc[shared["evaluation_scope"] == "external:adamson_k562"].iloc[0]
    assert pd.isna(internal["source_dataset"])
    assert pd.isna(internal["external_row_count"])
    assert pd.notna(adamson["source_dataset"])
    assert int(adamson["external_row_count"]) == 2


def test_load_state_model_can_suppress_checkpoint_stdout(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    module = types.ModuleType("state.tx.models.state_transition")

    class FakeStateTransitionPerturbationModel:
        @classmethod
        def load_from_checkpoint(cls, path: str, strict: bool) -> torch.nn.Module:
            print("StateTransitionPerturbationModel(...)")
            assert path == str(tmp_path / "state.ckpt")
            assert strict is False
            return torch.nn.Identity()

    module.StateTransitionPerturbationModel = FakeStateTransitionPerturbationModel
    monkeypatch.setitem(sys.modules, "state.tx.models.state_transition", module)

    model = load_state_model(
        backend="state_checkpoint",
        checkpoint_path=tmp_path / "state.ckpt",
        input_dim=3,
        output_dim=3,
        pert_dim=2,
        emit_checkpoint_output=False,
    )

    assert isinstance(model, torch.nn.Identity)
    assert capsys.readouterr().out == ""


def _loss_weights() -> LossWeights:
    return LossWeights(
        latent_mean_delta=1.0,
        latent_energy=1.0,
        hvg_mean_delta=0.1,
        hvg_energy=0.1,
        pred_c=1.0,
        obs_c=0.25,
        occupancy=0.1,
        gmm_nll=0.05,
    )


def _build_tiny_aivc_model(
    *,
    freeze_state: bool = False,
) -> tuple[AivcModel, torch.nn.Module]:
    state_model = load_state_model(
        backend="linear_mock",
        checkpoint_path=None,
        input_dim=3,
        output_dim=3,
        pert_dim=2,
    )
    perturbations = PerturbationVectorAdapter(["GENE1"], {}, pert_dim=2)
    response_encoder = ResponseEncoder(input_dim=3, latent_dim=2)
    response_pooler = TrainableDiagonalGMM(
        latent_dim=2,
        n_components=2,
        covariance_floor=1e-4,
        init_scale=0.02,
    )
    state_adapter = StateForwardAdapter(state_model)
    if freeze_state:
        for parameter in state_adapter.parameters():
            parameter.requires_grad = False
        state_adapter.eval()
    model = AivcModel(
        state_adapter=state_adapter,
        perturbations=perturbations,
        response_encoder=response_encoder,
        response_pooler=response_pooler,
        c_head=MLPHead(response_pooler.output_dim, (8,), 0.0),
        control_expression_mean=np.zeros(3, dtype=np.float32),
    )
    return model, state_model


def _build_shared_response_aivc_model() -> tuple[AivcModel, torch.nn.Module]:
    return _build_tiny_aivc_model()


def _build_legacy_featureizer() -> torch.nn.Module:
    control_latent = np.asarray([[0.0, 0.0], [0.1, 0.1]], dtype=np.float32)
    return fit_fixed_gmm(
        (
            control_latent,
            np.asarray([[1.0, 1.0], [1.1, 1.1]], dtype=np.float32),
        ),
        control_latent,
        n_components=2,
        covariance_floor=1e-4,
        random_state=7,
        max_fit_cells=None,
    )


def _shared_response_gene_inputs() -> dict[str, object]:
    return {
        "gene": "GENE1",
        "control_chunks": (torch.randn(3, 3), torch.randn(2, 3)),
        "target_expression_chunks": (torch.randn(3, 3), torch.randn(2, 3)),
        "target_latent_chunks": (torch.randn(3, 2), torch.randn(2, 2)),
        "batch_index_chunks": (None, None),
        "y": torch.tensor(-1.0),
    }


def _build_two_gene_aivc_model(
    *,
    freeze_state: bool = False,
) -> tuple[AivcModel, torch.nn.Module]:
    model, state_model = _build_tiny_aivc_model(freeze_state=freeze_state)
    model.perturbations = PerturbationVectorAdapter(["GENE1", "GENE2"], {}, pert_dim=2)
    return model, state_model


class _CountingStateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(3))
        self.call_shapes: list[int] = []
        self.call_batches: list[list[int]] = []
        self.call_padded: list[bool] = []
        self.contextual = False
        self.cell_sentence_len: int | None = None

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        padded: bool = False,
    ) -> torch.Tensor:
        control = batch["ctrl_cell_emb"]
        pert = batch["pert_emb"]
        self.call_shapes.append(int(control.shape[0]))
        self.call_padded.append(bool(padded))
        batch_indices = batch.get("batch")
        self.call_batches.append(
            [] if batch_indices is None else batch_indices.tolist()
        )
        contextual = control
        if self.contextual:
            if padded and self.cell_sentence_len is not None:
                sequences = control.reshape(
                    -1,
                    self.cell_sentence_len,
                    control.shape[-1],
                )
                contextual = sequences + sequences.mean(dim=1, keepdim=True)
                contextual = contextual.reshape_as(control)
            else:
                contextual = control + control.mean(dim=0, keepdim=True)
        return contextual @ self.weight + 0.1 * pert[:, :3]


def _build_counting_aivc_model() -> tuple[AivcModel, _CountingStateModel]:
    state_model = _CountingStateModel()
    perturbations = PerturbationVectorAdapter(["GENE1"], {}, pert_dim=3)
    response_encoder = ResponseEncoder(input_dim=3, latent_dim=2)
    response_pooler = TrainableDiagonalGMM(
        latent_dim=2,
        n_components=2,
        covariance_floor=1e-4,
        init_scale=0.02,
    )
    model = AivcModel(
        state_adapter=StateForwardAdapter(state_model),
        perturbations=perturbations,
        response_encoder=response_encoder,
        response_pooler=response_pooler,
        c_head=MLPHead(response_pooler.output_dim, (8,), 0.0),
        control_expression_mean=np.zeros(3, dtype=np.float32),
    )
    return model, state_model


def _legacy_chunk_loop_losses(
    model: AivcModel,
    *,
    gene: str,
    control_chunks: tuple[torch.Tensor, ...],
    target_expression_chunks: tuple[torch.Tensor, ...],
    target_latent_chunks: tuple[torch.Tensor, ...],
    batch_index_chunks: tuple[torch.Tensor | None, ...],
    y: torch.Tensor,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    del target_latent_chunks
    predicted_expression_chunks: list[torch.Tensor] = []
    predicted_latent_chunks: list[torch.Tensor] = []
    hvg_energy_terms: list[torch.Tensor] = []
    for control, target_expression, batch_indices in zip(
        control_chunks,
        target_expression_chunks,
        batch_index_chunks,
        strict=True,
    ):
        predicted_expression, predicted_latent = model.predict_response(
            control,
            gene,
            batch_indices,
        )
        predicted_expression_chunks.append(predicted_expression)
        predicted_latent_chunks.append(predicted_latent)
        hvg_energy_terms.append(
            _test_energy_distance(predicted_expression, target_expression)
        )
    control_expression = torch.cat(control_chunks, dim=0)
    target_expression = torch.cat(target_expression_chunks, dim=0)
    predicted_expression = torch.cat(predicted_expression_chunks, dim=0)
    predicted_latent = torch.cat(predicted_latent_chunks, dim=0)
    observed_latent = model.response_encoder(target_expression)
    control_latent = model.response_encoder(control_expression)
    hvg_mean_delta = _test_mean_delta_loss(
        predicted_expression,
        target_expression,
        model.control_expression_mean,
    )
    hvg_energy = torch.stack(hvg_energy_terms).mean()
    latent_mean_delta = F.mse_loss(
        predicted_latent.mean(dim=0),
        observed_latent.detach().mean(dim=0),
    )
    latent_energy = _test_energy_distance(predicted_latent, observed_latent.detach())
    pred_y = model.c_head(model.response_pooler(predicted_latent, control_latent))
    obs_y = model.c_head(model.response_pooler(observed_latent, control_latent))
    pred_c = F.mse_loss(pred_y.view(()), y.view(()))
    obs_c = F.mse_loss(obs_y.view(()), y.view(()))
    occupancy = F.mse_loss(
        model.response_pooler.occupancy(predicted_latent),
        model.response_pooler.occupancy(observed_latent).detach(),
    )
    gmm_nll = model.response_pooler.negative_log_likelihood(observed_latent)
    pred_rank = pred_y.sum() * 0.0
    total = (
        float(weights.hvg_mean_delta) * hvg_mean_delta
        + float(weights.hvg_energy) * hvg_energy
        + float(weights.latent_mean_delta) * latent_mean_delta
        + float(weights.latent_energy) * latent_energy
        + float(weights.pred_c) * pred_c
        + float(weights.obs_c) * obs_c
        + float(weights.occupancy) * occupancy
        + float(weights.gmm_nll) * gmm_nll
    )
    return {
        "total": total,
        "hvg_mean_delta": hvg_mean_delta,
        "hvg_energy": hvg_energy,
        "latent_mean_delta": latent_mean_delta,
        "latent_energy": latent_energy,
        "pred_c": pred_c,
        "obs_c": obs_c,
        "occupancy": occupancy,
        "gmm_nll": gmm_nll,
        "pred_rank": pred_rank,
        "pred_y": pred_y.view(()),
        "obs_y": obs_y.view(()),
    }


def _test_mean_delta_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    control_mean: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(
        predicted.mean(dim=0) - control_mean,
        target.mean(dim=0) - control_mean,
    )


def _test_energy_distance(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    cross = torch.cdist(predicted, target).mean()
    pred_self = torch.cdist(predicted, predicted).mean()
    target_self = torch.cdist(target, target).mean()
    return (2.0 * cross - pred_self - target_self).clamp_min(0.0)


def _toy_gene_bags_with_batches() -> GeneBags:
    input_bags = (
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.arange(12, 24, dtype=np.float32).reshape(4, 3),
    )
    latent_bags = tuple(bag[:, :2].astype(np.float32) for bag in input_bags)
    batch_bags = (
        np.asarray(["batch_a", "batch_a", "batch_b", "batch_b"], dtype=object),
        np.asarray(["batch_a", "batch_a", "batch_z", "batch_z"], dtype=object),
    )
    cell_type_bags = (
        np.asarray(["K562", "K562", "K562", "K562"], dtype=object),
        np.asarray(["K562", "K562", "K562", "K562"], dtype=object),
    )
    return GeneBags(
        genes=np.asarray(["GENE1", "GENE2"], dtype=object),
        y=np.asarray([-1.0, -0.5], dtype=np.float32),
        input_bags=input_bags,
        latent_bags=latent_bags,
        control_input=np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=np.float32,
        ),
        control_latent=np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
            dtype=np.float32,
        ),
        cell_type_bags=cell_type_bags,
        control_cell_type=np.asarray(["K562", "K562", "K562"], dtype=object),
        batch_bags=batch_bags,
        control_batch=np.asarray(["batch_a", "batch_a", "batch_b"], dtype=object),
        feature_names=None,
        feature_fill_values=np.zeros(3, dtype=np.float32),
        metadata=pd.DataFrame({"perturbation_gene": ["GENE1", "GENE2"]}),
        input_dim=3,
        latent_dim=2,
    )


def _write_toy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    genes = ["non-targeting"] * 4
    for gene in ["GENE1", "GENE2", "GENE3", "GENE4"]:
        genes.extend([gene] * 3)
    x = np.arange(len(genes) * 3, dtype=np.float32).reshape(len(genes), 3) / 10.0
    latent = np.stack([x[:, 0] - x[:, 1], x[:, 2]], axis=1).astype(np.float32)
    adata = ad.AnnData(x)
    adata.var_names = ["G0", "G1", "G2"]
    adata.obs["gene"] = genes
    adata.obsm["X_scVI"] = latent
    h5ad_path = tmp_path / "toy.h5ad"
    adata.write_h5ad(h5ad_path)
    overlap = pd.DataFrame(
        {
            "perturbation_gene": ["GENE1", "GENE2", "GENE3", "GENE4"],
            "depmap_gene_effect": [-1.2, -0.7, 0.1, 0.4],
            "has_depmap_label": [True, True, True, True],
        }
    )
    overlap_path = tmp_path / "overlap.csv"
    overlap.to_csv(overlap_path, index=False)
    return h5ad_path, overlap_path


def _write_scvi_cache_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "scvi_cache.yaml"
    config_path.write_text(
        f"""
data:
  h5ad_path: {tmp_path / "unused.h5ad"}
  overlap_csv: {tmp_path / "unused.csv"}
  output_dir: {tmp_path / "outputs"}
  obs_perturbation_col: gene
  control_label: non-targeting
state:
  backend: linear_mock
  input_dim: 3
  output_dim: 3
  pert_dim: 2
projector:
  teacher: scvi
  latent_dim: 2
  ridge_alpha: 0.1
  trainable: true
  scvi_max_epochs: 3
  scvi_hidden_units: 8
  scvi_layers: 1
  scvi_dropout: 0.0
train:
  run_id: cache
  seed: 13
  max_epochs: 1
  device: cpu
""",
    )
    return config_path


def _write_toy_external_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_a_labels = ["control", "control", "EXT1", "EXT1"]
    source_a = _write_external_source(
        tmp_path / "source_a.h5ad",
        source_a_labels,
        offset=0.0,
    )
    source_b_labels = ["control", "control", "EXT1_B", "EXT1_B", "EXT5", "EXT5"]
    source_b = _write_external_source(
        tmp_path / "source_b.h5ad",
        source_b_labels,
        offset=1.0,
    )
    overlap = pd.DataFrame(
        {
            "source_dataset": ["source_a", "source_b", "source_b"],
            "source_perturbation_label": ["EXT1", "EXT1_B", "EXT5"],
            "perturbation_gene": ["GENE1", "GENE1", "GENE5"],
            "depmap_gene_effect": [-1.2, -1.2, -0.2],
            "has_depmap_label": [True, True, True],
        }
    )
    overlap_path = tmp_path / "external_overlap.csv"
    overlap.to_csv(overlap_path, index=False)
    return source_a, source_b, overlap_path


def _assert_nested_tensors_close(actual: object, expected: object) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected)
        return
    if actual is None or expected is None:
        assert actual is expected
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_tensors_close(actual[key], expected[key])
        return
    if isinstance(actual, tuple) and isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_tensors_close(actual_value, expected_value)
        return
    assert actual == expected


def _write_external_source(path: Path, labels: list[str], offset: float) -> Path:
    x = (
        np.arange(len(labels) * 2, dtype=np.float32).reshape(len(labels), 2) / 10.0
        + offset
    )
    adata = ad.AnnData(x)
    adata.var_names = ["G0", "G2"]
    adata.var["gene_name"] = ["G0", "G2"]
    adata.obs["perturbation"] = labels
    adata.write_h5ad(path)
    return path


def _write_external_smoke_config(
    tmp_path: Path,
    h5ad_path: Path,
    overlap_path: Path,
    source_a: Path,
    source_b: Path,
    external_overlap: Path,
    *,
    run_id: str,
    max_epochs: int,
) -> Path:
    config_path = tmp_path / f"{run_id}.yaml"
    config_path.write_text(
        f"""
data:
  h5ad_path: {h5ad_path}
  overlap_csv: {overlap_path}
  output_dir: {tmp_path / "outputs"}
  obs_perturbation_col: gene
  control_label: non-targeting
  state_embed_key: null
  scvi_obsm_key: null
  depmap_label_col: depmap_gene_effect
  matched_label_col: has_depmap_label
  min_cells_per_gene: 2
external_test:
  name: adamson_k562
  overlap_csv: {external_overlap}
  sources:
    - name: source_a
      h5ad_path: {source_a}
      obs_perturbation_col: perturbation
      control_label: control
      var_gene_symbol_col: null
    - name: source_b
      h5ad_path: {source_b}
      obs_perturbation_col: perturbation
      control_label: control
      var_gene_symbol_col: null
split:
  train_fraction: 0.5
  val_fraction: 0.25
  test_fraction: 0.25
  train_genes: [GENE2, GENE3]
  val_genes: [GENE4]
  test_genes: [GENE1]
  random_state: 11
  stratify_bins: 2
state:
  backend: linear_mock
  input_dim: 3
  output_dim: 3
  pert_dim: 2
projector:
  teacher: obsm
  latent_dim: 3
  ridge_alpha: 0.1
  trainable: true
gmm:
  n_components: 2
  covariance_floor: 0.0001
  max_fit_cells: null
model:
  c_hidden_units: [8]
  dropout: 0.0
loss:
  latent_mean_delta_weight: 1.0
  latent_energy_weight: 1.0
  hvg_mean_delta_weight: 0.1
  hvg_energy_weight: 0.1
  pred_c_weight: 1.0
  obs_c_weight: 0.25
  occupancy_weight: 0.1
train:
  run_id: {run_id}
  seed: 13
  max_epochs: {max_epochs}
  learning_rate: 0.001
  weight_decay: 0.0
  cell_set_len: 2
  device: cpu
""",
    )
    return config_path
