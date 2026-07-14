"""Leakage-regression tests for the exp05 outer cross-validation protocol."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import pickle
import socket
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from accelerate.state import AcceleratorState

from aivc_model import cross_validate as cv
from aivc_model import gene_splits as gene_splits_module
from aivc_model import train as train_module
from aivc_model.distributed import (
    assert_all_ranks_stepped,
    require_exact_world_size,
    run_rank_zero_or_raise,
)
from aivc_model.gene_splits import FoldSpec, GeneAccessRecorder
from aivc_model.prepare import (
    AivcConfig,
    ExternalSourceConfig,
    ExternalTestConfig,
    GeneBags,
    SealedGeneBags,
    load_config,
)


class _SingleProcessAccelerator:
    is_main_process = True
    num_processes = 1
    device = torch.device("cpu")

    @staticmethod
    def wait_for_everyone() -> None:
        return None


class _FourRankMainAccelerator(_SingleProcessAccelerator):
    num_processes = 4
    device = torch.device("cuda", 0)
    _aivc_exp05_cuda_topology = (
        4,
        tuple(("cuda", index) for index in range(4)),
    )


def _mark_cuda_topology_validated(accelerator: object) -> object:
    setattr(
        accelerator,
        "_aivc_exp05_cuda_topology",
        (4, tuple(("cuda", index) for index in range(4))),
    )
    return accelerator


class _FakeAccelerator:
    def __init__(
        self,
        *,
        is_main_process: bool,
        num_processes: int,
        gathered: torch.Tensor | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.is_main_process = is_main_process
        self.num_processes = num_processes
        self.device = device
        self._gathered = gathered

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        return value if self._gathered is None else self._gathered


class _ResponseAccessTrap:
    def __init__(
        self,
        values: tuple[np.ndarray, ...],
        sealed_index: int,
    ) -> None:
        self._values = values
        self._sealed_index = sealed_index
        self.response_access_count = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> np.ndarray:
        if index == self._sealed_index:
            self.response_access_count += 1
            raise PermissionError("inner-validation observed response is sealed")
        return self._values[index]


def test_exp05_requires_exactly_four_ranks() -> None:
    with pytest.raises(RuntimeError, match="requires exactly 4 DDP ranks"):
        require_exact_world_size(types.SimpleNamespace(num_processes=1), expected=4)


def test_exp05_rejects_four_cpu_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    accelerator = _FakeAccelerator(is_main_process=True, num_processes=4)

    def gather(assignments: list[object], _local: object) -> None:
        assignments[:] = [("cpu", None)] * 4

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="requires CUDA on every rank"):
        require_exact_world_size(accelerator)  # type: ignore[arg-type]


def test_exp05_rejects_duplicate_cuda_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _FakeAccelerator(
        is_main_process=True,
        num_processes=4,
        device=torch.device("cuda", 0),
    )

    def gather(assignments: list[object], _local: object) -> None:
        assignments[:] = [("cuda", 0), ("cuda", 1), ("cuda", 1), ("cuda", 3)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="4 distinct CUDA device assignments"):
        require_exact_world_size(accelerator)  # type: ignore[arg-type]


def test_exp05_cuda_topology_guard_is_collective_once_per_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _FakeAccelerator(
        is_main_process=True,
        num_processes=4,
        device=torch.device("cuda", 0),
    )
    gather_calls = 0

    def gather(assignments: list[object], _local: object) -> None:
        nonlocal gather_calls
        gather_calls += 1
        assignments[:] = [("cuda", index) for index in range(4)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    require_exact_world_size(accelerator)  # type: ignore[arg-type]
    require_exact_world_size(accelerator)  # type: ignore[arg-type]

    assert gather_calls == 1


def test_rank_zero_exception_is_raised_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _FakeAccelerator(is_main_process=True, num_processes=4)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, src, device: None,
    )
    with pytest.raises(RuntimeError, match="checkpoint write failed.*disk full"):
        run_rank_zero_or_raise(
            accelerator,  # type: ignore[arg-type]
            "checkpoint write",
            lambda: (_ for _ in ()).throw(OSError("disk full")),
        )


def test_zero_optimizer_steps_on_any_rank_is_rejected() -> None:
    accelerator = _FakeAccelerator(
        is_main_process=True,
        num_processes=4,
        gathered=torch.tensor([8, 8, 0, 8]),
    )
    with pytest.raises(RuntimeError, match="rank optimizer-step counts.*0"):
        assert_all_ranks_stepped(accelerator, local_steps=8)  # type: ignore[arg-type]


def test_run_training_guards_audited_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _audited_config(tmp_path)
    accelerator = _SingleProcessAccelerator()
    guarded: list[object] = []
    data = _toy_bags()
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    monkeypatch.setattr(
        train_module,
        "require_exact_world_size",
        lambda value, expected: guarded.append((value, expected)),
        raising=False,
    )
    monkeypatch.setattr(train_module, "_run_audited_training", lambda **_kwargs: {})

    train_module.run_training(
        config,
        accelerator=accelerator,  # type: ignore[arg-type]
        train_data=data,
        val_data=data,
        sealed_test=SealedGeneBags(data, fold.test_genes),
        fold_spec=fold,
        run_dir_override=tmp_path / "fold_0",
        source_fingerprint="source",
        canonical_gene_order=tuple(str(gene) for gene in data.genes),
    )

    assert guarded == [(accelerator, 4)]


def test_run_training_fold_guards_direct_accelerator_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _audited_config(tmp_path)
    accelerator = _SingleProcessAccelerator()
    guarded: list[object] = []
    monkeypatch.setattr(cv, "_make_accelerator", lambda _config: accelerator)
    monkeypatch.setattr(
        cv,
        "require_exact_world_size",
        lambda value, expected: guarded.append((value, expected)),
    )
    monkeypatch.setattr(cv, "_prepare_fresh_run_dir", lambda *_args: None)
    monkeypatch.setattr(cv, "run_training", lambda *_args, **_kwargs: {})

    cv.run_training_fold(
        config=config,
        data=_toy_bags(),
        external=None,
        fold_spec=FoldSpec(0, ("A", "B"), ("C",), ("D",)),
        run_dir=tmp_path / "fold_0",
        source_fingerprint="source",
    )

    assert guarded == [(accelerator, 4)]


def test_run_training_fold_rejects_gene_batch_size_four_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _audited_config(tmp_path)
    config = replace(config, train=replace(config.train, gene_batch_size=4))
    monkeypatch.setattr(
        cv,
        "_make_accelerator",
        lambda _config: pytest.fail("Accelerator constructed before config rejection"),
    )

    with pytest.raises(ValueError, match="gene_batch_size must be 1"):
        cv.run_training_fold(
            config=config,
            data=_toy_bags(),
            external=None,
            fold_spec=FoldSpec(0, ("A", "B"), ("C",), ("D",)),
            run_dir=tmp_path / "fold_0",
            source_fingerprint="source",
        )

    assert not (tmp_path / "fold_0").exists()


@pytest.fixture(autouse=True)
def _isolate_accelerator_state() -> Iterator[None]:
    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


def _distributed_rank_safety_worker(
    rank: int,
    world_size: int,
    port: int,
    run_dir: str,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "ACCELERATE_USE_CPU": "true",
            "ACCELERATE_TORCH_DEVICE": "cpu",
        }
    )
    accelerator = cv.Accelerator(cpu=True)
    path = Path(run_dir)
    cv._prepare_fresh_run_dir(path, accelerator)

    def aggregate_once() -> None:
        marker = path / "aggregate_calls.txt"
        with marker.open("a", encoding="utf-8") as handle:
            handle.write(f"{rank}\n")

    run_rank_zero_or_raise(
        accelerator,
        "aggregate marker write",
        aggregate_once,
    )
    try:
        cv._prepare_fresh_run_dir(path, accelerator)
    except RuntimeError as error:
        assert f"fresh run directory required: {path}" in str(error)
        marker = path / f"freshness_error_rank_{rank}.txt"
        marker.write_text("rejected\n", encoding="utf-8")
    original_freshness_check = cv._require_fresh_run_dir
    cv._require_fresh_run_dir = lambda _path: (_ for _ in ()).throw(  # type: ignore[method-assign]
        PermissionError("permission sentinel")
    )
    try:
        cv._prepare_fresh_run_dir(path / "permission", accelerator)
    except RuntimeError as error:
        assert "permission sentinel" in str(error)
        marker = path / f"permission_error_rank_{rank}.txt"
        marker.write_text("propagated\n", encoding="utf-8")
    finally:
        cv._require_fresh_run_dir = original_freshness_check
    cv.run_preflight = lambda _path: (_ for _ in ()).throw(  # type: ignore[method-assign]
        ValueError("preflight sentinel")
    )
    try:
        cv._run_distributed_preflight(path / "config.yaml", accelerator)
    except RuntimeError as error:
        assert "preflight sentinel" in str(error)
        marker = path / f"preflight_error_rank_{rank}.txt"
        marker.write_text("propagated\n", encoding="utf-8")


def test_exp05_repaired_config_has_locked_contract(tmp_path: Path) -> None:
    path = Path("configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml")
    with pytest.raises(ValueError, match="gene_batch_size must be 1"):
        load_config(path)
    repaired_path = tmp_path / path.name
    repaired_path.write_text(
        path.read_text(encoding="utf-8").replace(
            "  gene_batch_size: 4\n",
            "  gene_batch_size: 1\n",
        ),
        encoding="utf-8",
    )
    config = load_config(repaired_path)
    assert config.data.h5ad_path.name == "K562_gwps_normalized_singlecell_01.h5ad"
    assert config.data.var_gene_symbol_col == "gene_name"
    assert config.data.state_hvg_n_top_genes is None
    assert config.state.gene_tokenizer == "esm2"
    assert config.state.esm2_npz is not None
    assert config.state.esm2_npz.name == "k562_gwps_depmap_esm2_650M.npz"
    assert config.state.require_resolved_esm2 is True
    assert config.state.representation_layer == "output"
    assert config.state.input_dim == 2000
    assert config.state.output_dim == 2000
    assert config.state.pert_dim == 2024
    assert config.train.gene_batch_size == 1
    assert config.train.freeze_state is True
    assert config.cv.n_splits == 5
    assert config.cv.expected_gene_count == 9338
    assert config.cv.outer_split_manifest is not None
    assert config.cv.outer_split_manifest.name == "k562_gwps_depmap_outer5_seed42.csv"
    assert config.cv.outer_split_sha256_file is not None
    assert config.cv.outer_split_sha256_file.name == (
        "k562_gwps_depmap_outer5_seed42.csv.sha256"
    )
    assert config.cv.inner_val_fraction == 0.1
    assert config.external_test is not None
    assert all(
        source.var_gene_symbol_col is None for source in config.external_test.sources
    )


def _toy_bags(genes: tuple[str, ...] = ("A", "B", "C", "D")) -> GeneBags:
    input_dim = 2000
    bags = tuple(
        np.full((2, input_dim), index + 1, dtype=np.float32)
        for index, _gene in enumerate(genes)
    )
    metadata = pd.DataFrame(
        {
            "perturbation_gene": genes,
            "depmap_gene_effect": np.linspace(-1.0, 0.0, len(genes)),
            "outer_fold": [1] * (len(genes) - 1) + [0],
        }
    )
    return GeneBags(
        genes=np.asarray(genes, dtype=object),
        y=metadata["depmap_gene_effect"].to_numpy(dtype=np.float32),
        input_bags=bags,
        latent_bags=tuple(bag.copy() for bag in bags),
        control_input=np.zeros((2, input_dim), dtype=np.float32),
        control_latent=np.zeros((2, input_dim), dtype=np.float32),
        cell_type_bags=None,
        control_cell_type=None,
        batch_bags=None,
        control_batch=None,
        feature_names=np.asarray(
            [f"FEATURE_{index}" for index in range(input_dim)],
            dtype=object,
        ),
        feature_fill_values=np.linspace(
            0.25,
            0.75,
            input_dim,
            dtype=np.float32,
        ),
        metadata=metadata,
        input_dim=input_dim,
        latent_dim=input_dim,
        gene_outer_folds=metadata["outer_fold"].to_numpy(dtype=np.int64),
    )


def _audited_config(tmp_path: Path) -> AivcConfig:
    config_path = tmp_path / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    split_path = tmp_path / "toy_outer.csv"
    split_path.write_text(
        "perturbation_gene,outer_fold\nA,1\nB,1\nC,1\nD,0\n",
        encoding="utf-8",
    )
    split_sha_path = tmp_path / "toy_outer.csv.sha256"
    split_sha_path.write_text(
        hashlib.sha256(split_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        f"""
data:
  h5ad_path: {tmp_path / "unused.h5ad"}
  overlap_csv: {tmp_path / "unused.csv"}
  output_dir: {tmp_path / "outputs"}
state:
  backend: linear_mock
  input_dim: 2000
  output_dim: 2000
  pert_dim: 2
response_encoder:
  input_dim: 2000
  latent_dim: 128
projector:
  teacher: obsm
  latent_dim: 2
  ridge_alpha: 0.1
gmm:
  n_components: 2
  init_scale: 0.02
  max_fit_cells: 8
cv:
  outer_split_manifest: {split_path}
  outer_split_sha256_file: {split_sha_path}
train:
  run_id: audited
  seed: 13
  max_epochs: 1
  cell_set_len: 2
  freeze_state: true
  device: cpu
""",
        encoding="utf-8",
    )
    return load_config(config_path)


def _validated_cpu_accelerator(config: AivcConfig) -> object:
    return _mark_cuda_topology_validated(train_module._make_accelerator(config))


def _run_tiny_audited_fold(tmp_path: Path) -> dict[str, Path]:
    config = _audited_config(tmp_path)
    return cv.run_training_fold(
        config=config,
        data=_toy_bags(),
        external=None,
        fold_spec=FoldSpec(0, ("A", "B"), ("C",), ("D",)),
        run_dir=tmp_path / "fold_0",
        source_fingerprint="source",
        accelerator=_validated_cpu_accelerator(config),  # type: ignore[arg-type]
    )


def _fingerprinted_config(tmp_path: Path) -> AivcConfig:
    config = _audited_config(tmp_path)
    cache_dir = tmp_path / "gwps_cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(
        '{"source_fingerprint":"gwps"}\n', encoding="utf-8"
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("perturbation_gene,label\nA,-1\n", encoding="utf-8")
    split = tmp_path / "outer.csv"
    split.write_text("perturbation_gene,outer_fold\nA,0\n", encoding="utf-8")
    split_sha = tmp_path / "outer.csv.sha256"
    split_sha.write_text(hashlib.sha256(split.read_bytes()).hexdigest() + "\n")
    esm = tmp_path / "esm2.npz"
    esm.write_bytes(b"esm")
    checkpoint = tmp_path / "state.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    model_dir = tmp_path / "state"
    model_dir.mkdir()
    (model_dir / "var_dims.pkl").write_bytes(b"sidecar")
    return replace(
        config,
        data=replace(
            config.data,
            prepared_cache_dir=cache_dir,
            overlap_csv=labels,
        ),
        cv=replace(
            config.cv,
            outer_split_manifest=split,
            outer_split_sha256_file=split_sha,
        ),
        state=replace(
            config.state,
            esm2_npz=esm,
            checkpoint_path=checkpoint,
            model_dir=model_dir,
        ),
    )


def _preflight_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AivcConfig:
    genes = tuple(f"G{index:02d}" for index in range(10))
    state_genes = ("A", "B", "C")
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "perturbation_gene": genes,
            "depmap_gene_effect": np.linspace(-1.0, 0.0, len(genes)),
            "has_depmap_label": True,
        }
    ).to_csv(labels, index=False)
    split = tmp_path / "outer.csv"
    pd.DataFrame(
        {
            "perturbation_gene": genes,
            "outer_fold": np.arange(len(genes)) % 5,
        }
    ).to_csv(split, index=False)
    split_sha = tmp_path / "outer.csv.sha256"
    split_sha.write_text(
        hashlib.sha256(split.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.save(cache_dir / "genes.npy", np.asarray(genes, dtype=object))
    np.save(cache_dir / "gene_outer_folds.npy", np.arange(len(genes)) % 5)
    np.save(cache_dir / "feature_names.npy", np.asarray(state_genes, dtype=object))
    np.save(cache_dir / "feature_fill_values.npy", np.zeros(3, dtype=np.float32))
    np.save(cache_dir / "cells.npy", np.zeros((1, 3), dtype=np.float32))
    np.save(cache_dir / "control_cells.npy", np.zeros((1, 3), dtype=np.float32))
    (cache_dir / "manifest.json").write_text(
        '{"schema_version":2,"source_fingerprint":"toy"}\n', encoding="utf-8"
    )

    esm_path = tmp_path / "esm2.npz"
    np.savez(
        esm_path,
        symbols=np.asarray(genes, dtype=object),
        vectors=np.ones((len(genes), 4), dtype=np.float32),
        resolved=np.ones(len(genes), dtype=bool),
    )

    response_labels = [*genes, "EXTRA1", "EXTRA2", "non-targeting", "non-targeting"]
    gwps = ad.AnnData(np.zeros((len(response_labels), 4), dtype=np.float32))
    gwps.obs["gene"] = response_labels
    gwps.var["gene_name"] = ["A", "B", "C", "D"]
    gwps_path = tmp_path / "gwps.h5ad"
    gwps.write_h5ad(gwps_path)

    model_dir = tmp_path / "state"
    model_dir.mkdir()
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump(
            {"gene_names": list(state_genes), "input_dim": 3, "output_dim": 3},
            handle,
        )
    checkpoint = model_dir / "final.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    source_paths = []
    for index, source_genes in enumerate((("A", "B"), ("B",), ("C",))):
        source = ad.AnnData(np.zeros((1, len(source_genes)), dtype=np.float32))
        source.var_names = list(source_genes)
        path = tmp_path / f"adamson_{index}.h5ad"
        source.write_h5ad(path)
        source_paths.append(path)

    base = _audited_config(tmp_path)
    config = replace(
        base,
        data=replace(
            base.data,
            h5ad_path=gwps_path,
            overlap_csv=labels,
            prepared_cache_dir=cache_dir,
            obs_perturbation_col="gene",
            var_gene_symbol_col="gene_name",
        ),
        cv=replace(
            base.cv,
            n_splits=5,
            expected_gene_count=10,
            outer_split_manifest=split,
            outer_split_sha256_file=split_sha,
        ),
        state=replace(
            base.state,
            backend="state_checkpoint",
            model_dir=model_dir,
            checkpoint_path=checkpoint,
            input_dim=3,
            output_dim=3,
            pert_dim=5,
            gene_tokenizer="esm2",
            esm2_npz=esm_path,
            require_resolved_esm2=True,
        ),
        external_test=ExternalTestConfig(
            name="adamson_k562",
            overlap_csv=labels,
            sources=tuple(
                ExternalSourceConfig(
                    name=name,
                    h5ad_path=path,
                    var_gene_symbol_col=None,
                )
                for name, path in zip(
                    (
                        "adamson_pilot",
                        "adamson_upr_epistasis",
                        "adamson_upr_perturb_seq",
                    ),
                    source_paths,
                    strict=True,
                )
            ),
        ),
    )
    monkeypatch.setattr(cv, "CANONICAL_GENE_COUNT", 10)
    monkeypatch.setattr(cv, "STATE_FEATURE_COUNT", 3)
    monkeypatch.setattr(cv, "EXPECTED_GWPS_SHAPE", (14, 4))
    monkeypatch.setattr(cv, "EXPECTED_GWPS_NONCONTROL_GENES", 12)
    monkeypatch.setattr(
        cv,
        "EXPECTED_ADAMSON_MATCHES",
        {
            "adamson_pilot": 2,
            "adamson_upr_epistasis": 1,
            "adamson_upr_perturb_seq": 1,
        },
    )
    monkeypatch.setattr(cv, "EXPECTED_STATE_PERT_DIM", 5)
    monkeypatch.setattr(cv, "load_config", lambda _path: config)
    monkeypatch.setattr(cv, "_checkpoint_pert_dim", lambda _config: 5)
    return config


def test_preflight_verifies_all_frozen_assets_and_reports_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _preflight_config(tmp_path, monkeypatch)

    report = cv.run_preflight(tmp_path / "config.yaml")

    assert report == {
        "gwps_shape": "14x4",
        "gwps_noncontrol_genes": 12,
        "gwps_depmap_overlap": 10,
        "canonical_split_genes": 10,
        "canonical_split_folds": 5,
        "canonical_split_sha256_length": 64,
        "esm2_resolved": "10/10",
        "state_expression_matches": "3/3",
        "state_input_dim": 3,
        "state_output_dim": 3,
        "state_pert_dim": 5,
        "adamson_pilot_matches": "2/3",
        "adamson_upr_epistasis_matches": "1/3",
        "adamson_upr_perturb_seq_matches": "1/3",
    }


def test_preflight_rejects_nonfinite_prepared_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _preflight_config(tmp_path, monkeypatch)
    assert config.data.prepared_cache_dir is not None
    cells_path = config.data.prepared_cache_dir / "cells.npy"
    cells = np.load(cells_path)
    cells[0, 0] = np.nan
    np.save(cells_path, cells)
    assert config.cv.outer_split_manifest is not None
    manifest = pd.read_csv(config.cv.outer_split_manifest)

    with pytest.raises(ValueError, match="cells.npy contains nonfinite"):
        cv._validate_prepared_cache(config, manifest)


def test_preflight_rejects_any_esm_gene_set_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _preflight_config(tmp_path, monkeypatch)
    assert config.state.esm2_npz is not None
    np.savez(
        config.state.esm2_npz,
        symbols=np.asarray([*(f"G{index:02d}" for index in range(9)), "EXTRA"]),
        vectors=np.ones((10, 4), dtype=np.float32),
        resolved=np.ones(10, dtype=bool),
    )

    with pytest.raises(ValueError, match="ESM-2 gene set"):
        cv.run_preflight(tmp_path / "config.yaml")


def test_preflight_cli_does_not_start_cross_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(cv, "run_preflight", lambda _path: {"esm2_resolved": "10/10"})
    monkeypatch.setattr(
        cv,
        "run_cross_validation",
        lambda *_args, **_kwargs: pytest.fail("cross-validation started"),
    )

    cv.main(["--config", str(tmp_path / "config.yaml"), "--preflight-only"])

    assert "esm2_resolved=10/10" in caplog.text


def test_cross_validation_uses_configured_prepared_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fingerprinted_config(tmp_path)
    sentinel = object()
    monkeypatch.setattr(
        cv,
        "load_gwps_cache",
        lambda _config, cache_dir: (
            sentinel
            if cache_dir == config.data.prepared_cache_dir
            else pytest.fail("wrong cache directory")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cv,
        "load_gene_bags",
        lambda _config: pytest.fail("raw GWPS loader used"),
    )

    assert cv._load_primary_bags(config) is sentinel


def test_experiment_source_fingerprint_changes_with_esm_cache(tmp_path: Path) -> None:
    config = _fingerprinted_config(tmp_path)

    first = cv.experiment_source_fingerprint(config)
    assert config.state.esm2_npz is not None
    config.state.esm2_npz.write_bytes(b"new-cache")
    second = cv.experiment_source_fingerprint(config)

    assert first != second


def test_experiment_source_fingerprint_hashes_small_state_sidecar_bytes(
    tmp_path: Path,
) -> None:
    config = _fingerprinted_config(tmp_path)

    first = cv.experiment_source_fingerprint(config)
    assert config.state.model_dir is not None
    (config.state.model_dir / "var_dims.pkl").write_bytes(b"changed-sidecar")
    second = cv.experiment_source_fingerprint(config)

    assert first != second


def test_gene_bag_views_preserve_only_requested_genes() -> None:
    bags = _toy_bags()
    with pytest.raises(ValueError, match="GeneAccessRecorder"):
        bags.for_genes(("C", "A"), stage="fine_tuning")
    bags = replace(
        bags,
        access_recorder=GeneAccessRecorder(FoldSpec(0, ("A", "C"), ("B",), ("D",))),
    )
    selected = bags.for_genes(("C", "A"), stage="fine_tuning")
    assert selected.genes.tolist() == ["C", "A"]
    assert selected.metadata["perturbation_gene"].tolist() == ["C", "A"]
    np.testing.assert_array_equal(selected.input_bags[0], bags.input_bags[2])
    np.testing.assert_array_equal(
        selected.feature_fill_values,
        bags.feature_fill_values,
    )


def test_actual_gene_view_rejects_unauthorized_access_without_recording() -> None:
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    recorder = GeneAccessRecorder(fold)
    bags = replace(_toy_bags(), access_recorder=recorder)

    with pytest.raises(ValueError, match="outer-test"):
        bags.for_genes(("D",), stage="response_encoder_fit")

    assert recorder.to_frame().empty
    selected = bags.for_genes(fold.train_genes, stage="fine_tuning")
    assert selected.genes.tolist() == ["A", "B"]
    assert recorder.to_frame()["stage"].tolist() == ["fine_tuning"]


def test_aggregation_rejects_unauthorized_emitted_event() -> None:
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    audit = pd.DataFrame(
        [
            {
                "stage": "fine_tuning",
                "outer_fold": 0,
                "gene_count": 2,
                "gene_set_sha256": cv._gene_set_sha256(fold.train_genes),
                "checkpoint_frozen": False,
            },
            {
                "stage": "response_encoder_fit",
                "outer_fold": 0,
                "gene_count": 1,
                "gene_set_sha256": cv._gene_set_sha256(fold.test_genes),
                "checkpoint_frozen": False,
            },
        ]
    )

    with pytest.raises(ValueError, match="authorized role"):
        cv._assert_access_audit(
            audit,
            [fold],
            types.SimpleNamespace(projector=types.SimpleNamespace(teacher="obsm")),
        )


def test_aggregation_rejects_missing_mandatory_stage(tmp_path: Path) -> None:
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    config = _audited_config(tmp_path)
    audit = pd.DataFrame(
        [
            {
                "stage": "fine_tuning",
                "outer_fold": 0,
                "gene_count": len(fold.train_genes),
                "gene_set_sha256": cv._gene_set_sha256(fold.train_genes),
                "checkpoint_frozen": False,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing mandatory audit stages"):
        cv._assert_access_audit(audit, [fold], config)


def test_two_process_rank_safety_creates_once_and_aggregates_once(
    tmp_path: Path,
) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    run_dir = tmp_path / "distributed_run"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_distributed_rank_safety_worker,
            args=(rank, 2, port, str(run_dir)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert run_dir.is_dir()
    assert (run_dir / "aggregate_calls.txt").read_text().splitlines() == ["0"]
    assert sorted(path.name for path in run_dir.glob("freshness_error_rank_*.txt")) == [
        "freshness_error_rank_0.txt",
        "freshness_error_rank_1.txt",
    ]
    assert sorted(path.name for path in run_dir.glob("preflight_error_rank_*.txt")) == [
        "preflight_error_rank_0.txt",
        "preflight_error_rank_1.txt",
    ]
    assert sorted(
        path.name for path in run_dir.glob("permission_error_rank_*.txt")
    ) == [
        "permission_error_rank_0.txt",
        "permission_error_rank_1.txt",
    ]


def test_cross_validation_runs_preflight_before_creating_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _audited_config(tmp_path)
    monkeypatch.setattr(cv, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cv,
        "run_preflight",
        lambda _path: (_ for _ in ()).throw(ValueError("locked preflight failed")),
    )
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, src, device: None,
    )

    with pytest.raises(RuntimeError, match="locked preflight failed"):
        cv.run_cross_validation(
            tmp_path / "config.yaml",
            accelerator=_FourRankMainAccelerator(),  # type: ignore[arg-type]
        )

    assert not (config.data.output_dir / "runs").exists()


def test_esm2_state_pert_dim_mismatch_fails_closed() -> None:
    state = types.SimpleNamespace(pert_dim=512)

    with pytest.raises(ValueError, match="2024"):
        train_module._effective_state_pert_dim(state, 2024)


def test_observed_b_shared_oracle_opens_only_after_checkpoint_freeze() -> None:
    bags = replace(
        _toy_bags(),
        access_recorder=GeneAccessRecorder(FoldSpec(0, ("A",), ("C",), ("B", "D"))),
    )
    sealed = SealedGeneBags(bags, ("B", "D"))
    with pytest.raises(PermissionError, match="selected checkpoint is frozen"):
        sealed.open(
            "observed_b_shared_oracle_outer_test",
            checkpoint_frozen=False,
        )
    for stage in (
        "generation_quality_outer_test",
        "observed_b_shared_oracle_outer_test",
    ):
        assert sealed.open(stage, checkpoint_frozen=True).genes.tolist() == ["B", "D"]
    label_view = sealed.label_view(checkpoint_frozen=True)
    assert label_view.genes.tolist() == ["B", "D"]
    assert all(bag.shape[0] == 0 for bag in label_view.input_bags)
    with pytest.raises(ValueError, match="only be opened"):
        sealed.open("fine_tuning", checkpoint_frozen=True)


def test_observed_b_shared_oracle_receives_frozen_eval_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = train_module._evaluate_observed_b_shared
    observed = False

    def assert_frozen(
        model: torch.nn.Module,
        data: GeneBags,
        device: torch.device,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        nonlocal observed
        observed = True
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        return original(model, data, device)

    monkeypatch.setattr(train_module, "_evaluate_observed_b_shared", assert_frozen)

    _run_tiny_audited_fold(tmp_path)

    assert observed


def test_inner_validation_never_reads_observed_response(tmp_path: Path) -> None:
    data = _toy_bags()
    input_trap = _ResponseAccessTrap(data.input_bags, sealed_index=2)
    latent_trap = _ResponseAccessTrap(data.latent_bags, sealed_index=2)
    sealed_validation = replace(
        data,
        input_bags=input_trap,  # type: ignore[arg-type]
        latent_bags=latent_trap,  # type: ignore[arg-type]
    )
    config = _audited_config(tmp_path)

    cv.run_training_fold(
        config=config,
        data=sealed_validation,
        external=None,
        fold_spec=FoldSpec(0, ("A", "B"), ("C",), ("D",)),
        run_dir=tmp_path / "fold_0",
        source_fingerprint="source",
        accelerator=_validated_cpu_accelerator(config),  # type: ignore[arg-type]
    )

    assert input_trap.response_access_count == 0
    assert latent_trap.response_access_count == 0


def test_run_training_fold_never_passes_outer_test_responses_to_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _toy_bags()
    fold = FoldSpec(0, ("A", "C"), ("B",), ("D",))
    captured: list[dict[str, object]] = []

    def fake_run_training(*_args: object, **kwargs: object) -> dict[str, Path]:
        train_data = kwargs["train_data"]
        val_data = kwargs["val_data"]
        sealed_test = kwargs["sealed_test"]
        assert isinstance(train_data, GeneBags)
        assert isinstance(val_data, GeneBags)
        assert isinstance(sealed_test, SealedGeneBags)
        captured.append(
            {
                "train_genes": train_data.genes.tolist(),
                "val_genes": val_data.genes.tolist(),
                "train_hash": hashlib.sha256(
                    train_data.input_bags[0].tobytes()
                ).hexdigest(),
                "sealed": sealed_test,
            }
        )
        run_dir = Path(str(kwargs["run_dir_override"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        audit = run_dir / "fit_audit_summary.json"
        audit.write_text(json.dumps({"checkpoint_sha256": captured[-1]["train_hash"]}))
        return {"fit_audit_summary": audit}

    monkeypatch.setattr(cv, "run_training", fake_run_training)
    common = {
        "config": types.SimpleNamespace(
            train=types.SimpleNamespace(
                gene_batch_size=1,
                required_world_size=4,
            )
        ),
        "external": None,
        "fold_spec": fold,
        "source_fingerprint": "source",
        "accelerator": _mark_cuda_topology_validated(
            _SingleProcessAccelerator()
        ),
    }
    cv.run_training_fold(data=data, run_dir=tmp_path / "first", **common)
    changed_bags = list(data.input_bags)
    changed_bags[3] = np.full_like(changed_bags[3], 999.0)
    changed = replace(data, input_bags=tuple(changed_bags))
    cv.run_training_fold(data=changed, run_dir=tmp_path / "changed", **common)

    assert captured[0]["train_genes"] == captured[1]["train_genes"] == ["A", "C"]
    assert captured[0]["val_genes"] == captured[1]["val_genes"] == ["B"]
    assert captured[0]["train_hash"] == captured[1]["train_hash"]


def test_audited_training_writes_three_final_scopes_and_fit_audit(
    tmp_path: Path,
) -> None:
    data = _toy_bags()
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    config = _audited_config(tmp_path)

    paths = cv.run_training_fold(
        config=config,
        data=data,
        external=None,
        fold_spec=fold,
        run_dir=tmp_path / "fold_0",
        source_fingerprint="source",
        accelerator=_validated_cpu_accelerator(config),  # type: ignore[arg-type]
    )

    predictions = pd.read_csv(paths["predictions"])
    assert set(predictions["evaluation_scope"]) == {
        "internal_outer_test",
        "generation_quality_outer_test",
        "observed_b_shared_oracle_outer_test",
    }
    audit = pd.read_csv(paths["fit_access_audit"])
    assert set(audit["stage"]) >= {
        "adapter_fit",
        "state_fit",
        "response_encoder_fit",
        "gmm_fit",
        "c_head_fit",
        "transition_supervision",
        "gene_prompt_fit",
        "fine_tuning",
        "early_stopping_prediction_only",
        "internal_outer_test",
        "generation_quality_outer_test",
        "observed_b_shared_oracle_outer_test",
    }
    assert set(audit["stage"]).isdisjoint(
        {
            "scvi_fit",
            "projector_fit",
            "normalizer_fit",
            "layer_selection",
            "observed_b_oracle_fit",
            "observed_b_oracle_selection",
            "observed_b_oracle_outer_test",
            "early_stopping",
        }
    )
    assert not audit.loc[audit["stage"].str.endswith("fit"), "checkpoint_frozen"].any()
    fit_summary = json.loads(paths["fit_audit_summary"].read_text())
    assert fit_summary["checkpoint_sha256"]
    assert fit_summary["state_sha256"]


def test_audited_exp05_never_calls_scvi_ridge_or_fixed_gmm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("removed precursor path was called")

    monkeypatch.setattr(train_module, "_fit_audited_scvi_latents", forbidden)
    monkeypatch.setattr(train_module, "_fit_or_load_projector_cache", forbidden)
    monkeypatch.setattr(train_module, "_fit_or_load_fixed_gmm_cache", forbidden)
    monkeypatch.setattr(train_module, "_fit_observed_b_oracle", forbidden)

    _run_tiny_audited_fold(tmp_path)


def test_audited_fit_summary_has_e2e_artifacts_only(tmp_path: Path) -> None:
    summary = _run_tiny_audited_fold(tmp_path)
    payload = json.loads(summary["fit_audit_summary"].read_text())
    artifacts_dir = summary["run_dir"] / "artifacts"

    assert "response_encoder_sha256" in payload
    assert "gmm_sha256" in payload
    assert "c_head_sha256" in payload
    assert "state_sha256" in payload
    assert "scvi_sha256" not in payload
    assert "projector_sha256" not in payload
    for directory in (
        "scvi_teacher_latents",
        "scvi_teacher_model",
        "ridge_projector_fit",
        "fixed_gmm_fit",
    ):
        assert not (artifacts_dir / directory).exists()


def test_outer_test_prediction_is_invariant_to_observed_response(
    tmp_path: Path,
) -> None:
    data = _toy_bags()
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    config = _audited_config(tmp_path)
    accelerator = _validated_cpu_accelerator(config)
    first = cv.run_training_fold(
        config=config,
        data=data,
        external=None,
        fold_spec=fold,
        run_dir=tmp_path / "first",
        source_fingerprint="source",
        accelerator=accelerator,  # type: ignore[arg-type]
    )
    changed_input = list(data.input_bags)
    changed_latent = list(data.latent_bags)
    changed_input[3] = np.full_like(changed_input[3], 999.0)
    changed_latent[3] = np.full_like(changed_latent[3], 999.0)
    changed = replace(
        data,
        input_bags=tuple(changed_input),
        latent_bags=tuple(changed_latent),
    )
    second = cv.run_training_fold(
        config=config,
        data=changed,
        external=None,
        fold_spec=fold,
        run_dir=tmp_path / "changed",
        source_fingerprint="source",
        accelerator=accelerator,  # type: ignore[arg-type]
    )
    first_audit = json.loads(first["fit_audit_summary"].read_text())
    second_audit = json.loads(second["fit_audit_summary"].read_text())
    for key in (
        "adapter_sha256",
        "state_sha256",
        "response_encoder_sha256",
        "gmm_sha256",
        "c_head_sha256",
        "best_epoch",
        "checkpoint_sha256",
    ):
        assert first_audit[key] == second_audit[key]
    first_predictions = pd.read_csv(first["predictions"])
    second_predictions = pd.read_csv(second["predictions"])
    first_internal = first_predictions.query(
        "evaluation_scope == 'internal_outer_test'"
    ).reset_index(drop=True)
    second_internal = second_predictions.query(
        "evaluation_scope == 'internal_outer_test'"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(first_internal, second_internal)


def test_audited_fold_artifacts_share_exact_fit_authority(tmp_path: Path) -> None:
    run_dir = tmp_path / "authority"
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    config = _fingerprinted_config(tmp_path)
    accelerator = _validated_cpu_accelerator(config)

    cv.run_training_fold(
        config=config,
        data=_toy_bags(),
        external=None,
        fold_spec=fold,
        run_dir=run_dir,
        source_fingerprint="source",
        accelerator=accelerator,  # type: ignore[arg-type]
    )

    expected_hash = train_module._sha256_strings(fold.train_genes)
    artifact_kinds = {
        "esm_adapter_fit": "esm_adapter",
        "state_fit": "state",
        "response_encoder_fit": "response_encoder",
        "gmm_fit": "trainable_gmm",
        "c_head_fit": "c_head",
    }
    metadata_paths = [
        *(
            run_dir / "artifacts" / directory / "metadata.json"
            for directory in artifact_kinds
        ),
        run_dir / "models" / "best" / "metadata.json",
        run_dir / "models" / "final" / "metadata.json",
    ]
    for path in metadata_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["source_fingerprint"] == "source"
        assert metadata["canonical_split_sha256"]
        assert metadata["outer_fold"] == 0
        assert metadata["fit_stage"] == "inner_train"
        assert metadata["fit_genes_sha256"] == expected_hash
        assert metadata["train_genes"] == ["A", "B"]
        assert metadata["val_genes"] == ["C"]
        assert metadata["test_genes"] == ["D"]
        assert set(metadata["fit_genes"]).isdisjoint(metadata["test_genes"])
    for directory, expected_kind in artifact_kinds.items():
        metadata = json.loads(
            (run_dir / "artifacts" / directory / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["kind"] == expected_kind


def test_projector_cache_rejects_source_change_and_test_gene_contamination(
    tmp_path: Path,
) -> None:
    config = _audited_config(tmp_path)
    data = _toy_bags(("A", "B"))
    split = train_module.GeneSplit(
        train=np.asarray([0, 1], dtype=np.int64),
        val=np.asarray([], dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    first_authority = train_module._fold_artifact_authority(
        config,
        fold,
        source_fingerprint="aaa",
        canonical_split_sha256="split",
    )
    metadata = train_module._projector_cache_metadata(
        config, data, split, authority=first_authority
    )
    train_module._write_projector_cache(tmp_path, metadata, np.eye(2), np.zeros(2))
    changed_authority = replace(first_authority, source_fingerprint="bbb")
    changed = train_module._projector_cache_metadata(
        config, data, split, authority=changed_authority
    )
    contaminated = {
        **metadata,
        "train_genes": ["A", "B", "D"],
        "fit_genes_sha256": train_module._sha256_strings(("A", "B", "D")),
    }

    assert train_module._load_projector_cache(tmp_path, changed) is None
    (tmp_path / "metadata.json").write_text(json.dumps(contaminated), encoding="utf-8")
    assert train_module._load_projector_cache(tmp_path, contaminated) is None


def test_projector_cache_rejects_persisted_non_validation_selection_gene(
    tmp_path: Path,
) -> None:
    config = _audited_config(tmp_path)
    data = _toy_bags(("A", "B"))
    split = train_module.GeneSplit(
        train=np.asarray([0, 1], dtype=np.int64),
        val=np.asarray([], dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )
    authority = train_module._fold_artifact_authority(
        config,
        FoldSpec(0, ("A", "B"), ("C",), ("D",)),
        source_fingerprint="source",
        canonical_split_sha256="split",
    )
    metadata = train_module._projector_cache_metadata(
        config, data, split, authority=authority
    )
    contaminated = {**metadata, "selection_genes": ["C", "D"]}
    train_module._write_projector_cache(tmp_path, contaminated, np.eye(2), np.zeros(2))

    assert train_module._load_projector_cache(tmp_path, contaminated) is None


def test_audited_checkpoint_loader_requires_exact_persisted_authority(
    tmp_path: Path,
) -> None:
    config = _audited_config(tmp_path)
    authority = train_module._fold_artifact_authority(
        config,
        FoldSpec(0, ("A", "B"), ("C",), ("D",)),
        source_fingerprint="source",
        canonical_split_sha256="split",
    )
    checkpoint_dir = tmp_path / "best"
    checkpoint_dir.mkdir()
    torch.save({"weight": torch.ones(1)}, checkpoint_dir / "pytorch_model.bin")
    metadata = {**authority.metadata(), "checkpoint_kind": "best"}
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    loaded = train_module._load_authorized_model_checkpoint(
        checkpoint_dir,
        authority,
        map_location="cpu",
    )
    torch.testing.assert_close(loaded["weight"], torch.ones(1))

    for contaminated in (
        {
            **metadata,
            "train_genes": ["A", "B", "D"],
            "fit_genes_sha256": train_module._sha256_strings(("A", "B", "D")),
        },
        {**metadata, "selection_genes": ["C", "D"]},
        {**metadata, "source_fingerprint": "changed"},
    ):
        (checkpoint_dir / "metadata.json").write_text(
            json.dumps(contaminated), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="checkpoint authority"):
            train_module._load_authorized_model_checkpoint(
                checkpoint_dir,
                authority,
                map_location="cpu",
            )


def test_audited_checkpoint_loader_rejects_missing_schema_v2_metadata(
    tmp_path: Path,
) -> None:
    config = _audited_config(tmp_path)
    authority = train_module._fold_artifact_authority(
        config,
        FoldSpec(0, ("A", "B"), ("C",), ("D",)),
        source_fingerprint="source",
        canonical_split_sha256="split",
    )
    checkpoint_dir = tmp_path / "final"
    checkpoint_dir.mkdir()
    torch.save({"weight": torch.ones(1)}, checkpoint_dir / "pytorch_model.bin")

    with pytest.raises(ValueError, match="schema-v2 metadata"):
        train_module._load_authorized_model_checkpoint(
            checkpoint_dir,
            authority,
            map_location="cpu",
        )


def test_run_training_fold_rejects_nonempty_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "fold_0"
    run_dir.mkdir()
    (run_dir / "stale.txt").write_text("stale", encoding="utf-8")

    config = _audited_config(tmp_path)
    with pytest.raises(RuntimeError, match="fresh run directory"):
        cv.run_training_fold(
            config=config,
            data=_toy_bags(),
            external=None,
            fold_spec=FoldSpec(0, ("A", "B"), ("C",), ("D",)),
            run_dir=run_dir,
            source_fingerprint="source",
            accelerator=_validated_cpu_accelerator(config),  # type: ignore[arg-type]
        )


def test_audited_esm_uses_exact_canonical_manifest_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_order = ("D", "B", "A", "C")
    manifest = pd.DataFrame(
        {
            "perturbation_gene": canonical_order,
            "outer_fold": [0, 1, 1, 1],
        }
    )
    manifest_path = tmp_path / "outer.csv"
    manifest.to_csv(manifest_path, index=False)
    sha_path = tmp_path / "outer.csv.sha256"
    sha_path.write_text(
        f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}\n",
        encoding="utf-8",
    )
    esm_path = tmp_path / "esm2.npz"
    np.savez(
        esm_path,
        symbols=np.asarray(canonical_order, dtype=object),
        vectors=np.arange(12, dtype=np.float32).reshape(4, 3),
        resolved=np.ones(4, dtype=bool),
    )
    base = _audited_config(tmp_path)
    config = replace(
        base,
        cv=replace(
            base.cv,
            outer_split_manifest=manifest_path,
            outer_split_sha256_file=sha_path,
        ),
        state=replace(
            base.state,
            gene_tokenizer="esm2",
            esm2_npz=esm_path,
            esm2_adapter_hidden=4,
            require_resolved_esm2=True,
        ),
    )
    accelerator = _validated_cpu_accelerator(config)
    monkeypatch.setattr(train_module, "CANONICAL_GENE_COUNT", 4)
    monkeypatch.setattr(gene_splits_module, "CANONICAL_GENE_COUNT", 4)
    monkeypatch.setattr(
        gene_splits_module,
        "CANONICAL_OUTER_FOLDS",
        frozenset({0, 1}),
    )

    paths = cv.run_training_fold(
        config=config,
        data=_toy_bags(),
        external=None,
        fold_spec=FoldSpec(0, ("A", "C"), ("B",), ("D",)),
        run_dir=tmp_path / "esm_fold",
        source_fingerprint="source",
        canonical_gene_order=canonical_order,
        accelerator=accelerator,  # type: ignore[arg-type]
    )

    evidence = json.loads(paths["runtime_evidence"].read_text())
    assert evidence["esm_resolved_count"] == 4
    assert (
        evidence["esm_gene_order_sha256"]
        == hashlib.sha256("\n".join(canonical_order).encode("utf-8")).hexdigest()
    )


def test_cross_validation_writes_each_outer_test_gene_once_per_final_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genes = tuple(f"GENE{index:05d}" for index in range(9338))
    labels = pd.DataFrame(
        {
            "perturbation_gene": genes,
            "depmap_gene_effect": np.linspace(-2.0, 1.0, len(genes)),
        }
    )
    manifest = pd.DataFrame(
        {"perturbation_gene": genes, "outer_fold": np.arange(len(genes)) % 5}
    )
    manifest_path = tmp_path / "outer.csv"
    manifest.to_csv(manifest_path, index=False)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sha_path = tmp_path / "outer.csv.sha256"
    sha_path.write_text(f"{digest}\n", encoding="utf-8")
    config = types.SimpleNamespace(
        data=types.SimpleNamespace(output_dir=tmp_path),
        train=types.SimpleNamespace(
            run_id="toy",
            seed=42,
            device="cpu",
            required_world_size=4,
            gene_batch_size=1,
        ),
        projector=types.SimpleNamespace(teacher="obsm"),
        cv=types.SimpleNamespace(
            outer_split_manifest=manifest_path,
            outer_split_sha256_file=sha_path,
            inner_val_fraction=0.1,
            random_state=42,
        ),
    )
    monkeypatch.setattr(cv, "load_config", lambda _path: config)
    monkeypatch.setattr(cv, "run_preflight", lambda _path: {})
    monkeypatch.setattr(
        cv,
        "load_gene_bags",
        lambda _config: types.SimpleNamespace(
            genes=np.asarray(genes, dtype=object),
            y=labels["depmap_gene_effect"].to_numpy(),
        ),
    )
    monkeypatch.setattr(cv, "load_external_gene_bags", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv, "experiment_source_fingerprint", lambda _config: "source")

    def fake_fold_runner(**kwargs: object) -> dict[str, Path]:
        fold = kwargs["fold_spec"]
        assert isinstance(fold, FoldSpec)
        fold_dir = Path(str(kwargs["run_dir"]))
        artifacts = fold_dir / "artifacts"
        artifacts.mkdir(parents=True)
        rows = []
        for scope in (
            "internal_outer_test",
            "generation_quality_outer_test",
            "observed_b_shared_oracle_outer_test",
        ):
            rows.extend(
                {
                    "perturbation_gene": gene,
                    "outer_fold": fold.outer_fold,
                    "inner_role": "outer_test",
                    "evaluation_scope": scope,
                    "y_true": 0.0,
                    "y_pred": 0.0,
                }
                for gene in fold.test_genes
            )
        rows.append(
            {
                "perturbation_gene": f"ADAMSON_ONLY_{fold.outer_fold}",
                "outer_fold": fold.outer_fold,
                "inner_role": "external_test",
                "evaluation_scope": "external:adamson_k562",
                "y_true": 0.0,
                "y_pred": 0.0,
            }
        )
        predictions = artifacts / "predictions.csv"
        pd.DataFrame(rows).to_csv(predictions, index=False)
        metrics = artifacts / "fold_metrics.csv"
        pd.DataFrame(
            [
                {"outer_fold": fold.outer_fold, "evaluation_scope": scope, "mse": 0.0}
                for scope in (
                    "internal_outer_test",
                    "generation_quality_outer_test",
                    "observed_b_shared_oracle_outer_test",
                )
            ]
        ).to_csv(metrics, index=False)
        audit = artifacts / "fit_access_audit.csv"
        pd.DataFrame(
            [
                {
                    "stage": stage,
                    "outer_fold": fold.outer_fold,
                    "gene_count": len(cv._stage_genes(stage, fold)),
                    "gene_set_sha256": cv._gene_set_sha256(
                        cv._stage_genes(stage, fold)
                    ),
                    "checkpoint_frozen": stage
                    in {
                        "internal_outer_test",
                        "generation_quality_outer_test",
                        "observed_b_shared_oracle_outer_test",
                    },
                }
                for stage in sorted(cv._mandatory_audit_stages(config))
            ]
        ).to_csv(audit, index=False)
        qa = artifacts / "external_alignment_qa.csv"
        pd.DataFrame(columns=["outer_fold", "source_name"]).to_csv(qa, index=False)
        summary = fold_dir / "fit_audit_summary.json"
        summary.write_text(json.dumps({"checkpoint_sha256": str(fold.outer_fold)}))
        runtime_evidence = fold_dir / "runtime_evidence.json"
        runtime_evidence.write_text(
            json.dumps(
                {
                    "esm_resolved_count": 9338,
                    "esm_total_count": 9338,
                    "esm_gene_order_sha256": hashlib.sha256(
                        "\n".join(genes).encode("utf-8")
                    ).hexdigest(),
                    "state_input_dim": 2000,
                    "state_output_dim": 2000,
                    "state_pert_dim": 2024,
                    "state_feature_match_count": 2000,
                }
            ),
            encoding="utf-8",
        )
        return {
            "predictions": predictions,
            "fold_metrics": metrics,
            "fit_access_audit": audit,
            "external_alignment_qa": qa,
            "fit_audit_summary": summary,
            "runtime_evidence": runtime_evidence,
        }

    monkeypatch.setattr(cv, "run_training_fold", fake_fold_runner)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, src, device: None,
    )
    run_dir = cv.run_cross_validation(
        tmp_path / "config.yaml",
        accelerator=_FourRankMainAccelerator(),  # type: ignore[arg-type]
    )
    predictions = pd.read_csv(run_dir / "artifacts" / "predictions.csv")
    for scope in (
        "internal_outer_test",
        "generation_quality_outer_test",
        "observed_b_shared_oracle_outer_test",
    ):
        rows = predictions.query("evaluation_scope == @scope")
        assert rows["perturbation_gene"].nunique() == 9338
        assert not rows.duplicated(["perturbation_gene", "evaluation_scope"]).any()
    canonical_output = run_dir / "artifacts" / "gene_splits.csv"
    assert len(pd.read_csv(canonical_output)) == 9338
    assert hashlib.sha256(canonical_output.read_bytes()).hexdigest() == digest
    assert len(pd.read_csv(run_dir / "artifacts" / "fold_roles.csv")) == 5 * 9338
    assert (run_dir / "summary.csv").exists()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert run_manifest["canonical_split_sha256"] == digest
    assert run_manifest["esm_resolved_count"] == 9338
    assert run_manifest["esm_total_count"] == 9338
    assert run_manifest["state_feature_match_count"] == 2000
    assert run_manifest["checkpoint_dimensions"]["pert_dim"] == 2024
