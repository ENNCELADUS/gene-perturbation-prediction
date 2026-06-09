from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import torch

import aivc_model.train as train_module
from aivc_model.model import (
    AivcModel,
    ExpressionToLatentProjector,
    LossWeights,
    MLPHead,
    PerturbationVectorAdapter,
    StateForwardAdapter,
    fit_fixed_gmm,
    load_state_model,
)
from aivc_model.prepare import (
    GeneBags,
    GeneSplit,
    ProjectorConfig,
    SplitConfig,
    _load_scvi_latent_cache,
    _project_scvi_latent_collections,
    _project_scvi_latent_groups,
    _scvi_datasplitter_kwargs,
    _scvi_latent_cache_dir,
    _suppress_scvi_lightning_warnings,
    _scvi_trainer_kwargs,
    _write_scvi_latent_cache,
    encode_batch_labels,
    fit_linear_projector,
    load_external_gene_bags,
    load_config,
    load_perturbation_vectors,
    load_gene_bags,
    load_state_batch_lookup,
    make_cell_set_chunks,
    make_gene_split,
    with_cached_scvi_teacher_latents,
)
from aivc_model.train import _write_csv_if_main, run_training


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


def test_accelerator_enables_ddp_unused_parameter_detection(tmp_path: Path) -> None:
    config = load_config(_write_scvi_cache_config(tmp_path))

    accelerator = train_module._make_accelerator(config)

    assert accelerator.ddp_handler is not None
    assert accelerator.ddp_handler.find_unused_parameters is True


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


def test_a_to_b_set_loss_is_target_order_invariant() -> None:
    model, _state_model = _build_tiny_aivc_model()
    control = torch.zeros(4, 3)
    target_expression = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]]
    )
    weight = model.projector.linear.weight.detach()
    bias = model.projector.linear.bias.detach()
    target_latent = target_expression @ weight.T + bias
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


def test_train_smoke_writes_minimal_csv_outputs(tmp_path: Path) -> None:
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

    paths = run_training(load_config(config_path))

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
        "val_total_loss",
    }.issubset(train_log.columns)
    expected_loss_cols = {
        "hvg_mean_delta",
        "hvg_energy",
        "latent_mean_delta",
        "latent_energy",
        "occupancy",
        "pred_c",
        "obs_c",
        "total_loss",
    }
    assert expected_loss_cols.issubset(test_metrics.columns)
    assert {"rmse", "spearman"}.issubset(test_metrics.columns)


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

    paths = run_training(load_config(config_path))

    run_dir = paths["run_dir"]
    test_metrics = pd.read_csv(paths["test_metrics"])
    predictions = pd.read_csv(run_dir / "artifacts" / "test_predictions.csv")
    assert test_metrics["evaluation_scope"].tolist() == ["external:adamson_k562"]
    assert set(predictions["perturbation_gene"]) == {"GENE1", "GENE5"}
    assert set(predictions["evaluation_scope"]) == {"external:adamson_k562"}
    assert "source_dataset" in predictions.columns
    assert "perturbation_has_known_vector" in predictions.columns
    assert not predictions["perturbation_has_known_vector"].any()
    assert (run_dir / "artifacts" / "external_test_qa.json").exists()
    assert (run_dir / "models" / "best" / "pytorch_model.bin").exists()
    assert (run_dir / "models" / "final" / "pytorch_model.bin").exists()


def _loss_weights() -> LossWeights:
    return LossWeights(
        latent_mean_delta=1.0,
        latent_energy=1.0,
        hvg_mean_delta=0.1,
        hvg_energy=0.1,
        pred_c=1.0,
        obs_c=0.25,
        occupancy=0.1,
    )


def _build_tiny_aivc_model() -> tuple[AivcModel, torch.nn.Module]:
    state_model = load_state_model(
        backend="linear_mock",
        checkpoint_path=None,
        input_dim=3,
        output_dim=3,
        pert_dim=2,
    )
    perturbations = PerturbationVectorAdapter(["GENE1"], {}, pert_dim=2)
    weight, bias = fit_linear_projector(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        alpha=0.1,
    )
    projector = ExpressionToLatentProjector(weight, bias, trainable=True)
    control_latent = np.asarray([[0.0, 0.0], [0.1, 0.1]], dtype=np.float32)
    featureizer = fit_fixed_gmm(
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
    model = AivcModel(
        state_adapter=StateForwardAdapter(state_model),
        perturbations=perturbations,
        projector=projector,
        featureizer=featureizer,
        c_head=MLPHead(featureizer.output_dim, (8,), 0.0),
        control_expression_mean=np.zeros(3, dtype=np.float32),
        control_latent_mean=np.zeros(2, dtype=np.float32),
    )
    return model, state_model


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
      var_gene_symbol_col: gene_name
    - name: source_b
      h5ad_path: {source_b}
      obs_perturbation_col: perturbation
      control_label: control
      var_gene_symbol_col: gene_name
split:
  train_fraction: 0.75
  val_fraction: 0.25
  test_fraction: 0.0
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
