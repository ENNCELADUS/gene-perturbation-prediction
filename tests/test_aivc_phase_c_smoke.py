"""e2e CPU smoke test (Wave 2 Phase C, Task 7): the whole wave composes.

Every prior task (1-6) tested only its own slice: GeneBags' target-space
fields (Task 1), the response encoder / ST input split (Task 2), the width
guard (Task 3), the warm-start loader (Task 4), the multi-line assembly
(Task 5), the GWPS cache fingerprint (Task 6). Nothing yet ran the ASSEMBLED
path -- a two-view GeneBags (2560-d Tx1-shaped ST input, 2000-d gene-shaped
response-encoder target) through ``aivc_model.train.run_training``'s
existing training loop -- end to end. This module is that gate.

No real Tx1/Perturb-seq data is used: the ``linear_mock`` STATE backend and
small synthetic two-view ``GeneBags`` let this run on CPU in seconds while
still exercising the real ``model.py``/``train.py`` forward and loss-
combination code (not a re-implementation of it).
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aivc_model import model as model_module
from aivc_model import train as train_module
from aivc_model.gene_splits import FoldSpec, GeneAccessRecorder
from aivc_model.prepare import (
    AivcConfig,
    GeneBags,
    SealedGeneBags,
    fit_linear_projector,
    load_config,
)

_INPUT_DIM = 2560  # Tx1 embedding width (tx1_embed_cache.EMBEDDING_WIDTH)
_TARGET_DIM = 2000  # gene/HVG width
_PERT_DIM = 8
_CELL_SET_LEN = 4
_N_CONTROL_CELLS = 12
_TRAIN_GENES = ("G0", "G1", "G2", "G3")
_VAL_GENES = ("G4",)
_TEST_GENES = ("G5",)
_ALL_GENES = _TRAIN_GENES + _VAL_GENES + _TEST_GENES


def _synthetic_two_view_bags(seed: int = 0) -> GeneBags:
    """A small, real (non-file-backed) two-view GeneBags matching Task 5's shape.

    ``input_bags`` are Tx1-embedding-space NaN placeholders and
    ``control_input`` is the only real Tx1-shaped array actually read as
    ST's input, mirroring ``tx1_response_data.assemble_train_response_gene_
    bags``'s own convention exactly (its module docstring's design decision
    3) -- this test exercises the SAME shape real Phase C training data will
    have, not a simplified stand-in.
    """
    rng = np.random.default_rng(seed)
    control_input = rng.normal(size=(_N_CONTROL_CELLS, _INPUT_DIM)).astype(np.float32)
    control_target = rng.normal(
        loc=1.0, scale=0.5, size=(_N_CONTROL_CELLS, _TARGET_DIM)
    ).astype(np.float32)

    input_bags = []
    target_bags = []
    metadata_rows = []
    for gene in _ALL_GENES:
        n_cells = _CELL_SET_LEN + 2
        input_bags.append(np.full((n_cells, _INPUT_DIM), np.nan, dtype=np.float32))
        # A real, learnable signal: target expression offset from the
        # control mean, so hvg_mean_delta/hvg_energy/latent_* have
        # something non-trivial to reduce via gradient descent.
        target_bags.append(
            rng.normal(loc=2.5, scale=0.5, size=(n_cells, _TARGET_DIM)).astype(
                np.float32
            )
        )
        metadata_rows.append({"perturbation_gene": gene, "n_cells": n_cells})

    metadata = pd.DataFrame(metadata_rows)
    feature_names = np.asarray(
        [f"tx1_embedding_{i}" for i in range(_INPUT_DIM)], dtype=object
    )
    target_feature_names = np.asarray(
        [f"GENE{i}" for i in range(_TARGET_DIM)], dtype=object
    )
    target_bags_t = tuple(target_bags)
    return GeneBags(
        genes=np.asarray(_ALL_GENES, dtype=object),
        y=np.full(len(_ALL_GENES), np.nan, dtype=np.float32),
        input_bags=tuple(input_bags),
        latent_bags=target_bags_t,
        control_input=control_input,
        control_latent=control_target,
        cell_type_bags=None,
        control_cell_type=None,
        batch_bags=None,
        control_batch=None,
        feature_names=feature_names,
        metadata=metadata,
        input_dim=_INPUT_DIM,
        latent_dim=_TARGET_DIM,
        target_bags=target_bags_t,
        control_target=control_target,
        target_dim=_TARGET_DIM,
        target_feature_names=target_feature_names,
    )


def _smoke_config(tmp_path: Path, *, max_epochs: int) -> AivcConfig:
    # _run_audited_training requires an artifact-authority split SHA-256
    # file (train._canonical_split_sha256); its *content* is irrelevant to
    # this GeneBags' own gene universe (this smoke test does not exercise
    # exp05's canonical-manifest machinery), only its presence and shape.
    split_path = tmp_path / "toy_outer.csv"
    split_path.write_text(
        "perturbation_gene,outer_fold\nG0,1\nG1,1\nG2,1\nG3,0\n",
        encoding="utf-8",
    )
    split_sha_path = tmp_path / "toy_outer.csv.sha256"
    split_sha_path.write_text(
        hashlib.sha256(split_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "smoke_config.yaml"
    config_path.write_text(
        f"""
data:
  h5ad_path: {tmp_path / "unused.h5ad"}
  overlap_csv: {tmp_path / "unused.csv"}
  output_dir: {tmp_path / "outputs"}
cv:
  outer_split_manifest: {split_path}
  outer_split_sha256_file: {split_sha_path}
state:
  backend: linear_mock
  input_dim: {_INPUT_DIM}
  output_dim: {_TARGET_DIM}
  pert_dim: {_PERT_DIM}
  input_view: obsm
response_encoder:
  input_dim: {_TARGET_DIM}
  latent_dim: 128
gmm:
  n_components: 2
  init_scale: 0.02
  trainable: true
loss:
  latent_mean_delta_weight: 1.0
  latent_energy_weight: 1.0
  hvg_mean_delta_weight: 1.0
  hvg_energy_weight: 1.0
  pred_c_weight: 0.0
  obs_c_weight: 0.0
  occupancy_weight: 0.0
  gmm_nll_weight: 0.0
  pred_rank_weight: 0.0
train:
  run_id: phase_c_smoke
  seed: 7
  max_epochs: {max_epochs}
  learning_rate: 0.02
  state_learning_rate: 0.02
  weight_decay: 0.0
  cell_set_len: {_CELL_SET_LEN}
  gene_batch_size: 1
  eval_control_panel_size: {_CELL_SET_LEN}
  eval_window_macro_batch_size: 4
  required_world_size: 4
  device: cpu
""",
        encoding="utf-8",
    )
    return load_config(config_path)


def _fold_bags(bags: GeneBags) -> tuple[FoldSpec, GeneBags, GeneBags, SealedGeneBags]:
    fold = FoldSpec(
        outer_fold=0,
        train_genes=_TRAIN_GENES,
        val_genes=_VAL_GENES,
        test_genes=_TEST_GENES,
    )
    recorder = GeneAccessRecorder(fold)
    data = replace(bags, access_recorder=recorder)
    train_data = data.for_genes(fold.train_genes, stage="adapter_fit")
    val_data = data.for_prediction_genes(
        fold.val_genes,
        stage="early_stopping_prediction_only",
        generation_targets=True,
    )
    sealed_test = SealedGeneBags(data, fold.test_genes)
    return fold, train_data, val_data, sealed_test


def _bypass_runtime_world_size_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the *runtime* rank-count check, not the config-level one.

    ``required_world_size`` must be exactly 4 in the config regardless
    (``train._require_authoritative_gene_batch_size`` hardcodes this for
    the audited path); a real GPU run launches under ``accelerate launch
    --num_processes 4`` to satisfy it honestly. This CPU smoke test instead
    bypasses ``require_exact_world_size``'s check that the *actual*
    accelerator has 4 ranks, mirroring test_aivc_cross_validate.py's
    ``test_run_training_guards_audited_entrypoint`` -- the codebase's own
    established pattern for exercising the audited path on one process.
    """
    monkeypatch.setattr(
        train_module,
        "require_exact_world_size",
        lambda *_args, **_kwargs: None,
    )


def test_phase_c_two_view_smoke_trains_and_losses_decrease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate: a handful of real training steps at 2560-in/2000-out.

    Runs ``train.run_training``'s audited entrypoint (the only entrypoint
    that accepts externally-assembled ``GeneBags``, matching how
    ``scripts/train_tx1_st_response.py`` wires Task 5's assembly into it) on
    CPU with the ``linear_mock`` backend, then asserts every reported
    per-epoch ``total_loss`` is finite and that the loss trajectory
    decreases.
    """
    max_epochs = 8
    config = _smoke_config(tmp_path, max_epochs=max_epochs)
    bags = _synthetic_two_view_bags()
    fold, train_data, val_data, sealed_test = _fold_bags(bags)
    _bypass_runtime_world_size_check(monkeypatch)

    paths = train_module.run_training(
        config,
        train_data=train_data,
        val_data=val_data,
        sealed_test=sealed_test,
        fold_spec=fold,
        run_dir_override=tmp_path / "run",
        source_fingerprint="phase_c_smoke",
        canonical_gene_order=tuple(str(gene) for gene in train_data.genes)
        + tuple(str(gene) for gene in val_data.genes)
        + fold.test_genes,
    )

    train_log = pd.read_csv(paths["train_log"])
    assert len(train_log) == max_epochs
    losses = train_log["train_total_loss"].to_numpy(dtype=np.float64)
    assert np.isfinite(losses).all(), f"non-finite total_loss values: {losses}"
    # A handful of AdamW steps on a real (non-trivial) regression target
    # should visibly reduce the loss; compare the mean of the second half
    # of epochs against the first half rather than a strict first-vs-last
    # comparison, tolerating step-to-step noise.
    first_half = losses[: max_epochs // 2].mean()
    second_half = losses[max_epochs // 2 :].mean()
    assert second_half < first_half, (
        f"expected total_loss to decrease: first-half mean {first_half}, "
        f"second-half mean {second_half}, full trajectory {losses.tolist()}"
    )

    # Also finite for the auxiliary components actually driving the
    # gradient (C9: hvg_*/latent_* stay gene/output-space in both arms).
    for column in (
        "train_hvg_mean_delta",
        "train_hvg_energy",
        "train_latent_mean_delta",
        "train_latent_energy",
    ):
        assert np.isfinite(train_log[column].to_numpy(dtype=np.float64)).all()

    # The frozen-checkpoint internal-outer-test evaluation at the end of
    # _run_audited_training also ran and produced metrics.
    fold_metrics = pd.read_csv(paths["fold_metrics"])
    assert not fold_metrics.empty

    # Regression pin: with an all-NaN `y` (no GeneEffect label), the "best"
    # checkpoint selection used to be permanently frozen at epoch 1 (val
    # c_loss is NaN every epoch, and _is_better_metric always rejects a
    # non-finite candidate). _checkpoint_selection_metric's fallback to
    # generation_loss should let a later epoch actually win, given the
    # clearly decreasing loss trajectory asserted above.
    fit_audit_summary = json.loads((paths["fit_audit_summary"]).read_text())
    assert fit_audit_summary["best_epoch"] > 1, (
        "best_epoch stuck at 1 -- the NaN-c_loss checkpoint-selection "
        "fallback to generation_loss is not working"
    )


def test_load_state_model_forwards_output_space_and_warm_start_from(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 7 wires StateConfig.output_space/warm_start_from into train.py.

    Before this task, ``load_state_model`` (Task 4) accepted these keyword
    arguments but neither of ``train.py``'s two call sites ever passed them
    -- the capability existed but nothing in the training pipeline could
    reach it from a config. This pins the wiring for the audited builder
    (``_build_e2e_model``); ``test_build_model_also_forwards_output_space_
    and_warm_start_from`` below covers the generic builder identically.
    """
    calls: list[dict[str, object]] = []

    def _fake_load_state_model(**kwargs: object) -> object:
        calls.append(kwargs)
        return model_module.LinearMockStateModel(
            int(kwargs["input_dim"]),  # type: ignore[arg-type]
            int(kwargs["output_dim"]),  # type: ignore[arg-type]
            int(kwargs["pert_dim"]),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(train_module, "load_state_model", _fake_load_state_model)

    bags = _synthetic_two_view_bags()
    _, train_data, _, _ = _fold_bags(bags)
    config = _smoke_config(tmp_path, max_epochs=1)
    warm_start_from = tmp_path / "does_not_need_to_exist.ckpt"
    config = replace(
        config,
        state=replace(
            config.state,
            output_space="gene",
            warm_start_from=warm_start_from,
        ),
    )

    train_module._build_e2e_model(
        config,
        train_data,
        extra_genes=(),
        canonical_gene_order=tuple(str(g) for g in train_data.genes),
    )

    assert calls, "load_state_model was never called"
    assert calls[0]["output_space"] == "gene"
    assert calls[0]["warm_start_from"] == warm_start_from


def test_build_model_also_forwards_output_space_and_warm_start_from(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generic (non-audited) builder gets the identical wiring fix."""
    calls: list[dict[str, object]] = []

    def _fake_load_state_model(**kwargs: object) -> object:
        calls.append(kwargs)
        return model_module.LinearMockStateModel(
            int(kwargs["input_dim"]),  # type: ignore[arg-type]
            int(kwargs["output_dim"]),  # type: ignore[arg-type]
            int(kwargs["pert_dim"]),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(train_module, "load_state_model", _fake_load_state_model)

    bags = _synthetic_two_view_bags()
    rng = np.random.default_rng(0)
    small_latent = rng.normal(size=(_N_CONTROL_CELLS, 8)).astype(np.float32)
    featureizer = model_module.fit_fixed_gmm(
        (),
        bags.effective_control_target,
        n_components=2,
        covariance_floor=1e-4,
        random_state=0,
        max_fit_cells=None,
    )
    projector_weight, projector_bias = fit_linear_projector(
        bags.effective_control_target, small_latent, alpha=1.0
    )
    config = _smoke_config(tmp_path, max_epochs=1)
    warm_start_from = tmp_path / "does_not_need_to_exist.ckpt"
    config = replace(
        config,
        state=replace(
            config.state, output_space="gene", warm_start_from=warm_start_from
        ),
    )

    train_module._build_model(
        config,
        bags,
        featureizer,
        projector_weight,
        projector_bias,
    )

    assert calls, "load_state_model was never called"
    assert calls[0]["output_space"] == "gene"
    assert calls[0]["warm_start_from"] == warm_start_from
