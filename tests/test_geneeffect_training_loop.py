from __future__ import annotations

import json
import hashlib
import math
from dataclasses import replace
import gc
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

import src.experiments.exp13_legacy.geneeffect_training_loop as loops
from src.data.batches import OnlineConditionBatch, FeatureBatch
from src.experiments.exp13_legacy.geneeffect_training import (
    JointStepMetrics,
    OnlineSupervisedBatch,
    PrecomputedSupervisedBatch,
    SupervisedMatrix,
    WarmupStepMetrics,
)


def _config(*, warmup_epochs=4, warmup_patience=2, joint_epochs=4, joint_patience=2):
    split_path = Path("configs/benchmarks/cell_line_geneeffect_226_split.json")
    return SimpleNamespace(
        warmup=SimpleNamespace(
            learning_rate=1e-3,
            max_epochs=warmup_epochs,
            patience=warmup_patience,
        ),
        joint=SimpleNamespace(
            state_learning_rate=1e-3,
            esm_adapter_learning_rate=1e-3,
            head_learning_rate=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            max_epochs=joint_epochs,
            patience=joint_patience,
        ),
        loss=SimpleNamespace(huber_delta=1.0, beta=1.0),
        selection=SimpleNamespace(
            metric="validation_macro_per_gene_spearman", direction="maximize"
        ),
        source_sha256="a" * 64,
        paths=SimpleNamespace(split=split_path),
    )


def _provenance() -> loops.CheckpointProvenance:
    split_path = Path("configs/benchmarks/cell_line_geneeffect_226_split.json")
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    return loops.CheckpointProvenance(
        distributed_runtime={
            "world_size": 4,
            "mixed_precision": "bf16",
            "conditions_per_rank": 256,
            "global_conditions_per_step": 1024,
            "rank_topology": [
                {
                    "rank": rank,
                    "local_rank": rank,
                    "device": f"cuda:{rank}",
                    "device_name": "NVIDIA H20",
                    "hostname": "hpc",
                }
                for rank in range(4)
            ],
        },
        warmup_runtime={
            "world_size": 4,
            "conditions_per_rank": 256,
            "global_conditions_per_step": 1024,
            "optimizer_steps_per_epoch": 3,
        },
        lambda_calibration_report={"lambda_dep": 0.5, "raw_ratios": [0.5]},
        split_sha256=hashlib.sha256(split_bytes).hexdigest(),
        gene_effect_sha256="e" * 64,
        mu_train_sha256="f" * 64,
        residual_target_sha256=_target_digest(),
        validation_target_sha256=_target_digest(),
        centering_fit_model_ids_sha256=_centering_digest(split),
        validation_model_ids=tuple(split["val"]),
        validation_gene_symbols=("G0", "G1"),
        validation_gene_count=2,
        validation_gene_universe_sha256=hashlib.sha256(b"G0\nG1").hexdigest(),
    )


def _centering_digest(split) -> str:
    unlabeled = set(split["unlabeled_train"])
    supervised = [model_id for model_id in split["train"] if model_id not in unlabeled]
    return hashlib.sha256("\n".join(supervised).encode()).hexdigest()


def _target_digest() -> str:
    target = torch.arange(27, dtype=torch.float32).repeat(2, 1)
    mask = torch.ones_like(target, dtype=torch.bool)
    return _digest_parts(("G0", "G1"), target, mask)


def _digest_parts(genes, target, mask) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(genes).encode())
    digest.update("\n".join(_validation_ids()).encode())
    digest.update(target.numpy().tobytes())
    digest.update(mask.numpy().tobytes())
    return digest.hexdigest()


def _validation_ids() -> tuple[str, ...]:
    split_path = Path("configs/benchmarks/cell_line_geneeffect_226_split.json")
    return tuple(json.loads(split_path.read_bytes())["val"])


def _supervision() -> SupervisedMatrix:
    target = torch.arange(27, dtype=torch.float32).repeat(2, 1)
    validation_ids = _validation_ids()
    return SupervisedMatrix(
        target=target,
        label_mask=torch.ones_like(target, dtype=torch.bool),
        g_var_mask=torch.ones(2, dtype=torch.bool),
        gene_symbols=("G0", "G1"),
        context_model_ids_by_gene=(validation_ids, validation_ids),
        residual_target_sha256=_target_digest(),
        centering_fit_model_ids_sha256=_provenance().centering_fit_model_ids_sha256,
    )


def _precomputed_validation_batch() -> PrecomputedSupervisedBatch:
    pairs = 54
    signal = torch.arange(27, dtype=torch.float32).repeat(2).unsqueeze(1)
    model_ids = _validation_ids() * 2
    gene_symbols = ("G0",) * 27 + ("G1",) * 27
    features = FeatureBatch(
        delta_proj=torch.zeros(pairs, 1),
        s=torch.zeros(pairs, 1),
        q_sc=torch.zeros(pairs, 1),
        e_g=signal,
        z_c=torch.zeros(pairs, 1),
        q_sc_mask=torch.ones(pairs, dtype=torch.bool),
        hvg_panel_mask=torch.ones(pairs, dtype=torch.bool),
        own_gene_shift_mask=torch.ones(pairs, dtype=torch.bool),
        gene_symbols=gene_symbols,
        model_ids=model_ids,
    )
    return PrecomputedSupervisedBatch(features, _supervision())


def _online_validation_batch() -> OnlineSupervisedBatch:
    pairs = 54
    signal = torch.arange(27, dtype=torch.float32).repeat(2).unsqueeze(1)
    conditions = OnlineConditionBatch(
        controls_tx1=tuple(torch.zeros(1, 1) for _ in range(pairs)),
        basal_hvg=tuple(torch.zeros(1, 1) for _ in range(pairs)),
        genes=("G0",) * 27 + ("G1",) * 27,
        model_ids=_validation_ids() * 2,
        q_sc=torch.zeros(pairs, 1),
        e_g=signal,
        z_c=torch.zeros(pairs, 1),
        q_sc_mask=torch.ones(pairs, dtype=torch.bool),
        gene_in_hvg_panel=torch.zeros(pairs, dtype=torch.bool),
        own_gene_hvg_indices=tuple(None for _ in range(pairs)),
        own_gene_shift_available=torch.zeros(pairs, dtype=torch.bool),
    )
    return OnlineSupervisedBatch(conditions, _supervision())


def _validation(*, online: bool = False):
    provenance = _provenance()

    def batch_factory():
        yield _online_validation_batch() if online else _precomputed_validation_batch()

    return loops.ResidualValidationMetric(
        batch_factory=batch_factory,
        batch_kind="online" if online else "precomputed",
        validation_model_ids=provenance.validation_model_ids,
        split_sha256=provenance.split_sha256,
        gene_effect_sha256=provenance.gene_effect_sha256,
        mu_train_sha256=provenance.mu_train_sha256,
    )


class _Perturbations(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(1, 1, bias=False)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_adapter = nn.Linear(1, 1, bias=False)
        self.perturbations = _Perturbations()


class _LoopModel(nn.Module):
    def __init__(self, *, frozen: bool) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.head = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.head.weight)
        self._backbone_frozen = frozen
        self.validation_nan = False
        if frozen:
            self.backbone.requires_grad_(False)

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def assert_frozen_backbone_clean(self) -> None:
        offenders = [
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad or parameter.grad is not None
        ]
        if offenders:
            raise RuntimeError("frozen backbone is dirty")

    def _validation_prediction(self, signal):
        if self.validation_nan:
            return torch.full_like(signal[:, 0], math.nan)
        selected_weights = (2.0, 6.0)
        direction = (
            1.0
            if any(
                abs(self.head.weight.item() - weight) < 1e-6
                for weight in selected_weights
            )
            else -1.0
        )
        return signal[:, 0] * direction

    def forward_precomputed(self, features):
        return self._validation_prediction(features.e_g)

    def forward(self, conditions):
        return SimpleNamespace(delta_hat=self._validation_prediction(conditions.e_g))


def _warmup_step(model, _batch, _optimizer, **_kwargs) -> WarmupStepMetrics:
    with torch.no_grad():
        model.head.weight.add_(1.0)
    return WarmupStepMetrics(1.0, 0.5, 0.5, 3, 1)


def _joint_step(model, _dependency, response, _optimizer, **_kwargs):
    with torch.no_grad():
        model.head.weight.add_(1.0)
    response.seen += 1
    return JointStepMetrics(2.0, 1.0, 1.0, 0.5, 0.5, 0.5, 3, 1)


class _Response:
    def __init__(self, *, batch_weight: float = 1.0) -> None:
        self.seen = 0
        self.batch_weight = batch_weight

    def validate(self) -> None:
        pass


class _WarmupBatch:
    def validate(self) -> None:
        pass


class _Dependency:
    def __init__(self, *, objective_weight: float = 1.0) -> None:
        self.objective_weight = objective_weight

    def validate(self) -> None:
        pass


def test_warmup_restores_earliest_best_and_stops_at_patience(tmp_path: Path) -> None:
    model = _LoopModel(frozen=True)
    outcome = loops.train_frozen_warmup(
        model,  # type: ignore[arg-type]
        lambda _epoch: (_WarmupBatch(),),  # type: ignore[return-value]
        _validation(),
        tmp_path / "warmup",
        _config(),  # type: ignore[arg-type]
        _provenance(),
        step_fn=_warmup_step,
    )
    assert outcome["best_epoch"] == 1
    assert outcome["stopped_epoch"] == 3
    assert outcome["stopped_early"] is True
    assert outcome["epochs"][0]["optimizer_steps"] == 1
    assert model.head.weight.item() == pytest.approx(2.0)
    saved = torch.load(tmp_path / "warmup/best/head.pt", weights_only=True)["weight"]
    assert torch.equal(model.head.weight, saved)


def test_warmup_refuses_nonfinite_metric(tmp_path: Path) -> None:
    model = _LoopModel(frozen=True)
    model.validation_nan = True
    with pytest.raises(ValueError, match="finite"):
        loops.train_frozen_warmup(
            model,  # type: ignore[arg-type]
            lambda _epoch: (_WarmupBatch(),),  # type: ignore[return-value]
            _validation(),
            tmp_path / "warmup",
            _config(),  # type: ignore[arg-type]
            _provenance(),
            step_fn=_warmup_step,
        )


def test_warmup_requires_frozen_backbone(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen backbone"):
        loops.train_frozen_warmup(
            _LoopModel(frozen=False),  # type: ignore[arg-type]
            lambda _epoch: (),
            _validation(),
            tmp_path / "warmup",
            _config(),  # type: ignore[arg-type]
            _provenance(),
        )


def test_loop_refuses_existing_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "warmup"
    output.mkdir()
    (output / "stale.txt").write_text("stale")
    with pytest.raises(FileExistsError, match="not fresh"):
        loops.train_frozen_warmup(
            _LoopModel(frozen=True),  # type: ignore[arg-type]
            lambda _epoch: (),
            _validation(),
            output,
            _config(),  # type: ignore[arg-type]
            _provenance(),
        )


def test_loop_accepts_precreated_empty_layout_directory(tmp_path: Path) -> None:
    output = tmp_path / "warmup"
    output.mkdir()
    loops.train_frozen_warmup(
        _LoopModel(frozen=True),  # type: ignore[arg-type]
        lambda _epoch: (_WarmupBatch(),),  # type: ignore[return-value]
        _validation(),
        output,
        _config(warmup_epochs=1),  # type: ignore[arg-type]
        _provenance(),
        step_fn=_warmup_step,
    )
    assert (output / "train_log.csv").is_file()


def test_joint_cycles_response_batches_and_restores_best(tmp_path: Path) -> None:
    model = _LoopModel(frozen=False)
    response_factory_calls = 0
    generated_responses: list[_Response] = []

    def response_factory(_epoch):
        nonlocal response_factory_calls
        response_factory_calls += 1
        for _ in range(2):
            response = _Response()
            generated_responses.append(response)
            yield response

    outcome = loops.train_joint(
        model,  # type: ignore[arg-type]
        lambda _epoch: (_Dependency(), _Dependency(), _Dependency()),
        response_factory,  # type: ignore[arg-type]
        _validation(online=True),
        tmp_path / "joint",
        _config(joint_epochs=4),  # type: ignore[arg-type]
        _provenance(),
        response_loss_fn=object(),  # type: ignore[arg-type]
        lambda_dep=0.5,
        step_fn=_joint_step,
    )
    assert outcome["best_epoch"] == 1
    assert outcome["stopped_epoch"] == 3
    assert outcome["stopped_early"] is True
    assert response_factory_calls == 8
    assert len(generated_responses) == 12
    assert sum(response.seen for response in generated_responses) == 12
    saved = torch.load(tmp_path / "joint/best/e2e_state.pt", weights_only=True)[
        "head.weight"
    ]
    assert torch.equal(model.head.weight, saved)


class _FakeAccelerator:
    num_processes = 2
    is_main_process = True
    device = torch.device("cpu")

    def __init__(self, gathered) -> None:
        self.gathered = list(gathered) if isinstance(gathered, tuple) else gathered

    def gather(self, _value):
        if isinstance(self.gathered, list):
            return self.gathered.pop(0)
        return self.gathered

    def unwrap_model(self, value):
        return getattr(value, "module", value)


class _PreparedWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _GatherTrackingAccelerator:
    num_processes = 2
    is_main_process = True
    device = torch.device("cpu")

    def __init__(self, responses: tuple[torch.Tensor, ...]) -> None:
        self.responses = list(responses)
        self.gathered_locals: list[torch.Tensor] = []

    def gather(self, value):
        self.gathered_locals.append(value.detach().cpu().clone())
        if not self.responses:
            raise AssertionError("unexpected post-action gather")
        return self.responses.pop(0)

    def prepare(self, value):
        return value

    def unwrap_model(self, value):
        return getattr(value, "module", value)

    def wait_for_everyone(self) -> None:
        pass


class _ProgressAccelerator:
    num_processes = 2
    is_main_process = True
    device = torch.device("cpu")


def test_progress_is_atomic_global_and_excludes_padding(
    tmp_path: Path, monkeypatch
) -> None:
    ticks = iter((10.0, 12.0, 14.0))
    writer = loops.TrainingProgressWriter(
        tmp_path / ".warmup_progress.json",
        {
            "world_size": 2,
            "conditions_per_rank": 256,
            "global_conditions_per_step": 512,
            "optimizer_steps_per_epoch": 3,
        },
        min_interval_seconds=60.0,
        monotonic=lambda: next(ticks),
        utcnow=lambda: datetime(2026, 8, 30, tzinfo=timezone.utc),
    )
    accelerator = _ProgressAccelerator()
    synchronized_actions: list[str] = []

    def synchronized(_accelerator, label, action):
        synchronized_actions.append(label)
        action()

    monkeypatch.setattr(loops, "run_rank_zero_or_raise", synchronized)
    writer.update(
        phase="training",
        epoch=0,
        step=1,
        global_real_pairs_increment=10,
        accelerator=accelerator,  # type: ignore[arg-type]
        force=True,
    )
    writer.update(
        phase="validation",
        epoch=0,
        step=2,
        global_real_pairs_increment=0,
        accelerator=accelerator,  # type: ignore[arg-type]
        force=True,
    )
    payload = json.loads((tmp_path / ".warmup_progress.json").read_text())
    assert payload == {
        "schema_version": 1,
        "phase": "validation",
        "epoch": 0,
        "step": 2,
        "global_real_pairs": 10,
        "elapsed_seconds": 4.0,
        "global_real_pairs_per_second": 2.5,
        "heartbeat_utc": "2026-08-30T00:00:00+00:00",
        "world_size": 2,
        "conditions_per_rank": 256,
        "global_conditions_per_step": 512,
        "optimizer_steps_per_epoch": 3,
    }
    assert not list(tmp_path.glob("*.tmp"))
    assert synchronized_actions == ["write training progress"] * 2


def test_progress_sync_is_not_a_per_step_collective(
    tmp_path: Path, monkeypatch
) -> None:
    ticks = iter((10.0, 12.0))
    writer = loops.TrainingProgressWriter(
        tmp_path / ".warmup_progress.json",
        {
            "world_size": 2,
            "conditions_per_rank": 256,
            "global_conditions_per_step": 512,
            "optimizer_steps_per_epoch": 2,
        },
        min_interval_seconds=0.0,
        sync_interval_steps=2,
        monotonic=lambda: next(ticks),
        utcnow=lambda: datetime(2026, 8, 30, tzinfo=timezone.utc),
    )
    accelerator = _ProgressAccelerator()
    synchronized_actions: list[str] = []

    def synchronized(_accelerator, label, action):
        synchronized_actions.append(label)
        action()

    monkeypatch.setattr(loops, "run_rank_zero_or_raise", synchronized)
    writer.update(
        phase="training",
        epoch=0,
        step=1,
        global_real_pairs_increment=10,
        accelerator=accelerator,  # type: ignore[arg-type]
    )
    assert synchronized_actions == []
    writer.update(
        phase="training",
        epoch=0,
        step=2,
        global_real_pairs_increment=12,
        accelerator=accelerator,  # type: ignore[arg-type]
    )

    assert synchronized_actions == ["write training progress"]
    payload = json.loads((tmp_path / ".warmup_progress.json").read_text())
    assert payload["global_real_pairs"] == 22


def test_rank_validation_metric_uses_rank_zero_canonical_value() -> None:
    accelerator = _FakeAccelerator(torch.tensor([0.1, 0.2], dtype=torch.float64))
    assert loops._metric_across_ranks(0.1, accelerator) == 0.1  # type: ignore[arg-type]


def test_validation_evaluator_recomputes_targets_and_exact_gene_coverage() -> None:
    provenance = _provenance()
    batch = _precomputed_validation_batch()
    tampered = replace(
        batch,
        supervision=replace(batch.supervision, target=batch.supervision.target + 100.0),
    )
    evaluator = loops.ResidualValidationMetric(
        batch_factory=lambda: (tampered,),
        batch_kind="precomputed",
        validation_model_ids=provenance.validation_model_ids,
        split_sha256=provenance.split_sha256,
        gene_effect_sha256=provenance.gene_effect_sha256,
        mu_train_sha256=provenance.mu_train_sha256,
    )
    with pytest.raises(ValueError, match="does not match values"):
        evaluator.evaluate(_LoopModel(frozen=True), provenance)
    duplicate = loops.ResidualValidationMetric(
        batch_factory=lambda: (batch, batch),
        batch_kind="precomputed",
        validation_model_ids=provenance.validation_model_ids,
        split_sha256=provenance.split_sha256,
        gene_effect_sha256=provenance.gene_effect_sha256,
        mu_train_sha256=provenance.mu_train_sha256,
    )
    with pytest.raises(ValueError, match="duplicated"):
        duplicate.evaluate(_LoopModel(frozen=True), provenance)

    features = replace(
        batch.features,
        delta_proj=batch.features.delta_proj[:27],
        s=batch.features.s[:27],
        q_sc=batch.features.q_sc[:27],
        e_g=batch.features.e_g[:27],
        z_c=batch.features.z_c[:27],
        q_sc_mask=batch.features.q_sc_mask[:27],
        hvg_panel_mask=batch.features.hvg_panel_mask[:27],
        own_gene_shift_mask=batch.features.own_gene_shift_mask[:27],
        gene_symbols=batch.features.gene_symbols[:27],
        model_ids=batch.features.model_ids[:27],
    )
    target = batch.supervision.target[:1]
    mask = batch.supervision.label_mask[:1]
    one_digest = _digest_parts(("G0",), target, mask)
    supervision = replace(
        batch.supervision,
        target=target,
        label_mask=mask,
        g_var_mask=batch.supervision.g_var_mask[:1],
        gene_symbols=("G0",),
        context_model_ids_by_gene=(provenance.validation_model_ids,),
        residual_target_sha256=one_digest,
    )
    evaluator = loops.ResidualValidationMetric(
        batch_factory=lambda: (PrecomputedSupervisedBatch(features, supervision),),
        batch_kind="precomputed",
        validation_model_ids=provenance.validation_model_ids,
        split_sha256=provenance.split_sha256,
        gene_effect_sha256=provenance.gene_effect_sha256,
        mu_train_sha256=provenance.mu_train_sha256,
    )
    with pytest.raises(ValueError, match="exactly cover"):
        evaluator.evaluate(
            _LoopModel(frozen=True),
            replace(
                provenance,
                residual_target_sha256=one_digest,
                validation_target_sha256=one_digest,
            ),
        )
    wrong_ids = loops.ResidualValidationMetric(
        batch_factory=lambda: (_precomputed_validation_batch(),),
        batch_kind="precomputed",
        validation_model_ids=tuple(f"TEST-{index}" for index in range(27)),
        split_sha256=provenance.split_sha256,
        gene_effect_sha256=provenance.gene_effect_sha256,
        mu_train_sha256=provenance.mu_train_sha256,
    )
    with pytest.raises(ValueError, match="ModelIDs"):
        wrong_ids.evaluate(_LoopModel(frozen=True), provenance)


def test_validation_batches_are_generated_lazily_for_each_evaluation() -> None:
    provenance = _provenance()
    factory_calls = 0

    def batch_factory():
        nonlocal factory_calls
        factory_calls += 1
        yield _precomputed_validation_batch()

    evaluator = loops.ResidualValidationMetric(
        batch_factory=batch_factory,
        batch_kind="precomputed",
        validation_model_ids=provenance.validation_model_ids,
        split_sha256=provenance.split_sha256,
        gene_effect_sha256=provenance.gene_effect_sha256,
        mu_train_sha256=provenance.mu_train_sha256,
    )
    assert factory_calls == 0
    model = _LoopModel(frozen=True)
    with torch.no_grad():
        model.head.weight.fill_(2.0)
    assert evaluator.evaluate(model, provenance) == pytest.approx(1.0)
    assert evaluator.evaluate(model, provenance) == pytest.approx(1.0)
    assert factory_calls == 2
    assert not hasattr(evaluator, "batches")


def test_online_validation_factory_does_not_retain_generated_batches() -> None:
    provenance = _provenance()
    generated: list[weakref.ReferenceType[OnlineSupervisedBatch]] = []

    def batch_factory():
        batch = _online_validation_batch()
        generated.append(weakref.ref(batch))
        yield batch

    evaluator = loops.ResidualValidationMetric(
        batch_factory=batch_factory,
        batch_kind="online",
        validation_model_ids=provenance.validation_model_ids,
        split_sha256=provenance.split_sha256,
        gene_effect_sha256=provenance.gene_effect_sha256,
        mu_train_sha256=provenance.mu_train_sha256,
    )
    model = _LoopModel(frozen=False)
    with torch.no_grad():
        model.head.weight.fill_(2.0)
    assert evaluator.evaluate(model, provenance) == pytest.approx(1.0)
    gc.collect()
    assert len(generated) == 1
    assert generated[0]() is None
    assert evaluator.evaluate(model, provenance) == pytest.approx(1.0)
    gc.collect()
    assert len(generated) == 2
    assert all(reference() is None for reference in generated)


def test_rank_validation_exception_enters_shared_collective() -> None:
    accelerator = _FakeAccelerator(
        torch.tensor([float("nan"), 0.2], dtype=torch.float64)
    )
    metric = SimpleNamespace(
        evaluate=lambda _model, _provenance: (_ for _ in ()).throw(
            RuntimeError("rank-local")
        )
    )
    with pytest.raises(ValueError, match="every rank"):
        loops._validation_value(
            metric,
            nn.Linear(1, 1),
            accelerator,  # type: ignore[arg-type]
            _provenance(),
        )


def test_rank_step_counts_must_be_equal(tmp_path: Path, monkeypatch) -> None:
    accelerator = _FakeAccelerator(
        (
            torch.tensor([1, 1], dtype=torch.int64),
            torch.tensor([1, 0], dtype=torch.int64),
        )
    )
    accelerator.prepare = lambda value: value
    accelerator.wait_for_everyone = lambda: None
    monkeypatch.setattr(
        loops,
        "run_rank_zero_or_raise",
        lambda _accelerator, _label, action: action(),
    )
    model = _LoopModel(frozen=False)
    with pytest.raises(RuntimeError, match="step counts differ"):
        loops.train_joint(
            model,  # type: ignore[arg-type]
            lambda _epoch: (_Dependency(), _Dependency()),
            lambda _epoch: (_Response(),),  # type: ignore[return-value]
            _validation(online=True),
            tmp_path / "joint",
            _config(joint_epochs=1),  # type: ignore[arg-type]
            _provenance(),
            response_loss_fn=object(),  # type: ignore[arg-type]
            lambda_dep=0.5,
            accelerator=accelerator,  # type: ignore[arg-type]
            forward_model=_PreparedWrapper(model),
            step_fn=_joint_step,
        )


def test_warmup_step_failure_stops_before_progress_or_next_readiness(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    accelerator = _GatherTrackingAccelerator((torch.tensor([1, 1], dtype=torch.int64),))
    monkeypatch.setattr(
        loops,
        "run_rank_zero_or_raise",
        lambda _accelerator, _label, action: action(),
    )

    def batch_factory(_epoch):
        events.append("batch-1")
        yield _WarmupBatch()
        events.append("batch-2")
        yield _WarmupBatch()

    def failing_step(*_args, **_kwargs):
        events.append("step")
        raise ValueError("rank-local warmup failure")

    model = _LoopModel(frozen=True)
    with pytest.raises(ValueError, match="rank-local warmup failure"):
        loops.train_frozen_warmup(
            model,  # type: ignore[arg-type]
            batch_factory,  # type: ignore[arg-type]
            _validation(),
            tmp_path / "warmup",
            _config(warmup_epochs=1),  # type: ignore[arg-type]
            _provenance(),
            accelerator=accelerator,  # type: ignore[arg-type]
            forward_head=_PreparedWrapper(model.head),
            progress=SimpleNamespace(
                update=lambda **_kwargs: events.append("progress")
            ),
            step_fn=failing_step,
        )

    assert events == ["batch-1", "step"]
    assert [value.tolist() for value in accelerator.gathered_locals] == [[1]]
    assert accelerator.responses == []


def test_joint_step_failure_stops_before_progress_or_next_readiness(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    accelerator = _GatherTrackingAccelerator((torch.tensor([1, 1], dtype=torch.int64),))
    monkeypatch.setattr(
        loops,
        "run_rank_zero_or_raise",
        lambda _accelerator, _label, action: action(),
    )

    def dependency_factory(_epoch):
        events.append("dependency-1")
        yield _Dependency()
        events.append("dependency-2")
        yield _Dependency()

    def response_factory(_epoch):
        events.append("response-1")
        yield _Response()
        events.append("response-2")
        yield _Response()

    def failing_step(*_args, **_kwargs):
        events.append("step")
        raise ValueError("rank-local joint failure")

    model = _LoopModel(frozen=False)
    with pytest.raises(ValueError, match="rank-local joint failure"):
        loops.train_joint(
            model,  # type: ignore[arg-type]
            dependency_factory,  # type: ignore[arg-type]
            response_factory,  # type: ignore[arg-type]
            _validation(online=True),
            tmp_path / "joint",
            _config(joint_epochs=1),  # type: ignore[arg-type]
            _provenance(),
            response_loss_fn=object(),  # type: ignore[arg-type]
            lambda_dep=0.5,
            accelerator=accelerator,  # type: ignore[arg-type]
            forward_model=_PreparedWrapper(model),
            progress=SimpleNamespace(
                update=lambda **_kwargs: events.append("progress")
            ),
            step_fn=failing_step,
        )

    assert events == ["dependency-1", "response-1", "step"]
    assert [value.tolist() for value in accelerator.gathered_locals] == [[1]]
    assert accelerator.responses == []


def test_successful_steps_do_not_add_post_action_gathers(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        loops,
        "run_rank_zero_or_raise",
        lambda _accelerator, _label, action: action(),
    )
    monkeypatch.setattr(loops, "_validation_value", lambda *_args: 1.0)
    monkeypatch.setattr(loops, "_write_history", lambda *_args: None)
    monkeypatch.setattr(loops, "_save_best", lambda *_args: None)
    monkeypatch.setattr(loops, "_restore_best", lambda *_args: None)

    warmup_events: list[str] = []
    warmup_accelerator = _GatherTrackingAccelerator(
        (
            torch.tensor([1, 1], dtype=torch.int64),
            torch.tensor([0, 0], dtype=torch.int64),
        )
    )

    def warmup_factory(_epoch):
        warmup_events.append("batch")
        yield _WarmupBatch()
        warmup_events.append("factory-end")

    def successful_warmup_step(*args, **kwargs):
        warmup_events.append("step")
        return _warmup_step(*args, **kwargs)

    warmup_model = _LoopModel(frozen=True)
    loops.train_frozen_warmup(
        warmup_model,  # type: ignore[arg-type]
        warmup_factory,  # type: ignore[arg-type]
        _validation(),
        tmp_path / "warmup-success",
        _config(warmup_epochs=1),  # type: ignore[arg-type]
        _provenance(),
        accelerator=warmup_accelerator,  # type: ignore[arg-type]
        forward_head=_PreparedWrapper(warmup_model.head),
        step_fn=successful_warmup_step,
    )
    assert warmup_events == ["batch", "step", "factory-end"]
    assert [value.tolist() for value in warmup_accelerator.gathered_locals] == [
        [1],
        [0],
    ]
    assert warmup_accelerator.responses == []

    joint_events: list[str] = []
    joint_accelerator = _GatherTrackingAccelerator(
        (
            torch.tensor([1, 1], dtype=torch.int64),
            torch.tensor([0, 0], dtype=torch.int64),
        )
    )

    def dependency_factory(_epoch):
        joint_events.append("dependency")
        yield _Dependency()
        joint_events.append("factory-end")

    def response_factory(_epoch):
        joint_events.append("response")
        yield _Response()

    def successful_joint_step(*args, **kwargs):
        joint_events.append("step")
        return _joint_step(*args, **kwargs)

    joint_model = _LoopModel(frozen=False)
    loops.train_joint(
        joint_model,  # type: ignore[arg-type]
        dependency_factory,  # type: ignore[arg-type]
        response_factory,  # type: ignore[arg-type]
        _validation(online=True),
        tmp_path / "joint-success",
        _config(joint_epochs=1),  # type: ignore[arg-type]
        _provenance(),
        response_loss_fn=object(),  # type: ignore[arg-type]
        lambda_dep=0.5,
        accelerator=joint_accelerator,  # type: ignore[arg-type]
        forward_model=_PreparedWrapper(joint_model),
        step_fn=successful_joint_step,
    )
    assert joint_events == ["dependency", "response", "step", "factory-end"]
    assert [value.tolist() for value in joint_accelerator.gathered_locals] == [
        [1],
        [0],
    ]
    assert joint_accelerator.responses == []


def test_rank_batch_validation_failure_enters_preflight_collective() -> None:
    class _Invalid:
        def validate(self):
            raise ValueError("bad shard")

    accelerator = _FakeAccelerator(torch.tensor([2, 1], dtype=torch.int64))
    with pytest.raises(RuntimeError, match="factory failed"):
        tuple(
            loops._stream_joint_batch_pairs(
                lambda _epoch: (_Invalid(),),  # type: ignore[return-value]
                lambda _epoch: (_Response(),),  # type: ignore[return-value]
                0,
                accelerator,  # type: ignore[arg-type]
            )
        )


def test_joint_batches_stream_without_epoch_retention() -> None:
    dependency_refs: list[weakref.ReferenceType[_Dependency]] = []
    response_refs: list[weakref.ReferenceType[_Response]] = []
    max_live_dependencies = 0
    max_live_responses = 0
    response_factory_calls = 0

    def dependency_factory(_epoch):
        nonlocal max_live_dependencies
        for _ in range(6):
            batch = _Dependency()
            dependency_refs.append(weakref.ref(batch))
            max_live_dependencies = max(
                max_live_dependencies,
                sum(reference() is not None for reference in dependency_refs),
            )
            yield batch

    def response_factory(_epoch):
        nonlocal max_live_responses, response_factory_calls
        response_factory_calls += 1
        for _ in range(2):
            batch = _Response()
            response_refs.append(weakref.ref(batch))
            max_live_responses = max(
                max_live_responses,
                sum(reference() is not None for reference in response_refs),
            )
            yield batch

    for dependency_batch, response_batch in loops._stream_joint_batch_pairs(
        dependency_factory,  # type: ignore[arg-type]
        response_factory,  # type: ignore[arg-type]
        0,
        None,
    ):
        assert dependency_batch is not None
        assert response_batch is not None
    del dependency_batch, response_batch
    gc.collect()
    assert max_live_dependencies <= 2
    assert max_live_responses <= 2
    assert response_factory_calls == 3
    assert all(reference() is None for reference in dependency_refs)
    assert all(reference() is None for reference in response_refs)


def test_joint_stream_preserves_zero_weight_padding_identity() -> None:
    dependency_padding = _Dependency(objective_weight=0.0)
    response_padding = _Response(batch_weight=0.0)
    pairs = list(
        loops._stream_joint_batch_pairs(
            lambda _epoch: (dependency_padding,),  # type: ignore[arg-type]
            lambda _epoch: (response_padding,),  # type: ignore[arg-type]
            0,
            None,
        )
    )
    assert len(pairs) == 1
    assert pairs[0][0] is dependency_padding
    assert pairs[0][0].objective_weight == 0.0
    assert pairs[0][1] is response_padding
    assert pairs[0][1].batch_weight == 0.0


def test_centering_provenance_is_recomputed_from_split() -> None:
    provenance = _provenance()
    with pytest.raises(ValueError, match="centering-fit"):
        loops._validate_contract(
            _config(),
            replace(provenance, centering_fit_model_ids_sha256="2" * 64),
            _validation(),
        )


def test_validation_contract_uses_official_tuple_not_fixed_count(
    tmp_path: Path,
) -> None:
    split = {"train": ["T0"], "unlabeled_train": [], "val": ["V0", "V1"]}
    split_path = tmp_path / "split.json"
    split_bytes = json.dumps(split).encode()
    split_path.write_bytes(split_bytes)
    config = _config()
    config.paths.split = split_path
    provenance = replace(
        _provenance(),
        split_sha256=hashlib.sha256(split_bytes).hexdigest(),
        centering_fit_model_ids_sha256=hashlib.sha256(b"T0").hexdigest(),
        validation_model_ids=("V0", "V1"),
    )
    metric = loops.ResidualValidationMetric(
        batch_factory=lambda: (),
        batch_kind="precomputed",
        validation_model_ids=("V0", "V1"),
        split_sha256=provenance.split_sha256,
        gene_effect_sha256=provenance.gene_effect_sha256,
        mu_train_sha256=provenance.mu_train_sha256,
    )

    loops._validate_contract(config, provenance, metric)
    with pytest.raises(ValueError, match="configured split"):
        loops._validate_contract(
            config,
            replace(provenance, validation_model_ids=("V1", "V0")),
            metric,
        )


def test_checkpoint_and_metadata_are_atomic(tmp_path: Path) -> None:
    model = _LoopModel(frozen=True)
    loops.train_frozen_warmup(
        model,  # type: ignore[arg-type]
        lambda _epoch: (_WarmupBatch(),),  # type: ignore[return-value]
        _validation(),
        tmp_path / "warmup",
        _config(warmup_epochs=1),  # type: ignore[arg-type]
        _provenance(),
        step_fn=_warmup_step,
    )
    best = tmp_path / "warmup/best"
    metadata = json.loads((best / "metadata.json").read_text())
    assert metadata["selection_direction"] == "maximize"
    assert metadata["provenance"]["lambda_calibration_report"]["lambda_dep"] == 0.5
    assert (best / "head.pt").is_file()
    assert best.is_symlink()
    assert not list((tmp_path / "warmup").rglob("*.tmp"))
