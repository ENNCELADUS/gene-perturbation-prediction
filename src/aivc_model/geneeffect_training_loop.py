"""Epoch orchestration and checkpoint selection for Exp13 Stage 2."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import csv
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from accelerate import Accelerator
import torch
from torch import nn

from aivc_model.distributed import run_rank_zero_or_raise
from aivc_model.geneeffect_e2e import GeneEffectE2EModel
from aivc_model.geneeffect_head import macro_per_gene_spearman
from aivc_model.geneeffect_training import (
    JointStepMetrics,
    OnlineSupervisedBatch,
    PrecomputedSupervisedBatch,
    ResponseSupervisionBatch,
    WarmupStepMetrics,
    build_joint_optimizer,
    build_warmup_optimizer,
    joint_step,
    warmup_step,
)
from aivc_model.response_training import ResponseLoss
from aivc_model.stage2_config import Stage2Config


SELECTION_NAME = "validation_macro_per_gene_spearman"
SELECTION_DIRECTION = "maximize"


@dataclass(frozen=True)
class CheckpointProvenance:
    """Inputs whose identities must travel with every selected checkpoint."""

    config_sha256: str
    stage2_code_sha256: Mapping[str, str]
    distributed_runtime: Mapping[str, object]
    lambda_calibration_report: Mapping[str, object] | None
    feature_sha256: Mapping[str, str]
    checkpoint_sha256: Mapping[str, str]
    split_sha256: str
    gene_effect_sha256: str
    mu_train_sha256: str
    residual_target_sha256: str
    validation_target_sha256: str
    centering_fit_model_ids_sha256: str
    validation_model_ids: tuple[str, ...]
    validation_gene_symbols: tuple[str, ...]
    validation_gene_count: int
    validation_gene_universe_sha256: str

    def json_ready(self) -> dict[str, object]:
        payload = asdict(self)
        json.dumps(payload, allow_nan=False)
        if (
            not self.feature_sha256
            or not self.checkpoint_sha256
            or not self.stage2_code_sha256
        ):
            raise ValueError(
                "feature, checkpoint, and Stage 2 code provenance hashes are required"
            )
        if not self.distributed_runtime:
            raise ValueError("distributed runtime provenance is required")
        if not self.validation_gene_symbols or len(
            set(self.validation_gene_symbols)
        ) != len(self.validation_gene_symbols):
            raise ValueError("validation gene universe must be non-empty and unique")
        gene_digest = hashlib.sha256(
            "\n".join(self.validation_gene_symbols).encode()
        ).hexdigest()
        if self.validation_gene_count != len(self.validation_gene_symbols):
            raise ValueError("validation gene count does not match gene universe")
        if self.validation_gene_universe_sha256 != gene_digest:
            raise ValueError("validation gene-universe SHA does not match symbols")
        return payload


WarmupBatchFactory = Callable[[int], Iterable[PrecomputedSupervisedBatch]]
JointBatchFactory = Callable[[int], Iterable[OnlineSupervisedBatch]]
ResponseBatchFactory = Callable[[int], Iterable[ResponseSupervisionBatch]]
WarmupStep = Callable[..., WarmupStepMetrics]
JointStep = Callable[..., JointStepMetrics]


@dataclass(frozen=True)
class ResidualValidationMetric:
    """Lazy validation-only residual evaluator used for selection."""

    batch_factory: Callable[
        [], Iterable[PrecomputedSupervisedBatch | OnlineSupervisedBatch]
    ]
    batch_kind: Literal["precomputed", "online"]
    validation_model_ids: tuple[str, ...]
    split_sha256: str
    gene_effect_sha256: str
    mu_train_sha256: str

    def __post_init__(self) -> None:
        if self.batch_kind not in ("precomputed", "online"):
            raise ValueError("validation batch_kind must be precomputed or online")

    def evaluate(self, model: nn.Module, provenance: CheckpointProvenance) -> float:
        values: list[float] = []
        actual_genes: list[str] = []
        mask_bytes: list[bytes] = []
        declared_target_sha256: set[str] = set()
        centering_sha256: set[str] = set()
        digest = hashlib.sha256()
        digest.update("\n".join(provenance.validation_gene_symbols).encode())
        digest.update("\n".join(self.validation_model_ids).encode())
        batch_count = 0
        for batch in self.batch_factory():
            batch_count += 1
            batch.validate()
            expected_type = (
                PrecomputedSupervisedBatch
                if self.uses_precomputed_batches
                else OnlineSupervisedBatch
            )
            if not isinstance(batch, expected_type):
                raise TypeError(
                    f"{self.batch_kind} validation factory returned "
                    f"{type(batch).__name__}"
                )
            genes, contexts = batch.supervision.shape
            if contexts != len(self.validation_model_ids):
                raise ValueError(
                    "validation supervision must span the official val lines"
                )
            if batch.objective_weight != 1.0:
                raise ValueError("validation batches cannot be DDP padding")
            source = (
                batch.features
                if isinstance(batch, PrecomputedSupervisedBatch)
                else batch.conditions
            )
            expected_models = self.validation_model_ids * genes
            if tuple(source.model_ids) != expected_models:
                raise ValueError(
                    "validation batch ModelIDs do not match authoritative val order"
                )
            actual_genes.extend(batch.supervision.gene_symbols)
            declared_target_sha256.add(batch.supervision.residual_target_sha256)
            centering_sha256.add(
                batch.supervision.centering_fit_model_ids_sha256
            )
            digest.update(
                batch.supervision.target.detach()
                .cpu()
                .contiguous()
                .numpy()
                .tobytes()
            )
            mask_bytes.append(
                batch.supervision.label_mask.detach()
                .cpu()
                .contiguous()
                .numpy()
                .tobytes()
            )
            if isinstance(batch, PrecomputedSupervisedBatch):
                target_model = getattr(model, "module", model)
                prediction = target_model.forward_precomputed(batch.features)
            else:
                # Validation has no backward pass.  Use the unwrapped module so
                # rank-local lazy iterator failures cannot strand peers inside
                # DDP forward collectives; the scalar metric is synchronized
                # after every rank finishes or records a local failure.
                target_model = getattr(model, "module", model)
                prediction = target_model(batch.conditions).delta_hat
            genes, contexts = batch.supervision.shape
            prediction = prediction.reshape(genes, contexts)
            target = batch.supervision.target.masked_fill(
                ~batch.supervision.label_mask, math.nan
            )
            score = macro_per_gene_spearman(prediction, target)
            values.extend(float(value) for value in score.per_gene.dropna().tolist())
        if batch_count == 0:
            raise ValueError("residual validation factory produced no batches")
        if len(set(actual_genes)) != len(actual_genes):
            raise ValueError("validation genes cannot be duplicated across batches")
        if tuple(actual_genes) != provenance.validation_gene_symbols:
            raise ValueError(
                "validation batches do not exactly cover the gene universe"
            )
        if declared_target_sha256 != {provenance.residual_target_sha256}:
            raise ValueError("global residual-target SHA does not match provenance")
        if centering_sha256 != {provenance.centering_fit_model_ids_sha256}:
            raise ValueError("validation centering-fit SHA does not match provenance")
        for value in mask_bytes:
            digest.update(value)
        if digest.hexdigest() != provenance.validation_target_sha256:
            raise ValueError("validation residual-target SHA does not match values")
        if not values:
            return math.nan
        return sum(values) / len(values)

    @property
    def uses_precomputed_batches(self) -> bool:
        return self.batch_kind == "precomputed"


def _prepare_fresh_dir(path: Path, accelerator: Accelerator | None) -> None:
    path = Path(path)

    def create() -> None:
        if path.exists():
            if not path.is_dir() or any(path.iterdir()):
                raise FileExistsError(
                    f"training output directory is not fresh and empty: {path}"
                )
            return
        path.mkdir(parents=True)

    _rank_zero_action(accelerator, "create fresh training directory", create)
    if accelerator is not None:
        accelerator.wait_for_everyone()


def _rank_zero_action(
    accelerator: Accelerator | None, label: str, action: Callable[[], object]
) -> None:
    if accelerator is None or accelerator.num_processes == 1:
        action()
    else:
        run_rank_zero_or_raise(accelerator, label, action)


def _atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_csv(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        raise ValueError("cannot write an empty training history")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def _state_dict(model: nn.Module, accelerator: Accelerator | None) -> dict[str, Any]:
    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
    return {
        name: value.detach().cpu()
        for name, value in unwrapped.state_dict().items()
    }


def _atomic_save_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(state), temporary)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_across_ranks(value: float, accelerator: Accelerator | None) -> float:
    value = float(value)
    if accelerator is None or accelerator.num_processes == 1:
        if not math.isfinite(value):
            raise ValueError(f"validation metric must be finite, got {value}")
        return value
    local = torch.tensor([value], dtype=torch.float64, device=accelerator.device)
    gathered = accelerator.gather(local).detach().cpu().reshape(-1)
    values = tuple(float(item) for item in gathered.tolist())
    if len(values) != accelerator.num_processes:
        raise RuntimeError(
            "validation metric gather did not return one value per rank: "
            f"{values}"
        )
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"validation metric must be finite on every rank: {values}")
    if any(item != values[0] for item in values[1:]):
        raise RuntimeError(f"validation metric differs across ranks: {values}")
    return values[0]


def _mean_step_metrics(metrics: list[object]) -> dict[str, float]:
    if not metrics:
        raise ValueError("an epoch must contain at least one optimization step")
    rows = [asdict(item) if is_dataclass(item) else vars(item) for item in metrics]
    means = {
        str(name): sum(float(row[name]) for row in rows) / len(rows)
        for name in rows[0]
    }
    nonfinite = [name for name, value in means.items() if not math.isfinite(value)]
    if nonfinite:
        raise ValueError(f"training step metrics must be finite: {nonfinite}")
    return means


def _metadata(
    *,
    kind: str,
    epoch: int,
    metric: float,
    provenance: CheckpointProvenance,
) -> dict[str, object]:
    return {
        "checkpoint_kind": kind,
        "epoch": epoch,
        "metric_value": metric,
        "selection_name": SELECTION_NAME,
        "selection_direction": SELECTION_DIRECTION,
        "provenance": provenance.json_ready(),
    }


def _save_best(
    model: nn.Module,
    output_dir: Path,
    filename: str,
    metadata: Mapping[str, object],
    accelerator: Accelerator | None,
) -> None:
    state = _state_dict(model, accelerator)

    def write() -> None:
        epoch = int(metadata["epoch"])
        generation = output_dir / f".best_epoch_{epoch}"
        generation.mkdir()
        checkpoint = generation / filename
        _atomic_save_checkpoint(checkpoint, state)
        bound_metadata = {**metadata, "checkpoint_sha256": _sha256_file(checkpoint)}
        _atomic_write_json(generation / "metadata.json", bound_metadata)
        temporary_link = output_dir / f".best_epoch_{epoch}.link.tmp"
        temporary_link.symlink_to(generation.name, target_is_directory=True)
        temporary_link.replace(output_dir / "best")

    _rank_zero_action(accelerator, "write selected checkpoint", write)
    if accelerator is not None:
        accelerator.wait_for_everyone()


def _write_history(
    output_dir: Path,
    history: list[dict[str, object]],
    accelerator: Accelerator | None,
) -> None:
    _rank_zero_action(
        accelerator,
        "write training history",
        lambda: _atomic_write_csv(output_dir / "train_log.csv", history),
    )


def _restore_best(
    model: nn.Module,
    path: Path,
    accelerator: Accelerator | None,
) -> None:
    if accelerator is not None:
        accelerator.wait_for_everyone()
    load_error = False
    caught_error: Exception | None = None
    try:
        metadata_path = path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_sha256 = metadata.get("checkpoint_sha256")
        actual_sha256 = _sha256_file(path)
        if expected_sha256 != actual_sha256:
            raise ValueError(
                "selected checkpoint hash mismatch: "
                f"{expected_sha256} != {actual_sha256}"
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        target = accelerator.unwrap_model(model) if accelerator is not None else model
        target.load_state_dict(state, strict=True)
    except Exception as error:
        if accelerator is None or accelerator.num_processes == 1:
            raise
        load_error = True
        caught_error = error
    if accelerator is not None and accelerator.num_processes > 1:
        local = torch.tensor(
            [int(load_error)], dtype=torch.int64, device=accelerator.device
        )
        failures = accelerator.gather(local).detach().cpu().reshape(-1)
        if bool(failures.any()):
            raise RuntimeError(
                "selected checkpoint restore failed on at least one rank"
            ) from caught_error


def _validation_value(
    validation_metric: ResidualValidationMetric,
    model: nn.Module,
    accelerator: Accelerator | None,
    provenance: CheckpointProvenance,
) -> float:
    try:
        local_value = float(validation_metric.evaluate(model, provenance))
    except Exception:
        if accelerator is None or accelerator.num_processes == 1:
            raise
        local_value = math.nan
    return _metric_across_ranks(local_value, accelerator)


def _stream_joint_batch_pairs(
    dependency_batch_factory: JointBatchFactory,
    response_batch_factory: ResponseBatchFactory,
    epoch: int,
    accelerator: Accelerator | None,
) -> Iterable[tuple[OnlineSupervisedBatch, ResponseSupervisionBatch]]:
    """Stream one dependency/response pair after rank-shared readiness checks."""
    try:
        dependency_iterator = iter(dependency_batch_factory(epoch))
        dependency_error = False
    except Exception:
        dependency_iterator = iter(())
        dependency_error = True
    try:
        response_iterator = iter(response_batch_factory(epoch))
        response_error = False
    except Exception:
        response_iterator = iter(())
        response_error = True
    step = 0
    while True:
        dependency_batch: OnlineSupervisedBatch | None = None
        response_batch: ResponseSupervisionBatch | None = None
        state = 2 if dependency_error or response_error else 0
        if state != 2:
            try:
                dependency_batch = next(dependency_iterator)
                dependency_batch.validate()
                state = 1
            except StopIteration:
                state = 0
            except Exception:
                state = 2
        if state == 1:
            try:
                response_batch = next(response_iterator)
            except StopIteration:
                try:
                    response_iterator = iter(response_batch_factory(epoch))
                    response_batch = next(response_iterator)
                except Exception:
                    state = 2
            except Exception:
                state = 2
            if response_batch is not None:
                try:
                    response_batch.validate()
                except Exception:
                    state = 2
        if accelerator is None or accelerator.num_processes == 1:
            if state == 2:
                raise RuntimeError(
                    f"joint batch factory failed before step {step}"
                )
            if state == 0:
                if step == 0:
                    raise ValueError("dependency batch factory produced no batches")
                return
        else:
            local = torch.tensor([state], dtype=torch.int64, device=accelerator.device)
            gathered = accelerator.gather(local).detach().cpu().reshape(-1)
            states = tuple(int(value) for value in gathered.tolist())
            if len(states) != accelerator.num_processes:
                raise RuntimeError("joint batch readiness did not gather every rank")
            if 2 in states:
                raise RuntimeError(
                    f"a joint batch factory failed before step {step}"
                )
            if all(value == 0 for value in states):
                if step == 0:
                    raise RuntimeError("dependency batch counts must be positive")
                return
            if any(value == 0 for value in states):
                raise RuntimeError(
                    f"rank optimizer-step counts differ before step {step}: {states}"
                )
        if dependency_batch is None or response_batch is None:
            raise RuntimeError("joint readiness accepted a missing local batch")
        yield dependency_batch, response_batch
        step += 1


def _outcome(
    history: list[dict[str, object]], best_epoch: int, stopped_early: bool
) -> dict[str, object]:
    return {
        "best_epoch": best_epoch,
        "best_metric": float(history[best_epoch][SELECTION_NAME]),
        "stopped_epoch": int(history[-1]["epoch"]),
        "stopped_early": stopped_early,
        "selection_name": SELECTION_NAME,
        "selection_direction": SELECTION_DIRECTION,
        "epochs": history,
    }


def _validate_contract(
    config: Stage2Config,
    provenance: CheckpointProvenance,
    validation_metric: ResidualValidationMetric,
) -> None:
    if config.selection.metric != SELECTION_NAME:
        raise ValueError(
            f"selection metric must be {SELECTION_NAME}, got {config.selection.metric}"
        )
    if config.selection.direction != SELECTION_DIRECTION:
        raise ValueError(
            "selection direction must be maximize, got "
            f"{config.selection.direction}"
        )
    if config.source_sha256 != provenance.config_sha256:
        raise ValueError(
            "checkpoint provenance config SHA does not match Stage 2 config"
        )
    split_path = Path(config.paths.split)
    if _sha256_file(split_path) != provenance.split_sha256:
        raise ValueError(
            "selection provenance split SHA does not match configured split"
        )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if tuple(split.get("val", ())) != provenance.validation_model_ids:
        raise ValueError(
            "selection provenance validation ModelIDs do not match configured split"
        )
    unlabeled = set(split.get("unlabeled_train", ()))
    supervised_train = tuple(
        model_id for model_id in split.get("train", ()) if model_id not in unlabeled
    )
    expected_centering_sha256 = hashlib.sha256(
        "\n".join(supervised_train).encode()
    ).hexdigest()
    if provenance.centering_fit_model_ids_sha256 != expected_centering_sha256:
        raise ValueError("centering-fit SHA does not match supervised train split")
    provenance.json_ready()
    expected_metric_contract = (
        provenance.validation_model_ids,
        provenance.split_sha256,
        provenance.gene_effect_sha256,
        provenance.mu_train_sha256,
    )
    actual_metric_contract = (
        validation_metric.validation_model_ids,
        validation_metric.split_sha256,
        validation_metric.gene_effect_sha256,
        validation_metric.mu_train_sha256,
    )
    if actual_metric_contract != expected_metric_contract:
        raise ValueError(
            "validation metric is not bound to the checkpoint selection provenance"
        )


def train_frozen_warmup(
    model: GeneEffectE2EModel,
    batch_factory: WarmupBatchFactory,
    validation_metric: ResidualValidationMetric,
    output_dir: Path,
    config: Stage2Config,
    provenance: CheckpointProvenance,
    *,
    step_fn: WarmupStep = warmup_step,
) -> dict[str, object]:
    """Train and select the frozen-backbone head in one local process."""
    if not model.backbone_frozen:
        raise ValueError("train_frozen_warmup requires a frozen backbone")
    model.assert_frozen_backbone_clean()
    _validate_contract(config, provenance, validation_metric)
    if not validation_metric.uses_precomputed_batches:
        raise TypeError("warmup selection requires precomputed validation batches")
    _prepare_fresh_dir(output_dir, None)
    optimizer = build_warmup_optimizer(model, config)
    history: list[dict[str, object]] = []
    best_metric = -math.inf
    best_epoch = -1
    stale_epochs = 0
    stopped_early = False

    for epoch in range(config.warmup.max_epochs):
        step_metrics = [
            step_fn(
                model,
                batch,
                optimizer,
                huber_delta=config.loss.huber_delta,
                beta=config.loss.beta,
            )
            for batch in batch_factory(epoch)
        ]
        train_metrics = _mean_step_metrics(step_metrics)
        model.eval()
        with torch.no_grad():
            metric = _validation_value(validation_metric, model, None, provenance)
        row: dict[str, object] = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            SELECTION_NAME: metric,
        }
        history.append(row)
        _write_history(Path(output_dir), history, None)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            stale_epochs = 0
            _save_best(
                model.head,
                Path(output_dir),
                "head.pt",
                _metadata(
                    kind="head",
                    epoch=epoch,
                    metric=metric,
                    provenance=provenance,
                ),
                None,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config.warmup.patience:
            stopped_early = True
            break

    if best_epoch < 0:
        raise RuntimeError("no epoch produced a finite validation metric")
    _restore_best(model.head, Path(output_dir) / "best/head.pt", None)
    return _outcome(history, best_epoch, stopped_early)


def train_joint(
    model: GeneEffectE2EModel,
    dependency_batch_factory: JointBatchFactory,
    response_batch_factory: ResponseBatchFactory,
    validation_metric: ResidualValidationMetric,
    output_dir: Path,
    config: Stage2Config,
    provenance: CheckpointProvenance,
    *,
    response_loss_fn: ResponseLoss,
    lambda_dep: float,
    accelerator: Accelerator | None = None,
    forward_model: nn.Module | None = None,
    step_fn: JointStep = joint_step,
) -> dict[str, object]:
    """Jointly tune the full model with one cycling response batch per step."""
    if model.backbone_frozen:
        raise ValueError("train_joint requires an unfrozen backbone")
    _validate_contract(config, provenance, validation_metric)
    if validation_metric.uses_precomputed_batches:
        raise TypeError("joint selection requires online validation batches")
    if provenance.lambda_calibration_report is None:
        raise ValueError("joint training requires a lambda calibration report")
    calibrated = float(provenance.lambda_calibration_report.get("lambda_dep", math.nan))
    if not math.isfinite(calibrated) or calibrated != float(lambda_dep):
        raise ValueError(
            "lambda_dep does not match the supplied calibration report: "
            f"{lambda_dep} vs {calibrated}"
        )
    if accelerator is not None and accelerator.num_processes > 1:
        if forward_model is None:
            raise ValueError(
                "multi-rank joint training requires a prepared forward_model"
            )
        if forward_model is model:
            raise ValueError("multi-rank forward_model must be an actual wrapper")
        if accelerator.unwrap_model(forward_model) is not model:
            raise ValueError("forward_model is not the prepared wrapper of model")
    _prepare_fresh_dir(output_dir, accelerator)
    optimizer = build_joint_optimizer(model, config)
    if accelerator is not None:
        optimizer = accelerator.prepare(optimizer)
    history: list[dict[str, object]] = []
    best_metric = -math.inf
    best_epoch = -1
    stale_epochs = 0
    stopped_early = False

    for epoch in range(config.joint.max_epochs):
        step_metrics: list[JointStepMetrics] = []
        for dependency_batch, response_batch in _stream_joint_batch_pairs(
            dependency_batch_factory,
            response_batch_factory,
            epoch,
            accelerator,
        ):
            step_metrics.append(
                step_fn(
                    model,
                    dependency_batch,
                    response_batch,
                    optimizer,
                    response_loss_fn=response_loss_fn,
                    lambda_dep=lambda_dep,
                    huber_delta=config.loss.huber_delta,
                    beta=config.loss.beta,
                    grad_clip=config.joint.grad_clip,
                    accelerator=accelerator,
                    forward_model=forward_model,
                )
            )
        train_metrics = _mean_step_metrics(step_metrics)
        model.eval()
        if forward_model is not None:
            forward_model.eval()
        with torch.no_grad():
            metric = _validation_value(
                validation_metric,
                forward_model or model,
                accelerator,
                provenance,
            )
        row: dict[str, object] = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            SELECTION_NAME: metric,
        }
        history.append(row)
        _write_history(Path(output_dir), history, accelerator)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            stale_epochs = 0
            _save_best(
                forward_model or model,
                Path(output_dir),
                "e2e_state.pt",
                _metadata(
                    kind="e2e",
                    epoch=epoch,
                    metric=metric,
                    provenance=provenance,
                ),
                accelerator,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config.joint.patience:
            stopped_early = True
            break

    if best_epoch < 0:
        raise RuntimeError("no epoch produced a finite validation metric")
    _restore_best(
        forward_model or model,
        Path(output_dir) / "best/e2e_state.pt",
        accelerator,
    )
    return _outcome(history, best_epoch, stopped_early)
