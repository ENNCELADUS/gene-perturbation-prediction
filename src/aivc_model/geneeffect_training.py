"""Pure training steps for frozen-head warmup and Stage-2 joint tuning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from accelerate import Accelerator

from aivc_model.geneeffect_e2e import (
    GeneEffectE2EModel,
    OnlineConditionBatch,
    PrecomputedFeatureBatch,
    ResponseForwardBatch,
)
from aivc_model.geneeffect_head import (
    MaskedGeneEffectLoss,
    masked_geneeffect_residual_loss,
)
from aivc_model.response_training import ResponseLoss, predict_bags
from aivc_model.stage2_config import Stage2Config


@dataclass(frozen=True)
class SupervisedMatrix:
    """Gene-major residual targets and masks aligned to a flat feature batch."""

    target: torch.Tensor
    label_mask: torch.Tensor
    g_var_mask: torch.Tensor
    gene_symbols: tuple[str, ...]
    context_model_ids_by_gene: tuple[tuple[str, ...], ...]
    residual_target_sha256: str
    centering_fit_model_ids_sha256: str

    @property
    def shape(self) -> tuple[int, int]:
        if self.target.dim() != 2:
            raise ValueError("target must be [n_genes, n_contexts]")
        return int(self.target.shape[0]), int(self.target.shape[1])

    @property
    def pair_count(self) -> int:
        genes, contexts = self.shape
        return genes * contexts

    def validate(self) -> None:
        genes, contexts = self.shape
        if (
            len(self.gene_symbols) != genes
            or len(self.context_model_ids_by_gene) != genes
        ):
            raise ValueError("supervision identities do not match target axes")
        if any(len(row) != contexts for row in self.context_model_ids_by_gene):
            raise ValueError(
                "every supervision gene row must name the expected number of contexts"
            )
        if any(not value for value in self.gene_symbols) or any(
            not value for row in self.context_model_ids_by_gene for value in row
        ):
            raise ValueError("supervision identities cannot be empty")
        if (
            self.label_mask.shape != (genes, contexts)
            or self.label_mask.dtype != torch.bool
        ):
            raise ValueError("label_mask must be boolean and match target")
        if bool(self.label_mask.any()) and not bool(
            torch.isfinite(self.target[self.label_mask]).all()
        ):
            raise ValueError("labeled supervision targets must be finite")
        if self.g_var_mask.shape != (genes,) or self.g_var_mask.dtype != torch.bool:
            raise ValueError("g_var_mask must be boolean [n_genes]")
        if not bool(self.label_mask.any()):
            raise ValueError("supervision contains no labeled pairs")
        true_pairs = [
            (gene, model_id)
            for row, gene in enumerate(self.gene_symbols)
            for column, model_id in enumerate(self.context_model_ids_by_gene[row])
            if bool(self.label_mask[row, column])
        ]
        if len(set(true_pairs)) != len(true_pairs):
            raise ValueError(
                "supervision contains duplicate labeled gene/context pairs"
            )


@dataclass(frozen=True)
class PrecomputedSupervisedBatch:
    features: PrecomputedFeatureBatch
    supervision: SupervisedMatrix
    objective_weight: float = 1.0

    def validate(self) -> None:
        self.features.validate()
        self.supervision.validate()
        if self.features.batch_size != self.supervision.pair_count:
            raise ValueError(
                "flat precomputed features do not align with target matrix"
            )
        expected_genes = tuple(
            gene
            for gene in self.supervision.gene_symbols
            for _ in range(self.supervision.shape[1])
        )
        expected_models = tuple(
            model_id
            for row in self.supervision.context_model_ids_by_gene
            for model_id in row
        )
        if (
            self.features.gene_symbols != expected_genes
            or self.features.model_ids != expected_models
        ):
            raise ValueError(
                "precomputed feature identities are not gene-major aligned"
            )
        if self.objective_weight not in (0.0, 1.0):
            raise ValueError("objective_weight must be 0.0 (DDP padding) or 1.0")


@dataclass(frozen=True)
class OnlineSupervisedBatch:
    conditions: OnlineConditionBatch
    supervision: SupervisedMatrix
    objective_weight: float = 1.0

    def validate(self) -> None:
        self.conditions.validate()
        self.supervision.validate()
        if self.conditions.batch_size != self.supervision.pair_count:
            raise ValueError("flat online conditions do not align with target matrix")
        expected_genes = tuple(
            gene
            for gene in self.supervision.gene_symbols
            for _ in range(self.supervision.shape[1])
        )
        expected_models = tuple(
            model_id
            for row in self.supervision.context_model_ids_by_gene
            for model_id in row
        )
        if (
            self.conditions.genes != expected_genes
            or self.conditions.model_ids != expected_models
        ):
            raise ValueError("online condition identities are not gene-major aligned")
        if self.objective_weight not in (0.0, 1.0):
            raise ValueError("objective_weight must be 0.0 (DDP padding) or 1.0")


@dataclass(frozen=True)
class ResponseSupervisionBatch:
    """One response-anchor minibatch paired with a dependency minibatch."""

    controls_tx1: tuple[torch.Tensor, ...]
    observed_hvg: tuple[torch.Tensor, ...]
    control_hvg: tuple[torch.Tensor, ...]
    genes: tuple[str, ...]
    objective_weights: torch.Tensor
    batch_weight: float = 1.0

    def validate(self) -> None:
        size = len(self.genes)
        if size == 0 or not (
            len(self.controls_tx1)
            == len(self.observed_hvg)
            == len(self.control_hvg)
            == size
        ):
            raise ValueError("response batch fields must be non-empty and aligned")
        if self.objective_weights.shape != (size,):
            raise ValueError("response objective_weights must align with genes")
        if not self.objective_weights.is_floating_point() or not bool(
            torch.isfinite(self.objective_weights).all()
        ):
            raise ValueError("response objective_weights must be finite floating point")
        if bool((self.objective_weights < 0).any()) or not bool(
            self.objective_weights.sum() > 0
        ):
            raise ValueError(
                "response objective_weights must be non-negative with positive sum"
            )
        if self.batch_weight not in (0.0, 1.0):
            raise ValueError("response batch_weight must be 0.0 or 1.0")


@dataclass(frozen=True)
class WarmupStepMetrics:
    total: float
    huber: float
    pearson: float
    n_valid_pairs: int
    n_genes_scored: int


@dataclass(frozen=True)
class JointStepMetrics:
    total: float
    response: float
    dependency: float
    huber: float
    pearson: float
    lambda_dep: float


@dataclass(frozen=True)
class LambdaCalibrationReport:
    lambda_dep: float
    raw_ratios: tuple[float, ...]
    response_gradient_norms: tuple[float, ...]
    dependency_gradient_norms: tuple[float, ...]


@dataclass(frozen=True)
class ResponseObjective:
    """Local weighted response mean plus its unreduced numerator/count."""

    mean: torch.Tensor
    weighted_sum: torch.Tensor
    weight_sum: torch.Tensor


@dataclass(frozen=True)
class _DistributedDependencyObjective:
    backward: torch.Tensor
    reported_total: torch.Tensor
    reported_huber: torch.Tensor
    reported_pearson: torch.Tensor


def _loss(
    prediction: torch.Tensor,
    supervision: SupervisedMatrix,
    *,
    huber_delta: float,
    beta: float,
) -> MaskedGeneEffectLoss:
    genes, contexts = supervision.shape
    if prediction.shape != (genes * contexts,):
        raise ValueError(
            f"flat prediction must have {genes * contexts} values, got "
            f"{tuple(prediction.shape)}"
        )
    return masked_geneeffect_residual_loss(
        prediction.reshape(genes, contexts),
        supervision.target,
        supervision.label_mask,
        supervision.g_var_mask,
        huber_delta=huber_delta,
        beta=beta,
    )


def build_warmup_optimizer(
    model: GeneEffectE2EModel, config: Stage2Config
) -> torch.optim.AdamW:
    """Create the head-only optimizer and verify the frozen boundary."""
    if not model.backbone_frozen:
        raise ValueError("freeze_backbone() must be called before warmup optimizer")
    model.assert_frozen_backbone_clean()
    return torch.optim.AdamW(
        model.head.parameters(), lr=config.warmup.learning_rate, weight_decay=0.0
    )


def _named_trainable_parameters(module: nn.Module) -> dict[int, nn.Parameter]:
    return {
        id(parameter): parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    }


def build_joint_optimizer(
    model: GeneEffectE2EModel, config: Stage2Config
) -> torch.optim.AdamW:
    """Build explicit STATE/ESM/head LR groups and reject an unowned weight."""
    if model.backbone_frozen:
        raise ValueError("unfreeze_backbone() must be called before joint optimizer")
    state_adapter = getattr(model.backbone, "state_adapter", None)
    perturbations = getattr(model.backbone, "perturbations", None)
    esm_adapter = getattr(perturbations, "adapter", None)
    if state_adapter is None or esm_adapter is None:
        raise TypeError(
            "joint backbone must expose state_adapter and perturbations.adapter"
        )
    state_params = list(state_adapter.parameters())
    esm_params = list(esm_adapter.parameters())
    head_params = list(model.head.parameters())
    grouped = {id(parameter) for parameter in state_params + esm_params + head_params}
    expected = set(_named_trainable_parameters(model))
    if grouped != expected:
        raise ValueError(
            "joint optimizer groups do not exactly cover trainable parameters: "
            f"missing={len(expected - grouped)} extra={len(grouped - expected)}"
        )
    joint = config.joint
    return torch.optim.AdamW(
        [
            {"params": state_params, "lr": joint.state_learning_rate, "name": "state"},
            {
                "params": esm_params,
                "lr": joint.esm_adapter_learning_rate,
                "name": "esm_adapter",
            },
            {"params": head_params, "lr": joint.head_learning_rate, "name": "head"},
        ],
        weight_decay=joint.weight_decay,
    )


def warmup_step(
    model: GeneEffectE2EModel,
    batch: PrecomputedSupervisedBatch,
    optimizer: torch.optim.Optimizer,
    *,
    huber_delta: float = 1.0,
    beta: float = 1.0,
    grad_clip: float | None = None,
    accelerator: Accelerator | None = None,
) -> WarmupStepMetrics:
    """Take one head-only optimization step and re-check the freeze invariant."""
    batch.validate()
    if accelerator is not None and accelerator.num_processes > 1:
        raise ValueError(
            "frozen-head warmup is single-process; do not pass a multi-rank Accelerator"
        )
    if not model.backbone_frozen:
        raise ValueError("warmup_step requires a frozen backbone")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    objective = _loss(
        model.forward_precomputed(batch.features),
        batch.supervision,
        huber_delta=huber_delta,
        beta=beta,
    )
    weighted = _distributed_dependency_objective(
        objective, beta, accelerator, local_weight=batch.objective_weight
    )
    if accelerator is None:
        weighted.backward.backward()
    else:
        accelerator.backward(weighted.backward)
    if grad_clip is not None:
        if accelerator is None:
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)
        else:
            accelerator.clip_grad_norm_(model.head.parameters(), grad_clip)
    optimizer.step()
    model.assert_frozen_backbone_clean()
    return WarmupStepMetrics(
        total=float(weighted.reported_total),
        huber=float(weighted.reported_huber),
        pearson=float(weighted.reported_pearson),
        n_valid_pairs=objective.n_valid_pairs,
        n_genes_scored=objective.n_genes_scored,
    )


def response_objective(
    backbone: nn.Module,
    batch: ResponseSupervisionBatch,
    *,
    loss_fn: ResponseLoss,
    collator_seed: int,
) -> ResponseObjective:
    """Compute the weighted Stage-1 response anchor objective."""
    batch.validate()
    predicted = predict_bags(
        backbone, batch.controls_tx1, batch.genes, seed=int(collator_seed)
    )
    return response_objective_from_predictions(predicted, batch, loss_fn=loss_fn)


def response_objective_from_predictions(
    predicted: Sequence[torch.Tensor],
    batch: ResponseSupervisionBatch,
    *,
    loss_fn: ResponseLoss,
) -> ResponseObjective:
    """Score response predictions already produced inside a top-level DDP call."""
    batch.validate()
    if len(predicted) != len(batch.genes):
        raise ValueError("response predictions do not align with supervision")
    losses: list[torch.Tensor] = []
    for pred, observed, control in zip(
        predicted, batch.observed_hvg, batch.control_hvg, strict=True
    ):
        value, _parts = loss_fn.tensor_parts(
            pred.float(), observed.float(), control.float().mean(dim=0)
        )
        losses.append(value)
    weights = batch.objective_weights.to(device=losses[0].device, dtype=losses[0].dtype)
    weighted_sum = (torch.stack(losses) * weights).sum()
    weight_sum = weights.sum()
    return ResponseObjective(
        mean=weighted_sum / weight_sum,
        weighted_sum=weighted_sum,
        weight_sum=weight_sum,
    )


def _distributed_dependency_objective(
    local: MaskedGeneEffectLoss,
    beta: float,
    accelerator: Accelerator | None,
    *,
    local_weight: float = 1.0,
) -> _DistributedDependencyObjective:
    """Apply global row/gene weighting before Accelerate's DDP averaging."""
    if local_weight not in (0.0, 1.0):
        raise ValueError("local dependency weight must be 0.0 or 1.0")
    if accelerator is None or accelerator.num_processes == 1:
        if local_weight == 0.0:
            raise ValueError("a single-process dependency step cannot be padding")
        return _DistributedDependencyObjective(
            backward=local.total,
            reported_total=local.total.detach(),
            reported_huber=local.huber.detach(),
            reported_pearson=local.pearson.detach(),
        )
    device = local.total.device
    local_counts = torch.tensor(
        [local.n_valid_pairs, local.n_genes_scored],
        dtype=torch.float32,
        device=device,
    ) * float(local_weight)
    global_counts = accelerator.reduce(local_counts, reduction="sum")
    if bool((global_counts <= 0).any()):
        raise RuntimeError(f"global dependency counts are invalid: {global_counts}")
    world = float(accelerator.num_processes)
    backward = (
        local.huber * local_counts[0] * world / global_counts[0]
        + float(beta) * local.pearson * local_counts[1] * world / global_counts[1]
    )
    global_sums = accelerator.reduce(
        torch.stack(
            (
                local.huber.detach() * local_counts[0],
                local.pearson.detach() * local_counts[1],
            )
        ),
        reduction="sum",
    )
    reported_huber = global_sums[0] / global_counts[0]
    reported_pearson = global_sums[1] / global_counts[1]
    return _DistributedDependencyObjective(
        backward=backward,
        reported_total=reported_huber + float(beta) * reported_pearson,
        reported_huber=reported_huber,
        reported_pearson=reported_pearson,
    )


def _distributed_response_objective(
    local: ResponseObjective,
    accelerator: Accelerator | None,
    *,
    local_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return DDP-correct backward loss and globally reported response mean."""
    if local_weight not in (0.0, 1.0):
        raise ValueError("local response weight must be 0.0 or 1.0")
    if accelerator is None or accelerator.num_processes == 1:
        if local_weight == 0.0:
            raise ValueError("a single-process response step cannot be padding")
        return local.mean, local.mean.detach()
    local_weight_sum = local.weight_sum * float(local_weight)
    local_weighted_sum = local.weighted_sum * float(local_weight)
    global_weight = accelerator.reduce(local_weight_sum.detach(), reduction="sum")
    if not bool(global_weight > 0):
        raise RuntimeError("global response objective weight is non-positive")
    backward = local_weighted_sum * float(accelerator.num_processes) / global_weight
    global_sum = accelerator.reduce(local_weighted_sum.detach(), reduction="sum")
    return backward, global_sum / global_weight


def _assert_finite_nonzero_gradient(module: nn.Module, label: str) -> None:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError(f"{label} received no gradient")
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError(f"{label} received a non-finite gradient")
    if not any(bool(gradient.detach().abs().sum() > 0) for gradient in gradients):
        raise RuntimeError(f"{label} gradients are all zero")


def joint_step(
    model: GeneEffectE2EModel,
    dependency_batch: OnlineSupervisedBatch,
    response_batch: ResponseSupervisionBatch,
    optimizer: torch.optim.Optimizer,
    *,
    response_loss_fn: ResponseLoss,
    lambda_dep: float,
    huber_delta: float = 1.0,
    beta: float = 1.0,
    grad_clip: float = 1.0,
    accelerator: Accelerator | None = None,
    forward_model: nn.Module | None = None,
) -> JointStepMetrics:
    """Take one true joint step through response backbone and residual head."""
    dependency_batch.validate()
    if (
        accelerator is not None
        and accelerator.num_processes > 1
        and forward_model is None
    ):
        raise ValueError(
            "multi-rank joint_step requires the Accelerator-prepared forward_model"
        )
    if model.backbone_frozen:
        raise ValueError("joint_step requires an unfrozen backbone")
    if not math.isfinite(lambda_dep) or lambda_dep <= 0:
        raise ValueError("lambda_dep must be finite and positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    response_forward = ResponseForwardBatch(
        controls_tx1=response_batch.controls_tx1,
        genes=response_batch.genes,
    )
    output = (forward_model or model)(
        dependency_batch.conditions, response=response_forward
    )
    dependency = _loss(
        output.delta_hat,
        dependency_batch.supervision,
        huber_delta=huber_delta,
        beta=beta,
    )
    if output.response_predicted is None:
        raise RuntimeError("joint forward omitted response-anchor predictions")
    response = response_objective_from_predictions(
        output.response_predicted,
        response_batch,
        loss_fn=response_loss_fn,
    )
    weighted_dependency = _distributed_dependency_objective(
        dependency,
        beta,
        accelerator,
        local_weight=dependency_batch.objective_weight,
    )
    response_backward, response_reported = _distributed_response_objective(
        response, accelerator, local_weight=response_batch.batch_weight
    )
    total = response_backward + float(lambda_dep) * weighted_dependency.backward
    if accelerator is None:
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        accelerator.backward(total)
        accelerator.clip_grad_norm_(model.parameters(), grad_clip)
    _assert_finite_nonzero_gradient(model.backbone, "backbone")
    _assert_finite_nonzero_gradient(model.head, "GeneEffect head")
    optimizer.step()
    return JointStepMetrics(
        total=float(
            response_reported + float(lambda_dep) * weighted_dependency.reported_total
        ),
        response=float(response_reported),
        dependency=float(weighted_dependency.reported_total),
        huber=float(weighted_dependency.reported_huber),
        pearson=float(weighted_dependency.reported_pearson),
        lambda_dep=float(lambda_dep),
    )


def _gradient_norm(loss: torch.Tensor, parameters: Sequence[nn.Parameter]) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    squared = loss.new_zeros(())
    found = False
    for gradient in gradients:
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("lambda calibration encountered a non-finite gradient")
        squared = squared + gradient.float().square().sum()
        found = True
    if not found:
        raise ValueError("lambda calibration loss has no shared-backbone gradient")
    value = float(squared.sqrt().detach())
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"lambda calibration gradient norm must be positive, got {value}"
        )
    return value


def calibrate_lambda_dep(
    loss_pairs: Sequence[tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]],
    shared_parameters: Sequence[nn.Parameter],
    *,
    clip_min: float = 1e-3,
    clip_max: float = 1e3,
) -> LambdaCalibrationReport:
    """Freeze median ``||grad L_resp|| / ||grad L_dep||`` on train-only batches."""
    if not loss_pairs:
        raise ValueError("lambda calibration requires at least one train-only batch")
    parameters = tuple(
        parameter for parameter in shared_parameters if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("lambda calibration requires trainable shared parameters")
    if not (0 < clip_min <= clip_max):
        raise ValueError("invalid lambda calibration clip interval")
    response_norms: list[float] = []
    dependency_norms: list[float] = []
    ratios: list[float] = []
    for response_closure, dependency_closure in loss_pairs:
        response_norm = _gradient_norm(response_closure(), parameters)
        dependency_norm = _gradient_norm(dependency_closure(), parameters)
        response_norms.append(response_norm)
        dependency_norms.append(dependency_norm)
        ratios.append(response_norm / dependency_norm)
    calibrated = float(np.clip(np.median(ratios), clip_min, clip_max))
    return LambdaCalibrationReport(
        lambda_dep=calibrated,
        raw_ratios=tuple(ratios),
        response_gradient_norms=tuple(response_norms),
        dependency_gradient_norms=tuple(dependency_norms),
    )
