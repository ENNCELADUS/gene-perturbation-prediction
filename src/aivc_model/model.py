"""Models for the STATE-ready AIVC A->B->C pipeline."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
import torch.nn.functional as F

from aivc_model.response import ResponseEncoder, TrainableDiagonalGMM
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable


@dataclass(frozen=True)
class LossWeights:
    latent_mean_delta: float
    latent_energy: float
    hvg_mean_delta: float
    hvg_energy: float
    pred_c: float
    obs_c: float
    occupancy: float
    gmm_nll: float = 0.0
    pred_rank: float = 0.0
    pred_rank_tau: float = 0.25
    pred_rank_pair_margin: float = 0.0
    pred_rank_pair_weight_clip: float = 2.0


class LinearMockStateModel(nn.Module):
    """Small STATE-shaped model for tests and smoke runs without a checkpoint."""

    def __init__(self, input_dim: int, output_dim: int, pert_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.pert_dim = int(pert_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim + self.pert_dim, 32),
            nn.GELU(),
            nn.Linear(32, self.output_dim),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        padded: bool = False,
    ) -> torch.Tensor:
        del padded
        basal = batch["ctrl_cell_emb"]
        pert = batch["pert_emb"]
        return self.net(torch.cat([basal, pert], dim=1))


def _flatten_state_output(output: object) -> torch.Tensor | None:
    if not isinstance(output, torch.Tensor):
        return None
    if output.dim() == 3 and output.shape[0] == 1:
        output = output.squeeze(0)
    elif output.dim() == 3:
        msg = "STATE token features must have batch dimension 1 when using padded=False"
        raise ValueError(msg)
    if output.dim() > 2:
        output = output.reshape(-1, output.shape[-1])
    return output


class StateForwardAdapter(nn.Module):
    """Thin wrapper around an ArcInstitute STATE transition model."""

    def __init__(self, state_model: nn.Module) -> None:
        super().__init__()
        self.state_model = state_model
        self._last_token_features: torch.Tensor | None = None

    @property
    def last_token_features(self) -> torch.Tensor | None:
        """Return token hidden features captured from the most recent forward."""
        return self._last_token_features

    def forward(
        self,
        control_cells: torch.Tensor,
        perturbation: torch.Tensor,
        gene: str,
        batch_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_token_features = None
        if hasattr(self.state_model, "_token_features"):
            try:
                setattr(self.state_model, "_token_features", None)
            except AttributeError:
                pass
        pert = perturbation.unsqueeze(0).expand(control_cells.shape[0], -1)
        batch: dict[str, Any] = {
            "ctrl_cell_emb": control_cells,
            "pert_emb": pert,
            "pert_name": [gene] * int(control_cells.shape[0]),
        }
        if batch_indices is not None:
            batch["batch"] = batch_indices.to(control_cells.device)
        elif getattr(self.state_model, "batch_encoder", None) is not None:
            batch["batch"] = torch.zeros(
                control_cells.shape[0],
                dtype=torch.long,
                device=control_cells.device,
            )
        if hasattr(self.state_model, "predict_step"):
            try:
                output = self.state_model.predict_step(batch, batch_idx=0, padded=False)
            except TypeError:
                output = self.state_model.predict_step(batch, 0)
        else:
            try:
                output = self.state_model(batch, padded=False)
            except TypeError:
                output = self.state_model(batch)
        if isinstance(output, dict):
            output = output["preds"]
        if isinstance(output, tuple):
            output = output[0]
        token_features = getattr(self.state_model, "_token_features", None)
        self._last_token_features = _flatten_state_output(token_features)
        if output.dim() == 3 and output.shape[0] == 1:
            output = output.squeeze(0)
        if output.dim() > 2:
            output = output.reshape(-1, output.shape[-1])
        return output


class PerturbationVectorAdapter(nn.Module):
    """Known perturbation vectors plus trainable mean-initialized missing entries."""

    def __init__(
        self,
        genes: list[str],
        known_vectors: dict[str, np.ndarray],
        pert_dim: int,
    ) -> None:
        super().__init__()
        self.genes = list(genes)
        self.pert_dim = int(pert_dim)
        self._known_genes = set(known_vectors)
        known_arrays = [
            np.asarray(vector, dtype=np.float32)
            for gene, vector in known_vectors.items()
            if gene in set(self.genes)
        ]
        if known_arrays:
            mean_vector = np.mean(np.vstack(known_arrays), axis=0).astype(np.float32)
        else:
            mean_vector = np.zeros(self.pert_dim, dtype=np.float32)
            if self.pert_dim:
                mean_vector[0] = 1.0
        if mean_vector.shape[0] != self.pert_dim:
            msg = (
                f"Known perturbation vector dim {mean_vector.shape[0]} does not "
                f"match configured pert_dim={self.pert_dim}"
            )
            raise ValueError(msg)
        vectors: list[np.ndarray] = []
        self._missing_param_keys: dict[str, str] = {}
        self._gene_to_index: dict[str, int] = {}
        self.missing_vectors = nn.ParameterDict()
        for index, gene in enumerate(self.genes):
            self._gene_to_index[gene] = index
            if gene in known_vectors:
                vector = np.asarray(known_vectors[gene], dtype=np.float32)
                if vector.shape[0] != self.pert_dim:
                    msg = f"Perturbation vector for {gene} has wrong dimension"
                    raise ValueError(msg)
                vectors.append(vector)
            else:
                vectors.append(mean_vector)
                key = f"g{index}"
                self._missing_param_keys[gene] = key
                self.missing_vectors[key] = nn.Parameter(
                    torch.as_tensor(mean_vector.copy(), dtype=torch.float32)
                )
        self.register_buffer("known_matrix", torch.as_tensor(np.vstack(vectors)))

    def forward(self, gene: str) -> torch.Tensor:
        if gene in self._missing_param_keys:
            return self.missing_vectors[self._missing_param_keys[gene]]
        return self.known_matrix[self._gene_to_index[gene]]

    def has_known_vector(self, gene: str) -> bool:
        """Return whether a gene used a loaded perturbation vector."""
        return gene in self._known_genes and gene not in self._missing_param_keys


class Esm2PerturbationAdapter(nn.Module):
    """Map fixed per-gene ESM-2 vectors through one trainable exp08 adapter."""

    def __init__(
        self,
        genes: list[str],
        table: Esm2EmbeddingTable,
        adapter_hidden: int,
        pert_dim: int,
    ) -> None:
        super().__init__()
        from sl_dl_model.encoder import PertAdapter

        self.genes = [str(gene).upper() for gene in genes]
        missing = [gene for gene in self.genes if gene not in table.vectors_by_symbol]
        if missing:
            raise ValueError(f"Unresolved ESM-2 genes: {missing[:10]}")
        matrix = np.vstack([table.vectors_by_symbol[gene] for gene in self.genes])
        self._gene_to_index = {gene: index for index, gene in enumerate(self.genes)}
        self.register_buffer("esm_matrix", torch.as_tensor(matrix, dtype=torch.float32))
        self.adapter = PertAdapter(table.dim, int(adapter_hidden), int(pert_dim))

    def forward(self, gene: str) -> torch.Tensor:
        index = self._gene_to_index[str(gene).upper()]
        return self.adapter(self.esm_matrix[index].unsqueeze(0)).squeeze(0)

    def has_embedding(self, gene: str) -> bool:
        """Return whether the adapter contains an ESM-2 vector for ``gene``."""
        return str(gene).upper() in self._gene_to_index

    def has_known_vector(self, gene: str) -> bool:
        """Compatibility alias for prediction artifact metadata."""
        return self.has_embedding(gene)


class ExpressionToLatentProjector(nn.Module):
    """Differentiable expression/HVG to scVI-latent adapter."""

    def __init__(
        self,
        weight: np.ndarray,
        bias: np.ndarray,
        *,
        trainable: bool,
    ) -> None:
        super().__init__()
        input_dim, latent_dim = weight.shape
        self.linear = nn.Linear(int(input_dim), int(latent_dim))
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(weight.T, dtype=torch.float32))
            self.linear.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
        if not trainable:
            for parameter in self.linear.parameters():
                parameter.requires_grad = False

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.linear(expression)


class FixedGMMFeatureizer(nn.Module):
    """Fixed diagonal-GMM occupancy features with differentiable responsibilities."""

    def __init__(
        self,
        means: np.ndarray,
        variances: np.ndarray,
        weights: np.ndarray,
        control_bag: np.ndarray,
    ) -> None:
        super().__init__()
        weights_array = np.asarray(weights, dtype=np.float32)
        if weights_array.shape != (means.shape[0],):
            msg = (
                f"GMM weights shape {weights_array.shape} does not match "
                f"n_components={means.shape[0]}"
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(weights_array)) or np.any(weights_array <= 0.0):
            msg = "GMM weights must be finite and positive"
            raise ValueError(msg)
        weight_sum = float(weights_array.sum())
        if weight_sum <= 0.0:
            msg = "GMM weights must have positive sum"
            raise ValueError(msg)
        weights_array = weights_array / weight_sum
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float32))
        self.register_buffer(
            "variances",
            torch.as_tensor(variances, dtype=torch.float32),
        )
        self.register_buffer(
            "log_weights",
            torch.as_tensor(np.log(weights_array), dtype=torch.float32),
        )
        control = torch.as_tensor(control_bag, dtype=torch.float32)
        self.register_buffer("control_feature", self._occupancy(control))

    @property
    def output_dim(self) -> int:
        n_components = int(self.means.shape[0])
        latent_dim = int(self.means.shape[1])
        return n_components * 2 + latent_dim * 2 + 1

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        occupancy = self._occupancy(bag)
        mean = bag.mean(dim=0)
        var = bag.var(dim=0, unbiased=False)
        entropy = -(occupancy * occupancy.clamp_min(1e-8).log()).sum().view(1)
        return torch.cat(
            [occupancy, occupancy - self.control_feature, mean, var, entropy],
            dim=0,
        )

    def _occupancy(self, bag: torch.Tensor) -> torch.Tensor:
        diff = bag.unsqueeze(1) - self.means.unsqueeze(0)
        log_var = self.variances.clamp_min(1e-8).log()
        log_prob = -0.5 * (diff.square() / self.variances.clamp_min(1e-8)).sum(dim=2)
        log_prob = log_prob - 0.5 * log_var.sum(dim=1).unsqueeze(0)
        log_prob = log_prob + self.log_weights.unsqueeze(0)
        return log_prob.softmax(dim=1).mean(dim=0)


class MLPHead(nn.Module):
    """Small MLP dependency head."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden in hidden_units:
            layers.extend(
                [
                    nn.Linear(previous, int(hidden)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = int(hidden)
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AivcModel(nn.Module):
    """End-to-end STATE A->B plus shared response B->C model."""

    def __init__(
        self,
        *,
        state_adapter: StateForwardAdapter,
        perturbations: PerturbationVectorAdapter | Esm2PerturbationAdapter,
        response_encoder: ResponseEncoder,
        response_pooler: TrainableDiagonalGMM,
        c_head: MLPHead,
        control_expression_mean: np.ndarray,
    ) -> None:
        super().__init__()
        self.state_adapter = state_adapter
        self.perturbations = perturbations
        self.response_encoder = response_encoder
        self.response_pooler = response_pooler
        self.c_head = c_head
        self.register_buffer(
            "control_expression_mean",
            torch.as_tensor(control_expression_mean, dtype=torch.float32),
        )

    def predict_response(
        self,
        control_cells: torch.Tensor,
        gene: str,
        batch_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        perturbation = self.perturbations(gene)
        predicted_expression = self.state_adapter(
            control_cells,
            perturbation,
            gene,
            batch_indices,
        )
        return predicted_expression, self.response_encoder(predicted_expression)

    def predict_c_from_response(
        self,
        expression_bag: torch.Tensor,
        control_expression_bag: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.response_encoder(expression_bag)
        control_latent = self.response_encoder(control_expression_bag)
        return self.c_head(self.response_pooler(latent, control_latent))

    def losses_for_gene(
        self,
        *,
        gene: str,
        control_chunks: tuple[torch.Tensor, ...],
        target_expression_chunks: tuple[torch.Tensor, ...],
        target_latent_chunks: tuple[torch.Tensor, ...],
        batch_index_chunks: tuple[torch.Tensor | None, ...],
        y: torch.Tensor,
        weights: LossWeights,
        gene_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward(
            gene=gene,
            control_chunks=control_chunks,
            target_expression_chunks=target_expression_chunks,
            target_latent_chunks=target_latent_chunks,
            batch_index_chunks=batch_index_chunks,
            y=y,
            weights=weights,
            gene_mask=gene_mask,
        )

    def forward(
        self,
        *,
        gene: str | tuple[str, ...],
        control_chunks: tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], ...],
        target_expression_chunks: tuple[torch.Tensor, ...]
        | tuple[tuple[torch.Tensor, ...], ...],
        target_latent_chunks: tuple[torch.Tensor, ...]
        | tuple[tuple[torch.Tensor, ...], ...],
        batch_index_chunks: tuple[torch.Tensor | None, ...]
        | tuple[tuple[torch.Tensor | None, ...], ...],
        y: torch.Tensor,
        weights: LossWeights,
        gene_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute one gene-level A->B->C loss through the module forward path."""
        if not isinstance(gene, str):
            return self._forward_gene_batch(
                gene=gene,
                control_chunks=control_chunks,
                target_expression_chunks=target_expression_chunks,
                target_latent_chunks=target_latent_chunks,
                batch_index_chunks=batch_index_chunks,
                y=y,
                weights=weights,
                gene_mask=gene_mask,
            )
        if gene_mask is not None and gene_mask.reshape(-1).shape[0] != 1:
            msg = "Single-gene forward requires a scalar gene_mask"
            raise ValueError(msg)
        return self._forward_one_gene(
            gene=gene,
            control_chunks=control_chunks,
            target_expression_chunks=target_expression_chunks,
            target_latent_chunks=target_latent_chunks,
            batch_index_chunks=batch_index_chunks,
            y=y,
            weights=weights,
        )

    def _forward_gene_batch(
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
        if not (
            len(gene)
            == len(control_chunks)
            == len(target_expression_chunks)
            == len(target_latent_chunks)
            == len(batch_index_chunks)
        ):
            msg = "All gene batch inputs must have the same length"
            raise ValueError(msg)
        y_values = y.reshape(-1)
        if y_values.shape[0] != len(gene):
            msg = "Batched y must have one value per gene"
            raise ValueError(msg)
        if gene_mask is None:
            valid_mask = torch.ones(len(gene), dtype=torch.bool, device=y_values.device)
        else:
            valid_mask = gene_mask.reshape(-1).to(
                device=y_values.device, dtype=torch.bool
            )
            if valid_mask.shape[0] != len(gene):
                msg = "gene_mask must have one value per gene"
                raise ValueError(msg)
        per_gene_losses = [
            self._forward_one_gene(
                gene=current_gene,
                control_chunks=current_control,
                target_expression_chunks=current_target_expression,
                target_latent_chunks=current_target_latent,
                batch_index_chunks=current_batch_indices,
                y=y_values[index],
                weights=weights,
            )
            for index, (
                current_gene,
                current_control,
                current_target_expression,
                current_target_latent,
                current_batch_indices,
            ) in enumerate(
                zip(
                    gene,
                    control_chunks,
                    target_expression_chunks,
                    target_latent_chunks,
                    batch_index_chunks,
                    strict=True,
                )
            )
        ]
        stacked = {
            key: torch.stack([losses[key] for losses in per_gene_losses])
            for key in (
                "total",
                "hvg_mean_delta",
                "hvg_energy",
                "latent_mean_delta",
                "latent_energy",
                "pred_c",
                "obs_c",
                "occupancy",
                "gmm_nll",
                "pred_y",
                "obs_y",
            )
        }
        valid_count = valid_mask.sum().clamp_min(1)
        pred_rank = _pairwise_ranknet_loss(
            stacked["pred_y"][valid_mask],
            y_values[valid_mask],
            tau=float(weights.pred_rank_tau),
            pair_margin=float(weights.pred_rank_pair_margin),
            pair_weight_clip=float(weights.pred_rank_pair_weight_clip),
        )
        weighted_rank = float(weights.pred_rank) * pred_rank
        rank_share = weighted_rank / valid_count.to(dtype=stacked["total"].dtype)
        per_gene_total = torch.where(
            valid_mask,
            stacked["total"] + rank_share,
            stacked["total"] * 0.0,
        )
        total = stacked["total"][valid_mask].sum() + weighted_rank
        return {
            "total": total,
            "hvg_mean_delta": _masked_mean(stacked["hvg_mean_delta"], valid_mask),
            "hvg_energy": _masked_mean(stacked["hvg_energy"], valid_mask),
            "latent_mean_delta": _masked_mean(stacked["latent_mean_delta"], valid_mask),
            "latent_energy": _masked_mean(stacked["latent_energy"], valid_mask),
            "pred_c": _masked_mean(stacked["pred_c"], valid_mask),
            "obs_c": _masked_mean(stacked["obs_c"], valid_mask),
            "occupancy": _masked_mean(stacked["occupancy"], valid_mask),
            "gmm_nll": _masked_mean(stacked["gmm_nll"], valid_mask),
            "pred_rank": pred_rank,
            "pred_y": stacked["pred_y"],
            "obs_y": stacked["obs_y"],
            "per_gene_total_loss": per_gene_total,
            "per_gene_hvg_mean_delta": stacked["hvg_mean_delta"],
            "per_gene_hvg_energy": stacked["hvg_energy"],
            "per_gene_latent_mean_delta": stacked["latent_mean_delta"],
            "per_gene_latent_energy": stacked["latent_energy"],
            "per_gene_pred_c": stacked["pred_c"],
            "per_gene_obs_c": stacked["obs_c"],
            "per_gene_occupancy": stacked["occupancy"],
            "per_gene_gmm_nll": stacked["gmm_nll"],
            "per_gene_pred_rank": torch.where(
                valid_mask,
                pred_rank.expand(len(gene)),
                pred_rank.expand(len(gene)) * 0.0,
            ),
        }

    def _forward_one_gene(
        self,
        *,
        gene: str,
        control_chunks: tuple[torch.Tensor, ...],
        target_expression_chunks: tuple[torch.Tensor, ...],
        target_latent_chunks: tuple[torch.Tensor, ...],
        batch_index_chunks: tuple[torch.Tensor | None, ...],
        y: torch.Tensor,
        weights: LossWeights,
    ) -> dict[str, torch.Tensor]:
        hvg_mean_delta_terms: list[torch.Tensor] = []
        hvg_energy_terms: list[torch.Tensor] = []
        compute_hvg_energy = float(weights.hvg_energy) != 0.0
        compute_latent_energy = float(weights.latent_energy) != 0.0
        if not (
            len(control_chunks)
            == len(target_expression_chunks)
            == len(target_latent_chunks)
            == len(batch_index_chunks)
        ):
            msg = "All chunk inputs must have the same length"
            raise ValueError(msg)
        chunk_sizes = [int(control.shape[0]) for control in control_chunks]
        batched_control = torch.cat(control_chunks, dim=0)
        batched_batch_indices = _concat_optional_batch_indices(
            batch_index_chunks,
            chunk_sizes,
            batched_control.device,
        )
        predicted_expression, predicted_latent = self.predict_response(
            batched_control,
            gene,
            batched_batch_indices,
        )
        target_expression = torch.cat(target_expression_chunks, dim=0)
        observed_latent = self.response_encoder(target_expression)
        control_latent = self.response_encoder(batched_control)
        predicted_expression_chunks = predicted_expression.split(chunk_sizes, dim=0)
        chunk_iter = zip(
            predicted_expression_chunks,
            target_expression_chunks,
            strict=True,
        )
        for predicted_chunk, target_chunk in chunk_iter:
            hvg_mean_delta_terms.append(
                _mean_delta_loss(
                    predicted_chunk,
                    target_chunk,
                    self.control_expression_mean,
                )
            )
            if compute_hvg_energy:
                hvg_energy_terms.append(
                    _energy_distance(predicted_chunk, target_chunk)
                )
        hvg_mean_delta = torch.stack(hvg_mean_delta_terms).mean()
        hvg_energy = (
            torch.stack(hvg_energy_terms).mean()
            if compute_hvg_energy
            else _zero_like_loss(predicted_expression)
        )
        latent_mean_delta = F.mse_loss(
            predicted_latent.mean(dim=0),
            observed_latent.detach().mean(dim=0),
        )
        latent_energy = (
            _energy_distance(predicted_latent, observed_latent.detach())
            if compute_latent_energy
            else _zero_like_loss(predicted_latent)
        )
        pred_y = self.c_head(self.response_pooler(predicted_latent, control_latent))
        obs_y = self.c_head(self.response_pooler(observed_latent, control_latent))
        pred_c = F.mse_loss(pred_y.view(()), y.view(()))
        obs_c = F.mse_loss(obs_y.view(()), y.view(()))
        occupancy = F.mse_loss(
            self.response_pooler.occupancy(predicted_latent),
            self.response_pooler.occupancy(observed_latent).detach(),
        )
        gmm_nll = self.response_pooler.negative_log_likelihood(observed_latent)
        pred_rank = pred_y.sum() * 0.0
        total = _weighted_loss_sum(
            obs_y,
            (weights.hvg_mean_delta, hvg_mean_delta),
            (weights.hvg_energy, hvg_energy),
            (weights.latent_mean_delta, latent_mean_delta),
            (weights.latent_energy, latent_energy),
            (weights.pred_c, pred_c),
            (weights.obs_c, obs_c),
            (weights.occupancy, occupancy),
            (weights.gmm_nll, gmm_nll),
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


def _mean_delta_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    control_mean: torch.Tensor,
) -> torch.Tensor:
    predicted_delta = predicted.mean(dim=0) - control_mean
    target_delta = target.mean(dim=0) - control_mean
    return F.mse_loss(predicted_delta, target_delta)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any().item()):
        return values[mask].mean()
    return values.sum() * 0.0


def _weighted_loss_sum(
    reference: torch.Tensor,
    *terms: tuple[float, torch.Tensor],
) -> torch.Tensor:
    total = reference.sum() * 0.0
    for weight, loss in terms:
        if float(weight) != 0.0:
            total = total + float(weight) * loss
    return total


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _energy_distance(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cross = torch.cdist(predicted, target).mean()
    pred_self = torch.cdist(predicted, predicted).mean()
    target_self = torch.cdist(target, target).mean()
    return (2.0 * cross - pred_self - target_self).clamp_min(0.0)


def _pairwise_ranknet_loss(
    pred_y: torch.Tensor,
    y: torch.Tensor,
    *,
    tau: float,
    pair_margin: float,
    pair_weight_clip: float,
) -> torch.Tensor:
    """Compute a margin-filtered pairwise logistic ranking loss."""
    if tau <= 0.0:
        msg = "pred_rank_tau must be positive"
        raise ValueError(msg)
    if pair_weight_clip <= 0.0:
        msg = "pred_rank_pair_weight_clip must be positive"
        raise ValueError(msg)
    scores = pred_y.reshape(-1)
    labels = y.reshape(-1).to(device=scores.device, dtype=scores.dtype)
    if scores.shape[0] < 2:
        return scores.sum() * 0.0
    label_delta = labels.unsqueeze(1) - labels.unsqueeze(0)
    score_delta = scores.unsqueeze(1) - scores.unsqueeze(0)
    valid = torch.triu(
        label_delta.abs() >= float(pair_margin),
        diagonal=1,
    )
    if not bool(valid.any().item()):
        return scores.sum() * 0.0
    target = label_delta[valid].sign()
    weighted_loss = F.softplus(-target * score_delta[valid] / float(tau))
    weights = label_delta[valid].abs().clamp_max(float(pair_weight_clip))
    return (weighted_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def _concat_optional_batch_indices(
    batch_index_chunks: tuple[torch.Tensor | None, ...],
    chunk_sizes: list[int],
    device: torch.device,
) -> torch.Tensor | None:
    if all(batch_indices is None for batch_indices in batch_index_chunks):
        return None
    normalized = [
        batch_indices.to(device)
        if batch_indices is not None
        else torch.zeros(size, dtype=torch.long, device=device)
        for batch_indices, size in zip(batch_index_chunks, chunk_sizes, strict=True)
    ]
    return torch.cat(normalized, dim=0)


def fit_fixed_gmm(
    bags: tuple[np.ndarray, ...],
    control_bag: np.ndarray,
    *,
    n_components: int,
    covariance_floor: float,
    random_state: int,
    max_fit_cells: int | None,
) -> FixedGMMFeatureizer:
    """Fit train-only diagonal-GMM prototypes and return a torch featureizer."""
    matrix = np.vstack([control_bag, *bags]).astype(np.float32)
    if max_fit_cells is not None and matrix.shape[0] > max_fit_cells:
        rng = np.random.default_rng(random_state)
        index = rng.choice(matrix.shape[0], size=int(max_fit_cells), replace=False)
        matrix = matrix[index]
    k = min(int(n_components), int(matrix.shape[0]))
    if k < 1:
        msg = "At least one GMM component is required"
        raise ValueError(msg)
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=random_state,
        reg_covar=float(covariance_floor),
    )
    gmm.fit(matrix)
    variances = np.maximum(gmm.covariances_, float(covariance_floor))
    return FixedGMMFeatureizer(
        gmm.means_.astype(np.float32),
        variances,
        gmm.weights_.astype(np.float32),
        control_bag,
    )


def load_state_model(
    *,
    backend: str,
    checkpoint_path: Path | None,
    input_dim: int,
    output_dim: int,
    pert_dim: int,
    emit_checkpoint_output: bool = True,
) -> nn.Module:
    """Load a STATE checkpoint model or a small mock backend."""
    if backend == "linear_mock":
        return LinearMockStateModel(input_dim, output_dim, pert_dim)
    if checkpoint_path is None:
        msg = "state_checkpoint backend requires checkpoint_path"
        raise ValueError(msg)
    module = importlib.import_module("state.tx.models.state_transition")
    cls = getattr(module, "StateTransitionPerturbationModel")
    output_context = (
        nullcontext() if emit_checkpoint_output else _suppress_checkpoint_output()
    )
    with output_context:
        return cls.load_from_checkpoint(str(checkpoint_path), strict=False)


@contextmanager
def _suppress_checkpoint_output() -> Any:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
