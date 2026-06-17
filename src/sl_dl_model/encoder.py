"""Trainable ESM2->pert adapter on top of a frozen Arc STATE backbone.

PertAdapter maps a 1280-d ESM2 gene embedding to a STATE pert token (replacing
STATE's one-hot pert_encoder). StateEncoder wraps a frozen STATE model so that
only the adapter's parameters receive gradients.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from aivc_model.model import StateForwardAdapter, load_state_model

logger = logging.getLogger(__name__)


class PertAdapter(nn.Module):
    """Map an ESM2 gene embedding to a STATE pert token (replaces pert_encoder).

    Args:
        esm_dim: Dimensionality of the input ESM2 embedding.
        hidden: Hidden layer width.
        pert_dim: Output dimensionality matching the STATE pert token size.
    """

    def __init__(self, esm_dim: int, hidden: int, pert_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(esm_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, pert_dim),
        )

    def forward(self, esm: torch.Tensor) -> torch.Tensor:
        """Map ESM2 embeddings to pert tokens.

        Args:
            esm: Input tensor of shape ``(B, esm_dim)``.

        Returns:
            Pert token tensor of shape ``(B, pert_dim)``.
        """
        return self.net(esm)


class StateEncoder(nn.Module):
    """Frozen STATE backbone fed by a trainable ESM2 pert-adapter.

    The STATE backbone parameters are frozen (``requires_grad=False``) and kept
    in ``eval()`` at all times. Only the ``PertAdapter`` is trainable.

    Args:
        backend: ``"linear_mock"`` for tests or ``"state_checkpoint"`` for real runs.
        checkpoint: Path to the STATE checkpoint file (required when backend is
            not ``"linear_mock"``).
        esm_dim: Dimensionality of the input ESM2 embedding.
        adapter_hidden: Hidden width of the PertAdapter MLP.
        pert_dim: Pert token dimensionality (must match STATE's expectation).
        input_dim: Dimensionality of the control cell embeddings fed into STATE.
        output_dim: Dimensionality of the predicted response vectors produced by STATE.
    """

    def __init__(
        self,
        *,
        backend: str,
        checkpoint: Path | None,
        esm_dim: int,
        adapter_hidden: int,
        pert_dim: int,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.adapter = PertAdapter(esm_dim, adapter_hidden, pert_dim)
        state_model = load_state_model(
            backend=backend,
            checkpoint_path=checkpoint,
            input_dim=input_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            emit_checkpoint_output=False,
        )
        for param in state_model.parameters():
            param.requires_grad = False
        state_model.eval()
        self.state = StateForwardAdapter(state_model)

    def train(self, mode: bool = True) -> "StateEncoder":
        """Keep the frozen STATE backbone in eval; let the adapter follow mode.

        Args:
            mode: Training mode flag passed to ``nn.Module.train``.

        Returns:
            ``self`` for chaining.
        """
        super().train(mode)
        self.state.eval()
        self.adapter.train(mode)
        return self

    def forward(
        self,
        esm_vec: torch.Tensor,
        control_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a response bag for one gene over a set of control cells.

        Args:
            esm_vec: ESM2 embedding for the perturbation gene, shape ``(esm_dim,)``.
            control_cells: Control cell embeddings, shape ``(T, input_dim)``.

        Returns:
            Predicted response bag of shape ``(T, output_dim)``.
        """
        pert = self.adapter(esm_vec.unsqueeze(0)).squeeze(0)
        return self.state(control_cells, pert, gene="adapter")


def state_original_token(
    state_model: nn.Module,
    onehot: torch.Tensor,
) -> torch.Tensor:
    """Apply the checkpoint's own pert_encoder to a one-hot (distill target).

    Args:
        state_model: The raw STATE model (unwrapped from StateForwardAdapter).
        onehot: One-hot perturbation vector of shape ``(vocab,)``.

    Returns:
        The pert token tensor produced by the checkpoint's encoder.

    Raises:
        AttributeError: If the model does not expose a ``pert_encoder`` attribute.
    """
    encoder = getattr(state_model, "pert_encoder", None)
    if encoder is None:
        raise AttributeError("state_model has no pert_encoder for distillation")
    with torch.no_grad():
        return encoder(onehot.float())
