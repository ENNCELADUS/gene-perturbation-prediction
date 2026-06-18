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
    """Map an ESM2 gene embedding to a STATE raw pert vector.

    The adapter output is a synthetic *raw* perturbation vector in STATE's
    ``pert_dim`` space; it is fed into the frozen checkpoint's own
    ``pert_encoder`` (it does not replace that encoder). Its width must match
    the loaded checkpoint's ``pert_dim``.

    Args:
        esm_dim: Dimensionality of the input ESM2 embedding.
        hidden: Hidden layer width.
        pert_dim: Output dimensionality matching the STATE pert vector size.
    """

    def __init__(self, esm_dim: int, hidden: int, pert_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(esm_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, pert_dim),
        )

    def forward(self, esm: torch.Tensor) -> torch.Tensor:
        """Map ESM2 embeddings to raw pert vectors.

        Args:
            esm: Input tensor of shape ``(B, esm_dim)``.

        Returns:
            Raw pert vector tensor of shape ``(B, pert_dim)``.
        """
        return self.net(esm)


class StateEncoder(nn.Module):
    """Frozen STATE backbone fed by a trainable ESM2 pert-adapter.

    The STATE backbone parameters are frozen (``requires_grad=False``) and kept
    in ``eval()`` at all times. Only the ``PertAdapter`` is trainable. The
    adapter emits a *raw* pert vector that is fed into the checkpoint's own
    frozen ``pert_encoder`` via ``predict_step``; its width is taken from the
    loaded checkpoint's ``pert_dim`` rather than the configured value.

    Args:
        backend: ``"linear_mock"`` for tests or ``"state_checkpoint"`` for real runs.
        checkpoint: Path to the STATE checkpoint file (required when backend is
            not ``"linear_mock"``).
        esm_dim: Dimensionality of the input ESM2 embedding.
        adapter_hidden: Hidden width of the PertAdapter MLP.
        pert_dim: Requested pert vector dimensionality. For ``"state_checkpoint"``
            this is only a hint: the loaded checkpoint's ``pert_dim`` overrides
            it (a mismatch is logged). It is authoritative for ``"linear_mock"``.
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
        state_model = load_state_model(
            backend=backend,
            checkpoint_path=checkpoint,
            input_dim=input_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            emit_checkpoint_output=False,
        )
        effective_pert_dim = int(getattr(state_model, "pert_dim", pert_dim))
        if backend != "linear_mock" and effective_pert_dim != int(pert_dim):
            logger.warning(
                "configured pert_dim=%d does not match checkpoint pert_dim=%d; "
                "using the checkpoint value for the adapter output width",
                int(pert_dim),
                effective_pert_dim,
            )
        self.pert_dim = effective_pert_dim
        self.adapter = PertAdapter(esm_dim, adapter_hidden, effective_pert_dim)
        adapter_out = self.adapter.net[-1].out_features
        if adapter_out != effective_pert_dim:
            msg = (
                f"adapter output width {adapter_out} does not match STATE "
                f"pert_dim {effective_pert_dim}"
            )
            raise ValueError(msg)
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


def state_encoded_token(
    state_model: nn.Module,
    vec: torch.Tensor,
) -> torch.Tensor:
    """Apply the checkpoint's pert_encoder to a raw pert vector (grad-carrying).

    Unlike :func:`state_original_token`, this does *not* detach: it is used on
    the adapter's own output so gradients flow back to the adapter through the
    frozen ``pert_encoder``.

    Args:
        state_model: The raw STATE model (unwrapped from StateForwardAdapter).
        vec: Raw pert vector of shape ``(..., pert_dim)``.

    Returns:
        The encoded pert token tensor produced by the checkpoint's encoder.

    Raises:
        AttributeError: If the model does not expose a ``pert_encoder`` attribute.
    """
    encoder = getattr(state_model, "pert_encoder", None)
    if encoder is None:
        raise AttributeError("state_model has no pert_encoder for distillation")
    return encoder(vec.float())


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
