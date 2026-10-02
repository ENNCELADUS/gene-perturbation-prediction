"""STATE on its own HVG basal path, with ESM2 perturbation tokens.

STATE's control-cell input is log-space basal HVG expression (2000 genes, STATE's
own order) through its released 2000->328 basal encoder. Tx1 embeddings never
reach STATE; they feed only the GeneEffect head's pooled context.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


class LinearMockStateModel(nn.Module):
    """Small STATE-shaped model for tests without arc-state or a checkpoint.

    It reads the same batch keys as STATE (``ctrl_cell_emb``, ``pert_emb``) and
    returns one output row per input cell.
    """

    def __init__(
        self, input_dim: int, output_dim: int, pert_dim: int, cell_set_len: int = 64
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.pert_dim = int(pert_dim)
        self.cell_sentence_len = int(cell_set_len)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim + self.pert_dim, 32),
            nn.GELU(),
            nn.Linear(32, self.output_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor], padded: bool = True):
        del padded
        return self.net(torch.cat([batch["ctrl_cell_emb"], batch["pert_emb"]], dim=1))


@contextmanager
def _quiet():
    """arc-state prints model summaries while constructing; keep logs readable."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def released_hparams(checkpoint_path: Path, *, cell_set_len: int) -> dict[str, Any]:
    """The released checkpoint's own constructor arguments, sentence length set."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    hparams = deepcopy(dict(checkpoint["hyper_parameters"]))
    hparams["cell_set_len"] = int(cell_set_len)
    return hparams


def build_state(hparams: Mapping[str, Any]) -> nn.Module:
    """Construct the pinned arc-state transition model from saved arguments."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    # STATE mutates nested transformer kwargs while constructing; pass a copy.
    with _quiet():
        return StateTransitionPerturbationModel(**deepcopy(dict(hparams)))


def load_released_state(checkpoint_path: Path, *, cell_set_len: int) -> nn.Module:
    """STATE built from its released arguments with every released weight loaded.

    Raises when any checkpoint key is missing from the model, unexpected, or
    shape-mismatched: STATE keeps its own 2000-gene basal encoder, so nothing is
    re-initialised.
    """
    from src.model.initialization import warm_start_state_dict

    state = build_state(released_hparams(checkpoint_path, cell_set_len=cell_set_len))
    report = warm_start_state_dict(state, checkpoint_path)
    if report.missing_keys or report.unexpected_keys or report.shape_skipped_keys:
        raise ValueError(
            f"released STATE checkpoint {checkpoint_path} does not load completely: "
            f"missing={report.missing_keys}, unexpected={report.unexpected_keys}, "
            f"shape-mismatched={report.shape_skipped_keys}"
        )
    return state


class StateResponse(nn.Module):
    """STATE driven by ESM2 perturbation tokens on log-space basal HVG cells."""

    def __init__(self, state: nn.Module, perturbations: nn.Module) -> None:
        super().__init__()
        self.state = state
        self.perturbations = perturbations

    @property
    def cell_set_len(self) -> int:
        return int(self.state.cell_sentence_len)

    def forward(
        self, basal_chunks: tuple[torch.Tensor, ...], genes: tuple[str, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Each ``[cell_set_len, 2000]`` basal chunk with its gene -> predicted cells.

        All chunks run as independent STATE sentences in one padded call; every
        cell uses batch index 0. Outputs are FP32.
        """
        size = int(basal_chunks[0].shape[0])
        basal = torch.cat(basal_chunks, dim=0)
        tokens = self.perturbations.forward_many(tuple(genes))
        output = self.state(
            {
                "ctrl_cell_emb": basal,
                "pert_emb": tokens.repeat_interleave(size, dim=0),
                "batch": torch.zeros(len(basal), dtype=torch.long, device=basal.device),
            },
            padded=True,
        )
        if isinstance(output, tuple):
            output = output[0]
        return tuple(output.float().split(size, dim=0))
