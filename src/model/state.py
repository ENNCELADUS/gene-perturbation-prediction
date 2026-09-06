"""model / state."""

from __future__ import annotations
from typing import Any
import torch
from torch import nn


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


class StateForwardAdapter(nn.Module):
    """Thin wrapper around an ArcInstitute STATE transition model."""

    def __init__(self, state_model: nn.Module) -> None:
        super().__init__()
        self.state_model = state_model

    def forward(
        self,
        control_cells: torch.Tensor,
        perturbation: torch.Tensor,
        gene: str,
        batch_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one unpadded condition through STATE."""
        return self._forward_state(
            control_cells,
            perturbation,
            gene,
            batch_indices,
            padded=False,
        )

    def forward_chunks(
        self,
        control_chunks: tuple[torch.Tensor, ...],
        perturbation: torch.Tensor,
        gene: str,
        batch_index_chunks: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Run equal-size condition chunks as independent STATE sequences."""
        return self.forward_condition_chunks(
            control_chunks,
            tuple(perturbation for _ in control_chunks),
            tuple(gene for _ in control_chunks),
            batch_index_chunks,
        )

    def forward_condition_chunks(
        self,
        control_chunks: tuple[torch.Tensor, ...],
        perturbations: tuple[torch.Tensor, ...],
        genes: tuple[str, ...],
        batch_index_chunks: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Run multiple gene conditions in one padded STATE forward pass."""
        if len(control_chunks) != len(batch_index_chunks):
            raise ValueError("control and batch-index chunks must have equal length")
        if len(control_chunks) != len(perturbations) or len(control_chunks) != len(
            genes
        ):
            raise ValueError(
                "control, perturbation, gene, and batch-index chunks must align"
            )
        if not control_chunks:
            raise ValueError("at least one STATE condition chunk is required")
        chunk_sizes = tuple(int(chunk.shape[0]) for chunk in control_chunks)
        sentence_len = getattr(self.state_model, "cell_sentence_len", None)
        if len(set(chunk_sizes)) != 1 or (
            sentence_len is not None and chunk_sizes[0] != sentence_len
        ):
            raise ValueError(
                "STATE chunks must all equal the configured cell_sentence_len"
            )
        control_cells = torch.cat(control_chunks, dim=0)
        batch_indices = _concat_optional_batch_indices(
            batch_index_chunks,
            chunk_sizes,
            control_cells.device,
        )
        perturbation_cells = torch.cat(
            tuple(
                perturbation.unsqueeze(0).expand(size, -1)
                for perturbation, size in zip(perturbations, chunk_sizes, strict=True)
            ),
            dim=0,
        )
        gene_cells = [
            gene
            for gene, size in zip(genes, chunk_sizes, strict=True)
            for _ in range(size)
        ]
        output = self._forward_state_prepared(
            control_cells,
            perturbation_cells,
            gene_cells,
            batch_indices,
            padded=True,
        )
        return tuple(output.split(chunk_sizes, dim=0))

    def _forward_state(
        self,
        control_cells: torch.Tensor,
        perturbation: torch.Tensor,
        gene: str,
        batch_indices: torch.Tensor | None,
        *,
        padded: bool,
    ) -> torch.Tensor:
        perturbation_cells = perturbation.unsqueeze(0).expand(
            control_cells.shape[0], -1
        )
        return self._forward_state_prepared(
            control_cells,
            perturbation_cells,
            [gene] * int(control_cells.shape[0]),
            batch_indices,
            padded=padded,
        )

    def _forward_state_prepared(
        self,
        control_cells: torch.Tensor,
        perturbation_cells: torch.Tensor,
        gene_cells: list[str],
        batch_indices: torch.Tensor | None,
        *,
        padded: bool,
    ) -> torch.Tensor:
        batch: dict[str, Any] = {
            "ctrl_cell_emb": control_cells,
            "pert_emb": perturbation_cells,
            "pert_name": gene_cells,
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
                output = self.state_model.predict_step(
                    batch,
                    batch_idx=0,
                    padded=padded,
                )
            except TypeError:
                output = self.state_model.predict_step(batch, 0)
        else:
            try:
                output = self.state_model(batch, padded=padded)
            except TypeError:
                output = self.state_model(batch)
        if isinstance(output, dict):
            output = output["preds"]
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 3 and output.shape[0] == 1:
            output = output.squeeze(0)
        if output.dim() > 2:
            output = output.reshape(-1, output.shape[-1])
        return output


def _concat_optional_batch_indices(
    batch_index_chunks: tuple[torch.Tensor | None, ...],
    chunk_sizes: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor | None:
    if all(batch_indices is None for batch_indices in batch_index_chunks):
        return None
    return torch.cat(
        tuple(
            batch_indices.to(device)
            if batch_indices is not None
            else torch.zeros(size, dtype=torch.long, device=device)
            for batch_indices, size in zip(
                batch_index_chunks,
                chunk_sizes,
                strict=True,
            )
        )
    )


class ForwardOnlyStateModel(nn.Module):
    """ST + perturbation adapter only -- no response encoder/pooler/head.

    A deliberately small ``nn.Module`` (not ``AivcModel``), holding the two
    submodules Phase D needs under the same attribute names ``AivcModel``
    uses, so its ``state_dict()`` keys are a strict subset of Phase C's
    saved checkpoint and load straight out of it.
    """

    def __init__(
        self, state_adapter: StateForwardAdapter, perturbations: nn.Module
    ) -> None:
        super().__init__()
        self.state_adapter = state_adapter
        self.perturbations = perturbations

    def forward(
        self,
        control_chunks: tuple[torch.Tensor, ...],
        gene: str | tuple[str, ...],
        batch_index_chunks: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor, ...]:
        """The training entry point, delegating to
        :meth:`predict_response_chunks`.

        Training must call the module, not the method: under DDP,
        ``accelerator.prepare`` wraps this in ``DistributedDataParallel``,
        which proxies only ``forward``. Reaching for
        ``predict_response_chunks`` on the wrapper raises ``AttributeError``,
        and unwrapping to dodge that would skip the gradient all-reduce
        entirely -- a silently un-synchronized multi-rank run. Going through
        ``forward`` also puts the call inside Accelerate's autocast context,
        so the ``--mixed_precision`` the launcher passes is actually applied.
        """
        if isinstance(gene, str):
            return self.predict_response_chunks(
                control_chunks, gene, batch_index_chunks
            )
        if len(control_chunks) != len(gene):
            raise ValueError("one gene is required per STATE condition chunk")
        if hasattr(self.perturbations, "forward_many"):
            perturbation_batch = self.perturbations.forward_many(gene)
            perturbations = tuple(perturbation_batch.unbind(0))
        else:
            perturbations = tuple(self.perturbations(name) for name in gene)
        return self.state_adapter.forward_condition_chunks(
            control_chunks,
            perturbations,
            gene,
            batch_index_chunks,
        )

    def predict_response_chunks(
        self,
        control_chunks: tuple[torch.Tensor, ...],
        gene: str,
        batch_index_chunks: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Forward ST for one gene (``AivcModel.predict_response_chunks`` minus
        the response-encoder step; Phase D's own head takes over from the raw
        expression-space output).
        """
        perturbation = self.perturbations(gene)
        return self.state_adapter.forward_chunks(
            control_chunks, perturbation, gene, batch_index_chunks
        )
