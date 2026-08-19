"""Canonical home for STATE/gene-bag primitives shared across the Tx1 tree.

These symbols were lifted out of the exp05 ``model.py``/``prepare.py`` before
those modules were deleted at ``873c99c``. They are the single definition,
used by ``tx1_embed_cache.py``, ``tx1_response_data.py``,
``tx1_predicted_response.py`` and the CPU tests; nothing redefines them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import pickle
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch import nn

from aivc_model.gene_embeddings import Esm2EmbeddingTable, PertAdapter
from aivc_model.gene_splits import GeneAccessRecorder


def sha256_strings(values: np.ndarray) -> str:
    """Hash an ordered string array without ambiguous concatenation."""
    digest = hashlib.sha256()
    for value in np.asarray(values).astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


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
        """Run one unpadded condition for frozen-feature extraction."""
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
        self._last_token_features = None
        if hasattr(self.state_model, "_token_features"):
            try:
                setattr(self.state_model, "_token_features", None)
            except AttributeError:
                pass
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
        token_features = getattr(self.state_model, "_token_features", None)
        if (
            not padded
            and isinstance(token_features, torch.Tensor)
            and token_features.dim() == 3
            and token_features.shape[0] != 1
        ):
            msg = "STATE token features must have batch dimension 1 when unpadded"
            raise ValueError(msg)
        self._last_token_features = _flatten_state_output(token_features)
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


class Esm2PerturbationAdapter(nn.Module):
    """Map fixed per-gene ESM-2 vectors to STATE perturbation vectors."""

    def __init__(
        self,
        genes: list[str],
        table: Esm2EmbeddingTable,
        adapter_hidden: int,
        pert_dim: int,
    ) -> None:
        super().__init__()
        self.genes = [str(gene).upper() for gene in genes]
        missing = [gene for gene in self.genes if gene not in table.vectors_by_symbol]
        if missing:
            raise ValueError(f"Unresolved ESM-2 genes: {missing[:10]}")
        matrix = np.vstack([table.vectors_by_symbol[gene] for gene in self.genes])
        self._gene_to_index = {gene: index for index, gene in enumerate(self.genes)}
        self.register_buffer("esm_matrix", torch.as_tensor(matrix, dtype=torch.float32))
        self.adapter = PertAdapter(table.dim, int(adapter_hidden), int(pert_dim))

    def forward(self, gene: str) -> torch.Tensor:
        return self.forward_many([gene])[0]

    def forward_many(self, genes: list[str] | tuple[str, ...]) -> torch.Tensor:
        """Map multiple genes through one adapter call."""
        indices = torch.as_tensor(
            [self._gene_to_index[str(gene).upper()] for gene in genes],
            dtype=torch.long,
            device=self.esm_matrix.device,
        )
        return self.adapter(self.esm_matrix.index_select(0, indices))

    def has_embedding(self, gene: str) -> bool:
        """Return whether the adapter contains an ESM-2 vector for ``gene``."""
        return str(gene).upper() in self._gene_to_index

    def has_known_vector(self, gene: str) -> bool:
        """Compatibility alias for prediction artifact metadata."""
        return self.has_embedding(gene)


@dataclass(frozen=True)
class GeneBags:
    genes: np.ndarray
    y: np.ndarray
    input_bags: tuple[np.ndarray, ...]
    latent_bags: tuple[np.ndarray, ...]
    control_input: np.ndarray
    control_latent: np.ndarray
    cell_type_bags: tuple[np.ndarray, ...] | None
    control_cell_type: np.ndarray | None
    batch_bags: tuple[np.ndarray, ...] | None
    control_batch: np.ndarray | None
    feature_names: np.ndarray | None
    metadata: pd.DataFrame
    input_dim: int
    latent_dim: int
    feature_fill_values: np.ndarray | None = None
    gene_outer_folds: np.ndarray | None = None
    access_recorder: GeneAccessRecorder | None = None
    target_bags: tuple[np.ndarray, ...] | None = None
    control_target: np.ndarray | None = None
    target_dim: int | None = None
    target_feature_names: np.ndarray | None = None
    target_fill_values: np.ndarray | None = None

    @property
    def effective_target_bags(self) -> tuple[np.ndarray, ...]:
        """Per-gene response-encoder-target bags.

        Falls back to ``input_bags`` (the identical object, not a copy) when no
        distinct target space is configured.
        """
        return self.input_bags if self.target_bags is None else self.target_bags

    @property
    def effective_control_target(self) -> np.ndarray:
        """Control-cell response-encoder-target matrix.

        Falls back to ``control_input`` (the identical object, not a copy) when
        no distinct target space is configured.
        """
        return (
            self.control_input if self.control_target is None else self.control_target
        )

    @property
    def effective_target_dim(self) -> int:
        """Response-encoder-target feature width, falling back to ``input_dim``."""
        return self.input_dim if self.target_dim is None else self.target_dim

    @property
    def effective_target_feature_names(self) -> np.ndarray | None:
        """Response-encoder-target feature names, falling back to ``feature_names``."""
        return (
            self.feature_names
            if self.target_feature_names is None
            else self.target_feature_names
        )

    @property
    def effective_target_fill_values(self) -> np.ndarray | None:
        """Response-encoder-target fill values (falls back to feature_fill_values)."""
        return (
            self.feature_fill_values
            if self.target_fill_values is None
            else self.target_fill_values
        )

    def for_genes(
        self,
        genes: tuple[str, ...],
        stage: str,
        *,
        checkpoint_frozen: bool = False,
    ) -> GeneBags:
        """Authorize, record, and return an ordered gene-only view."""
        requested = tuple(str(gene).upper() for gene in genes)
        if self.access_recorder is None:
            raise ValueError("GeneBags.for_genes requires a GeneAccessRecorder")
        if len(set(requested)) != len(requested):
            raise ValueError("gene view contains duplicate requested genes")
        index_by_gene = {
            str(gene).upper(): index for index, gene in enumerate(self.genes)
        }
        missing = [gene for gene in requested if gene not in index_by_gene]
        if missing:
            raise ValueError(f"gene view contains unknown genes: {missing[:5]}")
        self.access_recorder.record(
            stage,
            requested,
            checkpoint_frozen=checkpoint_frozen,
        )
        indices = [index_by_gene[gene] for gene in requested]
        return replace(
            self,
            genes=np.asarray([self.genes[index] for index in indices], dtype=object),
            y=np.asarray([self.y[index] for index in indices], dtype=np.float32),
            input_bags=tuple(self.input_bags[index] for index in indices),
            latent_bags=tuple(self.latent_bags[index] for index in indices),
            cell_type_bags=(
                tuple(self.cell_type_bags[index] for index in indices)
                if self.cell_type_bags is not None
                else None
            ),
            batch_bags=(
                tuple(self.batch_bags[index] for index in indices)
                if self.batch_bags is not None
                else None
            ),
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
            gene_outer_folds=(
                np.asarray([self.gene_outer_folds[index] for index in indices])
                if self.gene_outer_folds is not None
                else None
            ),
            target_bags=(
                tuple(self.target_bags[index] for index in indices)
                if self.target_bags is not None
                else None
            ),
        )

    def record_access(
        self,
        stage: str,
        *,
        checkpoint_frozen: bool = False,
    ) -> None:
        """Authorize and record an operation over every gene in this view."""
        if self.access_recorder is None:
            raise ValueError("audited gene access requires a GeneAccessRecorder")
        self.access_recorder.record(
            stage,
            tuple(str(gene) for gene in self.genes),
            checkpoint_frozen=checkpoint_frozen,
        )

    def for_prediction_genes(
        self,
        genes: tuple[str, ...],
        stage: str,
        *,
        checkpoint_frozen: bool = False,
        generation_targets: bool = False,
    ) -> GeneBags:
        """Return prediction inputs with optional target-only observed expression."""
        requested = tuple(str(gene).upper() for gene in genes)
        if self.access_recorder is None:
            raise ValueError(
                "GeneBags.for_prediction_genes requires a GeneAccessRecorder"
            )
        if len(set(requested)) != len(requested):
            raise ValueError("gene view contains duplicate requested genes")
        index_by_gene = {
            str(gene).upper(): index for index, gene in enumerate(self.genes)
        }
        missing = [gene for gene in requested if gene not in index_by_gene]
        if missing:
            raise ValueError(f"gene view contains unknown genes: {missing[:5]}")
        self.access_recorder.record(
            stage,
            requested,
            checkpoint_frozen=checkpoint_frozen,
        )
        indices = [index_by_gene[gene] for gene in requested]
        return GeneBags(
            genes=np.asarray([self.genes[index] for index in indices], dtype=object),
            y=np.asarray([self.y[index] for index in indices], dtype=np.float32),
            input_bags=(
                tuple(self.input_bags[index] for index in indices)
                if generation_targets
                else tuple(
                    np.empty((0, self.input_dim), dtype=np.float32) for _ in indices
                )
            ),
            latent_bags=tuple(
                np.empty((0, self.latent_dim), dtype=np.float32) for _ in indices
            ),
            control_input=self.control_input,
            control_latent=self.control_latent,
            cell_type_bags=(
                tuple(np.empty(0, dtype=object) for _ in indices)
                if self.cell_type_bags is not None
                else None
            ),
            control_cell_type=self.control_cell_type,
            batch_bags=(
                tuple(np.empty(0, dtype=object) for _ in indices)
                if self.batch_bags is not None
                else None
            ),
            control_batch=self.control_batch,
            feature_names=self.feature_names,
            feature_fill_values=self.feature_fill_values,
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            gene_outer_folds=(
                np.asarray([self.gene_outer_folds[index] for index in indices])
                if self.gene_outer_folds is not None
                else None
            ),
            access_recorder=self.access_recorder,
            target_bags=(
                (
                    tuple(self.target_bags[index] for index in indices)
                    if generation_targets
                    else tuple(
                        np.empty((0, self.effective_target_dim), dtype=np.float32)
                        for _ in indices
                    )
                )
                if self.target_bags is not None
                else None
            ),
            control_target=self.control_target,
            target_dim=self.target_dim,
            target_feature_names=self.target_feature_names,
            target_fill_values=self.target_fill_values,
        )


def encode_batch_labels(
    labels: np.ndarray | None,
    lookup: dict[str, int],
    fallback_index: int = 0,
) -> np.ndarray | None:
    """Encode batch labels for STATE checkpoints."""
    if labels is None:
        return None
    return np.asarray(
        [lookup.get(str(label), int(fallback_index)) for label in labels],
        dtype=np.int64,
    )


def resolve_state_gene_order(
    adata: ad.AnnData,
    model_dir: Path,
    symbol_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve expression columns to the exact STATE checkpoint gene order."""
    with (model_dir / "var_dims.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    checkpoint_names = np.asarray(payload["gene_names"], dtype=object).astype(str)
    source_names = adata.var[symbol_col].astype(str).to_numpy()
    positions: dict[str, int] = {}
    duplicates: set[str] = set()
    for index, symbol in enumerate(source_names):
        if symbol in positions:
            duplicates.add(symbol)
        else:
            positions[symbol] = index
    duplicate_matches = sorted(set(checkpoint_names).intersection(duplicates))
    missing = [name for name in checkpoint_names if name not in positions]
    if missing or duplicate_matches:
        matched = len(checkpoint_names) - len(missing) - len(duplicate_matches)
        raise ValueError(
            f"STATE expression alignment matched {matched}/{len(checkpoint_names)}; "
            f"missing={missing[:10]}, duplicate_matches={duplicate_matches[:10]}"
        )
    indices = np.asarray([positions[name] for name in checkpoint_names], dtype=np.int64)
    return indices, checkpoint_names.astype(object)
