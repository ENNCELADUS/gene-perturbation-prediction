"""data / gene bags."""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from src.data.gene_splits import GeneAccessRecorder


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
