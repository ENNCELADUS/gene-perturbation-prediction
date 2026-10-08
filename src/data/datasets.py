"""GeneEffect rows and anchor response conditions over prepared inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.batches import DependencyBatch, OnlineConditionBatch, ResponseBatch
from src.data.context_pca import pooled_context
from src.data.prepared import PreparedInputs


def split_lines(inputs: PreparedInputs, split: str) -> tuple[str, ...]:
    """Lines scored for a split; training uses the supervised training lines."""
    if split == "train":
        return inputs.split.supervised_train
    return getattr(inputs.split, split)


class DependencyDataset(Dataset[int]):
    """Labelled (line, gene) GeneEffect rows of one split, with a collator.

    ``lines`` restricts the rows to a subset of the split's lines. Every fixed
    quantity is placed on ``device`` once: each line's basal HVG cells, compressed
    Tx1 context (the pooled ``z_c`` through ``inputs.context_pca``) and q_sc
    table, the ESM2 table and every row's targets and indices.
    The collator only gathers by row index, so a batch is born on ``device``.
    Each row also carries its prior offset (``prior``), zero without a prior.
    """

    def __init__(
        self,
        inputs: PreparedInputs,
        split: str,
        *,
        device: torch.device | str = "cpu",
        lines: Sequence[str] | None = None,
    ) -> None:
        allowed = split_lines(inputs, split)
        lines = allowed if lines is None else tuple(lines)
        if not set(lines) <= set(allowed):
            raise ValueError(
                f"lines outside the {split} split: {set(lines) - set(allowed)}"
            )
        self.inputs = inputs
        self.split = split
        self.lines = lines
        self.device = torch.device(device)
        self.rows = inputs.labels.loc[inputs.labels["model_id"].isin(set(lines))]
        self.rows = self.rows.reset_index(drop=True)
        self.model_ids = self.rows["model_id"].astype(str).tolist()
        self.genes = self.rows["gene_symbol"].astype(str).tolist()
        hvg_index = {gene: index for index, gene in enumerate(inputs.hvg_order)}
        self.hvg_indices = [hvg_index.get(gene) for gene in self.genes]
        present = sorted(set(self.model_ids))
        line_index = {m: index for index, m in enumerate(present)}
        gene_index = {gene: index for index, gene in enumerate(inputs.genes)}
        esm2_index = {gene: index for index, gene in enumerate(inputs.esm2_symbols)}

        def on_device(values, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(values, dtype=dtype).to(self.device)

        self._basal = {
            m: torch.from_numpy(np.asarray(inputs.lines[m].basal_hvg)).to(self.device)
            for m in present
        }
        contexts = np.stack(
            [pooled_context(inputs.lines[m].controls_tx1) for m in present]
        )
        self._contexts = on_device(
            inputs.context_pca.transform(contexts), torch.float32
        )
        self._q_sc = on_device(
            np.stack(
                [
                    np.nan_to_num(inputs.lines[m].q_sc.values, nan=0.0).astype(
                        np.float32
                    )
                    for m in present
                ]
            ),
            torch.float32,
        )
        self._q_sc_available = on_device(
            np.stack([inputs.lines[m].q_sc.available for m in present]).astype(bool),
            torch.bool,
        )
        self._esm2 = on_device(
            np.asarray(inputs.esm2_vectors, dtype=np.float32), torch.float32
        )
        self._row_line = on_device([line_index[m] for m in self.model_ids], torch.long)
        self._row_gene_positions = np.array(
            [gene_index[g] for g in self.genes], dtype=np.int64
        )
        self._row_gene = on_device(self._row_gene_positions, torch.long)
        self._row_esm2 = on_device([esm2_index[g] for g in self.genes], torch.long)
        self._row_in_hvg_panel = on_device(
            [index is not None for index in self.hvg_indices], torch.bool
        )
        self.residual = on_device(self.rows["residual"].to_numpy(), torch.float32)
        self.gene_effect = on_device(self.rows["gene_effect"].to_numpy(), torch.float32)
        self.gene_mean = on_device(
            inputs.train_gene_means.loc[self.genes].to_numpy(), torch.float32
        )
        self.residual_scale = on_device(
            inputs.residual_scale.loc[self.genes].to_numpy(), torch.float32
        )
        self.selective = on_device(
            [gene in inputs.selective_genes for gene in self.genes], torch.bool
        )
        if inputs.prior is None:
            prior = np.zeros(len(self.rows))
        else:
            table = inputs.prior.values
            line_rows = table.index.get_indexer(self.model_ids)
            gene_columns = table.columns.get_indexer(self.genes)
            if (line_rows < 0).any() or (gene_columns < 0).any():
                raise ValueError("the prior export lacks some of the split's rows")
            prior = (
                table.to_numpy()[line_rows, gene_columns]
                * inputs.residual_scale.loc[self.genes].to_numpy()
            )
        self.prior = on_device(prior, torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> int:
        return int(index)

    def rows_by_gene(self) -> list[np.ndarray]:
        """Row positions of each gene, in ``inputs.genes`` order; empty when absent."""
        order = np.argsort(self._row_gene_positions, kind="stable")
        counts = np.bincount(self._row_gene_positions, minlength=len(self.inputs.genes))
        return np.split(order, np.cumsum(counts)[:-1])

    def collate(self, indices: Sequence[int]) -> DependencyBatch:
        positions = [int(index) for index in indices]
        rows = torch.as_tensor(positions, dtype=torch.long).to(self.device)
        line, gene = self._row_line[rows], self._row_gene[rows]
        model_ids = tuple(self.model_ids[i] for i in positions)
        q_available = self._q_sc_available[line, gene]
        in_hvg_panel = self._row_in_hvg_panel[rows]
        conditions = OnlineConditionBatch(
            basal_hvg=tuple(self._basal[m] for m in model_ids),
            genes=tuple(self.genes[i] for i in positions),
            gene_index=gene,
            model_ids=model_ids,
            q_sc=self._q_sc[line, gene],
            e_g=self._esm2[self._row_esm2[rows]],
            z_c=self._contexts[line],
            q_sc_mask=q_available,
            gene_in_hvg_panel=in_hvg_panel,
            own_gene_hvg_indices=tuple(self.hvg_indices[i] for i in positions),
            own_gene_shift_available=in_hvg_panel & q_available,
        )
        return DependencyBatch(
            conditions=conditions,
            residual=self.residual[rows],
            gene_effect=self.gene_effect[rows],
            gene_mean=self.gene_mean[rows],
            residual_scale=self.residual_scale[rows],
            selective=self.selective[rows],
            prior=self.prior[rows],
        )


class ResponseDataset(Dataset[int]):
    """Every prepared perturbation condition of the response anchors."""

    def __init__(
        self, inputs: PreparedInputs, *, device: torch.device | str = "cpu"
    ) -> None:
        self.inputs = inputs
        self.cache = inputs.response_targets
        self.keys = tuple(self.cache.keys)
        self.device = torch.device(device)
        self._basal = {
            m: torch.from_numpy(np.asarray(inputs.lines[m].basal_hvg)).to(self.device)
            for m in sorted({model_id for model_id, _ in self.keys})
        }

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> int:
        return int(index)

    def collate(self, indices: Sequence[int]) -> ResponseBatch:
        """One host-to-device copy of every observed bag, split back per condition."""
        keys = [self.keys[index] for index in indices]
        bags = [self.cache.target_bag(index) for index in indices]
        observed = torch.from_numpy(np.concatenate(bags).astype(np.float32, copy=False))
        return ResponseBatch(
            model_ids=tuple(model_id for model_id, _ in keys),
            genes=tuple(gene for _, gene in keys),
            control_hvg=tuple(self._basal[model_id] for model_id, _ in keys),
            observed_hvg=observed.to(self.device).split([len(bag) for bag in bags]),
        )
