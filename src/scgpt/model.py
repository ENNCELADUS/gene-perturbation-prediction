"""scGPT backbone and gene-scoring head."""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from src.scgpt.architecture import (
    LatentResponseCycleHead,
    SlotSetDecoder,
    SparseGraphMessagePassing,
    compute_cardinality_loss,
)
from src.scgpt.backbone import ScGPTBackbone


class GeneScoreModel(nn.Module):
    """scGPT cell embedding plus trainable gene-scoring head."""

    def __init__(
        self,
        n_genes: int,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        args_path: str | Path,
        score_gene_ids: Sequence[int] | torch.Tensor | None = None,
        freeze_encoder: bool = True,
        freeze_layers_up_to: int = 10,
        score_mode: str = "dot",
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
        gene_graph_edge_index: torch.Tensor | None = None,
        gene_graph_edge_weight: torch.Tensor | None = None,
        use_graph_encoder: bool = False,
        graph_message_layers: int = 2,
        use_contrast_encoder: bool = True,
        use_slots: bool = False,
        n_target_slots: int = 4,
        slot_aggregation: str = "logsumexp",
        use_cardinality_head: bool = False,
        max_cardinality: int = 4,
        cardinality_loss_weight: float = 0.1,
        use_cycle_loss: bool = False,
        cycle_loss_weight: float = 0.1,
        alignment_heads: int = 4,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.score_mode = score_mode
        self.use_graph_encoder = use_graph_encoder
        self.graph_message_layers = graph_message_layers
        self.use_contrast_encoder = use_contrast_encoder
        self.use_slots = use_slots
        self.use_cardinality_head = use_cardinality_head
        self.max_cardinality = max_cardinality
        self.cardinality_loss_weight = cardinality_loss_weight
        self.use_cycle_loss = use_cycle_loss
        self.cycle_loss_weight = cycle_loss_weight
        self.backbone = ScGPTBackbone(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            args_path=args_path,
            freeze_encoder=freeze_encoder,
            freeze_layers_up_to=freeze_layers_up_to,
            use_fast_transformer=use_fast_transformer,
            fast_transformer_backend=fast_transformer_backend,
            device=device,
        )
        emb_dim = int(self.backbone.args["embsize"])
        if score_mode == "dot":
            self.head = nn.Sequential(
                nn.Linear(emb_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, emb_dim),
            )
        elif score_mode == "mlp":
            self.head = nn.Sequential(
                nn.Linear(emb_dim * 2, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 1),
            )
        elif score_mode == "bilinear":
            self.phenotype_head = nn.Sequential(
                nn.Linear(emb_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, emb_dim),
            )
            self.gene_head = nn.Sequential(
                nn.Linear(emb_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, emb_dim),
            )
        elif score_mode == "causal_set":
            self.perturbation_gene_embedding = nn.Parameter(
                torch.empty(n_genes, emb_dim)
            )
            nn.init.normal_(self.perturbation_gene_embedding, mean=0.0, std=0.02)
            self.gene_input_projection = nn.Sequential(
                nn.Linear(emb_dim * 2, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, emb_dim),
            )
            self.graph_layers = nn.ModuleList(
                [
                    SparseGraphMessagePassing(emb_dim, head_dropout)
                    for _ in range(max(0, graph_message_layers))
                ]
            )
            attention_heads = alignment_heads if emb_dim % alignment_heads == 0 else 1
            self.phenotype_query = nn.Parameter(torch.empty(1, 1, emb_dim))
            nn.init.normal_(self.phenotype_query, mean=0.0, std=0.02)
            self.phenotype_attention = nn.MultiheadAttention(
                embed_dim=emb_dim,
                num_heads=attention_heads,
                dropout=head_dropout,
                batch_first=True,
            )
            self.phenotype_norm = nn.LayerNorm(emb_dim)
            self.phenotype_head = nn.Sequential(
                nn.Linear(emb_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, emb_dim),
            )
            self.single_query_head = nn.Sequential(
                nn.Linear(emb_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, emb_dim),
            )
            self.slot_decoder = SlotSetDecoder(
                embedding_dim=emb_dim,
                hidden_dim=head_hidden_dim,
                n_slots=n_target_slots,
                dropout=head_dropout,
                aggregation=slot_aggregation,
            )
            self.cardinality_head = nn.Linear(emb_dim, max_cardinality + 1)
            self.cycle_head = LatentResponseCycleHead(
                embedding_dim=emb_dim,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )
        else:
            raise ValueError(
                "score_mode must be 'dot', 'mlp', 'bilinear', or 'causal_set'"
            )
        self.register_buffer(
            "score_gene_ids", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "gene_graph_edge_index",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "gene_graph_edge_weight",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        if score_gene_ids is not None:
            self.set_score_gene_ids(score_gene_ids)
        if gene_graph_edge_index is not None and gene_graph_edge_weight is not None:
            self.set_gene_graph(gene_graph_edge_index, gene_graph_edge_weight)

    def set_score_gene_ids(self, gene_ids: Sequence[int] | torch.Tensor) -> None:
        """Set the token ids used for the output score order."""
        gene_ids_tensor = torch.as_tensor(gene_ids, dtype=torch.long)
        if gene_ids_tensor.ndim != 1 or gene_ids_tensor.numel() != self.n_genes:
            raise ValueError("score_gene_ids must be 1D with length n_genes")
        self.score_gene_ids = gene_ids_tensor.to(next(self.parameters()).device)

    def set_gene_graph(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> None:
        """Set sparse gene graph buffers aligned to score_gene_ids."""
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")
        if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.shape[1]:
            raise ValueError("edge_weight must be 1D with one value per edge")
        device = next(self.parameters()).device
        self.gene_graph_edge_index = edge_index.to(device=device, dtype=torch.long)
        self.gene_graph_edge_weight = edge_weight.to(device=device, dtype=torch.float32)

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        control_gene_ids: torch.Tensor | None = None,
        control_values: torch.Tensor | None = None,
        control_padding_mask: torch.Tensor | None = None,
        control_counts: int | None = None,
        control_chunk_size: int | None = None,
        control_no_grad: bool = True,
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if self.score_mode == "causal_set":
            return self._forward_causal_set(
                gene_ids=gene_ids,
                values=values,
                padding_mask=padding_mask,
                control_gene_ids=control_gene_ids,
                control_values=control_values,
                control_padding_mask=control_padding_mask,
                control_counts=control_counts,
                control_chunk_size=control_chunk_size,
                control_no_grad=control_no_grad,
                targets=targets,
                return_aux=return_aux,
            )
        return self._forward_simple(
            gene_ids=gene_ids,
            values=values,
            padding_mask=padding_mask,
            control_gene_ids=control_gene_ids,
            control_values=control_values,
            control_padding_mask=control_padding_mask,
            control_counts=control_counts,
            control_chunk_size=control_chunk_size,
            control_no_grad=control_no_grad,
        )

    def _forward_simple(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        control_gene_ids: torch.Tensor | None,
        control_values: torch.Tensor | None,
        control_padding_mask: torch.Tensor | None,
        control_counts: int | None,
        control_chunk_size: int | None,
        control_no_grad: bool,
    ) -> torch.Tensor:
        outputs = self.backbone.model(
            src=gene_ids,
            values=values,
            src_key_padding_mask=padding_mask,
            batch_labels=None,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False,
            do_sample=False,
        )
        cell_emb = outputs["cell_emb"]
        if control_gene_ids is not None:
            cell_emb = cell_emb - self._control_mean(
                control_gene_ids,
                control_values,
                control_padding_mask,
                control_counts,
                control_chunk_size,
                control_no_grad,
                batch_size=cell_emb.size(0),
            )
        gene_emb = self._raw_gene_embeddings(cell_emb.device)
        if self.score_mode == "dot":
            return self.head(cell_emb) @ gene_emb.transpose(0, 1)
        if self.score_mode == "bilinear":
            phenotype = self.phenotype_head(cell_emb)
            gene_keys = self.gene_head(gene_emb)
            return (phenotype @ gene_keys.transpose(0, 1)) / math.sqrt(
                gene_keys.size(1)
            )
        cell_rep = cell_emb.unsqueeze(1).expand(-1, gene_emb.size(0), -1)
        gene_rep = gene_emb.unsqueeze(0).expand(cell_emb.size(0), -1, -1)
        return self.head(torch.cat([cell_rep, gene_rep], dim=-1)).squeeze(-1)

    def _forward_causal_set(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        control_gene_ids: torch.Tensor | None,
        control_values: torch.Tensor | None,
        control_padding_mask: torch.Tensor | None,
        control_counts: int | None,
        control_chunk_size: int | None,
        control_no_grad: bool,
        targets: torch.Tensor | None,
        return_aux: bool,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        token_output = self._encode_inputs(gene_ids, values, padding_mask)
        cell_emb = self.backbone.model._get_cell_emb_from_layer(token_output, values)
        control_summary = None
        if control_gene_ids is not None:
            control_summary = self._control_outputs(
                control_gene_ids=control_gene_ids,
                control_values=control_values,
                control_padding_mask=control_padding_mask,
                control_counts=control_counts,
                control_chunk_size=control_chunk_size,
                control_no_grad=control_no_grad,
                batch_size=cell_emb.size(0),
                return_tokens=self.use_contrast_encoder,
            )
        phenotype = self._phenotype_embedding(
            cell_emb=cell_emb,
            token_output=token_output,
            values=values,
            padding_mask=padding_mask,
            control_summary=control_summary,
        )
        gene_emb = self._causal_gene_embeddings(cell_emb.device)
        if self.use_slots:
            logits = self.slot_decoder(phenotype, gene_emb)
        else:
            query = self.single_query_head(phenotype)
            logits = (query @ gene_emb.transpose(0, 1)) / math.sqrt(gene_emb.size(1))

        output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {"logits": logits}
        if self.use_cardinality_head:
            cardinality_logits = self.cardinality_head(phenotype)
            output["cardinality_logits"] = cardinality_logits
        if return_aux:
            output["auxiliary_losses"] = self._auxiliary_losses(
                output=output,
                logits=logits,
                gene_embeddings=gene_emb,
                cell_embedding=cell_emb,
                control_summary=control_summary,
                targets=targets,
            )
        return output

    def _encode_inputs(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone.model._encode(
            gene_ids,
            values,
            padding_mask,
            batch_labels=None,
        )

    def _raw_gene_embeddings(self, device: torch.device) -> torch.Tensor:
        return self.backbone.model.encoder(self.score_gene_ids.to(device))

    def _causal_gene_embeddings(self, device: torch.device) -> torch.Tensor:
        base_gene_emb = self._raw_gene_embeddings(device)
        perturbation_emb = self.perturbation_gene_embedding.to(device)
        gene_features = self.gene_input_projection(
            torch.cat([base_gene_emb, perturbation_emb], dim=1)
        )
        if not self.use_graph_encoder:
            return gene_features
        for graph_layer in self.graph_layers:
            gene_features = graph_layer(
                gene_features,
                edge_index=self.gene_graph_edge_index,
                edge_weight=self.gene_graph_edge_weight,
            )
        return gene_features

    def _phenotype_embedding(
        self,
        cell_emb: torch.Tensor,
        token_output: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        control_summary: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        control_cell = (
            control_summary["cell_emb"]
            if control_summary is not None
            else torch.zeros_like(cell_emb)
        )
        delta_cell = cell_emb - control_cell
        if not self.use_contrast_encoder:
            return self.phenotype_head(delta_cell)
        control_tokens = (
            control_summary["token_output"]
            if control_summary is not None and "token_output" in control_summary
            else torch.zeros_like(token_output)
        )
        delta_tokens = token_output - control_tokens
        query = self.phenotype_query.expand(cell_emb.size(0), -1, -1)
        query = query + delta_cell.unsqueeze(1)
        attended, _ = self.phenotype_attention(
            query=query,
            key=delta_tokens,
            value=delta_tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        phenotype = self.phenotype_norm(attended.squeeze(1) + delta_cell)
        return self.phenotype_head(phenotype)

    def _auxiliary_losses(
        self,
        output: Mapping[str, torch.Tensor | dict[str, torch.Tensor]],
        logits: torch.Tensor,
        gene_embeddings: torch.Tensor,
        cell_embedding: torch.Tensor,
        control_summary: Mapping[str, torch.Tensor] | None,
        targets: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        cardinality_logits = output.get("cardinality_logits")
        if self.use_cardinality_head and targets is not None:
            if not isinstance(cardinality_logits, torch.Tensor):
                raise ValueError("cardinality_logits missing from causal_set output")
            losses["cardinality"] = (
                self.cardinality_loss_weight
                * compute_cardinality_loss(
                    cardinality_logits=cardinality_logits,
                    targets=targets,
                    max_cardinality=self.max_cardinality,
                )
            )
        if self.use_cycle_loss and control_summary is not None:
            target_set_embedding = self._soft_target_embedding(logits, gene_embeddings)
            losses["cycle"] = self.cycle_loss_weight * self.cycle_head.loss(
                control_embedding=control_summary["cell_emb"],
                target_set_embedding=target_set_embedding,
                perturbed_embedding=cell_embedding,
            )
        return losses

    def _soft_target_embedding(
        self,
        logits: torch.Tensor,
        gene_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.sigmoid(logits)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return weights @ gene_embeddings

    def _control_outputs(
        self,
        control_gene_ids: torch.Tensor,
        control_values: torch.Tensor | None,
        control_padding_mask: torch.Tensor | None,
        control_counts: int | None,
        control_chunk_size: int | None,
        control_no_grad: bool,
        batch_size: int,
        return_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        if control_values is None or control_padding_mask is None:
            raise ValueError("control_values and control_padding_mask are required")
        chunk_size = control_chunk_size or control_gene_ids.size(0)
        control_cells = []
        control_tokens = []
        context = torch.no_grad() if control_no_grad else nullcontext()
        with context:
            for start in range(0, control_gene_ids.size(0), chunk_size):
                chunk_gene_ids = control_gene_ids[start : start + chunk_size]
                chunk_values = control_values[start : start + chunk_size]
                chunk_padding = control_padding_mask[start : start + chunk_size]
                token_output = self._encode_inputs(
                    chunk_gene_ids,
                    chunk_values,
                    chunk_padding,
                )
                control_cells.append(
                    self.backbone.model._get_cell_emb_from_layer(
                        token_output,
                        chunk_values,
                    )
                )
                if return_tokens:
                    control_tokens.append(token_output)
        control_cell = torch.cat(control_cells, dim=0)
        controls_per_sample = control_counts or control_cell.size(0) // batch_size
        output = {
            "cell_emb": control_cell.view(batch_size, controls_per_sample, -1).mean(
                dim=1
            )
        }
        if return_tokens:
            token_tensor = torch.cat(control_tokens, dim=0)
            output["token_output"] = token_tensor.view(
                batch_size,
                controls_per_sample,
                token_tensor.size(1),
                token_tensor.size(2),
            ).mean(dim=1)
        return output

    def _control_mean(
        self,
        control_gene_ids: torch.Tensor,
        control_values: torch.Tensor | None,
        control_padding_mask: torch.Tensor | None,
        control_counts: int | None,
        control_chunk_size: int | None,
        control_no_grad: bool,
        batch_size: int,
    ) -> torch.Tensor:
        if control_values is None or control_padding_mask is None:
            raise ValueError("control_values and control_padding_mask are required")
        chunk_size = control_chunk_size or control_gene_ids.size(0)
        control_embs = []
        context = torch.no_grad() if control_no_grad else nullcontext()
        with context:
            for start in range(0, control_gene_ids.size(0), chunk_size):
                output = self.backbone.model(
                    src=control_gene_ids[start : start + chunk_size],
                    values=control_values[start : start + chunk_size],
                    src_key_padding_mask=control_padding_mask[
                        start : start + chunk_size
                    ],
                    batch_labels=None,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                control_embs.append(output["cell_emb"])
        control_emb = torch.cat(control_embs, dim=0)
        controls_per_sample = control_counts or control_emb.size(0) // batch_size
        return control_emb.view(batch_size, controls_per_sample, -1).mean(dim=1)
