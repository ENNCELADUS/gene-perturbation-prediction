"""Tx1-3B (Tahoe-x1) loaded from its released safetensors without flash-attn, its
basal-cell ``EncoderFn``, and its encoder of each gene's mean contextual embedding."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch

from src.data.tx1_cache import EMBEDDING_WIDTH, EncoderFn


class TorchBertPadding:
    """Pure-Torch equivalent of ``flash_attn.bert_padding``."""

    @staticmethod
    def unpad_input(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        sequence_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        cumulative = torch.nn.functional.pad(
            torch.cumsum(sequence_lengths, dim=0, dtype=torch.int32), (1, 0)
        )
        unpadded = hidden_states.reshape(-1, *hidden_states.shape[2:])[indices]
        return unpadded, indices, cumulative, int(sequence_lengths.max().item())

    unpad_input_for_concatenated_sequences = unpad_input


def install_padding_metadata_fallback() -> None:
    """Supply padding indices when flash-attn is absent but torch attention is used."""
    from llmfoundry.models.mpt import modeling_mpt

    if not hasattr(modeling_mpt, "bert_padding"):
        modeling_mpt.bert_padding = TorchBertPadding


def validate_load_result(load_result: Any) -> dict[str, object]:
    """Reject partial backbone loads while allowing removed chemical-only heads."""
    missing = sorted(load_result.missing_keys)
    unexpected = sorted(load_result.unexpected_keys)
    allowed_unexpected = [
        key
        for key in unexpected
        if "chemical_encoder" in key or "chem_encoder" in key or "mlm_head" in key
    ]
    disallowed_unexpected = sorted(set(unexpected) - set(allowed_unexpected))
    if missing or disallowed_unexpected:
        raise ValueError(
            "Incomplete Tahoe-x1 checkpoint load: "
            f"missing={missing}, disallowed_unexpected={disallowed_unexpected}"
        )
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "allowed_unexpected_keys": allowed_unexpected,
        "complete_backbone_load": True,
    }


def load_local_safetensors(
    model_dir: Path, gene_embeddings: bool
) -> tuple[Any, Any, Any, dict[str, object]]:
    """The model, vocab, collator config and load report. ``gene_embeddings``: the
    output also holds ``gene_ids`` (cells, sequence) and ``gene_emb`` (cells,
    sequence, width), the transformer's output for each token."""
    from omegaconf import OmegaConf as om
    from safetensors.torch import load_file
    from tahoe_x1.model import ComposerTX
    from tahoe_x1.tokenizer import GeneVocab

    install_padding_metadata_fallback()

    model_config = om.load(model_dir / "model_config.yml")
    collator_config = om.load(model_dir / "collator_config.yml")
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    model_config["do_mlm"] = False
    model_config["return_gene_embeddings"] = gene_embeddings
    model_config["attn_config"]["attn_impl"] = "torch"
    collator_config["use_chem_token"] = False
    del model_config["chemical_encoder"]
    if "drug_to_id_path" in collator_config:
        del collator_config["drug_to_id_path"]
    model = ComposerTX(
        model_config=model_config,
        collator_config=collator_config,
    )
    state = load_file(model_dir / "model.safetensors", device="cpu")
    load_report = validate_load_result(model.load_state_dict(state, strict=False))
    model.eval()
    return model, vocab, collator_config, load_report


def build_tx1_encoder(
    model_dir: Path, batch_size: int, max_length: int
) -> tuple[EncoderFn, dict[str, object]]:
    """The Tx1-3B ``EncoderFn`` (CUDA only) and its load report."""
    from composer import Trainer
    from tahoe_x1.utils.util import loader_from_adata

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to embed Tx1-3B basal cells")
    install_padding_metadata_fallback()
    model, vocab, collator_config, load_report = load_local_safetensors(
        model_dir, gene_embeddings=False
    )
    trainer = Trainer(model=model, device="gpu")

    def encode(adata: ad.AnnData) -> np.ndarray:
        # Each line starts the frozen collator from the same explicit seed,
        # independent of cache hits and earlier encoded lines.
        import random

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        genes = adata.var["ensembl_id"].astype(str).tolist()
        gene_ids = np.asarray([vocab[gene] for gene in genes], dtype=int)
        loader = loader_from_adata(
            adata=adata,
            collator_cfg=collator_config,
            vocab=vocab,
            batch_size=batch_size,
            max_length=max_length,
            gene_ids=gene_ids,
            num_workers=0,
            prefetch_factor=None,
        )
        predictions = trainer.predict(loader, return_outputs=True)
        embeddings = torch.cat(
            [output["cell_emb"].detach().float().cpu() for output in predictions],
            dim=0,
        )
        return embeddings.numpy().astype(np.float32)

    return encode, load_report


def gene_sums(
    gene_ids: torch.Tensor, gene_emb: torch.Tensor, wanted: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each token id of ``wanted`` (genes,): the sum of its embedding over the
    cells whose sequence holds it (genes, width) and the number of those cells
    (genes,). ``gene_ids``: (cells, sequence); ``gene_emb``: (cells, sequence,
    width). A wanted token twice in one cell's sequence (two ``var`` rows of one
    Ensembl ID) raises ``ValueError``: its count would exceed the cells."""
    unique, inverse = torch.unique(wanted, return_inverse=True)
    flat = gene_ids.reshape(-1).to(unique.dtype)
    position = torch.searchsorted(unique, flat).clamp(max=len(unique) - 1)
    hit = unique[position] == flat
    position = position[hit]
    cell = torch.arange(gene_ids.shape[0], device=flat.device)
    cell = cell.repeat_interleave(gene_ids.shape[1])[hit]
    if torch.unique(cell * len(unique) + position).numel() < position.numel():
        raise ValueError("a wanted token is twice in one cell's sequence")
    width = gene_emb.shape[-1]
    sums = torch.zeros(len(unique), width, device=gene_emb.device).index_add_(
        0, position, gene_emb.reshape(-1, width)[hit].float()
    )
    counts = torch.zeros(len(unique), dtype=torch.long, device=flat.device)
    counts.index_add_(0, position, torch.ones_like(position))
    return sums[inverse], counts[inverse]


def build_tx1_gene_encoder(
    model_dir: Path, batch_size: int, max_length: int, cells_per_pass: int
) -> Callable[[ad.AnnData], tuple[np.ndarray, np.ndarray]]:
    """For the genes of ``adata.var``: the mean of Tx1-3B's contextual embedding over
    the cells whose sequence holds the gene (genes x 2,560, float32, NaN where no
    cell holds it) and the share of the cells that hold it (genes,). A gene whose
    Ensembl ID is not in the vocab is held by no cell. CUDA only."""
    from composer import Trainer
    from tahoe_x1.utils.util import loader_from_adata

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to embed Tx1-3B genes")
    model, vocab, collator_config, _ = load_local_safetensors(
        model_dir, gene_embeddings=True
    )
    trainer = Trainer(model=model, device="gpu")
    stoi = vocab.get_stoi()

    def encode(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
        import random

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        genes = adata.var["ensembl_id"].astype(str).tolist()
        gene_ids = np.asarray([vocab[gene] for gene in genes], dtype=int)
        # -1 is never a token: a gene outside the vocab gets no cell
        wanted = torch.tensor([stoi.get(gene, -1) for gene in genes])
        sums = torch.zeros(len(genes), EMBEDDING_WIDTH)
        counts = torch.zeros(len(genes), dtype=torch.long)
        # one pass of cells at a time: predict holds every batch's token outputs
        for start in range(0, adata.n_obs, cells_per_pass):
            loader = loader_from_adata(
                adata=adata[start : start + cells_per_pass].copy(),
                collator_cfg=collator_config,
                vocab=vocab,
                batch_size=batch_size,
                max_length=max_length,
                gene_ids=gene_ids,
                num_workers=0,
                prefetch_factor=None,
            )
            for output in trainer.predict(loader, return_outputs=True):
                ids = output["gene_ids"]
                part, number = gene_sums(ids, output["gene_emb"], wanted.to(ids.device))
                sums += part.cpu()
                counts += number.cpu()
        mean = sums / counts[:, None]
        share = counts.float() / adata.n_obs
        return mean.numpy().astype(np.float32), share.numpy()

    return encode
