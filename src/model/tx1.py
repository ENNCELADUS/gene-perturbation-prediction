"""model / tx1."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import anndata as ad
import numpy as np
import torch
from src.data.tx1_cache import EncoderFn


class _TorchBertPadding:
    """Pure-Torch equivalent of flash_attn.bert_padding for the width probe."""

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
        modeling_mpt.bert_padding = _TorchBertPadding


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


def load_local_safetensors(model_dir: Path) -> tuple[Any, Any, Any, dict[str, object]]:
    from omegaconf import OmegaConf as om
    from safetensors.torch import load_file
    from tahoe_x1.model import ComposerTX
    from tahoe_x1.tokenizer import GeneVocab

    install_padding_metadata_fallback()

    model_config = om.load(model_dir / "model_config.yml")
    collator_config = om.load(model_dir / "collator_config.yml")
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    model_config["do_mlm"] = False
    model_config["return_gene_embeddings"] = False
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


def _build_tx1_encoder(
    model_dir: Path, batch_size: int, max_length: int
) -> tuple[EncoderFn, dict[str, object]]:
    """Build the real Tx1-3B ``EncoderFn``, reusing the width-probe's loader."""
    from composer import Trainer
    from tahoe_x1.utils.util import loader_from_adata

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to embed Tx1-3B basal cells")
    install_padding_metadata_fallback()
    model, vocab, collator_config, load_report = load_local_safetensors(model_dir)
    trainer = Trainer(model=model, device="gpu")

    def encode(adata: ad.AnnData) -> np.ndarray:
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
