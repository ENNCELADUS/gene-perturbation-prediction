"""Build the joint GeneEffect model fresh from released STATE, or restore it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from src.data.embeddings import Esm2EmbeddingTable
from src.model.features import FixedSparseProjection
from src.model.geneeffect import GeneEffectE2EModel
from src.model.head import (
    GeneEffectBlockConfig,
    GeneEffectFeatureDims,
    GeneEffectNestedHead,
)
from src.model.normalization import BlockStandardizer
from src.model.perturbation import Esm2PerturbationAdapter
from src.model.state import (
    StateResponse,
    build_state,
    load_released_state,
    released_hparams,
)

if TYPE_CHECKING:
    from src.data.prepared import PreparedInputs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WarmStartReport:
    """Key-by-key outcome of a shape-filtered checkpoint load.

    Every destination parameter or buffer is in exactly one of ``loaded_keys``,
    ``shape_skipped_keys`` or ``missing_keys``; ``unexpected_keys`` are checkpoint
    keys the destination model does not have.
    """

    loaded_keys: tuple[str, ...]
    shape_skipped_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def warm_start_state_dict(model: nn.Module, checkpoint_path: Path) -> WarmStartReport:
    """Load every shape-matching checkpoint key into ``model`` and report the rest.

    Raises ``ValueError`` when zero keys load: a silent no-op load would train a
    randomly initialised model with no error.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    model_state = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    shape_skipped: list[str] = []
    unexpected: list[str] = []
    for name, tensor in checkpoint["state_dict"].items():
        if name not in model_state:
            unexpected.append(name)
        elif tuple(tensor.shape) != tuple(model_state[name].shape):
            shape_skipped.append(name)
        else:
            loaded[name] = tensor
    if not loaded:
        raise ValueError(
            f"loading {checkpoint_path} into {type(model).__name__} matched zero keys"
        )
    model.load_state_dict(loaded, strict=False)
    report = WarmStartReport(
        loaded_keys=tuple(sorted(loaded)),
        shape_skipped_keys=tuple(sorted(shape_skipped)),
        missing_keys=tuple(sorted(set(model_state) - set(loaded) - set(shape_skipped))),
        unexpected_keys=tuple(sorted(unexpected)),
    )
    logger.info(
        "Loaded %s: %d loaded, %d shape-skipped, %d missing, %d unexpected",
        checkpoint_path,
        len(report.loaded_keys),
        len(report.shape_skipped_keys),
        len(report.missing_keys),
        len(report.unexpected_keys),
    )
    return report


def _assemble(
    architecture: Mapping[str, Any],
    inputs: PreparedInputs,
    state: nn.Module,
    projection: FixedSparseProjection,
    standardizer: BlockStandardizer,
) -> GeneEffectE2EModel:
    if list(architecture["state_hparams"]["gene_names"]) != list(inputs.hvg_order):
        raise ValueError("STATE's gene order differs from the prepared HVG order")
    if list(architecture["genes"]) != list(inputs.genes):
        raise ValueError("the saved gene order differs from the prepared gene order")
    if list(inputs.residual_scale.index) != list(inputs.genes):
        raise ValueError("residual_scale is not indexed by the prepared gene order")
    head = architecture["head"]
    if inputs.context_pca.n_components != head["dims"]["z_c"]:
        raise ValueError(
            f"the head reads {head['dims']['z_c']} context components but the "
            f"prepared context PCA has {inputs.context_pca.n_components}"
        )
    vectors = np.asarray(inputs.esm2_vectors, dtype=np.float32)
    table = Esm2EmbeddingTable(
        vectors.shape[1], dict(zip(inputs.esm2_symbols, vectors, strict=True))
    )
    perturbations = Esm2PerturbationAdapter(
        list(inputs.esm2_symbols),
        table,
        architecture["esm2_adapter_hidden"],
        int(architecture["state_hparams"]["pert_dim"]),
    )
    model = GeneEffectE2EModel(
        StateResponse(state, perturbations),
        GeneEffectNestedHead(
            dims=GeneEffectFeatureDims(**head["dims"]),
            blocks=GeneEffectBlockConfig(**head["blocks"]),
            n_genes=head["n_genes"],
            factor_rank=head["factor_rank"],
            dropout=head["dropout"],
        ),
        projection,
        standardizer,
        collator_seed=architecture["collator_seed"],
        residual_scale=torch.as_tensor(
            inputs.residual_scale.to_numpy(), dtype=torch.float32
        ),
    )
    model.architecture = dict(architecture)
    return model


def build_joint_model(
    config: Mapping[str, Any], inputs: PreparedInputs
) -> GeneEffectE2EModel:
    """Released STATE (every weight), a new ESM2 adapter and a new GeneEffect head.

    STATE and the adapter are built even when no head block uses them, so every
    STATE setting shares one module layout; such a model never calls them.
    """
    model_config = config["model"]
    checkpoint = Path(config["paths"]["state_checkpoint"])
    cell_set_len = int(model_config["cell_sentence_len"])
    architecture = {
        "state_hparams": released_hparams(checkpoint, cell_set_len=cell_set_len),
        "esm2_adapter_hidden": int(model_config["esm2_adapter_hidden"]),
        "genes": list(inputs.genes),
        "head": {
            "blocks": dict(model_config["head_blocks"]),
            "dims": asdict(
                GeneEffectFeatureDims(
                    e_g=int(np.asarray(inputs.esm2_vectors).shape[1]),
                    z_c=int(model_config["context_components"]),
                )
            ),
            "n_genes": len(inputs.genes),
            "factor_rank": int(model_config["factor_rank"]),
            "dropout": float(model_config["dropout"]),
        },
        "collator_seed": int(config["seeds"]["collator"]),
    }
    return _assemble(
        architecture,
        inputs,
        load_released_state(checkpoint, cell_set_len=cell_set_len),
        FixedSparseProjection(seed=int(config["seeds"]["projection"])),
        BlockStandardizer(),
    )


def restore_joint_model(
    saved: Mapping[str, Any], inputs: PreparedInputs
) -> GeneEffectE2EModel:
    """Rebuild a saved joint model from its own architecture and weights only.

    No released STATE checkpoint or ESM2 file is opened; ``inputs`` must have
    been opened with the checkpoint's saved preprocessing.
    """
    architecture = saved["architecture"]
    model = _assemble(
        architecture,
        inputs,
        build_state(architecture["state_hparams"]),
        FixedSparseProjection.from_state(saved["projection_state"]),
        BlockStandardizer.from_state(saved["normalization_state"]),
    )
    model.load_state_dict(saved["model_state"], strict=True)
    return model
