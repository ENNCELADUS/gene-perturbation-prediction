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
    GeneEffectResidualHead,
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
    head = architecture["head"]
    model = GeneEffectE2EModel(
        StateResponse(state, perturbations),
        GeneEffectResidualHead(
            dims=GeneEffectFeatureDims(**head["dims"]),
            blocks=GeneEffectBlockConfig(**head["blocks"]),
            hidden=head["hidden"],
            n_hidden_layers=head["n_hidden_layers"],
        ),
        projection,
        standardizer,
        collator_seed=architecture["collator_seed"],
    )
    model.architecture = dict(architecture)
    return model


def build_joint_model(
    config: Mapping[str, Any], inputs: PreparedInputs
) -> GeneEffectE2EModel:
    """Released STATE (every weight), a new ESM2 adapter and a new GeneEffect head."""
    model_config = config["model"]
    checkpoint = Path(config["paths"]["state_checkpoint"])
    cell_set_len = int(model_config["cell_sentence_len"])
    line = next(iter(inputs.lines.values()))
    architecture = {
        "state_hparams": released_hparams(checkpoint, cell_set_len=cell_set_len),
        "esm2_adapter_hidden": int(model_config["esm2_adapter_hidden"]),
        "head": {
            "blocks": dict(model_config["head_blocks"]),
            "dims": asdict(
                GeneEffectFeatureDims(
                    e_g=int(np.asarray(inputs.esm2_vectors).shape[1]),
                    z_c=2 * int(line.controls_tx1.shape[1]),
                )
            ),
            "hidden": int(model_config["head_hidden"]),
            "n_hidden_layers": int(model_config["head_layers"]),
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
