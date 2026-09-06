"""model / initialization."""

from __future__ import annotations
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from copy import deepcopy
from typing import Mapping, TYPE_CHECKING
import numpy as np
import logging
import os
from pathlib import Path
from typing import Any
import torch
from torch import nn
from src.data.embeddings import Esm2EmbeddingTable
from src.model.geneeffect import GeneEffectE2EModel
from src.model.features import FixedSparseProjection, HVG_WIDTH
from src.model.head import GeneEffectFeatureDims, GeneEffectResidualHead
from src.model.normalization import BlockStandardizer

if TYPE_CHECKING:
    from src.data.prepared import PreparedInputs
from src.model.perturbation import Esm2PerturbationAdapter
from src.model.state import StateForwardAdapter
from src.model.state import ForwardOnlyStateModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WarmStartReport:
    """Key-by-key outcome of a shape-filtered checkpoint warm start.

    Mirrors ``src.model.tx1.validate_load_result``'s
    honest-load-reporting shape, adapted for a shape filter rather than a
    strict key-name-only load. Every destination-model parameter/buffer name
    is accounted for in exactly one of ``loaded_keys``, ``shape_skipped_keys``,
    or ``missing_keys``; ``unexpected_keys`` are checkpoint keys the
    destination model does not have at all.
    """

    loaded_keys: tuple[str, ...]
    shape_skipped_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def warm_start_state_dict(model: nn.Module, checkpoint_path: Path) -> WarmStartReport:
    """Warm-start ``model`` from a checkpoint, skipping shape-mismatched keys.

    Loads ``checkpoint_path`` with a raw ``torch.load`` (bypassing Lightning's
    ``load_from_checkpoint``, which loads the full state dict unfiltered and
    raises ``RuntimeError`` on any shape mismatch), filters the checkpoint's
    ``state_dict`` down to keys whose shape matches ``model``'s own, and
    loads only those with ``strict=False``. Keys left unmatched (by name or
    shape) keep ``model``'s freshly initialized values untouched.

    Args:
        model: Destination module, already constructed with its final
            shapes (e.g. a fresh 2560-input
            ``StateTransitionPerturbationModel``).
        checkpoint_path: Path to a Lightning ``.ckpt`` file with a top-level
            ``state_dict`` key.

    Returns:
        A :class:`WarmStartReport` naming every loaded, shape-skipped,
        missing, and unexpected key.

    Raises:
        ValueError: If zero keys are loadable. A warm start that silently
            loads nothing is worse than a crash: it would produce a
            randomly-initialized model that trains, converges to garbage,
            and reports no error.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint["state_dict"]
    model_state = model.state_dict()

    loaded: dict[str, torch.Tensor] = {}
    shape_skipped: list[str] = []
    unexpected: list[str] = []
    for name, tensor in checkpoint_state.items():
        destination = model_state.get(name)
        if destination is None:
            unexpected.append(name)
        elif tuple(tensor.shape) != tuple(destination.shape):
            shape_skipped.append(name)
        else:
            loaded[name] = tensor
    missing = sorted(set(model_state) - set(loaded) - set(shape_skipped))

    if not loaded:
        msg = (
            f"Warm start from {checkpoint_path} loaded zero keys into "
            f"{type(model).__name__} -- refusing a silent no-op warm start "
            "that would train a randomly-initialized model with no error."
        )
        raise ValueError(msg)

    model.load_state_dict(loaded, strict=False)

    report = WarmStartReport(
        loaded_keys=tuple(sorted(loaded)),
        shape_skipped_keys=tuple(sorted(shape_skipped)),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(sorted(unexpected)),
    )
    logger.info(
        "STATE warm start from %s: %d loaded, %d shape-skipped, %d missing, "
        "%d unexpected. shape_skipped=%s missing=%s unexpected=%s",
        checkpoint_path,
        len(report.loaded_keys),
        len(report.shape_skipped_keys),
        len(report.missing_keys),
        len(report.unexpected_keys),
        report.shape_skipped_keys,
        report.missing_keys,
        report.unexpected_keys,
    )
    return report


@contextmanager
def _suppress_checkpoint_output() -> Any:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def build_joint_model(
    config: Mapping[str, Any],
    inputs: PreparedInputs,
    *,
    architecture: Mapping[str, Any] | None = None,
    model_state: Mapping[str, torch.Tensor] | None = None,
    projection_state: Mapping[str, Any] | None = None,
    normalization_state: Mapping[str, Any] | None = None,
    model_cls: type[nn.Module] | None = None,
) -> GeneEffectE2EModel:
    """Build fresh from released STATE, or strictly restore a self-contained run.

    Resume requires all four saved model objects and PreparedInputs restored from
    checkpoint preprocessing (including actual ordered ESM2 vectors). No upstream
    model or ESM2 file is opened on that path. ``model_cls`` injects a STATE-shaped
    CPU fixture; production always uses the pinned StateTransitionPerturbationModel.
    """
    restoring = architecture is not None
    saved = (model_state, projection_state, normalization_state)
    if restoring != all(value is not None for value in saved) or (
        not restoring and any(value is not None for value in saved)
    ):
        raise ValueError(
            "restore requires architecture, model, projection and normalization"
        )
    if model_cls is None:
        from state.tx.models.state_transition import StateTransitionPerturbationModel

        model_cls = StateTransitionPerturbationModel
    symbols = list(inputs.esm2_symbols)
    vectors = np.asarray(inputs.esm2_vectors, dtype=np.float32)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(symbols)
        or not symbols
        or len(set(symbols)) != len(symbols)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("invalid ordered ESM2 construction inputs")
    if restoring:
        metadata = deepcopy(dict(architecture))
        if metadata["esm2_symbols"] != symbols:
            raise ValueError(
                "checkpoint architecture and preprocessing ESM2 order differ"
            )
        if metadata["hvg_order"] != list(inputs.hvg_order):
            raise ValueError("checkpoint architecture and input HVG order differ")
        hparams = deepcopy(metadata["state_hparams"])
    else:
        model_config = config.get("model", {})
        checkpoint_path = Path(config["paths"]["state_checkpoint"])
        reference = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hparams = deepcopy(dict(reference["hyper_parameters"]))
        checkpoint_genes = hparams.get("gene_names")
        if checkpoint_genes is not None and list(checkpoint_genes) != list(
            inputs.hvg_order
        ):
            raise ValueError("released STATE gene order differs from prepared HVGs")
        line = next(iter(inputs.lines.values()))
        hparams.update(
            input_dim=int(line.controls_tx1.shape[1]),
            output_dim=len(inputs.hvg_order),
            pert_dim=int(model_config.get("pert_dim", hparams["pert_dim"])),
            output_space="gene",
            cell_set_len=int(model_config.get("cell_sentence_len", 64)),
        )
        metadata = {
            "state_hparams": deepcopy(hparams),
            "esm2_symbols": symbols,
            "hvg_order": list(inputs.hvg_order),
            "esm2_adapter_hidden": int(model_config.get("esm2_adapter_hidden", 512)),
            "head": {
                "dims": asdict(
                    GeneEffectFeatureDims(
                        e_g=vectors.shape[1], z_c=2 * int(line.controls_tx1.shape[1])
                    )
                ),
                "hidden": int(model_config.get("head_hidden", 256)),
                "n_hidden_layers": int(model_config.get("head_layers", 2)),
            },
            "collator_seed": int(config["seeds"]["collator"]),
        }
    if hparams["output_dim"] != HVG_WIDTH or len(inputs.hvg_order) != HVG_WIDTH:
        raise ValueError(f"joint response features require {HVG_WIDTH} HVGs")
    with _suppress_checkpoint_output():
        # STATE mutates nested transformer kwargs while constructing; save the
        # original constructor arguments and pass a private copy each time.
        state = model_cls(**deepcopy(hparams))
    if not restoring:
        report = warm_start_state_dict(state, checkpoint_path)
        intentional = ("basal_encoder.", "pert_encoder.", "project_out.")
        unexpected_shapes = tuple(
            name
            for name in report.shape_skipped_keys
            if not name.startswith(intentional)
        )
        if report.missing_keys or report.unexpected_keys or unexpected_shapes:
            raise ValueError(f"unexpected released STATE incompatibility: {report}")
        logger.info("Fresh joint model: new ESM2 adapter and GeneEffect head")
    table = Esm2EmbeddingTable(
        vectors.shape[1], dict(zip(symbols, vectors, strict=True))
    )
    perturbations = Esm2PerturbationAdapter(
        symbols, table, metadata["esm2_adapter_hidden"], int(hparams["pert_dim"])
    )
    head_config = metadata["head"]
    head = GeneEffectResidualHead(
        dims=GeneEffectFeatureDims(**head_config["dims"]),
        hidden=head_config["hidden"],
        n_hidden_layers=head_config["n_hidden_layers"],
    )
    model = GeneEffectE2EModel(
        ForwardOnlyStateModel(StateForwardAdapter(state), perturbations),
        head,
        FixedSparseProjection.from_state(projection_state)
        if restoring
        else FixedSparseProjection(seed=int(config["seeds"]["projection"])),
        BlockStandardizer.from_state(normalization_state)
        if restoring
        else BlockStandardizer(),
        collator_seed=metadata["collator_seed"],
    )
    model.architecture = metadata
    if restoring:
        model.load_state_dict(model_state, strict=True)
    return model
