"""P1-C interface-isolation variant backbones.

Builds the five interface variants (``VARIANTS``) plus the diagnostic
``"N-native"`` state from a P1-B ``B-init.pt`` template, auditing every
backbone parameter's provenance the way ``src.model.p1b.configure_parameters``
audits P1-B's. Pure ``src.model``/``src.data`` module: no import of
``src.training``, ``src.eval`` or ``src.experiments``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from src.data.embeddings import Esm2EmbeddingTable
from src.data.p1c import HVG_WIDTH, INPUT_LAYOUTS
from src.model.initialization import _suppress_checkpoint_output
from src.model.perturbation import Esm2PerturbationAdapter
from src.model.state import ForwardOnlyStateModel, StateForwardAdapter


VARIANTS = ("V0", "V1", "V2-null", "V2", "V3")

INPUT_LAYOUT = {
    "V0": "tx1",
    "V1": "hvg_tx1",
    "V2-null": "hvg",
    "V2": "hvg_tx1",
    "V3": "hvg_tx1",
    "N-native": "hvg",
}
if not set(INPUT_LAYOUT.values()) <= set(INPUT_LAYOUTS):
    raise ValueError(
        "INPUT_LAYOUT values must be members of src.data.p1c.INPUT_LAYOUTS"
    )

_TRAINABLE_COUNTS = {
    "V0": 2_533_864,
    "V1": 2_533_864,
    "V2-null": 1_694_184,
    "V2": 2_533_864,
    "V3": 2_533_864,
    "N-native": 0,
}

_NATIVE_BASAL_VARIANTS = ("V2-null", "V2", "V3", "N-native")


class NativeOneHotPerturbations(nn.Module):
    """Fixed one-hot (or other native) perturbation vectors, keyed by gene name.

    Keys are used verbatim: unlike :class:`Esm2PerturbationAdapter`, gene
    names are never upper-cased, since native STATE vocabularies use their
    own casing (for example ``"non-targeting"``).
    """

    def __init__(self, onehot_map: dict[str, torch.Tensor]) -> None:
        super().__init__()
        if not onehot_map:
            raise ValueError(
                "NativeOneHotPerturbations requires a non-empty onehot_map"
            )
        self._genes = list(onehot_map)
        matrix = torch.stack(
            [
                torch.as_tensor(onehot_map[gene], dtype=torch.float32)
                for gene in self._genes
            ]
        )
        self.register_buffer("vectors", matrix, persistent=False)
        self._index = {gene: position for position, gene in enumerate(self._genes)}

    def has_embedding(self, gene: str) -> bool:
        return gene in self._index

    def forward(self, gene: str) -> torch.Tensor:
        return self.forward_many([gene])[0]

    def forward_many(self, genes: list[str] | tuple[str, ...]) -> torch.Tensor:
        try:
            indices = [self._index[gene] for gene in genes]
        except KeyError as exc:
            raise KeyError(
                f"no native one-hot embedding for gene {exc.args[0]!r}"
            ) from exc
        index_tensor = torch.as_tensor(
            indices, dtype=torch.long, device=self.vectors.device
        )
        return self.vectors.index_select(0, index_tensor)


class SplitBasalEncoder(nn.Module):
    """Native basal encoder plus a zero-initialised Tx1 context term.

    ``forward`` adds the frozen native encoding of the leading ``hvg_width``
    columns to a trainable linear map of the remaining (Tx1) columns, so an
    untrained instance reproduces the native encoder exactly.
    """

    def __init__(
        self, native: nn.Module, hvg_width: int, context_dim: int, hidden: int
    ) -> None:
        super().__init__()
        self.native = native
        self.hvg_width = int(hvg_width)
        self.context = nn.Linear(int(context_dim), int(hidden), bias=False)
        nn.init.zeros_(self.context.weight)

    def forward(self, expr: torch.Tensor) -> torch.Tensor:
        return self.native(expr[..., : self.hvg_width]) + self.context(
            expr[..., self.hvg_width :]
        )


class NullSubtractedBackbone(ForwardOnlyStateModel):
    """Predict ``control_hvg + (perturbed - null)`` instead of STATE's raw output.

    An untrained adapter (perturbation vector identically zero) makes the
    perturbed and null forwards identical, so the prediction is exactly
    ``control_hvg`` before any training -- the "exactly no-change" behaviour
    required of V1 and V3.
    """

    def __init__(
        self,
        state_adapter: StateForwardAdapter,
        perturbations: nn.Module,
        *,
        hvg_width: int,
        state_takes_hvg: bool,
    ) -> None:
        super().__init__(state_adapter, perturbations)
        self.hvg_width = int(hvg_width)
        self.state_takes_hvg = bool(state_takes_hvg)

    def forward(
        self,
        control_chunks: tuple[torch.Tensor, ...],
        gene: str | tuple[str, ...],
        batch_index_chunks: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor, ...]:
        """One STATE call carrying both branches.

        The perturbed chunks and the deduplicated null representatives go
        through a *single* ``forward_condition_chunks`` call. Two calls would
        present the transformer with two different batch sizes, and under CUDA
        bf16 autocast a different kernel may then be selected for each, so
        ``p - n`` would not be exactly zero for an untrained (identically zero)
        perturbation vector. One call keeps both branches on the same kernel.
        """
        if self.training:
            raise RuntimeError(
                "NullSubtractedBackbone requires eval mode: the null-subtracted "
                "residual is only exact without stochastic layers"
            )
        genes = (
            tuple(gene for _ in control_chunks)
            if isinstance(gene, str)
            else tuple(gene)
        )
        if len(genes) != len(control_chunks):
            raise ValueError("one gene is required per STATE condition chunk")
        hvg = tuple(c[:, : self.hvg_width] for c in control_chunks)
        inputs = (
            control_chunks
            if self.state_takes_hvg
            else tuple(c[:, self.hvg_width :] for c in control_chunks)
        )

        groups = self.null_groups(inputs, batch_index_chunks)
        representatives = tuple(inputs[ix[0]] for ix in groups.values())
        null_batches = tuple(batch_index_chunks[ix[0]] for ix in groups.values())

        # ForwardOnlyStateModel.forward's perturbation lookup, inlined: the
        # null branch's zero rows must be concatenated onto it before the
        # single STATE call.
        if hasattr(self.perturbations, "forward_many"):
            perturbations = self.perturbations.forward_many(genes)
        else:
            perturbations = torch.stack([self.perturbations(name) for name in genes])
        zeros = torch.zeros(
            len(representatives),
            perturbations.shape[1],
            device=perturbations.device,
            dtype=perturbations.dtype,
        )
        outputs = self.state_adapter.forward_condition_chunks(
            inputs + representatives,
            torch.cat([perturbations, zeros], dim=0),
            genes + tuple("__null__" for _ in representatives),
            tuple(batch_index_chunks) + null_batches,
        )
        perturbed = outputs[: len(inputs)]
        null_outputs = outputs[len(inputs) :]
        null: list[torch.Tensor | None] = [None] * len(inputs)
        for out, ix in zip(null_outputs, groups.values(), strict=True):
            for i in ix:
                null[i] = out
        return tuple(h + (p - n) for h, p, n in zip(hvg, perturbed, null, strict=True))

    @staticmethod
    def null_groups(
        inputs: tuple[torch.Tensor, ...],
        batch_index_chunks: tuple[torch.Tensor | None, ...],
    ) -> dict[tuple, list[int]]:
        """Group input chunks that share one null forward.

        Two chunks dedupe only when both their input tensor identity *and*
        their batch-index chunk identity match -- otherwise two conditions
        that happen to share a control bag but carry different batch indices
        would collapse into a single null computed at the first one's index.
        """
        groups: dict[tuple, list[int]] = {}
        for i, chunk in enumerate(inputs):
            batch_chunk = batch_index_chunks[i]
            batch_key = (
                None
                if batch_chunk is None
                else (batch_chunk.data_ptr(), tuple(batch_chunk.shape))
            )
            key = (
                chunk.data_ptr(),
                tuple(chunk.shape),
                tuple(chunk.stride()),
                str(chunk.device),
                batch_key,
            )
            groups.setdefault(key, []).append(i)
        return groups


class NativeBackbone(ForwardOnlyStateModel):
    """Plain forward, with every ``None`` batch-index chunk replaced by a constant."""

    def __init__(
        self,
        state_adapter: StateForwardAdapter,
        perturbations: nn.Module,
        *,
        batch_index: int = 0,
    ) -> None:
        super().__init__(state_adapter, perturbations)
        self.batch_index = int(batch_index)

    def forward(
        self,
        control_chunks: tuple[torch.Tensor, ...],
        gene: str | tuple[str, ...],
        batch_index_chunks: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor, ...]:
        filled = tuple(
            chunk
            if chunk is not None
            else torch.full(
                (control_chunks[i].shape[0],),
                self.batch_index,
                dtype=torch.long,
                device=control_chunks[i].device,
            )
            for i, chunk in enumerate(batch_index_chunks)
        )
        return super().forward(control_chunks, gene, filled)


def zero_final_layer(adapter: Esm2PerturbationAdapter) -> None:
    """Zero an ``Esm2PerturbationAdapter``'s final linear layer in place."""
    final = adapter.adapter.net[-1]
    if not isinstance(final, nn.Linear):
        raise ValueError(
            f"adapter's final layer must be nn.Linear, got {type(final).__name__}"
        )
    with torch.no_grad():
        final.weight.zero_()
        final.bias.zero_()


def _build_adapter_from_template(
    template: dict[str, Any], pert_dim: int
) -> tuple[Esm2PerturbationAdapter, tuple[str, ...]]:
    """Rebuild the ESM2 adapter exactly as ``restore_backbone`` does, loading its
    ``perturbations.``-prefixed template weights strictly.
    """
    preprocessing = template["preprocessing"]
    architecture = template["architecture"]
    vectors = preprocessing["esm2_vectors"].numpy()
    symbols = preprocessing["esm2_symbols"]
    table = Esm2EmbeddingTable(vectors.shape[1], dict(zip(symbols, vectors)))
    perturbations = Esm2PerturbationAdapter(
        symbols, table, architecture["esm2_adapter_hidden"], int(pert_dim)
    )
    prefix = "perturbations."
    loaded = {
        name[len(prefix) :]: tensor
        for name, tensor in template["model_state"].items()
        if name.startswith(prefix)
    }
    perturbations.load_state_dict(loaded, strict=True)
    return perturbations, tuple(loaded)


def build_variant(
    template: dict[str, Any],
    variant: str,
    *,
    native_state: dict[str, torch.Tensor] | None = None,
    native_map: dict[str, torch.Tensor] | None = None,
    batch_index: int = 0,
    expected_count: int | None | str = "table",
) -> tuple[nn.Module, dict[str, Any]]:
    """Construct one P1-C interface variant from a P1-B ``B-init.pt`` template.

    Returns ``(backbone, report)``. ``report["parameters"]`` classifies every
    backbone parameter's ``origin`` in
    ``{"inherited-template", "inherited-native", "new-template", "new-zero"}``;
    an unclassified parameter raises ``ValueError``. ``expected_count="table"``
    (the production default) checks the trainable count against the fixed
    per-variant table; ``None`` skips that check (tiny-dimension tests); an
    ``int`` overrides the table value.
    """
    if variant not in VARIANTS and variant != "N-native":
        raise ValueError(f"unknown P1-C variant: {variant!r}")

    architecture = template["architecture"]
    original_hparams = architecture["state_hparams"]
    tx1_width = int(original_hparams["input_dim"])
    hparams = deepcopy(original_hparams)
    native_basal = variant in _NATIVE_BASAL_VARIANTS
    if native_basal:
        hparams["input_dim"] = HVG_WIDTH

    with _suppress_checkpoint_output():
        from state.tx.models.state_transition import StateTransitionPerturbationModel

        state = StateTransitionPerturbationModel(**deepcopy(hparams))

    prefix = "state_adapter.state_model."
    template_state_dict = {
        name[len(prefix) :]: tensor
        for name, tensor in template["model_state"].items()
        if name.startswith(prefix)
    }

    origins: dict[str, str] = {}

    if variant == "N-native":
        if native_state is None:
            raise ValueError("N-native requires native_state")
        if native_map is None:
            raise ValueError("N-native requires native_map")
        pert_dim = int(hparams["pert_dim"])
        bad_widths = {
            gene: tuple(torch.as_tensor(vector).shape)
            for gene, vector in native_map.items()
            if torch.as_tensor(vector).shape[-1] != pert_dim
        }
        if bad_widths:
            raise ValueError(
                f"native_map vectors must have width {pert_dim}: {bad_widths}"
            )
        state.load_state_dict(native_state, strict=True)
        for name in native_state:
            origins[prefix + name] = "inherited-native"
        perturbations: nn.Module = NativeOneHotPerturbations(native_map)
    elif variant in ("V0", "V1"):
        state.load_state_dict(template_state_dict, strict=True)
        for name in template_state_dict:
            origins[prefix + name] = "inherited-template"
        origins[prefix + "basal_encoder.0.weight"] = "new-template"

        adapter, loaded_pert_names = _build_adapter_from_template(
            template, int(hparams["pert_dim"])
        )
        perturbations = adapter
        for name in loaded_pert_names:
            origins["perturbations." + name] = "new-template"
        if variant == "V1":
            zero_final_layer(adapter)
            origins["perturbations.adapter.net.2.weight"] = "new-zero"
            origins["perturbations.adapter.net.2.bias"] = "new-zero"
    else:
        # V2-null, V2, V3: native basal encoder loaded from native_state.
        if native_state is None:
            raise ValueError(f"{variant} requires native_state")
        filtered = {
            name: tensor
            for name, tensor in template_state_dict.items()
            if name != "basal_encoder.0.weight"
        }
        result = state.load_state_dict(filtered, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        if missing != ["basal_encoder.0.weight"] or unexpected:
            raise ValueError(
                f"unexpected native-basal load report: missing={missing} "
                f"unexpected={unexpected}"
            )
        template_bias = filtered["basal_encoder.0.bias"]
        native_bias = native_state["basal_encoder.0.bias"]
        native_weight = native_state["basal_encoder.0.weight"]
        if not torch.equal(template_bias, native_bias):
            raise ValueError(
                "template basal_encoder.0.bias does not match native_state's bias"
            )
        with torch.no_grad():
            state.basal_encoder[0].weight.copy_(native_weight)
            state.basal_encoder[0].bias.copy_(native_bias)
        for name in filtered:
            if name != "basal_encoder.0.bias":
                origins[prefix + name] = "inherited-template"

        adapter, loaded_pert_names = _build_adapter_from_template(
            template, int(hparams["pert_dim"])
        )
        perturbations = adapter
        for name in loaded_pert_names:
            origins["perturbations." + name] = "new-template"
        zero_final_layer(adapter)
        origins["perturbations.adapter.net.2.weight"] = "new-zero"
        origins["perturbations.adapter.net.2.bias"] = "new-zero"

        if variant in ("V2", "V3"):
            native_encoder = state.basal_encoder
            state.basal_encoder = SplitBasalEncoder(
                native_encoder, HVG_WIDTH, tx1_width, int(hparams["hidden_dim"])
            )
            state.input_dim = HVG_WIDTH + tx1_width
            # Wrapping renames every basal-encoder parameter; reclassify the
            # whole native encoder, not just its first layer, so a released
            # checkpoint with several encoder layers still builds. Layer 0 is
            # the one copied from native_state above; any deeper layer keeps
            # the template's (released) weights loaded by load_state_dict.
            for name in [
                key for key in origins if key.startswith(prefix + "basal_encoder.")
            ]:
                del origins[name]
            for name, _ in native_encoder.named_parameters():
                origins[prefix + "basal_encoder.native." + name] = (
                    "inherited-native"
                    if name.split(".")[0] == "0"
                    else "inherited-template"
                )
            origins[prefix + "basal_encoder.context.weight"] = "new-zero"
        else:
            origins[prefix + "basal_encoder.0.weight"] = "inherited-native"
            origins[prefix + "basal_encoder.0.bias"] = "inherited-native"

    state_adapter = StateForwardAdapter(state)
    if variant == "V0":
        backbone: nn.Module = ForwardOnlyStateModel(state_adapter, perturbations)
    elif variant == "V1":
        backbone = NullSubtractedBackbone(
            state_adapter, perturbations, hvg_width=HVG_WIDTH, state_takes_hvg=False
        )
    elif variant in ("V2-null", "V2"):
        backbone = NativeBackbone(state_adapter, perturbations, batch_index=batch_index)
    elif variant == "V3":
        backbone = NullSubtractedBackbone(
            state_adapter, perturbations, hvg_width=HVG_WIDTH, state_takes_hvg=True
        )
    else:  # N-native
        backbone = NativeBackbone(state_adapter, perturbations, batch_index=batch_index)

    all_names = [name for name, _ in backbone.named_parameters()]
    unclassified = [name for name in all_names if name not in origins]
    if unclassified:
        raise ValueError(f"unclassified backbone parameters: {unclassified}")

    backbone.requires_grad_(False)
    backbone.eval()
    actual = dict(backbone.named_parameters())
    for name in all_names:
        if origins[name].startswith("new"):
            actual[name].requires_grad_(True)

    parameters_report = {
        name: {
            "shape": list(p.shape),
            "origin": origins[name],
            "trainable": bool(p.requires_grad),
        }
        for name, p in actual.items()
    }
    trainable_count = sum(p.numel() for p in actual.values() if p.requires_grad)

    expected = (
        _TRAINABLE_COUNTS[variant] if expected_count == "table" else expected_count
    )
    if expected is not None and trainable_count != expected:
        raise ValueError(
            f"{variant} trainable parameter count {trainable_count} "
            f"!= expected {expected}"
        )

    report = {
        "variant": variant,
        "input_layout": INPUT_LAYOUT[variant],
        "state_input_dim": int(state.input_dim),
        "trainable_count": trainable_count,
        "parameters": parameters_report,
    }
    if variant in ("V1", "V3"):
        report["null_perturbation"] = "zero raw vector, same forward call"
    return backbone, report


def parameter_groups(backbone: nn.Module, lr: float) -> list[dict[str, Any]]:
    """One optimizer group over ``backbone``'s trainable parameters, name order."""
    params = [p for _, p in backbone.named_parameters() if p.requires_grad]
    if not params:
        raise ValueError("backbone has no trainable parameters")
    return [{"params": params, "lr": lr, "name": "interface"}]
