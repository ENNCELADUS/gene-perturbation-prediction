"""The GeneEffect heads (protocol §4): the joint model's nested low-rank head and the
flat MLP the readout fits, both reading explicit coverage masks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GeneEffectFeatureDims:
    """Per-block feature widths: ``delta_proj`` is ``Delta_{g,c}`` after the fixed
    4000 -> width projection; ``s`` six response statistics, the own-gene HVG-index
    shift last; ``q_sc`` the gene's ``[mean expr, fraction expressing, expr variance]``
    in the line's basal cells; ``e_g`` the ESM2 width; ``z_c`` the number of train-fit
    context PCA components (``model.context_components``)."""

    delta_proj: int = 256
    s: int = 6
    q_sc: int = 3
    e_g: int = 1280
    z_c: int = 128

    def __post_init__(self) -> None:
        for name in ("delta_proj", "s", "q_sc", "e_g", "z_c"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(
                    f"GeneEffectFeatureDims.{name} must be positive, got {value}"
                )


@dataclass(frozen=True)
class GeneEffectBlockConfig:
    """Whether STATE's two blocks, ``delta_proj`` and ``s``, feed ``h_delta``;
    ``q_sc``, ``e_g`` and ``z_c`` always do. A model without STATE turns both off
    (dropping ``delta_proj`` alone would leave STATE's own-gene shift in ``s``)."""

    use_delta_proj: bool = True
    use_s: bool = True

    def __post_init__(self) -> None:
        for name in ("use_delta_proj", "use_s"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"GeneEffectBlockConfig.{name} must be boolean")

    @property
    def enabled(self) -> tuple[str, ...]:
        """The blocks that feed ``h_delta``, in their canonical order."""
        optional = (("delta_proj", self.use_delta_proj), ("s", self.use_s))
        return (*(name for name, used in optional if used), "q_sc", "e_g", "z_c")


def _mlp(width: int, hidden: int, n_hidden_layers: int, out: int) -> nn.Sequential:
    """``n_hidden_layers`` x ``Linear -> LayerNorm -> GELU``, then ``Linear -> out``."""
    layers: list[nn.Module] = []
    for index in range(n_hidden_layers):
        layers += [
            nn.Linear(width if index == 0 else hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        ]
    layers.append(nn.Linear(hidden, out))
    return nn.Sequential(*layers)


class GeneEffectMLP(nn.Module):
    """Flat MLP ``delta_hat_{g,c} = h_delta(Delta_proj, s, q_sc, e_g, z_c)``.

    A block :class:`GeneEffectBlockConfig` disables takes ``None`` and adds no input
    width. Coverage is never a silent zero: ``q_sc_mask``, ``hvg_panel_mask`` and
    ``own_gene_shift_mask`` (a panel gene can still lack a shift) enter as input bits
    beside the zeroed values. The only per-line input is ``z_c``.
    """

    #: Mask channels the ``s`` block adds: hvg_panel_mask, own_gene_shift_mask.
    _S_BLOCK_MASK_BITS: int = 2
    #: Mask channels the ``q_sc`` block adds: q_sc_mask.
    _Q_SC_BLOCK_MASK_BITS: int = 1

    def __init__(
        self,
        dims: GeneEffectFeatureDims = GeneEffectFeatureDims(),
        blocks: GeneEffectBlockConfig = GeneEffectBlockConfig(),
        hidden: int = 256,
        n_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")

        self.dims = dims
        self.blocks = blocks
        self.hidden = int(hidden)
        width = 0
        if blocks.use_delta_proj:
            width += dims.delta_proj
        if blocks.use_s:
            width += dims.s + self._S_BLOCK_MASK_BITS
        width += dims.q_sc + self._Q_SC_BLOCK_MASK_BITS + dims.e_g + dims.z_c
        self.input_width = width
        self.net = _mlp(self.input_width, self.hidden, n_hidden_layers, 1)

    def forward(self, **blocks: torch.Tensor | None) -> torch.Tensor:
        """``delta_hat``, shape ``[batch]``; arguments as :func:`masked_blocks`."""
        x = torch.cat(
            list(masked_blocks(self.dims, self.blocks, **blocks).values()), -1
        )
        return self.net(x).squeeze(-1)


def _check_block(
    name: str, enabled: bool, value: torch.Tensor | None, expected_width: int
) -> torch.Tensor | None:
    if not enabled:
        if value is not None:
            raise ValueError(
                f"block {name!r} is disabled (blocks.use_{name}=False) but a "
                f"tensor was passed; pass None instead"
            )
        return None
    if value is None:
        raise ValueError(
            f"block {name!r} is enabled (blocks.use_{name}=True) but no "
            f"tensor was passed"
        )
    if value.dim() != 2 or value.shape[-1] != expected_width:
        raise ValueError(
            f"block {name!r} must be shaped [batch, {expected_width}], got "
            f"{tuple(value.shape)}"
        )
    return value


def _check_mask(
    name: str, required: bool, value: torch.Tensor | None, batch: int
) -> torch.Tensor | None:
    if not required:
        if value is not None:
            raise ValueError(
                f"mask {name!r} is not applicable (its block is disabled) but "
                f"a tensor was passed; pass None instead"
            )
        return None
    if value is None:
        raise ValueError(f"mask {name!r} is required but no tensor was passed")
    if tuple(value.shape) != (batch,):
        raise ValueError(
            f"mask {name!r} must be shaped ({batch},), got {tuple(value.shape)}"
        )
    return value


def masked_blocks(
    dims: GeneEffectFeatureDims,
    blocks: GeneEffectBlockConfig,
    *,
    delta_proj: torch.Tensor | None = None,
    s: torch.Tensor | None = None,
    q_sc: torch.Tensor | None = None,
    e_g: torch.Tensor | None = None,
    z_c: torch.Tensor | None = None,
    q_sc_mask: torch.Tensor | None = None,
    hvg_panel_mask: torch.Tensor | None = None,
    own_gene_shift_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """``{block name: [batch, width]}`` of each enabled block, in block order.

    A block's ``[batch, dims.<block>]`` tensor, and its ``[batch]`` masks, must be
    ``None`` exactly when ``blocks`` disables it, so a disabled block cannot reach the
    forward pass. Where ``own_gene_shift_mask`` or ``q_sc_mask`` is ``False``, ``s``'s
    last column (the own-gene shift) or ``q_sc`` is zeroed here; every mask bit is
    appended, so a missing value is never a genuine zero. Raises ``ValueError`` on a
    presence mismatch, a wrong shape or no tensor at all.
    """
    provided = [t for t in (delta_proj, s, q_sc, e_g, z_c) if t is not None]
    if not provided:
        raise ValueError("forward() received no tensors; cannot infer batch size")
    batch = provided[0].shape[0]

    delta_proj = _check_block(
        "delta_proj", blocks.use_delta_proj, delta_proj, dims.delta_proj
    )
    s = _check_block("s", blocks.use_s, s, dims.s)
    q_sc = _check_block("q_sc", True, q_sc, dims.q_sc)
    e_g = _check_block("e_g", True, e_g, dims.e_g)
    z_c = _check_block("z_c", True, z_c, dims.z_c)

    q_sc_mask = _check_mask("q_sc_mask", True, q_sc_mask, batch)
    hvg_panel_mask = _check_mask("hvg_panel_mask", blocks.use_s, hvg_panel_mask, batch)
    own_gene_shift_mask = _check_mask(
        "own_gene_shift_mask", blocks.use_s, own_gene_shift_mask, batch
    )

    parts: dict[str, torch.Tensor] = {}
    if blocks.use_delta_proj:
        parts["delta_proj"] = delta_proj
    if blocks.use_s:
        own_gate = own_gene_shift_mask.to(dtype=s.dtype).unsqueeze(-1)
        own_shift = s[:, -1:] * own_gate
        parts["s"] = torch.cat(
            [
                s[:, :-1],
                own_shift,
                hvg_panel_mask.to(dtype=s.dtype).unsqueeze(-1),
                own_gene_shift_mask.to(dtype=s.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
    q_gate = q_sc_mask.to(dtype=q_sc.dtype).unsqueeze(-1)
    parts["q_sc"] = torch.cat(
        [q_sc * q_gate, q_sc_mask.to(dtype=q_sc.dtype).unsqueeze(-1)], dim=-1
    )
    parts["e_g"] = e_g
    parts["z_c"] = z_c
    return parts


class SwiGLU(nn.Module):
    """``out((x W_1) * SiLU(x W_2))`` with dropout on the gated hidden layer.

    The output layer is zero-initialised, so the module starts as the zero
    function and its contribution grows from there.
    """

    def __init__(self, width: int, hidden: int, out: int, dropout: float) -> None:
        super().__init__()
        self.value = nn.Linear(width, hidden)
        self.gate = nn.Linear(width, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.value(x) * nn.functional.silu(self.gate(x))
        return self.out(self.dropout(hidden))


#: Width of each per-(g, c) block's encoder in the correction ``h``. ``s`` and
#: ``q_sc`` enter with their mask channels.
CORRECTION_ENCODER_WIDTHS = {"delta_proj": 32, "s": 16, "q_sc": 16, "e_g": 32}
#: Hidden width of the correction's SwiGLU layer.
CORRECTION_HIDDEN = 64
#: Hidden width of the context tower's residual SwiGLU branch.
CONTEXT_HIDDEN = 128


class GeneEffectNestedHead(nn.Module):
    """Nested low-rank head: a gene x context product plus a per-row correction.

    ``head(g, c) = <G(g), C(z_c)> / sqrt(rank) + h(q_sc, s, delta_proj, e_g)``, in
    units of the per-gene training residual SD. ``G(g)`` is a free embedding of the
    gene's position in ``inputs.genes`` plus a linear map of ``e_g``. ``C(z) = W z +
    SwiGLU(z)`` reads only the eigen-scaled context PCA scores; ``W`` and the SwiGLU
    output layers start at zero (``W``'s gradient arrives first through ``G``). ``h``
    encodes each enabled per-(g, c) block (``Linear -> LayerNorm``), then ``Dropout
    -> SwiGLU -> 1``; it never sees ``z_c`` and is ``None`` without such a block.
    There is no per-line lookup: the only per-line signal is ``z_c``.
    """

    def __init__(
        self,
        dims: GeneEffectFeatureDims,
        blocks: GeneEffectBlockConfig,
        *,
        n_genes: int,
        factor_rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_genes < 1:
            raise ValueError(f"n_genes must be positive, got {n_genes}")
        if factor_rank < 1:
            raise ValueError(f"factor_rank must be positive, got {factor_rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.dims = dims
        self.blocks = blocks
        self.n_genes = int(n_genes)
        self.factor_rank = int(factor_rank)
        self.gene_embedding = nn.Embedding(self.n_genes, self.factor_rank)
        nn.init.normal_(self.gene_embedding.weight, std=0.02)
        self.gene_projection = nn.Linear(dims.e_g, self.factor_rank)
        self.context_linear = nn.Linear(dims.z_c, self.factor_rank)
        # C starts at zero: with h's zero output layer the head predicts zero before
        # the first update, so a stack equals its prior there.
        nn.init.zeros_(self.context_linear.weight)
        nn.init.zeros_(self.context_linear.bias)
        self.context_residual = SwiGLU(
            dims.z_c, CONTEXT_HIDDEN, self.factor_rank, dropout
        )
        inputs = {
            "delta_proj": dims.delta_proj,
            "s": dims.s + GeneEffectMLP._S_BLOCK_MASK_BITS,
            "q_sc": dims.q_sc + GeneEffectMLP._Q_SC_BLOCK_MASK_BITS,
            "e_g": dims.e_g,
        }
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(width, CORRECTION_ENCODER_WIDTHS[name]),
                    nn.LayerNorm(CORRECTION_ENCODER_WIDTHS[name]),
                )
                for name, width in inputs.items()
                if name in blocks.enabled
            }
        )
        encoded = sum(CORRECTION_ENCODER_WIDTHS[name] for name in self.encoders)
        self.correction = (
            nn.Sequential(
                nn.Dropout(dropout), SwiGLU(encoded, CORRECTION_HIDDEN, 1, 0.0)
            )
            if self.encoders
            else None
        )

    def forward(
        self, *, gene_index: torch.Tensor, **blocks: torch.Tensor | None
    ) -> torch.Tensor:
        """``delta_hat`` in residual-SD units, ``[batch]``; ``gene_index`` is each row's
        integer position in ``inputs.genes``, the blocks as :func:`masked_blocks`."""
        parts = masked_blocks(self.dims, self.blocks, **blocks)
        context = parts["z_c"]
        batch = context.shape[0]
        if (
            tuple(gene_index.shape) != (batch,)
            or gene_index.is_floating_point()
            or gene_index.dtype == torch.bool
        ):
            raise ValueError(
                f"gene_index must be an integer tensor shaped ({batch},), got "
                f"shape={tuple(gene_index.shape)} dtype={gene_index.dtype}"
            )
        gene = self.gene_embedding(gene_index) + self.gene_projection(parts["e_g"])
        context = self.context_linear(context) + self.context_residual(context)
        prediction = (gene * context).sum(dim=-1) / math.sqrt(self.factor_rank)
        if self.correction is not None:
            encoded = torch.cat(
                [encoder(parts[name]) for name, encoder in self.encoders.items()], -1
            )
            prediction = prediction + self.correction(encoded).squeeze(-1)
        return prediction
