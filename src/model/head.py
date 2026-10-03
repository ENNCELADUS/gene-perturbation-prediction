"""model / head."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class GeneEffectFeatureDims:
    """Per-block feature widths (``docs/03-geneeffect-protocol.md`` §4).

    Attributes:
        delta_proj: ``Delta_{g,c}`` projected 4000 -> this width by a fixed
            seeded random projection (computed upstream; this module only
            consumes the result).
        s: Six distribution statistics and unprojected interpretables, with
            the own-gene HVG-index shift as the last channel.
        q_sc: ``[mean expr, fraction expressing, expr variance]`` of gene
            ``g`` in line ``c``, from basal single cells.
        e_g: ESM2 protein embedding width.
        z_c: Raw moment-pooled Tx1 basal-context width (mean + variance of
            the 2560-d embedding); no PCA is applied.
    """

    delta_proj: int = 256
    s: int = 6
    q_sc: int = 3
    e_g: int = 1280
    z_c: int = 5120

    def __post_init__(self) -> None:
        for name in ("delta_proj", "s", "q_sc", "e_g", "z_c"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(
                    f"GeneEffectFeatureDims.{name} must be positive, got {value}"
                )


@dataclass(frozen=True)
class GeneEffectBlockConfig:
    """Which of the five feature blocks feed ``h_delta``: one flag per ablation.

    The Phase 7 "virtual-cell ablation" removes ``delta_proj`` and ``s``
    together (``use_delta_proj=False, use_s=False``) -- dropping
    ``delta_proj`` alone would leave ST-derived signal (own-gene shift, the
    ``s`` distribution statistics) in the model. This dataclass does not
    enforce that pairing; it is a config flag, applied by the caller
    building each ablation's config, not a code edit to this module.
    """

    use_delta_proj: bool = True
    use_s: bool = True
    use_q_sc: bool = True
    use_e_g: bool = True
    use_z_c: bool = True

    def __post_init__(self) -> None:
        for name in ("use_delta_proj", "use_s", "use_q_sc", "use_e_g", "use_z_c"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"GeneEffectBlockConfig.{name} must be boolean")
        if not any(
            (
                self.use_delta_proj,
                self.use_s,
                self.use_q_sc,
                self.use_e_g,
                self.use_z_c,
            )
        ):
            raise ValueError("GeneEffectBlockConfig must enable at least one block")


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
    """MLP predicting ``delta_hat(g, c)`` from up to five feature blocks.

    ``delta_hat_{g,c} = h_delta(Delta_proj, s, q_sc, e_g, z_c)``
    (``docs/03-geneeffect-protocol.md`` §4). Every block is gated
    by :class:`GeneEffectBlockConfig`; a disabled block's tensor argument to
    :meth:`forward` must be ``None`` and contributes nothing to the input
    width or the parameter count -- each of the five ablations is therefore
    a config flag, not a code edit.

    Coverage is never signalled by a silently zero-filled value channel.
    Three partial-coverage conditions each get an explicit boolean mask
    input, concatenated into the net's input alongside the (zeroed-where-
    missing) values so the two are never confused:

    - ``q_sc_mask``: whether ``q_sc`` (mean expr / fraction expressing /
      expr variance) is available for this ``(g, c)``. Required whenever
      ``use_q_sc`` is enabled.
    - ``hvg_panel_mask``: whether gene ``g`` is in the 2000-gene HVG panel.
      Required whenever ``use_s`` is enabled (own-gene shift is only
      defined for panel genes).
    - ``own_gene_shift_mask``: whether the own-gene HVG-index shift value
      (the **last** channel of ``s``) is itself available. Required
      whenever ``use_s`` is enabled; kept distinct from
      ``hvg_panel_mask`` because a panel gene can still lack a computable
      shift (e.g. too few cells) even when panel membership holds.

    There is no cell-line-ID (or any line-identifying) input anywhere: the
    only per-line signal is ``z_c``, a continuous basal-population
    embedding defined for any line, never a per-line lookup table.

    Attributes:
        dims: Configured per-block feature widths.
        blocks: Configured per-block enable flags.
        input_width: Total net input width (enabled block widths + their
            mask-bit channels).
    """

    #: Number of explicit coverage-mask channels contributed by the ``s``
    #: block when enabled: hvg_panel_mask, own_gene_shift_mask.
    _S_BLOCK_MASK_BITS: int = 2
    #: Number of explicit coverage-mask channels contributed by the
    #: ``q_sc`` block when enabled: q_sc_mask.
    _Q_SC_BLOCK_MASK_BITS: int = 1

    def __init__(
        self,
        dims: GeneEffectFeatureDims = GeneEffectFeatureDims(),
        blocks: GeneEffectBlockConfig = GeneEffectBlockConfig(),
        hidden: int = 256,
        n_hidden_layers: int = 2,
    ) -> None:
        """Initialize the head.

        Args:
            dims: Per-block feature widths, see :class:`GeneEffectFeatureDims`.
            blocks: Per-block enable flags, see :class:`GeneEffectBlockConfig`.
            hidden: Hidden width of every layer of the MLP trunk.
            n_hidden_layers: Number of ``Linear -> LayerNorm -> GELU``
                hidden layers before the final scalar projection. Must be
                >= 1.

        Raises:
            ValueError: If ``hidden`` or ``n_hidden_layers`` is
                non-positive (``dims``/``blocks`` validate themselves).
        """
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
        if blocks.use_q_sc:
            width += dims.q_sc + self._Q_SC_BLOCK_MASK_BITS
        if blocks.use_e_g:
            width += dims.e_g
        if blocks.use_z_c:
            width += dims.z_c
        self.input_width = width
        self.net = _mlp(self.input_width, self.hidden, n_hidden_layers, 1)

    def _check_block(
        self,
        name: str,
        enabled: bool,
        value: torch.Tensor | None,
        expected_width: int,
    ) -> torch.Tensor | None:
        """Validate one block tensor against its enable flag and width."""
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
        self, name: str, required: bool, value: torch.Tensor | None, batch: int
    ) -> torch.Tensor | None:
        """Validate one boolean coverage-mask tensor."""
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

    def forward(self, **blocks: torch.Tensor | None) -> torch.Tensor:
        """``delta_hat``, shape ``[batch]``; arguments as :meth:`block_inputs`."""
        x = torch.cat(list(self.block_inputs(**blocks).values()), dim=-1)
        return self.net(x).squeeze(-1)

    def block_inputs(
        self,
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
        """Validated, masked net input of each enabled block, in block order.

        Every argument is keyword-only. A block's tensor must be ``None``
        iff that block is disabled in ``self.blocks`` (enforced, not just
        ignored, so a disabled block provably cannot influence the
        forward pass). Wherever a coverage mask is ``False``, the
        corresponding value channel(s) are zeroed **inside this method**
        before entering the net -- callers are not required to pre-zero
        them -- but the mask bit itself is always concatenated as an
        explicit feature, so a masked-missing value is never
        indistinguishable from a genuine zero.

        Args:
            delta_proj: ``[batch, dims.delta_proj]`` if ``blocks.use_delta_proj``,
                else ``None``.
            s: ``[batch, dims.s]`` if ``blocks.use_s``, else ``None``. The
                last column is the own-gene HVG-index shift.
            q_sc: ``[batch, dims.q_sc]`` if ``blocks.use_q_sc``, else ``None``.
            e_g: ``[batch, dims.e_g]`` if ``blocks.use_e_g``, else ``None``.
            z_c: ``[batch, dims.z_c]`` if ``blocks.use_z_c``, else ``None``.
            q_sc_mask: ``[batch]`` bool/0-1, required iff ``blocks.use_q_sc``.
            hvg_panel_mask: ``[batch]`` bool/0-1, required iff ``blocks.use_s``.
            own_gene_shift_mask: ``[batch]`` bool/0-1, required iff
                ``blocks.use_s``.

        Returns:
            ``{block name: [batch, width]}``; ``s`` and ``q_sc`` carry their
            mask-bit channels.

        Raises:
            ValueError: On a block/mask presence mismatch with ``self.blocks``,
                a wrong tensor shape, or no tensor provided at all (batch
                size cannot be inferred).
        """
        blocks = self.blocks
        dims = self.dims
        provided = [t for t in (delta_proj, s, q_sc, e_g, z_c) if t is not None]
        if not provided:
            raise ValueError("forward() received no tensors; cannot infer batch size")
        batch = provided[0].shape[0]

        delta_proj = self._check_block(
            "delta_proj", blocks.use_delta_proj, delta_proj, dims.delta_proj
        )
        s = self._check_block("s", blocks.use_s, s, dims.s)
        q_sc = self._check_block("q_sc", blocks.use_q_sc, q_sc, dims.q_sc)
        e_g = self._check_block("e_g", blocks.use_e_g, e_g, dims.e_g)
        z_c = self._check_block("z_c", blocks.use_z_c, z_c, dims.z_c)

        q_sc_mask = self._check_mask("q_sc_mask", blocks.use_q_sc, q_sc_mask, batch)
        hvg_panel_mask = self._check_mask(
            "hvg_panel_mask", blocks.use_s, hvg_panel_mask, batch
        )
        own_gene_shift_mask = self._check_mask(
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
        if blocks.use_q_sc:
            q_gate = q_sc_mask.to(dtype=q_sc.dtype).unsqueeze(-1)
            parts["q_sc"] = torch.cat(
                [q_sc * q_gate, q_sc_mask.to(dtype=q_sc.dtype).unsqueeze(-1)], dim=-1
            )
        if blocks.use_e_g:
            parts["e_g"] = e_g
        if blocks.use_z_c:
            parts["z_c"] = z_c
        return parts


class GeneEffectResidualHead(nn.Module):
    """Factorised gene x context head over the five-block MLP.

    ``head(g, c) = MLP(F) + <G(g), C(g, c)> / sqrt(rank)``. ``MLP`` is
    :class:`GeneEffectMLP` over every enabled block. ``G(g)`` is a free
    per-gene embedding plus, when ``e_g`` is enabled, a linear map of the
    ESM2 embedding. ``C(g, c)`` is an MLP over the enabled context blocks
    (``z_c``, ``q_sc``, ``delta_proj``, ``s``, each masked exactly as the MLP
    sees it), so a gene x context effect is one explicit inner product
    rather than something the MLP must build from additive pieces.

    The free embedding is indexed by gene position in ``inputs.genes``; there
    is still no per-line lookup: the only per-line signal is ``z_c``.

    Attributes:
        trunk: The additive five-block MLP.
        gene_embedding: Free ``[n_genes, factor_rank]`` gene factors.
        gene_projection: ``Linear(e_g -> factor_rank)``, or ``None`` when
            ``e_g`` is disabled.
        context: MLP ``context_width -> factor_rank``.
        context_width: Width of the context MLP's input.
    """

    def __init__(
        self,
        dims: GeneEffectFeatureDims,
        blocks: GeneEffectBlockConfig,
        hidden: int,
        n_hidden_layers: int,
        n_genes: int,
        factor_rank: int,
    ) -> None:
        """Initialize the head.

        Args:
            dims: Per-block feature widths.
            blocks: Per-block enable flags.
            hidden: Hidden width of the MLP trunk and the context MLP.
            n_hidden_layers: Hidden layers of the MLP trunk and the context MLP.
            n_genes: Rows of the free gene embedding (``len(inputs.genes)``).
            factor_rank: Width of the gene and context factors.

        Raises:
            ValueError: If ``n_genes`` or ``factor_rank`` is non-positive, or
                no context block is enabled.
        """
        super().__init__()
        if n_genes < 1:
            raise ValueError(f"n_genes must be positive, got {n_genes}")
        if factor_rank < 1:
            raise ValueError(f"factor_rank must be positive, got {factor_rank}")
        self.trunk = GeneEffectMLP(dims, blocks, hidden, n_hidden_layers)
        self.dims = dims
        self.blocks = blocks
        self.n_genes = int(n_genes)
        self.factor_rank = int(factor_rank)
        self.context_width = self.trunk.input_width - (
            dims.e_g if blocks.use_e_g else 0
        )
        if self.context_width == 0:
            raise ValueError(
                "the factorised head needs a context block (delta_proj, s, q_sc or z_c)"
            )
        self.gene_embedding = nn.Embedding(self.n_genes, self.factor_rank)
        nn.init.normal_(self.gene_embedding.weight, std=0.02)
        self.gene_projection = (
            nn.Linear(dims.e_g, self.factor_rank) if blocks.use_e_g else None
        )
        self.context = _mlp(
            self.context_width, self.trunk.hidden, n_hidden_layers, self.factor_rank
        )

    def forward(
        self, *, gene_index: torch.Tensor, **blocks: torch.Tensor | None
    ) -> torch.Tensor:
        """Predict ``delta_hat`` for a batch of ``(gene, context)`` rows.

        Args:
            gene_index: ``[batch]`` integer position of each row's gene in
                ``inputs.genes``.
            **blocks: Block tensors and coverage masks, validated and masked
                exactly as :meth:`GeneEffectMLP.block_inputs`.

        Returns:
            ``delta_hat``, shape ``[batch]``.

        Raises:
            ValueError: As :meth:`GeneEffectMLP.block_inputs`, or when
                ``gene_index`` is not an integer ``[batch]`` tensor.
        """
        parts = self.trunk.block_inputs(**blocks)
        trunk = self.trunk.net(torch.cat(list(parts.values()), dim=-1)).squeeze(-1)
        batch = trunk.shape[0]
        if (
            tuple(gene_index.shape) != (batch,)
            or gene_index.is_floating_point()
            or gene_index.dtype == torch.bool
        ):
            raise ValueError(
                f"gene_index must be an integer tensor shaped ({batch},), got "
                f"shape={tuple(gene_index.shape)} dtype={gene_index.dtype}"
            )
        gene = self.gene_embedding(gene_index)
        if self.gene_projection is not None:
            gene = gene + self.gene_projection(parts["e_g"])
        context = self.context(
            torch.cat([value for name, value in parts.items() if name != "e_g"], dim=-1)
        )
        return trunk + (gene * context).sum(dim=-1) / math.sqrt(self.factor_rank)
