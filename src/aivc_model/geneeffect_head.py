"""Exp13 GeneEffect residual head: axis-aware loss + five-block ``h_delta``.

Blocker B2 (``merry-mapping-comet.md``): ``tx1_geneeffect_head.rank_variance_loss``
``reshape(-1)``s a batch to ``[B]`` and computes one Pearson correlation over
mixed genes and cell-line contexts. The 226-line benchmark's primary metric is
macro **per-gene Spearman across cell lines** (``residual_metrics.py``), so
that flattened objective can score well while the per-gene axis is near-zero
whenever the per-gene main effect ``mu_g`` (large, shared between ``pred`` and
``target``) dominates a small per-context signal -- exactly the failure mode
``residual_metrics.py``'s module docstring documents for the raw-GeneEffect
per-line axis. :func:`per_gene_rank_variance_loss` fixes this by computing
:func:`~aivc_model.tx1_geneeffect_head.correlation_loss` and
:func:`~aivc_model.tx1_geneeffect_head.variance_matching_loss` **per gene,
along the context axis**, then macro-averaging over genes -- mirroring
:func:`aivc_model.residual_metrics.per_gene_spearman`'s axis choice, not
``rank_variance_loss``'s.

This module also implements the Phase 6 five-block ``h_delta`` readout
(:class:`GeneEffectResidualHead`): ``delta_hat(g, c) = h_delta(Delta_proj,
s, q_sc, e_g, z_c)``. Every block is individually disableable (one config
flag per block ablation) and every partial-coverage input -- ``q_sc``, HVG
panel membership, own-gene shift -- is signalled by an explicit boolean mask
input, never by silently zero-filling a value channel. There is no cell-line
identity input anywhere: ``z_c`` is a continuous basal-population embedding,
defined for any line, not a per-line lookup.

Pure torch/numpy (plus pandas/scipy transitively, only inside
:func:`macro_per_gene_spearman`, to reuse :mod:`aivc_model.residual_metrics`
rather than reimplement its NaN/undefined-correlation handling). No file
I/O, no provenance machinery, no GPU assumptions, no training loop -- see
``scripts/train_geneeffect_head.py`` (Phase 6, not built here) for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import pandas as pd
import torch
from torch import nn

from aivc_model.residual_metrics import per_gene_spearman

#: Numerical floor added to per-gene standard deviations before dividing,
#: matching ``tx1_geneeffect_head._STD_EPS`` -- keeps a per-gene correlation
#: finite (not NaN) when a *prediction* row collapses to (near-)constant.
_STD_EPS: float = 1e-6

#: Threshold below which a gene's TARGET standard deviation (across the
#: context axis) is treated as "(near-)constant" and therefore excluded from
#: the macro-average (Spearman/Pearson is mathematically undefined for a
#: constant reference vector; see module docstring and
#: ``residual_metrics.py``'s identical per-unit convention).
_CONSTANT_TARGET_STD_EPS: float = 1e-6

__all__ = [
    "moment_pool",
    "per_gene_rank_variance_loss",
    "PerGeneRankVarianceLoss",
    "macro_per_gene_spearman",
    "MacroPerGeneSpearman",
    "GeneEffectFeatureDims",
    "GeneEffectBlockConfig",
    "GeneEffectResidualHead",
]


# ---------------------------------------------------------------------------
# Pi = [mean, var]: permutation-invariant moment pooling (Stage 2 input prep)
# ---------------------------------------------------------------------------


def moment_pool(
    bag: torch.Tensor,
    mask: torch.Tensor | None = None,
    moments: int = 2,
) -> torch.Tensor:
    """Pool a per-cell bag into a permutation-invariant moment summary.

    ``Pi = [mean, var]`` (``01-blueprint.md`` §3, ``merry-mapping-comet.md``
    Target architecture): the default ``moments=2`` is exactly the pair used
    to build ``z_c = Pi({x_c^(i)})`` and ``Delta_{g,c} = Pi(B_hat_{c,g}) -
    Pi({b_c^(i)})``.

    Args:
        bag: Per-cell tensor of shape ``[N, D]`` (cells, feature dim).
        mask: Optional boolean or 0/1 tensor of shape ``[N]``. Rows where
            ``mask`` is falsy are excluded from every moment (padded rows).
            If ``None``, every row is used.
        moments: Number of moments to compute, in order:
            1 -> mean only, shape ``[D]``.
            2 (default) -> concat(mean, variance), shape ``[2D]``. Variance
                is the population (biased, ``unbiased=False``) variance.
            3 -> adds the standard deviation, shape ``[3D]``.
            4 -> adds the (Fisher, population) skewness, shape ``[4D]``.
            Higher moments are not implemented; callers needing kurtosis or
            beyond should extend this function rather than call it with
            ``moments > 4``.

    Returns:
        A 1-D tensor of length ``D * moments``, invariant to row order of
        ``bag`` (and, when masked, invariant to the order/placement of the
        masked-out padding rows).

    Raises:
        ValueError: If ``moments`` is not in ``{1, 2, 3, 4}``, if ``bag`` is
            not 2-D, or if every row is masked out (empty bag).
    """
    if bag.dim() != 2:
        msg = f"bag must be 2-D [N, D], got shape {tuple(bag.shape)}"
        raise ValueError(msg)
    if moments not in (1, 2, 3, 4):
        msg = f"moments must be one of {{1, 2, 3, 4}}, got {moments}"
        raise ValueError(msg)

    if mask is not None:
        keep = mask.to(dtype=torch.bool, device=bag.device)
        if not bool(keep.any()):
            msg = "moment_pool received an empty bag (mask excludes all rows)"
            raise ValueError(msg)
        bag = bag[keep]

    n_rows = bag.shape[0]
    if n_rows == 0:
        msg = "moment_pool received an empty bag (0 rows)"
        raise ValueError(msg)

    mean = bag.mean(dim=0)
    parts = [mean]
    if moments >= 2:
        centered = bag - mean.unsqueeze(0)
        variance = centered.square().mean(dim=0)
        parts.append(variance)
    if moments >= 3:
        std = (variance + _STD_EPS).sqrt()
        parts.append(std)
    if moments >= 4:
        skew = (centered.pow(3).mean(dim=0)) / (std.pow(3) + _STD_EPS)
        parts.append(skew)
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# B2: axis-aware training loss
# ---------------------------------------------------------------------------


class PerGeneRankVarianceLoss(NamedTuple):
    """Result of :func:`per_gene_rank_variance_loss`.

    Attributes:
        loss: 0-D scalar tensor, gradient-bearing w.r.t. ``pred``: the mean
            of ``correlation_loss + lam * variance_matching_loss`` over
            genes whose target is not (near-)constant across contexts.
        n_genes_scored: Number of gene rows that entered the mean.
        n_genes_excluded: Number of gene rows excluded for having a
            (near-)constant target across the context axis (undefined
            correlation) -- never folded into ``loss`` as 0 or NaN.
    """

    loss: torch.Tensor
    n_genes_scored: int
    n_genes_excluded: int


def per_gene_rank_variance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float = 1.0,
) -> PerGeneRankVarianceLoss:
    """Axis-aware replacement for ``tx1_geneeffect_head.rank_variance_loss``.

    ``rank_variance_loss`` flattens its ``[B]`` inputs to one Pearson
    correlation over whatever the batch happens to mix (genes, contexts, or
    both). This function instead requires ``pred``/``target`` shaped
    ``[n_genes, n_contexts]`` and computes
    :func:`~aivc_model.tx1_geneeffect_head.correlation_loss` and
    :func:`~aivc_model.tx1_geneeffect_head.variance_matching_loss`
    independently **per row (gene), along the context axis (dim=-1)**, then
    macro-averages the per-gene combined loss over genes -- the same axis
    choice as :func:`aivc_model.residual_metrics.per_gene_spearman`, so the
    training objective matches the benchmark's scored metric.

    A gene whose target is (near-)constant across contexts has an
    undefined correlation (a constant reference vector has no direction to
    correlate against) and is excluded from the mean rather than
    contributing a manufactured 0 or a NaN that would poison the whole
    batch's gradient -- mirroring ``residual_metrics.py``'s per-unit NaN
    convention. A gene whose *prediction* row collapses to (near-)constant
    is not excluded: like ``correlation_loss``, its standard deviation is
    floored at ``_STD_EPS`` before dividing, so it scores the finite worst
    case (correlation loss ~1) instead of being dropped -- collapse must be
    punished, not hidden from the gradient.

    Args:
        pred: Predictions, shape ``[n_genes, n_contexts]``.
        target: Targets, shape ``[n_genes, n_contexts]``, same shape as
            ``pred``.
        lam: Weight on the per-gene variance-matching term.

    Returns:
        A :class:`PerGeneRankVarianceLoss`.

    Raises:
        ValueError: If ``pred``/``target`` are not both 2-D of equal shape,
            or if every gene's target is (near-)constant (the macro-average
            would be over an empty set).
    """
    if pred.dim() != 2 or tuple(pred.shape) != tuple(target.shape):
        raise ValueError(
            "pred and target must both be 2-D [n_genes, n_contexts] of equal "
            f"shape; got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    target = target.to(device=pred.device, dtype=pred.dtype)
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError(
            "per_gene_rank_variance_loss requires finite pred/target; this "
            "loss operates on a dense labeled batch -- filter unlabeled "
            "(gene, context) cells out before calling, do not pass NaN"
        )

    n_genes = pred.shape[0]
    target_std = target.std(dim=-1, unbiased=False)
    defined = target_std > _CONSTANT_TARGET_STD_EPS
    n_excluded = int((~defined).sum().item())
    if not bool(defined.any()):
        raise ValueError(
            "per_gene_rank_variance_loss: every gene's target is "
            "(near-)constant across contexts; the per-gene macro-average is "
            "undefined for this batch"
        )

    pred_defined = pred[defined]
    target_defined = target[defined]

    pred_centered = pred_defined - pred_defined.mean(dim=-1, keepdim=True)
    target_centered = target_defined - target_defined.mean(dim=-1, keepdim=True)
    covariance = (pred_centered * target_centered).mean(dim=-1)
    pred_std = (pred_centered.square().mean(dim=-1) + _STD_EPS).sqrt()
    target_std_defined = (target_centered.square().mean(dim=-1) + _STD_EPS).sqrt()
    correlation = covariance / (pred_std * target_std_defined)
    correlation_loss_per_gene = 1.0 - correlation

    pred_row_std = pred_defined.std(dim=-1, unbiased=False)
    target_row_std = target_defined.std(dim=-1, unbiased=False)
    variance_loss_per_gene = (pred_row_std - target_row_std).square()

    per_gene_loss = correlation_loss_per_gene + float(lam) * variance_loss_per_gene
    loss = per_gene_loss.mean()
    return PerGeneRankVarianceLoss(
        loss=loss,
        n_genes_scored=n_genes - n_excluded,
        n_genes_excluded=n_excluded,
    )


# ---------------------------------------------------------------------------
# B2: selection/reporting metric (reuses residual_metrics, does not
# reimplement its NaN/undefined-correlation handling)
# ---------------------------------------------------------------------------


class MacroPerGeneSpearman(NamedTuple):
    """Result of :func:`macro_per_gene_spearman`.

    Attributes:
        macro: ``nanmean`` of ``per_gene`` -- NaN (never 0.0, never raised)
            if every gene is undefined.
        per_gene: Spearman rho per gene (index: ``gene_ids`` or a default
            positional index), NaN where undefined.
        n_scored: Number of genes with a defined correlation.
        n_undefined: Number of genes with an undefined (NaN) correlation.
    """

    macro: float
    per_gene: pd.Series
    n_scored: int
    n_undefined: int


def macro_per_gene_spearman(
    pred: torch.Tensor,
    target: torch.Tensor,
    gene_ids: Sequence[str] | None = None,
) -> MacroPerGeneSpearman:
    """Macro per-gene Spearman across contexts -- the selection/reporting metric.

    This is a thin torch-tensor adapter over
    :func:`aivc_model.residual_metrics.per_gene_spearman`: it reshapes
    ``[n_genes, n_contexts]`` into that module's long-form frame and reuses
    its per-gene grouping, NaN-on-undefined convention (constant truth or
    prediction, or fewer than
    ``aivc_model.residual_metrics.MIN_OBSERVATIONS`` finite pairs), and
    ``nanmean`` aggregation, rather than reimplementing them. Use this for
    checkpoint selection / reporting; use :func:`per_gene_rank_variance_loss`
    (a differentiable surrogate) for training.

    Args:
        pred: Predictions, shape ``[n_genes, n_contexts]``.
        target: Targets, shape ``[n_genes, n_contexts]``, same shape as
            ``pred``.
        gene_ids: Optional length-``n_genes`` gene labels for the returned
            ``per_gene`` index. Defaults to ``"0", "1", ...`` positional ids.

    Returns:
        A :class:`MacroPerGeneSpearman`.

    Raises:
        ValueError: If ``pred``/``target`` are not both 2-D of equal shape,
            or ``gene_ids`` is given with the wrong length.
    """
    if pred.dim() != 2 or tuple(pred.shape) != tuple(target.shape):
        raise ValueError(
            "pred and target must both be 2-D [n_genes, n_contexts] of equal "
            f"shape; got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    n_genes, n_contexts = pred.shape
    if gene_ids is None:
        gene_ids = [str(i) for i in range(n_genes)]
    elif len(gene_ids) != n_genes:
        raise ValueError(
            f"gene_ids must have length n_genes={n_genes}, got {len(gene_ids)}"
        )

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    frame = pd.DataFrame(
        {
            "gene_symbol": [g for g in gene_ids for _ in range(n_contexts)],
            "context_id": list(range(n_contexts)) * n_genes,
            "truth": target_np.reshape(-1),
            "pred": pred_np.reshape(-1),
        }
    )
    per_gene = per_gene_spearman(frame, truth_col="truth", pred_col="pred")
    finite = per_gene.dropna()
    macro = float(finite.mean()) if len(finite) else float("nan")
    return MacroPerGeneSpearman(
        macro=macro,
        per_gene=per_gene,
        n_scored=int(len(finite)),
        n_undefined=int(per_gene.isna().sum()),
    )


# ---------------------------------------------------------------------------
# Phase 6: five-block h_delta
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneEffectFeatureDims:
    """Per-block feature widths (target architecture, ``merry-mapping-comet.md``).

    Attributes:
        delta_proj: ``Delta_{g,c}`` projected 4000 -> this width by a fixed
            seeded random projection (computed upstream; this module only
            consumes the result).
        s: Distribution statistics + unprojected interpretables (energy
            distance to the basal bag, cross-cell dispersion, fraction
            beyond threshold, ``||Delta_mean||``, cosine to basal mean, and
            the own-gene HVG-index shift as the last channel).
        q_sc: ``[mean expr, fraction expressing, expr variance]`` of gene
            ``g`` in line ``c``, from basal single cells.
        e_g: ESM2 protein embedding width.
        z_c: Reduced Tx1 basal-context embedding width (PCs fit on train
            lines only, upstream of this module).
    """

    delta_proj: int = 256
    s: int = 20
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


class GeneEffectResidualHead(nn.Module):
    """MLP predicting ``delta_hat(g, c)`` from up to five feature blocks.

    ``delta_hat_{g,c} = h_delta(Delta_proj, s, q_sc, e_g, z_c)``
    (``merry-mapping-comet.md``, Target architecture). Every block is gated
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

        layers: list[nn.Module] = [
            nn.Linear(self.input_width, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.GELU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers += [
                nn.Linear(self.hidden, self.hidden),
                nn.LayerNorm(self.hidden),
                nn.GELU(),
            ]
        layers.append(nn.Linear(self.hidden, 1))
        self.net = nn.Sequential(*layers)

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

    def forward(
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
    ) -> torch.Tensor:
        """Predict ``delta_hat`` for a batch of ``(gene, context)`` rows.

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
            ``delta_hat``, shape ``[batch]``.

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

        parts: list[torch.Tensor] = []
        if blocks.use_delta_proj:
            parts.append(delta_proj)
        if blocks.use_s:
            own_gate = own_gene_shift_mask.to(dtype=s.dtype).unsqueeze(-1)
            own_shift = s[:, -1:] * own_gate
            s_masked = torch.cat([s[:, :-1], own_shift], dim=-1)
            parts.append(s_masked)
            parts.append(hvg_panel_mask.to(dtype=s.dtype).unsqueeze(-1))
            parts.append(own_gene_shift_mask.to(dtype=s.dtype).unsqueeze(-1))
        if blocks.use_q_sc:
            q_gate = q_sc_mask.to(dtype=q_sc.dtype).unsqueeze(-1)
            parts.append(q_sc * q_gate)
            parts.append(q_sc_mask.to(dtype=q_sc.dtype).unsqueeze(-1))
        if blocks.use_e_g:
            parts.append(e_g)
        if blocks.use_z_c:
            parts.append(z_c)

        x = torch.cat(parts, dim=-1)
        return self.net(x).squeeze(-1)
