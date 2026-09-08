"""Leave-one-anchor-out fold membership and control input layouts for P1-C.

Pure data module: no imports from ``src.training``, ``src.eval`` or
``src.experiments``, and no import of ``src.data.p1b`` (``p1b`` imports from
here, not the reverse).
"""

import numpy as np

ALL_ANCHORS = ("ACH-000551", "ACH-000739", "ACH-000971", "ACH-000995")

FOLDS = {
    "jurkat": "ACH-000995",
    "k562": "ACH-000551",
    "hepg2": "ACH-000739",
    "hct116": "ACH-000971",
}

INPUT_LAYOUTS = ("tx1", "hvg", "hvg_tx1")
HVG_WIDTH = 2000
TRANSFORMS = ("raw", "log1p_norm")


def fold_membership(fold):
    """Return (sources, external) for a leave-one-anchor-out fold.

    ``sources`` preserves ``ALL_ANCHORS`` order with the held-out anchor
    removed; ``external`` is the held-out anchor. Raises ``KeyError`` for an
    unknown fold name.
    """
    held = FOLDS[fold]
    return tuple(a for a in ALL_ANCHORS if a != held), held


def apply_transform(x, transform, target_sum):
    """Apply an evaluation-only preprocessing transform to an HVG-panel bag.

    ``x`` is cells-by-genes (over the 2000-gene HVG panel only, not the whole
    transcriptome). ``"raw"`` is the identity (``target_sum`` ignored).
    ``"log1p_norm"`` scales each row to ``target_sum`` counts (rows whose raw
    sum is zero are left at zero, not divided) and applies ``log1p``. Accepts
    and returns either a ``numpy.ndarray`` or a ``torch.Tensor``.
    """
    if transform == "raw":
        return x
    if transform != "log1p_norm":
        raise ValueError(f"unknown transform {transform!r}")
    if target_sum is None:
        raise ValueError("log1p_norm requires a target_sum")
    if isinstance(x, np.ndarray):
        row_sum = x.sum(axis=-1, keepdims=True)
        scale = np.zeros_like(row_sum, dtype=np.result_type(row_sum, np.float32))
        nonzero = row_sum > 0
        scale[nonzero] = target_sum / row_sum[nonzero]
        return np.log1p(x * scale)
    import torch

    if isinstance(x, torch.Tensor):
        row_sum = x.sum(dim=-1, keepdim=True)
        scale = torch.zeros_like(row_sum)
        nonzero = row_sum > 0
        scale[nonzero] = target_sum / row_sum[nonzero]
        return torch.log1p(x * scale)
    raise TypeError(f"unsupported array type {type(x)!r}")


def median_row_sum(arrays):
    """Median row sum (over the last axis) pooled across all rows of ``arrays``."""
    sums = np.concatenate([np.asarray(a).sum(axis=-1) for a in arrays])
    return float(np.median(sums))


def bundle_transform(bundle):
    """The transform record for ``bundle``, defaulting to raw when absent.

    A bundle predating this feature has no ``"transform"`` key at all; an
    explicit membership check distinguishes that from a bundle that recorded
    ``"raw"`` on purpose.
    """
    if "transform" in bundle:
        return bundle["transform"]
    return {"name": "raw", "target_sum": None, "row_sum_basis": "hvg_panel"}
