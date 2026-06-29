"""Trusted STATE perturbation-vocab loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_pert_vocab(checkpoint: Path) -> dict[str, np.ndarray] | None:
    """Load ``pert_onehot_map.pt`` from ``checkpoint.parent.parent``.

    Returns ``None`` (not ``{}``) when the sibling file is absent. This
    ``None``-on-missing contract is load-bearing: it is exactly the contract
    the existing exp08 ``StateDlProducer._ensure_pert_vocab`` relies on to
    *raise* when distill is configured but the vocab is missing
    (`tests/sl_dl_model/test_train.py::test_distill_required_but_missing_vocab_raises`).
    Returning ``{}`` here would silently turn a required-distill run into a
    bag-only run. Callers that genuinely want "no distill on this backend"
    (e.g. ``linear_mock``) must substitute ``{}`` themselves.

    The file is a trusted project artifact produced with the STATE checkpoint,
    so ``weights_only=False`` is intentional for compatibility with the existing
    serialized NumPy objects.
    """
    vocab_path = Path(checkpoint).parent.parent / "pert_onehot_map.pt"
    if not vocab_path.exists():
        return None
    raw: dict[str, object] = torch.load(
        vocab_path,
        map_location="cpu",
        weights_only=False,
    )
    return {str(k).upper(): np.asarray(v, dtype=np.float32) for k, v in raw.items()}
