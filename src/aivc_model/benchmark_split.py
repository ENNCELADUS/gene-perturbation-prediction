"""The Exp13 226-line membership authority: the hard fit/train admission guard.

``cell_line_geneeffect_226_split`` (``configs/benchmarks/
cell_line_geneeffect_226_split.json``, ``docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md``
§2, ``docs/data/cell-line-geneeffect-226.md``) is the sole membership authority for
Exp13 -- 172 train / 27 val / 27 test lines, with PC9 (``ACH-000779``) and HeLa
(``ACH-001086``) declared ``unlabeled_train`` members excluded from every supervised
fit. This module replaces the retired Phase-A ``train_head``/``test`` role column
(``tx1_geneeffect_data.assert_training_role``, ``results/phase_a_tx1_20260724/
cell_line_manifest.csv``), which this benchmark does not use -- do not resurrect it.

:func:`assert_fit_eligible` is the hard guard every consumer must call before
admitting a line to a fitting or training path (dependency fitting, context-model
fitting, checkpoint/hyperparameter selection, projection/PCA fits). A ``val``/
``test`` line, or an ``unlabeled_train`` line, reaching a fit is a Critical defect:
this raises, never warns.
"""

from __future__ import annotations

import json
from pathlib import Path

from aivc_model.residual_ladder import FixedSplit

_SPLIT_KEYS: tuple[str, ...] = ("train", "val", "test")

__all__ = ["load_geneeffect_226_split", "assert_fit_eligible"]


def load_geneeffect_226_split(path: Path) -> FixedSplit:
    """Load a ``cell_line_geneeffect_226_split``-shaped JSON into a :class:`FixedSplit`.

    Shape-validates only (JSON object; ``train``/``val``/``test``/
    ``unlabeled_train`` are lists of strings) -- mirrors
    ``scripts/run_r1_residual_ladder.py::_load_split``.

    Raises:
        ValueError: ``path`` is not a JSON object, is missing a required key,
            or a key's value is not a list of strings.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"split JSON {path} must be a JSON object")
    missing = [key for key in _SPLIT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"split JSON {path} is missing key(s): {missing}")
    parts: dict[str, tuple[str, ...]] = {}
    for key in _SPLIT_KEYS:
        value = payload[key]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"split JSON {path}: {key!r} must be a list of strings")
        parts[key] = tuple(value)
    unlabeled = payload.get("unlabeled_train", [])
    if not isinstance(unlabeled, list) or not all(
        isinstance(value, str) for value in unlabeled
    ):
        raise ValueError(
            f"split JSON {path}: 'unlabeled_train' must be a list of strings"
        )
    return FixedSplit(
        train=parts["train"],
        val=parts["val"],
        test=parts["test"],
        unlabeled_train=tuple(unlabeled),
    )


def assert_fit_eligible(model_id: str, split: FixedSplit) -> None:
    """Hard guard: raise unless ``model_id`` may enter a fitting/training path.

    A line is admissible only if it is a *labeled* train member: present in
    ``split.train`` and absent from ``split.unlabeled_train``. ``val``/
    ``test`` lines (and any id outside ``split.train`` entirely) are
    inference-only.

    Args:
        model_id: The candidate line's ACH model id.
        split: The loaded ``cell_line_geneeffect_226_split`` membership
            authority.

    Raises:
        ValueError: ``model_id`` is a declared ``unlabeled_train`` member, or
            is not a member of ``split.train`` at all (a ``val``/``test`` or
            unknown id).
    """
    if model_id in split.unlabeled_train:
        raise ValueError(
            f"{model_id!r} is an unlabeled_train member (no GeneEffect label "
            "under cell_line_geneeffect_226_split) and may not enter a "
            "fitting/training path"
        )
    if model_id not in split.train:
        raise ValueError(
            f"{model_id!r} is not a labeled train member of "
            "cell_line_geneeffect_226_split (val/test lines are "
            "inference-only); it may not enter a fitting/training path"
        )
