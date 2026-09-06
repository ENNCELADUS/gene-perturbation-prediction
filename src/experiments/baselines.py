#!/usr/bin/env python3
"""CLI for the R1 DepMap GeneEffect residual baseline ladder.

Loads plain CSV/JSON inputs, runs :func:`src.baselines.residual.run_r1_ladder`,
and writes ``predictions.csv``, ``per_line.csv``, ``per_gene.csv``, and
``summary.json``. See ``src.baselines.residual`` for the algorithm, the
delta-only prediction convention, and why the default ``--outer fixed``
(standard train/val/test) protocol is what the LOO-mean centering artifact
requires: ``--outer lolo`` keeps the 29-line leave-one-line-out diagnostic
available, at the cost of that artifact family being possible in principle
(and guarded against -- see the ladder module's docstring).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import FixedSplit
from src.baselines.residual import R1Result, run_r1_ladder

_LOGGER = logging.getLogger(__name__)

_LABEL_COLUMNS: tuple[str, ...] = ("model_id", "gene_symbol", "gene_effect")
_PRIOR_COLUMNS: tuple[str, ...] = ("gene_symbol", "gene_effect")
_SPLIT_KEYS: tuple[str, ...] = ("train", "val", "test")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    """Raise ValueError if ``frame`` lacks any of ``columns``."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _require_no_missing(frame: pd.DataFrame, column: str, label: str) -> None:
    """Raise ValueError if ``column`` has a null or blank-string entry."""
    blank = frame[column].astype(str).str.strip() == ""
    if frame[column].isna().any() or blank.any():
        raise ValueError(f"{label}: missing/blank {column}")


def _require_unique(
    frame: pd.DataFrame, columns: Sequence[str] | str, label: str
) -> None:
    """Raise ValueError if ``columns`` (jointly) are not unique per row."""
    cols = [columns] if isinstance(columns, str) else list(columns)
    if frame.duplicated(cols).any():
        raise ValueError(f"{label}: duplicate {cols}")


def _numeric_finite(
    frame: pd.DataFrame, columns: Sequence[str], label: str
) -> pd.DataFrame:
    """Coerce ``columns`` to numeric; raise ValueError if any is non-finite."""
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: non-numeric/non-finite values in {list(columns)}")
    return frame


def _load_labels(path: Path) -> pd.DataFrame:
    """Load/validate the long-form model_id/gene_symbol/gene_effect CSV."""
    frame = pd.read_csv(path)
    label = f"labels CSV {path}"
    _require_columns(frame, _LABEL_COLUMNS, label)
    frame = frame.loc[:, list(_LABEL_COLUMNS)].copy()
    for column in ("model_id", "gene_symbol"):
        _require_no_missing(frame, column, label)
        frame[column] = frame[column].astype(str)
    _require_unique(frame, ["model_id", "gene_symbol"], label)
    return _numeric_finite(frame, ["gene_effect"], label)


def _parse_context_specs(specs: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable ``--context NAME=PATH`` values into name->path."""
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--context value {spec!r} must be NAME=PATH")
        name, _, raw_path = spec.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"--context value {spec!r} has an empty NAME")
        if name in result:
            raise ValueError(f"--context name {name!r} supplied more than once")
        result[name] = Path(raw_path)
    return result


def _load_context_view(path: Path) -> pd.DataFrame:
    """Load a wide context CSV (model_id first, rest numeric) -> indexed frame."""
    frame = pd.read_csv(path)
    label = f"context CSV {path}"
    if len(frame.columns) == 0 or frame.columns[0] != "model_id":
        raise ValueError(f"{label}: first column must be model_id")
    _require_no_missing(frame, "model_id", label)
    frame["model_id"] = frame["model_id"].astype(str)
    _require_unique(frame, "model_id", label)
    feature_columns = [c for c in frame.columns if c != "model_id"]
    if not feature_columns:
        raise ValueError(f"{label}: no feature columns")
    frame = _numeric_finite(frame, feature_columns, label)
    return frame.set_index("model_id").sort_index()


def _load_copy_prior(path: Path) -> pd.Series:
    """Load the optional gene_symbol/gene_effect copy-prior CSV."""
    frame = pd.read_csv(path)
    label = f"copy-prior CSV {path}"
    _require_columns(frame, _PRIOR_COLUMNS, label)
    frame = frame.loc[:, list(_PRIOR_COLUMNS)].copy()
    _require_no_missing(frame, "gene_symbol", label)
    frame["gene_symbol"] = frame["gene_symbol"].astype(str)
    _require_unique(frame, "gene_symbol", label)
    frame = _numeric_finite(frame, ["gene_effect"], label)
    series = frame.set_index("gene_symbol")["gene_effect"].sort_index()
    series.name = "prior"
    return series


def _load_split(path: Path) -> FixedSplit:
    """Load a ``{"train": [...], "val": [...], "test": [...]}`` split JSON.

    Only shape-validates (JSON object, three list-of-string keys); the
    substantive cross-check against ``labels`` (unknown ids, overlaps,
    emptiness) lives in ``src.baselines.residual`` alongside the rest
    of the ladder's correctness logic.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def _write_outputs(result: R1Result, out_dir: Path) -> None:
    """Write predictions.csv, per_line.csv, per_gene.csv, and summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(out_dir / "predictions.csv", index=False)
    result.per_line.to_csv(out_dir / "per_line.csv", index=False)
    result.per_gene.to_csv(out_dir / "per_gene.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeatable; wide context CSV (model_id + numeric features).",
    )
    parser.add_argument("--copy-prior", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--min-lines", type=int, default=3)
    parser.add_argument(
        "--outer",
        choices=("fixed", "lolo"),
        default="fixed",
        help="'fixed' (default): standard train/val/test via --split-json. "
        "'lolo': the 29-line leave-one-line-out diagnostic.",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        help='{"train": [...], "val": [...], "test": [...]}; required when '
        "--outer=fixed (the default).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load inputs, run the ladder, and write its four output artifacts."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.outer == "fixed" and args.split_json is None:
        raise ValueError(
            "--split-json is required when --outer=fixed (the default); "
            "pass --outer lolo for the leave-one-line-out diagnostic instead"
        )

    context_specs = _parse_context_specs(args.context)
    labels = _load_labels(args.labels)
    context_views = {
        name: _load_context_view(path) for name, path in context_specs.items()
    }
    has_prior = args.copy_prior is not None
    copy_prior = _load_copy_prior(args.copy_prior) if has_prior else None
    split = _load_split(args.split_json) if args.split_json is not None else None

    result = run_r1_ladder(
        labels,
        context_views,
        copy_prior,
        pca_components=args.pca_components,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
        min_lines=args.min_lines,
        outer=args.outer,
        split=split,
    )
    _write_outputs(result, args.out_dir)
    _LOGGER.info(
        "Wrote R1 residual ladder results (outer=%s) for slice(s) %s to %s",
        args.outer,
        sorted(result.summary["slices"]),
        args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
