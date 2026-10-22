"""data / geneeffect."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
import numpy as np
import pandas as pd

SPLIT_COUNTS: Final = {"train": 171, "val": 27, "test": 28}


DEPENDENCY_THRESHOLD: Final = -0.5


UNLABELED_TRAIN: Final = ("ACH-000779", "ACH-001086")


SPLIT_SCHEMA_VERSION: Final = "cell-line-geneeffect-226-split-v2"


PINNED_SPLIT_SHA256: Final = (
    "8a264acfa63bf8c2c07d302e1adb5c4ce17edb19dd902f59ef6fd9b6e5434521"
)


_GENE_EFFECT_COLUMN_RE: Final = re.compile(r"^(?P<symbol>\S+) \((?P<entrez>\d+)\)$")


@dataclass(frozen=True)
class Exp13Split:
    """The fixed split authority, keyed only by DepMap ModelID."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    unlabeled_train: tuple[str, ...]

    @property
    def all_model_ids(self) -> tuple[str, ...]:
        return self.train + self.val + self.test

    @property
    def supervised_train(self) -> tuple[str, ...]:
        excluded = set(self.unlabeled_train)
        return tuple(model_id for model_id in self.train if model_id not in excluded)

    def to_manifest(self) -> dict[str, object]:
        return {
            "counts": {name: len(getattr(self, name)) for name in SPLIT_COUNTS},
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
            "unlabeled_train": list(self.unlabeled_train),
        }


def fit_variable_gene_membership(
    labels: pd.DataFrame,
    train_lines: Sequence[str],
    genes: Sequence[str],
    *,
    min_observations: int = 5,
    percentile: float = 75.0,
) -> frozenset[str]:
    """Fit residual-variance membership using labeled training rows only."""
    required = {"model_id", "gene_symbol", "residual"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels is missing columns: {missing}")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = _unique_strings(genes, "variable-gene panel")
    train_set = set(train_lines)
    if not train_set:
        raise ValueError("train_lines must not be empty")
    train = labels.loc[labels["model_id"].isin(train_set)]
    grouped = train.groupby("gene_symbol")["residual"]
    counts = grouped.count().reindex(ordered, fill_value=0)
    eligible = counts[counts >= min_observations].index
    variance = grouped.var(ddof=0).reindex(eligible)
    if variance.empty or not bool(np.isfinite(variance.to_numpy()).all()):
        raise ValueError("variable-gene fit has no finite eligible variances")
    threshold = float(np.percentile(variance.to_numpy(), percentile, method="linear"))
    return frozenset(variance.index[variance >= threshold].astype(str))


def fit_selective_genes(
    labels: pd.DataFrame,
    train_lines: Sequence[str],
    genes: Sequence[str],
    *,
    min_lines: int,
    max_fraction: float,
) -> frozenset[str]:
    """Fit selective-dependency membership using labeled training rows only.

    A gene is selective when at least ``min_lines`` training lines depend on it
    (``gene_effect < DEPENDENCY_THRESHOLD``) and fewer than ``max_fraction`` of
    its labeled training lines do.
    """
    required = {"model_id", "gene_symbol", "gene_effect"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels is missing columns: {missing}")
    if min_lines <= 0:
        raise ValueError("min_lines must be positive")
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError("max_fraction must be in (0, 1]")
    ordered = _unique_strings(genes, "selective-gene panel")
    train_set = set(train_lines)
    if not train_set:
        raise ValueError("train_lines must not be empty")
    train = labels.loc[
        labels["model_id"].isin(train_set) & np.isfinite(labels["gene_effect"])
    ]
    dependent = train["gene_effect"] < DEPENDENCY_THRESHOLD
    grouped = dependent.groupby(train["gene_symbol"])
    counts = grouped.sum().reindex(ordered, fill_value=0)
    rows = grouped.count().reindex(ordered, fill_value=0)
    selective = (counts >= min_lines) & (counts / rows < max_fraction)
    return frozenset(counts.index[selective].astype(str))


def fit_residual_scale(
    labels: pd.DataFrame,
    train_lines: Sequence[str],
    genes: Sequence[str],
    *,
    floor_percentile: float,
) -> pd.Series:
    """Fit the per-gene residual SD on labeled training rows only.

    Population SD (ddof 0) per gene, indexed in ``genes`` order. The floor is
    ``floor_percentile`` over the finite SDs; a gene with fewer than two rows or
    an SD below the floor takes the floor, so every value is finite and positive.
    """
    required = {"model_id", "gene_symbol", "residual"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels is missing columns: {missing}")
    if not 0.0 <= floor_percentile <= 100.0:
        raise ValueError("floor_percentile must be between 0 and 100")
    ordered = _unique_strings(genes, "residual-scale panel")
    train_set = set(train_lines)
    if not train_set:
        raise ValueError("train_lines must not be empty")
    train = labels.loc[
        labels["model_id"].isin(train_set) & np.isfinite(labels["residual"])
    ]
    grouped = train.groupby("gene_symbol")["residual"]
    counts = grouped.count().reindex(ordered, fill_value=0)
    sd = grouped.std(ddof=0).reindex(ordered).where(counts >= 2)
    finite = sd.to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("residual-scale fit has no gene with two training rows")
    floor = float(np.percentile(finite, floor_percentile, method="linear"))
    if not floor > 0.0:
        raise ValueError(f"residual-scale floor must be positive; got {floor}")
    scale = sd.fillna(floor).clip(lower=floor)
    scale.index.name = "gene_symbol"
    scale.name = "residual_scale"
    return scale.astype(np.float64)


def _unique_strings(values: Sequence[object], label: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ValueError(f"{label} contains an empty value")
    duplicates = sorted(value for value, count in Counter(result).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicates: {duplicates[:10]}")
    return result


def load_exp13_split(path: Path) -> Exp13Split:
    """Load and strictly validate the tracked 171/27/28 split authority."""
    path = Path(path)
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha256 != PINNED_SPLIT_SHA256:
        raise ValueError(
            f"Exp13 split SHA-256 mismatch: {observed_sha256} != {PINNED_SPLIT_SHA256}"
        )
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(f"split schema_version must be {SPLIT_SCHEMA_VERSION!r}")
    required = {"train", "val", "test", "unlabeled_train"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"split is missing keys: {missing}")
    groups = {
        name: _unique_strings(payload[name], f"split {name}") for name in required
    }
    for name, expected in SPLIT_COUNTS.items():
        if len(groups[name]) != expected:
            raise ValueError(
                f"split {name} has {len(groups[name])} ModelIDs; expected {expected}"
            )
    membership = groups["train"] + groups["val"] + groups["test"]
    if len(set(membership)) != sum(SPLIT_COUNTS.values()):
        raise ValueError("ModelID membership overlaps across train/val/test")
    if groups["unlabeled_train"] != UNLABELED_TRAIN:
        raise ValueError(
            "unlabeled_train must be exactly PC9 ACH-000779 and HeLa ACH-001086"
        )
    if not set(groups["unlabeled_train"]).issubset(groups["train"]):
        raise ValueError("unlabeled_train contains a non-train ModelID")
    return Exp13Split(
        train=groups["train"],
        val=groups["val"],
        test=groups["test"],
        unlabeled_train=groups["unlabeled_train"],
    )


def parse_gene_symbol(column: object) -> str:
    """Parse and normalize DepMap's ``SYMBOL (EntrezID)`` header."""
    text = str(column).strip()
    match = _GENE_EFFECT_COLUMN_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid DepMap GeneEffect column: {text!r}")
    return match.group("symbol").upper()


def load_geneeffect_long(path: Path, split: Exp13Split) -> pd.DataFrame:
    """Load CRISPRGeneEffect.csv into canonical ModelID/symbol long form."""
    wide = pd.read_csv(path, index_col=0)
    wide.index = wide.index.astype(str)
    if not wide.index.is_unique:
        duplicates = sorted(wide.index[wide.index.duplicated()].unique())
        raise ValueError(f"GeneEffect contains duplicate ModelIDs: {duplicates[:10]}")
    symbols = [parse_gene_symbol(column) for column in wide.columns]
    if len(symbols) != len(set(symbols)):
        duplicates = sorted(
            symbol for symbol, count in Counter(symbols).items() if count > 1
        )
        raise ValueError(
            f"GeneEffect columns map to duplicate symbols: {duplicates[:10]}"
        )
    expected_missing = set(split.unlabeled_train)
    observed_missing = set(split.all_model_ids) - set(wide.index)
    if observed_missing != expected_missing:
        raise ValueError(
            "split label coverage does not match unlabeled_train: "
            f"missing={sorted(observed_missing)}"
        )
    selected_ids = [
        model_id for model_id in split.all_model_ids if model_id in wide.index
    ]
    selected = wide.loc[selected_ids]
    numeric = selected.apply(pd.to_numeric, errors="coerce")
    invalid = selected.notna() & numeric.isna()
    if invalid.any().any():
        invalid_positions = np.argwhere(invalid.to_numpy())[:10]
        examples = [
            (
                str(selected.index[row]),
                str(selected.columns[column]),
                str(selected.iloc[row, column]),
            )
            for row, column in invalid_positions
        ]
        raise ValueError(
            f"GeneEffect contains nonnumeric nonmissing values: {examples}"
        )
    infinite = np.isinf(numeric.to_numpy())
    if infinite.any():
        raise ValueError(f"GeneEffect contains {int(infinite.sum())} infinite values")
    numeric.columns = symbols
    long = (
        numeric.rename_axis("model_id")
        .reset_index()
        .melt(id_vars="model_id", var_name="gene_symbol", value_name="gene_effect")
    )
    return long


