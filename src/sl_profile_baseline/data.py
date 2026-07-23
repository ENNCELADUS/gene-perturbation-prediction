"""Load the official Feng splits and a frozen DepMap GeneEffect panel."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_GENE_COLUMN_RE = re.compile(r"^(?P<symbol>.+?) \((?P<entrez>\d+)\)$")


@dataclass(frozen=True)
class FengFold:
    """One official Feng train/test fold with balanced Rand labels."""

    train_pairs: np.ndarray
    train_labels: np.ndarray
    test_pairs: np.ndarray
    test_labels: np.ndarray


@dataclass(frozen=True)
class ProfileUniverse:
    """GeneEffect profiles aligned to the Feng candidate-gene order."""

    symbols: tuple[str, ...]
    entrez_ids: tuple[int | None, ...]
    cell_lines: tuple[str, ...]
    profiles: np.ndarray
    coverage: np.ndarray
    finite_counts: np.ndarray
    essential_fraction: np.ndarray


def _labeled_pairs(
    positive: np.ndarray, negative: np.ndarray, matrix_index: int, fold_id: int
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positive[matrix_index, fold_id], dtype=int)
    neg = np.asarray(negative[matrix_index, fold_id], dtype=int)
    pairs = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate(
        [np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
    )
    return pairs, labels


def load_feng_fold(path: Path, fold_id: int) -> FengFold:
    """Load train/test pair arrays for one official ``CV*_1.npy`` fold."""
    positive, negative = np.load(Path(path), allow_pickle=True)
    if fold_id < 0 or fold_id >= positive.shape[1]:
        raise ValueError(f"fold_id {fold_id} outside [0, {positive.shape[1]})")
    train_pairs, train_labels = _labeled_pairs(positive, negative, 2, fold_id)
    test_pairs, test_labels = _labeled_pairs(positive, negative, 3, fold_id)
    return FengFold(train_pairs, train_labels, test_pairs, test_labels)


def candidate_gene_count(path: Path) -> int:
    """Return the official candidate-matrix dimension stored in a split file."""
    positive, _ = np.load(Path(path), allow_pickle=True)
    return int(positive[0, 0].shape[0])


def _candidate_symbols(path: Path, count: int) -> tuple[str, ...]:
    frame = pd.read_csv(path).set_index("unified_id")
    missing = sorted(set(range(count)).difference(frame.index))
    if missing:
        raise ValueError(f"fin_entities.csv missing unified ids: {missing[:5]}")
    rows = frame.loc[range(count)]
    if not rows["entity_type_name"].eq("Gene").all():
        raise ValueError("candidate universe contains non-gene entities")
    return tuple(rows["entity_name"].astype(str).str.upper())


def _gene_effect_columns(path: Path) -> tuple[str, dict[str, tuple[str, int]]]:
    header = pd.read_csv(path, nrows=0)
    index_column = str(header.columns[0])
    mapping: dict[str, tuple[str, int]] = {}
    for column in header.columns[1:]:
        match = _GENE_COLUMN_RE.match(str(column))
        if match is None:
            continue
        symbol = match.group("symbol").upper()
        mapping.setdefault(symbol, (str(column), int(match.group("entrez"))))
    return index_column, mapping


def _impute_profile(values: np.ndarray, covered: bool) -> np.ndarray:
    if not covered:
        return np.zeros_like(values, dtype=np.float32)
    finite = np.isfinite(values)
    fill = float(np.median(values[finite]))
    return np.where(finite, values, fill).astype(np.float32)


def load_profile_universe(
    entities_path: Path,
    gene_effect_path: Path,
    candidate_gene_count: int,
    min_finite_lines: int,
    dependency_threshold: float,
) -> ProfileUniverse:
    """Align DepMap profiles to Feng ids and apply label-independent QC."""
    symbols = _candidate_symbols(Path(entities_path), candidate_gene_count)
    index_column, mapping = _gene_effect_columns(Path(gene_effect_path))
    matched_symbols = [symbol for symbol in symbols if symbol in mapping]
    usecols = [index_column, *[mapping[symbol][0] for symbol in matched_symbols]]
    frame = pd.read_csv(gene_effect_path, usecols=usecols, index_col=0)

    n_gene = len(symbols)
    n_line = len(frame)
    profiles = np.zeros((n_gene, n_line), dtype=np.float32)
    finite_counts = np.zeros(n_gene, dtype=int)
    coverage = np.zeros(n_gene, dtype=bool)
    entrez_ids: list[int | None] = []
    for index, symbol in enumerate(symbols):
        if symbol not in mapping:
            entrez_ids.append(None)
            continue
        column, entrez = mapping[symbol]
        entrez_ids.append(entrez)
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite_counts[index] = int(np.isfinite(values).sum())
        coverage[index] = finite_counts[index] >= min_finite_lines
        profiles[index] = _impute_profile(values, bool(coverage[index]))

    essential_fraction = (profiles < dependency_threshold).mean(axis=1)
    essential_fraction[~coverage] = 0.0
    return ProfileUniverse(
        symbols=symbols,
        entrez_ids=tuple(entrez_ids),
        cell_lines=tuple(frame.index.astype(str)),
        profiles=profiles,
        coverage=coverage,
        finite_counts=finite_counts,
        essential_fraction=essential_fraction.astype(np.float32),
    )

