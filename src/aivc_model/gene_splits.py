"""Canonical exp05 gene-universe outer-fold manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CANONICAL_GENE_COUNT = 9338
CANONICAL_OUTER_FOLDS = frozenset(range(5))


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_strata(values: np.ndarray, n_splits: int) -> np.ndarray:
    """Assign deterministic, balanced quantile strata for stratified folding."""
    numeric = np.asarray(values, dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("depmap_gene_effect must contain only finite values")
    ranks = pd.Series(numeric).rank(method="first")
    return pd.qcut(ranks, q=n_splits, labels=False).to_numpy(dtype=np.int64)


def build_canonical_outer_manifest(
    labels: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """Assign every canonical gene to exactly one deterministic outer fold."""
    frame = labels[["perturbation_gene", "depmap_gene_effect"]].copy()
    frame["perturbation_gene"] = frame["perturbation_gene"].astype(str).str.upper()
    if (
        len(frame) != CANONICAL_GENE_COUNT
        or frame["perturbation_gene"].nunique() != CANONICAL_GENE_COUNT
    ):
        raise ValueError(
            "canonical gene universe must contain exactly 9338 unique genes"
        )
    frame = frame.sort_values("perturbation_gene").reset_index(drop=True)
    strata = quantile_strata(frame["depmap_gene_effect"].to_numpy(), n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    outer_fold = np.full(len(frame), -1, dtype=np.int64)
    for fold, (_, test_index) in enumerate(splitter.split(frame.index, strata)):
        outer_fold[test_index] = fold
    if np.any(outer_fold < 0):
        raise AssertionError("every canonical gene must receive one outer fold")
    return pd.DataFrame(
        {
            "perturbation_gene": frame["perturbation_gene"],
            "outer_fold": outer_fold,
        }
    )


def load_canonical_outer_manifest(
    path: Path,
    labels: pd.DataFrame,
    expected_sha256: str,
) -> pd.DataFrame:
    """Load a manifest only if its provenance and gene universe are exact."""
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"canonical manifest SHA-256 mismatch: {observed_sha256}")
    manifest = pd.read_csv(path)
    expected_columns = ["perturbation_gene", "outer_fold"]
    if manifest.columns.tolist() != expected_columns:
        raise ValueError(f"canonical manifest columns must be {expected_columns}")

    manifest["perturbation_gene"] = (
        manifest["perturbation_gene"].astype(str).str.upper()
    )
    label_genes = labels["perturbation_gene"].astype(str).str.upper()
    exact_universe = (
        len(manifest) == CANONICAL_GENE_COUNT
        and manifest["perturbation_gene"].nunique() == CANONICAL_GENE_COUNT
        and len(label_genes) == CANONICAL_GENE_COUNT
        and label_genes.nunique() == CANONICAL_GENE_COUNT
        and set(manifest["perturbation_gene"]) == set(label_genes)
    )
    if not exact_universe:
        raise ValueError("canonical gene universe does not match the label table")
    if set(manifest["outer_fold"]) != CANONICAL_OUTER_FOLDS:
        raise ValueError("canonical outer folds must be exactly 0..4")
    if not manifest["outer_fold"].map(lambda value: isinstance(value, int)).all():
        raise ValueError("canonical outer folds must be integers")
    return manifest
